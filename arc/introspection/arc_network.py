# ARC (Automatic Recovery Controller) - Self-Healing Neural Networks
# Copyright (c) 2026 Aryan Kaushik. All rights reserved.
#
# This file is part of ARC.
#
# ARC is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# ARC is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for
# more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ARC. If not, see <https://www.gnu.org/licenses/>.

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

from arc.introspection.geometry import (
    FisherRaoGeometry, GeometricState, GeodesicTracker
)
from arc.introspection.dynamics import (
    LyapunovEstimator, AttractorDetector, PhaseSpaceAnalyzer, DynamicalState
)
from arc.introspection.topology import (
    TopologicalFeatures, TopologicalState
)
from arc.introspection.transport import (
    WassersteinMonitor, TransportState
)
from arc.introspection.self_model import (
    SelfModelingHead, MetaCognition, SelfKnowledge
)

@dataclass
class IntrospectiveState:
    geometry: Optional[GeometricState] = None

    dynamics: Optional[DynamicalState] = None

    topology: Optional[TopologicalState] = None

    transport: Optional[TransportState] = None

    self_knowledge: Optional[SelfKnowledge] = None

    overall_health: float = 1.0
    risk_score: float = 0.0
    should_intervene: bool = False
    recommended_action: str = "continue"
    confidence: float = 0.5

    epoch: int = 0
    step: int = 0

@dataclass
class IntrospectionAction:
    action_type: str
    magnitude: float
    rationale: str

class IntrospectiveArc(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        enable_geometry: bool = True,
        enable_dynamics: bool = True,
        enable_topology: bool = False,
        enable_transport: bool = True,
        enable_self_model: bool = True,
        auto_intervene: bool = False,
    ):
        super().__init__()

        self.base_model = base_model
        self.optimizer = optimizer
        self.auto_intervene = auto_intervene

        self.scheduler = IntrospectionScheduler(mode="adaptive")
        self.smoother = SignalSmoother(beta=0.9)

        self.geometry = None
        self.dynamics = None
        self.topology = None
        self.transport = None
        self.self_model = None

        if enable_geometry:
            self.geometry = FisherRaoGeometry(base_model)
            self.geodesic_tracker = GeodesicTracker(self.geometry)

        if enable_dynamics:
            self.lyapunov = LyapunovEstimator(base_model)
            self.attractor = AttractorDetector(base_model)
            self.dynamics = PhaseSpaceAnalyzer(self.lyapunov, self.attractor)

        if enable_topology:
            self.topology = TopologicalFeatures(base_model)

        if enable_transport:
            self.transport = WassersteinMonitor(base_model)

        if enable_self_model:
            self.self_model = SelfModelingHead(base_model)

        self.controller = IntrospectionController()

        self._current_state: Optional[IntrospectiveState] = None
        self._state_history: deque = deque(maxlen=100)
        self._epoch = 0
        self._step = 0
        self._current_loss: Optional[float] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_model(x)

    def introspective_step(
        self,
        loss: torch.Tensor,
        inputs: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
    ) -> IntrospectiveState:
        self._current_loss = loss.item()
        self._step += 1

        state = IntrospectiveState(epoch=self._epoch, step=self._step)

        if self.geometry and self.scheduler.should_run_module("geometry", state.risk_score):
            self.geometry.accumulate(loss)
            state.geometry = self.geometry.compute_state()

        if self.dynamics and self.scheduler.should_run_module("dynamics", state.risk_score):
            state.dynamics = self.dynamics.update()

        if self.topology and inputs is not None and targets is not None:
            if self.scheduler.should_run_module("topology", state.risk_score):
                state.topology = self.topology.compute(
                    lambda o, t: F.cross_entropy(o, t),
                    inputs, targets
                )

        if self.transport and self.scheduler.should_run_module("transport", state.risk_score):
            state.transport = self.transport.update()

        if self.self_model and self.scheduler.should_run_module("self_model", state.risk_score):
            state.self_knowledge = self.self_model.introspect(loss.item())

        state = self._compute_meta_analysis(state)

        state.overall_health = self.smoother.update("health", state.overall_health)
        state.risk_score = self.smoother.update("risk", state.risk_score)

        self.scheduler.step()

        self._current_state = state
        self._state_history.append(state)

        if self.auto_intervene and state.should_intervene:
            self._apply_intervention(state.recommended_action)

        return state

    def _compute_meta_analysis(self, state: IntrospectiveState) -> IntrospectiveState:
        risk_factors = []
        confidence_factors = []

        if state.geometry:
            if state.geometry.riemannian_curvature > 10:
                risk_factors.append(0.3)

            if state.geometry.natural_gradient_alignment < 0.5:
                risk_factors.append(0.2)

            confidence_factors.append(0.8)

        if state.dynamics:
            if state.dynamics.lyapunov_exponent > 0.1:
                risk_factors.append(0.5)

            if state.dynamics.is_at_bifurcation:
                risk_factors.append(0.3)

            confidence_factors.append(0.9)

        if state.transport:
            if state.transport.mode_collapse_risk > 0.7:
                risk_factors.append(0.6)

            if state.transport.drift_velocity < 0.001:
                risk_factors.append(0.2)

            confidence_factors.append(0.7)

        if state.self_knowledge:
            if state.self_knowledge.surprise > 2.0:
                risk_factors.append(0.4)

            confidence_factors.append(state.self_knowledge.confidence)

        if risk_factors:
            state.risk_score = min(1.0, sum(risk_factors))
            state.overall_health = max(0.0, 1.0 - state.risk_score)

        if confidence_factors:
            state.confidence = sum(confidence_factors) / len(confidence_factors)

        state.should_intervene = state.risk_score > 0.5 and state.confidence > 0.6

        state.recommended_action = self.controller.recommend_action(state)

        return state

    def _apply_intervention(self, action: str) -> None:
        if self.optimizer is None:
            return

        if action == "reduce_lr":
            for group in self.optimizer.param_groups:
                group['lr'] *= 0.5

        elif action == "increase_regularization":
            for group in self.optimizer.param_groups:
                if 'weight_decay' in group:
                    group['weight_decay'] *= 2.0

        elif action == "reset_momentum":
            self.optimizer.state.clear()

    def on_epoch_end(self) -> Dict[str, Any]:
        self._epoch += 1

        summary = {
            "epoch": self._epoch,
            "steps": self._step,
        }

        if self._current_state:
            summary.update({
                "health": self._current_state.overall_health,
                "risk": self._current_state.risk_score,
                "confidence": self._current_state.confidence,
            })

            if self._current_state.geometry:
                summary["geodesic_distance"] = self._current_state.geometry.geodesic_velocity

            if self._current_state.dynamics:
                summary["lyapunov"] = self._current_state.dynamics.lyapunov_exponent
                summary["phase"] = self._current_state.dynamics.phase

        return summary

    def get_introspective_state(self) -> Optional[IntrospectiveState]:
        return self._current_state

    def get_health_trajectory(self) -> List[float]:
        return [s.overall_health for s in self._state_history]

class IntrospectionScheduler:
    def __init__(self, mode: str = "adaptive"):
        self.mode = mode
        self.risk_level = 0.0
        self.step_counter = 0

    def should_run_module(self, module_name: str, risk_score: float) -> bool:
        if self.mode == "always_on":
            return True
        if self.mode == "always_off":
            return False

        self.risk_level = risk_score

        if module_name in ["loss", "gradient"]:
            return True

        if module_name == "geometry":
            return risk_score > 0.1 or self.step_counter % 10 == 0

        if module_name == "dynamics":
            return risk_score > 0.2 or self.step_counter % 20 == 0

        if module_name == "transport":
            return risk_score > 0.4 or self.step_counter % 50 == 0

        if module_name == "topology":
            return risk_score > 0.6

        if module_name == "self_model":
            return risk_score > 0.3 or self.step_counter % 20 == 0

        return True

    def step(self):
        self.step_counter += 1

class SignalSmoother:
    def __init__(self, beta: float = 0.9):
        self.beta = beta
        self.values = {}

    def update(self, name: str, value: float) -> float:
        if name not in self.values:
            self.values[name] = value
            return value

        smoothed = self.beta * self.values[name] + (1 - self.beta) * value
        self.values[name] = smoothed
        return smoothed

class IntrospectionController:

    def __init__(self):
        self._intervention_history: List[str] = []

    def recommend_action(self, state: IntrospectiveState) -> str:
        if state.dynamics and state.dynamics.lyapunov_exponent > 0.2:
            return "reduce_lr"

        if state.transport and state.transport.mode_collapse_risk > 0.8:
            return "reset_momentum"

        if state.geometry and state.geometry.riemannian_curvature > 20:
            return "increase_regularization"

        if state.self_knowledge and state.self_knowledge.surprise > 3.0:
            return "reduce_lr"

        return "continue"

def create_self_aware_model(
    base_model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    level: str = "standard",
) -> IntrospectiveArc:
    configs = {
        "minimal": {
            "enable_geometry": True,
            "enable_dynamics": True,
            "enable_topology": False,
            "enable_transport": False,
            "enable_self_model": True,
        },
        "standard": {
            "enable_geometry": True,
            "enable_dynamics": True,
            "enable_topology": False,
            "enable_transport": True,
            "enable_self_model": True,
        },
        "full": {
            "enable_geometry": True,
            "enable_dynamics": True,
            "enable_topology": True,
            "enable_transport": True,
            "enable_self_model": True,
        },
    }

    config = configs.get(level, configs["standard"])

    return IntrospectiveArc(base_model, optimizer=optimizer, **config)