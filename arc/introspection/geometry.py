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
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from collections import deque

@dataclass
class GeometricState:
    fisher_trace: float
    fisher_rank: float
    condition_number: float
    geodesic_velocity: float
    riemannian_curvature: float
    natural_gradient_alignment: float

class FisherRaoGeometry:
    def __init__(
        self,
        model: nn.Module,
        mode: str = "diagonal",
        damping: float = 1e-4,
        ema_decay: float = 0.95,
    ):
        self.model = model
        self.mode = mode
        self.damping = damping
        self.ema_decay = ema_decay

        self._fisher_diag: Dict[str, torch.Tensor] = {}

        self._kronecker_factors: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        self._prev_params: Optional[Dict[str, torch.Tensor]] = None
        self._param_velocities: Dict[str, torch.Tensor] = {}

        self._prev_grads: Optional[Dict[str, torch.Tensor]] = None

        self._initialize_fisher()

    def _initialize_fisher(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._fisher_diag[name] = torch.ones_like(param) * self.damping

    def accumulate(self, loss: Optional[torch.Tensor] = None) -> None:
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue

            grad = param.grad.detach()

            grad_sq = grad ** 2

            if name in self._fisher_diag:
                self._fisher_diag[name] = (
                    self.ema_decay * self._fisher_diag[name] +
                    (1 - self.ema_decay) * grad_sq
                )
            else:
                self._fisher_diag[name] = grad_sq

        self._prev_grads = {
            name: param.grad.detach().clone()
            for name, param in self.model.named_parameters()
            if param.grad is not None
        }

    def compute_natural_gradient(self) -> Dict[str, torch.Tensor]:
        natural_grads = {}

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue

            grad = param.grad

            fisher = self._fisher_diag.get(name, torch.ones_like(grad))
            natural_grads[name] = grad / (fisher + self.damping)

        return natural_grads

    def compute_geodesic_distance(self) -> float:
        if self._prev_params is None:
            self._update_prev_params()
            return 0.0

        geodesic_sq = 0.0

        for name, param in self.model.named_parameters():
            if name not in self._prev_params:
                continue

            delta = param.data - self._prev_params[name]
            fisher = self._fisher_diag.get(name, torch.ones_like(delta))

            weighted_sq = (delta ** 2 * fisher).sum().item()
            geodesic_sq += weighted_sq

            self._param_velocities[name] = delta.clone()

        self._update_prev_params()

        return math.sqrt(max(0, geodesic_sq))

    def _update_prev_params(self) -> None:
        self._prev_params = {
            name: param.data.detach().clone()
            for name, param in self.model.named_parameters()
        }

    def compute_riemannian_curvature(self) -> float:
        if not self._param_velocities:
            return 0.0

        fisher_values = []
        velocity_values = []

        for name, fisher in self._fisher_diag.items():
            if name in self._param_velocities:
                fisher_values.append(fisher.flatten())
                velocity_values.append(self._param_velocities[name].flatten())

        if not fisher_values:
            return 0.0

        F = torch.cat(fisher_values)
        v = torch.cat(velocity_values)

        F_weighted = F * (v ** 2)
        curvature = (F_weighted.var() / (F_weighted.mean() + 1e-10)).item()

        return float(curvature)

    def compute_natural_gradient_alignment(self) -> float:
        if self._prev_grads is None:
            return 1.0

        natural_grads = self.compute_natural_gradient()

        dot_product = 0.0
        grad_norm_sq = 0.0
        nat_grad_norm_sq = 0.0

        for name, grad in self._prev_grads.items():
            if name in natural_grads:
                nat_grad = natural_grads[name]
                dot_product += (grad * nat_grad).sum().item()
                grad_norm_sq += (grad ** 2).sum().item()
                nat_grad_norm_sq += (nat_grad ** 2).sum().item()

        if grad_norm_sq < 1e-10 or nat_grad_norm_sq < 1e-10:
            return 1.0

        alignment = dot_product / (math.sqrt(grad_norm_sq) * math.sqrt(nat_grad_norm_sq))

        return float(alignment)

    def compute_effective_dimension(self) -> float:
        all_fisher = torch.cat([f.flatten() for f in self._fisher_diag.values()])

        trace = all_fisher.sum().item()
        trace_sq = (all_fisher ** 2).sum().item()

        if trace_sq < 1e-10:
            return 0.0

        return trace ** 2 / trace_sq

    def compute_state(self) -> GeometricState:
        fisher_trace = sum(f.sum().item() for f in self._fisher_diag.values())

        fisher_rank = self.compute_effective_dimension()

        all_fisher = torch.cat([f.flatten() for f in self._fisher_diag.values()])
        condition = (all_fisher.max() / (all_fisher.min() + self.damping)).item()

        geodesic_velocity = self.compute_geodesic_distance()

        riemannian_curvature = self.compute_riemannian_curvature()

        alignment = self.compute_natural_gradient_alignment()

        return GeometricState(
            fisher_trace=fisher_trace,
            fisher_rank=fisher_rank,
            condition_number=condition,
            geodesic_velocity=geodesic_velocity,
            riemannian_curvature=riemannian_curvature,
            natural_gradient_alignment=alignment,
        )

class NaturalGradient:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        geometry: FisherRaoGeometry,
        natural_ratio: float = 0.5,
    ):
        self.optimizer = optimizer
        self.geometry = geometry
        self.natural_ratio = natural_ratio

    def step(self) -> None:
        natural_grads = self.geometry.compute_natural_gradient()

        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue

                param_name = None
                for name, p in self.geometry.model.named_parameters():
                    if p is param:
                        param_name = name
                        break

                if param_name and param_name in natural_grads:
                    sgd_grad = param.grad
                    nat_grad = natural_grads[param_name]

                    blended = (
                        (1 - self.natural_ratio) * sgd_grad +
                        self.natural_ratio * nat_grad
                    )
                    param.grad = blended

        self.optimizer.step()

class GeodesicTracker:

    def __init__(self, geometry: FisherRaoGeometry, window_size: int = 100):
        self.geometry = geometry
        self.window_size = window_size

        self._states: deque = deque(maxlen=window_size)
        self._total_geodesic_distance: float = 0.0
        self._total_euclidean_distance: float = 0.0

    def update(self) -> GeometricState:
        state = self.geometry.compute_state()

        self._states.append(state)

        self._total_geodesic_distance += state.geodesic_velocity

        return state

    @property
    def total_geodesic_distance(self) -> float:
        return self._total_geodesic_distance

    @property
    def path_efficiency(self) -> float:
        if len(self._states) < 2:
            return 1.0

        straight_line = abs(self._states[-1].geodesic_velocity - self._states[0].geodesic_velocity)
        actual_path = sum(s.geodesic_velocity for s in self._states)

        if actual_path < 1e-10:
            return 1.0

        return min(1.0, straight_line / actual_path)

    def get_curvature_profile(self) -> List[float]:
        return [s.riemannian_curvature for s in self._states]

    def get_summary(self) -> Dict[str, float]:
        if not self._states:
            return {}

        curvatures = self.get_curvature_profile()

        return {
            "total_geodesic_distance": self._total_geodesic_distance,
            "path_efficiency": self.path_efficiency,
            "mean_curvature": np.mean(curvatures) if curvatures else 0,
            "max_curvature": max(curvatures) if curvatures else 0,
            "mean_fisher_rank": np.mean([s.fisher_rank for s in self._states]),
            "mean_alignment": np.mean([s.natural_gradient_alignment for s in self._states]),
        }