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

from typing import Dict, List, Optional, Tuple, Callable
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from collections import deque

@dataclass
class PINNDiagnostics:
    pde_residual: float
    boundary_residual: float
    initial_residual: float
    total_loss: float
    loss_balance: Dict[str, float]
    stability_score: float
    recommendation: str

class AdaptiveLossBalancer:

    def __init__(
        self,
        n_terms: int = 3,
        method: str = "softmax",
        alpha: float = 0.9,
        temperature: float = 1.0,
    ):
        self.n_terms = n_terms
        self.method = method
        self.alpha = alpha
        self.temperature = temperature

        self.weights = torch.ones(n_terms) / n_terms

        self._weight_history: deque = deque(maxlen=1000)
        self._loss_history: deque = deque(maxlen=1000)
        self._grad_history: deque = deque(maxlen=1000)

    def compute_weights(
        self,
        losses: List[torch.Tensor],
        model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        losses_tensor = torch.stack([l.detach() for l in losses])

        if self.method == "grad_norm" and model is not None:
            weights = self._grad_norm_balancing(losses, model)
        elif self.method == "uncertainty":
            weights = self._uncertainty_balancing(losses_tensor)
        elif self.method == "softmax":
            weights = self._softmax_balancing(losses_tensor)
        else:
            weights = 1.0 / (losses_tensor + 1e-10)
            weights = weights / weights.sum()

        self.weights = self.alpha * self.weights + (1 - self.alpha) * weights

        self._weight_history.append(self.weights.clone())
        self._loss_history.append(losses_tensor)

        return self.weights

    def _grad_norm_balancing(
        self,
        losses: List[torch.Tensor],
        model: nn.Module,
    ) -> torch.Tensor:
        grad_norms = []
        accumulated_grads: Optional[List[Optional[torch.Tensor]]] = None

        for loss in losses:
            model.zero_grad()
            loss.backward(retain_graph=True)

            total_norm = 0.0
            term_grads = []
            for param in model.parameters():
                if param.grad is not None:
                    total_norm += param.grad.norm().item() ** 2
                    term_grads.append(param.grad.detach().clone())
                else:
                    term_grads.append(None)
            grad_norms.append(np.sqrt(total_norm))

            if accumulated_grads is None:
                accumulated_grads = term_grads
            else:
                accumulated_grads = [
                    g if a is None else (a if g is None else a + g)
                    for a, g in zip(accumulated_grads, term_grads)
                ]

        grad_norms = torch.tensor(grad_norms)

        avg_norm = grad_norms.mean()

        weights = avg_norm / (grad_norms + 1e-10)
        weights = weights / weights.sum()

        # restore the accumulated (summed) per-term gradients, so a
        # subsequent optimizer.step() sees the total gradient rather than
        # only the last loss term's.
        model.zero_grad()
        for param, grad in zip(model.parameters(), accumulated_grads or []):
            if grad is not None:
                param.grad = grad

        self._grad_history.append(grad_norms)

        return weights

    def _uncertainty_balancing(self, losses: torch.Tensor) -> torch.Tensor:
        log_vars = torch.log(losses + 1e-10)

        weights = torch.exp(-log_vars)
        weights = weights / weights.sum()

        return weights

    def _softmax_balancing(self, losses: torch.Tensor) -> torch.Tensor:
        scaled = losses / self.temperature
        weights = torch.softmax(-scaled, dim=0)
        return weights

    def get_combined_loss(
        self,
        losses: List[torch.Tensor],
        model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        weights = self.compute_weights(losses, model)

        combined = sum(w.item() * l for w, l in zip(weights, losses))
        return combined

    def get_diagnostics(self) -> Dict[str, float]:
        return {
            f"weight_{i}": w.item() for i, w in enumerate(self.weights)
        }

class LyapunovStabilizer:

    def __init__(
        self,
        model: nn.Module,
        stability_threshold: float = 0.0,
        intervention_strength: float = 0.5,
    ):
        self.model = model
        self.stability_threshold = stability_threshold
        self.intervention_strength = intervention_strength

        self._param_history: deque = deque(maxlen=100)
        self._loss_history: deque = deque(maxlen=100)

        self._lyapunov_exponent: float = 0.0
        self._is_stable: bool = True

    def update(self, loss: torch.Tensor) -> Dict[str, float]:
        self._loss_history.append(loss.item() if isinstance(loss, torch.Tensor) else loss)

        params = torch.cat([
            p.detach().flatten() for p in self.model.parameters()
        ])
        self._param_history.append(params)

        if len(self._param_history) >= 10:
            self._lyapunov_exponent = self._estimate_lyapunov()

        self._is_stable = self._lyapunov_exponent < self.stability_threshold

        return {
            "lyapunov_exponent": self._lyapunov_exponent,
            "is_stable": self._is_stable,
            "n_samples": len(self._param_history),
        }

    def _estimate_lyapunov(self) -> float:
        if len(self._param_history) < 10:
            return 0.0

        deltas = []

        for i in range(1, min(len(self._param_history), 50)):
            prev = self._param_history[-i-1]
            curr = self._param_history[-i]

            delta = (curr - prev).norm().item()
            if delta > 1e-10:
                deltas.append(np.log(delta))

        if len(deltas) < 2:
            return 0.0

        t = np.arange(len(deltas))
        lyap = np.polyfit(t, deltas, 1)[0]

        return float(lyap)

    def stabilize_gradients(
        self,
        gradients: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        if self._is_stable:
            return None

        clip_value = (1.0 - self.intervention_strength) +                     self.intervention_strength * np.exp(-self._lyapunov_exponent)

        if gradients is None:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-clip_value, clip_value)
            return None
        else:
            return {
                name: grad.clamp(-clip_value, clip_value)
                for name, grad in gradients.items()
            }

    @property
    def stability_score(self) -> float:
        return float(np.exp(-max(self._lyapunov_exponent, 0)))

class CurriculumScheduler:

    def __init__(
        self,
        n_stages: int = 5,
        warmup_epochs: int = 10,
        strategy: str = "linear",
    ):
        self.n_stages = n_stages
        self.warmup_epochs = warmup_epochs
        self.strategy = strategy

        self._current_stage = 0
        self._epoch_in_stage = 0
        self._difficulty = 0.0

    def step(self, epoch: int, loss: Optional[float] = None) -> float:
        if self.strategy == "linear":
            self._difficulty = min(epoch / (self.n_stages * self.warmup_epochs), 1.0)

        elif self.strategy == "exponential":
            self._difficulty = 1.0 - np.exp(-epoch / (self.n_stages * self.warmup_epochs))

        elif self.strategy == "adaptive" and loss is not None:
            if loss < 1e-3:
                self._difficulty = min(self._difficulty + 0.1, 1.0)

        return self._difficulty

    def get_frequency_scale(self) -> float:
        base_scale = 1.0
        max_scale = 10.0

        return base_scale + self._difficulty * (max_scale - base_scale)

    def get_domain_fraction(self) -> float:
        min_fraction = 0.2
        return min_fraction + self._difficulty * (1.0 - min_fraction)

    def get_collocation_density(self) -> float:
        return 0.5 + self._difficulty * 0.5

class PINNStabilizer:
    def __init__(
        self,
        model: nn.Module,
        n_loss_terms: int = 3,
        use_curriculum: bool = True,
        use_lyapunov: bool = True,
    ):
        self.model = model

        self.loss_balancer = AdaptiveLossBalancer(n_terms=n_loss_terms)
        self.lyapunov = LyapunovStabilizer(model) if use_lyapunov else None
        self.curriculum = CurriculumScheduler() if use_curriculum else None

        self._diagnostics_history: deque = deque(maxlen=1000)

    def get_stabilized_loss(
        self,
        losses: List[torch.Tensor],
        loss_names: Optional[List[str]] = None,
    ) -> torch.Tensor:
        return self.loss_balancer.get_combined_loss(losses, self.model)

    def stabilize_step(self) -> None:
        if self.lyapunov is not None:
            self.lyapunov.stabilize_gradients()

    def update(
        self,
        epoch: int,
        loss: torch.Tensor,
        individual_losses: Optional[List[float]] = None,
    ) -> PINNDiagnostics:
        loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss

        lyap_info = {}
        if self.lyapunov is not None:
            lyap_info = self.lyapunov.update(loss)

        if self.curriculum is not None:
            self.curriculum.step(epoch, loss_val)

        balance = self.loss_balancer.get_diagnostics()

        stability = lyap_info.get("is_stable", True)
        stability_score = self.lyapunov.stability_score if self.lyapunov else 1.0

        if not stability:
            recommendation = "Unstable training detected. Reducing learning rate recommended."
        elif loss_val > 1.0:
            recommendation = "High loss. Consider increasing curriculum warmup."
        else:
            recommendation = "Training stable."

        ind_losses = individual_losses or [0.0, 0.0, 0.0]
        diagnostics = PINNDiagnostics(
            pde_residual=ind_losses[0] if len(ind_losses) > 0 else 0.0,
            boundary_residual=ind_losses[1] if len(ind_losses) > 1 else 0.0,
            initial_residual=ind_losses[2] if len(ind_losses) > 2 else 0.0,
            total_loss=loss_val,
            loss_balance=balance,
            stability_score=stability_score,
            recommendation=recommendation,
        )

        self._diagnostics_history.append(diagnostics)

        return diagnostics

    def get_training_params(self) -> Dict[str, float]:
        if self.curriculum is None:
            return {"frequency_scale": 1.0, "domain_fraction": 1.0}

        return {
            "frequency_scale": self.curriculum.get_frequency_scale(),
            "domain_fraction": self.curriculum.get_domain_fraction(),
            "collocation_density": self.curriculum.get_collocation_density(),
            "difficulty": self.curriculum._difficulty,
        }