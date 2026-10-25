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

"""Loss-landscape sharpness probes (research module).

Every probe uses ``torch.autograd.grad`` on its own probe batch, so it never
touches the caller's accumulated ``.grad`` and restores the model's
train/eval mode and weights exactly.
"""

from typing import Dict, Any, Optional, List
import time
import numpy as np
import torch
import torch.nn as nn

from arc.signals.base import SignalCollector, SignalSnapshot
from arc.config import SignalConfig


def _trainable(model: nn.Module) -> List[torch.Tensor]:
    return [p for p in model.parameters() if p.requires_grad]


def _hvp(model: nn.Module, loss_fn, inputs, targets, vectors: List[torch.Tensor]):
    """Hessian-vector product of the probe loss with `vectors`."""
    params = _trainable(model)
    loss = loss_fn(model(inputs), targets)
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    gv = sum((g * v).sum() for g, v in zip(grads, vectors) if g is not None)
    return torch.autograd.grad(gv, params, allow_unused=True)


class SharpnessCollector(SignalCollector):

    def __init__(
        self,
        config: SignalConfig,
        num_hutchinson_samples: int = 5,
        power_iterations: int = 10,
        perturbation_radius: float = 0.05,
    ):

        super().__init__(config)
        self.num_hutchinson_samples = num_hutchinson_samples
        self.power_iterations = power_iterations
        self.perturbation_radius = perturbation_radius

        self._hessian_trace: float = 0.0
        self._max_eigenvalue: float = 0.0
        self._gradient_curvature: float = 0.0
        self._sam_sharpness: float = 0.0
        self._flatness_ratio: float = 0.0

        self._trace_ema: float = 0.0
        self._max_eig_ema: float = 0.0
        self._ema_alpha: float = 0.1

    @property
    def name(self) -> str:
        return "sharpness"

    def compute_sharpness(
        self,
        loss_fn: callable,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        model: Optional[nn.Module] = None,
    ) -> Dict[str, float]:

        model = model or self._model
        if model is None:
            return {}

        was_training = model.training
        model.eval()
        try:
            self._hessian_trace = self._hutchinson_trace(model, loss_fn, inputs, targets)
            self._max_eigenvalue = self._power_iteration_max_eig(model, loss_fn, inputs, targets)
            self._gradient_curvature = self._gradient_direction_curvature(
                model, loss_fn, inputs, targets)
            self._sam_sharpness = self._sam_perturbation_sharpness(model, loss_fn, inputs, targets)
        finally:
            model.train(was_training)

        if self._max_eigenvalue > 1e-10:
            self._flatness_ratio = self._hessian_trace / self._max_eigenvalue

        self._trace_ema = self._ema_alpha * self._hessian_trace + (1 - self._ema_alpha) * self._trace_ema
        self._max_eig_ema = self._ema_alpha * self._max_eigenvalue + (1 - self._ema_alpha) * self._max_eig_ema

        return {
            "hessian_trace": self._hessian_trace,
            "max_eigenvalue": self._max_eigenvalue,
            "gradient_curvature": self._gradient_curvature,
            "sam_sharpness": self._sam_sharpness,
            "flatness_ratio": self._flatness_ratio,
        }

    def _hutchinson_trace(self, model, loss_fn, inputs, targets) -> float:
        traces = []
        for _ in range(self.num_hutchinson_samples):
            v = [torch.randint_like(p, 0, 2) * 2.0 - 1.0 for p in _trainable(model)]
            hvp = _hvp(model, loss_fn, inputs, targets, v)
            traces.append(sum(
                (vi * hv).sum().item() for vi, hv in zip(v, hvp) if hv is not None
            ))
        return float(np.mean(traces))

    def _power_iteration_max_eig(self, model, loss_fn, inputs, targets) -> float:
        v = []
        for p in _trainable(model):
            vi = torch.randn_like(p)
            v.append(vi / (vi.norm() + 1e-10))

        for _ in range(self.power_iterations):
            hvp = _hvp(model, loss_fn, inputs, targets, v)
            hvp = [hv if hv is not None else torch.zeros_like(vi) for hv, vi in zip(hvp, v)]
            total_norm = sum((hv ** 2).sum() for hv in hvp).sqrt()
            if total_norm > 1e-10:
                v = [hv / total_norm for hv in hvp]

        hvp = _hvp(model, loss_fn, inputs, targets, v)
        eigenvalue = sum((vi * hv).sum().item() for vi, hv in zip(v, hvp) if hv is not None)
        return float(max(0, eigenvalue))

    def _gradient_direction_curvature(self, model, loss_fn, inputs, targets) -> float:
        params = _trainable(model)
        loss = loss_fn(model(inputs), targets)
        grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)

        grad_norm_sq = sum((g ** 2).sum() for g in grads if g is not None)
        if grad_norm_sq < 1e-10:
            return 0.0

        hvp = torch.autograd.grad(grad_norm_sq, params, allow_unused=True)
        curvature = 0.5 * sum(
            (g * hv).sum().item()
            for g, hv in zip(grads, hvp)
            if g is not None and hv is not None
        ) / grad_norm_sq.item()
        return float(curvature)

    def _sam_perturbation_sharpness(self, model, loss_fn, inputs, targets) -> float:
        params = _trainable(model)
        with torch.enable_grad():
            loss_original = loss_fn(model(inputs), targets)
            grads = torch.autograd.grad(loss_original, params, allow_unused=True)

        grad_norm = float(np.sqrt(sum((g ** 2).sum().item() for g in grads if g is not None))) + 1e-10
        originals = [p.detach().clone() for p in params]

        with torch.no_grad():
            for p, g in zip(params, grads):
                if g is not None:
                    p.add_(self.perturbation_radius * g / grad_norm)
            loss_perturbed = loss_fn(model(inputs), targets)
            for p, original in zip(params, originals):
                p.copy_(original)

        return float(max(0, loss_perturbed.item() - loss_original.item()))

    def _register_hooks(self) -> None:
        # No forward/backward hooks needed: sharpness metrics require an
        # explicit compute_sharpness(loss_fn, inputs, targets) call (Hessian
        # products need a fresh forward pass), not hook-driven collection.
        pass

    def _collect_signals(self) -> Dict[str, Any]:
        return {
            "hessian_trace": self._hessian_trace,
            "hessian_trace_ema": self._trace_ema,
            "max_eigenvalue": self._max_eigenvalue,
            "max_eigenvalue_ema": self._max_eig_ema,
            "gradient_curvature": self._gradient_curvature,
            "sam_sharpness": self._sam_sharpness,
            "flatness_ratio": self._flatness_ratio,
        }

    def collect(self) -> SignalSnapshot:
        return SignalSnapshot(
            step=self._step,
            epoch=self._epoch,
            timestamp=time.time(),
            signals={"sharpness": self._collect_signals()},
            metadata=self._get_metadata(),
        )

    def reset(self) -> None:

        super().reset()
        self._hessian_trace = 0.0
        self._max_eigenvalue = 0.0
        self._gradient_curvature = 0.0
        self._sam_sharpness = 0.0
        self._flatness_ratio = 0.0
