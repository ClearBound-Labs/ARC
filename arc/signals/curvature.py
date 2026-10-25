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

from typing import Dict, Any, Optional, List
import warnings
import torch
import torch.nn as nn
import numpy as np

from arc.signals.base import SignalCollector
from arc.config import SignalConfig

class CurvatureCollector(SignalCollector):

    def __init__(self, config: Optional[SignalConfig] = None):
        super().__init__(config)
        self.config = config or SignalConfig()

        self._gradient_history: List[torch.Tensor] = []
        self._max_history: int = 10

    def _register_hooks(self) -> None:

        pass

    def record_gradients(self) -> None:

        if self._model is None:
            return

        grads = []
        for param in self._model.parameters():
            if param.grad is not None:
                grads.append(param.grad.detach().flatten())

        if grads:
            grad_flat = torch.cat(grads)
            self._gradient_history.append(grad_flat.cpu())

            if len(self._gradient_history) > self._max_history:
                self._gradient_history.pop(0)

    def _collect_signals(self) -> Dict[str, Any]:

        signals = {
            "grad_variance": {},
            "hvp": {},
            "global": {},
        }

        if len(self._gradient_history) >= 2:
            grad_matrix = torch.stack(self._gradient_history)

            param_variance = grad_matrix.var(dim=0)

            signals["grad_variance"]["mean"] = param_variance.mean().item()
            signals["grad_variance"]["max"] = param_variance.max().item()
            signals["grad_variance"]["min"] = param_variance.min().item()
            signals["grad_variance"]["std"] = param_variance.std().item()

            signals["global"]["trace_proxy"] = param_variance.sum().item()

            grad_mean = grad_matrix.mean(dim=0)
            mean_norm = torch.norm(grad_mean)
            if mean_norm > 1e-8:
                cv = torch.norm(param_variance.sqrt()) / mean_norm
                signals["global"]["grad_cv"] = cv.item()

        if self.config.compute_curvature_proxy and self._model is not None:
            hvp_signals = self._compute_hvp_estimates()
            signals["hvp"] = hvp_signals

        self._gradient_history.clear()

        return signals

    def _compute_hvp_estimates(self) -> Dict[str, Any]:

        hvp_signals = {}

        if self._model is None:
            return hvp_signals

        hvp_norms = []

        try:

            grads = []
            for param in self._model.parameters():
                if param.grad is not None:
                    grads.append(param.grad.detach().flatten())

            if not grads:
                return hvp_signals

            grad_flat = torch.cat(grads)
            n_params = grad_flat.numel()

            for _ in range(self.config.curvature_hvp_samples):

                v = torch.randint(0, 2, (n_params,), dtype=grad_flat.dtype, device=grad_flat.device)
                v = 2 * v - 1
                v = v / torch.norm(v)

                gv = torch.dot(grad_flat, v)
                hvp_norms.append(abs(gv.item()))

            if hvp_norms:
                hvp_signals["mean_hvp_norm"] = np.mean(hvp_norms)
                hvp_signals["max_hvp_norm"] = np.max(hvp_norms)
                hvp_signals["min_hvp_norm"] = np.min(hvp_norms)

                if hvp_signals["min_hvp_norm"] > 1e-10:
                    hvp_signals["curvature_ratio"] = (
                        hvp_signals["max_hvp_norm"] / hvp_signals["min_hvp_norm"]
                    )

        except Exception as e:
            warnings.warn(f"HVP curvature computation failed: {e}")

        return hvp_signals

    def reset(self) -> None:

        super().reset()
        self._gradient_history.clear()

    def _get_metadata(self) -> Dict[str, Any]:

        base = super()._get_metadata()
        base.update({
            "hvp_enabled": self.config.compute_curvature_proxy,
            "hvp_samples": self.config.curvature_hvp_samples,
            "gradient_history_size": len(self._gradient_history),
        })
        return base