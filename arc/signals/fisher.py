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
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from arc.signals.base import SignalCollector, SignalSnapshot
from arc.config import SignalConfig

class FisherCollector(SignalCollector):

    def __init__(
        self,
        config: SignalConfig,
        diagonal_only: bool = True,
        ema_decay: float = 0.99,
        spectral_samples: int = 10,
    ):

        super().__init__(config)
        self.diagonal_only = diagonal_only
        self.ema_decay = ema_decay
        self.spectral_samples = spectral_samples

        self._diagonal_fisher: Dict[str, torch.Tensor] = {}
        self._gradient_covariance: Dict[str, torch.Tensor] = {}

        self._fisher_trace: float = 0.0
        self._fisher_spectral_norm: float = 0.0
        self._effective_rank: float = 0.0
        self._gradient_snr: float = 0.0

        self._prev_params: Optional[Dict[str, torch.Tensor]] = None
        self._fisher_rao_distance: float = 0.0

        self._batch_count: int = 0

    @property
    def name(self) -> str:
        return "fisher"

    def accumulate_fisher(self, loss: Optional[torch.Tensor] = None) -> None:

        if self._model is None:
            return

        for name, param in self._model.named_parameters():
            if param.grad is None:
                continue

            grad = param.grad.detach()

            grad_sq = grad ** 2

            if name not in self._diagonal_fisher:
                self._diagonal_fisher[name] = grad_sq.clone()
            else:

                self._diagonal_fisher[name] = (
                    self.ema_decay * self._diagonal_fisher[name] +
                    (1 - self.ema_decay) * grad_sq
                )

            if name not in self._gradient_covariance:
                self._gradient_covariance[name] = {
                    "mean": grad.clone(),
                    "mean_sq": grad_sq.clone(),
                    "count": 1,
                }
            else:
                cov = self._gradient_covariance[name]
                n = cov["count"]
                cov["mean"] = (n * cov["mean"] + grad) / (n + 1)
                cov["mean_sq"] = (n * cov["mean_sq"] + grad_sq) / (n + 1)
                cov["count"] = n + 1

        self._batch_count += 1

    def compute_metrics(self) -> Dict[str, float]:

        if not self._diagonal_fisher:
            return {}

        self._fisher_trace = sum(
            f.sum().item() for f in self._diagonal_fisher.values()
        )

        self._effective_rank = self._compute_effective_rank()

        self._fisher_spectral_norm = self._estimate_spectral_norm()

        self._gradient_snr = self._compute_gradient_snr()

        self._fisher_rao_distance = self._compute_fisher_rao_distance()

        return {
            "fisher_trace": self._fisher_trace,
            "fisher_spectral_norm": self._fisher_spectral_norm,
            "effective_rank": self._effective_rank,
            "gradient_snr": self._gradient_snr,
            "fisher_rao_distance": self._fisher_rao_distance,
        }

    def _compute_effective_rank(self) -> float:

        if self._fisher_trace < 1e-10:
            return 0.0

        all_diag = []
        for f in self._diagonal_fisher.values():
            all_diag.append(f.flatten())

        if not all_diag:
            return 0.0

        diag = torch.cat(all_diag)

        p = diag / (diag.sum() + 1e-10)

        entropy = -(p * (p + 1e-10).log()).sum().item()

        return float(np.exp(entropy))

    def _estimate_spectral_norm(self) -> float:

        if not self._diagonal_fisher:
            return 0.0

        if self.diagonal_only:

            max_val = max(
                f.max().item() for f in self._diagonal_fisher.values()
            )
            return float(max_val)

        return 0.0

    def _compute_gradient_snr(self) -> float:

        if not self._gradient_covariance:
            return 0.0

        signal = 0.0
        noise = 0.0

        for name, cov in self._gradient_covariance.items():
            mean = cov["mean"]
            mean_sq = cov["mean_sq"]

            variance = mean_sq - mean ** 2

            signal += (mean ** 2).sum().item()
            noise += variance.sum().item()

        if noise < 1e-10:
            return float('inf')

        return float(signal / noise)

    def _compute_fisher_rao_distance(self) -> float:

        if self._model is None:
            return 0.0

        current_params = {}
        for name, param in self._model.named_parameters():
            current_params[name] = param.data.detach().clone()

        if self._prev_params is None:
            self._prev_params = current_params
            return 0.0

        distance_sq = 0.0

        for name in current_params:
            if name not in self._prev_params:
                continue
            if name not in self._diagonal_fisher:
                continue

            delta = current_params[name] - self._prev_params[name]
            fisher = self._diagonal_fisher[name]

            distance_sq += (delta ** 2 * fisher).sum().item()

        self._prev_params = current_params

        return float(np.sqrt(max(0, distance_sq)))

    def get_parameter_importance(self, top_k: int = 10) -> List[Tuple[str, float]]:

        importance = []

        for name, fisher in self._diagonal_fisher.items():
            importance.append((name, fisher.sum().item()))

        importance.sort(key=lambda x: x[1], reverse=True)

        return importance[:top_k]

    def _register_hooks(self) -> None:
        # No forward/backward hooks needed: Fisher stats are accumulated by
        # explicit accumulate_fisher() calls after loss.backward(), not hooks.
        pass

    def _collect_signals(self) -> Dict[str, Any]:
        return self.compute_metrics()

    def collect(self) -> SignalSnapshot:

        self.compute_metrics()

        signals = {
            "fisher": {
                "trace": self._fisher_trace,
                "spectral_norm": self._fisher_spectral_norm,
                "effective_rank": self._effective_rank,
                "gradient_snr": self._gradient_snr,
                "fisher_rao_distance": self._fisher_rao_distance,
                "batch_count": self._batch_count,
            }
        }

        return SignalSnapshot(
            step=self._step,
            epoch=self._epoch,
            timestamp=time.time(),
            signals=signals,
            metadata=self._get_metadata(),
        )

    def reset(self) -> None:

        super().reset()
        self._diagonal_fisher.clear()
        self._gradient_covariance.clear()
        self._batch_count = 0
        self._prev_params = None