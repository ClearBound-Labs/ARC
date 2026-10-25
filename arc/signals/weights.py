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

class WeightCollector(SignalCollector):

    def __init__(self, config: Optional[SignalConfig] = None):
        super().__init__(config)
        self.config = config or SignalConfig()

        self._prev_weights: Dict[str, torch.Tensor] = {}
        self._prev_norms: Dict[str, float] = {}
        self._layer_names: List[str] = []

    def _register_hooks(self) -> None:

        if self._model is None:
            return

        for name, param in self._model.named_parameters():
            if param.requires_grad:
                self._layer_names.append(name)

    def _collect_signals(self) -> Dict[str, Any]:

        signals = {
            "layer_stats": {},
            "global": {},
        }

        if self._model is None:
            return signals

        all_norms = []
        all_update_norms = []
        all_update_ratios = []
        all_norm_growths = []
        all_ranks = []

        for name, param in self._model.named_parameters():
            if not param.requires_grad:
                continue

            weight = param.detach()
            stats = {}

            weight_norm = torch.norm(weight, p=2).item()
            stats["weight_norm"] = weight_norm
            stats["weight_mean"] = weight.mean().item()
            stats["weight_std"] = weight.std().item()
            all_norms.append(weight_norm)

            if name in self._prev_weights:
                prev = self._prev_weights[name]
                if prev.shape == weight.shape:
                    # prev is cached on the host to keep checkpoints off the GPU,
                    # so it has to come back to the weight's device before the
                    # subtraction. Without this, every update-ratio computation
                    # raises on any CUDA model and the signal silently vanishes.
                    update = weight - prev.to(weight.device, non_blocking=True)
                    update_norm = torch.norm(update, p=2).item()
                    stats["weight_update_norm"] = update_norm
                    all_update_norms.append(update_norm)

                    if weight_norm > 1e-8:
                        update_ratio = update_norm / weight_norm
                        stats["weight_update_ratio"] = update_ratio
                        all_update_ratios.append(update_ratio)

            if name in self._prev_norms:
                prev_norm = self._prev_norms[name]
                if prev_norm > 1e-8:
                    norm_growth = (weight_norm - prev_norm) / prev_norm
                    stats["weight_norm_growth"] = norm_growth
                    all_norm_growths.append(norm_growth)

            if self.config.track_effective_rank and weight.dim() >= 2:
                rank = self._compute_effective_rank(weight)
                if rank is not None:
                    stats["effective_rank"] = rank
                    all_ranks.append(rank)

            signals["layer_stats"][name] = stats

            # Kept on the weight's own device. Round-tripping the whole model to
            # host memory once per collection cost more than the signal it feeds
            # (a single update-ratio scalar) and forced a sync every layer. The
            # price is one model-sized copy resident on the device, which is the
            # same order as a single checkpoint.
            self._prev_weights[name] = weight.clone()
            self._prev_norms[name] = weight_norm

        if all_norms:
            signals["global"]["total_weight_norm"] = np.sum(all_norms)
            signals["global"]["mean_weight_norm"] = np.mean(all_norms)
        if all_update_norms:
            signals["global"]["total_update_norm"] = np.sum(all_update_norms)
            signals["global"]["mean_update_norm"] = np.mean(all_update_norms)
        if all_update_ratios:
            signals["global"]["mean_update_ratio"] = np.mean(all_update_ratios)
            signals["global"]["max_update_ratio"] = np.max(all_update_ratios)
        if all_norm_growths:
            signals["global"]["mean_norm_growth"] = np.mean(all_norm_growths)
        if all_ranks:
            signals["global"]["mean_effective_rank"] = np.mean(all_ranks)
            signals["global"]["min_effective_rank"] = np.min(all_ranks)

        return signals

    def _compute_effective_rank(self, weight: torch.Tensor) -> Optional[float]:

        try:

            if weight.dim() > 2:
                weight = weight.flatten(start_dim=1)
            elif weight.dim() == 1:
                return None

            # Compute on whatever device the weight already lives on. Copying
            # every layer to the host and running the SVD on CPU made this the
            # single most expensive thing the collector did — it dominated total
            # instrumentation cost on a mid-size CNN, on top of forcing a device
            # sync per layer. cuSOLVER handles both paths below natively.
            weight_cpu = weight.float()

            m, n = weight_cpu.shape
            if m * n > 10000:

                k = min(self.config.effective_rank_sample_size, min(m, n))
                try:

                    U, S, Vh = torch.svd_lowrank(weight_cpu, q=k)
                    singular_values = S
                except Exception as e:
                    warnings.warn(f"svd_lowrank failed computing effective rank: {e}")
                    return None
            else:

                try:
                    U, S, Vh = torch.linalg.svd(weight_cpu, full_matrices=False)
                    singular_values = S
                except Exception as e:
                    warnings.warn(f"SVD failed computing effective rank: {e}")
                    return None

            singular_values = singular_values[singular_values > 1e-10]

            if len(singular_values) == 0:
                return 0.0

            probs = singular_values / singular_values.sum()

            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()

            effective_rank = np.exp(entropy)

            return effective_rank

        except Exception:
            return None

    def reset(self) -> None:

        super().reset()
        self._prev_weights.clear()
        self._prev_norms.clear()

    def _get_metadata(self) -> Dict[str, Any]:

        base = super()._get_metadata()
        base.update({
            "n_parameters": len(self._layer_names),
            "track_effective_rank": self.config.track_effective_rank,
        })
        return base
