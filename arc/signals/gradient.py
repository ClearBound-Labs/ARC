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
import math
import warnings
import torch
import torch.nn as nn
import numpy as np

from arc.signals.base import SignalCollector
from arc.config import SignalConfig

class GradientCollector(SignalCollector):

    def __init__(self, config: Optional[SignalConfig] = None):
        super().__init__(config)
        self.config = config or SignalConfig()
        self._gradient_data: Dict[str, torch.Tensor] = {}
        self._layer_names: List[str] = []
        self._layer_order: Dict[str, int] = {}

    def _register_hooks(self) -> None:

        if self._model is None:
            return

        # No backward hooks are registered.
        #
        # `register_full_backward_hook` raises on any module whose output is
        # later modified in place — which includes every `nn.ReLU(inplace=True)`,
        # and therefore torchvision's ResNet, VGG and MobileNet as shipped:
        #
        #   RuntimeError: Output 0 of BackwardHookFunctionBackward is a view
        #   and is being modified inplace.
        #
        # A monitoring collector must never be able to break the run it is
        # observing. The hook was only ever used to copy `param.grad` into
        # `_gradient_data`, and those gradients are still sitting on the
        # parameters after `backward()` returns, so they can simply be read at
        # collect time. Same numbers, no interference, and one less set of
        # autograd callbacks per step.
        layer_idx = 0
        for name, module in self._model.named_modules():
            if list(module.parameters(recurse=False)):
                self._layer_names.append(name or "root")
                self._layer_order[name or "root"] = layer_idx
                layer_idx += 1

    def _snapshot_gradients(self) -> None:
        """Read the current `param.grad` of every parameterised module."""
        if self._model is None:
            return
        for name, module in self._model.named_modules():
            layer_name = name or "root"
            if layer_name not in self._layer_order:
                continue
            grads = [
                param.grad.detach().flatten()
                for param in module.parameters(recurse=False)
                if param.grad is not None
            ]
            if grads:
                self._gradient_data[layer_name] = torch.cat(grads)

    def _collect_signals(self) -> Dict[str, Any]:

        signals = {
            "layer_stats": {},
            "global": {},
        }

        self._snapshot_gradients()

        if not self._gradient_data:
            return signals

        all_grads = []
        layer_norms = []

        for layer_name in self._layer_names:
            if layer_name not in self._gradient_data:
                continue

            grad = self._gradient_data[layer_name]
            all_grads.append(grad)

            l2_norm = torch.norm(grad, p=2).item()
            linf_norm = torch.norm(grad, p=float('inf')).item()
            mean_val = grad.mean().item()
            std_val = grad.std().item() if grad.numel() > 1 else 0.0

            layer_norms.append(l2_norm)

            signals["layer_stats"][layer_name] = {
                "grad_norm_l2": l2_norm,
                "grad_norm_linf": linf_norm,
                "grad_mean": mean_val,
                "grad_std": std_val,
            }

            if self.config.compute_gradient_entropy:
                entropy = self._compute_entropy(grad)
                signals["layer_stats"][layer_name]["grad_entropy"] = entropy

        if all_grads:
            all_grads_flat = torch.cat(all_grads)
            signals["global"]["total_grad_norm_l2"] = torch.norm(all_grads_flat, p=2).item()
            signals["global"]["total_grad_norm_linf"] = torch.norm(all_grads_flat, p=float('inf')).item()
            signals["global"]["total_grad_mean"] = all_grads_flat.mean().item()
            signals["global"]["total_grad_std"] = all_grads_flat.std().item()

            if self.config.compute_gradient_entropy:
                signals["global"]["total_grad_entropy"] = self._compute_entropy(all_grads_flat)

        if len(layer_norms) >= 4:
            quarter = len(layer_norms) // 4
            early_norm = np.mean(layer_norms[:quarter])
            late_norm = np.mean(layer_norms[-quarter:])

            if early_norm > 1e-10:
                signals["global"]["grad_flow_ratio"] = late_norm / early_norm
            else:
                signals["global"]["grad_flow_ratio"] = float('inf') if late_norm > 0 else 0.0

        self._gradient_data.clear()

        return signals

    def _compute_entropy(self, tensor: torch.Tensor) -> float:

        try:

            flat = tensor.flatten().cpu().float()

            if flat.numel() == 0:
                return 0.0

            flat = flat[torch.isfinite(flat)]
            if flat.numel() == 0:
                return 0.0

            n_bins = self.config.gradient_histogram_bins
            hist = torch.histc(flat, bins=n_bins)

            probs = hist / hist.sum()
            probs = probs[probs > 0]

            if probs.numel() == 0:
                return 0.0

            entropy = -torch.sum(probs * torch.log(probs)).item()

            max_entropy = math.log(n_bins)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

            return normalized_entropy

        except Exception as e:
            warnings.warn(f"Gradient entropy computation failed: {e}")
            return 0.0

    def _get_metadata(self) -> Dict[str, Any]:

        base = super()._get_metadata()
        base.update({
            "n_layers": len(self._layer_names),
            "layer_names": self._layer_names,
            "entropy_enabled": self.config.compute_gradient_entropy,
        })
        return base

    def reset(self) -> None:

        super().reset()
        self._gradient_data.clear()