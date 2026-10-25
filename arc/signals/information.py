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
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from arc.signals.base import SignalCollector, SignalSnapshot
from arc.config import SignalConfig

class InformationDynamicsCollector(SignalCollector):

    def __init__(
        self,
        config: SignalConfig,
        num_classes: int = 10,
        num_bins: int = 30,
        sample_size: int = 1000,
    ):

        super().__init__(config)
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.sample_size = sample_size

        self._layer_activations: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self._labels: List[int] = []

        self._mi_input_hidden: Dict[str, float] = {}
        self._mi_hidden_output: Dict[str, float] = {}
        self._compression_ratio: float = 0.0
        self._prediction_ratio: float = 0.0

        self._hooks: List[torch.utils.hooks.RemovableHandle] = []

        self._info_plane_history: List[Dict[str, Tuple[float, float]]] = []

    @property
    def name(self) -> str:
        return "information"

    def attach(self, model: nn.Module, optimizer: Any) -> None:

        super().attach(model, optimizer)

        def make_hook(layer_name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self._capture_activation(layer_name, output)
            return hook

        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.GELU)):
                handle = module.register_forward_hook(make_hook(name))
                self._hooks.append(handle)

    def _capture_activation(self, layer_name: str, activation: torch.Tensor) -> None:

        flat = activation.detach().view(activation.size(0), -1)

        if len(self._layer_activations[layer_name]) < self.sample_size:
            self._layer_activations[layer_name].append(flat.cpu())

    def update_labels(self, labels: torch.Tensor) -> None:

        for label in labels.detach().cpu().tolist():
            if len(self._labels) < self.sample_size:
                self._labels.append(int(label))

    def compute_information_metrics(self) -> Dict[str, Any]:

        if not self._layer_activations or not self._labels:
            return {}

        results = {
            "per_layer_mi_x_t": {},
            "per_layer_mi_t_y": {},
        }

        labels = np.array(self._labels[:self.sample_size])

        for layer_name, acts in self._layer_activations.items():
            if not acts:
                continue

            all_acts = torch.cat(acts, dim=0)[:len(labels)]

            if len(all_acts) < 100:
                continue

            mi_x_t = self._estimate_mi_input(all_acts)
            self._mi_input_hidden[layer_name] = mi_x_t
            results["per_layer_mi_x_t"][layer_name] = mi_x_t

            mi_t_y = self._estimate_mi_output(all_acts.numpy(), labels)
            self._mi_hidden_output[layer_name] = mi_t_y
            results["per_layer_mi_t_y"][layer_name] = mi_t_y

        if self._mi_input_hidden:
            max_mi_x_t = max(self._mi_input_hidden.values())
            min_mi_x_t = min(self._mi_input_hidden.values())

            self._compression_ratio = min_mi_x_t / (max_mi_x_t + 1e-10)
            results["compression_ratio"] = self._compression_ratio

        if self._mi_hidden_output:
            max_mi_t_y = max(self._mi_hidden_output.values())
            results["max_prediction_info"] = max_mi_t_y

        info_point = {
            layer: (self._mi_input_hidden.get(layer, 0), self._mi_hidden_output.get(layer, 0))
            for layer in self._layer_activations.keys()
        }
        self._info_plane_history.append(info_point)

        return results

    def _estimate_mi_input(self, activations: torch.Tensor) -> float:

        act = activations[:, :min(100, activations.size(1))].numpy()

        n_samples, n_dims = act.shape

        total_entropy = 0.0

        for d in range(n_dims):
            col = act[:, d]

            hist, edges = np.histogram(col, bins=self.num_bins, density=True)
            hist = hist + 1e-10
            hist = hist / hist.sum()

            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            total_entropy += entropy

        return float(total_entropy / (n_dims + 1e-10))

    def _estimate_mi_output(
        self,
        activations: np.ndarray,
        labels: np.ndarray
    ) -> float:

        unique, counts = np.unique(labels, return_counts=True)
        p_y = counts / len(labels)
        h_y = -np.sum(p_y * np.log2(p_y + 1e-10))

        from scipy.cluster.vq import kmeans, vq

        n_clusters = min(self.num_bins, len(labels) // 10)
        if n_clusters < 2 or len(activations) < n_clusters:
            return float(h_y)

        try:

            act_reduced = activations[:, :min(50, activations.shape[1])]

            centroids, _ = kmeans(act_reduced.astype(np.float64), n_clusters)
            cluster_ids, _ = vq(act_reduced.astype(np.float64), centroids)

            h_y_given_t = 0.0

            for c in range(n_clusters):
                mask = cluster_ids == c
                if mask.sum() == 0:
                    continue

                cluster_labels = labels[mask]
                p_cluster = mask.sum() / len(labels)

                unique_c, counts_c = np.unique(cluster_labels, return_counts=True)
                p_y_c = counts_c / len(cluster_labels)
                h_c = -np.sum(p_y_c * np.log2(p_y_c + 1e-10))

                h_y_given_t += p_cluster * h_c

            mi = h_y - h_y_given_t
            return float(max(0, mi))

        except Exception as e:
            warnings.warn(f"Mutual information computation failed, using fallback estimate: {e}")
            return float(h_y * 0.5)

    def get_information_plane_trajectory(self) -> List[Dict[str, Tuple[float, float]]]:

        return self._info_plane_history

    def detect_compression_phase(self) -> bool:

        if len(self._info_plane_history) < 5:
            return False

        recent = self._info_plane_history[-5:]

        mi_x_t_trend = []
        for point in recent:
            avg_mi = np.mean([v[0] for v in point.values()])
            mi_x_t_trend.append(avg_mi)

        if len(mi_x_t_trend) >= 3:
            diffs = np.diff(mi_x_t_trend)
            if np.mean(diffs) < -0.1:
                return True

        return False

    def _register_hooks(self) -> None:
        # No-op: attach() is overridden above to register this collector's
        # own forward hooks directly (activation capture needs layer-name
        # closures), bypassing the base class's default hook registration.
        pass

    def _collect_signals(self) -> Dict[str, Any]:
        return self.compute_information_metrics()

    def collect(self) -> SignalSnapshot:

        metrics = self.compute_information_metrics()

        signals = {
            "information": {
                "compression_ratio": self._compression_ratio,
                "in_compression_phase": self.detect_compression_phase(),
                "per_layer_mi_x_t": self._mi_input_hidden,
                "per_layer_mi_t_y": self._mi_hidden_output,
            }
        }

        if self._mi_input_hidden:
            signals["information"]["mean_mi_x_t"] = np.mean(list(self._mi_input_hidden.values()))

        if self._mi_hidden_output:
            signals["information"]["mean_mi_t_y"] = np.mean(list(self._mi_hidden_output.values()))

        return SignalSnapshot(
            step=self._step,
            epoch=self._epoch,
            timestamp=time.time(),
            signals=signals,
            metadata=self._get_metadata(),
        )

    def reset(self) -> None:

        super().reset()
        self._layer_activations.clear()
        self._labels.clear()
        self._mi_input_hidden.clear()
        self._mi_hidden_output.clear()

    def detach(self) -> None:

        super().detach()
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()