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

import time
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from arc.signals.base import SignalCollector, SignalSnapshot
from arc.config import SignalConfig

class NeuralCollapseCollector(SignalCollector):

    def __init__(
        self,
        config: SignalConfig,
        num_classes: int = 10,
        feature_dim: Optional[int] = None,
    ):

        super().__init__(config)
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        self._class_features: Dict[int, List[torch.Tensor]] = {
            i: [] for i in range(num_classes)
        }
        self._class_seen: Dict[int, int] = {i: 0 for i in range(num_classes)}
        self._class_means: Optional[torch.Tensor] = None
        self._global_mean: Optional[torch.Tensor] = None
        self._classifier_weights: Optional[torch.Tensor] = None

        self._feature_hook: Optional[torch.utils.hooks.RemovableHandle] = None
        self._last_features: Optional[torch.Tensor] = None

    @property
    def name(self) -> str:
        return "neural_collapse"

    def attach(self, model: nn.Module, optimizer: Any) -> None:

        super().attach(model, optimizer)

        modules = list(model.modules())

        for i, module in enumerate(reversed(modules)):
            if isinstance(module, nn.Linear):

                if i + 1 < len(modules):
                    penultimate = modules[-(i+2)]
                    self._feature_hook = penultimate.register_forward_hook(
                        self._capture_features
                    )

                self._classifier_weights = module.weight.detach().clone()
                self.feature_dim = module.in_features
                break

    def _capture_features(
        self,
        module: nn.Module,
        input: Tuple,
        output: torch.Tensor
    ) -> None:

        if isinstance(output, torch.Tensor):

            if output.dim() > 2:
                output = output.view(output.size(0), -1)
            self._last_features = output.detach()

    def update_batch(
        self,
        features: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> None:

        if features is None:
            features = self._last_features

        if features is None or labels is None:
            return

        features = features.detach().cpu()
        labels = labels.detach().cpu()

        for i in range(len(labels)):
            c = int(labels[i].item())
            if c < self.num_classes:

                self._class_seen[c] += 1
                buf = self._class_features[c]

                if len(buf) < 1000:
                    buf.append(features[i])
                else:
                    # Algorithm R reservoir sampling: the k-th item seen so
                    # far (1-indexed) replaces a uniformly random slot with
                    # probability capacity/k, giving each seen item equal
                    # probability of ending up in the final sample.
                    j = np.random.randint(0, self._class_seen[c])
                    if j < len(buf):
                        buf[j] = features[i]

    def collect(self) -> SignalSnapshot:

        signals = {"nc": {}}

        self._compute_class_means()

        if self._class_means is None or self._global_mean is None:
            return self._snapshot(signals)

        nc1 = self._compute_nc1()
        signals["nc"]["NC1_within_class_var"] = nc1

        nc2 = self._compute_nc2()
        signals["nc"]["NC2_simplex_etf_dev"] = nc2

        nc3 = self._compute_nc3()
        signals["nc"]["NC3_self_duality"] = nc3

        nc4 = self._compute_nc4()
        signals["nc"]["NC4_ncc_agreement"] = nc4

        signals["nc"]["collapse_score"] = self._compute_collapse_score(nc1, nc2, nc3, nc4)

        return self._snapshot(signals)

    def _register_hooks(self) -> None:
        # attach() registers the penultimate-layer feature hook itself.
        pass

    def _collect_signals(self) -> Dict[str, Any]:
        return self.collect().signals

    def _snapshot(self, signals: Dict[str, Any]) -> SignalSnapshot:
        return SignalSnapshot(step=self._step, epoch=self._epoch, timestamp=time.time(),
                              signals=signals, metadata=self._get_metadata())

    def _compute_class_means(self) -> None:

        means = []
        counts = []

        for c in range(self.num_classes):
            if self._class_features[c]:
                class_tensor = torch.stack(self._class_features[c])
                means.append(class_tensor.mean(dim=0))
                counts.append(len(self._class_features[c]))
            else:
                means.append(None)
                counts.append(0)

        valid_means = [m for m in means if m is not None]

        if len(valid_means) < 2:
            self._class_means = None
            self._global_mean = None
            return

        self._class_means = torch.stack(valid_means)

        total = sum(counts)
        if total > 0:
            weighted_sum = sum(
                m * c for m, c in zip(valid_means, [counts[i] for i, m in enumerate(means) if m is not None])
            )
            self._global_mean = weighted_sum / total

    def _compute_nc1(self) -> float:

        if self._class_means is None or self._global_mean is None:
            return 1.0

        within_trace = 0.0
        count = 0

        for c in range(self.num_classes):
            if self._class_features[c] and len(self._class_features[c]) > 1:
                class_tensor = torch.stack(self._class_features[c])
                class_mean = class_tensor.mean(dim=0)

                centered = class_tensor - class_mean
                var = (centered ** 2).sum() / len(class_tensor)
                within_trace += var.item()
                count += 1

        if count > 0:
            within_trace /= count

        between_trace = 0.0
        centered_means = self._class_means - self._global_mean
        between_trace = (centered_means ** 2).sum().item() / len(self._class_means)

        nc1 = within_trace / max(between_trace, 1e-10)

        return float(nc1)

    def _compute_nc2(self) -> float:

        if self._class_means is None or len(self._class_means) < 2:
            return 1.0

        centered = self._class_means - self._global_mean
        norms = centered.norm(dim=1, keepdim=True)
        normalized = centered / (norms + 1e-10)

        gram = normalized @ normalized.T

        C = len(self._class_means)
        ideal_off_diag = -1.0 / (C - 1)

        mask = ~torch.eye(C, dtype=torch.bool)
        off_diag = gram[mask]

        deviation = ((off_diag - ideal_off_diag) ** 2).mean().sqrt()

        return float(deviation.item())

    def _compute_nc3(self) -> float:

        if self._class_means is None or self._classifier_weights is None:
            return 0.0

        centered_means = self._class_means - self._global_mean

        mean_norms = centered_means.norm(dim=1, keepdim=True)
        normalized_means = centered_means / (mean_norms + 1e-10)

        weights = self._classifier_weights.detach().cpu()
        weight_norms = weights.norm(dim=1, keepdim=True)
        normalized_weights = weights / (weight_norms + 1e-10)

        min_classes = min(len(normalized_means), len(normalized_weights))

        alignment = (normalized_means[:min_classes] * normalized_weights[:min_classes]).sum(dim=1)

        return float(alignment.mean().item())

    def _compute_nc4(self) -> float:

        if self._class_means is None:
            return 0.0

        correct = 0
        total = 0

        for c in range(self.num_classes):
            if self._class_features[c] and c < len(self._class_means):
                for feat in self._class_features[c][:100]:

                    distances = ((self._class_means - feat) ** 2).sum(dim=1)
                    nearest = distances.argmin().item()

                    if nearest == c:
                        correct += 1
                    total += 1

        return correct / max(total, 1)

    def _compute_collapse_score(
        self, nc1: float, nc2: float, nc3: float, nc4: float
    ) -> float:

        nc1_score = 1.0 / (1.0 + nc1)

        nc2_score = 1.0 / (1.0 + nc2 * 10)

        nc3_score = max(0, nc3)

        nc4_score = nc4

        score = 0.25 * nc1_score + 0.25 * nc2_score + 0.25 * nc3_score + 0.25 * nc4_score

        return float(score)

    def reset(self) -> None:

        super().reset()
        self._class_features = {i: [] for i in range(self.num_classes)}
        self._class_seen = {i: 0 for i in range(self.num_classes)}
        self._class_means = None
        self._global_mean = None

    def detach(self) -> None:

        super().detach()
        if self._feature_hook is not None:
            self._feature_hook.remove()
            self._feature_hook = None
