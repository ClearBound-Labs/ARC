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

"""Numerical health fleet baseline, single-fleet/local degenerate
form: learn "normal" from an org's own prior runs (exported recovery
ledgers) and flag a new run whose summary stats fall outside that envelope.
The hosted cross-fleet version (aggregating across customers for a
population-level baseline) needs a real hosted service and real customer
telemetry; this is the part that's just local statistics.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


def summarize_ledger(ledger: Dict[str, Any]) -> Dict[str, float]:
    """Reduces an exported recovery ledger (parsed JSON dict) to the scalar
    summary stats a baseline is built from."""
    events = ledger.get("events", [])
    steps = max(ledger.get("total_steps", 0), 1)
    rollback_rate = ledger.get("total_rollbacks", 0) / steps

    stability_scores = [
        e["stability_score"] for e in events if e.get("stability_score") is not None
    ]
    loss_values = [e["loss_value"] for e in events if e.get("loss_value") is not None]

    return {
        "rollback_rate": rollback_rate,
        "mean_stability_score": float(np.mean(stability_scores)) if stability_scores else 0.0,
        "mean_loss_at_failure": float(np.mean(loss_values)) if loss_values else 0.0,
    }


@dataclass
class BaselineDeviation:
    metric: str
    value: float
    baseline_mean: float
    baseline_std: float
    zscore: float
    is_anomalous: bool


@dataclass
class BaselineReport:
    deviations: List[BaselineDeviation] = field(default_factory=list)

    @property
    def is_anomalous(self) -> bool:
        return any(d.is_anomalous for d in self.deviations)


class HealthBaseline:
    """Accumulates per-run summary stats and flags a new run that deviates
    from the learned mean by more than `zscore_threshold` standard
    deviations. Needs at least 2 prior runs per metric before it can judge
    anything (a single sample has no variance to compare against).
    """

    def __init__(self, zscore_threshold: float = 3.0):
        self.zscore_threshold = zscore_threshold
        self._samples: Dict[str, List[float]] = {}

    def add_run(self, ledger: Dict[str, Any]) -> None:
        for metric, value in summarize_ledger(ledger).items():
            self._samples.setdefault(metric, []).append(value)

    def check(self, ledger: Dict[str, Any]) -> BaselineReport:
        deviations = []
        for metric, value in summarize_ledger(ledger).items():
            history = self._samples.get(metric, [])
            if len(history) < 2:
                continue
            arr = np.asarray(history, dtype=np.float64)
            mean, std = float(arr.mean()), float(arr.std())
            if std > 1e-8:
                z = abs((value - mean) / std)
                is_anomalous = z > self.zscore_threshold
            else:
                # A perfectly consistent baseline has no variance to divide
                # by, but that doesn't mean nothing can be anomalous -- any
                # deviation from an always-identical history is suspicious.
                deviates = abs(value - mean) > 1e-8
                z = float("inf") if deviates else 0.0
                is_anomalous = deviates
            deviations.append(BaselineDeviation(
                metric=metric,
                value=value,
                baseline_mean=mean,
                baseline_std=std,
                zscore=z,
                is_anomalous=is_anomalous,
            ))
        return BaselineReport(deviations=deviations)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self._samples, f, indent=2)

    @classmethod
    def load(cls, path: str, zscore_threshold: float = 3.0) -> "HealthBaseline":
        baseline = cls(zscore_threshold=zscore_threshold)
        with open(path) as f:
            baseline._samples = json.load(f)
        return baseline
