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

"""Run comparison & regression detection: "this fine-tune is
drifting from the last known-good run." Operates entirely on the JSON
already produced by `Controller.export_recovery_ledger()` -- no live
comparison service, just a diff of two exported ledgers.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

from arc.signals.stats_utils import ols_slope


@dataclass
class RegressionFinding:
    metric: str
    baseline_value: float
    current_value: float
    delta: float
    is_regression: bool
    detail: str


@dataclass
class ComparisonReport:
    findings: List[RegressionFinding] = field(default_factory=list)

    @property
    def has_regression(self) -> bool:
        return any(f.is_regression for f in self.findings)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_regression": self.has_regression,
            "findings": [f.__dict__ for f in self.findings],
        }


def _rollback_rate(ledger: Dict[str, Any]) -> float:
    steps = max(ledger.get("total_steps", 0), 1)
    return ledger.get("total_rollbacks", 0) / steps


def _stability_slope(ledger: Dict[str, Any]) -> float:
    scores = [
        e["stability_score"] for e in ledger.get("events", [])
        if e.get("stability_score") is not None
    ]
    return ols_slope(scores)


def _failure_counts(ledger: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for e in ledger.get("events", []):
        if e.get("action") == "recovery_confirmed":
            continue
        reason = e.get("reason", "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


class RunComparator:
    """Diffs two exported recovery ledgers and flags metrics that moved in
    the wrong direction beyond a tolerance. Ledgers are the parsed JSON
    from `Controller.export_recovery_ledger()` (a dict, not the string).
    """

    def __init__(
        self,
        rollback_rate_tolerance: float = 0.5,
        stability_slope_tolerance: float = 0.01,
        new_failure_mode_is_regression: bool = True,
    ):
        self.rollback_rate_tolerance = rollback_rate_tolerance
        self.stability_slope_tolerance = stability_slope_tolerance
        self.new_failure_mode_is_regression = new_failure_mode_is_regression

    def compare(self, baseline: Dict[str, Any], current: Dict[str, Any]) -> ComparisonReport:
        findings: List[RegressionFinding] = []

        b_rate = _rollback_rate(baseline)
        c_rate = _rollback_rate(current)
        rate_regression = c_rate > b_rate * (1.0 + self.rollback_rate_tolerance) + 1e-9
        findings.append(RegressionFinding(
            metric="rollback_rate",
            baseline_value=b_rate,
            current_value=c_rate,
            delta=c_rate - b_rate,
            is_regression=rate_regression,
            detail=(
                f"rollback rate {c_rate:.4f} vs baseline {b_rate:.4f} "
                f"({'regressed' if rate_regression else 'within tolerance'})"
            ),
        ))

        b_slope = _stability_slope(baseline)
        c_slope = _stability_slope(current)
        slope_regression = (b_slope - c_slope) > self.stability_slope_tolerance
        findings.append(RegressionFinding(
            metric="stability_score_slope",
            baseline_value=b_slope,
            current_value=c_slope,
            delta=c_slope - b_slope,
            is_regression=slope_regression,
            detail=(
                f"stability trend slope {c_slope:.4f} vs baseline {b_slope:.4f} "
                f"({'regressed' if slope_regression else 'within tolerance'})"
            ),
        ))

        if self.new_failure_mode_is_regression:
            b_failures = set(_failure_counts(baseline))
            c_failures = _failure_counts(current)
            new_modes = set(c_failures) - b_failures
            findings.append(RegressionFinding(
                metric="new_failure_modes",
                baseline_value=0.0,
                current_value=float(len(new_modes)),
                delta=float(len(new_modes)),
                is_regression=len(new_modes) > 0,
                detail=(
                    f"new failure mode(s) not seen in baseline: {sorted(new_modes)}"
                    if new_modes else "no new failure modes"
                ),
            ))

        return ComparisonReport(findings=findings)

    def compare_files(self, baseline_path: str, current_path: str) -> ComparisonReport:
        with open(baseline_path) as f:
            baseline = json.load(f)
        with open(current_path) as f:
            current = json.load(f)
        return self.compare(baseline, current)
