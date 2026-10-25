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

"""MTTR / recovery-rate metrics: turns `Controller.intervention_log` into the
reliability numbers a renewal report or SLA claim needs -- recovery rate
overall and by failure type, rollback depth (steps of work a rollback throws
away), and mean time between failures (steps and wall-clock).

A failure counts as recovered only when a later ``recovery_confirmed`` event
(N healthy steps after it) covers it; a rollback alone proves nothing.
"""

import datetime
from dataclasses import dataclass
from typing import Any, Dict, List

FAILURE_ACTIONS = ("rollback", "skip")


@dataclass
class ReliabilityMetrics:
    total_failures: int
    total_recovered: int
    recovery_rate: float
    recovery_rate_by_reason: Dict[str, float]
    mean_steps_lost: float
    median_steps_lost: float
    mean_steps_between_failures: float
    mean_seconds_between_failures: float

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def compute_reliability_metrics(intervention_log: List[Dict[str, Any]]) -> ReliabilityMetrics:
    failures = [e for e in intervention_log if e.get("action") in FAILURE_ACTIONS]
    if not failures:
        return ReliabilityMetrics(
            total_failures=0, total_recovered=0, recovery_rate=0.0,
            recovery_rate_by_reason={}, mean_steps_lost=0.0,
            median_steps_lost=0.0, mean_steps_between_failures=0.0,
            mean_seconds_between_failures=0.0,
        )

    # Walk the log in order: a confirmation resolves every failure still pending.
    recovered_flags: List[bool] = []
    pending: List[int] = []
    for e in intervention_log:
        if e.get("action") in FAILURE_ACTIONS:
            pending.append(len(recovered_flags))
            recovered_flags.append(False)
        elif e.get("action") == "recovery_confirmed":
            for i in pending:
                recovered_flags[i] = True
            pending = []

    total_failures = len(failures)
    total_recovered = sum(recovered_flags)

    by_reason: Dict[str, List[bool]] = {}
    for e, recovered in zip(failures, recovered_flags):
        by_reason.setdefault(e.get("reason", "unknown"), []).append(recovered)
    recovery_rate_by_reason = {
        reason: sum(flags) / len(flags) for reason, flags in by_reason.items()
    }

    steps_lost = [
        e.get("steps_lost", e.get("steps_recovered", 0)) or 0
        for e in failures if e.get("action") == "rollback"
    ]
    if steps_lost:
        ordered = sorted(steps_lost)
        n = len(ordered)
        median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
        mean_steps_lost = sum(steps_lost) / n
        median_steps_lost = float(median)
    else:
        mean_steps_lost = 0.0
        median_steps_lost = 0.0

    steps = [e["step"] for e in failures if "step" in e]
    step_gaps = [b - a for a, b in zip(steps, steps[1:])]
    mean_steps_between_failures = sum(step_gaps) / len(step_gaps) if step_gaps else 0.0

    timestamps = []
    for e in failures:
        ts = e.get("timestamp")
        if ts:
            try:
                timestamps.append(datetime.datetime.fromisoformat(ts))
            except ValueError:
                pass
    time_gaps = [(b - a).total_seconds() for a, b in zip(timestamps, timestamps[1:])]
    mean_seconds_between_failures = sum(time_gaps) / len(time_gaps) if time_gaps else 0.0

    return ReliabilityMetrics(
        total_failures=total_failures,
        total_recovered=total_recovered,
        recovery_rate=total_recovered / total_failures,
        recovery_rate_by_reason=recovery_rate_by_reason,
        mean_steps_lost=mean_steps_lost,
        median_steps_lost=median_steps_lost,
        mean_steps_between_failures=mean_steps_between_failures,
        mean_seconds_between_failures=mean_seconds_between_failures,
    )
