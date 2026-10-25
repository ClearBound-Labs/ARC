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

"""Deterministic replay / reproducibility report: wraps
`DeterministicReplayVerifier` into a per-run summary -- sampled-cadence
replay checks across a training run, aggregated into a "this run was N%
bit-reproducible" artifact. Pure
local computation, reusing the SDC replay verifier already built.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import torch

from arc.security.sdc_detection import DeterministicReplayVerifier, SDCAlert


@dataclass
class ReproducibilityReport:
    total_checks: int
    reproducible_checks: int
    reproducible_ratio: float
    max_relative_diff: float
    detail: str


class ReproducibilityTracker:
    """Call `maybe_check(model, inputs)` on a cadence during training (e.g.
    every N steps); it samples every `check_every_n_calls`-th call and
    accumulates the results into a run-level `report()`.
    """

    def __init__(
        self,
        verifier: Optional[DeterministicReplayVerifier] = None,
        check_every_n_calls: int = 1,
    ):
        self.verifier = verifier or DeterministicReplayVerifier()
        self.check_every_n_calls = max(1, check_every_n_calls)
        self._call_count = 0
        self._results: List[SDCAlert] = []

    def maybe_check(self, model: torch.nn.Module, inputs: torch.Tensor) -> Optional[SDCAlert]:
        self._call_count += 1
        if self._call_count % self.check_every_n_calls != 0:
            return None
        alert = self.verifier.verify(model, inputs)
        self._results.append(alert)
        return alert

    def report(self) -> ReproducibilityReport:
        total = len(self._results)
        if total == 0:
            return ReproducibilityReport(
                total_checks=0, reproducible_checks=0, reproducible_ratio=1.0,
                max_relative_diff=0.0, detail="no sampled checks ran yet",
            )
        reproducible = sum(1 for a in self._results if not a.is_corrupted)
        ratio = reproducible / total
        max_diff = max(a.max_relative_deviation for a in self._results)
        return ReproducibilityReport(
            total_checks=total,
            reproducible_checks=reproducible,
            reproducible_ratio=ratio,
            max_relative_diff=max_diff,
            detail=f"{ratio:.1%} bit-reproducible over {total} sampled replay checks",
        )
