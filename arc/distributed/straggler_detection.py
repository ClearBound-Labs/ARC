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

"""Straggler / slow-node detection: in synchronous data-parallel training
a single degraded-but-not-dead GPU drags every rank's step time down to
its pace. This is the pure statistics half -- flag the rank whose step
times are a persistent, resistant-to-noise outlier across the group.
The actual evict/reshard/cordon-drain-replace action requires a live
elastic-training runtime and orchestrator to act on, which needs a real
cluster to build and validate; that part is unbuilt.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class StragglerReport:
    has_straggler: bool
    straggler_ranks: List[int]
    per_rank_zscore: Dict[int, float]
    detail: str = ""


class StragglerDetector:
    """Tracks per-rank step-time history and flags a rank that is
    persistently slow relative to the group, using the median absolute
    deviation (MAD) rather than mean/std so a single already-slow rank
    doesn't inflate the group's own noise estimate and mask itself.
    """

    def __init__(
        self,
        window: int = 20,
        zscore_threshold: float = 4.0,
        min_consecutive_flags: int = 3,
    ):
        self.zscore_threshold = zscore_threshold
        self.min_consecutive_flags = min_consecutive_flags
        self._history: Dict[int, deque] = {}
        self._window = window
        self._consecutive_flags: Dict[int, int] = {}

    def record_step(self, rank_step_times: Dict[int, float]) -> StragglerReport:
        if len(rank_step_times) < 3:
            raise ValueError(
                "Straggler detection needs >= 3 ranks reporting to form a "
                "robust group baseline."
            )

        # Drop tracked state for any rank absent from this call -- an
        # elastic runtime evicts/replaces ranks, so a rank id that stops
        # reporting has left the group, and a *new* rank reusing that id
        # later must start with a clean slate rather than inheriting a
        # departed rank's flag count.
        for rank in list(self._history):
            if rank not in rank_step_times:
                del self._history[rank]
                del self._consecutive_flags[rank]

        for rank, t in rank_step_times.items():
            if rank not in self._history:
                self._history[rank] = deque(maxlen=self._window)
                self._consecutive_flags[rank] = 0
            self._history[rank].append(t)

        latest = np.array([rank_step_times[r] for r in rank_step_times], dtype=np.float64)
        median = np.median(latest)
        mad = np.median(np.abs(latest - median))
        # 1.4826 makes MAD a consistent estimator of std under normality,
        # the standard robust-z-score scaling factor.
        scale = max(mad * 1.4826, 1e-8)

        per_rank_zscore: Dict[int, float] = {}
        newly_flagged: List[int] = []
        for rank, t in rank_step_times.items():
            z = (t - median) / scale
            per_rank_zscore[rank] = float(z)
            if z > self.zscore_threshold:
                self._consecutive_flags[rank] += 1
            else:
                self._consecutive_flags[rank] = 0
            if self._consecutive_flags[rank] >= self.min_consecutive_flags:
                newly_flagged.append(rank)

        has_straggler = len(newly_flagged) > 0
        detail = (
            f"rank(s) {newly_flagged} step-time z-score exceeds "
            f"{self.zscore_threshold} for {self.min_consecutive_flags}+ consecutive steps"
            if has_straggler else "all ranks within normal step-time variance"
        )

        return StragglerReport(
            has_straggler=has_straggler,
            straggler_ranks=newly_flagged,
            per_rank_zscore=per_rank_zscore,
            detail=detail,
        )
