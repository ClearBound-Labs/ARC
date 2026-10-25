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

"""Rollout / generation pipeline health: "detect degenerate
generations, mode collapse, and format drift in the sampling stage." Pure
sequence statistics over a batch of already-generated token sequences (or
any hashable-element sequence) -- no reward-model call, no live rollout
service.
"""

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np


@dataclass
class RolloutHealthReport:
    mode_collapse: bool = False
    high_repetition: bool = False
    format_drift: bool = False

    unique_ratio: float = 1.0
    max_ngram_repetition_rate: float = 0.0
    length_zscore: float = 0.0

    messages: List[str] = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        return not (self.mode_collapse or self.high_repetition or self.format_drift)


def _ngram_repetition_rate(sequence: Sequence[Any], n: int) -> float:
    if len(sequence) < n:
        return 0.0
    ngrams = [tuple(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(c for c in counts.values() if c > 1)
    return repeated / len(ngrams)


class RolloutHealthMonitor:
    def __init__(
        self,
        ngram_n: int = 3,
        repetition_threshold: float = 0.3,
        unique_ratio_threshold: float = 0.3,
        length_window: int = 50,
        length_zscore_threshold: float = 4.0,
    ):
        self.ngram_n = ngram_n
        self.repetition_threshold = repetition_threshold
        self.unique_ratio_threshold = unique_ratio_threshold
        self.length_zscore_threshold = length_zscore_threshold
        self._length_history: deque = deque(maxlen=length_window)

    def check_batch(self, sequences: List[Sequence[Any]]) -> RolloutHealthReport:
        if not sequences:
            return RolloutHealthReport()

        report = RolloutHealthReport()

        unique_count = len(set(tuple(s) for s in sequences))
        report.unique_ratio = unique_count / len(sequences)
        if report.unique_ratio < self.unique_ratio_threshold:
            report.mode_collapse = True
            report.messages.append(
                f"unique sequence ratio {report.unique_ratio:.2f} below "
                f"threshold {self.unique_ratio_threshold}"
            )

        rates = [_ngram_repetition_rate(s, self.ngram_n) for s in sequences]
        report.max_ngram_repetition_rate = max(rates) if rates else 0.0
        if report.max_ngram_repetition_rate > self.repetition_threshold:
            report.high_repetition = True
            report.messages.append(
                f"{self.ngram_n}-gram repetition rate {report.max_ngram_repetition_rate:.2f} "
                f"exceeds threshold {self.repetition_threshold}"
            )

        lengths = [len(s) for s in sequences]
        mean_length = float(np.mean(lengths))
        history = list(self._length_history)
        if len(history) >= 5:
            arr = np.asarray(history, dtype=np.float64)
            mean, std = arr.mean(), arr.std()
            if std > 1e-8:
                z = abs((mean_length - mean) / std)
                drifted = z > self.length_zscore_threshold
            else:
                # A perfectly stable history has no variance to divide by,
                # but any deviation from an always-identical length is
                # still worth flagging as drift.
                drifted = abs(mean_length - mean) > 1e-8
                z = float("inf") if drifted else 0.0
            report.length_zscore = float(z)
            if drifted:
                report.format_drift = True
                report.messages.append(
                    f"batch mean length z-score {z:.2f} exceeds threshold "
                    f"{self.length_zscore_threshold}"
                )
        self._length_history.append(mean_length)

        return report
