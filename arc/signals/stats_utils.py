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

"""Small shared statistics helpers used by more than one signal collector.

Kept deliberately tiny: this is not a home for general-purpose stats code,
only for logic that was previously copy-pasted between collectors.
"""

from typing import Sequence
import numpy as np


def ols_slope(values: Sequence[float]) -> float:
    """Ordinary-least-squares slope of `values` against their index.

    Returns 0.0 for fewer than 2 points or a degenerate (zero-variance x)
    fit. Used by both `signals.trend.TrendDetector` (which additionally
    normalises by a baseline) and `signals.silent_detector.MetricTracker`
    (which uses the raw slope directly).
    """
    n = len(values)
    if n < 2:
        return 0.0

    y = np.asarray(values, dtype=np.float64)
    x = np.arange(n, dtype=np.float64)

    sx = x.sum()
    sy = y.sum()
    sxx = (x * x).sum()
    sxy = (x * y).sum()

    denom = n * sxx - sx * sx
    if abs(denom) < 1e-12:
        return 0.0

    return float((n * sxy - sx * sy) / denom)
