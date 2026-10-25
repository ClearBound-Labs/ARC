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

"""
Trend-based drift detection for gradual training degradation.

Addresses the key limitation from the paper (Section 5, Limitation 1):
"ARC does not detect gradual numerical drift."

This module monitors slopes of moving averages for loss, gradient norm,
and weight magnitude. A consistently positive slope over a configurable
window triggers a WARNING or FAILURE signal.
"""

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List

import numpy as np

from arc.signals.stats_utils import ols_slope


class DriftSeverity(Enum):
    NONE = "none"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DriftReport:
    """Result of drift analysis across all tracked metrics."""
    severity: DriftSeverity = DriftSeverity.NONE
    loss_slope: float = 0.0
    grad_norm_slope: float = 0.0
    weight_norm_slope: float = 0.0
    loss_drifting: bool = False
    grad_drifting: bool = False
    weight_drifting: bool = False
    n_signals_drifting: int = 0
    confidence: float = 0.0
    message: str = ""


class TrendDetector:
    """
    Detects gradual degradation trends in training signals.

    Uses ordinary least squares linear regression on a sliding window
    of metric values. When the slope exceeds configurable thresholds
    consistently, a drift warning or failure is raised.

    This catches failure modes invisible to threshold-based detectors:
    - Loss creeping upward by +0.01 per step (no single spike)
    - Gradient norms slowly growing (pre-explosion drift)
    - Weight magnitudes gradually diverging
    """

    def __init__(
        self,
        window: int = 50,
        loss_slope_warning: float = 0.01,
        loss_slope_critical: float = 0.05,
        grad_slope_warning: float = 0.1,
        grad_slope_critical: float = 0.5,
        weight_slope_warning: float = 0.05,
        weight_slope_critical: float = 0.2,
        min_samples: int = 10,
        normalise: bool = True,
    ):
        self.window = window
        self.min_samples = min_samples
        self.normalise = normalise

        # Thresholds (positive slope = bad)
        self._thresholds = {
            "loss": (loss_slope_warning, loss_slope_critical),
            "grad_norm": (grad_slope_warning, grad_slope_critical),
            "weight_norm": (weight_slope_warning, weight_slope_critical),
        }

        # Buffers
        self._loss_history: deque = deque(maxlen=window)
        self._grad_norm_history: deque = deque(maxlen=window)
        self._weight_norm_history: deque = deque(maxlen=window)

        # Normalisation baselines (set from first few observations)
        self._loss_baseline: Optional[float] = None
        self._grad_baseline: Optional[float] = None
        self._weight_baseline: Optional[float] = None

    def update(
        self,
        loss: Optional[float] = None,
        grad_norm: Optional[float] = None,
        weight_norm: Optional[float] = None,
    ) -> DriftReport:
        """
        Feed new metric values and return drift analysis.

        Pass only the metrics you have available. Missing metrics
        are simply not tracked.
        """
        if loss is not None and math.isfinite(loss):
            if self._loss_baseline is None:
                self._loss_baseline = max(abs(loss), 1e-8)
            self._loss_history.append(loss)

        if grad_norm is not None and math.isfinite(grad_norm):
            if self._grad_baseline is None:
                self._grad_baseline = max(abs(grad_norm), 1e-8)
            self._grad_norm_history.append(grad_norm)

        if weight_norm is not None and math.isfinite(weight_norm):
            if self._weight_baseline is None:
                self._weight_baseline = max(abs(weight_norm), 1e-8)
            self._weight_norm_history.append(weight_norm)

        return self._analyse()

    def _compute_slope(self, values: deque, baseline: Optional[float]) -> float:
        """
        Compute normalised linear regression slope.

        Returns the slope of a least-squares linear fit, optionally
        normalised by the baseline value so that slopes are comparable
        across metrics with different scales.
        """
        if len(values) < self.min_samples:
            return 0.0

        slope = ols_slope(list(values))

        # Normalise by baseline
        if self.normalise and baseline is not None and baseline > 1e-12:
            slope = slope / baseline

        return float(slope)

    def _analyse(self) -> DriftReport:
        """Run drift analysis across all tracked metrics."""
        report = DriftReport()

        # Compute slopes
        report.loss_slope = self._compute_slope(
            self._loss_history, self._loss_baseline
        )
        report.grad_norm_slope = self._compute_slope(
            self._grad_norm_history, self._grad_baseline
        )
        report.weight_norm_slope = self._compute_slope(
            self._weight_norm_history, self._weight_baseline
        )

        # Check each metric against thresholds
        warn_loss, crit_loss = self._thresholds["loss"]
        warn_grad, crit_grad = self._thresholds["grad_norm"]
        warn_weight, crit_weight = self._thresholds["weight_norm"]

        report.loss_drifting = report.loss_slope > warn_loss
        report.grad_drifting = report.grad_norm_slope > warn_grad
        report.weight_drifting = report.weight_norm_slope > warn_weight

        report.n_signals_drifting = sum([
            report.loss_drifting,
            report.grad_drifting,
            report.weight_drifting,
        ])

        # Confidence: how full is the window?
        max_n = max(
            len(self._loss_history),
            len(self._grad_norm_history),
            len(self._weight_norm_history),
            1,
        )
        report.confidence = min(1.0, max_n / self.window)

        # Determine severity
        any_critical = (
            report.loss_slope > crit_loss or
            report.grad_norm_slope > crit_grad or
            report.weight_norm_slope > crit_weight
        )

        if any_critical and report.confidence > 0.5:
            report.severity = DriftSeverity.CRITICAL
            report.message = self._build_message(report, "CRITICAL")
        elif report.n_signals_drifting >= 2 and report.confidence > 0.5:
            report.severity = DriftSeverity.CRITICAL
            report.message = self._build_message(report, "MULTI-SIGNAL DRIFT")
        elif report.n_signals_drifting >= 1 and report.confidence > 0.3:
            report.severity = DriftSeverity.WARNING
            report.message = self._build_message(report, "WARNING")

        return report

    def _build_message(self, report: DriftReport, prefix: str) -> str:
        parts = [prefix + ":"]
        if report.loss_drifting:
            parts.append(f"loss↑{report.loss_slope:.4f}")
        if report.grad_drifting:
            parts.append(f"grad↑{report.grad_norm_slope:.4f}")
        if report.weight_drifting:
            parts.append(f"weight↑{report.weight_norm_slope:.4f}")
        return " ".join(parts)

    def reset(self):
        """Reset all state (e.g. after rollback)."""
        self._loss_history.clear()
        self._grad_norm_history.clear()
        self._weight_norm_history.clear()
        self._loss_baseline = None
        self._grad_baseline = None
        self._weight_baseline = None

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return full diagnostic state for logging."""
        return {
            "loss_slope": self._compute_slope(self._loss_history, self._loss_baseline),
            "grad_norm_slope": self._compute_slope(self._grad_norm_history, self._grad_baseline),
            "weight_norm_slope": self._compute_slope(self._weight_norm_history, self._weight_baseline),
            "loss_n": len(self._loss_history),
            "grad_norm_n": len(self._grad_norm_history),
            "weight_norm_n": len(self._weight_norm_history),
        }
