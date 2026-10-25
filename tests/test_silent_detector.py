"""Smoke tests for arc.signals.silent_detector.

Previously this module had zero test coverage, which blocked any safe
refactoring of it (see issues-found.md #47). These cover the basics of
MetricTracker and SilentCrashDetector so a future split/dedup has a
safety net to catch regressions.
"""

import numpy as np
import pytest

from arc.signals.silent_detector import (
    MetricTracker,
    SilentCrashDetector,
    SilentDetectorConfig,
    SilentFailureType,
)
from arc.signals.stats_utils import ols_slope


def test_ols_slope_matches_numpy_polyfit():
    values = [1.0, 2.1, 2.9, 4.2, 5.0]
    expected = np.polyfit(np.arange(len(values)), values, 1)[0]
    assert ols_slope(values) == pytest.approx(expected, abs=1e-9)


def test_ols_slope_degenerate_cases():
    assert ols_slope([]) == 0.0
    assert ols_slope([5.0]) == 0.0


def test_metric_tracker_mean_std_recent_mean():
    tracker = MetricTracker("loss", history_size=10)
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        tracker.update(v)

    assert tracker.mean == pytest.approx(3.0)
    assert tracker.recent_mean(3) == pytest.approx(4.0)


def test_metric_tracker_trend_increasing():
    tracker = MetricTracker("loss", history_size=20)
    for v in range(10):
        tracker.update(float(v))

    trend = tracker.get_trend(window=10)
    assert trend > 0.9  # slope should be ~1.0 for a straight increasing line


def test_metric_tracker_history_size_bounded():
    tracker = MetricTracker("loss", history_size=5)
    for v in range(20):
        tracker.update(float(v))

    assert len(tracker.history) == 5
    # only the last 5 values (15..19) should remain
    assert list(tracker.history) == [15.0, 16.0, 17.0, 18.0, 19.0]


def test_silent_crash_detector_accuracy_collapse():
    # accuracy_drop_sigma lowered from the 3.0 default: a clean two-level
    # step (baseline -> collapsed) has std bounded to ~drop/2, so drop_sigma
    # tops out around 2.0 regardless of how large the drop is -- this
    # exercises the detection path deterministically without depending on
    # a specific noise distribution.
    config = SilentDetectorConfig(
        detect_accuracy_collapse=True, smoothing_window=3, accuracy_drop_sigma=1.5
    )
    detector = SilentCrashDetector(config)

    # establish a healthy baseline
    for _ in range(5):
        detector.check(loss=1.0, val_metric=0.9)

    # then a sustained collapse
    result = None
    for _ in range(10):
        result = detector.check(loss=1.0, val_metric=0.1)

    assert result is not None
    assert result.detected
    assert result.failure_type == SilentFailureType.ACCURACY_COLLAPSE
