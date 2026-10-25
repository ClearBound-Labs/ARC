"""Deterministic replay / reproducibility report: samples
replay checks on a cadence and aggregates them into a run-level
"N% bit-reproducible" summary."""
import torch
import torch.nn as nn

from arc.security.reproducibility import ReproducibilityTracker


def test_sampling_cadence_only_checks_every_nth_call():
    model = nn.Sequential(nn.Linear(10, 10)).eval()
    x = torch.randn(4, 10)
    tracker = ReproducibilityTracker(check_every_n_calls=3)

    results = [tracker.maybe_check(model, x) for _ in range(6)]

    assert results[0] is None
    assert results[1] is None
    assert results[2] is not None
    assert results[5] is not None
    assert tracker.report().total_checks == 2


def test_healthy_model_reports_fully_reproducible():
    model = nn.Sequential(nn.Linear(10, 10)).eval()
    x = torch.randn(4, 10)
    tracker = ReproducibilityTracker(check_every_n_calls=1)

    for _ in range(5):
        tracker.maybe_check(model, x)

    report = tracker.report()
    assert report.total_checks == 5
    assert report.reproducible_ratio == 1.0


def test_empty_report_before_any_checks():
    tracker = ReproducibilityTracker()
    report = tracker.report()
    assert report.total_checks == 0
    assert report.reproducible_ratio == 1.0


if __name__ == "__main__":
    test_sampling_cadence_only_checks_every_nth_call()
    test_healthy_model_reports_fully_reproducible()
    test_empty_report_before_any_checks()
    print("ok")
