"""Run comparison & regression detection: diffing two exported
recovery ledgers flags a worse rollback rate, a worsening stability trend,
and any new failure mode not seen in the baseline."""
import json
import tempfile
import os

from arc.evaluation.run_comparison import RunComparator


def _ledger(total_steps, total_rollbacks, events):
    return {
        "total_steps": total_steps,
        "total_rollbacks": total_rollbacks,
        "events": events,
    }


def test_identical_runs_no_regression():
    comparator = RunComparator()
    ledger = _ledger(100, 2, [
        {"reason": "nan_loss", "stability_score": 0.9},
        {"reason": "nan_loss", "stability_score": 0.9},
    ])

    report = comparator.compare(ledger, ledger)

    assert report.has_regression is False


def test_higher_rollback_rate_flagged():
    comparator = RunComparator(rollback_rate_tolerance=0.2)
    baseline = _ledger(100, 2, [])
    current = _ledger(100, 10, [])

    report = comparator.compare(baseline, current)

    rollback_finding = next(f for f in report.findings if f.metric == "rollback_rate")
    assert rollback_finding.is_regression is True
    assert report.has_regression is True


def test_worsening_stability_trend_flagged():
    comparator = RunComparator(stability_slope_tolerance=0.001)
    baseline = _ledger(100, 0, [
        {"reason": "x", "stability_score": s} for s in [0.9, 0.91, 0.92, 0.93]
    ])
    current = _ledger(100, 0, [
        {"reason": "x", "stability_score": s} for s in [0.9, 0.7, 0.5, 0.3]
    ])

    report = comparator.compare(baseline, current)

    slope_finding = next(f for f in report.findings if f.metric == "stability_score_slope")
    assert slope_finding.is_regression is True


def test_new_failure_mode_flagged():
    comparator = RunComparator()
    baseline = _ledger(100, 1, [{"reason": "nan_loss", "stability_score": 0.9}])
    current = _ledger(100, 1, [{"reason": "weight_corruption", "stability_score": 0.9}])

    report = comparator.compare(baseline, current)

    new_mode_finding = next(f for f in report.findings if f.metric == "new_failure_modes")
    assert new_mode_finding.is_regression is True
    assert "weight_corruption" in new_mode_finding.detail


def test_compare_files_roundtrips_through_json():
    comparator = RunComparator()
    ledger = _ledger(100, 1, [{"reason": "nan_loss", "stability_score": 0.9}])

    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_path = os.path.join(tmpdir, "baseline.json")
        current_path = os.path.join(tmpdir, "current.json")
        with open(baseline_path, "w") as f:
            json.dump(ledger, f)
        with open(current_path, "w") as f:
            json.dump(ledger, f)

        report = comparator.compare_files(baseline_path, current_path)

    assert report.has_regression is False


if __name__ == "__main__":
    test_identical_runs_no_regression()
    test_higher_rollback_rate_flagged()
    test_worsening_stability_trend_flagged()
    test_new_failure_mode_flagged()
    test_compare_files_roundtrips_through_json()
    print("ok")
