"""Numerical health fleet baseline:
a new run within the learned envelope passes quietly, a run whose rollback
rate is way outside history gets flagged, and the baseline round-trips
through save/load."""
import tempfile
import os

from arc.security.health_baseline import HealthBaseline


def _ledger(rollback_rate_steps, rollbacks, stability=0.9, loss=0.5):
    return {
        "total_steps": rollback_rate_steps,
        "total_rollbacks": rollbacks,
        "events": [{"stability_score": stability, "loss_value": loss}] * max(rollbacks, 1),
    }


def test_too_few_samples_reports_nothing():
    baseline = HealthBaseline()
    baseline.add_run(_ledger(100, 2))

    report = baseline.check(_ledger(100, 2))

    assert report.deviations == []


def test_normal_run_within_envelope():
    baseline = HealthBaseline(zscore_threshold=3.0)
    for _ in range(10):
        baseline.add_run(_ledger(100, 2))

    report = baseline.check(_ledger(100, 2))

    assert report.is_anomalous is False


def test_anomalous_run_flagged():
    baseline = HealthBaseline(zscore_threshold=3.0)
    for _ in range(10):
        baseline.add_run(_ledger(100, 2))  # 2% rollback rate historically

    report = baseline.check(_ledger(100, 60))  # 60% rollback rate, way off

    assert report.is_anomalous is True
    rollback_dev = next(d for d in report.deviations if d.metric == "rollback_rate")
    assert rollback_dev.is_anomalous is True


def test_save_and_load_roundtrip():
    baseline = HealthBaseline(zscore_threshold=3.0)
    for _ in range(10):
        baseline.add_run(_ledger(100, 2))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "baseline.json")
        baseline.save(path)
        loaded = HealthBaseline.load(path, zscore_threshold=3.0)

    report = loaded.check(_ledger(100, 60))
    assert report.is_anomalous is True


if __name__ == "__main__":
    test_too_few_samples_reports_nothing()
    test_normal_run_within_envelope()
    test_anomalous_run_flagged()
    test_save_and_load_roundtrip()
    print("ok")
