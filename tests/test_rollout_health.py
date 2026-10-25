"""Rollout/generation health checks: flags mode collapse (batch
of near-identical generations), high n-gram repetition (degenerate loops),
and format drift (batch length suddenly way outside history)."""
from arc.signals.rollout_health import RolloutHealthMonitor


def test_healthy_diverse_batch():
    monitor = RolloutHealthMonitor()
    sequences = [list(range(i, i + 20)) for i in range(10)]  # all distinct, no repetition

    report = monitor.check_batch(sequences)

    assert report.is_healthy


def test_mode_collapse_detected():
    monitor = RolloutHealthMonitor(unique_ratio_threshold=0.3)
    sequences = [[1, 2, 3, 4, 5]] * 10  # every generation identical

    report = monitor.check_batch(sequences)

    assert report.mode_collapse is True


def test_high_repetition_detected():
    monitor = RolloutHealthMonitor(ngram_n=2, repetition_threshold=0.3)
    sequences = [[1, 2] * 20 for _ in range(5)]  # degenerate repeating loop

    report = monitor.check_batch(sequences)

    assert report.high_repetition is True


def test_format_drift_detected_after_length_shift():
    monitor = RolloutHealthMonitor(length_zscore_threshold=3.0)
    for _ in range(10):
        monitor.check_batch([[0] * 20 for _ in range(4)])  # stable length ~20

    report = monitor.check_batch([[0] * 500 for _ in range(4)])  # sudden length spike

    assert report.format_drift is True


def test_empty_batch_reports_healthy():
    monitor = RolloutHealthMonitor()
    report = monitor.check_batch([])
    assert report.is_healthy


if __name__ == "__main__":
    test_healthy_diverse_batch()
    test_mode_collapse_detected()
    test_high_repetition_detected()
    test_format_drift_detected_after_length_shift()
    test_empty_batch_reports_healthy()
    print("ok")
