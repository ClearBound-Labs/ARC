"""MTTR / recovery-rate metrics: pure aggregation over an intervention_log
shaped like the one Controller writes. A failure counts as recovered only
once a later `recovery_confirmed` event covers it."""
from arc.evaluation.reliability_metrics import compute_reliability_metrics


def _failure(step, reason, action="rollback", steps_lost=0):
    return {
        "step": step,
        "reason": reason,
        "action": action,
        "steps_lost": steps_lost,
        "timestamp": f"2026-07-11T00:00:{step:02d}",
    }


def _confirmed(step, resolved):
    return {
        "step": step,
        "reason": "x",
        "action": "recovery_confirmed",
        "failures_resolved": resolved,
        "timestamp": f"2026-07-11T00:00:{step:02d}",
    }


def test_empty_log_returns_zeroed_metrics():
    metrics = compute_reliability_metrics([])
    assert metrics.total_failures == 0
    assert metrics.recovery_rate == 0.0


def test_rollback_without_confirmation_is_not_recovered():
    metrics = compute_reliability_metrics([_failure(1, "nan_loss")])

    assert metrics.total_failures == 1
    assert metrics.total_recovered == 0
    assert metrics.recovery_rate == 0.0


def test_confirmation_recovers_every_pending_failure():
    log = [
        _failure(1, "nan_loss", steps_lost=2),
        _failure(2, "nan_loss", steps_lost=4),
        _confirmed(12, resolved=2),
        _failure(20, "loss_explosion", action="skip"),
    ]
    metrics = compute_reliability_metrics(log)

    assert metrics.total_failures == 3
    assert metrics.total_recovered == 2
    assert abs(metrics.recovery_rate - 2 / 3) < 1e-9
    assert metrics.recovery_rate_by_reason == {"nan_loss": 1.0, "loss_explosion": 0.0}


def test_steps_lost_stats_cover_rollbacks_only():
    log = [
        _failure(1, "nan_loss", steps_lost=2),
        _failure(2, "nan_loss", steps_lost=6),
        _failure(3, "nan_loss", action="skip"),
    ]
    metrics = compute_reliability_metrics(log)

    assert metrics.mean_steps_lost == 4.0
    assert metrics.median_steps_lost == 4.0


def test_legacy_steps_recovered_field_still_read():
    log = [{"step": 1, "reason": "nan_loss", "action": "rollback", "steps_recovered": 5}]

    assert compute_reliability_metrics(log).mean_steps_lost == 5.0


def test_mean_steps_between_failures_ignores_confirmations():
    log = [
        _failure(10, "nan_loss"),
        _failure(20, "nan_loss"),
        _confirmed(30, resolved=2),
        _failure(40, "nan_loss"),
    ]
    metrics = compute_reliability_metrics(log)

    assert metrics.mean_steps_between_failures == 15.0  # (10 + 20) / 2
