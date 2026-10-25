"""Straggler detection: a rank that's
persistently slow (not dead) gets flagged only after several consecutive
slow steps, so single-step jitter doesn't trigger a false eviction."""
import pytest

from arc.distributed.straggler_detection import StragglerDetector


def test_healthy_ranks_no_straggler():
    detector = StragglerDetector(zscore_threshold=4.0, min_consecutive_flags=3)
    report = None
    for _ in range(10):
        report = detector.record_step({0: 1.0, 1: 1.02, 2: 0.98, 3: 1.01})
    assert report.has_straggler is False


def test_persistent_slow_rank_flagged():
    detector = StragglerDetector(zscore_threshold=4.0, min_consecutive_flags=3)
    report = None
    for _ in range(5):
        report = detector.record_step({0: 1.0, 1: 1.0, 2: 1.0, 3: 5.0})
    assert report.has_straggler is True
    assert 3 in report.straggler_ranks


def test_single_slow_step_not_flagged_needs_consecutive():
    detector = StragglerDetector(zscore_threshold=4.0, min_consecutive_flags=3)
    detector.record_step({0: 1.0, 1: 1.0, 2: 1.0, 3: 5.0})
    report = detector.record_step({0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0})  # recovers
    assert report.has_straggler is False


def test_requires_at_least_three_ranks():
    detector = StragglerDetector()
    with pytest.raises(ValueError, match=">= 3 ranks"):
        detector.record_step({0: 1.0, 1: 1.0})


def test_departed_rank_state_is_dropped_not_leaked():
    detector = StragglerDetector(zscore_threshold=4.0, min_consecutive_flags=3)
    for _ in range(3):
        detector.record_step({0: 1.0, 1: 1.0, 2: 1.0, 3: 5.0})  # rank 3 flagged twice-ish

    # rank 3 leaves the group; a new rank 4 joins in its place
    detector.record_step({0: 1.0, 1: 1.0, 2: 1.0, 4: 1.0})

    assert 3 not in detector._history
    assert 3 not in detector._consecutive_flags


def test_rank_id_reuse_after_departure_starts_clean():
    detector = StragglerDetector(zscore_threshold=4.0, min_consecutive_flags=3)
    for _ in range(3):
        detector.record_step({0: 1.0, 1: 1.0, 2: 1.0, 3: 5.0})  # rank 3 slow, building up flags

    detector.record_step({0: 1.0, 1: 1.0, 2: 1.0})  # rank 3 leaves
    # rank 3's id reused by a brand-new, healthy rank -- must not inherit old flags
    report = detector.record_step({0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0})

    assert report.has_straggler is False


if __name__ == "__main__":
    test_healthy_ranks_no_straggler()
    test_persistent_slow_rank_flagged()
    test_single_slow_step_not_flagged_needs_consecutive()
    test_requires_at_least_three_ranks()
    print("ok")
