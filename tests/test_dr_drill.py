"""Checkpoint restore drill: a disaster-recovery self-test
that restores into a throwaway model and confirms it reproduces the live
model's output, without touching the live model at all."""
import torch
import torch.nn as nn

from arc.checkpointing.adaptive import (
    AdaptiveCheckpointer,
    AdaptiveCheckpointConfig,
    CheckpointStrategy,
)
from arc.checkpointing.dr_drill import verify_recoverability


def _make_model():
    return nn.Sequential(nn.Linear(10, 10))


def test_drill_passes_for_healthy_checkpoint():
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    config = AdaptiveCheckpointConfig(
        auto_select_strategy=False,
        preferred_strategy=CheckpointStrategy.FULL_CPU,
        verbose=False,
    )
    checkpointer = AdaptiveCheckpointer(model, optimizer, config=config)
    sample_input = torch.randn(4, 10)

    report = verify_recoverability(
        checkpointer,
        model_factory=_make_model,
        optimizer_factory=lambda m: torch.optim.Adam(m.parameters(), lr=0.01),
        sample_input=sample_input,
    )

    assert report.passed is True
    assert report.max_abs_diff < report.tolerance


def test_drill_does_not_mutate_live_model():
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    config = AdaptiveCheckpointConfig(
        auto_select_strategy=False,
        preferred_strategy=CheckpointStrategy.FULL_CPU,
        verbose=False,
    )
    checkpointer = AdaptiveCheckpointer(model, optimizer, config=config)
    sample_input = torch.randn(4, 10)
    was_training = model.training

    before = {k: v.clone() for k, v in model.state_dict().items()}
    verify_recoverability(
        checkpointer,
        model_factory=_make_model,
        optimizer_factory=lambda m: torch.optim.Adam(m.parameters(), lr=0.01),
        sample_input=sample_input,
    )
    after = model.state_dict()

    for k in before:
        assert torch.equal(before[k], after[k])
    assert model.training == was_training


def test_drill_passes_after_chained_incremental_deltas():
    """Regression test: a delta-of-a-delta must reconstruct through the
    whole chain, not just the latest delta against the original full
    checkpoint (which used to silently drop every change made in between)."""
    model = _make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    config = AdaptiveCheckpointConfig(
        auto_select_strategy=False,
        preferred_strategy=CheckpointStrategy.INCREMENTAL_DELTA,
        verbose=False,
    )
    checkpointer = AdaptiveCheckpointer(model, optimizer, config=config)

    checkpointer.save()  # first save falls back to a full checkpoint
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.1)
    checkpointer.save()  # first real incremental delta

    sample_input = torch.randn(4, 10)
    report = verify_recoverability(
        checkpointer,
        model_factory=_make_model,
        optimizer_factory=lambda m: torch.optim.Adam(m.parameters(), lr=0.01),
        sample_input=sample_input,
    )

    assert report.passed is True
    assert report.max_abs_diff < report.tolerance


if __name__ == "__main__":
    test_drill_passes_for_healthy_checkpoint()
    test_drill_does_not_mutate_live_model()
    test_drill_passes_after_chained_incremental_deltas()
    print("ok")
