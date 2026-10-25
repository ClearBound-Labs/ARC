"""Checkpoint integrity & signing: disk checkpoints get a sha256
content hash and, when a signing_key is set, an HMAC signature -- both
verified on restore, tampering must be rejected rather than silently loaded."""
import os
import tempfile

import torch
import torch.nn as nn
import pytest

from arc.checkpointing.adaptive import (
    AdaptiveCheckpointer,
    AdaptiveCheckpointConfig,
    CheckpointStrategy,
)


def _make_checkpointer(tmpdir, **overrides):
    model = nn.Sequential(nn.Linear(10, 10))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    config = AdaptiveCheckpointConfig(
        auto_select_strategy=False,
        preferred_strategy=CheckpointStrategy.STREAMING_DISK,
        disk_checkpoint_dir=tmpdir,
        verbose=False,
        **overrides,
    )
    return AdaptiveCheckpointer(model, optimizer, config=config), model, optimizer


def test_signed_checkpoint_round_trips():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt, model, optimizer = _make_checkpointer(
            tmpdir, sign_checkpoints=True, signing_key="secret-key"
        )
        ckpt.save(step=1)
        ckpt.restore()  # must not raise


def test_tampered_checkpoint_hash_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt, model, optimizer = _make_checkpointer(tmpdir, sign_checkpoints=True)
        ckpt.save(step=1)
        path = ckpt.checkpoints[-1]["path"]

        blob = torch.load(path, weights_only=False)
        blob["model"][next(iter(blob["model"]))] += 1.0  # tamper with weights
        torch.save(blob, path)

        with pytest.raises(ValueError, match="integrity check failed"):
            ckpt.restore()


def test_signature_required_key_missing_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt, model, optimizer = _make_checkpointer(
            tmpdir, sign_checkpoints=True, signing_key="secret-key"
        )
        ckpt.save(step=1)

        ckpt.config.signing_key = None
        with pytest.raises(ValueError, match="no signing_key"):
            ckpt.restore()


def test_signature_wrong_key_rejected():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt, model, optimizer = _make_checkpointer(
            tmpdir, sign_checkpoints=True, signing_key="secret-key"
        )
        ckpt.save(step=1)

        ckpt.config.signing_key = "wrong-key"
        with pytest.raises(ValueError, match="signature verification failed"):
            ckpt.restore()


def test_unsigned_checkpoints_unaffected():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt, model, optimizer = _make_checkpointer(tmpdir)  # sign_checkpoints defaults False
        ckpt.save(step=1)
        ckpt.restore()  # no integrity fields at all, must not raise


@pytest.mark.parametrize("strategy", [
    CheckpointStrategy.FULL_CPU,
    CheckpointStrategy.QUANTIZED_FP16,
    CheckpointStrategy.INCREMENTAL_DELTA,
])
def test_signing_works_for_in_memory_strategies_not_just_disk(strategy):
    """A customer who sets sign_checkpoints=True must get real protection
    whichever strategy ARC auto-selects, not only STREAMING_DISK."""
    model = nn.Sequential(nn.Linear(10, 10))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    config = AdaptiveCheckpointConfig(
        auto_select_strategy=False,
        preferred_strategy=strategy,
        verbose=False,
        sign_checkpoints=True,
        signing_key="secret-key",
    )
    ckpt = AdaptiveCheckpointer(model, optimizer, config=config)
    ckpt.save(step=1)

    saved = ckpt.checkpoints[-1]
    assert "integrity_hash" in saved
    assert "integrity_signature" in saved

    ckpt.restore()  # must not raise


if __name__ == "__main__":
    test_signed_checkpoint_round_trips()
    test_tampered_checkpoint_hash_rejected()
    test_signature_required_key_missing_rejected()
    test_signature_wrong_key_rejected()
    test_unsigned_checkpoints_unaffected()
    for s in [CheckpointStrategy.FULL_CPU, CheckpointStrategy.QUANTIZED_FP16, CheckpointStrategy.INCREMENTAL_DELTA]:
        test_signing_works_for_in_memory_strategies_not_just_disk(s)
    print("ok")
