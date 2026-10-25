"""HardwareRecoveryHandler must never claim a recovery it didn't perform
(audit item 8)."""
import os
import time

import pytest
import torch
import torch.nn as nn

from arc.intervention.hardware_handler import HardwareConfig, HardwareRecoveryHandler


def _handler(tmp_path, **kw):
    model = nn.Linear(4, 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    cfg = HardwareConfig(verbose=False, checkpoint_dir=str(tmp_path), retry_delay_seconds=0.0, **kw)
    return model, HardwareRecoveryHandler(model, opt, cfg)


@pytest.mark.parametrize("message", [
    "CUDA error: an illegal memory access was encountered",
    "CUDA error: device-side assert triggered",
    "CUDA error: unspecified launch failure",
])
def test_sticky_cuda_errors_are_not_reported_as_recovered(tmp_path, message):
    _, handler = _handler(tmp_path)

    result = handler.try_recover(RuntimeError(message))

    assert result.success is False
    assert "restart" in result.recovery_action.lower()


def test_cpu_fallback_is_opt_in(tmp_path):
    model, handler = _handler(tmp_path)
    device = next(model.parameters()).device

    result = handler.try_recover(RuntimeError("CUDA driver error: initialization error"))

    assert result.success is False
    assert next(model.parameters()).device == device


def test_nccl_failure_without_process_group_is_not_recovered(tmp_path):
    _, handler = _handler(tmp_path)

    result = handler.try_recover(RuntimeError("NCCL error: unhandled system error"))

    assert result.success is False


def test_failed_barrier_is_not_recovered(tmp_path, monkeypatch):
    _, handler = _handler(tmp_path)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def broken_barrier(**kwargs):
        raise RuntimeError("barrier timed out")

    monkeypatch.setattr(torch.distributed, "barrier", broken_barrier)

    assert handler.try_recover(RuntimeError("NCCL communicator was aborted")).success is False


def test_disk_full_with_nothing_to_delete_is_not_recovered(tmp_path):
    _, handler = _handler(tmp_path)

    result = handler.try_recover(OSError(28, "No space left on device"))

    assert result.success is False


def test_disk_full_only_deletes_arcs_own_checkpoints(tmp_path):
    _, handler = _handler(tmp_path)
    user_file = tmp_path / "my_finetuned_model.pt"
    user_file.write_bytes(b"precious")
    time.sleep(0.01)
    for i in range(5):
        (tmp_path / f"checkpoint_{i}.pt").write_bytes(b"x")
        time.sleep(0.01)

    result = handler.try_recover(OSError(28, "No space left on device"))

    assert result.success is True
    assert user_file.exists()
    remaining = sorted(p.name for p in tmp_path.glob("checkpoint_*.pt"))
    assert remaining == ["checkpoint_3.pt", "checkpoint_4.pt"]
