"""Regression tests for recovery escalation under persistent failures.

Covers two bugs found by the organic-failure harness:
  1. Checkpoint restore resurrected the checkpointed learning rate, so ARC's
     own LR reductions never accumulated across rollbacks.
  2. A failure ARC cannot fix (bad input data) looped forever, reporting
     "Recovered" on every step, with no terminal signal to the caller.
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from arc.core.controller import Controller, ControllerConfig


def _fixture(lr=1e-3):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    cfg = ControllerConfig(checkpoint_frequency=5, verbose=False,
                           auto_calibrate=False, calibration_steps=0)
    return model, opt, Controller(model, opt, config=cfg)


def _lr(opt):
    return opt.param_groups[0]["lr"]


def test_lr_reductions_survive_rollback():
    """Each rollback must leave the LR strictly lower than the previous one."""
    model, opt, ctrl = _fixture()
    x, y = torch.randn(16, 8), torch.randn(16, 1)

    for _ in range(6):
        opt.zero_grad()
        F.mse_loss(model(x), y).backward()
        opt.step()

    lrs = []
    for _ in range(4):
        action = ctrl.step(torch.tensor(float("nan")))
        assert action.rolled_back
        lrs.append(_lr(opt))

    assert lrs == sorted(lrs, reverse=True), f"LR did not ratchet down: {lrs}"
    assert len(set(lrs)) == len(lrs), f"LR repeated across rollbacks: {lrs}"
    assert lrs[-1] < 1e-3, f"LR ended at {lrs[-1]}, no net reduction"


def test_unrecoverable_failure_is_flagged():
    """A failure rollback cannot fix must eventually be reported as
    unrecoverable instead of looping silently forever."""
    model, opt, ctrl = _fixture()
    x, y = torch.randn(16, 8), torch.randn(16, 1)

    for _ in range(6):
        opt.zero_grad()
        F.mse_loss(model(x), y).backward()
        opt.step()

    flagged = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(80):
            action = ctrl.step(torch.tensor(float("nan")))
            if action.unrecoverable:
                flagged = True
                break

    assert flagged, "persistent unfixable failure was never flagged unrecoverable"
    assert ctrl.unrecoverable is True
    assert "data" in (action.message or "").lower(), (
        f"message should point at the real cause, got: {action.message!r}")


def test_healthy_training_never_flagged_unrecoverable():
    """The escalation must not fire during normal training."""
    model, opt, ctrl = _fixture()
    x, y = torch.randn(16, 8), torch.randn(16, 1)

    for _ in range(60):
        opt.zero_grad()
        loss = F.mse_loss(model(x), y)
        action = ctrl.step(loss)
        assert not action.unrecoverable
        if not action.should_skip:
            loss.backward()
            opt.step()

    assert ctrl.unrecoverable is False
    assert _lr(opt) == 1e-3


def test_unrecoverable_flag_is_sticky():
    """Once declared, every later action must still report unrecoverable --
    a caller polling `action.unrecoverable` must not miss it on a skip step."""
    model, opt, ctrl = _fixture()
    x, y = torch.randn(16, 8), torch.randn(16, 1)

    for _ in range(6):
        opt.zero_grad()
        F.mse_loss(model(x), y).backward()
        opt.step()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(80):
            if ctrl.step(torch.tensor(float("nan"))).unrecoverable:
                break
        assert ctrl.unrecoverable
        for _ in range(10):
            assert ctrl.step(torch.tensor(float("nan"))).unrecoverable
