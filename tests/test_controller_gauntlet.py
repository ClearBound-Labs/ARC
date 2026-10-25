"""Controller failure gauntlet: each injected failure is detected and handled."""
import warnings

import pytest
import torch
import torch.nn as nn

from arc.core.controller import Controller, ControllerConfig


def _env(**overrides):
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(100, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 10),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    config = ControllerConfig(checkpoint_frequency=5, verbose=False, auto_calibrate=False,
                              **overrides)
    return model, optimizer, Controller(model, optimizer, config=config)


def _train(model, optimizer, ctrl, n):
    for _ in range(n):
        loss = model(torch.randn(8, 100)).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        if not ctrl.step(loss).should_skip:
            optimizer.step()
    optimizer.zero_grad()


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_loss_rolls_back(value):
    model, opt, ctrl = _env()
    _train(model, opt, ctrl, 20)

    action = ctrl.step(torch.tensor(value))

    assert action.rolled_back and action.should_skip and action.failure_type == "nan_loss"


def test_nan_weights_roll_back_to_finite_weights():
    model, opt, ctrl = _env(weight_health_check_frequency=1)
    _train(model, opt, ctrl, 20)
    loss = model(torch.randn(8, 100)).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    with torch.no_grad():
        next(model.parameters()).fill_(float("nan"))

    action = ctrl.step(loss)

    assert action.rolled_back
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_gradient_explosion_rolls_back():
    model, opt, ctrl = _env()
    _train(model, opt, ctrl, 5)
    ctrl.config.gradient_explosion_threshold = 0.001
    loss = model(torch.randn(8, 100)).pow(2).mean()
    opt.zero_grad()
    loss.backward()

    action = ctrl.step(loss)

    assert action.rolled_back and action.failure_type == "gradient_explosion"


def test_loss_spike_rolls_back():
    model, opt, ctrl = _env()
    _train(model, opt, ctrl, 20)

    action = ctrl.step(torch.tensor(100000.0))

    assert action.rolled_back and action.failure_type == "loss_explosion"


def test_consecutive_failures_trigger_escape():
    model, opt, ctrl = _env(max_consecutive_rollbacks=3)
    _train(model, opt, ctrl, 10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(5):
            ctrl.step(torch.tensor(float("nan")))

    assert ctrl._persistent_failure_count > 0


def test_training_continues_and_recovers_after_rollback():
    model, opt, ctrl = _env()
    _train(model, opt, ctrl, 20)
    ctrl.step(torch.tensor(float("nan")))
    _train(model, opt, ctrl, 20)

    stats = ctrl.get_stats()
    assert stats["step_count"] == 41
    assert stats["failures_recovered"] == 1


def test_lyapunov_and_trend_report_after_training():
    model, opt, ctrl = _env()
    _train(model, opt, ctrl, 60)

    stats = ctrl.get_stats()
    assert stats["lyapunov_exponent"] != 0.0
    assert stats["trend"]["loss_n"] > 0


def test_scheduler_base_lrs_follow_lr_reduction():
    model, opt, _ = _env()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)
    ctrl = Controller(model, opt, scheduler=sched, config=ControllerConfig(
        verbose=False, auto_calibrate=False))
    _train(model, opt, ctrl, 10)

    ctrl.step(torch.tensor(float("nan")))

    assert sched.base_lrs[0] == pytest.approx(0.005)
