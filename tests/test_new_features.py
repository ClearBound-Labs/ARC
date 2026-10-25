"""Auto-calibration, gradient accumulation, summary, and W&B logging."""
import pytest
import torch
import torch.nn as nn

from arc.core.controller import Controller, ControllerConfig, HealthStatus


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(100, 64), nn.ReLU(), nn.Linear(64, 10))


def _train(model, opt, ctrl, n, accumulate=1):
    for i in range(n):
        loss = model(torch.randn(8, 100)).pow(2).mean() / accumulate
        loss.backward()
        action = ctrl.step(loss)
        if (i + 1) % accumulate == 0:
            if not action.should_skip:
                opt.step()
            opt.zero_grad()


def test_auto_calibration_replaces_default_thresholds():
    model = _model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ctrl = Controller(model, opt, config=ControllerConfig(
        calibration_steps=30, verbose=False, enable_lyapunov=False, enable_trend=False))

    _train(model, opt, ctrl, 50)

    assert ctrl._calibrated
    assert ctrl.config.loss_explosion_threshold != 100.0
    assert ctrl.config.gradient_explosion_threshold != 1e5


def test_nan_during_calibration_still_rolls_back():
    model = _model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    ctrl = Controller(model, opt, config=ControllerConfig(calibration_steps=100, verbose=False))
    _train(model, opt, ctrl, 5)

    assert ctrl.step(torch.tensor(float("nan"))).rolled_back


def test_gradient_accumulation_checks_only_full_steps():
    model = _model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    ctrl = Controller(model, opt, config=ControllerConfig(
        gradient_accumulation_steps=4, auto_calibrate=False, verbose=False,
        enable_lyapunov=False, enable_trend=False))
    ctrl.config.gradient_explosion_threshold = 1e-9  # any full check fails

    actions = []
    for _ in range(4):
        loss = model(torch.randn(4, 100)).pow(2).mean() / 4
        loss.backward()
        actions.append(ctrl.step(loss))

    assert all(a.status == HealthStatus.HEALTHY for a in actions[:3])
    assert actions[3].failure_type == "gradient_explosion"


def test_summary_reports_failures_without_invented_savings(capsys):
    model = _model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    ctrl = Controller(model, opt, config=ControllerConfig(auto_calibrate=False, verbose=False))
    _train(model, opt, ctrl, 20)
    ctrl.step(torch.tensor(float("nan")))

    report = ctrl.summary()

    assert "Failures found:  1" in report
    assert "unknown (set baseline_checkpoint_interval)" in report
    assert report in capsys.readouterr().out


def test_log_to_wandb_is_a_no_op_without_a_run():
    model = _model()
    ctrl = Controller(model, torch.optim.SGD(model.parameters(), lr=0.1),
                      config=ControllerConfig(verbose=False))
    ctrl.log_to_wandb()
