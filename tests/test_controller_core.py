"""Controller regressions for the Oct 2026 audit (items 5, 7, 11-16)."""
import hashlib
import json

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from arc.core.controller import Controller, ControllerConfig

CUDA = torch.cuda.is_available()


def _cfg(**kw):
    base = dict(
        verbose=False,
        auto_calibrate=False,
        enable_lyapunov=False,
        enable_trend=False,
        adaptive_frequency=False,
        checkpoint_frequency=5,
    )
    base.update(kw)
    return ControllerConfig(**base)


def _setup(seed=0, lr=0.01, device="cpu", scaler=None, scheduler=None, model=None, **kw):
    torch.manual_seed(seed)
    model = (model or nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ctrl = Controller(model, opt, config=_cfg(**kw), scaler=scaler) if scaler is not None \
        else Controller(model, opt, config=_cfg(**kw))
    return model, opt, ctrl


def _batch(device="cpu", n_in=8):
    return torch.randn(16, n_in, device=device), torch.randn(16, 1, device=device)


def _train(model, opt, ctrl, n, device="cpu", n_in=8):
    for _ in range(n):
        x, y = _batch(device, n_in)
        loss = F.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        if not ctrl.step(loss).should_skip:
            opt.step()
    opt.zero_grad()


def _nan():
    return torch.tensor(float("nan"))


def _lr(opt):
    return opt.param_groups[0]["lr"]


# --- item 14: no silent gradient clipping -------------------------------

def test_gradients_are_not_clipped_by_default():
    model, opt, ctrl = _setup()
    _train(model, opt, ctrl, 3)
    x, y = _batch()
    loss = F.mse_loss(model(x), y)
    opt.zero_grad()
    loss.backward()
    for p in model.parameters():
        p.grad.mul_(50.0)
    before = [p.grad.clone() for p in model.parameters()]
    assert torch.linalg.vector_norm(torch.cat([g.flatten() for g in before])) > 1.0

    action = ctrl.step(loss)

    assert not action.gradients_clipped
    assert all(torch.equal(b, p.grad) for b, p in zip(before, model.parameters()))


# --- item 7: honest recovery accounting ---------------------------------

def test_recovery_counted_only_after_confirmation_window():
    model, opt, ctrl = _setup(recovery_confirm_steps=5)
    _train(model, opt, ctrl, 8)

    assert ctrl.step(_nan()).rolled_back
    assert ctrl.failures_recovered == 0
    _train(model, opt, ctrl, 4)
    assert ctrl.failures_recovered == 0
    _train(model, opt, ctrl, 1)
    assert ctrl.failures_recovered == 1


def test_failures_before_confirmation_are_resolved_together():
    model, opt, ctrl = _setup(recovery_confirm_steps=5)
    _train(model, opt, ctrl, 8)
    ctrl.step(_nan())
    _train(model, opt, ctrl, 2)
    ctrl.step(_nan())
    _train(model, opt, ctrl, 5)

    assert ctrl.failures_detected == 2
    assert ctrl.failures_recovered == 2
    confirmations = [e for e in ctrl.intervention_log if e["action"] == "recovery_confirmed"]
    assert len(confirmations) == 1
    assert confirmations[0]["failures_resolved"] == 2


def test_ledger_records_steps_lost_and_saved_against_baseline():
    model, opt, ctrl = _setup(checkpoint_frequency=5, baseline_checkpoint_interval=100)
    _train(model, opt, ctrl, 12)

    ctrl.step(_nan())  # controller step 13; last committed snapshot is end of step 10

    event = ctrl.intervention_log[-1]
    assert event["restored_step"] == 10
    assert event["steps_lost"] == 3
    assert event["steps_saved"] == 10  # without ARC: restart from step 0 (13 // 100 * 100)


def test_savings_are_unknown_without_a_baseline():
    model, opt, ctrl = _setup(checkpoint_frequency=5)
    _train(model, opt, ctrl, 12)
    ctrl.step(_nan())

    event = ctrl.intervention_log[-1]
    assert event["steps_lost"] == 3
    assert event["steps_saved"] is None
    assert event["gpu_hours_saved"] is None
    json.loads(ctrl.export_recovery_ledger())  # still strict-JSON exportable


# --- item 15: LR recovers, schedulers keep the reduction ----------------

def test_lr_recovers_after_healthy_streak():
    model, opt, ctrl = _setup(lr=0.01, lr_recovery_steps=10)
    _train(model, opt, ctrl, 6)
    ctrl.step(_nan())
    assert _lr(opt) == pytest.approx(0.005)

    _train(model, opt, ctrl, 9)
    assert _lr(opt) == pytest.approx(0.005)
    _train(model, opt, ctrl, 1)
    assert _lr(opt) == pytest.approx(0.01)


def test_repeated_rollbacks_halve_lr_exactly_once_each():
    model, opt, ctrl = _setup(lr=0.01, checkpoint_frequency=2, lr_recovery_steps=0)
    _train(model, opt, ctrl, 4)
    ctrl.step(_nan())
    assert _lr(opt) == pytest.approx(0.005)
    _train(model, opt, ctrl, 4)  # a new snapshot is taken at the reduced LR
    ctrl.step(_nan())
    assert _lr(opt) == pytest.approx(0.0025)


def test_onecycle_scheduler_keeps_reduction():
    def make():
        torch.manual_seed(0)
        m = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))
        o = torch.optim.SGD(m.parameters(), lr=0.1)
        s = torch.optim.lr_scheduler.OneCycleLR(o, max_lr=0.1, total_steps=100)
        return m, o, s

    model, opt, sched = make()
    _, ref_opt, ref_sched = make()
    ctrl = Controller(model, opt, scheduler=sched, config=_cfg(lr_recovery_steps=0))

    def healthy_step():
        x, y = _batch()
        loss = F.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        assert not ctrl.step(loss).should_skip
        opt.step()
        sched.step()
        ref_opt.step()
        ref_sched.step()

    for _ in range(6):
        healthy_step()
    ctrl.step(_nan())
    for _ in range(5):
        healthy_step()

    assert _lr(opt) == pytest.approx(0.5 * _lr(ref_opt))


# --- detection holes ----------------------------------------------------

def test_loss_jump_against_one_sample_is_not_a_spike():
    _, _, ctrl = _setup()

    assert ctrl.step(torch.tensor(0.02)).failure_type is None
    assert ctrl.step(torch.tensor(0.3)).failure_type is None  # 15x, but one sample of history


def test_spike_against_established_history_is_caught():
    _, _, ctrl = _setup()
    for _ in range(5):
        ctrl.step(torch.tensor(0.1))

    assert ctrl.step(torch.tensor(5.0)).failure_type == "loss_explosion"

def test_nan_gradients_with_finite_loss_are_caught():
    model, opt, ctrl = _setup()
    _train(model, opt, ctrl, 6)
    x, y = _batch()
    loss = F.mse_loss(model(x), y)
    opt.zero_grad()
    loss.backward()
    next(model.parameters()).grad[0, 0] = float("nan")

    action = ctrl.step(loss)

    assert action.should_skip and action.rolled_back
    assert action.failure_type == "nan_gradients"


def test_step_without_loss_still_checks_gradients():
    model, opt, ctrl = _setup()
    _train(model, opt, ctrl, 6)
    x, y = _batch()
    opt.zero_grad()
    F.mse_loss(model(x), y).backward()
    next(model.parameters()).grad[0, 0] = float("inf")

    action = ctrl.step(None)

    assert action.should_skip and action.rolled_back


def test_nan_weights_detected_in_any_tensor_not_just_the_first_20():
    deep = nn.Sequential(*[nn.Linear(4, 4) for _ in range(14)], nn.Linear(4, 1))  # 30 tensors
    model, opt, ctrl = _setup(model=deep, weight_health_check_frequency=1)
    _train(model, opt, ctrl, 3, n_in=4)
    x = torch.randn(16, 4)
    loss = model(x).pow(2).mean()
    opt.zero_grad()
    loss.backward()
    with torch.no_grad():
        model[-1].bias[0] = float("nan")

    action = ctrl.step(loss)

    assert action.should_skip and action.failure_type == "nan_weights"


def test_gradient_explosion_uses_true_global_norm():
    model, opt, ctrl = _setup()
    _train(model, opt, ctrl, 6)
    x, y = _batch()
    loss = F.mse_loss(model(x), y)
    opt.zero_grad()
    loss.backward()
    true_norm = torch.linalg.vector_norm(
        torch.cat([p.grad.flatten() for p in model.parameters()])).item()

    ctrl.config.gradient_explosion_threshold = true_norm * 1.01
    assert not ctrl.step(loss).should_skip

    loss = F.mse_loss(model(x), y)
    opt.zero_grad()
    loss.backward()
    ctrl.config.gradient_explosion_threshold = true_norm * 0.99
    assert ctrl.step(loss).failure_type == "gradient_explosion"


# --- item 13: snapshot = the healthy post-update state ------------------

@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if CUDA else []))
def test_rollback_restores_state_after_last_snapshotted_update(device):
    model, opt, ctrl = _setup(device=device, checkpoint_frequency=5)
    _train(model, opt, ctrl, 4, device=device)
    x, y = _batch(device)
    loss = F.mse_loss(model(x), y)
    opt.zero_grad()
    loss.backward()
    assert not ctrl.step(loss).should_skip  # controller step 5 -> snapshot due
    opt.step()
    expected = [p.detach().clone() for p in model.parameters()]
    expected_m = [opt.state[p]["exp_avg"].clone() for p in model.parameters()]

    _train(model, opt, ctrl, 3, device=device)  # updates race the async copy
    assert ctrl.step(_nan()).rolled_back

    for e, p in zip(expected, model.parameters()):
        assert torch.equal(e, p.detach())
    for e, p in zip(expected_m, model.parameters()):
        assert torch.equal(e, opt.state[p]["exp_avg"])


@pytest.mark.skipif(not CUDA, reason="async snapshots are CUDA-only")
def test_async_snapshots_never_hold_pinned_memory():
    # Pinned host memory is scarce; holding max_checkpoints pinned copies of a
    # large model exhausted it and aborted the process on a 16 GB machine.
    model, opt, ctrl = _setup(device="cuda", checkpoint_frequency=2)
    _train(model, opt, ctrl, 12, device="cuda")
    ctrl._sync_inflight()

    for snap in ctrl._checkpoints:
        for value in snap["model"].values():
            assert not value.is_pinned()
        for state in snap["optim"]["state"].values():
            for value in state.values():
                if torch.is_tensor(value):
                    assert not value.is_pinned()


def test_controller_state_dict_roundtrip():
    model, opt, ctrl = _setup(auto_calibrate=True, calibration_steps=5, lr_recovery_steps=0)
    _train(model, opt, ctrl, 8)
    ctrl.step(_nan())
    _train(model, opt, ctrl, 2)

    sd = ctrl.state_dict()
    _, _, ctrl2 = _setup(seed=1, auto_calibrate=True, calibration_steps=5, lr_recovery_steps=0)
    ctrl2.load_state_dict(sd)

    s1, s2 = ctrl.get_stats(), ctrl2.get_stats()
    for key in ("step_count", "failures_detected", "total_rollbacks", "lr_reductions"):
        assert s1[key] == s2[key]
    assert ctrl2.config.loss_explosion_threshold == ctrl.config.loss_explosion_threshold
    assert ctrl2.intervention_log == ctrl.intervention_log
    assert ctrl2.verify_ledger_integrity()


def test_save_and_load_checkpoint_resumes_exact_state(tmp_path):
    model, opt, ctrl = _setup()
    _train(model, opt, ctrl, 7)
    path = tmp_path / "arc.pt"
    ctrl.save_checkpoint(path)
    torch.load(path, weights_only=True)  # safe-loadable

    model2, opt2, ctrl2 = _setup(seed=1)
    assert ctrl2.load_checkpoint(path) == 7

    for a, b in zip(model.parameters(), model2.parameters()):
        assert torch.equal(a, b)
        assert torch.equal(opt.state[a]["exp_avg_sq"], opt2.state[b]["exp_avg_sq"])
    assert ctrl2.get_stats()["step_count"] == 7


# --- item 16: keyed ledger ----------------------------------------------

def test_keyed_ledger_rejects_recomputed_unkeyed_chain():
    model, opt, ctrl = _setup(ledger_key="s3cret")
    _train(model, opt, ctrl, 6)
    ctrl.step(_nan())
    ctrl.step(_nan())
    assert ctrl.verify_ledger_integrity()

    prev = ""
    for i, entry in enumerate(ctrl.intervention_log):
        body = {k: v for k, v in entry.items() if k != "entry_hash"}
        if i == 0:
            body["reason"] = "forged"
        body["prev_hash"] = prev
        body["entry_hash"] = hashlib.sha256(
            (prev + json.dumps(body, sort_keys=True)).encode()).hexdigest()
        ctrl.intervention_log[i] = body
        prev = body["entry_hash"]

    assert ctrl.verify_ledger_integrity() is False


# --- item 5: GradScaler ---------------------------------------------------

def _scaled_step(model, opt, ctrl, scaler, poison=None):
    x, y = _batch("cuda")
    loss = F.mse_loss(model(x), y)
    opt.zero_grad()
    scaler.scale(loss).backward()
    if poison is not None:
        next(model.parameters()).grad[0, 0] = poison
    action = ctrl.step(loss)
    if not action.should_skip:
        scaler.step(opt)
        scaler.update()
    return action


@pytest.mark.skipif(not CUDA, reason="GradScaler path needs CUDA")
def test_gradscaler_overflow_is_not_a_failure():
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 16)
    model, opt, ctrl = _setup(device="cuda", scaler=scaler)
    for _ in range(4):
        _scaled_step(model, opt, ctrl, scaler)
    scale_before = scaler.get_scale()

    action = _scaled_step(model, opt, ctrl, scaler, poison=float("inf"))

    assert not action.should_skip and not action.rolled_back
    assert ctrl.failures_detected == 0
    assert scaler.get_scale() < scale_before


@pytest.mark.skipif(not CUDA, reason="GradScaler path needs CUDA")
def test_gradscaler_gradients_are_checked_unscaled():
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 16)
    model, opt, ctrl = _setup(device="cuda", scaler=scaler)
    _scaled_step(model, opt, ctrl, scaler)
    x, y = _batch("cuda")
    loss = F.mse_loss(model(x), y)
    ref = torch.autograd.grad(loss, list(model.parameters()), retain_graph=True)
    opt.zero_grad()
    scaler.scale(loss).backward()

    assert not ctrl.step(loss).should_skip

    for r, p in zip(ref, model.parameters()):
        assert torch.allclose(r, p.grad, rtol=1e-4, atol=1e-7)
    scaler.step(opt)
    scaler.update()


@pytest.mark.skipif(not CUDA, reason="GradScaler path needs CUDA")
def test_scaler_stays_usable_after_post_unscale_failure():
    scaler = torch.amp.GradScaler("cuda")
    model, opt, ctrl = _setup(device="cuda", scaler=scaler)
    for _ in range(6):
        _scaled_step(model, opt, ctrl, scaler)
    threshold = ctrl.config.gradient_explosion_threshold
    ctrl.config.gradient_explosion_threshold = 1e-9
    assert _scaled_step(model, opt, ctrl, scaler).should_skip
    ctrl.config.gradient_explosion_threshold = threshold

    for _ in range(3):  # would raise "unscale_() has already been called" if leaked
        assert not _scaled_step(model, opt, ctrl, scaler).should_skip
