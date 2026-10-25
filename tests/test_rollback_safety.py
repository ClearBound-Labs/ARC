"""Legacy rollback engines: a detected failure must always skip the optimizer
step, and checkpoints must restore optimizer state exactly (audit items 4, 6)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from arc.api.integrations import GenericTrainingWrapper
from arc.checkpointing.adaptive import (
    AdaptiveCheckpointConfig, AdaptiveCheckpointer, CheckpointStrategy,
)
from arc.distributed.universal import UniversalDistributedRollback, UniversalDDPConfig
from arc.intervention.rollback import RollbackConfig, WeightRollback


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))


def _healthy_loss(model):
    return F.mse_loss(model(torch.randn(16, 8)), torch.randn(16, 1))


def _poisoned_backward(model, opt):
    loss = _healthy_loss(model)
    opt.zero_grad()
    loss.backward()
    next(model.parameters()).grad[0, 0] = float("nan")
    return loss


def test_nan_gradient_inside_cooldown_still_skips_the_step():
    model = _model()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    wrapper = GenericTrainingWrapper(model, opt, verbose=False)
    for _ in range(5):
        loss = _healthy_loss(model)
        opt.zero_grad()
        loss.backward()
        assert not wrapper.step(loss).skip_step
        opt.step()

    assert wrapper.step(_poisoned_backward(model, opt)).rolled_back
    second = wrapper.step(_poisoned_backward(model, opt))  # inside the 20-step cooldown

    assert not second.rolled_back
    assert second.skip_step


def test_failure_after_rollback_budget_still_skips():
    model = _model()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    rb = WeightRollback(model, opt, RollbackConfig(max_rollbacks_per_epoch=1, cooldown_steps=0),
                        verbose=False)
    rb.step(torch.tensor(float("nan")))

    action = rb.step(torch.tensor(float("nan")))

    assert not action.rolled_back
    assert action.should_skip


def test_universal_rollback_checks_nan_during_cooldown():
    model = _model()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    rb = UniversalDistributedRollback(model, opt, UniversalDDPConfig(
        sync_frequency=1, cooldown_steps=20, verbose=False))
    rb._save_checkpoint()
    assert rb.step(torch.tensor(float("nan"))).rolled_back

    action = rb.step(torch.tensor(float("nan")))

    assert action.should_skip


def _adam_with_tiny_second_moment(model):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = _healthy_loss(model)
    loss.backward()
    opt.step()
    for state in opt.state.values():
        state["exp_avg_sq"].fill_(1e-10)  # below fp16's smallest subnormal (6e-8)
    return opt


def test_universal_checkpoint_restores_optimizer_state_exactly():
    model = _model()
    opt = _adam_with_tiny_second_moment(model)
    rb = UniversalDistributedRollback(model, opt, UniversalDDPConfig(verbose=False))
    rb._save_checkpoint()
    for state in opt.state.values():
        state["exp_avg_sq"].fill_(5.0)

    rb._restore_checkpoint()

    for state in opt.state.values():
        assert torch.all(state["exp_avg_sq"] == 1e-10)


def test_fp16_checkpoint_keeps_optimizer_state_full_precision():
    model = _model()
    opt = _adam_with_tiny_second_moment(model)
    cp = AdaptiveCheckpointer(model, opt, AdaptiveCheckpointConfig(
        auto_select_strategy=False, preferred_strategy=CheckpointStrategy.QUANTIZED_FP16,
        verbose=False))
    cp.save()
    for state in opt.state.values():
        state["exp_avg_sq"].fill_(5.0)

    cp.restore()

    for state in opt.state.values():
        assert torch.all(state["exp_avg_sq"] == 1e-10)


def test_int8_checkpoint_restores_weights_within_quantization_error():
    model = _model()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    original = [p.detach().clone() for p in model.parameters()]
    cp = AdaptiveCheckpointer(model, opt, AdaptiveCheckpointConfig(
        auto_select_strategy=False, preferred_strategy=CheckpointStrategy.QUANTIZED_INT8,
        verbose=False))
    cp.save()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

    cp.restore()

    for o, p in zip(original, model.parameters()):
        step = o.abs().max() / 127  # symmetric per-tensor int8 grid
        assert (o - p.detach()).abs().max() <= step / 2 + 1e-7
