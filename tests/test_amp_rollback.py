"""AMPRollback must judge gradients at their true scale (audit item 5)."""
import pytest
import torch
import torch.nn as nn

from arc.intervention.amp import AMPRollback

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _scaled_steps(model, opt, amp, scaler, n, loss_scale=20.0):
    actions = []
    dtype = next(model.parameters()).dtype
    for _ in range(n):
        x = torch.randn(32, 64, device="cuda", dtype=dtype)
        loss = model(x).float().pow(2).mean() * loss_scale
        opt.zero_grad()
        scaler.scale(loss).backward()
        action = amp.step(loss)
        actions.append(action)
        if not action.should_skip:
            scaler.step(opt)
            scaler.update()
    return actions


def test_healthy_scaled_training_is_never_rolled_back():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8)).cuda()
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 16)
    amp = AMPRollback(model, opt, scaler, verbose=False)

    # Unscaled gradients are O(1); scaled by 65536 they exceed fp16's max.
    x = torch.randn(32, 64, device="cuda")
    (model(x).pow(2).mean() * 20.0).backward()
    assert max(p.grad.abs().max().item() for p in model.parameters()) > 1.0
    opt.zero_grad()

    actions = _scaled_steps(model, opt, amp, scaler, 20)

    assert not any(a.rolled_back for a in actions)


def test_fp16_weight_near_overflow_rolls_back():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8)).cuda().half()
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    amp = AMPRollback(model, opt, scaler=None, verbose=False)
    _scaled_steps(model, opt, amp, torch.amp.GradScaler("cuda", enabled=False), 3,
                  loss_scale=1e-3)
    with torch.no_grad():
        model[0].weight[0, 0] = 65000.0

    x = torch.randn(32, 64, device="cuda").half()
    loss = model(x).float().pow(2).mean() * 1e-3
    opt.zero_grad()
    loss.backward()
    action = amp.step(loss)

    assert action.overflow_detected and action.rolled_back
    assert model[0].weight[0, 0].item() != 65000.0
