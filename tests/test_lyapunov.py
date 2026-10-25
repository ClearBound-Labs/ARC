"""LyapunovMonitor regressions (audit item 12)."""
import math

import pytest
import torch
import torch.nn as nn

from arc.signals.lyapunov import LyapunovMonitor, TrainingPhase


def test_velocity_is_measured_on_sampled_coordinates():
    model = nn.Linear(100, 100)  # 10,100 parameters
    mon = LyapunovMonitor(sample_ratio=0.5)
    mon.update(model)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    state = mon.update(model)

    # half of 10,100 coordinates, each moved by exactly 1.0
    assert state.velocity_norm == pytest.approx(math.sqrt(5050), rel=1e-6)


def test_lr_schedule_growth_is_not_divergence():
    torch.manual_seed(0)
    model = nn.Linear(100, 100)
    direction = [torch.randn_like(p) for p in model.parameters()]
    mon = LyapunovMonitor(sample_ratio=1.0)
    lr = 1e-3
    for _ in range(40):
        with torch.no_grad():
            for p, d in zip(model.parameters(), direction):
                p.add_(lr * d)
        state = mon.update(model, lr=lr)
        lr *= 1.2  # warmup-style growth: raw velocity grows 20% per check

    assert abs(state.exponent) < 1e-4
    assert state.phase != TrainingPhase.DIVERGENT


@pytest.mark.skipif(not torch.cuda.is_available(), reason="measures CUDA memory")
def test_sampling_does_not_copy_the_whole_model():
    model = nn.Sequential(nn.Linear(5000, 5000), nn.Linear(5000, 5000)).cuda()  # 200 MB fp32
    mon = LyapunovMonitor()
    mon.update(model)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    mon.update(model)
    torch.cuda.synchronize()

    assert torch.cuda.max_memory_allocated() - base < 50 * 1024 ** 2
