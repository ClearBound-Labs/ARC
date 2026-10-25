"""Multi-process Controller: every rank must make the same rollback decision."""
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from arc.core.controller import Controller, ControllerConfig


def _worker(rank, world_size, init_file, results):
    dist.init_process_group(
        "gloo", init_method=f"file:///{init_file}", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))
        ddp = nn.parallel.DistributedDataParallel(model)
        opt = torch.optim.SGD(ddp.parameters(), lr=0.01)
        ctrl = Controller(ddp, opt, config=ControllerConfig(
            verbose=False, auto_calibrate=False, enable_lyapunov=False,
            enable_trend=False, checkpoint_frequency=2))
        torch.manual_seed(rank)
        for _ in range(6):
            x, y = torch.randn(4, 8), torch.randn(4, 1)
            loss = F.mse_loss(ddp(x), y)
            opt.zero_grad()
            loss.backward()
            if not ctrl.step(loss).should_skip:
                opt.step()

        x, y = torch.randn(4, 8), torch.randn(4, 1)
        loss = F.mse_loss(ddp(x), y)
        opt.zero_grad()
        loss.backward()
        # only rank 0 sees a loss spike; its peers' losses look normal
        reported = loss * 1e6 if rank == 0 else loss
        action = ctrl.step(reported)
        weights = torch.cat([p.detach().flatten() for p in model.parameters()])
        results[rank] = (action.rolled_back, action.should_skip, weights.tolist())
    finally:
        dist.destroy_process_group()


def test_one_rank_failure_rolls_back_every_rank():
    world_size = 2
    with tempfile.TemporaryDirectory() as tmp:
        init_file = os.path.join(tmp, "init").replace("\\", "/")
        manager = mp.Manager()
        results = manager.dict()
        mp.spawn(_worker, args=(world_size, init_file, results), nprocs=world_size, join=True)

        r0, r1 = results[0], results[1]
    assert r0[0] and r1[0], "both ranks must roll back"
    assert r0[1] and r1[1], "both ranks must skip the optimizer step"
    assert r0[2] == r1[2], "replicas diverged"
