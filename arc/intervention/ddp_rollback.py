# ARC (Automatic Recovery Controller) - Self-Healing Neural Networks
# Copyright (c) 2026 Aryan Kaushik. All rights reserved.
#
# This file is part of ARC.
#
# ARC is free software: you can redistribute it and/or modify it under the
# terms of the GNU Affero General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option)
# any later version.
#
# ARC is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for
# more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ARC. If not, see <https://www.gnu.org/licenses/>.

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from arc.config import default_checkpoint_dir
import os
import warnings
from arc._logging import log as _log

@dataclass
class DDPConfig:
    # Default 1 (every step), not a larger interval: DistributedDataParallel
    # all-reduces gradients on every backward pass regardless of this setting,
    # so a NaN on one rank contaminates every rank's gradients immediately.
    # A larger sync_frequency here only delays *this* class's detection/rollback
    # after that contamination has already happened -- it doesn't prevent it.
    # The failure-sync itself is a single-element all_reduce, negligible next
    # to the gradient all-reduce DDP already performs every step, so there's
    # no real perf reason to raise this above 1. Left configurable for users
    # who explicitly want to trade detection latency for less sync overhead.
    sync_frequency: int = 1
    use_all_reduce: bool = True

    coordinator_rank: int = 0
    barrier_timeout_seconds: float = 300.0

    checkpoint_dir: str = default_checkpoint_dir("ddp_checkpoints")
    save_to_disk: bool = True

class DDPRollback:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[DDPConfig] = None,
        verbose: bool = True,
    ):
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before DDPRollback")

        self.model = model
        self.optimizer = optimizer
        self.config = config or DDPConfig()
        self.verbose = verbose

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.is_coordinator = (self.rank == self.config.coordinator_rank)

        from arc.intervention.rollback import WeightRollback, RollbackConfig

        local_config = RollbackConfig(
            fast_grad_norm=True,
            checkpoint_frequency=100,
        )
        self._local_rollback = WeightRollback(
            model.module if hasattr(model, 'module') else model,
            optimizer,
            local_config,
            verbose=False,
        )

        self._step_count = 0
        self._rollback_count = 0

        if self.config.sync_frequency > 1:
            warnings.warn(
                f"DDPRollback sync_frequency={self.config.sync_frequency} > 1: "
                "a NaN/exploded gradient on one rank propagates to all ranks via "
                "DDP's own per-step gradient all-reduce immediately, but this "
                "class's failure detection and rollback won't run until the next "
                "sync boundary (up to sync_frequency - 1 steps later). Only raise "
                "this above 1 if you've deliberately accepted that detection lag.",
                stacklevel=2,
            )

        if self.is_coordinator and self.config.save_to_disk:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        if verbose and self.is_coordinator:
            _log(f" DDPRollback: Initialized across {self.world_size} ranks")

    def _detect_any_failure(self, loss: torch.Tensor) -> bool:
        local_failure = self._local_rollback._detect_failure(loss)
        has_local_failure = 1.0 if local_failure['detected'] else 0.0

        failure_tensor = torch.tensor([has_local_failure], device=loss.device)
        dist.all_reduce(failure_tensor, op=dist.ReduceOp.MAX)

        return failure_tensor.item() > 0.5

    def _coordinated_rollback(self) -> int:
        dist.barrier()

        steps_back = self._local_rollback._restore_checkpoint()

        self._local_rollback._reduce_learning_rate()

        dist.barrier()

        self._rollback_count += 1

        if self.verbose and self.is_coordinator:
            _log(f" DDP Rollback #{self._rollback_count}: All {self.world_size} ranks restored")

        return steps_back

    def step(self, loss: torch.Tensor) -> 'DDPRollbackAction':
        self._step_count += 1

        action = DDPRollbackAction(
            step=self._step_count,
            rolled_back=False,
            rank=self.rank,
        )

        if self._step_count % self.config.sync_frequency == 0:
            any_failure = self._detect_any_failure(loss)

            if any_failure:
                steps_back = self._coordinated_rollback()
                action.rolled_back = True
                action.steps_back = steps_back
                action.all_ranks_rolled_back = True

        if self._step_count % self._local_rollback.config.checkpoint_frequency == 0:
            self._local_rollback._save_checkpoint()

        return action

    def get_stats(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "total_rollbacks": self._rollback_count,
            "steps": self._step_count,
        }

@dataclass
class DDPRollbackAction:
    step: int
    rolled_back: bool
    rank: int
    steps_back: int = 0
    all_ranks_rolled_back: bool = False

class PartialRankFailureHandler:

    @staticmethod
    def check_consistency(local_state: Dict[str, Any]) -> bool:
        if not dist.is_initialized():
            return True

        hash_tensor = torch.tensor([0.0], device='cuda' if torch.cuda.is_available() else 'cpu')
        for param in local_state.get('params', []):
            hash_tensor += param.sum()

        all_hashes = [torch.zeros_like(hash_tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(all_hashes, hash_tensor)

        reference = all_hashes[0]
        return all(torch.allclose(h, reference, atol=1e-6) for h in all_hashes)
def example_ddp_training():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    model = nn.Linear(10, 2).cuda(local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters())

    config = DDPConfig(sync_frequency=10)
    rollback = DDPRollback(model, optimizer, config)

    for step in range(100):
        x = torch.randn(4, 10).cuda(local_rank)
        loss = model(x).mean()

        if local_rank == 0 and step == 50:
            loss = loss * float('inf')

        loss.backward()

        action = rollback.step(loss)
        if action.rolled_back:
            optimizer.zero_grad()
            continue

        optimizer.step()
        optimizer.zero_grad()

    if local_rank == 0:
        _log(f"Training complete: {rollback.get_stats()}")

    dist.destroy_process_group()

if __name__ == "__main__":
    print("DDP Rollback Prototype")
    print("=" * 60)
    print("This module provides coordinated multi-GPU rollback.")
    print("Status: EXPERIMENTAL")
    print()
    print("To test: torchrun --nproc_per_node=2 ddp_rollback.py")