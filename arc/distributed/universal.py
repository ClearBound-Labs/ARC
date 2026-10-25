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

import copy
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, field
from arc.config import default_checkpoint_dir
import os
import time
import warnings
from arc._logging import log as _log


def _state_to_cpu(obj: Any) -> Any:
    """Full-precision CPU copy of a (nested) optimizer state dict."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: _state_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_state_to_cpu(v) for v in obj)
    return copy.deepcopy(obj)

@dataclass
class UniversalDDPConfig:
    sync_frequency: int = 5
    checkpoint_frequency: int = 50
    barrier_timeout_seconds: float = 60.0

    loss_explosion_threshold: float = 100.0
    gradient_explosion_threshold: float = 1e4
    nan_detection: bool = True

    lr_reduction_factor: float = 0.5
    max_rollbacks_per_epoch: int = 5
    cooldown_steps: int = 20

    allow_single_device_fallback: bool = True
    allow_cpu_fallback: bool = True

    checkpoint_dir: str = default_checkpoint_dir("universal_checkpoints")
    # Opt-in: stores model *weights* as fp16 (optimizer state is always kept
    # at full precision -- Adam's second moment underflows to 0 in fp16).
    use_quantized_checkpoints: bool = False

    auto_detect_backend: bool = True
    preferred_backend: Optional[str] = None

    verbose: bool = True

@dataclass
class DistributedState:
    is_distributed: bool = False
    backend: str = "none"
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    is_main_process: bool = True
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    step_count: int = 0
    rollback_count: int = 0
    last_rollback_step: int = -1000

@dataclass
class DistributedAction:
    step: int
    rolled_back: bool
    failure_detected: bool = False
    # True whenever a failure was detected on any rank, rollback or not:
    # the optimizer step must not run on poisoned gradients.
    should_skip: bool = False
    failure_rank: int = -1
    all_ranks_synced: bool = True
    details: Dict[str, Any] = field(default_factory=dict)

class UniversalDistributedRollback:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[UniversalDDPConfig] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config or UniversalDDPConfig()

        self.state = self._detect_distributed_state()

        self._checkpoints: List[Dict[str, Any]] = []
        self._max_checkpoints = 3

        self._loss_history: List[float] = []
        # Cumulative LR reduction, re-applied after every restore because
        # load_state_dict brings the checkpoint's param_groups (lr) back.
        self._lr_penalty: float = 1.0
        self._initial_loss: Optional[float] = None

        self._in_cooldown = False
        self._cooldown_end_step = 0

        if self.state.is_main_process:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        if self.state.is_distributed:
            self._barrier()

        if self.config.verbose and self.state.is_main_process:
            self._print_init_info()

    def _detect_distributed_state(self) -> DistributedState:
        state = DistributedState()

        if dist.is_initialized():
            state.is_distributed = True
            state.backend = dist.get_backend()
            state.rank = dist.get_rank()
            state.world_size = dist.get_world_size()
            state.local_rank = int(os.environ.get('LOCAL_RANK', state.rank))
            state.is_main_process = (state.rank == 0)
        else:
            state.is_distributed = False
            state.world_size = 1
            state.rank = 0
            state.is_main_process = True

        # Where the model actually lives, not where a GPU happens to exist.
        try:
            state.device = next(self.model.parameters()).device
        except StopIteration:
            state.device = torch.device("cpu")

        return state

    def _print_init_info(self):
        _log("=" * 60)
        _log("UniversalDistributedRollback Initialized")
        _log("=" * 60)
        _log(f"   Mode: {'Distributed' if self.state.is_distributed else 'Single Device'}")
        if self.state.is_distributed:
            _log(f"   Backend: {self.state.backend}")
            _log(f"   World Size: {self.state.world_size}")
            _log(f"   Rank: {self.state.rank}")
        _log(f"   Device: {self.state.device}")
        _log("=" * 60)

    def _barrier(self):
        if not self.state.is_distributed:
            return

        try:
            if hasattr(dist, 'barrier'):
                dist.barrier()
        except Exception as e:
            warnings.warn(f"Barrier failed: {e}")

    def _detect_local_failure(self, loss: torch.Tensor) -> Dict[str, Any]:
        loss_val = loss.detach().item() if isinstance(loss, torch.Tensor) else loss

        result = {
            "detected": False,
            "type": None,
            "details": {},
        }

        if self.config.nan_detection:
            if not torch.isfinite(loss).all():
                result["detected"] = True
                result["type"] = "nan_inf"
                return result

        self._loss_history.append(loss_val)
        if len(self._loss_history) > 10:
            self._loss_history.pop(0)

        if self._initial_loss is None:
            self._initial_loss = loss_val

        if self._initial_loss is not None and self._initial_loss > 0:
            if loss_val > self._initial_loss * self.config.loss_explosion_threshold:
                result["detected"] = True
                result["type"] = "loss_explosion"
                result["details"]["loss_ratio"] = loss_val / self._initial_loss
                return result

        norms = [p.grad.detach().norm() for p in self.model.parameters() if p.grad is not None]
        total_norm = torch.linalg.vector_norm(torch.stack(norms)).item() if norms else 0.0

        if self.config.nan_detection and not math.isfinite(total_norm):
            result["detected"] = True
            result["type"] = "nan_inf_gradients"
            return result

        if total_norm > self.config.gradient_explosion_threshold:
            result["detected"] = True
            result["type"] = "gradient_explosion"
            result["details"]["grad_norm"] = total_norm
            return result

        return result

    def _aggregate_failures(self, local_failure: Dict[str, Any]) -> bool:
        if not self.state.is_distributed:
            return local_failure["detected"]

        failure_flag = 1.0 if local_failure["detected"] else 0.0
        failure_tensor = torch.tensor([failure_flag], device=self.state.device)

        try:
            dist.all_reduce(failure_tensor, op=dist.ReduceOp.MAX)
        except Exception as e:
            warnings.warn(f"All-reduce failed: {e}, assuming local failure only")
            return local_failure["detected"]

        return failure_tensor.item() > 0.5

    def _save_checkpoint(self):
        model = self.model.module if hasattr(self.model, 'module') else self.model

        weights = {}
        for name, param in model.state_dict().items():
            if self.config.use_quantized_checkpoints and param.is_floating_point():
                weights[name] = param.detach().cpu().half()
            else:
                weights[name] = param.detach().cpu().clone()

        checkpoint = {
            'step': self.state.step_count,
            'model': weights,
            'optimizer_state_dict': _state_to_cpu(self.optimizer.state_dict()),
            'lr_penalty': self._lr_penalty,
            'rng': {'torch': torch.get_rng_state()},
        }
        if torch.cuda.is_available():
            checkpoint['rng']['cuda'] = torch.cuda.get_rng_state()

        self._checkpoints.append(checkpoint)
        if len(self._checkpoints) > self._max_checkpoints:
            self._checkpoints.pop(0)

    def _restore_checkpoint(self, idx: int = -1) -> int:
        if not self._checkpoints:
            return 0

        checkpoint = self._checkpoints[idx]
        model = self.model.module if hasattr(self.model, 'module') else self.model

        # load_state_dict copies into the live tensors, keeping their dtype.
        model.load_state_dict({
            name: value.to(device=self.state.device)
            for name, value in checkpoint['model'].items()
        })

        if 'optimizer_state_dict' in checkpoint:
            optimizer_state_dict = copy.deepcopy(checkpoint['optimizer_state_dict'])
        else:  # checkpoints written before optimizer_state_dict existed
            optimizer_state_dict = {
                'state': copy.deepcopy(checkpoint.get('optimizer', {})),
                'param_groups': checkpoint.get(
                    'optimizer_param_groups', self.optimizer.state_dict()['param_groups']),
            }
        self.optimizer.load_state_dict(optimizer_state_dict)

        # Re-apply only the LR reductions made since this checkpoint was taken.
        ratio = self._lr_penalty / checkpoint.get('lr_penalty', 1.0)
        if ratio != 1.0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= ratio

        torch.set_rng_state(checkpoint['rng']['torch'])
        if 'cuda' in checkpoint['rng'] and torch.cuda.is_available():
            torch.cuda.set_rng_state(checkpoint['rng']['cuda'])

        # Reset loss history so pre-rollback spikes don't re-trigger detection
        self._loss_history.clear()

        return checkpoint['step']

    def _reduce_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.config.lr_reduction_factor
        self._lr_penalty *= self.config.lr_reduction_factor

    def _coordinated_rollback(self) -> int:
        self._barrier()

        restored_step = self._restore_checkpoint()

        self._reduce_learning_rate()

        self._barrier()

        self.state.rollback_count += 1
        self.state.last_rollback_step = self.state.step_count

        self._in_cooldown = True
        self._cooldown_end_step = self.state.step_count + self.config.cooldown_steps

        return restored_step

    def step(self, loss: torch.Tensor) -> DistributedAction:
        self.state.step_count += 1

        action = DistributedAction(
            step=self.state.step_count,
            rolled_back=False,
        )

        if self._in_cooldown and self.state.step_count >= self._cooldown_end_step:
            self._in_cooldown = False

        if self.state.step_count % self.config.sync_frequency == 0:
            local_failure = self._detect_local_failure(loss)
            any_failure = self._aggregate_failures(local_failure)

            if any_failure:
                action.failure_detected = True
                action.should_skip = True

                if self._in_cooldown:
                    # No second rollback this soon, but the step still must not run.
                    return action

                if self.state.rollback_count >= self.config.max_rollbacks_per_epoch:
                    if self.config.verbose and self.state.is_main_process:
                        _log("Max rollbacks reached, skipping recovery")
                    return action

                steps_back = self._coordinated_rollback()
                action.rolled_back = True
                action.details['steps_back'] = steps_back
                action.details['new_lr'] = self.optimizer.param_groups[0]['lr']

                if self.config.verbose and self.state.is_main_process:
                    _log(f"Rollback #{self.state.rollback_count}: Restored to step {steps_back}")

        if self.state.step_count % self.config.checkpoint_frequency == 0:
            self._save_checkpoint()

        return action

    def end_epoch(self):
        self.state.rollback_count = 0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "is_distributed": self.state.is_distributed,
            "backend": self.state.backend,
            "rank": self.state.rank,
            "world_size": self.state.world_size,
            "step_count": self.state.step_count,
            "rollback_count": self.state.rollback_count,
            "checkpoints": len(self._checkpoints),
        }

def setup_distributed(backend: str = "auto") -> DistributedState:
    if dist.is_initialized():
        pass
    elif 'WORLD_SIZE' in os.environ:
        if backend == "auto":
            if torch.cuda.is_available():
                backend = "nccl"
            else:
                backend = "gloo"

        dist.init_process_group(backend=backend)

    state = DistributedState()

    if dist.is_initialized():
        state.is_distributed = True
        state.backend = dist.get_backend()
        state.rank = dist.get_rank()
        state.world_size = dist.get_world_size()
        state.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        state.is_main_process = (state.rank == 0)

        if torch.cuda.is_available():
            torch.cuda.set_device(state.local_rank)
            state.device = torch.device(f"cuda:{state.local_rank}")
    else:
        state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return state

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()