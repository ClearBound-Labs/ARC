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
from typing import Optional, Dict, Any, Callable
import math
from dataclasses import dataclass, field
from collections import deque
import copy
import warnings
from arc._logging import log as _log

@dataclass
class RollbackConfig:
    checkpoint_frequency: int = 50
    max_checkpoints: int = 3

    loss_explosion_threshold: float = 100.0
    gradient_explosion_threshold: float = 1e4
    nan_detection: bool = True

    lr_sanity_max: float = 10.0
    lr_spike_factor: float = 100.0
    loss_spike_threshold: float = 10.0
    loss_spike_window: int = 5

    lr_reduction_factor: float = 0.5
    max_rollbacks_per_epoch: int = 3
    cooldown_steps: int = 20

    async_checkpoint: bool = False
    incremental_checkpoint: bool = True
    adaptive_frequency: bool = True
    stable_loss_variance: float = 0.1
    min_check_frequency: int = 100

    fast_grad_norm: bool = True
    skip_stable_layers: bool = True
    layer_sample_ratio: float = 0.3

    weight_norm_monitoring: bool = True
    weight_norm_explosion_factor: float = 10.0

@dataclass
class RollbackState:
    step_count: int = 0
    rollback_count: int = 0
    last_rollback_step: int = -1000
    current_lr: float = 0.0
    initial_lr: float = 0.0
    checkpoints: deque = field(default_factory=lambda: deque(maxlen=3))
    loss_history: deque = field(default_factory=lambda: deque(maxlen=10))
    stable_step_count: int = 0
    current_check_frequency: int = 50
    stable_layers: set = field(default_factory=set)
    prev_weight_norms: Dict = field(default_factory=dict)

class WeightRollback:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[RollbackConfig] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config or RollbackConfig()
        self.verbose = verbose

        self.state = RollbackState()
        # Apply configured max_checkpoints to the deque (the dataclass default
        # hardcodes maxlen=3; this ensures the user-provided value is honoured).
        self.state.checkpoints = deque(maxlen=self.config.max_checkpoints)
        self._update_current_lr()
        self.state.initial_lr = self.state.current_lr
        # Cumulative product of every LR reduction. Restoring a checkpoint also
        # restores the optimizer's param_groups (lr included), which would undo
        # the reduction, so it is re-applied after each restore.
        self._lr_penalty: float = 1.0

        self._param_names = [name for name, _ in model.named_parameters()]
        self._n_params = len(self._param_names)

        self._save_checkpoint()

    def _update_current_lr(self):
        for pg in self.optimizer.param_groups:
            self.state.current_lr = pg['lr']
            break

    def _save_checkpoint(self):
        rng_state = {
            'torch': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        try:
            import numpy as np
            rng_state['numpy'] = np.random.get_state()
        except ImportError:
            rng_state['numpy'] = None

        checkpoint = {
            'step': self.state.step_count,
            'model_state': copy.deepcopy(self.model.state_dict()),
            'optimizer_state': copy.deepcopy(self.optimizer.state_dict()),
            'lr': self.state.current_lr,
            'lr_penalty': self._lr_penalty,
            'rng_state': rng_state,
        }
        self.state.checkpoints.append(checkpoint)

        if self.verbose and len(self.state.checkpoints) == 1:
            _log(f"   WeightRollback: Initial checkpoint saved (with RNG state)")

    def _restore_checkpoint(self, checkpoint_idx: int = -1) -> int:
        if not self.state.checkpoints:
            warnings.warn("No checkpoints available for rollback")
            return 0

        checkpoint = self.state.checkpoints[checkpoint_idx]

        self.model.load_state_dict(checkpoint['model_state'])
        # Copy: Optimizer.load_state_dict keeps same-device tensors as-is, so
        # training would otherwise mutate the stored checkpoint in place.
        self.optimizer.load_state_dict(copy.deepcopy(checkpoint['optimizer_state']))

        # load_state_dict brings back the LR in force when the checkpoint was
        # taken; re-apply only the reductions made since then.
        ratio = self._lr_penalty / checkpoint.get('lr_penalty', 1.0)
        if ratio != 1.0:
            for pg in self.optimizer.param_groups:
                pg['lr'] *= ratio
        self._update_current_lr()

        rng_state = checkpoint.get('rng_state', {})
        if rng_state.get('torch') is not None:
            torch.set_rng_state(rng_state['torch'])
        if rng_state.get('torch_cuda') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state['torch_cuda'])
        if rng_state.get('numpy') is not None:
            try:
                import numpy as np
                np.random.set_state(rng_state['numpy'])
            except ImportError:
                pass

        steps_back = self.state.step_count - checkpoint['step']

        if self.verbose:
            _log(f"  Rolled back {steps_back} steps to step {checkpoint['step']} (RNG restored)")

        # Reset detection state so stale pre-rollback losses don't immediately re-trigger
        self.state.loss_history.clear()
        self.state.stable_step_count = 0
        self.state.stable_layers.clear()

        return steps_back

    def _reduce_learning_rate(self):
        for pg in self.optimizer.param_groups:
            pg['lr'] *= self.config.lr_reduction_factor
            self.state.current_lr = pg['lr']
        self._lr_penalty *= self.config.lr_reduction_factor

        if self.verbose:
            _log(f"  Reduced LR to {self.state.current_lr:.2e}")

    def restore_checkpoint(self, checkpoint_idx: int = -1) -> int:
        """Public entry point for callers outside this class that need to
        force a rollback (e.g. in response to a signal WeightRollback itself
        didn't detect, like a silent-failure detector)."""
        return self._restore_checkpoint(checkpoint_idx)

    def reduce_learning_rate(self):
        """Public entry point for callers outside this class that need to
        force an LR reduction (see `restore_checkpoint`)."""
        self._reduce_learning_rate()

    def _detect_failure(self, loss: torch.Tensor) -> Dict[str, Any]:
        failure = {
            'detected': False,
            'type': None,
            'details': {},
        }

        if self.config.nan_detection:
            if torch.isnan(loss) or torch.isinf(loss):
                failure['detected'] = True
                failure['type'] = 'nan_loss'
                failure['details']['loss'] = float('nan') if torch.isnan(loss) else float('inf')
                return failure

        loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss

        if self.state.current_lr > self.config.lr_sanity_max:
            failure['detected'] = True
            failure['type'] = 'lr_insane'
            failure['details']['lr'] = self.state.current_lr
            failure['details']['max_allowed'] = self.config.lr_sanity_max
            return failure

        if self.state.initial_lr > 0:
            lr_ratio = self.state.current_lr / self.state.initial_lr
            if lr_ratio > self.config.lr_spike_factor:
                failure['detected'] = True
                failure['type'] = 'lr_spike'
                failure['details']['lr'] = self.state.current_lr
                failure['details']['initial_lr'] = self.state.initial_lr
                failure['details']['spike_factor'] = lr_ratio
                return failure

        if len(self.state.loss_history) >= 2:
            recent_avg = sum(list(self.state.loss_history)[-3:]) / min(3, len(self.state.loss_history))
            if recent_avg > 0 and loss_val / recent_avg > self.config.loss_spike_threshold:
                failure['detected'] = True
                failure['type'] = 'loss_spike'
                failure['details']['loss'] = loss_val
                failure['details']['recent_avg'] = recent_avg
                failure['details']['spike_factor'] = loss_val / recent_avg
                return failure

        if loss_val > self.config.loss_explosion_threshold:
            failure['detected'] = True
            failure['type'] = 'loss_explosion'
            failure['details']['loss'] = loss_val
            return failure

        if self.config.fast_grad_norm:
            grad_norm = self._compute_fast_grad_norm()
            if grad_norm is not None:
                if torch.isnan(torch.tensor(grad_norm)) or torch.isinf(torch.tensor(grad_norm)):
                    failure['detected'] = True
                    failure['type'] = 'nan_gradient'
                    return failure
                if grad_norm > self.config.gradient_explosion_threshold:
                    failure['detected'] = True
                    failure['type'] = 'gradient_explosion'
                    failure['details']['grad_norm'] = grad_norm
                    return failure
        else:
            max_grad = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        failure['detected'] = True
                        failure['type'] = 'nan_gradient'
                        return failure
                    max_grad = max(max_grad, p.grad.abs().max().item())

            if max_grad > self.config.gradient_explosion_threshold:
                failure['detected'] = True
                failure['type'] = 'gradient_explosion'
                failure['details']['max_grad'] = max_grad
                return failure

        params_to_check = self._get_params_to_check()
        for name, p in params_to_check:
            if torch.isnan(p).any() or torch.isinf(p).any():
                failure['detected'] = True
                failure['type'] = 'nan_weights'
                failure['details']['layer'] = name
                # A layer that just failed is not "known stable" -- make sure
                # it stays in the actively-checked rotation.
                self.state.stable_layers.discard(name)
                return failure

        # Weight norm deviation detection: catch subtle weight corruption
        if self.config.weight_norm_monitoring and self.state.step_count > 1:
            for name, p in params_to_check:
                current_norm = torch.linalg.vector_norm(p.data, ord=2).item()
                if name in self.state.prev_weight_norms:
                    prev_norm = self.state.prev_weight_norms[name]
                    if prev_norm > 0 and not math.isnan(current_norm):
                        ratio = current_norm / prev_norm
                        if ratio > self.config.weight_norm_explosion_factor:
                            failure['detected'] = True
                            failure['type'] = 'weight_norm_explosion'
                            failure['details']['layer'] = name
                            failure['details']['prev_norm'] = prev_norm
                            failure['details']['current_norm'] = current_norm
                            failure['details']['ratio'] = ratio
                            self.state.stable_layers.discard(name)
                            return failure
                self.state.prev_weight_norms[name] = current_norm
                # Passed both the NaN and weight-norm checks this round --
                # this is what "stable" means for _get_params_to_check's
                # sampling, which previously never had anything to read here
                # (stable_layers was never populated, so every step sampled
                # uniformly at random from ALL params instead of prioritizing
                # layers that haven't been vetted recently).
                self.state.stable_layers.add(name)

        return failure

    def _compute_fast_grad_norm(self) -> Optional[float]:
        grads = [p.grad.view(-1) for p in self.model.parameters() if p.grad is not None]
        if not grads:
            return None
        all_grads = torch.cat(grads)
        return torch.linalg.vector_norm(all_grads, ord=2).item()

    def _get_params_to_check(self):
        if not self.config.skip_stable_layers or self.state.step_count < 100:
            return list(self.model.named_parameters())

        import random
        all_params = list(self.model.named_parameters())
        n_sample = max(1, int(len(all_params) * self.config.layer_sample_ratio))

        unstable = [(n, p) for n, p in all_params if n not in self.state.stable_layers]
        if len(unstable) >= n_sample:
            return random.sample(unstable, n_sample)

        stable = [(n, p) for n, p in all_params if n in self.state.stable_layers]
        remaining = n_sample - len(unstable)
        if stable and remaining > 0:
            return unstable + random.sample(stable, min(remaining, len(stable)))

        return unstable

    def step(self, loss: torch.Tensor) -> 'RollbackAction':
        self.state.step_count += 1

        steps_since_rollback = self.state.step_count - self.state.last_rollback_step
        in_cooldown = steps_since_rollback < self.config.cooldown_steps

        failure = self._detect_failure(loss)

        # Any detected failure skips the optimizer step, even when a rollback
        # isn't allowed (cooldown, per-epoch budget): stepping on NaN/exploded
        # gradients would poison the weights.
        action = RollbackAction(
            step=self.state.step_count,
            rolled_back=False,
            failure_detected=failure['detected'],
            failure_type=failure.get('type'),
            should_skip=failure['detected'],
        )

        if failure['detected'] and not in_cooldown:
            if self.state.rollback_count < self.config.max_rollbacks_per_epoch:
                steps_back = self._restore_checkpoint()
                self._reduce_learning_rate()

                self.state.rollback_count += 1
                self.state.last_rollback_step = self.state.step_count

                action.rolled_back = True
                action.steps_back = steps_back
                action.new_lr = self.state.current_lr

                if self.verbose:
                    _log(f"  Rollback #{self.state.rollback_count}: {failure['type']}")
            else:
                if self.verbose:
                    _log(f"  Max rollbacks reached, not rolling back")

        if self.state.step_count % self.config.checkpoint_frequency == 0:
            if not failure['detected']:
                self._save_checkpoint()

        return action

    def end_epoch(self):
        self.state.rollback_count = 0

        self._save_checkpoint()

    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_steps': self.state.step_count,
            'total_rollbacks': self.state.rollback_count,
            'checkpoints_saved': len(self.state.checkpoints),
            'current_lr': self.state.current_lr,
        }

@dataclass
class RollbackAction:
    step: int
    rolled_back: bool
    failure_detected: bool
    failure_type: Optional[str] = None
    steps_back: int = 0
    new_lr: float = 0.0
    should_skip: bool = False

def create_self_healing_arc(model, optimizer, safety_level="standard"):
    from arc import ArcV2

    arc = ArcV2.auto(model, optimizer, safety_level=safety_level)
    rollback = WeightRollback(model, optimizer)

    return arc, rollback

if __name__ == "__main__":
    print("Testing WeightRollback...")

    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    rollback = WeightRollback(model, optimizer, verbose=True)

    for step in range(100):
        x = torch.randn(4, 10)
        y = model(x)
        loss = y.mean()

        if step == 50:
            loss = torch.tensor(float('inf'))

        loss.backward() if not torch.isinf(loss) else None

        action = rollback.step(loss)

        if action.rolled_back:
            print(f"  Step {step}: ROLLED BACK")
            continue

        optimizer.step()
        optimizer.zero_grad()

    print(f"\nStats: {rollback.get_stats()}")
    print("  WeightRollback test complete")