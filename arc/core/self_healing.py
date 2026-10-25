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
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from collections import deque
import copy
import time
import math
import warnings
from arc._logging import log as _log

@dataclass
class SelfHealingConfig:
    checkpoint_frequency: int = 20
    max_checkpoints: int = 3
    cpu_offload: bool = True

    loss_explosion_threshold: float = 100.0
    loss_nan_action: str = "rollback"
    gradient_explosion_threshold: float = 1e5
    weight_nan_action: str = "rollback"

    lr_reduction_factor: float = 0.1
    max_lr_reductions: int = 5
    gradient_clip_norm: float = 1.0
    # Opt-in: clipping changes the optimisation itself.
    enable_baseline_clipping: bool = False

    enable_forecasting: bool = True
    forecast_window: int = 10
    forecast_threshold: float = 2.0

    adaptive_checkpoint_frequency: bool = True
    min_checkpoint_frequency: int = 5
    max_checkpoint_frequency: int = 100

    lite_mode: bool = False
    check_frequency: int = 1

    # Silent weight corruption detection
    enable_weight_health_check: bool = True
    weight_sparsity_threshold: float = 0.3  # Max fraction of zeros allowed
    weight_magnitude_collapse_threshold: float = 0.01  # Min fraction of original magnitude
    weight_health_check_frequency: int = 10  # Check every N steps

    # Persistent failure recovery
    enable_persistent_failure_recovery: bool = True
    max_consecutive_rollbacks: int = 5  # Trigger recovery after N consecutive rollbacks
    skip_ahead_steps: int = 5  # Steps to skip forward when stuck
    weight_perturbation_scale: float = 0.01  # Scale of random perturbation to escape

    verbose: bool = True
    log_interventions: bool = True

@dataclass
class HealingAction:
    step: int
    should_skip: bool = False
    rolled_back: bool = False
    lr_reduced: bool = False
    gradients_clipped: bool = False
    failure_detected: str = None
    loss_value: float = 0.0
    health_score: float = 1.0
    message: str = ""

class SelfHealingArc:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[SelfHealingConfig] = None,
    ):
        warnings.warn(
            "SelfHealingArc is a legacy API kept for compatibility; use arc.Controller, "
            "which covers everything it does (plus GradScaler, DDP and honest recovery stats).",
            FutureWarning, stacklevel=2,
        )
        self.model = model
        self.optimizer = optimizer
        self.config = config or SelfHealingConfig()

        self.step_count = 0
        self.epoch_count = 0
        self.current_lr = self._get_lr()
        self.original_lr = self.current_lr
        self.lr_reductions = 0
        # Cumulative LR reduction, re-applied after every checkpoint restore
        # (load_state_dict would otherwise resurrect the pre-reduction lr).
        self._lr_penalty = 1.0

        self.checkpoints: List[Tuple[int, dict, dict]] = []
        self._save_checkpoint()

        self.loss_history: deque = deque(maxlen=100)
        self.gradient_history: deque = deque(maxlen=self.config.forecast_window)

        self.total_rollbacks = 0
        self.total_skips = 0
        self.failures_detected = 0
        self.failures_recovered = 0

        self.intervention_log: List[Dict[str, Any]] = []

        self._stability_score = 1.0
        self._current_checkpoint_frequency = self.config.checkpoint_frequency

        self._param_list = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

        # Capture baseline weight statistics for health monitoring
        self._baseline_weight_stats = self._compute_weight_stats()

        # Persistent failure recovery tracking
        self._consecutive_rollbacks = 0
        self._last_rollback_step = -1
        self._skip_until_step = 0
        self._persistent_failure_count = 0

        if self.config.verbose:
            n_params = sum(p.numel() for _, p in self._param_list)
            _log(f"    SelfHealingArc initialized: {n_params/1e6:.2f}M params")

    def _get_lr(self) -> float:
        for pg in self.optimizer.param_groups:
            return pg['lr']
        return 0.0

    def _set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        self.current_lr = lr

    def _save_checkpoint(self):
        if len(self.checkpoints) >= self.config.max_checkpoints:
            self.checkpoints.pop(0)

        if self.config.cpu_offload:
            model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            opt_state = copy.deepcopy(self.optimizer.state_dict())
            for state in opt_state.get('state', {}).values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cpu()
        else:
            model_state = copy.deepcopy(self.model.state_dict())
            opt_state = copy.deepcopy(self.optimizer.state_dict())

        self.checkpoints.append((self.step_count, model_state, opt_state, self._lr_penalty))

    def _restore_checkpoint(self, index: int = -1) -> bool:
        if not self.checkpoints:
            return False

        try:
            step, model_state, opt_state, penalty_at_save = self.checkpoints[index]

            device = next(self.model.parameters()).device

            # Always a copy: Optimizer.load_state_dict keeps same-device tensors
            # as-is, so training would otherwise mutate the stored checkpoint.
            opt_state = copy.deepcopy(opt_state)
            if self.config.cpu_offload:
                model_state = {k: v.to(device) for k, v in model_state.items()}
                for state in opt_state.get('state', {}).values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)

            self.model.load_state_dict(model_state)
            self.optimizer.load_state_dict(opt_state)

            # The restored param_groups carry the LR in force at save time;
            # re-apply only the reductions made since then.
            ratio = self._lr_penalty / penalty_at_save
            self._set_lr(self._get_lr() * ratio)

            if self.config.verbose:
                _log(f"      Rolled back to step {step}")

            return True
        except Exception as e:
            if self.config.verbose:
                _log(f"      Rollback failed: {e}")
            return False

    def _compute_gradient_norm(self) -> float:
        total_norm = 0.0
        for _, p in self._param_list:
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return math.sqrt(total_norm)

    def _check_weights_for_nan(self) -> bool:
        for _, p in self._param_list:
            if torch.isnan(p).any() or torch.isinf(p).any():
                return True
        return False

    def _check_gradients_for_nan(self) -> bool:
        for _, p in self._param_list:
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    return True
        return False

    def _compute_weight_stats(self) -> Dict[str, float]:
        """Compute weight statistics for health monitoring."""
        total_params = 0
        total_zeros = 0
        total_magnitude = 0.0
        
        for _, p in self._param_list:
            numel = p.numel()
            total_params += numel
            total_zeros += (p.data == 0).sum().item()
            total_magnitude += p.data.abs().sum().item()
        
        return {
            "total_params": total_params,
            "zero_fraction": total_zeros / max(total_params, 1),
            "mean_magnitude": total_magnitude / max(total_params, 1),
        }

    def _check_weight_health(self) -> Tuple[bool, str]:
        """Check for silent weight corruption (sparsity explosion, magnitude collapse)."""
        if not self.config.enable_weight_health_check:
            return False, ""
        
        current_stats = self._compute_weight_stats()
        baseline = self._baseline_weight_stats
        
        # Check for sparsity explosion (too many zeros)
        if current_stats["zero_fraction"] > self.config.weight_sparsity_threshold:
            # Only trigger if this is a significant change from baseline
            if current_stats["zero_fraction"] > baseline["zero_fraction"] + 0.1:
                return True, f"weight_sparsity_explosion (zeros: {current_stats['zero_fraction']:.1%})"
        
        # Check for magnitude collapse
        if baseline["mean_magnitude"] > 0:
            magnitude_ratio = current_stats["mean_magnitude"] / baseline["mean_magnitude"]
            if magnitude_ratio < self.config.weight_magnitude_collapse_threshold:
                return True, f"weight_magnitude_collapse (ratio: {magnitude_ratio:.4f})"
        
        return False, ""

    def _predict_explosion(self) -> Tuple[bool, float]:
        if len(self.gradient_history) < 3:
            return False, 1.0

        history = list(self.gradient_history)

        if history[-2] > 0:
            growth = history[-1] / history[-2]
        else:
            growth = 1.0

        will_explode = growth > self.config.forecast_threshold

        return will_explode, growth

    def _reduce_lr(self):
        if self.lr_reductions < self.config.max_lr_reductions:
            new_lr = self.current_lr * self.config.lr_reduction_factor
            self._set_lr(new_lr)
            self.lr_reductions += 1
            self._lr_penalty *= self.config.lr_reduction_factor

            if self.config.verbose:
                _log(f"      LR reduced to {new_lr:.2e}")

    def _track_rollback(self) -> bool:
        """Track consecutive rollbacks and detect persistent failure loop.
        
        Returns True if we're stuck in a persistent failure loop.
        """
        if not self.config.enable_persistent_failure_recovery:
            return False
        
        # Check if this rollback is consecutive (within 3 steps of the last)
        if self._last_rollback_step >= 0 and self.step_count - self._last_rollback_step <= 3:
            self._consecutive_rollbacks += 1
        else:
            self._consecutive_rollbacks = 1
        
        self._last_rollback_step = self.step_count
        
        # Detect persistent failure loop
        return self._consecutive_rollbacks >= self.config.max_consecutive_rollbacks
    
    def _escape_persistent_failure(self) -> bool:
        """Escape from a persistent failure loop by skipping ahead and perturbing weights.
        
        Returns True if escape was successful.
        """
        if not self.config.enable_persistent_failure_recovery:
            return False
        
        self._persistent_failure_count += 1
        
        # 1. Skip ahead to escape the failure region
        self._skip_until_step = self.step_count + self.config.skip_ahead_steps
        
        # 2. Restore from oldest checkpoint (go further back)
        if len(self.checkpoints) > 1:
            if self._restore_checkpoint(index=0):  # Oldest checkpoint
                pass
        
        # 3. Perturb weights to escape local minimum / failure region
        self._perturb_weights()
        
        # 4. More aggressive LR reduction
        for _ in range(2):
            self._reduce_lr()
        
        # 5. Reset consecutive rollback counter
        self._consecutive_rollbacks = 0
        
        # 6. Update baseline weight stats (new starting point)
        self._baseline_weight_stats = self._compute_weight_stats()
        
        if self.config.verbose:
            _log(f"      ⚡ PERSISTENT FAILURE ESCAPE: Skip to step {self._skip_until_step}, weights perturbed")
        
        return True
    
    def _perturb_weights(self):
        """Add small random perturbation to weights to escape failure region."""
        scale = self.config.weight_perturbation_scale
        
        with torch.no_grad():
            for _, p in self._param_list:
                # Add scaled random noise
                noise = torch.randn_like(p) * scale * (p.abs().mean() + 1e-8)
                p.add_(noise)
    
    def _should_skip_step(self) -> bool:
        """Check if we should skip this step (part of escape mechanism)."""
        if self.step_count < self._skip_until_step:
            return True
        return False

    def _log_intervention(self, reason: str, loss_val: float, action: str):
        if self.config.log_interventions:
            import datetime
            self.intervention_log.append({
                "timestamp": datetime.datetime.now().isoformat(),
                "step": self.step_count,
                "reason": reason,
                "loss_value": "NaN" if math.isnan(loss_val) else loss_val,
                "action": action,
                "new_lr": self.current_lr,
                "stability_score": self._stability_score,
                "checkpoint_step": self.checkpoints[-1][0] if self.checkpoints else None,
            })

    def _update_stability_score(self, loss: float):
        if len(self.loss_history) < 5:
            return

        recent = list(self.loss_history)[-10:]
        mean_loss = sum(recent) / len(recent)

        if mean_loss > 0:
            variance = sum((x - mean_loss) ** 2 for x in recent) / len(recent)
            cv = math.sqrt(variance) / mean_loss

            self._stability_score = max(0.0, 1.0 - cv)

        if self.config.adaptive_checkpoint_frequency:
            if self._stability_score < 0.5:
                self._current_checkpoint_frequency = self.config.min_checkpoint_frequency
            elif self._stability_score > 0.9:
                self._current_checkpoint_frequency = self.config.max_checkpoint_frequency
            else:
                self._current_checkpoint_frequency = int(
                    self.config.min_checkpoint_frequency +
                    self._stability_score * (self.config.max_checkpoint_frequency - self.config.min_checkpoint_frequency)
                )

    def _handle_failure(self, action: HealingAction, failure_type: str, action_config: str = "rollback") -> HealingAction:
        """Unified failure handling: detect -> rollback -> heal.

        Mirrors Controller._handle_failure. Replaces four copy-pasted
        rollback/lr-reduce/track/escape blocks that previously only the
        nan_loss branch gated on config (loss_nan_action); nan_weights now
        consistently respects the (previously unused) weight_nan_action.
        """
        action.failure_detected = failure_type
        action.should_skip = True
        self.failures_detected += 1

        if action_config == "rollback":
            if self._restore_checkpoint():
                action.rolled_back = True
                self.total_rollbacks += 1
                self.failures_recovered += 1
                self._reduce_lr()
                action.lr_reduced = True

                if self._track_rollback():
                    self._escape_persistent_failure()
                    action.message = "Persistent failure detected, escaping"

        return action

    def step(self, loss: torch.Tensor) -> HealingAction:
        self.step_count += 1

        action = HealingAction(
            step=self.step_count,
            loss_value=0.0,
            health_score=self._stability_score,
        )

        # Check if we should skip this step (part of escape mechanism)
        if self._should_skip_step():
            action.should_skip = True
            action.message = "Skipping step (escape mode)"
            self.total_skips += 1
            return action

        if torch.isnan(loss) or torch.isinf(loss):
            self._handle_failure(action, "nan_loss", self.config.loss_nan_action)

            if self.config.verbose:
                _log(f"      NaN/Inf loss detected at step {self.step_count}")

            return action

        loss_val = loss.item()
        action.loss_value = loss_val

        if loss_val > self.config.loss_explosion_threshold:
            if self.loss_history and loss_val > self.loss_history[-1] * 10:
                self._handle_failure(action, "loss_explosion")

                if self.config.verbose:
                    _log(f"      Loss explosion detected: {loss_val:.2f}")

                return action

        should_check_weights = True
        if self.config.lite_mode:
            if self._stability_score > 0.8 and self.step_count % 10 != 0:
                should_check_weights = False

        if should_check_weights and self._check_weights_for_nan():
            self._handle_failure(action, "nan_weights", self.config.weight_nan_action)

            if self.config.verbose:
                _log(f"      NaN weights detected at step {self.step_count}")

            return action

        # Check for silent weight corruption (sparsity/magnitude issues)
        if self.config.enable_weight_health_check:
            if self.step_count % self.config.weight_health_check_frequency == 0:
                is_corrupted, corruption_type = self._check_weight_health()
                if is_corrupted:
                    self._handle_failure(action, corruption_type)

                    if self.config.verbose:
                        _log(f"      Silent weight corruption detected: {corruption_type}")

                    return action

        self.loss_history.append(loss_val)
        self._update_stability_score(loss_val)

        if self.step_count % self._current_checkpoint_frequency == 0:
            self._save_checkpoint()

        return action

    def post_backward(self) -> HealingAction:
        action = HealingAction(step=self.step_count)

        if self._check_gradients_for_nan():
            action.failure_detected = "nan_gradients"
            action.should_skip = True
            self.failures_detected += 1

            if self._restore_checkpoint():
                action.rolled_back = True
                self.total_rollbacks += 1
                self.failures_recovered += 1
                self._reduce_lr()
                action.lr_reduced = True

            if self.config.verbose:
                _log(f"      NaN gradients detected")

            return action
        grad_norm = self._compute_gradient_norm()
        self.gradient_history.append(grad_norm)

        if grad_norm > self.config.gradient_explosion_threshold:
            action.failure_detected = "gradient_explosion"

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm
            )
            action.gradients_clipped = True

            if self.config.verbose:
                _log(f"    ✂️ Gradient explosion clipped: {grad_norm:.2e}")

        if self.config.enable_forecasting:
            will_explode, growth = self._predict_explosion()
            if will_explode:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip_norm
                )
                action.gradients_clipped = True
                action.message = f"Predicted explosion (growth={growth:.2f}), clipped"

        if self.config.enable_baseline_clipping and not action.gradients_clipped and self.config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm
            )
            action.gradients_clipped = True

        return action

    def end_epoch(self, epoch: int = None) -> Dict[str, Any]:
        if epoch is not None:
            self.epoch_count = epoch
        else:
            self.epoch_count += 1

        self._save_checkpoint()

        return {
            "epoch": self.epoch_count,
            "total_steps": self.step_count,
            "total_rollbacks": self.total_rollbacks,
            "total_skips": self.total_skips,
            "failures_detected": self.failures_detected,
            "failures_recovered": self.failures_recovered,
            "recovery_rate": self.failures_recovered / max(self.failures_detected, 1),
            "current_lr": self.current_lr,
            "stability_score": self._stability_score,
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "step_count": self.step_count,
            "epoch_count": self.epoch_count,
            "total_rollbacks": self.total_rollbacks,
            "total_skips": self.total_skips,
            "failures_detected": self.failures_detected,
            "failures_recovered": self.failures_recovered,
            "recovery_rate": self.failures_recovered / max(self.failures_detected, 1),
            "lr_reductions": self.lr_reductions,
            "current_lr": self.current_lr,
            "original_lr": self.original_lr,
            "stability_score": self._stability_score,
            "checkpoint_frequency": self._current_checkpoint_frequency,
            "checkpoints_saved": len(self.checkpoints),
        }

def make_uncrashable(model: nn.Module, optimizer: torch.optim.Optimizer) -> SelfHealingArc:
    return SelfHealingArc(model, optimizer)

if __name__ == "__main__":
    print("Testing SelfHealingArc...")

    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    shard = SelfHealingArc(model, optimizer)

    for step in range(50):
        x = torch.randn(32, 100)
        out = model(x)
        loss = out.mean()

        if step == 25:
            loss = torch.tensor(float('inf'))

        action = shard.step(loss)

        if action.should_skip:
            print(f"Step {step}: Skipped (failure: {action.failure_detected})")
            continue

        loss.backward()
        post_action = shard.post_backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"\nStats: {shard.get_stats()}")
    print("\nSelfHealingArc test complete")