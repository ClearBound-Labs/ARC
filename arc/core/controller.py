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

"""
ARC Unified Controller — the single canonical API.

Usage:

    import arc
    controller = arc.Controller(model, optimizer)

    for batch in dataloader:
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()

        action = controller.step(loss)
        if action.should_skip:
            continue

        optimizer.step()

Mixed precision with a GradScaler: hand the scaler to the Controller and let
it unscale the gradients (don't call ``scaler.unscale_()`` yourself):

    controller = arc.Controller(model, optimizer, scaler=scaler)
    ...
    scaler.scale(loss).backward()
    action = controller.step(loss)          # pass the *unscaled* loss
    if not action.should_skip:
        scaler.step(optimizer)
        scaler.update()

Checkpoints are kept in host memory. A snapshot is copied right after an
``optimizer.step()`` (asynchronously on CUDA, overlapping the next forward and
backward pass) and only becomes a rollback target once the following step
passes every check.
"""

import copy
import datetime
import hashlib
import hmac
import json
import logging
import math
import os
import time
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn

from arc.signals.lyapunov import (
    LyapunovMonitor, TrainingPhase, build_param_sample, read_param_sample,
)
from arc.signals.trend import TrendDetector, DriftSeverity
from arc._logging import logger, enable_console_logging as _enable_console_logging


class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    FAILURE = "failure"


@dataclass
class ControllerConfig:
    """Configuration for ARC Controller."""
    # In-memory checkpointing
    checkpoint_frequency: int = 20
    max_checkpoints: int = 3
    cpu_offload: bool = True
    async_checkpoint: bool = True

    # Detection thresholds
    loss_explosion_threshold: float = 100.0
    loss_spike_factor: float = 10.0
    gradient_explosion_threshold: float = 1e5
    enable_nan_detection: bool = True

    # Auto-calibration: learn thresholds from first N steps
    auto_calibrate: bool = True
    calibration_steps: int = 50

    # Gradient accumulation: only check every N micro-steps
    gradient_accumulation_steps: int = 1

    # Set when an fp16 GradScaler that ARC doesn't own (e.g. inside a
    # framework's trainer) handles overflow: non-finite gradients are then
    # routine overflow the scaler will skip, not a failure to roll back.
    external_grad_scaler: bool = False

    # Signal modules to enable
    enable_lyapunov: bool = True
    enable_trend: bool = True
    enable_gradient_monitoring: bool = True
    enable_weight_health: bool = True

    # Trend detection
    trend_window: int = 50
    trend_min_samples: int = 10

    # Lyapunov
    lyapunov_window: int = 30
    lyapunov_sample_ratio: float = 0.1
    lyapunov_check_frequency: int = 5  # Check every N steps (not every step)

    # Recovery
    lr_reduction_factor: float = 0.5
    max_lr_reductions: int = 8
    # Healthy steps after an LR reduction before one reduction is undone
    # (0 = reductions are permanent).
    lr_recovery_steps: int = 500
    # Healthy steps after a rollback before the failure counts as recovered.
    recovery_confirm_steps: int = 10
    gradient_clip_norm: float = 1.0
    # Opt-in: clipping changes the optimisation itself, so ARC never does it
    # unless asked.
    enable_baseline_clipping: bool = False

    # Persistent failure escape
    max_consecutive_rollbacks: int = 5
    # Escape cycles to attempt before declaring the failure unrecoverable.
    # Rollback only fixes corrupted weights; a bad batch, an unguarded log(0),
    # or an fp16 overflow in the data reproduces on every step no matter how
    # far back ARC rewinds, so it must stop claiming recovery and say so.
    max_persistent_failures: int = 3
    skip_ahead_steps: int = 5
    weight_perturbation_scale: float = 0.01

    # Weight health
    weight_health_check_frequency: int = 10
    weight_sparsity_threshold: float = 0.3
    weight_magnitude_collapse_threshold: float = 0.01

    # Adaptive checkpointing
    adaptive_frequency: bool = True
    min_checkpoint_frequency: int = 5
    max_checkpoint_frequency: int = 100

    # Logging
    verbose: bool = True
    log_interventions: bool = True

    # Cost accounting (for the recovery ledger's gpu-hours/dollars-saved fields)
    gpu_count: int = 1
    gpu_hourly_cost: float = 0.0
    gpu_power_watts: float = 0.0
    carbon_intensity_kg_per_kwh: float = 0.0
    # How often (in steps) the run would have a durable checkpoint without
    # ARC. "Saved" work is measured against restarting from that checkpoint;
    # 0 means unknown, and savings are then reported as None, not guessed.
    baseline_checkpoint_interval: int = 0

    # Cost attribution / chargeback: tags the exported ledger so
    # arc.evaluation.chargeback can roll up GPU-hours/dollars saved per
    # team/project across many runs' exported ledgers.
    team: Optional[str] = None
    project: Optional[str] = None

    # HMAC key for the ledger hash chain. Without a key the chain only shows
    # accidental edits; with one, nobody without the key can rewrite it.
    # Falls back to the ARC_LEDGER_KEY environment variable.
    ledger_key: Optional[str] = field(default=None, repr=False)


@dataclass
class StepAction:
    """Result of a single controller.step() call."""
    step: int = 0
    status: HealthStatus = HealthStatus.HEALTHY
    should_skip: bool = False
    rolled_back: bool = False
    lr_reduced: bool = False
    gradients_clipped: bool = False
    failure_type: Optional[str] = None
    loss_value: float = 0.0
    health_score: float = 1.0
    lyapunov_exponent: float = 0.0
    training_phase: str = "unknown"
    message: str = ""
    unrecoverable: bool = False


def _to_host(obj: Any) -> Any:
    """Detached copy of a (nested) state dict with every tensor in ordinary
    (pageable) CPU memory. Never pinned: pinned memory is scarce, and holding
    several model-sized pinned copies can abort the process."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: _to_host(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_host(v) for v in obj)
    return copy.deepcopy(obj)


def _clone_tree(obj: Any) -> Any:
    """Detached copy of a (nested) state dict, tensors kept on their device."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    if isinstance(obj, dict):
        return {k: _clone_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_clone_tree(v) for v in obj)
    return copy.deepcopy(obj)


def _stack_to_list(scalars: List[torch.Tensor]) -> List[float]:
    """One host sync for many 0-dim tensors, whatever devices they live on."""
    dev = scalars[0].device
    return torch.stack([s.to(dev) for s in scalars]).tolist()


class Controller:
    """
    ARC Unified Controller.

    Integrates all signal modules (Lyapunov, trend detection, gradient
    monitoring, weight health) into a single recovery loop with
    asynchronous host-memory checkpointing.

    This is the ONE entry point for ARC. All other APIs are legacy.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        config: Optional[ControllerConfig] = None,
        verbose: Optional[bool] = None,
        scaler: Optional[Any] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.config = config or ControllerConfig()

        if verbose is not None:
            self.config.verbose = verbose
        if self.config.verbose:
            _enable_console_logging()

        if type(model).__name__ == "FullyShardedDataParallel" or any(
            type(p).__name__ == "DTensor" for p in model.parameters()
        ):
            raise NotImplementedError(
                "ARC's Controller doesn't support FSDP / DTensor-sharded models yet. "
                "Single-device and DistributedDataParallel models are supported."
            )

        self._ledger_key = self.config.ledger_key or os.environ.get("ARC_LEDGER_KEY")
        self._distributed = (
            dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        )

        # ---- State ----
        self._step_count: int = 0
        self._original_lr: float = self._get_lr()
        self._lr_reductions: int = 0
        # Cumulative product of every LR reduction currently in force. Each
        # snapshot records the penalty it was taken under, so a restore can
        # re-apply exactly the reductions made since that snapshot.
        self._lr_penalty: float = 1.0
        self._stability_score: float = 1.0
        self._checkpoint_frequency: int = self.config.checkpoint_frequency

        # Committed snapshots (rollback targets) and the one awaiting validation
        self._checkpoints: List[Dict[str, Any]] = []
        self._pending: Optional[Dict[str, Any]] = None
        self._snapshot_due: bool = False
        self._inflight: Optional[Dict[str, Any]] = None  # snapshot still being copied
        self._executor: Optional[ThreadPoolExecutor] = None
        self._copy_stream = None
        self._hook_misses: int = 0

        self._loss_history: deque = deque(maxlen=100)
        self._gradient_history: deque = deque(maxlen=20)

        # Persistent failure tracking
        self._consecutive_rollbacks: int = 0
        self._last_rollback_step: int = -100
        self._skip_until_step: int = 0
        self._persistent_failure_count: int = 0
        self.unrecoverable: bool = False

        # Recovery bookkeeping
        self._pending_failures: int = 0
        self._last_failure_reason: Optional[str] = None
        self._healthy_streak: int = 0
        self._steps_since_lr_change: int = 0
        self._scaler_overflow: bool = False
        self._unscaled: bool = False

        # Counters
        self.total_rollbacks: int = 0
        self.total_skips: int = 0
        self.failures_detected: int = 0
        self.failures_recovered: int = 0
        self.intervention_log: List[Dict[str, Any]] = []
        self._last_restored_step: int = 0

        # ---- Signal Modules ----
        self._lyapunov: Optional[LyapunovMonitor] = None
        if self.config.enable_lyapunov:
            self._lyapunov = LyapunovMonitor(
                window=self.config.lyapunov_window,
                sample_ratio=self.config.lyapunov_sample_ratio,
            )

        self._trend: Optional[TrendDetector] = None
        if self.config.enable_trend:
            self._trend = TrendDetector(
                window=self.config.trend_window,
                min_samples=self.config.trend_min_samples,
            )

        self._param_list = [
            (n, p) for n, p in model.named_parameters() if p.requires_grad
        ]
        self._params = [p for _, p in self._param_list]
        n_total = sum(p.numel() for p in self._params)
        self._weight_sample = build_param_sample(
            self._params, min(n_total, 1_000_000), seed=7)

        self._baseline_weight_stats = self._compute_weight_stats()

        # Auto-calibration state
        self._calibrated: bool = not self.config.auto_calibrate
        self._calibration_losses: List[float] = []
        self._calibration_grads: List[float] = []

        self._start_time: float = time.time()
        self._micro_step: int = 0

        # Snapshot right after optimizer.step(); fall back to synchronous
        # snapshots inside step() if these hooks never fire.
        self._hooks = []
        self._use_hooks = hasattr(optimizer, "register_step_post_hook")
        if self._use_hooks:
            self._hooks.append(optimizer.register_step_pre_hook(self._pre_optimizer_step))
            self._hooks.append(optimizer.register_step_post_hook(self._post_optimizer_step))

        self._checkpoints.append(self._take_snapshot(asynchronous=False))

        n_params = n_total
        signals = [name for name, on in (
            ("lyapunov", self.config.enable_lyapunov),
            ("trend", self.config.enable_trend),
            ("gradient", self.config.enable_gradient_monitoring),
            ("weight_health", self.config.enable_weight_health),
        ) if on]
        extras = []
        if self.config.auto_calibrate:
            extras.append(f"auto_cal={self.config.calibration_steps}")
        if self.config.gradient_accumulation_steps > 1:
            extras.append(f"grad_accum={self.config.gradient_accumulation_steps}")
        if self.scaler is not None:
            extras.append("grad_scaler")
        extra_str = f" | {', '.join(extras)}" if extras else ""
        self._log(f"Controller: {n_params/1e6:.1f}M params | signals=[{', '.join(signals)}] | "
                  f"ckpt_freq={self.config.checkpoint_frequency} | "
                  f"cpu_offload={self.config.cpu_offload}{extra_str}")

    # ================================================================
    # PUBLIC API
    # ================================================================

    def step(self, loss: Optional[torch.Tensor] = None) -> StepAction:
        """
        Main entry point. Call AFTER loss.backward(), BEFORE optimizer.step().

        `loss` may be None when it isn't available (e.g. inside a framework
        callback); gradient, weight and dynamics checks still run.

        Returns StepAction. If action.should_skip is True, skip the
        optimizer step (and, with a GradScaler, scaler.step/update).
        """
        self._step_count += 1
        self._unscaled = False
        self._scaler_overflow = False
        action = StepAction(step=self._step_count, health_score=self._stability_score,
                            unrecoverable=self.unrecoverable)

        # --- Gradient accumulation: skip intermediate micro-steps ---
        if self.config.gradient_accumulation_steps > 1:
            self._micro_step += 1
            if self._micro_step < self.config.gradient_accumulation_steps:
                if loss is not None:
                    value = float(loss.detach())
                    if math.isfinite(value):
                        self._loss_history.append(value)
                action.status = HealthStatus.HEALTHY
                return action
            self._micro_step = 0

        # --- Escape mode: skip steps after persistent failure ---
        if self._step_count < self._skip_until_step:
            action.should_skip = True
            action.status = HealthStatus.WARNING
            action.message = "Skipping (escape mode)"
            self.total_skips += 1
            return action

        calibrating = not self._calibrated
        if calibrating:
            failure, extra = self._detect_during_calibration(loss, action)
        else:
            failure, extra = self._detect(loss, action)

        if self._distributed:
            failure = self._agree_on_failure(failure)

        if failure is not None:
            return self._handle_failure(action, failure, loss, extra)

        if self._scaler_overflow:
            # GradScaler skips this optimizer step and lowers its scale; it is
            # routine fp16 behaviour, not a failure.
            action.status = HealthStatus.WARNING
            action.message = "fp16 gradient overflow; GradScaler will skip this step"
            return action

        self._on_healthy_step(action, loss_seen=loss is not None)
        if calibrating:
            if self._step_count >= self.config.calibration_steps:
                self._finish_calibration()
            action.message = f"Calibrating ({self._step_count}/{self.config.calibration_steps})"
        return action

    def post_backward(self) -> StepAction:
        """
        Optional: call after loss.backward() for a gradient-only NaN/Inf check.

        Redundant if you pass the loss to step() after backward().
        """
        action = StepAction(step=self._step_count)
        if self._params and not math.isfinite(self._grad_norm(clip=False)):
            return self._handle_failure(action, "nan_gradients")
        return action

    def get_stats(self) -> Dict[str, Any]:
        """Return controller statistics for logging/monitoring."""
        stats = {
            "step_count": self._step_count,
            "total_rollbacks": self.total_rollbacks,
            "total_skips": self.total_skips,
            "failures_detected": self.failures_detected,
            "failures_recovered": self.failures_recovered,
            "failures_pending": self._pending_failures,
            "recovery_rate": self.failures_recovered / max(self.failures_detected, 1),
            "lr_reductions": self._lr_reductions,
            "current_lr": self._get_lr(),
            "original_lr": self._original_lr,
            "stability_score": self._stability_score,
            "checkpoint_frequency": self._checkpoint_frequency,
            "checkpoints_saved": len(self._checkpoints),
        }

        if self._lyapunov:
            stats["lyapunov_exponent"] = self._lyapunov.exponent

        if self._trend:
            stats["trend"] = self._trend.get_diagnostics()

        return stats

    def summary(self) -> str:
        """Human-readable end-of-training report (also logged when verbose)."""
        elapsed_min = (time.time() - self._start_time) / 60
        n_params = sum(p.numel() for p in self._params)
        recovery_rate = self.failures_recovered / max(self.failures_detected, 1)

        lines = [
            "",
            "=" * 60,
            "  ARC TRAINING SUMMARY",
            "=" * 60,
            f"  Model:           {n_params/1e6:.1f}M parameters",
            f"  Training time:   {elapsed_min:.1f} minutes ({self._step_count} steps)",
            f"  Current LR:      {self._get_lr():.2e} (started at {self._original_lr:.2e})",
            "",
            f"  Failures found:  {self.failures_detected}",
            f"  Recovered:       {self.failures_recovered} ({recovery_rate:.0%} confirmed by "
            f"{self.config.recovery_confirm_steps} healthy steps)",
            f"  Rollbacks:       {self.total_rollbacks}",
            f"  Steps skipped:   {self.total_skips}",
            f"  LR reductions:   {self._lr_reductions} in force",
        ]

        failures = [e for e in self.intervention_log if e["action"] in ("rollback", "skip")]
        if failures:
            saved = [e["gpu_hours_saved"] for e in failures if e.get("gpu_hours_saved") is not None]
            lost = sum(e.get("steps_lost") or 0 for e in failures)
            lines.append(f"  Steps rolled back: {lost}")
            if saved:
                lines.append(f"  GPU-hours saved: {sum(saved):.3f} "
                             f"(vs. restarting every {self.config.baseline_checkpoint_interval} steps)")
            else:
                lines.append("  GPU-hours saved: unknown (set baseline_checkpoint_interval)")

            type_counts: Dict[str, int] = {}
            for entry in failures:
                t = entry.get("reason", "unknown")
                type_counts[t] = type_counts.get(t, 0) + 1
            lines.append("")
            lines.append("  Failure breakdown:")
            for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
                lines.append(f"    {t}: {c}x")
        else:
            lines.append("  No failures detected -- clean run!")

        if self._lyapunov:
            lines.append(f"\n  Lyapunov exp:    {self._lyapunov.exponent:.6f}")

        lines.append(f"  Stability:       {self._stability_score:.3f}")
        lines.append("=" * 60)

        report = "\n".join(lines)
        print(report)  # explicitly requested report, not a log line
        return report

    def log_to_wandb(self, step: Optional[int] = None):
        """
        Log current ARC metrics to Weights & Biases.

        Requires `wandb` to be installed and initialized.
        Call this inside your training loop alongside your other logging.
        """
        try:
            import wandb
        except ImportError:
            return

        if not wandb.run:
            return

        log_step = step or self._step_count
        metrics = {
            "arc/stability_score": self._stability_score,
            "arc/current_lr": self._get_lr(),
            "arc/total_rollbacks": self.total_rollbacks,
            "arc/failures_detected": self.failures_detected,
            "arc/failures_recovered": self.failures_recovered,
            "arc/checkpoint_frequency": self._checkpoint_frequency,
        }

        if self._lyapunov:
            metrics["arc/lyapunov_exponent"] = self._lyapunov.exponent

        if self._trend:
            diag = self._trend.get_diagnostics()
            metrics["arc/trend_loss_slope"] = diag.get("loss_slope", 0)
            metrics["arc/trend_grad_slope"] = diag.get("grad_norm_slope", 0)

        if self._loss_history:
            recent = list(self._loss_history)[-10:]
            metrics["arc/recent_loss_mean"] = sum(recent) / len(recent)

        wandb.log(metrics, step=log_step)

    def close(self) -> None:
        """Detach ARC's optimizer hooks (e.g. before reusing the optimizer)."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        self._use_hooks = False
        self._sync_inflight()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    # ================================================================
    # PERSISTENCE
    # ================================================================

    def state_dict(self) -> Dict[str, Any]:
        """Controller state as plain Python values (safe for weights_only loads)."""
        return {
            "version": 1,
            "step_count": self._step_count,
            "elapsed_seconds": time.time() - self._start_time,
            "original_lr": self._original_lr,
            "lr_reductions": self._lr_reductions,
            "lr_penalty": self._lr_penalty,
            "stability_score": self._stability_score,
            "checkpoint_frequency": self._checkpoint_frequency,
            "consecutive_rollbacks": self._consecutive_rollbacks,
            "last_rollback_step": self._last_rollback_step,
            "skip_until_step": self._skip_until_step,
            "persistent_failure_count": self._persistent_failure_count,
            "unrecoverable": self.unrecoverable,
            "pending_failures": self._pending_failures,
            "last_failure_reason": self._last_failure_reason,
            "healthy_streak": self._healthy_streak,
            "steps_since_lr_change": self._steps_since_lr_change,
            "total_rollbacks": self.total_rollbacks,
            "total_skips": self.total_skips,
            "failures_detected": self.failures_detected,
            "failures_recovered": self.failures_recovered,
            "last_restored_step": self._last_restored_step,
            "calibrated": self._calibrated,
            "calibration_losses": list(self._calibration_losses),
            "calibration_grads": list(self._calibration_grads),
            "loss_explosion_threshold": self.config.loss_explosion_threshold,
            "gradient_explosion_threshold": self.config.gradient_explosion_threshold,
            "loss_history": list(self._loss_history),
            "gradient_history": list(self._gradient_history),
            "baseline_weight_stats": dict(self._baseline_weight_stats),
            "micro_step": self._micro_step,
            "intervention_log": copy.deepcopy(self.intervention_log),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore state produced by `state_dict()`."""
        self._step_count = state["step_count"]
        self._start_time = time.time() - state["elapsed_seconds"]
        self._original_lr = state["original_lr"]
        self._lr_reductions = state["lr_reductions"]
        self._lr_penalty = state["lr_penalty"]
        self._stability_score = state["stability_score"]
        self._checkpoint_frequency = state["checkpoint_frequency"]
        self._consecutive_rollbacks = state["consecutive_rollbacks"]
        self._last_rollback_step = state["last_rollback_step"]
        self._skip_until_step = state["skip_until_step"]
        self._persistent_failure_count = state["persistent_failure_count"]
        self.unrecoverable = state["unrecoverable"]
        self._pending_failures = state["pending_failures"]
        self._last_failure_reason = state["last_failure_reason"]
        self._healthy_streak = state["healthy_streak"]
        self._steps_since_lr_change = state["steps_since_lr_change"]
        self.total_rollbacks = state["total_rollbacks"]
        self.total_skips = state["total_skips"]
        self.failures_detected = state["failures_detected"]
        self.failures_recovered = state["failures_recovered"]
        self._last_restored_step = state["last_restored_step"]
        self._calibrated = state["calibrated"]
        self._calibration_losses = list(state["calibration_losses"])
        self._calibration_grads = list(state["calibration_grads"])
        self.config.loss_explosion_threshold = state["loss_explosion_threshold"]
        self.config.gradient_explosion_threshold = state["gradient_explosion_threshold"]
        self._loss_history = deque(state["loss_history"], maxlen=self._loss_history.maxlen)
        self._gradient_history = deque(state["gradient_history"],
                                       maxlen=self._gradient_history.maxlen)
        self._baseline_weight_stats = dict(state["baseline_weight_stats"])
        self._micro_step = state["micro_step"]
        self.intervention_log = copy.deepcopy(state["intervention_log"])

    def save_checkpoint(self, path) -> None:
        """Durable checkpoint of model, optimizer, scheduler, scaler, RNG and
        ARC state. Written atomically; loadable with ``weights_only=True``."""
        self._sync_inflight()
        payload = {
            "arc_version": 1,
            "model": _to_host(self.model.state_dict()),
            "optimizer": _to_host(self.optimizer.state_dict()),
            "controller": self.state_dict(),
            "rng": self._rng_state(),
        }
        if self.scheduler is not None and hasattr(self.scheduler, "state_dict"):
            payload["scheduler"] = self.scheduler.state_dict()
        if self.scaler is not None:
            payload["scaler"] = self.scaler.state_dict()
        path = os.fspath(path)
        tmp = f"{path}.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def load_checkpoint(self, path) -> int:
        """Restore a checkpoint written by `save_checkpoint`. Returns the step
        to resume from."""
        from arc._serialization import safe_torch_load

        payload = safe_torch_load(path, map_location="cpu")
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if "scheduler" in payload and self.scheduler is not None:
            self.scheduler.load_state_dict(payload["scheduler"])
        if "scaler" in payload and self.scaler is not None:
            self.scaler.load_state_dict(payload["scaler"])
        self.load_state_dict(payload["controller"])
        self._set_rng_state(payload["rng"])
        if self._lyapunov:
            self._lyapunov.reset()
        if self._trend:
            self._trend.reset()
        self._pending = None
        self._snapshot_due = False
        self._checkpoints = [self._take_snapshot(asynchronous=False)]
        return self._step_count

    # ================================================================
    # DETECTION
    # ================================================================

    def _detect(self, loss: Optional[torch.Tensor], action: StepAction
                ) -> Tuple[Optional[str], str]:
        """Run every enabled check. Returns (failure_type or None, detail)."""
        if loss is not None:
            loss_val = float(loss.detach())
            action.loss_value = loss_val
            if not math.isfinite(loss_val):
                if self.config.enable_nan_detection:
                    return "nan_loss", ""
            elif loss_val > 0:
                # The relative spike check runs independently of the absolute
                # threshold, so a threshold calibrated too high from noisy
                # warmup still can't hide a sudden spike. It needs a few losses
                # of history: one or two early losses (or the first steps after
                # a rollback) are no baseline, and Adam's first updates often
                # move the loss 10x on a healthy run.
                if len(self._loss_history) >= 5:
                    recent = list(self._loss_history)[-5:]
                    recent_mean = sum(recent) / len(recent)
                    if recent_mean > 0 and loss_val > recent_mean * self.config.loss_spike_factor:
                        return "loss_explosion", ""
                if loss_val > self.config.loss_explosion_threshold:
                    return "loss_explosion", ""

        if self.config.enable_weight_health and \
                self._step_count % self.config.weight_health_check_frequency == 0:
            failure = self._check_weights()
            if failure:
                return failure, ""

        failure, _ = self._check_gradients(action)
        if failure:
            return failure, ""

        if self._lyapunov and self._step_count % self.config.lyapunov_check_frequency == 0:
            lyap_state = self._lyapunov.update(self.model, lr=self._get_lr())
            action.lyapunov_exponent = lyap_state.exponent
            action.training_phase = lyap_state.phase.value
            if lyap_state.phase == TrainingPhase.DIVERGENT and lyap_state.confidence > 0.5:
                return "lyapunov_divergence", f"λ={lyap_state.exponent:.4f}"

        if self._trend:
            loss_val = action.loss_value if loss is not None else None
            grad_n = self._gradient_history[-1] if self._gradient_history else None
            weight_n = self._compute_total_weight_norm() if self._step_count % 10 == 0 else None
            drift = self._trend.update(loss=loss_val, grad_norm=grad_n, weight_norm=weight_n)
            if drift.severity == DriftSeverity.CRITICAL and drift.confidence > 0.5:
                return "drift_detected", drift.message

        return None, ""

    def _detect_during_calibration(self, loss: Optional[torch.Tensor], action: StepAction
                                   ) -> Tuple[Optional[str], str]:
        """During warmup only non-finite values count as failures; everything
        else is recorded to set the thresholds."""
        if loss is not None:
            loss_val = float(loss.detach())
            action.loss_value = loss_val
            if not math.isfinite(loss_val):
                return "nan_loss", ""
            self._calibration_losses.append(loss_val)

        if self.config.enable_weight_health and \
                self._step_count % self.config.weight_health_check_frequency == 0:
            if not self._weights_finite():
                return "nan_weights", ""

        failure, grad_norm = self._check_gradients(action, calibrating=True)
        if failure:
            return failure, ""
        if grad_norm is not None:
            self._calibration_grads.append(grad_norm)
        return None, ""

    def _check_gradients(self, action: StepAction, calibrating: bool = False
                         ) -> Tuple[Optional[str], Optional[float]]:
        """Returns (failure_type or None, finite gradient norm or None)."""
        clip = self.config.enable_baseline_clipping and self.config.gradient_clip_norm > 0
        if not (self.config.enable_gradient_monitoring or clip):
            return None, None
        if not any(p.grad is not None for p in self._params):
            return None, None

        if self.scaler is not None and self.scaler.is_enabled():
            self.scaler.unscale_(self.optimizer)
            self._unscaled = True

        grad_norm = self._grad_norm(clip=clip)
        action.gradients_clipped = clip

        if not math.isfinite(grad_norm):
            if self._unscaled or self.config.external_grad_scaler:
                self._scaler_overflow = True
                return None, None
            return "nan_gradients", None
        if not self.config.enable_gradient_monitoring:
            return None, None

        self._gradient_history.append(grad_norm)
        if not calibrating and grad_norm > self.config.gradient_explosion_threshold:
            return "gradient_explosion", grad_norm
        return None, grad_norm

    def _agree_on_failure(self, failure: Optional[str]) -> Optional[str]:
        """All ranks roll back together, or none do: a rank-local rollback
        would silently desynchronise the replicas."""
        device = torch.device("cpu")
        if dist.get_backend() == "nccl":
            device = torch.device("cuda", torch.cuda.current_device())
        flag = torch.tensor([1.0 if failure else 0.0], device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item() > 0 and failure is None:
            return "peer_failure"
        return failure

    # ================================================================
    # AUTO-CALIBRATION
    # ================================================================

    def _finish_calibration(self):
        """Set thresholds based on observed warmup statistics."""
        import numpy as np

        if self._calibration_losses:
            losses = np.array(self._calibration_losses)
            loss_mean = float(np.mean(losses))
            loss_std = float(np.std(losses))

            # Threshold = mean + 10 * std (very conservative), but capped relative
            # to the observed mean so a few noisy warmup outliers can't push the
            # threshold so high it effectively disables detection for the rest of
            # the run. The relative spike check in step() is the primary safety
            # net regardless of where this absolute threshold lands.
            cal_threshold = loss_mean + 10.0 * max(loss_std, loss_mean * 0.1)
            relative_cap = max(loss_mean * 100.0, 100.0)
            self.config.loss_explosion_threshold = max(10.0, min(cal_threshold, relative_cap, 1e6))

        if self._calibration_grads:
            grads = np.array(self._calibration_grads)
            grad_mean = float(np.mean(grads))
            grad_std = float(np.std(grads))

            cal_grad = grad_mean + 10.0 * max(grad_std, grad_mean * 0.1)
            self.config.gradient_explosion_threshold = max(100.0, min(cal_grad, 1e8))

        self._calibrated = True
        self._log(f"auto-calibrated: loss_thresh={self.config.loss_explosion_threshold:.1f}, "
                  f"grad_thresh={self.config.gradient_explosion_threshold:.1f}")

    # ================================================================
    # HEALTHY STEP / FAILURE HANDLING
    # ================================================================

    def _on_healthy_step(self, action: StepAction, loss_seen: bool) -> None:
        if loss_seen:
            self._loss_history.append(action.loss_value)
            if self._calibrated:
                self._update_stability()
        self._consecutive_rollbacks = 0

        # The snapshot taken after the previous update is now validated.
        if self._pending is not None:
            self._commit(self._pending)
            self._pending = None
        elif self._snapshot_due:
            # The post-step hook didn't fire since the snapshot was requested:
            # the optimizer doesn't run its hooks (wrapped optimizers do this).
            # Snapshot synchronously instead; this step's checks validated it.
            self._snapshot_due = False
            self._commit(self._take_snapshot(asynchronous=False))
            self._hook_misses += 1
            if self._hook_misses >= 3:
                self._use_hooks = False

        self._healthy_streak += 1
        self._steps_since_lr_change += 1
        if self._pending_failures and self._healthy_streak >= self.config.recovery_confirm_steps:
            self._confirm_recovery()

        if (self.config.lr_recovery_steps > 0 and self._lr_reductions > 0
                and self._steps_since_lr_change >= self.config.lr_recovery_steps):
            self._undo_lr_reduction()

        if self._step_count % self._checkpoint_frequency == 0:
            if self._use_hooks:
                self._snapshot_due = True
            else:
                self._commit(self._take_snapshot(asynchronous=False))

        action.status = HealthStatus.HEALTHY

    def _handle_failure(
        self,
        action: StepAction,
        failure_type: str,
        loss: Optional[torch.Tensor] = None,
        extra_msg: str = "",
    ) -> StepAction:
        """Unified failure handling: detect → rollback → heal."""
        action.failure_type = failure_type
        action.should_skip = True
        action.status = HealthStatus.FAILURE
        self.failures_detected += 1
        self._pending_failures += 1
        self._last_failure_reason = failure_type
        self._healthy_streak = 0
        self._steps_since_lr_change = 0

        # Anything captured after the last validated step is suspect.
        self._pending = None
        self._snapshot_due = False

        if self._unscaled:
            # The caller skips scaler.step/update on a skipped step; update()
            # here resets the scaler so the next unscale_() is legal.
            self.scaler.update()
            self._unscaled = False

        if self._restore_checkpoint():
            action.rolled_back = True
            self.total_rollbacks += 1

            self._reduce_lr()
            action.lr_reduced = True

            if self._track_consecutive_rollback():
                self._escape_persistent_failure()
                action.message = f"Persistent failure escape | {failure_type}"
                if self._persistent_failure_count >= self.config.max_persistent_failures:
                    self._declare_unrecoverable(action, failure_type)
            else:
                action.message = (f"Rolled back to step {self._last_restored_step} "
                                  f"after {failure_type}")
        else:
            action.message = f"Rollback failed for {failure_type}"

        if extra_msg:
            action.message += f" | {extra_msg}"

        # Reset trend detector after rollback to avoid re-triggering
        if self._trend:
            self._trend.reset()

        self._log_intervention(failure_type, action)

        if loss is None:
            loss_str = "n/a"
        else:
            loss_str = "NaN" if not math.isfinite(action.loss_value) else f"{action.loss_value:.4f}"
        self._log(f"[{self._step_count}] {failure_type} | loss={loss_str} | {action.message}",
                  level=logging.WARNING)

        return action

    def _confirm_recovery(self) -> None:
        resolved = self._pending_failures
        self.failures_recovered += resolved
        self._pending_failures = 0
        if self.config.log_interventions:
            self._append_ledger({
                "timestamp": datetime.datetime.now().isoformat(),
                "step": self._step_count,
                "reason": self._last_failure_reason,
                "action": "recovery_confirmed",
                "failures_resolved": resolved,
                "healthy_steps": self._healthy_streak,
            })

    # ================================================================
    # CHECKPOINTING
    # ================================================================

    def _rng_state(self) -> Dict[str, Any]:
        rng = {"torch": torch.get_rng_state()}
        if torch.cuda.is_available():
            rng["cuda"] = torch.cuda.get_rng_state_all()
        return rng

    def _set_rng_state(self, rng: Dict[str, Any]) -> None:
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if "cuda" in rng and torch.cuda.is_available():
            states = rng["cuda"]
            if len(states) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(states)

    def _single_cuda_device(self) -> Optional[torch.device]:
        devices = {p.device for p in self._params}
        if len(devices) == 1:
            (device,) = devices
            if device.type == "cuda":
                return device
        return None

    def _take_snapshot(self, asynchronous: bool) -> Dict[str, Any]:
        """Copy model + optimizer + RNG state.

        Asynchronous mode (single CUDA device) copies parameters and optimizer
        state to host memory on a background thread and a side CUDA stream,
        overlapping the next forward and backward pass. The pre-step hook makes
        the next optimizer.step() wait for that copy, so the in-place update
        can never race it. Buffers (e.g. BatchNorm running stats, which the
        next forward pass mutates) are copied right away instead.
        """
        model_sd = self.model.state_dict()
        opt_sd = self.optimizer.state_dict()
        snap: Dict[str, Any] = {
            "step": self._step_count,
            "lr_penalty": self._lr_penalty,
            "rng": self._rng_state(),
            "future": None,
        }

        if not self.config.cpu_offload:
            snap["model"] = _clone_tree(model_sd)
            snap["optim"] = _clone_tree(opt_sd)
            return snap

        device = self._single_cuda_device()
        if not (asynchronous and self.config.async_checkpoint and device is not None):
            snap["model"] = _to_host(model_sd)
            snap["optim"] = _to_host(opt_sd)
            return snap

        param_ptrs = {p.data_ptr() for p in self._params}
        buffers = {k: v for k, v in model_sd.items()
                   if not (isinstance(v, torch.Tensor) and v.data_ptr() in param_ptrs)}
        params = {k: v for k, v in model_sd.items() if k not in buffers}

        snap["model"] = _to_host(buffers)
        ready = torch.cuda.Event()
        ready.record(torch.cuda.current_stream(device))
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=device)
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arc-snapshot")
        stream = self._copy_stream

        def copy_rest():
            with torch.cuda.device(device), torch.cuda.stream(stream):
                stream.wait_event(ready)  # after the optimizer step's kernels
                return _to_host(params), _to_host(opt_sd)

        snap["future"] = self._executor.submit(copy_rest)
        self._inflight = snap
        return snap

    def _finish_copy(self, snap: Dict[str, Any]) -> bool:
        """Wait for a snapshot's background copy. False if the copy failed;
        that snapshot is then unusable, and training itself is unaffected."""
        future = snap.get("future")
        if future is None:
            return not snap.get("failed", False)
        try:
            params, optim = future.result()
        except Exception as e:  # noqa: BLE001 -- never let a snapshot crash training
            snap["future"] = None
            snap["failed"] = True
            self._log(f"snapshot copy failed and was dropped: {e}", level=logging.WARNING)
            return False
        snap["model"].update(params)
        snap["optim"] = optim
        snap["future"] = None
        return True

    def _pre_optimizer_step(self, optimizer, args, kwargs):
        # The in-place update must not start before the background copy of
        # the same tensors has finished.
        if self._inflight is not None:
            self._finish_copy(self._inflight)
            self._inflight = None

    def _post_optimizer_step(self, optimizer, args, kwargs):
        if self._snapshot_due:
            self._snapshot_due = False
            self._pending = self._take_snapshot(asynchronous=True)

    def _sync_inflight(self) -> None:
        for snap in self._checkpoints + ([self._pending] if self._pending else []):
            self._finish_copy(snap)
        self._inflight = None

    def _commit(self, snap: Dict[str, Any]) -> None:
        self._checkpoints.append(snap)
        while len(self._checkpoints) > self.config.max_checkpoints:
            self._checkpoints.pop(0)

    def _restore_checkpoint(self, index: int = -1) -> bool:
        """Restore model + optimizer + RNG from a committed snapshot.

        Tries the requested snapshot first, then falls back to progressively
        older ones if a restore raises partway through (load_state_dict copies
        tensors in place, so a mid-load failure can leave the model in a mixed
        state -- an older, known-good snapshot recovers instead of silently
        training on that mixed state).
        """
        n = len(self._checkpoints)
        if n == 0:
            return False

        start = index if index >= 0 else n + index
        start = max(0, min(start, n - 1))
        # Optimizer.load_state_dict keeps tensors that are already on the
        # parameter's device as-is; copy them so training never mutates a
        # stored snapshot.
        alias_risk = not self.config.cpu_offload or all(p.device.type == "cpu" for p in self._params)
        last_error: Optional[Exception] = None

        for i in range(start, -1, -1):
            snap = self._checkpoints[i]
            if not self._finish_copy(snap):
                last_error = RuntimeError("snapshot copy failed")
                continue
            try:
                self.model.load_state_dict(snap["model"])
                optim_state = _clone_tree(snap["optim"]) if alias_risk else snap["optim"]
                self.optimizer.load_state_dict(optim_state)

                # The restored param_groups carry the LR in force when the
                # snapshot was taken; re-apply only the reductions made since.
                ratio = self._lr_penalty / snap["lr_penalty"]
                if ratio != 1.0:
                    self._scale_lr(ratio, include_scheduler=False)

                self._set_rng_state(snap["rng"])

                # Losses recorded after the snapshot describe weights that no
                # longer exist; compared against them, the restored model's
                # (higher, older) loss would look like a spike and trigger a
                # rollback loop.
                self._loss_history.clear()
                self._gradient_history.clear()

                if self._lyapunov:
                    self._lyapunov.reset()

                if i != start:
                    self._log(f"checkpoint {start} unusable, restored older checkpoint {i}",
                              level=logging.WARNING)

                self._last_restored_step = snap["step"]
                return True

            except (RuntimeError, KeyError, ValueError) as e:
                last_error = e
                continue

        self._log(f"rollback failed, no usable checkpoint: {last_error}", level=logging.ERROR)
        return False

    # ================================================================
    # SIGNAL UTILITIES (one host sync each)
    # ================================================================

    def _grad_norm(self, clip: bool) -> float:
        params = [p for p in self._params if p.grad is not None]
        if not params:
            return 0.0
        if clip:
            # clip_grad_norm_ returns the pre-clipping total norm.
            return float(torch.nn.utils.clip_grad_norm_(params, self.config.gradient_clip_norm))
        norms = []
        for p in params:
            g = p.grad.detach()
            if g.is_sparse:
                g = g.coalesce()._values()
            norms.append(torch.linalg.vector_norm(g, 2, dtype=torch.float32))
        return math.sqrt(sum(n * n for n in _stack_to_list(norms)))

    def _weights_finite(self) -> bool:
        # fp32-accumulated sum: any NaN/Inf element makes it non-finite, and
        # no extra memory proportional to the tensor is allocated.
        if not self._params:
            return True
        sums = [torch.isfinite(p.detach().sum(dtype=torch.float32)) for p in self._params]
        return all(_stack_to_list(sums))

    def _compute_weight_stats(self) -> Dict[str, float]:
        if not self._params:
            return {"total_params": 0, "zero_fraction": 0.0, "mean_magnitude": 0.0}
        total = sum(p.numel() for p in self._params)
        l1 = _stack_to_list([
            torch.linalg.vector_norm(p.detach(), 1, dtype=torch.float32) for p in self._params
        ])
        sample = read_param_sample(self._params, self._weight_sample)
        zero_fraction = float((sample == 0).float().mean()) if sample.numel() else 0.0
        return {
            "total_params": total,
            "zero_fraction": zero_fraction,
            "mean_magnitude": sum(l1) / max(total, 1),
        }

    def _check_weights(self) -> Optional[str]:
        if not self._weights_finite():
            return "nan_weights"
        current = self._compute_weight_stats()
        baseline = self._baseline_weight_stats

        if current["zero_fraction"] > self.config.weight_sparsity_threshold:
            if current["zero_fraction"] > baseline["zero_fraction"] + 0.1:
                return f"weight_sparsity_explosion ({current['zero_fraction']:.1%})"

        if baseline["mean_magnitude"] > 0:
            ratio = current["mean_magnitude"] / baseline["mean_magnitude"]
            if ratio < self.config.weight_magnitude_collapse_threshold:
                return f"weight_magnitude_collapse (ratio={ratio:.4f})"

        return None

    def _compute_total_weight_norm(self) -> float:
        norms = [torch.linalg.vector_norm(p.detach(), 2, dtype=torch.float32)
                 for p in self._params]
        return math.sqrt(sum(n * n for n in _stack_to_list(norms))) if norms else 0.0

    # ================================================================
    # RECOVERY ACTIONS
    # ================================================================

    def _get_lr(self) -> float:
        for pg in self.optimizer.param_groups:
            return pg["lr"]
        return 0.0

    def _scale_lr(self, factor: float, include_scheduler: bool = True) -> None:
        """Scale the LR everywhere a scheduler might recompute it from."""
        for pg in self.optimizer.param_groups:
            pg["lr"] *= factor
            # OneCycleLR recomputes the LR from these every step.
            for key in ("initial_lr", "max_lr", "min_lr"):
                if isinstance(pg.get(key), (int, float)):
                    pg[key] *= factor
        if include_scheduler and self.scheduler is not None:
            for attr in ("base_lrs", "max_lrs"):
                values = getattr(self.scheduler, attr, None)
                if isinstance(values, list):
                    setattr(self.scheduler, attr, [v * factor for v in values])

    def _reduce_lr(self) -> None:
        if self._lr_reductions >= self.config.max_lr_reductions:
            return
        self._scale_lr(self.config.lr_reduction_factor)
        self._lr_reductions += 1
        self._lr_penalty *= self.config.lr_reduction_factor
        self._steps_since_lr_change = 0

    def _undo_lr_reduction(self) -> None:
        factor = self.config.lr_reduction_factor
        self._scale_lr(1.0 / factor)
        self._lr_reductions -= 1
        self._lr_penalty /= factor
        if self._lr_reductions == 0:
            self._lr_penalty = 1.0  # avoid float drift
        self._steps_since_lr_change = 0
        self._log(f"LR recovered to {self._get_lr():.2e} after "
                  f"{self.config.lr_recovery_steps} healthy steps")

    def _declare_unrecoverable(self, action: StepAction, failure_type: str):
        """Stop claiming recovery once rollback has demonstrably stopped working.

        ARC can only repair the model. If the weights are healthy and the loss
        is still bad after several escape cycles, the fault lives outside the
        parameters -- almost always the input batch, the loss expression, or a
        precision overflow -- and no further rollback will help. Say that once,
        loudly, so the run isn't left burning GPU hours on a fault ARC cannot fix.
        """
        if self._weights_finite():
            cause = (
                "model weights are finite, so the fault is in the incoming data, "
                "the loss computation, or numeric precision -- not the model. "
                "Check the batch for NaN/Inf, unguarded log/div in the loss, and "
                "fp16 overflow"
            )
        else:
            cause = (
                "weights are still non-finite after restore, so every retained "
                "checkpoint is likely already corrupted"
            )

        action.unrecoverable = True
        action.status = HealthStatus.FAILURE
        action.message = (
            f"UNRECOVERABLE after {self._persistent_failure_count} escape attempts "
            f"| {failure_type} | {cause}"
        )

        if not self.unrecoverable:  # warn once per run, not once per step
            self.unrecoverable = True
            warnings.warn(
                f"ARC cannot recover from '{failure_type}': {cause}. "
                f"Rollback has been retried {self._persistent_failure_count} times "
                f"with no improvement. Stop the run and fix the input pipeline; "
                f"ARC will keep skipping steps but cannot repair this.",
                RuntimeWarning,
                stacklevel=3,
            )

    def _track_consecutive_rollback(self) -> bool:
        if self._last_rollback_step >= 0 and self._step_count - self._last_rollback_step <= 3:
            self._consecutive_rollbacks += 1
        else:
            self._consecutive_rollbacks = 1

        self._last_rollback_step = self._step_count
        return self._consecutive_rollbacks >= self.config.max_consecutive_rollbacks

    def _escape_persistent_failure(self):
        self._persistent_failure_count += 1
        self._skip_until_step = self._step_count + self.config.skip_ahead_steps

        if len(self._checkpoints) > 1:
            self._restore_checkpoint(index=0)

        # Perturb weights. The noise is seeded from the step count so every
        # data-parallel rank applies the identical perturbation.
        generators: Dict[torch.device, torch.Generator] = {}
        with torch.no_grad():
            scale = self.config.weight_perturbation_scale
            for p in self._params:
                gen = generators.get(p.device)
                if gen is None:
                    gen = torch.Generator(device=p.device).manual_seed(self._step_count)
                    generators[p.device] = gen
                magnitude = torch.linalg.vector_norm(p, 1, dtype=torch.float32) / max(p.numel(), 1)
                noise = torch.randn(p.shape, generator=gen, device=p.device, dtype=p.dtype)
                p.add_(noise * (scale * (magnitude + 1e-8)).to(p.dtype))

        for _ in range(2):
            self._reduce_lr()

        self._consecutive_rollbacks = 0
        self._baseline_weight_stats = self._compute_weight_stats()

        self._log(f">> ESCAPE: skip to step {self._skip_until_step}, "
                  f"weights perturbed, LR={self._get_lr():.2e}", level=logging.WARNING)

    def _update_stability(self):
        if len(self._loss_history) < 5:
            return

        recent = list(self._loss_history)[-10:]
        mean_loss = sum(recent) / len(recent)
        if mean_loss > 0:
            variance = sum((x - mean_loss) ** 2 for x in recent) / len(recent)
            cv = math.sqrt(variance) / mean_loss
            self._stability_score = max(0.0, min(1.0, 1.0 - cv))

        if self.config.adaptive_frequency:
            if self._stability_score < 0.5:
                self._checkpoint_frequency = self.config.min_checkpoint_frequency
            elif self._stability_score > 0.9:
                self._checkpoint_frequency = self.config.max_checkpoint_frequency
            else:
                self._checkpoint_frequency = int(
                    self.config.min_checkpoint_frequency +
                    self._stability_score * (
                        self.config.max_checkpoint_frequency -
                        self.config.min_checkpoint_frequency
                    )
                )

    # ================================================================
    # RECOVERY LEDGER
    # ================================================================

    def _log(self, message: str, level: int = logging.INFO) -> None:
        if self.config.verbose:
            logger.log(level, message)

    def _hash(self, prev_hash: str, body: Dict[str, Any]) -> str:
        data = (prev_hash + json.dumps(body, sort_keys=True)).encode()
        if self._ledger_key:
            return hmac.new(self._ledger_key.encode(), data, hashlib.sha256).hexdigest()
        return hashlib.sha256(data).hexdigest()

    def _append_ledger(self, entry: Dict[str, Any]) -> None:
        prev_hash = self.intervention_log[-1]["entry_hash"] if self.intervention_log else ""
        entry["prev_hash"] = prev_hash
        entry["entry_hash"] = self._hash(prev_hash, entry)
        self.intervention_log.append(entry)

    def _log_intervention(self, reason: str, action: StepAction):
        if not self.config.log_interventions:
            return

        def _json_safe(value):
            # NaN/Inf are valid Python floats but not valid JSON; the ledger
            # exists to be exported and consumed externally.
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return None
            return value

        steps_lost = 0
        steps_saved: Optional[int] = None
        if action.rolled_back:
            steps_lost = max(0, self._step_count - self._last_restored_step)
            interval = self.config.baseline_checkpoint_interval
            if interval > 0:
                # Without ARC the run restarts from its last durable checkpoint.
                baseline_step = (self._step_count // interval) * interval
                steps_saved = max(0, self._last_restored_step - baseline_step)

        step_seconds = (time.time() - self._start_time) / max(self._step_count, 1)
        hours_per_step = step_seconds / 3600.0 * self.config.gpu_count
        gpu_hours_saved = None if steps_saved is None else steps_saved * hours_per_step
        dollars_saved = None if gpu_hours_saved is None else gpu_hours_saved * self.config.gpu_hourly_cost

        self._append_ledger({
            "timestamp": datetime.datetime.now().isoformat(),
            "step": self._step_count,
            "reason": reason,
            "loss_value": _json_safe(action.loss_value),
            "action": "rollback" if action.rolled_back else "skip",
            "restored_step": self._last_restored_step if action.rolled_back else None,
            "steps_lost": steps_lost,
            "steps_saved": steps_saved,
            "new_lr": _json_safe(self._get_lr()),
            "stability_score": _json_safe(self._stability_score),
            "lyapunov": _json_safe(self._lyapunov.exponent) if self._lyapunov else None,
            "gpu_hours_lost": _json_safe(steps_lost * hours_per_step),
            "gpu_hours_saved": _json_safe(gpu_hours_saved),
            "dollars_saved": _json_safe(dollars_saved),
        })

    def export_recovery_ledger(self, path: Optional[str] = None) -> str:
        """Exportable audit trail of every intervention: what fired, why, what it saved.

        Returns the JSON string (and writes it to `path` if given). Set
        `config.baseline_checkpoint_interval` for saved-work figures and
        `config.gpu_hourly_cost` for dollars.
        """
        saved = [e["gpu_hours_saved"] for e in self.intervention_log
                 if e.get("gpu_hours_saved") is not None]
        dollars = [e["dollars_saved"] for e in self.intervention_log
                   if e.get("dollars_saved") is not None]
        payload = {
            "generated_at": datetime.datetime.now().isoformat(),
            "team": self.config.team,
            "project": self.config.project,
            "total_steps": self._step_count,
            "total_rollbacks": self.total_rollbacks,
            "failures_detected": self.failures_detected,
            "failures_recovered": self.failures_recovered,
            "savings_baseline_interval": self.config.baseline_checkpoint_interval or None,
            "total_gpu_hours_saved": sum(saved) if saved else None,
            "total_dollars_saved": sum(dollars) if dollars else None,
            "ledger_keyed": bool(self._ledger_key),
            "events": self.intervention_log,
        }
        text = json.dumps(payload, indent=2, allow_nan=False)
        if path:
            with open(path, "w") as f:
                f.write(text)
        return text

    def verify_ledger_integrity(self) -> bool:
        """Verify the intervention_log hash chain: each `entry_hash` covers its
        own fields plus the previous entry's hash, so editing or deleting any
        past entry breaks every hash after it. With a ledger key the hashes are
        HMACs, so a rewritten chain can't be re-signed without the key.
        """
        prev_hash = ""
        for entry in self.intervention_log:
            if entry.get("prev_hash") != prev_hash:
                return False
            body = {k: v for k, v in entry.items() if k != "entry_hash"}
            if not hmac.compare_digest(self._hash(prev_hash, body), entry.get("entry_hash", "")):
                return False
            prev_hash = entry["entry_hash"]
        return True

    def reliability_roi_report(self, path: Optional[str] = None) -> str:
        """Board-report artifact: GPU-hours/dollars/carbon saved this run.

        Savings need `config.baseline_checkpoint_interval`; dollar and carbon
        figures also need `gpu_hourly_cost`, `gpu_power_watts` and
        `carbon_intensity_kg_per_kwh`.
        """
        saved = [e["gpu_hours_saved"] for e in self.intervention_log
                 if e.get("gpu_hours_saved") is not None]
        dollars = [e["dollars_saved"] for e in self.intervention_log
                   if e.get("dollars_saved") is not None]
        lines = [
            "=" * 60,
            "  ARC RELIABILITY ROI REPORT",
            "=" * 60,
            f"  Steps trained:        {self._step_count}",
            f"  Failures detected:    {self.failures_detected}",
            f"  Failures recovered:   {self.failures_recovered}",
        ]
        if saved:
            gpu_hours_saved = sum(saved)
            kwh_saved = gpu_hours_saved * self.config.gpu_power_watts / 1000.0
            carbon_kg_saved = kwh_saved * self.config.carbon_intensity_kg_per_kwh
            lines += [
                f"  GPU-hours saved:      {gpu_hours_saved:.4f}",
                f"  Dollars saved:        ${sum(dollars):,.2f}",
                f"  Carbon (kg CO2) saved:{carbon_kg_saved:.4f}",
            ]
        else:
            lines.append("  Savings:              unknown -- set baseline_checkpoint_interval")
        lines.append("=" * 60)
        report = "\n".join(lines)
        if path:
            with open(path, "w") as f:
                f.write(report)
        return report
