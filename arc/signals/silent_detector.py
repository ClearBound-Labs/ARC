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
import numpy as np
from typing import Optional, Dict, Any, Callable, List, Deque, Union
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
import warnings

from arc.signals.stats_utils import ols_slope
from arc._logging import log as _log

class SilentFailureType(Enum):

    ACCURACY_COLLAPSE = auto()
    MODE_COLLAPSE = auto()
    POSTERIOR_COLLAPSE = auto()
    DEAD_NEURONS = auto()
    REPRESENTATION_COLLAPSE = auto()
    GRADIENT_DEATH = auto()
    LOSS_PLATEAU = auto()
    OVERFITTING = auto()

@dataclass
class SilentDetectorConfig:

    detect_accuracy_collapse: bool = True
    detect_mode_collapse: bool = True
    detect_posterior_collapse: bool = True
    detect_dead_neurons: bool = True
    detect_representation_collapse: bool = True
    detect_gradient_death: bool = True
    detect_loss_plateau: bool = True
    detect_overfitting: bool = True

    accuracy_drop_sigma: float = 3.0
    mode_collapse_threshold: float = 0.01
    posterior_collapse_threshold: float = 0.1
    dead_neuron_threshold: float = 0.01
    representation_similarity_threshold: float = 0.99
    gradient_death_threshold: float = 1e-8
    # Patience in `check()` calls (i.e. training steps), not epochs -- this
    # class has no epoch-boundary information, so it can't convert an
    # epoch count into steps without assuming a fixed steps-per-epoch (the
    # previous `loss_plateau_epochs * 100` did exactly that and silently
    # misfired for any run that wasn't ~100 steps/epoch). Default of 1000
    # matches the old default of 10 epochs * 100 steps/epoch.
    loss_plateau_patience_steps: int = 1000
    loss_plateau_threshold: float = 0.001
    overfit_gap_threshold: float = 2.0

    history_size: int = 100
    min_samples: int = 10
    smoothing_window: int = 5

    verbose: bool = True

@dataclass
class SilentFailureDetection:

    detected: bool
    failure_type: Optional[SilentFailureType] = None
    severity: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""

class MetricTracker:

    def __init__(self, name: str, history_size: int = 100):
        self.name = name
        self.history: Deque[float] = deque(maxlen=history_size)
        self.timestamps: Deque[int] = deque(maxlen=history_size)
        self.step = 0

    def update(self, value: float):

        self.history.append(value)
        self.timestamps.append(self.step)
        self.step += 1

    @property
    def mean(self) -> float:
        if not self.history:
            return 0.0
        return np.mean(list(self.history))

    @property
    def std(self) -> float:
        if len(self.history) < 2:
            return 0.0
        return np.std(list(self.history))

    def recent_mean(self, window: int = 10) -> float:
        if not self.history:
            return 0.0
        recent = list(self.history)[-window:]
        return np.mean(recent)

    def get_trend(self, window: int = 10) -> float:

        if len(self.history) < window:
            return 0.0
        recent = list(self.history)[-window:]
        return ols_slope(recent)

    def detect_anomaly(self, current: float, sigma_threshold: float = 3.0) -> bool:

        if len(self.history) < 10:
            return False
        z_score = abs(current - self.mean) / (self.std + 1e-8)
        return z_score > sigma_threshold

class SilentCrashDetector:

    def __init__(self, config: Optional[SilentDetectorConfig] = None):
        self.config = config or SilentDetectorConfig()

        self.loss_tracker = MetricTracker("loss", self.config.history_size)
        self.val_loss_tracker = MetricTracker("val_loss", self.config.history_size)
        self.accuracy_tracker = MetricTracker("accuracy", self.config.history_size)
        self.grad_norm_tracker = MetricTracker("grad_norm", self.config.history_size)
        self.kl_tracker = MetricTracker("kl_divergence", self.config.history_size)
        self.d_loss_tracker = MetricTracker("discriminator_loss", self.config.history_size)

        self.step = 0
        self.best_loss = float('inf')
        self.steps_without_improvement = 0
        self.detection_history: List[SilentFailureDetection] = []

        self._activation_stats: Dict[str, Dict[str, float]] = {}
        self._representation_buffer: Deque[torch.Tensor] = deque(maxlen=10)

        # Fixed reference point for accuracy-collapse detection, captured once
        # from the first `smoothing_window` accuracy values. Deliberately not
        # re-derived from accuracy_tracker.history[:smoothing_window] on every
        # check -- that deque has maxlen=history_size, so once more than
        # history_size updates occur the "first N" entries silently become
        # "N entries starting ~history_size steps ago" and the baseline drifts
        # along with a slow collapse, making long-horizon collapse invisible.
        self._accuracy_baseline: Optional[float] = None

    def check(
        self,
        loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        val_metric: Optional[float] = None,
        gradients: Optional[Any] = None,
        activations: Optional[Dict[str, torch.Tensor]] = None,
        kl_divergence: Optional[float] = None,
        discriminator_loss: Optional[float] = None,
        representations: Optional[torch.Tensor] = None,
    ) -> SilentFailureDetection:

        self.step += 1
        failures = []

        if loss is not None:
            self.loss_tracker.update(loss)
            self._check_loss_improvement(loss)

        if val_loss is not None:
            self.val_loss_tracker.update(val_loss)

        if val_metric is not None:
            self.accuracy_tracker.update(val_metric)

        if self.config.detect_accuracy_collapse and val_metric is not None:
            result = self._check_accuracy_collapse(val_metric)
            if result.detected:
                failures.append(result)

        if self.config.detect_mode_collapse and discriminator_loss is not None:
            self.d_loss_tracker.update(discriminator_loss)
            result = self._check_mode_collapse(discriminator_loss)
            if result.detected:
                failures.append(result)

        if self.config.detect_posterior_collapse and kl_divergence is not None:
            self.kl_tracker.update(kl_divergence)
            result = self._check_posterior_collapse(kl_divergence)
            if result.detected:
                failures.append(result)

        if self.config.detect_dead_neurons and activations is not None:
            result = self._check_dead_neurons(activations)
            if result.detected:
                failures.append(result)

        if self.config.detect_gradient_death and gradients is not None:
            result = self._check_gradient_death(gradients)
            if result.detected:
                failures.append(result)

        if self.config.detect_representation_collapse and representations is not None:
            result = self._check_representation_collapse(representations)
            if result.detected:
                failures.append(result)

        if self.config.detect_loss_plateau and loss is not None:
            result = self._check_loss_plateau()
            if result.detected:
                failures.append(result)

        if self.config.detect_overfitting and loss is not None and val_loss is not None:
            result = self._check_overfitting(loss, val_loss)
            if result.detected:
                failures.append(result)

        if failures:

            failures.sort(key=lambda x: x.severity, reverse=True)
            result = failures[0]
            self.detection_history.append(result)

            if self.config.verbose:
                _log(f"⚠️ Silent failure detected: {result.failure_type.name}")
                _log(f"   Severity: {result.severity:.2f}, {result.recommendation}")

            return result

        return SilentFailureDetection(detected=False)

    def _check_accuracy_collapse(self, current: float) -> SilentFailureDetection:

        if len(self.accuracy_tracker.history) < self.config.min_samples:
            return SilentFailureDetection(detected=False)

        loss_std = self.loss_tracker.std
        loss_stable = loss_std < 0.1 * abs(self.loss_tracker.mean + 1e-8)

        if self._accuracy_baseline is None:
            self._accuracy_baseline = float(np.mean(
                list(self.accuracy_tracker.history)[:self.config.smoothing_window]
            ))
        baseline = self._accuracy_baseline
        recent = np.mean(list(self.accuracy_tracker.history)[-self.config.smoothing_window:])
        drop = baseline - recent
        drop_sigma = drop / (self.accuracy_tracker.std + 1e-8)

        if loss_stable and drop_sigma > self.config.accuracy_drop_sigma:
            return SilentFailureDetection(
                detected=True,
                failure_type=SilentFailureType.ACCURACY_COLLAPSE,
                severity=min(1.0, drop_sigma / 5.0),
                details={
                    "baseline_accuracy": baseline,
                    "current_accuracy": recent,
                    "drop_sigma": drop_sigma,
                    "loss_std": loss_std,
                },
                recommendation="Rollback and reduce learning rate, or check data pipeline"
            )

        return SilentFailureDetection(detected=False)

    def _check_mode_collapse(self, d_loss: float) -> SilentFailureDetection:

        if d_loss < self.config.mode_collapse_threshold:
            return SilentFailureDetection(
                detected=True,
                failure_type=SilentFailureType.MODE_COLLAPSE,
                severity=1.0 - d_loss / self.config.mode_collapse_threshold,
                details={"discriminator_loss": d_loss},
                recommendation="Rollback and rebalance G/D training, reduce D learning rate"
            )

        return SilentFailureDetection(detected=False)

    def _check_posterior_collapse(self, kl: float) -> SilentFailureDetection:

        if kl < self.config.posterior_collapse_threshold:
            return SilentFailureDetection(
                detected=True,
                failure_type=SilentFailureType.POSTERIOR_COLLAPSE,
                severity=1.0 - kl / self.config.posterior_collapse_threshold,
                details={"kl_divergence": kl},
                recommendation="Rollback and adjust β schedule, use free bits"
            )

        return SilentFailureDetection(detected=False)

    def _check_dead_neurons(self, activations: Dict[str, torch.Tensor]) -> SilentFailureDetection:

        dead_layers = []
        total_dead_ratio = 0.0

        for name, act in activations.items():

            with torch.no_grad():
                if act.numel() == 0:
                    continue

                near_zero = (act.abs() < 1e-6).float().mean().item()

                if near_zero > (1.0 - self.config.dead_neuron_threshold):
                    dead_layers.append((name, near_zero))
                    total_dead_ratio += near_zero

        if dead_layers:
            avg_dead = total_dead_ratio / len(activations) if activations else 0
            return SilentFailureDetection(
                detected=True,
                failure_type=SilentFailureType.DEAD_NEURONS,
                severity=min(1.0, len(dead_layers) / len(activations)),
                details={
                    "dead_layers": dead_layers[:5],
                    "total_dead_layers": len(dead_layers),
                    "avg_dead_ratio": avg_dead,
                },
                recommendation="Rollback and reduce learning rate, or switch to LeakyReLU"
            )

        return SilentFailureDetection(detected=False)

    def _check_gradient_death(self, gradients: Any) -> SilentFailureDetection:

        grad_norms = []

        with torch.no_grad():
            if hasattr(gradients, '__iter__'):
                for p in gradients:
                    if hasattr(p, 'grad') and p.grad is not None:
                        grad_norms.append(p.grad.norm().item())
                    elif isinstance(p, torch.Tensor):
                        grad_norms.append(p.norm().item())

        if not grad_norms:
            return SilentFailureDetection(detected=False)

        total_norm = np.sqrt(sum(g**2 for g in grad_norms))
        self.grad_norm_tracker.update(total_norm)

        if total_norm < self.config.gradient_death_threshold:
            return SilentFailureDetection(
                detected=True,
                failure_type=SilentFailureType.GRADIENT_DEATH,
                severity=1.0 - min(1.0, total_norm / self.config.gradient_death_threshold),
                details={
                    "total_grad_norm": total_norm,
                    "min_layer_norm": min(grad_norms),
                    "max_layer_norm": max(grad_norms),
                },
                recommendation="Rollback and increase learning rate, check skip connections"
            )

        return SilentFailureDetection(detected=False)

    def _check_representation_collapse(self, representations: torch.Tensor) -> SilentFailureDetection:

        with torch.no_grad():

            reps = representations.view(representations.size(0), -1)
            reps = reps / (reps.norm(dim=1, keepdim=True) + 1e-8)

            sim_matrix = torch.mm(reps, reps.t())

            mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
            off_diag_sim = sim_matrix[mask].mean().item()

            self._representation_buffer.append(reps.mean(dim=0).cpu())

        if off_diag_sim > self.config.representation_similarity_threshold:
            return SilentFailureDetection(
                detected=True,
                failure_type=SilentFailureType.REPRESENTATION_COLLAPSE,
                severity=(off_diag_sim - self.config.representation_similarity_threshold) /
                         (1.0 - self.config.representation_similarity_threshold),
                details={
                    "mean_similarity": off_diag_sim,
                    "threshold": self.config.representation_similarity_threshold,
                },
                recommendation="Rollback and add diversity regularization, check data augmentation"
            )

        return SilentFailureDetection(detected=False)

    def _check_loss_improvement(self, loss: float):

        if loss < self.best_loss - self.config.loss_plateau_threshold:
            self.best_loss = loss
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1

    def _check_loss_plateau(self) -> SilentFailureDetection:

        patience = self.config.loss_plateau_patience_steps
        if self.steps_without_improvement >= patience:
            return SilentFailureDetection(
                detected=True,
                failure_type=SilentFailureType.LOSS_PLATEAU,
                severity=min(1.0, self.steps_without_improvement / (patience * 2)),
                details={
                    "steps_without_improvement": self.steps_without_improvement,
                    "best_loss": self.best_loss,
                    "current_loss": list(self.loss_tracker.history)[-1] if self.loss_tracker.history else None,
                },
                recommendation="Try learning rate warmup restart, or reduce LR with patience"
            )

        return SilentFailureDetection(detected=False)

    def _check_overfitting(self, train_loss: float, val_loss: float) -> SilentFailureDetection:

        if train_loss < 1e-8:
            return SilentFailureDetection(detected=False)

        gap_ratio = val_loss / (train_loss + 1e-8)

        if gap_ratio > self.config.overfit_gap_threshold:
            return SilentFailureDetection(
                detected=True,
                failure_type=SilentFailureType.OVERFITTING,
                severity=min(1.0, (gap_ratio - self.config.overfit_gap_threshold) /
                            self.config.overfit_gap_threshold),
                details={
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "gap_ratio": gap_ratio,
                },
                recommendation="Add regularization (dropout, weight decay) or early stop"
            )

        return SilentFailureDetection(detected=False)

    def get_stats(self) -> Dict[str, Any]:

        return {
            "total_checks": self.step,
            "detections": len(self.detection_history),
            "detection_types": [d.failure_type.name for d in self.detection_history],
            "loss_trend": self.loss_tracker.get_trend(),
            "grad_norm_mean": self.grad_norm_tracker.mean,
            "steps_without_improvement": self.steps_without_improvement,
        }

    def reset(self):

        self.loss_tracker = MetricTracker("loss", self.config.history_size)
        self.val_loss_tracker = MetricTracker("val_loss", self.config.history_size)
        self.accuracy_tracker = MetricTracker("accuracy", self.config.history_size)
        self.grad_norm_tracker = MetricTracker("grad_norm", self.config.history_size)
        self.kl_tracker = MetricTracker("kl_divergence", self.config.history_size)
        self.d_loss_tracker = MetricTracker("discriminator_loss", self.config.history_size)
        self.step = 0
        self.best_loss = float('inf')
        self.steps_without_improvement = 0
        self.detection_history = []
        self._accuracy_baseline = None

class ActivationMonitor:

    def __init__(self, model: nn.Module, layers: Optional[List[str]] = None):
        self.model = model
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks = []

        for name, module in model.named_modules():
            if layers is not None and name not in layers:
                continue

            if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.Sigmoid)):
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook

    def get_activations(self) -> Dict[str, torch.Tensor]:

        return self.activations.copy()

    def clear(self):

        self.activations = {}

    def remove_hooks(self):

        for hook in self.hooks:
            hook.remove()
        self.hooks = []

class ValidationMetricCallback:

    def __init__(
        self,
        val_loader,
        metric_fn: Callable[[torch.Tensor, torch.Tensor], float],
        device: str = "cuda",
    ):
        self.val_loader = val_loader
        self.metric_fn = metric_fn
        self.device = device
        self.history: List[float] = []

    def evaluate(self, model: nn.Module) -> float:

        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch[0], batch[1]
                elif isinstance(batch, dict):
                    inputs = batch.get('input', batch.get('x'))
                    targets = batch.get('target', batch.get('y', batch.get('labels')))
                else:
                    continue

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = model(inputs)
                all_preds.append(outputs)
                all_targets.append(targets)

        model.train()

        if all_preds:
            preds = torch.cat(all_preds)
            targets = torch.cat(all_targets)
            metric = self.metric_fn(preds, targets)
            self.history.append(metric)
            return metric

        return 0.0
