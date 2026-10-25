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

"""Framework integrations.

The Hugging Face and Lightning callbacks run `arc.Controller` in the hook that
fires after backward/clipping and *before* the optimizer step. On a failure
they roll back and clear every ``.grad``: torch optimizers skip parameters
whose grad is None, so the poisoned update never happens.

Neither framework hands the loss to that hook, so these callbacks check
gradients, weights and training dynamics (NaN/Inf, explosions, weight
corruption, Lyapunov, trends) rather than the loss value. With the trainers'
default gradient clipping on, an exploding-but-finite gradient is already
clipped by the time ARC sees it; NaN/Inf is still caught.
"""

import copy
import dataclasses
import logging
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from arc.core.controller import Controller, ControllerConfig
from arc._logging import log

try:
    from transformers import TrainerCallback
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    TrainerCallback = object

try:
    from lightning.pytorch.callbacks import Callback as _LightningBase
    LIGHTNING_AVAILABLE = True
except ImportError:
    try:
        from pytorch_lightning.callbacks import Callback as _LightningBase
        LIGHTNING_AVAILABLE = True
    except ImportError:
        LIGHTNING_AVAILABLE = False
        _LightningBase = object


def _callback_config(
    config: Optional[ControllerConfig],
    verbose: bool,
    loss_threshold: Optional[float],
    gradient_threshold: Optional[float],
) -> ControllerConfig:
    cfg = copy.deepcopy(config) if config is not None else ControllerConfig()
    cfg.verbose = verbose
    if loss_threshold is not None:
        cfg.loss_explosion_threshold = loss_threshold
    if gradient_threshold is not None:
        cfg.gradient_explosion_threshold = gradient_threshold
    if loss_threshold is not None or gradient_threshold is not None:
        cfg.auto_calibrate = False  # explicit thresholds win
    return cfg


def _skip_optimizer_step(model: nn.Module) -> None:
    for p in model.parameters():
        p.grad = None


class HuggingFaceCallback(TrainerCallback):
    """`transformers.Trainer` callback backed by `arc.Controller`.

        trainer = Trainer(..., callbacks=[HuggingFaceCallback()])
    """

    def __init__(
        self,
        config: Optional[ControllerConfig] = None,
        enable_rollback: bool = True,
        verbose: bool = True,
        loss_threshold: Optional[float] = None,
        gradient_threshold: Optional[float] = None,
        lr_spike_factor: Optional[float] = None,
    ):
        if not HF_AVAILABLE:
            raise ImportError("transformers package required for HuggingFaceCallback")
        super().__init__()
        if lr_spike_factor is not None:
            warnings.warn("lr_spike_factor is no longer used and is ignored", DeprecationWarning,
                          stacklevel=2)
        self.enable_rollback = enable_rollback
        self.verbose = verbose
        self._config = _callback_config(config, verbose, loss_threshold, gradient_threshold)
        self.controller: Optional[Controller] = None
        self.skipped_steps = 0

    def on_train_begin(self, args, state, control, model=None, optimizer=None,
                       lr_scheduler=None, **kwargs):
        if not self.enable_rollback or model is None or optimizer is None:
            return
        cfg = copy.deepcopy(self._config)
        cfg.external_grad_scaler = bool(getattr(args, "fp16", False))
        # Trainer wraps the torch optimizer (accelerate's AcceleratedOptimizer);
        # ARC hooks the real one so its snapshot hooks fire.
        torch_optimizer = getattr(optimizer, "optimizer", optimizer)
        self.controller = Controller(model, torch_optimizer, scheduler=lr_scheduler, config=cfg)

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if self.controller is None:
            return
        action = self.controller.step(None)
        if action.should_skip:
            _skip_optimizer_step(self.controller.model)
            self.skipped_steps += 1

    def on_train_end(self, args, state, control, **kwargs):
        if self.controller is not None:
            self.controller.close()
            if self.verbose:
                self.controller.summary()


class LightningCallback(_LightningBase):
    """Lightning callback backed by `arc.Controller`.

        trainer = L.Trainer(..., callbacks=[LightningCallback()])
    """

    def __init__(
        self,
        config: Optional[ControllerConfig] = None,
        enable_rollback: bool = True,
        verbose: bool = True,
        loss_threshold: Optional[float] = None,
        gradient_threshold: Optional[float] = None,
    ):
        if not LIGHTNING_AVAILABLE:
            raise ImportError("lightning (or pytorch_lightning) is required for LightningCallback")
        super().__init__()
        self.enable_rollback = enable_rollback
        self.verbose = verbose
        self._config = _callback_config(config, verbose, loss_threshold, gradient_threshold)
        self.controller: Optional[Controller] = None
        self.skipped_steps = 0

    def on_train_start(self, trainer, pl_module):
        if not self.enable_rollback or not trainer.optimizers:
            return
        if len(trainer.optimizers) > 1:
            warnings.warn("ARC protects the first optimizer only", stacklevel=2)
        scheduler = None
        if getattr(trainer, "lr_scheduler_configs", None):
            scheduler = trainer.lr_scheduler_configs[0].scheduler
        cfg = copy.deepcopy(self._config)
        # Lightning unscales before on_before_optimizer_step; its fp16 scaler
        # skips overflowing steps itself.
        cfg.external_grad_scaler = str(trainer.precision).startswith("16")
        self.controller = Controller(pl_module, trainer.optimizers[0], scheduler=scheduler,
                                     config=cfg)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        optimizer = getattr(optimizer, "_optimizer", optimizer)  # LightningOptimizer
        if self.controller is None or optimizer is not self.controller.optimizer:
            return
        action = self.controller.step(None)
        if action.should_skip:
            _skip_optimizer_step(pl_module)
            self.skipped_steps += 1

    def on_train_end(self, trainer, pl_module):
        if self.controller is not None:
            self.controller.close()
            if self.verbose:
                self.controller.summary()


class YOLOCallback:
    """Ultralytics has no hook between backward and the optimizer step, so
    this callback detects failures after the step and rolls the weights back
    to the last good checkpoint."""

    def __init__(
        self,
        loss_threshold: float = 100.0,
        box_loss_threshold: float = 20.0,
        cls_loss_threshold: float = 50.0,
        enable_rollback: bool = True,
        verbose: bool = True,
    ):
        self.loss_threshold = loss_threshold
        self.box_loss_threshold = box_loss_threshold
        self.cls_loss_threshold = cls_loss_threshold
        self.enable_rollback = enable_rollback
        self.verbose = verbose

        self._rollback = None
        self._rollback_count = 0
        self._initialized = False

        self._trainer = None
        self._model = None
        self._optimizer = None

    def _lazy_init(self, trainer):
        if self._initialized:
            return

        self._trainer = trainer
        self._model = trainer.model
        self._optimizer = trainer.optimizer

        if self.enable_rollback and self._model is not None and self._optimizer is not None:
            from arc.intervention.rollback import WeightRollback, RollbackConfig
            from arc.intervention.heuristics import DetectionHeuristics, DetectionConfig

            config = RollbackConfig(
                loss_explosion_threshold=self.loss_threshold,
                checkpoint_frequency=100,
                fast_grad_norm=True,
                adaptive_frequency=True,
            )
            self._rollback = WeightRollback(self._model, self._optimizer, config, verbose=False)

            det_config = DetectionConfig(
                box_loss_threshold=self.box_loss_threshold,
                cls_loss_threshold=self.cls_loss_threshold,
            )
            self._heuristics = DetectionHeuristics(self._model, det_config, verbose=False)

            if self.verbose:
                log("YOLO callback: protection enabled")

        self._initialized = True

    def on_train_batch_end(self, trainer):
        self._lazy_init(trainer)

        if self._rollback is None:
            return

        loss = getattr(trainer, 'loss', None)
        if loss is None and hasattr(trainer, 'tloss'):
            loss = trainer.tloss

        if loss is not None:
            if isinstance(loss, (list, tuple)):
                loss = sum(loss) if all(isinstance(l, torch.Tensor) for l in loss) else loss[0]

            if isinstance(loss, torch.Tensor):
                action = self._rollback.step(loss.detach())

                if action.rolled_back:
                    self._rollback_count += 1
                    if self.verbose:
                        log(f"rollback #{self._rollback_count} ({action.failure_type})",
                            logging.WARNING)

    def on_train_epoch_end(self, trainer):
        if self._rollback is not None:
            self._rollback.end_epoch()

    def on_train_end(self, trainer):
        if self.verbose and self._rollback is not None:
            stats = self._rollback.get_stats()
            log(f"YOLO summary: {stats['total_rollbacks']} rollbacks")

    def wrap_trainer(self, trainer):
        if hasattr(trainer, 'add_callback'):
            trainer.add_callback('on_train_batch_end', self.on_train_batch_end)
            trainer.add_callback('on_train_epoch_end', self.on_train_epoch_end)
            trainer.add_callback('on_train_end', self.on_train_end)
        else:
            original_batch_end = getattr(trainer, 'on_train_batch_end', None)

            def wrapped_batch_end(*args, **kwargs):
                if original_batch_end:
                    original_batch_end(*args, **kwargs)
                self.on_train_batch_end(trainer)

            trainer.on_train_batch_end = wrapped_batch_end


class GenericTrainingWrapper:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_threshold: float = 100.0,
        gradient_threshold: float = 1e4,
        enable_heuristics: bool = True,
        verbose: bool = True,
    ):
        from arc.intervention.rollback import WeightRollback, RollbackConfig

        config = RollbackConfig(
            loss_explosion_threshold=loss_threshold,
            gradient_explosion_threshold=gradient_threshold,
            fast_grad_norm=True,
            adaptive_frequency=True,
        )

        self._rollback = WeightRollback(model, optimizer, config, verbose=verbose)

        self._heuristics = None
        if enable_heuristics:
            from arc.intervention.heuristics import AutoHeuristics
            self._heuristics = AutoHeuristics(model, verbose=False)

        self.verbose = verbose

    def step(self, loss: torch.Tensor, output: Optional[torch.Tensor] = None) -> 'WrapperAction':
        action = self._rollback.step(loss)

        result = WrapperAction(
            skip_step=action.should_skip,
            rolled_back=action.rolled_back,
            failure_type=action.failure_type,
        )

        if self._heuristics is not None and output is not None:
            heuristic_result = self._heuristics.check(output)
            if heuristic_result['critical']:
                result.warnings.extend(heuristic_result['warnings'])

        return result

    def end_epoch(self):
        self._rollback.end_epoch()

    def get_stats(self) -> Dict[str, Any]:
        return self._rollback.get_stats()


@dataclass
class WrapperAction:
    skip_step: bool
    rolled_back: bool
    failure_type: Optional[str] = None
    warnings: list = dataclasses.field(default_factory=list)


def get_callback(framework: str, **kwargs):
    framework = framework.lower()

    if framework in ["huggingface", "hf", "transformers"]:
        return HuggingFaceCallback(**kwargs)
    elif framework in ["lightning", "pl", "pytorch_lightning"]:
        return LightningCallback(**kwargs)
    elif framework in ["yolo", "ultralytics"]:
        return YOLOCallback(**kwargs)
    else:
        raise ValueError(f"Unknown framework: {framework}. Use 'huggingface', 'lightning', or 'yolo'")
