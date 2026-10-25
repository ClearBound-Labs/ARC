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

"""Mixed-precision rollback, built on `arc.Controller`.

Call order per step (the Controller unscales the gradients itself, so don't
call ``scaler.unscale_()``):

    scaler.scale(loss).backward()
    action = amp_rollback.step(loss)
    if not action.should_skip:
        scaler.step(optimizer)
        scaler.update()

Gradients are judged at their true scale. A GradScaler overflow is routine
fp16 behaviour (the scaler skips that step and lowers its scale), not a
failure, so it never triggers a rollback.
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from arc.core.controller import Controller, ControllerConfig, HealthStatus, StepAction

AMP_AVAILABLE = True  # torch.amp ships with every supported PyTorch


@dataclass
class AMPConfig:
    dtype: str = "fp16"
    use_autocast: bool = True

    # fp16 weights within 10% of this magnitude count as an overflow.
    detect_overflow: bool = True
    overflow_threshold: float = 65504.0
    bf16_overflow_threshold: float = 3.4e38

    coordinate_scaler: bool = True
    min_scale: float = 1.0

    rollback_on_overflow: bool = True
    reduce_scale_on_rollback: bool = True
    scale_reduction_factor: float = 0.5


@dataclass
class AMPRollbackAction:
    step: int
    rolled_back: bool
    overflow_detected: bool
    overflow_type: Optional[str] = None
    steps_back: int = 0
    current_scale: Optional[float] = None
    new_scale: Optional[float] = None
    should_skip: bool = False


class AMPRollback:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: Optional[torch.amp.GradScaler] = None,
        config: Optional[AMPConfig] = None,
        verbose: bool = True,
        controller_config: Optional[ControllerConfig] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        self.config = config or AMPConfig()
        self.verbose = verbose
        self._overflow_threshold = (
            self.config.bf16_overflow_threshold if self.config.dtype == "bf16"
            else self.config.overflow_threshold
        )
        self._controller = Controller(
            model, optimizer, scaler=scaler,
            config=controller_config or ControllerConfig(
                verbose=verbose, enable_lyapunov=False, enable_trend=False),
        )
        self._overflow_count = 0

    def _weight_near_overflow(self) -> bool:
        if not self.config.detect_overflow:
            return False
        limits = [
            torch.linalg.vector_norm(p.detach(), float("inf")) > self._overflow_threshold * 0.9
            for p in self.model.parameters()
            if p.dtype in (torch.float16, torch.bfloat16)
        ]
        return bool(limits) and bool(torch.stack([x.to(limits[0].device) for x in limits]).any())

    def step(
        self,
        loss: torch.Tensor,
        scaler: Optional[torch.amp.GradScaler] = None,
    ) -> AMPRollbackAction:
        if scaler is not None and scaler is not self._controller.scaler:
            self.scaler = scaler
            self._controller.scaler = scaler

        if self.config.rollback_on_overflow and self._weight_near_overflow():
            self._controller._step_count += 1
            action = self._controller._handle_failure(
                StepAction(step=self._controller._step_count), "weight_near_overflow")
        else:
            action = self._controller.step(loss)

        fp16_overflow = action.status == HealthStatus.WARNING and "overflow" in action.message
        if fp16_overflow:
            self._overflow_count += 1

        result = AMPRollbackAction(
            step=action.step,
            rolled_back=action.rolled_back,
            overflow_detected=fp16_overflow or action.failure_type == "weight_near_overflow",
            overflow_type=action.failure_type or ("gradient_overflow" if fp16_overflow else None),
            should_skip=action.should_skip,
        )

        if self.scaler is not None and self.scaler.is_enabled():
            if action.rolled_back and self.config.reduce_scale_on_rollback:
                new_scale = max(self.scaler.get_scale() * self.config.scale_reduction_factor,
                                self.config.min_scale)
                self.scaler.update(new_scale)
                result.new_scale = new_scale
            elif self.config.coordinate_scaler and self.scaler.get_scale() < self.config.min_scale:
                self.scaler.update(self.config.min_scale)
            result.current_scale = self.scaler.get_scale()

        return result

    def end_epoch(self):
        self._overflow_count = 0

    def get_stats(self) -> Dict[str, Any]:
        stats = self._controller.get_stats()
        stats.update({
            'overflow_count': self._overflow_count,
            'current_scale': self.scaler.get_scale() if self.scaler else None,
            'dtype': self.config.dtype,
        })
        return stats


class SafeAutocast:

    def __init__(
        self,
        amp_rollback: Optional[AMPRollback] = None,
        device_type: str = "cuda",
        dtype: Optional[torch.dtype] = None,
        enabled: bool = True,
    ):
        self.amp_rollback = amp_rollback
        self.device_type = device_type
        self.enabled = enabled

        if dtype is None:
            if amp_rollback and amp_rollback.config.dtype == "bf16":
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
        else:
            self.dtype = dtype

    def __enter__(self):
        if self.enabled:
            self._autocast = torch.autocast(device_type=self.device_type, dtype=self.dtype)
            return self._autocast.__enter__()
        return self

    def __exit__(self, *args):
        if self.enabled:
            return self._autocast.__exit__(*args)
        return False


def create_amp_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dtype: str = "fp16",
    device: str = "cuda",
) -> tuple:
    if device == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA not available, AMP disabled; training in fp32")
        return model, None, None

    # bf16 has fp32's exponent range and needs no loss scaling.
    scaler = torch.amp.GradScaler(device, enabled=(dtype == "fp16"))
    amp_rollback = AMPRollback(model, optimizer, scaler, AMPConfig(dtype=dtype))
    return model, scaler, amp_rollback
