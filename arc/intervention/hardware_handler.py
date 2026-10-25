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

"""Classify hardware/runtime errors and recover only where recovery is real.

What can and can't be fixed in-process:

- Most CUDA errors (illegal memory access, device-side assert, launch
  failures) leave the CUDA context unusable for the rest of the process.
  Nothing in Python can reset it. These are reported as *not* recovered with
  "restart the process and resume from the last checkpoint".
- An NCCL/collective failure is recovered only if a barrier across all ranks
  actually succeeds afterwards; ARC never claims to "continue with the
  remaining ranks" (NCCL communicators can't do that).
- Disk-full is recovered only by deleting ARC's *own* old checkpoint files.
- Moving training to the CPU or to another GPU is opt-in, never automatic.
"""

import functools
import glob
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum, auto
from typing import Optional, Dict, Any, Callable

import torch
import torch.nn as nn

from arc.config import default_checkpoint_dir
from arc._logging import log

RESTART = "restart the process and resume from the last durable checkpoint"


class HardwareErrorType(Enum):
    CUDA_ERROR = auto()
    GPU_MEMORY_CORRUPTION = auto()
    GPU_UNAVAILABLE = auto()
    DISK_FULL = auto()
    NETWORK_FAILURE = auto()
    DDP_COMMUNICATION_FAILURE = auto()
    DRIVER_ERROR = auto()
    DEVICE_TIMEOUT = auto()


@dataclass
class HardwareConfig:
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    exponential_backoff: bool = True

    # Opt-in: silently continuing a GPU run on the CPU (10-100x slower) or
    # on another GPU is a decision for the user, not for ARC.
    allow_cpu_fallback: bool = False
    allow_device_switch: bool = False

    checkpoint_to_disk_on_oom: bool = True
    # Disk-full recovery deletes old ARC checkpoints (checkpoint_*.pt) here,
    # keeping the newest `keep_checkpoints`. Nothing else is ever deleted.
    checkpoint_dir: str = default_checkpoint_dir("checkpoints")
    keep_checkpoints: int = 2

    ddp_timeout_seconds: float = 300.0

    monitor_gpu_health: bool = True
    health_check_interval: int = 100

    verbose: bool = True


@dataclass
class HardwareRecoveryResult:
    success: bool
    error_type: Optional[HardwareErrorType] = None
    recovery_action: str = ""
    new_device: Optional[torch.device] = None
    details: Dict[str, Any] = field(default_factory=dict)


class HardwareRecoveryHandler:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[HardwareConfig] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config or HardwareConfig()

        self.original_device = next(model.parameters()).device
        self.current_device = self.original_device
        self.fallback_device = torch.device("cpu")

        self.error_count: Dict[HardwareErrorType, int] = {e: 0 for e in HardwareErrorType}
        self.recovery_count = 0
        self.step = 0

        self._last_health_check = 0
        self._gpu_healthy = True

        if self.config.checkpoint_to_disk_on_oom:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)

        self._log(f"HardwareRecoveryHandler on {self.original_device}")

    def _log(self, message: str, level: int = logging.INFO) -> None:
        if self.config.verbose:
            log(message, level)

    def classify_error(self, error: Exception) -> Optional[HardwareErrorType]:
        error_str = str(error).lower()
        error_type_str = type(error).__name__.lower()

        if 'cuda' in error_str or 'cuda' in error_type_str:
            if 'out of memory' in error_str:
                return None
            if 'driver' in error_str:
                return HardwareErrorType.DRIVER_ERROR
            if 'device' in error_str and ('unavailable' in error_str or 'not found' in error_str):
                return HardwareErrorType.GPU_UNAVAILABLE
            if 'illegal memory access' in error_str or 'corruption' in error_str:
                return HardwareErrorType.GPU_MEMORY_CORRUPTION
            return HardwareErrorType.CUDA_ERROR

        if 'no space' in error_str or 'disk full' in error_str or 'enospc' in error_str:
            return HardwareErrorType.DISK_FULL

        if 'network' in error_str or 'connection' in error_str or 'timeout' in error_str:
            if 'nccl' in error_str or 'gloo' in error_str:
                return HardwareErrorType.DDP_COMMUNICATION_FAILURE
            return HardwareErrorType.NETWORK_FAILURE

        if 'nccl' in error_str or 'collective' in error_str:
            return HardwareErrorType.DDP_COMMUNICATION_FAILURE

        return None

    def try_recover(self, error: Exception) -> HardwareRecoveryResult:
        error_type = self.classify_error(error)

        if error_type is None:
            return HardwareRecoveryResult(
                success=False,
                details={"reason": "Not a hardware error", "original_error": str(error)}
            )

        self.error_count[error_type] += 1
        self._log(f"hardware error: {error_type.name}: {error}", logging.WARNING)

        handlers = {
            HardwareErrorType.CUDA_ERROR: self._recover_cuda_error,
            HardwareErrorType.GPU_MEMORY_CORRUPTION: self._recover_cuda_error,
            HardwareErrorType.DRIVER_ERROR: self._recover_driver_error,
            HardwareErrorType.GPU_UNAVAILABLE: self._recover_gpu_unavailable,
            HardwareErrorType.DISK_FULL: self._recover_disk_full,
            HardwareErrorType.NETWORK_FAILURE: self._recover_network_failure,
            HardwareErrorType.DDP_COMMUNICATION_FAILURE: self._recover_ddp_failure,
        }
        handler = handlers.get(error_type)
        if handler is None:
            return HardwareRecoveryResult(
                success=False, error_type=error_type,
                recovery_action=RESTART,
                details={"reason": "No in-process recovery exists for this error"},
            )
        result = handler()
        result.error_type = error_type
        if result.success:
            self.recovery_count += 1
        return result

    def _recover_cuda_error(self) -> HardwareRecoveryResult:
        # A CUDA error poisons the process's CUDA context; every later CUDA
        # call fails too. empty_cache()/synchronize() can't undo that.
        return HardwareRecoveryResult(
            success=False,
            recovery_action=f"CUDA context is unusable after this error: {RESTART}",
        )

    def _recover_driver_error(self) -> HardwareRecoveryResult:
        if self.config.allow_cpu_fallback:
            return self._fallback_to_cpu()
        return HardwareRecoveryResult(success=False, recovery_action=f"GPU driver error: {RESTART}")

    def _recover_gpu_unavailable(self) -> HardwareRecoveryResult:
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        if self.config.allow_device_switch and not distributed and torch.cuda.device_count() > 1:
            for i in range(torch.cuda.device_count()):
                device = torch.device(f"cuda:{i}")
                if device == self.current_device:
                    continue
                try:
                    self._move_training_state(device)
                except RuntimeError:
                    continue
                return HardwareRecoveryResult(
                    success=True, recovery_action=f"Moved model and optimizer to {device}",
                    new_device=device,
                )
        if self.config.allow_cpu_fallback:
            return self._fallback_to_cpu()
        return HardwareRecoveryResult(success=False, recovery_action=f"GPU unavailable: {RESTART}")

    def _recover_disk_full(self) -> HardwareRecoveryResult:
        files = sorted(glob.glob(os.path.join(self.config.checkpoint_dir, "checkpoint_*.pt")),
                       key=os.path.getmtime)
        stale = files[:max(0, len(files) - self.config.keep_checkpoints)]
        removed = 0
        for path in stale:
            try:
                os.remove(path)
                removed += 1
            except OSError:
                pass

        if removed:
            return HardwareRecoveryResult(
                success=True,
                recovery_action=f"Removed {removed} old ARC checkpoints",
                details={"removed_count": removed},
            )
        return HardwareRecoveryResult(
            success=False,
            recovery_action="No old ARC checkpoints to delete; free disk space and resume",
        )

    def _recover_network_failure(self) -> HardwareRecoveryResult:
        delay = self.config.retry_delay_seconds
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()

        if not distributed:
            # Nothing to re-check: back off once, then skip the failed step.
            time.sleep(delay)
            return HardwareRecoveryResult(
                success=True,
                recovery_action=f"Backed off {delay:.1f}s after a transient network error; "
                                "the failed step was skipped",
            )

        for attempt in range(self.config.max_retries):
            time.sleep(delay)
            try:
                torch.distributed.barrier(timeout=timedelta(seconds=10))
                return HardwareRecoveryResult(
                    success=True,
                    recovery_action=f"All ranks reached a barrier after {attempt + 1} attempt(s)",
                )
            except Exception:
                if self.config.exponential_backoff:
                    delay *= 2

        return HardwareRecoveryResult(
            success=False,
            recovery_action=f"Network did not recover: {RESTART}",
        )

    def _recover_ddp_failure(self) -> HardwareRecoveryResult:
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return HardwareRecoveryResult(
                success=False, recovery_action="No process group is initialized; nothing to resync",
            )
        try:
            torch.distributed.barrier(timeout=timedelta(seconds=self.config.ddp_timeout_seconds))
        except Exception as e:
            return HardwareRecoveryResult(
                success=False,
                recovery_action=f"Ranks could not resynchronize ({e}); relaunch with an elastic "
                                f"launcher (torchrun --max-restarts) and resume from the last checkpoint",
            )
        return HardwareRecoveryResult(success=True, recovery_action="All ranks resynchronized")

    def _move_training_state(self, device: torch.device) -> None:
        self.model.to(device)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and v.dim() > 0:
                    state[k] = v.to(device)
        self.current_device = device

    def _fallback_to_cpu(self) -> HardwareRecoveryResult:
        try:
            self._move_training_state(self.fallback_device)
        except Exception as e:
            return HardwareRecoveryResult(
                success=False,
                recovery_action=f"CPU fallback failed ({e}): {RESTART}",
            )
        self._log("fell back to CPU execution (expect a large slowdown)", logging.WARNING)
        return HardwareRecoveryResult(
            success=True,
            recovery_action="Moved model and optimizer to CPU",
            new_device=self.fallback_device,
        )

    def check_gpu_health(self) -> bool:
        if not torch.cuda.is_available():
            self._gpu_healthy = True
            return True

        try:
            torch.cuda.synchronize()
            free_bytes, _ = torch.cuda.mem_get_info()
            self._gpu_healthy = free_bytes >= 100 * 1024 * 1024
            if not self._gpu_healthy:
                self._log("GPU memory critically low", logging.WARNING)
            return self._gpu_healthy
        except Exception:
            self._gpu_healthy = False
            return False

    def safe_context(self):
        return SafeHardwareContext(self)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "current_device": str(self.current_device),
            "original_device": str(self.original_device),
            "recovery_count": self.recovery_count,
            "error_counts": {e.name: c for e, c in self.error_count.items() if c > 0},
            "gpu_healthy": self._gpu_healthy,
        }


class SafeHardwareContext:

    def __init__(self, handler: HardwareRecoveryHandler):
        self.handler = handler

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None and self.handler.classify_error(exc_val) is not None:
            return self.handler.try_recover(exc_val).success
        return False


def hardware_safe(func: Callable) -> Callable:
    """Retry `func` on transient device errors (not on sticky CUDA errors,
    which no retry can fix)."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                error_str = str(e).lower()
                transient = 'timeout' in error_str or 'temporarily unavailable' in error_str
                if transient and attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                raise

    return wrapper


def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_info() -> Dict[str, Any]:
    info = {
        "best_device": str(get_best_device()),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }

    if torch.cuda.is_available():
        info["cuda_devices"] = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["cuda_devices"].append({
                "name": props.name,
                "total_memory_gb": props.total_memory / 1e9,
                "compute_capability": f"{props.major}.{props.minor}",
            })

    if hasattr(torch.backends, 'mps'):
        info["mps_available"] = torch.backends.mps.is_available()

    return info
