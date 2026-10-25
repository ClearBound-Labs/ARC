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
from typing import Optional, Dict, Any, Callable, List, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import gc
import warnings
import functools
from arc._logging import log as _log

class OOMStrategy(Enum):
    CLEAR_CACHE = auto()
    REDUCE_BATCH = auto()
    GRADIENT_CHECKPOINT = auto()
    CPU_OFFLOAD_OPTIMIZER = auto()
    CPU_ONLY = auto()

@dataclass
class OOMConfig:
    enabled: bool = True
    max_retries: int = 3
    strategies: List[OOMStrategy] = field(default_factory=lambda: [
        OOMStrategy.CLEAR_CACHE,
        OOMStrategy.REDUCE_BATCH,
        OOMStrategy.GRADIENT_CHECKPOINT,
        OOMStrategy.CPU_OFFLOAD_OPTIMIZER,
    ])

    batch_reduction_factor: float = 0.5
    min_batch_size: int = 1

    checkpoint_modules: List[str] = field(default_factory=list)

    offload_optimizer: bool = True
    offload_device: str = "cpu"

    verbose: bool = True

@dataclass
class OOMRecoveryState:
    oom_count: int = 0
    current_batch_size: Optional[int] = None
    original_batch_size: Optional[int] = None
    gradient_checkpointing_enabled: bool = False
    optimizer_offloaded: bool = False
    strategies_applied: List[OOMStrategy] = field(default_factory=list)
    last_oom_step: int = -1
    successful_recoveries: int = 0
    failed_recoveries: int = 0

class OOMRecoveryHandler:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[OOMConfig] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config or OOMConfig()
        self.state = OOMRecoveryState()

        self._checkpointable_modules: List[nn.Module] = []

        self.device = next(model.parameters()).device
        self.is_cuda = self.device.type == "cuda"

        self._find_checkpointable_modules()

        if self.config.verbose:
            _log(f"OOMRecoveryHandler initialized")
            _log(f"   Device: {self.device}")
            _log(f"   Checkpointable modules: {len(self._checkpointable_modules)}")

    def _find_checkpointable_modules(self):
        from torch.utils.checkpoint import checkpoint

        for name, module in self.model.named_modules():
            if any(t in type(module).__name__.lower() for t in [
                'block', 'layer', 'attention', 'transformer', 'encoder', 'decoder'
            ]):
                if len(list(module.children())) > 0:
                    self._checkpointable_modules.append(module)

    def _clear_cuda_cache(self) -> bool:
        if not self.is_cuda:
            return False

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        if self.config.verbose:
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            _log(f"   🧹 Cleared CUDA cache. Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

        return True

    def _reduce_batch_size(self, current_batch: Any) -> Tuple[bool, Any]:
        if current_batch is None:
            return False, None

        if isinstance(current_batch, torch.Tensor):
            batch_size = current_batch.size(0)
        elif isinstance(current_batch, (list, tuple)) and len(current_batch) > 0:
            if isinstance(current_batch[0], torch.Tensor):
                batch_size = current_batch[0].size(0)
            else:
                return False, current_batch
        elif isinstance(current_batch, dict):
            first_tensor = next((v for v in current_batch.values() if isinstance(v, torch.Tensor)), None)
            if first_tensor is not None:
                batch_size = first_tensor.size(0)
            else:
                return False, current_batch
        else:
            return False, current_batch

        new_batch_size = max(
            self.config.min_batch_size,
            int(batch_size * self.config.batch_reduction_factor)
        )

        if new_batch_size >= batch_size:
            return False, current_batch

        if self.state.original_batch_size is None:
            self.state.original_batch_size = batch_size
        self.state.current_batch_size = new_batch_size

        reduced_batch = self._slice_batch(current_batch, new_batch_size)

        if self.config.verbose:
            _log(f"   Reduced batch size: {batch_size} → {new_batch_size}")

        return True, reduced_batch

    def _slice_batch(self, batch: Any, new_size: int) -> Any:
        if isinstance(batch, torch.Tensor):
            return batch[:new_size]
        elif isinstance(batch, (list, tuple)):
            batch_type = type(batch)
            return batch_type(self._slice_batch(item, new_size) for item in batch)
        elif isinstance(batch, dict):
            return {k: self._slice_batch(v, new_size) for k, v in batch.items()}
        else:
            return batch

    def _enable_gradient_checkpointing(self) -> bool:
        if self.state.gradient_checkpointing_enabled:
            return False

        enabled_count = 0

        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            self.state.gradient_checkpointing_enabled = True
            enabled_count = 1
        else:
            for module in self._checkpointable_modules:
                if hasattr(module, 'gradient_checkpointing'):
                    module.gradient_checkpointing = True
                    enabled_count += 1

        if enabled_count > 0:
            self.state.gradient_checkpointing_enabled = True
            if self.config.verbose:
                _log(f"   Enabled gradient checkpointing on {enabled_count} modules")
            return True

        return False

    def _offload_optimizer_to_cpu(self) -> bool:
        if self.state.optimizer_offloaded:
            return False

        if not self.is_cuda:
            return False

        moved_count = 0
        for param_state in self.optimizer.state.values():
            for key, value in param_state.items():
                if isinstance(value, torch.Tensor) and value.device.type == "cuda":
                    param_state[key] = value.cpu()
                    moved_count += 1

        if moved_count > 0:
            self.state.optimizer_offloaded = True
            gc.collect()
            torch.cuda.empty_cache()

            if self.config.verbose:
                _log(f"   Offloaded {moved_count} optimizer tensors to CPU")
            return True

        return False

    def _move_to_cpu(self) -> bool:
        if self.device.type == "cpu":
            return False

        self.model.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        self.device = torch.device("cpu")

        if self.config.verbose:
            _log(f"   Moved model to CPU (last resort)")

        return True

    def try_recover(self, batch: Any = None) -> Tuple[bool, Any]:
        self.state.oom_count += 1
        current_batch = batch

        for strategy in self.config.strategies:
            if strategy in self.state.strategies_applied:
                continue

            success = False

            if strategy == OOMStrategy.CLEAR_CACHE:
                success = self._clear_cuda_cache()
            elif strategy == OOMStrategy.REDUCE_BATCH:
                success, current_batch = self._reduce_batch_size(current_batch)
            elif strategy == OOMStrategy.GRADIENT_CHECKPOINT:
                success = self._enable_gradient_checkpointing()
            elif strategy == OOMStrategy.CPU_OFFLOAD_OPTIMIZER:
                success = self._offload_optimizer_to_cpu()
            elif strategy == OOMStrategy.CPU_ONLY:
                success = self._move_to_cpu()

            if success:
                self.state.strategies_applied.append(strategy)
                self.state.successful_recoveries += 1
                return True, current_batch

        self.state.failed_recoveries += 1
        return False, current_batch

    def safe_forward(
        self,
        forward_fn: Callable[[Any], torch.Tensor],
        batch: Any = None,
    ) -> torch.Tensor:
        retries = 0
        current_batch = batch

        while retries < self.config.max_retries:
            try:
                return forward_fn(current_batch)
            except RuntimeError as e:
                # Narrow match only ("out of memory"), not "CUDA" generically --
                # a broad "CUDA" substring match would funnel unrelated CUDA
                # errors (illegal memory access, device-side assert, cuDNN
                # errors) into OOM recovery instead of surfacing the real bug.
                # Matches the narrower check already used by safe_backward/
                # safe_step below and by hardware_handler.py.
                if "out of memory" in str(e).lower():
                    if self.config.verbose:
                        _log(f"OOM detected (attempt {retries + 1}/{self.config.max_retries})")

                    success, current_batch = self.try_recover(current_batch)

                    if not success:
                        raise OOMRecoveryFailed(
                            f"All OOM recovery strategies exhausted after {retries + 1} attempts"
                        )

                    retries += 1
                else:
                    raise

        raise OOMRecoveryFailed(f"Max retries ({self.config.max_retries}) exceeded")

    def safe_backward(self, loss: torch.Tensor) -> bool:
        retries = 0

        while retries < self.config.max_retries:
            try:
                loss.backward()
                return True
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if self.config.verbose:
                        _log(f"OOM during backward (attempt {retries + 1})")

                    success, _ = self.try_recover()
                    if not success:
                        raise OOMRecoveryFailed("OOM during backward, strategies exhausted")

                    retries += 1
                else:
                    raise

        raise OOMRecoveryFailed("Max retries during backward exceeded")

    def safe_step(self) -> bool:
        retries = 0

        while retries < self.config.max_retries:
            try:
                if self.state.optimizer_offloaded:
                    self._sync_optimizer_to_device()

                self.optimizer.step()
                return True
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    if self.config.verbose:
                        _log(f"OOM during optimizer step (attempt {retries + 1})")

                    success, _ = self.try_recover()
                    if not success:
                        raise OOMRecoveryFailed("OOM during optimizer step")

                    retries += 1
                else:
                    raise

        raise OOMRecoveryFailed("Max retries during optimizer step exceeded")

    def _sync_optimizer_to_device(self):
        if not self.state.optimizer_offloaded:
            return

        for param_state in self.optimizer.state.values():
            for key, value in param_state.items():
                if isinstance(value, torch.Tensor) and value.device.type == "cpu":
                    param_state[key] = value.to(self.device, non_blocking=True)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "oom_count": self.state.oom_count,
            "successful_recoveries": self.state.successful_recoveries,
            "failed_recoveries": self.state.failed_recoveries,
            "strategies_applied": [s.name for s in self.state.strategies_applied],
            "current_batch_size": self.state.current_batch_size,
            "original_batch_size": self.state.original_batch_size,
            "gradient_checkpointing": self.state.gradient_checkpointing_enabled,
            "optimizer_offloaded": self.state.optimizer_offloaded,
        }

    def reset_strategies(self):
        self.state.strategies_applied = []

class OOMRecoveryFailed(Exception):
    pass

def oom_protected(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        retries = 0
        max_retries = 3

        while retries < max_retries:
            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    retries += 1
                    _log(f"OOM in {func.__name__}, retry {retries}/{max_retries}")
                else:
                    raise

        raise OOMRecoveryFailed(f"OOM in {func.__name__} after {max_retries} retries")

    return wrapper

def get_memory_stats() -> Dict[str, float]:
    stats = {"cpu_percent": 0.0}

    try:
        import psutil
        stats["cpu_percent"] = psutil.virtual_memory().percent
        stats["cpu_available_gb"] = psutil.virtual_memory().available / 1e9
    except ImportError:
        pass

    if torch.cuda.is_available():
        stats["cuda_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
        stats["cuda_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
        stats["cuda_max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9

        total = torch.cuda.get_device_properties(0).total_memory
        stats["cuda_total_gb"] = total / 1e9
        stats["cuda_free_gb"] = (total - torch.cuda.memory_allocated()) / 1e9

    return stats

def estimate_model_memory(model: nn.Module, include_optimizer: bool = True) -> Dict[str, float]:
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())

    grad_bytes = param_bytes

    opt_bytes = param_bytes * 2 if include_optimizer else 0

    return {
        "params_gb": param_bytes / 1e9,
        "buffers_gb": buffer_bytes / 1e9,
        "gradients_gb": grad_bytes / 1e9,
        "optimizer_gb": opt_bytes / 1e9,
        "total_gb": (param_bytes + buffer_bytes + grad_bytes + opt_bytes) / 1e9,
    }