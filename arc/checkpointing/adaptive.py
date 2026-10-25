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
import hashlib
import hmac
import io
import os
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn

from arc.config import default_checkpoint_dir
from arc._logging import log as _log

class CheckpointStrategy(Enum):
    FULL_CPU = auto()
    QUANTIZED_FP16 = auto()
    QUANTIZED_INT8 = auto()
    INCREMENTAL_DELTA = auto()
    STREAMING_DISK = auto()
    SHARDED = auto()

@dataclass
class AdaptiveCheckpointConfig:
    auto_select_strategy: bool = True
    preferred_strategy: Optional[CheckpointStrategy] = None

    full_cpu_threshold: float = 4.0
    quantized_threshold: float = 2.0
    incremental_threshold: float = 1.0

    delta_threshold: float = 1e-6
    delta_check_layers: int = 10

    # Opt-in: store optimizer state as bf16 (fp32's exponent range, so Adam's
    # tiny second moments survive; fp16/int8 would flush them to zero).
    quantize_optimizer: bool = False

    disk_checkpoint_dir: str = default_checkpoint_dir("checkpoints")
    max_disk_checkpoints: int = 3
    compress_disk: bool = False

    # Checkpoint integrity & signing: sha256 content hash always
    # verified on restore if present; HMAC signature verified too if a key
    # is configured, proving the checkpoint wasn't tampered with off-key.
    sign_checkpoints: bool = False
    signing_key: Optional[str] = field(default=None, repr=False)
    # Permit weights_only=False when loading disk checkpoints (can execute
    # code). Only for checkpoint directories nobody else can write to.
    allow_unsafe_load: bool = False

    shard_count: int = 4

    max_checkpoints: int = 3
    checkpoint_frequency: int = 50
    verbose: bool = True

@dataclass
class CheckpointMetadata:
    step: int
    strategy: CheckpointStrategy
    model_size_bytes: int
    checkpoint_size_bytes: int
    compression_ratio: float
    timestamp: float
    is_incremental: bool = False
    # Stable checkpoint_id of the base full checkpoint this delta applies on
    # top of (matches CheckpointMetadata.step's companion checkpoint_id in
    # the stored checkpoint dict). NOT a positional index into `checkpoints`
    # -- that deque rotates as old entries are evicted, so a positional
    # index silently points to the wrong checkpoint once maxlen is exceeded.
    # Actual restore already resolves via this stable id (see
    # _resolve_incremental), this field just mirrors it for introspection.
    base_checkpoint_id: Optional[int] = None

class AdaptiveCheckpointer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[AdaptiveCheckpointConfig] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config or AdaptiveCheckpointConfig()
        self.device = next(model.parameters()).device

        self.checkpoints: deque = deque(maxlen=self.config.max_checkpoints)
        self.metadata: List[CheckpointMetadata] = []

        self.step = 0
        self.current_strategy: Optional[CheckpointStrategy] = None
        self._last_full_state: Optional[Dict[str, torch.Tensor]] = None
        self._next_checkpoint_id: int = 0
        self._last_full_checkpoint_id: Optional[int] = None
        # Most recent checkpoint's id, full or incremental -- an incremental
        # delta chains off whichever checkpoint immediately preceded it, not
        # always the last *full* one, or a delta-of-a-delta reconstructs from
        # a stale base and silently drops every change made since (see
        # _resolve_incremental).
        self._last_checkpoint_id: Optional[int] = None

        self.model_size_bytes = self._calculate_model_size()
        self.optimizer_size_bytes = self._calculate_optimizer_size()

        if self.config.disk_checkpoint_dir:
            os.makedirs(self.config.disk_checkpoint_dir, exist_ok=True)

        if self.config.auto_select_strategy:
            self.current_strategy = self._select_best_strategy()
        else:
            self.current_strategy = self.config.preferred_strategy or CheckpointStrategy.FULL_CPU

        if self.config.verbose:
            self._print_init_info()

    def _calculate_model_size(self) -> int:
        total = 0
        for param in self.model.parameters():
            total += param.numel() * param.element_size()
        for buffer in self.model.buffers():
            total += buffer.numel() * buffer.element_size()
        return total

    def _calculate_optimizer_size(self) -> int:
        total = 0
        for state in self.optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    total += value.numel() * value.element_size()
        return total

    def _get_available_memory(self) -> Dict[str, int]:
        import psutil

        memory = {
            "cpu_available": psutil.virtual_memory().available,
            "cpu_total": psutil.virtual_memory().total,
        }

        if torch.cuda.is_available():
            memory["gpu_available"] = torch.cuda.get_device_properties(0).total_memory -                                      torch.cuda.memory_allocated()
            memory["gpu_total"] = torch.cuda.get_device_properties(0).total_memory

        return memory

    def _select_best_strategy(self) -> CheckpointStrategy:
        try:
            memory = self._get_available_memory()
            cpu_available = memory["cpu_available"]

            total_size = self.model_size_bytes + self.optimizer_size_bytes

            if cpu_available > total_size * 3 * self.config.full_cpu_threshold:
                return CheckpointStrategy.FULL_CPU
            elif cpu_available > total_size * 1.5 * self.config.quantized_threshold:
                return CheckpointStrategy.QUANTIZED_FP16
            elif cpu_available > total_size * 0.3 * self.config.incremental_threshold:
                return CheckpointStrategy.INCREMENTAL_DELTA
            else:
                return CheckpointStrategy.STREAMING_DISK

        except Exception:
            # No memory info (e.g. psutil not installed): keep exact checkpoints
            # rather than silently trading away precision.
            return CheckpointStrategy.FULL_CPU

    def _print_init_info(self):
        _log(f"AdaptiveCheckpointer initialized")
        _log(f"   Model size: {self.model_size_bytes / 1e9:.2f} GB")
        _log(f"   Optimizer size: {self.optimizer_size_bytes / 1e9:.2f} GB")
        _log(f"   Strategy: {self.current_strategy.name}")

    def save(self, step: Optional[int] = None) -> CheckpointMetadata:
        self.step = step or self.step + 1

        if self.current_strategy == CheckpointStrategy.FULL_CPU:
            return self._save_full_cpu()
        elif self.current_strategy == CheckpointStrategy.QUANTIZED_FP16:
            return self._save_quantized(dtype=torch.float16)
        elif self.current_strategy == CheckpointStrategy.QUANTIZED_INT8:
            return self._save_quantized(dtype=torch.int8)
        elif self.current_strategy == CheckpointStrategy.INCREMENTAL_DELTA:
            return self._save_incremental()
        elif self.current_strategy == CheckpointStrategy.STREAMING_DISK:
            return self._save_to_disk()
        else:
            return self._save_full_cpu()

    def _save_full_cpu(self) -> CheckpointMetadata:
        import time
        start_time = time.time()

        checkpoint = {
            'model': {k: v.cpu().clone() for k, v in self.model.state_dict().items()},
            'optimizer': {
                k: {
                    k2: v2.cpu().clone() if isinstance(v2, torch.Tensor) else v2
                    for k2, v2 in v.items()
                }
                for k, v in self.optimizer.state.items()
            },
            'step': self.step,
        }

        checkpoint['rng'] = {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }

        checkpoint_size = sum(
            v.numel() * v.element_size()
            for v in checkpoint['model'].values()
        )

        checkpoint['checkpoint_id'] = self._next_checkpoint_id
        self._last_full_checkpoint_id = self._next_checkpoint_id
        self._last_checkpoint_id = self._next_checkpoint_id
        self._next_checkpoint_id += 1

        self._apply_integrity_fields(checkpoint)

        self.checkpoints.append(checkpoint)
        self._last_full_state = checkpoint['model']

        metadata = CheckpointMetadata(
            step=self.step,
            strategy=CheckpointStrategy.FULL_CPU,
            model_size_bytes=self.model_size_bytes,
            checkpoint_size_bytes=checkpoint_size,
            compression_ratio=1.0,
            timestamp=time.time(),
        )
        self.metadata.append(metadata)

        if self.config.verbose:
            _log(f"   Saved full CPU checkpoint at step {self.step} ({checkpoint_size / 1e6:.1f} MB)")

        return metadata

    def _save_quantized(self, dtype: torch.dtype = torch.float16) -> CheckpointMetadata:
        import time

        scales: Dict[str, float] = {}

        def quantize(name: str, v: torch.Tensor) -> torch.Tensor:
            v = v.detach().cpu()
            if dtype == torch.int8:
                # Symmetric per-tensor int8: q = round(v / s), s = max|v| / 127.
                scale = float(v.abs().max()) / 127.0 if v.numel() else 0.0
                scale = scale or 1.0
                scales[name] = scale
                return torch.round(v.float() / scale).clamp_(-127, 127).to(torch.int8)
            return v.to(dtype)

        model_q = {
            k: (quantize(k, v) if isinstance(v, torch.Tensor) and v.is_floating_point()
                else v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v)
            for k, v in self.model.state_dict().items()
        }

        opt_dtype = torch.bfloat16 if self.config.quantize_optimizer else None
        optimizer_q = {
            k: {
                k2: (v2.detach().cpu().to(opt_dtype)
                     if opt_dtype is not None and isinstance(v2, torch.Tensor) and v2.is_floating_point()
                     and v2.dim() > 0
                     else v2.detach().cpu().clone() if isinstance(v2, torch.Tensor) else copy.deepcopy(v2))
                for k2, v2 in v.items()
            }
            for k, v in self.optimizer.state.items()
        }

        checkpoint = {
            'model': model_q,
            'optimizer': optimizer_q,
            'step': self.step,
            'dtype': str(dtype),
            'scales': scales,
        }

        checkpoint_size = sum(
            v.numel() * v.element_size()
            for v in checkpoint['model'].values()
        )

        self._apply_integrity_fields(checkpoint)

        self.checkpoints.append(checkpoint)

        compression = self.model_size_bytes / checkpoint_size if checkpoint_size > 0 else 1.0

        metadata = CheckpointMetadata(
            step=self.step,
            strategy=CheckpointStrategy.QUANTIZED_FP16 if dtype == torch.float16 else CheckpointStrategy.QUANTIZED_INT8,
            model_size_bytes=self.model_size_bytes,
            checkpoint_size_bytes=checkpoint_size,
            compression_ratio=compression,
            timestamp=time.time(),
        )
        self.metadata.append(metadata)

        if self.config.verbose:
            _log(f"   Saved quantized checkpoint ({compression:.1f}x compression)")

        return metadata

    def _save_incremental(self) -> CheckpointMetadata:
        import time

        current_state = self.model.state_dict()

        if self._last_full_state is None:
            return self._save_full_cpu()

        delta = {}
        for name, param in current_state.items():
            if name in self._last_full_state:
                old_param = self._last_full_state[name]
                if not torch.allclose(param.cpu(), old_param, atol=self.config.delta_threshold):
                    delta[name] = param.cpu().clone()
            else:
                delta[name] = param.cpu().clone()

        checkpoint = {
            'delta': delta,
            'base_id': self._last_checkpoint_id,
            'checkpoint_id': self._next_checkpoint_id,
            'step': self.step,
            'is_incremental': True,
        }
        self._last_checkpoint_id = self._next_checkpoint_id
        self._next_checkpoint_id += 1

        checkpoint_size = sum(
            v.numel() * v.element_size()
            for v in delta.values()
        )

        self._apply_integrity_fields(checkpoint)

        self.checkpoints.append(checkpoint)

        compression = self.model_size_bytes / checkpoint_size if checkpoint_size > 0 else float('inf')

        self._last_full_state = {k: v.cpu().clone() for k, v in current_state.items()}

        metadata = CheckpointMetadata(
            step=self.step,
            strategy=CheckpointStrategy.INCREMENTAL_DELTA,
            model_size_bytes=self.model_size_bytes,
            checkpoint_size_bytes=checkpoint_size,
            compression_ratio=compression,
            timestamp=time.time(),
            is_incremental=True,
            base_checkpoint_id=checkpoint['base_id'],
        )
        self.metadata.append(metadata)

        if self.config.verbose:
            _log(f"   Saved incremental checkpoint ({len(delta)}/{len(current_state)} params changed, {compression:.1f}x compression)")

        return metadata

    def _content_hash(self, checkpoint: dict) -> str:
        """sha256 over the checkpoint dict's actual content (everything
        except the integrity fields themselves), independent of pickle
        metadata ordering, so re-saving identical content reproduces the
        same hash. Works for any save strategy's checkpoint shape (full
        model/optimizer, quantized, or an incremental delta) since it just
        hashes whatever's actually in the dict."""
        to_hash = {k: v for k, v in checkpoint.items() if k not in ("integrity_hash", "integrity_signature")}
        buf = io.BytesIO()
        torch.save(to_hash, buf)
        return hashlib.sha256(buf.getvalue()).hexdigest()

    def _sign(self, content_hash: str) -> str:
        return hmac.new(
            self.config.signing_key.encode(), content_hash.encode(), hashlib.sha256
        ).hexdigest()

    def _apply_integrity_fields(self, checkpoint: dict) -> None:
        """Adds `integrity_hash`/`integrity_signature` to `checkpoint` in
        place when `config.sign_checkpoints` is set -- called from every
        save strategy, not just disk saves, so enabling signing actually
        protects whichever strategy got auto-selected."""
        if not self.config.sign_checkpoints:
            return
        content_hash = self._content_hash(checkpoint)
        checkpoint['integrity_hash'] = content_hash
        if self.config.signing_key:
            checkpoint['integrity_signature'] = self._sign(content_hash)

    def _save_to_disk(self) -> CheckpointMetadata:
        import time

        self._clean_disk_checkpoints()

        filename = os.path.join(
            self.config.disk_checkpoint_dir,
            f"checkpoint_{self.step}.pt"
        )

        # Name-keyed, not Parameter-object-keyed: this checkpoint round-trips
        # through torch.save/torch.load, and pickling a Parameter deserializes
        # it as a brand-new tensor object with no identity relationship to the
        # live self.optimizer.state's Parameter keys. Keying by the object
        # itself (as _save_full_cpu/_save_quantized do for their in-memory-only
        # checkpoints) would make every entry silently unmatchable on restore.
        # Parameter names survive serialization and can be re-resolved against
        # the live model in restore().
        param_names = {p: n for n, p in self.model.named_parameters()}
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': {
                param_names[k]: {
                    k2: v2.clone() if isinstance(v2, torch.Tensor) else v2
                    for k2, v2 in v.items()
                }
                for k, v in self.optimizer.state.items()
                if k in param_names
            },
            'optimizer_keyed_by_name': True,
            'step': self.step,
        }

        self._apply_integrity_fields(checkpoint)

        torch.save(checkpoint, filename)

        checkpoint_size = os.path.getsize(filename)

        self.checkpoints.append({'path': filename, 'step': self.step})

        metadata = CheckpointMetadata(
            step=self.step,
            strategy=CheckpointStrategy.STREAMING_DISK,
            model_size_bytes=self.model_size_bytes,
            checkpoint_size_bytes=checkpoint_size,
            compression_ratio=1.0,
            timestamp=time.time(),
        )
        self.metadata.append(metadata)

        if self.config.verbose:
            _log(f"   Saved to disk: {filename}")

        return metadata

    def _clean_disk_checkpoints(self):
        files = sorted([
            os.path.join(self.config.disk_checkpoint_dir, f)
            for f in os.listdir(self.config.disk_checkpoint_dir)
            if f.startswith('checkpoint_') and f.endswith('.pt')
        ], key=os.path.getmtime)

        # `self.checkpoints` (the in-memory deque) can hold up to
        # config.max_checkpoints entries pointing at disk files. If
        # max_disk_checkpoints < max_checkpoints, pruning to max_disk_checkpoints
        # alone would delete a file still referenced by a live deque entry,
        # so restore() would fail with a missing file. Always retain at least
        # as many files as the deque can reference.
        retain = max(self.config.max_disk_checkpoints, self.config.max_checkpoints)
        while len(files) >= retain:
            try:
                os.remove(files.pop(0))
            except Exception:
                pass

    def restore(self, checkpoint_idx: int = -1) -> int:
        if not self.checkpoints:
            raise ValueError("No checkpoints available")

        checkpoint = self.checkpoints[checkpoint_idx]

        if isinstance(checkpoint, dict) and 'path' in checkpoint:
            from arc._serialization import safe_torch_load
            checkpoint = safe_torch_load(
                checkpoint['path'], map_location=self.device,
                allow_unsafe=self.config.allow_unsafe_load,
            )

        if 'integrity_hash' in checkpoint:
            actual_hash = self._content_hash(checkpoint)
            if actual_hash != checkpoint['integrity_hash']:
                raise ValueError(
                    "Checkpoint integrity check failed: content hash mismatch, "
                    "file may be corrupted or tampered with."
                )
            if 'integrity_signature' in checkpoint:
                if not self.config.signing_key:
                    raise ValueError(
                        "Checkpoint is signed but no signing_key is configured to verify it."
                    )
                expected_sig = self._sign(actual_hash)
                if not hmac.compare_digest(expected_sig, checkpoint['integrity_signature']):
                    raise ValueError(
                        "Checkpoint signature verification failed: signature does not match "
                        "configured signing_key."
                    )

        if checkpoint.get('is_incremental', False):
            checkpoint = self._resolve_incremental(checkpoint)

        if 'model' in checkpoint:
            scales = checkpoint.get('scales', {})
            live = self.model.state_dict()
            restored_state = {}
            for k, v in checkpoint['model'].items():
                if isinstance(v, torch.Tensor):
                    if k in scales:
                        v = v.float() * scales[k]  # dequantize int8
                    target = live.get(k)
                    if target is not None:
                        v = v.to(device=target.device, dtype=target.dtype)
                restored_state[k] = v
            self.model.load_state_dict(restored_state)

        if 'optimizer' in checkpoint and checkpoint['optimizer']:
            if checkpoint.get('optimizer_keyed_by_name', False):
                # Disk-persisted checkpoint: keys are parameter names (see
                # _save_to_disk), not live Parameter objects. Resolve back to
                # the current model's Parameter objects by name.
                name_to_param = {n: p for n, p in self.model.named_parameters()}
                saved = {name_to_param[n]: v for n, v in checkpoint['optimizer'].items()
                         if n in name_to_param}
            else:
                saved = checkpoint['optimizer']
            for p, v in saved.items():
                if not isinstance(v, dict) or p not in self.optimizer.state:
                    continue
                live_state = self.optimizer.state[p]
                for k2, v2 in v.items():
                    if isinstance(v2, torch.Tensor):
                        # Back to the live tensor's device and dtype (undoes an
                        # opt-in bf16 quantization; never upcasts silently).
                        ref = live_state.get(k2)
                        if isinstance(ref, torch.Tensor):
                            live_state[k2] = v2.to(device=ref.device, dtype=ref.dtype).clone()
                        else:
                            live_state[k2] = v2.to(device=p.device).clone()
                    else:
                        live_state[k2] = copy.deepcopy(v2)

        if 'rng' in checkpoint:
            torch.set_rng_state(checkpoint['rng']['torch'])
            if checkpoint['rng']['cuda'] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(checkpoint['rng']['cuda'])

        restored_step = checkpoint.get('step', 0)

        if self.config.verbose:
            _log(f"   Restored checkpoint from step {restored_step}")

        return restored_step

    def _resolve_incremental(self, checkpoint: Dict) -> Dict:
        base_checkpoint = next(
            (c for c in self.checkpoints if c.get('checkpoint_id') == checkpoint.get('base_id')),
            None
        )

        if base_checkpoint is None:
            raise ValueError(
                f"Base checkpoint with id={checkpoint.get('base_id')} not found. "
                f"It may have been evicted from the deque."
            )

        # The base itself may be another incremental delta (a delta chained
        # off a delta) -- resolve it first so every intermediate change gets
        # applied, not just the latest delta's.
        if base_checkpoint.get('is_incremental', False):
            base_checkpoint = self._resolve_incremental(base_checkpoint)

        full_state = base_checkpoint['model'].copy()
        for k, v in checkpoint['delta'].items():
            full_state[k] = v

        return {
            'model': full_state,
            'optimizer': base_checkpoint.get('optimizer', {}),
            'step': checkpoint['step'],
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "strategy": self.current_strategy.name,
            "num_checkpoints": len(self.checkpoints),
            "model_size_gb": self.model_size_bytes / 1e9,
            "optimizer_size_gb": self.optimizer_size_bytes / 1e9,
            "total_saved_checkpoints": len(self.metadata),
            "avg_compression_ratio": sum(m.compression_ratio for m in self.metadata) / len(self.metadata) if self.metadata else 0,
        }

def create_adaptive_checkpointer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    **kwargs
) -> AdaptiveCheckpointer:
    config = AdaptiveCheckpointConfig(**kwargs)
    return AdaptiveCheckpointer(model, optimizer, config)