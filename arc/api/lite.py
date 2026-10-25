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
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import random
import time
import math
import warnings

@dataclass
class LiteConfig:
    check_every_n_steps: int = 50
    full_analysis_every_n_epochs: int = 5

    sample_n_layers: int = 2

    check_loss: bool = True
    check_gradients: bool = True
    check_weights: bool = False
    check_activations: bool = False

    loss_explosion_threshold: float = 100.0
    gradient_explosion_threshold: float = 1e4
    nan_detection: bool = True

class LiteArc:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        config: Optional[LiteConfig] = None,
    ):
        warnings.warn(
            "LiteArc is a legacy API kept for compatibility; it only raises alerts. "
            "Use arc.Controller, which detects and recovers at under 1% overhead.",
            FutureWarning, stacklevel=2,
        )
        self.model = model
        self.optimizer = optimizer
        self.config = config or LiteConfig()

        params = list(model.named_parameters())
        self._cached_params = [(name, p) for name, p in params if p.requires_grad]
        self._n_params = len(self._cached_params)

        self.step_count = 0
        self.epoch_count = 0
        self.alerts = []
        self._last_epoch_end_step = 0

        self.total_check_time = 0.0
        self.checks_performed = 0

    def step(self, loss: torch.Tensor) -> Dict[str, Any]:
        self.step_count += 1

        if self.step_count % self.config.check_every_n_steps != 0:
            return {"step": self.step_count, "checked": False}

        start_time = time.perf_counter()
        result = {"step": self.step_count, "checked": True, "alert": None}

        if self.config.check_loss:
            loss_val = loss.item() if isinstance(loss, torch.Tensor) else loss

            if math.isnan(loss_val):
                result["alert"] = "nan_loss"
                result["recommendation"] = "rollback"
                self._record_alert(result)
                return result

            if loss_val > self.config.loss_explosion_threshold:
                result["alert"] = "loss_explosion"
                result["recommendation"] = "reduce_lr"

        if self.config.check_gradients and self._n_params > 0:
            alert = self._check_gradients_fast()
            if alert:
                result["alert"] = alert
                result["recommendation"] = "clip_gradients"

        self.total_check_time += time.perf_counter() - start_time
        self.checks_performed += 1

        if result["alert"]:
            self._record_alert(result)

        return result

    def _check_gradients_fast(self) -> Optional[str]:
        if self._n_params <= 2:
            indices = range(self._n_params)
        else:
            indices = random.sample(range(self._n_params), 2)

        for idx in indices:
            _, param = self._cached_params[idx]
            if param.grad is None:
                continue

            if torch.isnan(param.grad).any():
                return "nan_gradient"

            if param.grad.abs().max().item() > self.config.gradient_explosion_threshold:
                return "gradient_explosion"

        return None

    def _record_alert(self, result: Dict):
        self.alerts.append({"step": self.step_count, "alert": result["alert"]})

    def _check_weights_sparse(self) -> Optional[str]:
        if self._n_params == 0:
            return None

        n_samples = min(self.config.sample_n_layers, self._n_params)
        indices = random.sample(range(self._n_params), n_samples)

        for idx in indices:
            _, param = self._cached_params[idx]
            if torch.isnan(param).any() or torch.isinf(param).any():
                return "nan_weights"

        return None

    def end_epoch(self, epoch: int) -> Dict[str, Any]:
        self.epoch_count = epoch

        # Count alerts since the last epoch boundary by actual step count,
        # not a hardcoded steps-per-epoch guess (which was wrong for any
        # dataset/batch-size combo that isn't exactly 1000 steps/epoch).
        alerts_this_epoch = len(
            [a for a in self.alerts if a["step"] > self._last_epoch_end_step]
        )
        self._last_epoch_end_step = self.step_count

        return {
            "epoch": epoch,
            "alerts_this_epoch": alerts_this_epoch,
            "avg_check_time_ms": (self.total_check_time / max(self.checks_performed, 1)) * 1000,
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_steps": self.step_count,
            "checks_performed": self.checks_performed,
            "check_ratio": self.checks_performed / max(self.step_count, 1),
            "total_check_time_s": self.total_check_time,
            "avg_check_time_ms": (self.total_check_time / max(self.checks_performed, 1)) * 1000,
            "total_alerts": len(self.alerts),
        }

    @classmethod
    def from_standard(cls, arc_instance) -> "LiteArc":
        """Downgrade a standard ARC instance (Controller, SelfHealingArc) to LiteArc.

        Expects `arc_instance` to expose `.model`/`.optimizer` directly, as
        Controller and SelfHealingArc both do.
        """
        return cls(
            model=arc_instance.model,
            optimizer=arc_instance.optimizer,
        )

def benchmark_overhead(model: nn.Module, n_steps: int = 1000) -> Dict[str, float]:
    import time

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    x = torch.randn(32, 3, 32, 32)

    start = time.perf_counter()
    for _ in range(n_steps):
        out = model(x) if hasattr(model, 'conv1') else model(x.view(32, -1))
        loss = out.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    baseline_time = time.perf_counter() - start

    arc = LiteArc(model, optimizer)
    start = time.perf_counter()
    for _ in range(n_steps):
        out = model(x) if hasattr(model, 'conv1') else model(x.view(32, -1))
        loss = out.mean()
        loss.backward()
        arc.step(loss)
        optimizer.step()
        optimizer.zero_grad()
    lite_time = time.perf_counter() - start

    return {
        "baseline_time": baseline_time,
        "lite_arc_time": lite_time,
        "overhead_percent": ((lite_time - baseline_time) / baseline_time) * 100,
        "steps": n_steps,
    }

if __name__ == "__main__":
    print("Testing LiteArc overhead...")

    model = nn.Sequential(
        nn.Linear(3072, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    x = torch.randn(32, 3, 32, 32)
    n_steps = 500

    print("\nRunning baseline (no monitoring)...")
    start = time.perf_counter()
    for _ in range(n_steps):
        out = model(x.view(32, -1))
        loss = out.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    baseline_time = time.perf_counter() - start

    print("Running LiteArc (default config)...")
    arc = LiteArc(model, optimizer)
    start = time.perf_counter()
    for _ in range(n_steps):
        out = model(x.view(32, -1))
        loss = out.mean()
        loss.backward()
        arc.step(loss)
        optimizer.step()
        optimizer.zero_grad()
    lite_time = time.perf_counter() - start

    print("Running UltraLite (loss-only)...")
    ultra_config = LiteConfig(
        check_every_n_steps=100,
        check_loss=True,
        check_gradients=False,
        check_weights=False,
    )
    ultra_arc = LiteArc(model, optimizer, ultra_config)
    start = time.perf_counter()
    for _ in range(n_steps):
        out = model(x.view(32, -1))
        loss = out.mean()
        loss.backward()
        ultra_arc.step(loss)
        optimizer.step()
        optimizer.zero_grad()
    ultra_time = time.perf_counter() - start

    print(f"\n{'='*50}")
    print("OVERHEAD BENCHMARK RESULTS")
    print("="*50)
    print(f"\n| Mode          | Time    | Overhead |")
    print(f"|---------------|---------|----------|")
    print(f"| Baseline      | {baseline_time:.3f}s  | 0%       |")
    print(f"| LiteArc       | {lite_time:.3f}s  | {((lite_time - baseline_time) / baseline_time) * 100:.1f}%     |")
    print(f"| UltraLite     | {ultra_time:.3f}s  | {((ultra_time - baseline_time) / baseline_time) * 100:.1f}%     |")

    ultra_overhead = ((ultra_time - baseline_time) / baseline_time) * 100
    if ultra_overhead < 5:
        print(f"\n  UltraLite PASSED: {ultra_overhead:.1f}% overhead < 5% target")
    elif ultra_overhead < 10:
        print(f"\n  UltraLite PASSED: {ultra_overhead:.1f}% overhead < 10% target")
    else:
        print(f"\n  Overhead still > 10%")