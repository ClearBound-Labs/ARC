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
Overhead Ablation Study - Measure Impact of Each Optimization

Required by Reviewers:
- "Show actual overhead reduction per optimization"
- "Ablation study needed to validate each optimization's impact"

Tests each optimization independently to show its contribution.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import time
import json
from typing import Dict, Any, List
from dataclasses import dataclass, asdict


@dataclass
class AblationResult:
    """Result of overhead measurement."""
    config_name: str
    avg_step_time_ms: float
    std_step_time_ms: float
    overhead_vs_baseline_pct: float
    memory_peak_mb: float
    features_enabled: Dict[str, bool]


class TestModel(nn.Module):
    """Medium-sized model for overhead testing."""
    __test__ = False  # not a pytest test class, just named "Test*"

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
        )
    
    def forward(self, x):
        return self.net(x)


def measure_baseline(n_steps: int = 100) -> float:
    """Measure baseline training time (no ARC)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = TestModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Warmup
    for _ in range(10):
        x = torch.randn(64, 256, device=device)
        loss = model(x).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    times = []
    for _ in range(n_steps):
        start = time.perf_counter()
        
        x = torch.randn(64, 256, device=device)
        optimizer.zero_grad()
        loss = model(x).mean()
        loss.backward()
        optimizer.step()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        times.append((time.perf_counter() - start) * 1000)
    
    peak_mem = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    
    return {
        'avg_time_ms': sum(times) / len(times),
        'std_time_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
        'peak_mem_mb': peak_mem,
    }


def measure_with_arc(
    n_steps: int = 100,
    enable_checkpointing: bool = True,
    checkpoint_frequency: int = 10,
    use_quantized: bool = False,
    use_incremental: bool = False,
    use_fast_grad_norm: bool = False,
    layer_sample_ratio: float = 1.0,
) -> Dict[str, float]:
    """Measure training time with ARC and specific features."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = TestModel().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    
    from arc.intervention.rollback import WeightRollback, RollbackConfig
    
    config = RollbackConfig(
        checkpoint_frequency=checkpoint_frequency if enable_checkpointing else 1000000,
        fast_grad_norm=use_fast_grad_norm,
        layer_sample_ratio=layer_sample_ratio,
    )
    arc = WeightRollback(model, optimizer, config, verbose=False)
    
    # Warmup
    for _ in range(10):
        x = torch.randn(64, 256, device=device)
        loss = model(x).mean()
        arc.step(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    times = []
    for _ in range(n_steps):
        start = time.perf_counter()
        
        x = torch.randn(64, 256, device=device)
        optimizer.zero_grad()
        loss = model(x).mean()
        
        action = arc.step(loss)
        if action.rolled_back:
            continue
        
        loss.backward()
        optimizer.step()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        times.append((time.perf_counter() - start) * 1000)
    
    peak_mem = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
    
    return {
        'avg_time_ms': sum(times) / len(times),
        'std_time_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
        'peak_mem_mb': peak_mem,
    }


def run_ablation_study() -> List[AblationResult]:
    """Run complete ablation study."""
    print("=" * 60)
    print("OVERHEAD ABLATION STUDY")
    print("=" * 60)
    
    results = []
    
    # Measure baseline
    print("\n1. Measuring baseline (no ARC)...")
    baseline = measure_baseline()
    baseline_time = baseline['avg_time_ms']
    print(f"   Baseline: {baseline_time:.2f} ms/step")
    
    results.append(AblationResult(
        config_name="baseline",
        avg_step_time_ms=baseline['avg_time_ms'],
        std_step_time_ms=baseline['std_time_ms'],
        overhead_vs_baseline_pct=0.0,
        memory_peak_mb=baseline['peak_mem_mb'],
        features_enabled={},
    ))
    
    # Configuration variants
    configs = [
        {
            'name': 'arc_full',
            'desc': 'ARC Full (all features)',
            'params': {
                'enable_checkpointing': True,
                'checkpoint_frequency': 10,
                'use_fast_grad_norm': False,
                'layer_sample_ratio': 1.0,
            }
        },
        {
            'name': 'arc_no_checkpoint',
            'desc': 'ARC without checkpointing',
            'params': {
                'enable_checkpointing': False,
                'checkpoint_frequency': 1000000,
                'use_fast_grad_norm': False,
                'layer_sample_ratio': 1.0,
            }
        },
        {
            'name': 'arc_less_checkpoints',
            'desc': 'ARC with less frequent checkpoints (every 50)',
            'params': {
                'enable_checkpointing': True,
                'checkpoint_frequency': 50,
                'use_fast_grad_norm': False,
                'layer_sample_ratio': 1.0,
            }
        },
        {
            'name': 'arc_fast_grad',
            'desc': 'ARC with fast gradient norm',
            'params': {
                'enable_checkpointing': True,
                'checkpoint_frequency': 10,
                'use_fast_grad_norm': True,
                'layer_sample_ratio': 1.0,
            }
        },
        {
            'name': 'arc_sampled_layers',
            'desc': 'ARC with sampled layers (10%)',
            'params': {
                'enable_checkpointing': True,
                'checkpoint_frequency': 10,
                'use_fast_grad_norm': True,
                'layer_sample_ratio': 0.1,
            }
        },
        {
            'name': 'arc_lite',
            'desc': 'ARC Lite (infrequent ckpt + fast grad + sampling)',
            'params': {
                'enable_checkpointing': True,
                'checkpoint_frequency': 50,
                'use_fast_grad_norm': True,
                'layer_sample_ratio': 0.1,
            }
        },
    ]
    
    for i, cfg in enumerate(configs):
        print(f"\n{i+2}. Measuring {cfg['desc']}...")
        metrics = measure_with_arc(**cfg['params'])
        overhead = ((metrics['avg_time_ms'] - baseline_time) / baseline_time) * 100
        
        print(f"   Time: {metrics['avg_time_ms']:.2f} ms/step")
        print(f"   Overhead: {overhead:.1f}%")
        
        results.append(AblationResult(
            config_name=cfg['name'],
            avg_step_time_ms=metrics['avg_time_ms'],
            std_step_time_ms=metrics['std_time_ms'],
            overhead_vs_baseline_pct=overhead,
            memory_peak_mb=metrics['peak_mem_mb'],
            features_enabled=cfg['params'],
        ))
    
    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Config':<30} {'Time (ms)':<12} {'Overhead':<12}")
    print("-" * 60)
    for r in results:
        overhead_str = f"{r.overhead_vs_baseline_pct:.1f}%" if r.config_name != 'baseline' else '-'
        print(f"{r.config_name:<30} {r.avg_step_time_ms:<12.2f} {overhead_str:<12}")
    
    # Save results
    with open('overhead_ablation_results.json', 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to: overhead_ablation_results.json")
    
    return results


if __name__ == '__main__':
    results = run_ablation_study()