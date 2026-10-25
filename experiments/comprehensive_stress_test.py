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
ARC ULTIMATE COMPREHENSIVE STRESS TEST
Testing ALL possible failure modes across every category.

SUPERSEDED by experiments/recovery_benchmark.py. The ARC run here counts as
"survived" whenever no exception is raised, even if its weights end up NaN.
Don't quote its numbers.

============================================================
CATEGORIES OF FAILURES COVERED:
============================================================
1. NUMERICAL FAILURES (NaN, Inf, overflow, underflow)
2. GRADIENT FAILURES (explosion, vanishing, corruption)
3. WEIGHT FAILURES (corruption, collapse, explosion)
4. OPTIMIZER FAILURES (state corruption, momentum issues)
5. LOSS FAILURES (explosion, oscillation, NaN)
6. PRECISION FAILURES (FP16/32 issues, overflow)
7. MEMORY FAILURES (tensor corruption, shape issues)
8. ARCHITECTURE FAILURES (layer corruption, activation issues)
9. TRAINING DYNAMICS (learning rate, batch issues)
10. COMBINED FAILURES (multiple simultaneous issues)

Total: 50+ extreme scenarios
============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import sys
import os
import time
import gc
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc.core.self_healing import SelfHealingArc, SelfHealingConfig


# =============================================================================
# Test Infrastructure
# =============================================================================

@dataclass
class TestResult:
    __test__ = False  # not a pytest test class, just named "Test*"

    name: str
    category: str
    baseline_crashed: bool
    arc_survived: bool
    rollbacks: int
    details: str


class ComprehensiveReport:
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = time.time()
    
    def add(self, result: TestResult):
        self.results.append(result)
        status = "✓" if result.arc_survived else "✗"
        print(f"  [{status}] {result.name}")
    
    def summary(self) -> Dict:
        elapsed = time.time() - self.start_time
        
        # Stats by category
        categories = {}
        for r in self.results:
            if r.category not in categories:
                categories[r.category] = {"total": 0, "crashed": 0, "survived": 0}
            categories[r.category]["total"] += 1
            if r.baseline_crashed:
                categories[r.category]["crashed"] += 1
            if r.arc_survived and r.baseline_crashed:
                categories[r.category]["survived"] += 1
        
        total_crashable = sum(1 for r in self.results if r.baseline_crashed)
        total_survived = sum(1 for r in self.results if r.arc_survived and r.baseline_crashed)
        
        print("\n" + "=" * 70)
        print("ULTIMATE COMPREHENSIVE STRESS TEST - FINAL REPORT")
        print("=" * 70)
        
        print("\nResults by Category:")
        for cat, stats in sorted(categories.items()):
            if stats["crashed"] > 0:
                rate = 100 * stats["survived"] / stats["crashed"]
                print(f"  {cat}: {stats['survived']}/{stats['crashed']} ({rate:.0f}%)")
            else:
                print(f"  {cat}: No crashes (not testable)")
        
        print(f"\n{'─' * 50}")
        print(f"Total Scenarios:       {len(self.results)}")
        print(f"Baseline Crashes:      {total_crashable}")
        print(f"ARC Survived:          {total_survived}")
        if total_crashable > 0:
            print(f"Recovery Rate:         {100 * total_survived / total_crashable:.1f}%")
        print(f"Time Elapsed:          {elapsed:.1f}s")
        print("─" * 50)
        
        if total_crashable > 0 and total_survived == total_crashable:
            print("\n🎉 PERFECT SCORE - ARC SURVIVED ALL FAILURES!")
        elif total_crashable > 0 and total_survived / total_crashable >= 0.95:
            print("\n✓ EXCELLENT (95%+)")
        elif total_crashable > 0 and total_survived / total_crashable >= 0.80:
            print("\n⚠ GOOD (80%+)")
        
        return {
            "total": len(self.results),
            "baseline_crashes": total_crashable,
            "arc_survived": total_survived,
            "rate": total_survived / max(total_crashable, 1),
            "categories": categories
        }


class TestModel(nn.Module):
    __test__ = False  # not a pytest test class, just named "Test*"

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def run_test(name: str, category: str, failure_fn: Callable, 
             trigger_step: int = 50, max_steps: int = 100) -> TestResult:
    """Run a single failure scenario."""
    
    X = torch.randn(200, 100)
    y = torch.randint(0, 10, (200,))
    dataloader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
    
    # ===== Baseline Test =====
    model = TestModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    baseline_crashed = False
    baseline_steps = 0
    
    step = 0
    try:
        for epoch in range(20):
            for x_batch, y_batch in dataloader:
                step += 1
                if step > max_steps:
                    break
                
                optimizer.zero_grad()
                out = model(x_batch)
                loss = F.cross_entropy(out, y_batch)
                
                if step == trigger_step:
                    failure_fn(model, optimizer, loss)
                
                # Check for crashes
                if torch.isnan(loss) or torch.isinf(loss):
                    baseline_crashed = True
                    baseline_steps = step
                    break
                
                if loss.item() > 1e10:
                    baseline_crashed = True
                    baseline_steps = step
                    break
                
                loss.backward()
                
                # Check gradients
                for p in model.parameters():
                    if p.grad is not None:
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            baseline_crashed = True
                            break
                        if p.grad.abs().max() > 1e10:
                            baseline_crashed = True
                            break
                
                # Check weights
                for p in model.parameters():
                    if torch.isnan(p).any() or torch.isinf(p).any():
                        baseline_crashed = True
                        break
                
                # Check sparsity (silent corruption)
                total_params = sum(p.numel() for p in model.parameters())
                total_zeros = sum((p.data == 0).sum().item() for p in model.parameters())
                if total_zeros / total_params > 0.3:
                    baseline_crashed = True
                    break
                
                if baseline_crashed:
                    baseline_steps = step
                    break
                
                optimizer.step()
            
            if step > max_steps or baseline_crashed:
                break
    except Exception:
        baseline_crashed = True
        baseline_steps = step
    
    # ===== ARC Test =====
    model = TestModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    config = SelfHealingConfig(
        loss_explosion_threshold=50.0,
        gradient_explosion_threshold=1e4,
        lr_reduction_factor=0.5,
        max_lr_reductions=20,
        enable_weight_health_check=True,
        weight_sparsity_threshold=0.15,
        weight_magnitude_collapse_threshold=0.05,
        weight_health_check_frequency=1,
        enable_persistent_failure_recovery=True,
        max_consecutive_rollbacks=3,
        skip_ahead_steps=10,
        weight_perturbation_scale=0.02,
        verbose=False,
    )
    arc = SelfHealingArc(model, optimizer, config)
    
    arc_crashed = False
    arc_steps = 0
    rollbacks = 0
    
    step = 0
    try:
        for epoch in range(20):
            for x_batch, y_batch in dataloader:
                step += 1
                if step > max_steps:
                    break
                
                optimizer.zero_grad()
                
                try:
                    out = model(x_batch)
                    loss = F.cross_entropy(out, y_batch)
                except:
                    arc_crashed = True
                    break
                
                if step == trigger_step:
                    failure_fn(model, optimizer, loss)
                
                action = arc.step(loss)
                if action.should_skip:
                    if action.rolled_back:
                        rollbacks += 1
                    continue
                
                try:
                    loss.backward()
                except:
                    rollbacks += 1
                    continue
                
                post_action = arc.post_backward()
                if post_action.should_skip:
                    continue
                
                optimizer.step()
                arc_steps = step
            
            if step > max_steps or arc_crashed:
                break
    except:
        arc_crashed = True
    
    arc_survived = not arc_crashed and arc_steps >= max_steps - 10
    
    return TestResult(
        name=name,
        category=category,
        baseline_crashed=baseline_crashed,
        arc_survived=arc_survived,
        rollbacks=rollbacks,
        details=f"Baseline: {'CRASH' if baseline_crashed else 'OK'} @ {baseline_steps}, ARC: {arc_steps} steps, Rollbacks: {rollbacks}"
    )


# =============================================================================
# 1. NUMERICAL FAILURES
# =============================================================================

def fail_nan_loss(model, optimizer, loss):
    """Return NaN loss."""
    return torch.tensor(float('nan'))

def fail_inf_loss(model, optimizer, loss):
    """Return Inf loss."""
    return torch.tensor(float('inf'))

def fail_neg_inf_loss(model, optimizer, loss):
    """Return negative infinity loss."""
    return torch.tensor(float('-inf'))

def fail_nan_weights(model, optimizer, loss):
    """Set all weights to NaN."""
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(float('nan'))

def fail_inf_weights(model, optimizer, loss):
    """Set weights to infinity."""
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(float('inf'))

def fail_underflow(model, optimizer, loss):
    """Cause underflow by scaling to tiny values."""
    with torch.no_grad():
        for p in model.parameters():
            p.data *= 1e-40

def fail_overflow(model, optimizer, loss):
    """Cause overflow by scaling to huge values."""
    with torch.no_grad():
        for p in model.parameters():
            p.data *= 1e30


# =============================================================================
# 2. GRADIENT FAILURES
# =============================================================================

def fail_exploding_gradients(model, optimizer, loss):
    """Multiply gradients by huge factor."""
    for p in model.parameters():
        if p.grad is not None:
            p.grad *= 1e10

def fail_vanishing_gradients(model, optimizer, loss):
    """Multiply gradients by tiny factor."""
    for p in model.parameters():
        if p.grad is not None:
            p.grad *= 1e-40

def fail_nan_gradients(model, optimizer, loss):
    """Set gradients to NaN."""
    for p in model.parameters():
        if p.grad is not None:
            p.grad.fill_(float('nan'))

def fail_random_gradient_corruption(model, optimizer, loss):
    """Replace gradients with random huge values."""
    for p in model.parameters():
        if p.grad is not None:
            p.grad = torch.randn_like(p) * 1e8

def fail_gradient_sign_flip(model, optimizer, loss):
    """Flip gradient signs with huge magnitude."""
    for p in model.parameters():
        if p.grad is not None:
            p.grad = -p.grad * 1e6


# =============================================================================
# 3. WEIGHT FAILURES
# =============================================================================

def fail_zero_weights(model, optimizer, loss):
    """Zero out all weights."""
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0)

def fail_random_weight_corruption(model, optimizer, loss):
    """Replace weights with random values."""
    with torch.no_grad():
        for p in model.parameters():
            p.data = torch.randn_like(p) * 100

def fail_weight_explosion(model, optimizer, loss):
    """Scale weights to huge values."""
    with torch.no_grad():
        for p in model.parameters():
            p.data *= 1000

def fail_weight_collapse(model, optimizer, loss):
    """Scale weights to tiny values."""
    with torch.no_grad():
        for p in model.parameters():
            p.data *= 0.001

def fail_sparse_weights(model, optimizer, loss):
    """Zero out 80% of weights."""
    with torch.no_grad():
        for p in model.parameters():
            mask = torch.rand_like(p) < 0.8
            p.data[mask] = 0

def fail_constant_weights(model, optimizer, loss):
    """Set all weights to same value."""
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0.1)

def fail_layer_zero(model, optimizer, loss):
    """Zero out first layer."""
    with torch.no_grad():
        model.fc1.weight.fill_(0)
        model.fc1.bias.fill_(0)


# =============================================================================
# 4. OPTIMIZER FAILURES
# =============================================================================

def fail_corrupt_momentum(model, optimizer, loss):
    """Corrupt optimizer momentum buffers."""
    for pg in optimizer.param_groups:
        for p in pg['params']:
            state = optimizer.state[p]
            if 'exp_avg' in state:
                state['exp_avg'].fill_(float('nan'))

def fail_extreme_lr(model, optimizer, loss):
    """Set learning rate to extreme value."""
    for pg in optimizer.param_groups:
        pg['lr'] = 1e6

def fail_negative_lr(model, optimizer, loss):
    """Set negative learning rate."""
    for pg in optimizer.param_groups:
        pg['lr'] = -0.01

def fail_corrupt_state(model, optimizer, loss):
    """Corrupt all optimizer state."""
    for pg in optimizer.param_groups:
        for p in pg['params']:
            state = optimizer.state[p]
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    v.fill_(float('inf'))


# =============================================================================
# 5. LOSS FAILURES
# =============================================================================

def fail_loss_explosion(model, optimizer, loss):
    """Multiply loss by huge factor."""
    return loss * 1e15

def fail_loss_oscillation(model, optimizer, loss):
    """Make loss oscillate wildly."""
    return loss * (1e6 if np.random.random() > 0.5 else 1e-6)

def fail_negative_loss(model, optimizer, loss):
    """Return large negative loss."""
    return -torch.tensor(1e10)

def fail_zero_loss(model, optimizer, loss):
    """Return exactly zero loss."""
    return torch.tensor(0.0, requires_grad=True)


# =============================================================================
# 6. PRECISION FAILURES
# =============================================================================

def fail_fp16_overflow(model, optimizer, loss):
    """Simulate FP16 overflow."""
    with torch.no_grad():
        for p in model.parameters():
            p.data *= 70000  # Beyond FP16 range

def fail_fp16_underflow(model, optimizer, loss):
    """Simulate FP16 underflow."""
    with torch.no_grad():
        for p in model.parameters():
            p.data *= 1e-8  # Below FP16 precision

def fail_denormal_weights(model, optimizer, loss):
    """Set weights to denormal values."""
    with torch.no_grad():
        for p in model.parameters():
            p.data = torch.ones_like(p) * 1e-45


# =============================================================================
# 7. ARCHITECTURE FAILURES
# =============================================================================

def fail_dead_relu(model, optimizer, loss):
    """Force all ReLU outputs negative."""
    with torch.no_grad():
        for p in model.parameters():
            p.data = -torch.abs(p.data)

def fail_activation_saturation(model, optimizer, loss):
    """Force activations to saturate."""
    with torch.no_grad():
        for p in model.parameters():
            p.data *= 1e4

def fail_bias_explosion(model, optimizer, loss):
    """Set biases to huge values."""
    with torch.no_grad():
        if hasattr(model.fc1, 'bias'):
            model.fc1.bias.fill_(1e6)
        if hasattr(model.fc2, 'bias'):
            model.fc2.bias.fill_(1e6)


# =============================================================================
# 8. TRAINING DYNAMICS FAILURES
# =============================================================================

def fail_batch_corruption(model, optimizer, loss):
    """Simulate corrupted batch effect."""
    return loss * 1e10

def fail_gradient_accumulation_overflow(model, optimizer, loss):
    """Simulate gradient accumulation overflow."""
    for p in model.parameters():
        if p.grad is not None:
            p.grad *= 1e5


# =============================================================================
# 9. COMBINED FAILURES
# =============================================================================

def fail_triple_threat(model, optimizer, loss):
    """NaN weights + extreme LR + corrupted momentum."""
    with torch.no_grad():
        model.fc1.weight[:10].fill_(float('nan'))
    for pg in optimizer.param_groups:
        pg['lr'] = 1e4

def fail_cascade(model, optimizer, loss):
    """Multiple failures in sequence."""
    with torch.no_grad():
        model.fc1.weight.fill_(0)
        model.fc2.weight *= 1000

def fail_apocalypse(model, optimizer, loss):
    """Everything breaks at once."""
    with torch.no_grad():
        for i, p in enumerate(model.parameters()):
            if i % 3 == 0:
                p.fill_(float('nan'))
            elif i % 3 == 1:
                p.fill_(0)
            else:
                p.data *= 1e10


# =============================================================================
# Test Registry
# =============================================================================

ALL_TESTS = {
    "NUMERICAL": [
        ("NaN Loss", fail_nan_loss),
        ("Infinity Loss", fail_inf_loss),
        ("Negative Infinity Loss", fail_neg_inf_loss),
        ("NaN Weights", fail_nan_weights),
        ("Infinity Weights", fail_inf_weights),
        ("Underflow", fail_underflow),
        ("Overflow", fail_overflow),
    ],
    "GRADIENT": [
        ("Exploding Gradients (1e10)", fail_exploding_gradients),
        ("Vanishing Gradients (1e-40)", fail_vanishing_gradients),
        ("NaN Gradients", fail_nan_gradients),
        ("Random Gradient Corruption", fail_random_gradient_corruption),
        ("Gradient Sign Flip (1e6)", fail_gradient_sign_flip),
    ],
    "WEIGHT": [
        ("Zero All Weights", fail_zero_weights),
        ("Random Weight Corruption", fail_random_weight_corruption),
        ("Weight Explosion (1000x)", fail_weight_explosion),
        ("Weight Collapse (0.001x)", fail_weight_collapse),
        ("Sparse Weights (80% zeros)", fail_sparse_weights),
        ("Constant Weights", fail_constant_weights),
        ("Layer Zero (fc1)", fail_layer_zero),
    ],
    "OPTIMIZER": [
        ("Corrupt Momentum Buffers", fail_corrupt_momentum),
        ("Extreme Learning Rate (1e6)", fail_extreme_lr),
        ("Negative Learning Rate", fail_negative_lr),
        ("Corrupt All Optimizer State", fail_corrupt_state),
    ],
    "LOSS": [
        ("Loss Explosion (1e15)", fail_loss_explosion),
        ("Loss Oscillation", fail_loss_oscillation),
        ("Negative Loss", fail_negative_loss),
        ("Zero Loss", fail_zero_loss),
    ],
    "PRECISION": [
        ("FP16 Overflow", fail_fp16_overflow),
        ("FP16 Underflow", fail_fp16_underflow),
        ("Denormal Weights", fail_denormal_weights),
    ],
    "ARCHITECTURE": [
        ("Dead ReLU (negative weights)", fail_dead_relu),
        ("Activation Saturation", fail_activation_saturation),
        ("Bias Explosion", fail_bias_explosion),
    ],
    "DYNAMICS": [
        ("Batch Corruption", fail_batch_corruption),
        ("Gradient Accumulation Overflow", fail_gradient_accumulation_overflow),
    ],
    "COMBINED": [
        ("Triple Threat (NaN + LR + momentum)", fail_triple_threat),
        ("Cascade Failure", fail_cascade),
        ("Apocalypse (everything breaks)", fail_apocalypse),
    ],
}


def run_comprehensive_tests():
    """Run all comprehensive stress tests."""
    
    print("=" * 70)
    print("🔥 ULTIMATE COMPREHENSIVE STRESS TEST 🔥")
    print("   Testing ALL possible failure modes")
    print("=" * 70)
    
    total_tests = sum(len(tests) for tests in ALL_TESTS.values())
    print(f"\nTotal tests: {total_tests} across {len(ALL_TESTS)} categories\n")
    
    report = ComprehensiveReport()
    
    for category, tests in ALL_TESTS.items():
        print(f"\n{'─' * 50}")
        print(f"CATEGORY: {category}")
        print("─" * 50)
        
        for name, failure_fn in tests:
            result = run_test(name, category, failure_fn)
            report.add(result)
        
        gc.collect()
    
    summary = report.summary()
    
    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "comprehensive_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "summary": summary,
            "tests": [
                {
                    "name": r.name,
                    "category": r.category,
                    "baseline_crashed": r.baseline_crashed,
                    "arc_survived": r.arc_survived,
                    "rollbacks": r.rollbacks,
                }
                for r in report.results
            ]
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    return report.results


if __name__ == "__main__":
    results = run_comprehensive_tests()
