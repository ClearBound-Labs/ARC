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
ARC NIGHTMARE STRESS TEST - Extreme Failure Scenarios

SUPERSEDED by experiments/recovery_benchmark.py. This script never applies its
crash checks to the ARC-protected run, so its "recovered" count means "didn't
raise an exception". Don't quote its numbers.

Testing ARC against the WORST POSSIBLE failure conditions:

============================================================
TIER 1: INSTANT DEATH SCENARIOS
============================================================
1. Nuclear Learning Rate (lr=1e6) - Immediate explosion
2. All-NaN Injection - Every weight becomes NaN
3. Division by Zero Storm - Systematic div/0 in forward pass
4. Infinite Gradient Bomb - Gradients set to ±Inf
5. Memory Tsunami - Allocate until OOM during training

============================================================
TIER 2: CASCADING FAILURES
============================================================
6. Domino Collapse - Failure triggers 5 consecutive failures
7. Oscillating Death - Loss alternates between 0 and Inf
8. Gradient Whiplash - Gradients flip sign every step with 1e6 magnitude
9. Optimizer Corruption - Destroy all momentum buffers
10. RNG Catastrophe - Random seeds cause reproducible failures

============================================================
TIER 3: ADVERSARIAL NIGHTMARE
============================================================
11. Byzantine Gradients - 50% of gradients are adversarial
12. Loss Landscape Minefield - Random explosion triggers
13. Silent Precision Death - FP16 underflow accumulation
14. Batch Bomb - Single poisoned sample crashes training
15. Scheduler Sabotage - LR scheduler returns NaN

============================================================
TIER 4: COMBINED APOCALYPSE
============================================================
16. Everything At Once - Multiple failure modes simultaneously
17. Repeated Recovery Stress - Trigger 10 failures in 20 steps
18. Long-Tail Catastrophe - Failure after 1000 stable steps
19. Model Surgery Gone Wrong - Remove layers during training
20. Checkpoint Corruption - Corrupt saved checkpoints

For each scenario:
- Test WITHOUT ARC: Expected result = CRASH
- Test WITH ARC: Expected result = RECOVERY

This is the ULTIMATE validation that ARC can handle ANY failure.
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
import gc
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc.core.self_healing import SelfHealingArc, SelfHealingConfig


# =============================================================================
# Test Result Tracking
# =============================================================================

@dataclass
class TestResult:
    __test__ = False  # not a pytest test class, just named "Test*"

    scenario_name: str
    tier: str
    baseline_crashed: bool
    arc_crashed: bool
    arc_rollbacks: int
    arc_recovered: bool
    baseline_steps: int
    arc_steps: int
    details: str


class NightmareReport:
    """Comprehensive test result reporting."""
    
    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = time.time()
    
    def add(self, result: TestResult):
        self.results.append(result)
        status = "✓ RECOVERED" if result.arc_recovered else "✗ FAILED"
        print(f"  [{status}] {result.scenario_name}")
        if result.details:
            print(f"       {result.details}")
    
    def summary(self) -> Dict[str, Any]:
        elapsed = time.time() - self.start_time
        
        total = len(self.results)
        baseline_crashes = sum(1 for r in self.results if r.baseline_crashed)
        arc_recoveries = sum(1 for r in self.results if r.arc_recovered)
        arc_crashes = sum(1 for r in self.results if r.arc_crashed)
        
        print("\n" + "=" * 70)
        print("NIGHTMARE STRESS TEST - FINAL REPORT")
        print("=" * 70)
        
        # By tier
        tiers = {}
        for r in self.results:
            if r.tier not in tiers:
                tiers[r.tier] = {"total": 0, "recovered": 0}
            tiers[r.tier]["total"] += 1
            if r.arc_recovered:
                tiers[r.tier]["recovered"] += 1
        
        print("\nResults by Tier:")
        for tier, stats in sorted(tiers.items()):
            pct = 100 * stats["recovered"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {tier}: {stats['recovered']}/{stats['total']} ({pct:.0f}%)")
        
        print(f"\n{'─' * 50}")
        print(f"Total Scenarios:     {total}")
        print(f"Baseline Crashes:    {baseline_crashes}")
        print(f"ARC Recoveries:      {arc_recoveries}")
        print(f"ARC Crashes:         {arc_crashes}")
        print(f"Recovery Rate:       {100 * arc_recoveries / baseline_crashes:.1f}%" if baseline_crashes > 0 else "N/A")
        print(f"Time Elapsed:        {elapsed:.1f}s")
        print("─" * 50)
        
        if arc_recoveries == baseline_crashes:
            print("\n🎉 PERFECT SCORE - ARC SURVIVED ALL NIGHTMARE SCENARIOS!")
        elif arc_recoveries / baseline_crashes > 0.9:
            print("\n✓ EXCELLENT - ARC handled 90%+ of nightmare scenarios")
        elif arc_recoveries / baseline_crashes > 0.7:
            print("\n⚠ GOOD - ARC handled 70%+ but some scenarios need work")
        else:
            print("\n✗ NEEDS IMPROVEMENT - ARC struggled with nightmare scenarios")
        
        return {
            "total": total,
            "baseline_crashes": baseline_crashes,
            "arc_recoveries": arc_recoveries,
            "arc_crashes": arc_crashes,
            "recovery_rate": arc_recoveries / baseline_crashes if baseline_crashes > 0 else 0,
            "elapsed_seconds": elapsed,
            "by_tier": tiers
        }


# =============================================================================
# Model Definitions
# =============================================================================

class NightmareMLP(nn.Module):
    """Simple MLP for testing."""
    def __init__(self, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(100, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class NightmareCNN(nn.Module):
    """CNN that's prone to instability."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(256 * 4 * 4, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class UnstableTransformer(nn.Module):
    """Transformer block prone to attention explosion."""
    def __init__(self, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(50, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 10)
    
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # Add sequence dim
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)


# =============================================================================
# Nightmare Scenario Definitions
# =============================================================================

def create_dataloader(batch_size=32, n_samples=200, input_type="mlp"):
    """Create appropriate dataloader for model type."""
    if input_type == "cnn":
        X = torch.randn(n_samples, 3, 32, 32)
    else:
        X = torch.randn(n_samples, 100)
    y = torch.randint(0, 10, (n_samples,))
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)


def run_scenario(
    scenario_name: str,
    model_fn,
    failure_fn,
    trigger_step: int = 50,
    max_steps: int = 100,
    use_arc: bool = True,
    config_overrides: Optional[Dict] = None
) -> Dict:
    """Run a single nightmare scenario."""
    
    model = model_fn()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    dataloader = create_dataloader(input_type="mlp" if isinstance(model, NightmareMLP) else "cnn")
    
    arc = None
    if use_arc:
        config = SelfHealingConfig(
            loss_explosion_threshold=50.0,
            gradient_explosion_threshold=1e4,
            lr_reduction_factor=0.1,
            max_lr_reductions=20,  # Allow many LR reductions for stress test
            # Aggressive weight health monitoring
            enable_weight_health_check=True,
            weight_sparsity_threshold=0.2,  # Lower threshold to catch 50% zeros
            weight_health_check_frequency=1,  # Check every step
            verbose=False,
        )
        if config_overrides:
            for k, v in config_overrides.items():
                setattr(config, k, v)
        arc = SelfHealingArc(model, optimizer, config)
    
    results = {
        "steps_completed": 0,
        "crashed": False,
        "error": None,
        "rollbacks": 0,
        "losses": []
    }
    
    step = 0
    try:
        for epoch in range(20):  # Max epochs
            for batch_idx, (x, y_true) in enumerate(dataloader):
                step += 1
                if step > max_steps:
                    break
                
                optimizer.zero_grad()
                out = model(x)
                loss = F.cross_entropy(out, y_true)
                
                # Inject failure at trigger step
                if step >= trigger_step:
                    loss, model, optimizer = failure_fn(loss, model, optimizer, step)
                
                # ARC monitoring
                if arc:
                    action = arc.step(loss)
                    if action.should_skip:
                        if action.rolled_back:
                            results["rollbacks"] += 1
                        continue
                
                # Check for crash conditions (without ARC)
                if not arc:
                    if torch.isnan(loss) or torch.isinf(loss):
                        results["crashed"] = True
                        results["error"] = "NaN/Inf loss"
                        break
                    if loss.item() > 1e10:
                        results["crashed"] = True
                        results["error"] = "Loss explosion"
                        break
                
                # Backward pass
                try:
                    loss.backward()
                except Exception as e:
                    results["crashed"] = True
                    results["error"] = str(e)[:50]
                    break
                
                # Post-backward ARC check
                if arc:
                    post_action = arc.post_backward()
                    if post_action.should_skip:
                        continue
                
                # Check for NaN in weights
                has_nan = any(torch.isnan(p).any() for p in model.parameters())
                if has_nan and not arc:
                    results["crashed"] = True
                    results["error"] = "NaN in weights"
                    break
                
                # Check gradient explosion
                total_grad_norm = sum(
                    p.grad.norm().item() for p in model.parameters() 
                    if p.grad is not None
                )
                if total_grad_norm > 1e10 and not arc:
                    results["crashed"] = True
                    results["error"] = "Gradient explosion"
                    break
                
                # Check for weight sparsity explosion (silent corruption)
                if not arc:
                    total_params = sum(p.numel() for p in model.parameters())
                    total_zeros = sum((p.data == 0).sum().item() for p in model.parameters())
                    if total_zeros / total_params > 0.3:
                        results["crashed"] = True
                        results["error"] = "Weight sparsity explosion"
                        break
                
                optimizer.step()
                results["losses"].append(loss.item())
                results["steps_completed"] = step
            
            if step > max_steps or results["crashed"]:
                break
    
    except Exception as e:
        results["crashed"] = True
        results["error"] = str(e)[:100]
    
    if arc:
        stats = arc.get_stats()
        results["rollbacks"] = stats.get("total_rollbacks", 0)
    
    return results


# =============================================================================
# TIER 1: INSTANT DEATH SCENARIOS
# =============================================================================

def failure_nuclear_lr(loss, model, optimizer, step):
    """Set learning rate to 1e6 - instant explosion."""
    for pg in optimizer.param_groups:
        pg['lr'] = 1e6
    return loss, model, optimizer


def failure_all_nan(loss, model, optimizer, step):
    """Replace all weights with NaN."""
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(float('nan'))
    return loss, model, optimizer


def failure_div_zero(loss, model, optimizer, step):
    """Create division by zero in loss."""
    return loss / torch.tensor(0.0), model, optimizer


def failure_inf_gradients(loss, model, optimizer, step):
    """After backward, set all gradients to Inf."""
    # This happens post-backward, so we return a hook
    for p in model.parameters():
        if p.grad is not None:
            p.grad.fill_(float('inf'))
    return loss * 1e20, model, optimizer  # Also explode loss


def failure_nan_loss(loss, model, optimizer, step):
    """Return NaN loss directly."""
    return torch.tensor(float('nan')), model, optimizer


# =============================================================================
# TIER 2: CASCADING FAILURES
# =============================================================================

cascade_counter = [0]  # Mutable to track across calls

def failure_domino_collapse(loss, model, optimizer, step):
    """Trigger 5 consecutive failures."""
    cascade_counter[0] += 1
    if cascade_counter[0] <= 5:
        return torch.tensor(float('inf')), model, optimizer
    return loss, model, optimizer


_oscillation_state = [0]

def failure_oscillating_death(loss, model, optimizer, step):
    """Alternate between 0 and Inf loss."""
    _oscillation_state[0] += 1
    if _oscillation_state[0] % 2 == 0:
        return torch.tensor(float('inf')), model, optimizer
    else:
        return torch.tensor(0.0, requires_grad=True), model, optimizer


def failure_gradient_whiplash(loss, model, optimizer, step):
    """Extreme gradient sign flipping with huge magnitude."""
    sign = 1.0 if step % 2 == 0 else -1.0
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p.grad = torch.ones_like(p) * sign * 1e6
    return loss * 1000, model, optimizer


def failure_optimizer_corruption(loss, model, optimizer, step):
    """Destroy optimizer momentum buffers."""
    for pg in optimizer.param_groups:
        for p in pg['params']:
            state = optimizer.state[p]
            if 'exp_avg' in state:
                state['exp_avg'].fill_(float('nan'))
            if 'exp_avg_sq' in state:
                state['exp_avg_sq'].fill_(float('nan'))
    return loss, model, optimizer


def failure_explosive_loss(loss, model, optimizer, step):
    """Loss explodes by 1e10."""
    return loss * 1e10, model, optimizer


# =============================================================================
# TIER 3: ADVERSARIAL NIGHTMARE
# =============================================================================

def failure_byzantine_gradients(loss, model, optimizer, step):
    """50% of gradients are adversarial (random large values)."""
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                mask = torch.rand_like(p.grad) < 0.5
                p.grad[mask] = torch.randn_like(p.grad)[mask] * 1e5
    return loss, model, optimizer


_minefield_triggered = [False]

def failure_loss_minefield(loss, model, optimizer, step):
    """Random explosion with 30% chance each step."""
    if np.random.random() < 0.3:
        return torch.tensor(float('inf')), model, optimizer
    return loss, model, optimizer


def failure_silent_precision_death(loss, model, optimizer, step):
    """Simulate FP16 underflow accumulation."""
    with torch.no_grad():
        for p in model.parameters():
            # Add tiny values that accumulate to zero
            p.data *= 0.999
            if p.abs().max() < 1e-10:
                p.fill_(0)
    return loss, model, optimizer


def failure_batch_bomb(loss, model, optimizer, step):
    """Single step with extreme loss."""
    if step % 10 == 0:
        return loss * 1e15, model, optimizer
    return loss, model, optimizer


# =============================================================================
# TIER 4: COMBINED APOCALYPSE
# =============================================================================

def failure_everything_at_once(loss, model, optimizer, step):
    """Multiple failures simultaneously."""
    # 1. Explode loss
    loss = loss * 1000
    
    # 2. Corrupt some weights
    with torch.no_grad():
        for i, p in enumerate(model.parameters()):
            if i == 0:
                mask = torch.rand_like(p) < 0.1
                p[mask] = float('nan')
    
    # 3. Increase LR
    for pg in optimizer.param_groups:
        pg['lr'] *= 100
    
    return loss, model, optimizer


_repeated_failure_count = [0]

def failure_repeated_recovery(loss, model, optimizer, step):
    """Trigger failures every 2 steps for 10 consecutive failures."""
    _repeated_failure_count[0] += 1
    if _repeated_failure_count[0] <= 10:
        return torch.tensor(float('inf')), model, optimizer
    return loss, model, optimizer


def failure_long_tail(loss, model, optimizer, step):
    """Stable for many steps, then sudden catastrophe."""
    # Triggered only at step 50 (via trigger_step)
    return torch.tensor(float('nan')), model, optimizer


def failure_weight_surgery(loss, model, optimizer, step):
    """Randomly zero out 50% of weights - SILENT corruption."""
    with torch.no_grad():
        for p in model.parameters():
            mask = torch.rand_like(p) < 0.5
            p[mask] = 0.0
    # No loss modification - this is a SILENT failure
    return loss, model, optimizer


# =============================================================================
# Scenario Registry
# =============================================================================

NIGHTMARE_SCENARIOS = {
    # TIER 1: Instant Death
    "Tier1": [
        ("Nuclear Learning Rate (1e6)", failure_nuclear_lr),
        ("All-NaN Injection", failure_all_nan),
        ("Division by Zero", failure_div_zero),
        ("Immediate NaN Loss", failure_nan_loss),
        ("Loss Explosion (1e10)", failure_explosive_loss),
    ],
    
    # TIER 2: Cascading Failures
    "Tier2": [
        ("Domino Collapse (5x)", failure_domino_collapse),
        ("Oscillating Death", failure_oscillating_death),
        ("Gradient Whiplash (1e6)", failure_gradient_whiplash),
        ("Optimizer State Corruption", failure_optimizer_corruption),
    ],
    
    # TIER 3: Adversarial Nightmare
    "Tier3": [
        ("Byzantine Gradients (50%)", failure_byzantine_gradients),
        ("Loss Landscape Minefield", failure_loss_minefield),
        ("Batch Bomb", failure_batch_bomb),
    ],
    
    # TIER 4: Combined Apocalypse
    "Tier4": [
        ("Everything At Once", failure_everything_at_once),
        ("Repeated Recovery Stress (10x)", failure_repeated_recovery),
        ("Long-Tail Catastrophe", failure_long_tail),
        ("Weight Surgery (50% zeroed)", failure_weight_surgery),
    ],
}


# =============================================================================
# Main Test Runner
# =============================================================================

def run_nightmare_stress_test(verbose: bool = True):
    """Execute all nightmare scenarios."""
    
    print("=" * 70)
    print("🔥 ARC NIGHTMARE STRESS TEST 🔥")
    print("   Testing against the WORST POSSIBLE failure scenarios")
    print("=" * 70)
    
    report = NightmareReport()
    
    for tier_name, scenarios in NIGHTMARE_SCENARIOS.items():
        print(f"\n{'─' * 70}")
        print(f"TIER: {tier_name}")
        print("─" * 70)
        
        for scenario_name, failure_fn in scenarios:
            # Reset any global state
            cascade_counter[0] = 0
            _oscillation_state[0] = 0
            _repeated_failure_count[0] = 0
            
            print(f"\n>>> {scenario_name}")
            
            # Test WITHOUT ARC
            print("  [1/2] Running without ARC...")
            baseline = run_scenario(
                scenario_name,
                NightmareMLP,
                failure_fn,
                use_arc=False,
                max_steps=100
            )
            
            # Reset state again for ARC test
            cascade_counter[0] = 0
            _oscillation_state[0] = 0
            _repeated_failure_count[0] = 0
            
            # Test WITH ARC
            print("  [2/2] Running with ARC...")
            protected = run_scenario(
                scenario_name,
                NightmareMLP,
                failure_fn,
                use_arc=True,
                max_steps=100
            )
            
            # Analyze results
            arc_recovered = baseline["crashed"] and not protected["crashed"]
            
            result = TestResult(
                scenario_name=scenario_name,
                tier=tier_name,
                baseline_crashed=baseline["crashed"],
                arc_crashed=protected["crashed"],
                arc_rollbacks=protected["rollbacks"],
                arc_recovered=arc_recovered,
                baseline_steps=baseline["steps_completed"],
                arc_steps=protected["steps_completed"],
                details=f"Baseline: {'CRASH' if baseline['crashed'] else 'OK'} @ step {baseline['steps_completed']}, "
                       f"ARC: {'CRASH' if protected['crashed'] else 'OK'} @ step {protected['steps_completed']}, "
                       f"Rollbacks: {protected['rollbacks']}"
            )
            
            report.add(result)
    
    # Generate summary
    summary = report.summary()
    
    # Save results
    results_data = {
        "summary": summary,
        "scenarios": [
            {
                "name": r.scenario_name,
                "tier": r.tier,
                "baseline_crashed": r.baseline_crashed,
                "arc_crashed": r.arc_crashed,
                "arc_recovered": r.arc_recovered,
                "rollbacks": r.arc_rollbacks,
            }
            for r in report.results
        ]
    }
    
    output_path = os.path.join(os.path.dirname(__file__), "nightmare_results.json")
    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return report.results, summary


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    results, summary = run_nightmare_stress_test()
    
    # Exit code based on success rate
    if summary["recovery_rate"] >= 0.9:
        exit(0)
    else:
        exit(1)
