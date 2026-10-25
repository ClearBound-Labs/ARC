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
ARC False Positive Test (Phase 6)

Addresses reviewer concern: "How often did ARC intervene unnecessarily?"

This module:
1. Runs normal training (no failures)
2. Counts false rollbacks/skips
3. Tests threshold sensitivity
4. Measures precision/recall
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc.core.self_healing import SelfHealingArc, SelfHealingConfig


# =============================================================================
# Model
# =============================================================================

class TestModel(nn.Module):
    __test__ = False  # not a pytest test class, just named "Test*"

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# =============================================================================
# False Positive Test
# =============================================================================

def run_normal_training(threshold, n_steps=500, seed=42):
    """Run normal training with no failures injected."""
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = TestModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    X = torch.randn(500, 100)
    y = torch.randint(0, 10, (500,))
    dataloader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)
    
    config = SelfHealingConfig(
        checkpoint_frequency=20,
        loss_explosion_threshold=threshold,
        verbose=False,
    )
    shard = SelfHealingArc(model, optimizer, config)
    
    result = {
        'threshold': threshold,
        'total_steps': 0,
        'false_rollbacks': 0,
        'false_skips': 0,
        'losses': [],
    }
    
    step = 0
    
    for epoch in range(10):
        for batch_idx, (x, labels) in enumerate(dataloader):
            step += 1
            if step > n_steps:
                break
            
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, labels)
            
            action = shard.step(loss)
            
            if action.should_skip:
                result['false_skips'] += 1
                if action.rolled_back:
                    result['false_rollbacks'] += 1
                continue
            
            loss.backward()
            shard.post_backward()
            optimizer.step()
            
            result['losses'].append(loss.item())
            result['total_steps'] = step
        
        if step > n_steps:
            break
    
    stats = shard.get_stats()
    result['false_rollbacks'] = stats['total_rollbacks']
    result['false_positive_rate'] = result['false_rollbacks'] / result['total_steps'] if result['total_steps'] > 0 else 0
    
    return result


def run_threshold_sensitivity():
    """Test different threshold values."""
    
    thresholds = [10.0, 25.0, 50.0, 100.0, 200.0, 500.0]
    
    print("="*70)
    print("FALSE POSITIVE TEST")
    print("   Running normal training with different thresholds")
    print("="*70)
    
    all_results = []
    
    for threshold in thresholds:
        print(f"\n  Threshold: {threshold}")
        
        # Run multiple seeds
        fp_rates = []
        for seed in [42, 123, 456]:
            result = run_normal_training(threshold, seed=seed)
            fp_rates.append(result['false_positive_rate'])
        
        mean_fp = np.mean(fp_rates) * 100
        std_fp = np.std(fp_rates) * 100
        
        print(f"    False Positive Rate: {mean_fp:.2f}% ± {std_fp:.2f}%")
        
        all_results.append({
            'threshold': threshold,
            'fp_rate_mean': mean_fp,
            'fp_rate_std': std_fp,
        })
    
    # Summary
    print("\n" + "="*70)
    print("THRESHOLD SENSITIVITY")
    print("="*70)
    
    print("\n| Threshold | FP Rate |")
    print("|-----------|---------|")
    
    for r in all_results:
        print(f"| {r['threshold']:9.0f} | {r['fp_rate_mean']:.2f}% ± {r['fp_rate_std']:.2f}% |")
    
    return all_results


def run_precision_recall():
    """Measure precision and recall on labeled failures."""
    
    print("\n" + "="*70)
    print("PRECISION-RECALL ANALYSIS")
    print("="*70)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    model = TestModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    X = torch.randn(500, 100)
    y = torch.randint(0, 10, (500,))
    dataloader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)
    
    config = SelfHealingConfig(
        checkpoint_frequency=10,
        loss_explosion_threshold=50.0,
        verbose=False,
    )
    shard = SelfHealingArc(model, optimizer, config)
    
    # Track predictions
    n_steps = 200
    failure_steps = {50, 100, 150}  # Inject failures at these steps
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    
    step = 0
    
    for epoch in range(10):
        for batch_idx, (x, labels) in enumerate(dataloader):
            step += 1
            if step > n_steps:
                break
            
            optimizer.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, labels)
            
            # Inject failure
            is_failure_step = step in failure_steps
            if is_failure_step:
                loss = torch.tensor(float('inf'))
            
            action = shard.step(loss)
            
            # Classify intervention
            intervened = action.should_skip
            
            if is_failure_step and intervened:
                true_positives += 1
            elif is_failure_step and not intervened:
                false_negatives += 1
            elif not is_failure_step and intervened:
                false_positives += 1
            else:
                true_negatives += 1
            
            if not action.should_skip:
                loss.backward()
                shard.post_backward()
                optimizer.step()
        
        if step > n_steps:
            break
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n  True Positives:  {true_positives}")
    print(f"  False Positives: {false_positives}")
    print(f"  True Negatives:  {true_negatives}")
    print(f"  False Negatives: {false_negatives}")
    print(f"\n  Precision: {precision*100:.1f}%")
    print(f"  Recall:    {recall*100:.1f}%")
    print(f"  F1 Score:  {f1*100:.1f}%")
    
    return {
        'tp': true_positives,
        'fp': false_positives,
        'tn': true_negatives,
        'fn': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def run_false_positive_analysis():
    """Run full false positive analysis."""
    
    print("="*70)
    print("FALSE POSITIVE ANALYSIS SUITE")
    print("="*70)
    
    # Threshold sensitivity
    threshold_results = run_threshold_sensitivity()
    
    # Precision-Recall
    pr_results = run_precision_recall()
    
    # Save
    all_results = {
        'threshold_sensitivity': threshold_results,
        'precision_recall': pr_results,
    }
    
    with open("false_positive_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    
    print("\nResults saved to: false_positive_results.json")
    
    return all_results


if __name__ == "__main__":
    run_false_positive_analysis()