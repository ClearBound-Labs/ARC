#!/usr/bin/env python

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
Scientific Efficiency Benchmark.

Measures the impact of Active Sensing + Signal Smoothing.
Goal: Reduce compute time while maintaining detection accuracy.

Run: python experiments/efficiency_benchmark.py
"""

import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc.introspection import IntrospectiveArc as Arc, create_self_aware_model

def get_simple_model():
    return nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 5)
    )

def run_trial(mode="adaptive", n_steps=200):
    """Run a trial with specified introspection mode."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    model = get_simple_model()
    omega = Arc(model, enable_geometry=True, enable_dynamics=True, enable_transport=True)
    
    # Configure efficiency modules
    omega.scheduler.mode = mode
    
    # Dummy data
    X = torch.randn(32, 20)
    y = torch.randint(0, 5, (32,))
    
    start_time = time.time()
    
    health_scores = []
    
    for i in range(n_steps):
        output = omega(X)
        loss = F.cross_entropy(output, y)
        
        # Simulate a failure at step 100
        if i >= 100:
             loss = loss * (i - 99) # Exploding loss
        
        omega.introspective_step(loss)
        
        state = omega.get_introspective_state()
        health_scores.append(state.overall_health)
    
    end_time = time.time()
    
    return {
        "time": end_time - start_time,
        "health_scores": health_scores
    }

def run_benchmark():
    print("="*60)
    print("SCIENTIFIC EFFICIENCY BENCHMARK")
    print("="*60)
    
    # 1. Run Baseline (Always On - The "Old" Way)
    print("\nRunning BASELINE (All Sensors Always On)...")
    res_base = run_trial(mode="always_on")
    print(f"Time: {res_base['time']:.4f}s")
    
    # 2. Run Efficient (Adaptive - The "New" Way)
    print("\nRunning ADAPTIVE (Active Sensing)...")
    res_adaptive = run_trial(mode="adaptive")
    print(f"Time: {res_adaptive['time']:.4f}s")
    
    # 3. Compare Results
    speedup = (res_base['time'] / res_adaptive['time'])
    savings = (1 - res_adaptive['time'] / res_base['time']) * 100
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Speedup: {speedup:.2f}x")
    print(f"Compute Savings: {savings:.1f}%")
    
    # Check signal correlation (did we lose accuracy?)
    base_health = np.array(res_base['health_scores'])
    adapt_health = np.array(res_adaptive['health_scores'])
    
    # Correlation on detection (did both detect the failure at step 100?)
    # Detection = health < 0.5
    base_detect = np.where(base_health < 0.5)[0]
    adapt_detect = np.where(adapt_health < 0.5)[0]
    
    if len(base_detect) > 0 and len(adapt_detect) > 0:
        base_first = base_detect[0]
        adapt_first = adapt_detect[0]
        delay = adapt_first - base_first
        print(f"Detection Delay: {delay} steps (Negative is better/equal)")
    else:
        print("Warning: Failure not detected in one or both modes.")

    if savings > 50:
         print("\nSUCCESS: Massive efficiency gain achieved!")
    else:
         print("\nWARNING: Efficiency gain minimal.")

if __name__ == "__main__":
    run_benchmark()