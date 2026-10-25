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
Full Scientific Benchmark: ARC vs Baselines

This script runs a comprehensive experiment suite:
1. Tests all failure types
2. Compares ARC against 6 baseline methods
3. Runs multiple trials for statistical significance
4. Generates publication-ready results

Run: python experiments/run_benchmark.py
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import math
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from arc.experiments import (
    ExperimentConfig,
    ExperimentRunner,
    FailureType,
    get_all_baselines,
)
from arc.introspection import IntrospectiveArc as Arc


def create_results_dir() -> Path:
    """Create timestamped results directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("experiments") / f"benchmark_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def run_arc_network_vs_baselines(
    n_trials: int = 10,
    failure_types: list = None,
    model_types: list = None,
    results_dir: Path = None,
) -> dict:
    """
    Run comprehensive comparison of ARC vs baselines.
    
    Args:
        n_trials: Trials per condition
        failure_types: Failure types to test
        model_types: Model architectures to test
        results_dir: Where to save results
    
    Returns:
        Dictionary with all results
    """
    if failure_types is None:
        failure_types = [
            FailureType.NONE,
            FailureType.DIVERGENCE,
            FailureType.VANISHING_GRADIENT,
            FailureType.EXPLODING_GRADIENT,
            FailureType.MODE_COLLAPSE,
            FailureType.OVERFITTING,
        ]
    
    if model_types is None:
        model_types = ["mlp", "deep_mlp"]
    
    if results_dir is None:
        results_dir = create_results_dir()
    
    all_results = {
        "arc_network": defaultdict(list),
        "baselines": {b.name: defaultdict(list) for b in get_all_baselines()},
    }
    
    baselines = get_all_baselines()
    
    print("=" * 70)
    print("SCIENTIFIC BENCHMARK: ARC vs Baselines")
    print("=" * 70)
    print(f"Results directory: {results_dir}")
    print(f"Trials per condition: {n_trials}")
    print(f"Failure types: {[f.name for f in failure_types]}")
    print(f"Model types: {model_types}")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n")
    
    runner = ExperimentRunner(
        output_dir=str(results_dir / "logs"),
        use_arc_network=True,
        device=str(device),
    )
    
    total_experiments = len(failure_types) * len(model_types) * n_trials
    current_exp = 0
    
    for model_type in model_types:
        for failure_type in failure_types:
            print(f"\n{'='*50}")
            print(f"{model_type.upper()} + {failure_type.name}")
            print(f"{'='*50}")
            
            omega_metrics = []
            baseline_metrics = {b.name: [] for b in baselines}
            
            for trial in range(n_trials):
                current_exp += 1
                print(f"\n  Trial {trial+1}/{n_trials} ({current_exp}/{total_experiments})")
                
                # Create reproducible config
                config = ExperimentConfig(
                    name=f"{model_type}_{failure_type.name.lower()}_trial_{trial}",
                    failure_type=failure_type,
                    model_type=model_type,
                    seed=42 + trial,
                    n_epochs=30,
                    failure_epoch=10,
                )
                
                # Run ARC experiment
                result = runner.run(config)
                
                omega_metrics.append({
                    "detected": result.failure_detected,
                    "lead_time": result.detection_lead_time,
                    "accuracy": result.final_accuracy,
                    "converged": result.converged,
                })
                
                # Run baseline experiments (same data)
                torch.manual_seed(config.seed)
                np.random.seed(config.seed)
                
                for baseline in baselines:
                    baseline.reset()
                    
                    detected = False
                    detection_step = None
                    
                    def safe_val(x):
                        if math.isnan(x) or math.isinf(x):
                            return 1e10
                        return x
                    
                    # Simulate detection using recorded metrics
                    losses = result.train_losses[:30] if result.train_losses else []
                    grads = result.gradient_norms[:30] if result.gradient_norms else []
                    
                    for step in range(min(len(losses), len(grads))):
                        loss = safe_val(losses[step])
                        grad = safe_val(grads[step])
                        if baseline.update(loss=loss, gradient_norm=grad):
                            if not detected:
                                detected = True
                                detection_step = step
                    
                    # Calculate lead time
                    lead_time = None
                    if detected and result.actual_failure_epoch is not None:
                        lead_time = result.actual_failure_epoch - detection_step
                    
                    baseline_metrics[baseline.name].append({
                        "detected": detected,
                        "lead_time": lead_time,
                    })
                
                # Print summary
                omega_status = "✓" if result.failure_detected == (failure_type != FailureType.NONE) else "✗"
                print(f"    ARC: {omega_status} (lead_time={result.detection_lead_time})")
                
                for b in baselines:
                    bm = baseline_metrics[b.name][-1]
                    b_status = "✓" if bm["detected"] == (failure_type != FailureType.NONE) else "✗"
                    print(f"    {b.name}: {b_status} (lead_time={bm['lead_time']})")
            
            # Store results
            key = f"{failure_type.name}_{model_type}"
            all_results["arc_network"][key] = omega_metrics
            for b in baselines:
                all_results["baselines"][b.name][key] = baseline_metrics[b.name]
    
    return all_results


def compute_summary_statistics(results: dict) -> dict:
    """Compute summary statistics from results."""
    summary = {
        "arc_network": {},
        "baselines": {},
    }
    
    # Omega-Net stats
    all_omega = []
    for key, trials in results["arc_network"].items():
        all_omega.extend(trials)
    
    if all_omega:
        omega_detections = [t["detected"] for t in all_omega]
        omega_lead_times = [t["lead_time"] for t in all_omega if t["lead_time"] is not None]
        
        summary["arc_network"] = {
            "detection_rate": np.mean(omega_detections),
            "mean_lead_time": np.mean(omega_lead_times) if omega_lead_times else 0,
            "std_lead_time": np.std(omega_lead_times) if omega_lead_times else 0,
            "n_experiments": len(all_omega),
        }
    
    # Baseline stats
    for baseline_name, results_by_condition in results["baselines"].items():
        all_baseline = []
        for key, trials in results_by_condition.items():
            all_baseline.extend(trials)
        
        if all_baseline:
            detections = [t["detected"] for t in all_baseline]
            lead_times = [t["lead_time"] for t in all_baseline if t["lead_time"] is not None]
            
            summary["baselines"][baseline_name] = {
                "detection_rate": np.mean(detections),
                "mean_lead_time": np.mean(lead_times) if lead_times else 0,
                "std_lead_time": np.std(lead_times) if lead_times else 0,
                "n_experiments": len(all_baseline),
            }
    
    return summary


def print_results_table(summary: dict) -> str:
    """Print publication-ready results table."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("FINAL RESULTS")
    lines.append("=" * 70)
    
    # Header
    header = f"{'Method':<25} {'Detection Rate':<15} {'Lead Time':<20}"
    lines.append(header)
    lines.append("-" * 70)
    
    # Omega-Net (highlight)
    omega = summary["arc_network"]
    lines.append(
        f"{'ARC':<25} "
        f"{omega.get('detection_rate', 0)*100:>6.1f}%        "
        f"{omega.get('mean_lead_time', 0):>5.1f} ± {omega.get('std_lead_time', 0):.1f} epochs"
    )
    
    lines.append("-" * 70)
    
    # Baselines
    for name, stats in summary["baselines"].items():
        lines.append(
            f"   {name:<22} "
            f"{stats.get('detection_rate', 0)*100:>6.1f}%        "
            f"{stats.get('mean_lead_time', 0):>5.1f} ± {stats.get('std_lead_time', 0):.1f} epochs"
        )
    
    lines.append("=" * 70)
    
    # Compute improvement
    omega_rate = summary["arc_network"].get("detection_rate", 0)
    best_baseline_rate = max(
        s.get("detection_rate", 0)
        for s in summary["baselines"].values()
    ) if summary["baselines"] else 0
    
    if best_baseline_rate > 0:
        improvement = (omega_rate - best_baseline_rate) / best_baseline_rate * 100
        lines.append(f"\nARC improvement over best baseline: {improvement:+.1f}%")
    
    result = "\n".join(lines)
    print(result)
    return result


def main():
    """Run complete benchmark."""
    print("\n" + "🔬" * 30)
    print("\n       NEURALPROPHET SCIENTIFIC VALIDATION")
    print("\n" + "🔬" * 30)
    
    start_time = time.time()
    
    # Create results directory
    results_dir = create_results_dir()
    
    # Run experiments
    results = run_arc_network_vs_baselines(
        n_trials=5,  # Reduced for demo, use 20+ for publication
        failure_types=[
            FailureType.NONE,
            FailureType.DIVERGENCE,
            FailureType.VANISHING_GRADIENT,
            FailureType.EXPLODING_GRADIENT,
        ],
        model_types=["mlp"],
        results_dir=results_dir,
    )
    
    # Compute statistics
    summary = compute_summary_statistics(results)
    
    # Print results
    results_table = print_results_table(summary)
    
    # Save results
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    
    with open(results_dir / "results_table.txt", "w") as f:
        f.write(results_table)
    
    elapsed = time.time() - start_time
    
    print(f"\nTotal time: {elapsed/60:.1f} minutes")
    print(f"Results saved to: {results_dir}")
    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()