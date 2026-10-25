# ARC (Automatic Recovery Controller) - Self-Healing Neural Networks
# Copyright (c) 2026 Aryan Kaushik. All rights reserved.
#
# AAAI-2027 Benchmark Suite
# =========================
# Reproduces all claims for AAAI submission. Run on Kaggle with GPU.
# Generates JSON results files that map directly to paper tables.
#
# Usage:
#   python experiments/arc_aaai_benchmarks.py
#   (or run cells in the companion .ipynb notebook)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import time
import os
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc import Arc, ArcV2
from arc.config import Config, SignalConfig, FeatureConfig
from arc.intervention import WeightRollback, RollbackConfig
from arc.prediction import GradientForecaster

# ============================================================================
# Utility helpers
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "aaai_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_results(name: str, data: Any):
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  [OK] Saved: {path}")


def time_ms(fn):
    """Time a callable in milliseconds."""
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn()
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000
    return result, elapsed


# ============================================================================
# Model Zoo — architectures covering small → large
# ============================================================================

def make_mlp(hidden=128, layers=3):
    """Small MLP (~50K params)."""
    mods = [nn.Linear(100, hidden), nn.ReLU()]
    for _ in range(layers - 2):
        mods += [nn.Linear(hidden, hidden), nn.ReLU()]
    mods.append(nn.Linear(hidden, 10))
    return nn.Sequential(*mods)


def make_cnn():
    """Medium CNN (~300K params)."""
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
            )
            self.classifier = nn.Linear(128 * 4 * 4, 10)

        def forward(self, x):
            x = self.features(x)
            return self.classifier(x.view(x.size(0), -1))
    return CNN()


def make_resnet18():
    """ResNet-18 (~11M params) — uses torchvision if available."""
    try:
        from torchvision.models import resnet18
        return resnet18(num_classes=10)
    except ImportError:
        # Fallback: large MLP
        return make_mlp(hidden=1024, layers=6)


def make_transformer(d_model=256, nhead=8, num_layers=4, vocab=10000):
    """Small Transformer (~10M+ params)."""
    class SmallTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.head = nn.Linear(d_model, vocab)

        def forward(self, x):
            x = self.embed(x)
            x = self.encoder(x)
            return self.head(x[:, -1])
    return SmallTransformer()


MODEL_ZOO = {
    "MLP-50K": (make_mlp, lambda: torch.randn(8, 100)),
    "CNN-300K": (make_cnn, lambda: torch.randn(8, 3, 32, 32)),
    "ResNet18-11M": (make_resnet18, lambda: torch.randn(4, 3, 32, 32)),
    "Transformer-10M": (make_transformer, lambda: torch.randint(0, 10000, (4, 64))),
}

# ============================================================================
# Experiment 1: Failure Detection & Recovery (Table 1 in paper)
# ============================================================================

FAILURE_INJECTORS = {
    "nan_loss": lambda loss, step: torch.tensor(float("nan")) if step == 25 else loss,
    "inf_loss": lambda loss, step: torch.tensor(float("inf")) if step == 25 else loss,
    "loss_explosion": lambda loss, step: loss * 1e6 if step == 25 else loss,
    "gradient_explosion": lambda loss, step: loss * 1e5 if step == 25 else loss,
    "lr_spike": lambda loss, step: loss,  # LR spike is handled separately
}


def run_detection_recovery(n_seeds=5):
    """Test detection + recovery across architectures × failure types × seeds."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Failure Detection & Recovery")
    print("=" * 60)

    results = []

    for arch_name, (make_fn, make_input) in MODEL_ZOO.items():
        for fail_name, injector in FAILURE_INJECTORS.items():
            detections = 0
            recoveries = 0

            for seed in range(n_seeds):
                set_seed(seed)
                model = make_fn().to(DEVICE)
                optimizer = optim.Adam(model.parameters(), lr=0.01)
                config = RollbackConfig(checkpoint_frequency=5)
                rb = WeightRollback(model, optimizer, config, verbose=False)

                detected = False
                recovered = False
                initial_weights = {
                    k: v.clone() for k, v in model.state_dict().items()
                }

                # Special handling for LR spike
                if fail_name == "lr_spike":
                    for pg in optimizer.param_groups:
                        pg["lr"] = 0.01  # Reset

                for step in range(50):
                    x = make_input().to(DEVICE)
                    try:
                        out = model(x)
                        loss = out.mean()
                    except Exception:
                        loss = torch.tensor(1.0)

                    loss = injector(loss, step)

                    if fail_name == "lr_spike" and step == 25:
                        for pg in optimizer.param_groups:
                            pg["lr"] = 50.0  # Spike LR

                    try:
                        if not torch.isnan(loss) and not torch.isinf(loss):
                            loss.backward()
                    except Exception:
                        pass

                    action = rb.step(loss)

                    if action.failure_detected:
                        detected = True
                    if action.rolled_back:
                        recovered = True
                        optimizer.zero_grad()
                        continue

                    if not torch.isnan(loss) and not torch.isinf(loss):
                        optimizer.step()
                    optimizer.zero_grad()

                if detected:
                    detections += 1
                if recovered:
                    recoveries += 1

            result = {
                "architecture": arch_name,
                "failure_type": fail_name,
                "detection_rate": detections / n_seeds,
                "recovery_rate": recoveries / n_seeds,
                "n_seeds": n_seeds,
            }
            results.append(result)
            det_pct = result["detection_rate"] * 100
            rec_pct = result["recovery_rate"] * 100
            print(f"  {arch_name:20s} × {fail_name:20s}: Det={det_pct:5.1f}%  Rec={rec_pct:5.1f}%")

    save_results("exp1_detection_recovery", results)
    return results


# ============================================================================
# Experiment 2: False Positive Rate (Table 2 in paper)
# ============================================================================

def run_false_positive_test(n_steps=1000, n_seeds=5):
    """Run stable training and count false rollback triggers."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: False Positive Rate")
    print("=" * 60)

    results = []

    for arch_name, (make_fn, make_input) in MODEL_ZOO.items():
        total_fp = 0
        total_steps = 0

        for seed in range(n_seeds):
            set_seed(seed)
            model = make_fn().to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            rb = WeightRollback(model, optimizer, RollbackConfig(), verbose=False)

            fp = 0
            for step in range(n_steps):
                x = make_input().to(DEVICE)
                out = model(x)
                loss = out.mean()
                loss.backward()
                action = rb.step(loss)
                if action.rolled_back:
                    fp += 1
                optimizer.step()
                optimizer.zero_grad()

            total_fp += fp
            total_steps += n_steps

        result = {
            "architecture": arch_name,
            "total_steps": total_steps,
            "false_positives": total_fp,
            "fp_rate": total_fp / total_steps,
            "n_seeds": n_seeds,
        }
        results.append(result)
        print(f"  {arch_name:20s}: {total_fp} FPs in {total_steps} steps ({result['fp_rate']*100:.2f}%)")

    save_results("exp2_false_positives", results)
    return results


# ============================================================================
# Experiment 3: Overhead Measurement (Table 3 in paper)
# ============================================================================

def run_overhead_measurement(n_steps=100):
    """Measure ARC overhead as % of training step time."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Overhead Measurement")
    print("=" * 60)

    results = []

    for arch_name, (make_fn, make_input) in MODEL_ZOO.items():
        set_seed(42)
        model = make_fn().to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        # Time WITHOUT ARC
        times_no_arc = []
        for _ in range(n_steps):
            x = make_input().to(DEVICE)
            _, elapsed = time_ms(lambda: model(x).mean())
            times_no_arc.append(elapsed)
        avg_no_arc = np.mean(times_no_arc[10:])  # skip warmup

        # Time WITH ARC
        rb = WeightRollback(model, optimizer, RollbackConfig(), verbose=False)
        times_with_arc = []
        for _ in range(n_steps):
            x = make_input().to(DEVICE)

            def step_with_arc():
                out = model(x)
                loss = out.mean()
                loss.backward()
                action = rb.step(loss)
                optimizer.step()
                optimizer.zero_grad()
                return action

            _, elapsed = time_ms(step_with_arc)
            times_with_arc.append(elapsed)
        avg_with_arc = np.mean(times_with_arc[10:])

        overhead = avg_with_arc - avg_no_arc
        overhead_pct = (overhead / avg_no_arc) * 100 if avg_no_arc > 0 else 0

        result = {
            "architecture": arch_name,
            "n_params": n_params,
            "avg_step_no_arc_ms": round(avg_no_arc, 3),
            "avg_step_with_arc_ms": round(avg_with_arc, 3),
            "arc_overhead_ms": round(overhead, 3),
            "overhead_percent": round(overhead_pct, 1),
        }
        results.append(result)
        print(f"  {arch_name:20s} ({n_params:>12,} params): "
              f"Base={avg_no_arc:.2f}ms  +ARC={avg_with_arc:.2f}ms  "
              f"Overhead={overhead:.2f}ms ({overhead_pct:.1f}%)")

    save_results("exp3_overhead", results)
    return results


# ============================================================================
# Experiment 4: Baseline Comparison (Table 4 in paper)
# ============================================================================

def run_baseline_comparison(n_seeds=5):
    """Compare ARC vs no protection vs gradient clipping vs loss-only monitoring."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Baseline Comparison")
    print("=" * 60)

    methods = {
        "No Protection": "none",
        "Gradient Clipping (max_norm=1)": "clip",
        "Loss-Only Monitor": "loss_only",
        "Full ARC": "arc",
    }

    failure_types = ["nan_loss", "loss_explosion", "gradient_explosion"]
    results = []

    for method_name, method_type in methods.items():
        for fail_name in failure_types:
            injector = FAILURE_INJECTORS[fail_name]
            recoveries = 0
            detections = 0

            for seed in range(n_seeds):
                set_seed(seed)
                model = make_mlp().to(DEVICE)
                optimizer = optim.Adam(model.parameters(), lr=0.01)

                detected = False
                recovered = False

                if method_type == "arc":
                    rb = WeightRollback(model, optimizer,
                                       RollbackConfig(checkpoint_frequency=5),
                                       verbose=False)
                else:
                    rb = None

                for step in range(50):
                    x = torch.randn(8, 100).to(DEVICE)
                    loss = model(x).mean()
                    loss = injector(loss, step)

                    is_bad = torch.isnan(loss) or torch.isinf(loss)

                    if method_type == "none":
                        if is_bad:
                            detected = True
                            break
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                    elif method_type == "clip":
                        if is_bad:
                            detected = True
                            break
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()

                    elif method_type == "loss_only":
                        if is_bad:
                            detected = True
                            # Simple recovery: skip step
                            recovered = True
                            optimizer.zero_grad()
                            continue
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                    elif method_type == "arc":
                        try:
                            if not is_bad:
                                loss.backward()
                        except Exception:
                            pass
                        action = rb.step(loss)
                        if action.failure_detected:
                            detected = True
                        if action.rolled_back:
                            recovered = True
                            optimizer.zero_grad()
                            continue
                        if not is_bad:
                            optimizer.step()
                        optimizer.zero_grad()

                if detected:
                    detections += 1
                if recovered:
                    recoveries += 1

            result = {
                "method": method_name,
                "failure_type": fail_name,
                "detection_rate": detections / n_seeds,
                "recovery_rate": recoveries / n_seeds,
            }
            results.append(result)

    # Print summary table
    print(f"\n  {'Method':<35s} {'Detection':>10s} {'Recovery':>10s}")
    print("  " + "-" * 55)
    for r in results:
        print(f"  {r['method']:<35s} {r['detection_rate']*100:>9.1f}% {r['recovery_rate']*100:>9.1f}%")

    save_results("exp4_baseline_comparison", results)
    return results


# ============================================================================
# Experiment 5: ArcV2.auto() Config Correctness (Regression)
# ============================================================================

def run_config_correctness():
    """Verify ArcV2.auto() produces correct config objects at all scales."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: ArcV2.auto() Config Correctness")
    print("=" * 60)

    results = []

    test_cases = [
        ("Small (<10M)", make_mlp, lambda: torch.randn(4, 100)),
        ("Medium (>10M)", make_resnet18, lambda: torch.randn(4, 3, 32, 32)),
    ]

    for name, make_fn, make_input in test_cases:
        model = make_fn().to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        optimizer = optim.Adam(model.parameters())
        arc = ArcV2.auto(model, optimizer)

        from arc.config import SignalConfig, FeatureConfig
        signal_ok = isinstance(arc.config.signal, SignalConfig)
        feature_ok = isinstance(arc.config.feature, FeatureConfig)
        ratio = arc.config.signal.activation_sample_ratio
        ws = arc.config.feature.window_size
        curvature = arc.config.signal.compute_curvature_proxy

        result = {
            "name": name,
            "n_params": n_params,
            "signal_is_dataclass": signal_ok,
            "feature_is_dataclass": feature_ok,
            "activation_sample_ratio": ratio,
            "window_size": ws,
            "can_access_compute_curvature_proxy": True,
            "all_checks_pass": signal_ok and feature_ok and isinstance(ratio, float),
        }
        results.append(result)
        status = "[PASS]" if result["all_checks_pass"] else "[FAIL]"
        print(f"  {name:20s} ({n_params:>12,} params): {status}")
        print(f"    signal={type(arc.config.signal).__name__}, "
              f"ratio={ratio}, window={ws}")

    save_results("exp5_config_correctness", results)
    return results


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    print("=" * 60)
    print("ARC — AAAI 2027 Benchmark Suite")
    print(f"Device: {DEVICE}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 60)

    all_results = {}

    all_results["exp1"] = run_detection_recovery(n_seeds=5)
    all_results["exp2"] = run_false_positive_test(n_steps=500, n_seeds=3)
    all_results["exp3"] = run_overhead_measurement(n_steps=100)
    all_results["exp4"] = run_baseline_comparison(n_seeds=5)
    all_results["exp5"] = run_config_correctness()

    save_results("all_experiments_summary", {
        "device": DEVICE,
        "pytorch_version": torch.__version__,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiments": all_results,
    })

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
