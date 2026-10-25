<div align="center">

# ARC v4.1 Technical Report

### Efficiency, Recovery Performance, and Scalability Analysis

_March 2026_

</div>

> **Historical document.** This report describes ARC v4.1. Its numbers predate the October 2026 audit, which found that several stress scripts didn't apply their crash checks to the protected run. For current, symmetric results, see the [README benchmarks](../../README.md#benchmarks).

---

## Executive Summary

ARC (Autonomous Recovery Controller) is a runtime monitoring framework for autonomous detection and recovery from neural network training failures. This report documents empirical results from controlled experiments.

| Metric              | Result                          | Backing Script              |
| :------------------ | :------------------------------ | :-------------------------- |
| Recovery Rate       | **100%** (25/25 scenarios)      | `baseline_comparison.py`    |
| Prediction Accuracy | **97.5%** (200 scenarios, 5-CV) | `prediction_200_v2.py`      |
| Precision           | **100%** (zero false positives) | `prediction_200_v2.py`      |
| Overhead            | **<10%** for 250K+ params (CPU) | `overhead_measurement.py`   |
| Model Scale         | Up to **117M** parameters       | `validate_claims_phase2.py` |

> **Scope**: All recovery claims are restricted to programmatically induced failures. Real-world/organic failures are untested. All timing measurements were conducted on CPU.

---

## 1. Baseline Comparison

**Protocol**: 4 methods × 5 failure types × 5 seeds = 25 scenarios per method.
**Script**: `experiments/baseline_comparison.py`

| Method            | Detection | Recovery | False Pos. | Avg. Time |
| :---------------- | :-------: | :------: | :--------: | :-------: |
| No Protection     |   52.0%   |   0.0%   |     0      |   926ms   |
| Gradient Clipping |   20.0%   |   0.0%   |     0      |  1297ms   |
| Loss-Only Monitor |   80.0%   |  80.0%   |     0      |  1359ms   |
| **Full ARC**      | **100%**  | **100%** |   **0**    |  1722ms   |

**Key finding**: ARC's optimizer state monitoring catches silent failures (momentum buffer zeroing) that loss-only monitoring misses entirely.

---

## 2. Failure Prediction

**Protocol**: 4 architectures × 5 failure types × 5 seeds × 2 labels = 200 scenarios, 5-fold CV.
**Script**: `experiments/prediction_200_v2.py`

### v2 (12 features, current)

| Classifier         |     Accuracy     | Precision | Recall  |        F1        |
| :----------------- | :--------------: | :-------: | :-----: | :--------------: |
| Logistic Reg (12f) |   95.5% ± 1.9%   |   100%    |  91.0%  |   0.953 ± 2.6%   |
| **MLP (12f)**      | **97.5% ± 2.2%** | **100%**  | **95%** | **0.974 ± 2.8%** |

12 features: loss trend, loss variance, gradient mean/variance/max, weight norm change, weight variance, NaN count, **optimizer state norm change**, **loss acceleration**, **gradient entropy change**, **weight norm acceleration** (last 4 are new in v2).

### v1 → v2 improvement

| Metric   | v1 (8 feat, LogReg) | v2 (12 feat, MLP) |
| :------- | :-----------------: | :---------------: |
| Accuracy |        86.5%        |     **97.5%**     |
| Recall   |        73.0%        |     **95.0%**     |
| F1       |        0.844        |     **0.974**     |

---

## 3. Ablation Study

**Protocol**: 7 failure types × 5 seeds = 35 scenarios per configuration.
**Script**: `experiments/ablation_experiment.py`

| Configuration             | Detection | Δ from Full |
| :------------------------ | :-------: | :---------: |
| Full ARC (all components) |   85.7%   |     ---     |
| − Weight Health           |   85.7%   |    0.0%     |
| − Gradient Monitoring     |   85.7%   |    0.0%     |
| − Loss Monitoring         |   85.7%   |    0.0%     |
| − Optimizer State         |   71.4%   |   −14.3%    |
| Loss Only (baseline)      |   71.4%   |   −14.3%    |

**Interpretation**: Weight/gradient/loss monitoring provide redundant coverage (defense-in-depth). Optimizer state monitoring is uniquely valuable for silent failures. The 14.3% undetected by Full ARC are silent corruption scenarios too subtle for any threshold-based detector.

---

## 4. Overhead Analysis

**Protocol**: Median of 100 iterations, `time.perf_counter`, CPU.
**Script**: `experiments/overhead_measurement.py`

### Per-Component Timing (288K-parameter CNN)

| Component           | Time (ms) | % of ARC Total |
| :------------------ | :-------: | :------------: |
| Gradient Norm       |   0.12    |      9.0%      |
| Weight Statistics   |   1.06    |     76.9%      |
| Loss Analysis       |   0.01    |      0.6%      |
| Checkpoint (amort.) |   0.13    |      9.6%      |
| Forecasting         |   0.06    |      4.1%      |
| **Total ARC**       | **1.38**  |    **100%**    |

Weight statistics (norm computation + NaN/Inf check) dominates at 77%.

### Relative Overhead by Model Scale

| Model      | Parameters | ARC (ms) | Baseline (ms) | Relative |
| :--------- | :--------: | :------: | :-----------: | :------: |
| Small MLP  |    50K     |   0.86   |     1.45      |   ~60%   |
| Medium CNN |    288K    |   1.38   |     14.17     |   ~10%   |
| Large CNN  |    2.5M    |   7.04   |     74.24     |  ~9.5%   |

Overhead decreases with model size because forward/backward cost grows superlinearly while monitoring is O(n).

> **Note**: All measurements on CPU. GPU deployments would show lower relative overhead.

---

## 5. Large Model Stress Test

**Protocol**: Models trained 20 steps for stable baseline, failure injected at step 20, recovery verified by finite loss + return to 2× pre-failure baseline within 10 steps.
**Script**: `experiments/validate_claims_phase2.py`

| Model        | Params | Failure Type        | Recovery | Rollbacks |
| :----------- | :----- | :------------------ | :------: | :-------: |
| NanoGPT      | 10M    | LR Spike (50×)      |    ✓     |     2     |
| ResNet-50    | 25.6M  | Loss Singularity    |    ✓     |     1     |
| YOLOv11      | 30M    | Catastrophic LR     |    ✓     |     3     |
| GPT-2 Small  | 50M    | NaN Bomb            |    ✓     |     4     |
| SD-UNet      | 60M    | Gradient Apocalypse |    ✓     |     4     |
| Wide ResNet  | 68M    | Loss Supernova      |    ✓     |     3     |
| Llama-style  | 70M    | Catastrophic LR     |    ✓     |     5     |
| ViT-Base     | 86M    | Inf Nuke            |    ✓     |     1     |
| GPT-2 Medium | 117M   | NaN Bomb            |    ✓     |     3     |

100% recovery across all 9 models, all with programmatically injected failures.

---

## 6. What's Not Validated

| Claim                | Status                                                                                     |
| :------------------- | :----------------------------------------------------------------------------------------- |
| GPU overhead         | Untested: all measurements on CPU |
| Scale >117M          | Untested: behaviour at 1B+ unknown |
| Organic failures     | Untested: all failures injected programmatically |
| Distributed training | Untested: single-process only |
| Non-CIFAR datasets   | Limited: stress tests use domain-appropriate data, prediction experiments use CIFAR-10 |
