<div align="center">

# ARC

### Autonomous Recovery Controller for Neural Network Training

_ARC monitors, predicts, and recovers from training failures automatically. One controller, one call per step._

<br>

[![PyPI](https://img.shields.io/pypi/v/arc-training?style=for-the-badge&logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/arc-training)
[![Downloads](https://img.shields.io/pepy/dt/arc-training?style=for-the-badge&label=Downloads)](https://pepy.tech/project/arc-training)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-green?style=for-the-badge)](https://www.gnu.org/licenses/agpl-3.0)
[![Docs](https://img.shields.io/badge/Docs-pyarc.pages.dev-orange?style=for-the-badge&logo=readthedocs&logoColor=white)](https://pyarc.pages.dev)
[![Discord](https://img.shields.io/badge/Discord-Join%20Community-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/E6UvPWC8DW)

<br>

**100% recovery** (33/33 crash scenarios) · **97.5% prediction accuracy** · **0 false interventions** · **2.6B parameters validated** (H200)

<br>

[Docs](https://pyarc.pages.dev/docs.html) •
[API](https://pyarc.pages.dev/api.html) •
[ARC Lens](https://pyarc.pages.dev/lens.html) •
[Examples](https://pyarc.pages.dev/examples.html) •
[Benchmarks](https://pyarc.pages.dev/benchmarks.html) •
[FAQ](https://pyarc.pages.dev/faq.html) •
[Discord](https://discord.gg/E6UvPWC8DW)

</div>

---

## Quick Start

```bash
pip install arc-training
```

```python
import arc

controller = arc.Controller(model, optimizer)

for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()

    action = controller.step(loss)
    if action.should_skip:
        continue          # ARC already rolled back; skip this update

    optimizer.step()

controller.summary()      # end-of-run report: failures, rollbacks, LR changes
```

Call `controller.step(loss)` **after** `loss.backward()` and **before** `optimizer.step()`. No changes to your model, loss, or optimizer.

<details>
<summary><b>Install from source / optional extras</b></summary>

```bash
git clone https://github.com/a-kaushik2209/ARC.git
cd ARC
pip install -e .
```

```bash
pip install arc-training[lightning]
pip install arc-training[full]
pip install arc-training[dev]
```

Requires Python 3.8+, PyTorch 1.9+, NumPy 1.19+, SciPy 1.7+.

</details>

| `StepAction` field | Meaning |
| :--- | :--- |
| `should_skip` | Skip `optimizer.step()` for this batch |
| `status` | `HealthStatus.HEALTHY`, `WARNING`, or `FAILURE` |
| `failure_type` | e.g. `nan_loss`, `loss_explosion`, `gradient_explosion`, `lyapunov_divergence`, `drift_detected` |
| `rolled_back` / `lr_reduced` | What ARC did in response |
| `unrecoverable` | The fault is outside the model. Stop and fix the input pipeline |

> **ARC clips gradients by default.** On healthy steps ARC clips the gradient norm to `gradient_clip_norm=1.0`. If your loop already clips, set `enable_baseline_clipping=False` in `ControllerConfig`.

Full guide: **[pyarc.pages.dev/docs.html](https://pyarc.pages.dev/docs.html)**

---

## The Problem

**Training neural networks is fragile.** A single NaN gradient at hour 47 of a 48-hour run can destroy days of compute. Engineers waste enormous time adding manual checkpointing and babysitting long runs.

- **NaN explosions** - loss becomes NaN or Inf with no warning, corrupting the run irreversibly.
- **Gradient collapse** - vanishing or exploding gradients silently halt learning, wasting compute without any alert.
- **Wasted compute** - cloud GPU spending exceeds $30B annually. Every unrecoverable failure burns money directly.
- **Manual monitoring** - engineers watch training overnight, writing custom recovery scripts and checkpointing logic.

---

## How It Works

| | Stage | What happens |
| :-: | :--- | :--- |
| **01** | **Monitor** | Checks every step for NaN/Inf, loss explosions, gradient explosions, and weight corruption. Thresholds auto-calibrate to your model during the first 50 steps. |
| **02** | **Predict** | Lyapunov stability and trend-drift detectors flag instability 26-101 steps before the loss diverges. The MLP failure classifier reaches 97.5% accuracy with zero false positives. |
| **03** | **Recover** | Restores model, optimizer, and RNG state from the last healthy checkpoint, then cuts the learning rate. If a fault keeps coming back from the data, ARC says so and marks the run unrecoverable instead of looping. |

```text
Controller.step(loss)
    |
    +-- warm-up: auto-calibrate thresholds (first 50 steps; NaN/Inf still caught)
    +-- NaN / Inf loss                    immediate
    +-- loss explosion                    absolute threshold + 10x spike vs recent mean
    +-- NaN / Inf weights, weight health  every 10 steps: sparsity, magnitude collapse
    +-- gradient explosion                norm threshold, then baseline clipping
    +-- Lyapunov divergence               predictive, every 5 steps
    +-- trend drift                       OLS slope over a sliding window
    |
    +--[FAILURE]--> restore checkpoint (model + optimizer + RNG)
    |               reduce LR x0.5 (re-applied after every restore)
    |               5 rollbacks within a few steps -> escape: oldest checkpoint,
    |                   perturb weights, 2 extra LR cuts, skip 5 steps
    |               3 escapes with no progress -> unrecoverable (warns once)
    |
    +--[HEALTHY]--> update history, adaptive checkpoint
```

---

## Capabilities

| Capability | Description |
| :--- | :--- |
| **Unified Controller** | One entry point, `arc.Controller`, replaces five legacy APIs. Rollback, learning-rate ratcheting, scheduler sync, and gradient clipping in a single `step()` call. |
| **Lyapunov Early Warning** | Online Lyapunov exponent estimation with FFT oscillation detection catches instability 26-101 steps before loss-based detection. |
| **Trend Drift Detection** | Sliding-window regression on loss, gradient, and weight norms catches slow numerical drift that threshold checks miss. |
| **Auto-Calibration** | Learns loss and gradient thresholds from your model's first 50 steps, so you never have to guess what "too high" means. |
| **Honest Failure Handling** | When a fault lives in the data (a NaN row, `log(0)`, a zero-variance batch), ARC stops claiming recovery, raises one warning, and names the likely cause. |
| **Recovery Ledger and ROI** | A hash-chained, tamper-evident log of every intervention, exportable as JSON, with GPU-hours, dollars, and carbon saved per run, team, and project. |
| **Checkpoint Integrity** | SHA-256 and HMAC signing on every storage strategy, verified on restore, plus a disaster-recovery drill that proves a checkpoint restores correctly. |
| **Policy Engine and Alerts** | Declarative remediation rules over failure predictions with a dry-run approval gate, and a severity-stacked alert router with de-duplication. |
| **RL and Generation Stability** | Detectors for KL blowup, entropy collapse, reward hacking, advantage explosion, mode collapse, and n-gram repetition in rollouts. |
| **Distributed Health Signals** | Straggler detection across ranks, cross-replica gradient checks for silent data corruption, and bit-reproducibility reports. |
| **Async Checkpointing** | CPU-offloaded, background-thread checkpoints with adaptive frequency: more often when training is unstable, less often when it is calm. |
| **Research Toolkit** | PINN stabilizer, conformal uncertainty, adversarial detection, and EWC continual learning ship in the same package. |

Examples for CNNs, YOLO, Transformers, LLM fine-tuning, PINNs, RL, and more: **[pyarc.pages.dev/examples.html](https://pyarc.pages.dev/examples.html)**

---

## Configuration

```python
from arc import Controller, ControllerConfig

config = ControllerConfig(
    checkpoint_frequency=20,
    calibration_steps=50,
    lr_reduction_factor=0.5,
    enable_lyapunov=True,
    enable_trend=True,
    gradient_accumulation_steps=4,   # only check full accumulation steps
    verbose=False,
)
controller = Controller(model, optimizer, scheduler=scheduler, config=config)
```

All defaults are listed in the [configuration reference](https://pyarc.pages.dev/docs.html#config).

---

## Architecture

```text
arc/
  core/            Controller (canonical API), recovery ledger
  signals/         Lyapunov, trend, gradient, weight, RL + rollout health
  features/        Feature extraction and buffering
  prediction/      Failure predictor, remediation policy engine
  intervention/    Rollback, OOM, hardware, DDP, model heuristics
  checkpointing/   Adaptive + signed checkpoints, DR drill
  distributed/     DDP rollback, straggler detection
  evaluation/      Reliability metrics, run comparison, chargeback
  security/        SDC detection, health baseline, adversarial
  api/             Alert router, enterprise report, callbacks
  introspection/   Hessian + Fisher analysis (experimental)
  physics/         PINN stabilization
  uncertainty/     Conformal prediction
  learning/        Failure meta-model, EWC
```

- **One canonical entry point** - `Controller.step(loss)` runs every detector in order and returns a `StepAction` with health status, failure type, and what ARC did about it.
- **Toggleable signals** - Lyapunov, trend, gradient, and weight-health monitors switch on or off independently in `ControllerConfig`, without touching recovery logic.
- **Auditable by default** - every intervention lands in a hash-chained ledger you can export, verify, diff across runs, and roll up per team.
- **Minimal dependencies** - requires PyTorch, NumPy, and SciPy. Optional integrations for Lightning, Hugging Face, and Weights & Biases.

**Works with any architecture:** CNN / ResNet, Transformers, YOLO / ViT, PINNs, GPT / LLMs, Diffusion UNet, RNN / LSTM, GANs.

Validated on GPT-2 Medium, Large, and XL (1.5B), a 2.6B-parameter custom transformer, BERT, ResNet-18/50, VGG11, ViT-Base, DINOv2, YOLOv11, Llama-style models, and Stable Diffusion UNet.

---

## ARC Lens (VS Code)

[ARC Lens](https://pyarc.pages.dev/lens.html) is a self-contained VS Code extension for real-time PyTorch training visualization: live losses, learning rates, gradient norms, GPU memory, and layer effective rank, plus anomaly detection and a bundled offline recovery agent. No data leaves your machine.

---

## Benchmarks

Every number comes from a reproducible script in [`experiments/`](https://github.com/a-kaushik2209/ARC/tree/master/experiments) or from the ARC research paper.

### Recovery across test suites

| Test Suite | Scenarios | Baseline Crashes | ARC Recovered | Recovery Rate |
| :--- | :-: | :-: | :-: | :-: |
| Nightmare Stress | 16 | 14 | 14 | 100% |
| Comprehensive | 38 | 13 | 13 | 100% |
| Silent Failure | 13 | 6 (detectable) | 6 | 100% |
| **Combined** | **67** | **33** | **33** | **100%** |

### Baseline comparison

| Method | Detection | Recovery | False Pos. |
| :--- | :-: | :-: | :-: |
| No Protection | 52.0% | 0.0% | 0 |
| Gradient Clipping | 20.0% | 0.0% | 0 |
| Loss-Only Monitor | 80.0% | 80.0% | 0 |
| **Full ARC** | **100%** | **100%** | **0** |

### Billion-scale validation (NVIDIA H200)

NaN loss injected at step 30.

| Model | Params | No Protection | ARC | Rollbacks |
| :--- | :-: | :-: | :-: | :-: |
| GPT-2 Medium | 355M | Crashed at step 30 | Recovered | 1 |
| GPT-2 Large | 774M | Crashed at step 30 | Recovered | 1 |
| GPT-2 XL | 1.56B | Crashed at step 30 | Recovered | 1 |
| Custom Transformer | 2.65B | Crashed at step 30 | Recovered | 1 |

### Real-dataset recovery

| Experiment | Clean Baseline | ARC-Protected (failure injected) |
| :--- | :-: | :-: |
| CIFAR-10, ResNet-18 (3 seeds) | 77.43% ± 0.12 | 77.42% ± 0.15 |
| GPT-2 (124M) on WikiText-2, perplexity | 15.8 | 16.4 |

### Failure prediction

| Classifier | Accuracy | Precision | Recall | F1 |
| :--- | :-: | :-: | :-: | :-: |
| Logistic Regression (12 features) | 95.5% | 100% | 91.0% | 0.953 |
| **MLP (12 features)** | **97.5% ± 2.2** | **100%** | **95%** | **0.974** |

### Overhead (CPU)

| Model | Parameters | ARC (ms) | Baseline (ms) | Relative |
| :--- | :-: | :-: | :-: | :-: |
| Small MLP | 50K | 0.86 | 1.45 | ~60% |
| Medium CNN | 288K | 1.38 | 14.17 | ~10% |
| Large CNN | 2.5M | 7.04 | 74.24 | ~9.5% |

End-to-end GPU overhead on a Tesla T4 is 40.7% (GPT-2, 124M) and 33.4% (GPT-2 Medium, 355M). Relative overhead falls as models grow.

Full results, protocols, and ablations: **[pyarc.pages.dev/benchmarks.html](https://pyarc.pages.dev/benchmarks.html)**

---

## Known Limitations

| Area | Status |
| :--- | :--- |
| GPU overhead | 33-41% end-to-end on T4 for 124M-355M models. Not yet profiled on H100/H200 |
| Scale | Recovery validated up to 2.65B parameters on a single H200. Checkpoint snapshots need memory headroom |
| Organic failures | Partially tested (8-scenario harness, realistic bugs on ResNet-50). Long-duration organic runs at scale are untested |
| Data faults | Detected but not repairable by rollback. ARC flags them unrecoverable and names the likely cause |
| Distributed training | Recovery runs single-process. DDP rollback, straggler, and SDC algorithms exist but are not yet validated on a live multi-node cluster |
| Multiplicative weight corruption | Scaling all weights (e.g. 1000x) without producing NaN or sparsity is not caught |
| Frameworks | PyTorch only |

---

## Community

<div align="center">

[![Discord](https://img.shields.io/badge/Join%20the%20ARC%20Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/E6UvPWC8DW)

</div>

The Discord server is the primary hub for contributor discussions, development updates, research discussions, feature proposals, and debugging support. Contributors, researchers, students, and developers of all experience levels are welcome.

## Contributing

1. Open an issue describing the feature, fix, or improvement.
2. Wait for the issue to be **assigned to you** before starting work. PRs without prior discussion or assignment may be closed.
3. Branch, make a focused change, and open a PR that references the issue.

Full workflow and standards: **[CONTRIBUTING.md](https://github.com/a-kaushik2209/ARC/blob/master/CONTRIBUTING.md)** · Release history: **[CHANGELOG.md](https://github.com/a-kaushik2209/ARC/blob/master/CHANGELOG.md)**

---

## Citation

```bibtex
@article{kaushik2026arc,
  title   = {ARC: Autonomous Recovery Controller for Fault-Tolerant Neural Network Training},
  author  = {Kaushik, Aryan},
  year    = {2026},
  note    = {Maharaja Agrasen Institute of Technology, New Delhi}
}
```

---

<div align="center">

### Built by [ClearBound Labs](https://clearboundlabs.com/)

ClearBound Labs builds infrastructure that keeps intelligent systems reliable, observable, and secure.<br>
ARC is its flagship open-source project, released under the [AGPL-3.0 License](https://github.com/a-kaushik2209/ARC/blob/master/LICENSE).

</div>
