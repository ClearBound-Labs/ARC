<div align="center">

# ARC

### Autonomous Recovery Controller for Neural Network Training

_ARC monitors, predicts, and recovers from training failures automatically. One controller, one call per step._

<br>

[![PyPI](https://img.shields.io/pypi/v/arc-training?style=for-the-badge&logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/arc-training)
[![Downloads](https://img.shields.io/pepy/dt/arc-training?style=for-the-badge&label=Downloads)](https://pepy.tech/project/arc-training)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-green?style=for-the-badge)](https://www.gnu.org/licenses/agpl-3.0)
[![Docs](https://img.shields.io/badge/Docs-arc.clearboundlabs.com-orange?style=for-the-badge&logo=readthedocs&logoColor=white)](https://arc.clearboundlabs.com/)

<br>

**70/70** injected faults recovered (20/70 without ARC) · **0** rollbacks on clean runs · flags unfixable data faults instead of looping · rollback exercised at **2.65B** parameters (H200)

<br>

[Docs](https://arc.clearboundlabs.com/docs) •
[API](https://arc.clearboundlabs.com/api) •
[ARC Lens](https://arc.clearboundlabs.com/lens) •
[Examples](https://arc.clearboundlabs.com/examples) •
[Benchmarks](https://arc.clearboundlabs.com/benchmarks) •
[FAQ](https://arc.clearboundlabs.com/faq) •
[Contact](mailto:hello@clearboundlabs.com?subject=ARC)

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
pip install "arc-training[hf]"          # Hugging Face Trainer callback
pip install "arc-training[lightning]"   # Lightning callback
pip install "arc-training[dev]"
```

Requires Python 3.10+, PyTorch 2.4+, NumPy 1.23+, SciPy 1.9+.

</details>

| `StepAction` field | Meaning |
| :--- | :--- |
| `should_skip` | Skip `optimizer.step()` for this batch |
| `status` | `HealthStatus.HEALTHY`, `WARNING`, or `FAILURE` |
| `failure_type` | e.g. `nan_loss`, `loss_explosion`, `gradient_explosion`, `lyapunov_divergence`, `drift_detected` |
| `rolled_back` / `lr_reduced` | What ARC did in response |
| `unrecoverable` | The fault is outside the model. Stop and fix the input pipeline |

> **ARC never changes a healthy step.** It doesn't clip gradients unless you ask (`ControllerConfig(enable_baseline_clipping=True)`).

<details>
<summary><b>Mixed precision (fp16 + GradScaler), Hugging Face, Lightning</b></summary>

```python
controller = arc.Controller(model, optimizer, scaler=scaler)

scaler.scale(loss).backward()
action = controller.step(loss)      # ARC unscales; don't call scaler.unscale_()
if not action.should_skip:
    scaler.step(optimizer)
    scaler.update()
```

```python
from arc.api.integrations import HuggingFaceCallback, LightningCallback

Trainer(..., callbacks=[HuggingFaceCallback()])      # transformers
L.Trainer(..., callbacks=[LightningCallback()])      # lightning
```

Both callbacks check each step before the optimizer runs and skip a poisoned update.

</details>

Full guide: **[arc.clearboundlabs.com/docs](https://arc.clearboundlabs.com/docs)**

---

## The Problem

**Training neural networks is fragile.** A single NaN gradient at hour 47 of a 48-hour run can destroy days of compute. Engineers waste enormous time adding manual checkpointing and babysitting long runs.

- **NaN explosions:** the loss becomes NaN or Inf with no warning and corrupts the run.
- **Gradient collapse:** vanishing or exploding gradients stop learning without any alert.
- **Wasted compute:** every failure you can't recover from burns GPU-hours you already paid for.
- **Manual monitoring:** engineers watch runs overnight and write their own recovery scripts and checkpointing logic.

---

## How It Works

| | Stage | What happens |
| :-: | :--- | :--- |
| **01** | **Monitor** | Checks every step for NaN/Inf, loss explosions, gradient explosions, and weight corruption. Thresholds auto-calibrate to your model during the first 50 steps. |
| **02** | **Predict** | Lyapunov stability and trend-drift detectors flag rising instability before loss-based checks fire. Velocity is measured per unit of learning rate, so a warmup schedule isn't mistaken for divergence. |
| **03** | **Recover** | Restores model, optimizer, and RNG state from the last validated snapshot, then cuts the learning rate. A failure counts as recovered only after 10 healthy steps. If a fault keeps coming back from the data, ARC says so and marks the run unrecoverable instead of looping. |

```text
Controller.step(loss)
    |
    +-- warm-up: auto-calibrate thresholds (first 50 steps; NaN/Inf still caught)
    +-- NaN / Inf loss                    immediate
    +-- loss explosion                    absolute threshold + 10x spike vs recent mean
    +-- NaN / Inf weights, weight health  every 10 steps: sparsity, magnitude collapse
    +-- NaN / Inf gradients               true scale (GradScaler-aware)
    +-- gradient explosion                norm threshold (clipping only if enabled)
    +-- Lyapunov divergence               predictive, every 5 steps
    +-- trend drift                       OLS slope over a sliding window
    +-- DDP: every rank agrees            one all-reduce, so replicas never diverge
    |
    +--[FAILURE]--> restore snapshot (model + optimizer + RNG), skip the step
    |               reduce LR x0.5 (undone after 500 healthy steps)
    |               5 rollbacks within a few steps -> escape: oldest snapshot,
    |                   perturb weights, 2 extra LR cuts, skip 5 steps
    |               3 escapes with no progress -> unrecoverable (warns once)
    |
    +--[HEALTHY]--> snapshot copied async after optimizer.step(),
                    kept once the next step passes; adaptive frequency
```

---

## Capabilities

| Capability | Description |
| :--- | :--- |
| **Unified Controller** | One entry point, `arc.Controller`, replaces five legacy APIs. Rollback, learning-rate ratcheting and recovery, scheduler sync (including OneCycleLR), GradScaler support, and resumable `save_checkpoint()` / `load_checkpoint()`. |
| **Lyapunov Early Warning** | Online, learning-rate-normalized Lyapunov exponent estimation on a fixed parameter sample, with FFT oscillation detection. |
| **Trend Drift Detection** | Sliding-window regression on loss, gradient, and weight norms catches slow numerical drift that threshold checks miss. |
| **Auto-Calibration** | Learns loss and gradient thresholds from your model's first 50 steps, so you never have to guess what "too high" means. |
| **Honest Failure Handling** | When a fault lives in the data (a NaN row, `log(0)`, a zero-variance batch), ARC stops claiming recovery, raises one warning, and names the likely cause. |
| **Recovery Ledger and ROI** | A hash-chained log of every intervention (HMAC-keyed with `ledger_key`), exportable as JSON. Work lost per rollback is always recorded; GPU-hours, dollars, and carbon saved are measured against your durable-checkpoint interval (`baseline_checkpoint_interval`) and roll up per team and project. |
| **Checkpoint Integrity** | SHA-256 and HMAC signing on every storage strategy, verified on restore, plus a disaster-recovery drill that proves a checkpoint restores correctly. |
| **Policy Engine and Alerts** | Declarative remediation rules over failure predictions with a dry-run approval gate, and a severity-stacked alert router with de-duplication. |
| **RL and Generation Stability** | Detectors for KL blowup, entropy collapse, reward hacking, advantage explosion, mode collapse, and n-gram repetition in rollouts. |
| **Distributed Health Signals** | Straggler detection across ranks, cross-replica gradient checks for silent data corruption, and bit-reproducibility reports. |
| **Async Checkpointing** | Right after `optimizer.step()`, a background thread copies the snapshot to host memory on a side CUDA stream, overlapping the next forward and backward pass. Frequency adapts: more often when training is unstable, less often when it is calm. |
| **Research Toolkit** | PINN stabilizer, conformal uncertainty, adversarial detection, and EWC continual learning ship in the same package. |

Examples for CNNs, YOLO, Transformers, LLM fine-tuning, PINNs, RL, and more: **[arc.clearboundlabs.com/examples](https://arc.clearboundlabs.com/examples)**

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

All defaults are listed in the [configuration reference](https://arc.clearboundlabs.com/docs#config).

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

- **One canonical entry point:** `Controller.step(loss)` runs every detector in order and returns a `StepAction` with the health status, the failure type, and what ARC did about it.
- **Toggleable signals:** the Lyapunov, trend, gradient, and weight-health monitors switch on or off independently in `ControllerConfig`, without touching recovery logic.
- **Auditable by default:** every intervention lands in a hash-chained ledger you can export, verify, compare across runs, and roll up per team.
- **Minimal dependencies:** PyTorch, NumPy, and SciPy. Optional integrations for Lightning, Hugging Face, and Weights & Biases.

**Works with any architecture:** CNN / ResNet, Transformers, YOLO / ViT, PINNs, GPT / LLMs, Diffusion UNet, RNN / LSTM, GANs.

Validated on GPT-2 Medium, Large, and XL (1.5B), a 2.6B-parameter custom transformer, BERT, ResNet-18/50, VGG11, ViT-Base, DINOv2, YOLOv11, Llama-style models, and Stable Diffusion UNet.

---

## ARC Lens (VS Code)

[ARC Lens](https://arc.clearboundlabs.com/lens) is a self-contained VS Code extension for real-time PyTorch training visualization: live losses, learning rates, gradient norms, GPU memory, and layer effective rank, plus anomaly detection and a bundled offline recovery agent. No data leaves your machine.

---

## Benchmarks

Every number comes from a script in [`experiments/`](https://github.com/ClearBound-Labs/ARC/tree/master/experiments) that you can re-run.

### Fault-injection recovery (ARC 5.2.0)

[`experiments/recovery_benchmark.py`](https://github.com/ClearBound-Labs/ARC/blob/master/experiments/recovery_benchmark.py), 5 seeds, CPU. Each trial trains the same MLP on the same data twice, once unprotected and once with `arc.Controller`, and injects the same fault at step 60 of 150. Both runs are judged by the same checks at the end: no exception, every weight finite, and held-out loss no more than 2x its pre-fault value.

| Fault class | Trials | Healthy without ARC | Healthy with ARC | ARC rollbacks |
| :--- | :-: | :-: | :-: | :-: |
| No fault (control) | 5 | 5 | 5 | **0** |
| Transient: 14 types (NaN/Inf loss, x1e10 spikes, NaN weights or optimizer state, LR set to 1e6, Inf and adversarial gradients, repeated bursts) | 70 | 20 | **70** | 173 |
| Persistent bad data (NaN in every batch; x1e15 loss every 10th step) | 10 | 5 | **10** | 305 |

Under persistent NaN data, rollback can't fix the input. In all 5 trials ARC kept the model intact and flagged the run unrecoverable instead of looping. In 20 transient trials ARC rolled back for a fault that unprotected Adam would have absorbed (for example a x1e10 loss spike). That cost a few steps but never left a worse model. The control group is small; false-positive rates on full-size training recipes have not been measured yet.

### Results from ARC 5.1 scripts (pending re-measurement)

An October 2026 audit found that the 5.1 stress suites (Nightmare, Comprehensive, Silent Failure) never applied their crash checks to the ARC-protected run, so their "100% recovered" figures meant "didn't raise an exception". Those suites are superseded by the benchmark above. The tables below come from 5.1 code (some from legacy APIs). They are kept for reference until they're re-run on 5.2.

#### Baseline comparison

| Method | Detection | Recovery | False Pos. |
| :--- | :-: | :-: | :-: |
| No Protection | 52.0% | 0.0% | 0 |
| Gradient Clipping | 20.0% | 0.0% | 0 |
| Loss-Only Monitor | 80.0% | 80.0% | 0 |
| **Full ARC** | **100%** | **100%** | **0** |

#### Billion-scale rollback (NVIDIA H200)

A 40 to 50 step fine-tune with one NaN loss injected at step 30, using the legacy `WeightRollback`. The unprotected run is stopped at the NaN by the script, so this shows that rollback fits and runs at 2.65B parameters, not a recovery rate.

| Model | Params | No Protection | ARC | Rollbacks |
| :--- | :-: | :-: | :-: | :-: |
| GPT-2 Medium | 355M | Crashed at step 30 | Recovered | 1 |
| GPT-2 Large | 774M | Crashed at step 30 | Recovered | 1 |
| GPT-2 XL | 1.56B | Crashed at step 30 | Recovered | 1 |
| Custom Transformer | 2.65B | Crashed at step 30 | Recovered | 1 |

#### Real-dataset recovery

| Experiment | Clean Baseline | ARC-Protected (failure injected) |
| :--- | :-: | :-: |
| CIFAR-10, ResNet-18 (3 seeds) | 77.43% ± 0.12 | 77.42% ± 0.15 |
| GPT-2 (124M) on WikiText-2, perplexity | 15.8 | 16.4 |

#### Failure prediction (research)

Classifiers trained and tested on simulated training curves. No trained predictor ships with the package, so `FailurePredictor` reports `risk_level="unknown"` unless you load your own model.

| Classifier | Accuracy | Precision | Recall | F1 |
| :--- | :-: | :-: | :-: | :-: |
| Logistic Regression (12 features) | 95.5% | 100% | 91.0% | 0.953 |
| **MLP (12 features)** | **97.5% ± 2.2** | **100%** | **95%** | **0.974** |

### Overhead (GPU)

Measured with `experiments/controller_overhead.py`: GPT-2 (124M, random init), batch 4, sequence 128, AdamW, 400 steps in interleaved blocks of 50 (median step time).

| GPU | Baseline (ms/step) | With ARC (ms/step) | Overhead |
| :--- | :-: | :-: | :-: |
| RTX 3050 6GB Laptop | 255.86 | 261.49 | **2.2%** |

Reproduce: `python experiments/controller_overhead.py --steps 400 --batch 4 --seq 128`

Full results, protocols, and ablations: **[arc.clearboundlabs.com/benchmarks](https://arc.clearboundlabs.com/benchmarks)**

---

## Known Limitations

| Area | Status |
| :--- | :--- |
| GPU overhead | 2.2% on GPT-2 (124M) on an RTX 3050. Not yet profiled on A100/H100/H200 or at multi-billion scale |
| Scale | Rollback exercised at 2.65B parameters on one H200 (5.1, legacy engine). Snapshots live in host RAM: plan for `max_checkpoints` (3) copies of model plus optimizer state |
| Organic failures | Partially tested (8-scenario harness, realistic bugs on ResNet-50). Long-duration organic runs at scale are untested |
| Data faults | Detected but not repairable by rollback. ARC flags them unrecoverable and names the likely cause |
| Distributed training | `Controller` supports DDP: every rank agrees on each rollback (tested with 2 processes). FSDP/DTensor models are refused with a clear error. Multi-node, straggler, and SDC paths are not yet validated on a live cluster |
| Framework callbacks | The Hugging Face and Lightning hooks don't receive the loss, so they check gradients, weights and dynamics; with the trainers' default clipping, a finite exploding gradient is clipped before ARC sees it (NaN/Inf is still caught) |
| Multiplicative weight corruption | Scaling all weights (e.g. 1000x) without producing NaN or sparsity is not caught |
| Frameworks | PyTorch only |

---

## Contact

Questions, feedback, research collaborations, and support: email **[hello@clearboundlabs.com](mailto:hello@clearboundlabs.com?subject=ARC)** with the subject line **ARC**. Bugs and feature requests go to [GitHub Issues](https://github.com/ClearBound-Labs/ARC/issues).

## Contributing

1. Open an issue describing the feature, fix, or improvement.
2. Wait for the issue to be **assigned to you** before starting work. PRs without prior discussion or assignment may be closed.
3. Branch, make a focused change, and open a PR that references the issue.

Full workflow and standards: **[CONTRIBUTING.md](https://github.com/ClearBound-Labs/ARC/blob/master/CONTRIBUTING.md)** · Release history: **[CHANGELOG.md](https://github.com/ClearBound-Labs/ARC/blob/master/CHANGELOG.md)**

---

<div align="center">

### Built by [ClearBound Labs](https://clearboundlabs.com/)

ClearBound Labs builds infrastructure that keeps intelligent systems reliable, observable, and secure.<br>
ARC is its flagship open-source project, released under the [AGPL-3.0 License](https://github.com/ClearBound-Labs/ARC/blob/master/LICENSE).

</div>
