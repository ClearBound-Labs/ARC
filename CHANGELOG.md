<div align="center">

# Changelog

All notable changes to ARC are documented here.

Based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) · [Semantic Versioning](https://semver.org/spec/v2.0.0.html)

</div>

---

## [5.2.0] - 2026-10-25

A correctness and honesty release from a line-by-line audit. Several defaults
change; read **Changed** before upgrading.

### Fixed

- **fp16 GradScaler runs were rolled back for no reason.** `Controller` and
  `AMPRollback` judged *scaled* gradients (x65,536) and treated the scaler's
  routine inf gradients as explosions. Pass `scaler=` to `Controller`; it
  unscales, checks true-scale gradients, and lets the scaler skip overflow steps.
  `AMPRollback` is now built on `Controller`
- **NaN gradients with a finite loss slipped through** (`nan > threshold` is
  False). They are now a `nan_gradients` failure in `Controller` and
  `UniversalDistributedRollback`
- **A failure inside the rollback cooldown let the poisoned step run.**
  `WeightRollback`, `GenericTrainingWrapper`, `BulletproofTrainer` and
  `UniversalDistributedRollback` now report `should_skip=True` for every
  detected failure, rollback or not
- **LR was reduced twice** when restoring a checkpoint taken after an earlier
  reduction (regression from 5.1.2). Each snapshot now records the reduction in
  force when it was taken
- **Multi-GPU (DDP) replicas could diverge**: a failure seen on one rank rolled
  back only that rank. Ranks now agree through one all-reduce per step, and the
  escape-mode weight noise is identical on every rank
- **Quantized checkpoints destroyed state.** fp16 flushed Adam's `exp_avg_sq` to
  zero; "int8" cast floats to int8 with no scale. Optimizer state is never
  quantized by default, int8 uses a per-tensor scale, and restores keep the live
  dtype. `UniversalDistributedRollback` no longer quantizes by default, and it no
  longer moves a CPU model's optimizer state to CUDA
- **Hardware "recovery" claimed fixes it couldn't make** (resetting a CUDA
  context, "continuing with available ranks", "switching to remote storage").
  Those paths now report failure and say to restart and resume. Disk-full
  cleanup deletes only ARC's own `checkpoint_*.pt` files
- **The failure predictor reported risk from untrained weights.** Without a
  trained `model_path`, or with mismatched features, it now returns
  `risk_level="unknown"`, and policies never fire on unknown predictions
- **Silent unsafe unpickling**: four loaders fell back to
  `torch.load(weights_only=False)`. There is no fallback any more; see SECURITY.md
- **Hugging Face and Lightning callbacks acted after the optimizer step** (HF only
  every `logging_steps`). Both now run `Controller` before the step and skip a
  poisoned update
- `NeuralCollapseCollector` couldn't be instantiated, and `collect()` raised;
  `SharpnessCollector` wiped the caller's gradients and forced train mode;
  `EnterpriseReport.to_html()` crashed on unknown savings
- Lyapunov sampling copied every parameter on each check (about 2 GB per 1B
  bf16 params); it now reads a fixed per-tensor sample. Gradient and weight
  checks use one host sync instead of one per tensor, and the NaN-weight check
  covers every tensor, not the first 20

### Changed

- **No gradient clipping by default** (`enable_baseline_clipping=False` in
  `ControllerConfig` and `SelfHealingConfig`); clipping changes training, so it is
  opt-in
- **Recovery is counted only after `recovery_confirm_steps` (10) healthy steps**,
  logged as a `recovery_confirmed` ledger event. A rollback alone is not a recovery
- **Ledger fields:** `steps_recovered` is now `steps_lost` (work a rollback threw
  away). `steps_saved` / `gpu_hours_saved` / `dollars_saved` are measured against
  `ControllerConfig.baseline_checkpoint_interval` and are `None` when it isn't set.
  `summary()` no longer assumes 30 minutes saved per failure
- LR reductions are undone one at a time after `lr_recovery_steps` (500) healthy
  steps; OneCycleLR / CyclicLR keep the reduction
- Snapshots are taken right after `optimizer.step()`, copied to ordinary host
  memory on a background thread (overlapping the next forward and backward pass),
  and become rollback targets only once the next step passes its checks. A failed
  copy is dropped with a warning and never stops training
- Output goes through the `arc` logger; `import arc` loads only the Controller
  path (about 13 s down to about 3 s here)
- Python 3.10+ and PyTorch 2.4+; `setup.py` removed (pyproject only)
- `SelfHealingArc` and `LiteArc` emit a `FutureWarning`; use `arc.Controller`

### Added

- `Controller(scaler=...)`, `Controller.step(None)` for framework hooks,
  `state_dict()` / `load_state_dict()`, `save_checkpoint()` / `load_checkpoint()`
  (atomic, `weights_only`-safe), `close()`
- `ControllerConfig`: `recovery_confirm_steps`, `lr_recovery_steps`,
  `baseline_checkpoint_interval`, `ledger_key` (HMAC), `external_grad_scaler`
- `arc._serialization.safe_torch_load`

---

## [5.1.2] - 2026-08-22

Two recovery bugs found by testing ARC against *organic* failures (divergence from a
too-high LR, `log(0)` in a custom loss, fp16 overflow, NaN rows in the dataset) rather
than failures injected by assigning NaN to the loss.

### Fixed

- **LR reductions did not accumulate across rollbacks.** `_restore_checkpoint()` calls
  `optimizer.load_state_dict()`, which restores `param_groups` including `lr`, resurrecting
  the pre-reduction learning rate. The reduction counter kept incrementing toward
  `max_lr_reductions`, after which `_reduce_lr()` returned early, so the LR ended up back at
  its original value while `StepAction.lr_reduced` reported a reduction. Four consecutive
  rollbacks previously produced `[5e-4, 5e-4, 5e-4, 5e-4]`; they now ratchet
  `5e-4 → 2.5e-4 → 1.25e-4 → 6.25e-5`. Fixed in all four rollback implementations:
  `core/controller.py`, `intervention/rollback.py`, `core/self_healing.py`,
  `distributed/universal.py`
- **No terminal state for failures ARC cannot fix.** `_persistent_failure_count` was
  incremented and never read, so a fault outside the model (a NaN row in the batch, an
  unguarded `log(0)`, a zero-variance batch) looped forever reporting "Recovered from
  nan_loss" while burning GPU hours. After `max_persistent_failures` escape cycles (default 3)
  ARC now sets `StepAction.unrecoverable`, raises one `RuntimeWarning`, and names the likely
  cause based on whether the weights are still finite. The flag is sticky, so a caller polling
  `action.unrecoverable` cannot miss it on a skipped step

### Added

- `StepAction.unrecoverable` and `Controller.unrecoverable`
- `ControllerConfig.max_persistent_failures` (default 3)
- `experiments/organic_failure_test.py`: 8 organic failure scenarios, each caused by real
  training conditions rather than direct NaN assignment, run with and without ARC
- `tests/test_persistent_failure.py`: 4 regression tests (169 tests total)

---

## [5.1.1] - 2026-08-22

### Fixed

- `examples/arc_demo.py` imported `Arc` from `arc.introspection`, which was renamed to `IntrospectiveArc` in 5.1.0, so the shipped example failed on import. Now imports `IntrospectiveArc as Arc`
- Same stale import fixed in `experiments/efficiency_benchmark.py` and `experiments/run_benchmark.py` (not shipped in the package)

---

## [5.1.0] - 2026-08-22

### Highlights

- **Enterprise reliability layer:** recovery ledger, ROI/chargeback reporting, tamper-evident audit log
- **Checkpoint integrity:** SHA-256 + HMAC signing now enforced across all storage strategies, verified on restore
- **Auto-remediation policy engine:** declarative rules over failure predictions, with a dry-run approval gate
- **RL and generation stability:** KL blowup, entropy collapse, reward hacking, and rollout mode-collapse detectors
- **Two real correctness fixes in checkpointing:** signing was a silent no-op for 4 of 5 strategies; incremental-delta restore dropped intermediate changes
- **Test suite grown from 89 to 165 tests**

### Added

- `arc/core/controller.py` adds `export_recovery_ledger()`, `reliability_roi_report()`, `verify_ledger_integrity()`: exportable recovery event ledger, GPU-hours/dollars/carbon-saved report, hash-chained tamper-evident intervention log
- `arc/api/alerting.py` adds `AlertRouter`/`Alert`/`Severity`: severity-stacked channel dispatch with a dedup window
- `arc/api/enterprise_report.py` adds `EnterpriseReport`: ledger + ROI + policy decisions + alerts as one JSON blob and one self-contained static HTML page with inline SVG
- `arc/checkpointing/dr_drill.py` adds `verify_recoverability()`: disaster-recovery self-test that restores a checkpoint into a throwaway model and compares output against the live model, without touching it
- `arc/distributed/straggler_detection.py` adds `StragglerDetector`: MAD-based per-rank step-time outlier detection with consecutive-flag debounce
- `arc/evaluation/run_comparison.py` adds `RunComparator`: diffs two exported ledgers, flags rollback-rate and stability regressions and new failure modes
- `arc/evaluation/reliability_metrics.py` adds `compute_reliability_metrics()`: recovery rate overall and by reason, rollback-depth distribution, mean steps/seconds between failures
- `arc/evaluation/chargeback.py` adds `aggregate_ledgers()`/`aggregate_ledger_files()`: GPU-hours and dollars saved rolled up per team/project
- `arc/prediction/policy.py` adds `RemediationPolicy`/`PolicyEngine`: auto-remediation rules over `FailurePrediction`, dry-run gate, notifier dispatch, exportable decision log, plus `from_dict()`/`from_config()` JSON policy loading
- `arc/security/sdc_detection.py` adds `CrossReplicaGradientCheck` (median-consensus outlier-rank flagging) and `DeterministicReplayVerifier`
- `arc/security/health_baseline.py` adds `HealthBaseline`: learns per-metric mean/std from prior runs' ledgers and flags runs outside the envelope
- `arc/security/reproducibility.py` adds `ReproducibilityTracker`: samples replay checks on a cadence, aggregates into a run-level bit-reproducibility report
- `arc/signals/rl_stability.py` adds `RLStabilityMonitor`: KL blowup, entropy collapse, reward hacking (reward up / objective down divergence), advantage explosion
- `arc/signals/rollout_health.py` adds `RolloutHealthMonitor`: mode collapse, n-gram repetition, batch-length format drift over generated sequences
- `arc/signals/stats_utils.py`: shared `ols_slope()` helper
- `ControllerConfig.team` / `.project` tagging, consumed by chargeback aggregation
- Optional checkpoint signing config: `sign_checkpoints` / `signing_key`
- Public `restore_checkpoint()` / `reduce_learning_rate()` wrappers on `WeightRollback`
- 76 new tests covering every module above

### Changed

- `arc/intervention/heuristics.py` split into an `arc/intervention/heuristics/` package (`diffusion.py`, `llm.py`, `detection.py`, `auto.py`); the import path `from arc.intervention.heuristics import X` is unchanged
- `arc/introspection/arc_network.py`: `Arc` renamed to `IntrospectiveArc` and `Action` to `IntrospectionAction`, resolving the name collision with `api.callback.Arc` and `intervention.actions.InterventionAction`
- `PersistentHomology` now warns when constructed with `max_dimension > 0` instead of silently returning zeroed betti_1/betti_2
- `GeodesicTracker` uses a bounded deque instead of growing without limit
- `prediction/conformal.py` APS scoring vectorized and made seedable

### Fixed

- **Checkpoint signing was a silent no-op for 4 of 5 storage strategies:** only `STREAMING_DISK` applied it, so signing did nothing whenever memory allowed another strategy (the common case). Integrity fields are now applied on all save paths and verified on restore
- **Incremental-delta restore reconstructed deltas against the original full checkpoint** instead of the immediately preceding one, silently dropping intermediate changes. Now chains correctly through `_last_checkpoint_id`
- NaN/Inf could leak into exported ledger JSON. Non-finite floats are now sanitized before logging and hashing, and export uses `allow_nan=False` so future gaps fail loudly
- `learning/labeler.py` tuple-indexing bug that corrupted failure-trajectory labels
- `physics/pinn_stabilizer.py` gradient-norm balancing overwrote per-term gradients instead of accumulating them
- `uncertainty/conformal.py` RAPS calibration/prediction mismatch
- `signals/neural_collapse.py` reservoir sampling now implements Algorithm R correctly
- 2D loss-landscape sampling in `introspection/topology.py` no longer collapses to 1D
- Malformed policy config raises `ValueError` naming the bad field and policy index instead of an opaque `KeyError`/`TypeError`
- Empty-tensor input to `DeterministicReplayVerifier.verify` raises a clear `ValueError` instead of a confusing PyTorch error
- `StragglerDetector` prunes ranks that stop reporting, preventing unbounded memory growth and stale state on reused rank ids
- `DiffusionHeuristics` implements `__enter__`/`__exit__`/`__del__` so forward hooks cannot leak

---

## [5.0.0] - 2026-06-22

### Highlights

- **Unified Controller API:** one entry point replaces 5 legacy APIs
- **Lyapunov stability monitoring:** predicts training crashes before they happen
- **Trend-based drift detection:** catches gradual numerical degradation (fixes paper Limitation #1)
- **Async checkpointing:** background checkpoint saving reduces training loop blocking
- **LR scheduler coordination:** ARC's LR reductions sync with scheduler.base_lrs
- **10/10 failure gauntlet:** NaN, Inf, weight corruption, gradient explosion, loss spike, persistent failure

### Added

- `arc/core/controller.py` adds `Controller` class: the ONE canonical entry point for ARC
- `arc/signals/lyapunov.py` adds `LyapunovMonitor`: online Lyapunov exponent estimation with FFT oscillation detection
- `arc/signals/trend.py` adds `TrendDetector`: OLS linear regression drift detection on loss, gradient, weight norms
- `ControllerConfig` dataclass: unified configuration for all signals and recovery parameters
- `StepAction` dataclass: rich return type with health score, Lyapunov exponent, training phase
- `HealthStatus` enum: HEALTHY / WARNING / FAILURE tri-state
- Thread-safe async checkpointing via `threading.Lock` + background `Thread`
- Parameter subsampling in Lyapunov monitor (10% sample, max 500K params) for O(1) cost on billion-param models
- Scheduler base_lrs coordination on LR reduction to prevent scheduler override
- `tests/test_controller_gauntlet.py`: 10-scenario comprehensive failure test

### Changed

- Version bumped from 2.0.0 to 3.0.0 in `__init__.py` (internal), published as 5.0.0
- `arc/__init__.py`: Controller, LyapunovMonitor, TrendDetector exported at top level
- `arc/signals/__init__.py`: Lyapunov and Trend modules registered
- `arc/core/__init__.py`: Controller, ControllerConfig, StepAction, HealthStatus exported

### Architecture

```
Controller.step(loss)
    |
    +-- NaN/Inf detection (immediate)
    +-- Loss explosion (threshold + spike factor + absolute)
    +-- Weight NaN check (sampled)
    +-- Weight health (sparsity + magnitude collapse)
    +-- Gradient explosion (norm threshold)
    +-- Lyapunov divergence (predictive, every N steps)
    +-- Trend drift (OLS slope on sliding window)
    |
    +--[FAILURE]--> _handle_failure()
    |                   +-- _restore_checkpoint()
    |                   +-- _reduce_lr()
    |                   +-- _track_consecutive_rollback()
    |                   +-- _escape_persistent_failure() (if stuck)
    |
    +--[HEALTHY]--> update history, adaptive checkpoint
```

---

## [4.1.0] - 2026-03-08

### Highlights

- **Every paper claim backed by experiment scripts:** zero fabricated data
- **97.5% prediction accuracy** (up from 86.5%) with 12-feature MLP classifier
- **Real overhead measurement**: <10% for models above 250K parameters
- **Honest ablation table** from real experiment data

### Added

- `experiments/baseline_comparison.py`: 4 methods × 5 failures × 5 seeds (backs Table 4)
- `experiments/prediction_200_v2.py`: 200 scenarios, 12 features, MLP + LogReg, 5-fold CV (backs §4.2)
- `experiments/ablation_experiment.py`: 6 configs × 7 failures × 5 seeds (backs Table 5)
- `experiments/overhead_measurement.py`: per-component timing, 3 model scales (backs Table 9)
- Baseline comparison table in paper (ARC 100% vs loss-only 80% vs grad-clip 20%)
- Stress test protocol description with explicit verification criteria

### Changed

- Paper prediction: 86.5% accuracy / F1=0.844 → **97.5% / F1=0.974** (12 features + MLP)
- Paper ablation table: replaced fabricated numbers with real experiment data
- Paper overhead table: replaced fabricated numbers with measured data (weight stats = 77% of overhead)
- Abstract/intro/conclusion: overhead claim corrected from "below 3%" to "below 10% for 250K+ params"
- Paper methodology: trimmed ~30% (cut textbook Fisher/Lyapunov/Conformal/EWC theory)
- README.md: complete rewrite with honest, verified numbers
- All 19 paper claims rewritten with honest language

### Fixed

- The conclusion cited a fabricated "42% single-signal" claim; it now matches the real ablation
- The stress test table lacked a protocol description; the full procedure is now included

---

## [4.0.0] - 2026-01-28

### Highlights

- **27% overhead** with ARC Lite configuration (production-recommended)
- **100% recovery** on all numeric failure types (NaN, Inf, explosion, corruption)
- **3/3 vs 1/3** head-to-head win against torchft on numeric stability
- **1.5B parameter** model validation (GPT-2 XL with quantized checkpoints)

### Added

- ARC Lite preset: 27% overhead with full recovery, recommended for production use
- OOM recovery across all training stages (forward, backward, optimizer step)
- Quantized FP16 checkpointing: 50% memory reduction with no recovery degradation
- Silent failure detection suite (accuracy collapse, mode collapse, dead neurons)
- Head-to-head comparison framework against torchft
- Statistical validation harness (10 seeds × 4 architectures, p < 0.001)
- Mamba-based failure predictor with Evidential Deep Learning uncertainty
- Signal pipeline: 16+ real-time training health signals
- Fisher Information-based parameter importance tracking
- Multi-scale temporal fusion (5/20/50 step analysis windows)
- REST/gRPC monitoring API with Prometheus metrics

### Changed

- Restructured public API: `Arc` class replaces `WeightRollback` as primary entry point
- `ArcCallback` replaces `ARCCallback` for Lightning integration
- Overhead section rewritten to lead with validated ARC Lite numbers
- Recovery claims scoped to test suite of induced failures
- Failure type taxonomy: 6 validated + 6 detection-only

### Fixed

- Memory leaks during checkpoint restoration on repeated rollbacks
- RNG state preservation across rollback boundaries
- Optimizer momentum buffer handling during recovery
- Gradient accumulation state corruption after rollback

---

## [3.0.0] - 2025-12-01

### Added

- Multi-architecture support (CNN, ViT, Transformer, Diffusion UNet)
- Scaling validated to 387M parameters
- Basic OOM detection and graceful degradation

### Changed

- Checkpoint system restructured for incremental delta support
- Loss explosion detection threshold made adaptive

---

## [2.0.0] - 2025-09-01

### Added

- Initial public release
- NaN/Inf loss detection with automatic rollback
- Weight rollback with configurable LR reduction

---

## [1.0.0] - 2025-06-01

### Added

- Proof-of-concept implementation
- Single-model, single-failure-type validation
