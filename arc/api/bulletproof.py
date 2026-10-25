import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Callable, Union, Tuple, List
from dataclasses import dataclass, field
import warnings
import gc
import traceback

from arc.intervention.oom_handler import (
    OOMRecoveryHandler, OOMConfig, OOMRecoveryFailed,
    get_memory_stats, estimate_model_memory
)
from arc.signals.silent_detector import (
    SilentCrashDetector, SilentDetectorConfig, SilentFailureType,
    ActivationMonitor
)
from arc.intervention.hardware_handler import (
    HardwareRecoveryHandler, HardwareConfig,
    get_best_device, get_device_info
)
from arc.intervention.rollback import WeightRollback, RollbackConfig, RollbackAction
from arc._logging import log as _log

@dataclass
class BulletproofConfig:
    enable_oom_recovery: bool = True
    enable_numeric_recovery: bool = True
    enable_silent_detection: bool = True
    enable_hardware_recovery: bool = True

    oom_config: OOMConfig = field(default_factory=OOMConfig)

    silent_config: SilentDetectorConfig = field(default_factory=SilentDetectorConfig)

    hardware_config: HardwareConfig = field(default_factory=HardwareConfig)

    rollback_config: RollbackConfig = field(default_factory=RollbackConfig)

    auto_detect_model_type: bool = True
    validation_interval: int = 100
    log_interval: int = 50
    verbose: bool = True

    @classmethod
    def for_llm(cls) -> 'BulletproofConfig':
        config = cls()
        config.oom_config.batch_reduction_factor = 0.5
        config.rollback_config.loss_explosion_threshold = 50.0
        config.rollback_config.gradient_explosion_threshold = 1e3
        config.silent_config.detect_accuracy_collapse = True
        return config

    @classmethod
    def for_gan(cls) -> 'BulletproofConfig':
        config = cls()
        config.silent_config.detect_mode_collapse = True
        config.silent_config.mode_collapse_threshold = 0.01
        config.rollback_config.checkpoint_frequency = 25
        return config

    @classmethod
    def for_vae(cls) -> 'BulletproofConfig':
        config = cls()
        config.silent_config.detect_posterior_collapse = True
        config.silent_config.posterior_collapse_threshold = 0.1
        return config

    @classmethod
    def minimal_overhead(cls) -> 'BulletproofConfig':
        config = cls()
        config.enable_silent_detection = False
        config.rollback_config.checkpoint_frequency = 100
        config.rollback_config.fast_grad_norm = True
        config.rollback_config.layer_sample_ratio = 0.1
        config.validation_interval = 500
        return config

@dataclass
class TrainStepResult:
    """Result of a single BulletproofTrainer.train_step() call.

    `loss` is None whenever the step didn't complete normally (numeric
    rollback, OOM exhaustion, or a recovered hardware error) — check
    `should_skip`/`failure_type` rather than inferring failure from a bare
    `None`.
    """
    step: int = 0
    loss: Optional[torch.Tensor] = None
    should_skip: bool = False
    rolled_back: bool = False
    oom_recovered: bool = False
    oom_exhausted: bool = False
    hardware_recovered: bool = False
    failure_type: Optional[str] = None
    message: str = ""

class BulletproofTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[BulletproofConfig] = None,
        scheduler: Optional[Any] = None,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.config = config or BulletproofConfig()

        if self.config.enable_oom_recovery:
            self.oom_handler = OOMRecoveryHandler(
                model, optimizer, self.config.oom_config
            )
        else:
            self.oom_handler = None

        if self.config.enable_numeric_recovery:
            self.rollback = WeightRollback(
                model, optimizer, self.config.rollback_config,
                verbose=self.config.verbose
            )
        else:
            self.rollback = None

        if self.config.enable_silent_detection:
            self.silent_detector = SilentCrashDetector(self.config.silent_config)
            self.activation_monitor = None
        else:
            self.silent_detector = None
            self.activation_monitor = None

        if self.config.enable_hardware_recovery:
            self.hardware_handler = HardwareRecoveryHandler(
                model, optimizer, self.config.hardware_config
            )
        else:
            self.hardware_handler = None

        self.step = 0
        self.epoch = 0
        self.total_recoveries = 0
        self.recovery_log: List[Dict[str, Any]] = []
        self._current_loss = None
        self._is_healthy = True

        self.device = next(model.parameters()).device

        if self.config.auto_detect_model_type:
            self._auto_configure()

        if self.config.verbose:
            self._print_init_info()

    def _auto_configure(self):
        model_str = str(type(self.model).__name__).lower()

        if any(t in model_str for t in ['gpt', 'llama', 'bert', 'transformer']):
            if self.config.verbose:
                _log("Detected Transformer/LLM architecture")
            if self.rollback:
                self.rollback.config.loss_explosion_threshold = 50.0

        elif 'gan' in model_str or 'discriminator' in model_str or 'generator' in model_str:
            if self.config.verbose:
                _log("   Detected GAN architecture")
            if self.silent_detector:
                self.silent_detector.config.detect_mode_collapse = True

        elif 'vae' in model_str or 'autoencoder' in model_str:
            if self.config.verbose:
                _log("   Detected VAE architecture")
            if self.silent_detector:
                self.silent_detector.config.detect_posterior_collapse = True

    def _print_init_info(self):
        _log("=" * 60)
        _log("BulletproofTrainer Initialized")
        _log("=" * 60)
        _log(f"   Device: {self.device}")
        _log(f"   OOM Recovery: {'✓' if self.oom_handler else '✗'}")
        _log(f"   Numeric Recovery: {'✓' if self.rollback else '✗'}")
        _log(f"   Silent Detection: {'✓' if self.silent_detector else '✗'}")
        _log(f"   Hardware Recovery: {'✓' if self.hardware_handler else '✗'}")

        mem = estimate_model_memory(self.model)
        _log(f"   Model Memory: {mem['total_gb']:.2f} GB estimated")
        _log("=" * 60)

    def train_step(
        self,
        batch: Any,
        forward_fn: Callable[[Any], torch.Tensor],
        backward_fn: Optional[Callable[[torch.Tensor], None]] = None,
        val_metric: Optional[float] = None,
        **kwargs
    ) -> TrainStepResult:
        self.step += 1

        try:
            self.optimizer.zero_grad()

            if self.oom_handler:
                loss = self.oom_handler.safe_forward(
                    lambda b: forward_fn(b),
                    batch
                )
            else:
                loss = forward_fn(batch)

            self._current_loss = loss.detach()

            if self.oom_handler:
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    self.oom_handler.safe_backward(loss)
            else:
                if backward_fn:
                    backward_fn(loss)
                elif self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

            # Checks run on this step's gradients, at their true scale.
            fp16_overflow = False
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                fp16_overflow = bool(torch.isfinite(loss.detach()).item()) and not all(
                    bool(torch.isfinite(p.grad).all())
                    for p in self.model.parameters() if p.grad is not None
                )

            if self.rollback and not fp16_overflow:
                action = self.rollback.step(loss)
                if action.should_skip:
                    self._log_recovery("numeric", action)
                    if self.scaler:
                        self.scaler.update()  # reset after unscale_; step is skipped
                    return TrainStepResult(
                        step=self.step,
                        should_skip=True,
                        rolled_back=action.rolled_back,
                        failure_type=action.failure_type or "numeric",
                        message=("numeric rollback triggered" if action.rolled_back
                                 else "numeric failure detected; step skipped"),
                    )

            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            elif self.oom_handler:
                self.oom_handler.safe_step()
            else:
                self.optimizer.step()

            if self.scheduler:
                self.scheduler.step()

            if self.silent_detector and self.step % self.config.validation_interval == 0:
                self._check_silent_failures(loss.item(), val_metric, **kwargs)

            if self.config.verbose and self.step % self.config.log_interval == 0:
                self._log_status(loss.item())

            return TrainStepResult(step=self.step, loss=loss)

        except OOMRecoveryFailed as e:
            self._log_recovery("oom_exhausted", {"error": str(e)})
            warnings.warn(f"OOM recovery exhausted: {e}")
            self._is_healthy = False
            return TrainStepResult(
                step=self.step,
                should_skip=True,
                oom_exhausted=True,
                failure_type="oom_exhausted",
                message=str(e),
            )

        except Exception as e:
            if self.hardware_handler:
                result = self.hardware_handler.try_recover(e)
                if result.success:
                    self._log_recovery("hardware", result)
                    return TrainStepResult(
                        step=self.step,
                        should_skip=True,
                        hardware_recovered=True,
                        failure_type="hardware",
                        message=str(e),
                    )

            raise

    def _check_silent_failures(
        self,
        loss: float,
        val_metric: Optional[float] = None,
        **kwargs
    ):
        if not self.silent_detector:
            return

        gradients = [p for p in self.model.parameters() if p.grad is not None]

        result = self.silent_detector.check(
            loss=loss,
            val_metric=val_metric,
            gradients=gradients if gradients else None,
            **kwargs
        )

        if result.detected:
            self._log_recovery("silent", result)

            if result.severity > 0.7 and self.rollback:
                if self.config.verbose:
                    _log(f"   Triggering rollback due to severe {result.failure_type.name}")
                self.rollback.restore_checkpoint()
                self.rollback.reduce_learning_rate()

    def _log_recovery(self, recovery_type: str, details: Any):
        self.total_recoveries += 1

        entry = {
            "step": self.step,
            "type": recovery_type,
            "details": str(details) if not isinstance(details, dict) else details,
        }
        self.recovery_log.append(entry)

        if self.config.verbose:
            _log(f"Recovery #{self.total_recoveries}: {recovery_type} at step {self.step}")

    def _log_status(self, loss: float):
        status = f"Step {self.step} | Loss: {loss:.4f}"

        if self.rollback:
            status += f" | Rollbacks: {self.rollback.state.rollback_count}"

        if self.oom_handler:
            stats = self.oom_handler.get_stats()
            if stats['oom_count'] > 0:
                status += f" | OOMs: {stats['oom_count']}"

        _log(status)

    def safe_step(self):
        return BulletproofContext(self)

    def after_backward(self):
        if self.rollback and self._current_loss is not None:
            action = self.rollback.step(self._current_loss)
            if action.rolled_back:
                self._log_recovery("numeric", action)

    def end_epoch(self, val_loss: Optional[float] = None, val_metric: Optional[float] = None):
        self.epoch += 1

        if self.rollback:
            self.rollback.end_epoch()

        if self.silent_detector and val_loss is not None:
            train_loss = self._current_loss.item() if self._current_loss is not None else None
            if train_loss is not None:
                result = self.silent_detector.check(
                    loss=train_loss,
                    val_loss=val_loss,
                    val_metric=val_metric,
                )
                if result.detected:
                    self._log_recovery("silent_epoch", result)

    def is_healthy(self) -> bool:
        return self._is_healthy

    def get_stats(self) -> Dict[str, Any]:
        stats = {
            "step": self.step,
            "epoch": self.epoch,
            "total_recoveries": self.total_recoveries,
            "is_healthy": self._is_healthy,
            "device": str(self.device),
        }

        if self.rollback:
            stats["rollback"] = self.rollback.get_stats()

        if self.oom_handler:
            stats["oom"] = self.oom_handler.get_stats()

        if self.silent_detector:
            stats["silent"] = self.silent_detector.get_stats()

        if self.hardware_handler:
            stats["hardware"] = self.hardware_handler.get_stats()

        return stats

    def reset(self):
        self.step = 0
        self.epoch = 0
        self.total_recoveries = 0
        self.recovery_log = []
        self._is_healthy = True

        if self.silent_detector:
            self.silent_detector.reset()

        if self.oom_handler:
            self.oom_handler.reset_strategies()

class BulletproofContext:

    def __init__(self, trainer: BulletproofTrainer):
        self.trainer = trainer

    def __enter__(self):
        self.trainer.optimizer.zero_grad()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            if isinstance(exc_val, RuntimeError):
                error_str = str(exc_val).lower()

                if 'out of memory' in error_str and self.trainer.oom_handler:
                    success, _ = self.trainer.oom_handler.try_recover(None)
                    if success:
                        return True

                if self.trainer.hardware_handler:
                    result = self.trainer.hardware_handler.try_recover(exc_val)
                    if result.success:
                        return True

        return False

def protect(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    **kwargs
) -> Tuple[nn.Module, BulletproofTrainer]:
    trainer = BulletproofTrainer(model, optimizer, **kwargs)
    return model, trainer

def protect_trainer(hf_trainer, **kwargs):
    from arc.api.integrations import HuggingFaceCallback

    callback = HuggingFaceCallback(**kwargs)
    hf_trainer.add_callback(callback)

    return hf_trainer

def ARCCallback(config: Optional[BulletproofConfig] = None):
    """Deprecated: returns `arc.api.integrations.LightningCallback`, which
    checks each step before the optimizer runs (this callback used to act
    only after the step had already been applied)."""
    from arc.api.integrations import LightningCallback

    warnings.warn("ARCCallback is deprecated; use arc.api.integrations.LightningCallback",
                  DeprecationWarning, stacklevel=2)
    verbose = config.verbose if config is not None else True
    return LightningCallback(verbose=verbose)