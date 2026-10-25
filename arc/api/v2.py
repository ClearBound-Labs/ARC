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

from typing import Dict, Any, Optional, Union, List, Callable
import torch
import torch.nn as nn
import warnings

from arc.config import Config, SignalConfig, FeatureConfig
from arc.api.callback import Arc
try:
    from arc.learning.ewc import ElasticWeightConsolidation
except ImportError:
    ElasticWeightConsolidation = None
from arc.uncertainty.conformal import ConformalPredictor, ConformalRegression
from arc.security.adversarial import AdversarialDetector, AdversarialTrainer
from arc._logging import log as _log

class ArcV2(Arc):
    def __init__(
        self,
        config: Optional[Config] = None,
        model_path: Optional[str] = None,
        auto_intervene: bool = False,
        verbose: bool = True,
        continual_learning: bool = False,
        uncertainty: str = "none",
        adversarial_detection: bool = False,
        spectral_analysis: bool = False,
    ):

        super().__init__(
            config=config,
            model_path=model_path,
            auto_intervene=auto_intervene,
            verbose=verbose,
        )

        self._enable_ewc = continual_learning
        self._uncertainty_method = uncertainty
        self._enable_adversarial = adversarial_detection
        self._enable_spectral = spectral_analysis

        self._ewc: Optional[ElasticWeightConsolidation] = None
        self._conformal: Optional[ConformalPredictor] = None
        self._conformal_reg: Optional[ConformalRegression] = None
        self._adversarial_detector: Optional[AdversarialDetector] = None
        self._spectral_analyzer = None

        self._current_task: Optional[str] = None
        self._task_count: int = 0

    @classmethod
    def auto(
        cls,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        task_type: str = "classification",
        safety_level: str = "standard",
    ) -> 'ArcV2':
        n_params = sum(p.numel() for p in model.parameters())
        n_layers = len(list(model.modules()))

        if n_params > 100_000_000:
            config = Config(
                signal=SignalConfig(activation_sample_ratio=0.01),
                feature=FeatureConfig(window_size=5),
            )
        elif n_params > 10_000_000:
            config = Config(
                signal=SignalConfig(activation_sample_ratio=0.1),
                feature=FeatureConfig(window_size=10),
            )
        else:
            config = Config()

        continual = safety_level in ["standard", "maximum"]
        uncertainty = "conformal" if safety_level in ["standard", "maximum"] else "none"
        adversarial = safety_level == "maximum"
        spectral = task_type == "pinn"

        instance = cls(
            config=config,
            auto_intervene=safety_level == "maximum",
            verbose=True,
            continual_learning=continual,
            uncertainty=uncertainty,
            adversarial_detection=adversarial,
            spectral_analysis=spectral,
        )

        instance.attach(model, optimizer)

        instance._task_type = task_type

        if instance.verbose:
            _log(f"   ARC v2.0 Auto-configured:")
            _log(f"   Model: {n_params:,} params, {n_layers} layers")
            _log(f"   Task: {task_type}")
            _log(f"   Safety: {safety_level}")
            _log(f"   Features: EWC={continual}, UQ={uncertainty}, Adv={adversarial}")

        return instance

    def step(
        self,
        loss: Union[float, torch.Tensor],
        x: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Record one training step for monitoring/prediction.

        This does not call `.backward()` or touch the optimizer, so it
        cannot apply the EWC penalty to your loss for you. `result["ewc_penalty"]`
        is informational only. If continual learning is enabled, add it to
        your loss yourself *before* `.backward()`:

            ewc_loss = arc.get_ewc_loss()
            total_loss = loss + ewc_loss
            total_loss.backward()
            arc.step(total_loss)
        """
        if isinstance(loss, torch.Tensor):
            loss = loss.item()

        self.on_batch_end(loss)

        result = {
            "loss": loss,
            "step": self._step_count,
            "risk_level": "low",
            "recommendation": None,
            "adversarial_alert": None,
            "ewc_penalty": 0.0,
        }

        if self._enable_ewc and self._ewc is not None and self._ewc.n_tasks > 0:
            ewc_penalty = self._ewc.compute_penalty()
            result["ewc_penalty"] = ewc_penalty.item()

        if self._enable_adversarial and x is not None and self._adversarial_detector is not None:
            try:
                alert = self._adversarial_detector.detect(x)
                result["adversarial_alert"] = {
                    "is_adversarial": alert.is_adversarial,
                    "confidence": alert.confidence,
                    "method": alert.detection_method,
                }
            except Exception as e:
                if self.verbose:
                    warnings.warn(f"Adversarial detection failed: {e}")

        return result

    def end_epoch(
        self,
        epoch: int,
        val_loss: Optional[float] = None,
        train_loss: Optional[float] = None,
    ) -> Dict[str, Any]:
        prediction = self.on_epoch_end(epoch, val_loss, train_loss)

        result = {
            "epoch": epoch,
            "prediction": prediction,
            "risk_level": prediction.risk_level,
            "failure_probability": prediction.overall_risk,
            "recommendation": prediction.recommendation.action if prediction.recommendation else None,
        }

        if self._enable_ewc and self._ewc is not None:
            result["n_consolidated_tasks"] = self._ewc.n_tasks

        return result

    def begin_task(self, task_id: str) -> None:
        if not self._enable_ewc:
            raise RuntimeError("Continual learning not enabled. Set continual_learning=True")

        if self._ewc is None and self._model is not None:
            self._ewc = ElasticWeightConsolidation(self._model)

        self._current_task = task_id
        self._task_count += 1

        if self.verbose:
            _log(f"Beginning task: {task_id} (Task #{self._task_count})")

    def consolidate_task(
        self,
        dataloader: torch.utils.data.DataLoader,
        task_id: Optional[str] = None,
    ) -> None:
        if self._ewc is None:
            raise RuntimeError("Must call begin_task() first")

        task_id = task_id or self._current_task or f"task_{self._task_count}"

        self._ewc.consolidate_task(task_id, dataloader)

        if self.verbose:
            _log(f"   Consolidated task: {task_id}")
            _log(f"   Protected weights from {self._ewc.n_tasks} task(s)")

    def get_ewc_loss(self) -> torch.Tensor:
        if self._ewc is None or self._ewc.n_tasks == 0:
            return torch.tensor(0.0)
        return self._ewc.compute_penalty()

    def calibrate_uncertainty(
        self,
        dataloader: torch.utils.data.DataLoader,
        task_type: str = "classification",
    ) -> float:
        if self._uncertainty_method == "none":
            raise RuntimeError("Uncertainty quantification not enabled")

        if self._model is None:
            raise RuntimeError("Must attach to model first")

        if task_type == "classification":
            self._conformal = ConformalPredictor(self._model)
            return self._conformal.calibrate(dataloader)
        else:
            self._conformal_reg = ConformalRegression(self._model)
            return self._conformal_reg.calibrate(dataloader)

    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
    ) -> Dict[str, Any]:
        if self._conformal is None and self._conformal_reg is None:
            raise RuntimeError("Must call calibrate_uncertainty() first")

        if self._conformal is not None:
            pred_set = self._conformal.predict(x)
            return {
                "prediction": pred_set.prediction,
                "confidence": pred_set.confidence,
                "set": pred_set.set_members,
                "set_size": pred_set.set_size,
                "coverage_target": pred_set.coverage_target,
            }
        else:
            interval = self._conformal_reg.predict(x)
            return {
                "prediction": interval.prediction,
                "lower": interval.lower,
                "upper": interval.upper,
                "confidence": interval.confidence,
                "interval_width": interval.interval_width,
            }
    def fit_adversarial_detector(
        self,
        dataloader: torch.utils.data.DataLoader,
        n_samples: int = 1000,
    ) -> None:
        if self._model is None:
            raise RuntimeError("Must attach to model first")

        self._adversarial_detector = AdversarialDetector(self._model)
        self._adversarial_detector.fit(dataloader, n_samples)

        if self.verbose:
            _log("Adversarial detector fitted")

    def check_adversarial(self, x: torch.Tensor) -> Dict[str, Any]:
        if self._adversarial_detector is None:
            raise RuntimeError("Must call fit_adversarial_detector() first")

        alert = self._adversarial_detector.detect(x)
        return {
            "is_adversarial": alert.is_adversarial,
            "confidence": alert.confidence,
            "method": alert.detection_method,
            "recommendation": alert.recommendation,
        }

    def health_report(self) -> str:
        base_report = self.get_report()

        lines = [base_report, "\n--- ARC v2.0 Status ---"]

        if self._enable_ewc:
            n_tasks = self._ewc.n_tasks if self._ewc else 0
            lines.append(f"Continual Learning: {n_tasks} task(s) consolidated")

        if self._uncertainty_method != "none":
            calibrated = self._conformal is not None or self._conformal_reg is not None
            lines.append(f"Uncertainty: {'Calibrated' if calibrated else 'Not calibrated'}")

        if self._enable_adversarial:
            fitted = self._adversarial_detector is not None
            lines.append(f"Adversarial Detection: {'Active' if fitted else 'Not fitted'}")

        return "\n".join(lines)

    def save(self, path: str) -> None:
        import os
        os.makedirs(path, exist_ok=True)

        self.save_state(path)

        if self._ewc is not None:
            import json
            # Persist the full EWC state (Fisher matrices + optimal params)
            # so continual-learning progress survives restarts.
            torch.save(
                self._ewc.state_dict(),
                os.path.join(path, "ewc_state.pt"),
            )
            with open(os.path.join(path, "ewc_meta.json"), 'w') as f:
                json.dump({"n_tasks": self._ewc.n_tasks}, f)

    def load(self, path: str, model: Optional[nn.Module] = None,
             allow_unsafe: bool = False) -> None:
        """Restore ArcV2 state including EWC continual-learning data.

        Parameters
        ----------
        path : str
            Directory previously passed to :meth:`save`.
        model : nn.Module, optional
            If provided, the model is re-attached before restoring EWC
            state (required if a fresh ArcV2 instance was created).
        allow_unsafe : bool
            Permit ``weights_only=False`` unpickling (can execute code).
            Only for files you produced yourself.
        """
        import os
        from arc._serialization import safe_torch_load

        self.load_state(path)

        if model is not None and self._model is None:
            self._model = model

        ewc_path = os.path.join(path, "ewc_state.pt")
        if self._enable_ewc and os.path.exists(ewc_path):
            if self._ewc is None and self._model is not None:
                self._ewc = ElasticWeightConsolidation(self._model)

            if self._ewc is not None:
                device = "cpu"
                if self._model is not None:
                    try:
                        device = str(next(self._model.parameters()).device)
                    except StopIteration:
                        pass

                state = safe_torch_load(ewc_path, map_location=device, allow_unsafe=allow_unsafe)

                self._ewc.load_state_dict(state, device=device)

                if self.verbose:
                    _log(f"   Restored EWC state: {self._ewc.n_tasks} task(s)")

    def __repr__(self) -> str:
        features = []
        if self._enable_ewc:
            features.append("EWC")
        if self._uncertainty_method != "none":
            features.append(f"UQ:{self._uncertainty_method}")
        if self._enable_adversarial:
            features.append("ADV")
        if self._enable_spectral:
            features.append("SPECTRAL")

        return f"ArcV2(features=[{', '.join(features)}])"

Arc2 = ArcV2