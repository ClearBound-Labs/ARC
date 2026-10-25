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

from typing import Dict, Any, Optional, Union
import time
import os
import warnings
import torch
import torch.nn as nn

from arc.config import Config
from arc.signals import (
    CompositeCollector,
    GradientCollector,
    ActivationCollector,
    WeightCollector,
    OptimizerCollector,
    LossCollector,
    CurvatureCollector,
)
from arc.features.buffer import SignalBuffer, EpochSnapshot
from arc.prediction.predictor import FailurePredictor, FailurePrediction
from arc.intervention.recommender import InterventionRecommender
from arc._logging import log as _log

class Arc:
    def __init__(
        self,
        config: Optional[Config] = None,
        model_path: Optional[str] = None,
        auto_intervene: bool = False,
        verbose: bool = True,
    ):
        self.config = config or Config()
        self.verbose = verbose
        self.auto_intervene = auto_intervene

        if self.config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device

        self._init_collectors()
        self._init_predictor(model_path)
        self.recommender = InterventionRecommender()

        self._model: Optional[nn.Module] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._is_attached = False
        self._current_epoch = 0
        self._step_count = 0
        self._epoch_start_time: Optional[float] = None
        self._training_start_time: Optional[float] = None
        self._overhead_total = 0.0
        self._training_total = 0.0

    def _init_collectors(self) -> None:
        self._loss_collector = LossCollector(self.config.signal)

        self.collectors = CompositeCollector([
            GradientCollector(self.config.signal),
            ActivationCollector(self.config.signal),
            WeightCollector(self.config.signal),
            OptimizerCollector(self.config.signal),
            self._loss_collector,
        ])

        if self.config.signal.compute_curvature_proxy:
            self.collectors.collectors.append(
                CurvatureCollector(self.config.signal)
            )

    def _init_predictor(self, model_path: Optional[str]) -> None:
        self.predictor = FailurePredictor(
            model_path=model_path,
            config=self.config,
            device=self.device,
        )

    def attach(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> 'Arc':
        if self._is_attached:
            self.detach()

        self._model = model
        self._optimizer = optimizer
        self.collectors.attach(model, optimizer)
        self._is_attached = True
        self._training_start_time = time.time()

        if self.verbose:
            _log(f"Arc attached to {model.__class__.__name__}")
            _log(f"Device: {self.device}")
            _log(f"Signal collectors: {len(self.collectors.collectors)}")

        return self

    def detach(self) -> None:
        if self._is_attached:
            self.collectors.detach()
            self._is_attached = False

            if self.verbose:
                overhead_pct = (
                    self._overhead_total / max(self._training_total, 1e-6) * 100
                )
                _log(f"Arc detached. Overhead: {overhead_pct:.2f}%")

    def on_batch_end(
        self,
        loss: float,
        step: Optional[int] = None
    ) -> None:
        if not self._is_attached:
            return

        start_time = time.time()

        if step is not None:
            self._step_count = step
        else:
            self._step_count += 1

        self._loss_collector.record_batch_loss(loss)

        if self.config.signal.compute_curvature_proxy:
            for collector in self.collectors.collectors:
                if isinstance(collector, CurvatureCollector):
                    collector.record_gradients()

        elapsed = time.time() - start_time
        self._overhead_total += elapsed

    def on_epoch_end(
        self,
        epoch: int,
        val_loss: Optional[float] = None,
        train_loss: Optional[float] = None,
    ) -> FailurePrediction:
        if not self._is_attached:
            raise RuntimeError("Arc not attached. Call attach() first.")

        start_time = time.time()
        self._current_epoch = epoch

        self.collectors.set_epoch(epoch)
        snapshot = self.collectors.collect()

        if train_loss is None:
            train_loss = self._loss_collector._train_loss_ema or 0.0
        self._loss_collector.record_epoch_loss(train_loss, val_loss)

        epoch_snapshot = EpochSnapshot(
            epoch=epoch,
            step=self._step_count,
            signals=snapshot.signals,
            timestamp=time.time(),
        )

        self.predictor.update(snapshot.signals, epoch)

        prediction = self.predictor.predict()

        if self._epoch_start_time is not None:
            self._training_total += time.time() - self._epoch_start_time
        self._epoch_start_time = time.time()

        elapsed = time.time() - start_time
        self._overhead_total += elapsed

        if self.verbose and prediction.risk_level in ['high', 'critical']:
            self._print_warning(prediction)

        if self.auto_intervene and prediction.risk_level == 'critical':
            self._apply_intervention(prediction)

        return prediction

    def _print_warning(self, prediction: FailurePrediction) -> None:
        mode, prob = prediction.get_highest_risk_mode()

        icon = "⚠️" if prediction.risk_level == 'high' else "🚨"
        _log(f"\n{icon} Arc Warning (Epoch {prediction.epoch}):")
        _log(f"   Risk: {prediction.risk_level.upper()} ({prob:.1%} {mode})")
        _log(f"   Recommendation: {prediction.recommendation.action}")
        _log(f"   Rationale: {prediction.recommendation.rationale}")

        if prediction.top_contributors:
            _log("   Top signals:")
            for c in prediction.top_contributors[:2]:
                _log(f"     - {c.signal_name}: {c.interpretation}")

    def _apply_intervention(self, prediction: FailurePrediction) -> None:
        action = prediction.recommendation.action
        params = prediction.recommendation.parameters

        if action == "reduce_learning_rate" and self._optimizer:
            for group in self._optimizer.param_groups:
                group['lr'] *= params.get('lr_factor', 0.5)

            if self.verbose:
                _log(f"Applied intervention: Learning rate reduced")

        elif action == "enable_gradient_clipping":
            if self.verbose:
                _log(f"Suggested intervention: Enable gradient clipping (max_norm={params.get('clip_max_norm', 1.0)})")

        elif action != "no_action":
            # Recommender can propose actions (add_regularization, early_stop,
            # manual_review, ...) this callback doesn't act on automatically.
            # Surface that instead of silently dropping the recommendation.
            warnings.warn(
                f"Arc recommended intervention '{action}' but no automatic "
                f"handler is implemented for it; no action was taken.",
                stacklevel=2,
            )

    def get_report(self) -> str:
        from arc.api.report import ReportGenerator
        generator = ReportGenerator()
        return generator.generate_text_report(
            self.predictor.predict(),
            include_history=True,
        )

    def save_state(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

        self.predictor.buffer.save(os.path.join(path, "buffer.json"))
        self.predictor.normalizer.save(os.path.join(path, "normalizer.json"))
        self.recommender.save(os.path.join(path, "recommender.json"))

    def load_state(self, path: str) -> None:
        from arc.features.buffer import SignalBuffer
        from arc.features.normalizer import OnlineNormalizer

        buffer_path = os.path.join(path, "buffer.json")
        if os.path.exists(buffer_path):
            self.predictor.buffer = SignalBuffer.load(buffer_path)

        norm_path = os.path.join(path, "normalizer.json")
        if os.path.exists(norm_path):
            self.predictor.normalizer = OnlineNormalizer.load(norm_path)

        rec_path = os.path.join(path, "recommender.json")
        if os.path.exists(rec_path):
            self.recommender = InterventionRecommender.load(rec_path)

    @property
    def overhead_percentage(self) -> float:
        if self._training_total < 1e-6:
            return 0.0
        return self._overhead_total / self._training_total * 100

    def reset(self) -> None:
        self.predictor.reset()
        self.collectors.reset()
        self._current_epoch = 0
        self._step_count = 0
        self._overhead_total = 0.0
        self._training_total = 0.0

    def __enter__(self) -> 'Arc':
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.detach()
        return False