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

from typing import Dict, Any, Optional, List, Tuple, Set
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class ConformalPrediction:

    prediction_set: Set[int]
    set_size: int
    coverage_guarantee: float
    uncertainty_score: float
    class_scores: Dict[int, float]

class ConformalPredictor:

    def __init__(
        self,
        target_coverage: float = 0.9,
        score_type: str = "aps",
        random_state: Optional[int] = None,
    ):

        self.target_coverage = target_coverage
        self.score_type = score_type
        self.alpha = 1 - target_coverage

        self._threshold: Optional[float] = None
        self._calibration_scores: Optional[np.ndarray] = None
        self._n_cal: int = 0
        self._rng = np.random.default_rng(random_state)

    def calibrate(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> float:

        n_samples = len(labels)
        self._n_cal = n_samples

        scores = self._compute_scores(probs, labels)
        self._calibration_scores = np.sort(scores)

        adjusted_quantile = np.ceil((n_samples + 1) * (1 - self.alpha)) / n_samples
        adjusted_quantile = min(adjusted_quantile, 1.0)

        self._threshold = np.quantile(scores, adjusted_quantile)

        return self._threshold

    def _compute_scores(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:

        if self.score_type == "lac":

            scores = 1 - probs[np.arange(len(labels)), labels]

        elif self.score_type == "aps":

            n_samples = len(labels)
            row = np.arange(n_samples)

            sorted_idx = np.argsort(-probs, axis=1)
            sorted_probs = np.take_along_axis(probs, sorted_idx, axis=1)
            cumsum = np.cumsum(sorted_probs, axis=1)

            label_positions = np.argmax(sorted_idx == labels[:, None], axis=1)
            cumsum_at_label = cumsum[row, label_positions]
            prob_at_label = sorted_probs[row, label_positions]

            u = self._rng.uniform(0, 1, size=n_samples)
            scores = (cumsum_at_label - prob_at_label) + u * prob_at_label

        else:
            raise ValueError(f"Unknown score type: {self.score_type}")

        return scores

    def predict(
        self,
        probs: np.ndarray,
        return_all_scores: bool = False,
    ) -> ConformalPrediction:

        if self._threshold is None:
            raise RuntimeError("Must calibrate before prediction")

        n_classes = len(probs)

        if self.score_type == "lac":

            scores = {i: 1 - probs[i] for i in range(n_classes)}
            prediction_set = {i for i, s in scores.items() if s <= self._threshold}

        elif self.score_type == "aps":

            sorted_idx = np.argsort(-probs)
            prediction_set = set()
            scores = {}
            cumsum = 0.0

            for idx in sorted_idx:
                cumsum += probs[idx]
                scores[idx] = cumsum
                prediction_set.add(int(idx))

                if cumsum >= self._threshold:
                    break

        if not prediction_set:
            prediction_set = {int(np.argmax(probs))}

        return ConformalPrediction(
            prediction_set=prediction_set,
            set_size=len(prediction_set),
            coverage_guarantee=self.target_coverage,
            uncertainty_score=len(prediction_set) / n_classes,
            class_scores=scores if return_all_scores else {},
        )

    def predict_batch(
        self,
        probs: np.ndarray,
    ) -> List[ConformalPrediction]:

        return [self.predict(probs[i]) for i in range(len(probs))]

class ConformalTTFPredictor:

    def __init__(
        self,
        target_coverage: float = 0.9,
        symmetric: bool = False,
    ):

        self.target_coverage = target_coverage
        self.alpha = 1 - target_coverage
        self.symmetric = symmetric

        self._calibration_residuals: Optional[np.ndarray] = None
        self._q_lower: Optional[float] = None
        self._q_upper: Optional[float] = None

    def calibrate(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
    ) -> Tuple[float, float]:

        residuals = actuals - predictions
        n = len(residuals)

        if self.symmetric:
            abs_residuals = np.abs(residuals)
            q_level = min((1 - self.alpha) * (1 + 1/n), 1.0)
            q = np.quantile(abs_residuals, q_level)
            self._q_lower = -q
            self._q_upper = q
        else:

            q_alpha_lower = self.alpha / 2
            q_alpha_upper = 1 - self.alpha / 2

            q_level_lower = min(q_alpha_lower * (1 + 1/n), 1.0)
            q_level_upper = min(q_alpha_upper * (1 + 1/n), 1.0)

            self._q_lower = np.quantile(residuals, q_level_lower)
            self._q_upper = np.quantile(residuals, q_level_upper)

        self._calibration_residuals = np.sort(residuals)

        return self._q_lower, self._q_upper

    def predict_interval(
        self,
        prediction: float,
    ) -> Tuple[float, float]:

        if self._q_lower is None:
            raise RuntimeError("Must calibrate before prediction")

        lower = prediction + self._q_lower
        upper = prediction + self._q_upper

        lower = max(0, lower)
        upper = max(lower, upper)

        return lower, upper

    def predict_interval_batch(
        self,
        predictions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

        lower = predictions + self._q_lower
        upper = predictions + self._q_upper

        lower = np.maximum(0, lower)
        upper = np.maximum(lower, upper)

        return lower, upper

class ConformallyCalibratedPredictor:

    def __init__(
        self,
        n_failure_modes: int = 6,
        class_coverage: float = 0.9,
        ttf_coverage: float = 0.9,
    ):
        self.n_failure_modes = n_failure_modes

        self.class_predictor = ConformalPredictor(
            target_coverage=class_coverage,
            score_type="aps",
        )

        self.ttf_predictors = {
            i: ConformalTTFPredictor(target_coverage=ttf_coverage)
            for i in range(n_failure_modes)
        }

        self._is_calibrated = False

    def calibrate(
        self,
        class_probs: np.ndarray,
        class_labels: np.ndarray,
        ttf_predictions: Dict[int, np.ndarray],
        ttf_actuals: Dict[int, np.ndarray],
    ) -> None:

        self.class_predictor.calibrate(class_probs, class_labels)

        for mode, preds in ttf_predictions.items():
            if mode in ttf_actuals and len(preds) > 10:
                self.ttf_predictors[mode].calibrate(preds, ttf_actuals[mode])

        self._is_calibrated = True

    def predict(
        self,
        class_probs: np.ndarray,
        ttf_predictions: np.ndarray,
    ) -> Dict[str, Any]:

        if not self._is_calibrated:
            raise RuntimeError("Must calibrate before prediction")

        class_pred = self.class_predictor.predict(class_probs)

        ttf_intervals = {}
        for mode in class_pred.prediction_set:
            if mode < len(ttf_predictions):
                predictor = self.ttf_predictors.get(mode)
                if predictor and predictor._q_lower is not None:
                    lower, upper = predictor.predict_interval(ttf_predictions[mode])
                    ttf_intervals[mode] = (lower, upper)

        return {
            "failure_mode_set": class_pred.prediction_set,
            "set_size": class_pred.set_size,
            "class_coverage": class_pred.coverage_guarantee,
            "uncertainty": class_pred.uncertainty_score,
            "ttf_intervals": ttf_intervals,
        }
