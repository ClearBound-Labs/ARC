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

from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import numpy as np

from arc.config import FailureMode, FailureThresholds

@dataclass
class FailureLabel:
    mode: Optional[FailureMode]
    failure_epoch: Optional[int]
    confidence: float
    evidence: Dict[str, Any]

    @property
    def is_failure(self) -> bool:
        return self.mode is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode.name if self.mode else None,
            "failure_epoch": self.failure_epoch,
            "confidence": self.confidence,
            "evidence": self.evidence,
        }

class TrajectoryLabeler:
    def __init__(self, thresholds: Optional[FailureThresholds] = None):
        self.thresholds = thresholds or FailureThresholds()

    def label_trajectory(
        self,
        signals: List[Dict[str, Any]]
    ) -> FailureLabel:
        if not signals:
            return FailureLabel(
                mode=None,
                failure_epoch=None,
                confidence=0.0,
                evidence={"error": "empty trajectory"}
            )

        candidates = []

        div_result = self._check_divergence(signals)
        if div_result[0]:
            candidates.append((FailureMode.DIVERGENCE, *div_result[2:]))

        van_result = self._check_vanishing_gradients(signals)
        if van_result[0]:
            candidates.append((FailureMode.VANISHING_GRADIENTS, *van_result[2:]))

        exp_result = self._check_exploding_gradients(signals)
        if exp_result[0]:
            candidates.append((FailureMode.EXPLODING_GRADIENTS, *exp_result[2:]))

        col_result = self._check_representation_collapse(signals)
        if col_result[0]:
            candidates.append((FailureMode.REPRESENTATION_COLLAPSE, *col_result[2:]))

        ovf_result = self._check_overfitting(signals)
        if ovf_result[0]:
            candidates.append((FailureMode.SEVERE_OVERFITTING, *ovf_result[2:]))

        if not candidates:
            return FailureLabel(
                mode=None,
                failure_epoch=None,
                confidence=1.0 - self._compute_risk_score(signals),
                evidence={"status": "no_failure_detected"}
            )

        # each candidate is now (mode, failure_epoch, confidence, evidence)
        candidates.sort(key=lambda x: -x[2])

        best = candidates[0]
        return FailureLabel(
            mode=best[0],
            failure_epoch=best[1],
            confidence=best[2],
            evidence=best[3],
        )

    def _check_divergence(
        self,
        signals: List[Dict[str, Any]]
    ) -> Tuple[bool, bool, Optional[int], float, Dict]:
        initial_loss = self._get_signal(signals[0], "loss.epoch.train_loss")

        if initial_loss is None:
            return (False, False, None, 0.0, {})

        for epoch, snapshot in enumerate(signals):
            loss = self._get_signal(snapshot, "loss.epoch.train_loss")

            if loss is None:
                continue

            if self.thresholds.loss_nan_detection:
                if np.isnan(loss) or np.isinf(loss):
                    return (True, True, epoch, 1.0, {
                        "reason": "loss_nan_inf",
                        "value": str(loss),
                    })

            if loss > initial_loss * self.thresholds.loss_explosion_factor:
                confidence = min(1.0, loss / (initial_loss * self.thresholds.loss_explosion_factor * 2))
                return (True, True, epoch, confidence, {
                    "reason": "loss_explosion",
                    "initial_loss": initial_loss,
                    "current_loss": loss,
                    "ratio": loss / initial_loss,
                })

        return (False, False, None, 0.0, {})

    def _check_vanishing_gradients(
        self,
        signals: List[Dict[str, Any]]
    ) -> Tuple[bool, bool, Optional[int], float, Dict]:
        consecutive_low = 0
        first_low_epoch = None

        for epoch, snapshot in enumerate(signals):
            grad_norm = self._get_signal(snapshot, "gradient.global.total_grad_norm_l2")

            if grad_norm is None:
                continue

            if grad_norm < self.thresholds.vanishing_grad_threshold:
                if first_low_epoch is None:
                    first_low_epoch = epoch
                consecutive_low += 1

                if consecutive_low >= self.thresholds.vanishing_grad_epochs:
                    confidence = min(1.0, consecutive_low / (self.thresholds.vanishing_grad_epochs * 2))
                    return (True, True, first_low_epoch, confidence, {
                        "reason": "gradient_norm_below_threshold",
                        "grad_norm": grad_norm,
                        "threshold": self.thresholds.vanishing_grad_threshold,
                        "consecutive_epochs": consecutive_low,
                    })
            else:
                consecutive_low = 0
                first_low_epoch = None

        return (False, False, None, 0.0, {})

    def _check_exploding_gradients(
        self,
        signals: List[Dict[str, Any]]
    ) -> Tuple[bool, bool, Optional[int], float, Dict]:
        consecutive_high = 0
        first_high_epoch = None

        for epoch, snapshot in enumerate(signals):
            grad_norm = self._get_signal(snapshot, "gradient.global.total_grad_norm_l2")

            if grad_norm is None:
                continue

            if grad_norm > self.thresholds.exploding_grad_threshold:
                if first_high_epoch is None:
                    first_high_epoch = epoch
                consecutive_high += 1

                if consecutive_high >= self.thresholds.exploding_grad_epochs:
                    confidence = min(1.0, grad_norm / (self.thresholds.exploding_grad_threshold * 10))
                    return (True, True, first_high_epoch, confidence, {
                        "reason": "gradient_norm_above_threshold",
                        "grad_norm": grad_norm,
                        "threshold": self.thresholds.exploding_grad_threshold,
                        "consecutive_epochs": consecutive_high,
                    })
            else:
                consecutive_high = 0
                first_high_epoch = None

        return (False, False, None, 0.0, {})

    def _check_representation_collapse(
        self,
        signals: List[Dict[str, Any]]
    ) -> Tuple[bool, bool, Optional[int], float, Dict]:
        high_similarity_epochs = 0
        first_collapse_epoch = None

        for epoch, snapshot in enumerate(signals):
            similarity = self._get_signal(snapshot, "activation.global.mean_similarity")
            effective_rank = self._get_signal(snapshot, "weight.global.mean_effective_rank")

            is_collapsed = False
            evidence = {}

            if similarity is not None and similarity > self.thresholds.activation_similarity_threshold:
                is_collapsed = True
                evidence["similarity"] = similarity

            if effective_rank is not None and epoch > 0:
                initial_rank = self._get_signal(signals[0], "weight.global.mean_effective_rank")
                if initial_rank and effective_rank < initial_rank * self.thresholds.effective_rank_collapse_ratio:
                    is_collapsed = True
                    evidence["effective_rank"] = effective_rank
                    evidence["initial_rank"] = initial_rank

            if is_collapsed:
                if first_collapse_epoch is None:
                    first_collapse_epoch = epoch
                high_similarity_epochs += 1

                if high_similarity_epochs >= 5:
                    confidence = min(1.0, high_similarity_epochs / 10)
                    evidence["consecutive_epochs"] = high_similarity_epochs
                    return (True, True, first_collapse_epoch, confidence, evidence)
            else:
                high_similarity_epochs = 0
                first_collapse_epoch = None

        return (False, False, None, 0.0, {})

    def _check_overfitting(
        self,
        signals: List[Dict[str, Any]]
    ) -> Tuple[bool, bool, Optional[int], float, Dict]:
        for epoch, snapshot in enumerate(signals):
            train_loss = self._get_signal(snapshot, "loss.epoch.train_loss")
            val_loss = self._get_signal(snapshot, "loss.epoch.val_loss")
            epochs_since = self._get_signal(snapshot, "loss.epoch.epochs_since_improvement")

            if train_loss is None or val_loss is None:
                continue

            if val_loss > 1e-8 and train_loss / val_loss < (1 - self.thresholds.overfit_gap_threshold):
                if epochs_since is not None and epochs_since >= self.thresholds.overfit_val_plateau_epochs:
                    gap = (val_loss - train_loss) / val_loss
                    confidence = min(1.0, gap / 0.5)
                    return (True, True, epoch - int(epochs_since), confidence, {
                        "reason": "train_val_gap",
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "gap_ratio": gap,
                        "epochs_since_improvement": epochs_since,
                    })

        return (False, False, None, 0.0, {})

    def _compute_risk_score(self, signals: List[Dict[str, Any]]) -> float:
        risks = []

        if not signals:
            return 0.5

        last = signals[-1]

        train_loss = self._get_signal(last, "loss.epoch.train_loss")
        if signals[0]:
            initial_loss = self._get_signal(signals[0], "loss.epoch.train_loss")
            if initial_loss and train_loss and initial_loss > 0:
                ratio = train_loss / initial_loss
                if ratio > 1:
                    risks.append(min(1.0, (ratio - 1) / 5))

        grad_norm = self._get_signal(last, "gradient.global.total_grad_norm_l2")
        if grad_norm is not None:
            if grad_norm < 1e-5:
                risks.append(0.8)
            elif grad_norm > 1000:
                risks.append(0.9)

        train_val_gap = self._get_signal(last, "loss.epoch.train_val_gap")
        if train_val_gap is not None and train_val_gap > 0.2:
            risks.append(min(1.0, train_val_gap))

        return np.mean(risks) if risks else 0.1

    def _get_signal(self, snapshot: Dict[str, Any], path: str) -> Optional[float]:
        value = snapshot
        for key in path.split("."):
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

        if isinstance(value, (int, float)) and np.isfinite(value):
            return float(value)
        return None

    def label_batch(
        self,
        trajectories: List[List[Dict[str, Any]]]
    ) -> List[FailureLabel]:
        return [self.label_trajectory(t) for t in trajectories]