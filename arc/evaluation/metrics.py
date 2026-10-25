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

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
from collections import defaultdict

from arc.config import FailureMode

@dataclass
class EvaluationMetrics:
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0

    per_mode_precision: Dict[str, float] = field(default_factory=dict)
    per_mode_recall: Dict[str, float] = field(default_factory=dict)
    per_mode_f1: Dict[str, float] = field(default_factory=dict)

    precision_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    f1_at_k: Dict[int, float] = field(default_factory=dict)

    mean_detection_time: float = 0.0
    median_detection_time: float = 0.0
    std_detection_time: float = 0.0
    detection_times: List[float] = field(default_factory=list)

    false_alarm_rate: float = 0.0
    false_alarms_per_epoch: float = 0.0

    ece: float = 0.0

    overhead_percent: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "per_mode_precision": self.per_mode_precision,
            "per_mode_recall": self.per_mode_recall,
            "per_mode_f1": self.per_mode_f1,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "f1_at_k": self.f1_at_k,
            "mean_detection_time": self.mean_detection_time,
            "median_detection_time": self.median_detection_time,
            "std_detection_time": self.std_detection_time,
            "false_alarm_rate": self.false_alarm_rate,
            "false_alarms_per_epoch": self.false_alarms_per_epoch,
            "ece": self.ece,
            "overhead_percent": self.overhead_percent,
        }

    def summary(self) -> str:
        lines = [
            "=== Evaluation Metrics ===",
            f"Overall: P={self.precision:.3f}, R={self.recall:.3f}, F1={self.f1:.3f}",
            f"Detection: μ={self.mean_detection_time:.1f}ep, med={self.median_detection_time:.1f}ep",
            f"False Alarm Rate: {self.false_alarm_rate:.3f}",
            f"ECE: {self.ece:.4f}",
            f"Overhead: {self.overhead_percent:.2f}%",
        ]

        if self.precision_at_k:
            lines.append("Early Warning (P@k epochs lead):")
            for k in sorted(self.precision_at_k.keys()):
                lines.append(f"  k={k}: P={self.precision_at_k[k]:.3f}, R={self.recall_at_k.get(k, 0):.3f}")

        return "\n".join(lines)

class MetricsCalculator:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

        self._predictions: List[Dict[str, float]] = []
        self._ground_truth: List[Optional[str]] = []
        self._failure_epochs: List[Optional[int]] = []
        self._prediction_epochs: List[int] = []

        self._overhead_times: List[float] = []
        self._training_times: List[float] = []

    def record_prediction(
        self,
        prediction_probs: Dict[str, float],
        epoch: int,
        ground_truth_mode: Optional[str] = None,
        actual_failure_epoch: Optional[int] = None,
    ) -> None:
        self._predictions.append(prediction_probs)
        self._ground_truth.append(ground_truth_mode)
        self._failure_epochs.append(actual_failure_epoch)
        self._prediction_epochs.append(epoch)

    def record_overhead(self, overhead_time: float, training_time: float) -> None:
        self._overhead_times.append(overhead_time)
        self._training_times.append(training_time)

    def compute_metrics(
        self,
        lead_times: List[int] = [3, 5, 10],
    ) -> EvaluationMetrics:
        metrics = EvaluationMetrics()

        if not self._predictions:
            return metrics

        n_samples = len(self._predictions)
        n_modes = len(FailureMode)
        mode_names = [m.name for m in FailureMode]

        y_pred_binary = []
        y_true_binary = []

        for pred, gt in zip(self._predictions, self._ground_truth):
            max_prob = max(pred.values()) if pred else 0
            y_pred_binary.append(1 if max_prob >= self.threshold else 0)
            y_true_binary.append(1 if gt is not None else 0)

        y_pred = np.array(y_pred_binary)
        y_true = np.array(y_true_binary)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        metrics.precision = tp / max(tp + fp, 1)
        metrics.recall = tp / max(tp + fn, 1)
        metrics.f1 = 2 * metrics.precision * metrics.recall / max(metrics.precision + metrics.recall, 1e-10)
        metrics.accuracy = (tp + tn) / max(n_samples, 1)

        metrics.false_alarm_rate = fp / max(fp + tn, 1)

        for mode in FailureMode:
            mode_name = mode.name
            mode_pred = [1 if p.get(mode_name, 0) >= self.threshold else 0 for p in self._predictions]
            mode_true = [1 if gt == mode_name else 0 for gt in self._ground_truth]

            mode_tp = sum(p and t for p, t in zip(mode_pred, mode_true))
            mode_fp = sum(p and not t for p, t in zip(mode_pred, mode_true))
            mode_fn = sum(not p and t for p, t in zip(mode_pred, mode_true))

            mode_precision = mode_tp / max(mode_tp + mode_fp, 1)
            mode_recall = mode_tp / max(mode_tp + mode_fn, 1)
            mode_f1 = 2 * mode_precision * mode_recall / max(mode_precision + mode_recall, 1e-10)

            metrics.per_mode_precision[mode_name] = mode_precision
            metrics.per_mode_recall[mode_name] = mode_recall
            metrics.per_mode_f1[mode_name] = mode_f1

        for k in lead_times:
            metrics.precision_at_k[k], metrics.recall_at_k[k] = self._compute_early_warning_metrics(k)
            if metrics.precision_at_k[k] + metrics.recall_at_k[k] > 0:
                metrics.f1_at_k[k] = 2 * metrics.precision_at_k[k] * metrics.recall_at_k[k] / (
                    metrics.precision_at_k[k] + metrics.recall_at_k[k]
                )

        detection_times = self._compute_detection_times()
        if detection_times:
            metrics.detection_times = detection_times
            metrics.mean_detection_time = np.mean(detection_times)
            metrics.median_detection_time = np.median(detection_times)
            metrics.std_detection_time = np.std(detection_times)

        metrics.ece = self._compute_ece()

        if self._overhead_times and self._training_times:
            total_overhead = sum(self._overhead_times)
            total_training = sum(self._training_times)
            metrics.overhead_percent = total_overhead / max(total_training, 1e-10) * 100

        return metrics

    def _compute_early_warning_metrics(self, k: int) -> Tuple[float, float]:
        warnings_given = 0
        true_positives = 0
        false_positives = 0
        actual_failures = 0

        for i, (pred, gt, fail_epoch, pred_epoch) in enumerate(zip(
            self._predictions, self._ground_truth, self._failure_epochs, self._prediction_epochs
        )):
            is_failure = gt is not None and fail_epoch is not None

            if is_failure:
                actual_failures += 1
                lead_time = fail_epoch - pred_epoch

                max_prob = max(pred.values()) if pred else 0
                predicted_failure = max_prob >= self.threshold

                if predicted_failure and lead_time >= k:
                    true_positives += 1
                    warnings_given += 1
            else:
                max_prob = max(pred.values()) if pred else 0
                if max_prob >= self.threshold:
                    false_positives += 1
                    warnings_given += 1

        precision = true_positives / max(warnings_given, 1)
        recall = true_positives / max(actual_failures, 1)

        return precision, recall

    def _compute_detection_times(self) -> List[float]:
        detection_times = []

        current_warnings = []
        for i, (pred, gt, fail_epoch, pred_epoch) in enumerate(zip(
            self._predictions, self._ground_truth, self._failure_epochs, self._prediction_epochs
        )):
            max_prob = max(pred.values()) if pred else 0
            is_warning = max_prob >= self.threshold

            if is_warning:
                current_warnings.append(pred_epoch)

            if gt is not None and fail_epoch is not None and fail_epoch == pred_epoch:
                if current_warnings:
                    first_warning = min(current_warnings)
                    detection_time = fail_epoch - first_warning
                    detection_times.append(detection_time)
                current_warnings = []

        return detection_times

    def _compute_ece(self, n_bins: int = 10) -> float:
        if not self._predictions:
            return 0.0

        confidences = []
        accuracies = []

        for pred, gt in zip(self._predictions, self._ground_truth):
            if not pred:
                continue

            # ECE measures calibration between confidence and correctness;
            # without a ground-truth label there is no correctness signal
            # to compare against, so these samples are excluded rather than
            # substituted with an unrelated confidence-threshold proxy
            # (which measured something else entirely and silently mixed
            # two different notions of "correct" into the same bins).
            if gt is None:
                continue

            max_mode = max(pred, key=pred.get)
            max_prob = pred[max_mode]
            is_correct = gt == max_mode

            confidences.append(max_prob)
            accuracies.append(1 if is_correct else 0)

        if not confidences:
            return 0.0

        confidences = np.array(confidences)
        accuracies = np.array(accuracies)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            prop_in_bin = in_bin.mean()

            if in_bin.any():
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                ece += prop_in_bin * abs(avg_accuracy - avg_confidence)

        return ece

    def reset(self) -> None:
        self._predictions.clear()
        self._ground_truth.clear()
        self._failure_epochs.clear()
        self._prediction_epochs.clear()
        self._overhead_times.clear()
        self._training_times.clear()

def compute_all_metrics(
    predictions: List[Dict[str, float]],
    ground_truth: List[Optional[str]],
    threshold: float = 0.5,
) -> EvaluationMetrics:
    calc = MetricsCalculator(threshold=threshold)

    for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
        calc.record_prediction(pred, epoch=i, ground_truth_mode=gt)

    return calc.compute_metrics()