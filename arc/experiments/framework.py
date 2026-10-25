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

from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import time
import math
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

def safe_float(x: float, default: float = 0.0) -> float:
    if isinstance(x, torch.Tensor):
        x = x.item()
    if math.isnan(x) or math.isinf(x):
        return default
    return float(x)

class FailureType(Enum):
    NONE = auto()
    DIVERGENCE = auto()
    VANISHING_GRADIENT = auto()
    EXPLODING_GRADIENT = auto()
    MODE_COLLAPSE = auto()
    OVERFITTING = auto()

@dataclass
class ExperimentConfig:
    name: str
    failure_type: FailureType
    n_epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 0.01
    seed: int = 42
    model_type: str = "mlp"
    dataset: str = "synthetic"
    n_samples: int = 2000
    n_features: int = 20
    n_classes: int = 5
    failure_epoch: int = 10
    failure_severity: float = 1.0

@dataclass
class ExperimentResult:
    config: ExperimentConfig
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    gradient_norms: List[float] = field(default_factory=list)
    failure_detected: bool = False
    detection_epoch: Optional[int] = None
    detection_lead_time: Optional[int] = None
    false_positives: int = 0
    health_scores: List[float] = field(default_factory=list)
    lyapunov_exponents: List[float] = field(default_factory=list)
    surprises: List[float] = field(default_factory=list)
    actual_failure_epoch: Optional[int] = None
    final_accuracy: float = 0.0
    converged: bool = False
    total_time_seconds: float = 0.0

@dataclass
class AggregatedResults:
    n_experiments: int
    detection_accuracy: float
    precision: float
    recall: float
    f1_score: float
    mean_detection_lead_time: float
    std_detection_lead_time: float
    false_positive_rate: float
    p_value_vs_random: float
    confidence_interval_95: Tuple[float, float]

class FailureInducer:
    def __init__(self, failure_type: FailureType, severity: float = 1.0):
        self.failure_type = failure_type
        self.severity = severity
        self.active = False

    def activate(self) -> None:
        self.active = True

    def apply_to_model(self, model: nn.Module) -> None:
        if not self.active or self.failure_type == FailureType.NONE:
            return

        if self.failure_type == FailureType.VANISHING_GRADIENT:
            with torch.no_grad():
                for param in model.parameters():
                    param.data *= 0.1 * self.severity

        elif self.failure_type == FailureType.MODE_COLLAPSE:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if 'weight' in name and param.dim() >= 2:
                        param.data = param.data.mean() * torch.ones_like(param.data)

    def apply_to_gradients(self, model: nn.Module) -> None:
        if not self.active or self.failure_type == FailureType.NONE:
            return

        if self.failure_type == FailureType.VANISHING_GRADIENT:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad *= 0.001 * self.severity

        elif self.failure_type == FailureType.EXPLODING_GRADIENT:
            for param in model.parameters():
                if param.grad is not None:
                    param.grad *= 100 * self.severity

    def apply_to_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        if not self.active or self.failure_type == FailureType.NONE:
            return

        if self.failure_type == FailureType.DIVERGENCE:
            for group in optimizer.param_groups:
                group['lr'] *= 10 * self.severity

class MetricsCollector:
    def __init__(self):
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self._timestamps: List[float] = []

    def record(self, **kwargs) -> None:
        self._timestamps.append(time.time())
        for key, value in kwargs.items():
            self._metrics[key].append(float(value))

    def get(self, key: str) -> List[float]:
        return self._metrics.get(key, [])

    def get_stats(self, key: str) -> Dict[str, float]:
        values = self._metrics.get(key, [])
        if not values:
            return {"mean": 0, "std": 0, "min": 0, "max": 0}

        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
        }

    def detect_anomaly(self, key: str, threshold: float = 3.0) -> bool:
        values = self._metrics.get(key, [])
        if len(values) < 10:
            return False

        recent = values[-1]
        historical = values[:-1]

        mean = np.mean(historical)
        std = np.std(historical) + 1e-10

        z_score = abs(recent - mean) / std
        return z_score > threshold

class DetectionEvaluator:
    def __init__(self):
        self._predictions: List[Tuple[int, bool]] = []
        self._ground_truth: Optional[int] = None

    def record_prediction(self, epoch: int, predicted_failure: bool) -> None:
        self._predictions.append((epoch, predicted_failure))

    def set_ground_truth(self, failure_epoch: Optional[int]) -> None:
        self._ground_truth = failure_epoch

    def compute_metrics(self) -> Dict[str, float]:
        if not self._predictions:
            return {}

        first_detection = None
        for epoch, pred in self._predictions:
            if pred:
                first_detection = epoch
                break

        if self._ground_truth is None:
            if first_detection is not None:
                return {
                    "true_positive": 0,
                    "false_positive": 1,
                    "true_negative": 0,
                    "false_negative": 0,
                    "detection_lead_time": None,
                }
            else:
                return {
                    "true_positive": 0,
                    "false_positive": 0,
                    "true_negative": 1,
                    "false_negative": 0,
                    "detection_lead_time": None,
                }
        else:
            if first_detection is not None and first_detection <= self._ground_truth:
                lead_time = self._ground_truth - first_detection
                return {
                    "true_positive": 1,
                    "false_positive": 0,
                    "true_negative": 0,
                    "false_negative": 0,
                    "detection_lead_time": lead_time,
                }
            else:
                return {
                    "true_positive": 0,
                    "false_positive": 0,
                    "true_negative": 0,
                    "false_negative": 1,
                    "detection_lead_time": None,
                }

def create_model(model_type: str, n_features: int, n_classes: int) -> nn.Module:
    if model_type == "mlp":
        return nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )
    elif model_type == "deep_mlp":
        return nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )
    elif model_type == "cnn":
        return nn.Sequential(
            nn.Unflatten(1, (1, n_features)),
            nn.Conv1d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, n_classes),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def create_dataset(
    n_samples: int,
    n_features: int,
    n_classes: int,
    difficulty: str = "medium",
) -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.randn(n_samples, n_features)

    if difficulty == "easy":
        separation = 5.0
    elif difficulty == "medium":
        separation = 2.0
    else:
        separation = 0.5

    centers = torch.randn(n_classes, n_features) * separation
    y = torch.randint(0, n_classes, (n_samples,))

    for i in range(n_classes):
        mask = y == i
        X[mask] = X[mask] + centers[i]

    return X, y

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ExperimentRunner:

    def __init__(
        self,
        output_dir: str = "experiments",
        use_arc_network: bool = True,
        device: str = "auto",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.use_arc_network = use_arc_network

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        set_seed(config.seed)
        start_time = time.time()

        result = ExperimentResult(config=config)

        X, y = create_dataset(
            config.n_samples,
            config.n_features,
            config.n_classes,
        )

        split_idx = int(0.8 * len(X))
        train_dataset = TensorDataset(X[:split_idx], y[:split_idx])
        val_dataset = TensorDataset(X[split_idx:], y[split_idx:])

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

        model = create_model(config.model_type, config.n_features, config.n_classes)
        model = model.to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        inducer = FailureInducer(config.failure_type, config.failure_severity)

        evaluator = DetectionEvaluator()

        arc_network = None
        if self.use_arc_network:
            try:
                from arc.introspection import IntrospectiveArc
                arc_network = IntrospectiveArc(
                    model,
                    optimizer=optimizer,
                    enable_geometry=True,
                    enable_dynamics=True,
                    enable_topology=False,
                    enable_transport=True,
                    enable_self_model=True,
                )
            except ImportError:
                pass

        for epoch in range(config.n_epochs):
            if epoch == config.failure_epoch and config.failure_type != FailureType.NONE:
                inducer.activate()
                inducer.apply_to_model(model)
                result.actual_failure_epoch = epoch

            model.train()
            epoch_loss = 0
            n_batches = 0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()

                if arc_network:
                    output = arc_network(batch_x)
                else:
                    output = model(batch_x)

                loss = F.cross_entropy(output, batch_y)

                if torch.isnan(loss):
                    result.actual_failure_epoch = epoch
                    break

                loss.backward()

                inducer.apply_to_gradients(model)

                grad_norm = sum(
                    p.grad.norm().item() ** 2
                    for p in model.parameters()
                    if p.grad is not None
                ) ** 0.5
                result.gradient_norms.append(grad_norm)

                if arc_network:
                    state = arc_network.introspective_step(loss, batch_x, batch_y)
                    result.health_scores.append(state.overall_health)

                    if state.dynamics:
                        result.lyapunov_exponents.append(state.dynamics.lyapunov_exponent)

                    if state.self_knowledge:
                        result.surprises.append(state.self_knowledge.surprise)

                    predicted_failure = state.risk_score > 0.5
                    evaluator.record_prediction(epoch, predicted_failure)

                    if predicted_failure and not result.failure_detected:
                        result.failure_detected = True
                        result.detection_epoch = epoch

                inducer.apply_to_optimizer(optimizer)

                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if torch.isnan(loss):
                break

            result.train_losses.append(epoch_loss / max(n_batches, 1))

            model.eval()
            val_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    output = model(batch_x)
                    val_loss += F.cross_entropy(output, batch_y).item()

                    preds = output.argmax(dim=1)
                    correct += (preds == batch_y).sum().item()
                    total += len(batch_y)

            result.val_losses.append(val_loss / len(val_loader))
            result.final_accuracy = correct / total

        evaluator.set_ground_truth(result.actual_failure_epoch)

        if result.failure_detected and result.actual_failure_epoch:
            result.detection_lead_time = result.actual_failure_epoch - result.detection_epoch

        result.total_time_seconds = time.time() - start_time
        result.converged = result.actual_failure_epoch is None

        self._save_result(result)

        return result

    def _save_result(self, result: ExperimentResult) -> None:
        path = self.output_dir / f"{result.config.name}_{result.config.seed}.json"

        data = {
            "config": {
                "name": result.config.name,
                "failure_type": result.config.failure_type.name,
                "n_epochs": result.config.n_epochs,
                "seed": result.config.seed,
            },
            "metrics": {
                "failure_detected": result.failure_detected,
                "detection_epoch": result.detection_epoch,
                "detection_lead_time": result.detection_lead_time,
                "actual_failure_epoch": result.actual_failure_epoch,
                "final_accuracy": result.final_accuracy,
                "converged": result.converged,
                "total_time": result.total_time_seconds,
            },
            "trajectories": {
                "train_losses": result.train_losses,
                "val_losses": result.val_losses,
                "health_scores": result.health_scores[:100],
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def run_suite(
        self,
        n_trials: int = 10,
        failure_types: Optional[List[FailureType]] = None,
    ) -> Dict[str, AggregatedResults]:
        if failure_types is None:
            failure_types = list(FailureType)

        all_results = defaultdict(list)

        for failure_type in failure_types:
            print(f"\n{'='*50}")
            print(f"Testing: {failure_type.name}")
            print(f"{'='*50}")

            for trial in range(n_trials):
                config = ExperimentConfig(
                    name=f"{failure_type.name.lower()}_trial_{trial}",
                    failure_type=failure_type,
                    seed=42 + trial,
                )

                result = self.run(config)
                all_results[failure_type.name].append(result)

                status = "✓" if result.failure_detected == (failure_type != FailureType.NONE) else "✗"
                print(f"  Trial {trial+1}/{n_trials}: {status} "
                      f"(detected={result.failure_detected}, "
                      f"lead_time={result.detection_lead_time})")

        aggregated = {}

        for failure_name, results in all_results.items():
            aggregated[failure_name] = self._aggregate_results(results)

        return aggregated

    def _aggregate_results(self, results: List[ExperimentResult]) -> AggregatedResults:
        n = len(results)

        tp = sum(1 for r in results
                 if r.failure_detected and r.config.failure_type != FailureType.NONE)
        fp = sum(1 for r in results
                 if r.failure_detected and r.config.failure_type == FailureType.NONE)
        fn = sum(1 for r in results
                 if not r.failure_detected and r.config.failure_type != FailureType.NONE)
        tn = sum(1 for r in results
                 if not r.failure_detected and r.config.failure_type == FailureType.NONE)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        lead_times = [r.detection_lead_time for r in results if r.detection_lead_time is not None]
        mean_lead = np.mean(lead_times) if lead_times else 0
        std_lead = np.std(lead_times) if lead_times else 0

        fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

        observed_accuracy = (tp + tn) / n if n > 0 else 0
        p_value = self._binomial_p_value(tp + tn, n, 0.5)

        if n > 0:
            se = math.sqrt(observed_accuracy * (1 - observed_accuracy) / n)
            ci_low = max(0, observed_accuracy - 1.96 * se)
            ci_high = min(1, observed_accuracy + 1.96 * se)
        else:
            ci_low, ci_high = 0, 0

        return AggregatedResults(
            n_experiments=n,
            detection_accuracy=observed_accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            mean_detection_lead_time=mean_lead,
            std_detection_lead_time=std_lead,
            false_positive_rate=fp_rate,
            p_value_vs_random=p_value,
            confidence_interval_95=(ci_low, ci_high),
        )

    def _binomial_p_value(self, successes: int, n: int, p0: float) -> float:
        if n == 0:
            return 1.0

        observed_p = successes / n
        se = math.sqrt(p0 * (1 - p0) / n)

        if se < 1e-10:
            return 0.0 if observed_p > p0 else 1.0

        z = (observed_p - p0) / se
        p_value = 0.5 * (1 + math.erf(-z / math.sqrt(2)))

        return p_value