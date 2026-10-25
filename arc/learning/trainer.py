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
from dataclasses import dataclass
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from arc.config import FailureMode, Config
from arc.learning.meta_model import TrainingDynamicsPredictor, ModelOutput
from arc.learning.simulator import SimulatedTrajectory
from arc.learning.labeler import TrajectoryLabeler, FailureLabel
from arc.features.extractor import FeatureExtractor
from arc.features.normalizer import OnlineNormalizer
from arc._logging import log as _log

class TrajectoryDataset(Dataset):

    def __init__(
        self,
        trajectories: List[SimulatedTrajectory],
        feature_extractor: FeatureExtractor,
        normalizer: OnlineNormalizer,
        max_length: int = 50,
        n_failure_modes: int = 5,
    ):
        self.trajectories = trajectories
        self.feature_extractor = feature_extractor
        self.normalizer = normalizer
        self.max_length = max_length
        self.n_failure_modes = n_failure_modes

        self._preprocess()

    def _preprocess(self):
        self.features = []
        self.labels = []
        self.ttf_targets = []
        self.masks = []

        for traj in self.trajectories:
            epoch_features = []
            for signal_snapshot in traj.signals:
                flat_signals = self._flatten_signals(signal_snapshot)
                features = self.feature_extractor.extract_features(
                    np.array([flat_signals.get(k, 0) for k in sorted(flat_signals.keys())]),
                    "signal"
                )
                epoch_features.append(list(flat_signals.values()))

            n_features = len(epoch_features[0]) if epoch_features else 1
            padded = np.zeros((self.max_length, n_features))
            mask = np.zeros(self.max_length)

            actual_len = min(len(epoch_features), self.max_length)
            for i in range(actual_len):
                padded[i] = epoch_features[i][:n_features] if len(epoch_features[i]) >= n_features else epoch_features[i] + [0] * (n_features - len(epoch_features[i]))
                mask[i] = 1.0

            self.features.append(padded)
            self.masks.append(mask)

            if traj.failure_mode is None:
                label = self.n_failure_modes
            else:
                label = list(FailureMode).index(traj.failure_mode)
            self.labels.append(label)

            if traj.failure_epoch is not None:
                ttf = max(0, traj.n_epochs - traj.failure_epoch)
            else:
                ttf = 100
            self.ttf_targets.append([ttf] * self.n_failure_modes)

        self.features = torch.tensor(np.array(self.features), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.ttf_targets = torch.tensor(self.ttf_targets, dtype=torch.float32)
        self.masks = torch.tensor(np.array(self.masks), dtype=torch.float32)

    def _flatten_signals(self, signals: Dict[str, Any]) -> Dict[str, float]:
        flat = {}

        def recurse(obj, prefix=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_prefix = f"{prefix}.{k}" if prefix else k
                    recurse(v, new_prefix)
            elif isinstance(obj, (int, float)) and np.isfinite(obj):
                flat[prefix] = float(obj)

        recurse(signals)
        return flat

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.features[idx],
            self.labels[idx],
            self.ttf_targets[idx],
            self.masks[idx],
        )

class FocalLoss(nn.Module):

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

@dataclass
class TrainingMetrics:
    epoch: int
    train_loss: float
    val_loss: Optional[float]
    train_accuracy: float
    val_accuracy: Optional[float]
    early_warning_precision: Optional[float] = None
    early_warning_recall: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "early_warning_precision": self.early_warning_precision,
            "early_warning_recall": self.early_warning_recall,
        }

class MetaModelTrainer:

    def __init__(
        self,
        model: TrainingDynamicsPredictor,
        config: Optional[Config] = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.config = config or Config()
        self.device = device

        self.classification_loss = FocalLoss(gamma=2.0)
        self.regression_loss = nn.GaussianNLLLoss()

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,
            weight_decay=0.01,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
        )

        self.metrics_history: List[TrainingMetrics] = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train(
        self,
        train_dataset: TrajectoryDataset,
        val_dataset: Optional[TrajectoryDataset] = None,
        n_epochs: int = 50,
        batch_size: int = 32,
        patience: int = 10,
        checkpoint_dir: Optional[str] = None,
    ) -> List[TrainingMetrics]:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        val_loader = None
        if val_dataset is not None:
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
            )

        for epoch in range(n_epochs):
            train_metrics = self._train_epoch(train_loader, epoch)

            val_metrics = None
            if val_loader is not None:
                val_metrics = self._validate(val_loader)

            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                val_loss=val_metrics["loss"] if val_metrics else None,
                train_accuracy=train_metrics["accuracy"],
                val_accuracy=val_metrics["accuracy"] if val_metrics else None,
            )
            self.metrics_history.append(metrics)

            self.scheduler.step()

            if checkpoint_dir is not None:
                self._save_checkpoint(checkpoint_dir, epoch, metrics)

            if val_metrics is not None:
                if val_metrics["loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["loss"]
                    self.patience_counter = 0
                    if checkpoint_dir is not None:
                        self._save_checkpoint(checkpoint_dir, epoch, metrics, best=True)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= patience:
                        _log(f"Early stopping at epoch {epoch}")
                        break

        return self.metrics_history

    def _train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        for features, labels, ttf_targets, masks in loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            ttf_targets = ttf_targets.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(features)

            cls_loss = self.classification_loss(
                output.failure_probs,
                labels
            )
            failure_mask = labels < (output.failure_probs.shape[1] - 1)
            if failure_mask.any():
                ttf_loss = self.regression_loss(
                    output.time_to_failure[failure_mask],
                    ttf_targets[failure_mask],
                    output.ttf_uncertainty[failure_mask] ** 2,
                )
            else:
                ttf_loss = torch.tensor(0.0, device=self.device)

            loss = cls_loss + 0.1 * ttf_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

            preds = output.failure_probs.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return {
            "loss": total_loss / len(loader),
            "accuracy": correct / total,
        }

    def _validate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels, ttf_targets, masks in loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                ttf_targets = ttf_targets.to(self.device)

                output = self.model(features)

                cls_loss = self.classification_loss(
                    output.failure_probs,
                    labels
                )

                total_loss += cls_loss.item()

                preds = output.failure_probs.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return {
            "loss": total_loss / len(loader),
            "accuracy": correct / total,
        }

    def _save_checkpoint(
        self,
        checkpoint_dir: str,
        epoch: int,
        metrics: TrainingMetrics,
        best: bool = False,
    ) -> None:
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics.to_dict(),
            "config": self.config.to_dict() if hasattr(self.config, 'to_dict') else {},
        }

        filename = "best_model.pt" if best else f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, os.path.join(checkpoint_dir, filename))

    def load_checkpoint(self, path: str, allow_unsafe: bool = False) -> None:
        from arc._serialization import safe_torch_load

        checkpoint = safe_torch_load(path, map_location=self.device, allow_unsafe=allow_unsafe)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def calibrate_temperature(
        self,
        val_dataset: TrajectoryDataset,
    ) -> float:
        loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        self.model.eval()
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for features, labels, _, _ in loader:
                features = features.to(self.device)
                output = self.model(features)
                logits = torch.log(output.failure_probs + 1e-10)
                all_logits.append(logits.cpu())
                all_labels.append(labels)

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        temperature = nn.Parameter(torch.ones(1) * 1.5)
        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

        def eval_temp():
            optimizer.zero_grad()
            scaled_logits = all_logits / temperature
            loss = F.cross_entropy(scaled_logits, all_labels)
            loss.backward()
            return loss

        optimizer.step(eval_temp)

        optimal_temp = temperature.item()
        self.model.temperature.data.fill_(optimal_temp)

        return optimal_temp