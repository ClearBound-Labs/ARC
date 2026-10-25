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

from typing import Dict, List, Optional, Tuple, Union, Callable
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from collections import deque

@dataclass
class PredictionSet:

    prediction: int
    confidence: float
    set_members: List[int]
    set_size: int
    coverage_target: float
    nonconformity_score: float

@dataclass
class RegressionInterval:

    prediction: float
    lower: float
    upper: float
    confidence: float
    interval_width: float

class ConformalPredictor:

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.1,
        score_function: str = "aps",
        adaptive: bool = True,
        raps_k_reg: int = 2,
        raps_lambda_reg: float = 0.001,
    ):

        self.model = model
        self.alpha = alpha
        self.score_function = score_function
        self.adaptive = adaptive
        self.raps_k_reg = raps_k_reg
        self.raps_lambda_reg = raps_lambda_reg

        self._calibration_scores: List[float] = []
        self._quantile: Optional[float] = None
        self._is_calibrated = False

        self._online_scores: deque = deque(maxlen=1000)

        self._va_calibrator: Optional[VennAbersCalibrator] = None

    def calibrate(
        self,
        dataloader: torch.utils.data.DataLoader,
        use_venn_abers: bool = True,
    ) -> float:

        self.model.eval()
        scores = []
        all_probs = []
        all_labels = []

        device = next(self.model.parameters()).device

        with torch.no_grad():
            for batch in dataloader:
                x, y = batch[0].to(device), batch[1].to(device)
                logits = self.model(x)
                probs = torch.softmax(logits, dim=-1)

                batch_scores = self._compute_scores(probs, y)
                scores.extend(batch_scores.cpu().numpy().tolist())

                all_probs.append(probs.cpu())
                all_labels.append(y.cpu())

        self._calibration_scores = scores

        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self._quantile = np.quantile(scores, min(q_level, 1.0))

        if use_venn_abers:
            all_probs = torch.cat(all_probs, dim=0).numpy()
            all_labels = torch.cat(all_labels, dim=0).numpy()
            self._va_calibrator = VennAbersCalibrator()
            self._va_calibrator.fit(all_probs, all_labels)

        self._is_calibrated = True
        self.model.train()

        return self._quantile

    def _compute_scores(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:

        if self.score_function == "lac":

            return 1 - probs[torch.arange(len(labels)), labels]

        elif self.score_function == "aps":

            sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)

            label_positions = (sorted_idx == labels.unsqueeze(-1)).nonzero()[:, 1]
            scores = cumsum[torch.arange(len(labels)), label_positions]

            scores = scores - torch.rand_like(scores.float()) * sorted_probs[
                torch.arange(len(labels)), label_positions
            ]

            return scores

        elif self.score_function == "raps":

            sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
            cumsum = sorted_probs.cumsum(dim=-1)

            positions = torch.arange(probs.size(-1), device=probs.device)
            regularization = self.raps_lambda_reg * torch.clamp(positions - self.raps_k_reg, min=0)

            cumsum_reg = cumsum + regularization

            label_positions = (sorted_idx == labels.unsqueeze(-1)).nonzero()[:, 1]
            scores = cumsum_reg[torch.arange(len(labels)), label_positions]

            return scores

        else:
            raise ValueError(f"Unknown score function: {self.score_function}")

    def predict(
        self,
        x: torch.Tensor,
        return_calibrated_probs: bool = True,
    ) -> Union[PredictionSet, List[PredictionSet]]:

        if not self._is_calibrated:
            raise RuntimeError("Must call calibrate() before predict()")

        single = (x.dim() == 1) or (x.dim() >= 2 and x.size(0) == 1)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        self.model.eval()
        device = next(self.model.parameters()).device
        x = x.to(device)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)

            if return_calibrated_probs and self._va_calibrator is not None:
                cal_probs = self._va_calibrator.predict_proba(probs.cpu().numpy())
                probs = torch.tensor(cal_probs, device=device)

        results = []
        n_classes = probs.size(-1)

        for i in range(probs.size(0)):
            sample_probs = probs[i]

            sorted_probs, sorted_idx = sample_probs.sort(descending=True)
            cumsum = sorted_probs.cumsum(dim=0)

            if self.score_function == "lac":
                set_members = (sample_probs >= 1 - self._quantile).nonzero().squeeze(-1).tolist()
            else:
                if self.score_function == "raps":
                    positions = torch.arange(n_classes, device=sample_probs.device)
                    regularization = self.raps_lambda_reg * torch.clamp(positions - self.raps_k_reg, min=0)
                    set_cumsum = cumsum + regularization
                else:
                    set_cumsum = cumsum

                n_include = (set_cumsum <= self._quantile).sum().item() + 1
                n_include = min(n_include, n_classes)
                set_members = sorted_idx[:n_include].tolist()

            if not set_members:
                set_members = [sorted_idx[0].item()]

            prediction = sorted_idx[0].item()
            confidence = sample_probs[prediction].item()

            results.append(PredictionSet(
                prediction=prediction,
                confidence=confidence,
                set_members=set_members,
                set_size=len(set_members),
                coverage_target=1 - self.alpha,
                nonconformity_score=cumsum[0].item(),
            ))

        self.model.train()
        return results[0] if single else results

    def update_online(self, x: torch.Tensor, y: torch.Tensor) -> None:

        self.model.eval()
        device = next(self.model.parameters()).device

        with torch.no_grad():
            logits = self.model(x.to(device))
            probs = torch.softmax(logits, dim=-1)
            scores = self._compute_scores(probs, y.to(device))

        for score in scores.cpu().numpy().tolist():
            self._online_scores.append(score)

        if len(self._online_scores) > 10:
            n = len(self._online_scores)
            q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            self._quantile = np.quantile(list(self._online_scores), min(q_level, 1.0))

        self.model.train()

    def get_coverage(self, dataloader: torch.utils.data.DataLoader) -> float:

        covered = 0
        total = 0

        device = next(self.model.parameters()).device

        for batch in dataloader:
            x, y = batch[0].to(device), batch[1]
            pred_sets = self.predict(x)

            if not isinstance(pred_sets, list):
                pred_sets = [pred_sets]

            for pred_set, label in zip(pred_sets, y.numpy()):
                if label in pred_set.set_members:
                    covered += 1
                total += 1

        return covered / total if total > 0 else 0.0

    def get_average_set_size(self, dataloader: torch.utils.data.DataLoader) -> float:

        total_size = 0
        count = 0

        device = next(self.model.parameters()).device

        for batch in dataloader:
            x = batch[0].to(device)
            pred_sets = self.predict(x)

            if not isinstance(pred_sets, list):
                pred_sets = [pred_sets]

            for pred_set in pred_sets:
                total_size += pred_set.set_size
                count += 1

        return total_size / count if count > 0 else 0.0

class VennAbersCalibrator:

    def __init__(self):
        self._isotonic_regressors = {}
        self._is_fitted = False

    def fit(self, probs: np.ndarray, labels: np.ndarray) -> None:

        from sklearn.isotonic import IsotonicRegression

        n_classes = probs.shape[1]

        for c in range(n_classes):
            binary_labels = (labels == c).astype(float)

            ir = IsotonicRegression(out_of_bounds='clip')
            ir.fit(probs[:, c], binary_labels)
            self._isotonic_regressors[c] = ir

        self._is_fitted = True

    def predict_proba(self, probs: np.ndarray) -> np.ndarray:

        if not self._is_fitted:
            return probs

        calibrated = np.zeros_like(probs)

        for c, ir in self._isotonic_regressors.items():
            calibrated[:, c] = ir.predict(probs[:, c])

        calibrated = calibrated / (calibrated.sum(axis=1, keepdims=True) + 1e-10)

        return calibrated

class ConformalRegression:

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.1,
        method: str = "cqr",
    ):

        self.model = model
        self.alpha = alpha
        self.method = method

        self._quantile: Optional[float] = None
        self._is_calibrated = False

    def calibrate(self, dataloader: torch.utils.data.DataLoader) -> float:

        self.model.eval()
        residuals = []

        device = next(self.model.parameters()).device

        with torch.no_grad():
            for batch in dataloader:
                x, y = batch[0].to(device), batch[1].to(device)
                pred = self.model(x).squeeze()

                res = torch.abs(y - pred)
                residuals.extend(res.cpu().numpy().tolist())

        n = len(residuals)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self._quantile = np.quantile(residuals, min(q_level, 1.0))

        self._is_calibrated = True
        self.model.train()

        return self._quantile

    def predict(self, x: torch.Tensor) -> Union[RegressionInterval, List[RegressionInterval]]:

        if not self._is_calibrated:
            raise RuntimeError("Must call calibrate() before predict()")

        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)

        self.model.eval()
        device = next(self.model.parameters()).device

        with torch.no_grad():
            pred = self.model(x.to(device)).squeeze()

        results = []
        preds = pred.cpu().numpy() if pred.dim() > 0 else [pred.cpu().item()]

        for p in preds:
            results.append(RegressionInterval(
                prediction=float(p),
                lower=float(p - self._quantile),
                upper=float(p + self._quantile),
                confidence=1 - self.alpha,
                interval_width=2 * self._quantile,
            ))

        self.model.train()
        return results[0] if single else results