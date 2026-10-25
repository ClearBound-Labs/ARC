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
import torch
import torch.nn as nn

from arc.config import FailureMode, Config
from arc.learning.meta_model import TrainingDynamicsPredictor
from arc.features.buffer import SignalBuffer, EpochSnapshot
from arc.features.extractor import FeatureExtractor
from arc.features.normalizer import OnlineNormalizer

@dataclass
class SignalContribution:

    signal_name: str
    contribution_score: float
    current_value: float
    trend: str
    interpretation: str

@dataclass
class InterventionRecommendation:

    action: str
    confidence: float
    rationale: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    caveats: List[str] = field(default_factory=list)

@dataclass
class FailurePrediction:

    epoch: int
    failure_probabilities: Dict[FailureMode, float]
    confidence_intervals: Dict[FailureMode, Tuple[float, float]]
    time_to_failure: Dict[FailureMode, float]
    ttf_uncertainty: Dict[FailureMode, float]
    top_contributors: List[SignalContribution]
    recommendation: InterventionRecommendation
    overall_risk: float
    risk_level: str

    def to_dict(self) -> Dict[str, Any]:

        return {
            "epoch": self.epoch,
            "failure_probabilities": {
                m.name: p for m, p in self.failure_probabilities.items()
            },
            "confidence_intervals": {
                m.name: list(ci) for m, ci in self.confidence_intervals.items()
            },
            "time_to_failure": {
                m.name: ttf for m, ttf in self.time_to_failure.items()
            },
            "ttf_uncertainty": {
                m.name: u for m, u in self.ttf_uncertainty.items()
            },
            "top_contributors": [
                {
                    "signal_name": c.signal_name,
                    "contribution_score": c.contribution_score,
                    "current_value": c.current_value,
                    "trend": c.trend,
                    "interpretation": c.interpretation,
                }
                for c in self.top_contributors
            ],
            "recommendation": {
                "action": self.recommendation.action,
                "confidence": self.recommendation.confidence,
                "rationale": self.recommendation.rationale,
                "parameters": self.recommendation.parameters,
                "caveats": self.recommendation.caveats,
            },
            "overall_risk": self.overall_risk,
            "risk_level": self.risk_level,
        }

    def get_highest_risk_mode(self) -> Tuple[FailureMode, float]:

        max_mode = max(
            self.failure_probabilities.items(),
            key=lambda x: x[1]
        )
        return max_mode

class FailurePredictor:

    def __init__(
        self,
        model: Optional[TrainingDynamicsPredictor] = None,
        model_path: Optional[str] = None,
        config: Optional[Config] = None,
        device: str = "cpu",
    ):

        self.config = config or Config()
        self.device = device

        self.buffer = SignalBuffer(max_size=self.config.feature.window_size * 2)
        self.feature_extractor = FeatureExtractor(
            window_size=self.config.feature.window_size,
            compute_trend=self.config.feature.compute_trend,
            compute_spectral=self.config.feature.compute_spectral,
            compute_correlations=self.config.feature.compute_correlations,
        )
        self.normalizer = OnlineNormalizer()

        if model is not None:
            self.model = model.to(device)
        elif model_path is not None:
            self.model = self._load_model(model_path)
        else:

            self.model = None

        if self.model is not None:
             self.model.eval()

        self.attribution_engine = None

        self._current_epoch = 0

    def _load_model(self, path: str, allow_unsafe: bool = False) -> TrainingDynamicsPredictor:
        from arc._serialization import safe_torch_load

        checkpoint = safe_torch_load(path, map_location=self.device, allow_unsafe=allow_unsafe)

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        n_features = state_dict["input_proj.0.weight"].shape[1]

        model = TrainingDynamicsPredictor(
            config=self.config.prediction,
            n_features=n_features,
        )
        model.load_state_dict(state_dict)
        return model.to(self.device)

    def update(self, signals: Dict[str, Any], epoch: Optional[int] = None) -> None:

        if epoch is not None:
            self._current_epoch = epoch
        else:
            self._current_epoch += 1

        snapshot = EpochSnapshot(
            epoch=self._current_epoch,
            step=0,
            signals=signals,
            timestamp=0,
        )

        self.buffer.append(snapshot)

    def predict(self, n_mc_samples: int = 20) -> FailurePrediction:

        if len(self.buffer) < self.config.feature.min_history:
            return self._create_insufficient_data_prediction()

        if self.model is None:
            # Never predict with randomly initialised weights: the output
            # would look like a risk score and mean nothing.
            return self._unknown_prediction(
                "No trained failure-prediction model loaded (pass model_path). "
                "ARC reports no risk rather than guessing.")

        features = self._extract_features()
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        features_tensor = features_tensor.to(self.device)

        if features_tensor.shape[2] != self.model.n_features:
            return self._unknown_prediction(
                f"Loaded model expects {self.model.n_features} features but this run "
                f"produces {features_tensor.shape[2]}; it was trained on different signals.")

        with torch.no_grad():
            mc_output = self.model.predict_with_uncertainty(
                features_tensor,
                n_samples=n_mc_samples
            )

        probs_mean = mc_output["failure_probs_mean"][0].cpu().numpy()
        probs_std = mc_output["failure_probs_std"][0].cpu().numpy()
        ttf_mean = mc_output["ttf_mean"][0].cpu().numpy()
        ttf_std = mc_output["ttf_std"][0].cpu().numpy()

        failure_probs = {}
        confidence_intervals = {}
        ttf = {}
        ttf_unc = {}

        for i, mode in enumerate(FailureMode):
            failure_probs[mode] = float(probs_mean[i])

            lower = max(0, probs_mean[i] - 1.96 * probs_std[i])
            upper = min(1, probs_mean[i] + 1.96 * probs_std[i])
            confidence_intervals[mode] = (float(lower), float(upper))

            ttf[mode] = float(ttf_mean[i])
            ttf_unc[mode] = float(ttf_std[i])

        top_contributors = self._get_top_contributors(features_tensor)

        recommendation = self._generate_recommendation(failure_probs, top_contributors)

        overall_risk = max(failure_probs.values())
        risk_level = self._categorize_risk(overall_risk)

        return FailurePrediction(
            epoch=self._current_epoch,
            failure_probabilities=failure_probs,
            confidence_intervals=confidence_intervals,
            time_to_failure=ttf,
            ttf_uncertainty=ttf_unc,
            top_contributors=top_contributors,
            recommendation=recommendation,
            overall_risk=overall_risk,
            risk_level=risk_level,
        )

    def _extract_features(self) -> np.ndarray:

        signal_histories = {}
        for key in self.buffer.signal_keys:
            history = self.buffer.get_signal_history(key)
            if len(history) > 0:
                signal_histories[key] = history

        features_dict = self.feature_extractor.extract_all_features(signal_histories)

        self.normalizer.update(features_dict)

        if not hasattr(self, 'feature_names'):

             self.feature_names = sorted(features_dict.keys())

        if hasattr(self, 'feature_names'):

            normalized = []
            for name in self.feature_names:
                val = features_dict.get(name, 0.0)

                norm_dict = self.normalizer.normalize({name: val})
                norm_val = norm_dict.get(name, 0.0)
                normalized.append(norm_val)
            normalized = np.array(normalized)
        else:

            normalized = self.normalizer.normalize(features_dict, return_array=True)

        window_size = self.config.feature.window_size
        n_features = len(normalized)

        sequence = np.zeros((window_size, max(n_features, 1)))

        if n_features > 0:
             sequence[-1, :] = normalized

        return sequence

    def _get_top_contributors(
        self,
        features_tensor: torch.Tensor,
        n_top: int = 3
    ) -> List[SignalContribution]:

        contributors = []

        with torch.no_grad():
            output = self.model(features_tensor)

        probs = output.failure_probs[0].cpu().numpy()
        max_mode_idx = int(np.argmax(probs[:-1]))

        importance = self.model.get_feature_importance(
            features_tensor,
            max_mode_idx
        )

        avg_importance = importance[0].mean(dim=0).cpu().numpy()

        top_indices = np.argsort(avg_importance)[-n_top:][::-1]

        feature_names = self.normalizer.feature_names

        for idx in top_indices:
            if idx >= len(feature_names):
                continue

            name = feature_names[idx]
            score = float(avg_importance[idx]) / (avg_importance.max() + 1e-10)

            history = self.buffer.get_signal_history(name)
            current = history[-1] if len(history) > 0 else 0.0

            if len(history) >= 3:
                diff = history[-1] - history[-3]
                if abs(diff) < 0.01:
                    trend = "stable"
                elif diff > 0:
                    trend = "increasing"
                else:
                    trend = "decreasing"
            else:
                trend = "stable"

            interpretation = self._interpret_signal(name, current, trend, score)

            contributors.append(SignalContribution(
                signal_name=name,
                contribution_score=score,
                current_value=current,
                trend=trend,
                interpretation=interpretation,
            ))

        return contributors

    def _interpret_signal(
        self,
        name: str,
        value: float,
        trend: str,
        importance: float
    ) -> str:

        interpretations = {
            "grad_norm": "Gradient magnitude affecting training stability",
            "grad_entropy": "Gradient distribution uniformity",
            "grad_flow": "Gradient propagation through layers",
            "activation": "Hidden layer activity patterns",
            "similarity": "Representation diversity",
            "effective_rank": "Network capacity utilization",
            "dead_neuron": "Inactive network components",
            "weight_update": "Parameter update magnitude",
            "loss": "Training objective progress",
            "train_val_gap": "Generalization behavior",
        }

        for key, interp in interpretations.items():
            if key in name.lower():
                direction = f"({trend})" if trend != "stable" else ""
                return f"{interp} {direction}".strip()

        return f"Signal affecting prediction ({trend})"

    def _generate_recommendation(
        self,
        failure_probs: Dict[FailureMode, float],
        contributors: List[SignalContribution],
    ) -> InterventionRecommendation:

        max_mode, max_prob = max(failure_probs.items(), key=lambda x: x[1])

        if max_prob < 0.3:
            return InterventionRecommendation(
                action="no_action",
                confidence=0.8,
                rationale="Training appears healthy. Continue monitoring.",
                caveats=["Predictions are probabilistic; failures may still occur."],
            )

        recommendations = {
            FailureMode.DIVERGENCE: InterventionRecommendation(
                action="reduce_learning_rate",
                confidence=min(0.9, max_prob + 0.1),
                rationale="High divergence risk detected. Reducing learning rate may stabilize training.",
                parameters={"lr_factor": 0.5},
                caveats=[
                    "Learning rate reduction may slow convergence.",
                    "Consider also enabling gradient clipping.",
                ],
            ),
            FailureMode.VANISHING_GRADIENTS: InterventionRecommendation(
                action="increase_learning_rate",
                confidence=min(0.8, max_prob),
                rationale="Gradient flow appears diminished. Consider learning rate increase or architecture changes.",
                parameters={"lr_factor": 2.0},
                caveats=[
                    "May cause instability if gradients are not truly vanishing.",
                    "Consider residual connections or batch normalization.",
                ],
            ),
            FailureMode.EXPLODING_GRADIENTS: InterventionRecommendation(
                action="enable_gradient_clipping",
                confidence=min(0.9, max_prob + 0.1),
                rationale="Gradient magnitudes appear dangerous. Enable gradient clipping.",
                parameters={"max_norm": 1.0},
                caveats=[
                    "Clipping may affect convergence properties.",
                ],
            ),
            FailureMode.REPRESENTATION_COLLAPSE: InterventionRecommendation(
                action="add_regularization",
                confidence=min(0.75, max_prob),
                rationale="Representations may be collapsing. Consider spectral normalization or dropout.",
                parameters={"weight_decay": 0.01},
                caveats=[
                    "Collapse detection has higher uncertainty.",
                    "May require architecture changes.",
                ],
            ),
            FailureMode.SEVERE_OVERFITTING: InterventionRecommendation(
                action="early_stop",
                confidence=min(0.85, max_prob),
                rationale="Generalization gap growing. Consider early stopping or increased regularization.",
                parameters={},
                caveats=[
                    "Early stopping may leave performance on the table.",
                    "Consider data augmentation as alternative.",
                ],
            ),
        }

        return recommendations.get(max_mode, InterventionRecommendation(
            action="manual_review",
            confidence=0.5,
            rationale="Unusual pattern detected. Manual review recommended.",
            caveats=["Automated recommendation unavailable."],
        ))

    def _categorize_risk(self, risk: float) -> str:

        if risk < 0.25:
            return "low"
        elif risk < 0.5:
            return "medium"
        elif risk < 0.75:
            return "high"
        else:
            return "critical"

    def _create_insufficient_data_prediction(self) -> FailurePrediction:
        return self._unknown_prediction(
            f"Insufficient history ({len(self.buffer)} epochs). "
            f"Need {self.config.feature.min_history} epochs minimum.")

    def _unknown_prediction(self, rationale: str) -> FailurePrediction:

        probs = {mode: 0.0 for mode in FailureMode}
        cis = {mode: (0.0, 0.5) for mode in FailureMode}
        ttf = {mode: float('inf') for mode in FailureMode}
        ttf_unc = {mode: float('inf') for mode in FailureMode}

        return FailurePrediction(
            epoch=self._current_epoch,
            failure_probabilities=probs,
            confidence_intervals=cis,
            time_to_failure=ttf,
            ttf_uncertainty=ttf_unc,
            top_contributors=[],
            recommendation=InterventionRecommendation(
                action="no_action",
                confidence=0.0,
                rationale=rationale,
                caveats=["No prediction was made."],
            ),
            overall_risk=0.0,
            risk_level="unknown",
        )

    def reset(self) -> None:

        self.buffer.clear()
        self._current_epoch = 0
