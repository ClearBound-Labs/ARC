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

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from collections import deque
import warnings
import numpy as np

@dataclass
class ForecastConfig:

    history_size: int = 20

    ema_alpha: float = 0.3

    explosion_threshold: float = 1e4
    growth_rate_threshold: float = 2.0
    acceleration_threshold: float = 1.5

    lookahead_steps: int = 3

class GradientForecaster:

    def __init__(
        self,
        model: nn.Module,
        config: Optional[ForecastConfig] = None,
    ):
        self.model = model
        self.config = config or ForecastConfig()

        self.grad_norm_history: deque = deque(maxlen=self.config.history_size)
        self.ema_norm: float = 0.0
        self.ema_growth: float = 0.0

        self.step_count = 0

        self.last_prediction = None
        self.correct_predictions = 0
        self.total_predictions = 0

    def _compute_grad_norm(self) -> float:

        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def update(self) -> Dict[str, Any]:

        current_norm = self._compute_grad_norm()
        self.step_count += 1

        self.grad_norm_history.append(current_norm)

        if self.step_count == 1:
            self.ema_norm = current_norm
            self.ema_growth = 0.0
        else:
            prev_norm = self.grad_norm_history[-2] if len(self.grad_norm_history) > 1 else current_norm
            growth = current_norm / max(prev_norm, 1e-10)

            self.ema_norm = self.config.ema_alpha * current_norm + (1 - self.config.ema_alpha) * self.ema_norm
            self.ema_growth = self.config.ema_alpha * growth + (1 - self.config.ema_alpha) * self.ema_growth

        if self.last_prediction is not None:
            actual_exploded = current_norm > self.config.explosion_threshold
            if self.last_prediction.will_explode == actual_exploded:
                self.correct_predictions += 1
            self.total_predictions += 1

        return {
            "step": self.step_count,
            "grad_norm": current_norm,
            "ema_norm": self.ema_norm,
            "ema_growth": self.ema_growth,
        }

    def predict(self) -> 'GradientForecast':

        if len(self.grad_norm_history) < 3:

            return GradientForecast(
                will_explode=False,
                steps_until=float('inf'),
                confidence=0.0,
                predicted_norm=self.ema_norm,
            )

        history = list(self.grad_norm_history)
        current_norm = history[-1]

        try:
            log_norms = [np.log(max(n, 1e-10)) for n in history[-10:]]
            x = np.arange(len(log_norms))
            slope = np.polyfit(x, log_norms, 1)[0]

            growth_rate = np.exp(slope)
        except Exception as e:
            warnings.warn(f"Gradient-norm growth-rate fit failed, assuming no growth: {e}")
            growth_rate = 1.0

        lookahead = self.config.lookahead_steps
        predicted_norm = current_norm * (growth_rate ** lookahead)

        will_explode = False
        steps_until = float('inf')

        if predicted_norm > self.config.explosion_threshold:
            will_explode = True

            if growth_rate > 1:
                steps_until = np.log(self.config.explosion_threshold / max(current_norm, 1e-10)) / np.log(growth_rate)
                steps_until = max(1, int(steps_until))
            else:
                steps_until = lookahead

        if self.ema_growth > self.config.growth_rate_threshold:
            will_explode = True
            steps_until = min(steps_until, 5)

        if len(history) >= 5:
            recent_growth = [history[i] / max(history[i-1], 1e-10) for i in range(-4, 0)]
            consistency = 1.0 - np.std(recent_growth) / max(np.mean(recent_growth), 1e-10)
            confidence = max(0.0, min(1.0, consistency))
        else:
            confidence = 0.3

        forecast = GradientForecast(
            will_explode=will_explode,
            steps_until=steps_until,
            confidence=confidence,
            predicted_norm=predicted_norm,
            current_norm=current_norm,
            growth_rate=growth_rate,
        )

        self.last_prediction = forecast
        return forecast

    def get_stats(self) -> Dict[str, Any]:

        accuracy = self.correct_predictions / max(self.total_predictions, 1)
        return {
            "total_steps": self.step_count,
            "predictions_made": self.total_predictions,
            "prediction_accuracy": accuracy,
            "current_ema_norm": self.ema_norm,
            "current_growth_rate": self.ema_growth,
        }

@dataclass
class GradientForecast:

    will_explode: bool
    steps_until: float
    confidence: float
    predicted_norm: float
    current_norm: float = 0.0
    growth_rate: float = 1.0

class FP16ScalerMonitor:

    def __init__(
        self,
        scaler: torch.cuda.amp.GradScaler = None,
        min_scale: float = 1.0,
        max_skip_ratio: float = 0.3,
    ):
        self.scaler = scaler
        self.min_scale = min_scale
        self.max_skip_ratio = max_skip_ratio

        self.step_count = 0
        self.skip_count = 0
        self.scale_history: deque = deque(maxlen=20)

    def step(self, loss: torch.Tensor = None) -> Dict[str, Any]:

        self.step_count += 1

        result = {
            "step": self.step_count,
            "alert": None,
            "scale": None,
            "skip_ratio": self.skip_count / max(self.step_count, 1),
        }

        if self.scaler is None:
            return result

        current_scale = self.scaler.get_scale()
        result["scale"] = current_scale
        self.scale_history.append(current_scale)

        if current_scale < self.min_scale:
            result["alert"] = "scale_collapse"
            result["recommendation"] = "increase_loss_scale"

        if len(self.scale_history) > 1:
            if current_scale < self.scale_history[-2]:
                self.skip_count += 1

        skip_ratio = self.skip_count / max(self.step_count, 1)
        if skip_ratio > self.max_skip_ratio:
            result["alert"] = "frequent_skips"
            result["recommendation"] = "reduce_learning_rate"

        return result

if __name__ == "__main__":
    print("Testing Gradient Forecaster...")

    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10),
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    forecaster = GradientForecaster(model)

    for step in range(50):
        x = torch.randn(32, 100)

        scale = 1.0 + step * 0.1
        out = model(x) * scale
        loss = out.mean()

        optimizer.zero_grad()
        loss.backward()

        update_info = forecaster.update()

        forecast = forecaster.predict()

        if forecast.will_explode:
            print(f"Step {step}: ⚠️ Explosion predicted in {forecast.steps_until:.0f} steps "
                  f"(norm: {update_info['grad_norm']:.2f}, predicted: {forecast.predicted_norm:.2f})")

        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

    print(f"\nForecaster stats: {forecaster.get_stats()}")
    print("✅ Gradient Forecaster test complete")
