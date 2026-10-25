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

"""
Lyapunov Stability Monitor for Neural Network Training.

Estimates the maximal Lyapunov exponent of the training dynamical system
by tracking parameter velocities between successive optimiser steps.

A positive exponent indicates exponentially diverging trajectories (instability),
while a negative exponent indicates convergence toward a fixed point.

This module provides *predictive* failure detection: it can flag instability
26–101 steps before loss-based detectors trigger (see paper Section 4.5).
"""

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import numpy as np


class TrainingPhase(Enum):
    """Qualitative phase of the training dynamical system."""
    TRANSIENT = "transient"      # Early training, high velocity variance
    CONVERGENT = "convergent"    # Stable convergence, λ < 0
    OSCILLATORY = "oscillatory"  # Marginal stability, λ ≈ 0
    DIVERGENT = "divergent"      # Unstable, λ > 0


@dataclass
class LyapunovState:
    """Snapshot of Lyapunov analysis at a given step."""
    exponent: float = 0.0                 # Smoothed Lyapunov exponent (EMA)
    instantaneous: float = 0.0            # Raw single-step estimate
    phase: TrainingPhase = TrainingPhase.TRANSIENT
    velocity_norm: float = 0.0            # ‖θ_t − θ_{t-1}‖
    velocity_acceleration: float = 0.0    # d(velocity)/dt
    dominant_frequency: float = 0.0       # From FFT oscillation detection
    oscillation_power: float = 0.0        # Power at dominant frequency
    is_oscillating: bool = False
    confidence: float = 0.0               # 0-1, how reliable the estimate is


def build_param_sample(params: List[torch.Tensor], n_sample: int, seed: int = 42):
    """Fixed per-tensor index sample totalling about `n_sample` coordinates.

    Indices are drawn per tensor in proportion to its size, so reading them
    never concatenates the whole model (an O(model) copy on every call).
    Returns a list of (tensor_position, index_tensor) pairs.
    """
    total = sum(p.numel() for p in params)
    gen = torch.Generator().manual_seed(seed)
    sample = []
    for i, p in enumerate(params):
        k = int(round(n_sample * p.numel() / max(total, 1)))
        if k <= 0:
            continue
        if k >= p.numel():
            idx = torch.arange(p.numel())
        else:
            idx = torch.randperm(p.numel(), generator=gen)[:k] if p.numel() <= 10_000_000 \
                else torch.randint(p.numel(), (k,), generator=gen)
        sample.append((i, idx.to(p.device)))
    return sample


def read_param_sample(params: List[torch.Tensor], sample) -> torch.Tensor:
    """Gather the sampled coordinates into one float32 CPU vector."""
    with torch.no_grad():
        return torch.cat([
            params[i].detach().reshape(-1)[idx].float() for i, idx in sample
        ]).cpu()


class LyapunovMonitor:
    """
    Online Lyapunov exponent estimator for neural network training.

    Tracks parameter velocities v_t = ‖θ_t − θ_{t-1}‖ and estimates the
    maximal Lyapunov exponent via:

        λ̂_t = ln(‖v_t‖ / ‖v_{t-1}‖)
        λ̄_t = α · λ̂_t + (1 − α) · λ̄_{t-1}

    Thresholds:
        λ > 0.2  → DIVERGENT  (immediate rollback)
        λ > 0.1  → TRANSIENT  (increase monitoring)
        |λ| < 0.05 → OSCILLATORY (check for periodic behaviour)
        λ < -0.1 → CONVERGENT (reduce monitoring overhead)
    """

    def __init__(
        self,
        window: int = 30,
        ema_decay: float = 0.92,
        divergent_threshold: float = 0.2,
        transient_threshold: float = 0.1,
        oscillatory_band: float = 0.05,
        fft_window: int = 64,
        fft_power_threshold: float = 2.0,
        sample_ratio: float = 0.1,
        max_sample_params: int = 500_000,
    ):
        self.window = window
        self.ema_decay = ema_decay
        self.divergent_threshold = divergent_threshold
        self.transient_threshold = transient_threshold
        self.oscillatory_band = oscillatory_band
        self.fft_window = fft_window
        self.fft_power_threshold = fft_power_threshold
        self.sample_ratio = sample_ratio
        self.max_sample_params = max_sample_params

        # State
        self._velocities: deque = deque(maxlen=window)
        self._lyapunov_ema: float = 0.0
        self._prev_params: Optional[torch.Tensor] = None
        self._param_indices: Optional[torch.Tensor] = None
        self._step_count: int = 0
        self._warmup_steps: int = 5

        # FFT buffer for oscillation detection
        self._velocity_fft_buffer: deque = deque(maxlen=fft_window)

    def _sample_params(self, model: nn.Module) -> torch.Tensor:
        """Read a fixed, deterministic subset of parameter coordinates.

        The subset is chosen once per tensor (see `build_param_sample`), so
        memory cost is O(sample), never O(model).
        """
        params = [p for p in model.parameters() if p.requires_grad]
        if self._param_indices is None or len(self._param_indices[1]) != len(params):
            n_total = sum(p.numel() for p in params)
            n_sample = min(int(n_total * self.sample_ratio), self.max_sample_params)
            n_sample = min(max(n_sample, 1000), n_total)
            self._param_indices = (
                build_param_sample(params, n_sample),
                [p.shape for p in params],
            )
        return read_param_sample(params, self._param_indices[0])

    def update(self, model: nn.Module, lr: Optional[float] = None) -> LyapunovState:
        """
        Update Lyapunov estimate with current model parameters.

        Pass the current learning rate as `lr` so a schedule that grows the
        step size (warmup) isn't mistaken for divergence: velocity is then
        measured per unit of learning rate.

        Returns a LyapunovState with the current stability assessment.
        """
        self._step_count += 1
        current = self._sample_params(model)

        state = LyapunovState()

        if self._prev_params is None:
            self._prev_params = current
            return state

        # Compute velocity: v_t = ‖θ_t − θ_{t-1}‖, optionally per unit LR
        velocity = (current - self._prev_params).norm().item()
        state.velocity_norm = velocity
        if lr is not None and lr > 0:
            velocity = velocity / lr
        self._velocities.append(velocity)
        self._velocity_fft_buffer.append(velocity)
        self._prev_params = current

        if len(self._velocities) < 3:
            return state

        # Compute velocity acceleration
        velocities = list(self._velocities)
        if len(velocities) >= 2:
            state.velocity_acceleration = velocities[-1] - velocities[-2]

        # Compute instantaneous Lyapunov exponent: λ = ln(v_t / v_{t-1})
        v_prev = velocities[-2]
        v_curr = velocities[-1]

        if v_prev > 1e-12:
            instant_lyap = math.log(v_curr / v_prev)
        else:
            instant_lyap = 0.0

        state.instantaneous = instant_lyap

        # EMA smoothing
        if self._step_count <= self._warmup_steps:
            self._lyapunov_ema = instant_lyap
        else:
            self._lyapunov_ema = (
                self.ema_decay * self._lyapunov_ema +
                (1 - self.ema_decay) * instant_lyap
            )

        state.exponent = self._lyapunov_ema

        # Confidence: increases with number of samples
        n = len(self._velocities)
        state.confidence = min(1.0, n / self.window)

        # Phase classification
        state.phase = self._classify_phase(state.exponent, state.confidence)

        # FFT oscillation detection
        if len(self._velocity_fft_buffer) >= 16:
            fft_result = self._detect_oscillation()
            state.dominant_frequency = fft_result["frequency"]
            state.oscillation_power = fft_result["power_ratio"]
            state.is_oscillating = fft_result["detected"]

        return state

    def _classify_phase(self, exponent: float, confidence: float) -> TrainingPhase:
        """Classify training phase based on Lyapunov exponent."""
        if confidence < 0.3:
            return TrainingPhase.TRANSIENT

        if exponent > self.divergent_threshold:
            return TrainingPhase.DIVERGENT
        elif exponent > self.transient_threshold:
            return TrainingPhase.TRANSIENT
        elif abs(exponent) < self.oscillatory_band:
            return TrainingPhase.OSCILLATORY
        else:
            return TrainingPhase.CONVERGENT

    def _detect_oscillation(self) -> Dict[str, Any]:
        """
        FFT-based periodic behaviour detection.

        Computes the power spectrum of recent velocities and checks whether
        any frequency component dominates significantly above the mean.
        """
        velocities = np.array(list(self._velocity_fft_buffer), dtype=np.float64)
        n = len(velocities)

        if n < 8:
            return {"detected": False, "frequency": 0.0, "power_ratio": 0.0}

        # Remove DC component (mean)
        velocities = velocities - velocities.mean()

        # Windowing to reduce spectral leakage
        window = np.hanning(n)
        velocities = velocities * window

        # FFT
        spectrum = np.fft.rfft(velocities)
        power = np.abs(spectrum) ** 2

        # Skip DC component (index 0)
        if len(power) < 2:
            return {"detected": False, "frequency": 0.0, "power_ratio": 0.0}

        power_no_dc = power[1:]
        if power_no_dc.sum() < 1e-12:
            return {"detected": False, "frequency": 0.0, "power_ratio": 0.0}

        peak_idx = np.argmax(power_no_dc) + 1  # +1 because we skipped DC
        peak_power = power[peak_idx]
        mean_power = power_no_dc.mean()

        # Frequency in cycles per step
        frequency = peak_idx / n

        power_ratio = peak_power / mean_power if mean_power > 1e-12 else 0.0
        detected = power_ratio > self.fft_power_threshold

        return {
            "detected": detected,
            "frequency": float(frequency),
            "power_ratio": float(power_ratio),
            "period": float(1.0 / frequency) if frequency > 0 else float("inf"),
        }

    @property
    def exponent(self) -> float:
        """Current smoothed Lyapunov exponent."""
        return self._lyapunov_ema

    @property
    def is_divergent(self) -> bool:
        """Whether training is in a divergent phase."""
        return self._lyapunov_ema > self.divergent_threshold

    @property
    def is_warning(self) -> bool:
        """Whether Lyapunov exponent indicates early warning."""
        return self._lyapunov_ema > self.transient_threshold

    def reset(self):
        """Reset all state (e.g. after rollback)."""
        self._velocities.clear()
        self._velocity_fft_buffer.clear()
        self._prev_params = None
        self._lyapunov_ema = 0.0
        self._step_count = 0

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return full diagnostic state for logging/debugging."""
        return {
            "lyapunov_exponent": self._lyapunov_ema,
            "velocity_history": list(self._velocities),
            "step_count": self._step_count,
            "n_samples": len(self._velocities),
        }
