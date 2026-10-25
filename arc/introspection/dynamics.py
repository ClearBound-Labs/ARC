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

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import math
import numpy as np
import torch
import torch.nn as nn
from collections import deque

@dataclass
class DynamicalState:
    lyapunov_exponent: float
    phase: str
    attractor_distance: float
    velocity: float
    acceleration: float
    is_at_bifurcation: bool

class LyapunovEstimator:
    def __init__(
        self,
        model: nn.Module,
        window_size: int = 50,
        perturbation_scale: float = 1e-6,
    ):
        self.model = model
        self.window_size = window_size
        self.perturbation_scale = perturbation_scale

        self._grad_norms: deque = deque(maxlen=window_size)

        self._param_deltas: deque = deque(maxlen=window_size)
        self._prev_params: Optional[Dict[str, torch.Tensor]] = None

        self._lyapunov_history: deque = deque(maxlen=window_size)

        self._jacobian_traces: deque = deque(maxlen=window_size)

    def update(self) -> float:
        total_grad_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
        total_grad_norm = math.sqrt(total_grad_norm)

        self._grad_norms.append(total_grad_norm)

        if self._prev_params is not None:
            delta_norm = 0.0
            for name, param in self.model.named_parameters():
                if name in self._prev_params:
                    delta = param.data - self._prev_params[name]
                    delta_norm += delta.norm().item() ** 2
            delta_norm = math.sqrt(delta_norm)
            self._param_deltas.append(delta_norm)

        self._update_prev_params()

        lyap = self._estimate_lyapunov()
        self._lyapunov_history.append(lyap)

        return lyap

    def _update_prev_params(self) -> None:
        self._prev_params = {
            name: param.data.detach().clone()
            for name, param in self.model.named_parameters()
        }

    def _estimate_lyapunov(self) -> float:
        if len(self._grad_norms) < 10:
            return 0.0

        norms = np.array(list(self._grad_norms))

        norms = np.maximum(norms, 1e-10)
        log_norms = np.log(norms)

        t = np.arange(len(log_norms))

        if len(log_norms) > 2:
            slopes = np.diff(log_norms)
            lyap = np.median(slopes)
        else:
            lyap = 0.0

        return float(lyap)

    def estimate_spectrum(self, n_exponents: int = 3) -> List[float]:
        if len(self._param_deltas) < 20:
            return [0.0] * n_exponents

        deltas = np.array(list(self._param_deltas))

        embedding_dim = min(n_exponents + 2, len(deltas) // 2)

        if embedding_dim < 2:
            return [self._estimate_lyapunov()] * n_exponents

        trajectory = np.zeros((len(deltas) - embedding_dim, embedding_dim))
        for i in range(len(deltas) - embedding_dim):
            trajectory[i] = deltas[i:i + embedding_dim]

        try:
            U, S, Vh = np.linalg.svd(trajectory, full_matrices=False)

            log_s = np.log(S + 1e-10)

            spectrum = log_s[:n_exponents] / trajectory.shape[0]

            return list(spectrum)
        except:
            return [0.0] * n_exponents

    def get_phase(self) -> str:
        if len(self._lyapunov_history) < 5:
            return "unknown"

        lyap = np.mean(list(self._lyapunov_history)[-10:])
        lyap_std = np.std(list(self._lyapunov_history)[-10:])

        if lyap > 0.1:
            return "chaotic"
        elif lyap < -0.1:
            if np.mean(list(self._grad_norms)[-5:]) < 1e-5:
                return "converged"
            return "stable"
        elif lyap_std < 0.05:
            return "edge_of_chaos"
        else:
            return "transitional"

class AttractorDetector:
    def __init__(
        self,
        model: nn.Module,
        recurrence_threshold: float = 0.1,
        convergence_threshold: float = 1e-5,
    ):
        self.model = model
        self.recurrence_threshold = recurrence_threshold
        self.convergence_threshold = convergence_threshold

        self._trajectory_sketches: deque = deque(maxlen=200)

        self._velocities: deque = deque(maxlen=100)

        self._attractors: List[Dict[str, Any]] = []

    def update(self) -> Dict[str, Any]:
        sketch = self._compute_sketch()
        self._trajectory_sketches.append(sketch)

        if len(self._trajectory_sketches) >= 2:
            velocity = np.linalg.norm(
                sketch - self._trajectory_sketches[-2]
            )
        else:
            velocity = float('inf')

        self._velocities.append(velocity)

        attractor_info = self._detect_attractor_type()

        return attractor_info

    def _compute_sketch(self) -> np.ndarray:
        flat_params = []
        for param in self.model.parameters():
            flat_params.append(param.data.flatten()[:100].cpu().numpy())

        if not flat_params:
            return np.zeros(50)

        concat = np.concatenate(flat_params)[:500]

        sketch_dim = 50
        if len(concat) < sketch_dim:
            sketch = np.pad(concat, (0, sketch_dim - len(concat)))
        else:
            step = len(concat) // sketch_dim
            sketch = concat[::step][:sketch_dim]

        return sketch

    def _detect_attractor_type(self) -> Dict[str, Any]:
        if len(self._velocities) < 10:
            return {"type": "unknown", "confidence": 0.0}

        velocities = np.array(list(self._velocities)[-20:])
        mean_vel = np.mean(velocities)
        vel_trend = np.polyfit(range(len(velocities)), velocities, 1)[0]

        if mean_vel < self.convergence_threshold:
            return {
                "type": "fixed_point",
                "confidence": 1.0 - mean_vel / self.convergence_threshold,
                "distance": mean_vel,
            }

        if len(velocities) >= 10:
            fft = np.abs(np.fft.fft(velocities - np.mean(velocities)))
            peak_freq = np.argmax(fft[1:len(fft)//2]) + 1

            if fft[peak_freq] > np.mean(fft) * 2:
                period = len(velocities) / peak_freq
                return {
                    "type": "limit_cycle",
                    "confidence": min(1.0, fft[peak_freq] / (np.mean(fft) * 5)),
                    "period": period,
                }

        if vel_trend > 0:
            return {
                "type": "divergent",
                "confidence": min(1.0, abs(vel_trend) * 10),
            }
        else:
            return {
                "type": "strange_attractor",
                "confidence": 0.5,
                "mean_velocity": mean_vel,
            }

    def get_distance_to_attractor(self) -> float:
        if len(self._velocities) < 5:
            return float('inf')

        return float(np.mean(list(self._velocities)[-5:]))

class PhaseSpaceAnalyzer:
    def __init__(
        self,
        lyapunov: LyapunovEstimator,
        attractor: AttractorDetector,
    ):
        self.lyapunov = lyapunov
        self.attractor = attractor

        self._phase_history: deque = deque(maxlen=200)

    def update(self) -> DynamicalState:
        lyap = self.lyapunov.update()
        phase = self.lyapunov.get_phase()

        attractor_info = self.attractor.update()
        attractor_distance = self.attractor.get_distance_to_attractor()

        velocities = list(self.lyapunov._param_deltas)

        if len(velocities) >= 2:
            velocity = velocities[-1]
            acceleration = velocities[-1] - velocities[-2]
        else:
            velocity = 0.0
            acceleration = 0.0

        self._phase_history.append((velocity, acceleration))

        is_bifurcation = self._detect_bifurcation()

        return DynamicalState(
            lyapunov_exponent=lyap,
            phase=phase,
            attractor_distance=attractor_distance,
            velocity=velocity,
            acceleration=acceleration,
            is_at_bifurcation=is_bifurcation,
        )

    def _detect_bifurcation(self) -> bool:
        if len(self._phase_history) < 10:
            return False

        recent = list(self._phase_history)[-10:]
        accelerations = [a for v, a in recent]

        if len(accelerations) >= 5:
            recent_acc = accelerations[-5:]
            earlier_acc = accelerations[:5]

            if np.sign(np.mean(recent_acc)) != np.sign(np.mean(earlier_acc)):
                return True

        lyap_history = list(self.lyapunov._lyapunov_history)
        if len(lyap_history) >= 10:
            recent_lyap = np.mean(lyap_history[-5:])
            earlier_lyap = np.mean(lyap_history[-10:-5])

            if np.sign(recent_lyap) != np.sign(earlier_lyap):
                return True

        return False

    def get_phase_portrait_data(self) -> Tuple[List[float], List[float]]:
        velocities = [v for v, a in self._phase_history]
        accelerations = [a for v, a in self._phase_history]
        return velocities, accelerations