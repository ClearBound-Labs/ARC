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

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass

@dataclass
class SpectralAnalysis:
    dominant_frequencies: np.ndarray
    spectral_entropy: float
    high_freq_ratio: float
    spectral_gap: float
    recommendation: str

class FourierFeatureEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_frequencies: int = 256,
        sigma: float = 10.0,
        learnable: bool = False,
        encoding_type: str = "gaussian",
    ):
        super().__init__()

        self.input_dim = input_dim
        self.n_frequencies = n_frequencies
        self.sigma = sigma
        self.encoding_type = encoding_type
        self.output_dim = n_frequencies * 2

        if encoding_type == "gaussian":
            B = torch.randn(input_dim, n_frequencies) * sigma
        elif encoding_type == "positional":
            freqs = 2.0 ** torch.arange(n_frequencies // input_dim)
            B = torch.zeros(input_dim, n_frequencies)
            for i in range(input_dim):
                B[i, i * (n_frequencies // input_dim):(i + 1) * (n_frequencies // input_dim)] = freqs
        elif encoding_type == "log_linear":
            freqs = torch.logspace(0, np.log10(sigma), n_frequencies)
            B = freqs.unsqueeze(0).repeat(input_dim, 1) / input_dim
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")

        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * np.pi * x @ self.B

        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def get_frequency_spectrum(self) -> np.ndarray:
        return self.B.detach().cpu().numpy()

class SpectralRegularizer(nn.Module):
    def __init__(
        self,
        target_distribution: str = "uniform",
        lambda_spectral: float = 0.01,
    ):
        super().__init__()
        self.target_distribution = target_distribution
        self.lambda_spectral = lambda_spectral

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        error = predictions - targets

        if error.dim() >= 2:
            fft = torch.fft.rfft(error, dim=-1)
            power = torch.abs(fft) ** 2

            power_dist = power / (power.sum(dim=-1, keepdim=True) + 1e-10)

            n_freqs = power_dist.size(-1)

            if self.target_distribution == "uniform":
                target = torch.ones_like(power_dist) / n_freqs
            elif self.target_distribution == "balanced":
                idx = torch.arange(n_freqs, device=power_dist.device, dtype=power_dist.dtype)
                target = 1.0 / (idx + 1)
                target = target / target.sum()
                target = target.unsqueeze(0).expand_as(power_dist)
            else:
                return torch.tensor(0.0, device=predictions.device)

            reg_loss = F.kl_div(
                torch.log(power_dist + 1e-10),
                target,
                reduction='batchmean',
            )

            return self.lambda_spectral * reg_loss

        return torch.tensor(0.0, device=predictions.device)

class MultiScaleEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        scales: List[float] = None,
        features_per_scale: int = 32,
    ):
        super().__init__()

        if scales is None:
            scales = [2 ** i for i in range(8)]

        self.scales = scales
        self.encoders = nn.ModuleList([
            FourierFeatureEncoder(
                input_dim=input_dim,
                n_frequencies=features_per_scale,
                sigma=scale,
                encoding_type="gaussian",
            )
            for scale in scales
        ])

        self.output_dim = len(scales) * features_per_scale * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encodings = [encoder(x) for encoder in self.encoders]
        return torch.cat(encodings, dim=-1)

class SpectralAnalyzer:

    def __init__(self, n_bins: int = 64):
        self.n_bins = n_bins
        self._history: List[np.ndarray] = []

    def analyze(
        self,
        signal: Union[torch.Tensor, np.ndarray],
        reference: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> SpectralAnalysis:
        if isinstance(signal, torch.Tensor):
            signal = signal.detach().cpu().numpy()

        signal = signal.flatten()

        fft = np.fft.rfft(signal)
        power = np.abs(fft) ** 2
        freqs = np.fft.rfftfreq(len(signal))

        total_power = power.sum() + 1e-10
        power_norm = power / total_power

        n_dominant = min(5, len(power))
        dominant_idx = np.argsort(power)[-n_dominant:]
        dominant_frequencies = freqs[dominant_idx]

        spectral_entropy = -np.sum(power_norm * np.log(power_norm + 1e-10))
        spectral_entropy /= np.log(len(power))

        mid_idx = len(freqs) // 2
        high_freq_power = power[mid_idx:].sum()
        high_freq_ratio = high_freq_power / total_power

        sorted_power = np.sort(power)[::-1]
        if len(sorted_power) > 1:
            spectral_gap = (sorted_power[0] - sorted_power[1]) / (sorted_power[0] + 1e-10)
        else:
            spectral_gap = 0.0

        if high_freq_ratio < 0.1:
            recommendation = "Low high-frequency content. Consider using Fourier encoding."
        elif spectral_entropy < 0.3:
            recommendation = "Concentrated spectrum. Model may be missing frequency components."
        elif spectral_gap > 0.8:
            recommendation = "Single dominant frequency. Check for overfitting to specific scale."
        else:
            recommendation = "Healthy spectral distribution."

        self._history.append(power_norm)

        return SpectralAnalysis(
            dominant_frequencies=dominant_frequencies,
            spectral_entropy=float(spectral_entropy),
            high_freq_ratio=float(high_freq_ratio),
            spectral_gap=float(spectral_gap),
            recommendation=recommendation,
        )

    def compare_to_target(
        self,
        prediction: Union[torch.Tensor, np.ndarray],
        target: Union[torch.Tensor, np.ndarray],
    ) -> Dict[str, float]:
        pred_analysis = self.analyze(prediction)
        target_analysis = self.analyze(target)

        return {
            "entropy_gap": target_analysis.spectral_entropy - pred_analysis.spectral_entropy,
            "high_freq_gap": target_analysis.high_freq_ratio - pred_analysis.high_freq_ratio,
            "spectral_similarity": 1.0 - abs(
                pred_analysis.spectral_entropy - target_analysis.spectral_entropy
            ),
        }

    def get_frequency_loss_weights(
        self,
        target: Union[torch.Tensor, np.ndarray],
        boost_factor: float = 2.0,
    ) -> np.ndarray:
        if isinstance(target, torch.Tensor):
            target = target.detach().cpu().numpy()

        target = target.flatten()

        fft = np.fft.rfft(target)
        power = np.abs(fft) ** 2

        weights = 1.0 / (power + 1e-10)

        weights = weights / weights.mean()
        weights = np.clip(weights, 1.0, boost_factor)

        return weights