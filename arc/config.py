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
from enum import Enum, auto
from typing import List, Optional, Dict, Any
import json

from pathlib import Path


def default_checkpoint_dir(subdir: str = "checkpoints") -> str:
    """Per-user default checkpoint directory.

    Resolves to ``~/.cache/arc/<subdir>`` on every platform. Used as the
    default for checkpoint storage so that ARC does not write into
    world-writable ``/tmp/`` on shared systems (where another local user
    can write or symlink files that ARC would later deserialize via
    ``torch.load``). See SECURITY.md for the full trust boundary.
    """
    return str(Path.home() / ".cache" / "arc" / subdir)

class FailureMode(Enum):

    DIVERGENCE = auto()
    VANISHING_GRADIENTS = auto()
    EXPLODING_GRADIENTS = auto()
    REPRESENTATION_COLLAPSE = auto()
    SEVERE_OVERFITTING = auto()

    GROKKING_RISK = auto()
    DOUBLE_DESCENT = auto()

    def __str__(self) -> str:
        return self.name.replace("_", " ").title()

@dataclass
class SignalConfig:

    collect_every_n_steps: int = 1
    aggregate_per_epoch: bool = True

    activation_sample_ratio: float = 0.1
    activation_sample_layers: Optional[List[str]] = None

    compute_gradient_entropy: bool = True
    gradient_histogram_bins: int = 50

    compute_curvature_proxy: bool = False
    curvature_hvp_samples: int = 5

    track_effective_rank: bool = True
    # Rank of the truncated SVD backing effective rank. This is a *detector* for
    # rank collapse, not a spectral analysis: 32 components resolve a collapse
    # toward single digits perfectly well, whereas q=100 approached a full SVD
    # on typical conv weights and made this signal cost more per collection than
    # the training step it was measuring.
    effective_rank_sample_size: int = 32

    def __post_init__(self):
        if not (0 < self.activation_sample_ratio <= 1):
            raise ValueError(
                f"activation_sample_ratio must be in (0, 1], got {self.activation_sample_ratio}"
            )
        if self.collect_every_n_steps < 1:
            raise ValueError(
                f"collect_every_n_steps must be >= 1, got {self.collect_every_n_steps}"
            )

@dataclass
class FeatureConfig:

    window_size: int = 10
    min_history: int = 3

    compute_trend: bool = True
    compute_curvature: bool = True
    compute_spectral: bool = False

    compute_correlations: bool = True
    correlation_pairs: Optional[List[tuple]] = None

    def __post_init__(self):
        if self.window_size <= 0:
            raise ValueError(
                f"window_size must be > 0, got {self.window_size}"
            )
        if self.min_history < 1:
            raise ValueError(
                f"min_history must be >= 1, got {self.min_history}"
            )

@dataclass
class PredictionConfig:

    temporal_hidden_size: int = 64
    num_gru_layers: int = 2
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64])
    cnn_kernel_size: int = 3
    dropout_rate: float = 0.2

    mc_dropout_samples: int = 20
    confidence_level: float = 0.95

    temperature_scaling: bool = True

    high_risk_threshold: float = 0.7
    medium_risk_threshold: float = 0.4

    def __post_init__(self):
        if not (0 <= self.dropout_rate <= 1):
            raise ValueError(
                f"dropout_rate must be in [0, 1], got {self.dropout_rate}"
            )
        if not (0 < self.confidence_level <= 1):
            raise ValueError(
                f"confidence_level must be in (0, 1], got {self.confidence_level}"
            )
        if self.mc_dropout_samples < 1:
            raise ValueError(
                f"mc_dropout_samples must be >= 1, got {self.mc_dropout_samples}"
            )

@dataclass
class FailureThresholds:

    loss_explosion_factor: float = 10.0
    loss_nan_detection: bool = True

    vanishing_grad_threshold: float = 1e-7
    vanishing_grad_epochs: int = 5

    exploding_grad_threshold: float = 1e4
    exploding_grad_epochs: int = 2

    activation_similarity_threshold: float = 0.95
    effective_rank_collapse_ratio: float = 0.3

    overfit_gap_threshold: float = 0.5
    overfit_val_plateau_epochs: int = 10

    def __post_init__(self):
        if self.loss_explosion_factor <= 0:
            raise ValueError(
                f"loss_explosion_factor must be > 0, got {self.loss_explosion_factor}"
            )
        _positive_thresholds = [
            ("vanishing_grad_threshold", self.vanishing_grad_threshold),
            ("exploding_grad_threshold", self.exploding_grad_threshold),
            ("activation_similarity_threshold", self.activation_similarity_threshold),
            ("effective_rank_collapse_ratio", self.effective_rank_collapse_ratio),
            ("overfit_gap_threshold", self.overfit_gap_threshold),
        ]
        for _name, _value in _positive_thresholds:
            if _value <= 0:
                raise ValueError(f"{_name} must be positive, got {_value}")

@dataclass
class OverheadConfig:

    max_overhead_percent: float = 5.0

    adaptive_sampling: bool = True
    overhead_check_interval: int = 100

    reduce_activation_sampling: bool = True
    disable_curvature_proxy: bool = True
    reduce_mc_samples: bool = True

    def __post_init__(self):
        if self.max_overhead_percent < 0:
            raise ValueError(
                f"max_overhead_percent must be >= 0, got {self.max_overhead_percent}"
            )
        if self.overhead_check_interval < 1:
            raise ValueError(
                f"overhead_check_interval must be >= 1, got {self.overhead_check_interval}"
            )

@dataclass
class Config:

    signal: SignalConfig = field(default_factory=SignalConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    thresholds: FailureThresholds = field(default_factory=FailureThresholds)
    overhead: OverheadConfig = field(default_factory=OverheadConfig)

    verbose: bool = False
    log_level: str = "WARNING"

    device: str = "auto"

    def to_dict(self) -> Dict[str, Any]:

        def _asdict_recursive(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: _asdict_recursive(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, Enum):
                return obj.name
            elif isinstance(obj, list):
                return [_asdict_recursive(i) for i in obj]
            else:
                return obj
        return _asdict_recursive(self)

    def to_json(self, indent: int = 2) -> str:

        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Config':

        return cls(
            signal=SignalConfig(**d.get('signal', {})),
            feature=FeatureConfig(**d.get('feature', {})),
            prediction=PredictionConfig(**d.get('prediction', {})),
            thresholds=FailureThresholds(**d.get('thresholds', {})),
            overhead=OverheadConfig(**d.get('overhead', {})),
            verbose=d.get('verbose', False),
            log_level=d.get('log_level', 'WARNING'),
            device=d.get('device', 'auto'),
        )

    @classmethod
    def from_json(cls, json_str: str) -> 'Config':

        return cls.from_dict(json.loads(json_str))

    @classmethod
    def low_overhead(cls) -> 'Config':
        return cls(
            signal=SignalConfig(
                activation_sample_ratio=0.05,
                compute_curvature_proxy=False,
                compute_gradient_entropy=False,
            ),
            feature=FeatureConfig(
                compute_spectral=False,
                compute_correlations=False,
            ),
            prediction=PredictionConfig(
                mc_dropout_samples=5,
            ),
            overhead=OverheadConfig(
                max_overhead_percent=2.0,
            ),
        )

    @classmethod
    def high_accuracy(cls) -> 'Config':
        return cls(
            signal=SignalConfig(
                activation_sample_ratio=0.3,
                compute_curvature_proxy=True,
            ),
            feature=FeatureConfig(
                compute_spectral=True,
                window_size=20,
            ),
            prediction=PredictionConfig(
                mc_dropout_samples=50,
            ),
            overhead=OverheadConfig(
                max_overhead_percent=10.0,
            ),
        )
