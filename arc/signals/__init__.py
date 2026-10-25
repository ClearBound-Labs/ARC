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

from arc.signals.base import SignalCollector, SignalSnapshot, CompositeCollector
from arc.signals.gradient import GradientCollector
from arc.signals.activation import ActivationCollector
from arc.signals.weights import WeightCollector
from arc.signals.optimizer import OptimizerCollector
from arc.signals.loss import LossCollector
from arc.signals.curvature import CurvatureCollector

from arc.signals.neural_collapse import NeuralCollapseCollector
from arc.signals.sharpness import SharpnessCollector
from arc.signals.fisher import FisherCollector
from arc.signals.information import InformationDynamicsCollector
from arc.signals.lyapunov import LyapunovMonitor, LyapunovState, TrainingPhase
from arc.signals.trend import TrendDetector, DriftReport, DriftSeverity

from arc.signals.silent_detector import (
    SilentCrashDetector,
    SilentDetectorConfig,
    SilentFailureType,
    SilentFailureDetection,
    ActivationMonitor,
    ValidationMetricCallback,
    MetricTracker,
)

__all__ = [
    "SignalCollector",
    "SignalSnapshot",
    "CompositeCollector",
    "GradientCollector",
    "ActivationCollector",
    "WeightCollector",
    "OptimizerCollector",
    "LossCollector",
    "CurvatureCollector",

    "NeuralCollapseCollector",
    "SharpnessCollector",
    "FisherCollector",
    "InformationDynamicsCollector",

    "LyapunovMonitor",
    "LyapunovState",
    "TrainingPhase",
    "TrendDetector",
    "DriftReport",
    "DriftSeverity",
]
