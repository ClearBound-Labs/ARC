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

__version__ = "5.2.0"
__author__ = "Aryan Kaushik"

import importlib

from arc.config import Config, FailureMode

# The canonical entry point; everything else is imported on first use, so
# `import arc` doesn't pay for the legacy and research modules.
from arc.core.controller import Controller, ControllerConfig, StepAction, HealthStatus
from arc.signals.lyapunov import LyapunovMonitor, LyapunovState, TrainingPhase
from arc.signals.trend import TrendDetector, DriftReport, DriftSeverity

_LAZY = {
    # Legacy APIs (still supported)
    "Arc": "arc.api.callback",
    "ArcV2": "arc.api.v2",
    "Arc2": "arc.api.v2",
    "ArcCallback": "arc.api.pytorch_callback",
    "ArcWrapper": "arc.api.pytorch_callback",
    "ReportGenerator": "arc.api.report",
    "BulletproofTrainer": "arc.api.bulletproof",
    "BulletproofConfig": "arc.api.bulletproof",
    "protect": "arc.api.bulletproof",
    "protect_trainer": "arc.api.bulletproof",

    "SignalCollector": "arc.signals",
    "CompositeCollector": "arc.signals",
    "GradientCollector": "arc.signals",
    "ActivationCollector": "arc.signals",
    "WeightCollector": "arc.signals",
    "OptimizerCollector": "arc.signals",
    "LossCollector": "arc.signals",
    "CurvatureCollector": "arc.signals",
    "SilentCrashDetector": "arc.signals.silent_detector",
    "SilentDetectorConfig": "arc.signals.silent_detector",
    "SilentFailureType": "arc.signals.silent_detector",

    "SignalBuffer": "arc.features",
    "EpochSnapshot": "arc.features",
    "FeatureExtractor": "arc.features",
    "OnlineNormalizer": "arc.features",
    "FourierFeatureEncoder": "arc.features.spectral",
    "SpectralAnalyzer": "arc.features.spectral",
    "MultiScaleEncoder": "arc.features.spectral",

    "FailurePredictor": "arc.prediction",
    "FailurePrediction": "arc.prediction",
    "SignalAttribution": "arc.prediction",
    "AttributionEngine": "arc.prediction",
    "ProbabilityCalibrator": "arc.prediction",

    "InterventionAction": "arc.intervention",
    "InterventionRecommender": "arc.intervention",
    "OOMRecoveryHandler": "arc.intervention.oom_handler",
    "OOMConfig": "arc.intervention.oom_handler",
    "OOMRecoveryFailed": "arc.intervention.oom_handler",
    "get_memory_stats": "arc.intervention.oom_handler",
    "estimate_model_memory": "arc.intervention.oom_handler",
    "HardwareRecoveryHandler": "arc.intervention.hardware_handler",
    "HardwareConfig": "arc.intervention.hardware_handler",
    "get_best_device": "arc.intervention.hardware_handler",
    "get_device_info": "arc.intervention.hardware_handler",

    "EvaluationMetrics": "arc.evaluation",
    "BenchmarkSuite": "arc.evaluation",
    "BenchmarkResult": "arc.evaluation",

    "FailureSimulator": "arc.learning",
    "TrajectoryLabeler": "arc.learning",
    "TrainingDynamicsPredictor": "arc.learning",
    "MetaModelTrainer": "arc.learning",
    "ElasticWeightConsolidation": "arc.learning.ewc",
    "ProgressiveNet": "arc.learning.ewc",

    "ConformalPredictor": "arc.uncertainty",
    "ConformalRegression": "arc.uncertainty",
    "VennAbersCalibrator": "arc.uncertainty",

    "AdversarialDetector": "arc.security",
    "AdversarialTrainer": "arc.security",
    "RandomizedSmoothing": "arc.security",

    "PINNStabilizer": "arc.physics",
    "AdaptiveLossBalancer": "arc.physics",
    "CurriculumScheduler": "arc.physics",
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'arc' has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    # Canonical API
    "Controller",
    "ControllerConfig",
    "StepAction",
    "HealthStatus",
    "LyapunovMonitor",
    "LyapunovState",
    "TrainingPhase",
    "TrendDetector",
    "DriftReport",
    "DriftSeverity",
    "Config",
    "FailureMode",
    "__version__",
    *_LAZY,
]
