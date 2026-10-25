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

from arc.introspection.geometry import (
    FisherRaoGeometry,
    NaturalGradient,
    GeodesicTracker,
)
from arc.introspection.dynamics import (
    LyapunovEstimator,
    AttractorDetector,
    PhaseSpaceAnalyzer,
)
from arc.introspection.topology import (
    PersistentHomology,
    LossLandscapeSampler,
    TopologicalFeatures,
)
from arc.introspection.transport import (
    SinkhornDistance,
    DistributionTracker,
    WassersteinMonitor,
)
from arc.introspection.self_model import (
    SelfModelingHead,
    GradientPredictor,
    MetaCognition,
)
from arc.introspection.arc_network import (
    IntrospectiveArc,
    IntrospectionAction,
    create_self_aware_model,
    IntrospectionScheduler,
    SignalSmoother,
)
from arc.introspection.neural_ode import (
    TrainingDynamicsODE,
    TrajectoryPredictor,
    ContinuousLearningRateController,
)

__all__ = [
    "FisherRaoGeometry",
    "NaturalGradient",
    "GeodesicTracker",
    "LyapunovEstimator",
    "AttractorDetector",
    "PhaseSpaceAnalyzer",
    "PersistentHomology",
    "LossLandscapeSampler",
    "TopologicalFeatures",
    "SinkhornDistance",
    "DistributionTracker",
    "WassersteinMonitor",
    "SelfModelingHead",
    "GradientPredictor",
    "MetaCognition",
    "TrainingDynamicsODE",
    "TrajectoryPredictor",
    "ContinuousLearningRateController",
    "IntrospectiveArc",
    "IntrospectionAction",
    "create_self_aware_model",
    "IntrospectionScheduler",
    "SignalSmoother",
]