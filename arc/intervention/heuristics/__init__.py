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
Model-family-specific training heuristics (diffusion/LLM/detection),
plus `AutoHeuristics` which auto-detects the model family and dispatches
to the matching one.

Split into per-family modules (previously one 429-line file) so each
heuristics class can be read/tested independently; this package's public
surface (`from arc.intervention.heuristics import X`) is unchanged.
"""

from arc.intervention.heuristics.diffusion import DiffusionConfig, DiffusionHeuristics
from arc.intervention.heuristics.llm import LLMConfig, LLMHeuristics
from arc.intervention.heuristics.detection import DetectionConfig, DetectionHeuristics
from arc.intervention.heuristics.auto import AutoHeuristics

__all__ = [
    "DiffusionConfig",
    "DiffusionHeuristics",
    "LLMConfig",
    "LLMHeuristics",
    "DetectionConfig",
    "DetectionHeuristics",
    "AutoHeuristics",
]
