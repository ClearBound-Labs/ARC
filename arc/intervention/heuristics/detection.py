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
from typing import Optional, Dict, Any
from dataclasses import dataclass
from collections import deque
from arc._logging import log as _log

@dataclass
class DetectionConfig:
    track_component_losses: bool = True
    cls_loss_threshold: float = 50.0
    box_loss_threshold: float = 20.0
    obj_loss_threshold: float = 50.0

    track_box_predictions: bool = True
    max_box_size: float = 2.0
    min_box_size: float = 0.001

    track_nms_outputs: bool = True
    max_detections_per_image: int = 1000

class DetectionHeuristics:
    def __init__(
        self,
        model: nn.Module,
        config: Optional[DetectionConfig] = None,
        verbose: bool = True,
    ):
        self.model = model
        self.config = config or DetectionConfig()
        self.verbose = verbose

        self._loss_history = {
            'cls': deque(maxlen=50),
            'box': deque(maxlen=50),
            'obj': deque(maxlen=50),
        }
        self._detection_counts = deque(maxlen=50)

        if verbose:
            _log(f"  DetectionHeuristics initialized")

    def check(
        self,
        predictions: torch.Tensor,
        losses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        result = {
            'critical': False,
            'warnings': [],
            'metrics': {},
            'type': None,
        }

        if self.config.track_component_losses and losses:
            for loss_type, threshold in [
                ('cls', self.config.cls_loss_threshold),
                ('box', self.config.box_loss_threshold),
                ('obj', self.config.obj_loss_threshold),
            ]:
                if loss_type in losses:
                    loss_val = losses[loss_type].item() if isinstance(losses[loss_type], torch.Tensor) else losses[loss_type]
                    result['metrics'][f'{loss_type}_loss'] = loss_val
                    self._loss_history[loss_type].append(loss_val)

                    if loss_val > threshold:
                        result['critical'] = True
                        result['type'] = f'{loss_type}_loss_explosion'
                        result['warnings'].append(f"{loss_type} loss explosion ({loss_val:.2f})")

            if all(len(self._loss_history[k]) > 0 for k in ['cls', 'box']):
                cls_avg = sum(self._loss_history['cls']) / len(self._loss_history['cls'])
                box_avg = sum(self._loss_history['box']) / len(self._loss_history['box'])
                if cls_avg > 0 and box_avg / cls_avg > 100:
                    result['warnings'].append(f"Loss imbalance: box/cls ratio = {box_avg/cls_avg:.1f}")

        if self.config.track_box_predictions and predictions is not None:
            if predictions.dim() >= 2 and predictions.size(-1) >= 4:
                boxes = predictions[..., :4]

                box_max = boxes.abs().max().item()
                box_min = boxes.abs().min().item()

                result['metrics']['max_box_coord'] = box_max

                if box_max > self.config.max_box_size:
                    result['warnings'].append(f"Large box coordinates ({box_max:.2f})")

                if boxes.dim() >= 2:
                    widths = boxes[..., 2] - boxes[..., 0]
                    heights = boxes[..., 3] - boxes[..., 1]
                    if (widths < 0).any() or (heights < 0).any():
                        result['critical'] = True
                        result['type'] = 'invalid_boxes'
                        result['warnings'].append("Negative box dimensions detected")

        return result
