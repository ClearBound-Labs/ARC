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

"""RL & post-training stability detectors: RLHF/RLVR/GRPO loops
fail in ways pretraining never does -- KL blowup, policy entropy death,
reward hacking, advantage explosion. This covers the detector algorithms,
which are pure statistics over scalar streams the caller already has
(policy KL, entropy, reward, an optional held-out true-objective probe,
advantage), same shape as `signals/trend.py`'s TrendDetector. Multi-model
(policy+reward+reference+critic) coordinated checkpoint/rollback is a
separate, unbuilt piece -- it needs a real RLHF loop design partner to
validate the coordination semantics, not just the math.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from arc.signals.stats_utils import ols_slope


@dataclass
class RLStabilityReport:
    kl_blowup: bool = False
    entropy_collapse: bool = False
    reward_hacking: bool = False
    advantage_explosion: bool = False

    kl_value: Optional[float] = None
    entropy_value: Optional[float] = None
    entropy_slope: float = 0.0
    reward_slope: float = 0.0
    true_objective_slope: float = 0.0
    advantage_zscore: float = 0.0

    messages: list = field(default_factory=list)

    @property
    def is_healthy(self) -> bool:
        return not (
            self.kl_blowup or self.entropy_collapse
            or self.reward_hacking or self.advantage_explosion
        )


class RLStabilityMonitor:
    def __init__(
        self,
        window: int = 50,
        kl_threshold: float = 0.1,
        entropy_floor: float = 0.01,
        entropy_slope_threshold: float = -1e-3,
        advantage_zscore_threshold: float = 5.0,
        reward_hacking_slope_gap: float = 1e-4,
    ):
        self.kl_threshold = kl_threshold
        self.entropy_floor = entropy_floor
        self.entropy_slope_threshold = entropy_slope_threshold
        self.advantage_zscore_threshold = advantage_zscore_threshold
        self.reward_hacking_slope_gap = reward_hacking_slope_gap

        self._kl: deque = deque(maxlen=window)
        self._entropy: deque = deque(maxlen=window)
        self._reward: deque = deque(maxlen=window)
        self._true_objective: deque = deque(maxlen=window)
        self._advantage: deque = deque(maxlen=window)

    def update(
        self,
        kl: Optional[float] = None,
        entropy: Optional[float] = None,
        reward: Optional[float] = None,
        true_objective: Optional[float] = None,
        advantage: Optional[float] = None,
    ) -> RLStabilityReport:
        if kl is not None:
            self._kl.append(kl)
        if entropy is not None:
            self._entropy.append(entropy)
        if reward is not None:
            self._reward.append(reward)
        if true_objective is not None:
            self._true_objective.append(true_objective)
        if advantage is not None:
            self._advantage.append(advantage)

        report = RLStabilityReport()

        if self._kl:
            report.kl_value = self._kl[-1]
            if self._kl[-1] > self.kl_threshold:
                report.kl_blowup = True
                report.messages.append(
                    f"KL divergence {self._kl[-1]:.4f} exceeds threshold {self.kl_threshold}"
                )

        if self._entropy:
            report.entropy_value = self._entropy[-1]
            report.entropy_slope = ols_slope(list(self._entropy))
            if (
                self._entropy[-1] < self.entropy_floor
                or report.entropy_slope < self.entropy_slope_threshold
            ):
                report.entropy_collapse = True
                report.messages.append(
                    f"Policy entropy {self._entropy[-1]:.4f} collapsing "
                    f"(slope {report.entropy_slope:.2e})"
                )

        if len(self._reward) >= 2 and len(self._true_objective) >= 2:
            report.reward_slope = ols_slope(list(self._reward))
            report.true_objective_slope = ols_slope(list(self._true_objective))
            if (
                report.reward_slope > self.reward_hacking_slope_gap
                and report.true_objective_slope < -self.reward_hacking_slope_gap
            ):
                report.reward_hacking = True
                report.messages.append(
                    f"Reward climbing (slope {report.reward_slope:.2e}) while true "
                    f"objective degrades (slope {report.true_objective_slope:.2e})"
                )

        if len(self._advantage) >= 5:
            history = np.asarray(list(self._advantage)[:-1], dtype=np.float64)
            mean, std = history.mean(), history.std()
            if std > 1e-8:
                z = abs((self._advantage[-1] - mean) / std)
                report.advantage_zscore = float(z)
                if z > self.advantage_zscore_threshold:
                    report.advantage_explosion = True
                    report.messages.append(
                        f"Advantage z-score {z:.2f} exceeds threshold "
                        f"{self.advantage_zscore_threshold}"
                    )

        return report
