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

"""Alerting & routing: severity-aware, dedup'd delivery of
ARC events to whatever channel a customer wires up (PagerDuty, Opsgenie,
Slack, Teams, a plain webhook). This is the channel-agnostic router only
-- each channel is a plain callable the caller supplies, so there's no
SDK/API-key integration with a specific vendor here.
"""

import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, List, Optional


class Severity(IntEnum):
    INFO = 0
    WARNING = 1
    CRITICAL = 2


@dataclass
class Alert:
    severity: Severity
    title: str
    message: str
    dedup_key: str
    impact: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class AlertRouter:
    """Routes alerts to severity-appropriate channels, suppressing repeats
    of the same `dedup_key` within `dedup_window_seconds` so a flapping
    condition doesn't page someone once per step.
    """

    def __init__(
        self,
        channels: Dict[Severity, List[Callable[[Alert], None]]],
        dedup_window_seconds: float = 300.0,
    ):
        self.channels = channels
        self.dedup_window_seconds = dedup_window_seconds
        self._last_sent: Dict[str, float] = {}
        self.sent_log: List[Alert] = []

    def route(self, alert: Alert) -> bool:
        """Sends `alert` to every channel registered for its severity and
        any lower severity (a CRITICAL channel also gets WARNINGs, etc,
        matching how PagerDuty/Opsgenie escalation policies stack).
        Returns False (and sends nothing) if this dedup_key fired within
        the dedup window.
        """
        now = alert.timestamp
        last = self._last_sent.get(alert.dedup_key)
        if last is not None and (now - last) < self.dedup_window_seconds:
            return False

        for severity, handlers in self.channels.items():
            if severity <= alert.severity:
                for handler in handlers:
                    handler(alert)

        self._last_sent[alert.dedup_key] = now
        self.sent_log.append(alert)
        return True
