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

"""Auto-remediation policy engine: customer-defined, auditable
rules over FailurePrediction output -- "if predicted divergence > 0.8 and
confidence > 0.6, checkpoint + reduce LR + notify" -- with a dry-run gate
so nothing fires until a customer explicitly turns it on.
"""

import datetime
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from arc.config import FailureMode
from arc.prediction.predictor import FailurePrediction


@dataclass
class RemediationPolicy:
    name: str
    min_overall_risk: float = 0.0
    min_confidence: float = 0.0
    failure_mode: Optional[FailureMode] = None  # only match if this is the highest-risk mode
    actions: List[str] = field(default_factory=list)
    dry_run: bool = True  # approval gate: no notifier runs while True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RemediationPolicy":
        """Builds a policy from a plain dict (as loaded from JSON), so a
        platform team can ship/version policies as config instead of code.
        `failure_mode`, if present, is the FailureMode member's name
        (e.g. "DIVERGENCE"), matched case-insensitively.

        Raises `ValueError` with a clear, actionable message (not a bare
        KeyError/TypeError) on an unknown failure_mode or an unexpected
        field -- this is meant to be loaded from a hand-edited config file,
        where the failure mode is a typo, not a programming error.
        """
        data = dict(data)
        mode = data.get("failure_mode")
        if mode is not None:
            try:
                data["failure_mode"] = FailureMode[mode.upper()]
            except KeyError:
                valid = [m.name for m in FailureMode]
                raise ValueError(
                    f"policy {data.get('name', '<unnamed>')!r}: unknown failure_mode "
                    f"{mode!r}, valid values: {valid}"
                ) from None
        try:
            return cls(**data)
        except TypeError as exc:
            raise ValueError(
                f"policy {data.get('name', '<unnamed>')!r}: invalid field(s) in config: {exc}"
            ) from None

    def matches(self, prediction: FailurePrediction) -> bool:
        if prediction.risk_level == "unknown":
            return False  # no model, too little history, or mismatched features
        if prediction.overall_risk < self.min_overall_risk:
            return False
        if prediction.recommendation.confidence < self.min_confidence:
            return False
        if self.failure_mode is not None:
            mode, _ = prediction.get_highest_risk_mode()
            if mode != self.failure_mode:
                return False
        return True


@dataclass
class PolicyDecision:
    policy_name: str
    triggered: bool
    dry_run: bool
    actions: List[str]
    epoch: int
    overall_risk: float
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "triggered": self.triggered,
            "dry_run": self.dry_run,
            "actions": self.actions,
            "epoch": self.epoch,
            "overall_risk": self.overall_risk,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


class PolicyEngine:
    """Evaluates policies against a FailurePrediction and, for matched
    non-dry-run policies, calls the configured notifier for each action.

    `notifiers` maps an action name (e.g. "checkpoint", "reduce_lr",
    "notify_slack") to a callable invoked with the PolicyDecision. Actions
    with no registered notifier are still recorded as triggered, just not
    executed -- the log is the audit trail either way.
    """

    def __init__(
        self,
        policies: List[RemediationPolicy],
        notifiers: Optional[Dict[str, Callable[[PolicyDecision], None]]] = None,
    ):
        self.policies = policies
        self.notifiers = notifiers or {}
        self.decision_log: List[PolicyDecision] = []

    @classmethod
    def from_config(
        cls,
        path: str,
        notifiers: Optional[Dict[str, Callable[[PolicyDecision], None]]] = None,
    ) -> "PolicyEngine":
        """Loads policies from a JSON file: a list of objects with the same
        fields as `RemediationPolicy` (`name`, `min_overall_risk`,
        `min_confidence`, `failure_mode`, `actions`, `dry_run`). Notifiers
        are still supplied in code -- they're callables, not data.
        """
        with open(path) as f:
            raw_policies = json.load(f)
        policies = []
        for i, raw_policy in enumerate(raw_policies):
            try:
                policies.append(RemediationPolicy.from_dict(raw_policy))
            except ValueError as exc:
                raise ValueError(f"{path}: policy at index {i}: {exc}") from None
        return cls(policies, notifiers=notifiers)

    def evaluate(self, prediction: FailurePrediction) -> List[PolicyDecision]:
        decisions = []
        for policy in self.policies:
            matched = policy.matches(prediction)
            decision = PolicyDecision(
                policy_name=policy.name,
                triggered=matched,
                dry_run=policy.dry_run,
                actions=list(policy.actions) if matched else [],
                epoch=prediction.epoch,
                overall_risk=prediction.overall_risk,
                confidence=prediction.recommendation.confidence,
            )
            if matched and not policy.dry_run:
                for action in policy.actions:
                    notifier = self.notifiers.get(action)
                    if notifier:
                        notifier(decision)
            decisions.append(decision)
            self.decision_log.append(decision)
        return decisions

    def export_log(self, path: Optional[str] = None) -> str:
        text = json.dumps([d.to_dict() for d in self.decision_log], indent=2)
        if path:
            with open(path, "w") as f:
                f.write(text)
        return text
