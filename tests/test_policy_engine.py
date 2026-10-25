"""Auto-remediation policy engine: rules match on predicted risk
and confidence, dry_run gates whether notifiers actually fire, and every
evaluation is logged for audit/export."""
import json

import pytest

from arc.config import FailureMode
from arc.prediction.policy import PolicyEngine, RemediationPolicy
from arc.prediction.predictor import FailurePrediction, InterventionRecommendation


def _prediction(overall_risk=0.9, confidence=0.7, highest_mode=FailureMode.DIVERGENCE):
    return FailurePrediction(
        epoch=42,
        failure_probabilities={highest_mode: overall_risk},
        confidence_intervals={highest_mode: (overall_risk - 0.1, overall_risk + 0.1)},
        time_to_failure={highest_mode: 800.0},
        ttf_uncertainty={highest_mode: 50.0},
        top_contributors=[],
        recommendation=InterventionRecommendation(
            action="reduce_lr", confidence=confidence, rationale="test"
        ),
        overall_risk=overall_risk,
        risk_level="high",
    )


def test_dry_run_policy_logs_but_never_calls_notifier():
    calls = []
    policy = RemediationPolicy(
        name="divergence-guard",
        min_overall_risk=0.8,
        min_confidence=0.6,
        actions=["checkpoint", "reduce_lr"],
        dry_run=True,
    )
    engine = PolicyEngine([policy], notifiers={"checkpoint": lambda d: calls.append(d)})

    decisions = engine.evaluate(_prediction())

    assert decisions[0].triggered is True
    assert decisions[0].dry_run is True
    assert calls == []  # approval gate: dry_run must not execute anything


def test_live_policy_calls_registered_notifier():
    calls = []
    policy = RemediationPolicy(
        name="divergence-guard",
        min_overall_risk=0.8,
        min_confidence=0.6,
        actions=["checkpoint"],
        dry_run=False,
    )
    engine = PolicyEngine([policy], notifiers={"checkpoint": lambda d: calls.append(d)})

    engine.evaluate(_prediction())

    assert len(calls) == 1
    assert calls[0].policy_name == "divergence-guard"


def test_policy_below_threshold_does_not_trigger():
    policy = RemediationPolicy(name="strict", min_overall_risk=0.95, actions=["checkpoint"])
    engine = PolicyEngine([policy])

    decisions = engine.evaluate(_prediction(overall_risk=0.9))

    assert decisions[0].triggered is False
    assert decisions[0].actions == []


def test_failure_mode_filter():
    policy = RemediationPolicy(
        name="divergence-only", min_overall_risk=0.5, failure_mode=FailureMode.VANISHING_GRADIENTS,
        actions=["checkpoint"],
    )
    engine = PolicyEngine([policy])

    decisions = engine.evaluate(_prediction(highest_mode=FailureMode.DIVERGENCE))

    assert decisions[0].triggered is False


def test_export_log_is_valid_json():
    policy = RemediationPolicy(name="p", min_overall_risk=0.1, actions=["checkpoint"], dry_run=True)
    engine = PolicyEngine([policy])
    engine.evaluate(_prediction())

    payload = json.loads(engine.export_log())
    assert payload[0]["policy_name"] == "p"
    assert payload[0]["triggered"] is True


def test_from_config_loads_policies_from_json_file():
    import tempfile
    import os

    config = [
        {
            "name": "divergence-guard",
            "min_overall_risk": 0.8,
            "min_confidence": 0.6,
            "failure_mode": "DIVERGENCE",
            "actions": ["checkpoint", "reduce_lr"],
            "dry_run": True,
        }
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "policies.json")
        with open(path, "w") as f:
            json.dump(config, f)

        engine = PolicyEngine.from_config(path)

    assert len(engine.policies) == 1
    policy = engine.policies[0]
    assert policy.name == "divergence-guard"
    assert policy.failure_mode == FailureMode.DIVERGENCE

    decisions = engine.evaluate(_prediction(highest_mode=FailureMode.DIVERGENCE))
    assert decisions[0].triggered is True


def test_unknown_failure_mode_raises_clear_error():
    with pytest.raises(ValueError, match="unknown failure_mode 'DIVERGENCED'"):
        RemediationPolicy.from_dict({"name": "bad", "failure_mode": "DIVERGENCED"})


def test_unknown_field_raises_clear_error():
    with pytest.raises(ValueError, match="invalid field"):
        RemediationPolicy.from_dict({"name": "bad", "not_a_real_field": 1})


def test_from_config_error_names_which_policy_failed():
    import tempfile
    import os

    config = [
        {"name": "good", "min_overall_risk": 0.5},
        {"name": "bad", "failure_mode": "NOT_A_MODE"},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "policies.json")
        with open(path, "w") as f:
            json.dump(config, f)

        with pytest.raises(ValueError, match=r"policy at index 1"):
            PolicyEngine.from_config(path)


if __name__ == "__main__":
    test_dry_run_policy_logs_but_never_calls_notifier()
    test_live_policy_calls_registered_notifier()
    test_policy_below_threshold_does_not_trigger()
    test_failure_mode_filter()
    test_export_log_is_valid_json()
    test_from_config_loads_policies_from_json_file()
    print("ok")
