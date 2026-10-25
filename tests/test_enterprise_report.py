"""Unified EnterpriseReport: aggregates ledger + ROI report +
policy decisions + alerts into one JSON blob and one self-contained HTML
page with no external resources."""
import json
import tempfile
import os

import torch
import torch.nn as nn

from arc.core.controller import Controller, ControllerConfig
from arc.api.enterprise_report import EnterpriseReport
from arc.api.alerting import Alert, AlertRouter, Severity
from arc.prediction.policy import PolicyEngine, RemediationPolicy
from arc.prediction.predictor import FailurePrediction, InterventionRecommendation
from arc.config import FailureMode


def _controller_with_rollback(baseline_checkpoint_interval=1000):
    model = nn.Sequential(nn.Linear(10, 10))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    config = ControllerConfig(
        checkpoint_frequency=2,
        enable_lyapunov=False,
        enable_trend=False,
        enable_gradient_monitoring=False,
        enable_weight_health=False,
        async_checkpoint=False,
        verbose=False,
        gpu_count=8,
        gpu_hourly_cost=2.0,
        baseline_checkpoint_interval=baseline_checkpoint_interval,
    )
    ctrl = Controller(model, optimizer, config=config)
    for _ in range(5):
        x = torch.randn(4, 10)
        loss = model(x).mean()
        loss.backward()
        action = ctrl.step(loss)
        if not action.should_skip:
            optimizer.step()
        optimizer.zero_grad()
    ctrl.step(torch.tensor(float("nan")))
    return ctrl


def test_to_dict_contains_ledger_and_roi():
    ctrl = _controller_with_rollback()
    report = EnterpriseReport(ctrl)

    data = report.to_dict()

    assert "ledger" in data
    assert "roi_report" in data
    assert data["ledger"]["total_rollbacks"] == ctrl.total_rollbacks


def test_to_dict_includes_policy_and_alerts_when_provided():
    ctrl = _controller_with_rollback()

    policy = RemediationPolicy(name="p", min_overall_risk=0.1, actions=["checkpoint"])
    engine = PolicyEngine([policy])
    prediction = FailurePrediction(
        epoch=1,
        failure_probabilities={FailureMode.DIVERGENCE: 0.9},
        confidence_intervals={FailureMode.DIVERGENCE: (0.8, 1.0)},
        time_to_failure={FailureMode.DIVERGENCE: 10.0},
        ttf_uncertainty={FailureMode.DIVERGENCE: 1.0},
        top_contributors=[],
        recommendation=InterventionRecommendation(action="reduce_lr", confidence=0.9, rationale="x"),
        overall_risk=0.9,
        risk_level="high",
    )
    engine.evaluate(prediction)

    router = AlertRouter(channels={Severity.INFO: [lambda a: None]})
    router.route(Alert(Severity.INFO, "t", "m", dedup_key="k1"))

    report = EnterpriseReport(ctrl, policy_engine=engine, alert_router=router)
    data = report.to_dict()

    assert len(data["policy_decisions"]) == 1
    assert len(data["alerts"]) == 1


def test_to_html_is_self_contained_and_valid():
    ctrl = _controller_with_rollback()
    report = EnterpriseReport(ctrl)

    page = report.to_html()

    assert "<html>" in page
    assert "<svg" in page
    assert "http://" not in page.replace("http://www.w3.org/2000/svg", "")  # no external refs beyond the SVG namespace URI


def test_html_renders_when_savings_are_unknown():
    ctrl = _controller_with_rollback(baseline_checkpoint_interval=0)

    page = EnterpriseReport(ctrl).to_html()

    assert "baseline_checkpoint_interval" in page


def test_save_writes_both_files():
    ctrl = _controller_with_rollback()
    report = EnterpriseReport(ctrl)

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "report.json")
        html_path = os.path.join(tmpdir, "report.html")
        report.save(json_path=json_path, html_path=html_path)

        with open(json_path) as f:
            data = json.load(f)
        with open(html_path) as f:
            page = f.read()

    assert "ledger" in data
    assert "<html>" in page


if __name__ == "__main__":
    test_to_dict_contains_ledger_and_roi()
    test_to_dict_includes_policy_and_alerts_when_provided()
    test_to_html_is_self_contained_and_valid()
    test_save_writes_both_files()
    print("ok")
