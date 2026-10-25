"""Recovery ledger: rollback events get logged with
cost fields and export_recovery_ledger() produces valid, complete JSON."""
import json

import torch
import torch.nn as nn

from arc.core.controller import Controller, ControllerConfig


def test_rollback_logs_cost_fields_and_exports():
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
        baseline_checkpoint_interval=1000,
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

    action = ctrl.step(torch.tensor(float("nan")))
    assert action.rolled_back

    rollback_events = [e for e in ctrl.intervention_log if e["action"] == "rollback"]
    assert rollback_events
    event = rollback_events[-1]
    assert event["steps_lost"] >= 0
    assert event["gpu_hours_saved"] >= 0.0
    assert event["dollars_saved"] == event["gpu_hours_saved"] * config.gpu_hourly_cost

    text = ctrl.export_recovery_ledger()
    payload = json.loads(text)
    assert payload["total_rollbacks"] == ctrl.total_rollbacks
    assert len(payload["events"]) == len(ctrl.intervention_log)
    assert payload["total_dollars_saved"] == sum(e["dollars_saved"] for e in ctrl.intervention_log)


def test_reliability_roi_report_reflects_ledger_totals():
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
        gpu_power_watts=700.0,
        carbon_intensity_kg_per_kwh=0.4,
        baseline_checkpoint_interval=1000,
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

    report = ctrl.reliability_roi_report()
    gpu_hours_saved = sum(e["gpu_hours_saved"] for e in ctrl.intervention_log
                          if e.get("gpu_hours_saved") is not None)
    dollars_saved = sum(e["dollars_saved"] for e in ctrl.intervention_log
                        if e.get("dollars_saved") is not None)
    assert f"{gpu_hours_saved:.4f}" in report
    assert f"${dollars_saved:,.2f}" in report


def test_ledger_hash_chain_verifies_when_untampered():
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
    ctrl.step(torch.tensor(float("nan")))

    assert len(ctrl.intervention_log) >= 2
    assert ctrl.verify_ledger_integrity() is True


def test_ledger_hash_chain_detects_tampering():
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
    ctrl.step(torch.tensor(float("nan")))

    ctrl.intervention_log[0]["reason"] = "tampered"

    assert ctrl.verify_ledger_integrity() is False


def test_nan_or_inf_field_never_breaks_json_export():
    """NaN/Inf are valid Python floats but not valid JSON -- if any field
    the ledger logs ever ends up NaN/Inf, export must still produce
    strict-JSON-safe output (None in place of the non-finite value)
    instead of emitting a bare NaN token or raising."""
    from arc.core.controller import Controller, ControllerConfig, StepAction

    model = nn.Sequential(nn.Linear(10, 10))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    config = ControllerConfig(
        enable_lyapunov=False, enable_trend=False,
        enable_gradient_monitoring=False, enable_weight_health=False,
        async_checkpoint=False, verbose=False, log_interventions=True,
    )
    ctrl = Controller(model, optimizer, config=config)

    action = StepAction(step=1, rolled_back=False, loss_value=float("nan"))
    ctrl._log_intervention("nan_loss", action)

    assert ctrl.intervention_log[-1]["loss_value"] is None

    text = ctrl.export_recovery_ledger()  # must not raise (allow_nan=False)
    payload = json.loads(text)
    assert payload["events"][-1]["loss_value"] is None


if __name__ == "__main__":
    test_rollback_logs_cost_fields_and_exports()
    test_reliability_roi_report_reflects_ledger_totals()
    test_ledger_hash_chain_verifies_when_untampered()
    test_ledger_hash_chain_detects_tampering()
    test_nan_or_inf_field_never_breaks_json_export()
    print("ok")
