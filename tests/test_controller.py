"""Controller smoke test: healthy steps, a crash, recovery, and the ledger."""
import torch
import torch.nn as nn

from arc.core.controller import Controller, ControllerConfig


def test_controller_smoke():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 10))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    ctrl = Controller(model, optimizer, config=ControllerConfig(
        checkpoint_frequency=5, auto_calibrate=False, verbose=False))

    def train(n):
        for _ in range(n):
            loss = model(torch.randn(8, 100)).pow(2).mean()
            optimizer.zero_grad()
            loss.backward()
            if not ctrl.step(loss).should_skip:
                optimizer.step()

    train(30)
    assert ctrl.total_rollbacks == 0

    action = ctrl.step(torch.tensor(float("nan")))
    assert action.should_skip and action.rolled_back and action.failure_type == "nan_loss"

    train(10)
    stats = ctrl.get_stats()
    assert stats["failures_detected"] == 1
    assert stats["failures_recovered"] == 1
    assert [e["action"] for e in ctrl.intervention_log] == ["rollback", "recovery_confirmed"]
    assert ctrl.verify_ledger_integrity()
