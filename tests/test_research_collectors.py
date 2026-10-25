"""Research signal collectors must not crash or touch training state (audit item 17)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from arc.config import SignalConfig
from arc.signals.neural_collapse import NeuralCollapseCollector
from arc.signals.sharpness import SharpnessCollector


def test_neural_collapse_collect_returns_metrics():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
    collector = NeuralCollapseCollector(SignalConfig(), num_classes=3)
    collector.attach(model, None)
    assert collector.collect().signals == {"nc": {}}  # no data yet: empty, not a crash

    model(torch.randn(30, 8))
    collector.update_batch(labels=torch.arange(30) % 3)
    snapshot = collector.collect()

    assert 0.0 <= snapshot.signals["nc"]["collapse_score"] <= 1.0


def test_sam_sharpness_leaves_gradients_untouched():
    torch.manual_seed(0)
    model = nn.Linear(8, 1)
    F.mse_loss(model(torch.randn(16, 8)), torch.randn(16, 1)).backward()  # the user's step
    grads = [p.grad.clone() for p in model.parameters()]
    weights = [p.detach().clone() for p in model.parameters()]

    x, y = torch.randn(16, 8), torch.randn(16, 1)  # a different probe batch
    SharpnessCollector(SignalConfig())._sam_perturbation_sharpness(model, F.mse_loss, x, y)

    for g, p in zip(grads, model.parameters()):
        assert torch.equal(g, p.grad)
    for w, p in zip(weights, model.parameters()):
        assert torch.equal(w, p.detach())
