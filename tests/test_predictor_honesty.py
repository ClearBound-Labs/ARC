"""The failure predictor must never report risk from untrained or mismatched
weights (audit item 3)."""
import torch

from arc.config import Config, FailureMode
from arc.learning.meta_model import TrainingDynamicsPredictor
from arc.prediction.policy import PolicyEngine, RemediationPolicy
from arc.prediction.predictor import FailurePredictor


def _feed(predictor, epochs=5):
    for epoch in range(epochs):
        predictor.update({"loss": {"epoch": {"train_loss": 1.0 / (epoch + 1)}},
                          "gradient": {"global": {"total_grad_norm_l2": 0.5 + epoch}}}, epoch)


def test_predictor_without_trained_model_reports_unknown():
    predictor = FailurePredictor()
    _feed(predictor)

    prediction = predictor.predict()

    assert prediction.risk_level == "unknown"
    assert prediction.overall_risk == 0.0
    assert all(p == 0.0 for p in prediction.failure_probabilities.values())
    assert prediction.recommendation.action == "no_action"


def test_model_with_mismatched_features_reports_unknown(tmp_path):
    model = TrainingDynamicsPredictor(config=Config().prediction, n_features=7)
    path = tmp_path / "predictor.pt"
    torch.save({"model_state_dict": model.state_dict()}, path)
    predictor = FailurePredictor(model_path=str(path))
    _feed(predictor)

    prediction = predictor.predict()

    assert prediction.risk_level == "unknown"


def test_policies_never_fire_on_unknown_predictions():
    predictor = FailurePredictor()
    _feed(predictor)
    fired = []
    engine = PolicyEngine(
        [RemediationPolicy(name="any", min_overall_risk=0.0, actions=["notify"], dry_run=False)],
        notifiers={"notify": fired.append},
    )

    decisions = engine.evaluate(predictor.predict())

    assert fired == []
    assert decisions[0].triggered is False
