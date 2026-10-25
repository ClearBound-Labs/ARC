"""No ARC loader may silently fall back to unsafe (code-executing) unpickling."""
import pickle

import pytest
import torch
import torch.nn as nn


def _plant(tmp_path, name, wrap):
    sentinel = tmp_path / f"{name}.pwned"

    class Exploit:
        def __reduce__(self):
            return (open, (str(sentinel), "w"))

    path = tmp_path / f"{name}.pt"
    torch.save(wrap(Exploit()), path)
    return path, sentinel


def test_helper_refuses_untrusted_pickle(tmp_path):
    from arc._serialization import safe_torch_load
    path, sentinel = _plant(tmp_path, "helper", lambda e: {"x": e})

    with pytest.raises(pickle.UnpicklingError):
        safe_torch_load(path)
    assert not sentinel.exists()


def test_helper_unsafe_load_is_explicit_opt_in(tmp_path):
    from arc._serialization import safe_torch_load
    path, sentinel = _plant(tmp_path, "optin", lambda e: {"x": e})

    with pytest.warns(UserWarning, match="weights_only=False"):
        safe_torch_load(path, allow_unsafe=True)
    assert sentinel.exists()


def test_failure_predictor_does_not_fall_back(tmp_path):
    from arc.prediction.predictor import FailurePredictor
    path, sentinel = _plant(tmp_path, "pred", lambda e: {"model_state_dict": e})
    fp = FailurePredictor.__new__(FailurePredictor)
    fp.device = "cpu"

    with pytest.raises(pickle.UnpicklingError):
        fp._load_model(str(path))
    assert not sentinel.exists()


def test_meta_model_trainer_does_not_fall_back(tmp_path):
    from arc.learning.trainer import MetaModelTrainer
    path, sentinel = _plant(tmp_path, "trainer", lambda e: {"model_state_dict": e})
    trainer = MetaModelTrainer.__new__(MetaModelTrainer)
    trainer.device = "cpu"

    with pytest.raises(pickle.UnpicklingError):
        trainer.load_checkpoint(str(path))
    assert not sentinel.exists()


def test_adaptive_checkpointer_does_not_fall_back(tmp_path):
    from arc.checkpointing.adaptive import AdaptiveCheckpointer, AdaptiveCheckpointConfig
    model = nn.Linear(2, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    cp = AdaptiveCheckpointer(model, opt, AdaptiveCheckpointConfig(
        verbose=False, disk_checkpoint_dir=str(tmp_path)))
    path, sentinel = _plant(tmp_path, "adaptive", lambda e: {"model": e})
    cp.checkpoints.append({"path": str(path), "step": 1})

    with pytest.raises(pickle.UnpicklingError):
        cp.restore()
    assert not sentinel.exists()


def test_arcv2_ewc_state_does_not_fall_back(tmp_path):
    from arc.api.v2 import ArcV2
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    arc = ArcV2(continual_learning=True, verbose=False)
    arc.attach(model, opt)
    _, sentinel = _plant(tmp_path, "ewc_state", lambda e: {"task_memories": e})

    with pytest.raises(pickle.UnpicklingError):
        arc.load(str(tmp_path))
    assert not sentinel.exists()
