"""Regression test for ARC issue #8 — checkpoint pickle RCE.

Two layers of tests:

1. Low-level PyTorch sanity (test_malicious_pickle_blocked_by_weights_only,
   test_safe_checkpoint_loads_with_weights_only): verify the upstream
   ``torch.load(..., weights_only=True)`` flag itself rejects malicious
   pickles and accepts tensor-only checkpoints. These are baseline checks
   on the PyTorch contract ARC relies on.

2. ARC wrapper guards (test_meta_model_trainer_attempts_weights_only,
   test_failure_predictor_attempts_weights_only,
   test_adaptive_checkpointer_attempts_weights_only): spy on the
   ``torch.load`` calls made by each of ARC's three checkpoint-loading
   wrappers and verify the first attempt always uses ``weights_only=True``.
   If a future refactor drops or weakens the flag in any wrapper, the
   corresponding test fails.

There is no automatic ``weights_only=False`` fallback; unsafe loading is an
explicit opt-in (see tests/test_safe_load.py and SECURITY.md).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn


# ────────────────────────────────────────────────────────────
# Layer 1: low-level PyTorch sanity
# ────────────────────────────────────────────────────────────

def test_malicious_pickle_blocked_by_weights_only(tmp_path: Path) -> None:
    """A pickle whose ``__reduce__`` would call ``open()`` to create a
    sentinel file must NOT execute when loaded with ``weights_only=True``.

    This mirrors the proof-of-concept in issue #8: an attacker plants a
    crafted checkpoint, ARC's autonomous rollback path picks it up,
    ``torch.load()`` runs the payload as the training user. With
    ``weights_only=True`` the unpickler refuses to call arbitrary
    callables, so the sentinel is never created.
    """
    sentinel = tmp_path / "_exploit_ran.txt"
    malicious_path = tmp_path / "malicious.pt"

    class Exploit:
        """Marker class — if its ``__reduce__`` payload runs, the
        ``open()`` builtin is called and the sentinel file appears."""

        def __reduce__(self):
            return (open, (str(sentinel), "w"))

    with open(malicious_path, "wb") as f:
        pickle.dump({"model_state_dict": Exploit()}, f)

    try:
        torch.load(str(malicious_path), weights_only=True)
    except Exception:
        pass  # any refusal is fine — only the side-effect matters

    assert not sentinel.exists(), (
        f"Malicious pickle executed under weights_only=True — sentinel "
        f"{sentinel} was created by the exploit. This is a regression "
        "of issue #8; the safe-load path is broken."
    )


def test_safe_checkpoint_loads_with_weights_only(tmp_path: Path) -> None:
    """Sanity check: a tensor-only checkpoint must load cleanly under
    ``weights_only=True``. If this fails, the safe path is too strict
    and ARC's own checkpoints will hit the fallback warning."""
    safe_path = tmp_path / "safe.pt"
    expected_tensor = torch.randn(4, 4)
    torch.save({"weight": expected_tensor}, str(safe_path))

    loaded = torch.load(str(safe_path), weights_only=True)
    assert "weight" in loaded
    assert torch.equal(loaded["weight"], expected_tensor)


# ────────────────────────────────────────────────────────────
# Layer 2: ARC wrapper guards
#
# Each test spies on ``torch.load`` via ``unittest.mock.patch`` and
# verifies the first call from the wrapper used ``weights_only=True``.
# If a future refactor removes the kwarg, the spy records ``None`` and
# the assertion fails. Wrapper instances are built via ``__new__`` so
# we don't have to satisfy the full ``__init__`` of large training
# classes — only the attributes the load path actually touches.
# ────────────────────────────────────────────────────────────

def _record_torch_load_kwargs(module_dotted_path: str):
    """Build a (spy_callable, captured_kwargs_list) pair that delegates
    to the real ``torch.load`` while recording each call's kwargs.

    Used as the ``side_effect`` for ``unittest.mock.patch``. The patch
    target is the wrapper module's reference to ``torch.load`` — e.g.
    ``arc.learning.trainer.torch.load`` — so other modules are unaffected.
    """
    captured: list[dict] = []
    original_load = torch.load

    def spy(*args, **kwargs):
        captured.append(dict(kwargs))
        return original_load(*args, **kwargs)

    return spy, captured, module_dotted_path


def test_meta_model_trainer_attempts_weights_only(tmp_path: Path) -> None:
    """``MetaModelTrainer.load_checkpoint`` must call ``torch.load`` with
    ``weights_only=True`` on the first attempt. Regression guard against
    future refactors that drop or weaken the flag."""
    from arc.learning.trainer import MetaModelTrainer

    # Bypass __init__ — only set the attributes load_checkpoint reads.
    trainer = MetaModelTrainer.__new__(MetaModelTrainer)
    trainer.device = "cpu"
    trainer.model = nn.Linear(2, 2)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.01)

    # Build a checkpoint matching the structure load_checkpoint consumes.
    safe_path = tmp_path / "trainer_safe.pt"
    torch.save(
        {
            "model_state_dict": trainer.model.state_dict(),
            "optimizer_state_dict": trainer.optimizer.state_dict(),
        },
        str(safe_path),
    )

    spy, captured, _ = _record_torch_load_kwargs("arc.learning.trainer.torch.load")
    with patch("arc.learning.trainer.torch.load", side_effect=spy):
        trainer.load_checkpoint(str(safe_path))

    assert captured, "MetaModelTrainer.load_checkpoint never called torch.load"
    assert captured[0].get("weights_only") is True, (
        f"MetaModelTrainer.load_checkpoint must use weights_only=True on the "
        f"first attempt. Got first-call kwargs: {captured[0]}"
    )


def test_failure_predictor_attempts_weights_only(tmp_path: Path) -> None:
    """``FailurePredictor._load_model`` must call ``torch.load`` with
    ``weights_only=True`` on the first attempt."""
    from arc.prediction.predictor import FailurePredictor

    # Bypass __init__; _load_model only needs self.device for map_location.
    fp = FailurePredictor.__new__(FailurePredictor)
    fp.device = "cpu"

    # Minimal checkpoint — _load_model does shape inference downstream
    # which will fail on this stub, but we only care that torch.load
    # itself was invoked with weights_only=True. Subsequent failure is
    # caught by the try/except below.
    safe_path = tmp_path / "predictor_safe.pt"
    torch.save({"model_state_dict": {}}, str(safe_path))

    spy, captured, _ = _record_torch_load_kwargs("arc.prediction.predictor.torch.load")
    with patch("arc.prediction.predictor.torch.load", side_effect=spy):
        try:
            fp._load_model(str(safe_path))
        except KeyError:
            pass  # expected — stub state_dict lacks "input_proj.0.weight" for shape inference

    assert captured, "FailurePredictor._load_model never called torch.load"
    assert captured[0].get("weights_only") is True, (
        f"FailurePredictor._load_model must use weights_only=True on the "
        f"first attempt. Got first-call kwargs: {captured[0]}"
    )


def test_adaptive_checkpointer_attempts_weights_only(tmp_path: Path) -> None:
    """``AdaptiveCheckpointer.restore`` must call ``torch.load`` with
    ``weights_only=True`` on the first attempt."""
    from arc.checkpointing.adaptive import AdaptiveCheckpointer, AdaptiveCheckpointConfig

    model = nn.Linear(2, 2)
    cp = AdaptiveCheckpointer(model, torch.optim.SGD(model.parameters(), lr=0.1),
                              AdaptiveCheckpointConfig(verbose=False, disk_checkpoint_dir=str(tmp_path)))

    safe_path = tmp_path / "adaptive_safe.pt"
    torch.save(
        {
            "model_state_dict": {"weight": torch.randn(2, 2)},
            "optimizer_state_dict": {},
            "step": 0,
            "is_incremental": False,
        },
        str(safe_path),
    )
    cp.checkpoints = [{"path": str(safe_path)}]

    spy, captured, _ = _record_torch_load_kwargs("arc.checkpointing.adaptive.torch.load")
    with patch("arc.checkpointing.adaptive.torch.load", side_effect=spy):
        cp.restore(0)

    assert captured, "AdaptiveCheckpointer.restore never called torch.load"
    assert captured[0].get("weights_only") is True, (
        f"AdaptiveCheckpointer.restore must use weights_only=True on the "
        f"first attempt. Got first-call kwargs: {captured[0]}"
    )