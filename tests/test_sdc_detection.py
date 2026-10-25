"""SDC detection: cross-replica gradient agreement flags the
one rank silently computing wrong numbers, and deterministic replay
verification catches a model that can't reproduce its own forward pass."""
import torch
import torch.nn as nn
import pytest

from arc.security.sdc_detection import CrossReplicaGradientCheck, DeterministicReplayVerifier


def test_healthy_ranks_agree():
    check = CrossReplicaGradientCheck(relative_deviation_threshold=3.0)
    base = torch.randn(100)
    grads = {r: base + torch.randn(100) * 0.01 for r in range(4)}  # tiny batch noise

    alert = check.check(grads)

    assert alert.is_corrupted is False
    assert alert.suspect_ranks == []


def test_one_corrupted_rank_flagged():
    check = CrossReplicaGradientCheck(relative_deviation_threshold=3.0)
    base = torch.randn(100)
    grads = {r: base + torch.randn(100) * 0.01 for r in range(4)}
    grads[2] = base + 50.0  # simulated silent-corruption: wildly wrong gradient

    alert = check.check(grads)

    assert alert.is_corrupted is True
    assert alert.suspect_ranks == [2]


def test_requires_at_least_three_ranks():
    check = CrossReplicaGradientCheck()
    with pytest.raises(ValueError, match=">= 3 ranks"):
        check.check({0: torch.randn(10), 1: torch.randn(10)})


def test_replay_matches_reference_for_deterministic_model():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(10, 10)).eval()
    x = torch.randn(4, 10)
    verifier = DeterministicReplayVerifier(tolerance=1e-5)

    alert = verifier.verify(model, x)

    assert alert.is_corrupted is False


def test_replay_flags_mismatched_reference():
    model = nn.Sequential(nn.Linear(10, 10)).eval()
    x = torch.randn(4, 10)
    verifier = DeterministicReplayVerifier(tolerance=1e-5)
    wrong_reference = torch.ones(4, 10) * 999.0

    alert = verifier.verify(model, x, reference_output=wrong_reference)

    assert alert.is_corrupted is True


def test_empty_input_raises_clear_error():
    model = nn.Sequential(nn.Linear(10, 10)).eval()
    empty_input = torch.empty(0, 10)
    verifier = DeterministicReplayVerifier()

    with pytest.raises(ValueError, match="empty input"):
        verifier.verify(model, empty_input)


if __name__ == "__main__":
    test_healthy_ranks_agree()
    test_one_corrupted_rank_flagged()
    test_replay_matches_reference_for_deterministic_model()
    test_replay_flags_mismatched_reference()
    print("ok")
