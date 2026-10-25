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

"""Silent Data Corruption detection: a GPU computing wrong
numbers with no error flag poisons gradients for hours before a loss spike
makes it visible. This module implements the part of SDC detection that's a
pure numerical algorithm -- cross-replica gradient agreement -- rather than
hardware-telemetry fusion, which needs real ECC/XID
counters from a live cluster to validate.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch


@dataclass
class SDCAlert:
    is_corrupted: bool
    suspect_ranks: List[int]
    max_relative_deviation: float
    per_rank_deviation: Dict[int, float]
    detail: str = ""


class CrossReplicaGradientCheck:
    """Flags a rank whose gradient diverges from the cross-replica consensus.

    In healthy data-parallel training, every rank computes gradients over a
    different micro-batch but the same model, so gradients should agree up
    to normal batch-to-batch variance. A rank silently computing wrong
    numbers (bit-flip, degraded ALU) produces a gradient that diverges far
    more than sampling noise explains. Consensus is the coordinate-wise
    median across ranks (robust to exactly the single-outlier-rank case
    this check exists for; a mean would itself be pulled toward the
    corrupted value).
    """

    def __init__(self, relative_deviation_threshold: float = 3.0):
        self.relative_deviation_threshold = relative_deviation_threshold

    def check(self, rank_gradients: Dict[int, torch.Tensor]) -> SDCAlert:
        if len(rank_gradients) < 3:
            raise ValueError(
                "Cross-replica agreement needs >= 3 ranks to form a robust "
                "consensus (median) that isn't itself swayed by the one "
                "possibly-corrupted rank."
            )

        ranks = list(rank_gradients.keys())
        stacked = torch.stack([rank_gradients[r].flatten().float() for r in ranks])

        consensus = stacked.median(dim=0).values
        consensus_norm = consensus.norm().item()
        # Guard against a near-zero consensus norm (e.g. at init) making the
        # relative deviation blow up on noise alone.
        scale = max(consensus_norm, 1e-8)

        per_rank_deviation: Dict[int, float] = {}
        suspect_ranks: List[int] = []
        for i, r in enumerate(ranks):
            deviation = (stacked[i] - consensus).norm().item() / scale
            per_rank_deviation[r] = deviation
            if deviation > self.relative_deviation_threshold:
                suspect_ranks.append(r)

        max_deviation = max(per_rank_deviation.values())
        is_corrupted = len(suspect_ranks) > 0

        detail = (
            f"rank(s) {suspect_ranks} deviate {max_deviation:.2f}x consensus "
            f"(threshold {self.relative_deviation_threshold}x)"
            if is_corrupted else "all ranks within consensus tolerance"
        )

        return SDCAlert(
            is_corrupted=is_corrupted,
            suspect_ranks=suspect_ranks,
            max_relative_deviation=max_deviation,
            per_rank_deviation=per_rank_deviation,
            detail=detail,
        )


class DeterministicReplayVerifier:
    """Re-runs the same input through the model and compares outputs, to
    catch nondeterministic hardware drift (a healthy deterministic model
    should reproduce its own forward pass on the same input/weights).
    """

    def __init__(self, tolerance: float = 1e-5):
        self.tolerance = tolerance

    def verify(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        reference_output: Optional[torch.Tensor] = None,
    ) -> SDCAlert:
        if inputs.numel() == 0:
            raise ValueError(
                "DeterministicReplayVerifier.verify() got an empty input "
                f"tensor (shape {tuple(inputs.shape)}) -- there's nothing to replay."
            )

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                replay_output = model(inputs)
                if reference_output is None:
                    reference_output = model(inputs)
        finally:
            model.train(was_training)

        diff = (replay_output - reference_output).abs().max().item()
        scale = max(reference_output.abs().max().item(), 1e-8)
        relative_diff = diff / scale

        is_corrupted = relative_diff > self.tolerance
        detail = (
            f"replay differs by {relative_diff:.2e} (tolerance {self.tolerance:.0e})"
            if is_corrupted else "replay matches reference within tolerance"
        )

        return SDCAlert(
            is_corrupted=is_corrupted,
            suspect_ranks=[],
            max_relative_deviation=relative_diff,
            per_rank_deviation={},
            detail=detail,
        )
