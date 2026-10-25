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

"""Checkpoint restore drill / disaster-recovery self-test:
proves the checkpoint you'd rely on in an incident actually restores
cleanly, before you need it. Restores into a *fresh* model/optimizer
instance (not the live training ones), so the drill can run mid-training
without disturbing it.
"""

import torch
from dataclasses import dataclass
from typing import Callable

from arc.checkpointing.adaptive import AdaptiveCheckpointer


@dataclass
class RecoveryDrillReport:
    passed: bool
    max_abs_diff: float
    tolerance: float
    detail: str


def verify_recoverability(
    checkpointer: AdaptiveCheckpointer,
    model_factory: Callable[[], torch.nn.Module],
    optimizer_factory: Callable[[torch.nn.Module], torch.optim.Optimizer],
    sample_input: torch.Tensor,
    tolerance: float = 1e-4,
) -> RecoveryDrillReport:
    """Saves a checkpoint of `checkpointer`'s current state, restores it
    into a throwaway model built by `model_factory`/`optimizer_factory`,
    and checks the throwaway model's output on `sample_input` matches the
    live model's output within `tolerance`.
    """
    was_training = checkpointer.model.training
    checkpointer.model.eval()
    try:
        with torch.no_grad():
            reference_output = checkpointer.model(sample_input).clone()
    finally:
        checkpointer.model.train(was_training)

    checkpointer.save()

    drill_model = model_factory()
    drill_optimizer = optimizer_factory(drill_model)
    drill_checkpointer = AdaptiveCheckpointer(drill_model, drill_optimizer, config=checkpointer.config)
    drill_checkpointer.checkpoints = checkpointer.checkpoints
    drill_checkpointer.step = checkpointer.step
    drill_checkpointer.restore()

    drill_model.eval()
    with torch.no_grad():
        restored_output = drill_model(sample_input)

    max_abs_diff = (restored_output - reference_output).abs().max().item()
    passed = max_abs_diff <= tolerance
    detail = (
        f"restored output matches within tolerance (max abs diff {max_abs_diff:.2e})"
        if passed else
        f"restored output diverges: max abs diff {max_abs_diff:.2e} exceeds tolerance {tolerance:.2e}"
    )

    return RecoveryDrillReport(
        passed=passed, max_abs_diff=max_abs_diff, tolerance=tolerance, detail=detail,
    )
