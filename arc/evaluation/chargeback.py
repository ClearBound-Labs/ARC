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

"""Cost-attribution / chargeback aggregation: rolls up
GPU-hours/dollars saved per team/project across many runs' exported
recovery ledgers (tagged via `ControllerConfig.team`/`.project`). This is
the chargeback report in its local-file-aggregation form; a real
multi-tenant billing integration needs a hosted billing system.
"""

import glob
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ChargebackEntry:
    team: str
    project: str
    runs: int = 0
    total_gpu_hours_saved: float = 0.0
    total_dollars_saved: float = 0.0
    total_rollbacks: int = 0
    # Ledgers exported without ControllerConfig.baseline_checkpoint_interval
    # report savings as None; they're counted here instead of guessed.
    runs_with_unknown_savings: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def aggregate_ledgers(ledgers: List[Dict[str, Any]]) -> Dict[str, ChargebackEntry]:
    """Rolls up a list of parsed (already-`json.load`ed) exported ledger
    dicts by (team, project). Entries with no team/project set are grouped
    under "unassigned"/"unassigned"."""
    rollup: Dict[str, ChargebackEntry] = {}
    for ledger in ledgers:
        team = ledger.get("team") or "unassigned"
        project = ledger.get("project") or "unassigned"
        key = f"{team}::{project}"
        entry = rollup.setdefault(key, ChargebackEntry(team=team, project=project))
        entry.runs += 1
        if ledger.get("total_gpu_hours_saved") is None:
            entry.runs_with_unknown_savings += 1
        entry.total_gpu_hours_saved += ledger.get("total_gpu_hours_saved") or 0.0
        entry.total_dollars_saved += ledger.get("total_dollars_saved") or 0.0
        entry.total_rollbacks += ledger.get("total_rollbacks", 0)
    return rollup


def aggregate_ledger_files(glob_pattern: str) -> Dict[str, ChargebackEntry]:
    """Same as `aggregate_ledgers`, but reads every file matching
    `glob_pattern` (e.g. "reports/*.json") off local disk."""
    ledgers = []
    for path in sorted(glob.glob(glob_pattern)):
        with open(path) as f:
            ledgers.append(json.load(f))
    return aggregate_ledgers(ledgers)
