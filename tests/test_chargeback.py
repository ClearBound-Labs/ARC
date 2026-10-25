"""Cost-attribution / chargeback aggregation: rolls up
GPU-hours/dollars saved per team/project across multiple exported
ledgers, both in-memory and read off disk via glob."""
import json
import os
import tempfile

from arc.evaluation.chargeback import aggregate_ledgers, aggregate_ledger_files


def _ledger(team, project, gpu_hours=1.0, dollars=2.0, rollbacks=1):
    return {
        "team": team,
        "project": project,
        "total_gpu_hours_saved": gpu_hours,
        "total_dollars_saved": dollars,
        "total_rollbacks": rollbacks,
    }


def test_rollup_groups_by_team_and_project():
    ledgers = [
        _ledger("infra", "pretrain", gpu_hours=2.0, dollars=4.0),
        _ledger("infra", "pretrain", gpu_hours=3.0, dollars=6.0),
        _ledger("research", "rlhf", gpu_hours=1.0, dollars=2.0),
    ]

    rollup = aggregate_ledgers(ledgers)

    assert len(rollup) == 2
    infra = rollup["infra::pretrain"]
    assert infra.runs == 2
    assert infra.total_gpu_hours_saved == 5.0
    assert infra.total_dollars_saved == 10.0

    research = rollup["research::rlhf"]
    assert research.runs == 1
    assert research.total_gpu_hours_saved == 1.0


def test_untagged_ledgers_grouped_as_unassigned():
    ledgers = [{"total_gpu_hours_saved": 1.0, "total_dollars_saved": 2.0, "total_rollbacks": 1}]

    rollup = aggregate_ledgers(ledgers)

    assert "unassigned::unassigned" in rollup


def test_unknown_savings_count_as_zero_not_crash():
    ledgers = [
        _ledger("infra", "pretrain", gpu_hours=None, dollars=None),
        _ledger("infra", "pretrain", gpu_hours=2.0, dollars=4.0),
    ]

    entry = aggregate_ledgers(ledgers)["infra::pretrain"]

    assert entry.total_gpu_hours_saved == 2.0
    assert entry.total_dollars_saved == 4.0
    assert entry.runs_with_unknown_savings == 1


def test_aggregate_ledger_files_reads_from_disk():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            with open(os.path.join(tmpdir, f"run_{i}.json"), "w") as f:
                json.dump(_ledger("infra", "pretrain", gpu_hours=1.0, dollars=2.0), f)

        rollup = aggregate_ledger_files(os.path.join(tmpdir, "*.json"))

    entry = rollup["infra::pretrain"]
    assert entry.runs == 3
    assert entry.total_gpu_hours_saved == 3.0


if __name__ == "__main__":
    test_rollup_groups_by_team_and_project()
    test_untagged_ledgers_grouped_as_unassigned()
    test_aggregate_ledger_files_reads_from_disk()
    print("ok")
