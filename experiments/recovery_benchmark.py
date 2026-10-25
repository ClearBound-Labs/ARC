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

"""Symmetric recovery benchmark for arc.Controller.

Both runs of every trial (without ARC, with ARC) use the same model, data,
fault and seed, and are judged by the same outcome checks after training:

    crashed    an exception escaped the training loop
    corrupted  some weight is NaN/Inf
    diverged   held-out loss is non-finite or more than 2x its value just
               before the fault
    healthy    none of the above

Transient faults hit for one step or a short burst; a healthy outcome means
training recovered. Persistent faults (bad data every step) can't be fixed by
rolling back the model: the honest best outcome is a model left intact and
the run flagged unrecoverable.

    python experiments/recovery_benchmark.py --seeds 3
"""

import argparse
import json
import math
import os
import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arc  # noqa: E402

STEPS, TRIGGER, BATCH = 150, 60, 32


def make_task(seed):
    gen = torch.Generator().manual_seed(seed)
    teacher = torch.randn(100, 10, generator=gen)
    x = torch.randn(STEPS * BATCH, 100, generator=gen)
    x_val = torch.randn(512, 100, generator=gen)
    return x, (x @ teacher).argmax(1), x_val, (x_val @ teacher).argmax(1)


def make_model(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(100, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                         nn.Linear(256, 10))


def eval_loss(model, x_val, y_val):
    with torch.no_grad():
        return F.cross_entropy(model(x_val), y_val).item()


# --- faults: (phase, function); phase is when the fault is applied ------------
# "loss": f(loss, step) -> loss            "weights"/"optim": f(model, opt, step)
# "grads": f(model, step) after backward

def _nan_weights(model, opt, step):
    if step == TRIGGER:
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(float("nan"))


def _nuclear_lr(model, opt, step):
    if step == TRIGGER:
        for g in opt.param_groups:
            g["lr"] = 1e6


def _nan_optimizer_state(model, opt, step):
    if step == TRIGGER:
        for state in opt.state.values():
            for v in state.values():
                if torch.is_tensor(v) and v.dim() > 0:
                    v.fill_(float("nan"))


def _zero_half_weights(model, opt, step):
    if step == TRIGGER:
        with torch.no_grad():
            for p in model.parameters():
                p[torch.rand_like(p) < 0.5] = 0.0


def _everything(model, opt, step):
    if step == TRIGGER:
        with torch.no_grad():
            p = next(model.parameters())
            p[torch.rand_like(p) < 0.1] = float("nan")
        for g in opt.param_groups:
            g["lr"] *= 100


def _grads(fill):
    def apply(model, step):
        if step == TRIGGER:
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is not None:
                        fill(p.grad, step)
    return apply


def _byzantine(g, step):
    mask = torch.rand_like(g) < 0.5
    g[mask] = torch.randn_like(g)[mask] * 1e5


def _minefield(loss, step):
    hit = TRIGGER <= step < TRIGGER + 20 and torch.rand(1).item() < 0.3
    return loss * float("inf") if hit else loss


TRANSIENT = {
    "NaN loss": ("loss", lambda loss, s: loss * float("nan") if s == TRIGGER else loss),
    "Division by zero": ("loss", lambda loss, s: loss / 0.0 if s == TRIGGER else loss),
    "Loss spike x1e10": ("loss", lambda loss, s: loss * 1e10 if s == TRIGGER else loss),
    "Domino (5 Inf losses)": ("loss", lambda loss, s: loss * float("inf") if TRIGGER <= s < TRIGGER + 5 else loss),
    "Repeated (10 Inf in 20 steps)": ("loss", lambda loss, s: loss * float("inf") if TRIGGER <= s < TRIGGER + 20 and s % 2 == 0 else loss),
    "Loss minefield (30% Inf, 20 steps)": ("loss", _minefield),
    "All weights NaN": ("weights", _nan_weights),
    "Nuclear LR (1e6)": ("weights", _nuclear_lr),
    "Optimizer state NaN": ("weights", _nan_optimizer_state),
    "Half the weights zeroed": ("weights", _zero_half_weights),
    "Everything at once": ("weights", _everything),
    "Inf gradients": ("grads", _grads(lambda g, s: g.fill_(float("inf")))),
    "Gradient whiplash (1e6)": ("grads", _grads(lambda g, s: g.fill_(1e6 if s % 2 else -1e6))),
    "Byzantine gradients (50% x1e5)": ("grads", _grads(_byzantine)),
}

CONTROL = {
    "No fault": ("loss", lambda loss, s: loss),
}

PERSISTENT = {
    "NaN in every batch from step 60": ("loss", lambda loss, s: loss * float("nan") if s >= TRIGGER else loss),
    "Batch bomb every 10th step (x1e15)": ("loss", lambda loss, s: loss * 1e15 if s >= TRIGGER and s % 10 == 0 else loss),
}


def run(fault, seed, use_arc):
    phase, fn = fault
    x, y, x_val, y_val = make_task(seed)
    model = make_model(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ctrl = arc.Controller(model, opt, config=arc.ControllerConfig(verbose=False)) if use_arc else None
    torch.manual_seed(seed + 1000)  # same fault randomness for both runs

    reference = None
    try:
        for step in range(1, STEPS + 1):
            if step == TRIGGER:
                reference = eval_loss(model, x_val, y_val)
            if phase == "weights":
                fn(model, opt, step)
            xb = x[(step - 1) * BATCH: step * BATCH]
            yb = y[(step - 1) * BATCH: step * BATCH]
            loss = F.cross_entropy(model(xb), yb)
            if phase == "loss":
                loss = fn(loss, step)
            opt.zero_grad()
            loss.backward()
            if phase == "grads":
                fn(model, step)
            if ctrl is not None and ctrl.step(loss).should_skip:
                continue
            opt.step()
    except Exception as e:  # noqa: BLE001 -- the benchmark records any crash
        return {"outcome": "crashed", "error": str(e)[:80]}

    if not all(torch.isfinite(p).all() for p in model.parameters()):
        outcome = "corrupted"
    else:
        final = eval_loss(model, x_val, y_val)
        outcome = "diverged" if not math.isfinite(final) or final > 2 * reference else "healthy"
    result = {"outcome": outcome}
    if ctrl is not None:
        stats = ctrl.get_stats()
        result.update(rollbacks=stats["total_rollbacks"], unrecoverable=ctrl.unrecoverable,
                      recovered=stats["failures_recovered"], detected=stats["failures_detected"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                      "recovery_benchmark_results.json"))
    args = parser.parse_args()
    warnings.simplefilter("ignore")

    rows = []
    for kind, faults in (("control", CONTROL), ("transient", TRANSIENT),
                         ("persistent", PERSISTENT)):
        for name, fault in faults.items():
            for seed in range(args.seeds):
                rows.append({"kind": kind, "fault": name, "seed": seed,
                             "baseline": run(fault, seed, use_arc=False),
                             "arc": run(fault, seed, use_arc=True)})
                b, a = rows[-1]["baseline"], rows[-1]["arc"]
                print(f"{kind:10s} {name:38s} seed {seed}: baseline={b['outcome']:9s} "
                      f"arc={a['outcome']:9s} rollbacks={a.get('rollbacks')} "
                      f"unrecoverable={a.get('unrecoverable')}")

    def count(kind, who, outcome):
        return sum(1 for r in rows if r["kind"] == kind and r[who]["outcome"] == outcome)

    summary = {}
    for kind in ("control", "transient", "persistent"):
        n = sum(1 for r in rows if r["kind"] == kind)
        summary[kind] = {
            "trials": n,
            "baseline_healthy": count(kind, "baseline", "healthy"),
            "arc_healthy": count(kind, "arc", "healthy"),
            "arc_intact": sum(1 for r in rows if r["kind"] == kind
                              and r["arc"]["outcome"] in ("healthy", "diverged")),
            "arc_flagged_unrecoverable": sum(1 for r in rows if r["kind"] == kind
                                             and r["arc"].get("unrecoverable")),
            "arc_rollbacks": sum(r["arc"].get("rollbacks", 0) for r in rows if r["kind"] == kind),
            "trials_arc_intervened_but_baseline_survived": sum(
                1 for r in rows if r["kind"] == kind and r["baseline"]["outcome"] == "healthy"
                and r["arc"].get("rollbacks", 0) > 0),
        }

    print(json.dumps(summary, indent=2))
    with open(args.out, "w") as f:
        json.dump({"config": {"steps": STEPS, "trigger": TRIGGER, "batch": BATCH,
                              "seeds": args.seeds, "arc_version": arc.__version__},
                   "summary": summary, "trials": rows}, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
