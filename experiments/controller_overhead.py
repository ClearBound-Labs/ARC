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

"""Step-time overhead of arc.Controller on a real model (GPT-2, CUDA).

Times the same training loop with and without the Controller, interleaving
the two in blocks so clock/thermal drift hits both equally. Reports the
median per-step time of each and the relative overhead.

    python experiments/controller_overhead.py --model gpt2 --steps 300
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch

# transformers otherwise imports TensorFlow when it's installed; TF and torch
# in one process crash natively on some Windows setups.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import arc  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2", choices=["gpt2", "gpt2-medium"])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--block", type=int, default=50)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=256)
    args = parser.parse_args()

    from transformers import GPT2Config, GPT2LMHeadModel

    sizes = {  # architecture only (random init), so no download is needed
        "gpt2": GPT2Config(),
        "gpt2-medium": GPT2Config(n_embd=1024, n_layer=24, n_head=16),
    }
    torch.manual_seed(0)
    cfg = sizes[args.model]
    model = GPT2LMHeadModel(cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ctrl = arc.Controller(model, opt, config=arc.ControllerConfig(verbose=False))
    ids = torch.randint(0, cfg.vocab_size, (args.batch, args.seq), device="cuda")

    def step(protected):
        loss = model(ids, labels=ids).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if not protected or not ctrl.step(loss).should_skip:
            opt.step()

    for _ in range(20):  # warm up kernels, allocator, and ARC's calibration path
        step(True)
        step(False)

    times = {True: [], False: []}
    for _ in range(args.steps // (2 * args.block)):
        for protected in (False, True):
            for _ in range(args.block):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                step(protected)
                torch.cuda.synchronize()
                times[protected].append(time.perf_counter() - t0)

    base, prot = statistics.median(times[False]), statistics.median(times[True])
    result = {
        "gpu": torch.cuda.get_device_name(),
        "model": args.model,
        "params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "batch": args.batch, "seq": args.seq,
        "baseline_ms": round(base * 1e3, 2),
        "arc_ms": round(prot * 1e3, 2),
        "overhead_percent": round((prot - base) / base * 100, 2),
        "arc_version": arc.__version__,
        "torch": torch.__version__,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
