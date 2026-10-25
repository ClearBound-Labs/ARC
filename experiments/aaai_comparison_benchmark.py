"""
ARC AAAI 2027 — Comparison & Failure Diversity Benchmark
=========================================================
Phase 1: ARC vs 4 baselines (gradient clipping, manual checkpoint, GradScaler, no protection)
Phase 2: 6 failure types on GPT-2 XL with ARC
Run on H200/H100. ~1 hour total.
"""
import torch, torch.nn as nn, torch.optim as optim
import numpy as np, json, time, os, zipfile, gc, copy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "aaai_benchmark_results"; os.makedirs(OUT, exist_ok=True)

def save(n, d):
    p = os.path.join(OUT, f"{n}.json")
    with open(p, "w") as f: json.dump(d, f, indent=2, default=str)
    print(f"  Saved {p}")

def seed(s=42):
    torch.manual_seed(s); np.random.seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

print(f"Device: {DEVICE} | PyTorch: {torch.__version__}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

# Install
try:
    from arc.intervention import WeightRollback, RollbackConfig
except ImportError:
    os.system("pip install arc-training --upgrade -q")
    from arc.intervention import WeightRollback, RollbackConfig

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
except ImportError:
    os.system("pip install transformers -q")
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ====================================================================
# FAILURE INJECTION FUNCTIONS
# ====================================================================
def inject_nan(model, opt, loss, step, **kw):
    """Simulate hardware NaN error"""
    return loss * float("nan")

def inject_grad_explosion(model, opt, loss, step, **kw):
    """Simulate gradient explosion from bad data"""
    return loss * 1e6

def inject_weight_corruption(model, opt, loss, step, **kw):
    """Simulate bit-flip / AMP overflow corrupting weights"""
    with torch.no_grad():
        for p in list(model.parameters())[:5]:
            p.mul_(1e4)
    return loss  # loss itself is fine — the corruption is silent

def inject_lr_spike(model, opt, loss, step, **kw):
    """Simulate learning rate scheduler bug"""
    for pg in opt.param_groups:
        pg['lr'] *= 1000
    return loss

def inject_loss_spike(model, opt, loss, step, **kw):
    """Simulate corrupted data batch causing loss spike"""
    return loss + 1e5

def inject_silent_noise(model, opt, loss, step, **kw):
    """Simulate subtle numerical drift (silent model degradation)"""
    with torch.no_grad():
        for p in list(model.parameters())[:10]:
            p.add_(torch.randn_like(p) * 0.5)
    return loss

FAILURE_TYPES = {
    "nan": {"fn": inject_nan, "desc": "NaN injection (hardware error)"},
    "grad_explosion": {"fn": inject_grad_explosion, "desc": "Gradient explosion (bad LR)"},
    "weight_corruption": {"fn": inject_weight_corruption, "desc": "Weight corruption (bit flip)"},
    "lr_spike": {"fn": inject_lr_spike, "desc": "LR spike (scheduler bug)"},
    "loss_spike": {"fn": inject_loss_spike, "desc": "Loss spike (bad data)"},
    "silent_noise": {"fn": inject_silent_noise, "desc": "Silent noise (numerical drift)"},
}

# ====================================================================
# TRAINING LOOP (supports all protection methods)
# ====================================================================
def train_run(model_name, model, tokenizer, protection, failure_type, failure_fn,
              label, steps=1000, batch_size=4, max_len=128, fail_step=500,
              ckpt_freq=250):
    """
    protection: "none", "grad_clip", "manual_ckpt", "arc"
    failure_fn: callable or None
    """
    npar = sum(p.numel() for p in model.parameters())
    print(f"\n  [{label}] {model_name} | {npar/1e6:.0f}M params | prot={protection} | fail={failure_type}")

    model.to(DEVICE).train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    texts = ["The quick brown fox jumps over the lazy dog. Training large language models "
             "requires careful monitoring for numerical failures and gradient explosions. " * 3] * 2000
    enc = tokenizer(texts, return_tensors="pt", max_length=max_len,
                    truncation=True, padding="max_length")
    dataset = torch.utils.data.TensorDataset(enc["input_ids"], enc["attention_mask"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    # Setup protection method
    rb = None
    manual_ckpts = {}
    if protection == "arc":
        rb = WeightRollback(model, opt, RollbackConfig(
            checkpoint_frequency=10, max_checkpoints=3
        ), verbose=True)
    elif protection == "grad_clip":
        pass  # clip applied in loop
    elif protection == "manual_ckpt":
        pass  # save in loop

    hist = {"losses": [], "rollbacks": 0, "crashed": False, "label": label,
            "protection": protection, "failure_type": failure_type,
            "model": model_name, "params": npar, "steps_lost": 0}
    t0 = time.time()
    step = 0
    done = False

    while not done:
        for ids, mask in loader:
            if step >= steps:
                done = True; break
            ids, mask = ids.to(DEVICE), mask.to(DEVICE)

            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                out = model(ids, attention_mask=mask, labels=ids)
                loss = out.loss

            # Inject failure at specified step
            if failure_fn and step == fail_step:
                print(f"  [{label}] >>> INJECTING {failure_type} at step {step} <<<")
                loss = failure_fn(model, opt, loss, step)

            bad = torch.isnan(loss) or torch.isinf(loss)

            # ---- PROTECTION: ARC ----
            if protection == "arc" and rb:
                try:
                    if not bad:
                        loss.backward()
                except Exception:
                    pass
                action = rb.step(loss)
                if action.rolled_back:
                    hist["rollbacks"] += 1
                    hist["steps_lost"] += 1
                    opt.zero_grad()
                    step += 1; continue
                if not bad:
                    opt.step()
                opt.zero_grad()

            # ---- PROTECTION: GRADIENT CLIPPING ----
            elif protection == "grad_clip":
                if bad:
                    hist["crashed"] = True
                    hist["crash_step"] = step
                    print(f"  [{label}] CRASHED at step {step} (grad clip can't fix NaN)")
                    done = True; break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step(); opt.zero_grad()

            # ---- PROTECTION: MANUAL CHECKPOINT ----
            elif protection == "manual_ckpt":
                if bad:
                    if manual_ckpts:
                        ckpt_step = max(manual_ckpts.keys())
                        model.load_state_dict(manual_ckpts[ckpt_step]["model"])
                        opt.load_state_dict(manual_ckpts[ckpt_step]["opt"])
                        hist["rollbacks"] += 1
                        hist["steps_lost"] += (step - ckpt_step)
                        print(f"  [{label}] Restored manual checkpoint from step {ckpt_step} "
                              f"(lost {step - ckpt_step} steps)")
                        step += 1; continue
                    else:
                        hist["crashed"] = True
                        hist["crash_step"] = step
                        print(f"  [{label}] CRASHED (no checkpoint available)")
                        done = True; break
                loss.backward()
                opt.step(); opt.zero_grad()
                if step > 0 and step % ckpt_freq == 0:
                    manual_ckpts[step] = {
                        "model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                        "opt": {k: (v.cpu().clone() if isinstance(v, torch.Tensor) else v)
                                for k, v in opt.state_dict().items()},
                    }
                    print(f"  [{label}] Manual checkpoint saved at step {step}")

            # ---- NO PROTECTION ----
            else:
                if bad:
                    hist["crashed"] = True
                    hist["crash_step"] = step
                    print(f"  [{label}] CRASHED at step {step}")
                    done = True; break
                loss.backward()
                opt.step(); opt.zero_grad()

            if step % 100 == 0:
                lv = round(loss.item(), 4) if not bad else "NaN"
                hist["losses"].append({"step": step, "loss": lv})
                mem = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0
                print(f"    Step {step}: loss={lv} | VRAM={mem:.1f}GB")
            step += 1

    hist["time_sec"] = round(time.time() - t0, 1)
    hist["final_loss"] = hist["losses"][-1]["loss"] if hist["losses"] else None
    if DEVICE == "cuda":
        hist["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    print(f"  [{label}] Done: {hist['time_sec']}s | Final={hist['final_loss']} | "
          f"Crashed={hist['crashed']} | Rollbacks={hist['rollbacks']} | Lost={hist['steps_lost']} steps")

    del model; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    return hist

# ====================================================================
# PHASE 1: COMPARISON BENCHMARK — ARC vs 4 baselines
# ====================================================================
def phase1_comparison():
    print("\n" + "="*60)
    print("PHASE 1: ARC vs Baselines (GPT-2 XL 1.5B, NaN failure)")
    print("="*60)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    methods = [
        ("none",        "No Protection"),
        ("grad_clip",   "Gradient Clipping (norm=1.0)"),
        ("manual_ckpt", "Manual Checkpoint (every 10 steps)"),
        ("arc",         "ARC v4.5.0"),
    ]

    results = {}

    # First: healthy baseline (no failure)
    seed(42)
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
    results["Healthy"] = train_run("GPT-2-XL-1.5B", model, tokenizer,
                                   "none", "none", None, "Healthy",
                                   steps=2000, batch_size=2, max_len=128)

    # Then: each method with NaN failure at step 30
    for method, method_name in methods:
        seed(42)
        try:
            model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
            label = f"{method_name}"
            results[method] = train_run("GPT-2-XL-1.5B", model, tokenizer,
                                        method, "nan", inject_nan, label,
                                        steps=2000, batch_size=2, max_len=128,
                                        fail_step=1000, ckpt_freq=500)
        except Exception as e:
            print(f"  {method} FAILED: {e}")
            results[method] = {"error": str(e)}
            gc.collect()
            if DEVICE == "cuda": torch.cuda.empty_cache()

    save("phase1_comparison", results)

    # Print summary table
    print("\n  " + "="*70)
    print(f"  {'Method':35s} | {'Crashed':8s} | {'Final Loss':12s} | {'Steps Lost':10s}")
    print(f"  {'-'*35}-+-{'-'*8}-+-{'-'*12}-+-{'-'*10}")
    for key, r in results.items():
        if isinstance(r, dict) and "error" not in r:
            crashed = "YES" if r.get("crashed") else "No"
            floss = str(r.get("final_loss", "?"))[:10]
            lost = str(r.get("steps_lost", 0))
            print(f"  {key:35s} | {crashed:8s} | {floss:12s} | {lost:10s}")
    print("  " + "="*70)

    return results

# ====================================================================
# PHASE 2: FAILURE DIVERSITY — 6 failure types on GPT-2 XL with ARC
# ====================================================================
def phase2_failure_diversity():
    print("\n" + "="*60)
    print("PHASE 2: 6 Failure Types on GPT-2 XL (1.5B) with ARC")
    print("="*60)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Only test NEW failure types (NaN and weight_corruption already proven)
    new_failures = {k: v for k, v in FAILURE_TYPES.items()
                    if k not in ("nan", "weight_corruption")}

    results = {}
    for fname, finfo in new_failures.items():
        print(f"\n--- Failure: {finfo['desc']} ---")

        # Without ARC
        seed(42)
        try:
            model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
            results[f"{fname}_no_arc"] = train_run(
                "GPT-2-XL-1.5B", model, tokenizer,
                "none", fname, finfo["fn"], f"NoProt_{fname}",
                steps=2000, batch_size=2, max_len=128, fail_step=1000)
        except Exception as e:
            results[f"{fname}_no_arc"] = {"error": str(e)}
            gc.collect()
            if DEVICE == "cuda": torch.cuda.empty_cache()

        # With ARC
        seed(42)
        try:
            model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
            results[f"{fname}_arc"] = train_run(
                "GPT-2-XL-1.5B", model, tokenizer,
                "arc", fname, finfo["fn"], f"ARC_{fname}",
                steps=2000, batch_size=2, max_len=128, fail_step=1000)
        except Exception as e:
            results[f"{fname}_arc"] = {"error": str(e)}
            gc.collect()
            if DEVICE == "cuda": torch.cuda.empty_cache()

    save("phase2_failure_diversity", results)

    # Print summary
    print("\n  " + "="*75)
    print(f"  {'Failure Type':25s} | {'No Protection':15s} | {'ARC':15s} | {'ARC Detected':12s}")
    print(f"  {'-'*25}-+-{'-'*15}-+-{'-'*15}-+-{'-'*12}")
    for fname in FAILURE_TYPES:
        no_arc = results.get(f"{fname}_no_arc", {})
        arc = results.get(f"{fname}_arc", {})
        no_status = "CRASHED" if no_arc.get("crashed") else (
            f"loss={no_arc.get('final_loss', '?')}" if isinstance(no_arc, dict) and "error" not in no_arc else "ERROR")
        arc_status = f"loss={arc.get('final_loss', '?')}" if isinstance(arc, dict) and not arc.get("crashed") and "error" not in arc else (
            "CRASHED" if arc.get("crashed") else "ERROR")
        detected = "YES" if isinstance(arc, dict) and arc.get("rollbacks", 0) > 0 else "NO"
        print(f"  {fname:25s} | {no_status:15s} | {arc_status:15s} | {detected:12s}")
    print("  " + "="*75)

    return results

# ====================================================================
# MAIN
# ====================================================================
def main():
    t0 = time.time()
    print("="*60)
    print("ARC AAAI 2027 - COMPARISON & FAILURE DIVERSITY BENCHMARK")
    print("="*60)

    R = {}
    for name, fn in [("phase1", phase1_comparison), ("phase2", phase2_failure_diversity)]:
        elapsed = (time.time() - t0) / 60
        print(f"\n>>> Elapsed: {elapsed:.1f} min <<<")
        try:
            R[name] = fn()
        except Exception as e:
            print(f"  {name} FAILED: {e}")
            R[name] = {"error": str(e)}
            gc.collect()
            if DEVICE == "cuda": torch.cuda.empty_cache()

    total = round((time.time() - t0) / 60, 1)
    summary = {
        "gpu": torch.cuda.get_device_name() if DEVICE == "cuda" else "cpu",
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory/1e9, 1) if DEVICE == "cuda" else 0,
        "pytorch": torch.__version__,
        "total_minutes": total,
        "results": R
    }
    save("full_summary", summary)

    zp = "aaai_benchmark_results.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in os.listdir(OUT): z.write(os.path.join(OUT, f), f)
    print(f"\n{'='*60}")
    print(f"DONE | Total: {total} min | Zipped: {zp}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
