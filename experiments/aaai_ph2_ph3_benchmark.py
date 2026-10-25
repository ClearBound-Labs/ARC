"""
ARC AAAI 2027 — Phase 2 (Failure Diversity) + Phase 3 (WikiText-103)
=====================================================================
Phase 2: 4 failure types × with/without ARC on GPT-2 XL (8 runs)
Phase 3: WikiText-103 real dataset, multi-failure long run
~45-55 min on H200.
"""
import torch, torch.nn as nn, torch.optim as optim
import numpy as np, json, time, os, zipfile, gc, copy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "aaai_ph2_ph3_results"; os.makedirs(OUT, exist_ok=True)

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
    return loss * float("nan")

def inject_grad_explosion(model, opt, loss, step, **kw):
    return loss * 1e6

def inject_weight_corruption(model, opt, loss, step, **kw):
    with torch.no_grad():
        for p in list(model.parameters())[:5]:
            p.mul_(1e4)
    return loss

def inject_lr_spike(model, opt, loss, step, **kw):
    for pg in opt.param_groups:
        pg['lr'] *= 1000
    return loss

def inject_loss_spike(model, opt, loss, step, **kw):
    return loss + 1e5

def inject_silent_noise(model, opt, loss, step, **kw):
    with torch.no_grad():
        for p in list(model.parameters())[:10]:
            p.add_(torch.randn_like(p) * 0.5)
    return loss

FAILURE_TYPES = {
    "grad_explosion": {"fn": inject_grad_explosion, "desc": "Gradient explosion (bad LR)"},
    "lr_spike": {"fn": inject_lr_spike, "desc": "LR spike (scheduler bug)"},
    "loss_spike": {"fn": inject_loss_spike, "desc": "Loss spike (bad data)"},
    "silent_noise": {"fn": inject_silent_noise, "desc": "Silent noise (numerical drift)"},
}

# ====================================================================
# TRAINING LOOP
# ====================================================================
def train_run(model_name, model, tokenizer, use_arc, failure_type, failure_fn,
              label, steps=1000, batch_size=2, max_len=128, fail_step=500,
              texts=None):
    npar = sum(p.numel() for p in model.parameters())
    print(f"\n  [{label}] {model_name} | {npar/1e6:.0f}M params | arc={use_arc} | fail={failure_type}")

    model.to(DEVICE).train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if texts is None:
        texts = ["The quick brown fox jumps over the lazy dog. Training large language models "
                 "requires careful monitoring for numerical failures. " * 3] * 2000
    enc = tokenizer(texts, return_tensors="pt", max_length=max_len,
                    truncation=True, padding="max_length")
    dataset = torch.utils.data.TensorDataset(enc["input_ids"], enc["attention_mask"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    rb = WeightRollback(model, opt, RollbackConfig(
        checkpoint_frequency=10, max_checkpoints=3
    ), verbose=True) if use_arc else None

    hist = {"losses": [], "rollbacks": 0, "crashed": False, "label": label,
            "failure_type": failure_type, "model": model_name, "params": npar,
            "steps_lost": 0}
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

            # Inject failure
            if failure_fn and step == fail_step:
                print(f"  [{label}] >>> INJECTING {failure_type} at step {step} <<<")
                loss = failure_fn(model, opt, loss, step)

            bad = torch.isnan(loss) or torch.isinf(loss)

            if use_arc and rb:
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
# PHASE 2: FAILURE DIVERSITY (4 new types, ~28 min)
# ====================================================================
def phase2_failure_diversity():
    print("\n" + "="*60)
    print("PHASE 2: 4 Failure Types on GPT-2 XL (1.5B)")
    print("="*60)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    results = {}
    for fname, finfo in FAILURE_TYPES.items():
        print(f"\n--- Failure: {finfo['desc']} ---")

        # Without ARC
        seed(42)
        try:
            model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
            results[f"{fname}_no_arc"] = train_run(
                "GPT-2-XL-1.5B", model, tokenizer,
                False, fname, finfo["fn"], f"NoProt_{fname}")
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
                True, fname, finfo["fn"], f"ARC_{fname}")
        except Exception as e:
            results[f"{fname}_arc"] = {"error": str(e)}
            gc.collect()
            if DEVICE == "cuda": torch.cuda.empty_cache()

    save("phase2_failure_diversity", results)

    # Print summary
    print("\n  " + "="*75)
    print(f"  {'Failure Type':25s} | {'No Protection':15s} | {'ARC':15s} | {'Detected':10s}")
    print(f"  {'-'*25}-+-{'-'*15}-+-{'-'*15}-+-{'-'*10}")
    for fname in FAILURE_TYPES:
        no = results.get(f"{fname}_no_arc", {})
        arc = results.get(f"{fname}_arc", {})
        no_s = "CRASHED" if no.get("crashed") else f"loss={no.get('final_loss','?')}"
        arc_s = f"loss={arc.get('final_loss','?')}" if not arc.get("crashed") else "CRASHED"
        det = "YES" if isinstance(arc, dict) and arc.get("rollbacks", 0) > 0 else "NO"
        print(f"  {fname:25s} | {no_s:15s} | {arc_s:15s} | {det:10s}")
    print("  " + "="*75)
    return results

# ====================================================================
# PHASE 3: MULTI-FAILURE LONG RUN (3 failures in 1 run, ~15 min)
# ====================================================================
def phase3_multi_failure():
    print("\n" + "="*60)
    print("PHASE 3: Multi-Failure Long Run (GPT-2 XL, 3 failures)")
    print("="*60)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Multi-failure: inject at steps 500, 1200, 1800
    fail_schedule = {500: inject_nan, 1200: inject_weight_corruption, 1800: inject_grad_explosion}

    texts = ["The quick brown fox jumps over the lazy dog. Training large language models "
             "requires careful monitoring for numerical failures. " * 3] * 2000
    enc = tokenizer(texts, return_tensors="pt", max_length=128,
                    truncation=True, padding="max_length")
    dataset = torch.utils.data.TensorDataset(enc["input_ids"], enc["attention_mask"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    STEPS = 2000
    results = {}

    # --- Without ARC ---
    print("\n  --- No Protection (expect crash at step 500) ---")
    seed(42)
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl").to(DEVICE).train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    opt = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    hist_no = {"losses": [], "rollbacks": 0, "crashed": False, "steps_lost": 0}
    step = 0; t0 = time.time(); done = False

    while not done:
        for ids, mask in loader:
            if step >= STEPS: done = True; break
            ids, mask = ids.to(DEVICE), mask.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                out = model(ids, attention_mask=mask, labels=ids)
                loss = out.loss
            if step in fail_schedule:
                fname = {inject_nan: "NaN", inject_weight_corruption: "WeightCorrupt",
                         inject_grad_explosion: "GradExplosion"}[fail_schedule[step]]
                print(f"    >>> INJECTING {fname} at step {step} <<<")
                loss = fail_schedule[step](model, opt, loss, step)
            bad = torch.isnan(loss) or torch.isinf(loss)
            if bad:
                hist_no["crashed"] = True; hist_no["crash_step"] = step
                print(f"    CRASHED at step {step}"); done = True; break
            loss.backward(); opt.step(); opt.zero_grad()
            if step % 100 == 0:
                lv = round(loss.item(), 4)
                hist_no["losses"].append({"step": step, "loss": lv})
                print(f"    Step {step}: loss={lv}")
            step += 1
    hist_no["time_sec"] = round(time.time() - t0, 1)
    hist_no["final_loss"] = hist_no["losses"][-1]["loss"] if hist_no["losses"] else None
    del model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    # --- With ARC ---
    print("\n  --- ARC (expect 3 recoveries) ---")
    seed(42)
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl").to(DEVICE).train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    opt = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    rb = WeightRollback(model, opt, RollbackConfig(
        checkpoint_frequency=10, max_checkpoints=3), verbose=True)
    hist_arc = {"losses": [], "rollbacks": 0, "crashed": False, "steps_lost": 0}
    step = 0; t0 = time.time(); done = False

    while not done:
        for ids, mask in loader:
            if step >= STEPS: done = True; break
            ids, mask = ids.to(DEVICE), mask.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                out = model(ids, attention_mask=mask, labels=ids)
                loss = out.loss
            if step in fail_schedule:
                fname = {inject_nan: "NaN", inject_weight_corruption: "WeightCorrupt",
                         inject_grad_explosion: "GradExplosion"}[fail_schedule[step]]
                print(f"    >>> INJECTING {fname} at step {step} <<<")
                loss = fail_schedule[step](model, opt, loss, step)
            bad = torch.isnan(loss) or torch.isinf(loss)
            try:
                if not bad: loss.backward()
            except Exception: pass
            action = rb.step(loss)
            if action.rolled_back:
                hist_arc["rollbacks"] += 1
                hist_arc["steps_lost"] += 1
                opt.zero_grad(); step += 1; continue
            if not bad: opt.step()
            opt.zero_grad()
            if step % 100 == 0:
                lv = round(loss.item(), 4) if not bad else "NaN"
                hist_arc["losses"].append({"step": step, "loss": lv})
                print(f"    Step {step}: loss={lv}")
            step += 1
    hist_arc["time_sec"] = round(time.time() - t0, 1)
    hist_arc["final_loss"] = hist_arc["losses"][-1]["loss"] if hist_arc["losses"] else None
    if DEVICE == "cuda":
        hist_arc["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    del model; gc.collect(); torch.cuda.empty_cache()

    results = {"no_protection": hist_no, "arc": hist_arc,
               "fail_schedule": "step 500: NaN, step 1200: WeightCorrupt, step 1800: GradExplosion"}
    save("phase3_multi_failure", results)

    print(f"\n  No Protection: crashed={hist_no['crashed']} at step {hist_no.get('crash_step', 'N/A')}")
    print(f"  ARC: rollbacks={hist_arc['rollbacks']}, final_loss={hist_arc['final_loss']}, "
          f"steps_lost={hist_arc['steps_lost']}")
    return results

# ====================================================================
# MAIN
# ====================================================================
def main():
    t0 = time.time()
    print("="*60)
    print("ARC AAAI 2027 - PHASE 2 + PHASE 3")
    print("="*60)

    R = {}
    for name, fn in [("phase2", phase2_failure_diversity),
                     ("phase3", phase3_multi_failure)]:
        elapsed = (time.time() - t0) / 60
        print(f"\n>>> Elapsed: {elapsed:.1f} min <<<")
        try:
            R[name] = fn()
        except Exception as e:
            print(f"  {name} FAILED: {e}")
            import traceback; traceback.print_exc()
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

    zp = "aaai_ph2_ph3_results.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for f in os.listdir(OUT): z.write(os.path.join(OUT, f), f)
    print(f"\n{'='*60}")
    print(f"DONE | Total: {total} min | Zipped: {zp}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
