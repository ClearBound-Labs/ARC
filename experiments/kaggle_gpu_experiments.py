"""
ARC Kaggle GPU Experiments v3 — Diagnostic Version
====================================================
This version adds detailed diagnostic logging to identify
why large models have 62 rollbacks per run.

Kaggle Setup:
1. New Notebook → GPU T4 → Internet ON
2. First cell:  !pip install arc-training transformers torchvision -q --no-cache-dir
3. Copy everything below into cells
"""

# === CELL 1: Setup ===
# !pip install arc-training transformers torchvision -q --no-cache-dir

import torch, torch.nn as nn, torch.optim as optim
import time, gc, json, math, copy
import numpy as np
from collections import defaultdict

from arc.core.self_healing import SelfHealingArc, SelfHealingConfig, HealingAction
import arc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"arc-training: {arc.__version__} | PyTorch: {torch.__version__} | GPU: {torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'none'}")
print(f"SelfHealingConfig defaults: lite_mode={SelfHealingConfig().lite_mode}, check_frequency={SelfHealingConfig().check_frequency}")


# === CELL 2: EXPERIMENT 1 — GPU vs CPU Overhead ===

print("=" * 70)
print("  EXPERIMENT 1: GPU vs CPU Overhead")
print("=" * 70)

import torchvision, torchvision.transforms as T
ds = torchvision.datasets.CIFAR10('./data', train=True, download=True,
    transform=T.Compose([T.ToTensor(), T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]))
loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True, drop_last=True)

def cycle(loader):
    while True:
        for b in loader: yield b

class M1(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3,8,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(8*16*16,10))
    def forward(self,x): return self.net(x)

class M2(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.c = nn.Sequential(nn.Linear(64*8*8,128), nn.ReLU(), nn.Linear(128,10))
    def forward(self,x): return self.c(self.f(x).view(x.size(0),-1))

class M3(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(3,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4))
        self.c = nn.Sequential(nn.Linear(256*4*4,512), nn.ReLU(), nn.Linear(512,10))
    def forward(self,x): return self.c(self.f(x).view(x.size(0),-1))

class M4(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(3,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256,512,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4))
        self.c = nn.Sequential(nn.Linear(512*4*4,2048), nn.ReLU(), nn.Linear(2048,1024), nn.ReLU(), nn.Linear(1024,10))
    def forward(self,x): return self.c(self.f(x).view(x.size(0),-1))

class M5(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(3,256,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(256,512,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(512,1024,3,padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4))
        self.c = nn.Sequential(nn.Linear(1024*4*4,4096), nn.ReLU(), nn.Linear(4096,2048), nn.ReLU(), nn.Linear(2048,10))
    def forward(self,x): return self.c(self.f(x).view(x.size(0),-1))

MODELS = [("35K",M1),("150K",M2),("1.2M",M3),("10M",M4),("40M",M5)]
STEPS, WARM = 200, 30

def bench(dev):
    res = {}; it = cycle(loader); crit = nn.CrossEntropyLoss()
    for label, Cls in MODELS:
        torch.manual_seed(42)
        p = sum(q.numel() for q in Cls().parameters())
        m = Cls().to(dev); o = optim.SGD(m.parameters(),lr=0.01,momentum=0.9)
        for _ in range(WARM):
            x,y=next(it); x,y=x.to(dev),y.to(dev); o.zero_grad(); crit(m(x),y).backward(); o.step()
        bt=[]
        for _ in range(STEPS):
            x,y=next(it); x,y=x.to(dev),y.to(dev)
            if dev=="cuda": torch.cuda.synchronize()
            t=time.perf_counter(); o.zero_grad(); l=crit(m(x),y); l.backward(); o.step()
            if dev=="cuda": torch.cuda.synchronize()
            bt.append(time.perf_counter()-t)
        del m,o; gc.collect()
        if dev=="cuda": torch.cuda.empty_cache()
        m = Cls().to(dev); o = optim.SGD(m.parameters(),lr=0.01,momentum=0.9)
        ctrl = SelfHealingArc(m, o, SelfHealingConfig(verbose=False))
        for _ in range(WARM):
            x,y=next(it); x,y=x.to(dev),y.to(dev); o.zero_grad(); crit(m(x),y).backward(); o.step()
        at=[]
        for _ in range(STEPS):
            x,y=next(it); x,y=x.to(dev),y.to(dev)
            if dev=="cuda": torch.cuda.synchronize()
            t=time.perf_counter(); o.zero_grad(); l=crit(m(x),y)
            a = ctrl.step(l)
            if not a.rolled_back and not a.should_skip:
                l.backward(); pa = ctrl.post_backward()
                if not pa.rolled_back and not pa.should_skip: o.step()
            if dev=="cuda": torch.cuda.synchronize()
            at.append(time.perf_counter()-t)
        del m,o,ctrl; gc.collect()
        if dev=="cuda": torch.cuda.empty_cache()
        bm=np.median(bt)*1000; am=np.median(at)*1000; oh=((am-bm)/bm)*100
        res[label]={"params":p,"base_ms":round(bm,3),"arc_ms":round(am,3),"overhead":round(oh,2)}
        print(f"  {label:>5s} ({p:>12,}) base={bm:.2f}ms arc={am:.2f}ms overhead={oh:.1f}%")
    return res

print("\n--- GPU ---")
gpu_res = bench("cuda")
print("\n--- CPU ---")
cpu_res = bench("cpu")

print(f"\n  {'Scale':>5s} {'Params':>12s} {'CPU OH':>8s} {'GPU OH':>8s}")
print(f"  {'-'*36}")
for l,_ in MODELS:
    print(f"  {l:>5s} {gpu_res[l]['params']:>12,} {cpu_res[l]['overhead']:>7.1f}% {gpu_res[l]['overhead']:>7.1f}%")

gc.collect(); torch.cuda.empty_cache()


# === CELL 3: DIAGNOSTIC — Debug the fp16 rollback issue ===

print("\n" + "=" * 70)
print("  DIAGNOSTIC: Why does ARC roll back every step on fp16 models?")
print("=" * 70)

from transformers import AutoModelForCausalLM

# Load GPT-2 Medium in fp16 (same as Exp 2)
print("\nLoading GPT-2 Medium in fp16...")
m = AutoModelForCausalLM.from_pretrained("gpt2-medium", torch_dtype=torch.float16).to(DEVICE)
for p in m.parameters(): p.requires_grad = False
for layer in m.transformer.h[-4:]:
    for p in layer.parameters(): p.requires_grad = True

# Check: do fp16 weights have NaN?
print("\n--- Weight Health Check ---")
param_list = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
total_params = sum(p.numel() for _, p in param_list)
nan_count = sum(torch.isnan(p.data).sum().item() for _, p in param_list)
inf_count = sum(torch.isinf(p.data).sum().item() for _, p in param_list)
print(f"  Trainable params: {total_params:,}")
print(f"  NaN values: {nan_count}")
print(f"  Inf values: {inf_count}")

# Check: what does _compute_weight_stats return?
total_magnitude_fp16 = sum(p.data.abs().sum().item() for _, p in param_list)
total_magnitude_fp32 = sum(p.data.float().abs().sum().item() for _, p in param_list)
print(f"  Weight magnitude (fp16 sum): {total_magnitude_fp16}")
print(f"  Weight magnitude (fp32 sum): {total_magnitude_fp32}")
print(f"  Are they different? {total_magnitude_fp16 != total_magnitude_fp32}")

# Try training 5 steps WITHOUT ARC to confirm model works
print("\n--- Training WITHOUT ARC (5 steps) ---")
batch = torch.randint(0, m.config.vocab_size, (2, 128)).to(DEVICE)
opt = optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=2e-5)
for s in range(1, 6):
    opt.zero_grad()
    with torch.amp.autocast('cuda', dtype=torch.float16):
        loss = m(input_ids=batch, labels=batch).loss
    loss_f32 = loss.float()
    print(f"  Step {s}: loss={loss_f32.item():.4f} (isnan={torch.isnan(loss_f32).item()}, isinf={torch.isinf(loss_f32).item()})")
    loss_f32.backward()
    # Check gradients
    grad_nans = sum(torch.isnan(p.grad).sum().item() for _, p in param_list if p.grad is not None)
    grad_infs = sum(torch.isinf(p.grad).sum().item() for _, p in param_list if p.grad is not None)
    print(f"         grad NaN: {grad_nans}, grad Inf: {grad_infs}")
    opt.step()

# Now try WITH ARC - verbose step-by-step
print("\n--- Training WITH ARC (10 steps, verbose) ---")
m2 = AutoModelForCausalLM.from_pretrained("gpt2-medium", torch_dtype=torch.float16).to(DEVICE)
for p in m2.parameters(): p.requires_grad = False
for layer in m2.transformer.h[-4:]:
    for p in layer.parameters(): p.requires_grad = True

opt2 = optim.AdamW([p for p in m2.parameters() if p.requires_grad], lr=2e-5)
ctrl = SelfHealingArc(m2, opt2, SelfHealingConfig(verbose=True, cpu_offload=True))

for s in range(1, 11):
    opt2.zero_grad()
    with torch.amp.autocast('cuda', dtype=torch.float16):
        loss = m2(input_ids=batch, labels=batch).loss
    loss_f32 = loss.float()
    
    print(f"  Step {s}: loss={loss_f32.item():.6f}", end="")
    action = ctrl.step(loss_f32)
    print(f" | action: rb={action.rolled_back}, skip={action.should_skip}, fail={action.failure_detected}")
    
    if not action.rolled_back and not action.should_skip:
        loss_f32.backward()
        post = ctrl.post_backward()
        print(f"         post_bwd: rb={post.rolled_back}, skip={post.should_skip}, fail={post.failure_detected}")
        if not post.rolled_back and not post.should_skip:
            opt2.step()
            print(f"         -> optimizer stepped")
        else:
            print(f"         -> skipped optimizer step")
    else:
        print(f"         -> skipped backward + step")

print(f"\n  Total: {ctrl.total_rollbacks} rollbacks, {ctrl.failures_detected} failures")
print(f"  Failure breakdown: check intervention_log")
for entry in ctrl.intervention_log[:10]:
    print(f"    Step {entry.get('step')}: {entry.get('reason')} -> {entry.get('action')}")

del m, m2, opt, opt2, ctrl
gc.collect(); torch.cuda.empty_cache()


# === CELL 4: EXPERIMENT 2 — Large Model (use results from diagnostic) ===

print("\n" + "=" * 70)
print("  EXPERIMENT 2: Large Model Scale Test")
print("=" * 70)

from transformers import GPT2LMHeadModel, GPT2Config
import traceback

large_results = []

def run_large_test(name, plabel, load_fn, steps=50):
    print(f"\n  --- {name} ({plabel}) ---")
    scaler = None
    try:
        model, tp, use_amp = load_fn()
        dev = next(model.parameters()).device
        batch = torch.randint(0, model.config.vocab_size, (2, 128)).to(dev)

        opt = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)
        scaler = torch.amp.GradScaler('cuda') if use_amp else None
        # Disable weight health check for fp16 models to prevent false positives
        cfg = SelfHealingConfig(verbose=True, cpu_offload=True,
                                enable_weight_health_check=not use_amp)
        ctrl = SelfHealingArc(model, opt, cfg)

        def fwd():
            if use_amp:
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    return model(input_ids=batch, labels=batch).loss.float()
            return model(input_ids=batch, labels=batch).loss

        def bwd(loss):
            if scaler: scaler.scale(loss).backward()
            else: loss.backward()

        def stp():
            if scaler: scaler.step(opt); scaler.update()
            else: opt.step()

        print(f"    Training {steps} steps...")
        for s in range(1, steps+1):
            opt.zero_grad()
            loss = fwd()
            a = ctrl.step(loss)
            if not a.rolled_back and not a.should_skip:
                bwd(loss)
                pa = ctrl.post_backward()
                if not pa.rolled_back and not pa.should_skip: stp()
            if s % 10 == 0:
                lv = loss.item() if not (torch.isnan(loss) or torch.isinf(loss)) else "nan"
                print(f"      Step {s}: loss={lv} (rollbacks so far: {ctrl.total_rollbacks})")

        stable_loss = loss.item() if not (torch.isnan(loss) or torch.isinf(loss)) else float('inf')

        print(f"    Injecting failures...")
        for fn, fv in [("NaN",float('nan')),("Inf",float('inf'))]:
            a = ctrl.step(torch.tensor(fv, device=dev, dtype=torch.float32))
            print(f"      {fn}: rolled_back={a.rolled_back}")

        print(f"    Verifying recovery...")
        ok = True
        for _ in range(10):
            opt.zero_grad(); loss = fwd()
            if torch.isnan(loss) or torch.isinf(loss): ok=False; break
            a = ctrl.step(loss)
            if not a.rolled_back and not a.should_skip:
                bwd(loss); pa = ctrl.post_backward()
                if not pa.rolled_back and not pa.should_skip: stp()

        post = loss.item() if ok and not (torch.isnan(loss) or torch.isinf(loss)) else float('inf')
        r = {"model":name,"params":plabel,"trainable":tp,"recovered":ok,
             "stable_loss":round(stable_loss,4),"post_loss":round(post,4),
             "rollbacks":ctrl.total_rollbacks,"detected":ctrl.failures_detected}
        sym = "✓" if ok else "✗"
        print(f"    Result: {sym} | loss: {stable_loss:.4f} -> {post:.4f} | rollbacks: {ctrl.total_rollbacks}")
    except Exception as e:
        print(f"    ERROR: {e}"); traceback.print_exc()
        r = {"model":name,"params":plabel,"error":str(e)}
    large_results.append(r)
    try: del model, opt, ctrl, scaler
    except: pass
    gc.collect(); torch.cuda.empty_cache()

def load_gpt2_small():
    m = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32).to(DEVICE)
    tp = sum(p.numel() for p in m.parameters())
    print(f"    {tp:,} params (fp32)"); return m, tp, False

def load_gpt2_medium():
    m = AutoModelForCausalLM.from_pretrained("gpt2-medium", torch_dtype=torch.float16).to(DEVICE)
    for p in m.parameters(): p.requires_grad = False
    for l in m.transformer.h[-4:]:
        for p in l.parameters(): p.requires_grad = True
    tp = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"    {tp:,} trainable (fp16+autocast)"); return m, tp, True

def load_gpt2_large():
    m = AutoModelForCausalLM.from_pretrained("gpt2-large", torch_dtype=torch.float16).to(DEVICE)
    for p in m.parameters(): p.requires_grad = False
    for l in m.transformer.h[-3:]:
        for p in l.parameters(): p.requires_grad = True
    tp = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"    {tp:,} trainable (fp16+autocast)"); return m, tp, True

def load_tinyllama():
    m = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16).to(DEVICE)
    for p in m.parameters(): p.requires_grad = False
    for l in m.model.layers[-2:]:
        for p in l.parameters(): p.requires_grad = True
    tp = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"    {tp:,} trainable (fp16+autocast)"); return m, tp, True

run_large_test("GPT-2 Small", "124M", load_gpt2_small)
run_large_test("GPT-2 Medium", "355M", load_gpt2_medium)
run_large_test("GPT-2 Large", "774M", load_gpt2_large)
run_large_test("TinyLlama", "1.1B", load_tinyllama)

print(f"\n  {'Model':<14s} {'Params':>7s} {'Trainable':>12s} {'OK':>4s} {'Rollbacks':>10s}")
print(f"  {'-'*50}")
for r in large_results:
    if 'error' in r:
        print(f"  {r['model']:<14s} {r['params']:>7s} {'ERROR':>12s}")
    else:
        t = f"{r['trainable']/1e6:.1f}M"
        print(f"  {r['model']:<14s} {r['params']:>7s} {t:>12s} {'✓' if r['recovered'] else '✗':>4s} {r['rollbacks']:>10d}")

gc.collect(); torch.cuda.empty_cache()


# === CELL 5: EXPERIMENT 3 — Organic Failures ===

print("\n" + "=" * 70)
print("  EXPERIMENT 3: Organic Failure Detection (NO injection)")
print("=" * 70)

organic_results = []

configs = [
    ("LR=0.1 (1000x normal)",  0.1,  "Adam",  4, 500),
    ("LR=0.5 (5000x normal)",  0.5,  "Adam",  4, 300),
    ("LR=1.0 (10000x normal)", 1.0,  "Adam",  4, 200),
    ("SGD LR=0.1 no momentum", 0.1,  "SGD",   4, 500),
    ("Tiny batch LR=0.5",      0.5,  "Adam",  1, 300),
]

for cname, lr, optname, bs, max_steps in configs:
    print(f"\n  --- {cname} ---")

    torch.manual_seed(42)
    m = GPT2LMHeadModel(GPT2Config(vocab_size=50257, n_positions=256,
        n_embd=768, n_layer=12, n_head=12)).to(DEVICE)

    if optname == "Adam":
        o = optim.Adam(m.parameters(), lr=lr)
    else:
        o = optim.SGD(m.parameters(), lr=lr)

    ctrl = SelfHealingArc(m, o, SelfHealingConfig(verbose=False))

    log = []
    for step in range(1, max_steps+1):
        x = torch.randint(0, 50257, (bs, 128)).to(DEVICE)
        o.zero_grad()
        loss = m(input_ids=x, labels=x).loss

        if torch.isnan(loss) or torch.isinf(loss):
            action = ctrl.step(loss)
            post = HealingAction(step=step)
        else:
            action = ctrl.step(loss)
            if not action.rolled_back and not action.should_skip:
                loss.backward()
                post = ctrl.post_backward()
                if not post.rolled_back and not post.should_skip:
                    o.step()
            else:
                post = HealingAction(step=step)

        fail = action.failure_detected or post.failure_detected
        rolled = action.rolled_back or post.rolled_back
        lv = loss.item() if not (torch.isnan(loss) or torch.isinf(loss)) else float('inf')
        log.append({"step":step, "loss":lv, "fail":fail, "rb":rolled})

        if step % 50 == 0 or rolled:
            ls = f"{lv:.4f}" if lv!=float('inf') else "inf"
            print(f"    Step {step:>4d}: loss={ls}" + (f" [{fail}] <- ROLLED BACK" if rolled else ""))

        if ctrl.failures_detected >= 20:
            print(f"    -> Stopping: {ctrl.failures_detected} failures detected")
            break

    for g in o.param_groups: g['lr'] = 1e-4
    ok = True
    for _ in range(10):
        x = torch.randint(0, 50257, (4,128)).to(DEVICE)
        o.zero_grad()
        loss = m(input_ids=x, labels=x).loss
        if torch.isnan(loss) or torch.isinf(loss): ok=False; break
        loss.backward(); o.step()

    ftypes = defaultdict(int)
    for e in log:
        if e['fail']: ftypes[e['fail']] += 1

    rate = ctrl.failures_recovered / max(ctrl.failures_detected,1) * 100
    print(f"    Result: {ctrl.failures_detected} failures, {rate:.0f}% recovered, {'trainable' if ok else 'broken'}")
    print(f"    Types: {dict(ftypes)}")

    organic_results.append({
        "config": cname, "lr": lr, "steps": ctrl.step_count,
        "detected": ctrl.failures_detected, "recovered": ctrl.failures_recovered,
        "rate": round(rate,1), "trainable": ok, "types": dict(ftypes)
    })
    del m,o,ctrl; gc.collect(); torch.cuda.empty_cache()

print(f"\n  {'Config':<28s} {'Fails':>6s} {'Rec':>5s} {'Rate':>6s} {'OK':>4s}")
print(f"  {'-'*51}")
for r in organic_results:
    print(f"  {r['config']:<28s} {r['detected']:>6d} {r['recovered']:>5d} {r['rate']:>5.0f}% {'Y' if r['trainable'] else 'N':>4s}")


# === CELL 6: Save ===

summary = {
    "experiment_1_overhead": {"gpu": gpu_res, "cpu": cpu_res},
    "diagnostic": "See cell 3 output for detailed fp16 rollback analysis",
    "experiment_2_large_models": large_results,
    "experiment_3_organic": organic_results,
    "env": {"arc": arc.__version__, "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if DEVICE=="cuda" else "none"}
}

with open('kaggle_experiment_results.json','w') as f:
    json.dump(summary, f, indent=2, default=str)

print("\n" + "=" * 70)
print("  ALL DONE — Results saved to kaggle_experiment_results.json")
print("  IMPORTANT: Copy the Cell 3 diagnostic output!")
print("=" * 70)
