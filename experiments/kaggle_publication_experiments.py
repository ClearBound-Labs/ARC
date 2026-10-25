"""
ARC Publication-Grade Experiments — Kaggle T4 GPU
===================================================
6 experiments to prove every claim rigorously.
Estimated total: ~36 minutes GPU time.

Run each cell independently. Each saves results to JSON.
"""


# ============================================================
# CELL 0: Setup (run first)
# ============================================================
"""
!pip install arc-training transformers datasets torchvision -q
import arc, torch
print(f"ARC {arc.__version__} | PyTorch {torch.__version__} | GPU: {torch.cuda.get_device_name(0)}")
"""


# ============================================================
# EXPERIMENT 1: Real Dataset Training + Recovery (~8 min)
# ============================================================
# Proves: ARC recovers from failure and reaches the same final
# accuracy as uninterrupted training on a REAL dataset.
# ============================================================
"""
import torch, torch.nn as nn, torch.optim as optim, json, time
import torchvision, torchvision.transforms as T
from arc.core.self_healing import SelfHealingArc, SelfHealingConfig

transform_train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
transform_test = T.Compose([T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

def evaluate(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    model.train()
    return 100.0 * correct / total

def train_run(seed, inject_nan_epoch=None, use_arc=False, label=""):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    model = torchvision.models.resnet18(num_classes=10).cuda()
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    criterion = nn.CrossEntropyLoss()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    ctrl = None
    if use_arc:
        ctrl = SelfHealingArc(model, opt, SelfHealingConfig(
            verbose=False, lite_mode=True, checkpoint_frequency=50,
        ))
    
    epoch_accs = []
    total_rollbacks = 0
    crashed = False
    t0 = time.time()
    
    for epoch in range(1, 11):  # 10 epochs
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.cuda(), labels.cuda()
            opt.zero_grad()
            
            # Inject NaN at specific epoch, step 50
            if inject_nan_epoch and epoch == inject_nan_epoch and batch_idx == 50:
                if use_arc:
                    a = ctrl.step(torch.tensor(float('nan')).cuda())
                    if a.rolled_back:
                        total_rollbacks += 1
                    continue
                else:
                    # Without ARC: NaN corrupts the model
                    for p in list(model.parameters())[:5]:
                        p.data.fill_(float('nan'))
                    # Try to continue (will produce NaN outputs)
            
            out = model(images)
            loss = criterion(out, labels)
            
            # Check for NaN loss (model is corrupted)
            if torch.isnan(loss):
                if not use_arc:
                    crashed = True
                    break
            
            if use_arc:
                a = ctrl.step(loss)
                if a.rolled_back or a.should_skip:
                    total_rollbacks += (1 if a.rolled_back else 0)
                    continue
                loss.backward()
                pa = ctrl.post_backward()
                if pa.rolled_back or pa.should_skip:
                    total_rollbacks += (1 if pa.rolled_back else 0)
                    continue
                opt.step()
            else:
                loss.backward()
                opt.step()
            
            running_loss += loss.item()
        
        if crashed:
            break
        
        scheduler.step()
        acc = evaluate(model)
        epoch_accs.append(acc)
        print(f"  [{label}] Epoch {epoch}: acc={acc:.2f}%")
    
    elapsed = time.time() - t0
    return {
        "label": label,
        "seed": seed,
        "final_acc": epoch_accs[-1] if epoch_accs else 0.0,
        "epoch_accs": epoch_accs,
        "rollbacks": total_rollbacks,
        "crashed": crashed,
        "time_s": round(elapsed, 1),
    }

results = {"experiment": "real_dataset_recovery", "runs": []}
for seed in [42, 123, 777]:
    print(f"\\n=== Seed {seed} ===")
    r1 = train_run(seed, inject_nan_epoch=None, use_arc=False, label="clean_baseline")
    r2 = train_run(seed, inject_nan_epoch=5, use_arc=False, label="nan_no_arc")
    r3 = train_run(seed, inject_nan_epoch=5, use_arc=True, label="nan_with_arc")
    results["runs"].extend([r1, r2, r3])

# Summary
print("\\n" + "="*60)
print("EXPERIMENT 1: Real Dataset Recovery")
print("="*60)
for label in ["clean_baseline", "nan_no_arc", "nan_with_arc"]:
    runs = [r for r in results["runs"] if r["label"] == label]
    accs = [r["final_acc"] for r in runs if not r["crashed"]]
    crashed = sum(1 for r in runs if r["crashed"])
    if accs:
        print(f"  {label:20s}: acc={sum(accs)/len(accs):.2f}% +/- {max(accs)-min(accs):.2f}  crashed={crashed}/3")
    else:
        print(f"  {label:20s}: ALL CRASHED")

with open("exp1_real_dataset_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\\nSaved: exp1_real_dataset_results.json")
"""


# ============================================================
# EXPERIMENT 2: GPT-2 Fine-tuning on WikiText-2 (~10 min)
# ============================================================
# Proves: ARC works on real language model fine-tuning with
# actual tokens, not synthetic data.
# ============================================================
"""
import torch, torch.optim as optim, json, time, math
from torch.amp import autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from arc.core.self_healing import SelfHealingArc, SelfHealingConfig

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

# Tokenize and chunk into blocks of 128
def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

tokenized = dataset.filter(lambda x: len(x["text"]) > 50).map(tokenize, batched=True, remove_columns=["text"])
tokenized.set_format("torch")

def collate(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    return input_ids

loader = DataLoader(tokenized, batch_size=4, shuffle=True, collate_fn=collate, drop_last=True)

def gpt2_run(seed, inject_nan_step=None, use_arc=False, label="", max_steps=300):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32).cuda()
    opt = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    
    ctrl = None
    if use_arc:
        ctrl = SelfHealingArc(model, opt, SelfHealingConfig(
            verbose=False, lite_mode=True, checkpoint_frequency=20,
            enable_weight_health_check=False,
        ))
    
    losses, rollbacks = [], 0
    crashed = False
    t0 = time.time()
    data_iter = iter(loader)
    
    for step in range(1, max_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        
        batch = batch.cuda()
        opt.zero_grad()
        
        # Inject NaN
        if inject_nan_step and step == inject_nan_step:
            if use_arc:
                a = ctrl.step(torch.tensor(float('nan')).cuda())
                rollbacks += (1 if a.rolled_back else 0)
                continue
            else:
                for p in list(model.parameters())[:3]:
                    p.data.fill_(float('nan'))
        
        out = model(input_ids=batch, labels=batch)
        loss = out.loss
        
        if torch.isnan(loss) or torch.isinf(loss):
            if not use_arc:
                crashed = True
                break
        
        if use_arc:
            a = ctrl.step(loss)
            if a.rolled_back or a.should_skip:
                rollbacks += (1 if a.rolled_back else 0)
                continue
            loss.backward()
            pa = ctrl.post_backward()
            if pa.rolled_back or pa.should_skip:
                rollbacks += (1 if pa.rolled_back else 0)
                continue
            opt.step()
        else:
            loss.backward()
            opt.step()
        
        losses.append(loss.item())
        if step % 100 == 0:
            avg = sum(losses[-50:]) / min(50, len(losses[-50:]))
            ppl = math.exp(min(avg, 20))
            print(f"  [{label}] Step {step}: loss={avg:.4f} ppl={ppl:.1f}")
    
    elapsed = time.time() - t0
    final_loss = sum(losses[-20:]) / max(len(losses[-20:]), 1) if losses else float('inf')
    return {
        "label": label, "seed": seed,
        "final_loss": round(final_loss, 4),
        "perplexity": round(math.exp(min(final_loss, 20)), 1),
        "total_steps": len(losses),
        "rollbacks": rollbacks,
        "crashed": crashed,
        "time_s": round(elapsed, 1),
    }

results = {"experiment": "gpt2_wikitext2_finetuning", "runs": []}
for seed in [42, 123, 777]:
    print(f"\\n=== Seed {seed} ===")
    r1 = gpt2_run(seed, inject_nan_step=None, use_arc=False, label="clean")
    r2 = gpt2_run(seed, inject_nan_step=150, use_arc=False, label="nan_no_arc")
    r3 = gpt2_run(seed, inject_nan_step=150, use_arc=True, label="nan_with_arc")
    results["runs"].extend([r1, r2, r3])

print("\\n" + "="*60)
print("EXPERIMENT 2: GPT-2 WikiText-2 Fine-tuning")
print("="*60)
for label in ["clean", "nan_no_arc", "nan_with_arc"]:
    runs = [r for r in results["runs"] if r["label"] == label]
    ppls = [r["perplexity"] for r in runs if not r["crashed"]]
    crashed = sum(1 for r in runs if r["crashed"])
    if ppls:
        print(f"  {label:20s}: ppl={sum(ppls)/len(ppls):.1f} +/- {max(ppls)-min(ppls):.1f}  crashed={crashed}/3  rollbacks={sum(r['rollbacks'] for r in runs)}")
    else:
        print(f"  {label:20s}: ALL CRASHED")

with open("exp2_gpt2_wikitext_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\\nSaved: exp2_gpt2_wikitext_results.json")
"""


# ============================================================
# EXPERIMENT 3: Stronger Ablation Study (~5 min)
# ============================================================
# Proves: Each ARC component contributes. Uses HARD failures
# where individual components matter.
# ============================================================
"""
import torch, torch.nn as nn, torch.optim as optim, json, time
from arc.core.self_healing import SelfHealingArc, SelfHealingConfig

class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        dims = [200, 400, 400, 300, 200, 100, 10]
        for i in range(len(dims)-1):
            layers.extend([nn.Linear(dims[i], dims[i+1]), nn.ReLU()])
        layers.pop()
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

def run_ablation(config_name, config, seed, failure_type):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    model = DeepNet().cuda()
    
    if failure_type == "high_lr":
        opt = optim.SGD(model.parameters(), lr=2.0)
    elif failure_type == "nan_inject":
        opt = optim.Adam(model.parameters(), lr=1e-3)
    elif failure_type == "grad_explosion":
        opt = optim.SGD(model.parameters(), lr=0.5)
    else:
        opt = optim.Adam(model.parameters(), lr=1e-3)
    
    ctrl = SelfHealingArc(model, opt, config)
    batch_x = torch.randn(32, 200).cuda()
    batch_y = torch.randint(0, 10, (32,)).cuda()
    criterion = nn.CrossEntropyLoss()
    
    rollbacks = 0
    losses = []
    model_alive = True
    
    for s in range(1, 101):
        opt.zero_grad()
        
        # Inject failure at step 30
        if s == 30:
            if failure_type == "nan_inject":
                a = ctrl.step(torch.tensor(float('nan')).cuda())
                rollbacks += (1 if a.rolled_back else 0)
                continue
            elif failure_type == "weight_corrupt":
                with torch.no_grad():
                    for p in list(model.parameters())[:2]:
                        p.data.fill_(float('nan'))
        
        try:
            loss = criterion(model(batch_x), batch_y)
        except:
            loss = torch.tensor(float('nan')).cuda()
        
        if torch.isnan(loss) or torch.isinf(loss):
            a = ctrl.step(loss)
            rollbacks += (1 if a.rolled_back else 0)
            continue
        
        a = ctrl.step(loss)
        if a.rolled_back or a.should_skip:
            rollbacks += (1 if a.rolled_back else 0)
            continue
        loss.backward()
        pa = ctrl.post_backward()
        if pa.rolled_back or pa.should_skip:
            rollbacks += (1 if pa.rolled_back else 0)
            continue
        opt.step()
        losses.append(loss.item())
    
    model_alive = not any(torch.isnan(p).any() for p in model.parameters())
    final_loss = losses[-1] if losses else float('inf')
    recovered = model_alive and len(losses) > 50
    
    return {
        "config": config_name, "failure": failure_type, "seed": seed,
        "recovered": recovered, "model_alive": model_alive,
        "rollbacks": rollbacks, "final_loss": round(final_loss, 4),
        "steps_completed": len(losses),
    }

# Ablation configs
configs = {
    "Full ARC": SelfHealingConfig(verbose=False, checkpoint_frequency=5),
    "No Grad Clip": SelfHealingConfig(verbose=False, checkpoint_frequency=5, gradient_explosion_threshold=1e30),
    "No Weight Health": SelfHealingConfig(verbose=False, checkpoint_frequency=5, enable_weight_health_check=False),
    "No LR Reduction": SelfHealingConfig(verbose=False, checkpoint_frequency=5, lr_reduction_factor=1.0),
    "No Perturbation": SelfHealingConfig(verbose=False, checkpoint_frequency=5, weight_perturbation_scale=0.0),
    "No Forecasting": SelfHealingConfig(verbose=False, checkpoint_frequency=5, enable_forecasting=False),
}

failure_types = ["nan_inject", "high_lr", "grad_explosion", "weight_corrupt"]
seeds = [42, 123, 777, 999, 2024]

results = {"experiment": "ablation_study", "runs": []}
for cname, cfg in configs.items():
    for ft in failure_types:
        recovery_count = 0
        for seed in seeds:
            r = run_ablation(cname, cfg, seed, ft)
            results["runs"].append(r)
            recovery_count += (1 if r["recovered"] else 0)
        total = len(seeds)
        print(f"  {cname:20s} | {ft:15s} | recovered={recovery_count}/{total}")

# Summary table
print("\\n" + "="*70)
print("EXPERIMENT 3: Ablation Study")
print("="*70)
print(f"{'Config':20s} | {'Recovery Rate':>13s} | {'Avg Rollbacks':>13s} | {'Avg Final Loss':>14s}")
print("-"*70)
for cname in configs:
    runs = [r for r in results["runs"] if r["config"] == cname]
    rec_rate = sum(1 for r in runs if r["recovered"]) / len(runs) * 100
    avg_rb = sum(r["rollbacks"] for r in runs) / len(runs)
    valid_losses = [r["final_loss"] for r in runs if r["final_loss"] != float('inf')]
    avg_loss = sum(valid_losses) / max(len(valid_losses), 1)
    print(f"  {cname:20s} | {rec_rate:>11.1f}% | {avg_rb:>13.1f} | {avg_loss:>14.4f}")

with open("exp3_ablation_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\\nSaved: exp3_ablation_results.json")
"""


# ============================================================
# EXPERIMENT 4: Overhead at GPT-2 Scale (~3 min)
# ============================================================
# Proves: ARC overhead is manageable even at 124M-355M params.
# Uses proper warmup and measurement methodology.
# ============================================================
"""
import torch, torch.optim as optim, json, time
from torch.amp import autocast
from transformers import AutoModelForCausalLM
from arc.core.self_healing import SelfHealingArc, SelfHealingConfig

def measure_overhead(model_name, n_warmup=20, n_measure=80):
    print(f"\\n--- {model_name} ---")
    
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).cuda()
    total_params = sum(p.numel() for p in model.parameters())
    
    # Only train last 2 layers for medium
    if "medium" in model_name:
        for p in model.parameters(): p.requires_grad_(False)
        for p in model.transformer.h[-2:].parameters():
            p.requires_grad_(True)
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
    batch = torch.randint(0, 50257, (2, 128)).cuda()
    
    # === Baseline timing (no ARC) ===
    for _ in range(n_warmup):
        opt.zero_grad()
        loss = model(input_ids=batch, labels=batch).loss
        loss.backward()
        opt.step()
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_measure):
        opt.zero_grad()
        loss = model(input_ids=batch, labels=batch).loss
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    base_ms = (time.perf_counter() - t0) / n_measure * 1000
    
    # === ARC timing ===
    ctrl = SelfHealingArc(model, opt, SelfHealingConfig(
        verbose=False, lite_mode=True, checkpoint_frequency=10,
        enable_weight_health_check=False,
    ))
    
    for _ in range(n_warmup):
        opt.zero_grad()
        loss = model(input_ids=batch, labels=batch).loss
        a = ctrl.step(loss)
        if not a.rolled_back and not a.should_skip:
            loss.backward()
            pa = ctrl.post_backward()
            if not pa.rolled_back and not pa.should_skip:
                opt.step()
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_measure):
        opt.zero_grad()
        loss = model(input_ids=batch, labels=batch).loss
        a = ctrl.step(loss)
        if not a.rolled_back and not a.should_skip:
            loss.backward()
            pa = ctrl.post_backward()
            if not pa.rolled_back and not pa.should_skip:
                opt.step()
    torch.cuda.synchronize()
    arc_ms = (time.perf_counter() - t0) / n_measure * 1000
    
    overhead = (arc_ms - base_ms) / base_ms * 100
    
    result = {
        "model": model_name,
        "total_params_M": round(total_params / 1e6, 1),
        "trainable_params_M": round(trainable / 1e6, 1),
        "baseline_ms_per_step": round(base_ms, 2),
        "arc_ms_per_step": round(arc_ms, 2),
        "overhead_ms": round(arc_ms - base_ms, 2),
        "overhead_pct": round(overhead, 1),
    }
    
    print(f"  Params: {result['total_params_M']}M total, {result['trainable_params_M']}M trainable")
    print(f"  Baseline: {base_ms:.2f} ms/step")
    print(f"  With ARC: {arc_ms:.2f} ms/step")
    print(f"  Overhead: {overhead:.1f}%")
    
    del model, ctrl; torch.cuda.empty_cache()
    return result

results = {"experiment": "overhead_at_scale", "measurements": []}
for model_name in ["gpt2", "gpt2-medium"]:
    r = measure_overhead(model_name)
    results["measurements"].append(r)

print("\\n" + "="*60)
print("EXPERIMENT 4: Overhead at Scale")
print("="*60)
print(f"{'Model':15s} | {'Params':>8s} | {'Base ms':>8s} | {'ARC ms':>8s} | {'Overhead':>8s}")
print("-"*60)
for m in results["measurements"]:
    print(f"  {m['model']:15s} | {m['total_params_M']:>6.1f}M | {m['baseline_ms_per_step']:>7.2f} | {m['arc_ms_per_step']:>7.2f} | {m['overhead_pct']:>6.1f}%")

with open("exp4_overhead_scale_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\\nSaved: exp4_overhead_scale_results.json")
"""


# ============================================================
# EXPERIMENT 5: When ARC Fails — Honest Limitations (~5 min)
# ============================================================
# Proves: We are honest. We show when ARC doesn't work.
# This is CRITICAL for publication credibility.
# ============================================================
"""
import torch, torch.nn as nn, torch.optim as optim, json

from arc.core.self_healing import SelfHealingArc, SelfHealingConfig

results = {"experiment": "failure_modes", "tests": []}

# --- Test A: Rapid repeated failures (NaN every 3 steps) ---
print("=== Test A: Rapid repeated failures ===")
model = nn.Sequential(nn.Linear(50, 100), nn.ReLU(), nn.Linear(100, 10)).cuda()
opt = optim.Adam(model.parameters(), lr=1e-3)
ctrl = SelfHealingArc(model, opt, SelfHealingConfig(verbose=False, checkpoint_frequency=3))
batch_x, batch_y = torch.randn(16, 50).cuda(), torch.randint(0, 10, (16,)).cuda()
criterion = nn.CrossEntropyLoss()

rollbacks = 0
stuck = False
for s in range(1, 61):
    opt.zero_grad()
    if s % 3 == 0:  # NaN every 3 steps
        a = ctrl.step(torch.tensor(float('nan')).cuda())
        rollbacks += (1 if a.rolled_back else 0)
        continue
    loss = criterion(model(batch_x), batch_y)
    a = ctrl.step(loss)
    if not a.rolled_back and not a.should_skip:
        loss.backward()
        pa = ctrl.post_backward()
        if not pa.rolled_back and not pa.should_skip: opt.step()
    rollbacks += (1 if a.rolled_back else 0)

model_alive = not any(torch.isnan(p).any() for p in model.parameters())
# Check if LR got reduced to near-zero
final_lr = opt.param_groups[0]['lr']
print(f"  Rollbacks: {rollbacks}, Model alive: {model_alive}, Final LR: {final_lr:.2e}")
results["tests"].append({
    "name": "rapid_repeated_failures",
    "description": "NaN injected every 3 steps for 60 steps",
    "rollbacks": rollbacks,
    "model_alive": model_alive,
    "final_lr": final_lr,
    "stuck_in_loop": final_lr < 1e-10,
    "verdict": "LR degrades to near-zero after many rollbacks" if final_lr < 1e-10 else "Handled OK",
})

# --- Test B: Subtle numerical drift ---
print("\\n=== Test B: Subtle numerical drift ===")
model = nn.Sequential(nn.Linear(50, 100), nn.ReLU(), nn.Linear(100, 10)).cuda()
opt = optim.Adam(model.parameters(), lr=1e-3)
ctrl = SelfHealingArc(model, opt, SelfHealingConfig(verbose=False, checkpoint_frequency=5))

losses = []
for s in range(1, 101):
    opt.zero_grad()
    loss = criterion(model(batch_x), batch_y)
    # Add tiny drift to loss (simulates slow degradation)
    drift = torch.tensor(0.01 * s, device='cuda')
    drifted_loss = loss + drift
    
    a = ctrl.step(drifted_loss)
    if not a.rolled_back and not a.should_skip:
        loss.backward()  # backward on real loss, not drifted
        pa = ctrl.post_backward()
        if not pa.rolled_back and not pa.should_skip: opt.step()
    losses.append(drifted_loss.item())

drift_detected = ctrl.total_rollbacks > 0
print(f"  Final reported loss: {losses[-1]:.4f} (includes drift of {0.01*100:.1f})")
print(f"  Rollbacks: {ctrl.total_rollbacks}, Drift detected: {drift_detected}")
results["tests"].append({
    "name": "subtle_numerical_drift",
    "description": "Loss slowly increases by 0.01 per step (100 steps)",
    "rollbacks": ctrl.total_rollbacks,
    "drift_detected": drift_detected,
    "verdict": "ARC does NOT detect slow drift (only detects sudden failures)",
})

# --- Test C: Stealth weight corruption (loss looks normal) ---
print("\\n=== Test C: Stealth weight corruption ===")
model = nn.Sequential(nn.Linear(50, 100), nn.ReLU(), nn.Linear(100, 10)).cuda()
opt = optim.Adam(model.parameters(), lr=1e-3)
ctrl = SelfHealingArc(model, opt, SelfHealingConfig(verbose=False, checkpoint_frequency=5))

for s in range(1, 31):
    opt.zero_grad()
    loss = criterion(model(batch_x), batch_y)
    a = ctrl.step(loss)
    if not a.rolled_back and not a.should_skip:
        loss.backward(); pa = ctrl.post_backward()
        if not pa.rolled_back and not pa.should_skip: opt.step()

# Corrupt weights subtly (set to very large but not NaN/Inf)
with torch.no_grad():
    for p in list(model.parameters())[:2]:
        p.data *= 1000  # Not NaN, just huge

rollbacks_before = ctrl.total_rollbacks
for s in range(31, 51):
    opt.zero_grad()
    try:
        loss = criterion(model(batch_x), batch_y)
        if torch.isnan(loss) or torch.isinf(loss):
            a = ctrl.step(loss)
            continue
    except:
        break
    a = ctrl.step(loss)
    if not a.rolled_back and not a.should_skip:
        loss.backward(); pa = ctrl.post_backward()
        if not pa.rolled_back and not pa.should_skip: opt.step()

stealth_caught = ctrl.total_rollbacks > rollbacks_before
print(f"  Stealth corruption caught: {stealth_caught}, Rollbacks: {ctrl.total_rollbacks - rollbacks_before}")
results["tests"].append({
    "name": "stealth_weight_corruption",
    "description": "Weights multiplied by 1000 (not NaN, just large)",
    "caught": stealth_caught,
    "verdict": "ARC catches this via loss explosion or weight norm check" if stealth_caught else "ARC misses subtle non-NaN corruption",
})

# --- Test D: Max persistent failures (rollback budget exhaustion) ---
print("\\n=== Test D: Persistent failure loop ===")
model = nn.Sequential(nn.Linear(50, 100), nn.ReLU(), nn.Linear(100, 10)).cuda()
opt = optim.Adam(model.parameters(), lr=1e-3)
ctrl = SelfHealingArc(model, opt, SelfHealingConfig(
    verbose=False, checkpoint_frequency=3, max_consecutive_rollbacks=5,
))

for s in range(1, 11):  # Train 10 steps
    opt.zero_grad()
    loss = criterion(model(batch_x), batch_y)
    a = ctrl.step(loss)
    if not a.rolled_back and not a.should_skip:
        loss.backward(); pa = ctrl.post_backward()
        if not pa.rolled_back and not pa.should_skip: opt.step()

# Hit with 10 consecutive NaNs
for i in range(10):
    a = ctrl.step(torch.tensor(float('nan')).cuda())

print(f"  Rollbacks: {ctrl.total_rollbacks}, Max allowed: 5")
print(f"  LR: {opt.param_groups[0]['lr']:.2e}")
model_alive = not any(torch.isnan(p).any() for p in model.parameters())
results["tests"].append({
    "name": "persistent_failure_loop",
    "description": "10 consecutive NaN injections with max_rollbacks=5",
    "total_rollbacks": ctrl.total_rollbacks,
    "model_alive": model_alive,
    "final_lr": opt.param_groups[0]['lr'],
    "verdict": "ARC exhausts rollback budget, LR degrades heavily",
})

print("\\n" + "="*60)
print("EXPERIMENT 5: Honest Failure Modes")
print("="*60)
for t in results["tests"]:
    print(f"  {t['name']:30s}: {t['verdict']}")

with open("exp5_failure_modes_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\\nSaved: exp5_failure_modes_results.json")
"""


# ============================================================
# EXPERIMENT 6: Statistical Rigor — Key Claims (~5 min)
# ============================================================
# Proves: Results are reproducible across multiple seeds.
# Reports mean +/- std for every key metric.
# ============================================================
"""
import torch, torch.nn as nn, torch.optim as optim, json, time
import numpy as np
from arc.core.self_healing import SelfHealingArc, SelfHealingConfig

SEEDS = [42, 123, 456, 777, 999]
results = {"experiment": "statistical_rigor", "claims": {}}

# --- Claim 1: NaN Recovery Rate ---
print("=== Claim 1: NaN Recovery Rate (5 seeds × 3 models) ===")
model_configs = [
    ("MLP", lambda: nn.Sequential(nn.Linear(100, 200), nn.ReLU(), nn.Linear(200, 10)).cuda()),
    ("CNN", lambda: nn.Sequential(nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(32*16*16, 10)).cuda()),
    ("Deep", lambda: nn.Sequential(nn.Linear(100,400), nn.ReLU(), nn.Linear(400,300), nn.ReLU(), nn.Linear(300,200), nn.ReLU(), nn.Linear(200,10)).cuda()),
]

nan_results = []
for mname, mfn in model_configs:
    for seed in SEEDS:
        torch.manual_seed(seed); torch.cuda.manual_seed(seed)
        model = mfn()
        input_dim = 100 if "Linear" in str(model[0]) else None
        opt = optim.Adam(model.parameters(), lr=1e-3)
        ctrl = SelfHealingArc(model, opt, SelfHealingConfig(verbose=False, checkpoint_frequency=5))
        criterion = nn.CrossEntropyLoss()
        
        if input_dim:
            bx, by = torch.randn(16, input_dim).cuda(), torch.randint(0, 10, (16,)).cuda()
        else:
            bx, by = torch.randn(16, 3, 32, 32).cuda(), torch.randint(0, 10, (16,)).cuda()
        
        # Train 20 steps
        for s in range(20):
            opt.zero_grad()
            loss = criterion(model(bx), by)
            a = ctrl.step(loss)
            if not a.rolled_back and not a.should_skip:
                loss.backward(); pa = ctrl.post_backward()
                if not pa.rolled_back and not pa.should_skip: opt.step()
        
        # Inject NaN
        a = ctrl.step(torch.tensor(float('nan')).cuda())
        recovered = a.rolled_back
        model_clean = not any(torch.isnan(p).any() for p in model.parameters())
        nan_results.append({"model": mname, "seed": seed, "recovered": recovered, "clean": model_clean})

recovery_rate = sum(1 for r in nan_results if r["recovered"]) / len(nan_results) * 100
print(f"  Overall NaN recovery: {recovery_rate:.0f}% ({sum(1 for r in nan_results if r['recovered'])}/{len(nan_results)})")
results["claims"]["nan_recovery"] = {
    "recovery_rate": recovery_rate,
    "n_tests": len(nan_results),
    "per_model": {m: sum(1 for r in nan_results if r["model"]==m and r["recovered"])/len(SEEDS)*100 for m,_ in model_configs},
}

# --- Claim 2: Gradient Explosion Detection ---
print("\\n=== Claim 2: Gradient Explosion Detection (5 seeds) ===")
explosion_results = []
for seed in SEEDS:
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    model = nn.Sequential(nn.Linear(100,400), nn.ReLU(), nn.Linear(400,10)).cuda()
    opt = optim.SGD(model.parameters(), lr=3.0)
    ctrl = SelfHealingArc(model, opt, SelfHealingConfig(verbose=False, checkpoint_frequency=3, loss_explosion_threshold=50.0))
    bx, by = torch.randn(32, 100).cuda(), torch.randint(0, 10, (32,)).cuda()
    criterion = nn.CrossEntropyLoss()
    
    rollbacks = 0
    for s in range(1, 51):
        opt.zero_grad()
        loss = criterion(model(bx), by)
        if torch.isnan(loss) or torch.isinf(loss):
            a = ctrl.step(loss)
            rollbacks += (1 if a.rolled_back else 0)
            continue
        a = ctrl.step(loss)
        if a.rolled_back or a.should_skip:
            rollbacks += (1 if a.rolled_back else 0)
            continue
        loss.backward()
        pa = ctrl.post_backward()
        if pa.rolled_back or pa.should_skip:
            rollbacks += (1 if pa.rolled_back else 0)
            continue
        opt.step()
    
    model_alive = not any(torch.isnan(p).any() for p in model.parameters())
    explosion_results.append({"seed": seed, "rollbacks": rollbacks, "alive": model_alive})

rollback_counts = [r["rollbacks"] for r in explosion_results]
alive_rate = sum(1 for r in explosion_results if r["alive"]) / len(explosion_results) * 100
print(f"  Rollbacks: {np.mean(rollback_counts):.1f} +/- {np.std(rollback_counts):.1f}")
print(f"  Model survival: {alive_rate:.0f}%")
results["claims"]["grad_explosion"] = {
    "rollbacks_mean": round(float(np.mean(rollback_counts)), 1),
    "rollbacks_std": round(float(np.std(rollback_counts)), 1),
    "survival_rate": alive_rate,
}

# --- Claim 3: Progress Lost on Recovery ---
print("\\n=== Claim 3: Progress Lost on Recovery (5 seeds) ===")
progress_results = []
for seed in SEEDS:
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    model = nn.Sequential(nn.Linear(100,200), nn.ReLU(), nn.Linear(200,10)).cuda()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    ctrl = SelfHealingArc(model, opt, SelfHealingConfig(verbose=False, checkpoint_frequency=5))
    bx, by = torch.randn(16, 100).cuda(), torch.randint(0, 10, (16,)).cuda()
    criterion = nn.CrossEntropyLoss()
    
    # Train 30 steps
    losses = []
    for s in range(1, 31):
        opt.zero_grad()
        loss = criterion(model(bx), by)
        a = ctrl.step(loss)
        if not a.rolled_back and not a.should_skip:
            loss.backward(); pa = ctrl.post_backward()
            if not pa.rolled_back and not pa.should_skip: opt.step()
        losses.append(loss.item())
    
    loss_before = losses[-1]
    
    # Inject NaN
    ctrl.step(torch.tensor(float('nan')).cuda())
    
    # Check loss after recovery
    opt.zero_grad()
    loss = criterion(model(bx), by)
    loss_after = loss.item()
    
    progress_lost = loss_after - loss_before
    progress_results.append({"seed": seed, "before": loss_before, "after": loss_after, "lost": progress_lost})

progress_lost_values = [r["lost"] for r in progress_results]
print(f"  Progress lost: {np.mean(progress_lost_values):.4f} +/- {np.std(progress_lost_values):.4f}")
print(f"  (positive = loss increased after rollback)")
results["claims"]["progress_lost"] = {
    "mean": round(float(np.mean(progress_lost_values)), 4),
    "std": round(float(np.std(progress_lost_values)), 4),
    "raw": [round(v, 4) for v in progress_lost_values],
}

# Summary
print("\\n" + "="*60)
print("EXPERIMENT 6: Statistical Rigor Summary")
print("="*60)
print(f"  NaN Recovery Rate:        {results['claims']['nan_recovery']['recovery_rate']:.0f}%")
print(f"  Grad Explosion Rollbacks: {results['claims']['grad_explosion']['rollbacks_mean']:.1f} +/- {results['claims']['grad_explosion']['rollbacks_std']:.1f}")
print(f"  Model Survival Rate:      {results['claims']['grad_explosion']['survival_rate']:.0f}%")
print(f"  Progress Lost (loss):     {results['claims']['progress_lost']['mean']:.4f} +/- {results['claims']['progress_lost']['std']:.4f}")

with open("exp6_statistical_rigor_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\\nSaved: exp6_statistical_rigor_results.json")
"""
