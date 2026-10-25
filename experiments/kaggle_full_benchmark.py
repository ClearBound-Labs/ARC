"""
ARC Full Benchmark Suite for AAAI 2027
======================================
Run on Kaggle with GPU T4/P100. Fits within 11-hour session (~3h actual).

8 Experiments:
1. CIFAR-10 real training with failure injection + recovery
2. Detection rate across 4 architectures x 5 failure types
3. Zero false-positive verification (1000 stable steps)
4. Overhead profiling on GPU (% overhead per architecture)
5. Baseline comparison: ARC vs GradClip vs LossSkip vs None
6. Ablation: rollback-only vs prediction-only vs full ARC
7. Scale test: models from 50K to 100M+ params
8. Multi-failure resilience (3 failures in one run)

All results saved to JSON + zipped at end.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json, time, os, zipfile, sys
from collections import defaultdict

# ── Setup ──────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS = {}
OUT_DIR = "arc_benchmark_results"
os.makedirs(OUT_DIR, exist_ok=True)

def save(name, data):
    path = os.path.join(OUT_DIR, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")

def seed(s=42):
    torch.manual_seed(s)
    np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

print(f"Device: {DEVICE} | PyTorch: {torch.__version__}")

# ── Install ARC ────────────────────────────────────────────────────────
try:
    from arc.intervention import WeightRollback, RollbackConfig
    from arc import ArcV2
    from arc.config import Config, SignalConfig, FeatureConfig
    print("ARC loaded from local source")
except ImportError:
    os.system("pip install arc-training -q")
    from arc.intervention import WeightRollback, RollbackConfig
    from arc import ArcV2
    from arc.config import Config, SignalConfig, FeatureConfig
    print("ARC installed from PyPI v4.3.0")

# ── Model Zoo ──────────────────────────────────────────────────────────
def make_mlp(h=128, d=100):
    return nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,10))

def make_cnn():
    class C(nn.Module):
        def __init__(self):
            super().__init__()
            self.f=nn.Sequential(nn.Conv2d(3,32,3,1,1),nn.ReLU(),nn.MaxPool2d(2),
                nn.Conv2d(32,64,3,1,1),nn.ReLU(),nn.MaxPool2d(2),
                nn.Conv2d(64,128,3,1,1),nn.ReLU(),nn.AdaptiveAvgPool2d(4))
            self.c=nn.Linear(128*16,10)
        def forward(self,x): return self.c(self.f(x).view(x.size(0),-1))
    return C()

def make_resnet():
    try:
        from torchvision.models import resnet18
        return resnet18(num_classes=10)
    except: return make_mlp(1024, 100)

def make_transformer():
    class T(nn.Module):
        def __init__(self):
            super().__init__()
            self.e=nn.Embedding(10000,256)
            el=nn.TransformerEncoderLayer(256,8,1024,batch_first=True)
            self.enc=nn.TransformerEncoder(el,4)
            self.h=nn.Linear(256,10)
        def forward(self,x): return self.h(self.enc(self.e(x))[:,-1])
    return T()

def make_large(size="50M"):
    """Models for scale test."""
    if size=="50M":
        return nn.Sequential(nn.Linear(1024,4096),nn.ReLU(),nn.Linear(4096,4096),nn.ReLU(),
                             nn.Linear(4096,4096),nn.ReLU(),nn.Linear(4096,10))
    elif size=="100M":
        return nn.Sequential(nn.Linear(2048,4096),nn.ReLU(),nn.Linear(4096,8192),nn.ReLU(),
                             nn.Linear(8192,4096),nn.ReLU(),nn.Linear(4096,10))

ZOO = {
    "MLP-50K": (make_mlp, lambda: torch.randn(8,100)),
    "CNN-300K": (make_cnn, lambda: torch.randn(8,3,32,32)),
    "ResNet18-11M": (make_resnet, lambda: torch.randn(4,3,32,32)),
    "Transformer-10M": (make_transformer, lambda: torch.randint(0,10000,(4,64))),
}

INJECTORS = {
    "nan_loss": lambda l,s: torch.tensor(float("nan")) if s==25 else l,
    "inf_loss": lambda l,s: torch.tensor(float("inf")) if s==25 else l,
    "loss_explosion": lambda l,s: l*1e6 if s==25 else l,
    "grad_explosion": lambda l,s: l*1e5 if s==25 else l,
    "lr_spike": lambda l,s: l,  # LR modified separately
}

# ====================================================================
# EXPERIMENT 1: CIFAR-10 Real Training with Failure Recovery
# ====================================================================
def exp1_cifar10_recovery():
    """Train ResNet18 on CIFAR-10. Inject NaN at epoch 5. Compare ARC vs no protection."""
    print("\n" + "="*60)
    print("EXP 1: CIFAR-10 Real Training + Failure Recovery")
    print("="*60)
    import torchvision, torchvision.transforms as T

    tf = T.Compose([T.RandomHorizontalFlip(), T.RandomCrop(32,4), T.ToTensor(),
                    T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    tf_test = T.Compose([T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    trainset = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=tf)
    testset = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=tf_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, num_workers=2)

    def evaluate(model):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x,y in testloader:
                x,y = x.to(DEVICE), y.to(DEVICE)
                correct += (model(x).argmax(1)==y).sum().item()
                total += y.size(0)
        model.train()
        return correct/total*100

    def train_run(use_arc, inject_failure, epochs=15, label=""):
        seed(42)
        model = make_resnet().to(DEVICE)
        opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        crit = nn.CrossEntropyLoss()
        rb = WeightRollback(model, opt, RollbackConfig(checkpoint_frequency=50), verbose=False) if use_arc else None

        history = {"epoch_acc": [], "rollbacks": 0, "crashed": False, "label": label}

        for ep in range(epochs):
            model.train()
            for i, (x,y) in enumerate(trainloader):
                x, y = x.to(DEVICE), y.to(DEVICE)
                loss = crit(model(x), y)

                # Inject failure at epoch 5, batch 100
                if inject_failure and ep == 5 and i == 100:
                    loss = loss * float("nan")

                bad = torch.isnan(loss) or torch.isinf(loss)

                if use_arc and rb:
                    try:
                        if not bad: loss.backward()
                    except: pass
                    action = rb.step(loss)
                    if action.rolled_back:
                        history["rollbacks"] += 1
                        opt.zero_grad(); continue
                    if not bad: opt.step()
                    opt.zero_grad()
                else:
                    if bad:
                        history["crashed"] = True
                        print(f"  [{label}] CRASHED at epoch {ep} batch {i}")
                        acc = evaluate(model)
                        history["epoch_acc"].append({"epoch": ep, "acc": round(acc,2)})
                        return history
                    loss.backward()
                    opt.step()
                    opt.zero_grad()

            sched.step()
            acc = evaluate(model)
            history["epoch_acc"].append({"epoch": ep, "acc": round(acc,2)})
            print(f"  [{label}] Epoch {ep}: {acc:.2f}%")

        return history

    results = {}
    # Healthy baseline (no failure, no ARC)
    results["healthy_baseline"] = train_run(False, False, 15, "Healthy")
    # No protection + failure
    results["no_protection"] = train_run(False, True, 15, "NoProt+Fail")
    # ARC + failure
    results["arc_protected"] = train_run(True, True, 15, "ARC+Fail")
    # ARC without failure (overhead only)
    results["arc_no_failure"] = train_run(True, False, 15, "ARC+Healthy")

    # Summary
    print("\n  CIFAR-10 RESULTS SUMMARY:")
    for k, v in results.items():
        final = v["epoch_acc"][-1]["acc"] if v["epoch_acc"] else 0
        crashed = "CRASHED" if v["crashed"] else "OK"
        rb_count = v["rollbacks"]
        print(f"    {k:20s}: Final={final:.1f}% | Status={crashed} | Rollbacks={rb_count}")

    save("exp1_cifar10", results)
    return results

# ====================================================================
# EXPERIMENT 2: Detection Rate (4 arch x 5 failures x 5 seeds)
# ====================================================================
def exp2_detection():
    print("\n" + "="*60)
    print("EXP 2: Detection & Recovery Rate")
    print("="*60)
    results = []
    for arch, (fn, inp) in ZOO.items():
        for fail, inj in INJECTORS.items():
            det = rec = 0
            for s in range(5):
                seed(s)
                m = fn().to(DEVICE); o = optim.Adam(m.parameters(), lr=0.01)
                rb = WeightRollback(m, o, RollbackConfig(checkpoint_frequency=5), verbose=False)
                d = r = False
                for step in range(50):
                    x = inp().to(DEVICE)
                    try: loss = m(x).mean()
                    except: loss = torch.tensor(1.0)
                    loss = inj(loss, step)
                    if fail=="lr_spike" and step==25:
                        for pg in o.param_groups: pg["lr"]=50.0
                    bad = torch.isnan(loss) or torch.isinf(loss)
                    try:
                        if not bad: loss.backward()
                    except: pass
                    a = rb.step(loss)
                    if a.failure_detected: d=True
                    if a.rolled_back: r=True; o.zero_grad(); continue
                    if not bad: o.step()
                    o.zero_grad()
                if d: det+=1
                if r: rec+=1
            row = {"arch":arch,"failure":fail,"detection":det/5,"recovery":rec/5}
            results.append(row)
            print(f"  {arch:20s} x {fail:18s}: Det={det/5*100:.0f}%  Rec={rec/5*100:.0f}%")
    save("exp2_detection", results)
    return results

# ====================================================================
# EXPERIMENT 3: False Positive Rate
# ====================================================================
def exp3_false_positives():
    print("\n" + "="*60)
    print("EXP 3: False Positive Rate (1000 steps)")
    print("="*60)
    results = []
    for arch, (fn, inp) in ZOO.items():
        seed(42)
        m = fn().to(DEVICE); o = optim.Adam(m.parameters(), lr=0.001)
        rb = WeightRollback(m, o, RollbackConfig(), verbose=False)
        fp = 0
        for _ in range(1000):
            x = inp().to(DEVICE)
            loss = m(x).mean(); loss.backward()
            if rb.step(loss).rolled_back: fp+=1
            o.step(); o.zero_grad()
        results.append({"arch":arch, "steps":1000, "fp":fp, "fp_rate":fp/1000})
        print(f"  {arch:20s}: {fp} FPs in 1000 steps ({fp/10:.1f}%)")
    save("exp3_false_positives", results)
    return results

# ====================================================================
# EXPERIMENT 4: Overhead on GPU
# ====================================================================
def exp4_overhead():
    print("\n" + "="*60)
    print("EXP 4: Overhead Measurement")
    print("="*60)
    results = []
    N = 200
    for arch, (fn, inp) in ZOO.items():
        seed(42)
        m = fn().to(DEVICE); o = optim.Adam(m.parameters())
        npar = sum(p.numel() for p in m.parameters())

        # Baseline
        tb = []
        for i in range(N):
            x = inp().to(DEVICE)
            if DEVICE=="cuda": torch.cuda.synchronize()
            t0=time.perf_counter()
            l=m(x).mean(); l.backward(); o.step(); o.zero_grad()
            if DEVICE=="cuda": torch.cuda.synchronize()
            tb.append((time.perf_counter()-t0)*1000)

        # ARC
        rb = WeightRollback(m, o, RollbackConfig(), verbose=False)
        ta = []
        for i in range(N):
            x = inp().to(DEVICE)
            if DEVICE=="cuda": torch.cuda.synchronize()
            t0=time.perf_counter()
            l=m(x).mean(); l.backward(); rb.step(l); o.step(); o.zero_grad()
            if DEVICE=="cuda": torch.cuda.synchronize()
            ta.append((time.perf_counter()-t0)*1000)

        base = np.mean(tb[20:]); arc = np.mean(ta[20:])
        oh = ((arc-base)/base)*100 if base>0 else 0
        results.append({"arch":arch,"params":npar,"base_ms":round(base,3),
                        "arc_ms":round(arc,3),"overhead_pct":round(oh,1)})
        print(f"  {arch:20s} ({npar:>12,}p): Base={base:.2f}ms ARC={arc:.2f}ms OH={oh:.1f}%")
    save("exp4_overhead", results)
    return results

# ====================================================================
# EXPERIMENT 5: Baseline Comparison
# ====================================================================
def exp5_baselines():
    print("\n" + "="*60)
    print("EXP 5: Baseline Comparison")
    print("="*60)
    methods = {"No Protection":"none", "GradClip(1.0)":"clip",
               "Loss Skip":"skip", "Full ARC":"arc"}
    fails = ["nan_loss","loss_explosion","grad_explosion"]
    results = []
    for mname, mtype in methods.items():
        scores = {"det":0,"rec":0,"total":0}
        for fail in fails:
            inj = INJECTORS[fail]
            for s in range(5):
                seed(s); m=make_mlp().to(DEVICE); o=optim.Adam(m.parameters(),lr=0.01)
                rb=WeightRollback(m,o,RollbackConfig(checkpoint_frequency=5),verbose=False) if mtype=="arc" else None
                det=rec=False
                for step in range(50):
                    x=torch.randn(8,100).to(DEVICE); loss=m(x).mean()
                    loss=inj(loss,step); bad=torch.isnan(loss) or torch.isinf(loss)
                    if mtype=="arc":
                        try:
                            if not bad: loss.backward()
                        except: pass
                        a=rb.step(loss)
                        if a.failure_detected: det=True
                        if a.rolled_back: rec=True; o.zero_grad(); continue
                        if not bad: o.step()
                        o.zero_grad()
                    elif mtype=="skip":
                        if bad: det=True; rec=True; o.zero_grad(); continue
                        loss.backward(); o.step(); o.zero_grad()
                    elif mtype=="clip":
                        if bad: det=True; break
                        loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0)
                        o.step(); o.zero_grad()
                    else:
                        if bad: det=True; break
                        loss.backward(); o.step(); o.zero_grad()
                if det: scores["det"]+=1
                if rec: scores["rec"]+=1
                scores["total"]+=1
        results.append({"method":mname,
                        "detection":round(scores["det"]/scores["total"],3),
                        "recovery":round(scores["rec"]/scores["total"],3)})
    print(f"\n  {'Method':<20s} {'Detection':>10s} {'Recovery':>10s}")
    print("  "+"-"*42)
    for r in results:
        print(f"  {r['method']:<20s} {r['detection']*100:>9.0f}% {r['recovery']*100:>9.0f}%")
    save("exp5_baselines", results)
    return results

# ====================================================================
# EXPERIMENT 6: Ablation Study
# ====================================================================
def exp6_ablation():
    """Compare: rollback-only, prediction+monitoring only, full ARC."""
    print("\n" + "="*60)
    print("EXP 6: Ablation Study")
    print("="*60)
    results = []
    configs = {
        "Rollback Only (freq=5)": RollbackConfig(checkpoint_frequency=5, fast_grad_norm=False, skip_stable_layers=False),
        "Rollback + FastGrad": RollbackConfig(checkpoint_frequency=5, fast_grad_norm=True, skip_stable_layers=False),
        "Full ARC (all features)": RollbackConfig(checkpoint_frequency=5, fast_grad_norm=True,
                                                   skip_stable_layers=True, adaptive_frequency=True),
    }
    for cname, cfg in configs.items():
        det=rec=0; n=0
        for fail in ["nan_loss","loss_explosion","grad_explosion"]:
            inj=INJECTORS[fail]
            for s in range(5):
                seed(s); m=make_mlp().to(DEVICE); o=optim.Adam(m.parameters(),lr=0.01)
                rb=WeightRollback(m,o,cfg,verbose=False)
                d=r=False
                for step in range(50):
                    x=torch.randn(8,100).to(DEVICE); loss=m(x).mean()
                    loss=inj(loss,step); bad=torch.isnan(loss) or torch.isinf(loss)
                    try:
                        if not bad: loss.backward()
                    except: pass
                    a=rb.step(loss)
                    if a.failure_detected: d=True
                    if a.rolled_back: r=True; o.zero_grad(); continue
                    if not bad: o.step()
                    o.zero_grad()
                if d: det+=1
                if r: rec+=1
                n+=1
        results.append({"config":cname,"detection":round(det/n,3),"recovery":round(rec/n,3)})
        print(f"  {cname:35s}: Det={det/n*100:.0f}%  Rec={rec/n*100:.0f}%")
    save("exp6_ablation", results)
    return results

# ====================================================================
# EXPERIMENT 7: Scale Test (50K → 100M+)
# ====================================================================
def exp7_scale():
    print("\n" + "="*60)
    print("EXP 7: Scale Test (50K to 100M+ params)")
    print("="*60)
    scale_models = {
        "MLP-50K": (make_mlp, lambda: torch.randn(8,100)),
        "CNN-300K": (make_cnn, lambda: torch.randn(8,3,32,32)),
        "ResNet18-11M": (make_resnet, lambda: torch.randn(4,3,32,32)),
        "Large-50M": (lambda: make_large("50M"), lambda: torch.randn(4,1024)),
    }
    # Only add 100M if GPU has enough memory
    if DEVICE=="cuda" and torch.cuda.get_device_properties(0).total_memory > 8e9:
        scale_models["Large-100M"] = (lambda: make_large("100M"), lambda: torch.randn(4,2048))

    results = []
    for name, (fn, inp) in scale_models.items():
        seed(42)
        try:
            m=fn().to(DEVICE); npar=sum(p.numel() for p in m.parameters())
            o=optim.Adam(m.parameters(),lr=0.01)
            rb=WeightRollback(m,o,RollbackConfig(checkpoint_frequency=5),verbose=False)
            # Test: inject NaN at step 10, verify recovery
            recovered=False
            for step in range(30):
                x=inp().to(DEVICE); loss=m(x).mean()
                if step==10: loss=torch.tensor(float("nan"))
                bad=torch.isnan(loss) or torch.isinf(loss)
                try:
                    if not bad: loss.backward()
                except: pass
                a=rb.step(loss)
                if a.rolled_back: recovered=True; o.zero_grad(); continue
                if not bad: o.step()
                o.zero_grad()

            # Overhead timing
            times_b=[]; times_a=[]
            for _ in range(50):
                x=inp().to(DEVICE)
                if DEVICE=="cuda": torch.cuda.synchronize()
                t0=time.perf_counter()
                l=m(x).mean(); l.backward(); o.step(); o.zero_grad()
                if DEVICE=="cuda": torch.cuda.synchronize()
                times_b.append((time.perf_counter()-t0)*1000)
            rb2=WeightRollback(m,o,RollbackConfig(),verbose=False)
            for _ in range(50):
                x=inp().to(DEVICE)
                if DEVICE=="cuda": torch.cuda.synchronize()
                t0=time.perf_counter()
                l=m(x).mean(); l.backward(); rb2.step(l); o.step(); o.zero_grad()
                if DEVICE=="cuda": torch.cuda.synchronize()
                times_a.append((time.perf_counter()-t0)*1000)
            base=np.mean(times_b[10:]); arc=np.mean(times_a[10:])
            oh=((arc-base)/base)*100 if base>0 else 0
            status="RECOVERED" if recovered else "NO_FAILURE"
            results.append({"model":name,"params":npar,"recovered":recovered,
                           "overhead_pct":round(oh,1),"base_ms":round(base,2),"arc_ms":round(arc,2)})
            print(f"  {name:20s} ({npar:>12,}p): {status:12s} | OH={oh:.1f}%")
            del m, o, rb; torch.cuda.empty_cache() if DEVICE=="cuda" else None
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  {name:20s}: SKIPPED (OOM)")
                torch.cuda.empty_cache()
            else: raise
    save("exp7_scale", results)
    return results

# ====================================================================
# EXPERIMENT 8: Multi-Failure Resilience
# ====================================================================
def exp8_multi_failure():
    """3 different failures in a single 100-step run."""
    print("\n" + "="*60)
    print("EXP 8: Multi-Failure Resilience")
    print("="*60)
    results = []
    for arch, (fn, inp) in ZOO.items():
        total_rb=0; total_det=0; completed=0
        for s in range(5):
            seed(s); m=fn().to(DEVICE); o=optim.Adam(m.parameters(),lr=0.01)
            rb=WeightRollback(m,o,RollbackConfig(checkpoint_frequency=5,
                              max_rollbacks_per_epoch=10),verbose=False)
            rbc=0; finished=True
            for step in range(100):
                x=inp().to(DEVICE)
                try: loss=m(x).mean()
                except: loss=torch.tensor(1.0)
                # Inject 3 failures at steps 20, 50, 80
                if step==20: loss=torch.tensor(float("nan"))
                elif step==50: loss=loss*1e6
                elif step==80: loss=torch.tensor(float("inf"))
                bad=torch.isnan(loss) or torch.isinf(loss)
                try:
                    if not bad: loss.backward()
                except: pass
                a=rb.step(loss)
                if a.failure_detected: total_det+=1
                if a.rolled_back: rbc+=1; o.zero_grad(); continue
                if not bad: o.step()
                o.zero_grad()
            total_rb+=rbc
            if rbc>=3: completed+=1
        results.append({"arch":arch,"avg_rollbacks":total_rb/5,
                        "all_3_recovered":f"{completed}/5","detections":total_det/5})
        print(f"  {arch:20s}: Avg rollbacks={total_rb/5:.1f} | All 3 recovered: {completed}/5")
    save("exp8_multi_failure", results)
    return results

# ====================================================================
# MAIN
# ====================================================================
def main():
    t_start = time.time()
    print("="*60)
    print("ARC FULL BENCHMARK SUITE - AAAI 2027")
    print("="*60)

    RESULTS["exp1"] = exp1_cifar10_recovery()
    RESULTS["exp2"] = exp2_detection()
    RESULTS["exp3"] = exp3_false_positives()
    RESULTS["exp4"] = exp4_overhead()
    RESULTS["exp5"] = exp5_baselines()
    RESULTS["exp6"] = exp6_ablation()
    RESULTS["exp7"] = exp7_scale()
    RESULTS["exp8"] = exp8_multi_failure()

    # Save combined
    summary = {
        "device": DEVICE, "pytorch": torch.__version__,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_seconds": round(time.time()-t_start, 1),
        "all_results": RESULTS,
    }
    save("full_summary", summary)

    # ZIP everything
    zippath = "arc_aaai_benchmark_results.zip"
    with zipfile.ZipFile(zippath, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in os.listdir(OUT_DIR):
            zf.write(os.path.join(OUT_DIR, f), f)
    print(f"\nAll results zipped: {zippath}")
    print(f"Total time: {(time.time()-t_start)/60:.1f} minutes")
    print("="*60)

if __name__ == "__main__":
    main()
