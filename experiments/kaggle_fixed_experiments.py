"""
FIXED Experiment 3 & 6: Harder ablation + Grad explosion stats
===============================================================
These replace the weak experiments from the first run.
"""


# ============================================================
# FIXED EXPERIMENT 3: Harder Ablation Study (~5 min)
# ============================================================
# The old ablation was too easy — rollback alone handled everything.
# These scenarios are designed so that SPECIFIC components matter:
#
# Scenario A: "Recurring instability" (high LR SGD, 200 steps)
#   - Without grad clipping: explosion → NaN → rollback → explosion again
#   - Without LR reduction: keeps hitting same explosion threshold
#
# Scenario B: "Silent weight decay" (progressive weight corruption)
#   - Without weight health check: corruption goes undetected
#
# Scenario C: "Persistent failure loop" (NaN every 5 steps)
#   - Without perturbation: model stuck after rollback
#   - Without forecasting: no early warning
# ============================================================
"""
import torch, torch.nn as nn, torch.optim as optim, json
from arc.core.self_healing import SelfHealingArc, SelfHealingConfig

class UnstableNet(nn.Module):
    '''Deep net that's prone to instability with high LR.'''
    def __init__(self):
        super().__init__()
        # No BatchNorm, no residuals = easy to destabilize
        self.net = nn.Sequential(
            nn.Linear(100, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 10),
        )
    def forward(self, x): return self.net(x)

# Ablation configs — each disables ONE component
configs = {
    "Full ARC": SelfHealingConfig(
        verbose=False, checkpoint_frequency=5, check_frequency=1,
    ),
    "No Grad Clip": SelfHealingConfig(
        verbose=False, checkpoint_frequency=5, check_frequency=1,
        gradient_explosion_threshold=1e30,  # effectively disabled
        gradient_clip_norm=1e30,            # no clipping
    ),
    "No Weight Health": SelfHealingConfig(
        verbose=False, checkpoint_frequency=5, check_frequency=1,
        enable_weight_health_check=False,
    ),
    "No LR Reduction": SelfHealingConfig(
        verbose=False, checkpoint_frequency=5, check_frequency=1,
        lr_reduction_factor=1.0,  # LR stays the same after rollback
    ),
    "No Perturbation": SelfHealingConfig(
        verbose=False, checkpoint_frequency=5, check_frequency=1,
        weight_perturbation_scale=0.0,
    ),
    "No Forecasting": SelfHealingConfig(
        verbose=False, checkpoint_frequency=5, check_frequency=1,
        enable_forecasting=False,
    ),
}

def run_scenario_A(config_name, config, seed):
    '''Recurring instability: SGD lr=1.0 on deep net for 200 steps.
    The high LR causes periodic gradient explosions.'''
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    model = UnstableNet().cuda()
    opt = optim.SGD(model.parameters(), lr=1.0, momentum=0.9)
    ctrl = SelfHealingArc(model, opt, config)
    bx = torch.randn(32, 100).cuda()
    by = torch.randint(0, 10, (32,)).cuda()
    criterion = nn.CrossEntropyLoss()
    
    rollbacks, valid_steps = 0, 0
    for s in range(1, 201):
        opt.zero_grad()
        try:
            loss = criterion(model(bx), by)
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
        valid_steps += 1
    
    alive = not any(torch.isnan(p).any() for p in model.parameters())
    final_lr = opt.param_groups[0]['lr']
    return {"config": config_name, "scenario": "recurring_instability", "seed": seed,
            "rollbacks": rollbacks, "valid_steps": valid_steps, 
            "alive": alive, "final_lr": final_lr}

def run_scenario_B(config_name, config, seed):
    '''Silent weight decay: gradually zero out weights every 10 steps.
    Only weight health check can catch this.'''
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    model = UnstableNet().cuda()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    ctrl = SelfHealingArc(model, opt, config)
    bx = torch.randn(32, 100).cuda()
    by = torch.randint(0, 10, (32,)).cuda()
    criterion = nn.CrossEntropyLoss()
    
    rollbacks = 0
    losses = []
    for s in range(1, 151):
        # Every 10 steps, silently zero out 10% of a layer's weights
        if s % 10 == 0 and s <= 100:
            layer_idx = (s // 10) % len(list(model.parameters()))
            with torch.no_grad():
                p = list(model.parameters())[layer_idx]
                mask = torch.rand_like(p) > 0.6  # zero out 60%
                p.data *= mask.float()
        
        opt.zero_grad()
        try:
            loss = criterion(model(bx), by)
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
    
    alive = not any(torch.isnan(p).any() for p in model.parameters())
    final_loss = losses[-1] if losses else float('inf')
    return {"config": config_name, "scenario": "silent_weight_decay", "seed": seed,
            "rollbacks": rollbacks, "final_loss": round(final_loss, 4),
            "alive": alive, "steps_completed": len(losses)}

def run_scenario_C(config_name, config, seed):
    '''Persistent NaN: inject NaN every 5 steps for 100 steps.
    Tests if perturbation/LR reduction helps escape the loop.'''
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    model = UnstableNet().cuda()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    ctrl = SelfHealingArc(model, opt, config)
    bx = torch.randn(32, 100).cuda()
    by = torch.randint(0, 10, (32,)).cuda()
    criterion = nn.CrossEntropyLoss()
    
    rollbacks, valid_steps = 0, 0
    for s in range(1, 101):
        opt.zero_grad()
        if s % 5 == 0:  # NaN every 5 steps
            a = ctrl.step(torch.tensor(float('nan')).cuda())
            rollbacks += (1 if a.rolled_back else 0)
            continue
        loss = criterion(model(bx), by)
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
        valid_steps += 1
    
    alive = not any(torch.isnan(p).any() for p in model.parameters())
    final_lr = opt.param_groups[0]['lr']
    return {"config": config_name, "scenario": "persistent_nan", "seed": seed,
            "rollbacks": rollbacks, "valid_steps": valid_steps,
            "alive": alive, "final_lr": final_lr}

# Run all
seeds = [42, 123, 777, 999, 2024]
results = {"experiment": "ablation_v2_harder", "runs": []}

for cname, cfg in configs.items():
    for scenario_fn, sname in [(run_scenario_A, "A:instability"), 
                                 (run_scenario_B, "B:silent_decay"),
                                 (run_scenario_C, "C:persistent_nan")]:
        for seed in seeds:
            r = scenario_fn(cname, cfg, seed)
            results["runs"].append(r)
    
    # Per-config summary
    runs = [r for r in results["runs"] if r["config"] == cname]
    alive_count = sum(1 for r in runs if r["alive"])
    avg_rb = sum(r["rollbacks"] for r in runs) / len(runs)
    print(f"  {cname:20s} | alive={alive_count}/{len(runs)} | avg_rollbacks={avg_rb:.1f}")

# Detailed comparison table
print("\\n" + "="*80)
print("ABLATION v2: Component Contribution Analysis")
print("="*80)
scenarios = ["recurring_instability", "silent_weight_decay", "persistent_nan"]
snames = ["Instability", "Silent Decay", "Persistent NaN"]
print(f"{'Config':20s}", end="")
for sn in snames:
    print(f" | {sn:>15s}", end="")
print(" | Total Rollbacks")
print("-"*80)
for cname in configs:
    print(f"  {cname:20s}", end="")
    total_rb = 0
    for sc in scenarios:
        runs = [r for r in results["runs"] if r["config"] == cname and r["scenario"] == sc]
        alive = sum(1 for r in runs if r["alive"])
        rb = sum(r["rollbacks"] for r in runs)
        total_rb += rb
        print(f" | {alive}/{len(runs)} alive,{rb:>3}rb", end="")
    print(f" | {total_rb:>3}")

with open("exp3v2_ablation_harder_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\\nSaved: exp3v2_ablation_harder_results.json")
"""


# ============================================================
# FIXED EXPERIMENT 6b: Gradient Explosion Stats (~3 min)
# ============================================================
# The old exp used SGD lr=3.0 which didn't cause actual explosions.
# This one uses lr=10.0 and a deeper network guaranteed to explode.
# ============================================================
"""
import torch, torch.nn as nn, torch.optim as optim, json
import numpy as np
from arc.core.self_healing import SelfHealingArc, SelfHealingConfig

class VeryDeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        dims = [100, 500, 500, 500, 500, 300, 10]
        for i in range(len(dims)-1):
            layers.extend([nn.Linear(dims[i], dims[i+1]), nn.ReLU()])
        layers.pop()
        self.net = nn.Sequential(*layers)
        # Initialize with larger weights to make explosion more likely
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=2.0)
    def forward(self, x): return self.net(x)

SEEDS = [42, 123, 456, 777, 999]
results_arc, results_baseline = [], []

print("=== Gradient Explosion: ARC vs Baseline (5 seeds) ===")
for seed in SEEDS:
    # --- With ARC ---
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    model = VeryDeepNet().cuda()
    opt = optim.SGD(model.parameters(), lr=10.0)  # guaranteed to explode
    ctrl = SelfHealingArc(model, opt, SelfHealingConfig(
        verbose=False, checkpoint_frequency=3, check_frequency=1,
        loss_explosion_threshold=50.0,
    ))
    bx, by = torch.randn(32, 100).cuda(), torch.randint(0, 10, (32,)).cuda()
    criterion = nn.CrossEntropyLoss()
    
    rollbacks, arc_losses = 0, []
    for s in range(1, 101):
        opt.zero_grad()
        try:
            loss = criterion(model(bx), by)
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
        arc_losses.append(loss.item())
    
    arc_alive = not any(torch.isnan(p).any() for p in model.parameters())
    arc_final = arc_losses[-1] if arc_losses else float('inf')
    results_arc.append({"seed": seed, "rollbacks": rollbacks, "alive": arc_alive, 
                         "final_loss": round(arc_final, 4), "steps": len(arc_losses)})
    
    # --- Without ARC (baseline) ---
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    model2 = VeryDeepNet().cuda()
    opt2 = optim.SGD(model2.parameters(), lr=10.0)
    
    baseline_losses, crashed = [], False
    for s in range(1, 101):
        opt2.zero_grad()
        try:
            loss = criterion(model2(bx), by)
        except:
            crashed = True
            break
        if torch.isnan(loss) or torch.isinf(loss):
            crashed = True
            break
        loss.backward()
        # Check gradients
        grad_norm = sum(p.grad.norm().item()**2 for p in model2.parameters() if p.grad is not None)**0.5
        if grad_norm > 1e6 or torch.isnan(torch.tensor(grad_norm)):
            crashed = True
            break
        opt2.step()
        baseline_losses.append(loss.item())
    
    baseline_alive = not crashed and not any(torch.isnan(p).any() for p in model2.parameters())
    results_baseline.append({"seed": seed, "crashed": crashed, "alive": baseline_alive,
                              "steps_before_crash": len(baseline_losses)})
    
    print(f"  Seed {seed}: ARC={rollbacks}rb,alive={arc_alive},steps={len(arc_losses)} | Baseline=crashed={crashed},steps={len(baseline_losses)}")

# Summary
arc_rollbacks = [r["rollbacks"] for r in results_arc]
arc_alive_pct = sum(1 for r in results_arc if r["alive"]) / len(results_arc) * 100
baseline_crash_pct = sum(1 for r in results_baseline if r["crashed"]) / len(results_baseline) * 100

print(f"\\n{'='*60}")
print(f"Gradient Explosion: ARC vs Baseline")
print(f"{'='*60}")
print(f"  ARC rollbacks:     {np.mean(arc_rollbacks):.1f} +/- {np.std(arc_rollbacks):.1f}")
print(f"  ARC survival:      {arc_alive_pct:.0f}%")
print(f"  Baseline crash:    {baseline_crash_pct:.0f}%")
print(f"  Baseline avg steps before crash: {np.mean([r['steps_before_crash'] for r in results_baseline]):.0f}")

with open("exp6b_grad_explosion_results.json", "w") as f:
    json.dump({"arc": results_arc, "baseline": results_baseline}, f, indent=2)
print("\\nSaved: exp6b_grad_explosion_results.json")
"""
