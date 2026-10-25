"""
ARC Comprehensive Test Suite — Kaggle T4 GPU
=============================================
Run each cell in order on Kaggle with GPU T4.
Estimated total time: ~15-20 minutes.

Tests:
  1. Setup + Sanity
  2. Long training run (500 steps, real dataset)
  3. Chained failures (NaN + gradient explosion + weight corruption)
  4. GradScaler interaction (fp16 mixed precision)
  5. Large model (GPT-2 Medium 355M)
  6. Edge case: failure on step 1
  7. High-level Arc API test
  8. CPU training recovery
  9. ArcCallback (Lightning-style) test
 10. Final stress test: aggressive LR organic failures
"""

# ============================================================
# CELL 1: Setup + Sanity Check (~30 seconds)
# ============================================================
"""
!pip install arc-training transformers torchvision --no-cache-dir -q

import arc
import torch
print(f"ARC version: {arc.__version__}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Quick sanity: can we import everything?
from arc.core.self_healing import SelfHealingArc, SelfHealingConfig
from arc import Arc, ArcCallback
print("All imports OK")
"""


# ============================================================
# CELL 2: Long Training — 500 steps, Real Data (~3 min)
# ============================================================
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from arc.core.self_healing import SelfHealingArc, SelfHealingConfig

# Real dataset: CIFAR-10
transform = T.Compose([T.ToTensor(), T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

# Real model: ResNet18
model = torchvision.models.resnet18(num_classes=10).cuda()
opt = optim.AdamW(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

ctrl = SelfHealingArc(model, opt, SelfHealingConfig(
    verbose=True,
    lite_mode=True,
    check_frequency=5,
    checkpoint_frequency=10,
))

losses = []
data_iter = iter(loader)

for step in range(1, 501):
    try:
        images, labels = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        images, labels = next(data_iter)
    
    images, labels = images.cuda(), labels.cuda()
    opt.zero_grad()
    
    out = model(images)
    loss = criterion(out, labels)
    
    action = ctrl.step(loss)
    if not action.rolled_back and not action.should_skip:
        loss.backward()
        pa = ctrl.post_backward()
        if not pa.rolled_back and not pa.should_skip:
            opt.step()
    
    lv = loss.item()
    losses.append(lv)
    if step % 100 == 0:
        avg = sum(losses[-50:]) / 50
        print(f"  Step {step}: loss={lv:.4f} avg50={avg:.4f} rollbacks={ctrl.total_rollbacks}")

print(f"\\n--- RESULT ---")
print(f"Steps: 500 | Final loss: {losses[-1]:.4f}")
print(f"First 50 avg: {sum(losses[:50])/50:.4f}")
print(f"Last 50 avg:  {sum(losses[-50:])/50:.4f}")
print(f"Rollbacks: {ctrl.total_rollbacks}")
learned = sum(losses[-50:])/50 < sum(losses[:50])/50
print(f"Learning: {'YES' if learned else 'NO'}")
print(f"VERDICT: {'PASS' if learned and ctrl.total_rollbacks == 0 else 'CHECK'}")
"""


# ============================================================
# CELL 3: Chained Failures — NaN + GradExplosion + WeightCorrupt (~2 min)
# ============================================================
"""
import torch, torch.nn as nn, torch.optim as optim
from arc.core.self_healing import SelfHealingArc, SelfHealingConfig

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64*8*8, 256), nn.ReLU(), nn.Linear(256, 10)
        )
    def forward(self, x): return self.net(x)

model = SimpleCNN().cuda()
opt = optim.Adam(model.parameters(), lr=1e-3)
ctrl = SelfHealingArc(model, opt, SelfHealingConfig(
    verbose=True, checkpoint_frequency=5
))

batch_x = torch.randn(16, 3, 32, 32).cuda()
batch_y = torch.randint(0, 10, (16,)).cuda()
criterion = nn.CrossEntropyLoss()

results = {}

# Phase 1: Normal training (20 steps)
print("=== Phase 1: Normal training ===")
for s in range(1, 21):
    opt.zero_grad()
    loss = criterion(model(batch_x), batch_y)
    a = ctrl.step(loss)
    if not a.rolled_back and not a.should_skip:
        loss.backward(); pa = ctrl.post_backward()
        if not pa.rolled_back and not pa.should_skip: opt.step()
loss_before = loss.item()
print(f"  Loss after 20 steps: {loss_before:.4f}")

# Phase 2: Inject NaN
print("\\n=== Phase 2: NaN injection ===")
a = ctrl.step(torch.tensor(float('nan')).cuda())
results['nan_caught'] = a.rolled_back
print(f"  NaN caught and rolled back: {a.rolled_back}")

# Phase 3: Train 10 more, then inject gradient explosion
print("\\n=== Phase 3: Normal + Gradient Explosion ===")
for s in range(1, 11):
    opt.zero_grad()
    loss = criterion(model(batch_x), batch_y)
    a = ctrl.step(loss)
    if not a.rolled_back and not a.should_skip:
        loss.backward(); pa = ctrl.post_backward()
        if not pa.rolled_back and not pa.should_skip: opt.step()

# Inject gradient explosion by scaling gradients
opt.zero_grad()
loss = criterion(model(batch_x), batch_y)
a = ctrl.step(loss)
if not a.rolled_back:
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data *= 1e8
    pa = ctrl.post_backward()
    # ARC handles grad explosion by CLIPPING (not rollback) -- that's correct behavior
    # Check: gradients were clipped to safe values, OR rollback/skip happened
    max_grad = max(p.grad.data.norm().item() for p in model.parameters() if p.grad is not None)
    results['grad_explosion_handled'] = pa.rolled_back or pa.should_skip or max_grad < 1000
    print(f"  Grad explosion handled: {results['grad_explosion_handled']} (grad norm after: {max_grad:.1f})")
    if not pa.rolled_back and not pa.should_skip:
        opt.step()

# Phase 4: Train 10 more, then inject weight corruption
print("\\n=== Phase 4: Normal + Weight Corruption ===")
for s in range(1, 11):
    opt.zero_grad()
    loss = criterion(model(batch_x), batch_y)
    a = ctrl.step(loss)
    if not a.rolled_back and not a.should_skip:
        loss.backward(); pa = ctrl.post_backward()
        if not pa.rolled_back and not pa.should_skip: opt.step()

# Corrupt some weights
rollbacks_before = ctrl.total_rollbacks
with torch.no_grad():
    for p in list(model.parameters())[:3]:
        p.data.fill_(float('nan'))

# ARC catches weight corruption indirectly: NaN weights -> NaN loss -> rollback
# Give it a few steps to detect
for recovery_step in range(5):
    opt.zero_grad()
    try:
        loss = criterion(model(batch_x), batch_y)
    except:
        loss = torch.tensor(float('nan')).cuda()
    a = ctrl.step(loss)
    if a.rolled_back:
        break
    if not a.should_skip:
        try:
            loss.backward()
            pa = ctrl.post_backward()
            if not pa.rolled_back and not pa.should_skip: opt.step()
        except:
            pass

results['weight_corruption_recovered'] = ctrl.total_rollbacks > rollbacks_before
model_clean = not any(torch.isnan(p).any() for p in model.parameters())
print(f"  Weight corruption detected and recovered: {results['weight_corruption_recovered']}")
print(f"  Weights clean after recovery: {model_clean}")

# Phase 5: Can it still learn after all 3 failures?
print("\\n=== Phase 5: Post-recovery learning ===")
losses_after = []
for s in range(1, 31):
    opt.zero_grad()
    loss = criterion(model(batch_x), batch_y)
    a = ctrl.step(loss)
    if not a.rolled_back and not a.should_skip:
        loss.backward(); pa = ctrl.post_backward()
        if not pa.rolled_back and not pa.should_skip: opt.step()
    losses_after.append(loss.item())

results['still_learning'] = losses_after[-1] < losses_after[0]
print(f"  Loss: {losses_after[0]:.4f} -> {losses_after[-1]:.4f}")
print(f"  Still learning: {results['still_learning']}")

print(f"\\n{'='*50}")
print(f"CHAINED FAILURES SUMMARY:")
for k, v in results.items():
    status = "PASS" if v else "FAIL"
    print(f"  {k}: {status}")
all_pass = all(results.values())
print(f"Total rollbacks: {ctrl.total_rollbacks}")
print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
"""


# ============================================================
# CELL 4: GradScaler + ARC interaction (~2 min)
# ============================================================
"""
import torch, torch.nn as nn, torch.optim as optim
from torch.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM
from arc.core.self_healing import SelfHealingArc, SelfHealingConfig

# Load GPT-2, trainable params in fp32
model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16).cuda()
for p in model.parameters():
    p.data = p.data.float()
    p.requires_grad_(True)

opt = optim.AdamW(model.parameters(), lr=5e-5)
scaler = GradScaler()
ctrl = SelfHealingArc(model, opt, SelfHealingConfig(
    verbose=True, lite_mode=True, check_frequency=5,
    enable_weight_health_check=False,
))

batch = torch.randint(0, 50257, (2, 128)).cuda()
losses = []

print("=== GradScaler + ARC: 30 steps ===")
for s in range(1, 31):
    opt.zero_grad()
    
    with autocast(device_type='cuda', dtype=torch.float16):
        out = model(input_ids=batch, labels=batch)
        loss = out.loss
    
    action = ctrl.step(loss)
    if action.rolled_back or action.should_skip:
        # No backward was done, don't touch scaler
        continue
    
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    pa = ctrl.post_backward()
    if pa.rolled_back or pa.should_skip:
        # unscale was called, so update() is safe
        scaler.update()
        continue
    
    scaler.step(opt)
    scaler.update()
    
    lv = loss.item()
    losses.append(lv)
    if s % 10 == 0:
        print(f"  Step {s}: loss={lv:.4f} rollbacks={ctrl.total_rollbacks}")

# Inject NaN to test recovery
print("\\n=== Injecting NaN ===")
a = ctrl.step(torch.tensor(float('nan')).cuda())
print(f"  Rolled back: {a.rolled_back}")
# No backward was done here, don't call scaler.update()

# Continue training
print("\\n=== Post-recovery: 20 more steps ===")
for s in range(31, 51):
    opt.zero_grad()
    with autocast(device_type='cuda', dtype=torch.float16):
        out = model(input_ids=batch, labels=batch)
        loss = out.loss
    action = ctrl.step(loss)
    if action.rolled_back or action.should_skip:
        continue
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    pa = ctrl.post_backward()
    if pa.rolled_back or pa.should_skip:
        scaler.update()
        continue
    scaler.step(opt)
    scaler.update()
    losses.append(loss.item())

print(f"\\n--- RESULT ---")
print(f"GradScaler + ARC: {'WORKS' if losses and not any(torch.isnan(torch.tensor(l)) for l in losses) else 'CHECK'}")
if losses:
    print(f"Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    learned = losses[-1] < losses[0]
    print(f"Learning: {'YES' if learned else 'NO'}")
print(f"Rollbacks: {ctrl.total_rollbacks}")
nan_recovered = a.rolled_back
print(f"NaN recovery: {'PASS' if nan_recovered else 'FAIL'}")
print(f"VERDICT: {'PASS' if nan_recovered else 'CHECK'}")
"""


# ============================================================
# CELL 5: Large Model — GPT-2 Medium 355M (~3 min)
# ============================================================
"""
import torch, torch.optim as optim
from torch.amp import autocast
from transformers import AutoModelForCausalLM
from arc.core.self_healing import SelfHealingArc, SelfHealingConfig

model = AutoModelForCausalLM.from_pretrained("gpt2-medium", torch_dtype=torch.float16).cuda()
# Only train last 2 layers to fit in T4 memory
for p in model.parameters():
    p.requires_grad_(False)
for p in model.transformer.h[-2:].parameters():
    p.data = p.data.float()
    p.requires_grad_(True)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"GPT-2 Medium: {sum(p.numel() for p in model.parameters())/1e6:.0f}M total, {trainable/1e6:.1f}M trainable")

opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)
ctrl = SelfHealingArc(model, opt, SelfHealingConfig(
    verbose=True, lite_mode=True, check_frequency=5,
    enable_weight_health_check=False,
))

batch = torch.randint(0, 50257, (2, 128)).cuda()
losses = []

print("\\n=== Training 30 steps ===")
for s in range(1, 31):
    opt.zero_grad()
    with autocast(device_type='cuda', dtype=torch.float16):
        loss = model(input_ids=batch, labels=batch).loss
    action = ctrl.step(loss)
    if not action.rolled_back and not action.should_skip:
        loss.backward()
        pa = ctrl.post_backward()
        if not pa.rolled_back and not pa.should_skip:
            opt.step()
    losses.append(loss.item())
    if s % 10 == 0:
        print(f"  Step {s}: loss={losses[-1]:.4f} rollbacks={ctrl.total_rollbacks}")

# NaN injection
print("\\n=== NaN injection ===")
a = ctrl.step(torch.tensor(float('nan')).cuda())
print(f"  Rolled back: {a.rolled_back}")

# Recovery
print("\\n=== Recovery: 20 steps ===")
for s in range(31, 51):
    opt.zero_grad()
    with autocast(device_type='cuda', dtype=torch.float16):
        loss = model(input_ids=batch, labels=batch).loss
    action = ctrl.step(loss)
    if not action.rolled_back and not action.should_skip:
        loss.backward()
        pa = ctrl.post_backward()
        if not pa.rolled_back and not pa.should_skip:
            opt.step()
    losses.append(loss.item())

print(f"\\n--- RESULT ---")
print(f"GPT-2 Medium (355M) with ARC")
print(f"Loss: {losses[0]:.4f} -> {losses[29]:.4f} (before NaN)")
print(f"Recovery loss: {losses[30]:.4f} -> {losses[-1]:.4f}")
print(f"Rollbacks: {ctrl.total_rollbacks}")
print(f"NaN recovery: {'PASS' if a.rolled_back else 'FAIL'}")
learned = losses[29] < losses[0]
print(f"Learning: {'YES' if learned else 'NO'}")
print(f"VERDICT: {'PASS' if learned and a.rolled_back else 'CHECK'}")

del model; torch.cuda.empty_cache()
"""


# ============================================================
# CELL 6: Edge Case — Failure on Step 1 (~30 sec)
# ============================================================
"""
import torch, torch.nn as nn, torch.optim as optim
from arc.core.self_healing import SelfHealingArc, SelfHealingConfig

model = nn.Sequential(nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 2)).cuda()
opt = optim.Adam(model.parameters(), lr=1e-3)
ctrl = SelfHealingArc(model, opt, SelfHealingConfig(verbose=True))

print("=== Edge Case: NaN on Step 1 (no checkpoint exists) ===")
# First step is NaN -- no checkpoint to roll back to
a = ctrl.step(torch.tensor(float('nan')).cuda())
print(f"  Step 1 NaN: rolled_back={a.rolled_back}, should_skip={a.should_skip}")

# Can we still train after?
crashed = False
try:
    batch_x = torch.randn(8, 10).cuda()
    batch_y = torch.randint(0, 2, (8,)).cuda()
    for s in range(2, 22):
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(batch_x), batch_y)
        a = ctrl.step(loss)
        if not a.rolled_back and not a.should_skip:
            loss.backward()
            pa = ctrl.post_backward()
            if not pa.rolled_back and not pa.should_skip:
                opt.step()
    final_loss = loss.item()
    print(f"  Trained 20 more steps. Final loss: {final_loss:.4f}")
except Exception as e:
    crashed = True
    print(f"  CRASHED: {e}")

print(f"\\nVERDICT: {'PASS' if not crashed else 'FAIL'} (model survived step-1 failure)")
"""


# ============================================================
# CELL 7: High-Level Arc API Test (~1 min)
# ============================================================
"""
import torch, torch.nn as nn, torch.optim as optim
from arc import Arc

model = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 10)
).cuda()
opt = optim.Adam(model.parameters(), lr=1e-3)

# Test high-level API
controller = Arc(verbose=True)
controller.attach(model, opt)

criterion = nn.CrossEntropyLoss()
batch_x = torch.randn(32, 784).cuda()
batch_y = torch.randint(0, 10, (32,)).cuda()

print("=== High-Level Arc API: 50 steps ===")
for step in range(1, 51):
    opt.zero_grad()
    loss = criterion(model(batch_x), batch_y)
    loss.backward()
    opt.step()
    
    # Arc monitors on batch end
    controller.on_batch_end(loss.item(), step=step)

# Epoch-level check
prediction = controller.on_epoch_end(epoch=1, train_loss=loss.item())
print(f"  Overall risk: {prediction.overall_risk:.4f}")
print(f"  Risk level: {prediction.risk_level}")
print(f"  Recommendation: {prediction.recommendation.action}")
print(f"  Rationale: {prediction.recommendation.rationale}")

controller.detach()
print(f"VERDICT: PASS (Arc API works without crash)")
"""


# ============================================================
# CELL 8: CPU Training Recovery (~1 min)
# ============================================================
"""
import torch, torch.nn as nn, torch.optim as optim
from arc.core.self_healing import SelfHealingArc, SelfHealingConfig

# Force CPU
model = nn.Sequential(nn.Linear(20, 100), nn.ReLU(), nn.Linear(100, 5))
opt = optim.Adam(model.parameters(), lr=1e-3)
ctrl = SelfHealingArc(model, opt, SelfHealingConfig(verbose=True, checkpoint_frequency=5))

batch_x = torch.randn(16, 20)
batch_y = torch.randint(0, 5, (16,))
criterion = nn.CrossEntropyLoss()

print("=== CPU Training: 30 steps + NaN recovery ===")
losses = []
for s in range(1, 31):
    opt.zero_grad()
    loss = criterion(model(batch_x), batch_y)
    a = ctrl.step(loss)
    if not a.rolled_back and not a.should_skip:
        loss.backward()
        pa = ctrl.post_backward()
        if not pa.rolled_back and not pa.should_skip:
            opt.step()
    losses.append(loss.item())

loss_before_nan = losses[-1]
print(f"  Loss after 30 steps: {loss_before_nan:.4f}")

# NaN
nan_action = ctrl.step(torch.tensor(float('nan')))
nan_recovered = nan_action.rolled_back
print(f"  NaN recovery: rolled_back={nan_recovered}")

# Continue
for s in range(31, 51):
    opt.zero_grad()
    loss = criterion(model(batch_x), batch_y)
    a = ctrl.step(loss)
    if not a.rolled_back and not a.should_skip:
        loss.backward()
        pa = ctrl.post_backward()
        if not pa.rolled_back and not pa.should_skip:
            opt.step()
    losses.append(loss.item())

print(f"  Post-recovery loss: {losses[-1]:.4f}")
print(f"  Rollbacks: {ctrl.total_rollbacks}")
model_alive = not any(torch.isnan(p).any() for p in model.parameters())
print(f"  Model weights clean: {model_alive}")
print(f"VERDICT: {'PASS' if model_alive and nan_recovered else 'FAIL'}")
"""


# ============================================================
# CELL 9: ArcWrapper (simplified API) (~1 min)
# ============================================================
"""
import torch, torch.nn as nn, torch.optim as optim
from arc import ArcWrapper

model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(32*7*7, 128), nn.ReLU(), nn.Linear(128, 10)
).cuda()
opt = optim.Adam(model.parameters(), lr=1e-3)

# Wrap with ARC
wrapped = ArcWrapper(model, opt, verbose=True)

criterion = nn.CrossEntropyLoss()
batch_x = torch.randn(16, 1, 28, 28).cuda()
batch_y = torch.randint(0, 10, (16,)).cuda()

print("=== ArcWrapper API Test ===")
losses = []
for step in range(1, 31):
    opt.zero_grad()
    loss = criterion(model(batch_x), batch_y)
    loss.backward()
    opt.step()
    wrapped.on_batch_end(loss.item(), step=step)
    losses.append(loss.item())
    if step % 10 == 0:
        print(f"  Step {step}: loss={loss.item():.4f}")

# Epoch-level prediction
pred = wrapped.on_epoch_end(epoch=1)
print(f"  Overall risk: {pred.overall_risk:.4f}")
print(f"  Risk level: {pred.risk_level}")
print(f"  Learning: {'YES' if losses[-1] < losses[0] else 'NO'}")
print(f"VERDICT: PASS (ArcWrapper works)")
"""


# ============================================================
# CELL 10: Organic Failures — Aggressive LR (~2 min)
# ============================================================
"""
import torch, torch.nn as nn, torch.optim as optim
from arc.core.self_healing import SelfHealingArc, SelfHealingConfig

class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        dims = [784, 512, 512, 256, 256, 128, 10]
        for i in range(len(dims)-1):
            layers.extend([nn.Linear(dims[i], dims[i+1]), nn.ReLU()])
        layers.pop()  # remove last ReLU
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

model = DeepNet().cuda()
opt = optim.SGD(model.parameters(), lr=5.0)  # crazy high LR
ctrl = SelfHealingArc(model, opt, SelfHealingConfig(
    verbose=True, checkpoint_frequency=3,
    loss_explosion_threshold=50.0,
))

batch_x = torch.randn(32, 784).cuda()
batch_y = torch.randint(0, 10, (32,)).cuda()
criterion = nn.CrossEntropyLoss()

print("=== Organic Failure: SGD lr=5.0 on deep net ===")
losses = []
for s in range(1, 101):
    opt.zero_grad()
    loss = criterion(model(batch_x), batch_y)
    a = ctrl.step(loss)
    if not a.rolled_back and not a.should_skip:
        loss.backward()
        pa = ctrl.post_backward()
        if not pa.rolled_back and not pa.should_skip:
            opt.step()
    
    lv = loss.item() if not (torch.isnan(loss) or torch.isinf(loss)) else float('inf')
    losses.append(lv)
    if s % 20 == 0 or a.rolled_back:
        print(f"  Step {s}: loss={lv:.4f} rollbacks={ctrl.total_rollbacks}")

print(f"\\n--- RESULT ---")
print(f"Total rollbacks: {ctrl.total_rollbacks}")
model_alive = not any(torch.isnan(p).any() for p in model.parameters())
print(f"Model alive: {model_alive}")
valid_losses = [l for l in losses if l != float('inf')]
if valid_losses:
    print(f"Loss range: {min(valid_losses):.4f} - {max(valid_losses):.4f}")
    
if ctrl.total_rollbacks > 0:
    print(f"VERDICT: PASS (ARC caught {ctrl.total_rollbacks} organic failures, model alive={model_alive})")
else:
    print(f"VERDICT: CHECK (no organic failures triggered, try higher LR)")
"""


# ============================================================
# FINAL SUMMARY CELL
# ============================================================
"""
print("=" * 60)
print("ARC COMPREHENSIVE TEST RESULTS")
print("=" * 60)
print()
print("Test  1: Setup + Import           — check above")
print("Test  2: Long Run (500 steps)     — check above")
print("Test  3: Chained Failures         — check above")
print("Test  4: GradScaler + ARC         — check above")
print("Test  5: GPT-2 Medium (355M)      — check above")
print("Test  6: Step-1 Edge Case         — check above")
print("Test  7: High-Level Arc API       — check above")
print("Test  8: CPU Recovery             — check above")
print("Test  9: ArcWrapper               — check above")
print("Test 10: Organic Failures         — check above")
print()
print("If all say PASS — ARC is production-ready.")
print("If any say FAIL — fix that specific issue.")
"""
