"""Organic failure harness: failures caused by real training conditions,
never by assigning nan/inf to the loss directly."""
import io, sys, math, contextlib
import torch, torch.nn as nn, torch.nn.functional as F
from arc.core.controller import Controller, ControllerConfig

torch.manual_seed(0)
STEPS = 60


def mlp(d_in=16, h=64, d_out=1, depth=3, init_gain=None):
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers += [nn.Linear(d, d_out)]
    m = nn.Sequential(*layers)
    if init_gain is not None:
        for p in m.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=init_gain)
    return m


# ---- scenarios: return (model, optimizer, step_fn(model, step) -> loss) ----

def sc_lr_too_high():
    """Classic: learning rate orders of magnitude too high. Loss diverges."""
    m = mlp()
    opt = torch.optim.SGD(m.parameters(), lr=3.0)
    x, y = torch.randn(64, 16), torch.randn(64, 1)
    return m, opt, lambda m, s: F.mse_loss(m(x), y)


def sc_nan_in_data():
    """Corrupt sensor row enters the dataset at step 20."""
    m = mlp()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    x, y = torch.randn(64, 16), torch.randn(64, 1)
    bad = x.clone(); bad[3, 7] = float('nan')
    def step(m, s):
        xin = bad if s >= 20 else x
        return F.mse_loss(m(xin), y)
    return m, opt, step


def sc_log_zero():
    """Custom loss takes log of a probability that underflows to exactly 0."""
    m = mlp(d_out=4)
    opt = torch.optim.SGD(m.parameters(), lr=0.5)
    x = torch.randn(64, 16); y = torch.randint(0, 4, (64,))
    def step(m, s):
        logits = m(x)
        if s >= 20:
            logits = logits * 200.0          # saturates softmax to exact 0/1
        p = torch.softmax(logits, dim=-1)
        return F.nll_loss(torch.log(p), y)   # log(0) -> -inf, no eps guard
    return m, opt, step


def sc_fp16_overflow():
    """Half precision without a grad scaler: activations overflow to inf."""
    m = mlp(h=256).half()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    x = (torch.randn(64, 16) * 30).half(); y = torch.randn(64, 1).half()
    def step(m, s):
        xin = x * (60.0 if s >= 20 else 1.0)
        return F.mse_loss(m(xin), y).float()
    return m, opt, step


def sc_dead_relu():
    """Silent failure: large negative bias kills every ReLU. Gradients vanish,
    loss flatlines at a finite value. Nothing is nan."""
    m = mlp(depth=4)
    opt = torch.optim.SGD(m.parameters(), lr=0.05)
    x, y = torch.randn(64, 16), torch.randn(64, 1)
    def step(m, s):
        if s == 20:
            with torch.no_grad():
                for mod in m:
                    if isinstance(mod, nn.Linear):
                        mod.bias.fill_(-50.0)   # a bad init/LR would do this too
        return F.mse_loss(m(x), y)
    return m, opt, step


def sc_rnn_gradient_explosion():
    """Long unrolled RNN, spectral radius > 1, no clipping."""
    class R(nn.Module):
        def __init__(self):
            super().__init__()
            self.cell = nn.RNNCell(8, 32)
            self.out = nn.Linear(32, 1)
            with torch.no_grad():
                self.cell.weight_hh.mul_(3.0)
        def forward(self, x):
            h = torch.zeros(x.size(0), 32)
            for t in range(x.size(1)):
                h = self.cell(x[:, t], h)
            return self.out(h)
    m = R()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    x, y = torch.randn(32, 60, 8), torch.randn(32, 1)
    return m, opt, lambda m, s: F.mse_loss(m(x), y)


def sc_zero_variance_batch():
    """Degenerate batch: every row identical, so a std-normalization divides
    by zero. Happens for real with a bad sampler or a sorted dataset."""
    m = mlp()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    x, y = torch.randn(64, 16), torch.randn(64, 1)
    dup = x[0:1].repeat(64, 1)
    def step(m, s):
        xin = dup if s >= 20 else x
        xin = (xin - xin.mean(0)) / xin.std(0)   # std == 0 -> nan
        return F.mse_loss(m(xin), y)
    return m, opt, step


def sc_loss_scale_spike():
    """A single pathological sample with a huge target dominates the batch."""
    m = mlp()
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    x, y = torch.randn(64, 16), torch.randn(64, 1)
    ybad = y.clone(); ybad[0] = 1e12
    def step(m, s):
        return F.mse_loss(m(x), ybad if s >= 20 else y)
    return m, opt, step


SCENARIOS = {
    'lr_too_high': sc_lr_too_high,
    'nan_in_data': sc_nan_in_data,
    'log_zero': sc_log_zero,
    'fp16_overflow': sc_fp16_overflow,
    'dead_relu_silent': sc_dead_relu,
    'rnn_grad_explosion': sc_rnn_gradient_explosion,
    'zero_variance_batch': sc_zero_variance_batch,
    'loss_spike_outlier': sc_loss_scale_spike,
}


def params_finite(m):
    return all(torch.isfinite(p).all().item() for p in m.parameters())


def run(name, use_arc):
    torch.manual_seed(0)
    m, opt, stepfn = SCENARIOS[name]()
    ctrl = None
    if use_arc:
        cfg = ControllerConfig(checkpoint_frequency=5, verbose=False,
                               auto_calibrate=True, calibration_steps=10)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ctrl = Controller(m, opt, config=cfg)
    detected = recovered = False
    losses = []
    for s in range(STEPS):
        opt.zero_grad()
        try:
            loss = stepfn(m, s)
        except Exception as e:
            return dict(name=name, arc=use_arc, outcome=f'raised {type(e).__name__}',
                        detected=detected, finite=params_finite(m), losses=losses)
        lv = float(loss.detach())
        losses.append(lv)
        if ctrl is not None:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                a = ctrl.step(loss)
            if a.failure_type:
                detected = True
            if a.rolled_back or a.lr_reduced or a.gradients_clipped:
                recovered = True
            if a.should_skip or a.rolled_back:
                continue
        if torch.isfinite(loss):
            loss.backward()
            opt.step()
    finite = params_finite(m)
    tail = [l for l in losses[-10:] if math.isfinite(l)]
    healthy_end = finite and len(tail) >= 5
    return dict(name=name, arc=use_arc,
                outcome='healthy' if healthy_end else 'broken',
                detected=detected, recovered=recovered, finite=finite,
                losses=losses)


if __name__ == '__main__':
    only = sys.argv[1] if len(sys.argv) > 1 else None
    rows = []
    for n in SCENARIOS:
        if only and only != n:
            continue
        base = run(n, False)
        arc = run(n, True)
        rows.append((n, base, arc))
    print(f"{'scenario':<22} {'no-arc':<22} {'arc':<12} {'detected':<9} {'recovered'}")
    print('-' * 80)
    for n, b, a in rows:
        print(f"{n:<22} {b['outcome']:<22} {a['outcome']:<12} "
              f"{str(a.get('detected')):<9} {a.get('recovered')}")
