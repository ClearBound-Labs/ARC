import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arc.core.self_healing import SelfHealingArc, SelfHealingConfig

class TestSelfHealingArc:
    """Tests for SelfHealingArc."""
    
    def test_baseline_clipping_flag(self):
        """Test that action.gradients_clipped is set correctly when baseline clipping fires."""
        model = nn.Sequential(
            nn.Linear(10, 10),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Configure to ensure baseline clipping fires
        config = SelfHealingConfig(
            enable_baseline_clipping=True,
            gradient_clip_norm=1.0,
            gradient_explosion_threshold=1e9,  # Prevent explosion branch
            enable_forecasting=False           # Prevent forecasting branch
        )
        
        arc = SelfHealingArc(model, optimizer, config=config)
        
        # Forward pass to create gradients
        x = torch.randn(2, 10)
        out = model(x)
        loss = out.mean()
        loss.backward()
        
        # Call post_backward which should trigger baseline clipping
        action = arc.post_backward()
        
        # Verify the flag was set
        assert action.gradients_clipped is True, "Baseline clipping should set gradients_clipped to True"

    def test_baseline_clipping_opt_out(self):
        """Test that baseline clipping can be opted out using the config flag."""
        model = nn.Sequential(
            nn.Linear(10, 10),
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Configure to disable baseline clipping
        config = SelfHealingConfig(
            enable_baseline_clipping=False,
            gradient_clip_norm=1.0,
            gradient_explosion_threshold=1e9,
            enable_forecasting=False
        )
        
        arc = SelfHealingArc(model, optimizer, config=config)
        
        # Forward pass to create gradients
        x = torch.randn(2, 10)
        out = model(x)
        loss = out.mean()
        loss.backward()
        
        # Call post_backward
        action = arc.post_backward()
        
        # Verify the flag was NOT set
        assert action.gradients_clipped is False, "Baseline clipping should not run when disabled"

    def test_no_clipping_unless_asked(self):
        model = nn.Linear(10, 10)
        arc = SelfHealingArc(model, torch.optim.SGD(model.parameters(), lr=0.01),
                             SelfHealingConfig(enable_forecasting=False, verbose=False))
        (model(torch.randn(2, 10)).sum() * 100).backward()
        before = [p.grad.clone() for p in model.parameters()]

        assert arc.post_backward().gradients_clipped is False
        assert all(torch.equal(b, p.grad) for b, p in zip(before, model.parameters()))

    def test_repeated_rollbacks_reduce_lr_exactly_once_each(self):
        torch.manual_seed(0)
        model = nn.Linear(10, 1)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        arc = SelfHealingArc(model, opt, SelfHealingConfig(
            checkpoint_frequency=2, adaptive_checkpoint_frequency=False,
            lr_reduction_factor=0.1, verbose=False))

        def healthy(n):
            for _ in range(n):
                loss = model(torch.randn(4, 10)).pow(2).mean()
                opt.zero_grad()
                loss.backward()
                if not arc.step(loss).should_skip:
                    opt.step()

        healthy(4)
        arc.step(torch.tensor(float("nan")))
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)
        healthy(6)  # a fresh checkpoint is taken at the reduced LR
        arc.step(torch.tensor(float("nan")))
        assert opt.param_groups[0]["lr"] == pytest.approx(1e-4)

    def test_is_deprecated_in_favour_of_controller(self):
        model = nn.Linear(2, 2)
        with pytest.warns(FutureWarning, match="arc.Controller"):
            SelfHealingArc(model, torch.optim.SGD(model.parameters(), lr=0.1),
                           SelfHealingConfig(verbose=False))
