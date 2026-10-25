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

"""
ARC Comprehensive Test Suite
============================
Tests all core ARC functionality with detailed reporting.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.insert(0, '.')

from arc.intervention import WeightRollback, RollbackConfig

# Simple model (underscore prefix prevents pytest from collecting it as a test class)
class _ComprehensiveTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )
    def forward(self, x):
        return self.net(x)


def test_rng_preservation():
    """Test 1: RNG State Preservation"""
    print('\n[1] RNG STATE PRESERVATION TEST')
    model = _ComprehensiveTestModel()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    config = RollbackConfig(checkpoint_frequency=10)
    rb = WeightRollback(model, opt, config, verbose=False)

    # Check RNG is saved
    has_rng = 'rng_state' in rb.state.checkpoints[0]
    print(f'  RNG state in checkpoint: {has_rng}')
    print(f'  Torch RNG saved: {rb.state.checkpoints[0]["rng_state"]["torch"] is not None}')
    assert has_rng



def test_failure_detection():
    """Test 2: Failure Detection"""
    print('\n[2] FAILURE DETECTION TEST')
    test_cases = [
        ('NaN loss', float('nan')),
        ('Inf loss', float('inf')),
        ('Loss explosion', 1e8),
    ]

    results = []
    for name, loss_val in test_cases:
        model = _ComprehensiveTestModel()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        rb = WeightRollback(model, opt, RollbackConfig(), verbose=False)
        
        # Normal step
        x = torch.randn(4, 10)
        loss = model(x).mean()
        loss.backward()
        rb.step(loss)
        opt.step()
        opt.zero_grad()
        
        # Inject failure
        x = torch.randn(4, 10)
        loss = model(x).mean() * loss_val
        try:
            loss.backward()
        except:
            pass
        action = rb.step(loss)
        print(f'  {name}: Detected={action.rolled_back}')
        results.append(action.rolled_back)
    
    assert all(results)



def test_recovery_success():
    """Test 3: Recovery Success Rate"""
    print('\n[3] RECOVERY SUCCESS TEST (10 seeds)')
    success_count = 0
    for seed in range(10):
        torch.manual_seed(seed)
        model = _ComprehensiveTestModel()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        rb = WeightRollback(model, opt, RollbackConfig(checkpoint_frequency=5), verbose=False)
        
        recovered = False
        for step in range(50):
            x = torch.randn(4, 10)
            loss = model(x).mean()
            
            # Inject failure at step 25
            if step == 25:
                loss = loss * float('inf')
            
            try:
                loss.backward()
            except:
                pass
            
            action = rb.step(loss)
            if action.rolled_back:
                recovered = True
            
            if not action.rolled_back and not torch.isnan(loss) and not torch.isinf(loss):
                opt.step()
            opt.zero_grad()
        
        if recovered:
            success_count += 1

    print(f'  Recovery rate: {success_count}/10 ({success_count*10}%)')
    assert success_count == 10



def test_false_positives():
    """Test 4: False Positive Check"""
    print('\n[4] FALSE POSITIVE TEST (1000 stable steps)')
    model = _ComprehensiveTestModel()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    rb = WeightRollback(model, opt, RollbackConfig(), verbose=False)

    false_positives = 0
    for step in range(1000):
        x = torch.randn(4, 10)
        loss = model(x).mean()
        loss.backward()
        action = rb.step(loss)
        if action.rolled_back:
            false_positives += 1
        opt.step()
        opt.zero_grad()

    print(f'  False positives: {false_positives}/1000 ({false_positives/10}%)')
    assert false_positives == 0



def test_lr_reduction():
    """Test 5: LR Reduction After Rollback"""
    print('\n[5] LR REDUCTION AFTER ROLLBACK')
    model = _ComprehensiveTestModel()
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    rb = WeightRollback(model, opt, RollbackConfig(lr_reduction_factor=0.5), verbose=False)

    initial_lr = opt.param_groups[0]['lr']
    print(f'  Initial LR: {initial_lr}')

    # Trigger rollback
    x = torch.randn(4, 10)
    loss = model(x).mean() * float('inf')
    try:
        loss.backward()
    except:
        pass
    rb.step(loss)

    new_lr = opt.param_groups[0]['lr']
    print(f'  After rollback LR: {new_lr}')
    correct = abs(new_lr - initial_lr * 0.5) < 1e-6
    print(f'  Reduction correct: {correct}')
    assert correct



def test_optimizer_state():
    """Test 6: Optimizer State Preservation"""
    print('\n[6] OPTIMIZER STATE PRESERVATION')
    model = _ComprehensiveTestModel()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    rb = WeightRollback(model, opt, RollbackConfig(checkpoint_frequency=5), verbose=False)

    # Train a bit to build optimizer state
    for step in range(10):
        x = torch.randn(4, 10)
        loss = model(x).mean()
        loss.backward()
        rb.step(loss)
        opt.step()
        opt.zero_grad()

    # Check optimizer state exists
    has_state = len(opt.state) > 0
    saves_optimizer = 'optimizer_state' in rb.state.checkpoints[0]
    print(f'  Optimizer has state (m,v): {has_state}')
    print(f'  Checkpoint saves optimizer: {saves_optimizer}')
    assert has_state and saves_optimizer



def test_multi_failure():
    """Test 7: Multiple Consecutive Failures"""
    print('\n[7] MULTIPLE FAILURE RECOVERY')
    model = _ComprehensiveTestModel()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    rb = WeightRollback(model, opt, RollbackConfig(checkpoint_frequency=10, max_rollbacks_per_epoch=5), verbose=False)

    rollback_count = 0
    for step in range(100):
        x = torch.randn(4, 10)
        loss = model(x).mean()
        
        # Inject failures at steps 30, 50, 70
        if step in [30, 50, 70]:
            loss = loss * float('nan')
        
        try:
            loss.backward()
        except:
            pass
        
        action = rb.step(loss)
        if action.rolled_back:
            rollback_count += 1
        else:
            if not torch.isnan(loss) and not torch.isinf(loss):
                opt.step()
        opt.zero_grad()

    print(f'  Rollbacks triggered: {rollback_count}')
    print(f'  Training completed: True')
    assert rollback_count >= 3



def test_gradient_explosion():
    """Test 8: Gradient Explosion Detection"""
    print('\n[8] GRADIENT EXPLOSION DETECTION')
    model = _ComprehensiveTestModel()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    config = RollbackConfig(gradient_explosion_threshold=1e4)
    rb = WeightRollback(model, opt, config, verbose=False)

    # Normal step
    x = torch.randn(4, 10)
    loss = model(x).mean()
    loss.backward()
    rb.step(loss)
    opt.step()
    opt.zero_grad()

    # Simulate gradient explosion by scaling loss
    x = torch.randn(4, 10)
    loss = model(x).mean() * 1e6
    loss.backward()
    action = rb.step(loss)
    
    print(f'  Gradient explosion detected: {action.rolled_back}')
    assert action.rolled_back



if __name__ == '__main__':
    print('='*60)
    print('ARC COMPREHENSIVE TEST SUITE')
    print('='*60)

    tests = [
        ('RNG Preservation', test_rng_preservation),
        ('Failure Detection', test_failure_detection),
        ('Recovery Success', test_recovery_success),
        ('False Positives', test_false_positives),
        ('LR Reduction', test_lr_reduction),
        ('Optimizer State', test_optimizer_state),
        ('Multi Failure', test_multi_failure),
        ('Gradient Explosion', test_gradient_explosion),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            test_fn()
            results.append((name, 'PASS'))
        except AssertionError as e:
            results.append((name, f'FAIL: {e}'))
        except Exception as e:
            results.append((name, f'ERROR: {e}'))

    print('\n' + '='*60)
    print('TEST RESULTS SUMMARY')
    print('='*60)
    for name, status in results:
        symbol = '\u2713' if status == 'PASS' else '\u2717'
        print(f'  {symbol} {name}: {status}')
    
    passed = sum(1 for _, s in results if s == 'PASS')
    print(f'\nTotal: {passed}/{len(tests)} tests passed')
    print('='*60)