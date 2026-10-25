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

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from arc.introspection import (
    IntrospectiveArc as Arc,
    FisherRaoGeometry,
    LyapunovEstimator,
)
from arc.introspection.neural_ode import TrajectoryPredictor


def create_synthetic_data(n_samples: int = 1000, n_features: int = 20, n_classes: int = 5):
    X = torch.randn(n_samples, n_features)
    
    centers = torch.randn(n_classes, n_features) * 3
    y = torch.randint(0, n_classes, (n_samples,))
    
    for i in range(n_classes):
        mask = y == i
        X[mask] = X[mask] + centers[i]
    
    return X, y


def create_model(n_features: int, n_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(n_features, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, n_classes),
    )


def print_introspection(state, epoch: int, step: int):
    print(f"\n{'='*60}")
    print(f"Epoch {epoch}, Step {step}")
    print(f"{'='*60}")
    
    print(f"\nHEALTH: {state.overall_health:.2%}  |  RISK: {state.risk_score:.2%}")
    print(f"Confidence: {state.confidence:.2%}")
    
    if state.geometry:
        g = state.geometry
        print(f"\nGEOMETRY (Fisher-Rao)")
        print(f"   Geodesic velocity: {g.geodesic_velocity:.4f}")
        print(f"   Curvature: {g.riemannian_curvature:.4f}")
        print(f"   Natural grad alignment: {g.natural_gradient_alignment:.2%}")
    
    if state.dynamics:
        d = state.dynamics
        phase_emoji = {"chaotic": "🌀", "stable": "✅", "edge_of_chaos": "⚡", "converged": "🎯"}
        print(f"\nDYNAMICS (Lyapunov)")
        print(f"   Lyapunov exponent: {d.lyapunov_exponent:.4f}")
        print(f"   Phase: {phase_emoji.get(d.phase, '❓')} {d.phase}")
        print(f"   At bifurcation: {'YES' if d.is_at_bifurcation else 'No'}")
    
    if state.transport:
        t = state.transport
        print(f"\nTRANSPORT (Wasserstein)")
        print(f"   Distribution drift: {t.wasserstein_distance:.4f}")
        print(f"   Mode collapse risk: {t.mode_collapse_risk:.2%}")
        print(f"   Entropy: {t.distribution_entropy:.4f}")
    
    if state.self_knowledge:
        s = state.self_knowledge
        print(f"\nSELF-MODEL (Metacognition)")
        print(f"   Predicted grad: {s.predicted_gradient_norm:.4f}")
        print(f"   Actual grad: {s.actual_gradient_norm:.4f}")
        print(f"   Surprise: {s.surprise:.2f}σ")
        print(f"   Confidence: {s.confidence:.2%}")
    
    if state.should_intervene:
        print(f"\nINTERVENTION RECOMMENDED: {state.recommended_action}")


def main():
    print("=" * 60)
    print("ARC: Self-Introspecting Neural Network Demo")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    print("\nCreating synthetic data...")
    X, y = create_synthetic_data(n_samples=2000, n_features=20, n_classes=5)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    print("Creating model...")
    base_model = create_model(n_features=20, n_classes=5).to(device)
    
    optimizer = torch.optim.Adam(base_model.parameters(), lr=0.01)
    
    print("Wrapping with ARC...")
    omega = Arc(
        base_model,
        optimizer=optimizer,
        enable_geometry=True,
        enable_dynamics=True,
        enable_topology=False,
        enable_transport=True,
        enable_self_model=True,
        auto_intervene=True,
    )
    
    predictor = TrajectoryPredictor(signal_dim=20)
    
    print("\nStarting training with introspection...\n")
    
    n_epochs = 10
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        
        for batch_idx, (batch_x, batch_y) in enumerate(dataloader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = omega(batch_x)
            loss = F.cross_entropy(output, batch_y)
            
            loss.backward()
            
            state = omega.introspective_step(loss, batch_x, batch_y)
            
            grad_norm = sum(p.grad.norm().item()**2 for p in base_model.parameters() if p.grad is not None)**0.5
            weight_norm = sum(p.norm().item()**2 for p in base_model.parameters())**0.5
            predictor.update(
                loss=loss.item(),
                gradient_norm=grad_norm,
                learning_rate=optimizer.param_groups[0]['lr'],
                weight_norm=weight_norm,
                epoch=epoch,
            )
            
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
            
            if batch_idx % 10 == 0:
                print_introspection(state, epoch, batch_idx)
        
        avg_loss = epoch_loss / n_batches
        summary = omega.on_epoch_end()
        
        print(f"\n{'*'*60}")
        print(f"EPOCH {epoch} SUMMARY")
        print(f"   Avg Loss: {avg_loss:.4f}")
        print(f"   Health: {summary.get('health', 0):.2%}")
        print(f"   Phase: {summary.get('phase', 'unknown')}")
        
        future = predictor.predict(n_epochs=5)
        print(f"\nTRAJECTORY PREDICTION (next 5 epochs):")
        print(f"   Future losses: {[f'{l:.4f}' for l in future.future_losses[:5]]}")
        print(f"   Divergence risk: {future.divergence_risk:.2%}")
        print(f"{'*'*60}\n")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    base_model.eval()
    with torch.no_grad():
        test_output = base_model(X.to(device))
        predictions = test_output.argmax(dim=1)
        accuracy = (predictions == y.to(device)).float().mean()
    
    print(f"\nFinal Accuracy: {accuracy:.2%}")
    print(f"Final Health: {state.overall_health:.2%}")
    
    health_trajectory = omega.get_health_trajectory()
    print(f"\nHealth trajectory (last 10):")
    print(f"   {[f'{h:.2f}' for h in health_trajectory[-10:]]}")
    
    print("\nDemo complete!")


if __name__ == "__main__":
    main()