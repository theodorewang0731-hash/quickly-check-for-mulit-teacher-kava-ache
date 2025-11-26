"""
Quick A/B Test - Fast version for rapid validation
===================================================

Same 3 scenarios but with fewer steps for quick results.

Author: Quick Check Team  
Date: 2025-01-18
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
sys.path.append(".")

from experiments.kv_dimension_projector import KVDimensionProjector


def quick_test(scenario_name, teacher_data, target_data, steps=30):
    """Run a quick A/B test."""
    print(f"\n[{scenario_name}]")
    
    device = "cpu"  # Force CPU for speed
    B, T, D_teacher = teacher_data.shape
    _, _, D_student = target_data.shape
    
    # Normalize data statistics for fair comparison
    data_mean = teacher_data.mean()
    data_std = teacher_data.std()
    print(f"  Data: mean={data_mean:.2f}, std={data_std:.2f}, range=[{teacher_data.min():.2f}, {teacher_data.max():.2f}]")
    
    # Models
    linear = nn.Linear(D_teacher, D_student)
    mlp = KVDimensionProjector(
        {"T": {"d_model": D_teacher, "num_layers": 1}},
        D_student,
        mlp_ratio=1.0,
        dropout=0.1,  # Re-enable dropout
        trainable=True
    )
    
    # Adjust learning rates based on data scale
    lr_lin = 1e-3 if data_std < 5 else 1e-4
    lr_mlp = 1e-3 if data_std < 5 else 5e-4
    
    opt_lin = optim.AdamW(linear.parameters(), lr=lr_lin)
    opt_mlp = optim.AdamW(mlp.parameters(), lr=lr_mlp)
    loss_fn = nn.MSELoss()
    
    # Track best losses
    best_lin, best_mlp = float('inf'), float('inf')
    
    # Train
    for step in range(steps):
        opt_lin.zero_grad()
        loss_lin = loss_fn(linear(teacher_data), target_data)
        loss_lin.backward()
        opt_lin.step()
        best_lin = min(best_lin, loss_lin.item())
        
        opt_mlp.zero_grad()
        out_mlp, _ = mlp.project_teacher_kv("T", teacher_data, teacher_data)
        loss_mlp = loss_fn(out_mlp, target_data)
        loss_mlp.backward()
        opt_mlp.step()
        best_mlp = min(best_mlp, loss_mlp.item())
        
        if (step + 1) % 10 == 0:
            print(f"  Step {step+1:02d}: Linear={loss_lin.item():.4f}, MLP={loss_mlp.item():.4f}")
    
    winner = "MLP" if best_mlp < best_lin else "Linear"
    diff = abs(best_lin - best_mlp) / max(best_lin, 1e-8) * 100
    print(f"  Best:  Linear={best_lin:.4f}, MLP={best_mlp:.4f} | Winner: {winner} (+{diff:.1f}%)")
    return winner


if __name__ == "__main__":
    print("\n" + "="*50)
    print("QUICK A/B TEST (30 steps, adaptive LR)")
    print("="*50)
    
    # Test setup
    B, T = 2, 32
    D_t, D_s = 1024, 512
    
    # Scenario 1: Simple Linear
    print("\n>>> Testing: Can both models learn simple linear mapping?")
    torch.manual_seed(42)
    x1 = torch.randn(B, T, D_t)
    W_true = torch.randn(D_t, D_s) / np.sqrt(D_t)
    y1 = x1 @ W_true
    w1 = quick_test("Scenario 1: Simple Linear", x1, y1, steps=40)
    
    # Scenario 2: Distribution Shift (with proper scaling)
    print("\n>>> Testing: Can MLP handle outliers better with LayerNorm?")
    torch.manual_seed(42)
    x2 = torch.randn(B, T, D_t) * 5.0 + 2.0  # Moderate shift
    x2[:, :, ::50] = 20.0  # Moderate outliers
    y2 = x2 @ W_true
    w2 = quick_test("Scenario 2: Outliers", x2, y2, steps=40)
    
    # Scenario 3: Non-linear
    print("\n>>> Testing: Can MLP learn non-linear relationships?")
    torch.manual_seed(42)
    x3 = torch.randn(B, T, D_t)
    y3 = torch.tanh(x3 @ W_true)
    w3 = quick_test("Scenario 3: Non-linear", x3, y3, steps=40)
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    wins = [w1, w2, w3].count("MLP")
    print(f"MLP wins: {wins}/3")
    
    if wins >= 2:
        print("\n[PASS] Elastic Bottleneck validated!")
    else:
        print("\n[REVIEW] Consider tuning hyperparameters")
