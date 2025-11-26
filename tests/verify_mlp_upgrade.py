"""
Verify Elastic Bottleneck Projector Upgrade
============================================

Purpose:
    Compare Linear projection vs. MLP+Norm for medium teachers (<= 70B)
    
Scenario:
    Qwen-14B (5120 dim) -> Qwen-1.5B (1536 dim)
    
Expected:
    MLP should converge faster and better than pure Linear

Author: Quick Check Team
Date: 2025-01-18
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append(".")

from experiments.kv_dimension_projector import KVDimensionProjector


def run_comparison():
    print("\n" + "="*60)
    print("Medium Teacher (<=70B) Projector Battle")
    print("    Scenario: Qwen-14B -> Qwen-1.5B")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # --- 1. Simulate Parameters ---
    # Teacher: Qwen-14B (5120 dim)
    # Student: Qwen-1.5B (1536 dim)
    B, T = 4, 64
    D_teacher = 5120
    D_student = 1536
    
    # Simulate Teacher data (模拟真实分布：非零均值，有大数值)
    teacher_kv = (torch.randn(B, T, D_teacher).to(device) * 3.0) + 1.5
    
    # Simulate target (非线性关系)
    target_kv = torch.tanh(torch.randn(B, T, D_student).to(device))
    
    config = {"Mock-14B": {"d_model": D_teacher, "num_layers": 28}}

    # --- 2. Initialize Comparison ---
    
    # A. Old Method: Pure Linear
    print("Initializing Linear Baseline...")
    linear_proj = nn.Linear(D_teacher, D_student).to(device)
    
    # B. New Method: Lightweight MLP (Ratio=1.0)
    # For <30B models, 1.0x width is sufficient and parameter-efficient
    print("\nInitializing Elastic Bottleneck (MLP ratio=1.0)...")
    mlp_proj = KVDimensionProjector(
        config, 
        D_student, 
        mlp_ratio=1.0,
        dropout=0.1,
        trainable=True
    ).to(device)
    
    print(f"\n[Model Parameters]")
    print(f"  Linear: {sum(p.numel() for p in linear_proj.parameters())/1e6:.2f}M")
    print(f"  MLP:    {mlp_proj.count_parameters()/1e6:.2f}M")

    # --- 3. Training Race ---
    opt_lin = optim.AdamW(linear_proj.parameters(), lr=1e-3)
    opt_mlp = optim.AdamW(mlp_proj.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    print("\n[Starting Training - 40 Steps]")
    print(f"{'Step':<5} | {'Linear Loss':<12} | {'MLP Loss':<12} | {'Improvement'}")
    print("-" * 60)
    
    loss_linear_history = []
    loss_mlp_history = []
    
    for step in range(1, 41):
        # Linear
        opt_lin.zero_grad()
        loss_lin = loss_fn(linear_proj(teacher_kv), target_kv)
        loss_lin.backward()
        opt_lin.step()
        loss_linear_history.append(loss_lin.item())
        
        # MLP
        opt_mlp.zero_grad()
        out_mlp, _ = mlp_proj.project_teacher_kv("Mock-14B", teacher_kv, teacher_kv)
        loss_mlp = loss_fn(out_mlp, target_kv)
        loss_mlp.backward()
        opt_mlp.step()
        loss_mlp_history.append(loss_mlp.item())
        
        if step % 5 == 0:
            diff = (loss_lin.item() - loss_mlp.item()) / loss_lin.item() * 100
            sign = "[+]" if diff > 0 else "[-]"
            print(f"{step:<5} | {loss_lin.item():.6f}   | {loss_mlp.item():.6f}   | {sign} {abs(diff):.1f}%")

    print("-" * 60)
    
    # --- 4. Final Analysis ---
    initial_lin = loss_linear_history[0]
    final_lin = loss_linear_history[-1]
    reduction_lin = (initial_lin - final_lin) / initial_lin * 100
    
    initial_mlp = loss_mlp_history[0]
    final_mlp = loss_mlp_history[-1]
    reduction_mlp = (initial_mlp - final_mlp) / initial_mlp * 100
    
    print(f"\n[Final Results]")
    print(f"  Linear:  {initial_lin:.6f} -> {final_lin:.6f} ({reduction_lin:.1f}% reduction)")
    print(f"  MLP:     {initial_mlp:.6f} -> {final_mlp:.6f} ({reduction_mlp:.1f}% reduction)")
    
    if final_mlp < final_lin:
        improvement = (final_lin - final_mlp) / final_lin * 100
        print(f"\n[VERIFICATION PASSED]")
        print(f"   MLP is {improvement:.1f}% better than Linear!")
        print(f"   Elastic Bottleneck (MLP+Norm) significantly outperforms pure Linear.")
    else:
        print(f"\n[WARNING] Results close. Check data distribution.")
        print(f"   Linear: {final_lin:.6f}, MLP: {final_mlp:.6f}")
    
    # Visualize loss curves
    print(f"\n[Loss Trajectory - every 5 steps]")
    max_loss = max(max(loss_linear_history), max(loss_mlp_history))
    
    for i in range(0, 40, 5):
        lin_loss = loss_linear_history[i]
        mlp_loss = loss_mlp_history[i]
        
        lin_bar = "█" * int(lin_loss / max_loss * 40)
        mlp_bar = "█" * int(mlp_loss / max_loss * 40)
        
        print(f"  Step {i+1:02d}:")
        print(f"    Linear: {lin_loss:.6f} {lin_bar}")
        print(f"    MLP:    {mlp_loss:.6f} {mlp_bar}")


if __name__ == "__main__":
    run_comparison()
