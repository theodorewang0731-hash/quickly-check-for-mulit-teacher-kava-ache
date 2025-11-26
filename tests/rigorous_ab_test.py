"""
Rigorous A/B Test: Linear vs Elastic Bottleneck
================================================

Purpose:
    Scientific comparison across three scenarios:
    1. Simple Linear Relationship (Sanity Check)
    2. Distribution Shift & Outliers (Testing LayerNorm)
    3. Complex Non-linear Knowledge (Testing MLP Capability)
    
Expected Results:
    Scenario 1: Linear ≈ MLP (both should work)
    Scenario 2: MLP >> Linear (LayerNorm handles outliers)
    Scenario 3: MLP >> Linear (non-linearity needed)

Author: Quick Check Team
Date: 2025-01-18
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import random
sys.path.append(".")

from experiments.kv_dimension_projector import KVDimensionProjector


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_scenario(name, teacher_gen_func, student_gen_func, steps=50):
    """
    Run one A/B test scenario.
    
    Args:
        name: Scenario name
        teacher_gen_func: Function to generate teacher data
        student_gen_func: Function to map teacher -> student
        steps: Number of training steps
    
    Returns:
        (final_linear_loss, final_mlp_loss)
    """
    print(f"\n{'='*60}")
    print(f">>>>> Scenario: {name}")
    print(f"{'='*60}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Fixed Configuration
    B, T = 4, 64
    D_teacher = 4096
    D_student = 2048
    config = {"TestTeacher": {"d_model": D_teacher, "num_layers": 28}}
    
    # 2. Generate Data (固定随机种子保证公平)
    set_seed(42)
    teacher_kv = teacher_gen_func(B, T, D_teacher).to(device)
    target_kv = student_gen_func(teacher_kv, D_student).to(device)
    
    print(f"\nData Statistics:")
    print(f"  Teacher: mean={teacher_kv.mean().item():.2f}, std={teacher_kv.std().item():.2f}")
    print(f"  Teacher: min={teacher_kv.min().item():.2f}, max={teacher_kv.max().item():.2f}")
    print(f"  Target:  mean={target_kv.mean().item():.2f}, std={target_kv.std().item():.2f}")
    
    # 3. Initialize Models (A/B Test)
    set_seed(42)  # Same initialization starting point
    
    # Baseline: Linear
    linear_model = nn.Linear(D_teacher, D_student).to(device)
    opt_lin = optim.AdamW(linear_model.parameters(), lr=1e-3)
    
    # Ours: Elastic Bottleneck (MLP Ratio=1.0)
    set_seed(42)  # Reset seed again for MLP
    mlp_model = KVDimensionProjector(
        config, 
        D_student, 
        mlp_ratio=1.0,
        dropout=0.1,
        trainable=True
    ).to(device)
    opt_mlp = optim.AdamW(mlp_model.parameters(), lr=1e-3)
    
    loss_fn = nn.MSELoss()
    
    print(f"\nModel Parameters:")
    print(f"  Linear: {sum(p.numel() for p in linear_model.parameters())/1e6:.2f}M")
    print(f"  MLP:    {mlp_model.count_parameters()/1e6:.2f}M")
    
    # 4. Training Loop
    results = {"linear": [], "mlp": []}
    
    print(f"\n{'Step':<5} | {'Linear Loss':<12} | {'MLP Loss':<12} | {'Leader'}")
    print("-" * 55)
    
    for step in range(1, steps + 1):
        # Linear
        opt_lin.zero_grad()
        loss_lin = loss_fn(linear_model(teacher_kv), target_kv)
        loss_lin.backward()
        opt_lin.step()
        results["linear"].append(loss_lin.item())
        
        # MLP
        opt_mlp.zero_grad()
        out_mlp, _ = mlp_model.project_teacher_kv("TestTeacher", teacher_kv, teacher_kv)
        loss_mlp = loss_fn(out_mlp, target_kv)
        loss_mlp.backward()
        opt_mlp.step()
        results["mlp"].append(loss_mlp.item())
        
        if step % 10 == 0:
            winner = "MLP" if loss_mlp.item() < loss_lin.item() else "Linear"
            diff = abs(loss_lin.item() - loss_mlp.item()) / max(loss_lin.item(), loss_mlp.item()) * 100
            print(f"{step:<5} | {loss_lin.item():.6f}     | {loss_mlp.item():.6f}     | {winner} (+{diff:.1f}%)")

    # 5. Summary
    print("-" * 55)
    
    initial_lin = results["linear"][0]
    final_lin = results["linear"][-1]
    reduction_lin = (initial_lin - final_lin) / initial_lin * 100
    
    initial_mlp = results["mlp"][0]
    final_mlp = results["mlp"][-1]
    reduction_mlp = (initial_mlp - final_mlp) / initial_mlp * 100
    
    print(f"\n[Results Summary]")
    print(f"  Linear:  {initial_lin:.6f} -> {final_lin:.6f} ({reduction_lin:.1f}% reduction)")
    print(f"  MLP:     {initial_mlp:.6f} -> {final_mlp:.6f} ({reduction_mlp:.1f}% reduction)")
    
    if final_mlp < final_lin:
        improvement = (final_lin - final_mlp) / final_lin * 100
        print(f"\n  [RESULT] MLP wins by {improvement:.2f}%")
        verdict = "WIN"
    elif abs(final_mlp - final_lin) / final_lin < 0.05:
        print(f"\n  [RESULT] Tie (difference < 5%)")
        verdict = "TIE"
    else:
        degradation = (final_mlp - final_lin) / final_lin * 100
        print(f"\n  [RESULT] Linear wins by {degradation:.2f}% (acceptable in simple linear tasks)")
        verdict = "LOSS"
    
    return final_lin, final_mlp, verdict


# ============================================================================
# Data Generation Functions
# ============================================================================

def gen_simple_linear(B, T, D):
    """
    Scenario 1: Simple normal distribution.
    Both Linear and MLP should work well.
    """
    return torch.randn(B, T, D)


def map_linear(teacher_kv, D_out):
    """
    Target is linear: y = x * W
    Use a fixed random matrix as "ground truth rule"
    """
    B, T, D_in = teacher_kv.shape
    
    # Create fixed transformation matrix (deterministic given seed)
    true_W = torch.randn(D_in, D_out, generator=torch.manual_seed(123)).to(teacher_kv.device)
    true_W = true_W / np.sqrt(D_in)  # Xavier-like scaling
    
    return teacher_kv @ true_W


def gen_distribution_shift(B, T, D):
    """
    Scenario 2: Simulate RoPE/LLM features with distribution shift.
    - Non-zero mean
    - Large variance
    - Outliers
    
    This tests LayerNorm's ability to handle diverse distributions.
    """
    # Large mean and std
    data = torch.randn(B, T, D) * 10.0 + 5.0
    
    # Insert outliers (every 100th dimension)
    data[:, :, ::100] = 100.0
    
    # Add some extremely small values
    data[:, :, 50::100] = -50.0
    
    return data


def map_nonlinear(teacher_kv, D_out):
    """
    Scenario 3: Non-linear mapping with tanh.
    
    This simulates complex reasoning knowledge that cannot be
    represented by simple linear transformation.
    """
    B, T, D_in = teacher_kv.shape
    
    # Create fixed transformation matrix
    true_W = torch.randn(D_in, D_out, generator=torch.manual_seed(456)).to(teacher_kv.device)
    true_W = true_W / np.sqrt(D_in)
    
    # Apply non-linear transformation
    return torch.tanh(teacher_kv @ true_W)


# ============================================================================
# Main Program
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("RIGOROUS A/B TEST: Linear vs Elastic Bottleneck")
    print("="*60)
    print("\nObjective: Test under 3 different scenarios to ensure")
    print("           Elastic Bottleneck is universally superior or at least")
    print("           not worse than Linear baseline.")
    
    results_summary = []
    
    # ========================================
    # Scenario 1: Simple Linear Relationship
    # ========================================
    # Expected: Both should work well (Linear can fit linear rules)
    # Verdict: MLP should NOT be worse (sanity check)
    
    lin1, mlp1, verdict1 = run_scenario(
        "1. Simple Linear Relationship (Sanity Check)",
        gen_simple_linear,
        map_linear,
        steps=50
    )
    results_summary.append(("Scenario 1 (Linear)", verdict1))
    
    # ========================================
    # Scenario 2: Distribution Shift
    # ========================================
    # Expected: MLP wins (LayerNorm stabilizes gradients)
    # Linear will struggle with outliers
    
    lin2, mlp2, verdict2 = run_scenario(
        "2. Distribution Shift & Outliers (Testing LayerNorm)",
        gen_distribution_shift,
        map_linear,
        steps=50
    )
    results_summary.append(("Scenario 2 (Outliers)", verdict2))
    
    # ========================================
    # Scenario 3: Complex Non-linear
    # ========================================
    # Expected: MLP wins (needs non-linearity to fit tanh)
    # Linear will underfit
    
    lin3, mlp3, verdict3 = run_scenario(
        "3. Complex Non-linear Knowledge (Testing MLP Capability)",
        gen_simple_linear,
        map_nonlinear,
        steps=50
    )
    results_summary.append(("Scenario 3 (Non-linear)", verdict3))
    
    # ========================================
    # Final Summary
    # ========================================
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    print("\n[Scenario Results]")
    for name, verdict in results_summary:
        symbol = "[+]" if verdict == "WIN" else "[-]" if verdict == "LOSS" else "[=]"
        print(f"  {symbol} {name:<40} {verdict}")
    
    # Overall verdict
    wins = sum(1 for _, v in results_summary if v == "WIN")
    ties = sum(1 for _, v in results_summary if v == "TIE")
    losses = sum(1 for _, v in results_summary if v == "LOSS")
    
    print(f"\n[Overall Score]")
    print(f"  Wins:   {wins}/3")
    print(f"  Ties:   {ties}/3")
    print(f"  Losses: {losses}/3")
    
    if wins >= 2 and losses == 0:
        print(f"\n[CONCLUSION]")
        print(f"  Elastic Bottleneck (MLP+Norm) is VALIDATED!")
        print(f"  - Superior in challenging scenarios (outliers, non-linear)")
        print(f"  - At worst ties with Linear in simple cases")
        print(f"  - Ready for production use")
    elif wins >= 1 and losses <= 1:
        print(f"\n[CONCLUSION]")
        print(f"  Elastic Bottleneck shows promise but needs tuning")
        print(f"  - Consider adjusting mlp_ratio or learning rate")
    else:
        print(f"\n[CONCLUSION]")
        print(f"  WARNING: Elastic Bottleneck underperforming")
        print(f"  - Review architecture or hyperparameters")
    
    print("\n" + "="*60)
