"""
Quick Convergence Test - Pure Mock Tensors (No Model Download)
===============================================================

最快的验证方式：不下载任何模型，纯用随机张量测试梯度流动。

Author: Quick Check Team
Date: 2025-01-26
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append(".")

from experiments.kv_dimension_projector import KVDimensionProjector
from experiments.alignment_v2 import resample_kv_with_interpolation


def quick_convergence_test():
    print("\n" + "="*80)
    print("🚀 QUICK CONVERGENCE TEST - Pure Mock Tensors (No Model Download)")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Configuration
    B = 2
    T_teacher = 64
    T_student = 32
    d_teacher = 1536  # Qwen2.5-1.5B size
    d_student = 896   # Qwen2.5-0.5B size
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {B}")
    print(f"  Teacher: T={T_teacher}, d_model={d_teacher}")
    print(f"  Student: T={T_student}, d_model={d_student}")
    
    # Create projector with Elastic Bottleneck
    print(f"\nInitializing Elastic Bottleneck Projector (mlp_ratio=1.0)...")
    projector = KVDimensionProjector(
        teacher_configs={"MockTeacher": {"d_model": d_teacher, "num_layers": 28}},
        student_d_model=d_student,
        mlp_ratio=1.0,      # New parameter
        dropout=0.1,        # New parameter
        init_method="xavier",
        trainable=True
    ).to(device)
    
    print(f"  Parameters: {projector.count_parameters():,}")
    
    # Optimizer
    optimizer = optim.AdamW(projector.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    
    # Generate FIXED teacher and student KV (overfit to same pair)
    print(f"\nGenerating fixed teacher and student KV pair...")
    with torch.no_grad():
        # Fixed teacher KV (this won't change)
        k_teacher_fixed = torch.randn(B, T_teacher, d_teacher).to(device)
        v_teacher_fixed = torch.randn(B, T_teacher, d_teacher).to(device)
        
        # Time resampling: 64 -> 32
        k_resampled_fixed = resample_kv_with_interpolation(k_teacher_fixed, T_student, None, None)
        v_resampled_fixed = resample_kv_with_interpolation(v_teacher_fixed, T_student, None, None)
        
        # Fixed student target
        k_target = torch.randn(B, T_student, d_student).to(device)
        v_target = torch.randn(B, T_student, d_student).to(device)
    
    print(f"\n" + "="*80)
    print("Starting Optimization Loop")
    print("Goal: Projector should learn to map FIXED teacher KV to FIXED student KV")
    print("      (Single pair overfit - simplest possible test)")
    print("="*80)
    
    loss_history = []
    
    for step in range(31):
        optimizer.zero_grad()
        
        # Use FIXED teacher KV each step
        k_resampled = k_resampled_fixed
        v_resampled = v_resampled_fixed
        
        # Add layer dimension for projector
        k_in = k_resampled.unsqueeze(1)  # [B, 1, T, d_t]
        v_in = v_resampled.unsqueeze(1)
        
        # Dimension projection (trainable)
        k_proj, v_proj = projector.project_teacher_kv("MockTeacher", k_in, v_in)
        k_proj = k_proj.squeeze(1)  # [B, T, d_s]
        v_proj = v_proj.squeeze(1)
        
        # Calculate loss
        loss_k = loss_fn(k_proj, k_target)
        loss_v = loss_fn(v_proj, v_target)
        total_loss = loss_k + loss_v
        
        loss_history.append(total_loss.item())
        
        # Backward
        total_loss.backward()
        
        # Check gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
        
        optimizer.step()
        
        if step % 5 == 0:
            print(f"Step {step:02d} | Loss: {total_loss.item():.6f} "
                  f"(K: {loss_k.item():.6f}, V: {loss_v.item():.6f}) | "
                  f"GradNorm: {grad_norm:.4f}")
    
    # Analysis
    print("\n" + "="*80)
    print("CONVERGENCE ANALYSIS")
    print("="*80)
    
    initial_loss = loss_history[0]
    final_loss = loss_history[-1]
    min_loss = min(loss_history)
    reduction = ((initial_loss - final_loss) / initial_loss * 100)
    
    print(f"Initial Loss:    {initial_loss:.6f}")
    print(f"Final Loss:      {final_loss:.6f}")
    print(f"Minimum Loss:    {min_loss:.6f}")
    print(f"Reduction:       {reduction:.1f}%")
    
    # Check for common issues
    print(f"\nDiagnostics:")
    if grad_norm < 1e-6:
        print("  ⚠️  Gradient is too small (vanishing gradient)")
    elif grad_norm > 10.0:
        print("  ⚠️  Gradient is too large (exploding gradient)")
    else:
        print(f"  ✓ Gradient norm is healthy: {grad_norm:.4f}")
    
    if any(torch.isnan(torch.tensor(loss_history))):
        print("  ❌ NaN detected in loss history")
    else:
        print("  ✓ No NaN in loss history")
    
    # Convergence check
    print(f"\n" + "="*80)
    if final_loss < 0.01:
        print("✅ EXCELLENT: Loss converged to near zero!")
        print("   Pipeline is fully functional and learnable.")
        success = True
    elif final_loss < 0.1:
        print("✅ SUCCESS: Loss converged significantly!")
        print("   Pipeline is learnable, may need more steps for perfect fit.")
        success = True
    elif final_loss < initial_loss * 0.5:
        print("⚠️  PARTIAL SUCCESS: Loss is decreasing")
        print("   Pipeline is learning, but convergence is slow.")
        print("   Consider: increase learning rate or training steps")
        success = True
    else:
        print("❌ FAILURE: Loss is not decreasing")
        print("   Possible issues:")
        print("   1. Gradient flow is blocked")
        print("   2. Learning rate is too small")
        print("   3. Initialization is poor")
        success = False
    
    print("="*80)
    
    # Additional analysis
    if success:
        print("\n✨ Next Steps:")
        print("  1. ✓ Shape verification passed (previous test)")
        print("  2. ✓ Convergence verified (this test)")
        print("  3. → Ready to test with real models (smaller batches)")
        print("  4. → Then introduce Spherical Loss / Map Projection")
    
    return success, loss_history


def plot_loss_curve(loss_history):
    """Simple text-based loss curve"""
    print("\n" + "="*80)
    print("LOSS CURVE (Text Visualization)")
    print("="*80)
    
    max_loss = max(loss_history)
    min_loss = min(loss_history)
    
    print(f"Max: {max_loss:.4f} {'█' * 50}")
    
    for i, loss in enumerate(loss_history):
        if i % 5 == 0:  # Only show every 5 steps
            # Normalize to 0-50 range
            bar_len = int((loss - min_loss) / (max_loss - min_loss + 1e-8) * 50)
            bar = '█' * bar_len
            print(f"Step {i:02d}: {loss:.4f} {bar}")
    
    print(f"Min: {min_loss:.4f} {'█' * 0}")


if __name__ == "__main__":
    success, loss_history = quick_convergence_test()
    
    # Plot loss curve
    plot_loss_curve(loss_history)
    
    if success:
        print("\n" + "🎉"*40)
        print("PIPELINE IS READY!")
        print("🎉"*40)
    else:
        print("\n" + "⚠️ "*40)
        print("NEEDS DEBUGGING")
        print("⚠️ "*40)
    
    exit(0 if success else 1)
