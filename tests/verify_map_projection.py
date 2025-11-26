"""
Verify Map Projection Loss vs. Euclidean MSE
=============================================

Purpose:
    Demonstrate that Mercator/Spherical projection correctly handles
    the "same direction, different magnitude" scenario that MSE fails at.
    
Scenario:
    - Teacher: Very confident (magnitude = 100)
    - Student: Just learning (magnitude = 1)
    - Semantic: IDENTICAL direction
    
Expected:
    - MSE: Huge loss (~9800) - forces magnitude match
    - Mercator: Zero loss (~0.0000) - recognizes semantic alignment

Author: Quick Check Team
Date: 2025-01-18
"""

import torch
import torch.nn as nn
import sys
sys.path.append(".")

from src.losses import MercatorKVLoss, compute_angular_distance, compute_alignment_accuracy


def run_verification():
    print("\n" + "="*60)
    print("Map Projection vs. Euclidean (MSE) Comparison")
    print("="*60)
    
    # Scenario: Teacher and Student have identical semantics
    # but Teacher has much higher confidence (magnitude)
    B, T, D = 2, 10, 128
    
    # 1. Base semantic vector (direction)
    base_vector = torch.randn(B, T, D)
    base_vector = torch.nn.functional.normalize(base_vector, dim=-1)
    
    # 2. Create magnitude difference
    # Teacher: High confidence (magnitude = 100)
    teacher_kv = base_vector * 100.0
    # Student: Just starting (magnitude = 1)
    student_kv = base_vector * 1.0
    
    print(f"\n[Test Scenario]")
    print(f"  Teacher Magnitude: 100.0 (very confident)")
    print(f"  Student Magnitude: 1.0   (just learning)")
    print(f"  Direction:         IDENTICAL (perfect semantic alignment)")
    print("-" * 60)

    # --- Test MSE (Euclidean Distance) ---
    mse_fn = nn.MSELoss()
    loss_mse = mse_fn(student_kv, teacher_kv)
    
    print(f"\n[MSE Loss - Euclidean Approach]")
    print(f"  Loss: {loss_mse.item():.2f}")
    print(f"  ")
    print(f"  Interpretation:")
    print(f"    MSE sees huge numerical difference (100 vs 1)")
    print(f"    Forces Student to increase magnitude to match Teacher")
    print(f"    Ignores that semantics are already perfectly aligned")

    # --- Test Map Projection (Mercator/Spherical) ---
    map_fn = MercatorKVLoss(alpha=1.0, beta=0.0)  # Pure direction mode
    loss_map, metrics = map_fn(student_kv, teacher_kv)
    
    print(f"\n[Mercator Loss - Spherical Projection]")
    print(f"  Loss:       {loss_map.item():.6f}")
    print(f"  Cosine Sim: {metrics['cos_sim']:.6f}")
    print(f"  Direction:  {metrics['dir_loss']:.6f}")
    print(f"  ")
    print(f"  Interpretation:")
    print(f"    Mercator projects both onto unit sphere")
    print(f"    Recognizes identical semantic direction")
    print(f"    Loss ≈ 0 (perfect alignment)")
    
    # --- Angular Distance ---
    angle = compute_angular_distance(student_kv, teacher_kv).mean()
    print(f"\n[Angular Distance]")
    print(f"  Angle: {angle.item():.6f} radians ({angle.item() * 180 / 3.14159:.2f} degrees)")
    print(f"  Expected: 0.0 degrees (vectors point in same direction)")
    
    # --- Verification ---
    print("\n" + "="*60)
    print("[Verification Results]")
    print("="*60)
    
    mse_large = loss_mse.item() > 100
    mercator_small = loss_map.item() < 1e-5
    cos_sim_perfect = metrics['cos_sim'] > 0.9999
    
    if mse_large and mercator_small and cos_sim_perfect:
        print("[PASS] Verification successful!")
        print("")
        print("Key Findings:")
        print(f"  1. MSE Loss is huge ({loss_mse.item():.0f}) - treats this as error")
        print(f"  2. Mercator Loss is ~0 ({loss_map.item():.6f}) - recognizes alignment")
        print(f"  3. Cosine Similarity is 1.0 ({metrics['cos_sim']:.6f}) - perfect match")
        print("")
        print("Conclusion:")
        print("  Map Projection correctly ignores magnitude differences")
        print("  and focuses on semantic direction alignment!")
    else:
        print("[FAIL] Verification failed. Check implementation.")
        print(f"  MSE large? {mse_large} (expected: True)")
        print(f"  Mercator small? {mercator_small} (expected: True)")
        print(f"  Cos sim perfect? {cos_sim_perfect} (expected: True)")


def test_varied_angles():
    """Test with different angular separations."""
    print("\n" + "="*60)
    print("Test: Varied Angular Separations")
    print("="*60)
    
    B, T, D = 2, 10, 128
    teacher = torch.randn(B, T, D)
    teacher = torch.nn.functional.normalize(teacher, dim=-1) * 50.0
    
    loss_fn = MercatorKVLoss(alpha=1.0, beta=0.0)
    
    angles_deg = [0, 15, 30, 45, 60, 90, 120, 150, 180]
    
    print(f"\n{'Angle (deg)':<12} | {'Cos Sim':<10} | {'Mercator Loss':<15} | {'Status'}")
    print("-" * 60)
    
    for angle_deg in angles_deg:
        # Create student vector at specific angle
        angle_rad = angle_deg * 3.14159 / 180
        
        # Simple 2D rotation for visualization
        student = teacher.clone()
        if angle_deg > 0:
            # Perturb to create angle
            noise = torch.randn_like(teacher)
            noise = torch.nn.functional.normalize(noise, dim=-1)
            
            # Mix to get desired angle (approximate)
            cos_target = torch.cos(torch.tensor(angle_rad))
            student = cos_target * teacher + torch.sqrt(1 - cos_target**2) * noise
            student = torch.nn.functional.normalize(student, dim=-1) * 1.0
        
        loss, metrics = loss_fn(student, teacher)
        
        status = ""
        if angle_deg <= 30:
            status = "[Excellent]"
        elif angle_deg <= 60:
            status = "[Good]"
        elif angle_deg <= 90:
            status = "[Acceptable]"
        else:
            status = "[Poor]"
        
        print(f"{angle_deg:<12} | {metrics['cos_sim']:<10.4f} | {loss.item():<15.6f} | {status}")
    
    print("\nInterpretation:")
    print("  - Angle 0-30°:   Excellent alignment (target range)")
    print("  - Angle 30-60°:  Good alignment")
    print("  - Angle 60-90°:  Acceptable but needs improvement")
    print("  - Angle 90-180°: Poor alignment (orthogonal or opposite)")


def test_magnitude_constraint():
    """Test magnitude constraint (beta parameter)."""
    print("\n" + "="*60)
    print("Test: Magnitude Constraint Effect")
    print("="*60)
    
    B, T, D = 2, 10, 128
    
    base = torch.randn(B, T, D)
    base = torch.nn.functional.normalize(base, dim=-1)
    
    teacher = base * 100.0
    student = base * 1.0
    
    betas = [0.0, 0.01, 0.05, 0.1, 1.0]
    
    print(f"\n{'Beta':<8} | {'Total Loss':<12} | {'Dir Loss':<12} | {'Mag Loss':<12}")
    print("-" * 60)
    
    for beta in betas:
        loss_fn = MercatorKVLoss(alpha=1.0, beta=beta)
        loss, metrics = loss_fn(student, teacher)
        
        print(f"{beta:<8.2f} | {loss.item():<12.6f} | {metrics['dir_loss']:<12.6f} | {metrics['mag_loss']:<12.6f}")
    
    print("\nRecommendation:")
    print("  - Beta = 0.0:   Pure direction (recommended for semantic alignment)")
    print("  - Beta = 0.01:  Weak constraint (prevents extreme collapse)")
    print("  - Beta >= 0.1:  Too strong (defeats purpose of direction-only loss)")


if __name__ == "__main__":
    # Main verification
    run_verification()
    
    # Additional tests
    test_varied_angles()
    test_magnitude_constraint()
    
    print("\n" + "="*60)
    print("All verifications complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Integrate MercatorKVLoss into train_with_kv.py")
    print("  2. Replace MSE with Mercator in training loop")
    print("  3. Monitor Cosine Similarity during training (aim for 0.95+)")
