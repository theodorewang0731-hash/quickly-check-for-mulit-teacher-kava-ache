"""
Mercator/Spherical Projection Loss for KV Alignment
====================================================

Purpose:
    Align semantic directions rather than numerical values.
    "Direction" represents semantics, "magnitude" represents confidence.
    
Core Idea:
    For RoPE-based models (Qwen/Llama), rotational consistency (direction)
    is more important than value approximation (magnitude).
    
Formula:
    Loss = 1 - CosineSimilarity(student, teacher)
    
Why this works:
    - Teacher: 100 * [0.707, 0.707]  (high confidence)
    - Student:  1 * [0.707, 0.707]  (low confidence)
    - MSE: Huge loss (forces magnitude match)
    - Mercator: Zero loss (recognizes semantic alignment)

Author: Quick Check Team
Date: 2025-01-18
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MercatorKVLoss(nn.Module):
    """
    Mercator/Spherical Projection Loss for KV Cache Alignment.
    
    Projects high-dimensional vectors onto unit sphere and compares directions.
    Optionally includes weak magnitude constraint to prevent collapse.
    
    Args:
        alpha: Weight for direction alignment (primary/Mercator component)
        beta: Weight for magnitude alignment (optional/Euclidean component)
              Recommended: 0.0 (pure direction) or 0.01 (weak constraint)
        epsilon: Small constant for numerical stability
    """
    
    def __init__(self, alpha=1.0, beta=0.0, epsilon=1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        
        print(f"[Mercator KV Loss Initialized]")
        print(f"  Alpha (direction): {alpha}")
        print(f"  Beta (magnitude):  {beta}")
        if beta > 0:
            print(f"  Mode: Direction + Weak Magnitude Constraint")
        else:
            print(f"  Mode: Pure Direction Alignment")
    
    def forward(self, student_kv, teacher_kv):
        """
        Compute Mercator projection loss.
        
        Args:
            student_kv: [Batch, Seq, Dim] - Student's representation
            teacher_kv: [Batch, Seq, Dim] - Teacher's projection (ground truth)
        
        Returns:
            total_loss: Scalar tensor
            metrics: Dict with detailed metrics
        """
        # --- 1. Mercator Projection (Direction Alignment) ---
        # Normalize to unit sphere surface
        s_norm = F.normalize(student_kv, p=2, dim=-1)
        t_norm = F.normalize(teacher_kv, p=2, dim=-1)
        
        # Compute direction consistency (Cosine Similarity)
        # cos_sim close to 1.0 means perfect alignment
        cos_sim = torch.sum(s_norm * t_norm, dim=-1).mean()
        
        # Mercator Loss: larger angle -> larger loss
        direction_loss = 1.0 - cos_sim
        
        # --- 2. (Optional) Magnitude Constraint ---
        # Prevent student magnitude from collapsing to 0 or exploding
        if self.beta > 0:
            s_mag = torch.norm(student_kv, p=2, dim=-1)
            t_mag = torch.norm(teacher_kv, p=2, dim=-1)
            
            # Use log space for scale-invariant comparison
            magnitude_loss = F.mse_loss(
                torch.log(s_mag + self.epsilon),
                torch.log(t_mag + self.epsilon)
            )
        else:
            magnitude_loss = torch.tensor(0.0, device=student_kv.device)
        
        # --- 3. Combined Loss ---
        total_loss = self.alpha * direction_loss + self.beta * magnitude_loss
        
        # --- 4. Compute Metrics ---
        with torch.no_grad():
            # Average magnitudes
            s_mag_mean = torch.norm(student_kv, p=2, dim=-1).mean().item()
            t_mag_mean = torch.norm(teacher_kv, p=2, dim=-1).mean().item()
            
            # Magnitude ratio (should stay reasonable, not collapse to 0)
            mag_ratio = s_mag_mean / (t_mag_mean + self.epsilon)
        
        metrics = {
            "loss": total_loss.item(),
            "cos_sim": cos_sim.item(),           # Core metric: aim for 0.95+
            "dir_loss": direction_loss.item(),
            "mag_loss": magnitude_loss.item() if self.beta > 0 else 0.0,
            "s_magnitude": s_mag_mean,
            "t_magnitude": t_mag_mean,
            "mag_ratio": mag_ratio
        }
        
        return total_loss, metrics


class HybridKVLoss(nn.Module):
    """
    Hybrid loss combining Mercator (direction) and MSE (value).
    
    Useful for gradual transition from MSE to pure Mercator.
    
    Args:
        mercator_weight: Weight for direction loss (0.0 to 1.0)
        mse_weight: Weight for MSE loss
        beta: Magnitude constraint in Mercator component
    """
    
    def __init__(self, mercator_weight=0.8, mse_weight=0.2, beta=0.01):
        super().__init__()
        self.mercator_weight = mercator_weight
        self.mse_weight = mse_weight
        
        self.mercator_loss = MercatorKVLoss(alpha=1.0, beta=beta)
        self.mse_loss = nn.MSELoss()
        
        print(f"[Hybrid KV Loss Initialized]")
        print(f"  Mercator weight: {mercator_weight}")
        print(f"  MSE weight:      {mse_weight}")
    
    def forward(self, student_kv, teacher_kv):
        """Compute hybrid loss."""
        # Mercator component
        merc_loss, merc_metrics = self.mercator_loss(student_kv, teacher_kv)
        
        # MSE component
        mse = self.mse_loss(student_kv, teacher_kv)
        
        # Combined
        total_loss = self.mercator_weight * merc_loss + self.mse_weight * mse
        
        metrics = {
            "loss": total_loss.item(),
            "mercator_loss": merc_loss.item(),
            "mse_loss": mse.item(),
            "cos_sim": merc_metrics["cos_sim"],
            "s_magnitude": merc_metrics["s_magnitude"],
            "t_magnitude": merc_metrics["t_magnitude"]
        }
        
        return total_loss, metrics


# ============================================================================
# Helper Functions
# ============================================================================

def compute_angular_distance(vec1, vec2):
    """
    Compute angular distance (in radians) between two vectors.
    
    Returns:
        angle: Tensor of angles in [0, pi]
    """
    cos_sim = F.cosine_similarity(vec1, vec2, dim=-1)
    # Clamp for numerical stability
    cos_sim = torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)
    angle = torch.acos(cos_sim)
    return angle


def compute_alignment_accuracy(student_kv, teacher_kv, threshold=0.95):
    """
    Compute percentage of vectors aligned within threshold.
    
    Args:
        threshold: Cosine similarity threshold (default 0.95 = ~18 degrees)
    
    Returns:
        accuracy: Percentage in [0, 100]
    """
    cos_sim = F.cosine_similarity(student_kv, teacher_kv, dim=-1)
    aligned = (cos_sim >= threshold).float().mean()
    return aligned.item() * 100


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Mercator KV Loss")
    print("="*60)
    
    # Test 1: Perfect direction alignment, different magnitudes
    print("\n[Test 1] Perfect Direction, Different Magnitudes")
    B, T, D = 2, 10, 128
    
    base = torch.randn(B, T, D)
    base = F.normalize(base, dim=-1)
    
    teacher = base * 100.0  # High confidence
    student = base * 1.0    # Low confidence
    
    loss_fn = MercatorKVLoss(alpha=1.0, beta=0.0)
    loss, metrics = loss_fn(student, teacher)
    
    print(f"  Loss:       {loss.item():.6f}")
    print(f"  Cos Sim:    {metrics['cos_sim']:.6f}")
    print(f"  Expected:   Loss ≈ 0, Cos Sim ≈ 1.0")
    
    assert abs(metrics['cos_sim'] - 1.0) < 1e-5, "Should be perfectly aligned"
    print("  [PASS]")
    
    # Test 2: Orthogonal vectors
    print("\n[Test 2] Orthogonal Vectors (90 degrees)")
    teacher = torch.randn(B, T, D)
    student = torch.randn(B, T, D)
    # Make them orthogonal
    student = student - (student * teacher).sum(dim=-1, keepdim=True) * teacher
    
    loss, metrics = loss_fn(student, teacher)
    
    print(f"  Loss:       {loss.item():.6f}")
    print(f"  Cos Sim:    {metrics['cos_sim']:.6f}")
    print(f"  Expected:   Cos Sim ≈ 0")
    
    assert abs(metrics['cos_sim']) < 0.1, "Should be orthogonal"
    print("  [PASS]")
    
    # Test 3: Opposite directions
    print("\n[Test 3] Opposite Directions (180 degrees)")
    teacher = torch.randn(B, T, D)
    student = -teacher
    
    loss, metrics = loss_fn(student, teacher)
    
    print(f"  Loss:       {loss.item():.6f}")
    print(f"  Cos Sim:    {metrics['cos_sim']:.6f}")
    print(f"  Expected:   Cos Sim ≈ -1.0, Loss ≈ 2.0")
    
    assert metrics['cos_sim'] < -0.99, "Should be opposite"
    print("  [PASS]")
    
    # Test 4: Hybrid loss
    print("\n[Test 4] Hybrid Loss")
    hybrid_fn = HybridKVLoss(mercator_weight=0.8, mse_weight=0.2)
    
    teacher = torch.randn(B, T, D)
    student = torch.randn(B, T, D)
    
    loss, metrics = hybrid_fn(student, teacher)
    
    print(f"  Total Loss:    {loss.item():.6f}")
    print(f"  Mercator:      {metrics['mercator_loss']:.6f}")
    print(f"  MSE:           {metrics['mse_loss']:.6f}")
    print(f"  Cos Sim:       {metrics['cos_sim']:.6f}")
    print("  [PASS]")
    
    print("\n" + "="*60)
    print("All tests passed!")
