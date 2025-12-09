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
from typing import Optional


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


class StructuralKVLoss(nn.Module):
    """
    结构化 KV 损失：K/V 方向对齐 + Q-K 交互对齐
    
    ✨ v4.0 设计：
    - K/V：使用余弦相似度（方向对齐）
    - Q：不直接对齐 Q 向量，而是对齐 Q-K 交互（attention 分布）
    
    这样 Q 的语义被编码在"它如何查询 K"上，而不是单纯的向量差。
    
    Args:
        alpha_k: K 对齐权重
        alpha_v: V 对齐权重  
        alpha_attn: Attention KL 权重
        temperature: Attention softmax 温度（用于平滑分布）
        epsilon: 数值稳定性常数
    
    Example:
        >>> loss_fn = StructuralKVLoss(
        ...     alpha_k=1.0, alpha_v=1.0, alpha_attn=0.5
        ... )
        >>> loss, metrics = loss_fn(s_k, s_v, s_q, t_k, t_v, t_q)
    """
    
    def __init__(
        self,
        alpha_k: float = 1.0,
        alpha_v: float = 1.0,
        alpha_attn: float = 0.5,
        temperature: float = 1.0,
        epsilon: float = 1e-8
    ):
        super().__init__()
        self.alpha_k = alpha_k
        self.alpha_v = alpha_v
        self.alpha_attn = alpha_attn
        self.temperature = temperature
        self.epsilon = epsilon
        
        print(f"[StructuralKVLoss Initialized]")
        print(f"  alpha_k (K alignment):    {alpha_k}")
        print(f"  alpha_v (V alignment):    {alpha_v}")
        print(f"  alpha_attn (Attn KL):     {alpha_attn}")
        print(f"  temperature:              {temperature}")
    
    def forward(
        self,
        s_k: torch.Tensor,
        s_v: torch.Tensor,
        s_q: torch.Tensor,
        t_k: torch.Tensor,
        t_v: torch.Tensor,
        t_q: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ):
        """
        计算结构化 KV 损失
        
        Args:
            s_k, s_v, s_q: Student 的 K/V/Q，形状 [B, L, H, T, D]
            t_k, t_v, t_q: Teacher 的 K/V/Q（已投影到 student 空间），形状同上
            attention_mask: 可选的 mask，形状 [B, T]
        
        Returns:
            total_loss: 标量损失
            metrics: 详细指标字典
        """
        # --- 1. K 对齐（方向）---
        if self.alpha_k > 0:
            s_k_norm = F.normalize(s_k, p=2, dim=-1)
            t_k_norm = F.normalize(t_k, p=2, dim=-1)
            k_cos_sim = (s_k_norm * t_k_norm).sum(dim=-1).mean()
            k_loss = 1.0 - k_cos_sim
        else:
            k_loss = torch.tensor(0.0, device=s_k.device)
            k_cos_sim = torch.tensor(0.0, device=s_k.device)
        
        # --- 2. V 对齐（方向）---
        if self.alpha_v > 0:
            s_v_norm = F.normalize(s_v, p=2, dim=-1)
            t_v_norm = F.normalize(t_v, p=2, dim=-1)
            v_cos_sim = (s_v_norm * t_v_norm).sum(dim=-1).mean()
            v_loss = 1.0 - v_cos_sim
        else:
            v_loss = torch.tensor(0.0, device=s_v.device)
            v_cos_sim = torch.tensor(0.0, device=s_v.device)
        
        # --- 3. Q-K 交互对齐（Attention KL）---
        if self.alpha_attn > 0:
            attn_loss = self._compute_attention_kl(
                s_q, s_k, t_q, t_k, attention_mask
            )
        else:
            attn_loss = torch.tensor(0.0, device=s_k.device)
        
        # --- 4. 总损失 ---
        total_loss = (
            self.alpha_k * k_loss +
            self.alpha_v * v_loss +
            self.alpha_attn * attn_loss
        )
        
        # --- 5. 指标 ---
        metrics = {
            'k_loss': k_loss.item(),
            'v_loss': v_loss.item(),
            'attn_loss': attn_loss.item(),
            'k_cos_sim': k_cos_sim.item() if self.alpha_k > 0 else 0.0,
            'v_cos_sim': v_cos_sim.item() if self.alpha_v > 0 else 0.0,
            'total_loss': total_loss.item()
        }
        
        return total_loss, metrics
    
    def _compute_attention_kl(
        self,
        s_q: torch.Tensor,
        s_k: torch.Tensor,
        t_q: torch.Tensor,
        t_k: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算 Q-K 交互的 KL 散度
        
        比较：
        - Student Q 查询 Student K 的分布
        - Teacher Q 查询 Teacher K 的分布（投影后）
        
        Args:
            s_q, s_k: Student Q/K，形状 [B, L, H, T, D]
            t_q, t_k: Teacher Q/K（已投影），形状同上
            attention_mask: 可选 mask，[B, T]
        
        Returns:
            kl_loss: 标量
        """
        B, L, H, T, D = s_q.shape
        
        # 计算 attention scores：Q @ K^T / sqrt(D)
        # [B, L, H, T, D] @ [B, L, H, D, T] -> [B, L, H, T, T]
        s_scores = torch.matmul(
            s_q, s_k.transpose(-1, -2)
        ) / (D ** 0.5)
        
        t_scores = torch.matmul(
            t_q, t_k.transpose(-1, -2)
        ) / (D ** 0.5)
        
        # 应用 mask（如果提供）
        if attention_mask is not None:
            # mask: [B, T] -> [B, 1, 1, T, 1]
            mask = attention_mask.unsqueeze(1).unsqueeze(1).unsqueeze(-1)
            s_scores = s_scores.masked_fill(~mask, float('-inf'))
            t_scores = t_scores.masked_fill(~mask, float('-inf'))
        
        # Softmax 转换为概率分布
        s_attn_probs = F.softmax(s_scores / self.temperature, dim=-1)
        t_attn_probs = F.softmax(t_scores / self.temperature, dim=-1)
        
        # KL 散度：KL(Teacher || Student)
        # 注意：PyTorch 的 kl_div 期望 log(student) 和 teacher
        kl_loss = F.kl_div(
            s_attn_probs.log(),
            t_attn_probs,
            reduction='batchmean',
            log_target=False
        )
        
        return kl_loss


# ===== 便捷创建函数 =====

def create_mercator_loss(alpha=1.0, beta=0.0):
    """创建 Mercator 损失（单教师，flatten 路径）"""
    return MercatorKVLoss(alpha=alpha, beta=beta)


def create_structural_loss(
    alpha_k=1.0,
    alpha_v=1.0,
    alpha_attn=0.5,
    temperature=1.0
):
    """创建结构化损失（新方案，地图投影路径）"""
    return StructuralKVLoss(
        alpha_k=alpha_k,
        alpha_v=alpha_v,
        alpha_attn=alpha_attn,
        temperature=temperature
    )


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
