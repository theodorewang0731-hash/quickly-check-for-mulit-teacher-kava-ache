"""
RoPE Scaling Module

处理 RoPE (Rotary Position Embedding) 的位置编码不匹配。

当教师和学生使用不同的 RoPE 频率或训练长度时，需要重新缩放。

Strategies:
1. Linear scaling: 简单的线性插值
2. NTK-aware scaling: 保持高频信息，适用于长上下文
3. Dynamic scaling: 根据实际序列长度动态调整

Reference: 
- RoPE: https://arxiv.org/abs/2104.09864
- NTK-aware: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/

Usage:
    scaler = RoPEScaler(base=10000, teacher_max_len=2048, student_max_len=4096)
    scaled_k = scaler.scale_key(teacher_k, seq_len=3000)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class RoPEScaler:
    """
    RoPE 位置编码缩放器。
    """
    
    def __init__(
        self,
        base: float = 10000.0,
        teacher_max_len: int = 2048,
        student_max_len: int = 2048,
        scaling_method: str = "ntk",
        scaling_factor: Optional[float] = None
    ):
        """
        Args:
            base: RoPE 基础频率
            teacher_max_len: 教师训练的最大长度
            student_max_len: 学生训练的最大长度
            scaling_method: 缩放方法 ("linear", "ntk", "dynamic")
            scaling_factor: 缩放因子（可选，自动计算）
        """
        self.base = base
        self.teacher_max_len = teacher_max_len
        self.student_max_len = student_max_len
        self.scaling_method = scaling_method
        
        if scaling_factor is None:
            scaling_factor = student_max_len / teacher_max_len
        
        self.scaling_factor = scaling_factor
        
        # NTK-aware scaling: 计算新的 base
        if scaling_method == "ntk":
            self.scaled_base = base * (scaling_factor ** (2 / 3))
        else:
            self.scaled_base = base
    
    def compute_rope_freqs(
        self,
        dim: int,
        seq_len: int,
        base: Optional[float] = None
    ) -> torch.Tensor:
        """
        计算 RoPE 频率。
        
        Args:
            dim: 嵌入维度
            seq_len: 序列长度
            base: 基础频率（可选）
            
        Returns:
            freqs: [seq_len, dim // 2] 频率张量
        """
        if base is None:
            base = self.base
        
        # 计算逆频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        
        # 位置索引
        t = torch.arange(seq_len, dtype=torch.float32)
        
        # 频率矩阵 [seq_len, dim // 2]
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        
        return freqs
    
    def apply_rotary_emb(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor
    ) -> torch.Tensor:
        """
        应用 RoPE 编码。
        
        Args:
            x: [batch, seq_len, ..., dim] 输入张量
            freqs: [seq_len, dim // 2] 频率张量
            
        Returns:
            x_rotated: RoPE 编码后的张量
        """
        # 确保最后一维是 dim
        *batch_dims, seq_len, dim = x.shape
        
        # 分离奇偶维度
        x_even = x[..., ::2]  # [..., seq_len, dim // 2]
        x_odd = x[..., 1::2]  # [..., seq_len, dim // 2]
        
        # 扩展 freqs 到 batch 维度
        cos_freqs = freqs.cos()
        sin_freqs = freqs.sin()
        
        # Broadcast to batch dims
        for _ in batch_dims:
            cos_freqs = cos_freqs.unsqueeze(0)
            sin_freqs = sin_freqs.unsqueeze(0)
        
        # 旋转
        x_even_rot = x_even * cos_freqs - x_odd * sin_freqs
        x_odd_rot = x_even * sin_freqs + x_odd * cos_freqs
        
        # 交错合并
        x_rotated = torch.stack([x_even_rot, x_odd_rot], dim=-1).flatten(-2)
        
        return x_rotated
    
    def scale_key(
        self,
        teacher_k: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> torch.Tensor:
        """
        缩放教师的 key（RoPE 编码）。
        
        Args:
            teacher_k: [batch, seq_len, dim] 教师 key
            seq_len: 目标序列长度（可选，默认使用 student_max_len）
            
        Returns:
            scaled_k: 缩放后的 key
        """
        if seq_len is None:
            seq_len = teacher_k.size(1)
        
        dim = teacher_k.size(-1)
        
        if self.scaling_method == "linear":
            # 简单线性缩放（通过位置索引）
            teacher_freqs = self.compute_rope_freqs(dim, self.teacher_max_len, self.base)
            student_freqs = self.compute_rope_freqs(dim, seq_len, self.base)
            
            # 线性插值
            scale = seq_len / self.teacher_max_len
            scaled_freqs = teacher_freqs[:seq_len] * scale
            
        elif self.scaling_method == "ntk":
            # NTK-aware scaling
            scaled_freqs = self.compute_rope_freqs(dim, seq_len, self.scaled_base)
        
        elif self.scaling_method == "dynamic":
            # 动态缩放（根据实际序列长度）
            current_scale = max(1.0, seq_len / self.teacher_max_len)
            dynamic_base = self.base * (current_scale ** (2 / 3))
            scaled_freqs = self.compute_rope_freqs(dim, seq_len, dynamic_base)
        
        else:
            # 不缩放
            scaled_freqs = self.compute_rope_freqs(dim, seq_len, self.base)
        
        # 应用缩放后的 RoPE
        scaled_k = self.apply_rotary_emb(teacher_k, scaled_freqs.to(teacher_k.device))
        
        return scaled_k
    
    def scale_kv_pair(
        self,
        teacher_k: torch.Tensor,
        teacher_v: torch.Tensor,
        seq_len: Optional[int] = None
    ) -> tuple:
        """
        缩放 KV 对（只缩放 K，V 不需要 RoPE）。
        
        Args:
            teacher_k: [batch, seq_len, dim] 教师 key
            teacher_v: [batch, seq_len, dim] 教师 value
            seq_len: 目标序列长度（可选）
            
        Returns:
            scaled_k: 缩放后的 key
            scaled_v: 原始的 value（V 不需要 RoPE）
        """
        scaled_k = self.scale_key(teacher_k, seq_len)
        
        # V 不需要 RoPE 编码，直接返回
        scaled_v = teacher_v
        
        return scaled_k, scaled_v


def interpolate_rope_embeddings(
    embeddings: torch.Tensor,
    target_length: int,
    method: str = "linear"
) -> torch.Tensor:
    """
    插值 RoPE 嵌入到目标长度。
    
    Args:
        embeddings: [seq_len, dim] RoPE 嵌入
        target_length: 目标长度
        method: 插值方法 ("linear", "cubic")
        
    Returns:
        interpolated: [target_length, dim] 插值后的嵌入
    """
    if embeddings.size(0) == target_length:
        return embeddings
    
    # 使用 F.interpolate (需要 [batch, channels, length] 格式)
    embeddings = embeddings.unsqueeze(0).transpose(1, 2)  # [1, dim, seq_len]
    
    interpolated = F.interpolate(
        embeddings,
        size=target_length,
        mode=method,
        align_corners=False if method == "linear" else None
    )
    
    interpolated = interpolated.transpose(1, 2).squeeze(0)  # [target_length, dim]
    
    return interpolated


class MultiTeacherRoPEScaler:
    """
    多教师的 RoPE 缩放器。
    """
    
    def __init__(
        self,
        teacher_configs: list,  # [(base, max_len), ...]
        student_max_len: int,
        scaling_method: str = "ntk"
    ):
        """
        Args:
            teacher_configs: List of (base, max_len) for each teacher
            student_max_len: 学生最大长度
            scaling_method: 缩放方法
        """
        self.scalers = [
            RoPEScaler(
                base=base,
                teacher_max_len=max_len,
                student_max_len=student_max_len,
                scaling_method=scaling_method
            )
            for base, max_len in teacher_configs
        ]
    
    def scale_keys(
        self,
        teacher_ks: list,
        seq_len: Optional[int] = None
    ) -> list:
        """
        缩放多个教师的 keys。
        
        Args:
            teacher_ks: List of teacher K tensors
            seq_len: 目标序列长度
            
        Returns:
            scaled_ks: List of scaled K tensors
        """
        return [
            scaler.scale_key(k, seq_len)
            for scaler, k in zip(self.scalers, teacher_ks)
        ]
    
    def scale_kv_pairs(
        self,
        teacher_ks: list,
        teacher_vs: list,
        seq_len: Optional[int] = None
    ) -> tuple:
        """
        缩放多个教师的 KV 对。
        
        Args:
            teacher_ks: List of teacher K tensors
            teacher_vs: List of teacher V tensors
            seq_len: 目标序列长度
            
        Returns:
            scaled_ks: List of scaled K tensors
            scaled_vs: List of V tensors
        """
        scaled_ks = []
        scaled_vs = []
        
        for scaler, k, v in zip(self.scalers, teacher_ks, teacher_vs):
            scaled_k, scaled_v = scaler.scale_kv_pair(k, v, seq_len)
            scaled_ks.append(scaled_k)
            scaled_vs.append(scaled_v)
        
        return scaled_ks, scaled_vs


if __name__ == "__main__":
    # 测试代码
    print("Testing RoPE scaling...")
    
    batch_size = 2
    seq_len = 512
    dim = 64
    
    # 测试单一缩放器
    scaler = RoPEScaler(
        base=10000.0,
        teacher_max_len=2048,
        student_max_len=4096,
        scaling_method="ntk"
    )
    
    teacher_k = torch.randn(batch_size, seq_len, dim)
    teacher_v = torch.randn(batch_size, seq_len, dim)
    
    # 测试 key 缩放
    scaled_k = scaler.scale_key(teacher_k)
    assert scaled_k.shape == teacher_k.shape
    print("✓ Key scaling test passed")
    
    # 测试 KV 对缩放
    scaled_k, scaled_v = scaler.scale_kv_pair(teacher_k, teacher_v)
    assert scaled_k.shape == teacher_k.shape
    assert scaled_v.shape == teacher_v.shape
    print("✓ KV pair scaling test passed")
    
    # 测试不同缩放方法
    for method in ["linear", "ntk", "dynamic"]:
        scaler_method = RoPEScaler(
            base=10000.0,
            teacher_max_len=2048,
            student_max_len=4096,
            scaling_method=method
        )
        scaled = scaler_method.scale_key(teacher_k)
        assert scaled.shape == teacher_k.shape
        print(f"✓ {method} scaling test passed")
    
    # 测试频率计算
    freqs = scaler.compute_rope_freqs(dim, seq_len)
    assert freqs.shape == (seq_len, dim // 2)
    print("✓ Frequency computation test passed")
    
    # 测试 RoPE 应用
    x = torch.randn(batch_size, seq_len, dim)
    freqs = scaler.compute_rope_freqs(dim, seq_len)
    x_rotated = scaler.apply_rotary_emb(x, freqs)
    assert x_rotated.shape == x.shape
    print("✓ RoPE application test passed")
    
    # 测试插值
    embeddings = torch.randn(seq_len, dim)
    interpolated = interpolate_rope_embeddings(embeddings, target_length=1024)
    assert interpolated.shape == (1024, dim)
    print("✓ Interpolation test passed")
    
    # 测试多教师缩放
    teacher_configs = [
        (10000.0, 2048),
        (10000.0, 4096),
        (10000.0, 1024)
    ]
    multi_scaler = MultiTeacherRoPEScaler(teacher_configs, student_max_len=4096)
    
    teacher_ks = [torch.randn(batch_size, seq_len, dim) for _ in teacher_configs]
    teacher_vs = [torch.randn(batch_size, seq_len, dim) for _ in teacher_configs]
    
    scaled_ks, scaled_vs = multi_scaler.scale_kv_pairs(teacher_ks, teacher_vs)
    
    assert len(scaled_ks) == len(teacher_configs)
    for k in scaled_ks:
        assert k.shape == (batch_size, seq_len, dim)
    print("✓ Multi-teacher RoPE scaling test passed")
    
    print("All RoPE scaling tests passed!")
