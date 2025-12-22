"""
KV Head Projector - 解决 GQA/MQA 头数不匹配问题
=================================================

核心问题：
1. Teacher KV heads ≠ Student KV heads (例如 12 vs 2)
2. Head dimension 不匹配 (例如 128 vs 64)

解决方案：
1. 从张量 shape 动态获取实际的 KV head 数 (num_key_value_heads)
2. 先投影 head_dim: [Dt -> Ds]
3. 再混合 head 数: [Ht -> Hs] 通过可学习的线性层

作者: GitHub Copilot
日期: 2025-12-16
"""

import torch
import torch.nn as nn
from typing import Tuple


class KVProjector(nn.Module):
    """
    KV Cache 头数 + 维度投影器
    
    处理两个维度的对齐:
    1. head_dim: Dt -> Ds (per-head, per-token)
    2. num_heads: Ht -> Hs (head-mixing)
    """
    
    def __init__(self, Ht: int, Hs: int, Dt: int, Ds: int, share_kv: bool = True):
        """
        Args:
            Ht: Teacher KV head 数
            Hs: Student KV head 数
            Dt: Teacher head dimension
            Ds: Student head dimension
            share_kv: K 和 V 是否共享投影器 (默认 True)
        """
        super().__init__()
        self.Ht, self.Hs, self.Dt, self.Ds = Ht, Hs, Dt, Ds
        
        # Step 1: head_dim 投影 (如果需要)
        self.dim_proj_k = nn.Linear(Dt, Ds, bias=False) if Dt != Ds else None
        self.dim_proj_v = self.dim_proj_k if (share_kv and Dt != Ds) else (
            nn.Linear(Dt, Ds, bias=False) if Dt != Ds else None
        )
        
        # Step 2: head 混合 (如果需要)
        # 在 head 维度上做线性变换: [Ht] -> [Hs]
        self.head_proj = nn.Linear(Ht, Hs, bias=False) if Ht != Hs else None
        
        # 初始化策略
        self._init_weights()
    
    def _init_weights(self):
        """初始化投影层权重"""
        # head_dim 投影: 小随机初始化
        if self.dim_proj_k is not None:
            nn.init.normal_(self.dim_proj_k.weight, mean=0.0, std=0.02)
        if self.dim_proj_v is not None and self.dim_proj_v != self.dim_proj_k:
            nn.init.normal_(self.dim_proj_v.weight, mean=0.0, std=0.02)
        
        # head 混合: 如果能整除,初始化为分组平均 (更稳定)
        if self.head_proj is not None:
            if self.Ht % self.Hs == 0:
                # 例如: 12 -> 2, 每组 6 个头平均
                group_size = self.Ht // self.Hs
                weight = torch.zeros(self.Hs, self.Ht)
                for i in range(self.Hs):
                    weight[i, i*group_size:(i+1)*group_size] = 1.0 / group_size
                self.head_proj.weight.data = weight
            else:
                # 否则: 小随机初始化
                nn.init.normal_(self.head_proj.weight, mean=0.0, std=0.02)
    
    def _proj_dim(self, x: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        """
        投影 head_dim 维度
        
        Args:
            x: [B, H, T, Dt]
            proj: Linear(Dt -> Ds)
        
        Returns:
            [B, H, T, Ds]
        """
        if proj is None:
            return x
        
        B, H, T, Dt = x.shape
        # Reshape to [B*H*T, Dt]
        x_flat = x.reshape(-1, Dt)
        # Apply projection
        y_flat = proj(x_flat)  # [B*H*T, Ds]
        # Reshape back
        y = y_flat.reshape(B, H, T, -1)
        return y
    
    def _proj_head(self, x: torch.Tensor) -> torch.Tensor:
        """
        混合 head 维度
        
        Args:
            x: [B, Ht, T, D]
        
        Returns:
            [B, Hs, T, D]
        """
        if self.head_proj is None:
            return x
        
        B, Ht, T, D = x.shape
        # 将 head 维移到最后: [B, T, D, Ht]
        x_perm = x.permute(0, 2, 3, 1).contiguous()
        # 应用线性变换: [B, T, D, Ht] -> [B, T, D, Hs]
        y_perm = self.head_proj(x_perm)
        # 移回: [B, Hs, T, D]
        y = y_perm.permute(0, 3, 1, 2).contiguous()
        return y
    
    def forward(self, k_t: torch.Tensor, v_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向投影
        
        Args:
            k_t: Teacher K, [B, Ht, T, Dt]
            v_t: Teacher V, [B, Ht, T, Dt]
        
        Returns:
            k_hat: [B, Hs, T, Ds]
            v_hat: [B, Hs, T, Ds]
        """
        # Step 1: 投影 head_dim
        k = self._proj_dim(k_t, self.dim_proj_k)  # [B, Ht, T, Ds]
        v = self._proj_dim(v_t, self.dim_proj_v)  # [B, Ht, T, Ds]
        
        # Step 2: 混合 head 数
        k = self._proj_head(k)  # [B, Hs, T, Ds]
        v = self._proj_head(v)  # [B, Hs, T, Ds]
        
        return k, v


def safe_time_resample(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    安全的时间维重采样 (避免越界)
    
    Args:
        x: [B, H, T_in, D]
        indices: [B, T_out] 或 [T_out], 每个值在 [0, T_in-1] 范围内
    
    Returns:
        [B, H, T_out, D]
    """
    B, H, T_in, D = x.shape
    device = x.device
    
    # 确保 indices 在正确设备上
    indices = indices.to(device=device)
    
    # 转换为 long 类型
    indices = indices.long()
    
    # Clamp 到合法范围
    indices = indices.clamp(0, T_in - 1)
    
    # 处理 indices 的 shape
    if indices.dim() == 1:
        # [T_out] -> [B, T_out]
        indices = indices.unsqueeze(0).expand(B, -1)
    
    T_out = indices.shape[1]
    
    # 扩展 indices 用于 gather: [B, H, T_out, D]
    idx = indices[:, None, :, None].expand(B, H, T_out, D)
    
    # Gather
    return torch.gather(x, dim=2, index=idx)


def build_linear_indices(B: int, T_in: int, T_out: int, device: torch.device) -> torch.Tensor:
    """
    生成线性插值的索引 (避免天然越界)
    
    Args:
        B: batch size
        T_in: 输入序列长度
        T_out: 输出序列长度
        device: torch device
    
    Returns:
        indices: [B, T_out], dtype=long
    """
    if T_out == 1:
        # 边界情况: 只采样中间位置
        mid = T_in // 2
        return torch.full((B, 1), mid, device=device, dtype=torch.long)
    
    # 浮点生成, 再 round, 再 clamp
    base = torch.linspace(0, T_in - 1, steps=T_out, device=device)  # [T_out]
    idx = torch.round(base).long().clamp(0, T_in - 1)  # [T_out]
    
    # 扩展到 batch
    return idx.unsqueeze(0).expand(B, -1)  # [B, T_out]


def get_kv_heads_from_tensor(kv_tensor: torch.Tensor) -> int:
    """
    从 KV 张量动态获取头数 (避免用错 num_attention_heads)
    
    Args:
        kv_tensor: [B, H, T, D] 或 [B, T, H, D]
    
    Returns:
        H: KV head 数
    """
    # 假设 shape 为 [B, H, T, D] (最常见)
    # 如果是 [B, T, H, D], 需要根据实际情况判断
    if kv_tensor.dim() == 4:
        # 通常 H < T, D 较大
        # 简单启发: 第二维如果小于第三维,则为 head
        if kv_tensor.shape[1] < kv_tensor.shape[2]:
            return kv_tensor.shape[1]
        else:
            return kv_tensor.shape[2]
    else:
        raise ValueError(f"Expected 4D KV tensor, got shape {kv_tensor.shape}")


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("KV Head Projector - 测试")
    print("=" * 80)
    
    # 场景 1: GQA 头数不匹配 (12 -> 2)
    print("\n[测试 1] GQA 头数不匹配: Ht=12, Hs=2, Dt=Ds=128")
    Ht, Hs = 12, 2
    Dt, Ds = 128, 128
    B, T = 4, 50
    
    projector = KVProjector(Ht, Hs, Dt, Ds)
    
    k_teacher = torch.randn(B, Ht, T, Dt)
    v_teacher = torch.randn(B, Ht, T, Dt)
    
    k_hat, v_hat = projector(k_teacher, v_teacher)
    
    print(f"  Input:  K {k_teacher.shape}, V {v_teacher.shape}")
    print(f"  Output: K {k_hat.shape}, V {v_hat.shape}")
    assert k_hat.shape == (B, Hs, T, Ds)
    assert v_hat.shape == (B, Hs, T, Ds)
    print("  ✓ Shape correct!")
    
    # 场景 2: 头数 + head_dim 同时不匹配
    print("\n[测试 2] 头数 + head_dim 不匹配: Ht=28, Hs=2, Dt=128, Ds=64")
    Ht, Hs = 28, 2
    Dt, Ds = 128, 64
    
    projector2 = KVProjector(Ht, Hs, Dt, Ds)
    
    k_teacher2 = torch.randn(B, Ht, T, Dt)
    v_teacher2 = torch.randn(B, Ht, T, Dt)
    
    k_hat2, v_hat2 = projector2(k_teacher2, v_teacher2)
    
    print(f"  Input:  K {k_teacher2.shape}, V {v_teacher2.shape}")
    print(f"  Output: K {k_hat2.shape}, V {v_hat2.shape}")
    assert k_hat2.shape == (B, Hs, T, Ds)
    print("  ✓ Shape correct!")
    
    # 场景 3: 时间重采样
    print("\n[测试 3] 安全时间重采样: T_in=80 -> T_out=50")
    x = torch.randn(B, Hs, 80, Ds)
    indices = build_linear_indices(B, 80, 50, x.device)
    
    x_resampled = safe_time_resample(x, indices)
    
    print(f"  Input:  {x.shape}")
    print(f"  Indices: {indices.shape}")
    print(f"  Output: {x_resampled.shape}")
    assert x_resampled.shape == (B, Hs, 50, Ds)
    print("  ✓ Resampling correct!")
    
    # 场景 4: 边界情况 (T_in=1, T_out=1)
    print("\n[测试 4] 边界情况: T_in=1, T_out=1")
    x_edge = torch.randn(B, Hs, 1, Ds)
    indices_edge = build_linear_indices(B, 1, 1, x_edge.device)
    
    x_resampled_edge = safe_time_resample(x_edge, indices_edge)
    
    print(f"  Input:  {x_edge.shape}")
    print(f"  Output: {x_resampled_edge.shape}")
    assert x_resampled_edge.shape == (B, Hs, 1, Ds)
    print("  ✓ Edge case correct!")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)
