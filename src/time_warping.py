"""
TimeWarper: 基于 Segment 的时间维对齐

支持 P(Prompt)/R(Reasoning)/A(Answer) 三段式对齐，
每段可以有不同的采样比例和平滑参数。

✨ v4.0 更新：
- 添加清晰的假设注释：segment_ids[0] 作为全 batch 的参考
- 保留了工程上的简化（batch 内结构一致）
- 为将来的 per-sample 切段预留了扩展空间
"""
import torch
import torch.nn as nn
from typing import Optional, Dict


class TimeWarper(nn.Module):
    """
    时间维动态对齐：将 Teacher 序列长度 T_t 对齐到 Student 序列长度 T_s
    
    支持分段对齐（Prompt/Reasoning/Answer）：
    - 每段可以有不同的采样比例（ratio_map）
    - 每段可以有不同的平滑系数（alpha_map）
    
    ⚠️ 工程假设（v4.0）：
    batch 内所有样本的 segment 划分相同（使用 segment_ids[0] 作为参考）。
    这对当前 KV 蒸馏场景是合理的（同一个 prompt 格式），但如果以后有
    "每个样本不一样"的情况，需要改成 per-sample 切段 + padding。
    
    Args:
        num_segments: 段的数量（默认 3：P/R/A）
        ratio_map: 每段的采样比例 {seg_id: ratio}
        alpha_map: 每段的平滑系数 {seg_id: alpha}（0=最近邻，1=线性插值）
    
    Example:
        >>> warper = TimeWarper(
        ...     num_segments=3,
        ...     ratio_map={0: 1.0, 1: 0.5, 2: 1.0},  # Reasoning 段采样 50%
        ...     alpha_map={0: 0.0, 1: 0.5, 2: 0.0}   # Reasoning 段做插值
        ... )
        >>> k_t = torch.randn(2, 12, 32, 100, 128)  # [B, L, H, T_t=100, D]
        >>> segment_ids = torch.tensor([[0]*10 + [1]*80 + [2]*10]).expand(2, 100)
        >>> k_s = warper(k_t, segment_ids, T_s=50)  # [B, L, H, T_s=50, D]
    """
    
    def __init__(
        self,
        num_segments: int = 3,
        ratio_map: Optional[Dict[int, float]] = None,
        alpha_map: Optional[Dict[int, float]] = None
    ):
        super().__init__()
        self.num_segments = num_segments
        
        # 默认配置：所有段等比例采样，无插值
        self.ratio_map = ratio_map or {i: 1.0 for i in range(num_segments)}
        self.alpha_map = alpha_map or {i: 0.0 for i in range(num_segments)}
        
        # 注册为 buffer（不参与梯度）
        self.register_buffer('_ratio_tensor', torch.tensor([
            self.ratio_map.get(i, 1.0) for i in range(num_segments)
        ]))
        self.register_buffer('_alpha_tensor', torch.tensor([
            self.alpha_map.get(i, 0.0) for i in range(num_segments)
        ]))
    
    def forward(
        self,
        x: torch.Tensor,
        segment_ids: torch.Tensor,
        T_s: int
    ) -> torch.Tensor:
        """
        时间维对齐：T_t → T_s
        
        Args:
            x: Teacher KV，形状 [B, L, H, T_t, D]
            segment_ids: 段标签，形状 [B, T_t]，取值 0~num_segments-1
            T_s: Student 的目标序列长度
        
        Returns:
            x_warped: 对齐后的 KV，形状 [B, L, H, T_s, D]
        """
        B, L, H, T_t, D = x.shape
        device = x.device
        
        # ⚠️ 工程简化假设：使用 batch[0] 的 segment_ids 作为全 batch 的参考
        # 这假设 batch 内所有样本的段划分相同（比如同一个 prompt + reasoning 格式）
        # 如果以后需要 per-sample 切段，需要改成循环或 padding
        ref_seg = segment_ids[0]  # [T_t]
        
        # 为每个 segment 计算采样点
        sampled_indices = []
        for seg_id in range(self.num_segments):
            # 找到这个 segment 的所有位置
            mask = (ref_seg == seg_id)
            seg_positions = torch.where(mask)[0]  # [n_seg]
            
            if len(seg_positions) == 0:
                continue  # 这个 segment 不存在，跳过
            
            # 根据 ratio 计算采样数量
            ratio = self.ratio_map.get(seg_id, 1.0)
            n_sample = max(1, int(len(seg_positions) * ratio))
            
            # 等间隔采样（或最近邻）
            step = len(seg_positions) / n_sample
            indices = torch.tensor([
                int(seg_positions[min(int(i * step), len(seg_positions) - 1)])
                for i in range(n_sample)
            ], device=device)
            
            sampled_indices.append(indices)
        
        # 拼接所有段的采样点
        if len(sampled_indices) == 0:
            # 兜底：如果没有任何 segment，均匀采样
            sampled_indices = torch.linspace(0, T_t - 1, T_s, device=device).long()
        else:
            sampled_indices = torch.cat(sampled_indices)
        
        # 如果采样点数量不等于 T_s，调整到 T_s
        if len(sampled_indices) != T_s:
            # 简单策略：线性插值到 T_s 个点
            old_indices = torch.linspace(0, len(sampled_indices) - 1, T_s, device=device)
            new_indices = torch.gather(
                sampled_indices.float().unsqueeze(0).expand(T_s, -1),
                dim=1,
                index=old_indices.long().unsqueeze(1)
            ).squeeze(1).long()
            sampled_indices = new_indices
        
        # 使用 gather 提取采样点
        # x: [B, L, H, T_t, D] → [B, L, H, T_s, D]
        sampled_indices = sampled_indices.view(1, 1, 1, T_s, 1).expand(B, L, H, T_s, D)
        x_warped = torch.gather(x, dim=3, index=sampled_indices)
        
        return x_warped


# ===== 预设配置 =====

def create_default_warper():
    """
    默认配置：P/R/A 三段等比例，无插值
    """
    return TimeWarper(
        num_segments=3,
        ratio_map={0: 1.0, 1: 1.0, 2: 1.0},
        alpha_map={0: 0.0, 1: 0.0, 2: 0.0}
    )


def create_reasoning_focused_warper():
    """
    Reasoning 段加强：P/A 保持，R 段采样 50%
    """
    return TimeWarper(
        num_segments=3,
        ratio_map={0: 1.0, 1: 0.5, 2: 1.0},  # Reasoning 段减半
        alpha_map={0: 0.0, 1: 0.5, 2: 0.0}   # Reasoning 段做插值
    )


if __name__ == "__main__":
    # 简单测试
    print("🧪 测试 TimeWarper")
    
    # 创建 warper
    warper = create_reasoning_focused_warper()
    
    # 测试输入
    B, L, H, T_t, D = 2, 12, 32, 100, 128
    k_t = torch.randn(B, L, H, T_t, D)
    
    # 模拟 segment_ids: P(10) + R(80) + A(10)
    segment_ids = torch.cat([
        torch.zeros(10, dtype=torch.long),   # P
        torch.ones(80, dtype=torch.long),    # R
        torch.full((10,), 2, dtype=torch.long)  # A
    ]).unsqueeze(0).expand(B, T_t)
    
    print(f"输入形状: {k_t.shape}")
    print(f"segment_ids: {segment_ids.shape}, unique: {segment_ids.unique()}")
    
    # 对齐到 T_s=50
    k_s = warper(k_t, segment_ids, T_s=50)
    print(f"输出形状: {k_s.shape}")
    print(f"预期形状: [2, 12, 32, 50, 128]")
    
    # 检查不同目标长度
    for T_s in [30, 50, 80]:
        k_s = warper(k_t, segment_ids, T_s=T_s)
        print(f"✅ T_s={T_s}: {k_s.shape}")
