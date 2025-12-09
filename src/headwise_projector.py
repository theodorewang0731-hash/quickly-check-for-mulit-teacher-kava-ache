"""
HeadwiseMapProjector: Anti-Flatten 结构化投影

坚持 5D 输入输出 [B, L, H, T, D]，不进行 flatten 操作。
支持共享或 per-head 的维度投影。

✨ v4.0 更新：
- 添加 init_uniform 参数，支持均匀初始化 head_mixer
- 为 baseline 对比保留了清晰的初始化策略
"""
import torch
import torch.nn as nn
from typing import Optional


class HeadwiseMapProjector(nn.Module):
    """
    结构化 KV 投影器：Teacher → Student
    
    输入输出严格保持 5D 形状 [B, L, H, T, D]
    不进行任何 flatten 操作（Anti-Flatten 设计）
    
    Args:
        H_t: Teacher 的注意力头数
        H_s: Student 的注意力头数
        D_t: Teacher 每个头的维度
        D_s: Student 每个头的维度
        share_dim_proj: 是否在所有头之间共享维度投影矩阵
        init_uniform: 是否使用均匀初始化 head_mixer（推荐用于快速收敛）
    
    Example:
        >>> projector = HeadwiseMapProjector(
        ...     H_t=32, H_s=16, D_t=128, D_s=64,
        ...     share_dim_proj=True, init_uniform=True
        ... )
        >>> k_t = torch.randn(2, 12, 32, 512, 128)  # [B, L, H_t, T, D_t]
        >>> k_s = projector(k_t)                     # [B, L, H_s, T, D_s]
    """
    
    def __init__(
        self,
        H_t: int,
        H_s: int,
        D_t: int,
        D_s: int,
        share_dim_proj: bool = True,
        init_uniform: bool = True
    ):
        super().__init__()
        self.H_t = H_t
        self.H_s = H_s
        self.D_t = D_t
        self.D_s = D_s
        self.share_dim_proj = share_dim_proj
        
        # Head 混合器：学习如何将 H_t 个头映射到 H_s 个头
        self.head_mixer = nn.Linear(H_t, H_s, bias=False)
        
        # 维度投影器：D_t → D_s
        if share_dim_proj:
            # 所有头共享同一个投影矩阵（参数少）
            self.dim_proj = nn.Linear(D_t, D_s, bias=False)
        else:
            # 每个 student head 有独立的投影矩阵（表达力强）
            self.dim_proj = nn.ModuleList([
                nn.Linear(D_t, D_s, bias=False) for _ in range(H_s)
            ])
        
        # 可选：均匀初始化 head_mixer
        if init_uniform:
            self.init_uniform_head_mixer()
    
    def init_uniform_head_mixer(self):
        """
        均匀初始化 head_mixer 权重
        
        将 Teacher 的头均匀分配到 Student 的头：
        - Student head 0 → Teacher heads [0, H_t//H_s)
        - Student head 1 → Teacher heads [H_t//H_s, 2*H_t//H_s)
        - ...
        
        这样初始化能提供一个"合理的起点"，避免随机初始化时可能的不稳定。
        """
        with torch.no_grad():
            w = torch.zeros(self.H_s, self.H_t)
            for h_s in range(self.H_s):
                # 计算这个 student head 对应的 teacher heads 区间
                start = int(h_s * self.H_t / self.H_s)
                end = int((h_s + 1) * self.H_t / self.H_s)
                # 均匀权重
                w[h_s, start:end] = 1.0 / max(1, end - start)
            self.head_mixer.weight.copy_(w)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：保持 5D 结构
        
        Args:
            x: Teacher KV，形状 [B, L, H_t, T, D_t]
        
        Returns:
            x_proj: Student KV，形状 [B, L, H_s, T, D_s]
        """
        B, L, H_t, T, D_t = x.shape
        assert H_t == self.H_t and D_t == self.D_t, \
            f"输入形状不匹配：期望 H_t={self.H_t}, D_t={self.D_t}，实际 H_t={H_t}, D_t={D_t}"
        
        # Step 1: Head 混合 [B, L, H_t, T, D_t] → [B, L, H_s, T, D_t]
        # 将 head 维度移到最后，做线性变换，再移回来
        x = x.permute(0, 1, 3, 4, 2)  # [B, L, T, D_t, H_t]
        x = self.head_mixer(x)         # [B, L, T, D_t, H_s]
        x = x.permute(0, 1, 4, 2, 3)  # [B, L, H_s, T, D_t]
        
        # Step 2: 维度投影 [B, L, H_s, T, D_t] → [B, L, H_s, T, D_s]
        if self.share_dim_proj:
            # 共享投影：直接作用在最后一维
            x = self.dim_proj(x)  # [B, L, H_s, T, D_s]
        else:
            # Per-head 投影：每个 student head 独立投影
            outputs = []
            for h in range(self.H_s):
                x_h = x[:, :, h, :, :]  # [B, L, T, D_t]
                x_h_proj = self.dim_proj[h](x_h)  # [B, L, T, D_s]
                outputs.append(x_h_proj.unsqueeze(2))  # [B, L, 1, T, D_s]
            x = torch.cat(outputs, dim=2)  # [B, L, H_s, T, D_s]
        
        return x


# ===== 辅助函数：方便批量创建 =====

def create_kv_projectors(
    teacher_config,
    student_config,
    share_dim_proj: bool = True,
    init_uniform: bool = True
):
    """
    创建 K、V（和可选的 Q）投影器
    
    Args:
        teacher_config: Teacher 模型配置（需要 num_attention_heads, hidden_size）
        student_config: Student 模型配置
        share_dim_proj: 是否共享维度投影
        init_uniform: 是否均匀初始化
    
    Returns:
        proj_k, proj_v, proj_q
    """
    H_t = teacher_config.num_attention_heads
    H_s = student_config.num_attention_heads
    D_t = teacher_config.hidden_size // H_t
    D_s = student_config.hidden_size // H_s
    
    proj_k = HeadwiseMapProjector(H_t, H_s, D_t, D_s, share_dim_proj, init_uniform)
    proj_v = HeadwiseMapProjector(H_t, H_s, D_t, D_s, share_dim_proj, init_uniform)
    proj_q = HeadwiseMapProjector(H_t, H_s, D_t, D_s, share_dim_proj, init_uniform)
    
    return proj_k, proj_v, proj_q


if __name__ == "__main__":
    # 简单测试
    print("🧪 测试 HeadwiseMapProjector")
    
    # 创建投影器
    projector = HeadwiseMapProjector(
        H_t=32, H_s=16, D_t=128, D_s=64,
        share_dim_proj=True, init_uniform=True
    )
    
    # 测试输入
    k_t = torch.randn(2, 12, 32, 512, 128)  # [B=2, L=12, H_t=32, T=512, D_t=128]
    print(f"输入形状: {k_t.shape}")
    
    # 前向传播
    k_s = projector(k_t)
    print(f"输出形状: {k_s.shape}")
    print(f"预期形状: [2, 12, 16, 512, 64]")
    
    # 检查初始化
    print(f"\n✅ head_mixer 权重均值: {projector.head_mixer.weight.mean().item():.4f}")
    print(f"✅ head_mixer 权重标准差: {projector.head_mixer.weight.std().item():.4f}")
    
    # 验证每个 student head 的权重和为 1（均匀初始化）
    row_sums = projector.head_mixer.weight.sum(dim=1)
    print(f"✅ 每行权重和（应该接近1.0）: {row_sums[:5]}")
