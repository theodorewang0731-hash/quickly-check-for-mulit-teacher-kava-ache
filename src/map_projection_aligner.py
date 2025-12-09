"""
MapProjectionAligner: 完整的地图投影对齐器

整合层对齐、时间对齐、结构化投影，统一的 Teacher → Student 对齐接口。

✨ v4.0 更新：
- 添加 mode 参数，支持 "structured"（新方案）/ "flat"（旧 baseline）
- 兼容旧的 KVDimensionProjector 路径，方便 A/B 对比
- 显式处理 Q，支持完整的 Q-K-V 对齐
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
import sys
import os

# 导入我们的模块
from .headwise_projector import HeadwiseMapProjector, create_kv_projectors
from .time_warping import TimeWarper, create_default_warper


class MapProjectionAligner(nn.Module):
    """
    地图投影对齐器：统一的 Teacher → Student 对齐接口
    
    完成三步对齐：
    1. 层对齐：Teacher layers → Student layers (ratio-based mapping)
    2. 时间对齐：T_t → T_s (segment-aware warping)
    3. 结构化投影：(H_t, D_t) → (H_s, D_s) (HeadwiseMapProjector)
    
    ✨ v4.0 新功能：
    支持两种模式：
    - mode="structured"：新的地图投影方案（Anti-Flatten）
    - mode="flat"：旧的 flatten + KVDimensionProjector 方案（baseline）
    
    这样可以直接在 config 里切换做 A/B 对比！
    
    Args:
        teacher_config: Teacher 模型配置
        student_config: Student 模型配置
        mode: "structured" 或 "flat"
        layer_mapping_strategy: 层映射策略 ("ratio", "uniform", "skip")
        time_warper: 可选的自定义 TimeWarper
        share_dim_proj: 是否共享维度投影（仅 structured 模式）
        init_uniform: 是否均匀初始化（仅 structured 模式）
    
    Example:
        >>> # 新方案（地图投影）
        >>> aligner = MapProjectionAligner(
        ...     teacher_cfg, student_cfg, mode="structured"
        ... )
        >>> k_s, v_s, q_s = aligner(k_t, v_t, q_t, segment_ids)
        >>> 
        >>> # 旧方案（flatten baseline）
        >>> aligner_baseline = MapProjectionAligner(
        ...     teacher_cfg, student_cfg, mode="flat"
        ... )
        >>> k_s, v_s, q_s = aligner_baseline(k_t, v_t, q_t, segment_ids)
    """
    
    def __init__(
        self,
        teacher_config,
        student_config,
        mode: str = "structured",
        layer_mapping_strategy: str = "ratio",
        time_warper: Optional[TimeWarper] = None,
        share_dim_proj: bool = True,
        init_uniform: bool = True
    ):
        super().__init__()
        self.t_cfg = teacher_config
        self.s_cfg = student_config
        self.mode = mode  # "structured" or "flat"
        self.layer_mapping_strategy = layer_mapping_strategy
        
        assert mode in ["structured", "flat"], \
            f"mode 必须是 'structured' 或 'flat'，当前: {mode}"
        
        # ===== 共享部分：层映射 =====
        self.layer_mapping = self.build_layer_mapping()
        
        # ===== 共享部分：时间对齐 =====
        self.time_warper = time_warper or create_default_warper()
        
        # ===== 模式分支：初始化投影器 =====
        if mode == "structured":
            # 新方案：HeadwiseMapProjector
            self.proj_k, self.proj_v, self.proj_q = create_kv_projectors(
                teacher_config, student_config,
                share_dim_proj=share_dim_proj,
                init_uniform=init_uniform
            )
        elif mode == "flat":
            # 旧方案：KVDimensionProjector（flatten 路径）
            try:
                from experiments.kv_dimension_projector import KVDimensionProjector
                
                # 计算 flatten 后的维度
                H_t = teacher_config.num_attention_heads
                H_s = student_config.num_attention_heads
                D_t = teacher_config.hidden_size // H_t
                D_s = student_config.hidden_size // H_s
                L_t = teacher_config.num_hidden_layers
                L_s = student_config.num_hidden_layers
                
                flat_dim_t = L_t * H_t * D_t
                flat_dim_s = L_s * H_s * D_s
                
                self.kv_flat_projector = KVDimensionProjector(
                    teacher_dim=flat_dim_t,
                    student_dim=flat_dim_s
                )
                
                print(f"✅ [Flat Mode] 使用 KVDimensionProjector: {flat_dim_t} → {flat_dim_s}")
            except ImportError:
                raise ImportError(
                    "mode='flat' 需要 experiments.kv_dimension_projector.KVDimensionProjector\n"
                    "请确保该模块存在，或使用 mode='structured'"
                )
    
    def build_layer_mapping(self) -> Dict[int, list]:
        """
        构建层映射：Teacher layer → Student layer(s)
        
        策略：
        - "ratio"：比例映射 l_s = round(l_t * L_s / L_t)
        - "uniform"：均匀映射
        - "skip"：跳过某些层
        
        Returns:
            mapping: {student_layer_idx: [teacher_layer_indices]}
        """
        L_t = self.t_cfg.num_hidden_layers
        L_s = self.s_cfg.num_hidden_layers
        
        mapping = {}
        
        if self.layer_mapping_strategy == "ratio":
            # 比例映射
            for l_t in range(L_t):
                l_s = round(l_t * L_s / L_t)
                l_s = min(l_s, L_s - 1)  # 防止越界
                if l_s not in mapping:
                    mapping[l_s] = []
                mapping[l_s].append(l_t)
        
        elif self.layer_mapping_strategy == "uniform":
            # 均匀映射：每个 student 层对应 ceil(L_t/L_s) 个 teacher 层
            step = L_t / L_s
            for l_s in range(L_s):
                start = int(l_s * step)
                end = int((l_s + 1) * step)
                mapping[l_s] = list(range(start, end))
        
        elif self.layer_mapping_strategy == "skip":
            # 跳过映射：只取特定层（可以自定义）
            # 这里简单实现：取均匀分布的 L_s 层
            indices = torch.linspace(0, L_t - 1, L_s).long().tolist()
            for l_s, l_t in enumerate(indices):
                mapping[l_s] = [l_t]
        
        else:
            raise ValueError(f"未知的层映射策略: {self.layer_mapping_strategy}")
        
        return mapping
    
    def _apply_layer_map(
        self,
        x_t: torch.Tensor,
        layer_dim: int = 1
    ) -> torch.Tensor:
        """
        应用层映射：聚合 teacher 层到 student 层
        
        Args:
            x_t: Teacher KV，形状 [B, L_t, H, T, D]
            layer_dim: 层维度的索引（默认 1）
        
        Returns:
            x_s: Student 对齐后的 KV，形状 [B, L_s, H, T, D]
        """
        B, L_t, H, T, D = x_t.shape
        L_s = self.s_cfg.num_hidden_layers
        
        # 创建输出张量
        x_s = torch.zeros(B, L_s, H, T, D, device=x_t.device, dtype=x_t.dtype)
        
        # 按映射聚合
        for l_s, teacher_layers in self.layer_mapping.items():
            if len(teacher_layers) == 1:
                # 1对1映射：直接复制
                x_s[:, l_s] = x_t[:, teacher_layers[0]]
            else:
                # 1对多映射：平均
                x_s[:, l_s] = torch.stack([
                    x_t[:, l_t] for l_t in teacher_layers
                ], dim=0).mean(dim=0)
        
        return x_s
    
    def forward(
        self,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        q_t: torch.Tensor,
        segment_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        完整的对齐流程：Layer → Time → Projection
        
        Args:
            k_t: Teacher Key，形状 [B, L_t, H_t, T_t, D_t]
            v_t: Teacher Value，形状 [B, L_t, H_t, T_t, D_t]
            q_t: Teacher Query，形状 [B, L_t, H_t, T_t, D_t]
            segment_ids: 段标签，形状 [B, T_t]
        
        Returns:
            k_s, v_s, q_s: Student 对齐后的 KVQ，形状 [B, L_s, H_s, T_s, D_s]
        """
        B, L_t, H_t, T_t, D_t = k_t.shape
        T_s = segment_ids.shape[1]  # 假设 segment_ids 已经是目标长度（或自动推断）
        
        # Step 1: 层对齐 [B, L_t, H_t, T_t, D_t] → [B, L_s, H_t, T_t, D_t]
        k_t = self._apply_layer_map(k_t)
        v_t = self._apply_layer_map(v_t)
        q_t = self._apply_layer_map(q_t)
        
        # Step 2: 时间对齐 [B, L_s, H_t, T_t, D_t] → [B, L_s, H_t, T_s, D_t]
        L_s = self.s_cfg.num_hidden_layers
        k_t_warped = []
        v_t_warped = []
        q_t_warped = []
        
        for l in range(L_s):
            # 注意：time_warper 期望 [B, L, H, T, D]，这里传入 [B, 1, H, T, D]
            k_t_warped.append(self.time_warper(k_t[:, l:l+1], segment_ids, T_s))
            v_t_warped.append(self.time_warper(v_t[:, l:l+1], segment_ids, T_s))
            q_t_warped.append(self.time_warper(q_t[:, l:l+1], segment_ids, T_s))
        
        k_t = torch.cat(k_t_warped, dim=1)  # [B, L_s, H_t, T_s, D_t]
        v_t = torch.cat(v_t_warped, dim=1)
        q_t = torch.cat(q_t_warped, dim=1)
        
        # Step 3: 结构化投影（模式分支）
        if self.mode == "structured":
            # 新方案：HeadwiseMapProjector（Anti-Flatten）
            k_s = self.proj_k(k_t)  # [B, L_s, H_s, T_s, D_s]
            v_s = self.proj_v(v_t)
            q_s = self.proj_q(q_t)
        
        elif self.mode == "flat":
            # 旧方案：flatten + KVDimensionProjector
            k_s = self._flatten_and_project(k_t)
            v_s = self._flatten_and_project(v_t)
            q_s = self._flatten_and_project(q_t)
        
        return k_s, v_s, q_s
    
    def _flatten_and_project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Flatten + Project（旧方案路径）
        
        Args:
            x: [B, L, H_t, T, D_t]
        
        Returns:
            x_proj: [B, L_s, H_s, T, D_s]（unflatten 后）
        """
        B, L, H_t, T, D_t = x.shape
        H_s = self.s_cfg.num_attention_heads
        D_s = self.s_cfg.hidden_size // H_s
        L_s = L  # 层数已经对齐了
        
        # Flatten: [B, L, H_t, T, D_t] → [B, T, L*H_t*D_t]
        x_flat = x.permute(0, 3, 1, 2, 4).reshape(B, T, -1)
        
        # Project: [B, T, L*H_t*D_t] → [B, T, L_s*H_s*D_s]
        x_proj_flat = self.kv_flat_projector(x_flat)
        
        # Unflatten: [B, T, L_s*H_s*D_s] → [B, L_s, H_s, T, D_s]
        x_proj = x_proj_flat.reshape(B, T, L_s, H_s, D_s).permute(0, 2, 3, 1, 4)
        
        return x_proj


# ===== 便捷创建函数 =====

def create_structured_aligner(
    teacher_config,
    student_config,
    **kwargs
):
    """
    创建结构化对齐器（新方案）
    """
    return MapProjectionAligner(
        teacher_config, student_config,
        mode="structured",
        **kwargs
    )


def create_flat_aligner(
    teacher_config,
    student_config,
    **kwargs
):
    """
    创建 flatten 对齐器（旧 baseline）
    """
    return MapProjectionAligner(
        teacher_config, student_config,
        mode="flat",
        **kwargs
    )


if __name__ == "__main__":
    # 简单测试
    print("🧪 测试 MapProjectionAligner")
    
    # 模拟配置
    class FakeConfig:
        def __init__(self, num_layers, num_heads, hidden_size):
            self.num_hidden_layers = num_layers
            self.num_attention_heads = num_heads
            self.hidden_size = hidden_size
    
    teacher_cfg = FakeConfig(num_layers=24, num_heads=32, hidden_size=2048)
    student_cfg = FakeConfig(num_layers=12, num_heads=16, hidden_size=1024)
    
    # 创建对齐器
    aligner_structured = create_structured_aligner(teacher_cfg, student_cfg)
    print(f"✅ 创建 structured aligner: mode={aligner_structured.mode}")
    
    # 测试输入
    B, L_t, H_t, T_t, D_t = 2, 24, 32, 100, 64
    k_t = torch.randn(B, L_t, H_t, T_t, D_t)
    v_t = torch.randn(B, L_t, H_t, T_t, D_t)
    q_t = torch.randn(B, L_t, H_t, T_t, D_t)
    
    # Segment IDs
    segment_ids = torch.cat([
        torch.zeros(10, dtype=torch.long),
        torch.ones(80, dtype=torch.long),
        torch.full((10,), 2, dtype=torch.long)
    ]).unsqueeze(0).expand(B, T_t)
    
    print(f"\n输入形状:")
    print(f"  k_t: {k_t.shape}")
    print(f"  v_t: {v_t.shape}")
    print(f"  q_t: {q_t.shape}")
    print(f"  segment_ids: {segment_ids.shape}")
    
    # 前向传播（structured 模式）
    try:
        k_s, v_s, q_s = aligner_structured(k_t, v_t, q_t, segment_ids)
        print(f"\n✅ Structured 模式输出:")
        print(f"  k_s: {k_s.shape}")
        print(f"  v_s: {v_s.shape}")
        print(f"  q_s: {q_s.shape}")
    except Exception as e:
        print(f"\n❌ Structured 模式失败: {e}")
    
    # 测试 flat 模式（如果 KVDimensionProjector 存在）
    try:
        aligner_flat = create_flat_aligner(teacher_cfg, student_cfg)
        print(f"\n✅ 创建 flat aligner: mode={aligner_flat.mode}")
        
        k_s, v_s, q_s = aligner_flat(k_t, v_t, q_t, segment_ids)
        print(f"\n✅ Flat 模式输出:")
        print(f"  k_s: {k_s.shape}")
        print(f"  v_s: {v_s.shape}")
        print(f"  q_s: {q_s.shape}")
    except ImportError:
        print("\n⚠️ Flat 模式需要 KVDimensionProjector，跳过测试")
    except Exception as e:
        print(f"\n❌ Flat 模式失败: {e}")
