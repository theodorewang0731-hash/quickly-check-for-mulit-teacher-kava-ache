"""
Head and Dimension Adapter Module

处理 attention head 和 hidden dimension 不匹配。

Strategies:
1. Hidden dimension: 学习每个教师的线性投影 W_k, W_v ∈ R^{d_t × d_s}
2. Attention head: 
   - 如果 H_t < H_s: 复制/扩展 head
   - 如果 H_t > H_s: 分组聚合 head
3. 1×1 Conv: 可选的轻量级适配器

Usage:
    adapter = HeadDimAdapter(teacher_dim=768, student_dim=512, teacher_heads=12, student_heads=8)
    adapted_k, adapted_v = adapter(teacher_k, teacher_v)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class HeadDimAdapter(nn.Module):
    """
    适配教师和学生的 head/dim 不匹配。
    """
    
    def __init__(
        self,
        teacher_dim: int,
        student_dim: int,
        teacher_heads: int,
        student_heads: int,
        use_conv: bool = False,
        init_identity: bool = True
    ):
        """
        Args:
            teacher_dim: 教师 hidden dimension
            student_dim: 学生 hidden dimension
            teacher_heads: 教师 attention heads
            student_heads: 学生 attention heads
            use_conv: 是否使用 1×1 conv（轻量级）
            init_identity: 是否初始化为接近恒等映射
        """
        super().__init__()
        
        self.teacher_dim = teacher_dim
        self.student_dim = student_dim
        self.teacher_heads = teacher_heads
        self.student_heads = student_heads
        self.use_conv = use_conv
        
        # Hidden dimension adapters
        if use_conv:
            # 1×1 Conv (lightweight)
            self.adapt_k = nn.Conv1d(teacher_dim, student_dim, kernel_size=1)
            self.adapt_v = nn.Conv1d(teacher_dim, student_dim, kernel_size=1)
        else:
            # Linear projection
            self.adapt_k = nn.Linear(teacher_dim, student_dim, bias=False)
            self.adapt_v = nn.Linear(teacher_dim, student_dim, bias=False)
        
        # 初始化为接近恒等映射（如果维度相同）
        if init_identity and teacher_dim == student_dim:
            if use_conv:
                nn.init.eye_(self.adapt_k.weight.squeeze())
                nn.init.eye_(self.adapt_v.weight.squeeze())
            else:
                nn.init.eye_(self.adapt_k.weight)
                nn.init.eye_(self.adapt_v.weight)
        elif init_identity:
            # 如果维度不同，初始化为小随机值
            nn.init.normal_(self.adapt_k.weight if not use_conv else self.adapt_k.weight, std=0.02)
            nn.init.normal_(self.adapt_v.weight if not use_conv else self.adapt_v.weight, std=0.02)
    
    def adapt_heads(
        self,
        tensor: torch.Tensor,
        from_heads: int,
        to_heads: int
    ) -> torch.Tensor:
        """
        适配 attention head 数量。
        
        Args:
            tensor: [batch, time, from_heads, head_dim]
            from_heads: 源 head 数
            to_heads: 目标 head 数
            
        Returns:
            adapted: [batch, time, to_heads, head_dim]
        """
        if from_heads == to_heads:
            return tensor
        
        batch, time, _, head_dim = tensor.shape
        
        if from_heads < to_heads:
            # 扩展 head（复制）
            repeat_factor = to_heads // from_heads
            remainder = to_heads % from_heads
            
            # 复制整倍数
            expanded = tensor.repeat(1, 1, repeat_factor, 1)
            
            # 添加余数
            if remainder > 0:
                expanded = torch.cat([expanded, tensor[:, :, :remainder, :]], dim=2)
            
            return expanded
        
        else:
            # 聚合 head（分组平均）
            group_size = from_heads // to_heads
            
            # Reshape to [batch, time, to_heads, group_size, head_dim]
            grouped = tensor.view(batch, time, to_heads, group_size, head_dim)
            
            # 平均聚合
            aggregated = grouped.mean(dim=3)
            
            return aggregated
    
    def forward(
        self,
        teacher_k: torch.Tensor,
        teacher_v: torch.Tensor,
        adapt_heads: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。
        
        Args:
            teacher_k: [batch, time, teacher_dim] 或 [batch, time, teacher_heads, head_dim]
            teacher_v: 同上
            adapt_heads: 是否适配 head 数量（如果输入包含 head 维度）
            
        Returns:
            adapted_k: [batch, time, student_dim] 或 [batch, time, student_heads, head_dim]
            adapted_v: 同上
        """
        original_shape = teacher_k.shape
        has_heads = len(original_shape) == 4
        
        if has_heads and adapt_heads:
            # [batch, time, teacher_heads, head_dim]
            batch, time, teacher_heads, head_dim = teacher_k.shape
            
            # 先适配 heads
            k_adapted_heads = self.adapt_heads(teacher_k, self.teacher_heads, self.student_heads)
            v_adapted_heads = self.adapt_heads(teacher_v, self.teacher_heads, self.student_heads)
            
            # Reshape 到 [batch, time, dim]
            k_flat = k_adapted_heads.reshape(batch, time, -1)
            v_flat = v_adapted_heads.reshape(batch, time, -1)
            
            # 适配 dim
            if self.use_conv:
                # Conv1d 需要 [batch, dim, time]
                k_flat = k_flat.transpose(1, 2)
                v_flat = v_flat.transpose(1, 2)
                k_adapted = self.adapt_k(k_flat).transpose(1, 2)
                v_adapted = self.adapt_v(v_flat).transpose(1, 2)
            else:
                k_adapted = self.adapt_k(k_flat)
                v_adapted = self.adapt_v(v_flat)
            
            # Reshape 回 [batch, time, student_heads, head_dim]
            student_head_dim = self.student_dim // self.student_heads
            k_adapted = k_adapted.view(batch, time, self.student_heads, student_head_dim)
            v_adapted = v_adapted.view(batch, time, self.student_heads, student_head_dim)
            
        else:
            # [batch, time, teacher_dim]
            if self.use_conv:
                # Conv1d 需要 [batch, dim, time]
                teacher_k = teacher_k.transpose(1, 2)
                teacher_v = teacher_v.transpose(1, 2)
                k_adapted = self.adapt_k(teacher_k).transpose(1, 2)
                v_adapted = self.adapt_v(teacher_v).transpose(1, 2)
            else:
                k_adapted = self.adapt_k(teacher_k)
                v_adapted = self.adapt_v(teacher_v)
        
        return k_adapted, v_adapted


class MultiTeacherHeadDimAdapter(nn.Module):
    """
    多教师的 head/dim 适配器。
    """
    
    def __init__(
        self,
        teacher_configs: list,  # [(dim, heads), ...]
        student_dim: int,
        student_heads: int,
        use_conv: bool = False,
        init_identity: bool = True
    ):
        """
        Args:
            teacher_configs: List of (teacher_dim, teacher_heads) for each teacher
            student_dim: 学生 hidden dimension
            student_heads: 学生 attention heads
            use_conv: 是否使用 1×1 conv
            init_identity: 是否初始化为接近恒等映射
        """
        super().__init__()
        
        self.num_teachers = len(teacher_configs)
        self.student_dim = student_dim
        self.student_heads = student_heads
        
        # 为每个教师创建适配器
        self.adapters = nn.ModuleList([
            HeadDimAdapter(
                teacher_dim=t_dim,
                student_dim=student_dim,
                teacher_heads=t_heads,
                student_heads=student_heads,
                use_conv=use_conv,
                init_identity=init_identity
            )
            for t_dim, t_heads in teacher_configs
        ])
    
    def forward(
        self,
        teacher_ks: list,  # List of teacher K tensors
        teacher_vs: list,  # List of teacher V tensors
        adapt_heads: bool = True
    ) -> Tuple[list, list]:
        """
        前向传播。
        
        Args:
            teacher_ks: List of teacher K tensors
            teacher_vs: List of teacher V tensors
            adapt_heads: 是否适配 head 数量
            
        Returns:
            adapted_ks: List of adapted K tensors
            adapted_vs: List of adapted V tensors
        """
        adapted_ks = []
        adapted_vs = []
        
        for adapter, teacher_k, teacher_v in zip(self.adapters, teacher_ks, teacher_vs):
            k_adapted, v_adapted = adapter(teacher_k, teacher_v, adapt_heads=adapt_heads)
            adapted_ks.append(k_adapted)
            adapted_vs.append(v_adapted)
        
        return adapted_ks, adapted_vs


if __name__ == "__main__":
    # 测试代码
    print("Testing head/dim adapter...")
    
    batch_size = 2
    time_steps = 10
    
    # 测试单一适配器（flat 格式）
    teacher_dim = 768
    student_dim = 512
    teacher_heads = 12
    student_heads = 8
    
    adapter = HeadDimAdapter(teacher_dim, student_dim, teacher_heads, student_heads)
    
    teacher_k = torch.randn(batch_size, time_steps, teacher_dim)
    teacher_v = torch.randn(batch_size, time_steps, teacher_dim)
    
    adapted_k, adapted_v = adapter(teacher_k, teacher_v, adapt_heads=False)
    
    assert adapted_k.shape == (batch_size, time_steps, student_dim)
    assert adapted_v.shape == (batch_size, time_steps, student_dim)
    print("✓ Flat format test passed")
    
    # 测试 head 格式
    head_dim = teacher_dim // teacher_heads
    teacher_k_heads = torch.randn(batch_size, time_steps, teacher_heads, head_dim)
    teacher_v_heads = torch.randn(batch_size, time_steps, teacher_heads, head_dim)
    
    adapted_k_heads, adapted_v_heads = adapter(teacher_k_heads, teacher_v_heads, adapt_heads=True)
    
    student_head_dim = student_dim // student_heads
    assert adapted_k_heads.shape == (batch_size, time_steps, student_heads, student_head_dim)
    assert adapted_v_heads.shape == (batch_size, time_steps, student_heads, student_head_dim)
    print("✓ Head format test passed")
    
    # 测试多教师适配器
    teacher_configs = [
        (768, 12),   # Teacher 1
        (1024, 16),  # Teacher 2
        (512, 8)     # Teacher 3
    ]
    multi_adapter = MultiTeacherHeadDimAdapter(teacher_configs, student_dim, student_heads)
    
    teacher_ks = [torch.randn(batch_size, time_steps, dim) for dim, _ in teacher_configs]
    teacher_vs = [torch.randn(batch_size, time_steps, dim) for dim, _ in teacher_configs]
    
    adapted_ks, adapted_vs = multi_adapter(teacher_ks, teacher_vs, adapt_heads=False)
    
    assert len(adapted_ks) == len(teacher_configs)
    for k in adapted_ks:
        assert k.shape == (batch_size, time_steps, student_dim)
    print("✓ Multi-teacher test passed")
    
    # 测试 Conv1d 版本
    conv_adapter = HeadDimAdapter(teacher_dim, student_dim, teacher_heads, student_heads, use_conv=True)
    adapted_k_conv, adapted_v_conv = conv_adapter(teacher_k, teacher_v, adapt_heads=False)
    assert adapted_k_conv.shape == (batch_size, time_steps, student_dim)
    print("✓ Conv1d test passed")
    
    # 测试 head 聚合/扩展
    # 扩展: 4 -> 8 heads
    tensor_4h = torch.randn(batch_size, time_steps, 4, 64)
    adapter_expand = HeadDimAdapter(256, 512, 4, 8)
    expanded = adapter_expand.adapt_heads(tensor_4h, from_heads=4, to_heads=8)
    assert expanded.shape == (batch_size, time_steps, 8, 64)
    print("✓ Head expansion test passed")
    
    # 聚合: 12 -> 4 heads
    tensor_12h = torch.randn(batch_size, time_steps, 12, 64)
    adapter_agg = HeadDimAdapter(768, 256, 12, 4)
    aggregated = adapter_agg.adapt_heads(tensor_12h, from_heads=12, to_heads=4)
    assert aggregated.shape == (batch_size, time_steps, 4, 64)
    print("✓ Head aggregation test passed")
    
    print("All head/dim adapter tests passed!")
