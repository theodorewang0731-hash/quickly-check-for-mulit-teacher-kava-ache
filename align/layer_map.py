"""
Layer Mapping Module

处理层映射：ratio-based mapping + 插值。
将教师的 N_t 层映射到学生的 N_s 层。

Strategies:
1. Ratio mapping: l_s = round(l_t * L_s / L_t)
2. Interpolation: 当多个教师层映射到同一学生层时，加权平均
3. Skip mapping: 为超深教师提供跳跃映射

Usage:
    layer_map = build_layer_mapping(num_teacher_layers=32, num_student_layers=12)
    student_kv = interpolate_teacher_layers(teacher_kvs, layer_map)
"""
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


def build_layer_mapping(
    num_teacher_layers: int,
    num_student_layers: int,
    strategy: str = "ratio"
) -> Dict[int, List[Tuple[int, float]]]:
    """
    构建教师层到学生层的映射。
    
    Args:
        num_teacher_layers: 教师层数
        num_student_layers: 学生层数
        strategy: 映射策略 ("ratio", "uniform", "skip")
        
    Returns:
        layer_map: {student_layer: [(teacher_layer, weight), ...]}
            例如 {0: [(0, 1.0)], 1: [(2, 0.5), (3, 0.5)], ...}
    """
    layer_map = {}
    
    if strategy == "ratio":
        # Ratio-based mapping with interpolation
        ratio = num_teacher_layers / num_student_layers
        
        for l_s in range(num_student_layers):
            # 找到对应的教师层位置（浮点数）
            t_pos = l_s * ratio
            
            # 前后两层
            l_t_low = max(0, min(num_teacher_layers - 1, math.floor(t_pos)))
            l_t_high = max(0, min(num_teacher_layers - 1, math.ceil(t_pos)))
            
            if l_t_low == l_t_high:
                # 恰好对齐
                layer_map[l_s] = [(l_t_low, 1.0)]
            else:
                # 插值权重
                weight_high = t_pos - l_t_low
                weight_low = 1.0 - weight_high
                layer_map[l_s] = [(l_t_low, weight_low), (l_t_high, weight_high)]
    
    elif strategy == "uniform":
        # Uniform mapping: 均匀选择教师层
        teacher_indices = torch.linspace(0, num_teacher_layers - 1, num_student_layers).long()
        for l_s, l_t in enumerate(teacher_indices):
            layer_map[l_s] = [(l_t.item(), 1.0)]
    
    elif strategy == "skip":
        # Skip mapping: 跳跃映射（适用于非常深的教师）
        skip = num_teacher_layers // num_student_layers
        for l_s in range(num_student_layers):
            l_t = min(l_s * skip, num_teacher_layers - 1)
            layer_map[l_s] = [(l_t, 1.0)]
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return layer_map


def interpolate_teacher_layers(
    teacher_kvs: List[torch.Tensor],
    layer_map: Dict[int, List[Tuple[int, float]]],
    num_student_layers: int
) -> List[torch.Tensor]:
    """
    使用 layer_map 插值教师 KV。
    
    Args:
        teacher_kvs: List of [batch, time, dim] 每层的教师 KV
        layer_map: 层映射字典
        num_student_layers: 学生层数
        
    Returns:
        student_kvs: List of [batch, time, dim] 插值后的 KV
    """
    student_kvs = []
    
    for l_s in range(num_student_layers):
        if l_s not in layer_map:
            raise ValueError(f"No mapping for student layer {l_s}")
        
        mapping = layer_map[l_s]
        
        if len(mapping) == 1:
            # 单一映射
            l_t, weight = mapping[0]
            student_kv = teacher_kvs[l_t] * weight
        else:
            # 插值
            student_kv = None
            for l_t, weight in mapping:
                if student_kv is None:
                    student_kv = teacher_kvs[l_t] * weight
                else:
                    student_kv = student_kv + teacher_kvs[l_t] * weight
        
        student_kvs.append(student_kv)
    
    return student_kvs


def visualize_layer_mapping(
    layer_map: Dict[int, List[Tuple[int, float]]],
    num_teacher_layers: int,
    num_student_layers: int,
    save_path: Optional[str] = None
):
    """
    可视化层映射矩阵。
    
    Args:
        layer_map: 层映射字典
        num_teacher_layers: 教师层数
        num_student_layers: 学生层数
        save_path: 保存路径（可选）
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return
    
    # 构造映射矩阵 [student, teacher]
    mapping_matrix = np.zeros((num_student_layers, num_teacher_layers))
    
    for l_s, mappings in layer_map.items():
        for l_t, weight in mappings:
            mapping_matrix[l_s, l_t] = weight
    
    # 绘制热图
    plt.figure(figsize=(12, 8))
    plt.imshow(mapping_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Weight')
    plt.xlabel('Teacher Layer')
    plt.ylabel('Student Layer')
    plt.title(f'Layer Mapping: {num_teacher_layers} → {num_student_layers}')
    
    # 标记非零权重
    for l_s in range(num_student_layers):
        for l_t in range(num_teacher_layers):
            if mapping_matrix[l_s, l_t] > 0:
                plt.text(l_t, l_s, f'{mapping_matrix[l_s, l_t]:.2f}',
                        ha='center', va='center', color='white', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Layer mapping visualization saved to {save_path}")
    else:
        plt.show()


def build_multi_teacher_layer_map(
    teacher_layer_counts: List[int],
    student_layer_count: int,
    strategy: str = "ratio"
) -> List[Dict[int, List[Tuple[int, float]]]]:
    """
    为多个教师构建层映射。
    
    Args:
        teacher_layer_counts: 每个教师的层数
        student_layer_count: 学生层数
        strategy: 映射策略
        
    Returns:
        layer_maps: List of layer_map for each teacher
    """
    layer_maps = []
    for num_teacher_layers in teacher_layer_counts:
        layer_map = build_layer_mapping(num_teacher_layers, student_layer_count, strategy)
        layer_maps.append(layer_map)
    
    return layer_maps


def merge_multi_teacher_kvs(
    teacher_kvs_list: List[List[torch.Tensor]],
    layer_maps: List[Dict[int, List[Tuple[int, float]]]],
    num_student_layers: int,
    fusion_weights: Optional[List[float]] = None
) -> List[torch.Tensor]:
    """
    合并多个教师的 KV（带层映射）。
    
    Args:
        teacher_kvs_list: List of (List of teacher KV per layer) for each teacher
        layer_maps: 每个教师的层映射
        num_student_layers: 学生层数
        fusion_weights: 教师融合权重（可选，默认均匀）
        
    Returns:
        merged_kvs: List of [batch, time, dim] 合并后的 KV
    """
    num_teachers = len(teacher_kvs_list)
    
    if fusion_weights is None:
        fusion_weights = [1.0 / num_teachers] * num_teachers
    
    # 为每个教师插值
    interpolated_kvs = []
    for teacher_kvs, layer_map in zip(teacher_kvs_list, layer_maps):
        student_kvs = interpolate_teacher_layers(teacher_kvs, layer_map, num_student_layers)
        interpolated_kvs.append(student_kvs)
    
    # 合并
    merged_kvs = []
    for l_s in range(num_student_layers):
        merged_kv = None
        for teacher_idx, weight in enumerate(fusion_weights):
            teacher_kv = interpolated_kvs[teacher_idx][l_s]
            if merged_kv is None:
                merged_kv = teacher_kv * weight
            else:
                merged_kv = merged_kv + teacher_kv * weight
        merged_kvs.append(merged_kv)
    
    return merged_kvs


if __name__ == "__main__":
    # 测试代码
    print("Testing layer mapping...")
    
    # 测试 ratio mapping
    layer_map = build_layer_mapping(num_teacher_layers=24, num_student_layers=12, strategy="ratio")
    print(f"Layer mapping (24 → 12): {layer_map}")
    
    # 验证所有学生层都有映射
    assert len(layer_map) == 12
    # 验证权重和为 1
    for l_s, mappings in layer_map.items():
        total_weight = sum(w for _, w in mappings)
        assert abs(total_weight - 1.0) < 1e-6, f"Layer {l_s} weights sum to {total_weight}"
    print("✓ Ratio mapping test passed")
    
    # 测试插值
    num_teacher_layers = 24
    num_student_layers = 12
    batch_size = 2
    time_steps = 10
    hidden_dim = 64
    
    teacher_kvs = [torch.randn(batch_size, time_steps, hidden_dim) for _ in range(num_teacher_layers)]
    student_kvs = interpolate_teacher_layers(teacher_kvs, layer_map, num_student_layers)
    
    assert len(student_kvs) == num_student_layers
    assert student_kvs[0].shape == (batch_size, time_steps, hidden_dim)
    print("✓ Interpolation test passed")
    
    # 测试多教师合并
    teacher_kvs_list = [
        [torch.randn(batch_size, time_steps, hidden_dim) for _ in range(24)],
        [torch.randn(batch_size, time_steps, hidden_dim) for _ in range(32)]
    ]
    layer_maps = build_multi_teacher_layer_map([24, 32], num_student_layers, strategy="ratio")
    merged_kvs = merge_multi_teacher_kvs(teacher_kvs_list, layer_maps, num_student_layers)
    
    assert len(merged_kvs) == num_student_layers
    assert merged_kvs[0].shape == (batch_size, time_steps, hidden_dim)
    print("✓ Multi-teacher merge test passed")
    
    # 可视化（不保存）
    print("\nVisualizing layer mapping (32 → 12)...")
    layer_map_vis = build_layer_mapping(32, 12, strategy="ratio")
    visualize_layer_mapping(layer_map_vis, 32, 12)
    
    print("All layer mapping tests passed!")
