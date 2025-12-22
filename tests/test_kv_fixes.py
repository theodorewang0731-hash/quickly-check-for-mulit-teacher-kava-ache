"""
测试 KV 对齐修复 - 验证头数不匹配和时间重采样越界问题已解决

运行此测试以确认:
1. ✓ 头数不匹配 (12 vs 2) 已解决
2. ✓ 时间重采样越界 (index out of bounds) 已解决
3. ✓ 边界情况 (T=0, T=1, 空段) 正常处理

用法:
    python tests/test_kv_fixes.py
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.kv_head_projector import (
    KVProjector, 
    safe_time_resample, 
    build_safe_linear_indices,
    get_kv_heads_from_tensor
)
from experiments.alignment_v2 import (
    resample_kv_with_interpolation,
    _global_resample
)


def test_head_projection():
    """测试 1: 头数投影 (GQA 12->2)"""
    print("=" * 80)
    print("测试 1: 头数投影 (GQA: Ht=12 -> Hs=2)")
    print("=" * 80)
    
    # 场景: Teacher 有 12 个 KV heads, Student 有 2 个
    Ht, Hs = 12, 2
    Dt, Ds = 128, 128
    B, T = 4, 50
    
    projector = KVProjector(Ht, Hs, Dt, Ds)
    
    k_teacher = torch.randn(B, Ht, T, Dt)
    v_teacher = torch.randn(B, Ht, T, Dt)
    
    print(f"输入:  K shape={k_teacher.shape}, V shape={v_teacher.shape}")
    
    k_student, v_student = projector(k_teacher, v_teacher)
    
    print(f"输出:  K shape={k_student.shape}, V shape={v_student.shape}")
    
    # 验证
    assert k_student.shape == (B, Hs, T, Ds), f"Expected {(B, Hs, T, Ds)}, got {k_student.shape}"
    assert v_student.shape == (B, Hs, T, Ds), f"Expected {(B, Hs, T, Ds)}, got {v_student.shape}"
    
    print("✓ 头数投影测试通过!")
    print()


def test_head_and_dim_projection():
    """测试 2: 头数 + head_dim 同时不匹配"""
    print("=" * 80)
    print("测试 2: 头数 + head_dim 不匹配 (Ht=28 -> Hs=2, Dt=128 -> Ds=64)")
    print("=" * 80)
    
    Ht, Hs = 28, 2
    Dt, Ds = 128, 64
    B, T = 4, 50
    
    projector = KVProjector(Ht, Hs, Dt, Ds)
    
    k_teacher = torch.randn(B, Ht, T, Dt)
    v_teacher = torch.randn(B, Ht, T, Dt)
    
    print(f"输入:  K shape={k_teacher.shape}, V shape={v_teacher.shape}")
    
    k_student, v_student = projector(k_teacher, v_teacher)
    
    print(f"输出:  K shape={k_student.shape}, V shape={v_student.shape}")
    
    # 验证
    assert k_student.shape == (B, Hs, T, Ds), f"Expected {(B, Hs, T, Ds)}, got {k_student.shape}"
    assert v_student.shape == (B, Hs, T, Ds), f"Expected {(B, Hs, T, Ds)}, got {v_student.shape}"
    
    print("✓ 头数 + 维度投影测试通过!")
    print()


def test_safe_time_resample():
    """测试 3: 安全时间重采样 (80 -> 50)"""
    print("=" * 80)
    print("测试 3: 安全时间重采样 (T_in=80 -> T_out=50)")
    print("=" * 80)
    
    B, H, T_in, D = 4, 2, 80, 128
    T_out = 50
    
    x = torch.randn(B, H, T_in, D)
    indices = build_safe_linear_indices(B, T_in, T_out, x.device)
    
    print(f"输入:   X shape={x.shape}")
    print(f"索引:   indices shape={indices.shape}, dtype={indices.dtype}")
    print(f"索引范围: min={indices.min()}, max={indices.max()}")
    
    x_resampled = safe_time_resample(x, indices)
    
    print(f"输出:   X_resampled shape={x_resampled.shape}")
    
    # 验证
    assert x_resampled.shape == (B, H, T_out, D), f"Expected {(B, H, T_out, D)}, got {x_resampled.shape}"
    assert indices.min() >= 0, "Indices contain negative values!"
    assert indices.max() < T_in, f"Indices out of bounds! max={indices.max()}, T_in={T_in}"
    
    print("✓ 时间重采样测试通过!")
    print()


def test_edge_case_t_equals_1():
    """测试 4: 边界情况 - T_in=1, T_out=1"""
    print("=" * 80)
    print("测试 4: 边界情况 (T_in=1, T_out=1)")
    print("=" * 80)
    
    B, H, D = 4, 2, 128
    
    x = torch.randn(B, H, 1, D)
    indices = build_safe_linear_indices(B, 1, 1, x.device)
    
    print(f"输入:   X shape={x.shape}")
    print(f"索引:   indices={indices[0]}")
    
    x_resampled = safe_time_resample(x, indices)
    
    print(f"输出:   X_resampled shape={x_resampled.shape}")
    
    # 验证
    assert x_resampled.shape == (B, H, 1, D)
    assert torch.allclose(x, x_resampled), "T=1 case should preserve values!"
    
    print("✓ 边界情况 T=1 测试通过!")
    print()


def test_edge_case_t_equals_0():
    """测试 5: 边界情况 - T_in=0 (空序列)"""
    print("=" * 80)
    print("测试 5: 边界情况 (T_in=0, 空序列)")
    print("=" * 80)
    
    B, H, D = 4, 2, 128
    T_out = 10
    
    x = torch.randn(B, H, 0, D)  # 空序列
    indices = build_safe_linear_indices(B, 0, T_out, x.device)
    
    print(f"输入:   X shape={x.shape} (空)")
    print(f"目标长度: T_out={T_out}")
    
    x_resampled = safe_time_resample(x, indices)
    
    print(f"输出:   X_resampled shape={x_resampled.shape}")
    
    # 验证
    assert x_resampled.shape == (B, H, T_out, D)
    assert torch.all(indices == 0), "Empty input should map all indices to 0"
    
    print("✓ 边界情况 T=0 测试通过!")
    print()


def test_integration_with_alignment_v2():
    """测试 6: 集成测试 - 与 alignment_v2 配合"""
    print("=" * 80)
    print("测试 6: 集成测试 - resample_kv_with_interpolation")
    print("=" * 80)
    
    B, H, T_teacher, D = 4, 12, 100, 128
    T_student = 60
    
    teacher_kv = torch.randn(B, H, T_teacher, D)
    
    print(f"Teacher KV: shape={teacher_kv.shape}")
    print(f"Target length: {T_student}")
    
    # 使用 alignment_v2 的重采样函数 (已经修复)
    resampled_kv = resample_kv_with_interpolation(
        teacher_kv, 
        T_student,
        teacher_segments=None,  # 不使用段落信息
        student_segments=None
    )
    
    print(f"Resampled KV: shape={resampled_kv.shape}")
    
    # 验证
    assert resampled_kv.shape == (B, H, T_student, D), f"Expected {(B, H, T_student, D)}, got {resampled_kv.shape}"
    
    print("✓ 集成测试通过!")
    print()


def test_combined_head_and_time():
    """测试 7: 综合测试 - 头数投影 + 时间重采样"""
    print("=" * 80)
    print("测试 7: 综合测试 - 头数投影 + 时间重采样")
    print("=" * 80)
    
    # Teacher: 12 heads, 80 tokens, head_dim=128
    # Student: 2 heads, 50 tokens, head_dim=128
    
    B = 4
    Ht, Hs = 12, 2
    Dt, Ds = 128, 128
    T_teacher, T_student = 80, 50
    
    print(f"Teacher: {Ht} heads, {T_teacher} tokens, head_dim={Dt}")
    print(f"Student: {Hs} heads, {T_student} tokens, head_dim={Ds}")
    
    # Step 1: 生成 teacher KV
    k_teacher = torch.randn(B, Ht, T_teacher, Dt)
    v_teacher = torch.randn(B, Ht, T_teacher, Dt)
    
    # Step 2: 投影头数
    head_projector = KVProjector(Ht, Hs, Dt, Ds)
    k_proj, v_proj = head_projector(k_teacher, v_teacher)
    print(f"After head projection: K shape={k_proj.shape}")
    
    # Step 3: 时间重采样
    k_resampled = resample_kv_with_interpolation(k_proj, T_student)
    v_resampled = resample_kv_with_interpolation(v_proj, T_student)
    print(f"After time resampling: K shape={k_resampled.shape}")
    
    # Step 4: 生成 student KV (用于对比)
    k_student = torch.randn(B, Hs, T_student, Ds)
    v_student = torch.randn(B, Hs, T_student, Ds)
    
    # Step 5: 计算 loss (现在 shapes 应该完全匹配!)
    loss_k = torch.nn.functional.mse_loss(k_resampled, k_student)
    loss_v = torch.nn.functional.mse_loss(v_resampled, v_student)
    
    print(f"Loss K: {loss_k.item():.4f}")
    print(f"Loss V: {loss_v.item():.4f}")
    print(f"Final shapes match: K {k_resampled.shape} == {k_student.shape}")
    
    # 验证
    assert k_resampled.shape == k_student.shape
    assert v_resampled.shape == v_student.shape
    
    print("✓ 综合测试通过! 头数不匹配和时间重采样问题已完全解决!")
    print()


def main():
    print("\n")
    print("=" * 80)
    print(" KV 对齐修复验证测试")
    print("=" * 80)
    print()
    
    try:
        test_head_projection()
        test_head_and_dim_projection()
        test_safe_time_resample()
        test_edge_case_t_equals_1()
        test_edge_case_t_equals_0()
        test_integration_with_alignment_v2()
        test_combined_head_and_time()
        
        print("=" * 80)
        print("🎉 所有测试通过!")
        print("=" * 80)
        print()
        print("修复确认:")
        print("  ✓ 头数不匹配 (12 vs 2) 已解决")
        print("  ✓ 时间重采样越界 已解决")
        print("  ✓ 边界情况处理 正常")
        print("  ✓ 可以开始训练!")
        print()
        
        return 0
    
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ 测试失败!")
        print("=" * 80)
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
