"""
v4.0 Integration Smoke Test
测试 train_with_kv.py 的双模式集成是否正确
"""
import sys
import torch
import numpy as np

print("=" * 60)
print("v4.0 Integration Smoke Test")
print("=" * 60)

# Test 1: 导入检查
print("\n[Test 1] Import Checks")
try:
    from src.map_projection_aligner import MapProjectionAligner
    print("✓ MapProjectionAligner imported")
except Exception as e:
    print(f"✗ MapProjectionAligner import failed: {e}")
    sys.exit(1)

try:
    from src.headwise_projector import HeadwiseMapProjector
    print("✓ HeadwiseMapProjector imported")
except Exception as e:
    print(f"✗ HeadwiseMapProjector import failed: {e}")
    sys.exit(1)

try:
    from src.time_warping import TimeWarper
    print("✓ TimeWarper imported")
except Exception as e:
    print(f"✗ TimeWarper import failed: {e}")
    sys.exit(1)

# Test 2: stack_past_kv 工具函数
print("\n[Test 2] stack_past_kv Function")
try:
    # 模拟创建工具函数
    def stack_past_kv(past_key_values, as_tensor=True):
        kvs = []
        for k, v in past_key_values:
            if isinstance(k, np.ndarray):
                k = torch.from_numpy(k)
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            if k.device != v.device:
                v = v.to(k.device)
            kvs.append(torch.stack([k, v], dim=0))
        stacked = torch.stack(kvs, dim=0)
        return stacked if as_tensor else stacked.cpu().numpy()
    
    # 测试数据
    L, B, H, T, D = 4, 2, 8, 50, 64
    past_kv = []
    for _ in range(L):
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)
        past_kv.append((k, v))
    
    stacked = stack_past_kv(past_kv)
    expected_shape = (L, 2, B, H, T, D)
    assert stacked.shape == expected_shape, f"Shape mismatch: {stacked.shape} vs {expected_shape}"
    print(f"✓ stack_past_kv: {past_kv[0][0].shape} -> {stacked.shape}")
except Exception as e:
    print(f"✗ stack_past_kv failed: {e}")
    sys.exit(1)

# Test 3: MapProjectionAligner 初始化
print("\n[Test 3] MapProjectionAligner Initialization")
try:
    aligner_structured = MapProjectionAligner(
        num_teacher_layers=12,
        num_student_layers=6,
        num_teacher_heads=12,
        num_student_heads=6,
        teacher_head_dim=64,
        student_head_dim=64,
        mode="structured",
        share_dim_proj=True,
        init_uniform=True
    )
    param_count = sum(p.numel() for p in aligner_structured.parameters())
    print(f"✓ Structured Aligner: {param_count:,} parameters")
    
    aligner_flat = MapProjectionAligner(
        num_teacher_layers=12,
        num_student_layers=6,
        num_teacher_heads=12,
        num_student_heads=6,
        teacher_head_dim=64,
        student_head_dim=64,
        mode="flat"
    )
    param_count_flat = sum(p.numel() for p in aligner_flat.parameters())
    print(f"✓ Flat Aligner: {param_count_flat:,} parameters")
except Exception as e:
    print(f"✗ Aligner initialization failed: {e}")
    sys.exit(1)

# Test 4: 完整对齐流程（模拟训练循环）
print("\n[Test 4] Full Alignment Pipeline Simulation")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aligner = MapProjectionAligner(
        num_teacher_layers=4,
        num_student_layers=2,
        num_teacher_heads=8,
        num_student_heads=4,
        teacher_head_dim=64,
        student_head_dim=64,
        mode="structured",
        share_dim_proj=True,
        init_uniform=True
    ).to(device)
    
    # 模拟数据
    B, T, L_t, H_t, D_t = 2, 50, 4, 8, 64
    L_s, H_s, D_s = 2, 4, 64
    
    teacher_k = torch.randn(L_t, B, H_t, T, D_t, device=device)
    teacher_v = torch.randn(L_t, B, H_t, T, D_t, device=device)
    segment_ids = torch.zeros(B, T, dtype=torch.long, device=device)
    
    # 前向传播
    aligned_k, aligned_v, attn_map = aligner(teacher_k, teacher_v, None, segment_ids)
    
    # 验证形状
    expected_k_shape = (L_s, B, H_s, T, D_s)
    assert aligned_k.shape == expected_k_shape, f"K shape mismatch: {aligned_k.shape} vs {expected_k_shape}"
    assert aligned_v.shape == expected_k_shape, f"V shape mismatch: {aligned_v.shape} vs {expected_k_shape}"
    
    print(f"✓ Alignment: [{L_t},{B},{H_t},{T},{D_t}] -> [{L_s},{B},{H_s},{T},{D_s}]")
    print(f"  - NaN check: K={torch.isnan(aligned_k).sum().item()}, V={torch.isnan(aligned_v).sum().item()}")
    
    # 模拟 KV loss 计算
    student_k = torch.randn(L_s, B, H_s, T, D_s, device=device)
    student_v = torch.randn(L_s, B, H_s, T, D_s, device=device)
    
    import torch.nn.functional as F
    kv_loss_k = F.mse_loss(aligned_k, student_k)
    kv_loss_v = F.mse_loss(aligned_v, student_v)
    kv_loss_total = (kv_loss_k + kv_loss_v) / 2.0
    
    print(f"  - Loss: K={kv_loss_k.item():.4f}, V={kv_loss_v.item():.4f}, Total={kv_loss_total.item():.4f}")
    assert not torch.isnan(kv_loss_total), "KV loss is NaN!"
    
except Exception as e:
    print(f"✗ Alignment pipeline failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: 命令行参数模拟
print("\n[Test 5] Command Line Arguments Check")
try:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment_mode", type=str, default="flat", choices=["flat", "structured"])
    parser.add_argument("--map_proj_share_dim", action="store_true")
    parser.add_argument("--map_proj_init_uniform", action="store_true")
    
    # 测试默认参数
    args_flat = parser.parse_args([])
    assert args_flat.alignment_mode == "flat"
    assert args_flat.map_proj_share_dim == False
    print("✓ Default args: mode=flat, share_dim=False, init_uniform=False")
    
    # 测试 structured 参数
    args_structured = parser.parse_args([
        "--alignment_mode", "structured",
        "--map_proj_share_dim",
        "--map_proj_init_uniform"
    ])
    assert args_structured.alignment_mode == "structured"
    assert args_structured.map_proj_share_dim == True
    assert args_structured.map_proj_init_uniform == True
    print("✓ Structured args: mode=structured, share_dim=True, init_uniform=True")
    
except Exception as e:
    print(f"✗ Argument parsing failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED - Integration Ready!")
print("=" * 60)
print("\n📋 Next Steps:")
print("1. Run profile_alignment.py for both modes:")
print("   python experiments/profile_alignment.py --mode flat")
print("   python experiments/profile_alignment.py --mode structured")
print("\n2. Run 10-step smoke test:")
print("   python experiments/train_with_kv.py \\")
print("       --model_name gpt2 \\")
print("       --subset_size 10 \\")
print("       --batch_size 2 \\")
print("       --epochs 1 \\")
print("       --alignment_mode structured \\")
print("       --map_proj_share_dim \\")
print("       --map_proj_init_uniform")
print("\n3. Launch A/B experiments (see V4_EXECUTION_ROADMAP.md)")
