"""
Profile Alignment Script (阶段 1 验证工具)

✨ v4.0 目的：
- 只跑 1-2 个 batch 的 forward
- 验证形状对齐是否正确
- 检查是否有 NaN 或异常值
- 简单评估 cos 相似度
- 不进行真正的蒸馏训练

这是"训练前评估"的简易版，确保对齐模块工作正常。
"""
import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.map_projection_aligner import (
    create_structured_aligner,
    create_flat_aligner
)
from transformers import AutoConfig
import torch.nn.functional as F


def profile_alignment(
    teacher_model_name="Qwen/Qwen2.5-7B",
    student_model_name="Qwen/Qwen2.5-1.5B",
    mode="structured",
    batch_size=2,
    seq_length=100
):
    """
    Profile 对齐模块
    
    Args:
        teacher_model_name: Teacher 模型名称
        student_model_name: Student 模型名称
        mode: "structured" 或 "flat"
        batch_size: Batch 大小
        seq_length: 序列长度
    """
    print("=" * 80)
    print(f"🔍 Profile Alignment - Mode: {mode}")
    print("=" * 80)
    
    # 1. 加载配置
    print("\n[1/5] 加载模型配置...")
    teacher_cfg = AutoConfig.from_pretrained(teacher_model_name, trust_remote_code=True)
    student_cfg = AutoConfig.from_pretrained(student_model_name, trust_remote_code=True)
    
    print(f"  Teacher: {teacher_cfg.num_hidden_layers} 层, "
          f"{teacher_cfg.num_attention_heads} 头, "
          f"{teacher_cfg.hidden_size} 维")
    print(f"  Student: {student_cfg.num_hidden_layers} 层, "
          f"{student_cfg.num_attention_heads} 头, "
          f"{student_cfg.hidden_size} 维")
    
    # 2. 创建对齐器
    print(f"\n[2/5] 创建对齐器 (mode={mode})...")
    if mode == "structured":
        aligner = create_structured_aligner(
            teacher_cfg, student_cfg,
            share_dim_proj=True,
            init_uniform=True
        )
    else:
        aligner = create_flat_aligner(teacher_cfg, student_cfg)
    
    print(f"  ✅ 对齐器创建成功")
    
    # 统计参数量
    total_params = sum(p.numel() for p in aligner.parameters())
    trainable_params = sum(p.numel() for p in aligner.parameters() if p.requires_grad)
    print(f"  参数量: {total_params:,} ({trainable_params:,} 可训练)")
    
    # 3. 生成模拟输入
    print(f"\n[3/5] 生成模拟输入...")
    B = batch_size
    L_t = teacher_cfg.num_hidden_layers
    H_t = teacher_cfg.num_attention_heads
    T_t = seq_length
    D_t = teacher_cfg.hidden_size // H_t
    
    k_t = torch.randn(B, L_t, H_t, T_t, D_t)
    v_t = torch.randn(B, L_t, H_t, T_t, D_t)
    q_t = torch.randn(B, L_t, H_t, T_t, D_t)
    
    # Segment IDs: P(10) + R(80) + A(10)
    segment_ids = torch.cat([
        torch.zeros(10, dtype=torch.long),
        torch.ones(80, dtype=torch.long),
        torch.full((10,), 2, dtype=torch.long)
    ]).unsqueeze(0).expand(B, T_t)
    
    print(f"  k_t: {k_t.shape}")
    print(f"  v_t: {v_t.shape}")
    print(f"  q_t: {q_t.shape}")
    print(f"  segment_ids: {segment_ids.shape}")
    
    # 4. 前向传播
    print(f"\n[4/5] 前向传播...")
    try:
        with torch.no_grad():
            k_s, v_s, q_s = aligner(k_t, v_t, q_t, segment_ids)
        
        print(f"  ✅ 前向传播成功")
        print(f"  k_s: {k_s.shape}")
        print(f"  v_s: {v_s.shape}")
        print(f"  q_s: {q_s.shape}")
        
        # 验证形状
        L_s = student_cfg.num_hidden_layers
        H_s = student_cfg.num_attention_heads
        T_s = T_t  # 假设序列长度不变
        D_s = student_cfg.hidden_size // H_s
        
        expected_shape = (B, L_s, H_s, T_s, D_s)
        assert k_s.shape == expected_shape, f"K 形状不匹配: {k_s.shape} != {expected_shape}"
        assert v_s.shape == expected_shape, f"V 形状不匹配: {v_s.shape} != {expected_shape}"
        assert q_s.shape == expected_shape, f"Q 形状不匹配: {q_s.shape} != {expected_shape}"
        print(f"  ✅ 形状验证通过")
        
    except Exception as e:
        print(f"  ❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 统计分析
    print(f"\n[5/5] 统计分析...")
    
    # 检查 NaN
    has_nan_k = torch.isnan(k_s).any().item()
    has_nan_v = torch.isnan(v_s).any().item()
    has_nan_q = torch.isnan(q_s).any().item()
    
    if has_nan_k or has_nan_v or has_nan_q:
        print(f"  ❌ 检测到 NaN: K={has_nan_k}, V={has_nan_v}, Q={has_nan_q}")
    else:
        print(f"  ✅ 无 NaN")
    
    # 统计量
    print(f"\n  K 统计:")
    print(f"    均值: {k_s.mean().item():.4f}")
    print(f"    标准差: {k_s.std().item():.4f}")
    print(f"    最小值: {k_s.min().item():.4f}")
    print(f"    最大值: {k_s.max().item():.4f}")
    
    print(f"\n  V 统计:")
    print(f"    均值: {v_s.mean().item():.4f}")
    print(f"    标准差: {v_s.std().item():.4f}")
    print(f"    最小值: {v_s.min().item():.4f}")
    print(f"    最大值: {v_s.max().item():.4f}")
    
    # 简单的 Attention 分布检查
    print(f"\n  简单 Attention 检查:")
    # 计算一个样本的 attention scores
    sample_q = q_s[0, 0, 0]  # [T, D]
    sample_k = k_s[0, 0, 0]  # [T, D]
    scores = torch.matmul(sample_q, sample_k.T) / (D_s ** 0.5)  # [T, T]
    attn_probs = F.softmax(scores, dim=-1)
    
    # 检查是否有任何位置的 attention 过度集中
    max_attn = attn_probs.max(dim=-1)[0].mean().item()
    print(f"    平均最大 attention 权重: {max_attn:.4f}")
    
    if max_attn > 0.9:
        print(f"    ⚠️ Attention 过度集中（>0.9），可能有问题")
    elif max_attn < 0.1:
        print(f"    ⚠️ Attention 过于分散（<0.1），可能有问题")
    else:
        print(f"    ✅ Attention 分布合理")
    
    # 对比原始 teacher 和投影后的相似度（如果是 structured 模式）
    if mode == "structured":
        print(f"\n  Teacher → Student 相似度（仅作参考）:")
        # 注意：这里比较的是原始 teacher 和投影后的 student，维度不同，
        # 只能在 flatten 后比较语义相似度
        k_t_flat = k_t.reshape(B, -1)
        k_s_flat = k_s.reshape(B, -1)
        
        # 归一化后计算余弦相似度
        k_t_norm = F.normalize(k_t_flat, p=2, dim=-1)
        k_s_norm = F.normalize(k_s_flat, p=2, dim=-1)
        cos_sim = (k_t_norm * k_s_norm).sum(dim=-1).mean().item()
        
        print(f"    K 余弦相似度: {cos_sim:.4f}")
        print(f"    （注意：维度不同，仅作参考）")
    
    print("\n" + "=" * 80)
    print(f"✅ Profile 完成 ({mode} 模式)")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile Alignment Module")
    parser.add_argument("--teacher", type=str, default="Qwen/Qwen2.5-7B",
                       help="Teacher 模型名称")
    parser.add_argument("--student", type=str, default="Qwen/Qwen2.5-1.5B",
                       help="Student 模型名称")
    parser.add_argument("--mode", type=str, default="structured",
                       choices=["structured", "flat"],
                       help="对齐模式")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Batch 大小")
    parser.add_argument("--seq_length", type=int, default=100,
                       help="序列长度")
    
    args = parser.parse_args()
    
    # 运行 profile
    profile_alignment(
        teacher_model_name=args.teacher,
        student_model_name=args.student,
        mode=args.mode,
        batch_size=args.batch_size,
        seq_length=args.seq_length
    )
    
    # 如果是对比两种模式，再运行一次
    if args.mode == "structured":
        print("\n\n")
        try:
            profile_alignment(
                teacher_model_name=args.teacher,
                student_model_name=args.student,
                mode="flat",
                batch_size=args.batch_size,
                seq_length=args.seq_length
            )
        except ImportError:
            print("\n⚠️ Flat 模式需要 KVDimensionProjector，跳过")
