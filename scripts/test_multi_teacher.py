"""
Quick Test for Multi-Teacher KV Distillation

测试所有模块是否正常工作。

Usage:
    python scripts/test_multi_teacher.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import warnings
warnings.filterwarnings("ignore")

def test_alignment_modules():
    """测试对齐模块。"""
    print("\n=== Testing Alignment Modules ===")
    
    from align import (
        build_char_align_matrix,
        align_sequence_lengths,
        build_multi_teacher_layer_map,
        MultiTeacherHeadDimAdapter,
        MultiTeacherRoPEScaler
    )
    
    # Test tokenizer alignment
    print("✓ Tokenizer alignment module imported")
    
    # Test time alignment
    teacher_kv = torch.randn(2, 8, 64)
    student_kv = torch.randn(2, 10, 64)
    t_aligned, s_aligned, t_mask, s_mask = align_sequence_lengths(teacher_kv, student_kv)
    assert t_aligned.size(1) == s_aligned.size(1)
    print("✓ Time alignment works")
    
    # Test layer mapping
    layer_maps = build_multi_teacher_layer_map([24, 32], 12, strategy="ratio")
    assert len(layer_maps) == 2
    print("✓ Layer mapping works")
    
    # Test head/dim adapter
    adapter = MultiTeacherHeadDimAdapter([(768, 12), (1024, 16)], 512, 8)
    teacher_ks = [torch.randn(2, 10, 768), torch.randn(2, 10, 1024)]
    teacher_vs = [torch.randn(2, 10, 768), torch.randn(2, 10, 1024)]
    adapted_ks, adapted_vs = adapter(teacher_ks, teacher_vs, adapt_heads=False)
    assert len(adapted_ks) == 2
    print("✓ Head/dim adapter works")
    
    # Test RoPE scaler
    scaler = MultiTeacherRoPEScaler([(10000, 2048), (10000, 4096)], 2048, scaling_method="ntk")
    scaled_ks, scaled_vs = scaler.scale_kv_pairs(teacher_ks, teacher_vs)
    assert len(scaled_ks) == 2
    print("✓ RoPE scaler works")
    
    print("All alignment modules passed! ✓")


def test_teacher_modules():
    """测试教师模块。"""
    print("\n=== Testing Teacher Modules ===")
    
    from teacher import (
        compute_teacher_prototype,
        compute_similarity,
        compute_multi_teacher_prototypes,
        compute_routing_weights
    )
    
    # Test prototype computation
    kvs = torch.randn(4, 10, 64)
    proto = compute_teacher_prototype(kvs, method="mean")
    assert proto.shape == (64,)
    print("✓ Teacher prototype computation works")
    
    # Test similarity
    query = torch.randn(4, 64)
    sim = compute_similarity(query, proto, method="cosine")
    assert sim.shape == (4,)
    print("✓ Similarity computation works")
    
    # Test multi-teacher prototypes
    teacher_kvs_list = [torch.randn(4, 10, 64) for _ in range(3)]
    protos = compute_multi_teacher_prototypes(teacher_kvs_list, method="mean")
    assert len(protos) == 3
    print("✓ Multi-teacher prototypes work")
    
    # Test routing weights
    weights = compute_routing_weights(query, protos, temperature=1.0)
    assert weights.shape == (4, 3)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1e-6)
    print("✓ Routing weights work")
    
    print("All teacher modules passed! ✓")


def test_fusion_modules():
    """测试融合模块。"""
    print("\n=== Testing Fusion Modules ===")
    
    from fuse import (
        fuse_kvs_fixed,
        fuse_kvs_similarity,
        fuse_kvs_learnable,
        LearnableRouter,
        EntropyRegularizer
    )
    
    # Test fixed fusion
    kvs_list = [torch.randn(4, 10, 64) for _ in range(3)]
    fused = fuse_kvs_fixed(kvs_list, weights=[0.5, 0.3, 0.2])
    assert fused.shape == (4, 10, 64)
    print("✓ Fixed fusion works")
    
    # Test similarity fusion
    query = torch.randn(4, 64)
    prototypes = [torch.randn(64) for _ in range(3)]
    fused, weights = fuse_kvs_similarity(kvs_list, query, prototypes)
    assert fused.shape == (4, 10, 64)
    assert weights.shape == (4, 3)
    print("✓ Similarity fusion works")
    
    # Test learnable fusion
    router = LearnableRouter(64, 3, router_type="mlp")
    fused, weights = fuse_kvs_learnable(kvs_list, query, router)
    assert fused.shape == (4, 10, 64)
    print("✓ Learnable fusion (MLP) works")
    
    # Test attention router
    router_attn = LearnableRouter(64, 3, router_type="attention")
    fused, weights = fuse_kvs_learnable(kvs_list, query, router_attn)
    assert fused.shape == (4, 10, 64)
    print("✓ Learnable fusion (Attention) works")
    
    # Test entropy regularizer
    regularizer = EntropyRegularizer(target="specialized", strength=0.01)
    loss = regularizer.compute_loss(weights)
    assert loss.ndim == 0
    print("✓ Entropy regularizer works")
    
    print("All fusion modules passed! ✓")


def test_integration():
    """测试端到端集成。"""
    print("\n=== Testing End-to-End Integration ===")
    
    from align import (
        build_multi_teacher_layer_map,
        MultiTeacherHeadDimAdapter,
        interpolate_teacher_layers
    )
    from teacher import compute_multi_teacher_prototypes, compute_routing_weights
    from fuse import fuse_kvs_learnable, LearnableRouter
    
    # Simulate multi-teacher scenario
    num_teachers = 3
    num_layers = 12
    batch_size = 2
    time_steps = 10
    hidden_dim = 64
    
    # 1. Generate teacher KVs (different architectures)
    teacher_kvs_list = [
        [torch.randn(batch_size, time_steps, hidden_dim) for _ in range(24)],  # Teacher 1: 24 layers
        [torch.randn(batch_size, time_steps, hidden_dim) for _ in range(32)],  # Teacher 2: 32 layers
        [torch.randn(batch_size, time_steps, hidden_dim) for _ in range(12)]   # Teacher 3: 12 layers
    ]
    
    # 2. Build layer mappings
    layer_maps = build_multi_teacher_layer_map([24, 32, 12], num_layers, strategy="ratio")
    print("✓ Layer mappings built")
    
    # 3. Align teacher KVs to student layers
    aligned_kvs_list = []
    for teacher_kvs, layer_map in zip(teacher_kvs_list, layer_maps):
        aligned_kvs = interpolate_teacher_layers(teacher_kvs, layer_map, num_layers)
        aligned_kvs_list.append(aligned_kvs)
    print("✓ Teacher KVs aligned")
    
    # 4. Compute teacher prototypes
    # Use first layer for prototype
    teacher_kvs_layer0 = [kvs[0] for kvs in aligned_kvs_list]
    prototypes = compute_multi_teacher_prototypes(teacher_kvs_layer0, method="mean")
    print("✓ Teacher prototypes computed")
    
    # 5. Build router
    router = LearnableRouter(hidden_dim, num_teachers, router_type="attention")
    print("✓ Router built")
    
    # 6. Fuse KVs per layer
    query = torch.randn(batch_size, hidden_dim)
    fused_kvs = []
    
    for l in range(num_layers):
        kvs_layer = [teacher_kvs[l] for teacher_kvs in aligned_kvs_list]
        fused_kv, weights = fuse_kvs_learnable(kvs_layer, query, router)
        fused_kvs.append(fused_kv)
    
    assert len(fused_kvs) == num_layers
    print("✓ Multi-teacher KVs fused")
    
    # 7. Verify shapes
    for fused_kv in fused_kvs:
        assert fused_kv.shape == (batch_size, time_steps, hidden_dim)
    print("✓ Output shapes correct")
    
    print("End-to-end integration passed! ✓")


def main():
    print("=" * 60)
    print("Multi-Teacher KV Distillation - Quick Test")
    print("=" * 60)
    
    try:
        test_alignment_modules()
        test_teacher_modules()
        test_fusion_modules()
        test_integration()
        
        print("\n" + "=" * 60)
        print("All tests passed! ✓✓✓")
        print("=" * 60)
        print("\nYou can now run training with:")
        print("  python experiments/train_multi_teacher_kv.py --help")
        print("Or submit to SLURM:")
        print("  sbatch scripts/run_multi_teacher.sh")
        print()
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("Test failed! ✗")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
