"""
Test complete alignment pipeline: Time + Layer + Dimension
===========================================================

Test the three-stage alignment:
    1. Layer alignment: CKA-based top-k teacher layer selection
    2. Time alignment: Segment-aware resampling
    3. Dimension alignment: Linear projection d_teacher -> d_student
"""

import torch
import sys
sys.path.append(".")

from experiments.alignment_v2 import (
    CKALayerMapper,
    SequenceSegment,
    align_multi_teacher_kv_with_projection,
    fuse_multi_teacher_kv
)
from experiments.kv_dimension_projector import KVDimensionProjector
import numpy as np


def test_complete_alignment():
    print("Testing Complete 3-Stage Alignment Pipeline")
    print("=" * 80)
    
    # Setup
    B = 2
    student_layers = 12
    student_length = 50
    student_d_model = 2048
    
    teacher1_layers = 28
    teacher1_length = 80
    teacher1_d_model = 3584
    
    teacher2_layers = 28
    teacher2_length = 65
    teacher2_d_model = 1536
    
    # Test 1: Create projector
    print("\n[Test 1] Create KV Dimension Projector")
    teacher_configs = {
        "Qwen2-7B": {"d_model": teacher1_d_model, "num_layers": teacher1_layers},
        "Qwen2-1.5B": {"d_model": teacher2_d_model, "num_layers": teacher2_layers}
    }
    
    projector = KVDimensionProjector(
        teacher_configs=teacher_configs,
        student_d_model=student_d_model,
        init_method="xavier",
        trainable=True
    )
    print(f"  ✓ Projector created with {projector.count_parameters():,} params")
    
    # Test 2: Create layer mapper
    print("\n[Test 2] Create CKA Layer Mapper")
    layer_mapper = CKALayerMapper(
        student_num_layers=student_layers,
        teacher_num_layers=teacher1_layers,  # Assume same for both teachers
        top_k=2
    )
    
    # Fake similarity matrix
    layer_mapper.similarity_matrix = np.random.rand(student_layers, teacher1_layers)
    layer_mapper.build_layer_mapping()
    print(f"  Student L0 -> Teacher {layer_mapper.layer_mapping[0]}")
    print("  ✓ Layer mapper initialized")
    
    # Test 3: Create teacher KVs (per-layer)
    print("\n[Test 3] Create Multi-Teacher Per-Layer KVs")
    teacher_kvs = {
        "Qwen2-7B": [
            (
                torch.randn(B, teacher1_length, teacher1_d_model),
                torch.randn(B, teacher1_length, teacher1_d_model)
            )
            for _ in range(teacher1_layers)
        ],
        "Qwen2-1.5B": [
            (
                torch.randn(B, teacher2_length, teacher2_d_model),
                torch.randn(B, teacher2_length, teacher2_d_model)
            )
            for _ in range(teacher2_layers)
        ]
    }
    print(f"  Qwen2-7B: {len(teacher_kvs['Qwen2-7B'])} layers, "
          f"each K/V shape [{B}, {teacher1_length}, {teacher1_d_model}]")
    print(f"  Qwen2-1.5B: {len(teacher_kvs['Qwen2-1.5B'])} layers, "
          f"each K/V shape [{B}, {teacher2_length}, {teacher2_d_model}]")
    print("  ✓ Teacher KVs created")
    
    # Test 4: Complete alignment (Time + Layer + Dimension)
    print("\n[Test 4] Three-Stage Alignment")
    student_layer_idx = 5
    
    aligned_kvs = align_multi_teacher_kv_with_projection(
        teacher_kvs=teacher_kvs,
        student_layer_idx=student_layer_idx,
        student_length=student_length,
        layer_mapper=layer_mapper,
        projector=projector,
        use_segment_resampling=False  # Global resampling for now
    )
    
    print(f"  Aligned teachers: {list(aligned_kvs.keys())}")
    for teacher_name, (K, V) in aligned_kvs.items():
        print(f"    {teacher_name}: K {K.shape}, V {V.shape}")
        assert K.shape == (B, student_length, student_d_model), \
            f"Expected [{B}, {student_length}, {student_d_model}], got {K.shape}"
        assert V.shape == (B, student_length, student_d_model)
    print("  ✓ All dimensions aligned correctly!")
    
    # Test 5: Fuse multi-teacher KVs
    print("\n[Test 5] Fuse Multi-Teacher KVs")
    teacher_weights = {
        "Qwen2-7B": 0.7,
        "Qwen2-1.5B": 0.3
    }
    
    K_fused, V_fused = fuse_multi_teacher_kv(aligned_kvs, teacher_weights)
    print(f"  Fused K: {K_fused.shape}, V: {V_fused.shape}")
    assert K_fused.shape == (B, student_length, student_d_model)
    assert V_fused.shape == (B, student_length, student_d_model)
    print("  ✓ Multi-teacher fusion works!")
    
    # Test 6: With segment-aware resampling
    print("\n[Test 6] With Segment-Aware Resampling")
    
    # Define segments
    student_segments = [
        SequenceSegment("prompt", 0, 10, 10),
        SequenceSegment("reasoning", 10, 40, 30),
        SequenceSegment("answer", 40, 50, 10)
    ]
    
    teacher_segments = {
        "Qwen2-7B": [
            SequenceSegment("prompt", 0, 15, 15),
            SequenceSegment("reasoning", 15, 70, 55),
            SequenceSegment("answer", 70, 80, 10)
        ],
        "Qwen2-1.5B": [
            SequenceSegment("prompt", 0, 12, 12),
            SequenceSegment("reasoning", 12, 55, 43),
            SequenceSegment("answer", 55, 65, 10)
        ]
    }
    
    aligned_kvs_seg = align_multi_teacher_kv_with_projection(
        teacher_kvs=teacher_kvs,
        student_layer_idx=student_layer_idx,
        student_length=student_length,
        layer_mapper=layer_mapper,
        projector=projector,
        use_segment_resampling=True,
        teacher_segments=teacher_segments,
        student_segments=student_segments
    )
    
    for teacher_name, (K, V) in aligned_kvs_seg.items():
        assert K.shape == (B, student_length, student_d_model)
        assert V.shape == (B, student_length, student_d_model)
    
    print("  ✓ Segment-aware alignment works!")
    
    print("\n" + "=" * 80)
    print("✓ All complete alignment tests passed!")
    print("=" * 80)
    print("\nSummary:")
    print("  1. ✓ Layer alignment: CKA-based top-k mapping")
    print("  2. ✓ Time alignment: Segment-aware resampling")
    print("  3. ✓ Dimension alignment: Learnable projection")
    print("  4. ✓ Multi-teacher fusion: Weighted average")
    print("\nReady for integration into training script!")


if __name__ == "__main__":
    test_complete_alignment()
