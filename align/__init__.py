"""
Alignment Module Init

导入所有对齐模块，提供统一接口。
"""

from .tokenizer_align import (
    char_ranges_from_tokens,
    compute_iou,
    build_char_align_matrix,
    apply_char_alignment,
    visualize_alignment
)

from .time_align import (
    pad_to_length,
    apply_mask_to_kv,
    apply_soft_alignment,
    create_attention_mask,
    align_sequence_lengths
)

from .layer_map import (
    build_layer_mapping,
    interpolate_teacher_layers,
    visualize_layer_mapping,
    build_multi_teacher_layer_map,
    merge_multi_teacher_kvs
)

from .head_dim_adapter import (
    HeadDimAdapter,
    MultiTeacherHeadDimAdapter
)

from .rope_scale import (
    RoPEScaler,
    MultiTeacherRoPEScaler,
    interpolate_rope_embeddings
)


__all__ = [
    # tokenizer_align
    "char_ranges_from_tokens",
    "compute_iou",
    "build_char_align_matrix",
    "apply_char_alignment",
    "visualize_alignment",
    
    # time_align
    "pad_to_length",
    "apply_mask_to_kv",
    "apply_soft_alignment",
    "create_attention_mask",
    "align_sequence_lengths",
    
    # layer_map
    "build_layer_mapping",
    "interpolate_teacher_layers",
    "visualize_layer_mapping",
    "build_multi_teacher_layer_map",
    "merge_multi_teacher_kvs",
    
    # head_dim_adapter
    "HeadDimAdapter",
    "MultiTeacherHeadDimAdapter",
    
    # rope_scale
    "RoPEScaler",
    "MultiTeacherRoPEScaler",
    "interpolate_rope_embeddings",
]
