"""
Teacher Module Init

导入教师 KV 提取和原型计算模块。
"""

from .extract_teacher_kv import (
    TeacherKVExtractor,
    extract_from_dataset
)

from .router_proto import (
    compute_teacher_prototype,
    compute_similarity,
    compute_multi_teacher_prototypes,
    compute_routing_weights
)


__all__ = [
    "TeacherKVExtractor",
    "extract_from_dataset",
    "compute_teacher_prototype",
    "compute_similarity",
    "compute_multi_teacher_prototypes",
    "compute_routing_weights",
]
