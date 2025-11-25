"""
Fuse Module Init

导入 KV 融合和路由模块。
"""

from .fuse_kv import (
    fuse_kvs_fixed,
    fuse_kvs_similarity,
    fuse_kvs_learnable,
    LearnableRouter,
    EntropyRegularizer
)


__all__ = [
    "fuse_kvs_fixed",
    "fuse_kvs_similarity",
    "fuse_kvs_learnable",
    "LearnableRouter",
    "EntropyRegularizer",
]
