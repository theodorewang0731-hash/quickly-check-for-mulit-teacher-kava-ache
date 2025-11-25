"""
Teacher Router Prototype Computation

计算教师原型特征，用于相似度路由。

原型特征：
1. 全局平均池化：mean(KV, dim=time)
2. CLS token：KV[:, 0, :]
3. 学习的聚类中心：K-means

Usage:
    proto = compute_teacher_prototype(kvs, method="mean")
    similarity = compute_similarity(student_hidden, proto)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans


def compute_teacher_prototype(
    kvs: torch.Tensor,
    method: str = "mean",
    attention_mask: Optional[torch.Tensor] = None,
    num_clusters: int = 8
) -> torch.Tensor:
    """
    计算教师原型特征。
    
    Args:
        kvs: [batch, time, dim] 或 [num_layers, batch, time, dim]
        method: 原型计算方法
            - "mean": 全局平均池化
            - "cls": CLS token（第一个 token）
            - "max": 最大池化
            - "kmeans": K-means 聚类中心
        attention_mask: [batch, time] mask（可选）
        num_clusters: K-means 聚类数（仅用于 kmeans 方法）
        
    Returns:
        prototype: [dim] 或 [num_clusters, dim]（kmeans）或 [num_layers, dim]（多层）
    """
    if kvs.ndim == 4:
        # [num_layers, batch, time, dim]
        # 递归处理每层
        prototypes = []
        for layer_kv in kvs:
            proto = compute_teacher_prototype(
                layer_kv,
                method=method,
                attention_mask=attention_mask,
                num_clusters=num_clusters
            )
            prototypes.append(proto)
        
        return torch.stack(prototypes, dim=0)  # [num_layers, dim] or [num_layers, num_clusters, dim]
    
    # [batch, time, dim]
    batch, time, dim = kvs.shape
    
    if method == "mean":
        # 全局平均池化
        if attention_mask is not None:
            # 加权平均（忽略 padding）
            mask_expanded = attention_mask.unsqueeze(-1).float()  # [batch, time, 1]
            kvs_masked = kvs * mask_expanded
            prototype = kvs_masked.sum(dim=(0, 1)) / mask_expanded.sum(dim=(0, 1))
        else:
            prototype = kvs.mean(dim=(0, 1))
    
    elif method == "cls":
        # CLS token（第一个 token）
        prototype = kvs[:, 0, :].mean(dim=0)
    
    elif method == "max":
        # 最大池化
        if attention_mask is not None:
            # 将 padding 位置设为 -inf
            mask_expanded = attention_mask.unsqueeze(-1).float()
            kvs_masked = kvs.clone()
            kvs_masked[mask_expanded == 0] = -float('inf')
            prototype = kvs_masked.max(dim=1)[0].max(dim=0)[0]
        else:
            prototype = kvs.max(dim=1)[0].max(dim=0)[0]
    
    elif method == "kmeans":
        # K-means 聚类
        # Flatten to [batch * time, dim]
        kvs_flat = kvs.reshape(-1, dim).cpu().numpy()
        
        if attention_mask is not None:
            # 只使用有效 token
            mask_flat = attention_mask.reshape(-1).cpu().numpy()
            kvs_flat = kvs_flat[mask_flat == 1]
        
        # K-means
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        kmeans.fit(kvs_flat)
        
        # 聚类中心作为原型
        prototype = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)  # [num_clusters, dim]
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return prototype


def compute_similarity(
    query: torch.Tensor,
    prototype: torch.Tensor,
    method: str = "cosine"
) -> torch.Tensor:
    """
    计算 query 和 prototype 的相似度。
    
    Args:
        query: [batch, dim] 或 [batch, time, dim]
        prototype: [dim] 或 [num_prototypes, dim]
        method: 相似度方法
            - "cosine": 余弦相似度
            - "dot": 点积
            - "l2": 负 L2 距离
        
    Returns:
        similarity: [batch] 或 [batch, num_prototypes] 或 [batch, time, num_prototypes]
    """
    if prototype.ndim == 1:
        # [dim] -> [1, dim]
        prototype = prototype.unsqueeze(0)
    
    if query.ndim == 2:
        # [batch, dim]
        if method == "cosine":
            query_norm = F.normalize(query, dim=-1)
            proto_norm = F.normalize(prototype, dim=-1)
            similarity = torch.matmul(query_norm, proto_norm.T)  # [batch, num_prototypes]
        
        elif method == "dot":
            similarity = torch.matmul(query, prototype.T)  # [batch, num_prototypes]
        
        elif method == "l2":
            # 负 L2 距离（越近越好）
            query_expanded = query.unsqueeze(1)  # [batch, 1, dim]
            proto_expanded = prototype.unsqueeze(0)  # [1, num_prototypes, dim]
            similarity = -torch.norm(query_expanded - proto_expanded, dim=-1)  # [batch, num_prototypes]
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    elif query.ndim == 3:
        # [batch, time, dim]
        batch, time, dim = query.shape
        
        if method == "cosine":
            query_norm = F.normalize(query, dim=-1)  # [batch, time, dim]
            proto_norm = F.normalize(prototype, dim=-1)  # [num_prototypes, dim]
            similarity = torch.matmul(query_norm, proto_norm.T)  # [batch, time, num_prototypes]
        
        elif method == "dot":
            similarity = torch.matmul(query, prototype.T)  # [batch, time, num_prototypes]
        
        elif method == "l2":
            query_expanded = query.unsqueeze(2)  # [batch, time, 1, dim]
            proto_expanded = prototype.unsqueeze(0).unsqueeze(0)  # [1, 1, num_prototypes, dim]
            similarity = -torch.norm(query_expanded - proto_expanded, dim=-1)  # [batch, time, num_prototypes]
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    else:
        raise ValueError(f"Unsupported query shape: {query.shape}")
    
    # Squeeze if only one prototype
    if prototype.size(0) == 1:
        similarity = similarity.squeeze(-1)
    
    return similarity


def compute_multi_teacher_prototypes(
    teacher_kvs_list: List[torch.Tensor],
    method: str = "mean",
    attention_masks: Optional[List[torch.Tensor]] = None
) -> List[torch.Tensor]:
    """
    计算多个教师的原型。
    
    Args:
        teacher_kvs_list: List of [batch, time, dim] 或 [num_layers, batch, time, dim]
        method: 原型计算方法
        attention_masks: List of [batch, time] masks（可选）
        
    Returns:
        prototypes: List of prototype tensors
    """
    prototypes = []
    
    for i, kvs in enumerate(teacher_kvs_list):
        mask = attention_masks[i] if attention_masks else None
        proto = compute_teacher_prototype(kvs, method=method, attention_mask=mask)
        prototypes.append(proto)
    
    return prototypes


def compute_routing_weights(
    query: torch.Tensor,
    prototypes: List[torch.Tensor],
    temperature: float = 1.0,
    method: str = "cosine"
) -> torch.Tensor:
    """
    计算路由权重（相似度 + softmax）。
    
    Args:
        query: [batch, dim] 或 [batch, time, dim]
        prototypes: List of [dim] prototype tensors
        temperature: softmax 温度
        method: 相似度方法
        
    Returns:
        weights: [batch, num_teachers] 或 [batch, time, num_teachers]
    """
    num_teachers = len(prototypes)
    
    # 计算与每个教师的相似度
    similarities = []
    for proto in prototypes:
        sim = compute_similarity(query, proto, method=method)
        similarities.append(sim)
    
    # Stack to [batch, num_teachers] 或 [batch, time, num_teachers]
    similarities = torch.stack(similarities, dim=-1)
    
    # Softmax with temperature
    weights = F.softmax(similarities / temperature, dim=-1)
    
    return weights


if __name__ == "__main__":
    # 测试代码
    print("Testing teacher prototype computation...")
    
    batch_size = 4
    time_steps = 10
    hidden_dim = 64
    num_layers = 6
    
    # 测试单层原型
    kvs = torch.randn(batch_size, time_steps, hidden_dim)
    attention_mask = torch.ones(batch_size, time_steps, dtype=torch.bool)
    attention_mask[:, 7:] = False  # 模拟 padding
    
    # Mean prototype
    proto_mean = compute_teacher_prototype(kvs, method="mean", attention_mask=attention_mask)
    assert proto_mean.shape == (hidden_dim,)
    print("✓ Mean prototype test passed")
    
    # CLS prototype
    proto_cls = compute_teacher_prototype(kvs, method="cls")
    assert proto_cls.shape == (hidden_dim,)
    print("✓ CLS prototype test passed")
    
    # Max prototype
    proto_max = compute_teacher_prototype(kvs, method="max", attention_mask=attention_mask)
    assert proto_max.shape == (hidden_dim,)
    print("✓ Max prototype test passed")
    
    # K-means prototype
    proto_kmeans = compute_teacher_prototype(kvs, method="kmeans", num_clusters=4)
    assert proto_kmeans.shape == (4, hidden_dim)
    print("✓ K-means prototype test passed")
    
    # 测试多层原型
    kvs_multi = torch.randn(num_layers, batch_size, time_steps, hidden_dim)
    proto_multi = compute_teacher_prototype(kvs_multi, method="mean", attention_mask=attention_mask)
    assert proto_multi.shape == (num_layers, hidden_dim)
    print("✓ Multi-layer prototype test passed")
    
    # 测试相似度计算
    query = torch.randn(batch_size, hidden_dim)
    
    # Cosine similarity
    sim_cos = compute_similarity(query, proto_mean, method="cosine")
    assert sim_cos.shape == (batch_size,)
    print("✓ Cosine similarity test passed")
    
    # Dot product
    sim_dot = compute_similarity(query, proto_mean, method="dot")
    assert sim_dot.shape == (batch_size,)
    print("✓ Dot product similarity test passed")
    
    # L2 distance
    sim_l2 = compute_similarity(query, proto_mean, method="l2")
    assert sim_l2.shape == (batch_size,)
    print("✓ L2 distance test passed")
    
    # 测试多原型相似度
    sim_multi = compute_similarity(query, proto_kmeans, method="cosine")
    assert sim_multi.shape == (batch_size, 4)
    print("✓ Multi-prototype similarity test passed")
    
    # 测试序列 query
    query_seq = torch.randn(batch_size, time_steps, hidden_dim)
    sim_seq = compute_similarity(query_seq, proto_mean, method="cosine")
    assert sim_seq.shape == (batch_size, time_steps)
    print("✓ Sequence query similarity test passed")
    
    # 测试路由权重
    prototypes = [torch.randn(hidden_dim) for _ in range(3)]
    weights = compute_routing_weights(query, prototypes, temperature=1.0, method="cosine")
    assert weights.shape == (batch_size, 3)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size), atol=1e-6)
    print("✓ Routing weights test passed")
    
    # 测试多教师原型
    teacher_kvs_list = [
        torch.randn(batch_size, time_steps, hidden_dim),
        torch.randn(batch_size, time_steps, hidden_dim),
        torch.randn(batch_size, time_steps, hidden_dim)
    ]
    masks = [attention_mask] * 3
    multi_protos = compute_multi_teacher_prototypes(teacher_kvs_list, method="mean", attention_masks=masks)
    assert len(multi_protos) == 3
    for proto in multi_protos:
        assert proto.shape == (hidden_dim,)
    print("✓ Multi-teacher prototype test passed")
    
    print("All teacher prototype tests passed!")
