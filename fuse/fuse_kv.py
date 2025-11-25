"""
KV Fusion Module

多教师 KV 融合策略。

支持三种融合方式：
1. Fixed weights: 固定权重加权平均
2. Similarity routing: 基于相似度的动态路由
3. Learnable routing: 学习的 MLP/Gate 路由

Usage:
    # Fixed fusion
    fused_kv = fuse_kvs_fixed(kvs_list, weights=[0.5, 0.3, 0.2])
    
    # Similarity fusion
    fused_kv = fuse_kvs_similarity(kvs_list, query=student_hidden, prototypes=teacher_protos)
    
    # Learnable fusion
    router = LearnableRouter(hidden_dim=512, num_teachers=3)
    fused_kv = fuse_kvs_learnable(kvs_list, query=student_hidden, router=router)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


def fuse_kvs_fixed(
    kvs_list: List[torch.Tensor],
    weights: Optional[List[float]] = None
) -> torch.Tensor:
    """
    固定权重融合多个教师 KV。
    
    Args:
        kvs_list: List of [batch, time, dim] teacher KV tensors
        weights: 融合权重（可选，默认均匀）
        
    Returns:
        fused_kv: [batch, time, dim] 融合后的 KV
    """
    num_teachers = len(kvs_list)
    
    if weights is None:
        weights = [1.0 / num_teachers] * num_teachers
    
    # 归一化权重
    weights = torch.tensor(weights, dtype=torch.float32, device=kvs_list[0].device)
    weights = weights / weights.sum()
    
    # 加权求和
    fused_kv = None
    for kv, weight in zip(kvs_list, weights):
        if fused_kv is None:
            fused_kv = kv * weight
        else:
            fused_kv = fused_kv + kv * weight
    
    return fused_kv


def fuse_kvs_similarity(
    kvs_list: List[torch.Tensor],
    query: torch.Tensor,
    prototypes: List[torch.Tensor],
    temperature: float = 1.0,
    similarity_method: str = "cosine"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    基于相似度的动态融合。
    
    Args:
        kvs_list: List of [batch, time, dim] teacher KV tensors
        query: [batch, dim] 或 [batch, time, dim] 学生查询
        prototypes: List of [dim] 教师原型
        temperature: softmax 温度
        similarity_method: 相似度计算方法
        
    Returns:
        fused_kv: [batch, time, dim] 融合后的 KV
        weights: [batch, num_teachers] 或 [batch, time, num_teachers] 路由权重
    """
    from teacher.router_proto import compute_routing_weights
    
    # 计算路由权重
    weights = compute_routing_weights(
        query,
        prototypes,
        temperature=temperature,
        method=similarity_method
    )
    
    # 加权融合
    # weights: [batch, num_teachers] 或 [batch, time, num_teachers]
    # kvs: [batch, time, dim]
    
    if weights.ndim == 2:
        # [batch, num_teachers] -> [batch, 1, num_teachers]
        weights = weights.unsqueeze(1)
    
    # Stack KVs: [batch, time, dim, num_teachers]
    kvs_stacked = torch.stack(kvs_list, dim=-1)
    
    # Weighted sum: [batch, time, dim]
    # weights: [batch, time, num_teachers, 1]
    weights_expanded = weights.unsqueeze(-2)  # [batch, time, 1, num_teachers]
    fused_kv = (kvs_stacked * weights_expanded.transpose(-1, -2)).sum(dim=-1)
    
    return fused_kv, weights.squeeze(1) if weights.size(1) == 1 else weights


def fuse_kvs_learnable(
    kvs_list: List[torch.Tensor],
    query: torch.Tensor,
    router: nn.Module
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用可学习路由器融合。
    
    Args:
        kvs_list: List of [batch, time, dim] teacher KV tensors
        query: [batch, dim] 或 [batch, time, dim] 学生查询
        router: 路由器模块（输出权重）
        
    Returns:
        fused_kv: [batch, time, dim] 融合后的 KV
        weights: [batch, num_teachers] 或 [batch, time, num_teachers] 路由权重
    """
    # 计算路由权重
    weights = router(query)  # [batch, num_teachers] 或 [batch, time, num_teachers]
    
    # 加权融合（同 similarity 方法）
    if weights.ndim == 2:
        weights = weights.unsqueeze(1)
    
    kvs_stacked = torch.stack(kvs_list, dim=-1)
    weights_expanded = weights.unsqueeze(-2)
    fused_kv = (kvs_stacked * weights_expanded.transpose(-1, -2)).sum(dim=-1)
    
    return fused_kv, weights.squeeze(1) if weights.size(1) == 1 else weights


class LearnableRouter(nn.Module):
    """
    可学习的路由器（MLP + Gating）。
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_teachers: int,
        router_type: str = "mlp",
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            hidden_dim: 输入维度
            num_teachers: 教师数量
            router_type: 路由器类型 ("mlp", "gate", "attention")
            num_layers: MLP 层数
            dropout: Dropout 率
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_teachers = num_teachers
        self.router_type = router_type
        
        if router_type == "mlp":
            # MLP router
            layers = []
            for i in range(num_layers):
                in_dim = hidden_dim if i == 0 else hidden_dim // 2
                out_dim = hidden_dim // 2 if i < num_layers - 1 else num_teachers
                layers.append(nn.Linear(in_dim, out_dim))
                if i < num_layers - 1:
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout))
            
            self.router = nn.Sequential(*layers)
        
        elif router_type == "gate":
            # Gating network (简单的单层)
            self.router = nn.Linear(hidden_dim, num_teachers)
        
        elif router_type == "attention":
            # Attention-based router
            self.query_proj = nn.Linear(hidden_dim, hidden_dim)
            self.key_proj = nn.Linear(hidden_dim, hidden_dim)
            self.teacher_embeddings = nn.Parameter(torch.randn(num_teachers, hidden_dim))
            nn.init.normal_(self.teacher_embeddings, std=0.02)
        
        else:
            raise ValueError(f"Unknown router type: {router_type}")
    
    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """
        计算路由权重。
        
        Args:
            query: [batch, dim] 或 [batch, time, dim]
            
        Returns:
            weights: [batch, num_teachers] 或 [batch, time, num_teachers]
        """
        if self.router_type == "attention":
            # Attention-based routing
            q = self.query_proj(query)  # [batch, dim] or [batch, time, dim]
            k = self.key_proj(self.teacher_embeddings)  # [num_teachers, dim]
            
            # Compute attention scores
            if q.ndim == 2:
                # [batch, dim] @ [dim, num_teachers] = [batch, num_teachers]
                scores = torch.matmul(q, k.T)
            else:
                # [batch, time, dim] @ [dim, num_teachers] = [batch, time, num_teachers]
                scores = torch.matmul(q, k.T)
            
            # Softmax
            weights = F.softmax(scores, dim=-1)
        
        else:
            # MLP or Gate
            logits = self.router(query)
            weights = F.softmax(logits, dim=-1)
        
        return weights


class EntropyRegularizer:
    """
    路由权重的熵正则化（鼓励专业化或多样化）。
    """
    
    def __init__(self, target: str = "diverse", strength: float = 0.01):
        """
        Args:
            target: "diverse" (高熵) 或 "specialized" (低熵)
            strength: 正则化强度
        """
        self.target = target
        self.strength = strength
    
    def compute_loss(self, weights: torch.Tensor) -> torch.Tensor:
        """
        计算熵正则化损失。
        
        Args:
            weights: [batch, num_teachers] 或 [batch, time, num_teachers]
            
        Returns:
            loss: 标量损失
        """
        # 计算熵
        entropy = -(weights * (weights + 1e-10).log()).sum(dim=-1).mean()
        
        if self.target == "diverse":
            # 鼓励高熵（惩罚低熵）
            max_entropy = torch.log(torch.tensor(weights.size(-1), dtype=torch.float32))
            loss = self.strength * (max_entropy - entropy)
        
        elif self.target == "specialized":
            # 鼓励低熵（惩罚高熵）
            loss = self.strength * entropy
        
        else:
            raise ValueError(f"Unknown target: {self.target}")
        
        return loss


if __name__ == "__main__":
    # 测试代码
    print("Testing KV fusion...")
    
    batch_size = 4
    time_steps = 10
    hidden_dim = 64
    num_teachers = 3
    
    # 创建测试数据
    kvs_list = [torch.randn(batch_size, time_steps, hidden_dim) for _ in range(num_teachers)]
    query = torch.randn(batch_size, hidden_dim)
    
    # 测试固定权重融合
    fused_fixed = fuse_kvs_fixed(kvs_list, weights=[0.5, 0.3, 0.2])
    assert fused_fixed.shape == (batch_size, time_steps, hidden_dim)
    print("✓ Fixed fusion test passed")
    
    # 测试相似度融合
    prototypes = [torch.randn(hidden_dim) for _ in range(num_teachers)]
    fused_sim, weights_sim = fuse_kvs_similarity(kvs_list, query, prototypes)
    assert fused_sim.shape == (batch_size, time_steps, hidden_dim)
    assert weights_sim.shape == (batch_size, num_teachers)
    assert torch.allclose(weights_sim.sum(dim=-1), torch.ones(batch_size), atol=1e-6)
    print("✓ Similarity fusion test passed")
    
    # 测试可学习路由器
    router = LearnableRouter(hidden_dim, num_teachers, router_type="mlp")
    fused_learn, weights_learn = fuse_kvs_learnable(kvs_list, query, router)
    assert fused_learn.shape == (batch_size, time_steps, hidden_dim)
    assert weights_learn.shape == (batch_size, num_teachers)
    assert torch.allclose(weights_learn.sum(dim=-1), torch.ones(batch_size), atol=1e-6)
    print("✓ Learnable fusion (MLP) test passed")
    
    # 测试 Gate router
    router_gate = LearnableRouter(hidden_dim, num_teachers, router_type="gate")
    fused_gate, weights_gate = fuse_kvs_learnable(kvs_list, query, router_gate)
    assert fused_gate.shape == (batch_size, time_steps, hidden_dim)
    print("✓ Learnable fusion (Gate) test passed")
    
    # 测试 Attention router
    router_attn = LearnableRouter(hidden_dim, num_teachers, router_type="attention")
    fused_attn, weights_attn = fuse_kvs_learnable(kvs_list, query, router_attn)
    assert fused_attn.shape == (batch_size, time_steps, hidden_dim)
    print("✓ Learnable fusion (Attention) test passed")
    
    # 测试序列 query
    query_seq = torch.randn(batch_size, time_steps, hidden_dim)
    fused_seq, weights_seq = fuse_kvs_similarity(kvs_list, query_seq, prototypes)
    assert fused_seq.shape == (batch_size, time_steps, hidden_dim)
    assert weights_seq.shape == (batch_size, time_steps, num_teachers)
    print("✓ Sequence query fusion test passed")
    
    # 测试熵正则化
    regularizer_diverse = EntropyRegularizer(target="diverse", strength=0.01)
    loss_diverse = regularizer_diverse.compute_loss(weights_sim)
    assert loss_diverse.ndim == 0  # scalar
    print("✓ Entropy regularization (diverse) test passed")
    
    regularizer_spec = EntropyRegularizer(target="specialized", strength=0.01)
    loss_spec = regularizer_spec.compute_loss(weights_sim)
    assert loss_spec.ndim == 0
    print("✓ Entropy regularization (specialized) test passed")
    
    # 测试权重归一化
    # 创建不均匀的权重
    weights_uneven = torch.tensor([[0.8, 0.15, 0.05], [0.33, 0.33, 0.34]], dtype=torch.float32)
    regularizer = EntropyRegularizer(target="specialized", strength=0.1)
    loss = regularizer.compute_loss(weights_uneven)
    print(f"Entropy loss for specialized (uneven weights): {loss.item():.4f}")
    
    weights_even = torch.tensor([[0.33, 0.33, 0.34], [0.33, 0.33, 0.34]], dtype=torch.float32)
    loss_even = regularizer.compute_loss(weights_even)
    print(f"Entropy loss for specialized (even weights): {loss_even.item():.4f}")
    assert loss < loss_even  # 专业化目标下，不均匀权重应有更低损失
    print("✓ Entropy regularization behavior test passed")
    
    print("All KV fusion tests passed!")
