"""
KV Dimension Projector for Multi-Teacher Distillation
======================================================

Purpose:
    Project teacher KV cache from d_teacher to d_student using learnable linear layers.
    
Key Design:
    - Per-teacher shared projection: W_K^(teacher), W_V^(teacher) ∈ R^(d_t × d_s)
    - All layers of same teacher share same projection weights (reduce params)
    - Trainable during distillation or frozen with simple initialization
    - Head dimension is flattened first (H * d_head = d_model)

Math:
    K_aligned = K_teacher · W_K    shape: [B, L, T, d_t] -> [B, L, T, d_s]
    V_aligned = V_teacher · W_V    shape: [B, L, T, d_t] -> [B, L, T, d_s]

Author: Quick Check Team
Date: 2025-01-18
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List
import json


class KVDimensionProjector(nn.Module):
    """
    Elastic Bottleneck Projector for aligning teacher KV dimensions to student.
    
    Design Philosophy:
        - Per-teacher granularity: Each teacher has its own adapter
        - Elastic MLP: Adjustable width based on model size (<30B use 1.0, ~70B use 2.0)
        - Pre-LayerNorm: Essential for numerical stability across model sizes
        - Non-linear: SiLU activation for better feature transformation
        
    Architecture:
        Pre-LayerNorm -> Linear(d_t, hidden) -> SiLU -> Dropout -> Linear(hidden, d_s)
        
    Args:
        teacher_configs: Dict mapping teacher_name -> {"d_model": int, "num_layers": int}
        student_d_model: Student hidden dimension
        mlp_ratio: Hidden layer expansion ratio (0.5=compress, 1.0=same, 2.0=expand)
                   Recommended: <30B use 1.0, ~70B use 2.0
        dropout: Dropout rate for regularization
        init_method: "xavier" | "kaiming" | "normal"
        trainable: Whether to train projections during distillation
    """
    
    def __init__(
        self,
        teacher_configs: Dict[str, Dict[str, int]],
        student_d_model: int,
        mlp_ratio: float = 1.0,
        dropout: float = 0.1,
        init_method: str = "xavier",
        trainable: bool = True
    ):
        super().__init__()
        
        self.teacher_configs = teacher_configs
        self.student_d_model = student_d_model
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.init_method = init_method
        self.trainable = trainable
        
        # Create adapters for each teacher
        self.projectors = nn.ModuleDict()
        self.teacher_name_mapping = {}  # Map clean names to original names
        
        print(f"\n[Initializing Elastic Bottleneck Projector]")
        print(f"   MLP Ratio: {mlp_ratio}x | Dropout: {dropout} | Output: {student_d_model}")
        
        for teacher_name, config in teacher_configs.items():
            teacher_d_model = config["d_model"]
            
            # Create clean name for nn.Module (can't contain "." or "-")
            clean_name = teacher_name.replace(".", "_").replace("-", "_")
            self.teacher_name_mapping[clean_name] = teacher_name
            
            # Calculate hidden dimension based on MLP ratio
            # 70B models: use 2.0x for complex features
            # 7B-14B models: use 1.0x or 0.5x for efficiency
            hidden_dim = int(teacher_d_model * mlp_ratio)
            
            # Create K and V adapters with Elastic Bottleneck architecture
            adapter_K = nn.Sequential(
                # [Critical 1] Pre-LayerNorm: Stabilize gradients regardless of model size
                nn.LayerNorm(teacher_d_model),
                
                # [Critical 2] Transformation layers (Up/Down projection)
                nn.Linear(teacher_d_model, hidden_dim),
                
                # [Critical 3] Non-linear activation (SiLU)
                nn.SiLU(),
                
                nn.Dropout(dropout),
                
                # Final projection to student dimension
                nn.Linear(hidden_dim, student_d_model)
            )
            
            adapter_V = nn.Sequential(
                nn.LayerNorm(teacher_d_model),
                nn.Linear(teacher_d_model, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, student_d_model)
            )
            
            # Set trainable
            for p in adapter_K.parameters():
                p.requires_grad_(trainable)
            for p in adapter_V.parameters():
                p.requires_grad_(trainable)
            
            self.projectors[clean_name] = nn.ModuleDict({
                "K": adapter_K,
                "V": adapter_V
            })
            
            print(f"   Teacher: {teacher_name}")
            print(f"     d_model: {teacher_d_model} -> hidden: {hidden_dim} -> output: {student_d_model}")
        
        # Initialize weights
        if trainable:
            self._initialize_weights()
        
        print(f"\n[Elastic Bottleneck Projector Initialized]")
        print(f"  Teachers: {list(teacher_configs.keys())}")
        print(f"  Student d_model: {student_d_model}")
        print(f"  MLP ratio: {mlp_ratio}x")
        print(f"  Init method: {init_method}")
        print(f"  Trainable: {trainable}")
        print(f"  Total params: {self.count_parameters():,}")
    
    def _initialize_weights(self):
        """Initialize weights based on init_method (Xavier/Kaiming/Normal)."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if self.init_method == "xavier":
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                
                elif self.init_method == "kaiming":
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                
                elif self.init_method == "normal":
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                
                else:
                    # Default: Xavier
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def project_teacher_kv(
        self,
        teacher_name: str,
        teacher_K: torch.Tensor,
        teacher_V: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project teacher KV to student dimension.
        
        Args:
            teacher_name: Which teacher (e.g., "Qwen2-7B")
            teacher_K: shape [B, num_layers, T, d_teacher]
            teacher_V: shape [B, num_layers, T, d_teacher]
        
        Returns:
            (K_aligned, V_aligned): Both [B, num_layers, T, d_student]
        """
        
        # Map to clean name
        clean_name = teacher_name.replace(".", "_").replace("-", "_")
        
        if clean_name not in self.projectors:
            raise ValueError(
                f"Teacher {teacher_name} (clean: {clean_name}) not found. "
                f"Available: {list(self.teacher_name_mapping.values())}"
            )
        
        proj_K = self.projectors[clean_name]["K"]
        proj_V = self.projectors[clean_name]["V"]
        
        # Project: [B, L, T, d_t] @ [d_t, d_s] -> [B, L, T, d_s]
        K_aligned = proj_K(teacher_K)
        V_aligned = proj_V(teacher_V)
        
        return K_aligned, V_aligned
    
    def project_multi_teacher_kv(
        self,
        teacher_kvs: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Project multiple teachers' KV to student dimension.
        
        Args:
            teacher_kvs: Dict mapping teacher_name -> (K, V)
                where K, V are [B, num_layers, T, d_teacher]
        
        Returns:
            Dict mapping teacher_name -> (K_aligned, V_aligned)
                where aligned shapes are [B, num_layers, T, d_student]
        """
        
        aligned_kvs = {}
        
        for teacher_name, (K, V) in teacher_kvs.items():
            K_aligned, V_aligned = self.project_teacher_kv(teacher_name, K, V)
            aligned_kvs[teacher_name] = (K_aligned, V_aligned)
        
        return aligned_kvs
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_projection_weights(self, teacher_name: str) -> Dict[str, torch.Tensor]:
        """Get projection weights for analysis."""
        clean_name = teacher_name.replace(".", "_").replace("-", "_")
        
        if clean_name not in self.projectors:
            raise ValueError(f"Teacher {teacher_name} not found")
        
        return {
            "W_K": self.projectors[clean_name]["K"].weight.detach().clone(),
            "W_V": self.projectors[clean_name]["V"].weight.detach().clone()
        }
    
    def save_projections(self, path: str):
        """Save projection weights to file."""
        state = {
            "teacher_configs": self.teacher_configs,
            "student_d_model": self.student_d_model,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
            "init_method": self.init_method,
            "trainable": self.trainable,
            "state_dict": self.state_dict()
        }
        torch.save(state, path)
        print(f"[Saved projections to {path}]")
    
    def load_projections(self, path: str):
        """Load projection weights from file."""
        state = torch.load(path, map_location="cpu")
        self.load_state_dict(state["state_dict"])
        print(f"[Loaded projections from {path}]")
    
    @staticmethod
    def from_pretrained(path: str) -> "KVDimensionProjector":
        """Load projector from saved checkpoint."""
        state = torch.load(path, map_location="cpu")
        projector = KVDimensionProjector(
            teacher_configs=state["teacher_configs"],
            student_d_model=state["student_d_model"],
            mlp_ratio=state.get("mlp_ratio", 1.0),  # Backward compatibility
            dropout=state.get("dropout", 0.1),
            init_method=state["init_method"],
            trainable=state["trainable"]
        )
        projector.load_state_dict(state["state_dict"])
        return projector


def flatten_kv_heads(kv: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """
    Flatten head dimension: [B, L, H, T, d_head] -> [B, L, T, d_model]
    
    This allows us to treat KV as a single vector space instead of
    worrying about head-to-head alignment.
    
    Args:
        kv: shape [B, num_layers, num_heads, seq_len, head_dim]
        num_heads: Number of attention heads
        head_dim: Dimension per head
    
    Returns:
        Flattened KV: [B, num_layers, seq_len, d_model]
            where d_model = num_heads * head_dim
    """
    B, L, H, T, d_head = kv.shape
    assert H == num_heads and d_head == head_dim, \
        f"Shape mismatch: got H={H}, d_head={d_head}, expected {num_heads}, {head_dim}"
    
    # Reshape: [B, L, H, T, d_head] -> [B, L, T, H, d_head] -> [B, L, T, H*d_head]
    kv_flat = kv.transpose(2, 3).contiguous()  # [B, L, T, H, d_head]
    kv_flat = kv_flat.view(B, L, T, H * d_head)  # [B, L, T, d_model]
    
    return kv_flat


def unflatten_kv_heads(kv_flat: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """
    Unflatten head dimension: [B, L, T, d_model] -> [B, L, H, T, d_head]
    
    Inverse of flatten_kv_heads, used after projection if needed.
    
    Args:
        kv_flat: shape [B, num_layers, seq_len, d_model]
        num_heads: Number of attention heads
        head_dim: Dimension per head
    
    Returns:
        Unflattened KV: [B, num_layers, num_heads, seq_len, head_dim]
    """
    B, L, T, d_model = kv_flat.shape
    assert d_model == num_heads * head_dim, \
        f"d_model={d_model} != num_heads * head_dim = {num_heads * head_dim}"
    
    # Reshape: [B, L, T, d_model] -> [B, L, T, H, d_head] -> [B, L, H, T, d_head]
    kv_unflat = kv_flat.view(B, L, T, num_heads, head_dim)
    kv_unflat = kv_unflat.transpose(2, 3).contiguous()  # [B, L, H, T, d_head]
    
    return kv_unflat


# ============================================================================
# Integration with Alignment v2
# ============================================================================

def align_teacher_kv_with_projection(
    teacher_kvs: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    student_length: int,
    student_num_layers: int,
    projector: KVDimensionProjector,
    layer_mapper: Optional[object] = None,
    use_segment_resampling: bool = False,
    teacher_segments: Optional[Dict[str, List]] = None,
    student_segments: Optional[List] = None
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Complete alignment pipeline with dimension projection.
    
    Pipeline:
        1. Time alignment: Resample to student sequence length
        2. Layer alignment: Map to student layers (if layer_mapper provided)
        3. Dimension projection: Project d_teacher -> d_student
    
    Args:
        teacher_kvs: Dict mapping teacher_name -> (K, V)
            Original shapes: [B, L_t, T_t, d_t] or [B, L_t, H_t, T_t, d_head_t]
        student_length: Target sequence length
        student_num_layers: Target number of layers
        projector: KVDimensionProjector instance
        layer_mapper: CKALayerMapper (optional)
        use_segment_resampling: Whether to use segment-aware time alignment
        teacher_segments: Segments for each teacher (if segment resampling)
        student_segments: Student segments (if segment resampling)
    
    Returns:
        Dict mapping teacher_name -> (K_aligned, V_aligned)
            Final shapes: [B, student_num_layers, student_length, d_student]
    """
    from experiments.alignment_v2 import (
        resample_kv_with_interpolation,
        align_multi_teacher_kv_v2
    )
    
    # Step 1 & 2: Time + Layer alignment
    if layer_mapper is not None:
        # Use full alignment v2
        time_layer_aligned = align_multi_teacher_kv_v2(
            teacher_kvs=teacher_kvs,
            student_length=student_length,
            layer_mapper=layer_mapper,
            use_segment_resampling=use_segment_resampling,
            teacher_segments=teacher_segments,
            student_segments=student_segments
        )
    else:
        # Only time alignment (no layer mapping)
        time_layer_aligned = {}
        for teacher_name, (K_t, V_t) in teacher_kvs.items():
            # Ensure 4D: [B, L, T, d]
            if K_t.dim() == 5:  # [B, L, H, T, d_head]
                K_t = flatten_kv_heads(K_t, K_t.size(2), K_t.size(4))
                V_t = flatten_kv_heads(V_t, V_t.size(2), V_t.size(4))
            
            # Time resampling
            teacher_segs = teacher_segments.get(teacher_name) if teacher_segments else None
            K_resampled = resample_kv_with_interpolation(
                K_t, student_length, teacher_segs, student_segments
            )
            V_resampled = resample_kv_with_interpolation(
                V_t, student_length, teacher_segs, student_segments
            )
            time_layer_aligned[teacher_name] = (K_resampled, V_resampled)
    
    # Step 3: Dimension projection
    final_aligned = projector.project_multi_teacher_kv(time_layer_aligned)
    
    return final_aligned


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing KV Dimension Projector...")
    print("=" * 80)
    
    # Test 1: Create projector for multi-teacher setup
    print("\n[Test 1] Multi-teacher projector initialization")
    teacher_configs = {
        "Qwen2-7B": {"d_model": 3584, "num_layers": 28},
        "Qwen2-1.5B": {"d_model": 1536, "num_layers": 28}
    }
    student_d_model = 2048
    
    projector = KVDimensionProjector(
        teacher_configs=teacher_configs,
        student_d_model=student_d_model,
        init_method="xavier",
        trainable=True
    )
    
    print(f"✓ Projector created with {len(projector.projectors)} teachers")
    
    # Test 2: Project single teacher
    print("\n[Test 2] Project single teacher KV")
    B, L, T = 2, 28, 50
    d_teacher = 3584
    
    K_teacher = torch.randn(B, L, T, d_teacher)
    V_teacher = torch.randn(B, L, T, d_teacher)
    
    K_aligned, V_aligned = projector.project_teacher_kv("Qwen2-7B", K_teacher, V_teacher)
    
    print(f"  Teacher K: {K_teacher.shape} -> Aligned K: {K_aligned.shape}")
    print(f"  Teacher V: {V_teacher.shape} -> Aligned V: {V_aligned.shape}")
    assert K_aligned.shape == (B, L, T, student_d_model)
    assert V_aligned.shape == (B, L, T, student_d_model)
    print("  ✓ Shape correct")
    
    # Test 3: Head flattening
    print("\n[Test 3] Head dimension flattening")
    B, L, H, T, d_head = 2, 12, 32, 50, 64
    kv_heads = torch.randn(B, L, H, T, d_head)
    
    kv_flat = flatten_kv_heads(kv_heads, H, d_head)
    print(f"  Original: {kv_heads.shape} -> Flattened: {kv_flat.shape}")
    assert kv_flat.shape == (B, L, T, H * d_head)
    
    kv_unflat = unflatten_kv_heads(kv_flat, H, d_head)
    print(f"  Flattened: {kv_flat.shape} -> Unflattened: {kv_unflat.shape}")
    assert kv_unflat.shape == kv_heads.shape
    assert torch.allclose(kv_unflat, kv_heads)
    print("  ✓ Flatten/unflatten reversible")
    
    # Test 4: Multi-teacher projection
    print("\n[Test 4] Multi-teacher projection")
    teacher_kvs = {
        "Qwen2-7B": (
            torch.randn(B, 28, T, 3584),
            torch.randn(B, 28, T, 3584)
        ),
        "Qwen2-1.5B": (
            torch.randn(B, 28, T, 1536),
            torch.randn(B, 28, T, 1536)
        )
    }
    
    aligned_kvs = projector.project_multi_teacher_kv(teacher_kvs)
    
    for teacher_name, (K, V) in aligned_kvs.items():
        print(f"  {teacher_name}: K {K.shape}, V {V.shape}")
        assert K.shape[-1] == student_d_model
        assert V.shape[-1] == student_d_model
    print("  ✓ All teachers projected to student dimension")
    
    # Test 5: Save/load
    print("\n[Test 5] Save and load projections")
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "projector.pt")
        projector.save_projections(save_path)
        
        loaded_projector = KVDimensionProjector.from_pretrained(save_path)
        
        # Verify weights match
        for teacher_name in teacher_configs.keys():
            w1 = projector.get_projection_weights(teacher_name)
            w2 = loaded_projector.get_projection_weights(teacher_name)
            assert torch.allclose(w1["W_K"], w2["W_K"])
            assert torch.allclose(w1["W_V"], w2["W_V"])
        
        print("  ✓ Save/load preserves weights")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
