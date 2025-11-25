"""
时间维 + 层维 + hidden维 对齐 v2 (Advanced Alignment)

解决问题：
1. 时间维：多教师 CoT 长度不同，硬 index 对齐语义错位
2. 层维：固定等比例映射不考虑表征相似性
3. Hidden维：Teacher d_model (3584/4096) -> Student d_model (1536/2048) 需要投影
4. Head维：先展平 H*d_head，不做 head-to-head mapping

升级方案：
1. 时间维：Segment-aware 等比例重采样 + 线性插值
2. 层维：CKA-based 层相似度矩阵 → 加权组合
3. Hidden维：可学习线性投影 W_K, W_V (按教师粒度共享)
4. Head维：展平处理 [B,L,H,T,d_head] -> [B,L,T,d_model]

参考老师反馈：
"在 多教师 + 不同 CoT 设定下，单纯 index 对齐是太粗"
"既然 teacher/student 的模型家族和数据基本固定，现在就可以把 CKA 用上"
"hidden size 可能是 4096、3584，学生是 2048、1536，用可学习的线性投影"
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from experiments.cka_loss import linear_cka


# ============================================================================
# Part 1: 时间维对齐 v2 - Segment-aware Resampling
# ============================================================================

@dataclass
class SequenceSegment:
    """序列段定义"""
    name: str  # "prompt", "reasoning", "answer"
    start: int
    end: int
    length: int
    
    def __post_init__(self):
        self.length = self.end - self.start


class SegmentIdentifier:
    """序列段识别器（识别 Prompt/Reasoning/Answer 边界）"""
    
    # 常见 CoT trigger patterns
    COT_TRIGGERS = [
        "Let's think step by step",
        "Let me break this down",
        "Step 1:",
        "First,",
        "①",
        "1.",
        "解题思路：",
        "让我们一步步来",
    ]
    
    # Answer markers
    ANSWER_MARKERS = [
        "The answer is",
        "Therefore,",
        "Final answer:",
        "答案是",
        "因此，",
        "最终答案：",
    ]
    
    @staticmethod
    def identify_segments(
        text: str,
        tokenizer,
        input_ids: torch.Tensor = None
    ) -> List[SequenceSegment]:
        """
        识别序列的三个段：Prompt, Reasoning, Answer
        
        Args:
            text: 完整生成文本
            tokenizer: 分词器
            input_ids: Token IDs (optional, for precise boundary)
            
        Returns:
            List of SequenceSegment
        """
        segments = []
        
        # Find reasoning start (CoT trigger)
        reasoning_start = -1
        for trigger in SegmentIdentifier.COT_TRIGGERS:
            idx = text.find(trigger)
            if idx != -1:
                reasoning_start = idx
                break
        
        # Find answer start
        answer_start = -1
        for marker in SegmentIdentifier.ANSWER_MARKERS:
            idx = text.find(marker)
            if idx != -1:
                answer_start = idx
                break
        
        # Convert char positions to token positions
        if input_ids is not None:
            total_len = len(input_ids)
        else:
            total_len = len(tokenizer.encode(text))
        
        # Estimate token boundaries (rough approximation)
        if reasoning_start != -1 and answer_start != -1:
            # Prompt | Reasoning | Answer
            prompt_ratio = reasoning_start / len(text)
            answer_ratio = answer_start / len(text)
            
            prompt_end = int(total_len * prompt_ratio)
            reasoning_end = int(total_len * answer_ratio)
            
            segments = [
                SequenceSegment("prompt", 0, prompt_end, prompt_end),
                SequenceSegment("reasoning", prompt_end, reasoning_end, reasoning_end - prompt_end),
                SequenceSegment("answer", reasoning_end, total_len, total_len - reasoning_end),
            ]
        elif reasoning_start != -1:
            # Prompt | Reasoning (no explicit answer marker)
            prompt_ratio = reasoning_start / len(text)
            prompt_end = int(total_len * prompt_ratio)
            
            segments = [
                SequenceSegment("prompt", 0, prompt_end, prompt_end),
                SequenceSegment("reasoning", prompt_end, total_len, total_len - prompt_end),
            ]
        else:
            # Fallback: treat entire sequence as reasoning
            segments = [
                SequenceSegment("reasoning", 0, total_len, total_len),
            ]
        
        return segments


def resample_kv_with_interpolation(
    teacher_kv: torch.Tensor,
    student_length: int,
    teacher_segments: List[SequenceSegment] = None,
    student_segments: List[SequenceSegment] = None,
) -> torch.Tensor:
    """
    等比例重采样 + 线性插值
    
    公式（针对 reasoning 段）：
        u_i = i / (T_s - 1) * (T_teacher - 1)
        j = floor(u_i), λ = u_i - j
        KV_i = (1 - λ) * KV_j + λ * KV_{j+1}
    
    Args:
        teacher_kv: (batch, teacher_seq_len, dim) 或 (batch, heads, teacher_seq_len, head_dim)
        student_length: 学生目标序列长度
        teacher_segments: 教师序列段（可选，用于细粒度对齐）
        student_segments: 学生序列段（可选）
        
    Returns:
        resampled_kv: (batch, student_length, dim) 或 (batch, heads, student_length, head_dim)
    """
    # Handle 4D (K/V with heads) or 3D (hidden states)
    is_4d = teacher_kv.dim() == 4
    
    if is_4d:
        batch, heads, teacher_len, head_dim = teacher_kv.shape
        teacher_kv_flat = teacher_kv.transpose(1, 2).reshape(batch, teacher_len, heads * head_dim)
    else:
        batch, teacher_len, dim = teacher_kv.shape
        teacher_kv_flat = teacher_kv
    
    # Simple case: no segment info, global resampling
    if teacher_segments is None or student_segments is None:
        return _global_resample(teacher_kv_flat, student_length, is_4d, heads if is_4d else None, head_dim if is_4d else None)
    
    # Advanced case: segment-aware resampling
    return _segment_aware_resample(
        teacher_kv_flat, 
        student_length, 
        teacher_segments, 
        student_segments,
        is_4d,
        heads if is_4d else None,
        head_dim if is_4d else None
    )


def _global_resample(
    teacher_kv_flat: torch.Tensor,
    student_length: int,
    is_4d: bool,
    heads: Optional[int],
    head_dim: Optional[int]
) -> torch.Tensor:
    """全局等比例重采样（简单版）"""
    batch, teacher_len, dim = teacher_kv_flat.shape
    
    # Compute sampling positions
    if teacher_len == 1:
        # Edge case: teacher only has 1 token
        resampled = teacher_kv_flat.repeat(1, student_length, 1)
    else:
        # u_i = i / (student_length - 1) * (teacher_len - 1)
        student_positions = torch.arange(student_length, device=teacher_kv_flat.device, dtype=torch.float32)
        teacher_positions = student_positions / max(student_length - 1, 1) * max(teacher_len - 1, 1)
        
        # j = floor(u_i), λ = u_i - j
        teacher_indices = teacher_positions.long()
        lambdas = teacher_positions - teacher_indices.float()
        
        # Clamp indices
        teacher_indices = torch.clamp(teacher_indices, 0, teacher_len - 2)
        next_indices = teacher_indices + 1
        
        # Linear interpolation: (1 - λ) * KV_j + λ * KV_{j+1}
        kv_j = teacher_kv_flat[:, teacher_indices, :]  # (batch, student_length, dim)
        kv_j_next = teacher_kv_flat[:, next_indices, :]
        
        lambdas = lambdas.view(1, -1, 1)  # (1, student_length, 1)
        resampled = (1 - lambdas) * kv_j + lambdas * kv_j_next
    
    # Reshape back to 4D if needed
    if is_4d:
        resampled = resampled.reshape(batch, student_length, heads, head_dim).transpose(1, 2)
    
    return resampled


def _segment_aware_resample(
    teacher_kv_flat: torch.Tensor,
    student_length: int,
    teacher_segments: List[SequenceSegment],
    student_segments: List[SequenceSegment],
    is_4d: bool,
    heads: Optional[int],
    head_dim: Optional[int]
) -> torch.Tensor:
    """段感知重采样（高级版）"""
    batch, teacher_len, dim = teacher_kv_flat.shape
    device = teacher_kv_flat.device
    
    resampled_parts = []
    
    # Match segments by name
    for student_seg in student_segments:
        # Find corresponding teacher segment
        teacher_seg = next((s for s in teacher_segments if s.name == student_seg.name), None)
        
        if teacher_seg is None:
            # Fallback: use global resampling for this segment
            seg_resampled = _global_resample(
                teacher_kv_flat[:, :student_seg.length, :],
                student_seg.length,
                False, None, None
            )
        else:
            # Extract teacher segment KV
            teacher_seg_kv = teacher_kv_flat[:, teacher_seg.start:teacher_seg.end, :]
            
            # Resample this segment
            seg_resampled = _global_resample(
                teacher_seg_kv,
                student_seg.length,
                False, None, None
            )
        
        resampled_parts.append(seg_resampled)
    
    # Concatenate all segments
    resampled = torch.cat(resampled_parts, dim=1)
    
    # Ensure correct length
    if resampled.shape[1] != student_length:
        resampled = F.interpolate(
            resampled.transpose(1, 2),
            size=student_length,
            mode='linear',
            align_corners=True
        ).transpose(1, 2)
    
    # Reshape back to 4D if needed
    if is_4d:
        resampled = resampled.reshape(batch, student_length, heads, head_dim).transpose(1, 2)
    
    return resampled


# ============================================================================
# Part 2: 层维对齐 v2 - CKA-based Layer Mapping
# ============================================================================

class CKALayerMapper:
    """CKA-based 层相似度映射器"""
    
    def __init__(
        self,
        student_num_layers: int,
        teacher_num_layers: int,
        top_k: int = 2
    ):
        """
        Args:
            student_num_layers: 学生模型层数
            teacher_num_layers: 教师模型层数
            top_k: 每个学生层选择 top-k 个最相似的教师层
        """
        self.student_num_layers = student_num_layers
        self.teacher_num_layers = teacher_num_layers
        self.top_k = top_k
        
        # Similarity matrix: S[k, l] = CKA(student_layer_k, teacher_layer_l)
        self.similarity_matrix = None  # (student_layers, teacher_layers)
        
        # Mapping: student_layer_k -> [(teacher_layer_l1, weight_1), (teacher_layer_l2, weight_2), ...]
        self.layer_mapping = {}
    
    def compute_similarity_matrix(
        self,
        student_hiddens_list: List[torch.Tensor],
        teacher_hiddens_list: List[torch.Tensor],
        num_samples: int = 100
    ):
        """
        计算层间 CKA 相似度矩阵
        
        Args:
            student_hiddens_list: List of (batch*seq_len, hidden_dim) per layer, collected over num_samples
            teacher_hiddens_list: List of (batch*seq_len, hidden_dim) per layer
            num_samples: 用于计算的样本数
        """
        print(f"[CKA Layer Mapping] Computing similarity matrix with {num_samples} samples...")
        
        S = np.zeros((self.student_num_layers, self.teacher_num_layers))
        
        for k in range(self.student_num_layers):
            for l in range(self.teacher_num_layers):
                # Compute CKA between student layer k and teacher layer l
                student_hidden = student_hiddens_list[k]  # (N, d_s)
                teacher_hidden = teacher_hiddens_list[l]  # (N, d_t)
                
                # Use CKA (handles different dimensions automatically)
                cka_score = linear_cka(student_hidden, teacher_hidden, debiased=True)
                S[k, l] = cka_score.item()
            
            # Progress
            if (k + 1) % 5 == 0:
                print(f"  Processed {k + 1}/{self.student_num_layers} student layers")
        
        self.similarity_matrix = S
        print(f"✓ Similarity matrix computed: {S.shape}")
        
        # Print summary
        print("\nSimilarity Matrix Summary:")
        print(f"  Mean: {S.mean():.4f}, Std: {S.std():.4f}")
        print(f"  Min: {S.min():.4f}, Max: {S.max():.4f}")
    
    def build_layer_mapping(self):
        """根据相似度矩阵构建层映射"""
        if self.similarity_matrix is None:
            raise ValueError("Must compute similarity matrix first!")
        
        print(f"\n[CKA Layer Mapping] Building layer mapping (top-{self.top_k})...")
        
        for k in range(self.student_num_layers):
            # Get top-k teacher layers for student layer k
            similarities = self.similarity_matrix[k, :]
            top_k_indices = np.argsort(similarities)[-self.top_k:][::-1]
            top_k_scores = similarities[top_k_indices]
            
            # Normalize weights (softmax-like, but only over top-k)
            # Use max(score, 0) to avoid negative weights
            weights = np.maximum(top_k_scores, 0)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                # Fallback: uniform weights
                weights = np.ones(self.top_k) / self.top_k
            
            # Store mapping
            self.layer_mapping[k] = [(int(top_k_indices[i]), float(weights[i])) for i in range(self.top_k)]
            
            # Log
            mapping_str = ", ".join([f"L{l}:{w:.3f}" for l, w in self.layer_mapping[k]])
            print(f"  Student L{k:2d} -> Teacher [{mapping_str}]")
        
        print("✓ Layer mapping built")
    
    def get_aligned_teacher_kv(
        self,
        student_layer_idx: int,
        teacher_kvs: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据映射获取对齐的教师 KV（加权组合）
        
        Args:
            student_layer_idx: 学生层索引
            teacher_kvs: List of (K, V) per teacher layer
                         K, V: (batch, heads, seq_len, head_dim) 或 (batch, seq_len, dim)
        
        Returns:
            aligned_k, aligned_v: 加权组合后的 KV
        """
        if student_layer_idx not in self.layer_mapping:
            raise ValueError(f"No mapping for student layer {student_layer_idx}")
        
        mapping = self.layer_mapping[student_layer_idx]
        
        # Weighted sum: Σ β_{k,l} * KV_l^teacher
        aligned_k = None
        aligned_v = None
        
        for teacher_layer_idx, weight in mapping:
            teacher_k, teacher_v = teacher_kvs[teacher_layer_idx]
            
            if aligned_k is None:
                aligned_k = weight * teacher_k
                aligned_v = weight * teacher_v
            else:
                aligned_k = aligned_k + weight * teacher_k
                aligned_v = aligned_v + weight * teacher_v
        
        return aligned_k, aligned_v
    
    def save_mapping(self, path: str):
        """保存映射到文件"""
        import json
        
        data = {
            "student_num_layers": self.student_num_layers,
            "teacher_num_layers": self.teacher_num_layers,
            "top_k": self.top_k,
            "similarity_matrix": self.similarity_matrix.tolist(),
            "layer_mapping": {str(k): v for k, v in self.layer_mapping.items()}
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Layer mapping saved to {path}")
    
    def load_mapping(self, path: str):
        """从文件加载映射"""
        import json
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.student_num_layers = data["student_num_layers"]
        self.teacher_num_layers = data["teacher_num_layers"]
        self.top_k = data["top_k"]
        self.similarity_matrix = np.array(data["similarity_matrix"])
        self.layer_mapping = {int(k): v for k, v in data["layer_mapping"].items()}
        
        print(f"✓ Layer mapping loaded from {path}")


# ============================================================================
# Part 3: 完整对齐流程（时间 + 层）
# ============================================================================

def align_multi_teacher_kv_v2(
    student_hidden: torch.Tensor,
    student_layer_idx: int,
    teacher_kvs_per_layer: List[List[Tuple[torch.Tensor, torch.Tensor]]],
    layer_mapper: CKALayerMapper,
    student_segments: List[SequenceSegment] = None,
    teacher_segments_list: List[List[SequenceSegment]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    完整的多教师 KV 对齐 v2：层映射 + 时间重采样
    
    Args:
        student_hidden: (batch, student_seq_len, hidden_dim)
        student_layer_idx: 当前学生层索引
        teacher_kvs_per_layer: List[List[(K, V)]] - 每个教师的所有层 KV
        layer_mapper: CKA 层映射器
        student_segments: 学生序列段（可选）
        teacher_segments_list: 每个教师的序列段（可选）
    
    Returns:
        aligned_k, aligned_v: (batch, student_seq_len, dim)
    """
    batch, student_seq_len, hidden_dim = student_hidden.shape
    
    # Step 1: 使用 CKA layer mapper 获取对应的教师层 KV（已加权组合）
    # 假设只有一个教师（多教师在外层处理）
    teacher_kvs = teacher_kvs_per_layer[0]  # First teacher
    teacher_k, teacher_v = layer_mapper.get_aligned_teacher_kv(student_layer_idx, teacher_kvs)
    
    # Step 2: 时间维重采样（如果教师序列长度 != 学生序列长度）
    teacher_seq_len = teacher_k.shape[-2]  # (batch, heads, seq_len, head_dim) or (batch, seq_len, dim)
    
    if teacher_seq_len != student_seq_len:
        # 使用 segment-aware resampling
        teacher_segments = teacher_segments_list[0] if teacher_segments_list else None
        
        aligned_k = resample_kv_with_interpolation(
            teacher_k,
            student_seq_len,
            teacher_segments,
            student_segments
        )
        aligned_v = resample_kv_with_interpolation(
            teacher_v,
            student_seq_len,
            teacher_segments,
            student_segments
        )
    else:
        aligned_k = teacher_k
        aligned_v = teacher_v
    
    return aligned_k, aligned_v


# ============================================================================
# Part 4: 预计算工具（训练前运行一次）
# ============================================================================

def precompute_layer_mapping(
    student_model,
    teacher_model,
    dataloader,
    num_samples: int = 100,
    output_path: str = "layer_mapping.json",
    device: str = "cuda"
):
    """
    预计算 CKA 层相似度矩阵并保存
    
    Args:
        student_model: 学生模型
        teacher_model: 教师模型
        dataloader: 数据加载器
        num_samples: 用于计算的样本数
        output_path: 输出文件路径
        device: 计算设备
    
    Returns:
        layer_mapper: CKALayerMapper 对象
    """
    print("=" * 80)
    print("Precomputing CKA-based Layer Mapping")
    print("=" * 80)
    
    student_model.eval()
    teacher_model.eval()
    
    # Collect hidden states
    student_hiddens_per_layer = [[] for _ in range(student_model.config.num_hidden_layers)]
    teacher_hiddens_per_layer = [[] for _ in range(teacher_model.config.num_hidden_layers)]
    
    sample_count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if sample_count >= num_samples:
                break
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Student forward
            s_out = student_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            for layer_idx, hidden in enumerate(s_out.hidden_states[:-1]):  # Exclude final layer
                # Flatten: (batch, seq_len, dim) -> (batch*seq_len, dim)
                hidden_flat = hidden.reshape(-1, hidden.shape[-1])
                student_hiddens_per_layer[layer_idx].append(hidden_flat.cpu())
            
            # Teacher forward
            t_out = teacher_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            for layer_idx, hidden in enumerate(t_out.hidden_states[:-1]):
                hidden_flat = hidden.reshape(-1, hidden.shape[-1])
                teacher_hiddens_per_layer[layer_idx].append(hidden_flat.cpu())
            
            sample_count += input_ids.shape[0]
            print(f"  Collected {sample_count}/{num_samples} samples", end='\r')
    
    print(f"\n✓ Collected {sample_count} samples")
    
    # Concatenate all samples
    student_hiddens = [torch.cat(hiddens, dim=0) for hiddens in student_hiddens_per_layer]
    teacher_hiddens = [torch.cat(hiddens, dim=0) for hiddens in teacher_hiddens_per_layer]
    
    # Create mapper
    mapper = CKALayerMapper(
        student_num_layers=len(student_hiddens),
        teacher_num_layers=len(teacher_hiddens),
        top_k=2
    )
    
    # Compute similarity matrix
    mapper.compute_similarity_matrix(student_hiddens, teacher_hiddens, num_samples=sample_count)
    
    # Build mapping
    mapper.build_layer_mapping()
    
    # Save
    mapper.save_mapping(output_path)
    
    return mapper


# ============================================================================
# Test & Demo
# ============================================================================

if __name__ == "__main__":
    print("Testing Alignment v2...")
    
    # Test 1: Time-wise resampling with interpolation
    print("\n[Test 1] Time-wise resampling")
    teacher_kv = torch.randn(2, 10, 64)  # (batch=2, teacher_len=10, dim=64)
    student_len = 5
    
    resampled = resample_kv_with_interpolation(teacher_kv, student_len)
    print(f"  Teacher: {teacher_kv.shape} -> Student: {resampled.shape}")
    assert resampled.shape == (2, 5, 64)
    print("  ✓ Global resampling works")
    
    # Test 2: Segment-aware resampling
    print("\n[Test 2] Segment-aware resampling")
    teacher_segments = [
        SequenceSegment("prompt", 0, 2, 2),
        SequenceSegment("reasoning", 2, 8, 6),
        SequenceSegment("answer", 8, 10, 2),
    ]
    student_segments = [
        SequenceSegment("prompt", 0, 1, 1),
        SequenceSegment("reasoning", 1, 4, 3),
        SequenceSegment("answer", 4, 5, 1),
    ]
    
    resampled_seg = resample_kv_with_interpolation(
        teacher_kv, 
        student_len,
        teacher_segments,
        student_segments
    )
    print(f"  With segments: {resampled_seg.shape}")
    assert resampled_seg.shape == (2, 5, 64)
    print("  ✓ Segment-aware resampling works")
    
    # Test 3: CKA Layer Mapper
    print("\n[Test 3] CKA Layer Mapper")
    student_layers = 12
    teacher_layers = 24
    
    mapper = CKALayerMapper(student_layers, teacher_layers, top_k=2)
    
    # Fake similarity matrix
    mapper.similarity_matrix = np.random.rand(student_layers, teacher_layers)
    mapper.build_layer_mapping()
    
    print(f"  Mapping for student L0: {mapper.layer_mapping[0]}")
    print("  ✓ CKA layer mapper works")
    
    # Test 4: Get aligned teacher KV
    print("\n[Test 4] Aligned teacher KV")
    teacher_kvs = [(torch.randn(2, 8, 64), torch.randn(2, 8, 64)) for _ in range(teacher_layers)]
    aligned_k, aligned_v = mapper.get_aligned_teacher_kv(0, teacher_kvs)
    print(f"  Aligned K: {aligned_k.shape}, V: {aligned_v.shape}")
    print("  ✓ Aligned teacher KV works")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)


# ============================================================================
# Part 5: Integration with Dimension Projection (完整对齐流程)
# ============================================================================

def align_multi_teacher_kv_with_projection(
    teacher_kvs: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]],
    student_layer_idx: int,
    student_length: int,
    layer_mapper: CKALayerMapper,
    projector: Optional[object] = None,  # KVDimensionProjector
    use_segment_resampling: bool = False,
    teacher_segments: Optional[Dict[str, List]] = None,
    student_segments: Optional[List] = None
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    完整的三阶段对齐：时间 + 层 + 维度
    
    Pipeline:
        1. Layer Alignment: CKA-based top-k teacher layer selection
        2. Time Alignment: Segment-aware resampling to student sequence length
        3. Dimension Alignment: Linear projection d_teacher -> d_student
    
    Args:
        teacher_kvs: Dict mapping teacher_name -> List[(K, V)] 
            where K, V are per-layer tensors [B, T_t, d_t] or [B, H_t, T_t, d_head_t]
        student_layer_idx: Current student layer index (for layer mapping)
        student_length: Target sequence length
        layer_mapper: CKALayerMapper instance
        projector: KVDimensionProjector instance (optional, if dimensions match)
        use_segment_resampling: Whether to use segment-aware time alignment
        teacher_segments: Segments for each teacher (if segment resampling)
        student_segments: Student segments (if segment resampling)
    
    Returns:
        Dict mapping teacher_name -> (K_aligned, V_aligned)
            Final shape: [B, student_length, d_student]
    
    Example:
        >>> from experiments.kv_dimension_projector import KVDimensionProjector
        >>> 
        >>> # Initialize projector
        >>> projector = KVDimensionProjector(
        ...     teacher_configs={"Qwen2-7B": {"d_model": 3584, "num_layers": 28}},
        ...     student_d_model=2048
        ... )
        >>> 
        >>> # Align KV
        >>> teacher_kvs = {
        ...     "Qwen2-7B": [(K_l, V_l) for l in range(28)]  # Per-layer KV
        ... }
        >>> aligned = align_multi_teacher_kv_with_projection(
        ...     teacher_kvs=teacher_kvs,
        ...     student_layer_idx=5,
        ...     student_length=50,
        ...     layer_mapper=mapper,
        ...     projector=projector
        ... )
    """
    
    aligned_kvs = {}
    
    for teacher_name, teacher_kv_layers in teacher_kvs.items():
        # Step 1: Layer Alignment - Get top-k teacher layers for this student layer
        teacher_K_layers = [kv[0] for kv in teacher_kv_layers]
        teacher_V_layers = [kv[1] for kv in teacher_kv_layers]
        
        # Convert to list of (K, V) tuples
        teacher_kvs_list = list(zip(teacher_K_layers, teacher_V_layers))
        
        # Get weighted combination of top-k teacher layers
        K_layer_aligned, V_layer_aligned = layer_mapper.get_aligned_teacher_kv(
            student_layer_idx, teacher_kvs_list
        )
        
        # Step 2: Time Alignment - Resample to student sequence length
        teacher_segs = teacher_segments.get(teacher_name) if teacher_segments else None
        
        K_time_aligned = resample_kv_with_interpolation(
            K_layer_aligned,
            student_length,
            teacher_segs,
            student_segments
        )
        V_time_aligned = resample_kv_with_interpolation(
            V_layer_aligned,
            student_length,
            teacher_segs,
            student_segments
        )
        
        # Step 3: Dimension Alignment - Project to student d_model (if projector provided)
        if projector is not None:
            # Ensure 3D: [B, T, d]
            if K_time_aligned.dim() == 4:  # [B, H, T, d_head]
                from experiments.kv_dimension_projector import flatten_kv_heads
                B, H, T, d_head = K_time_aligned.shape
                K_time_aligned = flatten_kv_heads(K_time_aligned.unsqueeze(1), H, d_head).squeeze(1)
                V_time_aligned = flatten_kv_heads(V_time_aligned.unsqueeze(1), H, d_head).squeeze(1)
            
            # Add dummy layer dimension for projector: [B, T, d] -> [B, 1, T, d]
            K_with_layer = K_time_aligned.unsqueeze(1)
            V_with_layer = V_time_aligned.unsqueeze(1)
            
            # Project: [B, 1, T, d_t] -> [B, 1, T, d_s]
            K_final, V_final = projector.project_teacher_kv(
                teacher_name, K_with_layer, V_with_layer
            )
            
            # Remove layer dimension: [B, 1, T, d_s] -> [B, T, d_s]
            K_final = K_final.squeeze(1)
            V_final = V_final.squeeze(1)
        else:
            K_final = K_time_aligned
            V_final = V_time_aligned
        
        aligned_kvs[teacher_name] = (K_final, V_final)
    
    return aligned_kvs


def fuse_multi_teacher_kv(
    aligned_kvs: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    teacher_weights: Optional[Dict[str, float]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    融合多个教师的对齐后 KV
    
    简单版：加权平均
    后续可扩展：模式发现、动态路由等
    
    Args:
        aligned_kvs: Dict mapping teacher_name -> (K_aligned, V_aligned)
            where K, V are [B, T, d_student]
        teacher_weights: Optional weights for each teacher (default: uniform)
    
    Returns:
        (K_fused, V_fused): Both [B, T, d_student]
    
    Math:
        KV_target = Σ_i α_i · KV_(teacher_i, aligned)
    """
    
    teacher_names = list(aligned_kvs.keys())
    
    # Default: uniform weights
    if teacher_weights is None:
        teacher_weights = {name: 1.0 / len(teacher_names) for name in teacher_names}
    
    # Normalize weights
    total_weight = sum(teacher_weights.values())
    teacher_weights = {k: v / total_weight for k, v in teacher_weights.items()}
    
    # Weighted sum
    K_fused = None
    V_fused = None
    
    for teacher_name, (K, V) in aligned_kvs.items():
        weight = teacher_weights.get(teacher_name, 0.0)
        
        if K_fused is None:
            K_fused = weight * K
            V_fused = weight * V
        else:
            K_fused += weight * K
            V_fused += weight * V
    
    return K_fused, V_fused

