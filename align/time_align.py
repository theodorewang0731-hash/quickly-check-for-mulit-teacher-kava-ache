"""
Time Alignment Module

处理时间维度对齐：padding、masking、软对齐（字符矩阵）。
确保教师和学生的 KV 在时间步上一致。

Usage:
    kv_aligned = apply_time_alignment(kv, mask=mask)
    kv_aligned = apply_soft_alignment(kv, align_matrix=A)
"""
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def pad_to_length(
    tensor: torch.Tensor,
    target_length: int,
    pad_value: float = 0.0,
    dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将 tensor 在指定维度 pad 到目标长度。
    
    Args:
        tensor: 输入 tensor
        target_length: 目标长度
        pad_value: padding 值
        dim: 要 pad 的维度
        
    Returns:
        padded_tensor: padding 后的 tensor
        mask: [batch, target_length] bool mask (True = valid, False = padding)
    """
    current_length = tensor.size(dim)
    
    if current_length >= target_length:
        # 不需要 padding（或需要截断，但我们不截断）
        slices = [slice(None)] * len(tensor.shape)
        slices[dim] = slice(0, target_length)
        truncated = tensor[tuple(slices)]
        
        batch_size = tensor.size(0)
        mask = torch.ones(batch_size, target_length, dtype=torch.bool, device=tensor.device)
        return truncated, mask
    
    # 计算需要 pad 的长度
    pad_length = target_length - current_length
    
    # 构造 padding 参数（从最后一维往前）
    # F.pad 的 pad 参数格式：(左, 右, 上, 下, ...)
    ndim = len(tensor.shape)
    pad_params = [0] * (2 * ndim)
    
    # dim 对应的 padding（从后往前数）
    pad_idx = 2 * (ndim - 1 - dim)
    pad_params[pad_idx + 1] = pad_length  # 右侧 padding
    
    padded = F.pad(tensor, pad_params, value=pad_value)
    
    # 构造 mask
    batch_size = tensor.size(0)
    mask = torch.zeros(batch_size, target_length, dtype=torch.bool, device=tensor.device)
    mask[:, :current_length] = True
    
    return padded, mask


def apply_mask_to_kv(
    kv: torch.Tensor,
    mask: torch.Tensor,
    mask_value: float = 0.0,
    time_dim: int = 1
) -> torch.Tensor:
    """
    应用 mask 到 KV tensor。
    
    Args:
        kv: [batch, time, ...] KV tensor
        mask: [batch, time] bool mask (True = valid)
        mask_value: mask 位置的填充值
        time_dim: 时间维度
        
    Returns:
        masked_kv: mask 后的 KV
    """
    # 扩展 mask 到 kv 的所有维度
    while mask.ndim < kv.ndim:
        mask = mask.unsqueeze(-1)
    
    # 移动 time_dim 到第二维（如果不是）
    if time_dim != 1:
        perm = list(range(kv.ndim))
        perm[1], perm[time_dim] = perm[time_dim], perm[1]
        kv = kv.permute(*perm)
        # mask 也对应调整
        mask_perm = list(range(mask.ndim))
        mask_perm[1], mask_perm[time_dim] = mask_perm[time_dim], mask_perm[1]
        mask = mask.permute(*mask_perm)
    
    # 应用 mask
    masked_kv = torch.where(mask, kv, torch.full_like(kv, mask_value))
    
    # 恢复维度顺序
    if time_dim != 1:
        masked_kv = masked_kv.permute(*perm)
    
    return masked_kv


def apply_soft_alignment(
    kv: torch.Tensor,
    align_matrix: torch.Tensor,
    time_dim: int = 1
) -> torch.Tensor:
    """
    使用对齐矩阵进行软对齐（适用于异构 tokenizer）。
    
    Args:
        kv: [batch, T_t, ...] 教师 KV
        align_matrix: [T_s, T_t] 对齐矩阵
        time_dim: 时间维度
        
    Returns:
        aligned_kv: [batch, T_s, ...] 对齐后的 KV
    """
    # 使用 tokenizer_align.apply_char_alignment
    from align.tokenizer_align import apply_char_alignment
    return apply_char_alignment(kv, align_matrix, dim=time_dim)


def create_attention_mask(
    input_ids: torch.Tensor,
    pad_token_id: int
) -> torch.Tensor:
    """
    从 input_ids 创建 attention mask。
    
    Args:
        input_ids: [batch, seq_len] token IDs
        pad_token_id: padding token ID
        
    Returns:
        mask: [batch, seq_len] bool mask (True = not padding)
    """
    return input_ids != pad_token_id


def align_sequence_lengths(
    teacher_kv: torch.Tensor,
    student_kv: torch.Tensor,
    teacher_mask: Optional[torch.Tensor] = None,
    student_mask: Optional[torch.Tensor] = None,
    align_matrix: Optional[torch.Tensor] = None,
    time_dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    对齐教师和学生的序列长度。
    
    策略：
    1. 如果提供 align_matrix，使用软对齐（异构 tokenizer）
    2. 否则，pad 到相同长度（同构 tokenizer）
    
    Args:
        teacher_kv: 教师 KV
        student_kv: 学生 KV
        teacher_mask: 教师 mask（可选）
        student_mask: 学生 mask（可选）
        align_matrix: 对齐矩阵（可选，用于异构 tokenizer）
        time_dim: 时间维度
        
    Returns:
        teacher_kv_aligned: 对齐后的教师 KV
        student_kv_aligned: 对齐后的学生 KV
        teacher_mask_aligned: 对齐后的教师 mask
        student_mask_aligned: 对齐后的学生 mask
    """
    T_t = teacher_kv.size(time_dim)
    T_s = student_kv.size(time_dim)
    
    if align_matrix is not None:
        # 软对齐（异构 tokenizer）
        teacher_kv_aligned = apply_soft_alignment(teacher_kv, align_matrix, time_dim)
        student_kv_aligned = student_kv
        
        # mask 也需要对齐
        if teacher_mask is not None:
            # [batch, T_t] @ [T_s, T_t]^T -> [batch, T_s]
            teacher_mask_float = teacher_mask.float()
            teacher_mask_aligned = torch.matmul(teacher_mask_float, align_matrix.T.to(teacher_mask.device))
            teacher_mask_aligned = teacher_mask_aligned > 0.5  # 转回 bool
        else:
            teacher_mask_aligned = torch.ones(
                teacher_kv_aligned.size(0),
                teacher_kv_aligned.size(time_dim),
                dtype=torch.bool,
                device=teacher_kv.device
            )
        
        student_mask_aligned = student_mask if student_mask is not None else torch.ones(
            student_kv.size(0),
            student_kv.size(time_dim),
            dtype=torch.bool,
            device=student_kv.device
        )
        
    else:
        # 硬对齐（同构 tokenizer）- pad 到相同长度
        target_length = max(T_t, T_s)
        
        teacher_kv_aligned, teacher_mask_aligned = pad_to_length(teacher_kv, target_length, dim=time_dim)
        student_kv_aligned, student_mask_aligned = pad_to_length(student_kv, target_length, dim=time_dim)
        
        # 合并原有 mask
        if teacher_mask is not None:
            teacher_mask_padded, _ = pad_to_length(teacher_mask.unsqueeze(-1), target_length, dim=1)
            teacher_mask_aligned = teacher_mask_aligned & teacher_mask_padded.squeeze(-1)
        
        if student_mask is not None:
            student_mask_padded, _ = pad_to_length(student_mask.unsqueeze(-1), target_length, dim=1)
            student_mask_aligned = student_mask_aligned & student_mask_padded.squeeze(-1)
    
    return teacher_kv_aligned, student_kv_aligned, teacher_mask_aligned, student_mask_aligned


if __name__ == "__main__":
    # 测试代码
    print("Testing time alignment...")
    
    # 测试 padding
    tensor = torch.randn(2, 5, 64)  # [batch, time, dim]
    padded, mask = pad_to_length(tensor, target_length=10, dim=1)
    assert padded.shape == (2, 10, 64)
    assert mask.shape == (2, 10)
    assert mask[:, :5].all()
    assert not mask[:, 5:].any()
    print("✓ Padding test passed")
    
    # 测试 masking
    kv = torch.randn(2, 10, 64)
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[:, :7] = True
    masked_kv = apply_mask_to_kv(kv, mask)
    assert (masked_kv[:, 7:] == 0).all()
    print("✓ Masking test passed")
    
    # 测试序列对齐
    teacher_kv = torch.randn(2, 8, 64)
    student_kv = torch.randn(2, 10, 64)
    t_aligned, s_aligned, t_mask, s_mask = align_sequence_lengths(teacher_kv, student_kv)
    assert t_aligned.size(1) == s_aligned.size(1) == 10
    print("✓ Sequence alignment test passed")
    
    print("All time alignment tests passed!")
