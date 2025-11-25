"""
Tokenizer Alignment Module

构造字符级对齐矩阵 A ∈ R^{T_s × T_t}，用于异构 tokenizer 之间的软对齐。
基于字符级交并比（IoU）计算对齐权重。

Usage:
    A = build_char_align_matrix(text, teacher_tokenizer, student_tokenizer)
    K_aligned = A @ K_teacher  # [T_s, d_k]
"""
import torch
import numpy as np
from typing import List, Tuple


def char_ranges_from_tokens(
    text: str,
    tokens: List[str],
    tokenizer
) -> List[Tuple[int, int]]:
    """
    将 token 映射回原文本的字符范围。
    
    Args:
        text: 原始文本
        tokens: tokenizer 输出的 token 列表
        tokenizer: tokenizer 对象
        
    Returns:
        List[(start_char, end_char)] 每个 token 对应的字符范围
    """
    char_ranges = []
    current_pos = 0
    
    for token in tokens:
        # 解码 token 到文本片段
        token_text = tokenizer.decode([tokenizer.encode(token)[0]])
        
        # 在原文本中查找该片段（跳过空格）
        while current_pos < len(text) and text[current_pos].isspace():
            current_pos += 1
            
        # 查找匹配
        start = current_pos
        if token_text in text[current_pos:current_pos + len(token_text) + 10]:
            # 找到匹配位置
            idx = text[current_pos:].find(token_text)
            start = current_pos + idx
            end = start + len(token_text)
            current_pos = end
        else:
            # 无法精确匹配，使用估算
            end = min(start + len(token_text), len(text))
            current_pos = end
            
        char_ranges.append((start, end))
    
    return char_ranges


def compute_iou(range1: Tuple[int, int], range2: Tuple[int, int]) -> float:
    """
    计算两个字符范围的交并比（IoU）。
    
    Args:
        range1, range2: (start, end) 字符范围
        
    Returns:
        IoU score [0, 1]
    """
    start1, end1 = range1
    start2, end2 = range2
    
    # 计算交集
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection = max(0, intersection_end - intersection_start)
    
    # 计算并集
    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union = union_end - union_start
    
    if union == 0:
        return 0.0
    
    return intersection / union


def build_char_align_matrix(
    text: str,
    teacher_tokenizer,
    student_tokenizer,
    teacher_tokens: List[str] = None,
    student_tokens: List[str] = None,
    normalize: bool = True,
    min_threshold: float = 0.1
) -> torch.Tensor:
    """
    构造字符级对齐矩阵 A ∈ R^{T_s × T_t}。
    
    A[i, j] 表示学生 token i 与教师 token j 的对齐权重（基于字符 IoU）。
    
    Args:
        text: 原始文本
        teacher_tokenizer: 教师 tokenizer
        student_tokenizer: 学生 tokenizer
        teacher_tokens: 教师 token 列表（可选，用于加速）
        student_tokens: 学生 token 列表（可选，用于加速）
        normalize: 是否按行归一化（使每个学生 token 的权重和为 1）
        min_threshold: 最小 IoU 阈值，低于此值的对齐权重设为 0
        
    Returns:
        A: [T_s, T_t] 对齐矩阵
    """
    # 编码文本
    if teacher_tokens is None:
        teacher_ids = teacher_tokenizer.encode(text, add_special_tokens=False)
        teacher_tokens = [teacher_tokenizer.decode([tid]) for tid in teacher_ids]
    
    if student_tokens is None:
        student_ids = student_tokenizer.encode(text, add_special_tokens=False)
        student_tokens = [student_tokenizer.decode([sid]) for sid in student_ids]
    
    # 获取字符范围
    teacher_ranges = char_ranges_from_tokens(text, teacher_tokens, teacher_tokenizer)
    student_ranges = char_ranges_from_tokens(text, student_tokens, student_tokenizer)
    
    T_s = len(student_tokens)
    T_t = len(teacher_tokens)
    
    # 构造对齐矩阵
    A = np.zeros((T_s, T_t), dtype=np.float32)
    
    for i, s_range in enumerate(student_ranges):
        for j, t_range in enumerate(teacher_ranges):
            iou = compute_iou(s_range, t_range)
            if iou >= min_threshold:
                A[i, j] = iou
    
    # 归一化（每行和为 1）
    if normalize:
        row_sums = A.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)  # 避免除零
        A = A / row_sums
    
    return torch.from_numpy(A)


def apply_char_alignment(
    kv_teacher: torch.Tensor,
    align_matrix: torch.Tensor,
    dim: int = 1
) -> torch.Tensor:
    """
    应用字符对齐矩阵到教师 KV。
    
    Args:
        kv_teacher: [batch, T_t, ...] 教师 KV tensor
        align_matrix: [T_s, T_t] 对齐矩阵
        dim: 时间维度（默认为 1）
        
    Returns:
        kv_aligned: [batch, T_s, ...] 对齐后的 KV
    """
    # 确保在正确的设备上
    align_matrix = align_matrix.to(kv_teacher.device)
    
    # 重塑以便矩阵乘法
    shape = kv_teacher.shape
    assert dim < len(shape), f"dim {dim} out of range for shape {shape}"
    
    # 将时间维移到第二维
    if dim != 1:
        perm = list(range(len(shape)))
        perm[1], perm[dim] = perm[dim], perm[1]
        kv_teacher = kv_teacher.permute(*perm)
        shape_perm = kv_teacher.shape
    else:
        shape_perm = shape
    
    # [batch, T_t, ...] -> [batch * ..., T_t]
    batch = shape_perm[0]
    T_t = shape_perm[1]
    rest = shape_perm[2:]
    kv_flat = kv_teacher.reshape(batch, T_t, -1)  # [batch, T_t, d]
    
    # 应用对齐：[batch, T_s, d] = [T_s, T_t] @ [batch, T_t, d]
    # 需要广播：对每个 batch 独立应用
    kv_aligned = torch.einsum('st,btd->bsd', align_matrix, kv_flat)
    
    # 恢复形状
    T_s = align_matrix.shape[0]
    kv_aligned = kv_aligned.reshape(batch, T_s, *rest)
    
    # 恢复维度顺序
    if dim != 1:
        kv_aligned = kv_aligned.permute(*perm)
    
    return kv_aligned


def visualize_alignment(
    align_matrix: torch.Tensor,
    teacher_tokens: List[str],
    student_tokens: List[str],
    save_path: str = None
):
    """
    可视化对齐矩阵。
    
    Args:
        align_matrix: [T_s, T_t] 对齐矩阵
        teacher_tokens: 教师 token 列表
        student_tokens: 学生 token 列表
        save_path: 保存路径（可选）
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            align_matrix.cpu().numpy(),
            xticklabels=teacher_tokens[:50],  # 限制显示数量
            yticklabels=student_tokens[:50],
            cmap='YlOrRd',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Alignment Weight'}
        )
        plt.xlabel('Teacher Tokens')
        plt.ylabel('Student Tokens')
        plt.title('Character-level Token Alignment Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Alignment matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    except ImportError:
        print("Warning: matplotlib/seaborn not available for visualization")


if __name__ == "__main__":
    # 测试代码
    from transformers import AutoTokenizer
    
    text = "The quick brown fox jumps over the lazy dog"
    
    # 使用两个不同的 tokenizer
    tok1 = AutoTokenizer.from_pretrained("gpt2")
    tok2 = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # 构造对齐矩阵
    A = build_char_align_matrix(text, tok1, tok2)
    
    print(f"Text: {text}")
    print(f"Teacher tokens: {len(tok1.encode(text))}")
    print(f"Student tokens: {len(tok2.encode(text))}")
    print(f"Alignment matrix shape: {A.shape}")
    print(f"Matrix sum per row (should be ~1.0): {A.sum(dim=1)}")
    
    # 测试应用对齐
    kv_teacher = torch.randn(2, len(tok1.encode(text)), 64)  # [batch, T_t, d]
    kv_aligned = apply_char_alignment(kv_teacher, A)
    print(f"Original KV shape: {kv_teacher.shape}")
    print(f"Aligned KV shape: {kv_aligned.shape}")
    print("✓ Tokenizer alignment test passed")
