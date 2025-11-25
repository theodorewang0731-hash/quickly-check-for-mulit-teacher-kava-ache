"""
KV alignment and loss utilities.

提供：
- align_teacher_kv_to_student: 将压缩后 teacher KV 对齐到 student 的时间步（当前实现基于 right-crop 对齐）
- compute_kv_loss: 支持 'smooth_l1', 'mse', 'smooth_l1_alpha'（alpha2 控制缩放）
- shuffled_kv: 用于生成打乱的 KV 对照

注意：本模块假定 teacher_kv 是由 experiments.kv_utils 返回的压缩后 tuple 格式，
其中每层为 (k, v)，k 为 numpy 数组或 torch tensor，shape 常见为 (batch, n_head, sel_len, head_dim)
"""
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F

def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.from_numpy(np.array(x))

def flatten_kv_tensor(k):
    """Flatten k from (batch, n_head, sel_len, head_dim) to (batch, sel_len, feat)
    where feat = n_head * head_dim
    """
    k = _to_tensor(k)
    if k.dim() == 4:
        b, nh, se, hd = k.shape
        k = k.permute(0,2,1,3).reshape(b, se, nh*hd)
    elif k.dim() == 3:
        # already (batch, seqlen, feat)
        pass
    return k

def align_teacher_kv_to_student(teacher_kv_layer, student_hidden, method='right_crop'):
    """Align a compressed teacher KV layer to student hidden states.

    - teacher_kv_layer: (k, v) for one layer (numpy or tensor)
    - student_hidden: tensor (batch, seq_len, hidden_size)

    Returns teacher_kv_tensor (batch, sel_len, feat) and student_aligned (batch, sel_len, hidden_size)
    by simple right-crop alignment: take last sel_len student hidden states.
    """
    k, v = teacher_kv_layer
    k_t = flatten_kv_tensor(k)  # (batch, sel_len, feat)
    sel_len = k_t.shape[1]
    # align to last sel_len tokens of student_hidden
    if student_hidden is None:
        raise ValueError('student_hidden is required for alignment')
    sh = student_hidden
    if sh.size(1) < sel_len:
        # pad student hidden with zeros at front
        pad = torch.zeros((sh.size(0), sel_len - sh.size(1), sh.size(2)), device=sh.device, dtype=sh.dtype)
        sh_padded = torch.cat([pad, sh], dim=1)
        student_segment = sh_padded[:, -sel_len:, :]
    else:
        student_segment = sh[:, -sel_len:, :]
    return k_t.to(student_segment.device).to(student_segment.dtype), student_segment

def compute_kv_loss(student_segment, teacher_k_t, loss_type='smooth_l1', alpha2=1.0, attention_weights=None):
    """Compute KV loss between student_segment (batch, sel_len, hidden_size)
    and teacher_k_t (batch, sel_len, feat).

    We first project student to teacher feature dim via a linear layer provided externally;
    Here we assume shapes already matched: if not, user must provide a projector.

    loss_type: 'smooth_l1', 'mse', 'smooth_l1_alpha'
    alpha2: scale factor applied to the loss (used by smooth_l1_alpha)
    attention_weights: Optional tensor of shape (batch, num_heads, seq_len, seq_len) or (batch, seq_len)
                       Used for attention-weighted KV loss (稳健小升级)
    """
    # Convert teacher to tensor
    tk = _to_tensor(teacher_k_t).to(student_segment.device)
    # If dims mismatch, try simple linear mapping via pseudo-inverse (not learned)
    if student_segment.shape[2] != tk.shape[2]:
        # perform a simple linear map via least-squares per batch: find W s.t. student_segment @ W ~= tk
        # This is expensive; instead, project tk to student dim by linear projection (PCA-like): use SVD on tk across feature dim
        # Simpler: if tk higher dim, reduce tk by mean pooling over feature axis groups
        s_dim = student_segment.shape[2]
        t_dim = tk.shape[2]
        if t_dim > s_dim:
            # reduce teacher dim by averaging chunks
            factor = t_dim // s_dim
            tk = tk.view(tk.shape[0], tk.shape[1], s_dim, factor).mean(-1)
        else:
            # expand teacher by repeating
            repeats = (s_dim + t_dim - 1) // t_dim
            tk = tk.repeat(1,1,repeats)[:,:,:s_dim]

    # ============================================================
    # Attention-weighted KV loss (稳健小升级 - 1天实现)
    # ============================================================
    if attention_weights is not None:
        # Compute token importance from attention weights
        attn = _to_tensor(attention_weights).to(student_segment.device)
        
        # **CRITICAL**: Detach attention weights to prevent gradient flow back to attention
        # Otherwise KV loss will try to optimize attention, conflicting with self-attention training
        attn = attn.detach()
        
        if attn.dim() == 4:  # (batch, num_heads, seq_len, seq_len)
            # Average over heads and query positions to get importance per token
            # token_importance[b, t] = how much attention token t receives
            token_importance = attn.mean(dim=(1, 2))  # (batch, seq_len)
        elif attn.dim() == 3:  # (batch, seq_len, seq_len)
            token_importance = attn.mean(dim=1)  # (batch, seq_len)
        elif attn.dim() == 2:  # (batch, seq_len) - already token importance
            token_importance = attn
        else:
            raise ValueError(f'Unexpected attention_weights shape: {attn.shape}')
        
        # Align to sel_len (take last sel_len tokens)
        sel_len = student_segment.shape[1]
        if token_importance.shape[1] > sel_len:
            token_importance = token_importance[:, -sel_len:]
        elif token_importance.shape[1] < sel_len:
            # Pad with uniform weights
            pad_len = sel_len - token_importance.shape[1]
            pad = torch.ones(token_importance.shape[0], pad_len, device=token_importance.device)
            token_importance = torch.cat([pad, token_importance], dim=1)
        
        # Normalize to sum to 1 (importance distribution)
        token_importance = token_importance / (token_importance.sum(dim=1, keepdim=True) + 1e-8)
        
        # Expand to match feature dimension: (batch, sel_len, 1)
        importance_weight = token_importance.unsqueeze(-1)
        
        # Compute weighted loss
        if loss_type == 'mse':
            diff = (student_segment - tk) ** 2
            weighted_loss = (diff * importance_weight).sum() / diff.numel()
        elif loss_type == 'smooth_l1':
            diff = F.smooth_l1_loss(student_segment, tk, reduction='none')
            weighted_loss = (diff * importance_weight).sum() / diff.numel()
        elif loss_type == 'smooth_l1_alpha':
            diff = F.smooth_l1_loss(student_segment, tk, reduction='none')
            weighted_loss = alpha2 * (diff * importance_weight).sum() / diff.numel()
        else:
            raise ValueError(f'Unknown loss_type: {loss_type}')
        
        return weighted_loss
    
    # ============================================================
    # Original unweighted KV loss
    # ============================================================
    if loss_type == 'mse':
        return F.mse_loss(student_segment, tk)
    elif loss_type == 'smooth_l1':
        return F.smooth_l1_loss(student_segment, tk)
    elif loss_type == 'smooth_l1_alpha':
        return alpha2 * F.smooth_l1_loss(student_segment, tk)
    else:
        raise ValueError(f'Unknown loss_type: {loss_type}')

def shuffled_kv(teacher_kv):
    """Return a shuffled version of teacher_kv for negative control.

    teacher_kv: tuple of layers (k,v) where k is numpy/tensor (batch, sel_len, feat) or (batch, n_head, sel_len, head_dim)
    This function will shuffle time steps within each batch-layer or swap between-batch examples.
    """
    out = []
    for k, v in teacher_kv:
        kt = np.array(k)
        vt = np.array(v)
        # shuffle along time axis
        if kt.ndim == 4:
            # (batch, n_head, sel_len, hd) -> shuffle sel_len axis
            b, nh, se, hd = kt.shape
            perm = np.random.permutation(se)
            kt_s = kt[..., perm, :]
            vt_s = vt[..., perm, :]
        elif kt.ndim == 3:
            perm = np.random.permutation(kt.shape[1])
            kt_s = kt[:, perm, :]
            vt_s = vt[:, perm, :]
        else:
            # fallback
            kt_s = np.take(kt, np.random.permutation(kt.shape[-1]), axis=-1)
            vt_s = np.take(vt, np.random.permutation(vt.shape[-1]), axis=-1)
        out.append((kt_s, vt_s))
    return tuple(out)
