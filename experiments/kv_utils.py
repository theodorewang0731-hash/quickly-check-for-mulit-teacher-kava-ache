"""
KV utilities for extracting and compressing teacher past_key_values.

提供三种压缩方法：
- full_kv: 不压缩，直接返回原始 past_key_values
- right_crop_kv: 取每层 KV 的最后 L 个 time steps
- rkv: 基于重要性评分 + 去冗余的贪心选择（近似 Gram-Schmidt）

这些实现是可运行的近似复现，用于实验对比。
"""
from typing import Tuple, List
import numpy as np

def flatten_key_tensor(k):
    # k: (batch, n_head, seq_len, head_dim) or (batch, seq_len, n_head, head_dim)
    k = np.array(k)
    if k.ndim == 4:
        # try to detect ordering: if second dim equals number heads small, assume (batch, n_head, seq_len, head_dim)
        b, a, c, d = k.shape
        # flatten head and head_dim into feature dim, and put seq_len as second dim
        if a < 64:  # heuristic
            k = k.transpose(0,2,1,3).reshape(b, c, a*d)
        else:
            k = k.reshape(b, c, a*d)
    return k

def full_kv(past_key_values: Tuple[Tuple[np.ndarray, np.ndarray], ...]):
    """Return KV unchanged."""
    return past_key_values

def right_crop_kv(past_key_values: Tuple[Tuple[np.ndarray, np.ndarray], ...], target_len: int):
    """Crop the last `target_len` time steps from each layer's key/value tensors.

    Expects past_key_values as tuple of (k, v) per layer, where k/v are numpy arrays
    with shape roughly (batch, n_head, seq_len, head_dim).
    """
    compressed = []
    for k, v in past_key_values:
        k = np.array(k)
        v = np.array(v)
        if k.ndim == 4:
            cropped_k = k[..., -target_len:, :]
            cropped_v = v[..., -target_len:, :]
        elif k.ndim == 3:
            # (batch, seq_len, dim)
            cropped_k = k[:, -target_len:, :]
            cropped_v = v[:, -target_len:, :]
        else:
            # fallback: take last target_len on last axis
            cropped_k = k[..., -target_len:]
            cropped_v = v[..., -target_len:]
        compressed.append((cropped_k, cropped_v))
    return tuple(compressed)

def rkv_greedy(past_key_values: Tuple[Tuple[np.ndarray, np.ndarray], ...], target_len: int, lambda_param: float=0.1):
    """Approximate R-KV: select `target_len` time steps per layer by importance + de-redundancy.

    Importance score: L2 norm of flattened key vector across heads and head-dim.
    Greedy de-redundancy: pick highest-scoring vector, orthogonalize remaining vectors w.r.t selected ones (Gram-Schmidt)
    and repeat until target_len selected.
    """
    compressed = []
    for layer_idx, (k, v) in enumerate(past_key_values):
        k = np.array(k)
        v = np.array(v)
        # convert to (batch, seq_len, feat)
        if k.ndim == 4:
            b, nh, seqlen, hd = k.shape
            feats = k.transpose(0,2,1,3).reshape(b, seqlen, nh*hd)
            vals = v.transpose(0,2,1,3).reshape(b, seqlen, nh*hd)
        elif k.ndim == 3:
            b, seqlen, feat = k.shape
            feats = k
            vals = v
        else:
            feats = k
            vals = v

        # operate per batch independently; here assume batch=1 most experiments
        sel_indices_batch = []
        comp_k = []
        comp_v = []
        for bi in range(feats.shape[0]):
            X = feats[bi].astype(np.float64).copy()  # (seqlen, feat)
            Y = vals[bi].astype(np.float64).copy()
            seqlen = X.shape[0]
            if target_len >= seqlen:
                idxs = list(range(seqlen))
            else:
                # importance scores
                scores = np.linalg.norm(X, axis=1)
                selected = []
                # working copy for GS
                X_work = X.copy()
                for _ in range(target_len):
                    # pick highest norm
                    i = int(np.argmax(np.linalg.norm(X_work, axis=1)))
                    selected.append(i)
                    # if selected vector is zero, break
                    v_sel = X_work[i]
                    if np.allclose(v_sel, 0):
                        break
                    # project out component from others
                    v_norm = v_sel / (np.dot(v_sel, v_sel) + 1e-9)
                    for j in range(X_work.shape[0]):
                        X_work[j] = X_work[j] - np.dot(X_work[j], v_norm) * v_sel
                # ensure increasing order (time order)
                idxs = sorted(list(dict.fromkeys(selected)))
            sel_indices_batch.append(idxs)
            # build compressed arrays for this batch
            comp_k.append(X[ idxs ] if len(idxs)>0 else np.zeros((0, X.shape[1])))
            comp_v.append(Y[ idxs ] if len(idxs)>0 else np.zeros((0, Y.shape[1])))

        # stack back to shape (batch, sel_len, feat)
        comp_k = np.stack([arr if arr.ndim==2 else np.zeros((0, feats.shape[2])) for arr in comp_k], axis=0)
        comp_v = np.stack([arr if arr.ndim==2 else np.zeros((0, vals.shape[2])) for arr in comp_v], axis=0)

        # reshape back to (batch, n_head, sel_len, head_dim) if original had 4 dims
        if k.ndim == 4:
            sel_len = comp_k.shape[1]
            comp_k = comp_k.reshape(b, sel_len, nh, hd).transpose(0,2,1,3)
            comp_v = comp_v.reshape(b, sel_len, nh, hd).transpose(0,2,1,3)

        compressed.append((comp_k, comp_v))
    return tuple(compressed)

def reconstruct_from_compressed(original_k, compressed_k, method='right_crop'):
    """Simple reconstruction of original sequence length by placing compressed vectors back to their positions.

    For right_crop: place at the end; for rkv: we do sparse placement at selected indices if available, else zeros.
    Used only for diagnostic comparisons.
    """
    ok = np.array(original_k)
    ck = np.array(compressed_k)
    if ok.ndim == 4:
        b, nh, seqlen, hd = ok.shape
        sel_len = ck.shape[2]
        recon = np.zeros_like(ok)
        if method == 'right_crop':
            recon[..., -sel_len:, :] = ck
        else:
            # if compressed has fewer timesteps, place them evenly at the end positions
            recon[..., -sel_len:, :] = ck
        return recon
    else:
        return ck
