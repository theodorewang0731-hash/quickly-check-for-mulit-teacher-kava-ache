"""
KaVa Training Script - KV Distillation for Language Model Compression

功能：
- 支持三种 teacher-KV 压缩： full / right_crop / rkv (通过 experiments.kv_utils)
- 支持 KV 损失类型： smooth_l1 / mse / smooth_l1_alpha
- 支持 shuffled-KV 对照（--shuffle_kv）
- 支持超大数据集、流式模式、多 GPU、混合精度训练
- 完整的检查点保存和训练日志记录

验证目标：
1. KV 压缩保留监督信号
2. KV 对齐补充 latent 监督
3. R-KV 提供最佳稳定性

使用示例：
    python experiments/train_with_kv.py \\
        --model_name Qwen/Qwen2-1.5B \\
        --dataset_name openai/gsm8k \\
        --epochs 3 \\
        --batch_size 8 \\
        --kv_method rkv \\
        --fp16 \\
        --gradient_checkpointing
"""
import argparse
import json
import os
from typing import Optional
import torch
import numpy as np
from datasets import load_dataset
import torch.nn.functional as F
from types import SimpleNamespace

from transformers import AutoTokenizer, AutoModelForCausalLM

from experiments.kv_utils import full_kv, right_crop_kv, rkv_greedy
try:
    from experiments.rkv_official import rkv_compress_with_attention
except ImportError:
    rkv_compress_with_attention = None
from experiments.kv_loss import align_teacher_kv_to_student, compute_kv_loss, shuffled_kv
from experiments.projector import StudentToTeacherProjector
from experiments.cka_loss import multi_layer_cka_loss
from experiments.alignment_v2 import (
    CKALayerMapper,
    SegmentIdentifier,
    resample_kv_with_interpolation,
    align_multi_teacher_kv_v2
)
import torch.nn as nn

# v4.0 Map Projection imports
from src.map_projection_aligner import MapProjectionAligner


# ========== Core Functions ==========

def stack_past_kv(past_key_values, as_tensor=True):
    """
    将 tuple of (k, v) 转为 [L, 2, B, H, T, D] 张量用于 MapProjectionAligner
    
    Args:
        past_key_values: tuple of (k, v) pairs, each k/v shape [B, H, T, D]
        as_tensor: 是否返回 torch.Tensor (否则返回 numpy)
    
    Returns:
        stacked_kv: [L, 2, B, H, T, D]
    """
    kvs = []
    for k, v in past_key_values:
        if isinstance(k, np.ndarray):
            k = torch.from_numpy(k)
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
        # 确保在同一设备
        if k.device != v.device:
            v = v.to(k.device)
        kvs.append(torch.stack([k, v], dim=0))  # [2, B, H, T, D]
    
    stacked = torch.stack(kvs, dim=0)  # [L, 2, B, H, T, D]
    return stacked if as_tensor else stacked.cpu().numpy()


def to_numpy_kv(past_key_values):
    """Convert tensors in past_key_values to numpy for downstream compression utils."""
    converted = []
    for layer in past_key_values:
        if layer is None:
            continue
        k, v = layer
        if isinstance(k, torch.Tensor):
            k = k.detach().cpu().numpy()
        elif not isinstance(k, np.ndarray):
            k = np.array(k)
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        elif not isinstance(v, np.ndarray):
            v = np.array(v)
        converted.append((k, v))
    return tuple(converted)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subset_size", type=int, default=None, help="Number of training samples to use (None=use all)")
    p.add_argument("--model_name", type=str, default="gpt2")
    p.add_argument("--teacher_name", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--torch_dtype", type=str, default=None, help="Optional torch dtype override, e.g. float16")
    p.add_argument("--device_map", type=str, default=None, help="Optional accelerate-style device map, e.g. 'auto'")
    p.add_argument("--output_dir", type=str, default="outputs/with_kv")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    p.add_argument("--target_len", type=int, default=8)
    p.add_argument("--kv_method", type=str, default="rkv", choices=["full","right_crop","rkv","rkv_official"]) 
    p.add_argument("--kv_loss", type=str, default="smooth_l1", choices=["smooth_l1","mse","smooth_l1_alpha"])
    p.add_argument("--rkv_lambda", type=float, default=0.1, help="Lambda for R-KV (importance weight, paper recommends 0.1)")
    p.add_argument("--rkv_keep_recent", type=int, default=None, help="Keep recent β tokens in generation mode (None for training)") 
    p.add_argument("--alpha2", type=float, default=1.0)
    p.add_argument("--shuffle_kv", action='store_true')
    p.add_argument("--train_file", type=str, default=None)
    p.add_argument("--dataset_name", type=str, default="openai/gsm8k", help="Hugging Face dataset name")
    p.add_argument("--dataset_config", type=str, default="main", help="Dataset config/subset name")
    p.add_argument("--dataset_split", type=str, default="train", help="Dataset split to use")
    p.add_argument("--streaming", action="store_true", help="Use streaming mode for large datasets")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--kv_weight", type=float, default=1.0, help="Weight for KV loss (0=no KV)")
    p.add_argument("--codi_weight", type=float, default=0.5, help="Weight for CODI loss")
    p.add_argument("--use_attention_weighted_kv", action="store_true", help="Use attention-weighted KV loss (稳健小升级)")
    p.add_argument("--attention_weighted_kv_warmup", type=int, default=1000, help="Warmup steps before enabling attention-weighted KV (avoid unstable early attention)")
    p.add_argument("--use_teacher_attention", action="store_true", help="Use teacher attention for KV weighting instead of student (more stable but less aligned)")
    p.add_argument("--cka_weight", type=float, default=0.05, help="Weight for CKA auxiliary loss (稳健小升级)")
    p.add_argument("--cka_layers", type=str, default="middle", help="Which layers for CKA: 'middle', 'last', or comma-separated indices like '6,12'")
    
    # Alignment v2 (时间维 + 层维对齐升级)
    p.add_argument("--use_cka_layer_mapping", action="store_true", help="Use CKA-based layer mapping (Alignment v2)")
    p.add_argument("--layer_mapping_path", type=str, default=None, help="Path to precomputed layer mapping JSON")
    p.add_argument("--use_segment_resampling", action="store_true", help="Use segment-aware time resampling (Alignment v2)")
    
    # v4.0 Map Projection (Anti-Flatten Structured Alignment)
    p.add_argument("--alignment_mode", type=str, default="flat", choices=["flat", "structured"], 
                   help="Alignment mode: 'flat' (baseline) or 'structured' (v4.0 map projection)")
    p.add_argument("--map_proj_share_dim", action="store_true", 
                   help="Share dimension projection across heads (v4.0)")
    p.add_argument("--map_proj_init_uniform", action="store_true",
                   help="Uniform initialization for head mixer (v4.0)")
    
    p.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    p.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    p.add_argument("--logging_steps", type=int, default=10, help="Log metrics every N steps")
    p.add_argument("--fp16", action="store_true", help="Use mixed precision training (fp16)")
    p.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory")
    return p.parse_args()

def _load_local_jsonl(train_file, subset_size, tokenizer):
    examples = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            prompt = item.get('prompt') or item.get('question') or item.get('problem') or ''
            target = item.get('target') or item.get('answer') or item.get('answer_text') or ''
            text = prompt + tokenizer.eos_token + target
            examples.append({'input_text': text, 'prompt': prompt, 'target': target})
            if subset_size is not None and len(examples) >= subset_size:
                break
    return examples

def build_dataset(args, tokenizer):
    """Build dataset from local file or Hugging Face, with optional streaming support."""
    if args.train_file:
        if not os.path.exists(args.train_file):
            raise FileNotFoundError(f"Specified train_file not found: {args.train_file}")
        return _load_local_jsonl(args.train_file, args.subset_size, tokenizer)

    # Load from Hugging Face
    try:
        if args.streaming:
            # Streaming mode for very large datasets
            ds = load_dataset(
                args.dataset_name, 
                args.dataset_config, 
                split=args.dataset_split,
                streaming=True,
                cache_dir=args.cache_dir
            )
            if args.subset_size:
                ds = ds.take(args.subset_size)
            return ds  # Return streaming dataset directly
        else:
            # Regular mode
            ds = load_dataset(
                args.dataset_name,
                args.dataset_config,
                split=args.dataset_split,
                cache_dir=args.cache_dir
            )
            if args.subset_size:
                ds = ds.select(range(min(args.subset_size, len(ds))))
            
            # Convert to examples list
            examples = []
            for item in ds:
                prompt = item.get('question') or item.get('problem') or ''
                target = item.get('answer') or item.get('answer_text') or ''
                text = prompt + tokenizer.eos_token + target
                examples.append({'input_text': text, 'prompt': prompt, 'target': target})
            return examples
    except Exception as e:
        print(f"Warning: failed to load {args.dataset_name} from Hugging Face ({e}). Falling back to local sample file.")
        fallback = os.path.join('data', 'sample_train.jsonl')
        return _load_local_jsonl(fallback, args.subset_size, tokenizer)

def collate_fn(batch, tokenizer, max_len=512):
    texts = [b['input_text'] for b in batch]
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
    # build labels: mask prompt tokens
    labels = enc['input_ids'].clone()
    for i, b in enumerate(batch):
        p_enc = tokenizer(b['prompt'], truncation=True, max_length=max_len)['input_ids']
        plen = len(p_enc)
        labels[i, :plen] = -100
    enc['labels'] = labels
    return enc

def main():
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    def _resolve_dtype(name: Optional[str]):
        if not name:
            return None
        name = name.lower()
        mapping = {
            "float32": torch.float32,
            "fp32": torch.float32,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
        }
        if name in mapping:
            return mapping[name]
        raise ValueError(f"Unsupported torch dtype string: {name}")

    dtype = _resolve_dtype(args.torch_dtype)

    model_kwargs = {
        "cache_dir": args.cache_dir,
        "trust_remote_code": args.trust_remote_code,
    }
    tokenizer_kwargs = {
        "cache_dir": args.cache_dir,
        "trust_remote_code": args.trust_remote_code,
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if args.device_map:
        model_kwargs["device_map"] = args.device_map

    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **tokenizer_kwargs)

    if hasattr(tokenizer, "pad_token") and getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    elif not hasattr(tokenizer, "pad_token"):
        tokenizer.pad_token = tokenizer.eos_token

    if hasattr(tokenizer, "pad_token_id") and getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = getattr(tokenizer, "eos_token_id", None)

    # Load teacher model
    teacher_name = args.teacher_name or args.model_name
    def _move_model(m):
        if args.device_map:
            return m
        return m.to(device)

    print(f"Loading teacher model: {teacher_name}")
    teacher = AutoModelForCausalLM.from_pretrained(teacher_name, **model_kwargs)
    teacher = _move_model(teacher)
    teacher.eval()

    # Load student model
    print(f"Loading student model: {args.model_name}")
    student = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    student = _move_model(student)
    student.train()

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing and hasattr(student, 'gradient_checkpointing_enable'):
        student.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled for student model")

    # Load CKA layer mapper (Alignment v2)
    layer_mapper = None
    if args.use_cka_layer_mapping:
        if args.layer_mapping_path is None:
            raise ValueError("--layer_mapping_path required when using CKA layer mapping!")
        
        print(f"\n[Alignment v2] Loading CKA layer mapping from {args.layer_mapping_path}")
        layer_mapper = CKALayerMapper(
            student_num_layers=student.config.num_hidden_layers,
            teacher_num_layers=teacher.config.num_hidden_layers if teacher else student.config.num_hidden_layers
        )
        layer_mapper.load_mapping(args.layer_mapping_path)
        print("✓ CKA layer mapping loaded")
    
    # v4.0 Map Projection Aligner (will be initialized lazily on first batch)
    map_aligner = None
    if args.alignment_mode == "structured":
        print(f"\n[v4.0 Map Projection] Will initialize structured aligner on first batch")
        print(f"  - share_dim_proj: {args.map_proj_share_dim}")
        print(f"  - init_uniform: {args.map_proj_init_uniform}")

    examples = build_dataset(args, tokenizer)

    from torch.utils.data import DataLoader
    dl = DataLoader(
        examples, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_len=args.max_length),
        pin_memory=True if args.device != "cpu" else False
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)
    
    # Setup mixed precision training
    scaler = None
    if args.fp16:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
        print("Mixed precision training (fp16) enabled")
    
    # projectors will be created lazily on first batch when we know teacher KV feat dims
    projectors = None

    global_step = 0
    for ep in range(args.epochs):
        print(f"Epoch {ep+1}/{args.epochs}")
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(dl):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            teacher_hidden = None
            teacher_pkv = None
            teacher_all_hiddens = None
            teacher_attention = None
            t_out = None  # Store full teacher output for rkv_official
            
            if teacher is not None:
                with torch.no_grad():
                    # 论文严格要求：R-KV 必须获取 attention（用于 Importance Score）
                    need_attention = args.use_teacher_attention or args.kv_method == 'rkv_official'
                    
                    t_out = teacher(
                        input_ids, 
                        attention_mask=attention_mask, 
                        use_cache=True, 
                        output_hidden_states=True,
                        output_attentions=need_attention  # R-KV official 强制需要
                    )
                    teacher_pkv = getattr(t_out, 'past_key_values', None)
                    teacher_hidden = t_out.hidden_states[-1]  # (batch, seq_len, hidden)
                    teacher_all_hiddens = t_out.hidden_states if args.cka_weight > 0 else None
                    teacher_attention = t_out.attentions[-1] if need_attention else None

            # compress teacher kv
            if teacher_pkv is None:
                # create simulated kv as random
                # fallback handled in kv_utils if needed; here we call full_kv on simulated
                simulated = []
                n_layer = student.config.n_layer
                n_head = student.config.n_head
                head_dim = student.config.hidden_size // n_head
                seq_len = input_ids.shape[-1]
                for _ in range(n_layer):
                    k = np.random.randn(input_ids.shape[0], n_head, seq_len, head_dim).astype(np.float32)
                    v = np.random.randn(input_ids.shape[0], n_head, seq_len, head_dim).astype(np.float32)
                    simulated.append((k, v))
                teacher_pkv = tuple(simulated)
                if teacher_hidden is None:
                    teacher_hidden = torch.randn(input_ids.size(0), seq_len, student.config.hidden_size, device=device)
            else:
                teacher_pkv = to_numpy_kv(teacher_pkv)

            if args.kv_method == 'full':
                comp = full_kv(teacher_pkv)
            elif args.kv_method == 'right_crop':
                comp = right_crop_kv(teacher_pkv, args.target_len)
            elif args.kv_method == 'rkv_official':
                # 论文官方实现：使用 attention-based importance + cosine redundancy
                if t_out is None or not hasattr(t_out, 'attentions') or t_out.attentions is None:
                    raise ValueError(
                        "rkv_official requires teacher attention scores. "
                        "This should not happen as we forced output_attentions=True above."
                    )
                
                # 将 tuple of tensors 转换回 torch tensors
                teacher_pkv_torch = tuple(
                    (torch.from_numpy(k).to(device) if isinstance(k, np.ndarray) else k,
                     torch.from_numpy(v).to(device) if isinstance(v, np.ndarray) else v)
                    for k, v in teacher_pkv
                )
                
                # 论文方法：S = λ * I + (1 - λ) * R
                # I: attention from answer tokens to CoT
                # R: cosine similarity based redundancy
                comp_torch = rkv_compress_with_attention(
                    t_out,
                    target_len=args.target_len,
                    lbd=args.rkv_lambda,  # 论文推荐 0.1
                    attention_mask=attention_mask,
                    keep_recent=args.rkv_keep_recent  # 训练时为 None
                )
                
                # 转回 numpy（与其他方法保持一致）
                comp = to_numpy_kv(comp_torch)
            else:
                # 默认 rkv_greedy（旧版近似实现）
                comp = rkv_greedy(teacher_pkv, args.target_len, lambda_param=0.1)

            if args.shuffle_kv:
                comp = shuffled_kv(comp)

            # lazily build projectors matching teacher KV feature dims
            if projectors is None and args.alignment_mode == "flat":
                # Baseline mode: use old projectors
                proj_list = []
                def feat_dim_of(k):
                    import numpy as _np
                    import torch as _torch
                    if isinstance(k, _np.ndarray):
                        if k.ndim == 4:
                            return k.shape[1] * k.shape[3]
                        elif k.ndim == 3:
                            return k.shape[2]
                    if isinstance(k, _torch.Tensor):
                        if k.dim() == 4:
                            return k.shape[1] * k.shape[3]
                        elif k.dim() == 3:
                            return k.shape[2]
                    # fallback
                    return student.config.hidden_size

                for layer in comp:
                    tk, _ = layer
                    td = feat_dim_of(tk)
                    proj = StudentToTeacherProjector(student.config.hidden_size, td).to(device)
                    proj_list.append(proj)
                projectors = nn.ModuleList(proj_list)
                # recreate optimizer to include projector params
                optimizer = torch.optim.AdamW(list(student.parameters()) + list(projectors.parameters()), lr=args.lr)
            
            # v4.0: Lazily initialize MapProjectionAligner
            if map_aligner is None and args.alignment_mode == "structured":
                # Extract dimensions from first batch
                sample_k, sample_v = comp[0]
                if isinstance(sample_k, np.ndarray):
                    sample_k = torch.from_numpy(sample_k).to(device)
                if isinstance(sample_v, np.ndarray):
                    sample_v = torch.from_numpy(sample_v).to(device)
                
                # sample_k shape: [B, H_t, T, D_t]
                num_teacher_layers = len(comp)
                num_student_layers = student.config.num_hidden_layers
                num_teacher_heads = sample_k.shape[1]
                num_student_heads = student.config.num_attention_heads
                teacher_head_dim = sample_k.shape[-1]
                student_head_dim = student.config.hidden_size // student.config.num_attention_heads
                
                print(f"\n[v4.0 Aligner Init] T_layers={num_teacher_layers}, S_layers={num_student_layers}")
                print(f"  T_heads={num_teacher_heads}, S_heads={num_student_heads}")
                print(f"  T_head_dim={teacher_head_dim}, S_head_dim={student_head_dim}")
                
                map_aligner = MapProjectionAligner(
                    num_teacher_layers=num_teacher_layers,
                    num_student_layers=num_student_layers,
                    num_teacher_heads=num_teacher_heads,
                    num_student_heads=num_student_heads,
                    teacher_head_dim=teacher_head_dim,
                    student_head_dim=student_head_dim,
                    mode="structured",
                    share_dim_proj=args.map_proj_share_dim,
                    init_uniform=args.map_proj_init_uniform
                ).to(device)
                
                # Recreate optimizer to include aligner params
                optimizer = torch.optim.AdamW(
                    list(student.parameters()) + list(map_aligner.parameters()), 
                    lr=args.lr
                )
                print(f"✓ MapProjectionAligner initialized with {sum(p.numel() for p in map_aligner.parameters())} params")

            # forward student with output_hidden_states and output_attentions (for attention weighting)
            s_out = student(
                input_ids, 
                attention_mask=attention_mask, 
                output_hidden_states=True,
                output_attentions=(args.use_attention_weighted_kv and not args.use_teacher_attention)  # Only get student attention if needed
            )
            logits = s_out.logits
            student_hidden = s_out.hidden_states[-1]
            student_all_hiddens = s_out.hidden_states if args.cka_weight > 0 else None
            student_attention = s_out.attentions[-1] if (args.use_attention_weighted_kv and not args.use_teacher_attention) else None

            if teacher_hidden is None:
                teacher_hidden = student_hidden.detach()

            # CE loss
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = ce_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            kv_loss_total = torch.tensor(0.0, device=device)
            
            # ========== 双模式对齐分支 ==========
            if args.alignment_mode == "structured":
                # v4.0 Structured Alignment: Map Projection (Anti-Flatten)
                # 1. 准备输入: 将 comp (numpy/mixed) 转为统一的 torch tensor
                teacher_kv_list = []
                for k, v in comp:
                    if isinstance(k, np.ndarray):
                        k = torch.from_numpy(k).to(device)
                    if isinstance(v, np.ndarray):
                        v = torch.from_numpy(v).to(device)
                    teacher_kv_list.append((k, v))
                
                # Stack to [L, 2, B, H, T, D]
                teacher_k_stack = torch.stack([kv[0] for kv in teacher_kv_list], dim=0)  # [L, B, H, T, D]
                teacher_v_stack = torch.stack([kv[1] for kv in teacher_kv_list], dim=0)
                
                # 2. 获取 student KV (需要从 forward 中获取)
                # NOTE: 这里需要重新 forward student 以获取 past_key_values
                with torch.no_grad():
                    s_out_kv = student(
                        input_ids,
                        attention_mask=attention_mask,
                        use_cache=True
                    )
                student_pkv = s_out_kv.past_key_values
                student_k_stack = torch.stack([kv[0] for kv in student_pkv], dim=0)
                student_v_stack = torch.stack([kv[1] for kv in student_pkv], dim=0)
                
                # 3. 创建 segment_ids (假设整个序列为单个 segment)
                # Shape: [B, T] - 全 0 表示所有 token 属于 segment 0
                segment_ids = torch.zeros(input_ids.shape[0], input_ids.shape[1], 
                                         dtype=torch.long, device=device)
                
                # 4. Map Projection Alignment
                aligned_k, aligned_v, attn_map = map_aligner(
                    teacher_k_stack,  # [L_t, B, H_t, T, D_t]
                    teacher_v_stack,
                    None,  # Q not needed for KV-only loss
                    segment_ids
                )
                
                # 5. Compute KV loss: MSE between aligned teacher and student
                # aligned_k: [L_s, B, H_s, T_s, D_s]
                # student_k_stack: [L_s, B, H_s, T_s, D_s]
                kv_loss_k = F.mse_loss(aligned_k, student_k_stack)
                kv_loss_v = F.mse_loss(aligned_v, student_v_stack)
                kv_loss_total = (kv_loss_k + kv_loss_v) / 2.0
                
            else:
                # Baseline mode: Flat alignment (original pipeline)
                layer_losses = []
                for layer_idx, layer in enumerate(comp):
                    teacher_k, teacher_v = layer
                    tk, student_seg = align_teacher_kv_to_student((teacher_k, teacher_v), student_hidden, method='right_crop')
                    # project student_segment to teacher feat dim
                    proj = projectors[layer_idx]
                    student_proj = proj(student_seg)
                    # student_proj: (batch, sel_len, feat), tk: (batch, sel_len, feat)
                    
                    # Attention-weighted KV loss (稳健小升级)
                    # Use warmup: only enable after N steps to avoid unstable early student attention
                    use_attn_weighting = (
                        args.use_attention_weighted_kv 
                        and global_step >= args.attention_weighted_kv_warmup
                    )
                    
                    if use_attn_weighting:
                        # Choose attention source: teacher (more stable) or student (more aligned)
                        attn_source = teacher_attention if args.use_teacher_attention else student_attention
                        
                        if attn_source is not None:
                            l = compute_kv_loss(
                                student_proj, 
                                tk, 
                                loss_type=args.kv_loss, 
                                alpha2=args.alpha2,
                                attention_weights=attn_source  # Pass attention for weighting (will be detached inside)
                            )
                        else:
                            # Fallback to unweighted if attention not available
                            l = compute_kv_loss(student_proj, tk, loss_type=args.kv_loss, alpha2=args.alpha2)
                    else:
                        # Original unweighted KV loss (baseline or during warmup)
                        l = compute_kv_loss(student_proj, tk, loss_type=args.kv_loss, alpha2=args.alpha2)
                    
                    layer_losses.append(l)
                if len(layer_losses) > 0:
                    kv_loss_total = torch.stack(layer_losses).mean()

            # CODI proxy: MSE between student_hidden and teacher_hidden
            codi_loss = F.mse_loss(student_hidden, teacher_hidden)

            # CKA auxiliary loss (稳健小升级)
            cka_loss = torch.tensor(0.0, device=device)
            if args.cka_weight > 0 and student_all_hiddens is not None and teacher_all_hiddens is not None:
                # Parse layer indices
                if args.cka_layers == "middle":
                    layer_indices = [len(student_all_hiddens) // 2]
                elif args.cka_layers == "last":
                    layer_indices = [len(student_all_hiddens) - 1]
                else:
                    # Parse comma-separated indices
                    try:
                        layer_indices = [int(x.strip()) for x in args.cka_layers.split(",")]
                    except:
                        print(f"Warning: Invalid --cka_layers '{args.cka_layers}', using middle layer")
                        layer_indices = [len(student_all_hiddens) // 2]
                
                cka_loss = multi_layer_cka_loss(student_all_hiddens, teacher_all_hiddens, layer_indices=layer_indices)

            # Compose total loss: ce + kv + codi + cka (use configurable weights)
            total_loss = (
                ce_loss 
                + args.codi_weight * codi_loss 
                + args.kv_weight * kv_loss_total
                + args.cka_weight * cka_loss
            )

            # Backward pass with optional mixed precision
            if scaler is not None:
                scaler.scale(total_loss).backward()
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                total_loss.backward()
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            epoch_loss += total_loss.item()
            global_step += 1

            # Logging
            if global_step % args.logging_steps == 0:
                avg_loss = total_loss.item()
                log_msg = f"Step {global_step}: loss={avg_loss:.4f}, CE={ce_loss.item():.4f}, KV={kv_loss_total.item():.4f}, CODI={codi_loss.item():.4f}"
                if args.cka_weight > 0:
                    log_msg += f", CKA={cka_loss.item():.4f}"
                
                # Alignment mode indicator
                log_msg += f" [Mode: {args.alignment_mode}]"
                
                # Attention weighting status
                if args.use_attention_weighted_kv and args.alignment_mode == "flat":
                    if global_step >= args.attention_weighted_kv_warmup:
                        attn_src = "Teacher-Attn" if args.use_teacher_attention else "Student-Attn"
                        log_msg += f" [{attn_src}-weighted]"
                    else:
                        warmup_remaining = args.attention_weighted_kv_warmup - global_step
                        log_msg += f" [Attn-warmup: {warmup_remaining} steps]"
                
                print(log_msg)
                
            # Save checkpoint
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                ckpt = os.path.join(args.output_dir, f'checkpoint-step{global_step}')
                student.save_pretrained(ckpt)
                if projectors is not None:
                    torch.save(projectors.state_dict(), os.path.join(ckpt, 'projectors.pt'))
                if map_aligner is not None:
                    torch.save(map_aligner.state_dict(), os.path.join(ckpt, 'map_aligner.pt'))
                print(f"Checkpoint saved at step {global_step}")

        # Epoch summary
        avg_epoch_loss = epoch_loss / len(dl)
        epoch_summary = f"Epoch {ep+1}/{args.epochs} done. Avg Loss={avg_epoch_loss:.4f}, Last CE={ce_loss.item():.4f}, KV={kv_loss_total.item():.4f}, CODI={codi_loss.item():.4f}"
        print(epoch_summary)
        
        # save epoch checkpoint
        ckpt = os.path.join(args.output_dir, f'checkpoint-ep{ep+1}')
        student.save_pretrained(ckpt)
        if projectors is not None:
            torch.save(projectors.state_dict(), os.path.join(ckpt, 'projectors.pt'))
        if map_aligner is not None:
            torch.save(map_aligner.state_dict(), os.path.join(ckpt, 'map_aligner.pt'))
        
        # save epoch summary to log
        log_file = os.path.join(args.output_dir, 'training_log.txt')
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(epoch_summary + '\n')

    print('Training finished.')
    
    # Write completion marker and final summary
    done_file = os.path.join(args.output_dir, 'TRAINING_COMPLETED.txt')
    with open(done_file, 'w', encoding='utf-8') as f:
        f.write('=' * 50 + '\n')
        f.write('TRAINING COMPLETED SUCCESSFULLY\n')
        f.write('=' * 50 + '\n')
        f.write(f'Model: {args.model_name}\n')
        f.write(f'Subset size: {args.subset_size}\n')
        f.write(f'Epochs: {args.epochs}\n')
        f.write(f'Batch size: {args.batch_size}\n')
        f.write(f'Alignment mode: {args.alignment_mode}\n')  # v4.0 新增
        if args.alignment_mode == "structured":
            f.write(f'  - share_dim_proj: {args.map_proj_share_dim}\n')
            f.write(f'  - init_uniform: {args.map_proj_init_uniform}\n')
        f.write(f'KV method: {args.kv_method}\n')
        if args.kv_method == 'rkv_official':
            f.write(f'  R-KV lambda (importance weight): {args.rkv_lambda}\n')
            f.write(f'  R-KV keep_recent (generation mode): {args.rkv_keep_recent}\n')
        f.write(f'KV weight: {args.kv_weight}\n')
        f.write(f'CODI weight: {args.codi_weight}\n')
        f.write(f'Shuffle KV: {args.shuffle_kv}\n')
        f.write(f'Output dir: {args.output_dir}\n')
        f.write('=' * 50 + '\n')
        f.write(f'Final losses (Epoch {args.epochs}):\n')
        f.write(f'  CE: {ce_loss.item():.4f}\n')
        f.write(f'  KV: {kv_loss_total.item():.4f}\n')
        f.write(f'  CODI: {codi_loss.item():.4f}\n')
        f.write('=' * 50 + '\n')
    print(f'Training summary saved to {done_file}')

if __name__ == '__main__':
    main()
