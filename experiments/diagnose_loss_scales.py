"""
Loss Scale Diagnostic Tool - 检查各 loss 数量级是否合理

用法：
    python experiments/diagnose_loss_scales.py \\
        --model_name Qwen/Qwen2-1.5B \\
        --teacher_model Qwen/Qwen2-7B \\
        --num_samples 10

目的：
    在正式训练前，快速检查：
    1. CE loss 数量级
    2. KV loss 数量级
    3. CODI loss 数量级
    4. CKA loss 数量级
    5. 各 loss 权重是否合理

老师反馈要点：
    "如果 L_KV ≈ 0.1，L_CKA ≈ 1e-3，那乘 0.05 基本就没什么影响；
     如果 L_CKA 数值很大，那 0.05 可能会压过 KV-loss。"
     
建议：
    - 先跑这个脚本看各 loss 数量级
    - 必要时调整 cka_weight 到 0.01 或 0.005
"""
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from experiments.kv_utils import rkv_greedy
from experiments.kv_loss import align_teacher_kv_to_student, compute_kv_loss
from experiments.cka_loss import multi_layer_cka_loss
from experiments.projector import StudentToTeacherProjector


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2-1.5B")
    p.add_argument("--teacher_model", type=str, default="Qwen/Qwen2-7B")
    p.add_argument("--num_samples", type=int, default=10, help="Number of samples to test")
    p.add_argument("--seq_len", type=int, default=128, help="Sequence length for testing")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def to_numpy_kv(past_key_values):
    """Convert tensors to numpy."""
    converted = []
    for layer in past_key_values:
        if layer is None:
            continue
        k, v = layer
        converted.append((k.cpu().numpy(), v.cpu().numpy()))
    return tuple(converted)


def main():
    args = parse_args()
    device = torch.device(args.device)
    
    print("=" * 80)
    print("Loss Scale Diagnostic Tool")
    print("=" * 80)
    print(f"Student: {args.model_name}")
    print(f"Teacher: {args.teacher_model}")
    print(f"Samples: {args.num_samples}, Batch: {args.batch_size}, Seq: {args.seq_len}")
    print(f"Device: {device}")
    print("=" * 80)
    
    # Load models
    print("\n[1/5] Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    student = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True).to(device)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model, trust_remote_code=True).to(device)
    
    student.eval()
    teacher.eval()
    
    print(f"✓ Student loaded: {sum(p.numel() for p in student.parameters()) / 1e9:.2f}B params")
    print(f"✓ Teacher loaded: {sum(p.numel() for p in teacher.parameters()) / 1e9:.2f}B params")
    
    # Collect loss statistics
    print(f"\n[2/5] Running {args.num_samples} forward passes...")
    
    ce_losses = []
    kv_losses = []
    kv_weighted_losses = []
    codi_losses = []
    cka_losses = []
    
    for i in range(args.num_samples):
        # Generate random input
        input_ids = torch.randint(0, tokenizer.vocab_size, (args.batch_size, args.seq_len), device=device)
        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            # Teacher forward
            t_out = teacher(
                input_ids, 
                attention_mask=attention_mask, 
                use_cache=True,
                output_hidden_states=True,
                output_attentions=True
            )
            teacher_pkv = t_out.past_key_values
            teacher_hidden = t_out.hidden_states[-1]
            teacher_all_hiddens = t_out.hidden_states
            teacher_attention = t_out.attentions[-1]
            
            # Student forward
            s_out = student(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True
            )
            logits = s_out.logits
            student_hidden = s_out.hidden_states[-1]
            student_all_hiddens = s_out.hidden_states
            student_attention = s_out.attentions[-1]
            
            # CE loss (with random labels)
            labels = input_ids.clone()
            labels[:, :args.seq_len // 2] = -100  # Mask first half
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = ce_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            ce_losses.append(ce_loss.item())
            
            # KV loss (unweighted)
            teacher_pkv_np = to_numpy_kv(teacher_pkv)
            comp = rkv_greedy(teacher_pkv_np, target_len=8, lambda_param=0.1)
            
            # Use first layer for simplicity
            teacher_k, teacher_v = comp[0]
            tk, student_seg = align_teacher_kv_to_student((teacher_k, teacher_v), student_hidden, method='right_crop')
            
            # Project (create projector on the fly)
            def feat_dim_of(k):
                import numpy as np
                if isinstance(k, np.ndarray):
                    if k.ndim == 4:
                        return k.shape[1] * k.shape[3]
                    elif k.ndim == 3:
                        return k.shape[2]
                return student.config.hidden_size
            
            td = feat_dim_of(teacher_k)
            proj = StudentToTeacherProjector(student.config.hidden_size, td).to(device)
            student_proj = proj(student_seg)
            
            # Unweighted KV loss
            kv_loss = compute_kv_loss(student_proj, tk, loss_type='smooth_l1')
            kv_losses.append(kv_loss.item())
            
            # Weighted KV loss (with student attention)
            kv_weighted_loss = compute_kv_loss(
                student_proj, 
                tk, 
                loss_type='smooth_l1',
                attention_weights=student_attention
            )
            kv_weighted_losses.append(kv_weighted_loss.item())
            
            # CODI loss
            codi_loss = F.mse_loss(student_hidden, teacher_hidden)
            codi_losses.append(codi_loss.item())
            
            # CKA loss (middle layer)
            cka_loss = multi_layer_cka_loss(
                student_all_hiddens, 
                teacher_all_hiddens, 
                layer_indices=[len(student_all_hiddens) // 2]
            )
            cka_losses.append(cka_loss.item())
        
        if (i + 1) % max(1, args.num_samples // 5) == 0:
            print(f"  Progress: {i + 1}/{args.num_samples}")
    
    # Statistics
    print("\n[3/5] Computing statistics...")
    
    def stats(losses, name):
        mean = sum(losses) / len(losses)
        std = (sum((x - mean) ** 2 for x in losses) / len(losses)) ** 0.5
        return mean, std
    
    ce_mean, ce_std = stats(ce_losses, "CE")
    kv_mean, kv_std = stats(kv_losses, "KV")
    kv_weighted_mean, kv_weighted_std = stats(kv_weighted_losses, "KV-weighted")
    codi_mean, codi_std = stats(codi_losses, "CODI")
    cka_mean, cka_std = stats(cka_losses, "CKA")
    
    # Print results
    print("\n" + "=" * 80)
    print("[4/5] Loss Scales (Mean ± Std)")
    print("=" * 80)
    print(f"CE Loss:          {ce_mean:10.4f} ± {ce_std:8.4f}")
    print(f"KV Loss:          {kv_mean:10.4f} ± {kv_std:8.4f}")
    print(f"KV-weighted:      {kv_weighted_mean:10.4f} ± {kv_weighted_std:8.4f}")
    print(f"CODI Loss:        {codi_mean:10.4f} ± {codi_std:8.4f}")
    print(f"CKA Loss:         {cka_mean:10.4f} ± {cka_std:8.4f}")
    print("=" * 80)
    
    # Weighted contributions
    print("\n[5/5] Weighted Contributions (with default weights)")
    print("=" * 80)
    kv_weight = 1.0
    codi_weight = 0.5
    cka_weight = 0.05
    
    kv_contrib = kv_weight * kv_mean
    codi_contrib = codi_weight * codi_mean
    cka_contrib = cka_weight * cka_mean
    total_loss = ce_mean + kv_contrib + codi_contrib + cka_contrib
    
    print(f"CE contribution:    {ce_mean:10.4f}  ({100 * ce_mean / total_loss:5.1f}%)")
    print(f"KV contribution:    {kv_contrib:10.4f}  ({100 * kv_contrib / total_loss:5.1f}%)  [weight={kv_weight}]")
    print(f"CODI contribution:  {codi_contrib:10.4f}  ({100 * codi_contrib / total_loss:5.1f}%)  [weight={codi_weight}]")
    print(f"CKA contribution:   {cka_contrib:10.4f}  ({100 * cka_contrib / total_loss:5.1f}%)  [weight={cka_weight}]")
    print("-" * 80)
    print(f"Total loss:         {total_loss:10.4f}")
    print("=" * 80)
    
    # Recommendations
    print("\n📋 Recommendations:")
    print("=" * 80)
    
    # Check CKA weight
    if cka_contrib / total_loss > 0.15:
        recommended_cka = cka_weight * 0.1 / (cka_contrib / total_loss)
        print(f"⚠️  CKA contribution ({100 * cka_contrib / total_loss:.1f}%) is too high!")
        print(f"   Recommended: --cka_weight {recommended_cka:.4f}")
    elif cka_contrib / total_loss < 0.01:
        recommended_cka = cka_weight * 0.05 / (cka_contrib / total_loss)
        print(f"ℹ️  CKA contribution ({100 * cka_contrib / total_loss:.1f}%) is very low.")
        print(f"   You can increase: --cka_weight {recommended_cka:.4f}")
    else:
        print(f"✅ CKA weight is reasonable ({100 * cka_contrib / total_loss:.1f}% of total)")
    
    # Check KV weight
    if kv_contrib / total_loss > 0.5:
        recommended_kv = kv_weight * 0.3 / (kv_contrib / total_loss)
        print(f"⚠️  KV contribution ({100 * kv_contrib / total_loss:.1f}%) dominates CE loss!")
        print(f"   Recommended: --kv_weight {recommended_kv:.4f}")
    elif kv_contrib / total_loss < 0.05:
        print(f"ℹ️  KV contribution ({100 * kv_contrib / total_loss:.1f}%) is very low.")
        print(f"   You can increase: --kv_weight {kv_weight * 2:.2f}")
    else:
        print(f"✅ KV weight is reasonable ({100 * kv_contrib / total_loss:.1f}% of total)")
    
    # Attention weighting effect
    kv_change = abs(kv_weighted_mean - kv_mean) / kv_mean * 100
    print(f"\n📊 Attention weighting effect: {kv_change:.1f}% change in KV loss")
    if kv_change < 5:
        print(f"   ⚠️  Very small change - attention weighting may not be effective")
    elif kv_change > 30:
        print(f"   ⚠️  Large change - may indicate unstable attention weights")
    else:
        print(f"   ✅ Reasonable change - attention weighting should work")
    
    print("=" * 80)
    print("\n✓ Diagnostic complete!")
    print("\nNext steps:")
    print("1. Adjust weights based on recommendations above")
    print("2. Run small-scale training experiment (--subset_size 5000 --epochs 2)")
    print("3. Monitor loss curves for stability")


if __name__ == "__main__":
    main()
