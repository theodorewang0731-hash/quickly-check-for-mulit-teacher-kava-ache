"""
Multi-Teacher KV Distillation Training Script

多教师 KV 蒸馏训练，支持 5 个阶段：
Phase 1: Dual-prompt（双提示词）
Phase 2: Multi-sample（多样本内多教师）
Phase 3: Real multi-teacher（真正的多教师）
Phase 4: Routing（动态路由）
Phase 5: Z-space alignment（跨架构对齐）

Usage:
    python experiments/train_multi_teacher_kv.py \
        --student_model Qwen/Qwen2.5-0.5B-Instruct \
        --teacher_models gpt2 bert-base-uncased \
        --phase 3 \
        --fusion_method learnable \
        --router_type attention \
        --output_dir ./outputs/multi_teacher
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import argparse
from typing import Dict, List, Optional
from tqdm.auto import tqdm

# Import alignment modules
from align import (
    build_char_align_matrix,
    align_sequence_lengths,
    build_multi_teacher_layer_map,
    merge_multi_teacher_kvs,
    MultiTeacherHeadDimAdapter,
    MultiTeacherRoPEScaler
)

# Import teacher modules
from teacher import (
    TeacherKVExtractor,
    compute_multi_teacher_prototypes,
    compute_routing_weights
)

# Import fusion modules
from fuse import (
    fuse_kvs_fixed,
    fuse_kvs_similarity,
    fuse_kvs_learnable,
    LearnableRouter,
    EntropyRegularizer
)

# Import existing modules
from experiments.kv_loss import compute_kv_loss
from experiments.projector import StudentToTeacherProjector


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Teacher KV Distillation Training")
    
    # Model arguments
    parser.add_argument("--student_model", type=str, required=True, help="Student model name/path")
    parser.add_argument("--teacher_models", type=str, nargs="+", required=True, help="Teacher model names/paths")
    
    # Phase arguments
    parser.add_argument("--phase", type=int, default=3, choices=[1, 2, 3, 4, 5],
                      help="Training phase (1=dual-prompt, 2=multi-sample, 3=real-multi, 4=routing, 5=z-space)")
    
    # Fusion arguments
    parser.add_argument("--fusion_method", type=str, default="fixed",
                      choices=["fixed", "similarity", "learnable"],
                      help="KV fusion method")
    parser.add_argument("--router_type", type=str, default="mlp",
                      choices=["mlp", "gate", "attention"],
                      help="Router type (for learnable fusion)")
    parser.add_argument("--fixed_weights", type=float, nargs="+", default=None,
                      help="Fixed fusion weights (for fixed fusion)")
    
    # Alignment arguments
    parser.add_argument("--layer_mapping_strategy", type=str, default="ratio",
                      choices=["ratio", "uniform", "skip"],
                      help="Layer mapping strategy")
    parser.add_argument("--rope_scaling_method", type=str, default="ntk",
                      choices=["linear", "ntk", "dynamic"],
                      help="RoPE scaling method")
    
    # Loss arguments
    parser.add_argument("--kv_loss_type", type=str, default="mse",
                      choices=["mse", "smoothl1", "cosine"],
                      help="KV loss type")
    parser.add_argument("--lambda_k", type=float, default=1.0, help="Key loss weight")
    parser.add_argument("--lambda_v", type=float, default=1.0, help="Value loss weight")
    parser.add_argument("--beta_cos", type=float, default=0.1, help="Cosine loss weight")
    parser.add_argument("--gamma_kl", type=float, default=0.01, help="KL loss weight (attention)")
    parser.add_argument("--delta_ce", type=float, default=1.0, help="CE loss weight")
    parser.add_argument("--entropy_reg_strength", type=float, default=0.01,
                      help="Entropy regularization strength (for routing)")
    parser.add_argument("--entropy_target", type=str, default="specialized",
                      choices=["diverse", "specialized"],
                      help="Entropy regularization target")
    
    # Data arguments
    parser.add_argument("--dataset_name", type=str, default="wikitext",
                      help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1",
                      help="Dataset config")
    parser.add_argument("--max_samples", type=int, default=10000,
                      help="Maximum training samples")
    parser.add_argument("--max_length", type=int, default=512,
                      help="Maximum sequence length")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./outputs/multi_teacher",
                      help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                      help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                      help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                      help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                      help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=500,
                      help="Warmup steps")
    parser.add_argument("--save_steps", type=int, default=1000,
                      help="Save checkpoint steps")
    parser.add_argument("--logging_steps", type=int, default=100,
                      help="Logging steps")
    parser.add_argument("--fp16", action="store_true",
                      help="Use FP16 mixed precision")
    parser.add_argument("--bf16", action="store_true",
                      help="Use BF16 mixed precision")
    
    # HPC arguments
    parser.add_argument("--gradient_checkpointing", action="store_true",
                      help="Enable gradient checkpointing")
    parser.add_argument("--cache_dir", type=str, default=None,
                      help="Cache directory for models/datasets")
    parser.add_argument("--trust_remote_code", action="store_true",
                      help="Trust remote code")
    
    return parser.parse_args()


class MultiTeacherKVTrainer:
    """
    多教师 KV 蒸馏训练器。
    """
    
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load student model
        print(f"Loading student model: {args.student_model}")
        self.student_tokenizer = AutoTokenizer.from_pretrained(
            args.student_model,
            trust_remote_code=args.trust_remote_code,
            cache_dir=args.cache_dir
        )
        self.student_model = AutoModelForCausalLM.from_pretrained(
            args.student_model,
            trust_remote_code=args.trust_remote_code,
            cache_dir=args.cache_dir,
            torch_dtype=torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32),
            device_map="auto"
        )
        
        if args.gradient_checkpointing:
            self.student_model.gradient_checkpointing_enable()
        
        # Get student config
        self.student_config = self.student_model.config
        self.student_num_layers = self.student_config.num_hidden_layers
        self.student_hidden_size = self.student_config.hidden_size
        self.student_num_heads = self.student_config.num_attention_heads
        
        # Load teacher models
        print(f"Loading {len(args.teacher_models)} teacher models...")
        self.teacher_extractors = []
        self.teacher_configs = []
        
        for teacher_name in args.teacher_models:
            print(f"  - {teacher_name}")
            extractor = TeacherKVExtractor(
                teacher_name,
                device=self.device,
                max_length=args.max_length,
                batch_size=args.per_device_train_batch_size,
                trust_remote_code=args.trust_remote_code,
                cache_dir=args.cache_dir
            )
            self.teacher_extractors.append(extractor)
            self.teacher_configs.append({
                "num_layers": extractor.num_layers,
                "hidden_size": extractor.hidden_size,
                "num_heads": extractor.num_heads
            })
        
        # Build alignment modules
        self._build_alignment_modules()
        
        # Build fusion modules
        self._build_fusion_modules()
        
        # Build loss modules
        self._build_loss_modules()
        
        print(f"Multi-teacher KV distillation initialized (Phase {args.phase})")
    
    def _build_alignment_modules(self):
        """构建对齐模块。"""
        # Layer mapping
        teacher_layer_counts = [cfg["num_layers"] for cfg in self.teacher_configs]
        self.layer_maps = build_multi_teacher_layer_map(
            teacher_layer_counts,
            self.student_num_layers,
            strategy=self.args.layer_mapping_strategy
        )
        
        # Head/Dim adapter
        teacher_configs_hd = [
            (cfg["hidden_size"], cfg["num_heads"])
            for cfg in self.teacher_configs
        ]
        self.head_dim_adapter = MultiTeacherHeadDimAdapter(
            teacher_configs_hd,
            self.student_hidden_size,
            self.student_num_heads,
            use_conv=False,
            init_identity=True
        ).to(self.device)
        
        # RoPE scaler
        teacher_configs_rope = [
            (10000.0, self.args.max_length)  # Assuming standard RoPE
            for _ in self.teacher_configs
        ]
        self.rope_scaler = MultiTeacherRoPEScaler(
            teacher_configs_rope,
            self.args.max_length,
            scaling_method=self.args.rope_scaling_method
        )
        
        print("Alignment modules built")
    
    def _build_fusion_modules(self):
        """构建融合模块。"""
        num_teachers = len(self.args.teacher_models)
        
        if self.args.fusion_method == "learnable":
            self.router = LearnableRouter(
                self.student_hidden_size,
                num_teachers,
                router_type=self.args.router_type,
                num_layers=2,
                dropout=0.1
            ).to(self.device)
        else:
            self.router = None
        
        # Entropy regularizer (for routing)
        if self.args.phase >= 4:
            self.entropy_regularizer = EntropyRegularizer(
                target=self.args.entropy_target,
                strength=self.args.entropy_reg_strength
            )
        else:
            self.entropy_regularizer = None
        
        print(f"Fusion modules built (method={self.args.fusion_method})")
    
    def _build_loss_modules(self):
        """构建损失模块。"""
        # KV loss
        self.kv_loss_fn = lambda student_kv, teacher_kv: compute_kv_loss(
            student_kv,
            teacher_kv,
            loss_type=self.args.kv_loss_type,
            reduction="mean"
        )
        
        print("Loss modules built")
    
    def extract_teacher_kvs(self, texts: List[str]) -> List[Dict]:
        """
        提取所有教师的 KV。
        
        Returns:
            List of teacher KV dicts
        """
        teacher_kvs_list = []
        
        for extractor in self.teacher_extractors:
            kvs = extractor.extract_kvs(texts, show_progress=False)
            teacher_kvs_list.append(kvs)
        
        return teacher_kvs_list
    
    def align_teacher_kvs(
        self,
        teacher_kvs_list: List[Dict],
        student_input_ids: torch.Tensor
    ) -> tuple:
        """
        对齐所有教师的 KV。
        
        Returns:
            aligned_ks: List of aligned K tensors per layer
            aligned_vs: List of aligned V tensors per layer
        """
        num_teachers = len(teacher_kvs_list)
        
        # For simplicity, assume all teachers use same sequence length
        # In practice, need tokenizer alignment matrix
        
        aligned_ks_all = []
        aligned_vs_all = []
        
        for teacher_kvs, layer_map in zip(teacher_kvs_list, self.layer_maps):
            # Extract K, V per layer
            # Note: extract_teacher_kv.py returns placeholder, need proper extraction
            keys = teacher_kvs["keys"]  # [num_layers, batch, time, hidden_size]
            values = teacher_kvs["values"]
            
            # Apply layer mapping
            keys_list = [keys[l] for l in range(keys.size(0))]
            values_list = [values[l] for l in range(values.size(0))]
            
            from align import interpolate_teacher_layers
            aligned_keys = interpolate_teacher_layers(keys_list, layer_map, self.student_num_layers)
            aligned_values = interpolate_teacher_layers(values_list, layer_map, self.student_num_layers)
            
            # Apply head/dim adapter
            # aligned_keys: List of [batch, time, dim]
            # Need to adapt per layer
            for l in range(self.student_num_layers):
                aligned_keys[l] = aligned_keys[l].to(self.device)
                aligned_values[l] = aligned_values[l].to(self.device)
            
            aligned_ks_all.append(aligned_keys)
            aligned_vs_all.append(aligned_values)
        
        return aligned_ks_all, aligned_vs_all
    
    def fuse_teacher_kvs(
        self,
        aligned_ks_all: List[List[torch.Tensor]],
        aligned_vs_all: List[List[torch.Tensor]],
        student_hidden: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        融合多个教师的 KV。
        
        Returns:
            fused_ks: List of fused K per layer
            fused_vs: List of fused V per layer
            routing_weights: Optional routing weights
        """
        num_layers = self.student_num_layers
        fused_ks = []
        fused_vs = []
        routing_weights = None
        
        for l in range(num_layers):
            # Get all teachers' KV for layer l
            ks_layer = [teacher_ks[l] for teacher_ks in aligned_ks_all]
            vs_layer = [teacher_vs[l] for teacher_vs in aligned_vs_all]
            
            # Fuse
            if self.args.fusion_method == "fixed":
                fused_k = fuse_kvs_fixed(ks_layer, weights=self.args.fixed_weights)
                fused_v = fuse_kvs_fixed(vs_layer, weights=self.args.fixed_weights)
                weights = None
            
            elif self.args.fusion_method == "similarity":
                # Compute prototypes (simplified)
                prototypes = [k.mean(dim=(0, 1)) for k in ks_layer]
                query = student_hidden.mean(dim=1) if student_hidden is not None else torch.zeros(1, self.student_hidden_size).to(self.device)
                
                fused_k, weights = fuse_kvs_similarity(ks_layer, query, prototypes)
                fused_v, _ = fuse_kvs_similarity(vs_layer, query, prototypes)
            
            elif self.args.fusion_method == "learnable":
                query = student_hidden.mean(dim=1) if student_hidden is not None else torch.zeros(1, self.student_hidden_size).to(self.device)
                
                fused_k, weights = fuse_kvs_learnable(ks_layer, query, self.router)
                fused_v, _ = fuse_kvs_learnable(vs_layer, query, self.router)
            
            else:
                raise ValueError(f"Unknown fusion method: {self.args.fusion_method}")
            
            fused_ks.append(fused_k)
            fused_vs.append(fused_v)
            
            if weights is not None and routing_weights is None:
                routing_weights = weights
        
        return fused_ks, fused_vs, routing_weights
    
    def compute_multi_teacher_loss(
        self,
        student_model,
        inputs,
        fused_ks,
        fused_vs
    ) -> Dict[str, torch.Tensor]:
        """
        计算多教师蒸馏损失。
        
        Returns:
            losses: Dict of losses
        """
        # Forward student with output_attentions
        outputs = student_model(
            **inputs,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False
        )
        
        # Extract student KV (placeholder - need proper extraction)
        # For now, use hidden states as proxy
        student_hidden_states = outputs.hidden_states
        
        # Compute KV loss per layer
        total_kv_loss = 0.0
        layer_losses = []
        
        for l in range(self.student_num_layers):
            if l + 1 < len(student_hidden_states):
                student_h = student_hidden_states[l + 1]  # [batch, time, dim]
                
                # Use as K, V proxy
                k_loss = self.kv_loss_fn(student_h, fused_ks[l])
                v_loss = self.kv_loss_fn(student_h, fused_vs[l])
                
                layer_loss = self.args.lambda_k * k_loss + self.args.lambda_v * v_loss
                total_kv_loss += layer_loss
                layer_losses.append(layer_loss.item())
        
        # CE loss
        ce_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0).to(self.device)
        
        # Total loss
        total_loss = self.args.delta_ce * ce_loss + total_kv_loss
        
        return {
            "loss": total_loss,
            "ce_loss": ce_loss,
            "kv_loss": total_kv_loss,
            "layer_losses": layer_losses
        }
    
    def train(self):
        """
        执行训练。
        """
        # Load dataset
        print(f"Loading dataset: {self.args.dataset_name}")
        dataset = load_dataset(
            self.args.dataset_name,
            self.args.dataset_config,
            split="train",
            streaming=True
        )
        
        if self.args.max_samples:
            dataset = dataset.take(self.args.max_samples)
        
        # Tokenize dataset
        def tokenize_function(examples):
            return self.student_tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.args.max_length,
                padding="max_length"
            )
        
        print("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.args.output_dir,
            num_train_epochs=self.args.num_train_epochs,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            warmup_steps=self.args.warmup_steps,
            save_steps=self.args.save_steps,
            logging_steps=self.args.logging_steps,
            fp16=self.args.fp16,
            bf16=self.args.bf16,
            save_total_limit=3,
            load_best_model_at_end=False,
            remove_unused_columns=False
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.student_tokenizer,
            mlm=False
        )
        
        # Custom training loop (simplified)
        # In practice, subclass Trainer for custom compute_loss
        
        print(f"Starting training (Phase {self.args.phase})...")
        print(f"Fusion method: {self.args.fusion_method}")
        print(f"Teacher models: {self.args.teacher_models}")
        
        # Placeholder: Use standard Trainer
        # Real implementation needs custom Trainer with multi-teacher KV loss
        
        trainer = Trainer(
            model=self.student_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator
        )
        
        trainer.train()
        
        # Save final model
        print(f"Saving model to {self.args.output_dir}")
        trainer.save_model()
        
        print("Training complete!")


def main():
    args = parse_args()
    
    # Create trainer
    trainer = MultiTeacherKVTrainer(args)
    
    # Train
    trainer.train()


if __name__ == "__main__":
    main()
