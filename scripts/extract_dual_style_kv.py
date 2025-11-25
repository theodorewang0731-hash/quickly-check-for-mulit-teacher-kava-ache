"""
双风格 KV 提取脚本
为每个问题的 CoT 和 Direct 两种风格提取教师模型的 KV Cache
自动处理对齐（tokenizer, time, layer, head/dim, RoPE）
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from typing import List, Dict, Tuple, Optional
import os
from pathlib import Path
from tqdm import tqdm
import json

# 导入对齐模块
import sys
sys.path.append(str(Path(__file__).parent.parent))
from align.tokenizer_align import TokenizerAligner
from align.time_align import TimeAligner
from align.layer_map import LayerMapper
from align.head_dim_adapter import HeadDimAdapter
from align.rope_scale import RoPEScaler


class DualStyleKVExtractor:
    """双风格 KV 提取器"""
    
    def __init__(
        self,
        teacher_models: List[str],
        student_model: str,
        output_dir: str,
        device: str = "cuda",
        max_length: int = 2048,
        batch_size: int = 4,
        kv_compression: str = "right",  # "full", "right", "r-kv"
    ):
        """
        Args:
            teacher_models: 教师模型列表（支持多个）
            student_model: 学生模型（用于对齐）
            output_dir: 输出目录
            device: 设备
            max_length: 最大序列长度
            batch_size: 批次大小
            kv_compression: KV 压缩策略
        """
        self.teacher_models_names = teacher_models
        self.student_model_name = student_model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.kv_compression = kv_compression
        
        # 加载学生模型（用于对齐）
        print(f"Loading student model: {student_model}")
        self.student_tokenizer = AutoTokenizer.from_pretrained(student_model, trust_remote_code=True)
        if self.student_tokenizer.pad_token is None:
            self.student_tokenizer.pad_token = self.student_tokenizer.eos_token
        
        # 加载教师模型
        self.teachers = []
        for teacher_name in teacher_models:
            print(f"Loading teacher model: {teacher_name}")
            tokenizer = AutoTokenizer.from_pretrained(teacher_name, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                teacher_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            model.eval()
            
            self.teachers.append({
                "name": teacher_name,
                "tokenizer": tokenizer,
                "model": model,
            })
        
        # 初始化对齐模块
        self._init_aligners()
    
    def _init_aligners(self):
        """初始化对齐模块"""
        print("\nInitializing aligners...")
        
        self.aligners = []
        for teacher in self.teachers:
            # Tokenizer 对齐
            tokenizer_aligner = TokenizerAligner(
                teacher["tokenizer"],
                self.student_tokenizer,
            )
            
            # Time 对齐
            time_aligner = TimeAligner(strategy="soft")
            
            # Layer 对齐
            teacher_layers = teacher["model"].config.num_hidden_layers
            student_layers = self.student_tokenizer.init_kwargs.get("num_hidden_layers", 24)
            layer_mapper = LayerMapper(
                teacher_layers=teacher_layers,
                student_layers=student_layers,
                strategy="ratio",
            )
            
            # Head/Dim 对齐
            teacher_config = teacher["model"].config
            student_config = {
                "hidden_size": 1024,  # 假设学生模型维度
                "num_attention_heads": 16,
            }
            head_dim_adapter = HeadDimAdapter(
                teacher_num_heads=teacher_config.num_attention_heads,
                teacher_head_dim=teacher_config.hidden_size // teacher_config.num_attention_heads,
                student_num_heads=student_config["num_attention_heads"],
                student_head_dim=student_config["hidden_size"] // student_config["num_attention_heads"],
            )
            
            # RoPE 对齐
            rope_scaler = RoPEScaler(
                teacher_max_position=teacher_config.max_position_embeddings,
                student_max_position=2048,
                scaling_strategy="ntk",
            )
            
            self.aligners.append({
                "tokenizer": tokenizer_aligner,
                "time": time_aligner,
                "layer": layer_mapper,
                "head_dim": head_dim_adapter,
                "rope": rope_scaler,
            })
        
        print("✓ Aligners initialized")
    
    @torch.no_grad()
    def extract_kv_single(
        self,
        teacher_idx: int,
        prompt: str,
        answer: str,
    ) -> Dict[str, torch.Tensor]:
        """
        为单个教师模型提取 KV
        
        Args:
            teacher_idx: 教师模型索引
            prompt: 输入提示
            answer: 答案（用于计算完整序列）
        
        Returns:
            包含 keys 和 values 的字典
        """
        teacher = self.teachers[teacher_idx]
        tokenizer = teacher["tokenizer"]
        model = teacher["model"]
        
        # Tokenize
        full_text = prompt + answer
        inputs = tokenizer(
            full_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass with KV cache
        outputs = model(
            **inputs,
            output_hidden_states=True,
            use_cache=True,
        )
        
        # 提取 KV
        past_key_values = outputs.past_key_values  # Tuple of (key, value) for each layer
        
        # 转换为张量格式
        keys = torch.stack([kv[0] for kv in past_key_values], dim=0)  # [num_layers, batch, num_heads, seq_len, head_dim]
        values = torch.stack([kv[1] for kv in past_key_values], dim=0)
        
        # KV 压缩
        if self.kv_compression == "right":
            # Right-crop：保留后半部分（关键推理步骤）
            seq_len = keys.size(3)
            crop_len = seq_len // 2
            keys = keys[:, :, :, -crop_len:, :]
            values = values[:, :, :, -crop_len:, :]
        elif self.kv_compression == "r-kv":
            # R-KV：使用滑动窗口 + 关键位置
            # 简化版：保留前 25% 和后 25%
            seq_len = keys.size(3)
            keep_len = seq_len // 4
            keys = torch.cat([keys[:, :, :, :keep_len, :], keys[:, :, :, -keep_len:, :]], dim=3)
            values = torch.cat([values[:, :, :, :keep_len, :], values[:, :, :, -keep_len:, :]], dim=3)
        # else: "full" - 不压缩
        
        return {
            "keys": keys.cpu(),
            "values": values.cpu(),
            "seq_len": keys.size(3),
        }
    
    def align_kv(
        self,
        kv_dict: Dict[str, torch.Tensor],
        teacher_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        对提取的 KV 进行对齐
        
        Args:
            kv_dict: 包含 keys 和 values 的字典
            teacher_idx: 教师模型索引
        
        Returns:
            对齐后的 KV 字典
        """
        aligner = self.aligners[teacher_idx]
        keys = kv_dict["keys"]
        values = kv_dict["values"]
        
        # Layer 对齐
        keys, values = aligner["layer"].align_kv(keys, values)
        
        # Head/Dim 对齐
        keys = aligner["head_dim"].adapt_kv(keys)
        values = aligner["head_dim"].adapt_kv(values)
        
        # Time 对齐（如果需要）
        # keys, values = aligner["time"].align_kv(keys, values, target_length=...)
        
        return {
            "keys": keys,
            "values": values,
            "seq_len": keys.size(3),
        }
    
    def extract_dataset(
        self,
        dataset: Dataset,
        output_name: str,
    ):
        """
        为整个数据集提取 KV
        
        Args:
            dataset: 数据集（应包含 prompt, answer, style 等字段）
            output_name: 输出文件名前缀
        """
        print(f"\n{'='*60}")
        print(f"Extracting KV for {len(dataset)} examples")
        print(f"{'='*60}")
        
        all_kv_data = []
        
        for idx, example in enumerate(tqdm(dataset, desc="Extracting KV")):
            prompt = example["prompt"]
            answer = example["answer"]
            style = example["style"]
            
            # 为每个教师提取 KV
            teacher_kvs = []
            for teacher_idx in range(len(self.teachers)):
                # 提取原始 KV
                kv_dict = self.extract_kv_single(teacher_idx, prompt, answer)
                
                # 对齐 KV
                aligned_kv = self.align_kv(kv_dict, teacher_idx)
                
                teacher_kvs.append(aligned_kv)
            
            # 保存元数据
            all_kv_data.append({
                "example_id": idx,
                "prompt": prompt,
                "answer": answer,
                "style": style,
                "task_type": example.get("task_type", "unknown"),
                "dataset": example.get("dataset", "unknown"),
                "teacher_kvs": teacher_kvs,  # List of aligned KV dicts
            })
            
            # 定期保存（避免内存溢出）
            if (idx + 1) % 1000 == 0:
                self._save_checkpoint(all_kv_data, output_name, idx + 1)
        
        # 最终保存
        self._save_final(all_kv_data, output_name)
        
        print(f"\n✓ KV extraction completed: {len(all_kv_data)} examples")
        print(f"✓ Saved to: {self.output_dir / output_name}")
    
    def _save_checkpoint(self, data: List[Dict], output_name: str, step: int):
        """保存检查点"""
        checkpoint_path = self.output_dir / f"{output_name}_checkpoint_{step}.pt"
        torch.save(data, checkpoint_path)
        print(f"\n✓ Checkpoint saved: {checkpoint_path}")
    
    def _save_final(self, data: List[Dict], output_name: str):
        """保存最终数据"""
        # 保存 PyTorch 格式（包含完整 KV）
        pt_path = self.output_dir / f"{output_name}.pt"
        torch.save(data, pt_path)
        
        # 保存 JSON 格式（仅元数据，用于检查）
        metadata = []
        for item in data:
            metadata.append({
                "example_id": item["example_id"],
                "prompt": item["prompt"][:200] + "...",  # 截断
                "style": item["style"],
                "task_type": item["task_type"],
                "dataset": item["dataset"],
                "num_teachers": len(item["teacher_kvs"]),
                "kv_shapes": [
                    {
                        "keys": list(kv["keys"].shape),
                        "values": list(kv["values"].shape),
                    }
                    for kv in item["teacher_kvs"]
                ],
            })
        
        json_path = self.output_dir / f"{output_name}_metadata.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved PyTorch data: {pt_path}")
        print(f"✓ Saved metadata: {json_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract dual-style KV from teacher models")
    parser.add_argument("--teacher_models", nargs="+", required=True, help="Teacher model names")
    parser.add_argument("--student_model", required=True, help="Student model name")
    parser.add_argument("--dataset_path", required=True, help="Path to prepared dataset (.pt or HF format)")
    parser.add_argument("--output_dir", default="./kv_cache", help="Output directory")
    parser.add_argument("--output_name", default="dual_style_kv", help="Output file prefix")
    parser.add_argument("--kv_compression", default="right", choices=["full", "right", "r-kv"])
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    
    args = parser.parse_args()
    
    # 加载数据集
    print(f"Loading dataset from: {args.dataset_path}")
    if args.dataset_path.endswith(".pt"):
        dataset = torch.load(args.dataset_path)
    else:
        from datasets import load_from_disk
        dataset = load_from_disk(args.dataset_path)
    
    print(f"Dataset size: {len(dataset)}")
    
    # 创建提取器
    extractor = DualStyleKVExtractor(
        teacher_models=args.teacher_models,
        student_model=args.student_model,
        output_dir=args.output_dir,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        kv_compression=args.kv_compression,
    )
    
    # 提取 KV
    extractor.extract_dataset(dataset, args.output_name)


if __name__ == "__main__":
    main()
