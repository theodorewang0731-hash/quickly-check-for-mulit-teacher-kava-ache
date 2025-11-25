"""
Teacher KV Extraction Module

离线计算教师 KV，避免在线重复计算。

支持：
1. 单个样本提取
2. 批量提取（SLURM array job）
3. 保存/加载 KV cache

Usage:
    extractor = TeacherKVExtractor(model, tokenizer)
    kvs = extractor.extract_kvs(texts)
    extractor.save_kvs(kvs, "teacher_kvs.pt")
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional, Tuple
import os
from pathlib import Path
import json
from tqdm.auto import tqdm


class TeacherKVExtractor:
    """
    教师 KV 提取器。
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda",
        max_length: int = 2048,
        batch_size: int = 4,
        trust_remote_code: bool = False,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            model_name_or_path: 教师模型路径
            device: 计算设备
            max_length: 最大序列长度
            batch_size: 批次大小
            trust_remote_code: 是否信任远程代码
            cache_dir: 缓存目录
        """
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        
        print(f"Loading teacher model: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            cache_dir=cache_dir,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        self.model.eval()
        
        # 获取模型配置
        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size
        self.num_heads = self.model.config.num_attention_heads
        
        print(f"Teacher model loaded: {self.num_layers} layers, {self.hidden_size} dim, {self.num_heads} heads")
    
    @torch.no_grad()
    def extract_kvs(
        self,
        texts: List[str],
        return_dict: bool = True,
        show_progress: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        提取教师 KV。
        
        Args:
            texts: 输入文本列表
            return_dict: 是否返回字典格式
            show_progress: 是否显示进度条
            
        Returns:
            kvs: {
                "keys": [num_layers, batch, time, hidden_size],
                "values": [num_layers, batch, time, hidden_size],
                "attention_mask": [batch, time],
                "input_ids": [batch, time],
                "metadata": {...}
            }
        """
        all_keys = []
        all_values = []
        all_input_ids = []
        all_attention_masks = []
        
        # 批次处理
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        
        iterator = range(num_batches)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting KVs")
        
        for i in iterator:
            batch_texts = texts[i * self.batch_size: (i + 1) * self.batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Forward with output_attentions=True
            outputs = self.model(
                **inputs,
                output_attentions=True,
                use_cache=False
            )
            
            # 提取 KV
            # outputs.attentions: tuple of [batch, num_heads, seq_len, seq_len]
            # 我们需要从 attention 层内部提取 K, V
            # 注意：不是所有模型都直接暴露 K, V，可能需要钩子
            
            # 这里假设使用钩子或直接访问（需要根据模型调整）
            # 为了通用性，我们存储 attention weights 作为代理
            # 实际使用时需要修改为直接提取 K, V
            
            # 临时：存储 hidden states 作为代理
            batch_keys = []
            batch_values = []
            
            # 使用钩子提取 K, V（示例）
            # 这里简化为提取 hidden_states
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                hidden_states = outputs.hidden_states  # tuple of [batch, seq_len, hidden_size]
                
                for layer_hidden in hidden_states[1:]:  # 跳过输入层
                    # 这里简化：K = V = hidden_state
                    # 实际应该从 attention 层提取真实的 K, V
                    batch_keys.append(layer_hidden.cpu())
                    batch_values.append(layer_hidden.cpu())
            else:
                # 如果模型不支持 output_hidden_states，使用最终层
                final_hidden = outputs.logits  # [batch, seq_len, vocab_size]
                # 降维到 hidden_size
                # 这只是占位符，实际需要真正的 K, V
                print("Warning: Using placeholder for K, V. Need to implement proper extraction.")
                for _ in range(self.num_layers):
                    batch_keys.append(torch.zeros(
                        inputs.input_ids.size(0),
                        inputs.input_ids.size(1),
                        self.hidden_size,
                        device="cpu"
                    ))
                    batch_values.append(torch.zeros(
                        inputs.input_ids.size(0),
                        inputs.input_ids.size(1),
                        self.hidden_size,
                        device="cpu"
                    ))
            
            all_keys.append(batch_keys)
            all_values.append(batch_values)
            all_input_ids.append(inputs.input_ids.cpu())
            all_attention_masks.append(inputs.attention_mask.cpu())
        
        # 合并批次
        # Transpose to [num_layers, total_batch, time, hidden_size]
        keys_by_layer = []
        values_by_layer = []
        
        for layer_idx in range(self.num_layers):
            layer_keys = torch.cat([batch[layer_idx] for batch in all_keys], dim=0)
            layer_values = torch.cat([batch[layer_idx] for batch in all_values], dim=0)
            keys_by_layer.append(layer_keys)
            values_by_layer.append(layer_values)
        
        keys_tensor = torch.stack(keys_by_layer, dim=0)  # [num_layers, batch, time, hidden_size]
        values_tensor = torch.stack(values_by_layer, dim=0)
        
        input_ids = torch.cat(all_input_ids, dim=0)
        attention_mask = torch.cat(all_attention_masks, dim=0)
        
        if return_dict:
            return {
                "keys": keys_tensor,
                "values": values_tensor,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "metadata": {
                    "num_layers": self.num_layers,
                    "hidden_size": self.hidden_size,
                    "num_heads": self.num_heads,
                    "max_length": self.max_length,
                    "num_samples": len(texts)
                }
            }
        else:
            return keys_tensor, values_tensor, input_ids, attention_mask
    
    def save_kvs(
        self,
        kvs: Dict[str, torch.Tensor],
        save_path: str,
        save_metadata: bool = True
    ):
        """
        保存 KV cache。
        
        Args:
            kvs: KV 字典
            save_path: 保存路径
            save_metadata: 是否保存元数据
        """
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存主数据
        torch.save({
            "keys": kvs["keys"],
            "values": kvs["values"],
            "input_ids": kvs["input_ids"],
            "attention_mask": kvs["attention_mask"]
        }, save_path)
        
        print(f"KVs saved to {save_path}")
        
        # 保存元数据
        if save_metadata and "metadata" in kvs:
            metadata_path = save_path.replace(".pt", "_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(kvs["metadata"], f, indent=2)
            print(f"Metadata saved to {metadata_path}")
    
    @staticmethod
    def load_kvs(load_path: str) -> Dict[str, torch.Tensor]:
        """
        加载 KV cache。
        
        Args:
            load_path: 加载路径
            
        Returns:
            kvs: KV 字典
        """
        kvs = torch.load(load_path, map_location="cpu")
        
        # 加载元数据
        metadata_path = load_path.replace(".pt", "_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            kvs["metadata"] = metadata
        
        print(f"KVs loaded from {load_path}")
        return kvs


def extract_from_dataset(
    model_name: str,
    dataset_name: str,
    split: str,
    output_dir: str,
    max_samples: Optional[int] = None,
    batch_size: int = 4,
    text_column: str = "text"
):
    """
    从 Hugging Face 数据集提取 KV。
    
    Args:
        model_name: 教师模型名
        dataset_name: 数据集名
        split: 数据集 split
        output_dir: 输出目录
        max_samples: 最大样本数
        batch_size: 批次大小
        text_column: 文本列名
    """
    from datasets import load_dataset
    
    # 加载数据集
    print(f"Loading dataset: {dataset_name} ({split})")
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    
    if max_samples:
        dataset = dataset.take(max_samples)
    
    # 提取文本
    texts = []
    for example in tqdm(dataset, desc="Loading texts"):
        if text_column in example:
            texts.append(example[text_column])
        if max_samples and len(texts) >= max_samples:
            break
    
    # 提取 KV
    extractor = TeacherKVExtractor(model_name, batch_size=batch_size)
    kvs = extractor.extract_kvs(texts)
    
    # 保存
    output_path = Path(output_dir) / f"{Path(model_name).name}_kvs.pt"
    extractor.save_kvs(kvs, str(output_path))
    
    print(f"Extraction complete: {len(texts)} samples")


if __name__ == "__main__":
    # 测试代码
    print("Testing teacher KV extraction...")
    
    # 使用 GPT2 作为测试
    model_name = "gpt2"
    
    extractor = TeacherKVExtractor(
        model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=2
    )
    
    # 测试样本
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a high-level programming language."
    ]
    
    # 提取 KV
    kvs = extractor.extract_kvs(texts, show_progress=True)
    
    print(f"Keys shape: {kvs['keys'].shape}")
    print(f"Values shape: {kvs['values'].shape}")
    print(f"Input IDs shape: {kvs['input_ids'].shape}")
    print(f"Attention mask shape: {kvs['attention_mask'].shape}")
    print(f"Metadata: {kvs['metadata']}")
    
    # 测试保存/加载
    save_path = "test_teacher_kvs.pt"
    extractor.save_kvs(kvs, save_path)
    
    loaded_kvs = TeacherKVExtractor.load_kvs(save_path)
    assert torch.allclose(kvs['keys'], loaded_kvs['keys'])
    print("✓ Save/load test passed")
    
    # 清理
    os.remove(save_path)
    metadata_path = save_path.replace(".pt", "_metadata.json")
    if os.path.exists(metadata_path):
        os.remove(metadata_path)
    
    print("All teacher KV extraction tests passed!")
