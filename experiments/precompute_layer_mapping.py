"""
预计算 CKA 层映射（训练前运行一次）

使用方法:
    python experiments/precompute_layer_mapping.py \
        --student_model Qwen/Qwen2-1.5B \
        --teacher_model Qwen/Qwen2-7B \
        --dataset_name openai/gsm8k \
        --num_samples 100 \
        --output layer_mapping_qwen15b_7b.json
        
然后在训练时使用:
    python experiments/train_with_kv.py \
        --use_cka_layer_mapping \
        --layer_mapping_path layer_mapping_qwen15b_7b.json \
        ...
"""

import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from experiments.alignment_v2 import precompute_layer_mapping


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute CKA-based layer mapping")
    
    # Models
    parser.add_argument("--student_model", type=str, required=True, help="Student model name/path")
    parser.add_argument("--teacher_model", type=str, required=True, help="Teacher model name/path")
    
    # Dataset
    parser.add_argument("--dataset_name", type=str, default="openai/gsm8k", help="Dataset for computing CKA")
    parser.add_argument("--dataset_config", type=str, default="main", help="Dataset config")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to use")
    
    # Output
    parser.add_argument("--output", type=str, default="layer_mapping.json", help="Output path")
    
    # Hardware
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    
    print("=" * 80)
    print("CKA Layer Mapping Precomputation")
    print("=" * 80)
    print(f"Student: {args.student_model}")
    print(f"Teacher: {args.teacher_model}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Samples: {args.num_samples}")
    print(f"Output: {args.output}")
    print("=" * 80)
    
    # Load tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer loaded: {len(tokenizer)} tokens")
    
    # Load models
    print("\n[2/5] Loading models...")
    student = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32
    ).to(device)
    print(f"✓ Student loaded: {sum(p.numel() for p in student.parameters()) / 1e9:.2f}B params")
    
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32
    ).to(device)
    print(f"✓ Teacher loaded: {sum(p.numel() for p in teacher.parameters()) / 1e9:.2f}B params")
    
    # Load dataset
    print(f"\n[3/5] Loading dataset: {args.dataset_name}...")
    try:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config,
            split=args.dataset_split
        )
        
        # Take subset
        if len(dataset) > args.num_samples * 2:
            dataset = dataset.select(range(args.num_samples * 2))
        
        print(f"✓ Dataset loaded: {len(dataset)} examples")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        print("Using random data for demonstration...")
        dataset = None
    
    # Prepare dataloader
    print("\n[4/5] Preparing dataloader...")
    
    def collate_fn(examples):
        if dataset is None:
            # Random data
            input_ids = torch.randint(0, tokenizer.vocab_size, (args.batch_size, args.max_length))
            attention_mask = torch.ones_like(input_ids)
        else:
            # Real data
            texts = []
            for ex in examples:
                if 'question' in ex:
                    texts.append(ex['question'])
                elif 'text' in ex:
                    texts.append(ex['text'])
                else:
                    texts.append(str(ex))
            
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors='pt'
            )
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
        
        return {'input_ids': input_ids, 'attention_mask': attention_mask}
    
    if dataset is None:
        # Create dummy dataset
        class DummyDataset:
            def __len__(self):
                return args.num_samples
            def __getitem__(self, idx):
                return {}
        dataset = DummyDataset()
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    print(f"✓ Dataloader ready: {len(dataloader)} batches")
    
    # Compute layer mapping
    print("\n[5/5] Computing CKA layer mapping...")
    mapper = precompute_layer_mapping(
        student_model=student,
        teacher_model=teacher,
        dataloader=dataloader,
        num_samples=args.num_samples,
        output_path=args.output,
        device=device
    )
    
    # Summary
    print("\n" + "=" * 80)
    print("✓ Precomputation complete!")
    print("=" * 80)
    print(f"Layer mapping saved to: {args.output}")
    print(f"\nTo use in training:")
    print(f"  python experiments/train_with_kv.py \\")
    print(f"      --use_cka_layer_mapping \\")
    print(f"      --layer_mapping_path {args.output} \\")
    print(f"      ...")
    print("=" * 80)


if __name__ == "__main__":
    main()
