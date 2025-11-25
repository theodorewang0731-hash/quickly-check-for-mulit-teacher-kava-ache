#!/usr/bin/env python3
"""
Pre-download models and datasets for offline training.
Run this script with internet access before submitting HPC jobs.

Usage:
    python scripts/download_assets.py --model Qwen/Qwen2-1.5B --dataset openai/gsm8k
"""
import argparse
import os
from pathlib import Path


def download_model(model_name, cache_dir):
    """Download model weights from Hugging Face."""
    print(f"Downloading model: {model_name}")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("  - Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        print(f"  ✓ Tokenizer saved to {cache_dir}")
        
        print("  - Downloading model weights...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            torch_dtype="auto"
        )
        print(f"  ✓ Model saved to {cache_dir}")
        print(f"  Model size: ~{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def download_dataset(dataset_name, dataset_config, cache_dir):
    """Download dataset from Hugging Face."""
    print(f"Downloading dataset: {dataset_name} ({dataset_config})")
    try:
        from datasets import load_dataset
        
        print("  - Fetching dataset...")
        ds = load_dataset(
            dataset_name,
            dataset_config,
            cache_dir=cache_dir
        )
        print(f"  ✓ Dataset saved to {cache_dir}")
        
        # Print dataset info
        for split, data in ds.items():
            print(f"    - {split}: {len(data)} examples")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def export_dataset_to_jsonl(dataset_name, dataset_config, split, output_file):
    """Export dataset to local JSONL file."""
    print(f"Exporting {dataset_name} to {output_file}")
    try:
        from datasets import load_dataset
        import json
        
        ds = load_dataset(dataset_name, dataset_config, split=split)
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        count = 0
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in ds:
                prompt = item.get('question') or item.get('problem') or ''
                target = item.get('answer') or item.get('answer_text') or ''
                json.dump({
                    'prompt': prompt,
                    'target': target
                }, f, ensure_ascii=False)
                f.write('\n')
                count += 1
        
        print(f"  ✓ Exported {count} examples to {output_file}")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download models and datasets for offline training")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B",
                        help="Model name from Hugging Face")
    parser.add_argument("--dataset", type=str, default="openai/gsm8k",
                        help="Dataset name from Hugging Face")
    parser.add_argument("--dataset_config", type=str, default="main",
                        help="Dataset configuration")
    parser.add_argument("--cache_dir", type=str, default="cache",
                        help="Cache directory for models and datasets")
    parser.add_argument("--export_jsonl", action="store_true",
                        help="Also export dataset to local JSONL file")
    parser.add_argument("--skip_model", action="store_true",
                        help="Skip model download")
    parser.add_argument("--skip_dataset", action="store_true",
                        help="Skip dataset download")
    args = parser.parse_args()
    
    # Get project directory
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    cache_dir = project_dir / args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("KaVa Asset Download")
    print("=" * 60)
    print(f"Project directory: {project_dir}")
    print(f"Cache directory: {cache_dir}")
    print()
    
    success = True
    
    # Download model
    if not args.skip_model:
        if not download_model(args.model, str(cache_dir)):
            success = False
        print()
    
    # Download dataset
    if not args.skip_dataset:
        if not download_dataset(args.dataset, args.dataset_config, str(cache_dir)):
            success = False
        print()
    
    # Export to JSONL
    if args.export_jsonl and not args.skip_dataset:
        data_dir = project_dir / "data"
        jsonl_file = data_dir / f"{args.dataset.replace('/', '_')}_{args.dataset_config}_train.jsonl"
        if not export_dataset_to_jsonl(args.dataset, args.dataset_config, "train", str(jsonl_file)):
            success = False
        print()
    
    print("=" * 60)
    if success:
        print("✓ All assets downloaded successfully")
        print()
        print("You can now:")
        print("1. Submit training jobs with: sbatch scripts/run_hpc_training.sh")
        print("2. Or use offline mode with: --train_file data/*.jsonl")
    else:
        print("✗ Some downloads failed. Check errors above.")
        return 1
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    exit(main())
