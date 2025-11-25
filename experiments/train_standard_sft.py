"""
标准监督微调训练脚本（无 KV 蒸馏）
作为对照组，验证 KV 蒸馏的必要性
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Standard Supervised Fine-Tuning")
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, required=True,
                       help="Student model name or path")
    
    # Data arguments
    parser.add_argument("--dataset_name", type=str, default="multi_reasoning_cot_direct",
                       help="Dataset name")
    parser.add_argument("--train_samples", type=int, default=15000,
                       help="Number of training samples")
    parser.add_argument("--val_samples", type=int, default=2000,
                       help="Number of validation samples")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                       help="Batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                       help="Eval batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                       help="Warmup ratio")
    parser.add_argument("--bf16", type=bool, default=True,
                       help="Use bfloat16")
    parser.add_argument("--gradient_checkpointing", type=bool, default=True,
                       help="Use gradient checkpointing")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Max gradient norm")
    
    # Evaluation arguments
    parser.add_argument("--evaluation_strategy", type=str, default="steps",
                       choices=["no", "steps", "epoch"])
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Eval every N steps")
    parser.add_argument("--save_steps", type=int, default=1000,
                       help="Save every N steps")
    parser.add_argument("--logging_steps", type=int, default=50,
                       help="Log every N steps")
    parser.add_argument("--save_total_limit", type=int, default=2,
                       help="Maximum number of checkpoints to keep")
    parser.add_argument("--load_best_model_at_end", type=bool, default=True,
                       help="Load best model at end")
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss",
                       help="Metric for best model")
    parser.add_argument("--greater_is_better", type=bool, default=False,
                       help="Greater is better for metric")
    
    # Other arguments
    parser.add_argument("--report_to", type=str, default="tensorboard",
                       choices=["none", "tensorboard", "wandb"])
    parser.add_argument("--logging_dir", type=str, default=None,
                       help="Logging directory")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    return parser.parse_args()


def load_data(dataset_name, train_samples, val_samples):
    """加载数据集"""
    print(f"Loading dataset: {dataset_name}")
    
    # 使用我们的多任务数据集加载器
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from data.multi_task_dataset import MultiTaskReasoningDataset
    
    loader = MultiTaskReasoningDataset(
        base_datasets=["gsm8k", "svamp", "strategyqa"],
        extended_datasets=["math", "arc_challenge"],
        use_extended=True,
        train_samples=train_samples,
        val_samples=val_samples,
    )
    
    train_dataset, val_dataset = loader.load_and_prepare()
    
    return train_dataset, val_dataset


def preprocess_function(examples, tokenizer, max_length=2048):
    """预处理函数"""
    # 组合 prompt 和 answer
    texts = [
        prompt + " " + answer
        for prompt, answer in zip(examples["prompt"], examples["answer"])
    ]
    
    # Tokenize
    model_inputs = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding=False,  # 动态填充
    )
    
    # 标签就是输入（语言建模）
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    
    return model_inputs


def main():
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    
    print("="*80)
    print("Standard Supervised Fine-Tuning (Baseline)")
    print("="*80)
    print(f"Model: {args.model_name_or_path}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Output: {args.output_dir}")
    print(f"Training samples: {args.train_samples}")
    print(f"Validation samples: {args.val_samples}")
    print("="*80)
    
    # 加载模型和 tokenizer
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print(f"✓ Model loaded: {model.config.num_parameters() / 1e9:.2f}B parameters")
    
    # 加载数据
    print("\nLoading data...")
    train_dataset, val_dataset = load_data(
        args.dataset_name,
        args.train_samples,
        args.val_samples
    )
    
    # 预处理数据
    print("\nPreprocessing data...")
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Preprocessing train dataset"
    )
    
    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names,
        desc="Preprocessing validation dataset"
    )
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    
    # 数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 因果语言建模
    )
    
    # 训练参数
    logging_dir = args.logging_dir or f"{args.output_dir}/logs"
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=args.max_grad_norm,
        evaluation_strategy=args.evaluation_strategy,
        eval_steps=args.eval_steps if args.evaluation_strategy == "steps" else None,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        report_to=args.report_to,
        logging_dir=logging_dir,
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )
    
    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 训练
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    
    train_result = trainer.train()
    
    # 保存模型
    print("\n" + "="*80)
    print("Saving Model")
    print("="*80)
    
    trainer.save_model(f"{args.output_dir}/best_model")
    tokenizer.save_pretrained(f"{args.output_dir}/best_model")
    
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # 最终评估
    print("\n" + "="*80)
    print("Final Evaluation")
    print("="*80)
    
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)
    
    print("\n" + "="*80)
    print("✓ Training Completed!")
    print("="*80)
    print(f"Best model saved to: {args.output_dir}/best_model")
    print(f"Final eval loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
    print(f"\nNext step: Evaluate on test sets")
    print(f"  python evaluation/multi_task_eval.py \\")
    print(f"    --model_path {args.output_dir}/best_model \\")
    print(f"    --eval_datasets gsm8k_test math500 bbh gpqa truthfulqa cmmlu_subset ceval_subset \\")
    print(f"    --output_file {args.output_dir}/eval_results.json")
    print("="*80)


if __name__ == "__main__":
    main()
