"""
train_baseline.py

最小可运行基线：对 GSM8k 的小子集（默认 100 条）用 GPT-2 做微调。
功能：
- 下载/加载 GSM8k 数据集（Hugging Face `datasets`）并截取子集
- 构建 prompt + target 的因果语言建模样本，并对 prompt 部分屏蔽 loss（labels=-100）
- 使用 Hugging Face Trainer 微调 student 模型（默认与 teacher 同构的 gpt2）
- 简单的生成评估（decode + exact-match）作为参考指标

用法示例：
  python experiments/train_baseline.py --subset_size 100 --model_name gpt2 --output_dir outputs/baseline --epochs 3 --batch_size 4

"""
import argparse
import os
import math
from datasets import load_dataset

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subset_size", type=int, default=100)
    p.add_argument("--model_name", type=str, default="gpt2")
    p.add_argument("--output_dir", type=str, default="outputs/baseline")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--max_input_length", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def build_examples_from_gsm8k(dataset, subset_size):
    # gsm8k uses fields 'question' and 'answer'
    examples = []
    for i, item in enumerate(dataset):
        if i >= subset_size:
            break
        prompt = item.get("question") or item.get("problem") or ""
        target = item.get("answer") or item.get("answer_text") or ""
        # canonicalize target: keep text after '###' or newline if present
        examples.append({"prompt": prompt.strip(), "target": target.strip()})
    return examples

def main():
    args = parse_args()

    # Lazy imports that may be heavy
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
    import random
    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading GSM8k dataset (may download)...")
    ds = load_dataset("gsm8k", "main", split="train")
    examples = build_examples_from_gsm8k(ds, args.subset_size)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def to_text(item):
        return item["prompt"] + tokenizer.eos_token + item["target"]

    # Build tokenized datasets with labels masked for prompt tokens
    input_texts = [to_text(x) for x in examples]
    prompts = [x["prompt"] for x in examples]
    targets = [x["target"] for x in examples]

    encodings = tokenizer(input_texts, truncation=True, max_length=args.max_input_length, padding=True)

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    labels = []
    for i, text in enumerate(input_texts):
        # mask prompt tokens in labels
        prompt_enc = tokenizer(prompts[i], truncation=True, max_length=args.max_input_length)["input_ids"]
        lbl = input_ids[i].copy()
        # set prompt token positions to -100
        prompt_len = len(prompt_enc)
        for j in range(min(prompt_len, len(lbl))):
            lbl[j] = -100
        labels.append(lbl)

    from datasets import Dataset
    ds_train = Dataset.from_dict({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})

    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        seed=args.seed,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(args.output_dir)

    # Simple generation-based evaluation on the same small set
    print("Running simple generation evaluation (exact-match)...")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    exact = 0
    total = len(prompts)
    for p, t in zip(prompts, targets):
        input_ids = tokenizer(p, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            out = model.generate(input_ids, max_new_tokens=args.max_new_tokens, do_sample=False)
        gen = tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
        # normalization for exact match: keep digits/letters and lowercase
        def norm(s):
            return "".join(c for c in s.lower() if c.isalnum())
        if norm(gen) == norm(t):
            exact += 1
    acc = exact / total if total>0 else 0.0
    print(f"Exact-match on subset: {exact}/{total} = {acc:.4f}")

if __name__ == "__main__":
    main()
