"""
多任务评测框架
支持：GSM8K, MATH500, BBH, GPQA, TruthfulQA, CMMLU, C-Eval
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from typing import List, Dict, Optional
import json
from pathlib import Path
from tqdm import tqdm
import re
import numpy as np


class MultiTaskEvaluator:
    """多任务评测器"""
    
    # 评测数据集配置
    EVAL_CONFIGS = {
        "gsm8k_test": {
            "path": "openai/gsm8k",
            "subset": "main",
            "split": "test",
            "metric": "exact_match",
            "task_type": "math",
        },
        "math500": {
            "path": "lighteval/MATH",
            "subset": "all",
            "split": "test",
            "metric": "exact_match",
            "task_type": "math",
            "max_samples": 500,
        },
        "bbh": {
            "path": "lukaemon/bbh",
            "subset": None,
            "split": "test",
            "metric": "exact_match",
            "task_type": "reasoning",
        },
        "gpqa": {
            "path": "Idavidrein/gpqa",
            "subset": "gpqa_main",
            "split": "train",  # GPQA 只有 train split
            "metric": "accuracy",
            "task_type": "qa",
        },
        "truthfulqa": {
            "path": "truthful_qa",
            "subset": "multiple_choice",
            "split": "validation",
            "metric": "accuracy",
            "task_type": "qa",
        },
        "cmmlu_subset": {
            "path": "haonan-li/cmmlu",
            "subset": "all",
            "split": "test",
            "metric": "accuracy",
            "task_type": "qa",
            "language": "zh",
            "max_samples": 1000,
        },
        "ceval_subset": {
            "path": "ceval/ceval-exam",
            "subset": "all",
            "split": "val",
            "metric": "accuracy",
            "task_type": "qa",
            "language": "zh",
            "max_samples": 1000,
        },
    }
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_length: int = 2048,
        batch_size: int = 8,
    ):
        """
        Args:
            model_path: 模型路径
            device: 设备
            max_length: 最大序列长度
            batch_size: 批次大小
        """
        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        
        # 加载模型
        print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print("✓ Model loaded")
    
    def load_eval_dataset(self, dataset_name: str) -> Dict:
        """加载评测数据集"""
        config = self.EVAL_CONFIGS[dataset_name]
        
        try:
            if config["subset"]:
                dataset = load_dataset(
                    config["path"],
                    config["subset"],
                    split=config["split"],
                    trust_remote_code=True
                )
            else:
                dataset = load_dataset(
                    config["path"],
                    split=config["split"],
                    trust_remote_code=True
                )
            
            # 采样（如果指定）
            if "max_samples" in config and len(dataset) > config["max_samples"]:
                dataset = dataset.shuffle(seed=42).select(range(config["max_samples"]))
            
            print(f"✓ Loaded {dataset_name}: {len(dataset)} examples")
            return {
                "dataset": dataset,
                "config": config,
            }
        except Exception as e:
            print(f"✗ Failed to load {dataset_name}: {e}")
            return None
    
    def format_prompt(self, example: Dict, dataset_name: str) -> str:
        """格式化评测提示"""
        config = self.EVAL_CONFIGS[dataset_name]
        task_type = config["task_type"]
        language = config.get("language", "en")
        
        # 根据数据集类型格式化
        if dataset_name == "gsm8k_test":
            question = example["question"]
            prompt = f"Question: {question}\n\nLet's solve this step by step:\n\nSolution:"
        
        elif dataset_name == "math500":
            problem = example["problem"]
            prompt = f"Problem: {problem}\n\nSolution:"
        
        elif dataset_name == "bbh":
            # BBH 有多个子任务
            question = example.get("input", "")
            prompt = f"Question: {question}\n\nAnswer:"
        
        elif dataset_name == "gpqa":
            question = example["question"]
            prompt = f"Question: {question}\n\nAnswer:"
        
        elif dataset_name == "truthfulqa":
            question = example["question"]
            prompt = f"Question: {question}\n\nAnswer:"
        
        elif dataset_name in ["cmmlu_subset", "ceval_subset"]:
            question = example["Question"]
            choices = "\n".join([
                f"A. {example.get('A', '')}",
                f"B. {example.get('B', '')}",
                f"C. {example.get('C', '')}",
                f"D. {example.get('D', '')}",
            ])
            prompt = f"问题：{question}\n\n选项：\n{choices}\n\n答案："
        
        else:
            prompt = str(example)
        
        return prompt
    
    def extract_answer(self, text: str, dataset_name: str) -> str:
        """从生成文本中提取答案"""
        if dataset_name in ["gsm8k_test", "math500"]:
            # 数学题：提取最后一个数字
            numbers = re.findall(r'-?\d+\.?\d*', text)
            return numbers[-1] if numbers else ""
        
        elif dataset_name in ["cmmlu_subset", "ceval_subset"]:
            # 选择题：提取 A/B/C/D
            match = re.search(r'[ABCD]', text.upper())
            return match.group(0) if match else ""
        
        else:
            # 其他：返回前 50 个字符
            return text.strip()[:50]
    
    def get_gold_answer(self, example: Dict, dataset_name: str) -> str:
        """获取标准答案"""
        if dataset_name == "gsm8k_test":
            answer = example["answer"]
            # GSM8K 答案格式："#### 42"
            match = re.search(r'####\s*(-?\d+\.?\d*)', answer)
            return match.group(1) if match else ""
        
        elif dataset_name == "math500":
            return example.get("answer", "")
        
        elif dataset_name in ["bbh", "gpqa", "truthfulqa"]:
            return example.get("answer", example.get("target", ""))
        
        elif dataset_name in ["cmmlu_subset", "ceval_subset"]:
            return example.get("Answer", "")
        
        else:
            return ""
    
    @torch.no_grad()
    def generate_answer(self, prompt: str) -> str:
        """生成答案"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            temperature=0.0,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 移除原始提示
        answer = generated_text[len(prompt):].strip()
        return answer
    
    def compute_metric(
        self,
        predictions: List[str],
        references: List[str],
        metric_type: str,
    ) -> float:
        """计算评测指标"""
        if metric_type == "exact_match":
            correct = sum(
                pred.strip().lower() == ref.strip().lower()
                for pred, ref in zip(predictions, references)
            )
            return correct / len(predictions) * 100
        
        elif metric_type == "accuracy":
            correct = sum(pred == ref for pred, ref in zip(predictions, references))
            return correct / len(predictions) * 100
        
        else:
            return 0.0
    
    def evaluate_dataset(self, dataset_name: str) -> Dict:
        """评测单个数据集"""
        print(f"\n{'='*60}")
        print(f"Evaluating: {dataset_name}")
        print(f"{'='*60}")
        
        # 加载数据集
        eval_data = self.load_eval_dataset(dataset_name)
        if eval_data is None:
            return None
        
        dataset = eval_data["dataset"]
        config = eval_data["config"]
        
        # 生成预测
        predictions = []
        references = []
        
        for example in tqdm(dataset, desc=f"Evaluating {dataset_name}"):
            # 格式化提示
            prompt = self.format_prompt(example, dataset_name)
            
            # 生成答案
            generated = self.generate_answer(prompt)
            
            # 提取答案
            pred_answer = self.extract_answer(generated, dataset_name)
            gold_answer = self.get_gold_answer(example, dataset_name)
            
            predictions.append(pred_answer)
            references.append(gold_answer)
        
        # 计算指标
        score = self.compute_metric(predictions, references, config["metric"])
        
        result = {
            "dataset": dataset_name,
            "metric": config["metric"],
            "score": score,
            "num_examples": len(dataset),
        }
        
        print(f"\n✓ {dataset_name} - {config['metric']}: {score:.2f}%")
        
        return result
    
    def evaluate_all(
        self,
        dataset_names: List[str],
        output_file: Optional[str] = None,
    ) -> Dict:
        """评测所有数据集"""
        print(f"\n{'='*60}")
        print(f"Multi-Task Evaluation")
        print(f"{'='*60}")
        print(f"Model: {self.model_path}")
        print(f"Datasets: {', '.join(dataset_names)}")
        print(f"{'='*60}")
        
        results = {}
        for dataset_name in dataset_names:
            result = self.evaluate_dataset(dataset_name)
            if result:
                results[dataset_name] = result
        
        # 计算平均分
        avg_score = np.mean([r["score"] for r in results.values()])
        results["average"] = avg_score
        
        # 打印总结
        print(f"\n{'='*60}")
        print(f"Evaluation Summary")
        print(f"{'='*60}")
        for dataset_name, result in results.items():
            if dataset_name != "average":
                print(f"{dataset_name:20s} - {result['metric']:15s}: {result['score']:6.2f}%")
        print(f"{'='*60}")
        print(f"{'Average':20s} - {'':15s}: {avg_score:6.2f}%")
        print(f"{'='*60}")
        
        # 保存结果
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Results saved to: {output_file}")
        
        return results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-task evaluation")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        default=["gsm8k_test", "math500", "bbh", "gpqa", "truthfulqa", "cmmlu_subset", "ceval_subset"],
        help="Datasets to evaluate",
    )
    parser.add_argument("--output_file", default=None, help="Output JSON file")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    
    args = parser.parse_args()
    
    # 创建评测器
    evaluator = MultiTaskEvaluator(
        model_path=args.model_path,
        device=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )
    
    # 运行评测
    results = evaluator.evaluate_all(
        dataset_names=args.eval_datasets,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
