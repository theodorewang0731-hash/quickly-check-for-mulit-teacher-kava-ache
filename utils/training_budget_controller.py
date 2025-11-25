"""
训练预算控制器 - 确保所有实验组使用相同的训练步数和计算资源

硬性控制：
1. 所有训练组（SFT, Single-Teacher, Multi-Teacher）使用相同的训练步数
2. 相同的数据 token 数（通过控制 batch_size × num_steps）
3. 相同的学习率调度策略
4. 相同的训练 epochs（如果基于 epoch）

用法：
    controller = TrainingBudgetController(
        total_tokens=1e9,  # 10亿 tokens
        batch_size=32,
        seq_length=512
    )
    
    # 计算所有实验组应使用的统一步数
    training_steps = controller.get_unified_training_steps()
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class TrainingBudgetController:
    """
    训练预算控制器
    
    确保所有实验组使用相同的计算预算，避免被质疑"多教师组训练更久"
    """
    
    def __init__(
        self,
        total_tokens: int,  # 总训练 token 数
        batch_size: int,
        seq_length: int,
        gradient_accumulation_steps: int = 1,
        num_gpus: int = 1,
        output_dir: str = "./training_budget"
    ):
        """
        Args:
            total_tokens: 所有实验组应看到的总 token 数
            batch_size: 每个 GPU 的 batch size
            seq_length: 序列长度
            gradient_accumulation_steps: 梯度累积步数
            num_gpus: GPU 数量
            output_dir: 输出目录（记录预算配置）
        """
        self.total_tokens = total_tokens
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_gpus = num_gpus
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 计算统一的训练步数
        self._calculate_unified_steps()
        
    def _calculate_unified_steps(self):
        """计算统一的训练步数"""
        # 每个训练步的有效 batch size
        self.effective_batch_size = (
            self.batch_size * 
            self.gradient_accumulation_steps * 
            self.num_gpus
        )
        
        # 每个训练步看到的 token 数
        self.tokens_per_step = self.effective_batch_size * self.seq_length
        
        # 所需的总训练步数
        self.unified_training_steps = int(self.total_tokens / self.tokens_per_step)
        
        print(f"\n{'='*60}")
        print(f"训练预算控制器配置")
        print(f"{'='*60}")
        print(f"总 Token 数: {self.total_tokens:,.0f}")
        print(f"每 GPU Batch Size: {self.batch_size}")
        print(f"序列长度: {self.seq_length}")
        print(f"梯度累积步数: {self.gradient_accumulation_steps}")
        print(f"GPU 数量: {self.num_gpus}")
        print(f"有效 Batch Size: {self.effective_batch_size}")
        print(f"每步 Token 数: {self.tokens_per_step:,.0f}")
        print(f"统一训练步数: {self.unified_training_steps:,.0f}")
        print(f"{'='*60}\n")
        
    def get_unified_training_steps(self) -> int:
        """返回所有实验组应使用的统一训练步数"""
        return self.unified_training_steps
    
    def get_training_config(self) -> Dict[str, Any]:
        """返回训练配置（用于所有实验组）"""
        return {
            "total_tokens": self.total_tokens,
            "unified_training_steps": self.unified_training_steps,
            "batch_size": self.batch_size,
            "seq_length": self.seq_length,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_gpus": self.num_gpus,
            "effective_batch_size": self.effective_batch_size,
            "tokens_per_step": self.tokens_per_step,
        }
    
    def save_config(self, filename: str = "training_budget_config.json"):
        """保存预算配置到文件"""
        config = self.get_training_config()
        config_path = self.output_dir / filename
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 训练预算配置已保存到: {config_path}")
        return config_path
    
    def verify_experiment_budget(
        self, 
        experiment_name: str,
        actual_steps: int,
        actual_tokens: Optional[int] = None
    ) -> bool:
        """
        验证某个实验是否符合预算控制
        
        Args:
            experiment_name: 实验名称（如 "baseline_sft", "single_teacher"）
            actual_steps: 实际训练步数
            actual_tokens: 实际训练 token 数（可选）
        
        Returns:
            是否符合预算
        """
        print(f"\n验证实验预算: {experiment_name}")
        print(f"  期望步数: {self.unified_training_steps}")
        print(f"  实际步数: {actual_steps}")
        
        # 允许 ±1% 的误差（可能由于数据集大小不整除）
        tolerance = 0.01
        step_ratio = actual_steps / self.unified_training_steps
        
        if abs(step_ratio - 1.0) <= tolerance:
            print(f"  ✓ 步数匹配（比例: {step_ratio:.4f}）")
            steps_ok = True
        else:
            print(f"  ✗ 步数不匹配（比例: {step_ratio:.4f}，超出 ±{tolerance*100}% 范围）")
            steps_ok = False
        
        # 如果提供了 token 数，也验证
        if actual_tokens is not None:
            expected_tokens = self.tokens_per_step * actual_steps
            token_ratio = actual_tokens / self.total_tokens
            
            print(f"  期望 Token 数: {self.total_tokens:,.0f}")
            print(f"  实际 Token 数: {actual_tokens:,.0f}")
            
            if abs(token_ratio - 1.0) <= tolerance:
                print(f"  ✓ Token 数匹配（比例: {token_ratio:.4f}）")
                tokens_ok = True
            else:
                print(f"  ✗ Token 数不匹配（比例: {token_ratio:.4f}）")
                tokens_ok = False
            
            return steps_ok and tokens_ok
        
        return steps_ok
    
    def create_hpc_slurm_snippet(self) -> str:
        """生成 HPC SLURM 脚本片段（确保所有实验使用相同配置）"""
        snippet = f"""
# ============================================================
# 训练预算控制 - 所有实验组使用相同配置
# ============================================================
export TOTAL_TOKENS={self.total_tokens}
export UNIFIED_TRAINING_STEPS={self.unified_training_steps}
export BATCH_SIZE={self.batch_size}
export SEQ_LENGTH={self.seq_length}
export GRADIENT_ACCUMULATION_STEPS={self.gradient_accumulation_steps}
export NUM_GPUS={self.num_gpus}
export EFFECTIVE_BATCH_SIZE={self.effective_batch_size}
export TOKENS_PER_STEP={self.tokens_per_step}

echo "========================================================"
echo "训练预算控制配置"
echo "========================================================"
echo "总 Token 数: $TOTAL_TOKENS"
echo "统一训练步数: $UNIFIED_TRAINING_STEPS"
echo "每 GPU Batch Size: $BATCH_SIZE"
echo "有效 Batch Size: $EFFECTIVE_BATCH_SIZE"
echo "========================================================"
"""
        return snippet
    
    def estimate_training_time(
        self,
        model_size_params: float,  # 模型参数量（单位：B）
        gpu_tflops: float = 300,  # GPU 算力（单位：TFLOPS）
        mfu: float = 0.4  # Model FLOPs Utilization
    ) -> Dict[str, float]:
        """
        估计训练时间
        
        Args:
            model_size_params: 模型参数量（单位：B，如 1.5 表示 1.5B）
            gpu_tflops: GPU 理论算力（TFLOPS）
            mfu: 模型 FLOPs 利用率（通常 0.3-0.5）
        
        Returns:
            训练时间估计（秒、小时、天）
        """
        # 每个 token 的 FLOPs（前向+反向 ≈ 6 × params）
        flops_per_token = 6 * model_size_params * 1e9
        
        # 总 FLOPs
        total_flops = flops_per_token * self.total_tokens
        
        # 实际算力（考虑 MFU）
        actual_tflops = gpu_tflops * mfu * self.num_gpus
        
        # 训练时间（秒）
        training_time_seconds = total_flops / (actual_tflops * 1e12)
        training_time_hours = training_time_seconds / 3600
        training_time_days = training_time_hours / 24
        
        return {
            "total_flops": total_flops,
            "training_time_seconds": training_time_seconds,
            "training_time_hours": training_time_hours,
            "training_time_days": training_time_days,
            "estimated_gpu_hours": training_time_hours * self.num_gpus
        }
    
    def print_budget_comparison_table(
        self,
        experiments: Dict[str, Dict[str, Any]]
    ):
        """
        打印预算对比表格
        
        Args:
            experiments: {
                "experiment_name": {
                    "actual_steps": int,
                    "actual_tokens": int,
                    "actual_time_hours": float
                }
            }
        """
        print(f"\n{'='*80}")
        print(f"训练预算对比表格")
        print(f"{'='*80}")
        print(f"{'实验名称':<30} {'训练步数':<15} {'Token数':<20} {'训练时间(h)':<15}")
        print(f"{'-'*80}")
        
        # 期望值
        print(f"{'期望值 (所有实验)':<30} {self.unified_training_steps:<15,} "
              f"{self.total_tokens:<20,} {'N/A':<15}")
        print(f"{'-'*80}")
        
        # 实际值
        for exp_name, exp_data in experiments.items():
            actual_steps = exp_data.get("actual_steps", 0)
            actual_tokens = exp_data.get("actual_tokens", 0)
            actual_time = exp_data.get("actual_time_hours", 0)
            
            # 计算偏差
            step_deviation = (actual_steps / self.unified_training_steps - 1.0) * 100
            token_deviation = (actual_tokens / self.total_tokens - 1.0) * 100
            
            print(f"{exp_name:<30} {actual_steps:<15,} ({step_deviation:+.1f}%) "
                  f"{actual_tokens:<20,} ({token_deviation:+.1f}%) "
                  f"{actual_time:<15.1f}")
        
        print(f"{'='*80}\n")


def load_budget_config(config_path: str) -> Dict[str, Any]:
    """从文件加载预算配置"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_fair_baseline_config(
    total_tokens: int = 1e9,
    batch_size: int = 32,
    seq_length: int = 512,
    num_gpus: int = 8
) -> TrainingBudgetController:
    """
    创建公平的基线配置（所有实验组通用）
    
    推荐配置：
    - total_tokens: 1B (快速实验) 或 10B (完整实验)
    - batch_size: 根据 GPU 显存调整
    - seq_length: 512 (推理任务) 或 2048 (长文本)
    """
    controller = TrainingBudgetController(
        total_tokens=total_tokens,
        batch_size=batch_size,
        seq_length=seq_length,
        gradient_accumulation_steps=4,
        num_gpus=num_gpus
    )
    
    # 保存配置
    controller.save_config()
    
    # 生成 SLURM 片段
    slurm_snippet = controller.create_hpc_slurm_snippet()
    with open(controller.output_dir / "slurm_snippet.sh", 'w') as f:
        f.write(slurm_snippet)
    
    print(f"✓ SLURM 配置片段已保存到: {controller.output_dir / 'slurm_snippet.sh'}")
    
    return controller


if __name__ == "__main__":
    # 示例：创建公平的训练预算配置
    print("创建公平的训练预算配置\n")
    
    # 配置 1: 快速实验（1B tokens）
    print("【配置 1】快速实验（1B tokens）")
    controller_fast = create_fair_baseline_config(
        total_tokens=1e9,  # 1B tokens
        batch_size=32,
        seq_length=512,
        num_gpus=8
    )
    
    # 估计训练时间
    time_estimate = controller_fast.estimate_training_time(
        model_size_params=1.5,  # 1.5B 模型
        gpu_tflops=300,  # A100 80GB
        mfu=0.4
    )
    print(f"\n1.5B 模型训练时间估计:")
    print(f"  总 FLOPs: {time_estimate['total_flops']:.2e}")
    print(f"  训练时间: {time_estimate['training_time_hours']:.1f} 小时 "
          f"({time_estimate['training_time_days']:.2f} 天)")
    print(f"  总 GPU-hours: {time_estimate['estimated_gpu_hours']:.1f}")
    
    print("\n" + "="*60 + "\n")
    
    # 配置 2: 完整实验（10B tokens）
    print("【配置 2】完整实验（10B tokens）")
    controller_full = create_fair_baseline_config(
        total_tokens=10e9,  # 10B tokens
        batch_size=32,
        seq_length=512,
        num_gpus=8
    )
    
    time_estimate_full = controller_full.estimate_training_time(
        model_size_params=1.5,
        gpu_tflops=300,
        mfu=0.4
    )
    print(f"\n1.5B 模型训练时间估计:")
    print(f"  训练时间: {time_estimate_full['training_time_hours']:.1f} 小时 "
          f"({time_estimate_full['training_time_days']:.2f} 天)")
    print(f"  总 GPU-hours: {time_estimate_full['estimated_gpu_hours']:.1f}")
    
    # 示例：验证实验预算
    print("\n" + "="*60)
    print("示例：验证实验预算")
    print("="*60)
    
    # 模拟三个实验
    experiments = {
        "Baseline: Standard SFT": {
            "actual_steps": controller_fast.unified_training_steps,
            "actual_tokens": int(controller_fast.total_tokens),
            "actual_time_hours": 4.5
        },
        "Baseline: Single Teacher": {
            "actual_steps": controller_fast.unified_training_steps,
            "actual_tokens": int(controller_fast.total_tokens),
            "actual_time_hours": 5.2
        },
        "Experimental: Multi-Teacher": {
            "actual_steps": controller_fast.unified_training_steps,
            "actual_tokens": int(controller_fast.total_tokens),
            "actual_time_hours": 6.1
        }
    }
    
    # 打印对比表格
    controller_fast.print_budget_comparison_table(experiments)
    
    # 验证每个实验
    for exp_name, exp_data in experiments.items():
        controller_fast.verify_experiment_budget(
            exp_name,
            exp_data["actual_steps"],
            exp_data["actual_tokens"]
        )
