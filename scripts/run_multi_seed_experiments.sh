#!/bin/bash

################################################################################
# 多随机种子实验运行脚本
# 
# 硬性控制：每个实验配置至少运行 3 个不同的随机种子
# 用于后续的统计显著性检验（mean ± std, t-test, bootstrap CI）
#
# 用法:
#   sbatch scripts/run_multi_seed_experiments.sh
################################################################################

#SBATCH --job-name=multi_seed_experiments
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8                # 根据 HPC 要求可能需要完整格式，如 gpu:a100-sxm4-80gb:8
#SBATCH --cpus-per-task=64          # 根据 HPC 限制可能需要调整（建议 4-8 核/GPU）
#SBATCH --mem=0                     # 使用节点全部内存，或指定如 512G
#SBATCH --time=120:00:00
#SBATCH --output=logs/multi_seed_%j.out
#SBATCH --error=logs/multi_seed_%j.err

# ============================================================
# 配置区域
# ============================================================

# 实验配置
EXPERIMENT_NAME="baseline_sft"  # 或 "single_teacher", "multi_teacher_fixed", etc.
SEEDS=(42 43 44)  # 至少 3 个随机种子

# 训练预算控制（确保所有种子使用相同配置）
TOTAL_TOKENS=1000000000  # 1B tokens
BATCH_SIZE=32
SEQ_LENGTH=512
GRADIENT_ACCUMULATION_STEPS=4
NUM_GPUS=8

# 模型配置
STUDENT_MODEL="Qwen/Qwen2.5-1.5B"
TEACHER_MODELS="Qwen/Qwen2.5-7B,Qwen/Qwen2.5-14B"  # 仅用于多教师实验

# 数据配置
DATASET_NAME="multi_reasoning_cot_direct"
TRAIN_SPLIT="train"
VAL_SPLIT="validation"

# 输出配置
BASE_OUTPUT_DIR="./outputs/${EXPERIMENT_NAME}"

# ============================================================
# 环境设置
# ============================================================

# 使用统一的环境配置脚本（自动检测 CUDA 和 Python 环境）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_hpc_environment.sh"

# 创建日志目录
mkdir -p logs
mkdir -p "${BASE_OUTPUT_DIR}"

# 打印配置
echo "========================================================"
echo "多随机种子实验配置"
echo "========================================================"
echo "实验名称: ${EXPERIMENT_NAME}"
echo "随机种子: ${SEEDS[@]}"
echo "学生模型: ${STUDENT_MODEL}"
echo "总 Token 数: ${TOTAL_TOKENS}"
echo "输出目录: ${BASE_OUTPUT_DIR}"
echo "========================================================"

# ============================================================
# 计算统一训练步数（所有种子使用相同步数）
# ============================================================

EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))
TOKENS_PER_STEP=$((EFFECTIVE_BATCH_SIZE * SEQ_LENGTH))
UNIFIED_TRAINING_STEPS=$((TOTAL_TOKENS / TOKENS_PER_STEP))

echo ""
echo "训练预算控制:"
echo "  有效 Batch Size: ${EFFECTIVE_BATCH_SIZE}"
echo "  每步 Token 数: ${TOKENS_PER_STEP}"
echo "  统一训练步数: ${UNIFIED_TRAINING_STEPS}"
echo "========================================================"
echo ""

# ============================================================
# 主循环：对每个随机种子运行实验
# ============================================================

for SEED in "${SEEDS[@]}"; do
    echo "========================================================"
    echo "开始运行实验 - 随机种子: ${SEED}"
    echo "========================================================"
    
    # 创建种子专用输出目录
    SEED_OUTPUT_DIR="${BASE_OUTPUT_DIR}/seed_${SEED}"
    mkdir -p "${SEED_OUTPUT_DIR}"
    
    # 根据实验类型选择不同的训练脚本
    if [ "${EXPERIMENT_NAME}" == "baseline_sft" ]; then
        # ============================================================
        # 基线 1: 标准监督微调（无 KV 蒸馏）
        # ============================================================
        echo "训练配置: 标准 SFT（无 KV 蒸馏）"
        
        python experiments/train_standard_sft.py \
            --model_name_or_path "${STUDENT_MODEL}" \
            --dataset_name "${DATASET_NAME}" \
            --train_split "${TRAIN_SPLIT}" \
            --val_split "${VAL_SPLIT}" \
            --output_dir "${SEED_OUTPUT_DIR}" \
            --per_device_train_batch_size ${BATCH_SIZE} \
            --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
            --max_steps ${UNIFIED_TRAINING_STEPS} \
            --learning_rate 2e-5 \
            --warmup_steps $((UNIFIED_TRAINING_STEPS / 10)) \
            --logging_steps 50 \
            --save_steps $((UNIFIED_TRAINING_STEPS / 5)) \
            --eval_steps $((UNIFIED_TRAINING_STEPS / 10)) \
            --bf16 \
            --seed ${SEED} \
            --report_to tensorboard \
            --gradient_checkpointing
    
    elif [ "${EXPERIMENT_NAME}" == "single_teacher" ]; then
        # ============================================================
        # 基线 2: 单教师 KV 蒸馏
        # ============================================================
        echo "训练配置: 单教师 KV 蒸馏"
        
        # 提取第一个教师
        TEACHER_MODEL=$(echo ${TEACHER_MODELS} | cut -d',' -f1)
        
        python experiments/train_with_kv.py \
            --model_name_or_path "${STUDENT_MODEL}" \
            --teacher_models "${TEACHER_MODEL}" \
            --teacher_kv_dir "./kv_caches/single_teacher/seed_${SEED}" \
            --dataset_name "${DATASET_NAME}" \
            --output_dir "${SEED_OUTPUT_DIR}" \
            --per_device_train_batch_size ${BATCH_SIZE} \
            --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
            --max_steps ${UNIFIED_TRAINING_STEPS} \
            --learning_rate 2e-5 \
            --kv_loss_weight 0.5 \
            --ce_loss_weight 0.5 \
            --seed ${SEED} \
            --bf16 \
            --gradient_checkpointing
    
    elif [ "${EXPERIMENT_NAME}" == "multi_teacher_fixed" ]; then
        # ============================================================
        # 实验组 1: 多教师固定权重
        # ============================================================
        echo "训练配置: 多教师固定权重 (0.5/0.5)"
        
        python experiments/train_with_kv.py \
            --model_name_or_path "${STUDENT_MODEL}" \
            --teacher_models "${TEACHER_MODELS}" \
            --teacher_kv_dir "./kv_caches/multi_teacher/seed_${SEED}" \
            --dataset_name "${DATASET_NAME}" \
            --output_dir "${SEED_OUTPUT_DIR}" \
            --per_device_train_batch_size ${BATCH_SIZE} \
            --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
            --max_steps ${UNIFIED_TRAINING_STEPS} \
            --learning_rate 2e-5 \
            --kv_loss_weight 0.5 \
            --ce_loss_weight 0.5 \
            --fusion_strategy "fixed" \
            --fixed_weights "0.5,0.5" \
            --seed ${SEED} \
            --bf16 \
            --gradient_checkpointing
    
    elif [ "${EXPERIMENT_NAME}" == "multi_teacher_similarity" ]; then
        # ============================================================
        # 实验组 2: 多教师相似度路由
        # ============================================================
        echo "训练配置: 多教师相似度路由"
        
        python experiments/train_with_kv.py \
            --model_name_or_path "${STUDENT_MODEL}" \
            --teacher_models "${TEACHER_MODELS}" \
            --teacher_kv_dir "./kv_caches/multi_teacher/seed_${SEED}" \
            --dataset_name "${DATASET_NAME}" \
            --output_dir "${SEED_OUTPUT_DIR}" \
            --per_device_train_batch_size ${BATCH_SIZE} \
            --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
            --max_steps ${UNIFIED_TRAINING_STEPS} \
            --learning_rate 2e-5 \
            --kv_loss_weight 0.5 \
            --ce_loss_weight 0.5 \
            --fusion_strategy "similarity" \
            --seed ${SEED} \
            --bf16 \
            --gradient_checkpointing
    
    elif [ "${EXPERIMENT_NAME}" == "multi_teacher_learnable" ]; then
        # ============================================================
        # 实验组 3: 多教师可学习路由
        # ============================================================
        echo "训练配置: 多教师可学习路由"
        
        python experiments/train_with_kv.py \
            --model_name_or_path "${STUDENT_MODEL}" \
            --teacher_models "${TEACHER_MODELS}" \
            --teacher_kv_dir "./kv_caches/multi_teacher/seed_${SEED}" \
            --dataset_name "${DATASET_NAME}" \
            --output_dir "${SEED_OUTPUT_DIR}" \
            --per_device_train_batch_size ${BATCH_SIZE} \
            --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
            --max_steps ${UNIFIED_TRAINING_STEPS} \
            --learning_rate 2e-5 \
            --kv_loss_weight 0.5 \
            --ce_loss_weight 0.5 \
            --fusion_strategy "learnable" \
            --learnable_router_type "mlp" \
            --router_entropy_weight 0.01 \
            --seed ${SEED} \
            --bf16 \
            --gradient_checkpointing
    
    else
        echo "错误: 未知的实验名称 ${EXPERIMENT_NAME}"
        exit 1
    fi
    
    # 检查训练是否成功
    if [ $? -eq 0 ]; then
        echo "✓ 随机种子 ${SEED} 训练完成"
        
        # ============================================================
        # 评测
        # ============================================================
        echo "开始评测 - 随机种子 ${SEED}"
        
        python evaluation/multi_task_eval.py \
            --model_path "${SEED_OUTPUT_DIR}" \
            --eval_datasets gsm8k_test math500 bbh gpqa truthfulqa cmmlu ceval \
            --output_dir "${SEED_OUTPUT_DIR}" \
            --batch_size 16 \
            --max_samples 500
        
        if [ $? -eq 0 ]; then
            echo "✓ 随机种子 ${SEED} 评测完成"
        else
            echo "✗ 随机种子 ${SEED} 评测失败"
        fi
    else
        echo "✗ 随机种子 ${SEED} 训练失败"
    fi
    
    echo ""
done

# ============================================================
# 汇总所有随机种子的结果
# ============================================================

echo "========================================================"
echo "所有随机种子实验完成！"
echo "========================================================"

# 统计成功/失败的种子数
SUCCESS_COUNT=0
FAIL_COUNT=0

for SEED in "${SEEDS[@]}"; do
    RESULT_FILE="${BASE_OUTPUT_DIR}/seed_${SEED}/evaluation_results.json"
    if [ -f "${RESULT_FILE}" ]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "  ✗ 随机种子 ${SEED} 缺少结果文件"
    fi
done

echo ""
echo "成功: ${SUCCESS_COUNT} / ${#SEEDS[@]}"
echo "失败: ${FAIL_COUNT} / ${#SEEDS[@]}"
echo ""

# ============================================================
# 生成汇总表格
# ============================================================

if [ ${SUCCESS_COUNT} -ge 2 ]; then
    echo "生成汇总表格..."
    
    python -c "
import json
import numpy as np
from pathlib import Path

# 加载所有种子的结果
base_dir = Path('${BASE_OUTPUT_DIR}')
seeds = [${SEEDS[@]}]
all_results = {}

for seed in seeds:
    result_file = base_dir / f'seed_{seed}' / 'evaluation_results.json'
    if result_file.exists():
        with open(result_file, 'r') as f:
            data = json.load(f)
            all_results[seed] = data.get('results', {})

# 计算统计量
datasets = list(list(all_results.values())[0].keys())
print('\n' + '='*80)
print('多随机种子结果汇总: ${EXPERIMENT_NAME}')
print('='*80)
print(f'{'数据集':<20} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}')
print('-'*80)

summary = {}
for dataset in datasets:
    values = [all_results[seed][dataset] for seed in all_results.keys()]
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1)
    min_val = np.min(values)
    max_val = np.max(values)
    
    summary[dataset] = {
        'mean': mean_val,
        'std': std_val,
        'min': min_val,
        'max': max_val,
        'values': values
    }
    
    print(f'{dataset:<20} {mean_val:<15.2f} {std_val:<15.2f} {min_val:<15.2f} {max_val:<15.2f}')

print('='*80)

# 保存汇总结果
summary_file = base_dir / 'multi_seed_summary.json'
with open(summary_file, 'w') as f:
    json.dump({
        'experiment_name': '${EXPERIMENT_NAME}',
        'seeds': list(all_results.keys()),
        'num_seeds': len(all_results),
        'summary': summary
    }, f, indent=2)

print(f'\n✓ 汇总结果已保存: {summary_file}')
"
    
    echo ""
    echo "========================================================"
    echo "提示: 运行统计显著性测试"
    echo "========================================================"
    echo "python utils/statistical_significance.py \\"
    echo "  --baseline_dir ./outputs/baseline_sft \\"
    echo "  --experimental_dir ${BASE_OUTPUT_DIR} \\"
    echo "  --output_dir ./statistical_analysis/${EXPERIMENT_NAME}"
    echo "========================================================"
else
    echo "警告: 至少需要 2 个成功的种子才能计算统计量"
fi

echo ""
echo "✓ 多随机种子实验流程完成！"
