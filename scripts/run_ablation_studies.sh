#!/bin/bash

################################################################################
# 消融实验自动化脚本
#
# 包含以下消融研究:
# 1. 路由消融: 固定权重 vs 可学习路由
# 2. 层级消融: 只蒸浅层 vs 全层
# 3. K/V 消融: 只蒸 K vs 只蒸 V vs K+V
# 4. 对齐策略消融: 硬截断 vs 软对齐
#
# 用法:
#   sbatch scripts/run_ablation_studies.sh
################################################################################

#SBATCH --job-name=ablation_studies
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8                # 根据 HPC 要求可能需要完整格式，如 gpu:a100-sxm4-80gb:8
#SBATCH --cpus-per-task=64          # 根据 HPC 限制可能需要调整（建议 4-8 核/GPU）
#SBATCH --mem=0                     # 使用节点全部内存，或指定如 512G
#SBATCH --time=96:00:00
#SBATCH --output=logs/ablation_%j.out
#SBATCH --error=logs/ablation_%j.err

# ============================================================
# 环境设置
# ============================================================

# 使用统一的环境配置脚本（自动配置共享模型库）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_hpc_environment.sh"

mkdir -p logs
mkdir -p ./outputs/ablation_studies

# 通用配置
STUDENT_MODEL="Qwen/Qwen2.5-1.5B"
TEACHER_MODELS="Qwen/Qwen2.5-7B,Qwen/Qwen2.5-14B"
DATASET_NAME="multi_reasoning_cot_direct"
SEEDS=(42 43 44)  # 3 个随机种子

# 训练预算控制（所有消融实验使用相同配置）
BATCH_SIZE=32
SEQ_LENGTH=512
GRADIENT_ACCUMULATION_STEPS=4
NUM_GPUS=8
MAX_STEPS=5000  # 消融实验可以用较少步数

echo "========================================================"
echo "消融实验配置"
echo "========================================================"
echo "学生模型: ${STUDENT_MODEL}"
echo "教师模型: ${TEACHER_MODELS}"
echo "随机种子: ${SEEDS[@]}"
echo "训练步数: ${MAX_STEPS}"
echo "========================================================"

# ============================================================
# 消融实验 1: 路由消融
# 比较固定权重 vs 可学习路由
# ============================================================

run_routing_ablation() {
    echo ""
    echo "========================================================"
    echo "消融实验 1: 路由策略"
    echo "========================================================"
    
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "--- 固定权重 (0.5/0.5) - Seed ${SEED} ---"
        
        python experiments/train_with_kv.py \
            --model_name_or_path "${STUDENT_MODEL}" \
            --teacher_models "${TEACHER_MODELS}" \
            --teacher_kv_dir "./kv_caches/multi_teacher/seed_${SEED}" \
            --dataset_name "${DATASET_NAME}" \
            --output_dir "./outputs/ablation_studies/routing_fixed/seed_${SEED}" \
            --per_device_train_batch_size ${BATCH_SIZE} \
            --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
            --max_steps ${MAX_STEPS} \
            --fusion_strategy "fixed" \
            --fixed_weights "0.5,0.5" \
            --seed ${SEED} \
            --bf16 \
            --gradient_checkpointing
        
        echo ""
        echo "--- 可学习路由 (MLP) - Seed ${SEED} ---"
        
        python experiments/train_with_kv.py \
            --model_name_or_path "${STUDENT_MODEL}" \
            --teacher_models "${TEACHER_MODELS}" \
            --teacher_kv_dir "./kv_caches/multi_teacher/seed_${SEED}" \
            --dataset_name "${DATASET_NAME}" \
            --output_dir "./outputs/ablation_studies/routing_learnable/seed_${SEED}" \
            --per_device_train_batch_size ${BATCH_SIZE} \
            --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
            --max_steps ${MAX_STEPS} \
            --fusion_strategy "learnable" \
            --learnable_router_type "mlp" \
            --router_entropy_weight 0.01 \
            --seed ${SEED} \
            --bf16 \
            --gradient_checkpointing
    done
    
    echo "✓ 路由消融完成"
}

# ============================================================
# 消融实验 2: 层级消融
# 比较只蒸浅层 vs 全层
# ============================================================

run_layer_ablation() {
    echo ""
    echo "========================================================"
    echo "消融实验 2: 层级贡献"
    echo "========================================================"
    
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "--- 只蒸浅层 (0-12) - Seed ${SEED} ---"
        
        python experiments/train_with_kv.py \
            --model_name_or_path "${STUDENT_MODEL}" \
            --teacher_models "${TEACHER_MODELS}" \
            --teacher_kv_dir "./kv_caches/multi_teacher/seed_${SEED}" \
            --dataset_name "${DATASET_NAME}" \
            --output_dir "./outputs/ablation_studies/layers_shallow/seed_${SEED}" \
            --per_device_train_batch_size ${BATCH_SIZE} \
            --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
            --max_steps ${MAX_STEPS} \
            --distill_layers "0-12" \
            --seed ${SEED} \
            --bf16 \
            --gradient_checkpointing
        
        echo ""
        echo "--- 只蒸中层 (12-24) - Seed ${SEED} ---"
        
        python experiments/train_with_kv.py \
            --model_name_or_path "${STUDENT_MODEL}" \
            --teacher_models "${TEACHER_MODELS}" \
            --teacher_kv_dir "./kv_caches/multi_teacher/seed_${SEED}" \
            --dataset_name "${DATASET_NAME}" \
            --output_dir "./outputs/ablation_studies/layers_middle/seed_${SEED}" \
            --per_device_train_batch_size ${BATCH_SIZE} \
            --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
            --max_steps ${MAX_STEPS} \
            --distill_layers "12-24" \
            --seed ${SEED} \
            --bf16 \
            --gradient_checkpointing
        
        echo ""
        echo "--- 蒸全层 (0-28) - Seed ${SEED} ---"
        
        python experiments/train_with_kv.py \
            --model_name_or_path "${STUDENT_MODEL}" \
            --teacher_models "${TEACHER_MODELS}" \
            --teacher_kv_dir "./kv_caches/multi_teacher/seed_${SEED}" \
            --dataset_name "${DATASET_NAME}" \
            --output_dir "./outputs/ablation_studies/layers_all/seed_${SEED}" \
            --per_device_train_batch_size ${BATCH_SIZE} \
            --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
            --max_steps ${MAX_STEPS} \
            --distill_layers "0-28" \
            --seed ${SEED} \
            --bf16 \
            --gradient_checkpointing
    done
    
    echo "✓ 层级消融完成"
}

# ============================================================
# 消融实验 3: K/V 消融
# 比较只蒸 K vs 只蒸 V vs K+V
# ============================================================

run_kv_ablation() {
    echo ""
    echo "========================================================"
    echo "消融实验 3: K vs V 蒸馏"
    echo "========================================================"
    
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "--- 只蒸 K - Seed ${SEED} ---"
        
        python experiments/train_with_kv.py \
            --model_name_or_path "${STUDENT_MODEL}" \
            --teacher_models "${TEACHER_MODELS}" \
            --teacher_kv_dir "./kv_caches/multi_teacher/seed_${SEED}" \
            --dataset_name "${DATASET_NAME}" \
            --output_dir "./outputs/ablation_studies/kv_only_k/seed_${SEED}" \
            --per_device_train_batch_size ${BATCH_SIZE} \
            --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
            --max_steps ${MAX_STEPS} \
            --distill_k_only \
            --seed ${SEED} \
            --bf16 \
            --gradient_checkpointing
        
        echo ""
        echo "--- 只蒸 V - Seed ${SEED} ---"
        
        python experiments/train_with_kv.py \
            --model_name_or_path "${STUDENT_MODEL}" \
            --teacher_models "${TEACHER_MODELS}" \
            --teacher_kv_dir "./kv_caches/multi_teacher/seed_${SEED}" \
            --dataset_name "${DATASET_NAME}" \
            --output_dir "./outputs/ablation_studies/kv_only_v/seed_${SEED}" \
            --per_device_train_batch_size ${BATCH_SIZE} \
            --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
            --max_steps ${MAX_STEPS} \
            --distill_v_only \
            --seed ${SEED} \
            --bf16 \
            --gradient_checkpointing
        
        echo ""
        echo "--- 同时蒸 K+V - Seed ${SEED} ---"
        
        python experiments/train_with_kv.py \
            --model_name_or_path "${STUDENT_MODEL}" \
            --teacher_models "${TEACHER_MODELS}" \
            --teacher_kv_dir "./kv_caches/multi_teacher/seed_${SEED}" \
            --dataset_name "${DATASET_NAME}" \
            --output_dir "./outputs/ablation_studies/kv_both/seed_${SEED}" \
            --per_device_train_batch_size ${BATCH_SIZE} \
            --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
            --max_steps ${MAX_STEPS} \
            --seed ${SEED} \
            --bf16 \
            --gradient_checkpointing
    done
    
    echo "✓ K/V 消融完成"
}

# ============================================================
# 消融实验 4: 对齐策略消融
# 比较硬截断 vs 软对齐矩阵
# ============================================================

run_alignment_ablation() {
    echo ""
    echo "========================================================"
    echo "消融实验 4: 对齐策略"
    echo "========================================================"
    
    for SEED in "${SEEDS[@]}"; do
        echo ""
        echo "--- 硬截断 - Seed ${SEED} ---"
        
        python experiments/train_with_kv.py \
            --model_name_or_path "${STUDENT_MODEL}" \
            --teacher_models "${TEACHER_MODELS}" \
            --teacher_kv_dir "./kv_caches/multi_teacher/seed_${SEED}" \
            --dataset_name "${DATASET_NAME}" \
            --output_dir "./outputs/ablation_studies/align_hard_truncate/seed_${SEED}" \
            --per_device_train_batch_size ${BATCH_SIZE} \
            --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
            --max_steps ${MAX_STEPS} \
            --alignment_strategy "truncate" \
            --seed ${SEED} \
            --bf16 \
            --gradient_checkpointing
        
        echo ""
        echo "--- 软对齐矩阵 - Seed ${SEED} ---"
        
        python experiments/train_with_kv.py \
            --model_name_or_path "${STUDENT_MODEL}" \
            --teacher_models "${TEACHER_MODELS}" \
            --teacher_kv_dir "./kv_caches/multi_teacher/seed_${SEED}" \
            --dataset_name "${DATASET_NAME}" \
            --output_dir "./outputs/ablation_studies/align_soft_matrix/seed_${SEED}" \
            --per_device_train_batch_size ${BATCH_SIZE} \
            --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
            --max_steps ${MAX_STEPS} \
            --alignment_strategy "soft" \
            --seed ${SEED} \
            --bf16 \
            --gradient_checkpointing
    done
    
    echo "✓ 对齐策略消融完成"
}

# ============================================================
# 主流程
# ============================================================

echo ""
echo "========================================================"
echo "开始运行所有消融实验"
echo "========================================================"

# 运行所有消融实验
run_routing_ablation
run_layer_ablation
run_kv_ablation
run_alignment_ablation

echo ""
echo "========================================================"
echo "所有消融实验完成！"
echo "========================================================"

# ============================================================
# 评测所有消融实验
# ============================================================

echo ""
echo "开始评测所有消融实验..."

ABLATION_DIRS=(
    "routing_fixed"
    "routing_learnable"
    "layers_shallow"
    "layers_middle"
    "layers_all"
    "kv_only_k"
    "kv_only_v"
    "kv_both"
    "align_hard_truncate"
    "align_soft_matrix"
)

for ABLATION_DIR in "${ABLATION_DIRS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        MODEL_PATH="./outputs/ablation_studies/${ABLATION_DIR}/seed_${SEED}"
        
        if [ -d "${MODEL_PATH}" ]; then
            echo ""
            echo "评测: ${ABLATION_DIR}/seed_${SEED}"
            
            python evaluation/multi_task_eval.py \
                --model_path "${MODEL_PATH}" \
                --eval_datasets gsm8k_test math500 bbh \
                --output_dir "${MODEL_PATH}" \
                --batch_size 16 \
                --max_samples 500
        fi
    done
done

echo ""
echo "✓ 所有评测完成！"

# ============================================================
# 生成消融实验对比报告
# ============================================================

echo ""
echo "生成消融实验对比报告..."

python visualization/ablation_analysis.py \
    --ablation_base_dir "./outputs/ablation_studies" \
    --output_dir "./outputs/ablation_analysis" \
    --seeds "${SEEDS[@]}"

echo ""
echo "========================================================"
echo "消融实验流程完成！"
echo "========================================================"
echo "结果保存在: ./outputs/ablation_studies"
echo "分析报告: ./outputs/ablation_analysis"
echo "========================================================"
