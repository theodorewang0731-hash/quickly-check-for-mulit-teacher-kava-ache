#!/bin/bash
#SBATCH --job-name=kv_distill_large
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8              # 根据 HPC 要求可能需要完整格式，如 gpu:a100-sxm4-80gb:8
#SBATCH --mem=500G                # 根据 HPC 限制可能需要调整
#SBATCH --time=72:00:00
#SBATCH --output=logs/multi_teacher_large_%j.log
#SBATCH --error=logs/multi_teacher_large_%j.err

# ============================================================================
# 大规模多教师知识蒸馏配置
# Teacher: 7B-34B 级别 | Student: 1.5B-3B 级别
# ============================================================================

# ------------------------ 环境配置 ------------------------
# 使用统一的环境配置脚本（自动配置共享模型库）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_hpc_environment.sh"

# ------------------------ 模型配置 ------------------------
# 学生模型选择（1.5B-3B 级别）
STUDENT_MODEL="Qwen/Qwen2.5-1.5B"           # 推荐：Qwen2.5-1.5B 或 Qwen2.5-3B
# STUDENT_MODEL="Qwen/Qwen2.5-3B"
# STUDENT_MODEL="meta-llama/Llama-3.2-3B"

# ------------------------ 教师模型配置 ------------------------
# 策略 1: 单家族多 checkpoint（极易对齐，推荐起步）
# Pure Llama
TEACHER_MODELS="meta-llama/Llama-3.1-8B meta-llama/Llama-3.1-70B"
# Pure Qwen
# TEACHER_MODELS="Qwen/Qwen2.5-7B Qwen/Qwen2.5-14B Qwen/Qwen2.5-32B"

# 策略 2: 跨家族少量（最稳，次优选择）
# TEACHER_MODELS="Qwen/Qwen2.5-7B meta-llama/Llama-3.1-8B"
# TEACHER_MODELS="Qwen/Qwen2.5-14B meta-llama/Llama-3.1-70B"

# 策略 3: 混合大小（测试鲁棒性）
# TEACHER_MODELS="Qwen/Qwen2.5-7B Qwen/Qwen2.5-32B meta-llama/Llama-3.1-70B"

# ------------------------ 数据集配置 ------------------------
# 基础数据集：GSM8K + SVAMP + StrategyQA + Math23K（20%中文）
# 每题双风格：CoT（链式推理）+ 直答（直接答案）
DATASET_CONFIG="multi_reasoning_cot_direct"

# 训练集比例
TRAIN_SAMPLES=15000      # GSM8K(7473) + SVAMP(1000) + StrategyQA(2290) + Math23K(4237*0.2≈847) ≈ 11610 → 扩充到 15k
VAL_SAMPLES=2000         # 验证集

# 扩展数据集（可选，通过参数启用）
USE_EXTENDED=true        # 是否使用扩展数据集
# 扩展包括：MATH subset, ARC-Challenge, HotpotQA

# ------------------------ 训练配置 ------------------------
OUTPUT_DIR="./outputs/large_scale_multi_teacher"
NUM_EPOCHS=3
BATCH_SIZE=2             # 每卡 batch size（大教师模型需小 batch）
GRAD_ACCUM=16            # 梯度累积：有效 batch = 2 * 8 * 16 = 256
LEARNING_RATE=2e-5
WARMUP_RATIO=0.1

# 混合精度
USE_BF16=true            # H100/A100-80G 推荐
USE_FP16=false

# 内存优化
GRADIENT_CHECKPOINTING=true
MAX_GRAD_NORM=1.0

# ------------------------ 多教师配置 ------------------------
# Phase 1: 双教师基础训练（固定权重）
FUSION_STRATEGY="fixed"
FIXED_WEIGHTS="0.5,0.5"  # 等权重起步
TRAIN_PHASE=1
NUM_TEACHERS=2

# KV 压缩策略
KV_COMPRESSION="right"   # right-crop（保留关键推理步骤）
KV_LOSS_TYPE="smooth_l1"
KV_LOSS_WEIGHT=0.1

# 对齐策略
LAYER_MAP_STRATEGY="ratio"     # 比例映射（自动处理不同层数）
HEAD_ADAPT_STRATEGY="linear"   # 线性投影（处理维度差异）
ROPE_SCALING=true              # NTK-aware RoPE 缩放

# ------------------------ 评测配置 ------------------------
EVAL_STRATEGY="steps"
EVAL_STEPS=500
SAVE_STEPS=1000
LOGGING_STEPS=50

# 评测数据集（在训练后运行）
# GSM8K test, MATH500, BBH, GPQA, TruthfulQA, CMMLU, C-Eval
EVAL_DATASETS="gsm8k_test,math500,bbh,gpqa,truthfulqa,cmmlu_subset,ceval_subset"

# ------------------------ 运行训练 ------------------------
echo "=========================================="
echo "Starting Large-Scale Multi-Teacher KV Distillation"
echo "=========================================="
echo "Student Model: $STUDENT_MODEL"
echo "Teacher Models: $TEACHER_MODELS"
echo "Dataset: $DATASET_CONFIG"
echo "Training Phase: $TRAIN_PHASE"
echo "Fusion Strategy: $FUSION_STRATEGY"
echo "=========================================="

python experiments/train_multi_teacher_kv.py \
    --student_model_name_or_path "$STUDENT_MODEL" \
    --teacher_models $TEACHER_MODELS \
    --dataset_name "$DATASET_CONFIG" \
    --train_samples $TRAIN_SAMPLES \
    --val_samples $VAL_SAMPLES \
    --use_extended_datasets $USE_EXTENDED \
    --output_dir "$OUTPUT_DIR/phase${TRAIN_PHASE}" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --warmup_ratio $WARMUP_RATIO \
    --bf16 $USE_BF16 \
    --fp16 $USE_FP16 \
    --gradient_checkpointing $GRADIENT_CHECKPOINTING \
    --max_grad_norm $MAX_GRAD_NORM \
    --kv_compression "$KV_COMPRESSION" \
    --kv_loss_type "$KV_LOSS_TYPE" \
    --kv_loss_weight $KV_LOSS_WEIGHT \
    --fusion_strategy "$FUSION_STRATEGY" \
    --fixed_weights "$FIXED_WEIGHTS" \
    --layer_map_strategy "$LAYER_MAP_STRATEGY" \
    --head_adapt_strategy "$HEAD_ADAPT_STRATEGY" \
    --rope_scaling $ROPE_SCALING \
    --train_phase $TRAIN_PHASE \
    --num_teachers $NUM_TEACHERS \
    --evaluation_strategy "$EVAL_STRATEGY" \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --logging_steps $LOGGING_STEPS \
    --save_total_limit 3 \
    --load_best_model_at_end true \
    --metric_for_best_model "eval_kv_loss" \
    --greater_is_better false \
    --dataloader_num_workers 8 \
    --remove_unused_columns false \
    --report_to tensorboard

echo "=========================================="
echo "Phase $TRAIN_PHASE Training Completed!"
echo "=========================================="

# ------------------------ 多阶段训练脚本 ------------------------
# 运行完整的三阶段路由训练：
# 1. 固定权重（验证基础融合）
# 2. 相似度路由（自动权重分配）
# 3. 可学习路由（端到端优化）

# 使用方法：
# Phase 1: bash run_large_scale_multi_teacher.sh
# Phase 2: 修改 FUSION_STRATEGY="similarity" 并设置 TRAIN_PHASE=2
# Phase 3: 修改 FUSION_STRATEGY="learnable" 并设置 TRAIN_PHASE=3
#          同时设置 ROUTER_TYPE="mlp" 或 "gate" 或 "attention"

# ------------------------ 评测脚本（训练后执行）------------------------
# python evaluation/run_multi_task_eval.py \
#     --model_path "$OUTPUT_DIR/phase${TRAIN_PHASE}/best_model" \
#     --eval_datasets $EVAL_DATASETS \
#     --output_file "$OUTPUT_DIR/phase${TRAIN_PHASE}/eval_results.json"
