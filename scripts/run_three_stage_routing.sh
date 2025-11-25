#!/bin/bash
#SBATCH --job-name=three_stage_routing
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8                # 根据 HPC 要求可能需要完整格式，如 gpu:a100-sxm4-80gb:8
#SBATCH --mem=500G                  # 根据 HPC 限制可能需要调整
#SBATCH --time=96:00:00
#SBATCH --output=logs/three_stage_routing_%j.log
#SBATCH --error=logs/three_stage_routing_%j.err

# ============================================================================
# 三阶段路由训练脚本
# Stage 1: 固定权重（验证基础融合）
# Stage 2: 相似度路由（自动权重分配）
# Stage 3: 可学习路由（端到端优化）
# ============================================================================

# 使用统一的环境配置脚本（自动配置共享模型库）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_hpc_environment.sh"

# ------------------------ 基础配置 ------------------------
STUDENT_MODEL="Qwen/Qwen2.5-1.5B"
TEACHER_MODELS="Qwen/Qwen2.5-7B Qwen/Qwen2.5-14B"  # 单家族起步
DATASET_CONFIG="multi_reasoning_cot_direct"
OUTPUT_BASE="./outputs/three_stage_routing"

# 数据配置
TRAIN_SAMPLES=15000
VAL_SAMPLES=2000
USE_EXTENDED=true

# 训练超参数
NUM_EPOCHS=3
BATCH_SIZE=2
GRAD_ACCUM=16
LEARNING_RATE=2e-5
WARMUP_RATIO=0.1

# 内存优化
USE_BF16=true
GRADIENT_CHECKPOINTING=true
MAX_GRAD_NORM=1.0

# KV 配置
KV_COMPRESSION="right"
KV_LOSS_TYPE="smooth_l1"
KV_LOSS_WEIGHT=0.1

# 对齐策略
LAYER_MAP_STRATEGY="ratio"
HEAD_ADAPT_STRATEGY="linear"
ROPE_SCALING=true

# 评测配置
EVAL_STRATEGY="steps"
EVAL_STEPS=500
SAVE_STEPS=1000
LOGGING_STEPS=50

# ============================================================================
# Stage 1: 固定权重训练
# ============================================================================
echo "========================================================================"
echo "Stage 1: Fixed Routing (Equal Weights)"
echo "========================================================================"

STAGE=1
OUTPUT_DIR="$OUTPUT_BASE/stage1_fixed"
FUSION_STRATEGY="fixed"
FIXED_WEIGHTS="0.5,0.5"

python experiments/train_multi_teacher_kv.py \
    --student_model_name_or_path "$STUDENT_MODEL" \
    --teacher_models $TEACHER_MODELS \
    --dataset_name "$DATASET_CONFIG" \
    --train_samples $TRAIN_SAMPLES \
    --val_samples $VAL_SAMPLES \
    --use_extended_datasets $USE_EXTENDED \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --warmup_ratio $WARMUP_RATIO \
    --bf16 $USE_BF16 \
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
    --train_phase $STAGE \
    --num_teachers 2 \
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

echo "✓ Stage 1 completed: $OUTPUT_DIR"

# 评测 Stage 1
echo "Evaluating Stage 1..."
python evaluation/multi_task_eval.py \
    --model_path "$OUTPUT_DIR/best_model" \
    --eval_datasets gsm8k_test math500 bbh gpqa truthfulqa cmmlu_subset ceval_subset \
    --output_file "$OUTPUT_DIR/eval_results.json"

# ============================================================================
# Stage 2: 相似度路由训练
# ============================================================================
echo "========================================================================"
echo "Stage 2: Similarity-Based Routing"
echo "========================================================================"

STAGE=2
OUTPUT_DIR="$OUTPUT_BASE/stage2_similarity"
FUSION_STRATEGY="similarity"
SIMILARITY_METRIC="cosine"  # cosine, dot, euclidean
TEMPERATURE=1.0

# 继续从 Stage 1 训练
RESUME_FROM="$OUTPUT_BASE/stage1_fixed/best_model"

python experiments/train_multi_teacher_kv.py \
    --student_model_name_or_path "$RESUME_FROM" \
    --teacher_models $TEACHER_MODELS \
    --dataset_name "$DATASET_CONFIG" \
    --train_samples $TRAIN_SAMPLES \
    --val_samples $VAL_SAMPLES \
    --use_extended_datasets $USE_EXTENDED \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --warmup_ratio $WARMUP_RATIO \
    --bf16 $USE_BF16 \
    --gradient_checkpointing $GRADIENT_CHECKPOINTING \
    --max_grad_norm $MAX_GRAD_NORM \
    --kv_compression "$KV_COMPRESSION" \
    --kv_loss_type "$KV_LOSS_TYPE" \
    --kv_loss_weight $KV_LOSS_WEIGHT \
    --fusion_strategy "$FUSION_STRATEGY" \
    --similarity_metric "$SIMILARITY_METRIC" \
    --temperature $TEMPERATURE \
    --layer_map_strategy "$LAYER_MAP_STRATEGY" \
    --head_adapt_strategy "$HEAD_ADAPT_STRATEGY" \
    --rope_scaling $ROPE_SCALING \
    --train_phase $STAGE \
    --num_teachers 2 \
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

echo "✓ Stage 2 completed: $OUTPUT_DIR"

# 评测 Stage 2
echo "Evaluating Stage 2..."
python evaluation/multi_task_eval.py \
    --model_path "$OUTPUT_DIR/best_model" \
    --eval_datasets gsm8k_test math500 bbh gpqa truthfulqa cmmlu_subset ceval_subset \
    --output_file "$OUTPUT_DIR/eval_results.json"

# ============================================================================
# Stage 3: 可学习路由训练
# ============================================================================
echo "========================================================================"
echo "Stage 3: Learnable Routing"
echo "========================================================================"

STAGE=3
OUTPUT_DIR="$OUTPUT_BASE/stage3_learnable"
FUSION_STRATEGY="learnable"
ROUTER_TYPE="mlp"  # mlp, gate, attention
ROUTER_HIDDEN_DIM=256
ENTROPY_REG_WEIGHT=0.01

# 继续从 Stage 2 训练
RESUME_FROM="$OUTPUT_BASE/stage2_similarity/best_model"

python experiments/train_multi_teacher_kv.py \
    --student_model_name_or_path "$RESUME_FROM" \
    --teacher_models $TEACHER_MODELS \
    --dataset_name "$DATASET_CONFIG" \
    --train_samples $TRAIN_SAMPLES \
    --val_samples $VAL_SAMPLES \
    --use_extended_datasets $USE_EXTENDED \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --warmup_ratio $WARMUP_RATIO \
    --bf16 $USE_BF16 \
    --gradient_checkpointing $GRADIENT_CHECKPOINTING \
    --max_grad_norm $MAX_GRAD_NORM \
    --kv_compression "$KV_COMPRESSION" \
    --kv_loss_type "$KV_LOSS_TYPE" \
    --kv_loss_weight $KV_LOSS_WEIGHT \
    --fusion_strategy "$FUSION_STRATEGY" \
    --router_type "$ROUTER_TYPE" \
    --router_hidden_dim $ROUTER_HIDDEN_DIM \
    --entropy_reg_weight $ENTROPY_REG_WEIGHT \
    --layer_map_strategy "$LAYER_MAP_STRATEGY" \
    --head_adapt_strategy "$HEAD_ADAPT_STRATEGY" \
    --rope_scaling $ROPE_SCALING \
    --train_phase $STAGE \
    --num_teachers 2 \
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

echo "✓ Stage 3 completed: $OUTPUT_DIR"

# 评测 Stage 3
echo "Evaluating Stage 3..."
python evaluation/multi_task_eval.py \
    --model_path "$OUTPUT_DIR/best_model" \
    --eval_datasets gsm8k_test math500 bbh gpqa truthfulqa cmmlu_subset ceval_subset \
    --output_file "$OUTPUT_DIR/eval_results.json"

# ============================================================================
# 总结对比
# ============================================================================
echo "========================================================================"
echo "Three-Stage Training Complete!"
echo "========================================================================"
echo "Stage 1 (Fixed):      $OUTPUT_BASE/stage1_fixed/eval_results.json"
echo "Stage 2 (Similarity): $OUTPUT_BASE/stage2_similarity/eval_results.json"
echo "Stage 3 (Learnable):  $OUTPUT_BASE/stage3_learnable/eval_results.json"
echo "========================================================================"

# 生成对比报告
python -c "
import json
from pathlib import Path

stages = ['stage1_fixed', 'stage2_similarity', 'stage3_learnable']
base_dir = Path('$OUTPUT_BASE')

print('\n' + '='*80)
print('Performance Comparison Across Stages')
print('='*80)
print(f'{'Dataset':<20} {'Stage 1':<15} {'Stage 2':<15} {'Stage 3':<15}')
print('-'*80)

all_results = {}
for stage in stages:
    result_file = base_dir / stage / 'eval_results.json'
    if result_file.exists():
        with open(result_file) as f:
            all_results[stage] = json.load(f)

# 获取所有数据集
datasets = set()
for results in all_results.values():
    datasets.update(k for k in results.keys() if k != 'average')

for dataset in sorted(datasets):
    scores = []
    for stage in stages:
        if dataset in all_results.get(stage, {}):
            score = all_results[stage][dataset]['score']
            scores.append(f'{score:.2f}%')
        else:
            scores.append('N/A')
    print(f'{dataset:<20} {scores[0]:<15} {scores[1]:<15} {scores[2]:<15}')

print('-'*80)
# 平均分
avg_scores = []
for stage in stages:
    if 'average' in all_results.get(stage, {}):
        avg_scores.append(f\"{all_results[stage]['average']:.2f}%\")
    else:
        avg_scores.append('N/A')
print(f'{'Average':<20} {avg_scores[0]:<15} {avg_scores[1]:<15} {avg_scores[2]:<15}')
print('='*80)
"

echo "Training pipeline completed successfully!"
