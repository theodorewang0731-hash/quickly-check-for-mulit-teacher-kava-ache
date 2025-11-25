#!/bin/bash

#SBATCH --job-name=multi_teacher_kv
#SBATCH --output=logs/multi_teacher_%j.out
#SBATCH --error=logs/multi_teacher_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=256GB
#SBATCH --time=96:00:00
#SBATCH --partition=gpu

# Multi-Teacher KV Distillation Training Script (HPC)
# 
# 支持 5 个阶段的多教师蒸馏训练
#
# Usage:
#   sbatch scripts/run_multi_teacher.sh

echo "============================================"
echo "Multi-Teacher KV Distillation Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "============================================"

# ========== Environment Setup ==========

# Detect workspace root
if [ -f "setup.sh" ]; then
    WORKSPACE_ROOT="$(pwd)"
elif [ -f "../setup.sh" ]; then
    WORKSPACE_ROOT="$(cd .. && pwd)"
else
    echo "Error: Cannot find workspace root (setup.sh not found)"
    exit 1
fi

echo "Workspace root: $WORKSPACE_ROOT"
cd "$WORKSPACE_ROOT"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
elif [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "Error: Virtual environment not found"
    exit 1
fi

# ========== Configuration ==========

# Student model (smaller model to distill into)
STUDENT_MODEL="Qwen/Qwen2.5-0.5B-Instruct"

# Teacher models (space-separated, use DIFFERENT models for true multi-teacher learning)
# Examples of diverse teacher combinations:
# - Same family, different sizes: "gpt2 gpt2-medium gpt2-large"
# - Different architectures: "gpt2 facebook/opt-350m EleutherAI/pythia-410m"
# - Different model families: "Qwen/Qwen2.5-1.5B-Instruct meta-llama/Llama-3.2-1B mistralai/Mistral-7B-v0.1"
TEACHER_MODELS="gpt2 gpt2-medium gpt2-large"

# Phase (1-5)
# Phase 1-2: Use 2 teachers (dual-prompt style but different models)
# Phase 3-5: Use 3+ teachers (true multi-teacher)
PHASE=3

# Fusion method (fixed, similarity, learnable)
FUSION_METHOD="learnable"

# Router type (mlp, gate, attention)
ROUTER_TYPE="attention"

# Dataset
DATASET_NAME="wikitext"
DATASET_CONFIG="wikitext-2-raw-v1"
MAX_SAMPLES=10000

# Training hyperparameters
BATCH_SIZE=4
GRAD_ACCUM=8
MAX_LENGTH=512
NUM_EPOCHS=3
LEARNING_RATE=2e-5
WARMUP_STEPS=500

# Loss weights
LAMBDA_K=1.0
LAMBDA_V=1.0
BETA_COS=0.1
GAMMA_KL=0.01
DELTA_CE=1.0

# Entropy regularization (for Phase 4+)
ENTROPY_REG_STRENGTH=0.01
ENTROPY_TARGET="specialized"

# Output directory
OUTPUT_DIR="./outputs/multi_teacher_phase${PHASE}_${FUSION_METHOD}"

# HPC settings
FP16="--bf16"  # Use BF16 for better stability
GRADIENT_CHECKPOINTING="--gradient_checkpointing"

# ========== Print Configuration ==========

echo ""
echo "Configuration:"
echo "  Student Model: $STUDENT_MODEL"
echo "  Teacher Models: $TEACHER_MODELS"
echo "  Phase: $PHASE"
echo "  Fusion Method: $FUSION_METHOD"
echo "  Router Type: $ROUTER_TYPE"
echo "  Dataset: $DATASET_NAME ($DATASET_CONFIG)"
echo "  Max Samples: $MAX_SAMPLES"
echo "  Batch Size: $BATCH_SIZE"
echo "  Gradient Accumulation: $GRAD_ACCUM"
echo "  Max Length: $MAX_LENGTH"
echo "  Num Epochs: $NUM_EPOCHS"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Output Dir: $OUTPUT_DIR"
echo ""

# ========== Create Output Directory ==========

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# ========== Training Command ==========

# Build teacher models argument
TEACHER_ARGS=""
for teacher in $TEACHER_MODELS; do
    TEACHER_ARGS="$TEACHER_ARGS $teacher"
done

echo "Starting training..."
echo ""

python experiments/train_multi_teacher_kv.py \
    --student_model "$STUDENT_MODEL" \
    --teacher_models $TEACHER_ARGS \
    --phase $PHASE \
    --fusion_method "$FUSION_METHOD" \
    --router_type "$ROUTER_TYPE" \
    --dataset_name "$DATASET_NAME" \
    --dataset_config "$DATASET_CONFIG" \
    --max_samples $MAX_SAMPLES \
    --max_length $MAX_LENGTH \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --learning_rate $LEARNING_RATE \
    --warmup_steps $WARMUP_STEPS \
    --save_steps 1000 \
    --logging_steps 100 \
    $FP16 \
    $GRADIENT_CHECKPOINTING \
    --trust_remote_code \
    --lambda_k $LAMBDA_K \
    --lambda_v $LAMBDA_V \
    --beta_cos $BETA_COS \
    --gamma_kl $GAMMA_KL \
    --delta_ce $DELTA_CE \
    --entropy_reg_strength $ENTROPY_REG_STRENGTH \
    --entropy_target "$ENTROPY_TARGET"

EXIT_CODE=$?

# ========== Finish ==========

echo ""
echo "============================================"
echo "Training finished with exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "Output directory: $OUTPUT_DIR"
echo "============================================"

exit $EXIT_CODE
