#!/bin/bash
#SBATCH --job-name=kava_qwen3_exp
#SBATCH --output=logs/experiments_%j.out
#SBATCH --error=logs/experiments_%j.err
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu

# Experiment suite for validating KaVa three claims:
# E1: Baseline (no KV, no CODI)
# E2: Full KV compression
# E3: Right-crop KV compression  
# E4: R-KV compression (best stability)
# E5: Shuffled KV (negative control)

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_DIR"

# Load modules - comment out if not available on your HPC
# module load python/3.11
# module load cuda/12.1
# module load cudnn/8.9

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Warning: venv/bin/activate not found"
fi

export TRANSFORMERS_CACHE="$PROJECT_DIR/cache"
export HF_HOME="$PROJECT_DIR/.huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/cache/datasets"

mkdir -p logs
mkdir -p outputs

BASE_CMD="python experiments/train_with_kv.py \
    --model_name Qwen/Qwen2.5-7B \
    --teacher_name Qwen/Qwen2.5-7B \
    --dataset_name openai/gsm8k \
    --dataset_config main \
    --dataset_split train \
    --streaming \
    --subset_size 100000 \
    --epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr 2e-5 \
    --max_length 1024 \
    --target_len 16 \
    --save_steps 1000 \
    --logging_steps 50 \
    --num_workers 16 \
    --trust_remote_code \
    --torch_dtype bfloat16 \
    --fp16 \
    --gradient_checkpointing \
    --device_map auto"

echo "=========================================="
echo "E1: Baseline (no KV distillation)"
echo "=========================================="
$BASE_CMD \
    --output_dir outputs/E1_baseline \
    --kv_weight 0.0 \
    --codi_weight 0.5

echo "=========================================="
echo "E2: Full KV compression"
echo "=========================================="
$BASE_CMD \
    --output_dir outputs/E2_full_kv \
    --kv_method full \
    --kv_weight 1.0 \
    --codi_weight 0.5

echo "=========================================="
echo "E3: Right-crop KV compression"
echo "=========================================="
$BASE_CMD \
    --output_dir outputs/E3_right_crop \
    --kv_method right_crop \
    --kv_weight 1.0 \
    --codi_weight 0.5

echo "=========================================="
echo "E4: R-KV compression (best stability)"
echo "=========================================="
$BASE_CMD \
    --output_dir outputs/E4_rkv \
    --kv_method rkv \
    --kv_weight 1.0 \
    --codi_weight 0.5

echo "=========================================="
echo "E5: Shuffled KV (negative control)"
echo "=========================================="
$BASE_CMD \
    --output_dir outputs/E5_shuffled \
    --kv_method rkv \
    --kv_weight 1.0 \
    --codi_weight 0.5 \
    --shuffle_kv

echo "=========================================="
echo "All experiments completed at $(date)"
echo "=========================================="
