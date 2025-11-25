#!/bin/bash
#SBATCH --job-name=kava_qwen3_ultra
#SBATCH --output=logs/train_ultra_%j.out
#SBATCH --error=logs/train_ultra_%j.err
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu

# Ultra-large scale training with Qwen3 (7B) on massive dataset
# This script is optimized for:
# - Very large models (7B+ parameters)
# - Massive datasets (100K+ samples)
# - Multi-GPU training
# - Extended training time

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_DIR"

# Load modules - uncomment and adjust for your HPC
# module load python/3.11
# module load cuda/12.1
# module load cudnn/8.9
# module load nccl/2.18  # For multi-GPU

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Warning: venv/bin/activate not found"
fi

# Set environment variables
export TRANSFORMERS_CACHE="$PROJECT_DIR/cache"
export HF_HOME="$PROJECT_DIR/.huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/cache/datasets"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO  # For debugging multi-GPU issues

# Create output directories
mkdir -p outputs/qwen3_ultra
mkdir -p logs

echo "=========================================="
echo "KaVa Ultra-Large Scale Training"
echo "=========================================="
echo "Model: Qwen/Qwen2.5-7B"
echo "Dataset: Full scale (streaming mode)"
echo "GPUs: 8"
echo "Memory: 512GB"
echo "Time limit: 168 hours (7 days)"
echo "=========================================="
echo ""

# Training command for ultra-large scale
python experiments/train_with_kv.py \
    --model_name "Qwen/Qwen2.5-7B" \
    --teacher_name "Qwen/Qwen2.5-7B" \
    --dataset_name "openai/gsm8k" \
    --dataset_config "main" \
    --dataset_split "train" \
    --streaming \
    --output_dir "outputs/qwen3_ultra" \
    --epochs 5 \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --lr 1e-5 \
    --max_length 2048 \
    --target_len 32 \
    --kv_method "rkv" \
    --kv_loss "smooth_l1" \
    --kv_weight 1.0 \
    --codi_weight 0.5 \
    --save_steps 2000 \
    --logging_steps 100 \
    --num_workers 16 \
    --trust_remote_code \
    --torch_dtype "bfloat16" \
    --fp16 \
    --gradient_checkpointing \
    --device_map "auto"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully at $(date)"
else
    echo "✗ Training failed with exit code $EXIT_CODE at $(date)"
fi
echo "=========================================="

exit $EXIT_CODE
