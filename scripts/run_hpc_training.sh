#!/bin/bash
#SBATCH --job-name=kava_qwen3
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu

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

# Set environment variables
export TRANSFORMERS_CACHE="$PROJECT_DIR/cache"
export HF_HOME="$PROJECT_DIR/.huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/cache/datasets"
export CUDA_VISIBLE_DEVICES=0

# Create output directories
mkdir -p outputs/kava_experiment
mkdir -p logs

# Training command - Qwen3 with large-scale dataset
python experiments/train_with_kv.py \
    --model_name "Qwen/Qwen2.5-7B" \
    --teacher_name "Qwen/Qwen2.5-7B" \
    --dataset_name "openai/gsm8k" \
    --dataset_config "main" \
    --dataset_split "train" \
    --streaming \
    --subset_size 50000 \
    --output_dir "outputs/kava_qwen3_large" \
    --epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr 2e-5 \
    --max_length 1024 \
    --target_len 16 \
    --kv_method "rkv" \
    --kv_loss "smooth_l1" \
    --kv_weight 1.0 \
    --codi_weight 0.5 \
    --save_steps 1000 \
    --logging_steps 50 \
    --num_workers 8 \
    --trust_remote_code \
    --torch_dtype "bfloat16" \
    --fp16 \
    --gradient_checkpointing \
    --device_map "auto"

echo "Training completed at $(date)"
