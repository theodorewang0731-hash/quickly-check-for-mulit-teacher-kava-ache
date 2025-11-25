#!/bin/bash
#SBATCH --job-name=kava_multinode
#SBATCH --output=logs/multinode_%j.out
#SBATCH --error=logs/multinode_%j.err
#SBATCH --time=336:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=512G
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu

# Multi-node training for extreme scale
# This requires distributed training setup

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_DIR"

# Load modules
# module load python/3.11
# module load cuda/12.1
# module load nccl/2.18
# module load openmpi/4.1

source venv/bin/activate

export TRANSFORMERS_CACHE="$PROJECT_DIR/cache"
export HF_HOME="$PROJECT_DIR/.huggingface"
export HF_DATASETS_CACHE="$PROJECT_DIR/cache/datasets"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

# Multi-node setup
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

mkdir -p outputs/qwen3_multinode
mkdir -p logs

echo "=========================================="
echo "Multi-Node Distributed Training"
echo "=========================================="
echo "Nodes: $SLURM_NNODES"
echo "Total GPUs: $(($SLURM_NNODES * 8))"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "=========================================="

# Use torchrun for distributed training
torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    experiments/train_with_kv.py \
    --model_name "Qwen/Qwen2.5-7B" \
    --teacher_name "Qwen/Qwen2.5-7B" \
    --dataset_name "openai/gsm8k" \
    --streaming \
    --output_dir "outputs/qwen3_multinode" \
    --epochs 10 \
    --batch_size 1 \
    --gradient_accumulation_steps 32 \
    --lr 5e-6 \
    --max_length 2048 \
    --target_len 64 \
    --kv_method "rkv" \
    --save_steps 5000 \
    --logging_steps 200 \
    --num_workers 16 \
    --trust_remote_code \
    --torch_dtype "bfloat16" \
    --fp16 \
    --gradient_checkpointing \
    --device_map "auto"

echo "Training completed at $(date)"
