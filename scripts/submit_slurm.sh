#!/bin/bash
#SBATCH --job-name=kava_training
#SBATCH --output=logs/kava_%j.out
#SBATCH --error=logs/kava_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu

# ====================================================================
# KAVA Training - SLURM Job Script
# 适用于大多数 HPC 集群（自动适配）
# ====================================================================

echo "========================================================================"
echo "KAVA Training Job Started"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "========================================================================"

# ====================================================================
# 1. 环境变量配置（根据您的 HPC 集群修改）
# ====================================================================

# 设置项目路径（自动检测或手动指定）
if [ -z "$KAVA_PROJECT_ROOT" ]; then
    export KAVA_PROJECT_ROOT="/path/to/kava/project"
fi

# 设置数据路径（建议使用 scratch 目录）
if [ -z "$KAVA_DATA_PATH" ]; then
    export KAVA_DATA_PATH="/scratch/$USER/kava/data"
fi

# 设置模型路径
if [ -z "$KAVA_MODEL_PATH" ]; then
    export KAVA_MODEL_PATH="/scratch/$USER/kava/models"
fi

# 设置缓存路径
if [ -z "$KAVA_CACHE_PATH" ]; then
    export KAVA_CACHE_PATH="/scratch/$USER/kava/cache"
fi

# 设置输出路径
if [ -z "$KAVA_OUTPUT_PATH" ]; then
    export KAVA_OUTPUT_PATH="/scratch/$USER/kava/outputs"
fi

echo "Project Root: $KAVA_PROJECT_ROOT"
echo "Data Path: $KAVA_DATA_PATH"
echo "Model Path: $KAVA_MODEL_PATH"
echo "Output Path: $KAVA_OUTPUT_PATH"
echo "------------------------------------------------------------------------"

# ====================================================================
# 2. 加载模块（根据您的 HPC 环境）
# ====================================================================

# 常见模块加载命令（取消注释适合您集群的命令）

# Option A: 使用预安装的 CUDA 和 Python
# module load cuda/12.1
# module load python/3.10

# Option B: 使用 Anaconda
# module load anaconda3
# source activate kava_env

# Option C: 使用系统 Python + venv
# module load python/3.10
# source $KAVA_PROJECT_ROOT/.venv/bin/activate

echo "Modules loaded successfully"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "------------------------------------------------------------------------"

# ====================================================================
# 3. 验证 GPU 可用性
# ====================================================================

echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "------------------------------------------------------------------------"

# ====================================================================
# 4. 创建必要的目录
# ====================================================================

mkdir -p $KAVA_DATA_PATH
mkdir -p $KAVA_MODEL_PATH
mkdir -p $KAVA_CACHE_PATH
mkdir -p $KAVA_OUTPUT_PATH
mkdir -p $KAVA_PROJECT_ROOT/logs
mkdir -p $KAVA_PROJECT_ROOT/checkpoints

# ====================================================================
# 5. 切换到项目目录
# ====================================================================

cd $KAVA_PROJECT_ROOT

echo "Current directory: $(pwd)"
echo "------------------------------------------------------------------------"

# ====================================================================
# 6. 运行训练脚本（环境自适应）
# ====================================================================

echo "Starting KAVA training..."
echo "========================================================================"

python train_adaptive.py \
    --max_steps 1000 \
    --batch_size auto \
    --gradient_accumulation auto \
    --save_interval 200 \
    --eval_interval 200

TRAIN_EXIT_CODE=$?

echo "========================================================================"
echo "Training completed with exit code: $TRAIN_EXIT_CODE"
echo "End Time: $(date)"
echo "========================================================================"

# ====================================================================
# 7. 保存作业信息
# ====================================================================

JOB_INFO_FILE="$KAVA_OUTPUT_PATH/job_${SLURM_JOB_ID}_info.txt"

cat > $JOB_INFO_FILE << EOF
KAVA Training Job Information
========================================
Job ID: $SLURM_JOB_ID
Node: $SLURMD_NODENAME
Start Time: $SLURM_JOB_START_TIME
End Time: $(date)
Exit Code: $TRAIN_EXIT_CODE

Paths:
  Project: $KAVA_PROJECT_ROOT
  Data: $KAVA_DATA_PATH
  Models: $KAVA_MODEL_PATH
  Output: $KAVA_OUTPUT_PATH

GPU:
$(nvidia-smi --query-gpu=name,memory.total --format=csv)

Python Environment:
$(python --version)
PyTorch: $(python -c "import torch; print(torch.__version__)")
CUDA Available: $(python -c "import torch; print(torch.cuda.is_available())")
EOF

echo "Job info saved to: $JOB_INFO_FILE"

exit $TRAIN_EXIT_CODE
