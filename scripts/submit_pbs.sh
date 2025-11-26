#!/bin/bash
#PBS -N kava_training
#PBS -o logs/kava_${PBS_JOBID}.out
#PBS -e logs/kava_${PBS_JOBID}.err
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -l walltime=04:00:00
#PBS -l mem=32gb
#PBS -q gpu

# ====================================================================
# KAVA Training - PBS Job Script
# 适用于 PBS/Torque 调度器的 HPC 集群
# ====================================================================

echo "========================================================================"
echo "KAVA Training Job Started (PBS)"
echo "========================================================================"
echo "Job ID: $PBS_JOBID"
echo "Node: $PBS_O_HOST"
echo "Start Time: $(date)"
echo "========================================================================"

# ====================================================================
# 环境变量配置
# ====================================================================

# 设置路径（根据您的环境修改）
export KAVA_PROJECT_ROOT="${HOME}/projects/kava"
export KAVA_DATA_PATH="/scratch/${USER}/kava/data"
export KAVA_MODEL_PATH="/scratch/${USER}/kava/models"
export KAVA_OUTPUT_PATH="/scratch/${USER}/kava/outputs"

# PBS 特定环境变量
cd $PBS_O_WORKDIR

# 加载模块
module load cuda/12.1
module load python/3.10

# 激活虚拟环境
source $KAVA_PROJECT_ROOT/.venv/bin/activate

# 创建必要目录
mkdir -p $KAVA_DATA_PATH $KAVA_MODEL_PATH $KAVA_OUTPUT_PATH
mkdir -p $KAVA_PROJECT_ROOT/logs $KAVA_PROJECT_ROOT/checkpoints

# 运行训练
cd $KAVA_PROJECT_ROOT

python train_adaptive.py \
    --max_steps 1000 \
    --save_interval 200

TRAIN_EXIT_CODE=$?

echo "========================================================================"
echo "Training completed with exit code: $TRAIN_EXIT_CODE"
echo "End Time: $(date)"
echo "========================================================================"

exit $TRAIN_EXIT_CODE
