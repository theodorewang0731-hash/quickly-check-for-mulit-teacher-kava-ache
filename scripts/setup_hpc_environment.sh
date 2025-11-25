#!/bin/bash

################################################################################
# HPC 环境配置脚本
# 
# 目的：在计算节点上自动加载 CUDA 并激活 Python 环境
# 用法：在所有 SLURM 脚本开头 source 此文件
################################################################################

echo "========================================================"
echo "正在配置 HPC 环境..."
echo "========================================================"

# ============================================================
# 1. 尝试加载 CUDA 模块（如果 HPC 提供）
# ============================================================

# 常见的 CUDA 模块名称（根据你的 HPC 调整）
CUDA_MODULES=(
    "cuda/11.8"
    "cuda/12.1"
    "cuda/12.2"
    "cudatoolkit/11.8"
    "nvidia/cuda/11.8"
)

CUDA_LOADED=false

for cuda_module in "${CUDA_MODULES[@]}"; do
    if module avail 2>&1 | grep -q "${cuda_module}"; then
        echo "✓ 找到 CUDA 模块: ${cuda_module}"
        module load "${cuda_module}"
        CUDA_LOADED=true
        break
    fi
done

if [ "$CUDA_LOADED" = false ]; then
    echo "⚠ 未找到预配置的 CUDA 模块"
    echo "⚠ 尝试使用系统 CUDA（如果存在）"
    
    # 检查系统 CUDA
    if command -v nvcc &> /dev/null; then
        echo "✓ 系统中存在 nvcc: $(nvcc --version | grep release)"
    else
        echo "✗ 未检测到 CUDA 工具链"
        echo "✗ 训练将在 CPU 上运行（非常慢）"
    fi
fi

# ============================================================
# 2. 检查 GPU 可用性
# ============================================================

if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "========================================================"
    echo "GPU 信息："
    echo "========================================================"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    echo "========================================================"
else
    echo "⚠ 未检测到 nvidia-smi，可能没有 GPU"
fi

# ============================================================
# 3. 激活 Python 环境
# ============================================================

# 方案 A：使用 virtualenv (kava_env)
VENV_PATH="./kava_env/bin/activate"
if [ -f "$VENV_PATH" ]; then
    echo "✓ 激活 virtualenv: kava_env"
    source "$VENV_PATH"
    
# 方案 B：使用 conda 环境
elif command -v conda &> /dev/null; then
    echo "✓ 激活 conda 环境: kava"
    source ~/.bashrc
    conda activate kava
    
else
    echo "✗ 未找到 Python 环境 (kava_env 或 conda kava)"
    exit 1
fi

# ============================================================
# 4. 验证关键包
# ============================================================

echo ""
echo "========================================================"
echo "Python 环境验证："
echo "========================================================"

python -c "
import sys
print(f'Python: {sys.version}')

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA Available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA Version: {torch.version.cuda}')
        print(f'GPU Count: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
except ImportError:
    print('⚠ PyTorch 未安装')
    sys.exit(1)

try:
    import transformers
    print(f'Transformers: {transformers.__version__}')
except ImportError:
    print('⚠ Transformers 未安装')
    sys.exit(1)

try:
    import accelerate
    print(f'Accelerate: {accelerate.__version__}')
except ImportError:
    print('⚠ Accelerate 未安装')
"

if [ $? -ne 0 ]; then
    echo "✗ 环境验证失败"
    exit 1
fi

echo "========================================================"
echo "✓ HPC 环境配置完成"
echo "========================================================"
echo ""

# ============================================================
# 5. 导出环境变量
# ============================================================

# ============================================================
# 【重要】HPC 共享模型库配置
# ============================================================
# 优先使用 HPC 共享模型库，避免重复下载和 HuggingFace 登录问题
# 如果 /home/share/models 不存在，则回退到用户缓存目录
if [ -d "/home/share/models" ]; then
    export HF_HOME="/home/share/models"
    export TRANSFORMERS_CACHE="/home/share/models"
    export HF_DATASETS_CACHE="${HOME}/.cache/huggingface/datasets"
    echo "✓ 使用 HPC 共享模型库: /home/share/models"
else
    export HF_HOME="${HOME}/.cache/huggingface"
    export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
    export HF_DATASETS_CACHE="${HF_HOME}/datasets"
    echo "⚠ 共享模型库不存在，使用用户缓存目录"
fi

# 优化设置
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# 如果需要登录 Hugging Face（需要事先设置 token）
# 注意：使用共享模型库时无需登录
# export HF_TOKEN="your_token_here"

echo "环境变量已设置："
echo "  HF_HOME=${HF_HOME}"
echo "  TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE}"
echo "  HF_DATASETS_CACHE=${HF_DATASETS_CACHE}"
echo "  OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo ""
