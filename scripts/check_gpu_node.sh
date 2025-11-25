#!/bin/bash

################################################################################
# GPU 计算节点环境检测作业
# 
# 目的：提交到计算节点，检测 GPU/CUDA 环境并报告
# 用时：~1 分钟
################################################################################

#SBATCH --job-name=gpu_check
#SBATCH --partition=gpu           # 改为你的 GPU 分区名称
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1              # 只申请 1 个 GPU 用于测试
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00           # 5 分钟足够
#SBATCH --output=logs/gpu_check_%j.out
#SBATCH --error=logs/gpu_check_%j.err

echo "========================================================"
echo "GPU 计算节点环境检测报告"
echo "========================================================"
echo "作业 ID: ${SLURM_JOB_ID}"
echo "节点名称: ${SLURM_NODELIST}"
echo "开始时间: $(date)"
echo ""

# ============================================================
# 1. 系统信息
# ============================================================

echo "========================================================"
echo "1. 系统信息"
echo "========================================================"
echo "主机名: $(hostname)"
echo "操作系统: $(cat /etc/os-release | grep PRETTY_NAME | cut -d '"' -f 2)"
echo "内核版本: $(uname -r)"
echo ""

# ============================================================
# 2. GPU 检测
# ============================================================

echo "========================================================"
echo "2. GPU 检测"
echo "========================================================"

if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi 可用"
    echo ""
    nvidia-smi
    echo ""
    
    echo "GPU 列表（简表）："
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
    echo ""
else
    echo "✗ nvidia-smi 不可用"
    echo "✗ 此节点可能没有 NVIDIA GPU 或驱动未安装"
fi

# ============================================================
# 3. CUDA 工具链检测
# ============================================================

echo "========================================================"
echo "3. CUDA 工具链检测"
echo "========================================================"

# 检查 nvcc
if command -v nvcc &> /dev/null; then
    echo "✓ nvcc 可用"
    nvcc --version
    echo ""
else
    echo "✗ nvcc 不可用（CUDA 编译器未安装或未加载）"
fi

# 检查 CUDA 库路径
if [ -d "/usr/local/cuda" ]; then
    echo "✓ 发现 CUDA 安装目录: /usr/local/cuda"
    if [ -f "/usr/local/cuda/version.txt" ]; then
        cat /usr/local/cuda/version.txt
    elif [ -f "/usr/local/cuda/version.json" ]; then
        cat /usr/local/cuda/version.json
    fi
    echo ""
else
    echo "⚠ 未发现标准 CUDA 安装路径 (/usr/local/cuda)"
fi

# 检查环境变量
echo "CUDA 相关环境变量："
env | grep -i cuda || echo "  (无)"
echo ""

# ============================================================
# 4. Module 系统检测
# ============================================================

echo "========================================================"
echo "4. Module 系统检测"
echo "========================================================"

if command -v module &> /dev/null; then
    echo "✓ module 命令可用"
    echo ""
    
    echo "已加载的模块："
    module list 2>&1 || echo "  (无)"
    echo ""
    
    echo "可用的 CUDA 相关模块："
    module avail 2>&1 | grep -i cuda || echo "  (无 CUDA 模块)"
    echo ""
    
    echo "可用的 Python 相关模块："
    module avail 2>&1 | grep -i python || echo "  (无 Python 模块)"
    echo ""
else
    echo "✗ module 命令不可用（此 HPC 可能不使用 module 系统）"
fi

# ============================================================
# 5. Python 环境检测
# ============================================================

echo "========================================================"
echo "5. Python 环境检测"
echo "========================================================"

# 尝试激活虚拟环境
if [ -f "kava_env/bin/activate" ]; then
    echo "✓ 找到虚拟环境: kava_env"
    source kava_env/bin/activate
    echo ""
else
    echo "⚠ 未找到 kava_env（将使用系统 Python）"
    echo ""
fi

# Python 版本
echo "Python 版本:"
python --version
echo ""

# PyTorch 检测
echo "PyTorch 环境检测："
python -c "
import sys
import os

print(f'Python 可执行文件: {sys.executable}')
print(f'Python 路径: {sys.prefix}')
print('')

try:
    import torch
    print(f'✓ PyTorch 已安装')
    print(f'  版本: {torch.__version__}')
    print(f'  CUDA 编译支持: {torch.version.cuda}')
    print(f'  CUDA 运行时可用: {torch.cuda.is_available()}')
    
    if torch.cuda.is_available():
        print(f'  GPU 数量: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
            print(f'    显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB')
    else:
        print('')
        print('⚠ PyTorch 检测不到 CUDA/GPU')
        print('  可能原因:')
        print('  1. PyTorch 是 CPU-only 版本')
        print('  2. CUDA 驱动版本与 PyTorch 编译版本不匹配')
        print('  3. 需要加载 CUDA module')
        
except ImportError:
    print('✗ PyTorch 未安装')
    sys.exit(1)
" 2>&1

echo ""

# ============================================================
# 6. 其他依赖检测
# ============================================================

echo "========================================================"
echo "6. 其他关键依赖"
echo "========================================================"

python -c "
packages = ['transformers', 'accelerate', 'datasets', 'matplotlib', 'scipy', 'sklearn']

for pkg in packages:
    try:
        if pkg == 'sklearn':
            mod = __import__('sklearn')
        else:
            mod = __import__(pkg)
        version = getattr(mod, '__version__', 'unknown')
        print(f'✓ {pkg:15s} {version}')
    except ImportError:
        print(f'✗ {pkg:15s} NOT INSTALLED')
"

echo ""

# ============================================================
# 总结与建议
# ============================================================

echo "========================================================"
echo "7. 总结与建议"
echo "========================================================"

# 检查 GPU 是否可用
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "✓ GPU 环境：可用"
    GPU_AVAILABLE=true
else
    echo "✗ GPU 环境：不可用"
    GPU_AVAILABLE=false
fi

# 检查 PyTorch CUDA
PYTORCH_CUDA=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "false")

if [ "$GPU_AVAILABLE" = true ] && [ "$PYTORCH_CUDA" = "True" ]; then
    echo ""
    echo "🎉 恭喜！计算节点环境完全正常！"
    echo ""
    echo "下一步："
    echo "  1. 可以直接提交训练作业："
    echo "     sbatch scripts/run_multi_seed_experiments.sh"
    echo ""
    
elif [ "$GPU_AVAILABLE" = true ] && [ "$PYTORCH_CUDA" = "False" ]; then
    echo ""
    echo "⚠ GPU 可用，但 PyTorch 检测不到 CUDA"
    echo ""
    echo "解决方案："
    echo "  1. 检查是否需要加载 CUDA module（查看上面第 4 节）"
    echo "  2. 重新安装匹配的 PyTorch 版本："
    echo ""
    
    # 获取驱动版本
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
    echo "     当前驱动版本: ${DRIVER_VERSION}"
    
    # 推荐 CUDA 版本
    if [[ "$DRIVER_VERSION" > "450" ]]; then
        echo "     推荐安装 PyTorch with CUDA 11.8+:"
        echo "     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    else
        echo "     推荐安装 PyTorch with CUDA 11.3:"
        echo "     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113"
    fi
    echo ""
    
elif [ "$GPU_AVAILABLE" = false ]; then
    echo ""
    echo "✗ 计算节点没有 GPU"
    echo ""
    echo "可能原因："
    echo "  1. SLURM 配置错误（检查 --partition 和 --gres 参数）"
    echo "  2. 此 HPC 没有 GPU 资源"
    echo "  3. GPU 驱动未安装"
    echo ""
    echo "请联系 HPC 管理员确认 GPU 资源配置"
    echo ""
fi

echo "========================================================"
echo "检测完成时间: $(date)"
echo "========================================================"
