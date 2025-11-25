#!/bin/bash

################################################################################
# 登录节点验证脚本
# 
# 目的：在没有 GPU 的登录节点上验证代码和环境
# 不会下载大模型，只测试导入和基础功能
################################################################################

echo "========================================================"
echo "KaVa 项目 - 登录节点验证"
echo "========================================================"
echo ""

# ============================================================
# 1. 检查当前环境
# ============================================================

echo "步骤 1/4: 检查当前环境"
echo "----------------------------------------"

# 检查 Python 环境
if [ -d "kava_env" ]; then
    echo "✓ 找到虚拟环境: kava_env"
    source kava_env/bin/activate
else
    echo "✗ 未找到 kava_env 虚拟环境"
    echo "请先运行: python -m venv kava_env && source kava_env/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# 检查 GPU（预期：登录节点无 GPU）
if command -v nvidia-smi &> /dev/null; then
    echo "⚠ 检测到 nvidia-smi（意外：登录节点通常无 GPU）"
    nvidia-smi
else
    echo "✓ 登录节点无 GPU（符合预期）"
fi

echo ""

# ============================================================
# 2. 验证 Python 依赖
# ============================================================

echo "步骤 2/4: 验证 Python 依赖"
echo "----------------------------------------"

python -c "
import sys
import importlib

packages = [
    'torch',
    'transformers',
    'accelerate',
    'datasets',
    'numpy',
    'matplotlib',
    'seaborn',
    'scipy',
    'sklearn',
]

print('检查已安装的包：')
print('-' * 50)

missing = []
for pkg in packages:
    try:
        if pkg == 'sklearn':
            mod = importlib.import_module('sklearn')
        else:
            mod = importlib.import_module(pkg)
        
        version = getattr(mod, '__version__', 'unknown')
        print(f'✓ {pkg:15s} {version}')
    except ImportError:
        print(f'✗ {pkg:15s} NOT INSTALLED')
        missing.append(pkg)

print('-' * 50)

if missing:
    print(f'\n✗ 缺少 {len(missing)} 个包: {missing}')
    print('请运行: pip install -r requirements.txt')
    sys.exit(1)
else:
    print('\n✓ 所有依赖已安装')

# 检查 PyTorch CUDA
import torch
print(f'\nPyTorch CUDA 可用: {torch.cuda.is_available()}')
if not torch.cuda.is_available():
    print('✓ CPU 版本（登录节点适用）')
"

if [ $? -ne 0 ]; then
    echo "✗ 依赖验证失败"
    exit 1
fi

echo ""

# ============================================================
# 3. 测试代码导入
# ============================================================

echo "步骤 3/4: 测试代码导入"
echo "----------------------------------------"

python -c "
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path.cwd()))

print('测试核心模块导入：')
print('-' * 50)

modules = [
    ('align.tokenizer_align', 'TokenizerAligner'),
    ('align.layer_map', 'LayerMapper'),
    ('teacher.router_proto', 'RouterProto'),
    ('data.data_split_controller', 'DataSplitController'),
    ('utils.training_budget_controller', 'TrainingBudgetController'),
    ('utils.statistical_significance', 'MultiSeedAggregator'),
    ('utils.learning_curve_tracker', 'LearningCurveTracker'),
]

failed = []
for module_path, class_name in modules:
    try:
        parts = module_path.split('.')
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        print(f'✓ {module_path}.{class_name}')
    except Exception as e:
        print(f'✗ {module_path}.{class_name}: {e}')
        failed.append(module_path)

print('-' * 50)

if failed:
    print(f'\n✗ {len(failed)} 个模块导入失败')
    sys.exit(1)
else:
    print('\n✓ 所有核心模块导入成功')
"

if [ $? -ne 0 ]; then
    echo "✗ 代码导入测试失败"
    exit 1
fi

echo ""

# ============================================================
# 4. 运行轻量级模拟训练
# ============================================================

echo "步骤 4/4: 运行轻量级模拟训练"
echo "----------------------------------------"

if [ -f "experiments/train_minimal.py" ]; then
    echo "运行 train_minimal.py（模拟模式）..."
    python experiments/train_minimal.py
    
    if [ $? -eq 0 ]; then
        echo "✓ 模拟训练成功"
        
        # 检查输出
        if [ -f "dummy_model.txt" ]; then
            echo "✓ 生成了 dummy_model.txt"
            echo ""
            echo "内容预览："
            head -n 5 dummy_model.txt
        fi
    else
        echo "✗ 模拟训练失败"
        exit 1
    fi
else
    echo "⚠ 未找到 train_minimal.py，跳过模拟训练"
fi

echo ""

# ============================================================
# 总结
# ============================================================

echo "========================================================"
echo "✓ 登录节点验证完成！"
echo "========================================================"
echo ""
echo "⚠ 重要提示："
echo "  登录节点没有 GPU/CUDA（这是正常的）"
echo "  真正的训练需要在计算节点（通过 SLURM）运行"
echo ""
echo "下一步操作："
echo ""
echo "  1. 【推荐】先检测计算节点的 GPU 环境："
echo "     chmod +x scripts/check_gpu_node.sh"
echo "     sbatch scripts/check_gpu_node.sh"
echo "     # 等待作业完成后查看报告："
echo "     cat logs/gpu_check_*.out"
echo ""
echo "  2. 确认 GPU 可用后，提交真实训练："
echo "     sbatch scripts/run_multi_seed_experiments.sh"
echo ""
echo "  3. 查看作业状态："
echo "     squeue -u \$USER"
echo ""
echo "  4. 查看日志输出："
echo "     tail -f logs/multi_seed_*.out"
echo ""
echo "========================================================"
