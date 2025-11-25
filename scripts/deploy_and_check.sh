#!/bin/bash

################################################################################
# HPC 一键部署脚本
# 
# 自动执行所有验证和检测步骤
# 用法: bash scripts/deploy_and_check.sh
################################################################################

set -e  # 遇到错误立即退出

echo "========================================================"
echo "KaVa 项目 - HPC 自动部署与检测"
echo "========================================================"
echo ""

# 获取项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "项目目录: $PROJECT_ROOT"
echo ""

# ============================================================
# 步骤 1: 检查必需文件
# ============================================================

echo "步骤 1/5: 检查必需文件"
echo "----------------------------------------"

REQUIRED_FILES=(
    "scripts/verify_login_node.sh"
    "scripts/check_gpu_node.sh"
    "scripts/setup_hpc_environment.sh"
    "requirements.txt"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file"
    else
        echo "✗ $file 不存在"
        exit 1
    fi
done

echo "✓ 所有必需文件存在"
echo ""

# ============================================================
# 步骤 2: 给脚本执行权限
# ============================================================

echo "步骤 2/5: 设置脚本权限"
echo "----------------------------------------"

chmod +x scripts/*.sh
echo "✓ 所有脚本已设置执行权限"
echo ""

# ============================================================
# 步骤 3: 创建必需目录
# ============================================================

echo "步骤 3/5: 创建必需目录"
echo "----------------------------------------"

mkdir -p logs
mkdir -p outputs
mkdir -p results
mkdir -p data/unified_splits

echo "✓ 所有目录已创建"
echo ""

# ============================================================
# 步骤 4: 运行登录节点验证
# ============================================================

echo "步骤 4/5: 运行登录节点验证"
echo "----------------------------------------"

if bash scripts/verify_login_node.sh; then
    echo ""
    echo "✓ 登录节点验证成功"
    echo ""
else
    echo ""
    echo "✗ 登录节点验证失败"
    echo "请检查输出并修复问题"
    exit 1
fi

# ============================================================
# 步骤 5: 提交 GPU 检测作业
# ============================================================

echo "步骤 5/5: 提交 GPU 检测作业"
echo "----------------------------------------"

if command -v sbatch &> /dev/null; then
    echo "✓ sbatch 命令可用"
    
    # 提交作业
    JOB_ID=$(sbatch scripts/check_gpu_node.sh | awk '{print $NF}')
    
    if [ ! -z "$JOB_ID" ]; then
        echo "✓ GPU 检测作业已提交"
        echo "  作业 ID: $JOB_ID"
        echo ""
        echo "========================================================"
        echo "部署完成！"
        echo "========================================================"
        echo ""
        echo "下一步操作："
        echo ""
        echo "  1. 等待 GPU 检测作业完成（通常 1-2 分钟）："
        echo "     squeue -u \$USER"
        echo "     # 或查看特定作业："
        echo "     squeue -j ${JOB_ID}"
        echo ""
        echo "  2. 查看 GPU 检测报告："
        echo "     cat logs/gpu_check_${JOB_ID}.out"
        echo ""
        echo "  3. 根据报告采取行动："
        echo "     - 如果显示 '🎉 恭喜'：立即提交训练"
        echo "       sbatch scripts/run_multi_seed_experiments.sh"
        echo ""
        echo "     - 如果需要重装 PyTorch：按报告中的命令执行"
        echo ""
        echo "     - 如果有其他问题：查看 HPC_EXECUTION_CHECKLIST.md"
        echo ""
        echo "========================================================"
    else
        echo "✗ 作业提交失败"
        exit 1
    fi
else
    echo "⚠ sbatch 命令不可用"
    echo ""
    echo "可能原因："
    echo "  1. 此 HPC 不使用 SLURM 调度系统"
    echo "  2. 需要先加载 SLURM 模块: module load slurm"
    echo ""
    echo "请手动提交 GPU 检测作业："
    echo "  sbatch scripts/check_gpu_node.sh"
    echo ""
    echo "或联系 HPC 管理员询问作业提交方式"
fi
