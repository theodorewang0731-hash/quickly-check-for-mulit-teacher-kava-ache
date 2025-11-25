#!/bin/bash

################################################################################
# KaVa 项目自动修复脚本
# 
# 根据 KAVA 项目实施经验，自动修复常见问题：
# 1. Windows 换行符 → Unix LF
# 2. 脚本执行权限
# 3. 路径引号检查
# 4. 目录结构创建
################################################################################

set -e

echo "========================================================"
echo "KaVa 多教师蒸馏项目 - 自动修复脚本"
echo "========================================================"
echo ""

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "项目目录: $PROJECT_ROOT"
echo ""

# ============================================================
# 1. 修复换行符（问题 10）
# ============================================================

echo "步骤 1/5: 修复 Windows 换行符"
echo "----------------------------------------"

if command -v dos2unix &> /dev/null; then
    echo "使用 dos2unix 转换..."
    dos2unix scripts/*.sh 2>/dev/null || true
    echo "✓ 使用 dos2unix 完成"
elif command -v sed &> /dev/null; then
    echo "使用 sed 转换..."
    for file in scripts/*.sh; do
        if [ -f "$file" ]; then
            sed -i 's/\r$//' "$file"
            echo "  ✓ $file"
        fi
    done
    echo "✓ 使用 sed 完成"
else
    echo "⚠ 警告: 未找到 dos2unix 或 sed，跳过换行符转换"
fi

echo ""

# ============================================================
# 2. 设置脚本执行权限
# ============================================================

echo "步骤 2/5: 设置脚本执行权限"
echo "----------------------------------------"

chmod +x scripts/*.sh
echo "✓ 所有 .sh 脚本已设置执行权限"

if [ -f "scripts/pre_training_validation.py" ]; then
    chmod +x scripts/pre_training_validation.py
    echo "✓ pre_training_validation.py 已设置执行权限"
fi

echo ""

# ============================================================
# 3. 创建必需目录
# ============================================================

echo "步骤 3/5: 创建必需目录"
echo "----------------------------------------"

REQUIRED_DIRS=(
    "logs"
    "outputs"
    "results"
    "data/unified_splits"
    "outputs/figures"
    "outputs/checkpoints"
)

for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        echo "  ✓ 创建: $dir"
    else
        echo "  ✓ 已存在: $dir"
    fi
done

echo ""

# ============================================================
# 4. 检查并修复路径引号（问题 5）
# ============================================================

echo "步骤 4/5: 检查路径变量引号"
echo "----------------------------------------"

# 检测未加引号的 cd 命令
ISSUES_FOUND=0

for script in scripts/*.sh; do
    if [ -f "$script" ]; then
        # 检测 cd $VAR（未加引号）
        if grep -n '^[^#]*cd \$[A-Z_]*[^"]' "$script" | grep -v 'cd "$' > /dev/null 2>&1; then
            echo "⚠ $script: 发现未加引号的 cd 命令"
            ISSUES_FOUND=$((ISSUES_FOUND + 1))
        fi
    fi
done

if [ $ISSUES_FOUND -eq 0 ]; then
    echo "✓ 所有路径变量已正确处理"
else
    echo "⚠ 发现 $ISSUES_FOUND 个脚本可能需要手动检查"
    echo ""
    echo "  修复示例:"
    echo "    修复前: cd \$SLURM_SUBMIT_DIR"
    echo "    修复后: cd \"\$SLURM_SUBMIT_DIR\""
fi

echo ""

# ============================================================
# 5. 验证配置文件
# ============================================================

echo "步骤 5/5: 验证项目配置"
echo "----------------------------------------"

# 检查关键文件
CRITICAL_FILES=(
    "requirements.txt"
    "scripts/run_multi_seed_experiments.sh"
    "scripts/setup_hpc_environment.sh"
    "scripts/check_gpu_node.sh"
    "experiments/train_multi_teacher_kv.py"
)

MISSING=0
for file in "${CRITICAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ $file (缺失)"
        MISSING=$((MISSING + 1))
    fi
done

echo ""

# ============================================================
# 总结
# ============================================================

echo "========================================================"
echo "修复完成！"
echo "========================================================"
echo ""

if [ $MISSING -eq 0 ]; then
    echo "✓ 所有关键文件存在"
else
    echo "✗ 缺少 $MISSING 个关键文件"
fi

echo ""
echo "下一步操作:"
echo ""
echo "  1. 运行完整验证:"
echo "     python scripts/pre_training_validation.py"
echo ""
echo "  2. 如果验证通过，部署到 HPC:"
echo "     bash scripts/deploy_and_check.sh"
echo ""
echo "  3. 查看 GPU 检测报告:"
echo "     cat logs/gpu_check_*.out"
echo ""
echo "========================================================"
