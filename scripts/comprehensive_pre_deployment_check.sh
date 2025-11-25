#!/bin/bash

################################################################################
# HPC 综合部署前验证脚本
#
# 功能:
# 1. 运行所有检查工具
# 2. 验证环境配置
# 3. 检查共享模型库
# 4. 检测资源配置
# 5. 验证脚本语法
# 6. 生成部署报告
#
# 用法:
#   bash scripts/comprehensive_pre_deployment_check.sh
################################################################################

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 错误计数
ERROR_COUNT=0
WARNING_COUNT=0

echo "========================================================================"
echo -e "${BOLD}KaVa 项目 - HPC 综合部署前验证${NC}"
echo "========================================================================"
echo ""
echo "项目路径: ${PROJECT_ROOT}"
echo "检查时间: $(date)"
echo ""

# ============================================================
# 辅助函数
# ============================================================

log_section() {
    echo ""
    echo "========================================================================"
    echo -e "${CYAN}${BOLD}$1${NC}"
    echo "========================================================================"
    echo ""
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
    ((ERROR_COUNT++))
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNING_COUNT++))
}

log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# ============================================================
# 1. 环境基础检查
# ============================================================

log_section "[1/8] 环境基础检查"

# 检查 Python
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1)
    log_success "Python 可用: ${PYTHON_VERSION}"
else
    log_error "Python 未找到"
fi

# 检查虚拟环境
if [ -d "${PROJECT_ROOT}/.venv" ]; then
    log_success "虚拟环境存在: ${PROJECT_ROOT}/.venv"
else
    log_warning "虚拟环境不存在，建议运行: python -m venv .venv"
fi

# 检查 requirements.txt
if [ -f "${PROJECT_ROOT}/requirements.txt" ]; then
    log_success "依赖文件存在: requirements.txt"
    req_count=$(wc -l < "${PROJECT_ROOT}/requirements.txt")
    log_info "  包含 ${req_count} 个依赖项"
else
    log_error "requirements.txt 不存在"
fi

# 检查目录结构
REQUIRED_DIRS=("experiments" "kava" "scripts" "data" "logs")
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ -d "${PROJECT_ROOT}/${dir}" ]; then
        log_success "目录存在: ${dir}"
    else
        log_error "目录缺失: ${dir}"
    fi
done

# ============================================================
# 2. HPC 共享模型库检查
# ============================================================

log_section "[2/8] HPC 共享模型库检查"

if [ -f "${SCRIPT_DIR}/check_hpc_models.sh" ]; then
    log_info "运行模型库检查脚本..."
    echo ""
    
    # 运行模型检查（捕获退出码但不中断）
    if bash "${SCRIPT_DIR}/check_hpc_models.sh"; then
        log_success "模型库检查通过"
    else
        log_error "模型库检查失败 - 请查看上面的详细信息"
    fi
else
    log_error "模型库检查脚本不存在: check_hpc_models.sh"
fi

# ============================================================
# 3. SLURM 资源配置检查
# ============================================================

log_section "[3/8] SLURM 资源配置检查"

# 检查 SLURM 命令是否可用
if command -v squeue &> /dev/null; then
    log_success "SLURM 命令可用"
    
    # 检查分区
    if command -v sinfo &> /dev/null; then
        gpu_partition=$(sinfo -h -p gpu 2>/dev/null | head -n 1)
        if [ -n "${gpu_partition}" ]; then
            log_success "GPU 分区可用"
        else
            log_warning "未找到 GPU 分区，请确认分区名称"
        fi
    fi
else
    log_warning "SLURM 命令不可用（可能在本地环境）"
fi

# 检查资源检测脚本
if [ -f "${SCRIPT_DIR}/detect_slurm_resources.sh" ]; then
    log_success "资源检测脚本存在: detect_slurm_resources.sh"
    log_info "  提示: 可在计算节点上运行以获取详细信息"
else
    log_error "资源检测脚本不存在: detect_slurm_resources.sh"
fi

# ============================================================
# 4. SLURM 脚本语法检查
# ============================================================

log_section "[4/8] SLURM 脚本语法检查"

SLURM_SCRIPTS=(
    "run_multi_seed_experiments.sh"
    "run_ablation_studies.sh"
    "run_three_stage_routing.sh"
    "run_large_scale_multi_teacher.sh"
    "train_with_visualization.sh"
    "run_all_baselines.sh"
)

for script in "${SLURM_SCRIPTS[@]}"; do
    script_path="${SCRIPT_DIR}/${script}"
    if [ -f "${script_path}" ]; then
        # 检查语法
        if bash -n "${script_path}" 2>/dev/null; then
            log_success "${script} - 语法正确"
            
            # 检查是否使用了共享环境配置
            if grep -q "setup_hpc_environment.sh" "${script_path}"; then
                log_info "  ✓ 使用了统一环境配置"
            else
                log_warning "  未使用 setup_hpc_environment.sh"
            fi
            
            # 检查 GPU 格式注释
            if grep -q "# 根据 HPC 要求可能需要完整格式" "${script_path}"; then
                log_info "  ✓ 包含 GPU 格式提示"
            else
                log_warning "  缺少 GPU 格式提示注释"
            fi
        else
            log_error "${script} - 语法错误"
        fi
    else
        log_error "${script} - 文件不存在"
    fi
done

# ============================================================
# 5. Python 脚本检查
# ============================================================

log_section "[5/8] Python 训练脚本检查"

PYTHON_SCRIPTS=(
    "experiments/train_standard_sft.py"
    "experiments/train_single_teacher.py"
    "experiments/train_multi_teacher.py"
)

for script in "${PYTHON_SCRIPTS[@]}"; do
    script_path="${PROJECT_ROOT}/${script}"
    if [ -f "${script_path}" ]; then
        # 检查语法
        if python -m py_compile "${script_path}" 2>/dev/null; then
            log_success "${script} - 语法正确"
        else
            log_error "${script} - 语法错误"
        fi
    else
        log_error "${script} - 文件不存在"
    fi
done

# ============================================================
# 6. 工具脚本完整性检查
# ============================================================

log_section "[6/8] 工具脚本完整性检查"

TOOL_SCRIPTS=(
    "setup_hpc_environment.sh"
    "auto_fix.sh"
    "pre_training_validation.py"
    "monitor_training.sh"
    "check_hpc_models.sh"
    "detect_slurm_resources.sh"
)

for script in "${TOOL_SCRIPTS[@]}"; do
    script_path="${SCRIPT_DIR}/${script}"
    if [ -f "${script_path}" ]; then
        # 检查可执行权限
        if [ -x "${script_path}" ]; then
            log_success "${script} - 存在且可执行"
        else
            log_warning "${script} - 存在但不可执行，建议运行: chmod +x ${script_path}"
        fi
    else
        log_error "${script} - 不存在"
    fi
done

# ============================================================
# 7. 文件编码和行尾检查
# ============================================================

log_section "[7/8] 文件编码和行尾检查"

# 检查是否有 CRLF 行尾
crlf_files=$(find "${SCRIPT_DIR}" -name "*.sh" -exec file {} \; 2>/dev/null | grep -i "CRLF" | wc -l)

if [ "${crlf_files}" -gt 0 ]; then
    log_warning "发现 ${crlf_files} 个脚本使用 Windows 行尾 (CRLF)"
    log_info "  运行以下命令修复:"
    log_info "  bash scripts/auto_fix.sh"
else
    log_success "所有脚本使用正确的 Unix 行尾 (LF)"
fi

# ============================================================
# 8. 配置文件检查
# ============================================================

log_section "[8/8] 配置文件检查"

# 检查环境配置脚本中的共享模型库设置
if [ -f "${SCRIPT_DIR}/setup_hpc_environment.sh" ]; then
    if grep -q "/home/share/models" "${SCRIPT_DIR}/setup_hpc_environment.sh"; then
        log_success "环境脚本配置了共享模型库路径"
    else
        log_warning "环境脚本未配置共享模型库路径"
    fi
    
    if grep -q "TRANSFORMERS_CACHE" "${SCRIPT_DIR}/setup_hpc_environment.sh"; then
        log_success "环境脚本配置了 TRANSFORMERS_CACHE"
    else
        log_error "环境脚本未配置 TRANSFORMERS_CACHE"
    fi
fi

# ============================================================
# 生成报告
# ============================================================

echo ""
echo "========================================================================"
echo -e "${BOLD}检查结果汇总${NC}"
echo "========================================================================"
echo ""

if [ ${ERROR_COUNT} -eq 0 ] && [ ${WARNING_COUNT} -eq 0 ]; then
    echo -e "${GREEN}${BOLD}✓ 所有检查通过！可以部署到 HPC${NC}"
    echo ""
    echo "下一步操作："
    echo "  1. 上传代码到 HPC"
    echo "  2. 在 HPC 上运行: bash scripts/check_hpc_models.sh"
    echo "  3. 在计算节点运行: srun --gres=gpu:1 bash scripts/detect_slurm_resources.sh"
    echo "  4. 提交训练作业: sbatch scripts/run_multi_seed_experiments.sh"
    EXIT_CODE=0
elif [ ${ERROR_COUNT} -eq 0 ]; then
    echo -e "${YELLOW}${BOLD}⚠ 检查完成，有 ${WARNING_COUNT} 个警告${NC}"
    echo ""
    echo "建议处理警告后再部署，或确认警告可以忽略"
    EXIT_CODE=0
else
    echo -e "${RED}${BOLD}✗ 检查失败，发现 ${ERROR_COUNT} 个错误和 ${WARNING_COUNT} 个警告${NC}"
    echo ""
    echo "请修复上述错误后再部署"
    EXIT_CODE=1
fi

echo ""
echo "详细信息："
echo "  - 错误: ${ERROR_COUNT}"
echo "  - 警告: ${WARNING_COUNT}"
echo "  - 检查时间: $(date)"
echo ""

# 生成文本报告
REPORT_FILE="${PROJECT_ROOT}/pre_deployment_check_report.txt"
{
    echo "KaVa 项目 - HPC 部署前验证报告"
    echo "================================"
    echo ""
    echo "检查时间: $(date)"
    echo "项目路径: ${PROJECT_ROOT}"
    echo ""
    echo "结果统计:"
    echo "  - 错误: ${ERROR_COUNT}"
    echo "  - 警告: ${WARNING_COUNT}"
    echo ""
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "状态: 通过 ✓"
    else
        echo "状态: 失败 ✗"
    fi
} > "${REPORT_FILE}"

log_info "详细报告已保存到: ${REPORT_FILE}"

echo "========================================================================"

exit ${EXIT_CODE}
