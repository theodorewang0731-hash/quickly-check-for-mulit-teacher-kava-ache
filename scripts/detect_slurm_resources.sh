#!/bin/bash

################################################################################
# SLURM 资源检测与配置建议脚本
#
# 功能:
# 1. 检测 HPC 集群的 GPU 类型和数量
# 2. 检测 CPU 和内存限制
# 3. 提供 SLURM 配置建议
# 4. 验证当前脚本的资源配置是否合理
#
# 用法:
#   bash scripts/detect_slurm_resources.sh
#   或在计算节点上运行以获取实时信息
################################################################################

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo "========================================================"
echo "SLURM 资源检测与配置建议"
echo "========================================================"
echo ""

# ============================================================
# 1. 检测当前环境
# ============================================================

echo -e "${BLUE}[1] 当前环境检测${NC}"
echo ""

# 检测是否在 SLURM 作业中
if [ -n "${SLURM_JOB_ID}" ]; then
    echo -e "  ${GREEN}✓${NC} 正在 SLURM 作业中运行"
    echo "    作业 ID: ${SLURM_JOB_ID}"
    echo "    作业名称: ${SLURM_JOB_NAME}"
    echo "    节点列表: ${SLURM_NODELIST}"
    IN_SLURM_JOB=true
else
    echo -e "  ${YELLOW}○${NC} 不在 SLURM 作业中（在登录节点）"
    echo "    某些检测功能受限"
    IN_SLURM_JOB=false
fi

echo ""

# ============================================================
# 2. GPU 信息检测
# ============================================================

echo -e "${BLUE}[2] GPU 资源检测${NC}"
echo ""

if command -v nvidia-smi &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} nvidia-smi 可用"
    echo ""
    
    # 获取 GPU 信息
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1 | sed 's/ MiB//')
    
    echo "  检测到的 GPU 信息："
    echo "    数量: ${GPU_COUNT}"
    echo "    型号: ${GPU_NAME}"
    echo "    显存: ${GPU_MEMORY} MiB"
    echo ""
    
    # 推断 GPU 格式
    GPU_FORMAT=""
    if [[ "${GPU_NAME}" =~ "A100" ]]; then
        if [[ "${GPU_MEMORY}" -gt 70000 ]]; then
            GPU_FORMAT="gpu:a100-sxm4-80gb"
        else
            GPU_FORMAT="gpu:a100-sxm4-40gb"
        fi
    elif [[ "${GPU_NAME}" =~ "V100" ]]; then
        GPU_FORMAT="gpu:v100"
    elif [[ "${GPU_NAME}" =~ "RTX 3090" ]]; then
        GPU_FORMAT="gpu:rtx3090"
    elif [[ "${GPU_NAME}" =~ "RTX 4090" ]]; then
        GPU_FORMAT="gpu:rtx4090"
    else
        GPU_FORMAT="gpu"
    fi
    
    echo "  推荐的 SLURM GPU 格式："
    echo -e "    ${GREEN}完整格式:${NC} ${GPU_FORMAT}:N  (推荐用于严格 HPC)"
    echo -e "    ${YELLOW}简化格式:${NC} gpu:N           (部分 HPC 支持)"
    echo ""
    
else
    echo -e "  ${RED}✗${NC} nvidia-smi 不可用"
    echo "    可能原因："
    echo "      1. 在登录节点上（登录节点通常没有 GPU）"
    echo "      2. GPU 驱动未安装"
    echo ""
    echo "  建议使用以下命令在计算节点上运行此脚本："
    echo "    srun --partition=gpu --gres=gpu:1 --pty bash scripts/detect_slurm_resources.sh"
    echo ""
fi

# ============================================================
# 3. CPU 和内存检测
# ============================================================

echo -e "${BLUE}[3] CPU 和内存资源检测${NC}"
echo ""

# CPU 信息
CPU_COUNT=$(nproc 2>/dev/null || echo "unknown")
echo "  物理 CPU 核心数: ${CPU_COUNT}"

if [ "${IN_SLURM_JOB}" = true ]; then
    echo "  SLURM 分配的 CPU: ${SLURM_CPUS_PER_TASK:-未设置}"
fi

# 内存信息
if command -v free &> /dev/null; then
    TOTAL_MEM=$(free -h | awk '/^Mem:/ {print $2}')
    AVAIL_MEM=$(free -h | awk '/^Mem:/ {print $7}')
    echo "  总内存: ${TOTAL_MEM}"
    echo "  可用内存: ${AVAIL_MEM}"
    
    if [ "${IN_SLURM_JOB}" = true ]; then
        echo "  SLURM 分配的内存: ${SLURM_MEM_PER_NODE:-未设置}"
    fi
else
    echo "  无法获取内存信息（free 命令不可用）"
fi

echo ""

# ============================================================
# 4. SLURM 分区信息
# ============================================================

echo -e "${BLUE}[4] SLURM 分区配置${NC}"
echo ""

if command -v sinfo &> /dev/null; then
    echo "  可用分区："
    sinfo -o "%P %G %C %m" | head -n 10
    echo ""
    
    echo "  GPU 分区详细信息："
    sinfo -p gpu -o "%P %G %c %m %l %N" 2>/dev/null || echo "    无法获取 GPU 分区信息"
else
    echo -e "  ${RED}✗${NC} sinfo 命令不可用"
fi

echo ""

# ============================================================
# 5. 资源配置建议
# ============================================================

echo "========================================================"
echo "资源配置建议"
echo "========================================================"
echo ""

echo -e "${CYAN}[推荐配置 - 单节点 8-GPU 训练]${NC}"
echo ""
echo "  #SBATCH --partition=gpu"
echo "  #SBATCH --nodes=1"
echo "  #SBATCH --ntasks-per-node=1"

if [ -n "${GPU_FORMAT}" ] && [ "${GPU_FORMAT}" != "gpu" ]; then
    echo -e "  ${GREEN}#SBATCH --gres=${GPU_FORMAT}:8${NC}  # 完整格式（推荐）"
    echo -e "  ${YELLOW}# 或简化格式（某些 HPC 支持）:${NC}"
    echo "  # #SBATCH --gres=gpu:8"
else
    echo "  #SBATCH --gres=gpu:8"
fi

echo ""
echo "  CPU 配置（根据 GPU 数量）："
echo "    - 保守配置: --cpus-per-task=32  (4 核/GPU)"
echo "    - 标准配置: --cpus-per-task=48  (6 核/GPU)"
echo -e "    - ${YELLOW}激进配置: --cpus-per-task=64  (8 核/GPU，可能超限)${NC}"
echo ""
echo "  内存配置："
echo "    - 使用全部节点内存: --mem=0"
echo "    - 或指定具体大小: --mem=512G / --mem=1T"
echo ""
echo "  时间限制:"
echo "    - 短期测试: --time=4:00:00   (4 小时)"
echo "    - 标准训练: --time=48:00:00  (2 天)"
echo "    - 长期训练: --time=120:00:00 (5 天)"
echo ""

# ============================================================
# 6. 检查现有脚本
# ============================================================

echo "========================================================"
echo "检查现有 SLURM 脚本配置"
echo "========================================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_SCRIPTS=(
    "${SCRIPT_DIR}/run_multi_seed_experiments.sh"
    "${SCRIPT_DIR}/run_ablation_studies.sh"
    "${SCRIPT_DIR}/run_three_stage_routing.sh"
)

for script in "${SLURM_SCRIPTS[@]}"; do
    if [ -f "${script}" ]; then
        script_name=$(basename "${script}")
        echo -e "${CYAN}检查: ${script_name}${NC}"
        echo ""
        
        # 提取 SBATCH 配置
        gres=$(grep "^#SBATCH --gres=" "${script}" | head -n 1 | sed 's/.*--gres=//' | sed 's/#.*//' | xargs)
        cpus=$(grep "^#SBATCH --cpus-per-task=" "${script}" | head -n 1 | sed 's/.*--cpus-per-task=//' | sed 's/#.*//' | xargs)
        mem=$(grep "^#SBATCH --mem=" "${script}" | head -n 1 | sed 's/.*--mem=//' | sed 's/#.*//' | xargs)
        
        echo "  当前配置:"
        echo "    GPU: ${gres:-未设置}"
        echo "    CPU: ${cpus:-未设置}"
        echo "    内存: ${mem:-未设置}"
        
        # 给出建议
        needs_fix=false
        
        if [ -n "${GPU_FORMAT}" ] && [ "${GPU_FORMAT}" != "gpu" ]; then
            if [[ "${gres}" != *"${GPU_FORMAT}"* ]] && [[ "${gres}" != "gpu:"* ]]; then
                echo -e "    ${YELLOW}⚠ 建议使用完整 GPU 格式: ${GPU_FORMAT}:N${NC}"
                needs_fix=true
            fi
        fi
        
        if [ -n "${cpus}" ] && [ "${cpus}" -gt 64 ]; then
            echo -e "    ${YELLOW}⚠ CPU 核心数可能过高，建议 32-48${NC}"
            needs_fix=true
        fi
        
        if [ "${needs_fix}" = false ]; then
            echo -e "    ${GREEN}✓ 配置看起来合理${NC}"
        fi
        
        echo ""
    fi
done

# ============================================================
# 7. 实战测试建议
# ============================================================

echo "========================================================"
echo "实战测试建议"
echo "========================================================"
echo ""

echo "1. 快速测试（验证配置是否能通过 SLURM）："
echo ""
echo "   srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=32G \\"
echo "        --time=10:00 --pty bash -c 'nvidia-smi && echo 测试成功'"
echo ""

echo "2. 完整训练提交前检查："
echo ""
echo "   # 验证环境"
echo "   bash scripts/pre_training_validation.py"
echo ""
echo "   # 检查模型"
echo "   bash scripts/check_hpc_models.sh"
echo ""
echo "   # 提交作业"
echo "   sbatch scripts/run_multi_seed_experiments.sh"
echo ""

echo "3. 监控作业状态："
echo ""
echo "   # 查看队列"
echo "   squeue -u \$USER"
echo ""
echo "   # 查看作业详情"
echo "   scontrol show job <JOB_ID>"
echo ""
echo "   # 实时监控日志"
echo "   bash scripts/monitor_training.sh --auto"
echo ""

echo "========================================================"
echo "检测完成"
echo "========================================================"
