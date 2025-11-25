#!/bin/bash

################################################################################
# HPC 共享模型库检查脚本
#
# 功能:
# 1. 检查 HPC 共享模型库是否存在
# 2. 列出所有可用的模型
# 3. 验证项目需要的模型是否都存在
# 4. 提供模型路径映射信息
#
# 用法:
#   bash scripts/check_hpc_models.sh
################################################################################

set -e

# ============================================================
# 配置区域
# ============================================================

# HPC 共享模型库路径
SHARED_MODEL_DIR="/home/share/models"

# 项目需要的模型列表
REQUIRED_MODELS=(
    "Qwen/Qwen2.5-1.5B"
    "Qwen/Qwen2.5-7B"
    "Qwen/Qwen2.5-14B"
    "Qwen/Qwen2.5-0.5B"  # 可选：用于快速测试
    "meta-llama/Llama-3.2-1B"  # 可选：对比实验
    "meta-llama/Llama-3.2-3B"  # 可选：对比实验
)

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================
# 检查函数
# ============================================================

echo "========================================================"
echo "HPC 共享模型库检查"
echo "========================================================"
echo ""

# 1. 检查共享模型库目录是否存在
echo -e "${BLUE}[1] 检查共享模型库目录${NC}"
if [ -d "${SHARED_MODEL_DIR}" ]; then
    echo -e "  ${GREEN}✓${NC} 共享模型库存在: ${SHARED_MODEL_DIR}"
    
    # 检查权限
    if [ -r "${SHARED_MODEL_DIR}" ]; then
        echo -e "  ${GREEN}✓${NC} 具有读取权限"
    else
        echo -e "  ${RED}✗${NC} 没有读取权限"
        exit 1
    fi
else
    echo -e "  ${RED}✗${NC} 共享模型库不存在: ${SHARED_MODEL_DIR}"
    echo ""
    echo "请联系 HPC 管理员，或使用以下备选方案："
    echo "  1. 使用用户缓存目录: \${HOME}/.cache/huggingface"
    echo "  2. 手动下载模型到指定目录"
    exit 1
fi

echo ""

# 2. 列出所有可用模型
echo -e "${BLUE}[2] 扫描可用模型${NC}"
echo ""

# 查找所有模型目录（假设模型目录包含 config.json）
AVAILABLE_MODELS=()
while IFS= read -r -d '' model_dir; do
    # 提取相对路径（去掉 SHARED_MODEL_DIR 前缀）
    relative_path="${model_dir#${SHARED_MODEL_DIR}/}"
    # 去掉末尾的 /config.json
    model_name="${relative_path%/config.json}"
    AVAILABLE_MODELS+=("${model_name}")
done < <(find "${SHARED_MODEL_DIR}" -name "config.json" -print0 2>/dev/null)

if [ ${#AVAILABLE_MODELS[@]} -eq 0 ]; then
    echo -e "  ${YELLOW}⚠${NC} 未找到任何模型（没有 config.json 文件）"
    echo ""
    echo "可能的原因："
    echo "  1. 模型尚未下载到共享库"
    echo "  2. 目录结构不符合 HuggingFace 标准"
    echo ""
else
    echo "找到 ${#AVAILABLE_MODELS[@]} 个可用模型："
    echo ""
    for model in "${AVAILABLE_MODELS[@]}"; do
        echo -e "  ${GREEN}✓${NC} ${model}"
    done
fi

echo ""

# 3. 检查项目需要的模型
echo -e "${BLUE}[3] 验证项目所需模型${NC}"
echo ""

MISSING_MODELS=()
OPTIONAL_MISSING=()

for model in "${REQUIRED_MODELS[@]}"; do
    # 构建完整路径
    model_path="${SHARED_MODEL_DIR}/${model}"
    
    if [ -d "${model_path}" ]; then
        # 检查是否有 config.json
        if [ -f "${model_path}/config.json" ]; then
            echo -e "  ${GREEN}✓${NC} ${model}"
        else
            echo -e "  ${YELLOW}⚠${NC} ${model} (目录存在但缺少 config.json)"
            MISSING_MODELS+=("${model}")
        fi
    else
        # 判断是否为可选模型
        if [[ "${model}" == *"Llama"* ]] || [[ "${model}" == *"0.5B"* ]]; then
            echo -e "  ${YELLOW}○${NC} ${model} (可选，未找到)"
            OPTIONAL_MISSING+=("${model}")
        else
            echo -e "  ${RED}✗${NC} ${model} (必需，未找到)"
            MISSING_MODELS+=("${model}")
        fi
    fi
done

echo ""

# 4. 生成报告
echo "========================================================"
echo "检查结果汇总"
echo "========================================================"
echo ""

if [ ${#MISSING_MODELS[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ 所有必需模型都已就绪！${NC}"
    echo ""
    echo "环境变量配置："
    echo "  export HF_HOME=\"${SHARED_MODEL_DIR}\""
    echo "  export TRANSFORMERS_CACHE=\"${SHARED_MODEL_DIR}\""
    echo "  export HF_DATASETS_CACHE=\"\${HOME}/.cache/huggingface/datasets\""
    echo ""
    echo "这些环境变量已在 scripts/setup_hpc_environment.sh 中自动配置"
else
    echo -e "${RED}✗ 缺少 ${#MISSING_MODELS[@]} 个必需模型${NC}"
    echo ""
    echo "缺少的模型："
    for model in "${MISSING_MODELS[@]}"; do
        echo "  - ${model}"
    done
    echo ""
    echo "解决方案："
    echo "  1. 联系 HPC 管理员下载缺失的模型"
    echo "  2. 或者使用以下命令手动下载："
    echo ""
    for model in "${MISSING_MODELS[@]}"; do
        echo "     huggingface-cli download ${model} --local-dir ${SHARED_MODEL_DIR}/${model}"
    done
    echo ""
fi

if [ ${#OPTIONAL_MISSING[@]} -gt 0 ]; then
    echo -e "${YELLOW}可选模型未找到（不影响核心实验）:${NC}"
    for model in "${OPTIONAL_MISSING[@]}"; do
        echo "  - ${model}"
    done
    echo ""
fi

# 5. 模型大小统计（如果有 du 命令）
echo "========================================================"
echo "模型存储信息"
echo "========================================================"
echo ""

if command -v du &> /dev/null; then
    echo "各模型占用空间："
    echo ""
    for model in "${REQUIRED_MODELS[@]}"; do
        model_path="${SHARED_MODEL_DIR}/${model}"
        if [ -d "${model_path}" ]; then
            size=$(du -sh "${model_path}" 2>/dev/null | cut -f1)
            echo "  ${model}: ${size}"
        fi
    done
    echo ""
    
    total_size=$(du -sh "${SHARED_MODEL_DIR}" 2>/dev/null | cut -f1)
    echo "共享模型库总大小: ${total_size}"
else
    echo "无法获取存储信息（du 命令不可用）"
fi

echo ""
echo "========================================================"
echo "检查完成"
echo "========================================================"

# 返回状态
if [ ${#MISSING_MODELS[@]} -eq 0 ]; then
    exit 0
else
    exit 1
fi
