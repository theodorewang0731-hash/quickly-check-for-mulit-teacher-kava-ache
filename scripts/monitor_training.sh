#!/bin/bash

################################################################################
# KaVa 训练任务监控脚本
# 
# 功能：
# - 显示 SLURM 任务状态
# - 显示训练进度和日志
# - 支持自动刷新模式（--auto）
#
# 用法：
#   bash scripts/monitor_training.sh           # 单次查看
#   bash scripts/monitor_training.sh --auto    # 自动刷新（每30秒）
################################################################################

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 检查是否自动刷新
AUTO_REFRESH=false
REFRESH_INTERVAL=30

if [ "$1" == "--auto" ] || [ "$1" == "-a" ]; then
    AUTO_REFRESH=true
fi

# 监控函数
monitor_once() {
    clear
    
    echo -e "${BLUE}========================================================"
    echo "KaVa 多教师蒸馏 - 训练任务监控"
    echo -e "========================================================${NC}"
    echo ""
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # ============================================================
    # 1. SLURM 任务状态
    # ============================================================
    
    echo -e "${CYAN}[1] SLURM 任务队列${NC}"
    echo "----------------------------------------"
    
    if command -v squeue &> /dev/null; then
        JOBS=$(squeue -u $USER 2>/dev/null)
        
        if [ -z "$JOBS" ] || [ $(echo "$JOBS" | wc -l) -eq 1 ]; then
            echo -e "${YELLOW}⚠ 未发现运行中的任务${NC}"
        else
            echo "$JOBS" | head -n 20
            
            # 统计任务状态
            RUNNING=$(echo "$JOBS" | grep -c " R " || true)
            PENDING=$(echo "$JOBS" | grep -c " PD " || true)
            TOTAL=$(($(echo "$JOBS" | wc -l) - 1))
            
            echo ""
            echo -e "${GREEN}运行中: $RUNNING${NC} | ${YELLOW}等待中: $PENDING${NC} | 总计: $TOTAL"
        fi
    else
        echo -e "${RED}✗ squeue 命令不可用${NC}"
    fi
    
    echo ""
    
    # ============================================================
    # 2. 最近的训练日志
    # ============================================================
    
    echo -e "${CYAN}[2] 最近的训练日志（最新 5 行）${NC}"
    echo "----------------------------------------"
    
    if [ -d "logs" ]; then
        LATEST_LOG=$(ls -t logs/*.out 2>/dev/null | head -n 1)
        
        if [ -n "$LATEST_LOG" ]; then
            echo "文件: $LATEST_LOG"
            echo ""
            tail -n 5 "$LATEST_LOG" 2>/dev/null || echo "（无内容）"
        else
            echo -e "${YELLOW}⚠ 未找到日志文件${NC}"
        fi
    else
        echo -e "${RED}✗ logs/ 目录不存在${NC}"
    fi
    
    echo ""
    
    # ============================================================
    # 3. 错误日志检查
    # ============================================================
    
    echo -e "${CYAN}[3] 错误检查${NC}"
    echo "----------------------------------------"
    
    if [ -d "logs" ]; then
        ERROR_COUNT=$(grep -r "Error\|ERROR\|error" logs/*.err 2>/dev/null | wc -l)
        
        if [ $ERROR_COUNT -eq 0 ]; then
            echo -e "${GREEN}✓ 未发现错误${NC}"
        else
            echo -e "${RED}⚠ 发现 $ERROR_COUNT 行错误信息${NC}"
            echo ""
            grep -r "Error\|ERROR\|error" logs/*.err 2>/dev/null | tail -n 3
        fi
    else
        echo -e "${YELLOW}⚠ 无法检查错误日志${NC}"
    fi
    
    echo ""
    
    # ============================================================
    # 4. 检查点和输出
    # ============================================================
    
    echo -e "${CYAN}[4] 训练输出${NC}"
    echo "----------------------------------------"
    
    if [ -d "outputs" ]; then
        CHECKPOINT_COUNT=$(find outputs -name "checkpoint-*" -type d 2>/dev/null | wc -l)
        RESULT_COUNT=$(find outputs -name "*.json" -type f 2>/dev/null | wc -l)
        
        echo "检查点数量: $CHECKPOINT_COUNT"
        echo "结果文件数量: $RESULT_COUNT"
        
        # 显示最新的检查点
        LATEST_CKPT=$(find outputs -name "checkpoint-*" -type d 2>/dev/null | sort | tail -n 1)
        if [ -n "$LATEST_CKPT" ]; then
            echo "最新检查点: $LATEST_CKPT"
        fi
    else
        echo -e "${YELLOW}⚠ outputs/ 目录不存在${NC}"
    fi
    
    echo ""
    
    # ============================================================
    # 5. 任务历史（最近完成的）
    # ============================================================
    
    echo -e "${CYAN}[5] 最近完成的任务${NC}"
    echo "----------------------------------------"
    
    if command -v sacct &> /dev/null; then
        sacct -u $USER -S today --format=JobID,JobName,State,ExitCode,Elapsed -n | tail -n 5
    else
        echo -e "${YELLOW}⚠ sacct 命令不可用${NC}"
    fi
    
    echo ""
    
    # ============================================================
    # 底部提示
    # ============================================================
    
    echo -e "${BLUE}========================================================"
    echo "快捷命令"
    echo -e "========================================================${NC}"
    echo ""
    echo "  查看完整日志:   tail -f logs/<job_id>.out"
    echo "  查看错误日志:   tail -f logs/<job_id>.err"
    echo "  查看任务详情:   scontrol show job <job_id>"
    echo "  取消任务:       scancel <job_id>"
    echo "  取消所有任务:   scancel -u \$USER"
    echo ""
    
    if [ "$AUTO_REFRESH" = true ]; then
        echo -e "${GREEN}自动刷新模式 - ${REFRESH_INTERVAL}秒后刷新${NC}"
        echo "按 Ctrl+C 退出"
    else
        echo "提示: 使用 'bash scripts/monitor_training.sh --auto' 启用自动刷新"
    fi
    
    echo ""
}

# 主循环
if [ "$AUTO_REFRESH" = true ]; then
    while true; do
        monitor_once
        sleep $REFRESH_INTERVAL
    done
else
    monitor_once
fi
