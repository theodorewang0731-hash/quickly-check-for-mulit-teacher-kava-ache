#!/bin/bash
#SBATCH --job-name=kv_distill_with_viz
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:8              # 根据 HPC 要求可能需要完整格式，如 gpu:a100-sxm4-80gb:8
#SBATCH --mem=500G                # 根据 HPC 限制可能需要调整
#SBATCH --time=72:00:00
#SBATCH --output=logs/training_with_viz_%j.log
#SBATCH --error=logs/training_with_viz_%j.err

# ============================================================================
# 训练 + 自动可视化包装脚本
# 训练完成后自动生成 HTML 可视化报告
# ============================================================================

# 使用统一的环境配置脚本（自动配置共享模型库）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_hpc_environment.sh"

# ------------------------ 配置 ------------------------
STUDENT_MODEL="${STUDENT:-Qwen/Qwen2.5-1.5B}"
TEACHER_MODELS="${TEACHERS:-Qwen/Qwen2.5-7B Qwen/Qwen2.5-14B}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/experiment_$(date +%Y%m%d_%H%M%S)}"
ENABLE_VISUALIZATION=true  # 是否启用可视化

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/visualizations"
mkdir -p logs

echo "========================================================================"
echo "Training with Automatic Visualization"
echo "========================================================================"
echo "Student: $STUDENT_MODEL"
echo "Teachers: $TEACHER_MODELS"
echo "Output: $OUTPUT_DIR"
echo "Visualization: $ENABLE_VISUALIZATION"
echo "========================================================================"

# ============================================================================
# Step 1: 运行训练
# ============================================================================
echo "Step 1: Starting training..."

python experiments/train_multi_teacher_kv.py \
    --student_model_name_or_path "$STUDENT_MODEL" \
    --teacher_models $TEACHER_MODELS \
    --dataset_name "multi_reasoning_cot_direct" \
    --train_samples 15000 \
    --val_samples 2000 \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --bf16 true \
    --gradient_checkpointing true \
    --kv_compression "right" \
    --kv_loss_type "smooth_l1" \
    --kv_loss_weight 0.1 \
    --fusion_strategy "fixed" \
    --fixed_weights "0.5,0.5" \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --save_steps 1000 \
    --logging_steps 50 \
    --save_total_limit 3 \
    --load_best_model_at_end true \
    --metric_for_best_model "eval_kv_loss" \
    --report_to tensorboard \
    --logging_dir "$OUTPUT_DIR/logs" 2>&1 | tee "$OUTPUT_DIR/training.log"

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
    echo "✗ Training failed with exit code $TRAIN_EXIT_CODE"
    exit $TRAIN_EXIT_CODE
fi

echo "✓ Training completed successfully"

# ============================================================================
# Step 2: 运行评测
# ============================================================================
echo ""
echo "Step 2: Running evaluation..."

python evaluation/multi_task_eval.py \
    --model_path "$OUTPUT_DIR/best_model" \
    --eval_datasets gsm8k_test math500 bbh gpqa truthfulqa cmmlu_subset ceval_subset \
    --output_file "$OUTPUT_DIR/eval_results.json" \
    --device cuda 2>&1 | tee "$OUTPUT_DIR/evaluation.log"

EVAL_EXIT_CODE=${PIPESTATUS[0]}

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "✓ Evaluation completed successfully"
else
    echo "✗ Evaluation failed (exit code $EVAL_EXIT_CODE), continuing..."
fi

# ============================================================================
# Step 3: 生成可视化
# ============================================================================
if [ "$ENABLE_VISUALIZATION" = true ]; then
    echo ""
    echo "========================================================================"
    echo "Step 3: Generating Visualizations"
    echo "========================================================================"
    
    # 导出 TensorBoard 日志为 JSON（如果需要）
    echo "Exporting training logs..."
    python -c "
import json
import re
from pathlib import Path

# 解析训练日志
log_file = Path('$OUTPUT_DIR/training.log')
if not log_file.exists():
    print('✗ Training log not found')
    exit(1)

logs = {
    'step': [],
    'train_loss': [],
    'eval_loss': [],
    'kv_loss': [],
    'learning_rate': [],
    'grad_norm': [],
}

with open(log_file) as f:
    for line in f:
        # 解析日志行（根据实际格式调整）
        if 'loss' in line.lower():
            # 示例：提取训练步数和损失
            step_match = re.search(r'step[:\s]+(\d+)', line)
            loss_match = re.search(r'loss[:\s]+([\d.]+)', line)
            
            if step_match and loss_match:
                logs['step'].append(int(step_match.group(1)))
                logs['train_loss'].append(float(loss_match.group(1)))

# 保存为 JSON
output_path = Path('$OUTPUT_DIR/training_log.json')
with open(output_path, 'w') as f:
    json.dump(logs, f, indent=2)

print(f'✓ Training logs exported to: {output_path}')
" || echo "⚠ Log export failed, will try to visualize anyway"
    
    # 1. 训练曲线可视化
    echo "Creating training curves..."
    python visualization/hpc_visualizer.py \
        --mode training \
        --input "$OUTPUT_DIR/training_log.json" \
        --output_dir "$OUTPUT_DIR/visualizations" \
        --output_name "training_curves" || echo "⚠ Training curves visualization failed"
    
    # 2. 评测结果可视化（如果有多个模型对比）
    if [ -f "$OUTPUT_DIR/eval_results.json" ]; then
        echo "Creating evaluation visualization..."
        python visualization/hpc_visualizer.py \
            --mode eval \
            --input "$OUTPUT_DIR/eval_results.json" \
            --labels "Trained Model" \
            --output_dir "$OUTPUT_DIR/visualizations" \
            --output_name "evaluation_results" || echo "⚠ Evaluation visualization failed"
    fi
    
    # 3. 路由权重可视化（如果使用了可学习路由）
    if [ -f "$OUTPUT_DIR/routing_weights.json" ]; then
        echo "Creating routing weights visualization..."
        python visualization/hpc_visualizer.py \
            --mode routing \
            --input "$OUTPUT_DIR/routing_weights.json" \
            --output_dir "$OUTPUT_DIR/visualizations" \
            --output_name "routing_weights" || echo "⚠ Routing visualization failed"
    fi
    
    # 4. 生成综合报告
    echo "Creating comprehensive experiment summary..."
    python visualization/hpc_visualizer.py \
        --mode summary \
        --input "$OUTPUT_DIR" \
        --output_dir "$OUTPUT_DIR/visualizations" \
        --output_name "experiment_summary" || echo "⚠ Summary creation failed"
    
    echo "========================================================================"
    echo "✓ Visualization Complete!"
    echo "========================================================================"
    
    # 获取主 HTML 文件的绝对路径
    SUMMARY_HTML="$OUTPUT_DIR/visualizations/experiment_summary.html"
    
    if [ -f "$SUMMARY_HTML" ]; then
        echo ""
        echo "📊 Main Report Generated:"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "$SUMMARY_HTML"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "📥 To download and open on your local machine:"
        echo ""
        echo "  Step 1: Copy this command (replace YOUR_USERNAME):"
        echo "  ┌────────────────────────────────────────────────────────────────┐"
        echo "  │ scp YOUR_USERNAME@hpc_address:$SUMMARY_HTML ~/Downloads/report.html │"
        echo "  └────────────────────────────────────────────────────────────────┘"
        echo ""
        echo "  Step 2: Run on your local machine, then open:"
        echo "  ┌────────────────────────────────────────────────────────────────┐"
        echo "  │ open ~/Downloads/report.html          # macOS                  │"
        echo "  │ start ~/Downloads/report.html         # Windows                │"
        echo "  │ xdg-open ~/Downloads/report.html      # Linux                  │"
        echo "  └────────────────────────────────────────────────────────────────┘"
        echo ""
        echo "💡 This HTML file is self-contained (images embedded)!"
        echo "   You can copy it anywhere and it will work."
        echo ""
        
        # 额外生成一个下载脚本
        cat > "$OUTPUT_DIR/download_report.sh" << DOWNLOAD_EOF
#!/bin/bash
# Quick download script - Run this on your LOCAL machine

echo "Downloading experiment report..."
scp $USER@\$(hostname):$SUMMARY_HTML ~/Downloads/experiment_report_\$(date +%Y%m%d_%H%M%S).html

if [ \$? -eq 0 ]; then
    echo "✓ Downloaded successfully!"
    echo "Opening report..."
    
    # Auto-open based on OS
    if [[ "\$OSTYPE" == "darwin"* ]]; then
        open ~/Downloads/experiment_report_*.html
    elif [[ "\$OSTYPE" == "linux-gnu"* ]]; then
        xdg-open ~/Downloads/experiment_report_*.html
    elif [[ "\$OSTYPE" == "msys" ]] || [[ "\$OSTYPE" == "win32" ]]; then
        start ~/Downloads/experiment_report_*.html
    fi
else
    echo "✗ Download failed. Please check your connection."
fi
DOWNLOAD_EOF
        chmod +x "$OUTPUT_DIR/download_report.sh"
        
        echo "  Alternative: Use the auto-download script"
        echo "  ┌────────────────────────────────────────────────────────────────┐"
        echo "  │ scp YOUR_USERNAME@hpc:$OUTPUT_DIR/download_report.sh ~/        │"
        echo "  │ bash ~/download_report.sh                                      │"
        echo "  └────────────────────────────────────────────────────────────────┘"
        echo ""
    else
        echo "⚠ Main report not found at: $SUMMARY_HTML"
    fi
    
    echo "All HTML files in this experiment:"
    ls -lh "$OUTPUT_DIR/visualizations"/*.html 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
    echo "========================================================================"
fi

# ============================================================================
# Step 4: 创建快速查看脚本
# ============================================================================
cat > "$OUTPUT_DIR/view_results.sh" << 'EOF'
#!/bin/bash
# 快速查看实验结果

echo "=========================================="
echo "Experiment Results Summary"
echo "=========================================="

# 显示评测结果
if [ -f eval_results.json ]; then
    echo ""
    echo "Evaluation Results:"
    python -c "
import json
with open('eval_results.json') as f:
    results = json.load(f)
for dataset, result in results.items():
    if dataset != 'average':
        print(f'  {dataset:20s}: {result.get(\"score\", 0):.2f}%')
if 'average' in results:
    print(f'  {\"Average\":20s}: {results[\"average\"]:.2f}%')
    "
fi

# 显示可视化文件
echo ""
echo "Visualization Files:"
if [ -d visualizations ]; then
    ls -lh visualizations/*.html 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
    echo ""
    echo "To view HTML files:"
    echo "  1. Download to local: scp -r $USER@hpc:$(pwd)/visualizations ~/Downloads/"
    echo "  2. Open in browser: open ~/Downloads/visualizations/experiment_summary.html"
else
    echo "  No visualizations found"
fi

echo "=========================================="
EOF

chmod +x "$OUTPUT_DIR/view_results.sh"

# ============================================================================
# 完成
# ============================================================================
echo ""
echo "========================================================================"
echo "All Steps Completed!"
echo "========================================================================"

# 使用 Python 脚本显示下载信息
python visualization/show_report_info.py "$OUTPUT_DIR" --create-script

echo ""
echo "Training directory: $OUTPUT_DIR"
echo ""
echo "Quick commands:"
echo "  View results:  cd $OUTPUT_DIR && cat eval_results.json | python -m json.tool"
echo "  Check logs:    less $OUTPUT_DIR/training.log"
echo "========================================================================"
