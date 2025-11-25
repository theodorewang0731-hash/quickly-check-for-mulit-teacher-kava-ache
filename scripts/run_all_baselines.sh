#!/bin/bash
#SBATCH --job-name=run_baselines
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4              # 根据 HPC 要求可能需要完整格式，如 gpu:a100-sxm4-80gb:4
#SBATCH --mem=256G                # 根据 HPC 限制可能需要调整
#SBATCH --time=48:00:00
#SBATCH --output=logs/baselines_%j.log
#SBATCH --error=logs/baselines_%j.err

# ============================================================================
# 运行所有基线实验
# 用于建立对照组，评估多教师 KV 蒸馏的提升
# ============================================================================

# 使用统一的环境配置脚本（自动配置共享模型库）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/setup_hpc_environment.sh"

BASELINE_DIR="./baselines"
mkdir -p "$BASELINE_DIR"
mkdir -p logs

echo "========================================================================"
echo "Running All Baseline Experiments"
echo "========================================================================"
echo "Output directory: $BASELINE_DIR"
echo "Start time: $(date)"
echo "========================================================================"

# ------------------------ 配置 ------------------------
STUDENT_MODEL="Qwen/Qwen2.5-1.5B"
TEACHER_MODEL="Qwen/Qwen2.5-7B"  # 单教师对照
DATASET="multi_reasoning_cot_direct"
TRAIN_SAMPLES=15000
VAL_SAMPLES=2000
EVAL_DATASETS="gsm8k_test math500 bbh gpqa truthfulqa cmmlu_subset ceval_subset"

# ============================================================================
# Baseline 1: 原始学生模型（无训练）
# ============================================================================
echo ""
echo "========================================================================"
echo "Baseline 1: Raw Student Model (No Training)"
echo "========================================================================"

python evaluation/multi_task_eval.py \
    --model_path "$STUDENT_MODEL" \
    --eval_datasets $EVAL_DATASETS \
    --output_file "$BASELINE_DIR/baseline1_raw_student.json" \
    --device cuda

if [ $? -eq 0 ]; then
    echo "✓ Baseline 1 completed"
else
    echo "✗ Baseline 1 failed"
fi

# ============================================================================
# Baseline 2: 标准监督微调（无 KV 蒸馏）
# ============================================================================
echo ""
echo "========================================================================"
echo "Baseline 2: Standard Supervised Fine-Tuning (No KV Distillation)"
echo "========================================================================"

python experiments/train_standard_sft.py \
    --model_name_or_path "$STUDENT_MODEL" \
    --dataset_name "$DATASET" \
    --train_samples $TRAIN_SAMPLES \
    --val_samples $VAL_SAMPLES \
    --output_dir "$BASELINE_DIR/baseline2_standard_sft" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --bf16 true \
    --gradient_checkpointing true \
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --save_steps 1000 \
    --logging_steps 50 \
    --save_total_limit 2 \
    --load_best_model_at_end true \
    --report_to tensorboard

if [ $? -eq 0 ]; then
    echo "✓ Baseline 2 training completed"
    
    # 评测
    python evaluation/multi_task_eval.py \
        --model_path "$BASELINE_DIR/baseline2_standard_sft/best_model" \
        --eval_datasets $EVAL_DATASETS \
        --output_file "$BASELINE_DIR/baseline2_standard_sft/eval_results.json"
    
    echo "✓ Baseline 2 evaluation completed"
else
    echo "✗ Baseline 2 failed"
fi

# ============================================================================
# Baseline 3: 单教师 KV 蒸馏
# ============================================================================
echo ""
echo "========================================================================"
echo "Baseline 3: Single-Teacher KV Distillation"
echo "========================================================================"

python experiments/train_with_kv.py \
    --student_model_name_or_path "$STUDENT_MODEL" \
    --teacher_model_name_or_path "$TEACHER_MODEL" \
    --dataset_name "$DATASET" \
    --train_samples $TRAIN_SAMPLES \
    --val_samples $VAL_SAMPLES \
    --output_dir "$BASELINE_DIR/baseline3_single_teacher_kv" \
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
    --evaluation_strategy "steps" \
    --eval_steps 500 \
    --save_steps 1000 \
    --logging_steps 50 \
    --save_total_limit 2 \
    --load_best_model_at_end true \
    --report_to tensorboard

if [ $? -eq 0 ]; then
    echo "✓ Baseline 3 training completed"
    
    # 评测
    python evaluation/multi_task_eval.py \
        --model_path "$BASELINE_DIR/baseline3_single_teacher_kv/best_model" \
        --eval_datasets $EVAL_DATASETS \
        --output_file "$BASELINE_DIR/baseline3_single_teacher_kv/eval_results.json"
    
    echo "✓ Baseline 3 evaluation completed"
else
    echo "✗ Baseline 3 failed"
fi

# ============================================================================
# 生成基线对比报告
# ============================================================================
echo ""
echo "========================================================================"
echo "Generating Baseline Comparison Report"
echo "========================================================================"

python -c "
import json
from pathlib import Path
import sys

baseline_dir = Path('$BASELINE_DIR')
results = {}

# 加载所有基线结果
for json_file in baseline_dir.glob('**/eval_results.json'):
    baseline_name = json_file.parent.name if json_file.parent != baseline_dir else json_file.stem
    with open(json_file) as f:
        results[baseline_name] = json.load(f)

# 直接评测的结果
if (baseline_dir / 'baseline1_raw_student.json').exists():
    with open(baseline_dir / 'baseline1_raw_student.json') as f:
        results['baseline1_raw_student'] = json.load(f)

# 打印对比
print('\n' + '='*80)
print('Baseline Results Comparison')
print('='*80)
print(f'{'Dataset':<20} {'Raw Student':<15} {'Standard SFT':<15} {'Single Teacher':<15}')
print('-'*80)

# 获取所有数据集
datasets = set()
for result in results.values():
    datasets.update(k for k in result.keys() if k != 'average')

for dataset in sorted(datasets):
    row = [dataset[:18]]
    
    for baseline in ['baseline1_raw_student', 'baseline2_standard_sft', 'baseline3_single_teacher_kv']:
        if baseline in results and dataset in results[baseline]:
            score = results[baseline][dataset].get('score', 0)
            row.append(f'{score:.2f}%')
        else:
            row.append('N/A')
    
    print(f'{row[0]:<20} {row[1]:<15} {row[2]:<15} {row[3]:<15}')

print('-'*80)

# 平均分
avg_row = ['Average']
for baseline in ['baseline1_raw_student', 'baseline2_standard_sft', 'baseline3_single_teacher_kv']:
    if baseline in results and 'average' in results[baseline]:
        avg_row.append(f\"{results[baseline]['average']:.2f}%\")
    else:
        avg_row.append('N/A')

print(f'{avg_row[0]:<20} {avg_row[1]:<15} {avg_row[2]:<15} {avg_row[3]:<15}')
print('='*80)

# 保存到文件
summary = {
    'baselines': results,
    'summary': {
        'raw_student': results.get('baseline1_raw_student', {}).get('average', 0),
        'standard_sft': results.get('baseline2_standard_sft', {}).get('average', 0),
        'single_teacher': results.get('baseline3_single_teacher_kv', {}).get('average', 0),
    }
}

with open(baseline_dir / 'baseline_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f'\n✓ Summary saved to: {baseline_dir / \"baseline_summary.json\"}')
"

# 生成可视化
if [ -f "$BASELINE_DIR/baseline_summary.json" ]; then
    python visualization/hpc_visualizer.py \
        --mode eval \
        --input \
            "$BASELINE_DIR/baseline1_raw_student.json" \
            "$BASELINE_DIR/baseline2_standard_sft/eval_results.json" \
            "$BASELINE_DIR/baseline3_single_teacher_kv/eval_results.json" \
        --labels "Raw Student" "Standard SFT" "Single Teacher KV" \
        --output_dir "$BASELINE_DIR/visualizations" \
        --output_name "baseline_comparison"
    
    echo "✓ Visualization generated: $BASELINE_DIR/visualizations/baseline_comparison.html"
fi

# ============================================================================
# 完成
# ============================================================================
echo ""
echo "========================================================================"
echo "All Baselines Completed!"
echo "========================================================================"
echo "End time: $(date)"
echo ""
echo "Results saved in: $BASELINE_DIR/"
echo ""
echo "Next steps:"
echo "  1. Review baseline results: cat $BASELINE_DIR/baseline_summary.json"
echo "  2. Download visualization: scp user@hpc:$BASELINE_DIR/visualizations/baseline_comparison.html ~/"
echo "  3. Run multi-teacher experiments: sbatch scripts/run_three_stage_routing.sh"
echo ""
echo "Expected baseline performance:"
echo "  Raw Student:     35-45% average"
echo "  Standard SFT:    45-55% average"
echo "  Single Teacher:  52-58% average"
echo ""
echo "Multi-teacher target: 58-63% average (+5-10% improvement)"
echo "========================================================================"
