# 对比实验：Baseline vs Alignment v2 (PowerShell 版本)
# Comparison: Baseline vs Alignment v2

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "Alignment v2 Comparison Experiments" -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "实验组："
Write-Host "  1. Baseline: 硬 index 对齐 + 等比例层映射"
Write-Host "  2. +CKA Layer: 硬 index 对齐 + CKA 层映射"
Write-Host "  3. +Segment Time: Segment 重采样 + 等比例层映射"
Write-Host "  4. Alignment v2 (Full): Segment 重采样 + CKA 层映射"
Write-Host ""
Write-Host "==============================================================================" -ForegroundColor Cyan

# Configuration
$MODEL_NAME = "Qwen/Qwen2-1.5B"
$TEACHER_MODEL = "Qwen/Qwen2-7B"
$DATASET = "openai/gsm8k"
$SUBSET_SIZE = 5000
$EPOCHS = 2
$BATCH_SIZE = 8
$LAYER_MAPPING = "layer_mapping_qwen15b_7b.json"

# Check if layer mapping exists
if (-not (Test-Path $LAYER_MAPPING)) {
    Write-Host ""
    Write-Host "⚠️  Layer mapping not found: $LAYER_MAPPING" -ForegroundColor Yellow
    Write-Host "   Running precomputation first..."
    Write-Host ""
    
    python experiments/precompute_layer_mapping.py `
        --student_model "$MODEL_NAME" `
        --teacher_model "$TEACHER_MODEL" `
        --dataset_name "$DATASET" `
        --num_samples 100 `
        --output "$LAYER_MAPPING"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Precomputation failed!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host ""
    Write-Host "✓ Layer mapping precomputed" -ForegroundColor Green
    Write-Host ""
}

# Experiment 1: Baseline
Write-Host ""
Write-Host "[Experiment 1/4] 📊 Baseline (硬 index 对齐 + 等比例层映射)" -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "输出目录: outputs/alignment_baseline"
Write-Host ""
Read-Host "按 Enter 运行，或 Ctrl+C 跳过..."

python experiments/train_with_kv.py `
    --model_name "$MODEL_NAME" `
    --teacher_model "$TEACHER_MODEL" `
    --dataset_name "$DATASET" `
    --subset_size $SUBSET_SIZE `
    --epochs $EPOCHS `
    --batch_size $BATCH_SIZE `
    --kv_weight 1.0 `
    --codi_weight 0.5 `
    --fp16 `
    --output_dir outputs/alignment_baseline

# Experiment 2: +CKA Layer
Write-Host ""
Write-Host "[Experiment 2/4] 🔬 +CKA Layer (硬 index 对齐 + CKA 层映射)" -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "输出目录: outputs/alignment_cka_layer"
Write-Host ""
Read-Host "按 Enter 运行，或 Ctrl+C 跳过..."

python experiments/train_with_kv.py `
    --model_name "$MODEL_NAME" `
    --teacher_model "$TEACHER_MODEL" `
    --dataset_name "$DATASET" `
    --subset_size $SUBSET_SIZE `
    --epochs $EPOCHS `
    --batch_size $BATCH_SIZE `
    --kv_weight 1.0 `
    --codi_weight 0.5 `
    --use_cka_layer_mapping `
    --layer_mapping_path "$LAYER_MAPPING" `
    --fp16 `
    --output_dir outputs/alignment_cka_layer

# Experiment 3: +Segment Time
Write-Host ""
Write-Host "[Experiment 3/4] ⏱️  +Segment Time (Segment 重采样 + 等比例层映射)" -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "输出目录: outputs/alignment_segment_time"
Write-Host ""
Read-Host "按 Enter 运行，或 Ctrl+C 跳过..."

python experiments/train_with_kv.py `
    --model_name "$MODEL_NAME" `
    --teacher_model "$TEACHER_MODEL" `
    --dataset_name "$DATASET" `
    --subset_size $SUBSET_SIZE `
    --epochs $EPOCHS `
    --batch_size $BATCH_SIZE `
    --kv_weight 1.0 `
    --codi_weight 0.5 `
    --use_segment_resampling `
    --fp16 `
    --output_dir outputs/alignment_segment_time

# Experiment 4: Alignment v2 (Full)
Write-Host ""
Write-Host "[Experiment 4/4] 🚀 Alignment v2 (Segment 重采样 + CKA 层映射)" -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "输出目录: outputs/alignment_v2_full"
Write-Host ""
Read-Host "按 Enter 运行，或 Ctrl+C 跳过..."

python experiments/train_with_kv.py `
    --model_name "$MODEL_NAME" `
    --teacher_model "$TEACHER_MODEL" `
    --dataset_name "$DATASET" `
    --subset_size $SUBSET_SIZE `
    --epochs $EPOCHS `
    --batch_size $BATCH_SIZE `
    --kv_weight 1.0 `
    --codi_weight 0.5 `
    --use_cka_layer_mapping `
    --layer_mapping_path "$LAYER_MAPPING" `
    --use_segment_resampling `
    --fp16 `
    --output_dir outputs/alignment_v2_full

# Summary
Write-Host ""
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "✓ 所有实验完成" -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "结果目录："
Write-Host "  1. outputs/alignment_baseline        - Baseline"
Write-Host "  2. outputs/alignment_cka_layer       - +CKA Layer"
Write-Host "  3. outputs/alignment_segment_time    - +Segment Time"
Write-Host "  4. outputs/alignment_v2_full         - Alignment v2 (Full)"
Write-Host ""
Write-Host "下一步分析："
Write-Host "  1. 比较各组验证集困惑度/准确率"
Write-Host "  2. 检查训练稳定性"
Write-Host "  3. 分析时间/层对齐的独立贡献"
Write-Host "  4. 决定是否作为默认方法"
Write-Host ""
Write-Host "预期提升（根据文献和老师反馈）：" -ForegroundColor Yellow
Write-Host "  - 时间对齐改进：+1-2%"
Write-Host "  - 层对齐改进：+2-3%"
Write-Host "  - 组合效果：+3-5%"
Write-Host "==============================================================================" -ForegroundColor Cyan
