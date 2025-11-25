#!/bin/bash
# 对比实验：Baseline vs Alignment v2
# Comparison: Baseline vs Alignment v2

echo "=============================================================================="
echo "Alignment v2 Comparison Experiments"
echo "=============================================================================="
echo ""
echo "实验组："
echo "  1. Baseline: 硬 index 对齐 + 等比例层映射"
echo "  2. +CKA Layer: 硬 index 对齐 + CKA 层映射"
echo "  3. +Segment Time: Segment 重采样 + 等比例层映射"
echo "  4. Alignment v2 (Full): Segment 重采样 + CKA 层映射"
echo ""
echo "=============================================================================="

# Configuration
MODEL_NAME="Qwen/Qwen2-1.5B"
TEACHER_MODEL="Qwen/Qwen2-7B"
DATASET="openai/gsm8k"
SUBSET_SIZE=5000
EPOCHS=2
BATCH_SIZE=8
LAYER_MAPPING="layer_mapping_qwen15b_7b.json"

# Check if layer mapping exists
if [ ! -f "$LAYER_MAPPING" ]; then
    echo ""
    echo "⚠️  Layer mapping not found: $LAYER_MAPPING"
    echo "   Running precomputation first..."
    echo ""
    
    python experiments/precompute_layer_mapping.py \
        --student_model "$MODEL_NAME" \
        --teacher_model "$TEACHER_MODEL" \
        --dataset_name "$DATASET" \
        --num_samples 100 \
        --output "$LAYER_MAPPING"
    
    if [ $? -ne 0 ]; then
        echo "✗ Precomputation failed!"
        exit 1
    fi
    
    echo ""
    echo "✓ Layer mapping precomputed"
    echo ""
fi

# Experiment 1: Baseline
echo ""
echo "[Experiment 1/4] 📊 Baseline (硬 index 对齐 + 等比例层映射)"
echo "=============================================================================="
echo "输出目录: outputs/alignment_baseline"
echo ""
read -p "按 Enter 运行，或 Ctrl+C 跳过..."

python experiments/train_with_kv.py \
    --model_name "$MODEL_NAME" \
    --teacher_model "$TEACHER_MODEL" \
    --dataset_name "$DATASET" \
    --subset_size $SUBSET_SIZE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --kv_weight 1.0 \
    --codi_weight 0.5 \
    --fp16 \
    --output_dir outputs/alignment_baseline

# Experiment 2: +CKA Layer
echo ""
echo "[Experiment 2/4] 🔬 +CKA Layer (硬 index 对齐 + CKA 层映射)"
echo "=============================================================================="
echo "输出目录: outputs/alignment_cka_layer"
echo ""
read -p "按 Enter 运行，或 Ctrl+C 跳过..."

python experiments/train_with_kv.py \
    --model_name "$MODEL_NAME" \
    --teacher_model "$TEACHER_MODEL" \
    --dataset_name "$DATASET" \
    --subset_size $SUBSET_SIZE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --kv_weight 1.0 \
    --codi_weight 0.5 \
    --use_cka_layer_mapping \
    --layer_mapping_path "$LAYER_MAPPING" \
    --fp16 \
    --output_dir outputs/alignment_cka_layer

# Experiment 3: +Segment Time
echo ""
echo "[Experiment 3/4] ⏱️  +Segment Time (Segment 重采样 + 等比例层映射)"
echo "=============================================================================="
echo "输出目录: outputs/alignment_segment_time"
echo ""
read -p "按 Enter 运行，或 Ctrl+C 跳过..."

python experiments/train_with_kv.py \
    --model_name "$MODEL_NAME" \
    --teacher_model "$TEACHER_MODEL" \
    --dataset_name "$DATASET" \
    --subset_size $SUBSET_SIZE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --kv_weight 1.0 \
    --codi_weight 0.5 \
    --use_segment_resampling \
    --fp16 \
    --output_dir outputs/alignment_segment_time

# Experiment 4: Alignment v2 (Full)
echo ""
echo "[Experiment 4/4] 🚀 Alignment v2 (Segment 重采样 + CKA 层映射)"
echo "=============================================================================="
echo "输出目录: outputs/alignment_v2_full"
echo ""
read -p "按 Enter 运行，或 Ctrl+C 跳过..."

python experiments/train_with_kv.py \
    --model_name "$MODEL_NAME" \
    --teacher_model "$TEACHER_MODEL" \
    --dataset_name "$DATASET" \
    --subset_size $SUBSET_SIZE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --kv_weight 1.0 \
    --codi_weight 0.5 \
    --use_cka_layer_mapping \
    --layer_mapping_path "$LAYER_MAPPING" \
    --use_segment_resampling \
    --fp16 \
    --output_dir outputs/alignment_v2_full

# Summary
echo ""
echo "=============================================================================="
echo "✓ 所有实验完成"
echo "=============================================================================="
echo ""
echo "结果目录："
echo "  1. outputs/alignment_baseline        - Baseline"
echo "  2. outputs/alignment_cka_layer       - +CKA Layer"
echo "  3. outputs/alignment_segment_time    - +Segment Time"
echo "  4. outputs/alignment_v2_full         - Alignment v2 (Full)"
echo ""
echo "下一步分析："
echo "  1. 比较各组验证集困惑度/准确率"
echo "  2. 检查训练稳定性"
echo "  3. 分析时间/层对齐的独立贡献"
echo "  4. 决定是否作为默认方法"
echo ""
echo "预期提升（根据文献和老师反馈）："
echo "  - 时间对齐改进：+1-2%"
echo "  - 层对齐改进：+2-3%"
echo "  - 组合效果：+3-5%"
echo "=============================================================================="
