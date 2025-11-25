#!/bin/bash
# 快速验证脚本 - 根据老师反馈的执行顺序
# Quick validation script following teacher's recommendations

echo "=============================================================================="
echo "稳健小升级验证流程 (Stable Upgrades Validation)"
echo "=============================================================================="
echo ""
echo "执行顺序（按老师建议）："
echo "1. 诊断 loss 数量级"
echo "2. 小规模对比实验"
echo "3. 根据结果决定是否继续"
echo ""
echo "=============================================================================="

# Configuration
MODEL_NAME="Qwen/Qwen2-1.5B"
TEACHER_MODEL="Qwen/Qwen2-7B"
DATASET="openai/gsm8k"
SUBSET_SIZE=5000
EPOCHS=2
BATCH_SIZE=8

# Step 0: Diagnose loss scales
echo ""
echo "[Step 0] 🔍 诊断 Loss 数量级 (必须先做)"
echo "=============================================================================="
echo "命令："
echo "python experiments/diagnose_loss_scales.py \\"
echo "    --model_name $MODEL_NAME \\"
echo "    --teacher_model $TEACHER_MODEL \\"
echo "    --num_samples 10 \\"
echo "    --batch_size 4"
echo ""
read -p "按 Enter 运行诊断，或 Ctrl+C 跳过..."

python experiments/diagnose_loss_scales.py \
    --model_name "$MODEL_NAME" \
    --teacher_model "$TEACHER_MODEL" \
    --num_samples 10 \
    --batch_size 4

echo ""
echo "📋 请根据诊断结果调整权重（如有需要）"
echo "   - 如果 CKA 贡献 >15%: 降低 cka_weight 到 0.01"
echo "   - 如果 CKA 贡献 <1%: 可以提高到 0.1"
echo "   - 如果 KV 贡献 >50%: 降低 kv_weight"
echo ""
read -p "确认权重配置 OK？按 Enter 继续实验..."

# Default weights (adjust based on diagnostic)
KV_WEIGHT=1.0
CODI_WEIGHT=0.5
CKA_WEIGHT=0.05

echo ""
echo "使用的权重配置："
echo "  --kv_weight $KV_WEIGHT"
echo "  --codi_weight $CODI_WEIGHT"
echo "  --cka_weight $CKA_WEIGHT"
echo ""

# Step 1: Baseline
echo ""
echo "[Experiment 1/3] 📊 Baseline (无升级)"
echo "=============================================================================="
echo "输出目录: outputs/baseline"
echo ""
read -p "按 Enter 运行，或 Ctrl+C 跳过..."

python experiments/train_with_kv.py \
    --model_name "$MODEL_NAME" \
    --teacher_model "$TEACHER_MODEL" \
    --dataset_name "$DATASET" \
    --subset_size $SUBSET_SIZE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --kv_weight $KV_WEIGHT \
    --codi_weight $CODI_WEIGHT \
    --fp16 \
    --output_dir outputs/baseline

# Step 2: Attention-weighted (Student)
echo ""
echo "[Experiment 2/3] 🎯 Attention-weighted KV (学生注意力)"
echo "=============================================================================="
echo "输出目录: outputs/attn_weighted_student"
echo "配置: warmup=500, 使用学生注意力"
echo ""
read -p "按 Enter 运行，或 Ctrl+C 跳过..."

python experiments/train_with_kv.py \
    --model_name "$MODEL_NAME" \
    --teacher_model "$TEACHER_MODEL" \
    --dataset_name "$DATASET" \
    --subset_size $SUBSET_SIZE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --kv_weight $KV_WEIGHT \
    --codi_weight $CODI_WEIGHT \
    --use_attention_weighted_kv \
    --attention_weighted_kv_warmup 500 \
    --fp16 \
    --output_dir outputs/attn_weighted_student

# Step 3: Attention-weighted (Teacher) + CKA
echo ""
echo "[Experiment 3/3] 🚀 Attention-weighted KV (教师注意力) + CKA"
echo "=============================================================================="
echo "输出目录: outputs/attn_weighted_teacher_cka"
echo "配置: warmup=300, 使用教师注意力, CKA=0.05"
echo ""
read -p "按 Enter 运行，或 Ctrl+C 跳过..."

python experiments/train_with_kv.py \
    --model_name "$MODEL_NAME" \
    --teacher_model "$TEACHER_MODEL" \
    --dataset_name "$DATASET" \
    --subset_size $SUBSET_SIZE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --kv_weight $KV_WEIGHT \
    --codi_weight $CODI_WEIGHT \
    --use_attention_weighted_kv \
    --use_teacher_attention \
    --attention_weighted_kv_warmup 300 \
    --cka_weight $CKA_WEIGHT \
    --cka_layers middle \
    --fp16 \
    --output_dir outputs/attn_weighted_teacher_cka

# Summary
echo ""
echo "=============================================================================="
echo "✓ 所有实验完成"
echo "=============================================================================="
echo ""
echo "结果目录："
echo "  1. outputs/baseline"
echo "  2. outputs/attn_weighted_student"
echo "  3. outputs/attn_weighted_teacher_cka"
echo ""
echo "下一步："
echo "  1. 比较三组实验的验证集困惑度/准确率"
echo "  2. 检查训练稳定性（loss 曲线是否震荡）"
echo "  3. 根据老师建议决策："
echo "     - 提升 >2%: 继续 Phase 2 (GOVERN)"
echo "     - 提升 1-2%: 作为可选增强，focus on 多教师路由"
echo "     - 提升 <1% 或不稳定: 保持当前方法"
echo ""
echo "⚠️  重要提醒（老师反馈）："
echo "   这些升级是'底层 loss 工程强化'，不是核心创新"
echo "   核心方向是：多教师 KV 蒸馏 + 教师权重/路由设计"
echo "=============================================================================="
