#!/bin/bash
# v4.0 Quick Test Commands
# 快速测试命令集合

set -e  # 遇到错误立即停止

echo "=========================================="
echo "v4.0 Map Projection Quick Test Suite"
echo "=========================================="

# 设置项目根目录
PROJECT_ROOT="/Users/alexwang/quickly-check-for-mulit-teacher-kava-ache"
cd $PROJECT_ROOT

echo ""
echo "📋 Test 1: Integration Smoke Test"
echo "------------------------------------------"
python experiments/test_v4_integration.py
if [ $? -eq 0 ]; then
    echo "✅ Integration test PASSED"
else
    echo "❌ Integration test FAILED"
    exit 1
fi

echo ""
echo "📋 Test 2: Profile Alignment (Flat Mode)"
echo "------------------------------------------"
python experiments/profile_alignment.py --mode flat
if [ $? -eq 0 ]; then
    echo "✅ Flat profile PASSED"
else
    echo "❌ Flat profile FAILED"
    exit 1
fi

echo ""
echo "📋 Test 3: Profile Alignment (Structured Mode)"
echo "------------------------------------------"
python experiments/profile_alignment.py --mode structured
if [ $? -eq 0 ]; then
    echo "✅ Structured profile PASSED"
else
    echo "❌ Structured profile FAILED"
    exit 1
fi

echo ""
echo "📋 Test 4: 10-Step Training Smoke (Flat Mode)"
echo "------------------------------------------"
python experiments/train_with_kv.py \
    --model_name gpt2 \
    --subset_size 10 \
    --batch_size 2 \
    --epochs 1 \
    --alignment_mode flat \
    --kv_method rkv \
    --output_dir outputs/smoke_flat \
    --logging_steps 1

if [ $? -eq 0 ]; then
    echo "✅ Flat training smoke test PASSED"
else
    echo "❌ Flat training smoke test FAILED"
    exit 1
fi

echo ""
echo "📋 Test 5: 10-Step Training Smoke (Structured Mode)"
echo "------------------------------------------"
python experiments/train_with_kv.py \
    --model_name gpt2 \
    --subset_size 10 \
    --batch_size 2 \
    --epochs 1 \
    --alignment_mode structured \
    --map_proj_share_dim \
    --map_proj_init_uniform \
    --kv_method rkv \
    --output_dir outputs/smoke_structured \
    --logging_steps 1

if [ $? -eq 0 ]; then
    echo "✅ Structured training smoke test PASSED"
else
    echo "❌ Structured training smoke test FAILED"
    exit 1
fi

echo ""
echo "=========================================="
echo "🎉 ALL TESTS PASSED!"
echo "=========================================="
echo ""
echo "✅ Phase 2 Integration: COMPLETE"
echo "📝 Next: Review outputs in outputs/smoke_*/"
echo "🚀 Ready for A/B experiments (see V4_EXECUTION_ROADMAP.md)"
