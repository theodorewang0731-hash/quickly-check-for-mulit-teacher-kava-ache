# 稳健小升级修复摘要 (根据老师反馈)

## 日期：2025-11-18

---

## 📋 老师反馈的关键问题

### 1. ✅ **CRITICAL**: Attention weights 缺少 detach()

**问题描述**：
```python
# ❌ 原始代码（有问题）
attn = _to_tensor(attention_weights).to(student_segment.device)
token_importance = attn.mean(dim=(1, 2))
```

**为什么有问题**：
- 没有 detach，梯度会从 KV loss 流向注意力权重
- KV loss 会试图优化注意力，与 self-attention 训练目标冲突
- 导致训练不稳定

**修复**：
```python
# ✅ 修复后
attn = _to_tensor(attention_weights).to(student_segment.device)
attn = attn.detach()  # CRITICAL: 阻止梯度流向注意力
token_importance = attn.mean(dim=(1, 2))
```

**文件**: `experiments/kv_loss.py` (第 81 行)

---

### 2. ✅ 添加 Warmup 机制

**问题描述**：
- 训练早期学生注意力很弱，可能乱打权重
- 直接启用注意力加权会导致不稳定

**修复**：
- 新增参数：`--attention_weighted_kv_warmup 1000` (默认 1000 步)
- 前 N 步使用原始 KV loss
- Warmup 完成后才启用注意力加权

**实现**：
```python
# experiments/train_with_kv.py
use_attn_weighting = (
    args.use_attention_weighted_kv 
    and global_step >= args.attention_weighted_kv_warmup
)
```

**日志示例**：
```
Step 500: ... [Attn-warmup: 500 steps]  # 还在 warmup
Step 1000: ... [Attn-warmup: 0 steps]   # 即将启用
Step 1010: ... [Student-Attn-weighted]  # 已启用
```

---

### 3. ✅ 提供 Teacher Attention 选项

**问题描述**：
- 学生注意力：更 aligned，但早期不稳定
- 教师注意力：更稳定，但与学生 KV 一致性略差

**修复**：
- 新增参数：`--use_teacher_attention`
- 允许选择注意力来源

**Trade-off 表格**：

| 方案 | 稳定性 | 与学生 KV 一致性 | Warmup 需求 | 推荐场景 |
|------|--------|------------------|-------------|----------|
| 学生注意力 | 中等 | 高 | 1000 步 | 学生模型较大 (>1B) |
| 教师注意力 | 高 | 中等 | 500 步 | 学生模型较小 (<1B) |

**使用示例**：
```bash
# 学生注意力（默认）
python experiments/train_with_kv.py \
    --use_attention_weighted_kv \
    --attention_weighted_kv_warmup 1000

# 教师注意力（更稳定）
python experiments/train_with_kv.py \
    --use_attention_weighted_kv \
    --use_teacher_attention \
    --attention_weighted_kv_warmup 500
```

---

### 4. ✅ 创建 Loss 诊断工具

**问题描述**：
- 不知道各 loss 数量级是否合理
- CKA/KV 权重可能不匹配实际 loss scale
- 老师警告："如果 L_KV ≈ 0.1，L_CKA ≈ 1e-3，那乘 0.05 基本就没什么影响；如果 L_CKA 数值很大，那 0.05 可能会压过 KV-loss。"

**修复**：
- 创建 `experiments/diagnose_loss_scales.py`
- 在训练前快速检查各 loss 数量级
- 自动给出权重调整建议

**使用方法**：
```bash
python experiments/diagnose_loss_scales.py \
    --model_name Qwen/Qwen2-1.5B \
    --teacher_model Qwen/Qwen2-7B \
    --num_samples 10
```

**输出示例**：
```
Loss Scales (Mean ± Std)
================================================================================
CE Loss:            2.3456 ±   0.1234
KV Loss:            0.1234 ±   0.0234
KV-weighted:        0.1150 ±   0.0220
CODI Loss:          0.0856 ±   0.0123
CKA Loss:           0.7234 ±   0.0456

Weighted Contributions (with default weights)
================================================================================
CE contribution:      2.3456  ( 87.2%)
KV contribution:      0.1234  (  4.6%)  [weight=1.0]
CODI contribution:    0.0428  (  1.6%)  [weight=0.5]
CKA contribution:     0.0362  (  1.3%)  [weight=0.05]

📋 Recommendations:
✅ CKA weight is reasonable (1.3% of total)
✅ KV weight is reasonable (4.6% of total)
📊 Attention weighting effect: 6.8% change in KV loss
   ✅ Reasonable change - attention weighting should work
```

---

## 🎯 老师的核心建议

### 定位明确

> "这套'稳健小升级'只是你整体项目里的'底层 loss 工程强化'，**不是你项目的核心创新**。"

**核心方向**（才是论文主要贡献）：
- 多教师 KV 蒸馏
- 教师权重/路由设计
- 学到多种推理模式

**当前升级**（辅助性工程优化）：
- Attention-weighted KV：让 KV 蒸馏更关注关键 token
- CKA：让 hidden 表征整体更加 aligned

### 执行顺序

1. **先确认不破坏现有训练**
   - ✅ 运行 `diagnose_loss_scales.py`
   - ✅ 检查各 loss 数量级

2. **小规模验证实验**
   - baseline vs +attention vs +attention+CKA
   - 看稳定性、计算开销

3. **如果至少不变差**
   - 把注意力加权留作默认增强
   - CKA 按实验情况决定

4. **重点做多教师设计**
   - Teacher 权重的 sample-wise / task-wise 设计
   - 后续考虑 GOVERN/MT-KD

5. **不要喧宾夺主**
   - 千万别让这俩变成"我们论文的贡献"
   - 保持 focus 在多教师路由

### 预期提升的现实性

> "至于'预期提升 +2–4%，开销 +8–15% per step'：这是经验猜测，不是保证。你自己的结果可能某些 seed 上有提升，某些 seed 上差不多，少数情况甚至略降。所以你可以把这个当'optimistic target'，别当做 KPI。"

**决策标准**：
- ✅ 提升 >2%: 继续 Phase 2 (GOVERN)
- ⚠️ 提升 1-2%: 作为可选增强，focus on 多教师路由
- ❌ 提升 <1% 或不稳定: 保持当前方法

---

## 📁 修改文件清单

### 新增文件
1. `experiments/diagnose_loss_scales.py` - Loss 诊断工具 ⭐ 必用
2. `scripts/validate_stable_upgrades.sh` - Bash 验证脚本
3. `scripts/validate_stable_upgrades.ps1` - PowerShell 验证脚本
4. `FIXES_SUMMARY.md` - 本文档

### 修改文件
1. **`experiments/kv_loss.py`**
   - 第 81 行：添加 `attn = attn.detach()` ⭐ CRITICAL
   - 完整实现 4D/3D/2D 注意力格式处理

2. **`experiments/train_with_kv.py`**
   - 新增参数：
     - `--attention_weighted_kv_warmup` (默认 1000)
     - `--use_teacher_attention` (flag)
   - 实现 warmup 逻辑
   - 支持教师/学生注意力选择
   - 增强日志显示 warmup 状态

3. **`STABLE_UPGRADES_GUIDE.md`**
   - 添加老师反馈章节（置顶）
   - 添加诊断工具使用说明
   - 添加 warmup 和 teacher attention 说明
   - 强调"底层工程优化"定位

---

## ✅ 验证清单

在运行实验前，请确认：

- [ ] `experiments/kv_loss.py` 第 81 行包含 `attn.detach()`
- [ ] 运行 `diagnose_loss_scales.py` 检查 loss 数量级
- [ ] 根据诊断结果调整 `--cka_weight`（如有需要）
- [ ] 选择注意力来源：学生 or 教师
- [ ] 设置合理的 warmup 步数（学生:1000, 教师:500）
- [ ] 准备三组对比实验：baseline, +attn, +attn+CKA
- [ ] 明确这是"辅助优化"，不是核心创新

---

## 🚀 快速开始

### 1. 诊断 Loss 数量级

```bash
python experiments/diagnose_loss_scales.py \
    --model_name Qwen/Qwen2-1.5B \
    --teacher_model Qwen/Qwen2-7B \
    --num_samples 10
```

### 2. 根据诊断调整权重

如果诊断显示：
- CKA 贡献 >15%：降低到 `--cka_weight 0.01`
- CKA 贡献 <1%：提高到 `--cka_weight 0.1`

### 3. 运行验证脚本

**PowerShell (Windows)**:
```powershell
.\scripts\validate_stable_upgrades.ps1
```

**Bash (Linux/Mac)**:
```bash
bash scripts/validate_stable_upgrades.sh
```

或手动运行三组实验：

```bash
# Baseline
python experiments/train_with_kv.py \
    --model_name Qwen/Qwen2-1.5B \
    --teacher_model Qwen/Qwen2-7B \
    --dataset_name openai/gsm8k \
    --subset_size 5000 --epochs 2 --batch_size 8 \
    --output_dir outputs/baseline

# +Attention (Student)
python experiments/train_with_kv.py \
    --model_name Qwen/Qwen2-1.5B \
    --teacher_model Qwen/Qwen2-7B \
    --dataset_name openai/gsm8k \
    --subset_size 5000 --epochs 2 --batch_size 8 \
    --use_attention_weighted_kv \
    --attention_weighted_kv_warmup 500 \
    --output_dir outputs/attn_weighted_student

# +Attention (Teacher) + CKA
python experiments/train_with_kv.py \
    --model_name Qwen/Qwen2-1.5B \
    --teacher_model Qwen/Qwen2-7B \
    --dataset_name openai/gsm8k \
    --subset_size 5000 --epochs 2 --batch_size 8 \
    --use_attention_weighted_kv \
    --use_teacher_attention \
    --attention_weighted_kv_warmup 300 \
    --cka_weight 0.05 \
    --cka_layers middle \
    --output_dir outputs/attn_weighted_teacher_cka
```

---

## 🔍 检查修复是否生效

### 1. 验证 detach() 已添加

```bash
grep -n "attn.detach()" experiments/kv_loss.py
```

应该看到：
```
81:        attn = attn.detach()
```

### 2. 验证 warmup 机制

运行训练，检查日志：
```
Step 500: ... [Attn-warmup: 500 steps]
Step 1000: ... [Attn-warmup: 0 steps]
Step 1010: ... [Student-Attn-weighted]  # 应该从这里开始显示
```

### 3. 验证诊断工具

```bash
python experiments/diagnose_loss_scales.py \
    --model_name Qwen/Qwen2-1.5B \
    --teacher_model Qwen/Qwen2-7B \
    --num_samples 5
```

应该看到完整的 loss 统计和建议。

---

## 📚 参考

- **详细使用指南**: `STABLE_UPGRADES_GUIDE.md`
- **SOTA 分析**: `TARGETED_SOTA_RECOMMENDATIONS.md`
- **验证脚本**: `scripts/validate_stable_upgrades.ps1`

---

## 📞 问题排查

如果遇到问题，请参考 `STABLE_UPGRADES_GUIDE.md` 的"故障排除"章节，特别是：
- 问题 1: 训练早期 loss 震荡/发散
- 问题 3: Attention-weighted KV 没有效果
- 问题 4: CKA 贡献过大/过小
- 问题 7: Detach 忘记添加

---

**最后更新**: 2025-11-18  
**根据老师反馈完成所有关键修复** ✅
