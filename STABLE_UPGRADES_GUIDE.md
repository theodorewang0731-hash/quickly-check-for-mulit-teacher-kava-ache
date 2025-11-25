# 稳健小升级实施指南 (Stable Upgrades Guide)

## ⚠️ 关键修复（根据老师反馈）

### 已修复的重要问题：

1. **✅ Attention weights 已添加 detach()**
   - **问题**：之前没有 detach，会导致 KV loss 反向优化注意力，与 self-attention 训练目标冲突
   - **修复**：在 `experiments/kv_loss.py` 中添加 `attn = attn.detach()`
   - **影响**：防止训练不稳定，保持注意力机制的原始学习目标

2. **✅ 添加 Warmup 机制**
   - **问题**：训练早期学生注意力很弱，可能乱打权重
   - **修复**：新增 `--attention_weighted_kv_warmup 1000` 参数（默认 1000 步）
   - **影响**：前 1000 步使用原始 KV loss，之后才启用注意力加权

3. **✅ 提供 Teacher Attention 选项**
   - **问题**：学生注意力可能不够稳定
   - **修复**：新增 `--use_teacher_attention` flag
   - **Trade-off**：
     - 学生注意力：更 aligned，但早期不稳定
     - 教师注意力：更稳定，但与学生 KV 一致性略差

4. **✅ 提供 Loss 诊断工具**
   - **问题**：不知道各 loss 数量级是否合理
   - **修复**：创建 `experiments/diagnose_loss_scales.py`
   - **用途**：训练前快速检查权重配置，避免某项 loss 喧宾夺主

### 老师的核心建议：

> "这套'稳健小升级'只是你整体项目里的'底层 loss 工程强化'，**不是你项目的核心创新**。"

**核心方向**：多教师 → KV 蒸馏 → 老师权重/路由 → 学到多种推理模式

**当前升级**：
- Attention-weighted KV：让 KV 蒸馏更关注关键 token
- CKA：让 hidden 表征整体更加 aligned

**定位**：✅ 更强的基线，🚫 不是主要贡献

---

## 概述

实现了两个针对多教师 KV 蒸馏的「稳健小升级」，保留 KaVa 框架，仅添加轻量级增强：

1. **Attention-weighted KV Loss** - 根据 token 重要性加权 KV 损失
2. **CKA Auxiliary Loss** - 表示对齐辅助损失

预期提升：+2-4% 性能改进  
实施时间：1.5 天  
风险级别：低（向后兼容）

---

## Phase 1: Attention-weighted KV + CKA (已实现)

### 1.1 Attention-weighted KV Loss

**原理**：
- 参考 MiniCache (NeurIPS 2024) 的 token 重要性思想
- 从学生模型的最后一层注意力权重提取 token 重要性
- 重要 token 的 KV 损失权重更高

**修改文件**：
- `experiments/kv_loss.py`: 
  - 新增 `attention_weights` 参数到 `compute_kv_loss()`
  - 自动处理 4D (batch, heads, seq, seq) / 3D (batch, seq, seq) / 2D (batch, seq) 格式
  - 向后兼容（不传 attention_weights 则使用原始均匀损失）

**实现细节**：
```python
# Token importance from attention maps
if attention_weights is not None:
    # **CRITICAL**: Detach to prevent gradient flow back to attention
    attn = attn.detach()
    
    # Average over heads and queries
    token_importance = attn.mean(dim=(1, 2))  # (batch, seq)
    token_importance = token_importance / (token_importance.sum(dim=-1, keepdim=True) + 1e-8)
    
    # Align to selected length and weight the loss
    importance_weight = token_importance[:, :sel_len].unsqueeze(-1)
    weighted_loss = (diff * importance_weight).sum() / diff.numel()
```

**关键修复**：
- ✅ `attn.detach()` - 防止 KV loss 反向优化注意力
- ✅ Warmup 机制 - 避免早期学生注意力不稳定
- ✅ Teacher attention 选项 - 提供更稳定的替代方案

---

### 1.2 CKA Auxiliary Loss

**原理**：
- Centered Kernel Alignment (ICML 2019, 2024)
- 测量学生和教师隐藏表示的相似性
- 作为小权重辅助项，不破坏主要 KV 损失

**修改文件**：
- `experiments/cka_loss.py`: 新文件，实现 RCKA (unbiased estimator)
  - `linear_cka()`: 线性 CKA 计算
  - `cka_loss()`: 返回 1 - CKA (最小化以最大化对齐)
  - `multi_layer_cka_loss()`: 多层 CKA (默认只用中间层)
  - `contrastive_alignment_loss()`: 可选的对比损失增强

**推荐配置**：
- **层选择**：中间层 (默认) 或最后一层
- **权重**：λ_CKA = 0.05 (小辅助项，不喧宾夺主)
- **计算开销**：Gram matrix 计算 O(n²)，但 n = batch * seq_len 通常不大

**公式**：
$$
\text{CKA}(X, Y) = \frac{\text{HSIC}(X, Y)}{\sqrt{\text{HSIC}(X, X) \cdot \text{HSIC}(Y, Y)}}
$$

其中 HSIC (Hilbert-Schmidt Independence Criterion):
$$
\text{HSIC}(X, Y) = \text{tr}(K_X K_Y)
$$

---

## 使用方法

### 🔍 步骤 0: 诊断 Loss 数量级（强烈推荐）

**在正式训练前**，先运行诊断工具检查各 loss 是否平衡：

```bash
python experiments/diagnose_loss_scales.py \
    --model_name Qwen/Qwen2-1.5B \
    --teacher_model Qwen/Qwen2-7B \
    --num_samples 10 \
    --batch_size 4
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
Total loss:           2.5480

📋 Recommendations:
✅ CKA weight is reasonable (1.3% of total)
✅ KV weight is reasonable (4.6% of total)
📊 Attention weighting effect: 6.8% change in KV loss
   ✅ Reasonable change - attention weighting should work
```

**根据诊断调整权重**：
- 如果 CKA 贡献 >15%：降低 `--cka_weight` 到 0.01
- 如果 CKA 贡献 <1%：可以提高到 0.1
- 如果 KV 贡献 >50%：降低 `--kv_weight`

---

### 基础训练（无升级）

```bash
python experiments/train_with_kv.py \
    --model_name Qwen/Qwen2-1.5B \
    --teacher_model Qwen/Qwen2-7B \
    --dataset_name openai/gsm8k \
    --epochs 3 \
    --batch_size 8 \
    --kv_method rkv \
    --kv_weight 1.0 \
    --codi_weight 0.5 \
    --fp16 \
    --gradient_checkpointing
```

### 启用 Attention-weighted KV

```bash
python experiments/train_with_kv.py \
    --model_name Qwen/Qwen2-1.5B \
    --teacher_model Qwen/Qwen2-7B \
    --dataset_name openai/gsm8k \
    --epochs 3 \
    --batch_size 8 \
    --kv_method rkv \
    --kv_weight 1.0 \
    --codi_weight 0.5 \
    --use_attention_weighted_kv \
    --attention_weighted_kv_warmup 1000 \
    --fp16 \
    --gradient_checkpointing
```

**新增参数**：
- `--use_attention_weighted_kv`: 启用注意力加权 KV 损失
- `--attention_weighted_kv_warmup 1000`: Warmup 步数（默认 1000，避免早期不稳定）

**替代方案（使用教师注意力，更稳定）**：
```bash
python experiments/train_with_kv.py \
    --model_name Qwen/Qwen2-1.5B \
    --teacher_model Qwen/Qwen2-7B \
    --dataset_name openai/gsm8k \
    --epochs 3 \
    --batch_size 8 \
    --kv_method rkv \
    --use_attention_weighted_kv \
    --use_teacher_attention \
    --attention_weighted_kv_warmup 500 \
    --fp16
```

**新增参数**：
- `--use_teacher_attention`: 使用教师注意力而非学生注意力
  - ✅ 更稳定（教师注意力已训练好）
  - ❌ 与学生 KV 一致性略差
  - 💡 推荐：teacher attention + 较短 warmup (500 步)

**Trade-off 分析**：

| 方案 | 稳定性 | 与学生 KV 一致性 | Warmup 需求 | 推荐场景 |
|------|--------|------------------|-------------|----------|
| 学生注意力 | 中等 | 高 | 1000 步 | 学生模型较大 (>1B) |
| 教师注意力 | 高 | 中等 | 500 步 | 学生模型较小 (<1B) |

### 启用 CKA Auxiliary Loss

```bash
python experiments/train_with_kv.py \
    --model_name Qwen/Qwen2-1.5B \
    --teacher_model Qwen/Qwen2-7B \
    --dataset_name openai/gsm8k \
    --epochs 3 \
    --batch_size 8 \
    --kv_method rkv \
    --kv_weight 1.0 \
    --codi_weight 0.5 \
    --cka_weight 0.05 \
    --cka_layers middle \
    --fp16 \
    --gradient_checkpointing
```

**新增参数**：
- `--cka_weight`: CKA 损失权重 (默认 0.05)
- `--cka_layers`: 使用哪些层计算 CKA
  - `middle`: 中间层 (默认，推荐)
  - `last`: 最后一层
  - `6,12`: 逗号分隔的层索引

### 完全启用（推荐配置）

```bash
# 推荐：先运行诊断工具
python experiments/diagnose_loss_scales.py \
    --model_name Qwen/Qwen2-1.5B \
    --teacher_model Qwen/Qwen2-7B \
    --num_samples 10

# 根据诊断结果调整权重，然后训练
python experiments/train_with_kv.py \
    --model_name Qwen/Qwen2-1.5B \
    --teacher_model Qwen/Qwen2-7B \
    --dataset_name openai/gsm8k \
    --epochs 3 \
    --batch_size 8 \
    --kv_method rkv \
    --kv_weight 1.0 \
    --codi_weight 0.5 \
    --use_attention_weighted_kv \
    --attention_weighted_kv_warmup 1000 \
    --cka_weight 0.05 \
    --cka_layers middle \
    --fp16 \
    --gradient_checkpointing \
    --output_dir outputs/stable_upgrade_v1
```

**日志示例（显示 warmup 状态）**：
```
Step 500: loss=2.3123, CE=1.8000, KV=0.3200, CODI=0.1690, CKA=0.7233 [Attn-warmup: 500 steps]
Step 1000: loss=2.2456, CE=1.7500, KV=0.3100, CODI=0.1600, CKA=0.6956 [Attn-warmup: 0 steps]
Step 1010: loss=2.2012, CE=1.7500, KV=0.2900, CODI=0.1600, CKA=0.6912 [Student-Attn-weighted]
Step 1020: loss=2.1678, CE=1.7200, KV=0.2800, CODI=0.1578, CKA=0.6700 [Student-Attn-weighted]
```

**解读**：
- Step 500-1000: Warmup 阶段，使用原始 KV loss
- Step 1010+: Warmup 完成，启用注意力加权（KV loss 有所下降）

---

## 验证实验

### 小规模验证（1-2天）

**目标**：快速验证是否有 +2% 以上的提升

**步骤 1：运行诊断**
```bash
python experiments/diagnose_loss_scales.py \
    --model_name Qwen/Qwen2-1.5B \
    --teacher_model Qwen/Qwen2-7B \
    --num_samples 10
```

**步骤 2：根据诊断调整权重（如有需要）**

**步骤 3：三组对比实验**

```bash
# Baseline (无升级)
python experiments/train_with_kv.py \
    --model_name Qwen/Qwen2-1.5B \
    --teacher_model Qwen/Qwen2-7B \
    --dataset_name openai/gsm8k \
    --subset_size 5000 \
    --epochs 2 \
    --batch_size 8 \
    --output_dir outputs/baseline

# With Attention-weighted KV (学生注意力)
python experiments/train_with_kv.py \
    --model_name Qwen/Qwen2-1.5B \
    --teacher_model Qwen/Qwen2-7B \
    --dataset_name openai/gsm8k \
    --subset_size 5000 \
    --epochs 2 \
    --batch_size 8 \
    --use_attention_weighted_kv \
    --attention_weighted_kv_warmup 500 \
    --output_dir outputs/attn_weighted_student

# With Attention-weighted KV (教师注意力) + CKA
python experiments/train_with_kv.py \
    --model_name Qwen/Qwen2-1.5B \
    --teacher_model Qwen/Qwen2-7B \
    --dataset_name openai/gsm8k \
    --subset_size 5000 \
    --epochs 2 \
    --batch_size 8 \
    --use_attention_weighted_kv \
    --use_teacher_attention \
    --attention_weighted_kv_warmup 300 \
    --cka_weight 0.05 \
    --cka_layers middle \
    --output_dir outputs/attn_weighted_teacher_cka
```

**评估指标**：
- 验证集困惑度 (Perplexity)
- 下游任务准确率 (如果有评估脚本)

**决策标准**：
- ✅ 如果提升 >2%：继续实施 Phase 2 (GOVERN router)
- ⚠️ 如果提升 1-2%：作为可选增强，focus on 多教师路由
- ❌ 如果提升 <1% 或不稳定：保持当前方法，不过度工程化

**老师的提醒**：
> "至于'预期提升 +2–4%，开销 +8–15% per step'：
> 这是经验猜测，不是保证。你自己的结果可能某些 seed 上有提升，某些 seed 上差不多，
> 少数情况甚至略降。所以你可以把这个当'optimistic target'，别当做 KPI。"

**重要**：这些升级是「底层 loss 工程强化」，不是核心创新。真正重点在多教师权重/路由设计。

---

## 日志示例

### 无升级（Baseline）
```
Step 10: loss=2.3456, CE=1.8000, KV=0.3500, CODI=0.1956
Step 20: loss=2.1234, CE=1.6500, KV=0.3200, CODI=0.1534
```

### 启用 Attention-weighted KV
```
Step 10: loss=2.2890, CE=1.8000, KV=0.3200, CODI=0.1690 [Attn-weighted]
Step 20: loss=2.0678, CE=1.6500, KV=0.2800, CODI=0.1378 [Attn-weighted]
```

### 启用 Attention-weighted KV + CKA
```
Step 10: loss=2.3123, CE=1.8000, KV=0.3200, CODI=0.1690, CKA=0.7233 [Attn-weighted]
Step 20: loss=2.0912, CE=1.6500, KV=0.2800, CODI=0.1378, CKA=0.6234 [Attn-weighted]
```

**解读**：
- CKA 初始值接近 1 (完全不对齐)
- 随训练逐渐降低 (对齐改善)
- CKA 权重小 (0.05)，对总损失影响有限

---

## 技术细节

### Attention Weighting 实现

**输入格式兼容性**：
- **4D**: `(batch, num_heads, seq_len, seq_len)` - 完整注意力矩阵
- **3D**: `(batch, seq_len, seq_len)` - 已平均过 heads
- **2D**: `(batch, seq_len)` - 已平均过 heads 和 queries

**重要性归一化**：
```python
token_importance = attn.mean(dim=(1, 2))  # (batch, seq_len)
token_importance = token_importance / (token_importance.sum(dim=-1, keepdim=True) + 1e-8)
```

**长度对齐**：
- KV 损失只在 `sel_len` (compressed KV length) 上计算
- Attention weights 裁剪到 `[:, :sel_len]` 匹配

### CKA 实现

**Unbiased Estimator (Debiased=True)**：
```python
hsic_xy = (X_gram * Y_gram).sum() - X_gram.diagonal().sum() * Y_gram.diagonal().sum() / (n - 2)
hsic_xx = (X_gram * X_gram).sum() - (X_gram.diagonal() ** 2).sum() / (n - 2)
hsic_yy = (Y_gram * Y_gram).sum() - (Y_gram.diagonal() ** 2).sum() / (n - 2)
cka = hsic_xy / sqrt(hsic_xx * hsic_yy)
```

**计算复杂度**：
- Gram matrix: O(n² * d) where n = batch * seq_len, d = hidden_dim
- 通常 n ≈ 8 * 512 = 4096，可接受

**层选择建议**：
- **Middle layer**: 最稳定，避免过拟合输出层
- **Multiple layers**: 可以用 `6,12` 选择中层和输出层
- **避免早期层**: 特征还不够抽象

---

## 计算开销

### Baseline (无升级)
- 前向传播: Student + Teacher
- 损失计算: CE + KV + CODI
- 内存: ~16GB (Qwen2-1.5B + 7B, batch=8)

### + Attention-weighted KV
- **额外前向**: 需要 `output_attentions=True`
- **额外内存**: ~1-2GB (attention maps)
- **额外计算**: Token importance 计算 (negligible)
- **总增加**: ~5-10% 时间开销

### + CKA Auxiliary Loss
- **额外计算**: Gram matrix O(n²)
- **额外内存**: 2 * n² floats (n ≈ 4096)
- **总增加**: ~3-5% 时间开销
- **仅在选定层**: 默认只有中间层 (1 层)

### 总计 (Full Upgrade)
- **时间开销**: +8-15% per step
- **内存开销**: +1-3GB
- **性能提升**: 预期 +2-4%

**性价比评估**: ✅ 合理，如果提升 >2% 则值得

---

## 下一步：Phase 2 (Optional)

如果 Phase 1 验证成功 (>2% 提升)，可继续实施：

### GOVERN Gradient Voting Router (3-4 天)

**原理**：
- 基于梯度方向的教师融合
- 不是固定权重，而是动态投票
- 论文：GOVERN (ICML 2024)

**实施计划**：
1. 创建 `fuse/govern_router.py`
2. 实现梯度方向投票机制
3. 集成到多教师训练循环
4. 预期额外提升：+2-3%

**风险**：中等（需要修改训练流程，增加计算开销）

---

## 故障排除

### 问题 1: 训练早期 loss 震荡/发散
**可能原因**: 学生注意力不稳定，导致 KV 权重异常  
**解决**: 
- ✅ 增加 warmup 步数：`--attention_weighted_kv_warmup 2000`
- ✅ 切换到教师注意力：`--use_teacher_attention`
- ✅ 暂时禁用注意力加权，只用 CKA

### 问题 2: CKA loss 为 NaN
**原因**: Gram matrix 数值不稳定  
**解决**: 
- 检查 `debiased=True` (unbiased estimator)
- 增加 epsilon: `hsic_xx + 1e-10`
- 降低 `--cka_weight` 到 0.01

### 问题 3: Attention-weighted KV 没有效果
**原因**: 可能的问题  
**解决**:
- 运行诊断工具检查：`python experiments/diagnose_loss_scales.py`
- 查看 "Attention weighting effect" 指标
- 如果 <5% 变化：注意力加权可能不适合当前设置
- 确认 warmup 已完成（查看日志中的 `[Attn-warmup: X steps]`）

### 问题 4: CKA 贡献过大/过小
**原因**: 权重不匹配 loss 数量级  
**解决**:
- ✅ **必须先运行诊断工具** `diagnose_loss_scales.py`
- 根据诊断建议调整 `--cka_weight`
- 目标：CKA 贡献 1-10% 的总 loss
- 老师建议："必要时把 CKA 权重调到 0.01 甚至 0.005"

### 问题 5: 内存不足
**原因**: Attention maps 占用额外显存  
**解决**:
- 减小 `--batch_size`
- 启用 `--gradient_checkpointing`
- 使用教师注意力（减少学生 attention 计算）
- 只在最后一层使用 attention weighting

### 问题 6: 训练速度变慢
**原因**: CKA 计算 Gram matrix 开销  
**解决**:
- 减少 `--cka_layers` 数量 (只用 1 层)
- 降低 `--cka_weight` 到 0.0 (禁用)
- 仅在验证集上使用 CKA (不在训练中)

### 问题 7: Detach 忘记添加
**检查方法**:
```bash
grep -n "attn.detach()" experiments/kv_loss.py
```
**应该看到**:
```
81:        attn = attn.detach()
```
**如果没有**：这是 **critical bug**，必须修复！

---

## 文件清单

### 新增文件
- `experiments/cka_loss.py` - CKA auxiliary loss 实现
- `experiments/diagnose_loss_scales.py` - Loss 数量级诊断工具 ⭐ 必用

### 修改文件
- `experiments/kv_loss.py` - 添加 attention_weights 参数 + **detach() 修复**
- `experiments/train_with_kv.py` - 集成 attention-weighted KV 和 CKA + warmup 机制

### 文档
- `STABLE_UPGRADES_GUIDE.md` - 本文档（包含老师反馈修复）
- `TARGETED_SOTA_RECOMMENDATIONS.md` - SOTA 方法分析

---

## 老师反馈总结

### ✅ 肯定的点

1. **Attention-weighted KV 思路对路**
   - 跟 MiniCache/SpindleKV 同类直觉，但用在训练 loss 上
   - 逻辑合理：重要 token → KV 对齐更重要

2. **CKA 作为小辅助正则合理**
   - 比单纯 MSE 更稳的结构对齐方式
   - 权重小，不喧宾夺主

3. **开关设计做得好**
   - 向后兼容，可开可关
   - 便于三组实验：baseline → +attn → +attn+CKA

### ⚠️ 需要注意的点

1. **权重必须 detach**
   - ✅ 已修复：`attn = attn.detach()`
   - 否则会破坏 self-attention 训练

2. **需要 warmup**
   - ✅ 已添加：`--attention_weighted_kv_warmup 1000`
   - 避免早期学生注意力乱打权重

3. **提供 teacher attention 选项**
   - ✅ 已添加：`--use_teacher_attention`
   - Trade-off：稳定 vs 一致性

4. **CKA 权重要小心**
   - ✅ 已提供诊断工具
   - 目标：CKA 贡献 1-10% 总 loss
   - 必要时调到 0.01 或 0.005

5. **层数不要太多**
   - 默认：只用中间 1 层
   - 最多：前/中/后 3 层

### 🎯 核心定位

**这不是核心创新，是底层 loss 工程强化**

真正的核心方向：
- 多教师 KV 蒸馏
- 教师权重/路由设计
- 学到多种推理模式

这些升级的作用：
- ✅ 更强的单教师/多教师基线
- ✅ 让 KV 蒸馏更 robust
- 🚫 不是论文主要贡献

### 📋 建议的执行顺序

1. **先确认不破坏现有训练**
   - 运行 `diagnose_loss_scales.py`
   - 检查各 loss 数量级

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

---

## 参考文献

### Attention-weighted KV
- MiniCache (NeurIPS 2024): KV cache compression with importance scoring
- Token dropping (ICLR 2023): Selective token attention

### CKA Loss
- Kornblith et al. (ICML 2019): "Similarity of Neural Network Representations Revisited"
- Cui et al. (ICML 2024): "Representation Alignment via CKA for Knowledge Distillation"

### KaVa Framework
- KaVa (arxiv:2501.00231): Key-Value Matching for Teacher-Student Distillation

---

## 总结

✅ **已实现**：
- Attention-weighted KV loss (保留 KaVa 框架，仅加权)
- CKA auxiliary loss (轻量级表示对齐)
- 完全向后兼容
- 命令行开关控制

🎯 **预期效果**：
- +2-4% 性能提升
- 低风险（不破坏现有方法）
- 实施时间 1.5 天

📊 **下一步验证**：
1. 运行小规模对比实验 (5000 samples, 2 epochs)
2. 比较 Baseline vs Attn-weighted vs +CKA
3. 如果 >2% 提升 → 实施 Phase 2 (GOVERN)
4. 如果 <2% 提升 → 保持当前方法

---

**联系**: 如有问题，参考 `TARGETED_SOTA_RECOMMENDATIONS.md` 了解更多 SOTA 方法背景。
