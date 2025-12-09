# v4.0 Map Projection 集成完成报告

**日期**: 2025-12-09  
**版本**: v4.0 Phase 2 Integration  
**状态**: ✅ **集成完成，等待测试验证**

---

## 📋 完成内容总结

### 1. 训练脚本集成 (train_with_kv.py) ✅

#### 1.1 新增导入
```python
# v4.0 Map Projection imports
from src.map_projection_aligner import MapProjectionAligner
```

#### 1.2 新增工具函数
```python
def stack_past_kv(past_key_values, as_tensor=True):
    """将 tuple of (k, v) 转为 [L, 2, B, H, T, D] 张量"""
    # 完整实现已添加到 train_with_kv.py
```

#### 1.3 命令行参数扩展
```bash
--alignment_mode {flat,structured}     # 对齐模式：flat (baseline) / structured (v4.0)
--map_proj_share_dim                   # 共享维度投影
--map_proj_init_uniform                # 均匀初始化 head_mixer
```

#### 1.4 双模式 Aligner 初始化
- **Flat Mode**: 使用原有的 `StudentToTeacherProjector` (baseline)
- **Structured Mode**: 使用 `MapProjectionAligner` (v4.0 新方案)
- 参数懒加载：首次 batch 时根据实际维度初始化
- 自动添加到 optimizer 参数列表

#### 1.5 训练循环双模式分支

**Structured Mode (v4.0)**:
```python
# 1. 准备输入: numpy -> torch tensor
teacher_k_stack = torch.stack([...], dim=0)  # [L_t, B, H_t, T, D_t]

# 2. 获取 student KV
student_pkv = student(..., use_cache=True).past_key_values
student_k_stack = torch.stack([...], dim=0)  # [L_s, B, H_s, T, D_s]

# 3. 创建 segment_ids (全 0 表示单 segment)
segment_ids = torch.zeros(B, T, dtype=torch.long, device=device)

# 4. Map Projection Alignment
aligned_k, aligned_v, attn_map = map_aligner(
    teacher_k_stack, teacher_v_stack, None, segment_ids
)

# 5. Compute KV loss: MSE
kv_loss = (F.mse_loss(aligned_k, student_k) + F.mse_loss(aligned_v, student_v)) / 2
```

**Flat Mode (Baseline)**:
```python
# 原有流程保持不变
for layer_idx, layer in enumerate(comp):
    tk, student_seg = align_teacher_kv_to_student(...)
    student_proj = projectors[layer_idx](student_seg)
    l = compute_kv_loss(student_proj, tk, ...)
    layer_losses.append(l)
kv_loss = torch.stack(layer_losses).mean()
```

#### 1.6 日志和检查点更新
- 日志输出添加 `[Mode: flat/structured]` 标记
- 检查点保存时包含 `map_aligner.pt` (structured mode)
- 训练完成报告包含对齐模式和配置参数

---

## 🧪 测试验证工具

### 1. 集成冒烟测试
**文件**: `experiments/test_v4_integration.py`

**测试内容**:
- ✅ 模块导入检查 (MapProjectionAligner, HeadwiseMapProjector, TimeWarper)
- ✅ stack_past_kv 工具函数
- ✅ 双模式 Aligner 初始化
- ✅ 完整对齐流程模拟（含 loss 计算）
- ✅ 命令行参数解析

**运行方式**:
```bash
python experiments/test_v4_integration.py
```

### 2. Profile Alignment (已存在)
**文件**: `experiments/profile_alignment.py`

**运行方式**:
```bash
# Flat mode
python experiments/profile_alignment.py --mode flat

# Structured mode
python experiments/profile_alignment.py --mode structured
```

---

## 🚀 下一步行动计划

### Step 1: 冒烟测试 (立即执行)

#### 1.1 模块测试
```bash
cd /Users/alexwang/quickly-check-for-mulit-teacher-kava-ache
python experiments/test_v4_integration.py
```

**预期输出**:
```
✓ MapProjectionAligner imported
✓ HeadwiseMapProjector imported
✓ TimeWarper imported
✓ stack_past_kv: torch.Size([2, 8, 50, 64]) -> torch.Size([4, 2, 2, 8, 50, 64])
✓ Structured Aligner: 13,824 parameters
✓ Flat Aligner: XXX parameters
✓ Alignment: [4,2,8,50,64] -> [2,2,4,50,64]
✅ ALL TESTS PASSED
```

#### 1.2 Profile Alignment (双模式)
```bash
# Baseline
python experiments/profile_alignment.py --mode flat

# v4.0
python experiments/profile_alignment.py --mode structured
```

**检查项**:
- [ ] 无形状错误
- [ ] 无 NaN
- [ ] 参数量统计正确
- [ ] Attention 分布合理

#### 1.3 10-Step 训练冒烟测试

**Baseline (Flat)**:
```bash
python experiments/train_with_kv.py \
    --model_name gpt2 \
    --subset_size 10 \
    --batch_size 2 \
    --epochs 1 \
    --alignment_mode flat \
    --kv_method rkv \
    --output_dir outputs/smoke_flat
```

**v4.0 (Structured)**:
```bash
python experiments/train_with_kv.py \
    --model_name gpt2 \
    --subset_size 10 \
    --batch_size 2 \
    --epochs 1 \
    --alignment_mode structured \
    --map_proj_share_dim \
    --map_proj_init_uniform \
    --kv_method rkv \
    --output_dir outputs/smoke_structured
```

**验证检查**:
- [ ] 训练完成无报错
- [ ] Loss 正常收敛（不是 NaN/Inf）
- [ ] 日志显示正确的 `[Mode: ...]` 标记
- [ ] 检查点保存成功（含 map_aligner.pt）

---

### Step 2: A/B 对比实验 (冒烟测试通过后)

#### 实验矩阵

| 实验名称 | Mode | share_dim | init_uniform | 输出目录 |
|---------|------|-----------|--------------|---------|
| Baseline | flat | N/A | N/A | `outputs/ab_flat` |
| v4.0-Full | structured | ✓ | ✓ | `outputs/ab_structured_full` |
| v4.0-NoShare | structured | ✗ | ✓ | `outputs/ab_structured_noshare` |
| v4.0-Random | structured | ✓ | ✗ | `outputs/ab_structured_random` |

#### 实验配置（建议）
```bash
# 共同参数
MODEL=gpt2
SUBSET=1000         # 足够的数据量
BATCH=8
EPOCHS=3
KV_METHOD=rkv
LR=5e-5
```

#### Baseline 实验
```bash
python experiments/train_with_kv.py \
    --model_name gpt2 \
    --subset_size 1000 \
    --batch_size 8 \
    --epochs 3 \
    --alignment_mode flat \
    --kv_method rkv \
    --lr 5e-5 \
    --output_dir outputs/ab_flat \
    --logging_steps 10 \
    --save_steps 200
```

#### v4.0 实验 (推荐配置)
```bash
python experiments/train_with_kv.py \
    --model_name gpt2 \
    --subset_size 1000 \
    --batch_size 8 \
    --epochs 3 \
    --alignment_mode structured \
    --map_proj_share_dim \
    --map_proj_init_uniform \
    --kv_method rkv \
    --lr 5e-5 \
    --output_dir outputs/ab_structured_full \
    --logging_steps 10 \
    --save_steps 200
```

#### 结果分析指标
1. **训练稳定性**
   - Loss 曲线平滑度
   - NaN/Inf 发生率
   - 梯度范数

2. **最终性能**
   - Final CE Loss
   - Final KV Loss
   - Final CODI Loss

3. **参数效率**
   - 总参数量
   - 训练时间
   - 内存占用

4. **Alignment 质量** (可选)
   - CKA 分数
   - Attention 熵
   - Head Mixer 权重分布

---

## 📊 代码变更统计

### 修改文件
- **experiments/train_with_kv.py**: +150 lines
  - 新增 `stack_past_kv` 工具函数
  - 添加 3 个命令行参数
  - 双模式 Aligner 初始化逻辑
  - 训练循环双模式分支 (if/else)
  - 日志和检查点更新

### 新增文件
- **experiments/test_v4_integration.py**: 230 lines
  - 5 个集成测试
  - 完整的对齐流程模拟

### 关键设计决策
1. **控制变量原则**: flat vs structured 只改对齐方式，loss 函数完全相同 (MSE)
2. **懒加载初始化**: Aligner 在首次 batch 时初始化，避免硬编码维度
3. **双模式兼容**: 同一脚本支持两种模式，配置文件一键切换
4. **向后兼容**: Flat mode 保持原有逻辑不变，确保 baseline 可复现

---

## ⚠️ 已知限制和注意事项

### 1. Student KV 重复 Forward
**问题**: Structured mode 需要获取 student past_key_values，需要额外一次 forward
```python
s_out_kv = student(input_ids, attention_mask, use_cache=True)
```

**影响**: 
- 计算成本增加 ~1x student forward
- 内存占用增加

**解决方案** (可选优化):
- 在主 forward 时就设置 `use_cache=True`
- 重用 KV cache

### 2. Segment IDs 简化假设
**当前实现**: 整个序列视为单个 segment (全 0)
```python
segment_ids = torch.zeros(B, T, dtype=torch.long, device=device)
```

**适用场景**: 
- ✅ 标准训练 (prompt + answer 不分段)
- ✅ 短序列 (< 512 tokens)

**不适用场景**:
- ✗ 多 segment 复杂推理 (需要真实 segment 标注)
- ✗ 超长序列 (需要动态分段)

**扩展方法**:
- 添加 `SegmentIdentifier` (已有模块)
- 基于 attention mask 或特殊 token 分段

### 3. 时间维度对齐
**当前实现**: 使用 TimeWarper 的 3 段式采样 (P/R/A)

**假设**: 
- batch 内所有样本使用 `segment_ids[0]` 的段长度
- 适用于 batch 内序列长度一致的情况

**改进方向**:
- Dynamic segment length per sample
- Adaptive time resampling

---

## ✅ 集成检查清单

### Phase 2.1: 代码集成
- [x] 导入 MapProjectionAligner
- [x] 添加 stack_past_kv 工具函数
- [x] 添加命令行参数 (alignment_mode, share_dim, init_uniform)
- [x] 双模式 Aligner 初始化逻辑
- [x] 训练循环双模式分支
- [x] 日志输出更新
- [x] 检查点保存更新
- [x] 训练完成报告更新

### Phase 2.2: 测试验证
- [ ] 运行 test_v4_integration.py
- [ ] 运行 profile_alignment.py (flat & structured)
- [ ] 10-step 冒烟测试 (flat)
- [ ] 10-step 冒烟测试 (structured)

### Phase 2.3: A/B 实验
- [ ] Baseline 实验 (flat mode)
- [ ] v4.0 实验 (structured mode)
- [ ] 结果收集和对比分析
- [ ] 更新 DEVELOPMENT_HISTORY.md

---

## 📚 相关文档

- **V4_EXECUTION_ROADMAP.md**: 完整执行路线图
- **V4_UPDATE_COMPLETION.md**: v4.0 更新完成报告
- **DEVELOPMENT_HISTORY.md**: 项目发展历史
- **experiments/profile_alignment.py**: 对齐 profile 工具

---

## 🎯 成功标准

### Phase 2 完成标准
1. ✅ 代码集成无语法错误
2. ⏳ test_v4_integration.py 全部通过
3. ⏳ 10-step 冒烟测试成功 (flat & structured)
4. ⏳ 无 NaN/Inf/形状错误

### Phase 3 完成标准
1. ⏳ A/B 实验完成 (≥1000 samples, 3 epochs)
2. ⏳ 结果对比报告完成
3. ⏳ 统计显著性测试 (t-test or Wilcoxon)
4. ⏳ 性能改进 vs 计算成本权衡分析

---

## 📝 变更日志

### 2025-12-09: Phase 2 Integration Complete
- ✅ train_with_kv.py 双模式集成完成
- ✅ test_v4_integration.py 冒烟测试脚本创建
- ✅ V4_INTEGRATION_COMPLETE.md 文档创建
- ⏳ 等待测试验证

---

**状态**: 🟡 Ready for Testing  
**下一步**: 运行 `python experiments/test_v4_integration.py`
