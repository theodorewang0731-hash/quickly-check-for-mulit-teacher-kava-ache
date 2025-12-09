# Phase 2 完成状态报告 - 总控面板

**日期**: 2025-12-09  
**阶段**: Phase 2 - 训练脚本集成  
**状态**: ✅ **代码集成完成，等待测试验证**

---

## 🎯 Phase 2 / v4.0 Map Projection – 成功标准总表

### 核心 Checklist（可以直接挂在桌前）

- ✅ **代码集成无语法错误** → 详见 [`V4_INTEGRATION_COMPLETE.md`](V4_INTEGRATION_COMPLETE.md)
- ✅ **Git 推送成功** → 远程分支已更新 (Commit: fb36290)
- ⏳ **test_v4_integration.py 全部通过** → 第一关，单元集成测试
- ⏳ **10-step 冒烟测试成功** → 第二关，flat + structured 双模式
- ⏳ **A/B 实验完成** → 第三关，GPT-2 本地 + Qwen2 HPC 主实验

### 文档索引（四层体系）

| 文档 | 用途 | 对应阶段 |
|------|------|---------|
| [`V4_INTEGRATION_COMPLETE.md`](V4_INTEGRATION_COMPLETE.md) | 📋 **我都改了啥** | 溯源代码变更 |
| [`V4_EXECUTION_ROADMAP.md`](V4_EXECUTION_ROADMAP.md) | 🗺️ **整体怎么走** | Phase 2/2.5/3 时间线 |
| [`PHASE2_STATUS_REPORT.md`](PHASE2_STATUS_REPORT.md) | 📍 **现在在哪** | 当前进度盘点（本文件） |
| [`V4_QUICK_REFERENCE.md`](V4_QUICK_REFERENCE.md) | ⚡ **命令怎么敲** | 实战速查表 |

---

## 📊 完成度统计

### 整体进度
```
Phase 1 (核心模块开发):     ████████████████████ 100% ✅
Phase 2 (训练脚本集成):     ████████████████████ 100% ✅
Phase 2.5 (测试验证):       ░░░░░░░░░░░░░░░░░░░░   0% ⏳
Phase 3 (A/B 实验):         ░░░░░░░░░░░░░░░░░░░░   0% ⏳
```

### 代码变更统计
- **新增文件**: 2 个
  - `experiments/test_v4_integration.py` (230 lines)
  - `scripts/test_v4_quick.sh` (80 lines)
  
- **修改文件**: 2 个
  - `experiments/train_with_kv.py` (+150 lines)
  - `README.md` (更新 v4.0 状态)

- **新增文档**: 1 个
  - `V4_INTEGRATION_COMPLETE.md` (500+ lines)

- **总代码量**: +960 lines

---

## 🧪 三关通关指南

### ⏳ 第一关：test_v4_integration.py 全部通过

**目的**: 验证 v4.0 代码路径（模块 import、对齐函数、CLI 参数）联通，属于**单元 + 集成级**测试。

**怎么做**:
1. 打开 [`V4_QUICK_REFERENCE.md`](V4_QUICK_REFERENCE.md) / [`V4_EXECUTION_ROADMAP.md`](V4_EXECUTION_ROADMAP.md)，找到 "Integration Test" 那一节
2. 在项目根目录执行:
   ```bash
   cd /Users/alexwang/quickly-check-for-mulit-teacher-kava-ache
   python experiments/test_v4_integration.py
   ```

**通过标准**:
- ✅ 所有断言/测试通过（exit code 0）
- ✅ 日志确认：
  - `MapProjectionAligner` 能成功初始化
  - `stack_past_kv` 输出形状正确
  - `alignment_mode=flat/structured` 配对的分支都能跑一遍 forward

**预期时间**: 10 秒

---

### ⏳ 第二关：10-step 冒烟测试（flat + structured）

**目的**: 真正跑一遍完整训练循环（Teacher → Student → 对齐 → KV loss → backward），确保训练脚本逻辑没问题。
对应 [`V4_EXECUTION_ROADMAP.md`](V4_EXECUTION_ROADMAP.md) 里的 **Phase 2.5：Pipeline Smoke Test**。

**参考命令**（来自 [`V4_QUICK_REFERENCE.md`](V4_QUICK_REFERENCE.md)）:

#### 2.1 Flat 模式 smoke test
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

#### 2.2 Structured 模式 smoke test
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

**通过标准**:
- ✅ 两个命令都能完整跑完（10 步左右），没有 RuntimeError / 维度错配
- ✅ 日志里有清晰的 mode 标记（flat / structured），说明分支切换正确
- ✅ `loss_task`, `loss_kv` 都是正常数值（不是 NaN/Inf）
- ✅ 检查点保存成功（含 `map_aligner.pt` for structured mode）

**预期时间**: 2 分钟 × 2

---

### ⏳ 第三关：A/B 实验完成（flat vs structured）

这是整个 Phase 2 的终极目标，对应 [`V4_EXECUTION_ROADMAP.md`](V4_EXECUTION_ROADMAP.md) 和 [`PHASE2_STATUS_REPORT.md`](PHASE2_STATUS_REPORT.md) 里的 **Phase 3**。

#### 3.1 第一轮：本地/轻量模型 A/B（GPT-2 + 小数据）

**目的**: 用最小成本验证：**在同样的 loss 设置下，v4.0 结构化对齐是否至少不比 flat 差，有没有更快收敛的趋势**。

**Baseline（flat）**:
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

**v4.0（structured 推荐版）**:
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
    --output_dir outputs/ab_structured_uniform \
    --logging_steps 10 \
    --save_steps 200
```

**对比内容**:
- `loss_task` / `loss_kv` 随 step 的下降速度、稳定性
- 如果有简单 dev 集，可以看下 PPL 或准确率

**预期时间**: 2-4 小时（取决于硬件）

#### 3.2 第二轮：HPC 上的 Qwen2 主实验（可以晚一点做）

当 GPT-2 上的趋势结果出来之后，切到:
- **Teacher**: `Qwen2-1.5B`
- **Student**: `Qwen2-0.5B`（先做这一组，用作"正式单教师实验"）
- **数据**: Reasoning mix（GSM8K / SVAMP / StrategyQA / ARC 等）

同样跑:
- `alignment_mode=flat`
- `alignment_mode=structured` + `map_proj_share_dim` + `map_proj_init_uniform`

然后把结果写进 `V4_AB_TEST_RESULTS.md` 或更新到 [`DEVELOPMENT_HISTORY.md`](DEVELOPMENT_HISTORY.md) 的 Phase 4 小节。

---

## ✅ Phase 2 完成清单

### 2.1 代码集成
- [x] 导入 MapProjectionAligner
- [x] 添加 stack_past_kv 工具函数
- [x] 添加 3 个命令行参数
  - `--alignment_mode {flat,structured}`
  - `--map_proj_share_dim`
  - `--map_proj_init_uniform`
- [x] 双模式 Aligner 初始化逻辑
- [x] 训练循环双模式分支 (if/else)
- [x] 日志输出更新 (显示模式标记)
- [x] 检查点保存更新 (包含 map_aligner.pt)
- [x] 训练完成报告更新

### 2.2 测试工具准备
- [x] 创建 test_v4_integration.py (5 个测试)
- [x] 创建 test_v4_quick.sh (自动化测试脚本)
- [x] 更新 README.md (v4.0 状态)
- [x] 创建 V4_INTEGRATION_COMPLETE.md (完整文档)

### 2.3 待执行测试
- [ ] 运行 test_v4_integration.py
- [ ] 运行 profile_alignment.py (flat & structured)
- [ ] 10-step 冒烟测试 (flat mode)
- [ ] 10-step 冒烟测试 (structured mode)

---

## 🔑 关键技术实现

### 1. 双模式架构设计

**设计原则**: 控制变量，最小化差异
- **相同**: Loss 函数 (MSE)、优化器、训练流程
- **不同**: 仅对齐方式 (flat 展平 vs structured 保持结构)

**实现方式**: if/else 分支
```python
if args.alignment_mode == "structured":
    # v4.0: Map Projection Alignment
    aligned_k, aligned_v, _ = map_aligner(teacher_k, teacher_v, None, segment_ids)
    kv_loss = (F.mse_loss(aligned_k, student_k) + F.mse_loss(aligned_v, student_v)) / 2
else:
    # Baseline: Flat alignment
    for layer_idx, layer in enumerate(comp):
        tk, student_seg = align_teacher_kv_to_student(...)
        student_proj = projectors[layer_idx](student_seg)
        l = compute_kv_loss(student_proj, tk, ...)
        layer_losses.append(l)
    kv_loss = torch.stack(layer_losses).mean()
```

### 2. 懒加载初始化

**问题**: 不同模型的维度不同 (GPT-2 vs Qwen vs LLaMA)

**解决方案**: 首次 batch 时根据实际数据初始化
```python
if map_aligner is None and args.alignment_mode == "structured":
    # 从首个 batch 提取维度
    sample_k, sample_v = comp[0]
    num_teacher_heads = sample_k.shape[1]
    teacher_head_dim = sample_k.shape[-1]
    
    # 动态初始化
    map_aligner = MapProjectionAligner(
        num_teacher_layers=len(comp),
        num_student_layers=student.config.num_hidden_layers,
        num_teacher_heads=num_teacher_heads,
        num_student_heads=student.config.num_attention_heads,
        teacher_head_dim=teacher_head_dim,
        student_head_dim=student.config.hidden_size // student.config.num_attention_heads,
        mode="structured",
        share_dim_proj=args.map_proj_share_dim,
        init_uniform=args.map_proj_init_uniform
    ).to(device)
```

### 3. Student KV 获取策略

**挑战**: Structured mode 需要 student 的 past_key_values

**当前实现**: 额外一次 forward
```python
with torch.no_grad():
    s_out_kv = student(input_ids, attention_mask, use_cache=True)
student_pkv = s_out_kv.past_key_values
```

**成本**: 
- 计算: +1x student forward (~10-20% overhead)
- 内存: +1x KV cache storage

**优化方向** (可选):
- 在主 forward 时直接获取 KV cache
- 重用已计算的 KV cache

---

## 🧪 测试计划

### Phase 2.5: 冒烟测试

#### 测试 1: 模块集成
```bash
python experiments/test_v4_integration.py
```

**检查项**:
- [x] 模块导入无错误
- [ ] stack_past_kv 功能正确
- [ ] Aligner 初始化成功
- [ ] 完整对齐流程无 NaN

**预期时间**: 10 秒

#### 测试 2: Profile Alignment
```bash
python experiments/profile_alignment.py --mode flat
python experiments/profile_alignment.py --mode structured
```

**检查项**:
- [ ] 形状对齐正确
- [ ] 参数量统计合理
- [ ] Attention 分布正常

**预期时间**: 30 秒 x 2

#### 测试 3: 10-Step 训练冒烟
```bash
# Flat
python experiments/train_with_kv.py \
    --model_name gpt2 \
    --subset_size 10 \
    --batch_size 2 \
    --epochs 1 \
    --alignment_mode flat \
    --output_dir outputs/smoke_flat

# Structured
python experiments/train_with_kv.py \
    --model_name gpt2 \
    --subset_size 10 \
    --batch_size 2 \
    --epochs 1 \
    --alignment_mode structured \
    --map_proj_share_dim \
    --map_proj_init_uniform \
    --output_dir outputs/smoke_structured
```

**检查项**:
- [ ] 训练完成无报错
- [ ] Loss 正常（非 NaN/Inf）
- [ ] 日志显示正确模式
- [ ] 检查点保存成功

**预期时间**: 2 分钟 x 2

#### 快速测试脚本
```bash
bash scripts/test_v4_quick.sh
```
自动运行上述所有测试。

---

### Phase 3: A/B 实验

#### 实验配置

| 参数 | Baseline | v4.0 |
|------|----------|------|
| alignment_mode | flat | structured |
| map_proj_share_dim | N/A | ✓ |
| map_proj_init_uniform | N/A | ✓ |
| 数据量 | 1000 | 1000 |
| Epochs | 3 | 3 |
| Batch size | 8 | 8 |
| Learning rate | 5e-5 | 5e-5 |

#### 评估指标

**主指标**:
1. Final CE Loss (越低越好)
2. Final KV Loss (越低越好)
3. Training stability (loss 曲线平滑度)

**辅助指标**:
1. 参数量 (params count)
2. 训练时间 (wall-clock time)
3. 内存占用 (peak memory)

**分析方法**:
- 配对 t 检验 (p < 0.05)
- Cohen's d (效应量)
- Loss 曲线可视化

---

## 📁 文件组织

### 新增/修改文件清单

```
experiments/
  ├── train_with_kv.py              🔧 修改 (+150 lines, 双模式集成)
  ├── test_v4_integration.py        ✨ 新增 (230 lines, 冒烟测试)
  └── profile_alignment.py          📝 已存在 (Phase 1 创建)

scripts/
  └── test_v4_quick.sh              ✨ 新增 (80 lines, 自动化测试)

根目录/
  ├── V4_INTEGRATION_COMPLETE.md    ✨ 新增 (500 lines, 集成报告)
  ├── V4_EXECUTION_ROADMAP.md       📝 已存在 (Phase 1 创建)
  ├── DEVELOPMENT_HISTORY.md        📝 已存在 (Phase 1 创建)
  └── README.md                     🔧 修改 (更新 v4.0 状态)
```

### Git 状态
```bash
# 待提交文件
M  experiments/train_with_kv.py
A  experiments/test_v4_integration.py
A  scripts/test_v4_quick.sh
A  V4_INTEGRATION_COMPLETE.md
A  PHASE2_STATUS_REPORT.md
M  README.md
```

---

## 🚀 下一步行动

### 立即执行（今天）
1. **提交当前代码**
   ```bash
   git add .
   git commit -m "feat(v4.0): Phase 2 integration complete - dual-mode aligner in train_with_kv.py"
   git push origin main
   ```

2. **运行冒烟测试**
   ```bash
   bash scripts/test_v4_quick.sh
   ```

3. **如果测试通过**
   - 更新 PHASE2_STATUS_REPORT.md (标记测试为 ✅)
   - 准备 A/B 实验配置

### 短期计划（本周）
1. **完成 A/B 实验**
   - Baseline (flat mode): 1000 samples, 3 epochs
   - v4.0 (structured mode): 同上
   - 收集结果并分析

2. **编写分析报告**
   - Loss 曲线对比
   - 统计显著性测试
   - 性能 vs 成本权衡

### 中期计划（下周）
1. **优化改进** (如果 v4.0 表现好)
   - 减少 student forward 次数
   - 支持真实 segment 标注
   - 接入 StructuralKVLoss

2. **扩展实验**
   - 更大模型 (Qwen-1.5B)
   - 更多数据 (10k samples)
   - 多 teacher 场景

---

## ⚠️ 风险和缓解

### 风险 1: 测试可能失败
**概率**: 中等  
**影响**: 高 (需要 debug)  
**缓解**:
- 分步测试 (不要一次运行全部)
- 保留详细日志
- 回滚到 Phase 1 已验证状态

### 风险 2: v4.0 性能未必更好
**概率**: 中等  
**影响**: 低 (仍有学术价值)  
**缓解**:
- A/B 实验提供客观对比
- 分析失败原因（学术贡献）
- 保留 flat mode 作为 baseline

### 风险 3: 计算资源不足
**概率**: 低  
**影响**: 中 (延长实验时间)  
**缓解**:
- 使用小规模实验 (1000 samples)
- 申请 HPC 资源
- GPU 优化 (混合精度、梯度累积)

---

## 📝 提交消息建议

```
feat(v4.0): Phase 2 integration complete - dual-mode aligner in train_with_kv.py

Major changes:
- Add MapProjectionAligner integration to train_with_kv.py
- Implement dual-mode training loop (flat vs structured)
- Add 3 new CLI arguments for v4.0 configuration
- Create comprehensive smoke test (test_v4_integration.py)
- Add automated test script (test_v4_quick.sh)

Files changed:
- experiments/train_with_kv.py (+150 lines)
- experiments/test_v4_integration.py (new, 230 lines)
- scripts/test_v4_quick.sh (new, 80 lines)
- V4_INTEGRATION_COMPLETE.md (new, 500 lines)
- PHASE2_STATUS_REPORT.md (new, 400 lines)
- README.md (updated v4.0 status)

Status:
- Phase 1 (Core modules): ✅ Complete
- Phase 2 (Integration): ✅ Complete
- Phase 2.5 (Testing): ⏳ Pending
- Phase 3 (A/B experiments): ⏳ Pending

Next steps:
1. Run bash scripts/test_v4_quick.sh
2. Launch A/B experiments if tests pass
3. Analyze results and update docs

See V4_INTEGRATION_COMPLETE.md for full details.
```

---

## 🎯 成功标准回顾

### Phase 2 目标
- [x] train_with_kv.py 双模式集成 ✅
- [x] 命令行参数添加 ✅
- [x] 训练循环分支实现 ✅
- [x] 测试工具准备 ✅
- [ ] 冒烟测试通过 ⏳

### 完成判定
- **代码层面**: ✅ 已完成 (无语法错误)
- **功能层面**: ⏳ 等待测试验证
- **文档层面**: ✅ 已完成 (完整文档)

---

**当前状态**: 🟢 Code Complete, 🟡 Testing Pending  
**下一步**: 运行 `bash scripts/test_v4_quick.sh`  
**预计完成 Phase 2.5**: 2025-12-09 晚上  
**预计启动 Phase 3**: 2025-12-10
