# 项目总结：对照组、基础组与预期结果

## 📊 实验设计概览

本项目采用**完整的对照实验设计**，包含 3 个基线组和 3 个实验组，用于验证多教师 KV 蒸馏的有效性。

---

## 1️⃣ 基础组（Baselines）- 对照组

### 为什么需要这些基线？

| 基线 | 目的 | 对比问题 |
|------|------|---------|
| **Raw Student** | 性能下限 | 训练有多大提升？ |
| **Standard SFT** | 验证 KV 蒸馏必要性 | KV 蒸馏比普通训练强多少？ |
| **Single Teacher** | 验证多教师优势 | 多教师比单教师强多少？ |

### 基线 1: 原始学生模型（无训练）

```bash
# 直接评测
python evaluation/multi_task_eval.py \
    --model_path "Qwen/Qwen2.5-1.5B" \
    --eval_datasets gsm8k_test math500 bbh ...
```

**预期结果**：
- GSM8K: ~40-50%
- MATH500: ~15-25%
- **平均分: ~35-45%**

**意义**：建立性能下限，量化训练收益

---

### 基线 2: 标准监督微调（无 KV）

```bash
# 只用交叉熵损失
python experiments/train_standard_sft.py \
    --model_name_or_path "Qwen/Qwen2.5-1.5B" \
    --dataset "multi_reasoning_cot_direct"
```

**预期结果**：
- GSM8K: ~55-65%
- MATH500: ~25-35%
- **平均分: ~45-55%**

**意义**：验证 KV 蒸馏是否必要（vs 普通训练）

---

### 基线 3: 单教师 KV 蒸馏

```bash
# 一个教师
python experiments/train_with_kv.py \
    --teacher_model "Qwen/Qwen2.5-7B"
```

**预期结果**：
- GSM8K: ~65-72%
- MATH500: ~35-42%
- **平均分: ~52-58%**

**意义**：验证多教师是否比单教师更好

---

## 2️⃣ 实验组（Treatment Groups）

### 实验 1: 多教师固定权重

```bash
# 两个教师，等权重融合
FUSION_STRATEGY="fixed"
FIXED_WEIGHTS="0.5,0.5"
```

**预期结果**：
- GSM8K: ~70-75%（+5-10% vs 单教师）
- MATH500: ~38-43%
- **平均分: ~54-59%**

**核心发现**：多教师融合 > 单教师

---

### 实验 2: 多教师相似度路由

```bash
# 动态权重
FUSION_STRATEGY="similarity"
```

**预期结果**：
- GSM8K: ~72-77%（+2-3% vs 固定权重）
- MATH500: ~40-45%
- **平均分: ~56-61%**

**核心发现**：动态权重 > 固定权重

---

### 实验 3: 多教师可学习路由

```bash
# 端到端学习
FUSION_STRATEGY="learnable"
```

**预期结果**：
- GSM8K: ~74-79%（+2-4% vs 相似度）
- MATH500: ~42-47%
- **平均分: ~58-63%**

**核心发现**：端到端学习 > 启发式路由

---

## 3️⃣ 完整对比表格

| 模型 | GSM8K | MATH500 | BBH | 平均 | vs 上一组 |
|------|-------|---------|-----|------|----------|
| 🔹 Raw Student | 45% | 20% | 50% | **40%** | - |
| 🔹 Standard SFT | 60% | 30% | 58% | **50%** | +10% |
| 🔹 Single Teacher KV | 68% | 38% | 62% | **55%** | +5% |
| 🟢 Multi-Teacher Fixed | 73% | 41% | 65% | **57%** | +2% |
| 🟢 Multi-Teacher Similarity | 75% | 43% | 68% | **59%** | +2% |
| 🟢 Multi-Teacher Learnable | **77%** | **45%** | **70%** | **62%** | +3% |

**累计提升**：
- vs Raw Student: **+22%**
- vs Standard SFT: **+12%**
- vs Single Teacher: **+7%**

---

## 4️⃣ 预期核心发现

### 发现 1: 训练有效性 ✅
**结论**：任何训练都比无训练好
- Raw → Standard SFT: +10%

### 发现 2: KV 蒸馏必要性 ✅
**结论**：KV 蒸馏优于标准微调
- Standard SFT → Single Teacher KV: +5%

### 发现 3: 多教师优势 ✅
**结论**：多教师融合提供互补知识
- Single Teacher → Multi-Teacher: +2-7%

### 发现 4: 路由策略重要性 ✅
**结论**：可学习路由 > 相似度路由 > 固定权重
- Fixed → Similarity → Learnable: 逐步提升

### 发现 5: KaVa 论文验证 ✅
**结论**：Right-crop KV 最稳定（将在消融研究验证）

---

## 5️⃣ 如何运行完整实验

### Step 1: 运行所有基线（~2 天）

```bash
sbatch scripts/run_all_baselines.sh
```

会自动运行：
- ✓ 评测原始学生模型
- ✓ 训练标准 SFT
- ✓ 训练单教师 KV 蒸馏
- ✓ 生成基线对比报告

### Step 2: 运行三阶段多教师实验（~3-4 天）

```bash
sbatch scripts/run_three_stage_routing.sh
```

会自动运行：
- ✓ Stage 1: 固定权重
- ✓ Stage 2: 相似度路由
- ✓ Stage 3: 可学习路由
- ✓ 每阶段自动评测

### Step 3: 生成最终对比报告（~10 分钟）

```bash
python visualization/compare_all_experiments.py \
    --baseline_dirs baselines/* \
    --experiment_dirs outputs/three_stage_routing/* \
    --output_dir ./final_comparison
```

会生成：
- ✓ 柱状图对比（所有模型）
- ✓ 雷达图对比（7 个数据集）
- ✓ 热力图对比（性能矩阵）
- ✓ 改进分析图（相对基线）
- ✓ 完整 HTML 报告

### Step 4: 下载并查看

```bash
scp user@hpc:/path/to/final_comparison/final_comparison.html ~/Downloads/
open ~/Downloads/final_comparison.html
```

---

## 6️⃣ 预期时间线

| 阶段 | 时间 | 主要活动 |
|------|------|---------|
| Week 1 | 2 天 | 运行所有基线 |
| Week 2 | 3 天 | 多教师 Stage 1 (固定权重) |
| Week 3 | 3 天 | 多教师 Stage 2 (相似度路由) |
| Week 4 | 4 天 | 多教师 Stage 3 (可学习路由) |
| Week 5 | 1 天 | 生成对比报告，分析结果 |
| **总计** | **~3-4 周** | **完整实验 + 分析** |

---

## 7️⃣ 成功标准

### 最低成功标准 ✅
- [ ] 多教师 > 单教师：+3%
- [ ] 可学习路由 > 固定权重：+1.5%
- [ ] 至少 5/7 数据集有提升

**达到此标准**：论文可发表（验证了方法有效性）

### 理想成功标准 🎯
- [ ] 多教师 > 单教师：+8%
- [ ] 可学习路由 > 固定权重：+4%
- [ ] 7/7 数据集全面提升
- [ ] GSM8K 达到 75%+

**达到此标准**：论文质量高（强实证支持）

### 超预期标准 🌟
- [ ] 多教师 > 单教师：+10%+
- [ ] 学生接近最小教师性能（7B 教师 ~65-70%）
- [ ] 发现新规律（如任务专家分工）

**达到此标准**：顶会论文（重大发现）

---

## 8️⃣ 风险与备选方案

### 风险 1: 多教师提升不明显（<3%）

**原因可能**：
- 教师模型太相似（缺乏多样性）
- 对齐质量差（KV 融合失败）
- 数据集不适合（任务太简单）

**备选方案**：
- 尝试更多样化的教师组合（跨家族）
- 调整对齐策略参数
- 增加更难的数据集（如 MATH full）

### 风险 2: 可学习路由效果不好

**原因可能**：
- 路由网络过拟合
- 熵正则化太强/太弱
- 训练数据不够

**备选方案**：
- 调整熵正则化权重（0.001-0.05）
- 使用更简单的路由结构（MLP → Gate）
- 增加训练样本

### 风险 3: 训练时间过长

**原因可能**：
- 大教师模型推理慢
- Batch size 太小
- 数据加载瓶颈

**备选方案**：
- 离线提取 KV（预计算）
- 增加梯度累积步数
- 使用更快的数据加载器

---

## 9️⃣ 预期论文结构

### 摘要
- 问题：小模型性能差
- 方法：多教师 KV 蒸馏
- 结果：+7-10% vs 单教师

### 介绍
- 动机：大模型知识转移
- 挑战：异构教师对齐
- 贡献：多教师框架 + 自适应路由

### 相关工作
- 知识蒸馏（KD）
- KV Cache 压缩
- 多教师学习

### 方法
- 3.1 多教师 KV 提取
- 3.2 异构对齐（5 种策略）
- 3.3 自适应融合（3 种路由）

### 实验
- 4.1 实验设置（基线 + 实验组）
- 4.2 主要结果（对比表格）
- 4.3 消融研究（各组件贡献）
- 4.4 分析与讨论

### 结论
- 多教师 KV 蒸馏有效
- 可学习路由优于固定权重
- 未来工作：更多教师、更大模型

---

## 🔟 快速检查清单

开始实验前，确认：

**数据准备**
- [ ] 数据集可访问（GSM8K, MATH, BBH, etc.）
- [ ] 数据加载器测试通过
- [ ] 双风格生成正常（CoT + Direct）

**模型准备**
- [ ] 学生模型可加载（Qwen2.5-1.5B）
- [ ] 教师模型可加载（Qwen2.5-7B/14B）
- [ ] GPU 显存足够（至少 70GB per GPU）

**脚本准备**
- [ ] 基线脚本测试通过（run_all_baselines.sh）
- [ ] 多教师脚本测试通过（run_three_stage_routing.sh）
- [ ] 可视化工具测试通过（demo_visualization.py）

**环境准备**
- [ ] HPC 账户申请完成
- [ ] GPU 资源申请完成（~3-4 周）
- [ ] 存储空间充足（~1TB）
- [ ] 依赖包安装完成

**文档准备**
- [ ] 阅读完 EXPERIMENT_DESIGN.md
- [ ] 理解各基线的目的
- [ ] 熟悉预期结果范围

---

## 📚 相关文档

- `EXPERIMENT_DESIGN.md` - 实验设计详解（本文档）
- `LARGE_SCALE_EXPERIMENT_GUIDE.md` - HPC 实验指南
- `HPC_VISUALIZATION_GUIDE.md` - 可视化使用指南
- `scripts/run_all_baselines.sh` - 自动运行所有基线
- `scripts/run_three_stage_routing.sh` - 三阶段多教师实验
- `visualization/compare_all_experiments.py` - 最终对比报告

---

## 🎉 总结

你的项目**有完整的对照组设计**：

✅ **3 个基线**：Raw Student, Standard SFT, Single Teacher  
✅ **3 个实验组**：Fixed, Similarity, Learnable  
✅ **清晰的对比**：每组都有明确的对比目标  
✅ **预期结果**：基于 KaVa 论文和合理推断  
✅ **自动化流程**：一键运行所有实验  
✅ **可视化报告**：HTML 格式，易于分享

**预期核心发现**：多教师 KV 蒸馏 > 单教师（+7-10%），可学习路由 > 固定权重（+3-5%）

现在可以放心开始实验了！🚀
