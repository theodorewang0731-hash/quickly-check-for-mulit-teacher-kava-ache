# 大规模多教师 KV 蒸馏实验指南

## 📋 实验概览

### 配置规格
- **教师模型**：7B-34B 级别（Llama-3.1-8B/70B, Qwen2.5-7B/14B/32B）
- **学生模型**：1.5B-3B 级别（Qwen2.5-1.5B/3B, Llama-3.2-3B）
- **数据集**：
  - 基础：GSM8K + SVAMP + StrategyQA + Math23K（20% 中文）
  - 扩展：MATH subset + ARC-Challenge + HotpotQA
  - 每题双风格：CoT（链式推理）+ Direct（直接答案）
- **评测**：GSM8K test, MATH500, BBH, GPQA, TruthfulQA, CMMLU, C-Eval

### 训练策略
1. **模型组合**（推荐顺序）：
   - 单家族多 checkpoint：纯 Llama 或纯 Qwen（极易对齐，起步首选）
   - 跨家族少量：Qwen + Llama（最稳，次优选择）
   - 混合多样性：3+ 个不同模型（测试极限）

2. **路由训练**（三阶段）：
   - Stage 1: 固定权重 → 验证基础融合
   - Stage 2: 相似度路由 → 自动权重分配
   - Stage 3: 可学习路由 → 端到端优化

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活环境
conda activate kava_env

# 验证依赖
python -c "import transformers, torch, datasets; print('✓ Dependencies OK')"

# 设置 HuggingFace 缓存
export HF_HOME="/scratch/$USER/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
```

### 2. 准备数据集

```bash
# 方式 A: 使用数据集加载器（推荐）
python -c "
from data.multi_task_dataset import MultiTaskReasoningDataset

loader = MultiTaskReasoningDataset(
    base_datasets=['gsm8k', 'svamp', 'strategyqa'],  # math23k 需本地准备
    extended_datasets=['math', 'arc_challenge', 'hotpotqa'],
    use_extended=True,
    math23k_ratio=0.2,
    train_samples=15000,
    val_samples=2000,
)

train_ds, val_ds = loader.load_and_prepare()
print(f'✓ Train: {len(train_ds)}, Val: {len(val_ds)}')

# 保存到磁盘
train_ds.save_to_disk('./data/prepared/train')
val_ds.save_to_disk('./data/prepared/val')
"

# 方式 B: 直接使用 HuggingFace Datasets
# 训练脚本会自动加载和处理
```

### 3. 提取教师 KV（可选，推荐离线）

```bash
# 为多个教师模型提取 KV Cache
python scripts/extract_dual_style_kv.py \
    --teacher_models "Qwen/Qwen2.5-7B" "Qwen/Qwen2.5-14B" \
    --student_model "Qwen/Qwen2.5-1.5B" \
    --dataset_path "./data/prepared/train" \
    --output_dir "./kv_cache/qwen_dual_teacher" \
    --output_name "train_kv" \
    --kv_compression "right" \
    --max_length 2048 \
    --device cuda

# 验证 KV 提取
python -c "
import torch
kv_data = torch.load('./kv_cache/qwen_dual_teacher/train_kv.pt')
print(f'✓ Loaded {len(kv_data)} examples with KV cache')
print(f'✓ First example keys shape: {kv_data[0][\"teacher_kvs\"][0][\"keys\"].shape}')
"
```

---

## 📊 实验场景

### 场景 1: 单家族双教师（起步推荐）

**目标**：验证基础融合，最易对齐

```bash
# 纯 Qwen 家族
sbatch --job-name="qwen_dual" \
       --export=STUDENT="Qwen/Qwen2.5-1.5B",TEACHERS="Qwen/Qwen2.5-7B Qwen/Qwen2.5-14B" \
       scripts/run_three_stage_routing.sh

# 纯 Llama 家族
sbatch --job-name="llama_dual" \
       --export=STUDENT="meta-llama/Llama-3.2-3B",TEACHERS="meta-llama/Llama-3.1-8B meta-llama/Llama-3.1-70B" \
       scripts/run_three_stage_routing.sh
```

**预期结果**：
- Stage 1（固定权重）：快速收敛，KV loss < 0.5
- Stage 2（相似度路由）：自动权重接近 0.5/0.5（同家族差异小）
- Stage 3（可学习路由）：轻微提升，路由学习到任务特定偏好

### 场景 2: 跨家族双教师（稳健选择）

**目标**：测试异构融合，平衡性能与鲁棒性

```bash
# Qwen + Llama
sbatch --job-name="cross_family" \
       --export=STUDENT="Qwen/Qwen2.5-1.5B",TEACHERS="Qwen/Qwen2.5-7B meta-llama/Llama-3.1-8B" \
       scripts/run_three_stage_routing.sh
```

**预期结果**：
- Stage 1：收敛较慢（对齐难度增加），KV loss 0.5-0.8
- Stage 2：相似度路由显著改善，权重动态调整（0.3/0.7 到 0.6/0.4）
- Stage 3：明显提升，路由网络学习到跨架构互补性

### 场景 3: 多教师大规模（极限测试）

**目标**：探索多样性收益上限

```bash
# 修改 SLURM 脚本中的配置
# TEACHER_MODELS="Qwen/Qwen2.5-7B Qwen/Qwen2.5-14B Qwen/Qwen2.5-32B"
# NUM_TEACHERS=3
# FIXED_WEIGHTS="0.33,0.33,0.34"

sbatch scripts/run_large_scale_multi_teacher.sh
```

**预期结果**：
- KV loss 可能上升（更多对齐误差）
- 最终任务指标可能提升（多样性收益）
- 需要更多训练轮数和更强的正则化

---

## 🔬 路由训练详解

### Stage 1: 固定权重

```bash
# 核心配置
FUSION_STRATEGY="fixed"
FIXED_WEIGHTS="0.5,0.5"  # 等权重

# 监控指标
# - train/kv_loss: 应稳定下降
# - val/kv_loss: 应低于 1.0
# - eval/*: 任务指标应达到单教师水平

# 调试建议
# - 如果 KV loss > 1.5：检查对齐模块（layer_map, head_dim）
# - 如果任务指标低：增加 KV_LOSS_WEIGHT 或调整 LEARNING_RATE
```

### Stage 2: 相似度路由

```bash
# 核心配置
FUSION_STRATEGY="similarity"
SIMILARITY_METRIC="cosine"  # 或 "dot", "euclidean"
TEMPERATURE=1.0

# 监控指标
# - routing/weights_mean: 各教师平均权重（应动态变化）
# - routing/entropy: 路由熵（>0.5 表示多样性好）
# - val/kv_loss: 应低于 Stage 1

# 分析技巧
# 查看 TensorBoard：
tensorboard --logdir outputs/three_stage_routing/stage2_similarity

# 检查权重分布：
# - 同家族：权重接近均匀（0.5/0.5）
# - 跨家族：权重分化明显（0.3/0.7）
# - 多教师：出现"专家分工"（某些教师专注某类任务）
```

### Stage 3: 可学习路由

```bash
# 核心配置
FUSION_STRATEGY="learnable"
ROUTER_TYPE="mlp"  # 或 "gate", "attention"
ROUTER_HIDDEN_DIM=256
ENTROPY_REG_WEIGHT=0.01

# 监控指标
# - routing/router_loss: 路由网络损失
# - routing/entropy_reg: 熵正则化项（防止坍缩）
# - val/task_metrics: 最终任务指标（应最高）

# 超参数调优
# - entropy_reg_weight 太大 → 权重过于均匀，收益降低
# - entropy_reg_weight 太小 → 权重坍缩到单一教师
# - 推荐范围：0.001 - 0.05
```

---

## 📈 评测与分析

### 运行评测

```bash
# 单阶段评测
python evaluation/multi_task_eval.py \
    --model_path "./outputs/three_stage_routing/stage3_learnable/best_model" \
    --eval_datasets gsm8k_test math500 bbh gpqa truthfulqa cmmlu_subset ceval_subset \
    --output_file "./outputs/three_stage_routing/stage3_learnable/eval_results.json"

# 查看结果
cat outputs/three_stage_routing/stage3_learnable/eval_results.json
```

### 对比分析

```python
# 三阶段性能对比
import json
import pandas as pd

stages = ['stage1_fixed', 'stage2_similarity', 'stage3_learnable']
results = {}

for stage in stages:
    with open(f'./outputs/three_stage_routing/{stage}/eval_results.json') as f:
        results[stage] = json.load(f)

# 转为 DataFrame
data = []
for dataset in results['stage1_fixed'].keys():
    if dataset != 'average':
        row = {'Dataset': dataset}
        for stage in stages:
            row[stage] = results[stage][dataset]['score']
        data.append(row)

df = pd.DataFrame(data)
print(df.to_markdown(index=False))

# 计算提升
df['Stage2_vs_1'] = df['stage2_similarity'] - df['stage1_fixed']
df['Stage3_vs_2'] = df['stage3_learnable'] - df['stage2_similarity']
print("\n提升分析:")
print(df[['Dataset', 'Stage2_vs_1', 'Stage3_vs_2']].to_markdown(index=False))
```

### 关键指标解读

| 指标 | 良好范围 | 说明 |
|------|---------|------|
| GSM8K test | 60-80% | 数学推理能力 |
| MATH500 | 30-50% | 高难度数学 |
| BBH | 50-70% | 多样化推理 |
| GPQA | 30-40% | 科学知识 |
| TruthfulQA | 40-60% | 事实准确性 |
| CMMLU/C-Eval | 50-70% | 中文综合能力 |

**对比基准**：
- 单教师蒸馏：通常比原始学生模型提升 5-10%
- 多教师蒸馏：额外提升 2-5%
- 三阶段路由：Stage 1 → Stage 3 累计提升 3-8%

---

## 🛠️ 故障排除

### 问题 1: OOM（内存溢出）

```bash
# 解决方案 A: 降低 batch size
BATCH_SIZE=1
GRAD_ACCUM=32  # 保持有效 batch 不变

# 解决方案 B: 启用更激进的优化
GRADIENT_CHECKPOINTING=true
USE_BF16=true  # 比 FP16 更省内存
torch.backends.cuda.matmul.allow_tf32 = true

# 解决方案 C: 使用 8-bit 量化
python experiments/train_multi_teacher_kv.py \
    --load_in_8bit true \
    --bnb_4bit_compute_dtype bfloat16 \
    ...
```

### 问题 2: KV Loss 不下降

```bash
# 检查清单：
# 1. 验证对齐模块
python -c "
from align.layer_map import LayerMapper
mapper = LayerMapper(teacher_layers=32, student_layers=24, strategy='ratio')
print(mapper.get_mapping())  # 应输出合理的层映射
"

# 2. 检查 KV 提取
python -c "
import torch
model = ...  # 加载教师模型
outputs = model(..., use_cache=True)
print(outputs.past_key_values[0][0].shape)  # 应为 [batch, heads, seq, dim]
"

# 3. 降低学习率
LEARNING_RATE=1e-5  # 从 2e-5 降低

# 4. 增加 warmup
WARMUP_RATIO=0.2  # 从 0.1 增加
```

### 问题 3: 路由权重坍缩

```bash
# 症状：所有权重集中在单一教师（如 [0.95, 0.05]）

# 解决方案 A: 增加熵正则化
ENTROPY_REG_WEIGHT=0.05  # 从 0.01 增加

# 解决方案 B: 使用温度退火
# 在训练脚本中添加：
# temperature = max(0.5, 1.0 - epoch * 0.1)

# 解决方案 C: 重新初始化路由
# 从 Stage 1 重新开始，不使用预训练路由
```

---

## 📚 预期时间与资源

### 单次实验（双教师，15K 样本）

| 阶段 | 时间（8xA100） | 显存（每卡） | 检查点大小 |
|------|----------------|-------------|-----------|
| 数据准备 | 30 分钟 | 10GB | 5GB |
| KV 提取 | 2-4 小时 | 60GB | 50-100GB |
| Stage 1 训练 | 8-12 小时 | 70GB | 6GB |
| Stage 2 训练 | 10-14 小时 | 70GB | 6GB |
| Stage 3 训练 | 12-16 小时 | 75GB | 6.5GB |
| 评测 | 1-2 小时 | 40GB | - |
| **总计** | **~2-3 天** | **75GB** | **~80GB** |

### 多次实验（完整消融研究）

- 3 种模型组合 × 3 个训练阶段 = **9 个实验**
- 总时间：**~3-4 周**（并行运行）
- 总存储：**~1TB**（包括数据、KV、模型）

---

## 📖 论文复现清单

### 必做实验
1. ✅ 单家族双教师 + 三阶段路由（Qwen）
2. ✅ 跨家族双教师 + 三阶段路由（Qwen+Llama）
3. ✅ 消融研究：固定 vs 相似度 vs 可学习路由
4. ✅ 7 个评测数据集的完整测试

### 可选实验
- [ ] 多教师（3-5 个）扩展性分析
- [ ] 不同 KV 压缩策略对比（full, right, r-kv）
- [ ] 路由可视化与分析（教师选择模式）
- [ ] 中文数据集比例消融（10%, 20%, 50%）

### 预期贡献
1. **方法创新**：首次大规模（7B+）多教师 KV 蒸馏
2. **工程贡献**：完整的对齐框架（5 种策略）
3. **实验洞察**：
   - 单家族 vs 跨家族对齐难度量化
   - 路由策略的实际收益分析
   - 双风格（CoT+Direct）对蒸馏的影响

---

## 🔗 相关文件

- **SLURM 脚本**：
  - `scripts/run_large_scale_multi_teacher.sh`：单次实验
  - `scripts/run_three_stage_routing.sh`：三阶段流程
  
- **核心代码**：
  - `data/multi_task_dataset.py`：数据集加载
  - `scripts/extract_dual_style_kv.py`：KV 提取
  - `evaluation/multi_task_eval.py`：评测框架
  - `experiments/train_multi_teacher_kv.py`：训练主程序
  
- **对齐模块**：
  - `align/tokenizer_align.py`
  - `align/time_align.py`
  - `align/layer_map.py`
  - `align/head_dim_adapter.py`
  - `align/rope_scale.py`
  
- **融合模块**：
  - `fuse/fuse_kv.py`：三种融合策略

---

## 📧 联系与支持

如遇到问题或需要帮助：
1. 检查 `logs/` 目录下的训练日志
2. 使用 TensorBoard 查看训练曲线
3. 参考 `MULTI_TEACHER_README.md` 中的详细说明

祝实验顺利！🎉
