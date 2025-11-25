# Multi-Teacher KV Distillation

多教师 KV 蒸馏完整实现，支持异构教师（不同 tokenizer、层数、维度、注意力头）的知识融合。

## 目录

- [概述](#概述)
- [架构设计](#架构设计)
- [快速开始](#快速开始)
- [模块说明](#模块说明)
- [训练阶段](#训练阶段)
- [HPC 部署](#hpc-部署)
- [实验配置](#实验配置)

## 概述

该实现支持将**多个不同的教师模型**的知识通过 KV cache 蒸馏到单一学生模型，解决了：

1. **异构教师支持**：不同 tokenizer、层数、维度、注意力头的教师模型
2. **真正的多教师学习**：使用完全不同的模型（如 GPT2 + OPT + Pythia）作为教师
3. **融合策略**：固定权重、相似度路由、可学习路由
4. **5 阶段训练**：从简单到复杂的渐进式训练
5. **HPC 优化**：支持多 GPU、混合精度、梯度检查点

**关键特性**：本框架使用**多个不同的预训练模型**作为教师，而不是同一模型的多个 prompt，能够融合不同模型架构的优势。

## 架构设计

```
┌─────────────────────────────────────────────────────────┐
│                    多教师 KV 蒸馏                       │
└─────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
   Teacher 1          Teacher 2          Teacher 3
   (GPT2, 12层)       (BERT, 24层)      (Qwen, 32层)
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                    对齐层 (Alignment)
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   时间对齐            层映射             维度适配
   (Tokenizer)        (Ratio)           (Linear)
        │                  │                  │
        │              位置编码缩放            │
        │              (RoPE NTK)            │
        └──────────────────┼──────────────────┘
                           │
                    融合层 (Fusion)
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   固定权重          相似度路由         可学习路由
   (Fixed)          (Similarity)       (MLP/Gate)
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                    KV_fused = Σ α_i * KV_i
                           │
                           ▼
                      Student Model
```

## 快速开始

### 1. 测试所有模块

```bash
python scripts/test_multi_teacher.py
```

### 2. HPC 训练

```bash
# 修改 scripts/run_multi_teacher.sh 中的配置
sbatch scripts/run_multi_teacher.sh
```

## 模块说明

### 对齐层 (`align/`)

#### 1. **Tokenizer Alignment** (`tokenizer_align.py`)
- **功能**：处理不同 tokenizer 的序列对齐
- **方法**：字符级 IoU 匹配
- **输入**：学生/教师 token 序列
- **输出**：对齐矩阵 $A \in \mathbb{R}^{T_s \times T_t}$

```python
from align import build_char_align_matrix, apply_char_alignment

A = build_char_align_matrix(student_tokens, teacher_tokens, text)
aligned_kv = apply_char_alignment(teacher_kv, A)
```

#### 2. **Time Alignment** (`time_align.py`)
- **功能**：序列长度对齐（padding、masking）
- **方法**：Teacher forcing、软对齐
- **策略**：优先 padding，避免截断

```python
from align import align_sequence_lengths

t_aligned, s_aligned, t_mask, s_mask = align_sequence_lengths(
    teacher_kv, student_kv, align_matrix=A
)
```

#### 3. **Layer Mapping** (`layer_map.py`)
- **功能**：教师层映射到学生层
- **方法**：Ratio-based + 插值
- **公式**：$l_s = \text{round}(l_t \times \frac{L_s}{L_t})$

```python
from align import build_multi_teacher_layer_map, interpolate_teacher_layers

layer_maps = build_multi_teacher_layer_map([24, 32], 12, strategy="ratio")
aligned_kvs = interpolate_teacher_layers(teacher_kvs, layer_map, 12)
```

#### 4. **Head/Dim Adapter** (`head_dim_adapter.py`)
- **功能**：适配不同的隐藏维度和注意力头数
- **方法**：线性投影 + head 聚合/扩展
- **组件**：
  - $W_k, W_v \in \mathbb{R}^{d_t \times d_s}$：维度投影
  - Head grouping/expansion：头数对齐

```python
from align import MultiTeacherHeadDimAdapter

adapter = MultiTeacherHeadDimAdapter(
    [(768, 12), (1024, 16)],  # Teacher configs
    512, 8                     # Student config
)
adapted_ks, adapted_vs = adapter(teacher_ks, teacher_vs)
```

#### 5. **RoPE Scaling** (`rope_scale.py`)
- **功能**：RoPE 位置编码缩放
- **方法**：NTK-aware scaling
- **公式**：$\text{base}_{\text{new}} = \text{base} \times s^{2/3}$

```python
from align import MultiTeacherRoPEScaler

scaler = MultiTeacherRoPEScaler(
    [(10000, 2048), (10000, 4096)],  # Teacher configs
    2048,                             # Student max_len
    scaling_method="ntk"
)
scaled_ks, scaled_vs = scaler.scale_kv_pairs(teacher_ks, teacher_vs)
```

### 教师层 (`teacher/`)

#### 1. **KV Extraction** (`extract_teacher_kv.py`)
- **功能**：离线提取教师 KV
- **用途**：避免在线重复计算
- **支持**：批量提取、SLURM array job

```python
from teacher import TeacherKVExtractor

extractor = TeacherKVExtractor("gpt2", batch_size=4)
kvs = extractor.extract_kvs(texts)
extractor.save_kvs(kvs, "teacher_kvs.pt")
```

#### 2. **Router Prototype** (`router_proto.py`)
- **功能**：计算教师原型特征（用于相似度路由）
- **方法**：
  - Mean pooling：$\text{proto} = \text{mean}(\text{KV}, \text{dim}=\text{time})$
  - CLS token：$\text{proto} = \text{KV}[:, 0, :]$
  - K-means：聚类中心

```python
from teacher import compute_multi_teacher_prototypes, compute_routing_weights

prototypes = compute_multi_teacher_prototypes(teacher_kvs_list, method="mean")
weights = compute_routing_weights(student_hidden, prototypes, temperature=1.0)
```

### 融合层 (`fuse/`)

#### 1. **Fixed Fusion** (`fuse_kv.py`)
- **公式**：$\text{KV}_{\text{fused}} = \sum_{i=1}^{N} w_i \cdot \text{KV}_i$
- **权重**：预定义或均匀

```python
from fuse import fuse_kvs_fixed

fused_kv = fuse_kvs_fixed(kvs_list, weights=[0.5, 0.3, 0.2])
```

#### 2. **Similarity Fusion**
- **公式**：$w_i = \text{softmax}\left(\frac{\text{sim}(q, p_i)}{\tau}\right)$
- **相似度**：余弦/点积/L2

```python
from fuse import fuse_kvs_similarity

fused_kv, weights = fuse_kvs_similarity(kvs_list, query, prototypes, temperature=1.0)
```

#### 3. **Learnable Fusion**
- **路由器**：MLP / Gate / Attention
- **训练**：端到端学习路由权重

```python
from fuse import LearnableRouter, fuse_kvs_learnable

router = LearnableRouter(hidden_dim=512, num_teachers=3, router_type="attention")
fused_kv, weights = fuse_kvs_learnable(kvs_list, query, router)
```

#### 4. **Entropy Regularization**
- **目标**：
  - `diverse`：鼓励使用所有教师（高熵）
  - `specialized`：鼓励专业化（低熵）
- **损失**：$L_{\text{entropy}} = \gamma \cdot H(\boldsymbol{w})$

```python
from fuse import EntropyRegularizer

regularizer = EntropyRegularizer(target="specialized", strength=0.01)
loss = regularizer.compute_loss(routing_weights)
```

## 训练阶段

### Phase 1: Dual-Teacher（双教师）
- **策略**：使用 2 个**不同的模型**作为教师（如 GPT2 + GPT2-medium）
- **教师数**：2
- **融合**：固定权重（0.5, 0.5）
- **目标**：验证多教师可行性

### Phase 2: Multi-Teacher Basic（多教师基础）
- **策略**：使用 2-3 个**不同架构的模型**（如 GPT2 + OPT + Pythia）
- **教师数**：2-3
- **融合**：固定权重
- **目标**：测试异构模型融合

### Phase 3: Real Multi-Teacher（真正的多教师）
- **策略**：所有教师**同时**标注所有样本，使用**完全不同的模型家族**
- **教师数**：3+（如 GPT2、LLaMA、Qwen、Mistral）
- **融合**：固定权重或相似度路由
- **目标**：融合不同模型优势

### Phase 4: Routing（动态路由）
- **策略**：可学习路由器，自动学习何时使用哪个教师模型
- **教师数**：3+
- **融合**：MLP/Gate/Attention 路由
- **正则化**：熵正则化
- **目标**：动态选择最优教师

### Phase 5: Z-Space Alignment（跨架构对齐）
- **策略**：在统一 Z 空间对齐**不同架构**教师（GPT、BERT、T5 等）
- **教师数**：异构教师（Encoder-Decoder、Decoder-only 混合）
- **融合**：学习统一表示
- **目标**：跨架构知识融合

## HPC 部署

### 推荐的教师模型组合

根据不同的实验目标，推荐以下教师模型组合：

#### 1. **同家族不同大小**（验证基础融合）
```bash
# GPT-2 系列
TEACHER_MODELS="gpt2 gpt2-medium gpt2-large"

# OPT 系列
TEACHER_MODELS="facebook/opt-125m facebook/opt-350m facebook/opt-1.3b"

# Pythia 系列
TEACHER_MODELS="EleutherAI/pythia-160m EleutherAI/pythia-410m EleutherAI/pythia-1b"
```

#### 2. **不同架构同尺寸**（测试异构融合）
```bash
# 小模型混合（~300-500M 参数）
TEACHER_MODELS="gpt2 facebook/opt-350m EleutherAI/pythia-410m"

# 中型模型混合（~1-2B 参数）
TEACHER_MODELS="gpt2-large facebook/opt-1.3b EleutherAI/pythia-1.4b"
```

#### 3. **不同模型家族**（最强异构性）
```bash
# 开源小模型混合
TEACHER_MODELS="gpt2 facebook/opt-350m TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# 中型开源模型混合
TEACHER_MODELS="Qwen/Qwen2.5-1.5B-Instruct meta-llama/Llama-3.2-1B mistralai/Mistral-7B-v0.1"
```

#### 4. **跨语言模型**（Phase 5 专用）
```bash
# 英文 + 中文 + 多语言
TEACHER_MODELS="gpt2 Qwen/Qwen2.5-1.5B THUDM/chatglm3-6b"
```

#### 5. **不同训练目标**（多样化知识）
```bash
# Base + Instruct + Chat
TEACHER_MODELS="gpt2 facebook/opt-350m TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

### 配置文件：`scripts/run_multi_teacher.sh`

```bash
# 学生模型
STUDENT_MODEL="Qwen/Qwen2.5-0.5B-Instruct"

# 教师模型（空格分隔）
TEACHER_MODELS="gpt2 gpt2-medium gpt2-large"

# 训练阶段
PHASE=4

# 融合方法
FUSION_METHOD="learnable"
ROUTER_TYPE="attention"

# 训练超参数
BATCH_SIZE=4
GRAD_ACCUM=8
MAX_LENGTH=512
NUM_EPOCHS=3
LEARNING_RATE=2e-5
```

### 提交任务

```bash
# 修改配置
vim scripts/run_multi_teacher.sh

# 提交
sbatch scripts/run_multi_teacher.sh

# 查看日志
tail -f logs/multi_teacher_*.out
```

## 实验配置

### 损失权重

```python
# KV loss
lambda_k = 1.0  # Key loss weight
lambda_v = 1.0  # Value loss weight

# Additional losses
beta_cos = 0.1   # Cosine similarity loss
gamma_kl = 0.01  # KL divergence (attention)
delta_ce = 1.0   # Cross-entropy loss

# Entropy regularization (Phase 4+)
entropy_reg_strength = 0.01
entropy_target = "specialized"  # or "diverse"
```

### 对齐策略

```python
# Layer mapping
layer_mapping_strategy = "ratio"  # or "uniform", "skip"

# RoPE scaling
rope_scaling_method = "ntk"  # or "linear", "dynamic"
```

### 融合方法选择

| 阶段 | 推荐方法 | 路由器类型 | 熵正则化 |
|------|----------|-----------|---------|
| 1-2  | fixed    | -         | ✗       |
| 3    | similarity | -       | ✗       |
| 4    | learnable | attention | ✓       |
| 5    | learnable | attention | ✓       |

## 预期效果

根据 MULTI_TEACHER_KV_PLAN.md：

1. **Phase 1-2**：验证多教师可行性，loss 应下降
2. **Phase 3**：真正多教师，性能提升 5-10%
3. **Phase 4**：动态路由，进一步提升 3-5%
4. **Phase 5**：跨架构对齐，支持异构教师融合

## 可视化

### 路由权重分布

```python
import matplotlib.pyplot as plt

# weights: [batch, num_teachers]
plt.bar(range(num_teachers), weights.mean(dim=0).cpu())
plt.xlabel("Teacher")
plt.ylabel("Average Weight")
plt.title("Routing Weight Distribution")
plt.show()
```

### 层映射矩阵

```python
from align import visualize_layer_mapping

visualize_layer_mapping(layer_map, num_teacher_layers=32, num_student_layers=12)
```

### 对齐矩阵

```python
from align import visualize_alignment

visualize_alignment(align_matrix, student_tokens, teacher_tokens)
```

## 故障排查

### 问题 1：OOM（内存不足）

**解决方案**：
- 减小 `batch_size`
- 增加 `gradient_accumulation_steps`
- 启用 `--gradient_checkpointing`
- 使用 `--bf16` 或 `--fp16`

### 问题 2：Loss 不下降

**解决方案**：
- 检查 `lambda_*` 权重是否合理
- 降低 `learning_rate`
- 增加 `warmup_steps`
- 从 Phase 1 开始，逐步到 Phase 4

### 问题 3：路由权重退化（所有权重相同）

**解决方案**：
- 调整 `entropy_target` 为 `"specialized"`
- 增加 `entropy_reg_strength`
- 使用 `attention` 路由器而非 `mlp`

### 问题 4：不同教师性能差异大

**解决方案**：
- 使用相似度路由（`fusion_method="similarity"`）
- 调整固定权重（更多权重给强教师）
- 检查对齐模块是否正常（运行 `test_multi_teacher.py`）

## 参考文献

1. KaVa: https://arxiv.org/abs/2501.00231
2. RoPE: https://arxiv.org/abs/2104.09864
3. NTK-aware scaling: https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/
4. Multi-teacher distillation: Distilling the Knowledge in a Neural Network (Hinton et al.)

## 更新日志

- **2024-01**: 初始实现（Phase 1-3）
- **2024-01**: 添加动态路由（Phase 4）
- **2024-01**: 支持跨架构对齐（Phase 5）
- **2024-01**: HPC 优化和文档完善

## 联系

有问题或建议，请查看：
- `MULTI_TEACHER_KV_PLAN.md` - 完整技术规范
- `scripts/test_multi_teacher.py` - 测试所有模块
- `experiments/train_multi_teacher_kv.py` - 训练脚本
