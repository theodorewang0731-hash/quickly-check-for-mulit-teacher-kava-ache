# KAVA 对齐方法完整说明

## 📋 目录
1. [对齐方法概览](#对齐方法概览)
2. [核心对齐方法](#核心对齐方法)
3. [维度对齐方法](#维度对齐方法)
4. [实现细节](#实现细节)
5. [完整流程](#完整流程)

---

## 对齐方法概览

### 当前使用的对齐方法

| 对齐类型 | 方法名称 | 目的 | 实现位置 |
|---------|---------|------|---------|
| **语义对齐** | Mercator Projection Loss | 对齐语义方向（主要） | `src/losses.py` |
| **混合对齐** | Hybrid Loss | Mercator + MSE 混合 | `src/losses.py` |
| **维度对齐** | Elastic Bottleneck Projector | Teacher维度 → Student维度 | `experiments/kv_dimension_projector.py` |
| **跨层聚合** | Cross-Layer Aggregation | 聚合所有层的KV | `src/dynamic_kv_extractor.py` |

---

## 核心对齐方法

### 1. Mercator Projection Loss（主要方法）⭐

#### 📖 核心思想
**对齐语义方向而非数值大小**

- **方向** = 语义含义（向量指向哪里）
- **幅度** = 置信度（向量有多长）

对于 RoPE-based 模型（Qwen/Llama），旋转一致性（方向）比数值近似（幅度）更重要。

#### 📐 数学公式

```python
# 1. 投影到单位球面（归一化）
s_norm = student_kv / ||student_kv||    # [B, T, D]
t_norm = teacher_kv / ||teacher_kv||    # [B, T, D]

# 2. 计算余弦相似度（方向一致性）
cos_sim = mean(s_norm · t_norm)         # 标量，范围 [-1, 1]

# 3. Mercator Loss
direction_loss = 1 - cos_sim             # cos_sim=1时loss=0（完美）

# 4. 可选：弱幅度约束（防止坍缩）
magnitude_loss = MSE(log(||s||), log(||t||))

# 5. 总损失
total_loss = α × direction_loss + β × magnitude_loss
```

**推荐参数**：
- `α = 1.0` （方向损失权重）
- `β = 0.0` 或 `0.01` （幅度约束，可选）

#### 💡 为什么有效？

**场景对比：**

| Teacher | Student | MSE Loss | Mercator Loss |
|---------|---------|----------|---------------|
| 100×[0.707, 0.707] | 1×[0.707, 0.707] | 很大 ❌ | 0.0 ✅ |
| [1.0, 0.0] | [0.0, 1.0] | 2.0 ❌ | 2.0 ❌ |
| [1.0, 0.0] | [1.0, 0.0] | 0.0 ✅ | 0.0 ✅ |

**关键优势：**
- ✅ 识别语义对齐（即使幅度不同）
- ✅ 不惩罚置信度差异
- ✅ 专注于方向一致性

#### 🔧 实现代码

```python
class MercatorKVLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.0, epsilon=1e-8):
        super().__init__()
        self.alpha = alpha  # 方向损失权重
        self.beta = beta    # 幅度损失权重（可选）
        self.epsilon = epsilon
    
    def forward(self, student_kv, teacher_kv):
        """
        Args:
            student_kv: [Batch, Seq, Dim] - Student的KV表示
            teacher_kv: [Batch, Seq, Dim] - Teacher的KV表示
        
        Returns:
            total_loss: 标量损失
            metrics: 详细指标字典
        """
        # Step 1: Mercator投影（方向对齐）
        s_norm = F.normalize(student_kv, p=2, dim=-1)
        t_norm = F.normalize(teacher_kv, p=2, dim=-1)
        
        # Step 2: 计算方向一致性
        cos_sim = torch.sum(s_norm * t_norm, dim=-1).mean()
        direction_loss = 1.0 - cos_sim
        
        # Step 3: 可选幅度约束
        if self.beta > 0:
            s_mag = torch.norm(student_kv, p=2, dim=-1)
            t_mag = torch.norm(teacher_kv, p=2, dim=-1)
            magnitude_loss = F.mse_loss(
                torch.log(s_mag + self.epsilon),
                torch.log(t_mag + self.epsilon)
            )
        else:
            magnitude_loss = 0.0
        
        # Step 4: 组合损失
        total_loss = self.alpha * direction_loss + self.beta * magnitude_loss
        
        # Step 5: 收集指标
        metrics = {
            "cos_sim": cos_sim.item(),           # 核心指标：目标 > 0.95
            "dir_loss": direction_loss.item(),
            "mag_loss": magnitude_loss.item() if self.beta > 0 else 0.0,
        }
        
        return total_loss, metrics
```

#### 📊 训练目标

| 阶段 | CosSim 范围 | 状态 |
|------|-------------|------|
| 0-50步 | 0.20-0.50 | 🔄 适应中 |
| 50-100步 | 0.50-0.70 | ⚠️ 学习中 |
| 100-200步 | 0.70-0.90 | 📈 良好 |
| 200+步 | **>0.90** | ✅ 优秀 ← **目标** |

---

### 2. Hybrid Loss（混合方法）

#### 📖 核心思想
渐进式从 MSE 过渡到 Mercator

适用场景：
- 训练初期需要更强的数值约束
- 逐步转向方向对齐
- 平衡传统损失和新方法

#### 📐 数学公式

```python
# Mercator分量
merc_loss = MercatorLoss(student, teacher)

# MSE分量
mse_loss = MSE(student, teacher)

# 混合
total_loss = w_merc × merc_loss + w_mse × mse_loss
```

**推荐权重：**
- `w_merc = 0.8`（Mercator主导）
- `w_mse = 0.2`（MSE辅助）

#### 🔧 实现代码

```python
class HybridKVLoss(nn.Module):
    def __init__(self, mercator_weight=0.8, mse_weight=0.2, beta=0.01):
        super().__init__()
        self.mercator_weight = mercator_weight
        self.mse_weight = mse_weight
        
        self.mercator_loss = MercatorKVLoss(alpha=1.0, beta=beta)
        self.mse_loss = nn.MSELoss()
    
    def forward(self, student_kv, teacher_kv):
        # Mercator部分
        merc_loss, merc_metrics = self.mercator_loss(student_kv, teacher_kv)
        
        # MSE部分
        mse = self.mse_loss(student_kv, teacher_kv)
        
        # 组合
        total_loss = (self.mercator_weight * merc_loss + 
                      self.mse_weight * mse)
        
        return total_loss, metrics
```

---

## 维度对齐方法

### 3. Elastic Bottleneck Projector ⭐

#### 📖 核心思想
**将Teacher的高维KV投影到Student的低维空间**

问题：
- Teacher: Qwen-1.5B (7168维)
- Student: Qwen-0.5B (3072维)
- 无法直接对齐！

解决：可学习的维度投影网络

#### 🏗️ 架构设计

```python
Input: Teacher KV [B, T, 7168]
    ↓
LayerNorm(7168)          # 稳定梯度
    ↓
Linear(7168 → 7168)      # 特征变换
    ↓
SiLU()                   # 非线性激活
    ↓
Dropout(0.1)             # 正则化
    ↓
Linear(7168 → 3072)      # 降维投影
    ↓
Output: Aligned KV [B, T, 3072]
```

**关键组件：**
1. **Pre-LayerNorm**: 稳定数值，跨模型尺度通用
2. **Elastic MLP**: 可调节隐藏层宽度（mlp_ratio）
3. **Non-linear**: SiLU激活捕获复杂特征
4. **Separate K/V**: Keys和Values独立投影

#### 📐 数学公式

```python
# K 投影
K_aligned = Linear2(Dropout(SiLU(Linear1(LayerNorm(K_teacher)))))
          : [B, T, d_t] → [B, T, d_s]

# V 投影（独立网络）
V_aligned = Linear2(Dropout(SiLU(Linear1(LayerNorm(V_teacher)))))
          : [B, T, d_t] → [B, T, d_s]
```

**维度变化：**
```
Teacher: 7168维 → Hidden: 7168维 → Student: 3072维
         (d_t)     (d_t × mlp_ratio)    (d_s)
```

#### 🔧 实现代码

```python
class KVDimensionProjector(nn.Module):
    def __init__(
        self,
        teacher_configs: Dict[str, Dict[str, int]],
        student_d_model: int,
        mlp_ratio: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.projectors = nn.ModuleDict()
        
        for teacher_name, config in teacher_configs.items():
            teacher_d_model = config["d_model"]
            hidden_dim = int(teacher_d_model * mlp_ratio)
            
            # K 投影器
            adapter_K = nn.Sequential(
                nn.LayerNorm(teacher_d_model),     # 稳定性
                nn.Linear(teacher_d_model, hidden_dim),
                nn.SiLU(),                          # 非线性
                nn.Dropout(dropout),                # 正则化
                nn.Linear(hidden_dim, student_d_model)
            )
            
            # V 投影器（独立）
            adapter_V = nn.Sequential(
                nn.LayerNorm(teacher_d_model),
                nn.Linear(teacher_d_model, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, student_d_model)
            )
            
            self.projectors[teacher_name] = nn.ModuleDict({
                "K": adapter_K,
                "V": adapter_V
            })
    
    def project_teacher_kv(self, teacher_name, teacher_K, teacher_V):
        """
        投影Teacher KV到Student维度
        
        Args:
            teacher_K: [B, T, d_teacher] = [B, T, 7168]
            teacher_V: [B, T, d_teacher] = [B, T, 7168]
        
        Returns:
            K_aligned: [B, T, d_student] = [B, T, 3072]
            V_aligned: [B, T, d_student] = [B, T, 3072]
        """
        proj_K = self.projectors[teacher_name]["K"]
        proj_V = self.projectors[teacher_name]["V"]
        
        K_aligned = proj_K(teacher_K)
        V_aligned = proj_V(teacher_V)
        
        return K_aligned, V_aligned
```

#### 🎯 参数配置

| 模型规模 | mlp_ratio | 说明 |
|---------|-----------|------|
| <7B | 0.5-1.0 | 压缩瓶颈，减少参数 |
| 7B-30B | 1.0 | 等宽变换，平衡性能 |
| 30B-70B+ | 2.0 | 扩展特征，捕获复杂性 |

**当前配置（Qwen 1.5B → 0.5B）：**
- `mlp_ratio = 1.0`
- `dropout = 0.1`
- 参数量: **147M**（两个投影器）

---

### 4. Cross-Layer Aggregation（跨层聚合）

#### 📖 核心思想
**聚合所有层的KV而非只用最后一层**

问题：
- 量化模型单层维度小（256维）
- 配置维度大（1536维）
- 维度不匹配！

解决：拼接所有28层的KV

#### 📐 数学公式

```python
# 传统方法（单层）
k_last, v_last = past_key_values[-1]  # 只取最后一层
k_flat = flatten(k_last)               # [B, T, 256]
# 问题：维度太小 256 ≠ 1536

# 跨层聚合（全层）
all_kvs = []
for layer_kv in past_key_values:      # 遍历所有28层
    k, v = layer_kv
    k_flat = flatten(k)                # [B, T, 256]
    all_kvs.append(k_flat)

k_combined = concat(all_kvs, dim=-1)  # [B, T, 28×256] = [B, T, 7168]
# 解决：28层 × 256维/层 = 7168维 ✓
```

#### 🔍 维度分析

**Teacher (Qwen-1.5B 4-bit):**
```
配置维度: 1536
层数: 28
每层注意力头: 2（量化后）
每头维度: 128
单层维度: 2 × 128 = 256
总维度: 28 × 256 = 7168 ← 实际使用
```

**Student (Qwen-0.5B):**
```
配置维度: 896
层数: 24
每层注意力头: 2
每头维度: 128  
单层维度: 2 × 128 = 128
总维度: 24 × 128 = 3072 ← 实际使用
```

#### 🔧 实现代码

```python
class DynamicKVExtractor:
    def __init__(
        self,
        aggregation_method: str = "concat",  # concat / mean / weighted
        use_all_layers: bool = True,
    ):
        self.aggregation_method = aggregation_method
        self.use_all_layers = use_all_layers
    
    def extract_kv(self, past_key_values):
        """
        提取并聚合KV Cache
        
        Args:
            past_key_values: Tuple of (key, value) for each layer
                key: [B, H, T, D_h] for each layer
        
        Returns:
            kv_flat: [B, T, total_dim]
        """
        if self.aggregation_method == "concat":
            return self._extract_concat(past_key_values)
    
    def _extract_concat(self, past_key_values):
        """拼接聚合方法"""
        all_kvs = []
        
        for layer_kv in past_key_values:  # 遍历所有层
            k, v = layer_kv
            # k shape: [B, H, T, D_h]
            
            # 展平单层：[B, H, T, D_h] → [B, T, H×D_h]
            B, H, T, D_h = k.shape
            k_flat = k.permute(0, 2, 1, 3).contiguous().view(B, T, H * D_h)
            all_kvs.append(k_flat)
        
        # 拼接所有层：[B, T, num_layers × H × D_h]
        kv_combined = torch.cat(all_kvs, dim=-1)
        
        return kv_combined
```

#### 🎯 聚合策略对比

| 方法 | 输出维度 | 优点 | 缺点 |
|-----|---------|------|------|
| **concat** | 28×256=7168 | 保留所有信息 | 维度大 |
| mean | 256 | 维度小 | 信息损失 |
| weighted | 256 | 可学习权重 | 需要调优 |

**当前使用：concat**（保留完整信息）

---

## 实现细节

### 🔄 完整对齐流程

```python
# 训练循环中的对齐流程

for batch in dataloader:
    # ===== Step 1: 前向传播获取KV Cache =====
    with torch.no_grad():
        t_out = teacher(input_ids, attention_mask, use_cache=True)
        # t_out.past_key_values: Tuple[Tuple[Tensor, Tensor], ...]
        #   每个元素: (key, value) for one layer
        #   key shape: [B, H, T, D_h]
    
    s_out = student(input_ids, attention_mask, use_cache=True)
    # 同上
    
    # ===== Step 2: 跨层聚合 (Cross-Layer Aggregation) =====
    t_kv = kv_extractor.extract_kv(
        t_out.past_key_values,
        model_name="teacher"
    )
    # Output: [B, T, 7168] ← 28层聚合
    
    s_kv = kv_extractor.extract_kv(
        s_out.past_key_values,
        model_name="student"
    )
    # Output: [B, T, 3072] ← 24层聚合
    
    # ===== Step 3: 数据类型转换 =====
    t_kv = t_kv.to(torch.bfloat16)  # 统一精度
    s_kv = s_kv.to(torch.bfloat16)
    
    # ===== Step 4: 维度对齐 (Elastic Bottleneck) =====
    t_proj, _ = projector.project_teacher_kv("teacher", t_kv, t_kv)
    # Input:  [B, T, 7168]
    # Output: [B, T, 3072] ← 与Student维度匹配
    
    # ===== Step 5: 语义对齐 (Mercator Loss) =====
    loss, metrics = loss_fn(s_kv, t_proj)
    # s_kv:   [B, T, 3072] Student KV
    # t_proj: [B, T, 3072] Teacher KV (对齐后)
    # 
    # 内部计算：
    # 1. 归一化到单位球面
    # 2. 计算余弦相似度
    # 3. direction_loss = 1 - cos_sim
    
    # ===== Step 6: 反向传播 =====
    loss.backward()
    optimizer.step()
    
    # ===== 监控指标 =====
    print(f"Loss: {loss.item():.4f}")
    print(f"CosSim: {metrics['cos_sim']:.4f}")  # 目标 > 0.90
```

### 📊 维度变化追踪

```
Teacher (Qwen-1.5B 4-bit):
  Model Output → [B, 28_layers, H=2, T, D_h=128]
      ↓ Cross-Layer Aggregation
  Flattened → [B, T, 28×2×128] = [B, T, 7168]
      ↓ Type Conversion
  BF16 → [B, T, 7168]
      ↓ Elastic Bottleneck Projector
  Aligned → [B, T, 3072]
      ↓ Mercator Loss
  Direction Loss ← Compare with Student

Student (Qwen-0.5B):
  Model Output → [B, 24_layers, H=2, T, D_h=128]
      ↓ Cross-Layer Aggregation
  Flattened → [B, T, 24×2×128] = [B, T, 3072]
      ↓ Type Conversion
  BF16 → [B, T, 3072]
      ↓ (No projection needed)
  Ready → [B, T, 3072]
      ↓ Mercator Loss
  Direction Loss ← Compare with Teacher
```

---

## 完整流程图

```
┌─────────────────────────────────────────────────────────────┐
│                     KAVA对齐完整流程                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────┐         ┌─────────────────┐
│   Teacher       │         │    Student      │
│  Qwen-1.5B      │         │   Qwen-0.5B     │
│   (4-bit)       │         │   (bfloat16)    │
└────────┬────────┘         └────────┬────────┘
         │                           │
         │ Forward Pass              │ Forward Pass
         │ use_cache=True            │ use_cache=True
         ↓                           ↓
┌─────────────────┐         ┌─────────────────┐
│ past_key_values │         │ past_key_values │
│ 28 layers       │         │ 24 layers       │
│ [B,H,T,D_h]     │         │ [B,H,T,D_h]     │
│ per layer       │         │ per layer       │
└────────┬────────┘         └────────┬────────┘
         │                           │
         │ ┌──────────────────────┐  │
         └→│  Cross-Layer         │←─┘
           │  Aggregation         │
           │  (concat all layers) │
           └──────────┬───────────┘
                      ↓
         ┌────────────────────────────┐
         │  Flattened KV Cache        │
         │  Teacher: [B, T, 7168]     │
         │  Student: [B, T, 3072]     │
         └────────────┬───────────────┘
                      │
                      │ Type Conversion
                      │ to BF16
                      ↓
         ┌────────────────────────────┐
         │  Unified Precision         │
         │  Both in BF16              │
         └────────────┬───────────────┘
                      │
         ┌────────────┴────────────┐
         │                         │
         ↓                         ↓
┌─────────────────┐       ┌─────────────────┐
│  Teacher KV     │       │   Student KV    │
│  [B, T, 7168]   │       │   [B, T, 3072]  │
└────────┬────────┘       └────────┬────────┘
         │                         │
         │ Elastic                 │ (No projection)
         │ Bottleneck              │
         │ Projector               │
         ↓                         │
┌─────────────────┐                │
│  Aligned KV     │                │
│  [B, T, 3072]   │                │
└────────┬────────┘                │
         │                         │
         └────────┬────────────────┘
                  │
                  ↓
         ┌─────────────────┐
         │  Mercator Loss  │
         │  (Direction)    │
         │                 │
         │  1. Normalize   │
         │  2. CosSim      │
         │  3. Loss=1-cos  │
         └────────┬────────┘
                  │
                  ↓
         ┌─────────────────┐
         │  Backprop &     │
         │  Update:        │
         │  - Student      │
         │  - Projector    │
         └─────────────────┘
```

---

## 📈 训练配置

### 当前使用的配置

```python
# 模型配置
GLOBAL_CONFIG = {
    # 模型
    'teacher_model_name': 'Qwen/Qwen2.5-1.5B-Instruct',
    'student_model_name': 'Qwen/Qwen2.5-0.5B',
    'teacher_quantization': '4bit',
    
    # 损失函数（Mercator）
    'loss_alpha': 1.0,   # 方向损失权重
    'loss_beta': 0.01,   # 幅度损失权重（弱约束）
    
    # KV提取（Cross-Layer）
    'kv_aggregation_method': 'concat',
    'use_all_layers': True,
    
    # 优化器
    'learning_rate_student': 5e-5,      # Student学习率
    'learning_rate_projector': 1e-3,    # Projector学习率（更高）
    'weight_decay': 0.01,
    
    # 训练
    'max_length': 512,
    'batch_size': 2,                     # 自动调整
    'gradient_accumulation_steps': 16,   # 自动调整
    'max_steps': 1000,
}
```

### 维度信息

```python
# 动态检测到的实际维度
Teacher: 
  - Config: 1536维
  - Actual: 7168维 (28层 × 256维/层)
  - Layers: 28
  - Heads per layer: 2 (量化后)
  - Head dim: 128

Student:
  - Config: 896维
  - Actual: 3072维 (24层 × 128维/层)
  - Layers: 24
  - Heads per layer: 2
  - Head dim: 128

Projector:
  - Input: 7168维
  - Hidden: 7168维 (mlp_ratio=1.0)
  - Output: 3072维
  - Parameters: 147M
```

---

## 🎯 关键指标

### 训练目标

| 指标 | 目标值 | 说明 |
|-----|-------|------|
| **CosSim** | **>0.90** | 核心指标：方向对齐度 |
| Loss | 趋向0 | 总损失下降 |
| Student Magnitude | 稳定 | 不应坍缩或爆炸 |
| Training Speed | ~2-5s/it | RTX 4070 8GB |

### 收敛阶段

```
Step   0-50:  CosSim 0.20-0.50  🔄 适应中
Step  50-100: CosSim 0.50-0.70  ⚠️  学习中
Step 100-200: CosSim 0.70-0.90  📈 良好
Step  200+:   CosSim >0.90      ✅优秀 ← 目标
```

---

## 🔬 对比总结

### vs 传统MSE

| 特性 | MSE | Mercator |
|-----|-----|----------|
| 对齐目标 | 数值相等 | 方向一致 |
| 幅度敏感 | 高 ❌ | 低 ✅ |
| 语义理解 | 弱 | 强 ✅ |
| RoPE兼容 | 一般 | 优秀 ✅ |
| 收敛速度 | 慢 | 快 ✅ |

### vs 其他方法

| 方法 | 优点 | 缺点 | 使用场景 |
|-----|------|------|---------|
| **Mercator** | 语义对齐强 | 可能忽略幅度 | 主要方法⭐ |
| **Hybrid** | 平衡两者 | 需调权重 | 过渡阶段 |
| **Pure MSE** | 简单直接 | 语义弱 | 基线对比 |

---

## ✅ 总结

### 我们使用的四种对齐方法：

1. **Mercator Projection Loss** ⭐
   - 对齐语义方向（主要方法）
   - `loss = 1 - cosine_similarity`
   - 目标：CosSim > 0.90

2. **Hybrid Loss**
   - Mercator + MSE混合
   - 可调权重平衡

3. **Elastic Bottleneck Projector** ⭐
   - 维度对齐：7168维 → 3072维
   - 可学习的MLP投影网络
   - 147M参数

4. **Cross-Layer Aggregation** ⭐
   - 聚合所有层的KV
   - Teacher: 28层 × 256 = 7168维
   - Student: 24层 × 128 = 3072维

### 完整流程：

```
Teacher Forward → Cross-Layer Aggregation → Type Conversion
                                              ↓
                                    Elastic Bottleneck
                                              ↓
                                    Mercator Loss ← Student KV
                                              ↓
                                         Backprop
```

**核心创新**：方向对齐 + 跨层聚合 + 动态维度检测

**适用场景**：RoPE-based模型（Qwen/Llama）的KV Cache蒸馏
