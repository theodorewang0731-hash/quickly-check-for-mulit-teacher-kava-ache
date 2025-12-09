# 项目更新说明 - v4.0 地图投影整合

**更新日期**: 2025年12月9日  
**版本**: v4.0  
**核心目标**: 整合地图投影方案，实现 structured/flat 双模式对比

---

## 📦 更新内容总结

### 🆕 新增文件 (4个)

#### 1. `src/headwise_projector.py` ✨
**功能**: Anti-Flatten 结构化投影器  
**核心特性**:
- ✅ 输入输出严格保持 5D 形状 `[B, L, H, T, D]`
- ✅ 支持 `share_dim_proj` 参数（共享/独立维度投影）
- ✅ **新增 `init_uniform` 参数**：均匀初始化 head_mixer
  - 将 Teacher heads 均匀分配到 Student heads
  - 提供合理的初始地图，避免随机初始化不稳定
- ✅ 可独立测试（内置 `__main__` 测试代码）

**使用示例**:
```python
from src.headwise_projector import HeadwiseMapProjector

projector = HeadwiseMapProjector(
    H_t=32, H_s=16, D_t=128, D_s=64,
    share_dim_proj=True,  # 共享维度投影
    init_uniform=True      # 均匀初始化 ⭐ 新增
)
k_s = projector(k_t)  # [B, L, H_t, T, D_t] → [B, L, H_s, T, D_s]
```

---

#### 2. `src/time_warping.py` ✨
**功能**: 基于 Segment 的时间维对齐  
**核心特性**:
- ✅ 支持 P/R/A 三段式对齐
- ✅ 每段可配置不同的采样比例（ratio_map）和平滑系数（alpha_map）
- ✅ **清晰的工程假设注释**：
  - `segment_ids[0]` 作为全 batch 参考
  - 假设 batch 内结构一致（当前 KV 蒸馏场景合理）
  - 为将来 per-sample 切段预留扩展空间

**使用示例**:
```python
from src.time_warping import create_reasoning_focused_warper

warper = create_reasoning_focused_warper()  # R 段采样 50%
k_s = warper(k_t, segment_ids, T_s=50)
```

---

#### 3. `src/map_projection_aligner.py` ✨✨ **核心**
**功能**: 完整的地图投影对齐器  
**核心特性**:
- ✅ 整合层对齐 + 时间对齐 + 结构化投影
- ✅ **双模式支持**（v4.0 核心改进）:
  - `mode="structured"`: 新方案（HeadwiseMapProjector）
  - `mode="flat"`: 旧方案（KVDimensionProjector，baseline）
- ✅ 显式处理 Q（支持完整的 Q-K-V 对齐）
- ✅ 统一接口，便于 A/B 对比

**使用示例**:
```python
from src.map_projection_aligner import (
    create_structured_aligner,
    create_flat_aligner
)

# 新方案
aligner = create_structured_aligner(teacher_cfg, student_cfg)
k_s, v_s, q_s = aligner(k_t, v_t, q_t, segment_ids)

# 旧方案（baseline）
aligner_baseline = create_flat_aligner(teacher_cfg, student_cfg)
k_s, v_s, q_s = aligner_baseline(k_t, v_t, q_t, segment_ids)
```

**配置文件控制**:
```yaml
# 只需修改一个字段即可切换模式
kv_projection_mode: structured  # 或 "flat"
```

---

#### 4. `experiments/profile_alignment.py` ✨
**功能**: 阶段 1 验证工具  
**核心特性**:
- ✅ 只跑 1-2 个 batch 的 forward（不训练）
- ✅ 验证形状对齐是否正确
- ✅ 检查 NaN 和异常值
- ✅ 简单评估 cos 相似度
- ✅ Attention 分布检查
- ✅ 支持 structured/flat 双模式对比

**使用示例**:
```bash
# 测试 structured 模式
python experiments/profile_alignment.py --mode structured

# 测试 flat 模式
python experiments/profile_alignment.py --mode flat

# 自定义配置
python experiments/profile_alignment.py \
    --teacher Qwen/Qwen2.5-7B \
    --student Qwen/Qwen2.5-1.5B \
    --mode structured \
    --batch_size 2 \
    --seq_length 100
```

---

### 🔧 修改文件 (1个)

#### `src/losses.py` - 新增 `StructuralKVLoss`
**功能**: 结构化 KV 损失（阶段 2 用）  
**新增内容**:
- ✅ `StructuralKVLoss` 类：K/V 方向对齐 + Q-K 交互对齐
- ✅ K/V 使用余弦相似度（方向对齐）
- ✅ Q 通过 Q-K 交互的 Attention KL 对齐（而非直接向量差）
- ✅ 支持独立 ablation（alpha_k, alpha_v, alpha_attn）
- ✅ 可选 attention_mask 支持（预留接口）

**使用示例**:
```python
from src.losses import create_structural_loss

loss_fn = create_structural_loss(
    alpha_k=1.0,      # K 对齐权重
    alpha_v=1.0,      # V 对齐权重
    alpha_attn=0.5,   # Attention KL 权重
    temperature=1.0   # Softmax 温度
)

loss, metrics = loss_fn(s_k, s_v, s_q, t_k, t_v, t_q)
```

---

## 🎯 实施路线图 v4.0

### 阶段 1: 对齐+投影（不碰 loss）✅ **已完成**

**目标**: 确保 MapProjectionAligner 的 forward 能跑通，形状正确，无 NaN

**步骤**:
1. ✅ 在 `src/` 下创建三个核心模块
2. ⏳ 在 `experiments/train_with_kv.py` 中接入（下一步）
3. ⏳ 运行 `profile_alignment.py` 验证（下一步）

**当前状态**: 模块已创建，待接入训练脚本

---

### 阶段 2: 挂上 StructuralKVLoss（真正蒸馏）⏳ **待实施**

**目标**: 使用新的 loss 进行真正的 KV 蒸馏训练

**步骤**:
1. 在训练循环中获取 student 的 `s_k, s_v, s_q`
2. 使用 `MapProjectionAligner` 获取 `t_k_proj, t_v_proj, t_q_proj`
3. 计算 `StructuralKVLoss`
4. 组合总损失：`loss_ce + lambda_kv * loss_struct`

---

## 📋 下一步行动清单

### 立即可做（阶段 1 收尾）

- [ ] **在 `experiments/train_with_kv.py` 中接入 `MapProjectionAligner`**
  ```python
  # 在训练脚本开头
  from src.map_projection_aligner import create_structured_aligner
  
  # 在初始化阶段
  aligner = create_structured_aligner(
      teacher_cfg, student_cfg,
      mode=config.kv_projection_mode  # "structured" 或 "flat"
  )
  
  # 在训练循环中
  k_s_proj, v_s_proj, q_s_proj = aligner(k_t, v_t, q_t, segment_ids)
  ```

- [ ] **运行 `profile_alignment.py` 验证**
  ```bash
  python experiments/profile_alignment.py --mode structured
  ```

- [ ] **验证检查点**:
  - 形状是否正确对齐到 student
  - 是否有 NaN
  - Attention 分布是否合理

### 准备阶段 2（蒸馏训练）

- [ ] **修改配置文件**:
  添加 `kv_projection_mode` 参数
  ```yaml
  kv_projection_mode: structured  # 或 "flat"
  loss_config:
    alpha_k: 1.0
    alpha_v: 1.0
    alpha_attn: 0.5
  ```

- [ ] **在训练脚本中引入 StructuralKVLoss**
  ```python
  from src.losses import create_structural_loss
  
  structural_loss_fn = create_structural_loss(
      alpha_k=config.loss_config.alpha_k,
      alpha_v=config.loss_config.alpha_v,
      alpha_attn=config.loss_config.alpha_attn
  )
  ```

---

## 🔬 实验对比计划

### A/B 测试矩阵

| 实验组 | mode | share_dim_proj | init_uniform | 描述 |
|--------|------|----------------|--------------|------|
| **Baseline** | flat | - | - | 旧方案（KVDimensionProjector） |
| **V4.0-1** | structured | True | False | 共享投影 + 随机初始化 |
| **V4.0-2** | structured | True | True | 共享投影 + 均匀初始化 ⭐ |
| **V4.0-3** | structured | False | True | 独立投影 + 均匀初始化 |

**预期结果**:
- V4.0-2 (shared + uniform) 应该是最稳定的
- V4.0-3 (per-head) 理论表达力更强，但参数多
- Baseline 作为参照，验证新方案的提升

---

## 📝 代码注释标准

所有新增模块都遵循以下标准：

1. ✅ **模块级文档字符串**：说明功能和 v4.0 更新
2. ✅ **类级文档字符串**：包含 Args、Example
3. ✅ **关键假设注释**：标注工程简化（如 segment_ids[0]）
4. ✅ **内置测试代码**：`if __name__ == "__main__"`
5. ✅ **清晰的 TODO**：为未来扩展预留位置

---

## 🎉 总结

### 核心改进（v4.0）

1. **双模式支持**: structured/flat 一键切换，方便 A/B 对比
2. **均匀初始化**: `init_uniform` 参数提供合理起点
3. **完整工程假设注释**: 记录所有简化，方便未来扩展
4. **阶段化验证**: profile_alignment 确保对齐正确再进入训练

### 与之前方案的兼容性

- ✅ 保留了所有旧的 baseline 路径（flat 模式）
- ✅ 不破坏现有代码，纯增量更新
- ✅ 配置文件一个字段切换模式

### 技术债务清理

- ✅ Anti-Flatten 设计彻底实施
- ✅ Q 显式处理（不再被忽略）
- ✅ 时间对齐的假设明确标注
- ✅ 为 mask 支持预留接口

---

**准备好进入下一阶段！** 🚀

请先运行 `profile_alignment.py` 验证模块正确性，然后我们一起接入训练脚本。
