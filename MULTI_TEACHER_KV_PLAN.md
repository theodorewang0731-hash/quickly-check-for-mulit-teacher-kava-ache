# 多教师 KV 蒸馏完整实施方案

## 一、方法总览

**目标**：让学生模型学到**多位教师的推理轨迹（KV）**，并能按输入/任务选择或加权不同"思维模式"。

**训练信号**：融合后的教师 KV（逐层、逐token），用 **KV-MSE/Cosine** 监督学生的 KV；可叠加少量 CE / 注意力 KL 稳定生成。

**核心差异**（vs 多教师 logits KD）：我们不是只加权"最终输出分布"，而是**在 KV 轨迹空间逐层逐步加权**，能保留和转移"怎么想"的信息（注意力走向、层内状态）。

---

## 二、对齐策略（默认方案，所有阶段通用）

### 1. 时间/序列对齐（必须）
- **Teacher Forcing**：老师和学生使用同一输入/同一金标准输出，强制展开到同一长度 T
- **pad + mask**：不足 T 的位置 padding，并用 kv_mask 屏蔽（**不要截断**）
- **异 tokenizer**：构造**字符级对齐矩阵 A ∈ R^{T_s×T_t}**（按字符交并比/IoU），将教师时刻柔性汇聚到学生时刻

### 2. 层对齐
- **比例映射 + 线性插值**：教师层 L_t → 学生层 L_s
- l_s = round(l_t * L_s / L_t)
- 需要平滑时对相邻层插值

### 3. 维度/头数对齐
- **线性适配器（每位教师一套）**：K_hat = K @ W_k, V_hat = V @ W_v
- **头数不一致**：
  - H_t > H_s：分组平均/1×1 conv 降头
  - H_t < H_s：1×1 conv 扩头
  - 进阶：按注意力相似度做头匹配再聚合

### 4. 位置编码对齐
- **同为 RoPE**：做 **NTK/相位缩放**，把教师频率重标定到学生频率
- 混合（RoPE ↔ ALiBi/绝对PE）：优先选同类教师；必要时"去位置→线性投影→再注入学生 RoPE"

---

## 三、训练损失与融合

### 融合（逐层逐步）
```
KV_fused = Σ α_i * KV_i^align
```

- **v1**：固定权重（0.7/0.3/...）
- **v2**：**相似度路由**（根据输入或前几步 KV 与各教师原型的余弦相似度 → softmax 得 α）
- **v3**：**可学习路由器**（小 MLP/Gate 输出 α_i，加熵正则防塌缩；可"浅层偏A，深层偏B"）

### 损失函数
```
L = λK * MSE(K_s, K_f) + λV * MSE(V_s, V_f) 
  + β * Cos(KV_s, KV_f) 
  + γ * KL(Attn_s || Attn_f) 
  + δ * CE
```

- **层权重退火**：先蒸浅层，后期再放大深层权重（深层更难对齐）

---

## 四、实施分阶段计划

### Phase 0：单教师 KaVA 可行性 ✓ 已完成
- 确认"KV 当监督"在模型/数据上有效
- 验证 KV loss/CE 都能下降

### Phase 1：多教师最小闭环（同模型+双 prompt）
**目标**：验证多教师 KV 融合基础流程

**步骤**：
1. 同模型两种 prompt，取两份 KV
2. 按时间/层/维度/头/位置对齐
3. 固定权重融合（α = [0.7, 0.3]）
4. KV-MSE + Cosine 蒸馏
5. 评估 10→30 步曲线（train/val）

**输出**：
- 对齐可视化（层级 MSE 热图）
- train/val loss 曲线
- 确认不是初始化偶然

### Phase 2：多样本 + 验证集 + 等长输入
**目标**：扩展到多样本，验证泛化

**步骤**：
1. ≥32 train / ≥8 val 样本
2. Teacher forcing 到同一 T
3. 异 tokenizer 用 A 对齐
4. 监控 train/val gap（≤2~3%）

**输出**：
- 稳定的 train/val 下降曲线
- 对齐矩阵可视化

### Phase 3：真多教师（不同 checkpoint）
**目标**：使用真实不同教师

**步骤**：
1. 替换为多个 checkpoint 的教师
2. 增加线性适配器（各教师一套）
3. 必要时启用头匹配
4. 维持 Phase 2 评估协议

**输出**：
- 多教师融合后的 KV loss
- 任务指标（PPL/准确率）

### Phase 4：路由器（固定→相似度→可学）
**目标**：动态选择教师权重

**步骤**：
1. v2 相似度路由：以输入 embedding 或前几步 KV 计算 α
2. v3 可学路由：小 MLP/Gate + 熵正则
3. 支持层感知路由（浅层/深层不同 α）

**输出**：
- 路由权重可视化（样本×教师）
- 相比固定权重的提升

### Phase 5（可选）：跨架构公共空间 Z
**目标**：处理差异极大的教师组合

**步骤**：
1. 为每位教师学 g_i([K,V]) → Z_i
2. 学生学 h_s([K,V]) → Z_s
3. 在 Z 空间融合监督（MSE/Cos/Barlow/CCA）

**输出**：
- 跨架构蒸馏成功案例
- Z 空间特征可视化

---

## 五、默认超参（可直接用）

```python
# 序列长度
MAX_LENGTH = 512  # 按任务调整 256/512/1024

# 损失权重
LAMBDA_K = 1.0    # Key MSE
LAMBDA_V = 1.0    # Value MSE
BETA = 0.1        # Cosine
GAMMA = 0.0       # Attention KL（稳定后逐步加到 0.05）
DELTA = 0.0       # CE（稳定后逐步加到 0.1）

# 学习率
LR_LORA = 2e-4    # LoRA 微调
LR_FULL = 5e-5    # 全参数小模型
WARMUP_RATIO = 0.03

# 批量
EFFECTIVE_BATCH = 128  # 全局有效 batch
GRAD_ACCUM_STEPS = 4   # 根据显存调整

# 层权重退火（前 30% 步只蒸前 1/3 层）
LAYER_WARMUP_STEPS = 0.3  # 训练步数比例
```

---

## 六、HPC 优化建议

### 1. 离线预计算教师 KV
```bash
# SLURM job array 并行
sbatch scripts/dump_teacher_kv.sh --array=0-99
```
- 教师前向 + 对齐/投影/头聚合/层映射全部预算
- 存储格式：zarr/webdataset/npz
- 训练时只读磁盘，大幅加速

### 2. 分布式训练
- FSDP 或 DeepSpeed ZeRO-3
- BFloat16 混合精度
- FlashAttention-2
- 梯度检查点
- 梯度裁剪
- Warmup + Cosine LR

### 3. 路由策略
1. 先固定权重（确认基础流程）
2. 切换相似度路由（动态但不可学）
3. 尝试可学路由 + 熵正则

---

## 七、检查清单（避免踩坑）

- [ ] **绝不截断** KV（防止后半段丢失导致虚假提升）
- [ ] **mask 传入所有损失**（MSE/Cos/KL 都要处理 padding）
- [ ] **验证 train/val 都在降**（差距 ≤2~3%）
- [ ] **单教师 vs 多教师对照**（确认多教师有增益）
- [ ] **跨架构不稳时用公共空间 Z**（避免生拼硬对）
- [ ] **层权重退火**（深层更难对齐，后期才加大权重）
- [ ] **路由熵正则**（防止塌缩到单一教师）

---

## 八、评估指标

### 训练监控
- **KV Loss**（分层、分步）
  - Key MSE per layer
  - Value MSE per layer
  - Cosine similarity
- **Attention KL**（可选）
- **CE Loss**（可选）

### 任务评估
- **PPL**（困惑度）
- **准确率**（分类/QA 任务）
- **CoT 成功率**（推理任务）
- **生成质量**（BLEU/ROUGE）

### 可视化
- 层级 MSE 热图（layer × step）
- 路由权重分布（sample × teacher）
- 对齐矩阵可视化（时间对齐）
- Z 空间 t-SNE/UMAP

---

## 九、快速启动命令

### Phase 1 测试（本地）
```bash
python experiments/train_multi_teacher_kv.py \
    --phase 1 \
    --model_name Qwen/Qwen2.5-7B \
    --teacher_prompts "默认prompt" "CoT prompt" \
    --n_samples 10 \
    --epochs 1 \
    --device cuda:0
```

### Phase 3 完整训练（HPC）
```bash
sbatch scripts/run_multi_teacher.sh \
    --phase 3 \
    --teachers checkpoint1 checkpoint2 checkpoint3 \
    --n_samples 1000
```

### Phase 4 路由训练
```bash
sbatch scripts/run_router_training.sh \
    --router_type learnable \
    --entropy_weight 0.01
```

---

## 十、项目结构

```
quickly check/
├── align/                      # 对齐模块
│   ├── tokenizer_align.py     # 字符级对齐矩阵
│   ├── time_align.py          # 时间对齐
│   ├── layer_map.py           # 层映射
│   ├── head_dim_adapter.py    # 维度/头数适配
│   └── rope_scale.py          # RoPE 缩放
├── teacher/                    # 教师模块
│   ├── dump_teacher_kv.py     # 离线计算教师 KV
│   └── router_proto.py        # 教师原型特征
├── fuse/                       # 融合模块
│   ├── fuse_kv.py             # KV 融合
│   └── router.py              # 路由器（固定/相似度/可学）
├── student/                    # 学生模块
│   ├── forward.py             # 学生前向
│   └── loss.py                # 多目标损失
├── experiments/
│   └── train_multi_teacher_kv.py  # 主训练脚本
├── scripts/
│   ├── run_multi_teacher.sh   # HPC 训练脚本
│   ├── dump_teacher_kv.sh     # 离线计算教师 KV
│   └── run_router_training.sh # 路由训练脚本
└── eval/
    ├── kv_metrics.py          # KV 对齐指标
    ├── task_metrics.py        # 任务指标
    └── visualize.py           # 可视化工具
```

---

## 十一、时间线估算

| 阶段 | 工作量 | 预计时间 | 输出 |
|------|--------|----------|------|
| Phase 0 | 已完成 | - | 单教师验证 |
| Phase 1 | 实现对齐+融合 | 2-3天 | 双prompt闭环 |
| Phase 2 | 扩展数据集 | 1-2天 | 多样本验证 |
| Phase 3 | 真多教师 | 2-3天 | 适配器训练 |
| Phase 4 | 路由器 | 3-5天 | 动态路由 |
| Phase 5 | 跨架构Z空间 | 5-7天 | 通用方案 |

**总计**：2-3 周完成 Phase 1-4，4 周完成全部。

---

> **核心原则**：
> 1. **对齐先行**：时间/层/维度/头/位置全方位对齐
> 2. **渐进复杂**：固定权重→相似度路由→可学路由
> 3. **稳定优先**：train/val 都要降，不能只看 train
> 4. **充分可视**：层级热图、路由权重、对齐矩阵
> 5. **离线预算**：教师 KV 预计算，训练时只读盘
