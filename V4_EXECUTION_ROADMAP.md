# v4.0 执行路线图（Phase 2 完整作战指南）

**创建时间**: 2025年12月9日  
**当前状态**: Phase 1 完成 ✅ | Phase 2 开始 ⏳  
**目标**: flat vs structured A/B 测试

---

## 📍 当前位置

### ✅ Phase 1 已完成
- `src/headwise_projector.py` - Anti-Flatten 投影器
- `src/time_warping.py` - Segment 时间对齐
- `src/map_projection_aligner.py` - 统一对齐接口
- `src/losses.py` - StructuralKVLoss（暂不启用）
- `experiments/profile_alignment.py` - 验证工具

### ⏳ Phase 2 现在开始
**核心任务**: 把这些模块接入 `experiments/train_with_kv.py`，在相同 loss 下对比 flat vs structured

---

## 🥇 第一步：集成代码（Integrate）

**目标**: 让训练脚本支持双模式切换

### 1.1 在 `train_with_kv.py` 顶部导入

```python
# 原有
from experiments.kv_dimension_projector import (
    KVDimensionProjector,
    flatten_kv_heads,
    unflatten_kv_heads  # 如果有
)

# 新增 v4.0
from src.map_projection_aligner import MapProjectionAligner
```

添加 HF past_key_values → 5D 工具：

```python
def stack_past_kv(past_key_values):
    """
    HF: tuple[(k,v), ...] -> [B, L, H, T, D]
    
    Args:
        past_key_values: HF 格式的 past_key_values
    
    Returns:
        k, v: [B, L, H, T, D] 形状的 tensors
    """
    k_list, v_list = [], []
    for k, v in past_key_values:
        # k, v: [B, H, T, D]
        k_list.append(k.unsqueeze(1))  # [B, 1, H, T, D]
        v_list.append(v.unsqueeze(1))
    k = torch.cat(k_list, dim=1)  # [B, L, H, T, D]
    v = torch.cat(v_list, dim=1)
    return k, v
```

### 1.2 在 `main()` 中初始化双模式 Aligner

```python
# 读取模式配置
kv_mode = getattr(args, "kv_projection_mode", "flat")
print(f"🚀 Initializing Alignment System in mode: [ {kv_mode.upper()} ]")

if kv_mode == "structured":
    # 🔵 蓝方：v4.0 地图投影
    aligner = MapProjectionAligner(
        teacher_config=teacher.config,
        student_config=student.config,
        mode="structured",
        share_dim_proj=getattr(args, "share_dim_proj", True),
        init_uniform=getattr(args, "init_uniform", True),
    ).to(device)
    print("✅ Enabled: MapProjectionAligner (Headwise + TimeWarp)")
    print(f"   share_dim_proj: {getattr(args, 'share_dim_proj', True)}")
    print(f"   init_uniform: {getattr(args, 'init_uniform', True)}")

else:
    # 🔴 红方：Baseline flatten
    aligner = KVDimensionProjector(
        teacher_heads=teacher.config.num_attention_heads,
        student_heads=student.config.num_attention_heads,
        teacher_head_dim=teacher.config.hidden_size // teacher.config.num_attention_heads,
        student_head_dim=student.config.hidden_size // student.config.num_attention_heads,
        mlp_ratio=getattr(args, "mlp_ratio", 4),
        use_elastic=getattr(args, "use_elastic", False),
    ).to(device)
    print("✅ Enabled: KVDimensionProjector (Flatten + MLP)")

# 优化器包含 aligner 参数
optimizer = torch.optim.AdamW(
    list(student.parameters()) + list(aligner.parameters()),
    lr=args.lr,
)
```

### 1.3 在训练循环中接入双路径对齐

**关键控制变量设计**：

```python
for step, batch in enumerate(dataloader):
    batch = {k: v.to(device) for k, v in batch.items()}
    
    # ===== Teacher Forward =====
    with torch.no_grad():
        t_out = teacher(**batch, output_hidden_states=True, use_cache=True)
        t_k_raw, t_v_raw = stack_past_kv(t_out.past_key_values)  # [B,L_t,H_t,T_t,D_t]
        
        # 获取或创建 segment_ids
        segment_ids = batch.get("segment_ids", None)
        if segment_ids is None:
            B, L_t, H_t, T_t, D_t = t_k_raw.shape
            segment_ids = torch.zeros(B, T_t, dtype=torch.long, device=t_k_raw.device)
    
    # ===== Student Forward =====
    s_out = student(**batch, output_hidden_states=True, use_cache=True)
    s_logits = s_out.logits
    s_k, s_v = stack_past_kv(s_out.past_key_values)  # [B,L_s,H_s,T_s,D_s]
    
    # CE loss
    loss_task = F.cross_entropy(
        s_logits.view(-1, s_logits.size(-1)),
        batch["labels"].view(-1),
        ignore_index=-100
    )
    
    # ===== 🔥 关键分支：双路径对齐 =====
    if kv_mode == "structured":
        # 🔵 v4.0: MapProjectionAligner
        t_k_proj, t_v_proj, _ = aligner(t_k_raw, t_v_raw, None, segment_ids)
        # 输出: [B, L_s, H_s, T_s, D_s]
        
    else:
        # 🔴 Baseline: flatten -> projector -> unflatten
        t_k_flat = flatten_kv_heads(t_k_raw)  # [B,L_t,T_t,H_t*D_t]
        t_v_flat = flatten_kv_heads(t_v_raw)
        
        t_k_proj_flat = aligner(t_k_flat)     # [B,L_s,T_s,H_s*D_s]
        t_v_proj_flat = aligner(t_v_flat)
        
        t_k_proj = unflatten_kv_heads(t_k_proj_flat, student.config.num_attention_heads)
        t_v_proj = unflatten_kv_heads(t_v_proj_flat, student.config.num_attention_heads)
    
    # ===== KV Loss（保持不变）=====
    loss_k = kv_loss_fn(s_k, t_k_proj)
    loss_v = kv_loss_fn(s_v, t_v_proj)
    loss_kv = loss_k + loss_v
    
    # 总损失
    loss = loss_task + args.lambda_kv * loss_kv
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 日志
    if step % args.log_interval == 0:
        print(f"[Step {step}] loss_task={loss_task.item():.4f}, "
              f"loss_kv={loss_kv.item():.4f}, total={loss.item():.4f}")
```

### 1.4 添加命令行参数

在 `argparse` 部分添加：

```python
parser.add_argument("--kv_projection_mode", type=str, default="flat",
                   choices=["flat", "structured"],
                   help="KV 对齐模式：flat (baseline) 或 structured (v4.0)")
parser.add_argument("--share_dim_proj", action="store_true",
                   help="(structured 模式) 是否共享维度投影")
parser.add_argument("--init_uniform", action="store_true", default=True,
                   help="(structured 模式) 是否使用均匀初始化")
```

---

## 🥈 第二步：冒烟测试（Smoke Test）

**目标**: 确保管线能跑通，没有 shape bug / NaN

### 2.1 Profile 工具测试

**测试 v4.0 路径**:
```bash
python experiments/profile_alignment.py --mode structured
```

**测试 Baseline 路径**:
```bash
python experiments/profile_alignment.py --mode flat
```

**检查项**:
- ✅ KV 形状正确: `[B, L, H, T, D]`
- ✅ 无报错
- ✅ 无 NaN

### 2.2 训练脚本冒烟（max_steps=10）

**测试 v4.0**:
```bash
python experiments/train_with_kv.py \
    --kv_projection_mode structured \
    --share_dim_proj \
    --init_uniform \
    --max_steps 10 \
    --output_dir debug_v4
```

**测试 Baseline**:
```bash
python experiments/train_with_kv.py \
    --kv_projection_mode flat \
    --max_steps 10 \
    --output_dir debug_baseline
```

**检查项**:
- ✅ 正确打印模式信息
- ✅ 10 步内无崩溃
- ✅ Loss 有正常数值（非 NaN/Inf）
- ✅ 日志中 KV 形状正确

---

## 🥉 第三步：正式 A/B 实验（Execution）

**目标**: 红方 vs 蓝方对局，产出实验数据

### 3.1 实验 A：Baseline（红方 🔴）

**配置**:
- `kv_projection_mode = "flat"`
- 用于建立性能基线

**命令**:
```bash
python experiments/train_with_kv.py \
    --kv_projection_mode flat \
    --run_name baseline_flat_run \
    --output_dir outputs/ab_test/baseline_flat \
    --epochs 3 \
    --batch_size 8 \
    --lr 5e-5 \
    --lambda_kv 0.5
```

### 3.2 实验 B：v4.0-2 推荐版（蓝方 🔵）

**配置**:
- `kv_projection_mode = "structured"`
- `share_dim_proj = True`
- `init_uniform = True`

**命令**:
```bash
python experiments/train_with_kv.py \
    --kv_projection_mode structured \
    --share_dim_proj \
    --init_uniform \
    --run_name v4_structured_uniform_run \
    --output_dir outputs/ab_test/v4_structured \
    --epochs 3 \
    --batch_size 8 \
    --lr 5e-5 \
    --lambda_kv 0.5
```

### 3.3 结果分析

**重点观察三条曲线**:

#### 1. Training Loss（前 100-500 steps）
- **预期**: 蓝方下降更快或更平滑
- **说明**: 新对齐方式 + 初始化提供了更好的"地图"

#### 2. 验证集指标（PPL / GSM8K）
- **对比**: 相同 training steps 下的性能
- **目标**: structured 在 reasoning/数学题上表现更好

#### 3. 对齐内部指标（可选）
- `cos(s_k, t_k_proj)` 平均值
- 查看对齐质量

**生成报告**:
```bash
# 运行结果对比脚本
python utils/compare_runs.py \
    --baseline outputs/ab_test/baseline_flat \
    --experimental outputs/ab_test/v4_structured \
    --output V4_AB_TEST_RESULTS.md
```

---

## 🧬 未来扩展：接入 StructuralKVLoss（Phase 2.5）

**时机**: 完成 A/B 实验，确认结构化对齐有收益后

### 步骤

1. **初始化损失函数**:
```python
from src.losses import create_structural_loss

structural_loss_fn = create_structural_loss(
    alpha_k=1.0,
    alpha_v=1.0,
    alpha_attn=0.5,
    temperature=1.0
).to(device)
```

2. **在训练循环中添加**:
```python
# 在 aligner 之后
if kv_mode == "structured" and args.use_structural_loss:
    # 获取 Q（需要修改 aligner 返回值）
    t_k_proj, t_v_proj, t_q_proj = aligner(t_k_raw, t_v_raw, t_q_raw, segment_ids)
    
    # 也需要 student 的 Q
    s_q = stack_past_kv_q(s_out.past_key_values)  # 需要实现
    
    # 计算结构化损失
    loss_struct, struct_metrics = structural_loss_fn(
        s_k, s_v, s_q,
        t_k_proj, t_v_proj, t_q_proj
    )
    
    # 添加到总损失
    loss = loss_task + args.lambda_kv * loss_kv + args.lambda_struct * loss_struct
```

3. **命令行参数**:
```python
parser.add_argument("--use_structural_loss", action="store_true",
                   help="是否使用 StructuralKVLoss")
parser.add_argument("--lambda_struct", type=float, default=0.1,
                   help="StructuralKVLoss 权重（建议 0.05-0.1）")
```

---

## ✅ 执行检查清单

### Phase 2.1: 代码集成
- [ ] 1.1 在 `train_with_kv.py` 添加导入和 `stack_past_kv`
- [ ] 1.2 初始化双模式 Aligner
- [ ] 1.3 改写训练循环的对齐分支
- [ ] 1.4 添加命令行参数

### Phase 2.2: 冒烟测试
- [ ] 2.1 运行 `profile_alignment.py` (structured & flat)
- [ ] 2.2 运行 10 步训练测试 (structured & flat)
- [ ] 检查无崩溃、无 NaN

### Phase 2.3: A/B 实验
- [ ] 3.1 启动 Baseline 实验（红方）
- [ ] 3.2 启动 v4.0 实验（蓝方）
- [ ] 3.3 收集并分析结果
- [ ] 更新 `DEVELOPMENT_HISTORY.md` 记录结论

### Phase 2.5: 扩展（可选）
- [ ] 接入 StructuralKVLoss
- [ ] 进行消融实验

---

## 📊 预期结果矩阵

| 实验组 | mode | share_dim | init_uniform | 预期性能 | 状态 |
|--------|------|-----------|--------------|---------|------|
| **Baseline** | flat | - | - | 基准 | ⏳ 待运行 |
| **V4.0-1** | structured | True | False | +2% | ⏸️ 可选 |
| **V4.0-2** | structured | True | True | +5% ⭐ | ⏳ 待运行 |
| **V4.0-3** | structured | False | True | +6% | ⏸️ 可选 |

---

## 🎯 成功标准

### 最低目标
- ✅ 两种模式都能正常训练完成
- ✅ structured 模式不比 flat 差

### 理想目标
- ✅ structured 在 validation 指标上 **+3~5%**
- ✅ training loss 下降更快/更稳定
- ✅ 内部对齐指标（cos sim）更高

### 论文级目标
- ✅ 达到理想目标 +
- ✅ 有显著性检验（p < 0.05）
- ✅ 消融实验完整

---

## 📝 下一步行动

**立即可做**:
1. 打开 `experiments/train_with_kv.py`
2. 按照 1.1-1.4 的步骤修改代码
3. 运行冒烟测试
4. 启动 A/B 实验

**需要的文件**:
- 如果你把当前的 `train_with_kv.py` 相关部分发给我，我可以帮你做精确的代码改写

---

**准备开始执行！** 🚀

最后更新: 2025年12月9日
