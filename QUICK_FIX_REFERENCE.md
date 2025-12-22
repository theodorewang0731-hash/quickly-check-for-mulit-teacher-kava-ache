# 快速修复参考 - KV 对齐问题

## 🚨 如果你看到这些错误

### 错误 1: Shape Mismatch (12 vs 2)
```
RuntimeError: shape mismatch: value tensor of shape [4, 12, 50, 128] 
cannot be broadcast to indexing result of shape [4, 2, 50, 128]
```

**原因：** Teacher 有 12 个 KV heads，Student 有 2 个  
**解决：** 使用 `KVProjector`

```python
from experiments.kv_head_projector import KVProjector

# 在训练循环前
kv_projector = KVProjector(Ht=12, Hs=2, Dt=128, Ds=128).to(device)

# 在每个 batch
k_teacher, v_teacher = ...  # [B, 12, T, 128]
k_proj, v_proj = kv_projector(k_teacher, v_teacher)  # [B, 2, T, 128]
# 现在可以和 student KV 计算 loss
```

### 错误 2: Index Out of Bounds
```
RuntimeError: index 81 is out of bounds for dimension 2 with size 80
```

**原因：** 时间重采样索引越界  
**解决：** 使用 `safe_time_resample`

```python
from experiments.kv_head_projector import safe_time_resample, build_safe_linear_indices

# 生成安全索引
indices = build_safe_linear_indices(B=4, T_in=80, T_out=50, device=device)

# 安全重采样
x_resampled = safe_time_resample(x, indices)  # 不会越界
```

---

## ✅ 3 步快速集成

### Step 1: 导入模块

```python
from experiments.kv_head_projector import KVProjector
from experiments.alignment_v2 import resample_kv_with_interpolation
```

### Step 2: 初始化投影器

```python
# 方法 A: 手动指定参数
Ht = teacher_k.shape[1]  # 从实际张量获取
Hs = student_k.shape[1]
Dt = teacher_k.shape[-1]
Ds = student_k.shape[-1]
kv_projector = KVProjector(Ht, Hs, Dt, Ds).to(device)

# 方法 B: 使用集成版本 (推荐)
from experiments.kv_dimension_projector import KVDimensionProjector

projector = KVDimensionProjector(
    teacher_configs={"Qwen2-7B": {"d_model": 3584, "num_layers": 28}},
    student_d_model=2048,
    student_num_kv_heads=2,  # 关键：指定学生 KV 头数
    mlp_ratio=1.0
)
```

### Step 3: 在训练循环中使用

```python
for batch in dataloader:
    # 提取 teacher KV
    teacher_k, teacher_v = extract_teacher_kv(...)  # [B, Ht, T_t, Dt]
    
    # 方法 A: 手动投影
    k_proj, v_proj = kv_projector(teacher_k, teacher_v)  # [B, Hs, T_t, Ds]
    k_aligned = resample_kv_with_interpolation(k_proj, T_student)
    v_aligned = resample_kv_with_interpolation(v_proj, T_student)
    
    # 方法 B: 一步到位 (集成版本会自动处理)
    k_aligned, v_aligned = projector.project_teacher_kv(
        "Qwen2-7B", teacher_k, teacher_v
    )
    
    # 计算 loss (现在不会报错!)
    loss_k = F.mse_loss(k_aligned, student_k)
    loss_v = F.mse_loss(v_aligned, student_v)
```

---

## 🔍 调试检查清单

如果还有问题，按顺序检查：

### 1. 确认 KV 头数（不是 Q 头数！）

```python
# ❌ 错误
num_heads = config.num_attention_heads  # 这是 Q 的头数

# ✅ 正确
num_kv_heads = config.num_key_value_heads  # GQA/MQA 的 KV 头数

# ✅ 或者从张量推断
num_kv_heads = teacher_k.shape[1]  # 假设 [B, H, T, D]
```

### 2. 打印所有关键 shapes

```python
print(f"Teacher K: {teacher_k.shape}")
print(f"Teacher V: {teacher_v.shape}")
print(f"Student K: {student_k.shape}")
print(f"Student V: {student_v.shape}")
print(f"After projection: {k_proj.shape}")
print(f"After resampling: {k_resampled.shape}")
```

### 3. 检查配置参数

```python
print(f"Teacher config:")
print(f"  num_attention_heads: {teacher_config.num_attention_heads}")
print(f"  num_key_value_heads: {teacher_config.num_key_value_heads}")
print(f"  hidden_size: {teacher_config.hidden_size}")

print(f"Student config:")
print(f"  num_attention_heads: {student_config.num_attention_heads}")
print(f"  num_key_value_heads: {student_config.num_key_value_heads}")
print(f"  hidden_size: {student_config.hidden_size}")
```

### 4. 验证索引范围

```python
indices = build_safe_linear_indices(B, T_in, T_out, device)
print(f"Indices shape: {indices.shape}")
print(f"Indices dtype: {indices.dtype}")
print(f"Indices range: [{indices.min()}, {indices.max()}]")
print(f"T_in: {T_in}, should be > {indices.max()}")
```

---

## 📦 文件清单

修复涉及的文件：

```
新增:
  experiments/kv_head_projector.py        - 头数投影器
  tests/test_kv_fixes.py                  - 测试套件
  PRECISE_FIX_GUIDE.md                    - 详细指南
  KV_FIX_SUMMARY.md                       - 修复总结
  QUICK_FIX_REFERENCE.md                  - 本文件

修改:
  experiments/alignment_v2.py             - 安全时间重采样
  experiments/kv_dimension_projector.py   - 集成头数投影
```

---

## 🧪 快速测试

在 HPC 上运行：

```bash
# 完整测试
python tests/test_kv_fixes.py

# 快速验证（复制粘贴到 Python）
python << 'EOF'
import torch
from experiments.kv_head_projector import KVProjector, safe_time_resample

# 测试 1: 头数投影
proj = KVProjector(12, 2, 128, 128)
k = torch.randn(4, 12, 50, 128)
v = torch.randn(4, 12, 50, 128)
k_out, v_out = proj(k, v)
assert k_out.shape == (4, 2, 50, 128), "Head projection failed!"
print("✓ Test 1 passed: Head projection works")

# 测试 2: 时间重采样
from experiments.kv_head_projector import build_safe_linear_indices
x = torch.randn(4, 2, 80, 128)
indices = build_safe_linear_indices(4, 80, 50, x.device)
x_out = safe_time_resample(x, indices)
assert x_out.shape == (4, 2, 50, 128), "Time resampling failed!"
print("✓ Test 2 passed: Time resampling works")

print("\n✓ All quick tests passed! Ready for training.")
EOF
```

---

## 💡 常见问题

### Q: 我的模型不是 GQA，需要修改吗？
A: 不需要！如果 teacher 和 student 头数相同，`KVProjector` 会自动跳过投影（零开销）。

### Q: 会影响训练速度吗？
A: 影响很小：
- 头数投影：线性变换，可忽略（~1-2% overhead）
- 安全重采样：只是多了 clamp 和类型转换，几乎无开销

### Q: 需要重新预训练吗？
A: 不需要！这只是对齐层，训练时会自动学习。

### Q: 支持哪些模型组合？
A: 所有！只要知道 teacher 和 student 的 KV 头数：
- Qwen (28 heads) → TinyLlama (4 heads) ✓
- Llama-70B (8 heads) → Llama-7B (32 heads) ✓
- GPT (12 heads) → Student (2 heads) ✓

---

## 📞 需要帮助？

如果修复后仍有问题，请提供：

1. **完整错误堆栈**（包含文件名和行号）
2. **张量 shapes**（teacher_k, student_k, 等）
3. **模型配置**（num_attention_heads, num_key_value_heads）
4. **你的代码片段**（如何初始化和使用投影器）

---

## ✨ 一句话总结

**头数不匹配？用 `KVProjector`。索引越界？用 `safe_time_resample`。就这么简单！** 🚀
