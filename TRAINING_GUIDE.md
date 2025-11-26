# 🚀 KAVA 地图投影实战训练指南

## 硬件配置
- **GPU**: RTX 4070 (8GB VRAM)
- **模型组合**: Qwen2.5-1.5B (Teacher) → Qwen2.5-0.5B (Student)
- **优化策略**: 4-bit 量化 + 梯度累积

---

## 📋 执行步骤

### Step 1: 安装依赖

```bash
pip install bitsandbytes scipy accelerate datasets transformers
```

**依赖说明**:
- `bitsandbytes`: 4-bit 量化（显存救星）
- `scipy`: 数学计算辅助
- `accelerate`: 分布式训练支持
- `datasets`: Hugging Face 数据集
- `transformers`: 模型加载

---

### Step 2: 验证环境

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

**预期输出**:
```
CUDA: True, Device: NVIDIA GeForce RTX 4070
```

---

### Step 3: 启动训练

```bash
python train_full_dataset.py
```

---

## 📊 训练监控指南

### 核心指标: Cosine Similarity

训练过程中，请重点关注控制台输出的 **CosSim** 值：

| 阶段 | 步数 | CosSim 范围 | 状态 | 说明 |
|-----|------|------------|------|------|
| **Phase 1** | 0-50 | 0.20-0.50 | 🔄 Adapting | 模型适应中，正常现象 |
| **Phase 2** | 50-100 | 0.50-0.70 | ⚠️ Learning | 开始学习方向对齐 |
| **Phase 3** | 100-200 | 0.70-0.90 | 📈 Good | 显著进步，继续训练 |
| **Phase 4** | 200+ | 0.90-0.95 | 🎯 Great | 接近目标，效果良好 |
| **Target** | - | >0.95 | ✅ Excellent | 完美对齐，训练成功！ |

### 预期输出示例

```
[Step 0000] Loss: 0.8234 | CosSim: 0.1766 🔄 Adapting
[Step 0050] Loss: 0.4521 | CosSim: 0.5479 ⚠️ Learning
[Step 0100] Loss: 0.2145 | CosSim: 0.7855 📈 Good
[Step 0150] Loss: 0.1023 | CosSim: 0.8977 📈 Good
[Step 0200] Loss: 0.0432 | CosSim: 0.9568 ✅ Excellent
```

---

## ⚙️ 配置说明

### 默认配置（适配 8GB VRAM）

```python
CONFIG = {
    "batch_size": 2,                      # 单批次样本数
    "gradient_accumulation_steps": 16,    # 梯度累积（等效 Batch=32）
    "max_length": 512,                    # 序列最大长度
    "lr_projector": 1e-3,                 # Projector 学习率（从头学）
    "lr_student": 5e-5,                   # Student 学习率（微调）
    "save_steps": 200,                    # 每 200 步保存检查点
}
```

### 显存紧急配置（如遇 OOM）

如果启动时报 `CUDA out of memory`，修改配置：

```python
"batch_size": 1,                      # 降至 1
"gradient_accumulation_steps": 32,    # 增至 32（保持等效 Batch=32）
"max_length": 384,                    # 缩短序列（可选）
```

---

## 🎯 成功标志

### 1. 训练完成标志

```
✅ Training Complete!
💾 Final Projector saved: final_projector.pth
💾 Final Student saved: final_student/
```

### 2. 关键文件

- `final_projector.pth`: 训练好的弹性瓶颈（Elastic Bottleneck）
- `final_student/`: 蒸馏后的学生模型
- `checkpoints/proj_step_*.pth`: 中间检查点

### 3. 验证标准

最终 CosSim 应满足：
- **优秀**: CosSim ≥ 0.95
- **良好**: CosSim ≥ 0.90
- **及格**: CosSim ≥ 0.80

---

## 🔍 常见问题排查

### Q1: 显存溢出 (OOM)

**症状**:
```
RuntimeError: CUDA out of memory. Tried to allocate 1.23 GiB
```

**解决方案**:
1. 降低 `batch_size` 到 1
2. 减少 `max_length` 到 384 或 256
3. 检查后台进程（关闭不必要的 GPU 占用）

### Q2: CosSim 不上升

**症状**:
- 200 步后 CosSim 仍 <0.50
- Loss 下降但 CosSim 停滞

**可能原因**:
1. 学习率过低：尝试 `lr_projector=2e-3`
2. 数据质量问题：检查 GSM8K 是否正确加载
3. 模型维度不匹配：确认 Teacher/Student 配置正确

**调试命令**:
```python
# 在训练脚本中添加调试输出
print(f"Teacher KV shape: {t_kv.shape}")
print(f"Student KV shape: {s_kv.shape}")
print(f"Projected shape: {t_proj.shape}")
```

### Q3: Loss 震荡

**症状**:
- Loss 上下波动剧烈
- CosSim 时高时低

**解决方案**:
1. 增加梯度累积步数到 32
2. 降低学习率：`lr_projector=5e-4`
3. 启用更强的梯度裁剪：`max_grad_norm=0.5`

### Q4: 数据集下载失败

**症状**:
```
ConnectionError: Couldn't reach https://huggingface.co/datasets/gsm8k
```

**解决方案**:
```bash
# 方法 1: 设置 HF 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 方法 2: 手动下载数据集
git clone https://huggingface.co/datasets/gsm8k
# 然后修改代码：load_dataset("./gsm8k", ...)
```

---

## 📈 性能预期

### 训练速度

- **RTX 4070 (8GB)**: ~2-3 it/s
- **每 Epoch**: ~3-4 小时（7473 样本 / 2 batch_size）
- **推荐步数**: 500-1000 步（足够看到效果）

### 显存占用

| 组件 | 显存占用 | 说明 |
|-----|---------|------|
| Teacher (4-bit) | ~1.5GB | 量化后 |
| Student (bf16) | ~1.2GB | 半精度 |
| Projector | ~0.3GB | MLP 参数 |
| 激活值 (Batch=2) | ~3.5GB | 前向+反向 |
| **Total** | ~6.5GB | 预留 1.5GB 余量 |

---

## 🎓 理论回顾

### 为什么地图投影有效？

**传统 MSE**:
```
Teacher KV: [100, 100]  (高置信度)
Student KV: [1, 1]      (低置信度)
MSE Loss = 76.57        ❌ 误判为错误
```

**Mercator Loss**:
```
Teacher Direction: [0.707, 0.707]
Student Direction: [0.707, 0.707]
Cosine Similarity = 1.0  ✅ 识别相同语义
Mercator Loss = 0.0      ✅ 完美对齐
```

**核心洞察**:
- RoPE 模型：方向 = 语义，模长 = 置信度
- 蒸馏目标：学习语义方向，而非数值大小
- 地图投影：归一化到单位球，只比较方向

---

## 🔧 高级调参

### 实验组合推荐

| 实验名 | mlp_ratio | alpha | beta | 适用场景 |
|-------|-----------|-------|------|---------|
| **Pure Direction** | 1.0 | 1.0 | 0.0 | 纯方向对齐 |
| **Weak Constraint** | 1.0 | 1.0 | 0.01 | 推荐（防塌缩） |
| **Strong Constraint** | 1.0 | 1.0 | 0.1 | 数值差异小时 |
| **High Capacity** | 2.0 | 1.0 | 0.01 | 复杂任务 |
| **Low Capacity** | 0.5 | 1.0 | 0.01 | 显存受限 |

### 调参优先级

1. **首先调整**: `batch_size` + `gradient_accumulation_steps`（显存适配）
2. **其次调整**: `lr_projector`（收敛速度）
3. **最后调整**: `beta`（仅当数值异常时）

---

## 📝 实验记录模板

```yaml
experiment:
  name: "KAVA-Mercator-1.5B-to-0.5B"
  date: "2025-11-26"
  
hardware:
  gpu: "RTX 4070 8GB"
  batch_size: 2
  grad_accum: 16
  
models:
  teacher: "Qwen/Qwen2.5-1.5B-Instruct"
  student: "Qwen/Qwen2.5-0.5B"
  
config:
  mlp_ratio: 1.0
  dropout: 0.1
  alpha: 1.0
  beta: 0.01
  lr_projector: 1e-3
  lr_student: 5e-5
  
results:
  final_cos_sim: 0.XXX
  final_loss: 0.XXX
  training_time: "X hours"
  
notes: |
  在此记录观察到的现象、遇到的问题、解决方案等。
```

---

## 🚀 下一步行动

1. ✅ **运行训练**: `python train_full_dataset.py`
2. ⏭️ **监控 CosSim**: 每 50 步查看进度
3. ⏭️ **等待完成**: 预计 3-4 小时（或提前停止）
4. ⏭️ **评估效果**: 加载 `final_student` 测试 GSM8K
5. ⏭️ **对比基线**: MSE vs Mercator 性能差异

---

## 📚 参考资料

- **Elastic Bottleneck**: `experiments/kv_dimension_projector.py`
- **Map Projection**: `src/losses.py`
- **验证测试**: `tests/verify_map_projection.py`
- **完整指南**: `docs/MAP_PROJECTION_GUIDE.md`

---

**准备好了吗？开始你的 KAVA 之旅吧！** 🎉

```bash
python train_full_dataset.py
```

祝训练顺利，CosSim 早日突破 0.95！ 🚀
