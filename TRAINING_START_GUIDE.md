# 🚀 KAVA 训练快速启动指南

## ✅ 环境已就绪

您的环境配置：
- ✅ PyTorch: 2.5.1+cu121 (CUDA 12.1)
- ✅ CUDA Available: True
- ✅ GPU: NVIDIA GeForce RTX 4070 Laptop GPU (8GB)
- ✅ 模型: local_models/ (Teacher 1.5B + Student 0.5B)
- ✅ 数据: local_data/gsm8k/ (7473 训练样本)

---

## 🎯 启动训练（一键启动）

```bash
python train_simplified.py
```

---

## 📊 预期输出流程

### Phase 1: 初始化（1-2 分钟）

```
🚀🚀🚀 Starting KAVA Training with Local Resources 🚀🚀🚀

📋 Environment Check:
   Python: 3.11.x
   PyTorch: 2.5.1+cu121
   CUDA Available: True
   CUDA Device: NVIDIA GeForce RTX 4070 Laptop GPU
   CUDA Memory: 8.0 GB

🎯 KAVA Local Training - Simplified & Stable

⚙️ Configuration:
   Teacher: local_models/qwen-1.5b-teacher
   Student: local_models/qwen-0.5b-student
   Dataset: local_data/gsm8k
   Device: cuda
   Effective Batch Size: 32

📚 Step 1: Loading Dataset
   ✅ Dataset loaded: 7473 samples

🔤 Step 2: Loading Tokenizer
   ✅ Tokenizer loaded

🔧 Step 3: Processing Dataset
   ✅ 3737 batches prepared
```

### Phase 2: 模型加载（2-3 分钟）

```
🤖 Step 4: Loading Models
   Loading Teacher (4-bit quantized)...
      ✅ Teacher: d_model=1536
   Loading Student (bfloat16)...
      ✅ Student: d_model=896
```

**显存占用预期**:
- Teacher (4-bit): ~1.2 GB
- Student (bf16): ~1.0 GB
- Projector: ~0.3 GB
- 激活值: ~3.5 GB
- **总计**: ~6 GB / 8 GB ✅ 安全

### Phase 3: 训练开始

```
🗺️ Step 5: Initializing KAVA Components
   Projector: 1536 -> 896
   Loss: Mercator (alpha=1.0, beta=0.01)

======================================================================
🎯 Training Start - Monitor 'CosSim' (Target: >0.90)
======================================================================

Training:   0%|          | 0/3737 [00:00<?, ?it/s]
```

### Phase 4: 训练进度（核心监控）

```
Training:   1%|▏         | 16/3737 [00:45<2:15:30, 0.45it/s]
Loss: 0.8234 | CosSim: 0.1766 | Status: 🔄 Adapting

[Step 0050] Loss: 0.4521 | CosSim: 0.5479 ⚠️ Learning
Training:   3%|▎         | 100/3737 [03:40<2:01:23, 0.50it/s]

[Step 0100] Loss: 0.2145 | CosSim: 0.7855 📈 Good
Training:   5%|▌         | 200/3737 [07:20<1:58:45, 0.50it/s]

[Step 0200] Loss: 0.0432 | CosSim: 0.9568 ✅ Excellent
💾 Checkpoint saved: checkpoints/proj_step_200.pth
```

---

## 🎯 关键指标解读

### Cosine Similarity (CosSim) - 最重要！

| CosSim 值 | 状态 | 含义 | 对应 Loss |
|----------|------|------|----------|
| 0.10-0.30 | 🔄 Adapting | 初始随机状态 | 0.7-0.9 |
| 0.30-0.50 | 🔄 Adapting | 开始学习 | 0.5-0.7 |
| 0.50-0.70 | ⚠️ Learning | 快速进步中 | 0.3-0.5 |
| 0.70-0.90 | 📈 Good | 显著对齐 | 0.1-0.3 |
| 0.90-0.95 | 🎯 Great | 接近目标 | 0.05-0.1 |
| **>0.95** | **✅ Excellent** | **完美对齐！** | **<0.05** |

### 训练速度预期

- **迭代速度**: ~0.4-0.5 it/s (每次迭代 2-2.5 秒)
- **梯度累积**: 每 16 个 batch 更新一次
- **实际更新速度**: 每 32-40 秒一次权重更新
- **每 50 步**: ~25-30 分钟
- **每 200 步**: ~1.5-2 小时
- **完整 Epoch**: ~3-4 小时

---

## ⚠️ 可能遇到的情况

### 情况 1: 显存不足 (OOM)

**症状**:
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**解决方案**: 打开 `train_simplified.py`，修改配置：
```python
CONFIG = {
    "batch_size": 1,                      # 从 2 改为 1
    "gradient_accumulation_steps": 32,    # 从 16 改为 32
}
```

### 情况 2: CosSim 不上升

**症状**: 200 步后 CosSim 仍 <0.50

**可能原因**:
1. 学习率过低
2. 数据处理问题

**调试**: 检查每 50 步的输出，观察趋势。

### 情况 3: Loss 震荡

**症状**: Loss 上下波动剧烈

**解决**: 降低 `lr_projector` 从 1e-3 到 5e-4

---

## 🛑 停止训练

### 优雅停止
- **按 Ctrl+C 一次**: 保存紧急检查点后退出
- 检查点保存在: `checkpoints/emergency_projector.pth`

### 继续训练
训练脚本会自动保存检查点，但不支持自动恢复。如需继续训练，需要修改脚本加载检查点。

---

## ✅ 训练完成标志

```
======================================================================
✅ Training Complete!
======================================================================

💾 Final models saved:
   - final_projector.pth
   - final_student/

🎉 All Done!
```

**最终文件**:
- `final_projector.pth`: 训练好的 Elastic Bottleneck
- `final_student/`: 蒸馏后的 Student 模型
- `checkpoints/proj_step_*.pth`: 中间检查点

---

## 🔍 实时监控技巧

### 方法 1: 观察进度条
```
Training:   5%|▌  | 200/3737 [07:20<1:58:45, 0.50it/s]
Loss: 0.0432 | CosSim: 0.9568 | Status: ✅ Excellent
```

### 方法 2: 每 50 步详细输出
```
[Step 0050] Loss: 0.4521 | CosSim: 0.5479 ⚠️ Learning
[Step 0100] Loss: 0.2145 | CosSim: 0.7855 📈 Good
[Step 0150] Loss: 0.1234 | CosSim: 0.8766 📈 Good
[Step 0200] Loss: 0.0432 | CosSim: 0.9568 ✅ Excellent
```

### 方法 3: GPU 监控（另开终端）
```powershell
nvidia-smi -l 5
```
持续显示 GPU 使用率和显存占用。

---

## 🎉 成功标准

训练成功的标志：
1. ✅ **CosSim 达到 0.95+**（Excellent 状态）
2. ✅ **Loss 降至 0.05 以下**
3. ✅ **训练稳定**（无 NaN、无 OOM）
4. ✅ **检查点保存成功**

---

## 📝 训练日志记录

建议记录以下信息：
- 开始时间
- CosSim 在 50/100/200 步的值
- 最终 CosSim 和 Loss
- 训练总时长
- 是否遇到问题

---

## 🚀 现在开始！

确认以下准备就绪：
- [x] PyTorch CUDA 已安装（2.5.1+cu121）
- [x] GPU 可用（RTX 4070）
- [x] 模型和数据集已下载
- [x] 虚拟环境已激活

**启动命令**:
```bash
python train_simplified.py
```

祝训练顺利！🎉 期待 CosSim 突破 0.95！
