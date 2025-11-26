# 🎯 KAVA 本地化训练完整指南

## 目标：实现完全离线训练环境

将所有模型和数据集下载到项目本地，确保训练环境：
- ✅ **可移植**：无需依赖外部网络
- ✅ **独立**：所有资源在本地磁盘
- ✅ **稳定**：不受网络波动影响

---

## 📋 三步部署流程

### Step 1: 安装必要依赖

```bash
pip install huggingface-hub bitsandbytes scipy accelerate datasets transformers
```

**依赖说明**:
- `huggingface-hub`: 下载 HuggingFace 模型和数据集
- `bitsandbytes`: 4-bit 量化支持
- `scipy`, `accelerate`: 训练加速
- `datasets`, `transformers`: 数据处理和模型加载

---

### Step 2: 下载本地资源

运行下载脚本，将所有资源下载到项目目录：

```bash
python download_local_resources.py
```

**下载内容**:
```
📦 Models (约 3-4 GB):
  • Qwen/Qwen2.5-1.5B-Instruct → local_models/qwen-1.5b-teacher/
  • Qwen/Qwen2.5-0.5B-Instruct  → local_models/qwen-0.5b-student/

📦 Datasets (约 50-100 MB):
  • gsm8k → local_data/gsm8k/
```

**预计时间**: 10-30 分钟（取决于网络速度）

**下载过程**:
1. 自动创建 `local_models/` 和 `local_data/` 目录
2. 从 HuggingFace 下载模型和数据集
3. 验证文件完整性
4. 显示下载摘要

**成功标志**:
```
🎉 SUCCESS! All resources downloaded successfully!

📂 Project Structure:
   .
   ├── local_models/
   │   ├── qwen-1.5b-teacher/
   │   └── qwen-0.5b-student/
   └── local_data/
       └── gsm8k/

✅ Ready to run: python train_local_only.py
```

---

### Step 3: 启动本地化训练

所有资源下载完成后，运行本地化训练脚本：

```bash
python train_local_only.py
```

**特性**:
- ✅ 启动时自动验证本地资源
- ✅ 强制使用本地文件（`local_files_only=True`）
- ✅ 无需网络连接即可训练
- ✅ 完整的错误提示和排查指南

---

## 🔍 本地资源验证

### 自动验证功能

`train_local_only.py` 启动时会自动检查：
1. ✅ Teacher 模型是否存在
2. ✅ Student 模型是否存在
3. ✅ 数据集是否完整
4. ✅ 关键配置文件是否齐全

**验证输出示例**:
```
🔍 Verifying local resources...
   ✅ Teacher: local_models/qwen-1.5b-teacher
   ✅ Student: local_models/qwen-0.5b-student
   ✅ Dataset: local_data/gsm8k
   ✅ All local resources verified!
```

### 手动验证（可选）

```bash
# 检查模型文件
ls local_models/qwen-1.5b-teacher/
ls local_models/qwen-0.5b-student/

# 检查数据集文件
ls local_data/gsm8k/

# 预期看到的关键文件:
# - config.json
# - tokenizer_config.json
# - model.safetensors 或 pytorch_model.bin
# - dataset_info.json (数据集)
```

---

## 📂 项目目录结构

```
quickly-check-for-mulit-teacher-kava-ache/
├── local_models/                      # 本地模型目录
│   ├── qwen-1.5b-teacher/             # Teacher 模型
│   │   ├── config.json
│   │   ├── tokenizer_config.json
│   │   ├── model.safetensors
│   │   └── ...
│   └── qwen-0.5b-student/             # Student 模型
│       ├── config.json
│       ├── tokenizer_config.json
│       ├── model.safetensors
│       └── ...
├── local_data/                        # 本地数据集目录
│   └── gsm8k/                         # GSM8K 数据集
│       ├── dataset_info.json
│       ├── train/
│       └── test/
├── checkpoints/                       # 训练检查点
│   └── proj_step_*.pth
├── download_local_resources.py        # 资源下载脚本
├── train_local_only.py                # 本地化训练脚本
├── train_full_dataset.py              # 原在线训练脚本（备份）
└── ...
```

---

## ⚙️ 配置说明

### 本地化配置 (`train_local_only.py`)

```python
CONFIG = {
    # 本地路径（无需修改，除非自定义目录）
    "teacher_path": "local_models/qwen-1.5b-teacher",
    "student_path": "local_models/qwen-0.5b-student",
    "dataset_path": "local_data/gsm8k",
    
    # 训练配置（与在线版本相同）
    "batch_size": 2,
    "gradient_accumulation_steps": 16,
    "max_length": 512,
    "lr_projector": 1e-3,
    "lr_student": 5e-5,
    
    # 验证开关
    "verify_local_files": True  # 建议保持开启
}
```

### 显存优化配置

如遇 OOM，修改配置：
```python
"batch_size": 1,                      # 降至 1
"gradient_accumulation_steps": 32,    # 增至 32
"max_length": 384,                    # 可选：缩短序列
```

---

## 🔧 常见问题排查

### Q1: 下载脚本失败

**症状**:
```
❌ Error downloading Qwen/Qwen2.5-1.5B-Instruct: ...
ConnectionError: Couldn't reach https://huggingface.co
```

**解决方案**:

1. **使用 HF 镜像**（国内推荐）:
```bash
export HF_ENDPOINT=https://hf-mirror.com
python download_local_resources.py
```

2. **手动下载**:
```bash
# 使用 Git LFS 手动下载
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct local_models/qwen-1.5b-teacher
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct local_models/qwen-0.5b-student
```

3. **分步下载**（逐个模型）:
```python
# 修改 download_local_resources.py
# 注释掉已下载的模型，只下载失败的部分
```

### Q2: 训练启动失败（找不到本地文件）

**症状**:
```
❌ Teacher not found: local_models/qwen-1.5b-teacher
```

**解决方案**:

1. **确认下载完成**:
```bash
python download_local_resources.py
# 等待看到 "🎉 SUCCESS!"
```

2. **检查目录结构**:
```bash
ls -R local_models/
ls -R local_data/
```

3. **手动验证关键文件**:
```bash
# 必须存在的文件:
cat local_models/qwen-1.5b-teacher/config.json
cat local_models/qwen-0.5b-student/config.json
cat local_data/gsm8k/dataset_info.json
```

### Q3: 磁盘空间不足

**症状**:
```
OSError: [Errno 28] No space left on device
```

**解决方案**:

1. **检查可用空间**（至少需要 5 GB）:
```bash
df -h .
```

2. **清理缓存**:
```bash
# 清理 HF 缓存（如果之前下载过）
rm -rf ~/.cache/huggingface/hub/
```

3. **修改下载路径**（使用更大的磁盘）:
```python
# 修改 download_local_resources.py 中的路径
DOWNLOAD_CONFIG = {
    "models": {
        "Qwen/Qwen2.5-1.5B-Instruct": "/mnt/large_disk/local_models/qwen-1.5b-teacher",
        ...
    }
}

# 同步修改 train_local_only.py 中的 CONFIG
```

### Q4: 模型加载报错

**症状**:
```
OSError: local_models/qwen-1.5b-teacher does not appear to be a valid model
```

**解决方案**:

1. **检查文件完整性**:
```bash
# 查看模型文件
ls -lh local_models/qwen-1.5b-teacher/

# 必须包含:
# - config.json (非空)
# - tokenizer_config.json
# - *.safetensors 或 *.bin (模型权重)
```

2. **重新下载损坏的模型**:
```bash
# 删除损坏的目录
rm -rf local_models/qwen-1.5b-teacher/

# 重新运行下载脚本
python download_local_resources.py
```

3. **验证 JSON 文件有效性**:
```bash
python -c "import json; print(json.load(open('local_models/qwen-1.5b-teacher/config.json')))"
```

---

## 📊 训练监控

### 核心指标

与在线版本相同，重点关注 **Cosine Similarity**:

| CosSim 范围 | 状态 | 说明 |
|------------|------|------|
| 0.20-0.50 | 🔄 Adapting | 初始适应 |
| 0.50-0.70 | ⚠️ Learning | 学习中 |
| 0.70-0.90 | 📈 Good | 显著进步 |
| 0.90-0.95 | 🎯 Great | 接近目标 |
| >0.95 | ✅ Excellent | 完美对齐 |

### 预期输出

```
[Step 0000] Loss: 0.8234 | CosSim: 0.1766 🔄 Adapting
[Step 0050] Loss: 0.4521 | CosSim: 0.5479 ⚠️ Learning
[Step 0100] Loss: 0.2145 | CosSim: 0.7855 📈 Good
[Step 0200] Loss: 0.0432 | CosSim: 0.9568 ✅ Excellent
```

---

## 🎯 成功标志

### 1. 下载成功

```
🎉 SUCCESS! All resources downloaded successfully!
✅ Ready to run: python train_local_only.py
```

### 2. 训练启动成功

```
🔍 Verifying local resources...
   ✅ All local resources verified!

🤖 Loading models from local disk...
   ✅ Teacher loaded: d_model=1536
   ✅ Student loaded: d_model=896

🎯 Training Start - Monitor 'CosSim' (Target: >0.90)
```

### 3. 训练完成

```
✅ Training Complete!
💾 Final models saved:
   - final_projector.pth
   - final_student/
🎉 All Done!
```

---

## 🚀 快速启动清单

```bash
# ✅ Step 1: 安装依赖
pip install huggingface-hub bitsandbytes scipy accelerate datasets transformers

# ✅ Step 2: 下载本地资源（等待 10-30 分钟）
python download_local_resources.py

# ✅ Step 3: 启动训练
python train_local_only.py
```

---

## 📝 对比：在线 vs 本地

| 特性 | 在线训练 | 本地训练 |
|-----|---------|---------|
| **网络依赖** | 每次需联网 | 首次下载后离线 |
| **启动速度** | 慢（每次下载） | 快（本地加载） |
| **稳定性** | 受网络影响 | 完全稳定 |
| **磁盘占用** | 缓存不可控 | 明确 5 GB |
| **可移植性** | 差 | 优秀（可打包） |
| **适用场景** | 快速测试 | 生产训练 |

**推荐**: 生产环境使用 **本地训练**（`train_local_only.py`）

---

## 🎓 技术细节

### local_files_only 参数

```python
# 强制仅使用本地文件，防止意外联网
teacher = AutoModelForCausalLM.from_pretrained(
    CONFIG['teacher_path'],
    local_files_only=True  # 关键参数
)
```

### load_from_disk vs load_dataset

```python
# 在线加载（train_full_dataset.py）
dataset = load_dataset("gsm8k", "main")

# 本地加载（train_local_only.py）
dataset = load_from_disk("local_data/gsm8k")
```

### 自动验证机制

```python
if CONFIG["verify_local_files"]:
    if not verify_local_resources():
        sys.exit(1)  # 启动前阻止，避免训练到一半才发现问题
```

---

## 🎉 总结

### 优势

1. ✅ **完全离线**：首次下载后无需网络
2. ✅ **可移植**：整个项目可打包迁移
3. ✅ **稳定**：不受 HuggingFace 服务波动影响
4. ✅ **可控**：明确知道所有文件位置和大小
5. ✅ **快速**：本地加载模型比联网下载快 10x

### 适用场景

- ✅ **生产训练**：需要稳定、可重复的训练环境
- ✅ **离线环境**：无法联网或网络受限的服务器
- ✅ **批量实验**：需要多次训练，避免重复下载
- ✅ **团队协作**：统一的本地资源版本

---

**准备好了吗？开始你的本地化 KAVA 训练之旅！** 🚀

```bash
python download_local_resources.py
```
