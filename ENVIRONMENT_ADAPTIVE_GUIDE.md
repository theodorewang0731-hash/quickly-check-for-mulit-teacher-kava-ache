# KAVA 环境自适应系统

## 🎯 核心特性

### ✅ 完全环境无关
- **自动检测硬件**：GPU (CUDA/ROCm/MPS) / CPU，自动选择最佳设备
- **自动检测精度**：BF16 > FP16 > FP32，根据硬件能力自动选择
- **自动适配路径**：支持环境变量、相对路径、HPC 特定路径
- **自动优化配置**：Batch Size、梯度累积、DataLoader 并行度

### ✅ 跨层聚合 (Cross-Layer Aggregation)
- **动态维度检测**：运行时自动检测实际 KV Cache 维度
- **多种聚合策略**：
  - `concat`: 拼接所有层（默认，28层×256维=7168维）
  - `mean`: 平均所有层
  - `weighted`: 加权聚合
- **量化模型兼容**：支持 4-bit/8-bit 量化模型

### ✅ 动态 Projector 配置
- **延迟初始化**：根据检测到的维度动态初始化
- **形状验证**：自动验证输入输出形状匹配
- **测试前向传播**：初始化后自动测试确保正确

---

## 📦 文件结构

```
kava/
├── configs/
│   └── environment_config.yaml      # 环境配置文件
├── src/
│   ├── environment_adapter.py       # 环境自适应模块
│   ├── dynamic_kv_extractor.py      # 动态 KV 提取器
│   └── losses.py                    # 损失函数
├── scripts/
│   ├── submit_slurm.sh             # SLURM 作业脚本
│   └── submit_pbs.sh               # PBS 作业脚本
├── train_adaptive.py               # 环境自适应训练脚本
└── check_environment.py            # 环境检查脚本
```

---

## 🚀 快速开始

### 1. 检查环境

```bash
python check_environment.py
```

输出示例：
```
🌍 Environment Detection Report
========================================
📍 Environment Type: LOCAL
   Platform: Windows
   GPU: RTX 4070 Laptop (8GB)
   Precision: BF16

✅ All dependencies installed
✅ Paths configured
✅ KV Extractor ready
```

### 2. 本地训练

```bash
python train_adaptive.py
```

### 3. HPC 集群训练

#### SLURM 系统
```bash
# 编辑 scripts/submit_slurm.sh 中的路径
# 然后提交作业
sbatch scripts/submit_slurm.sh
```

#### PBS 系统
```bash
# 编辑 scripts/submit_pbs.sh 中的路径
# 然后提交作业
qsub scripts/submit_pbs.sh
```

---

## 🔧 环境配置

### 方式 1: 环境变量（推荐用于 HPC）

```bash
export KAVA_PROJECT_ROOT="/path/to/kava"
export KAVA_MODEL_PATH="/scratch/$USER/kava/models"
export KAVA_DATA_PATH="/scratch/$USER/kava/data"
export KAVA_CACHE_PATH="/scratch/$USER/kava/cache"
export KAVA_OUTPUT_PATH="/scratch/$USER/kava/outputs"

python train_adaptive.py
```

### 方式 2: 配置文件

编辑 `configs/environment_config.yaml`：

```yaml
paths:
  env_vars:
    models: KAVA_MODEL_PATH
    data: KAVA_DATA_PATH
    
  defaults:
    models: ./local_models
    data: ./local_data
    
  hpc_patterns:
    - /scratch/{username}/kava
    - /work/{username}/kava
```

### 方式 3: 自动检测（默认）

脚本会自动检测：
1. 当前环境类型（local / hpc / cloud）
2. 使用相对路径（本地开发）
3. 使用 HPC 标准路径（HPC 环境）

---

## 📊 工作原理

### 1. 环境检测流程

```
启动脚本
    ↓
检测环境类型
    ├── SLURM_JOB_ID 存在? → HPC (SLURM)
    ├── PBS_JOBID 存在? → HPC (PBS)
    ├── KUBERNETES → Cloud
    └── 默认 → Local
    ↓
检测硬件
    ├── torch.cuda.is_available()? → CUDA
    ├── torch.backends.mps? → MPS (Apple Silicon)
    └── 默认 → CPU
    ↓
选择精度
    ├── 支持 BF16? → BF16
    ├── 支持 FP16? → FP16
    └── 默认 → FP32
    ↓
配置路径
    ├── 环境变量存在? → 使用环境变量
    ├── HPC 环境? → 使用 HPC 标准路径
    └── 默认 → 使用相对路径
```

### 2. KV 维度检测流程

```
加载模型
    ↓
创建测试输入 (1, 32)
    ↓
前向传播获取 KV Cache
    ↓
分析 KV 结构
    ├── Layers: 28
    ├── Heads per layer: 2 (量化后)
    ├── Head dim: 128
    └── Layer dim: 256
    ↓
跨层聚合
    ├── concat: 28 × 256 = 7168 维
    ├── mean: 单层 256 维
    └── weighted: 加权后 256 维
    ↓
动态初始化 Projector
    └── LayerNorm(7168) → 确保维度匹配
```

### 3. Projector 初始化流程

```
检测 Teacher KV dim: 7168
检测 Student KV dim: 3072
    ↓
创建 Projector
    ├── LayerNorm(7168)  ← 使用检测到的维度
    ├── Linear(7168 → 7168)
    ├── SiLU()
    ├── Dropout(0.1)
    └── Linear(7168 → 3072)
    ↓
测试前向传播
    └── 验证输出形状匹配
```

---

## 🎓 关键技术说明

### 跨层聚合 (Cross-Layer Aggregation)

**问题**：量化模型的单层 KV 维度（256）与配置不匹配（1536）

**解决方案**：将所有层聚合为一个高维向量

```python
# 传统方法（单层）
k, v = past_key_values[-1]  # 只取最后一层
# 结果：[B, T, 256] ← 维度太小

# 跨层聚合（全部层）
all_kvs = []
for layer_kv in past_key_values:
    k, v = layer_kv
    all_kvs.append(flatten(k))  # 展平每一层
kv_combined = torch.cat(all_kvs, dim=-1)
# 结果：[B, T, 7168] ← 28层 × 256维 = 7168维
```

**优势**：
- ✅ 捕获所有层的信息
- ✅ 自动适配量化模型
- ✅ 维度匹配稳定

### 动态维度检测

**问题**：模型配置维度 ≠ 实际输出维度

```python
# 配置说：hidden_size = 1536
teacher.config.hidden_size  # → 1536

# 实际输出：28层 × 2头 × 128维/头 = 7168
actual_output.shape[-1]  # → 7168  ← 不匹配！
```

**解决方案**：运行时检测实际维度

```python
# Step 1: 创建测试输入
test_input = torch.randint(0, 1000, (1, 32)).to(device)

# Step 2: 前向传播获取实际输出
with torch.no_grad():
    output = model(test_input, use_cache=True)
    kv = extract_kv(output.past_key_values)

# Step 3: 测量实际维度
actual_dim = kv.shape[-1]  # 7168

# Step 4: 使用实际维度初始化 Projector
projector = KVDimensionProjector(
    teacher_configs={"teacher": {"d_model": actual_dim}}  # 7168
)
```

---

## 🖥️ HPC 环境适配

### 支持的调度器
- ✅ SLURM (最常见)
- ✅ PBS/Torque
- ✅ SGE
- ✅ LSF
- ✅ Cobalt

### 自动检测的环境变量

| 调度器 | 检测变量 | 说明 |
|--------|----------|------|
| SLURM | `SLURM_JOB_ID` | 作业 ID |
| PBS | `PBS_JOBID` | 作业 ID |
| SGE | `SGE_TASK_ID` | 任务 ID |
| LSF | `LSB_JOBID` | 作业 ID |

### HPC 路径模式

自动检测常见 HPC 路径：
```python
/scratch/{username}/kava
/home/{username}/projects/kava
/data/{username}/kava
/work/{username}/kava
```

---

## 📝 配置选项

### environment_config.yaml 主要选项

```yaml
# 硬件检测
hardware:
  gpu_detection:
    auto_detect: true           # 自动检测 GPU
    fallback_to_cpu: false      # GPU 不可用时是否回退到 CPU
  
  precision:
    auto_detect: true           # 自动选择精度
    fallback: fp32              # 回退精度

# 模型维度
model_dimensions:
  auto_detect: true             # 动态检测维度
  
  kv_extraction:
    method: cross_layer_aggregation  # 跨层聚合
    detect_at_runtime: true     # 运行时检测
  
  projector:
    initialization: dynamic     # 动态初始化
    validate_shapes: true       # 验证形状

# 训练配置
training:
  auto_tune: true               # 自动调优
  
  batch_size:
    auto_detect: true           # 根据显存自动选择
    strategy: max_fit           # 最大化利用显存
  
  gradient_accumulation:
    auto_compute: true          # 自动计算累积步数
    target_batch_size: 32       # 目标有效 batch size
```

---

## 🔍 调试与验证

### 查看环境检测结果

```python
from src.environment_adapter import create_environment_adapter

adapter = create_environment_adapter()
# 自动打印完整环境报告

# 获取推荐配置
config = adapter.get_training_config()
print(config)
```

### 测试 KV 提取器

```python
from src.dynamic_kv_extractor import create_kv_extractor

extractor = create_kv_extractor(
    aggregation_method="concat",
    use_all_layers=True
)

# 使用真实模型测试
kv_tensor = extractor.extract_kv(
    model_output.past_key_values,
    model_name="test",
    debug=True  # 打印详细信息
)
```

### 验证 Projector 维度

```python
# 检查 Projector 第一层（LayerNorm）的维度
print(projector.adapter_K[0].normalized_shape)
# 应该输出: [7168] 而不是 [1536]
```

---

## 📈 性能优化

### 自动优化的配置

1. **Batch Size**
   ```python
   # 根据显存自动选择
   8GB GPU  → batch_size=2
   16GB GPU → batch_size=4
   40GB GPU → batch_size=8
   ```

2. **梯度累积**
   ```python
   # 自动计算以达到目标 batch size (32)
   batch_size=2 → grad_accum=16
   batch_size=4 → grad_accum=8
   ```

3. **DataLoader 并行**
   ```python
   # 根据 CPU 核心数自动选择
   num_workers = min(os.cpu_count(), 8)
   ```

4. **混合精度训练**
   ```python
   # 自动选择最佳精度
   A100/4090 → BF16  # 最快
   V100/3090 → FP16  # 次快
   CPU       → FP32  # 兼容
   ```

---

## 🆘 常见问题

### Q1: 如何在没有 GPU 的 HPC 节点上测试？

```bash
# 强制使用 CPU（调试用）
export CUDA_VISIBLE_DEVICES=""
python check_environment.py
```

### Q2: 如何修改 Batch Size？

编辑 `train_adaptive.py`：
```python
GLOBAL_CONFIG = {
    'batch_size': 4,  # 固定 batch size
    # 或者
    'auto_tune': True,  # 自动选择
}
```

### Q3: 如何使用不同的聚合方法？

```python
GLOBAL_CONFIG = {
    'kv_aggregation_method': 'mean',  # concat / mean / weighted
    'use_all_layers': True,
}
```

### Q4: 如何在不同 HPC 集群之间迁移？

**完全无需修改代码！** 只需设置环境变量：

```bash
# 集群 A
export KAVA_DATA_PATH=/scratch/user/data

# 集群 B
export KAVA_DATA_PATH=/work/user/data

# 脚本自动适配
python train_adaptive.py
```

---

## 🎯 总结

### 核心优势

1. **环境无关** ✅
   - 本地开发、HPC 集群、云平台无缝切换
   - 无需修改代码，自动适配

2. **维度自适应** ✅
   - 运行时动态检测实际维度
   - 避免配置维度与实际输出不匹配

3. **跨层聚合** ✅
   - 捕获所有层信息
   - 兼容量化模型
   - 维度稳定可靠

4. **生产就绪** ✅
   - 完善的错误处理
   - 自动保存检查点
   - 详细的日志输出

### 使用建议

- **本地开发**：直接运行 `python train_adaptive.py`
- **HPC 训练**：使用 `sbatch scripts/submit_slurm.sh`
- **环境迁移**：只需设置环境变量，代码无需改动
- **调试问题**：运行 `python check_environment.py` 检查配置

---

**现在您的代码已经完全环境无关，可以在任何 HPC 集群上无缝运行！** 🚀
