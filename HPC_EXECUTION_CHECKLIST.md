# HPC 部署执行清单 ✅

**当前状态汇总**（根据你的诊断报告）

## ✅ 已完成

### 登录节点验证（通过 `test_setup.sh`）
```
✓ Python 3.10.12
✓ torch 2.9.1+cu128 (CPU mode)
✓ transformers 4.57.1
✓ datasets 4.4.1
✓ accelerate 1.11.0
✓ 项目模块导入成功 (kv_utils, kv_loss, projector)
✓ Setup test passed!
```

### GPU 检测（登录节点 - 符合预期）
```
✗ nvidia-smi: 无法与驱动通信（登录节点无 GPU - 正常）
✗ nvcc: 未安装（登录节点无 CUDA - 正常）
✗ module avail cuda: 无输出（登录节点无 CUDA 模块 - 正常）
```

**结论**: 登录节点环境正常，PyTorch 已安装但为 CPU 模式（符合预期）。

---

## ⚠️ 待执行

### 关键任务：检测计算节点的 GPU 环境

计算节点是否有 GPU/CUDA **目前未知**，需要提交 SLURM 作业验证。

---

## 🚀 立即执行步骤

### 步骤 1：确认脚本存在

```bash
cd /path/to/kava/quickly_check

# 检查关键脚本
ls -lh scripts/verify_login_node.sh
ls -lh scripts/check_gpu_node.sh
ls -lh scripts/setup_hpc_environment.sh
```

**预期输出**: 三个脚本都存在

---

### 步骤 2：运行增强版登录节点验证

```bash
# 给脚本执行权限
chmod +x scripts/verify_login_node.sh

# 运行验证（比 test_setup.sh 更详细）
bash scripts/verify_login_node.sh
```

**预期输出**:
```
✓ 找到虚拟环境: kava_env
✓ 登录节点无 GPU（符合预期）
✓ 所有依赖已安装
✓ 所有核心模块导入成功
✓ 模拟训练成功（如果有 train_minimal.py）

下一步操作：
  1. 【推荐】先检测计算节点的 GPU 环境
     sbatch scripts/check_gpu_node.sh
```

---

### 步骤 3：提交 GPU 检测作业到计算节点 ⭐

```bash
# 创建日志目录
mkdir -p logs

# 给脚本执行权限
chmod +x scripts/check_gpu_node.sh

# 【关键】提交检测作业
sbatch scripts/check_gpu_node.sh
```

**预期输出**:
```
Submitted batch job 12345678
```

**记下作业 ID**（如 12345678）

---

### 步骤 4：监控作业状态

```bash
# 查看作业队列
squeue -u $USER

# 或查看特定作业
squeue -j 12345678
```

**作业状态说明**:
- `PENDING (PD)`: 排队中，等待资源
- `RUNNING (R)`: 正在运行（通常 1-2 分钟完成）
- `COMPLETED (CD)`: 已完成
- `FAILED (F)`: 失败（查看错误日志）

---

### 步骤 5：查看 GPU 检测报告

```bash
# 等待作业完成后（通常 1-2 分钟）
# 查看输出日志
cat logs/gpu_check_*.out

# 如果有错误，查看错误日志
cat logs/gpu_check_*.err
```

---

## 📊 GPU 检测报告解读

### 情况 A：🎉 完美情况

**报告内容**:
```
✓ nvidia-smi 可用
✓ GPU 列表: NVIDIA A100 80GB
✓ PyTorch 已安装
  版本: 2.9.1+cu128
  CUDA 编译支持: 12.8
  CUDA 运行时可用: True
  GPU 数量: 8

🎉 恭喜！计算节点环境完全正常！

下一步：
  1. 可以直接提交训练作业：
     sbatch scripts/run_multi_seed_experiments.sh
```

**你的操作**: ✅ **立即提交训练**
```bash
sbatch scripts/run_multi_seed_experiments.sh
```

---

### 情况 B：⚠️ 需要重装 PyTorch

**报告内容**:
```
✓ nvidia-smi 可用
✓ GPU 列表: NVIDIA A100 80GB
✓ PyTorch 已安装
  版本: 2.9.1+cu128
  CUDA 编译支持: None  ← 问题！
  CUDA 运行时可用: False  ← 问题！

⚠ GPU 可用，但 PyTorch 检测不到 CUDA

解决方案：
  当前驱动版本: 535.xxx
  推荐安装 PyTorch with CUDA 11.8+:
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**你的操作**: 🔧 **重新安装 PyTorch**

方法 1：在登录节点安装（推荐）
```bash
source kava_env/bin/activate
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 再次检测确认
sbatch scripts/check_gpu_node.sh
```

方法 2：创建计算节点专用安装脚本
```bash
# 我可以帮你创建一个 SLURM 作业来在计算节点安装
```

---

### 情况 C：📋 需要加载 CUDA 模块

**报告内容**:
```
✓ nvidia-smi 可用
✗ nvcc 不可用
✓ module 命令可用

可用的 CUDA 相关模块：
  cuda/11.8
  cuda/12.1

⚠ PyTorch 检测不到 CUDA
  可能原因: 需要加载 CUDA module
```

**你的操作**: 🔧 **修改环境配置脚本**

编辑 `scripts/setup_hpc_environment.sh`，找到第 12-20 行：
```bash
CUDA_MODULES=(
    "cuda/11.8"     # 改为你的 HPC 提供的版本
    "cuda/12.1"     # 改为你的 HPC 提供的版本
    # ...
)
```

然后重新检测：
```bash
sbatch scripts/check_gpu_node.sh
```

---

### 情况 D：❌ 没有 GPU

**报告内容**:
```
✗ nvidia-smi 不可用
✗ 此节点可能没有 NVIDIA GPU 或驱动未安装

✗ 计算节点没有 GPU

可能原因：
  1. SLURM 配置错误（检查 --partition 和 --gres 参数）
  2. 此 HPC 没有 GPU 资源
  3. GPU 驱动未安装
```

**你的操作**: 📞 **联系 HPC 管理员**

询问以下问题：
1. GPU 分区名称是什么？（修改 `#SBATCH --partition=???`）
2. 如何申请 GPU 资源？（确认 `#SBATCH --gres=gpu:???`）
3. 是否需要特殊权限访问 GPU 节点？
4. 是否有 GPU 使用文档或示例脚本？

---

## 📋 检查清单

完成后打勾：

- [ ] 步骤 1：确认脚本存在 ✓
- [ ] 步骤 2：运行 `verify_login_node.sh` ✓
- [ ] 步骤 3：提交 `check_gpu_node.sh` 作业
- [ ] 步骤 4：等待作业完成（1-2 分钟）
- [ ] 步骤 5：查看并理解 GPU 检测报告
- [ ] **根据报告采取相应行动**（A/B/C/D）

---

## 🆘 如果遇到问题

### Q: `sbatch: command not found`
**A**: 你的 HPC 可能不使用 SLURM，或者需要先加载模块：
```bash
module load slurm
# 或联系管理员询问作业调度系统类型
```

### Q: `Permission denied: check_gpu_node.sh`
**A**: 给脚本执行权限：
```bash
chmod +x scripts/check_gpu_node.sh
```

### Q: 作业一直 PENDING
**A**: 查看原因并调整资源需求：
```bash
squeue -u $USER --start  # 查看预计开始时间和原因
sinfo                    # 查看可用分区
```

### Q: 作业立即 FAILED
**A**: 查看错误日志：
```bash
cat logs/gpu_check_*.err
# 常见原因：分区名称错误、虚拟环境路径错误
```

---

## 📞 需要我帮助的情况

**请把以下信息发给我**：

1. **`verify_login_node.sh` 的完整输出**（如果运行了）
2. **`logs/gpu_check_*.out` 的完整内容**（作业完成后）
3. **`logs/gpu_check_*.err` 的内容**（如果有错误）
4. **`squeue -u $USER` 的输出**（如果作业卡住）

---

## ✅ 成功后的下一步

一旦 GPU 检测报告显示 "🎉 恭喜！计算节点环境完全正常！"：

```bash
# 1. 提交训练作业
sbatch scripts/run_multi_seed_experiments.sh

# 2. 监控训练
squeue -u $USER
tail -f logs/multi_seed_*.out

# 3. 预计时间
# - 基线训练: 2-3 天（3 种 × 3 seeds）
# - 主实验: 3-5 天（3 种 × 3 seeds）
# - 消融实验: 2-3 天（4 种 × 3 seeds）
```

---

**当前时间**: 2025年11月14日  
**状态**: 等待 GPU 检测报告  
**关键文件**: `logs/gpu_check_*.out` ⬅️ 这是最重要的！
