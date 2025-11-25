# HPC 快速操作指南 🚀

## 📍 你现在在哪里？

### 登录节点（Login Node）
- ❌ 没有 GPU/CUDA
- ✅ 可以编辑代码、安装依赖
- ✅ 提交 SLURM 作业

### 计算节点（Compute Node）
- ✅ 有 GPU/CUDA（通过 SLURM 访问）
- ✅ 真正训练的地方
- ❌ 不能直接登录

---

## 🎯 三步走战略

### 步骤 1：登录节点验证（5 分钟）

```bash
cd /path/to/kava/quickly_check
source kava_env/bin/activate

# 安装 CPU 版 PyTorch（登录节点用）
pip uninstall torch -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 运行验证
bash scripts/verify_login_node.sh
```

**预期结果**：✅ 所有依赖已安装，模拟训练成功

---

### 步骤 2：检测计算节点 GPU（2 分钟）

```bash
# 提交 GPU 检测作业
chmod +x scripts/check_gpu_node.sh
sbatch scripts/check_gpu_node.sh

# 查看作业（等待 Running 或 Completed）
squeue -u $USER

# 查看报告
cat logs/gpu_check_*.out
```

**根据报告采取行动**：

| 报告内容 | 操作 |
|---------|------|
| 🎉 GPU 和 PyTorch CUDA 都正常 | 直接进入步骤 3 |
| ⚠️ 有 GPU，但 PyTorch 检测不到 CUDA | 按报告中的命令重装 PyTorch |
| ✗ 没有 GPU | 联系管理员，确认分区名称和 GPU 申请格式 |

---

### 步骤 3：提交真实训练（确认 GPU 可用后）

```bash
# 编辑配置（可选）
vim scripts/run_multi_seed_experiments.sh
# 修改: EXPERIMENT_NAME, STUDENT_MODEL, TEACHER_MODELS

# 提交训练
sbatch scripts/run_multi_seed_experiments.sh

# 监控
squeue -u $USER
tail -f logs/multi_seed_*.out
```

---

## 🔍 常用命令速查

### 查看作业
```bash
squeue -u $USER              # 查看我的所有作业
scontrol show job <job_id>   # 查看作业详情
```

### 取消作业
```bash
scancel <job_id>             # 取消单个作业
scancel -u $USER             # 取消我的所有作业
```

### 查看日志
```bash
ls logs/                     # 列出所有日志
cat logs/gpu_check_*.out     # GPU 检测报告
tail -f logs/multi_seed_*.out  # 实时查看训练日志
```

### 查看分区和资源
```bash
sinfo                        # 查看所有分区
sinfo -p gpu                 # 查看 GPU 分区（改为你的分区名）
```

---

## ⚠️ 重要提示

### ✅ 在登录节点做什么
- 编辑代码
- 安装依赖（CPU 版本）
- 提交 SLURM 作业
- 查看日志和结果

### ❌ 不要在登录节点做什么
- 运行大规模计算
- 尝试使用 GPU（没有）
- 长时间占用 CPU（会被 kill）

### ✅ 在计算节点做什么（通过 SLURM）
- 真实训练
- GPU 加速计算
- 大规模数据处理

---

## 🆘 故障排查

### 问题：作业一直在 PENDING 状态
**原因**：资源不足或分区错误
```bash
# 查看原因
squeue -u $USER --start

# 检查分区
sinfo

# 减少资源需求或更换分区
vim scripts/run_multi_seed_experiments.sh
```

---

### 问题：作业立即失败（状态变为 FAILED）
**原因**：脚本错误或环境问题
```bash
# 查看错误日志
cat logs/multi_seed_*.err

# 常见原因：
# 1. 虚拟环境路径错误
# 2. 依赖未安装
# 3. 文件路径错误
```

---

### 问题：PyTorch 检测不到 GPU
**解决**：
1. 查看 GPU 检测报告：`cat logs/gpu_check_*.out`
2. 按报告建议重装 PyTorch（匹配 CUDA 版本）
3. 或修改 `scripts/setup_hpc_environment.sh` 加载正确的 CUDA module

---

### 问题：下载模型失败（Hugging Face）
**解决**：
```bash
# 方法 1：交互登录
huggingface-cli login

# 方法 2：环境变量（在 SLURM 脚本中）
export HF_TOKEN="your_token_here"
```

---

## 📞 需要管理员帮助时询问

1. **GPU 分区名称**：`#SBATCH --partition=???` 应该填什么？
2. **GPU 申请格式**：`#SBATCH --gres=gpu:???` 应该填什么？
3. **CUDA 模块名称**：如何加载 CUDA？(`module load ???`)
4. **Python 模块**：是否有预装的 Python/Conda 模块？
5. **存储配额**：我的存储空间和配额是多少？

---

## 📚 关键文档

- **完整指南**：`HPC_DEPLOYMENT_GUIDE.md`
- **可视化指南**：`VISUALIZATION_QUICKSTART.md`
- **硬性控制文档**：`RIGOROUS_CONTROLS.md`
- **实验设计**：`EXPERIMENT_DESIGN.md`

---

**最后更新**: 2025年11月14日  
**状态**: 适配无 CUDA 登录节点的 HPC 环境
