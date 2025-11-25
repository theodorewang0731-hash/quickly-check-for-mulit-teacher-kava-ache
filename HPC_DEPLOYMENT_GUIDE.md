# HPC 部署与运行指南

## 📋 快速开始

### 1️⃣ 在登录节点（无 GPU）

```bash
# 进入项目目录
cd /path/to/kava/quickly_check

# 激活虚拟环境
source kava_env/bin/activate

# 安装 CPU 版本 PyTorch（登录节点用，节省空间）
pip uninstall torch -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 运行验证脚本
bash scripts/verify_login_node.sh
```

**预期输出**：
- ✅ 所有依赖已安装
- ✅ 所有核心模块导入成功
- ✅ 模拟训练运行成功
- ✅ 生成 `dummy_model.txt`

---

### 2️⃣ 检测计算节点的 GPU 环境

验证成功后，**先检测计算节点**是否有 GPU 和 CUDA：

```bash
# 给脚本执行权限
chmod +x scripts/check_gpu_node.sh

# 提交 GPU 检测作业（只需 1 分钟）
sbatch scripts/check_gpu_node.sh

# 查看作业状态
squeue -u $USER

# 作业完成后查看报告
cat logs/gpu_check_*.out
```

**查看报告内容**：
- ✅ GPU 信息（nvidia-smi 输出）
- ✅ CUDA 版本和驱动
- ✅ PyTorch CUDA 支持
- ✅ 环境诊断和建议

**根据报告采取行动**：

#### 情况 A：GPU 和 PyTorch CUDA 都正常
```
🎉 恭喜！计算节点环境完全正常！
```
→ **直接进入步骤 3**，提交训练作业

#### 情况 B：有 GPU，但 PyTorch 检测不到 CUDA
```
⚠ GPU 可用，但 PyTorch 检测不到 CUDA
当前驱动版本: 535.xxx
推荐安装 PyTorch with CUDA 11.8+:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
→ **先重新安装 PyTorch**：
```bash
source kava_env/bin/activate
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# 再次运行检测确认
sbatch scripts/check_gpu_node.sh
```

#### 情况 C：报告中提示需要加载 CUDA module
```
可用的 CUDA 相关模块：
  cuda/11.8
  cuda/12.1
```
→ **修改 SLURM 脚本**，在开头添加：
```bash
# 编辑 scripts/setup_hpc_environment.sh
vim scripts/setup_hpc_environment.sh
# 找到 CUDA_MODULES 数组，添加你的 HPC 提供的版本
```

#### 情况 D：计算节点没有 GPU
```
✗ 计算节点没有 GPU
可能原因：SLURM 配置错误（检查 --partition 和 --gres 参数）
```
→ **联系 HPC 管理员**，询问：
- GPU 分区名称（修改 `#SBATCH --partition=???`）
- GPU 资源申请方式（确认 `#SBATCH --gres=gpu:?`）

---

### 3️⃣ 提交训练作业（确认 GPU 可用后）

```bash
# 提交多种子实验
sbatch scripts/run_multi_seed_experiments.sh

# 查看作业状态
squeue -u $USER

# 实时查看日志
tail -f logs/multi_seed_*.out
```

---

## 🔧 环境说明

### 登录节点 vs 计算节点

| 特性 | 登录节点 | 计算节点 |
|------|---------|---------|
| GPU | ❌ 无 | ✅ 有 |
| CUDA | ❌ 无 | ✅ 有 |
| 用途 | 编辑代码、提交任务 | 真实训练 |
| PyTorch | CPU 版本 | 自动使用 GPU |

### 自动环境配置

所有 SLURM 脚本会自动：
1. 检测并加载可用的 CUDA 模块
2. 激活 Python 环境（kava_env 或 conda kava）
3. 验证 GPU 可用性
4. 设置环境变量

通过 `scripts/setup_hpc_environment.sh` 实现。

---

## 📂 关键文件

### 环境配置
- `requirements.txt` - Python 依赖列表
- `scripts/setup_hpc_environment.sh` - HPC 环境自动配置
- `scripts/verify_login_node.sh` - 登录节点验证脚本
- `scripts/check_gpu_node.sh` - 计算节点 GPU 环境检测 ⭐

### 训练脚本
- `scripts/run_multi_seed_experiments.sh` - 多种子实验（≥3 seeds）
- `scripts/run_ablation_studies.sh` - 消融实验
- `experiments/train_minimal.py` - 轻量级验证训练

---

## 🚀 完整工作流

### 阶段 1：环境准备（登录节点）
```bash
# 1. 上传代码到 HPC
scp -r ./kava/quickly_check user@hpc:/path/to/

# 2. SSH 登录
ssh user@hpc

# 3. 创建虚拟环境
cd /path/to/quickly_check
python -m venv kava_env
source kava_env/bin/activate

# 4. 安装依赖（CPU 版本）
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 5. 验证环境
bash scripts/verify_login_node.sh

# 6. 【重要】检测计算节点 GPU
chmod +x scripts/check_gpu_node.sh
sbatch scripts/check_gpu_node.sh
# 等待作业完成（~1 分钟）
cat logs/gpu_check_*.out
```

---

**⚠️ 根据 GPU 检测报告采取相应行动（见下文"常见问题 Q3"）**

---

### 阶段 2：数据准备（登录节点或计算节点）
```bash
# 创建统一数据切分
python data/data_split_controller.py \
    --dataset_names gsm8k svamp strategyqa math arc_challenge \
    --output_dir ./data/unified_splits \
    --teacher_separate \
    --val_size 0.1 \
    --test_size 0.1
```

### 阶段 3：提交训练（登录节点提交，计算节点运行）
```bash
# 编辑 SLURM 脚本配置（如需要）
vim scripts/run_multi_seed_experiments.sh
# 修改: EXPERIMENT_NAME, STUDENT_MODEL, TEACHER_MODELS 等

# 提交基线实验
sbatch scripts/run_all_baselines.sh

# 提交主实验
sbatch scripts/run_three_stage_routing.sh

# 提交消融实验
sbatch scripts/run_ablation_studies.sh
```

### 阶段 4：监控作业
```bash
# 查看作业队列
squeue -u $USER

# 查看作业详情
scontrol show job <job_id>

# 实时查看输出
tail -f logs/multi_seed_*.out

# 取消作业（如需要）
scancel <job_id>
```

### 阶段 5：分析结果（可在登录节点或本地）
```bash
# 统计显著性分析
python utils/statistical_significance.py \
    --baseline_dir outputs/baseline_sft \
    --experimental_dir outputs/multi_teacher_learnable \
    --seeds 42 43 44 \
    --output_dir results/statistical_analysis

# 消融分析
python visualization/ablation_analysis.py \
    --ablation_base_dir outputs/ablations \
    --output_dir results/ablation_visualizations

# 生成学习曲线
python utils/learning_curve_tracker.py \
    --log_dir outputs/multi_teacher_learnable/seed_42/logs \
    --output_dir results/learning_curves
```

---

## ⚠️ 常见问题

### Q1: 登录节点没有 GPU，怎么测试代码？
**A**: 使用 `scripts/verify_login_node.sh` 验证环境和代码导入。真实训练在计算节点（通过 SLURM）。

### Q2: 如何知道计算节点有没有 GPU？
**A**: SLURM 脚本会在计算节点运行时自动检测并打印 GPU 信息（通过 `setup_hpc_environment.sh`）。

### Q3: PyTorch 版本需要匹配 CUDA 吗？
**A**: 是的！根据 `scripts/check_gpu_node.sh` 的报告选择：

**如果报告显示驱动 ≥450**：
```bash
source kava_env/bin/activate
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**如果报告显示驱动 <450**：
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113
```

**如果报告提示需要加载 module**：
编辑 `scripts/setup_hpc_environment.sh`，修改 `CUDA_MODULES` 数组为你的 HPC 提供的版本。

### Q4: 如何下载 Hugging Face 模型？
**A**: 首次运行前需要登录：
```bash
# 方法 1：交互式登录
huggingface-cli login

# 方法 2：设置环境变量（在 SLURM 脚本中）
export HF_TOKEN="your_token_here"
```

### Q5: 如何确认计算节点有 GPU？
**A**: 使用专门的检测脚本：
```bash
sbatch scripts/check_gpu_node.sh
# 等待 1-2 分钟后查看报告
cat logs/gpu_check_*.out
```

报告会显示：
- GPU 型号和数量
- CUDA 驱动版本
- PyTorch CUDA 支持状态
- 详细的诊断建议

如果报告显示 "✗ 计算节点没有 GPU"，请联系 HPC 管理员确认：
1. GPU 分区名称（`#SBATCH --partition=???`）
2. GPU 申请格式（`#SBATCH --gres=gpu:???`）

---

## 📊 预期时间线

| 阶段 | 时间 | 说明 |
|-----|------|-----|
| 环境准备 | 30 分钟 | 安装依赖、验证环境 |
| 数据准备 | 1-2 小时 | 下载数据集、创建切分 |
| 基线训练 | 2-3 天 | 3 种基线 × 3 seeds |
| 主实验训练 | 3-5 天 | 3 种路由策略 × 3 seeds |
| 消融实验 | 2-3 天 | 4 种消融 × 3 seeds |
| 分析可视化 | 4-6 小时 | 统计分析、生成图表 |
| **总计** | **~2 周** | 取决于 HPC 队列等待时间 |

---

## ✅ 检查清单

### 环境准备
- [ ] 代码已上传到 HPC
- [ ] 虚拟环境已创建 (`kava_env`)
- [ ] 依赖已安装 (`requirements.txt`)
- [ ] CPU 版本 PyTorch 已安装（登录节点）
- [ ] `verify_login_node.sh` 运行成功
- [ ] **`check_gpu_node.sh` 报告已查看** ⭐
- [ ] **GPU 和 PyTorch CUDA 均可用** ⭐

### 数据准备
- [ ] 数据切分已创建 (`data_split_controller.py`)
- [ ] 切分哈希已验证（无泄露）
- [ ] Teacher/Student 训练集已分离

### 训练配置
- [ ] SLURM 参数已配置（GPU 数量、时间限制）
- [ ] 模型名称已确认（Student、Teachers）
- [ ] 训练预算已统一（total_tokens）
- [ ] 随机种子已设置（≥3 个）

### 作业提交
- [ ] 基线作业已提交
- [ ] 主实验作业已提交
- [ ] 消融实验作业已提交
- [ ] 作业日志正常输出

### 结果验证
- [ ] Checkpoint 文件已生成
- [ ] 评估结果已保存（JSON）
- [ ] 日志文件完整（TensorBoard/JSON）
- [ ] 统计分析已完成
- [ ] 可视化图表已生成

---

## 🆘 获取帮助

如果遇到问题：
1. 检查 SLURM 日志: `logs/multi_seed_*.out` 和 `.err`
2. 验证环境: `bash scripts/verify_login_node.sh`
3. 测试代码导入: `python -c "import align.tokenizer_align"`
4. 查看 GPU 状态（计算节点）: `nvidia-smi`
5. 检查 CUDA 模块: `module list`

---

**最后更新**: 2025年11月14日  
**适用 HPC 类型**: SLURM 调度系统  
**GPU 要求**: NVIDIA GPU with CUDA 11.3+
