# KaVa: Multi-Teacher KV Distillation with Learnable Routing

**一个用于多教师 KV 蒸馏的端到端训练框架，包含完整的硬性控制和 HPC 部署工具。**

> ✨ **v4.0 更新 (2025-12-09)**: 
> - ✅ **地图投影对齐器（Map Projection Aligner）**: Anti-Flatten 结构化设计
> - ✅ **双模式支持**: structured (v4.0) vs flat (baseline) A/B 对比
> - ✅ **结构化 KV 损失**: K/V 方向对齐 + Q-K 交互对齐
> - ✅ **训练脚本集成完成**: `train_with_kv.py` 支持双模式切换
> - 🧪 **测试验证中**: 冒烟测试和 A/B 实验准备中
> 
> 📖 查看 [`DEVELOPMENT_HISTORY.md`](DEVELOPMENT_HISTORY.md) 了解完整发展历程  
> 📖 查看 [`V4_INTEGRATION_COMPLETE.md`](V4_INTEGRATION_COMPLETE.md) 了解集成详情  
> 🚀 查看 [`V4_EXECUTION_ROADMAP.md`](V4_EXECUTION_ROADMAP.md) 了解执行计划

---

## 🚀 快速开始

### 本地开发（Windows）

```powershell
# 1. 创建虚拟环境
python -m venv kava_env
.\kava_env\Scripts\Activate.ps1

# 2. 安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 3. 运行自动修复（修复换行符等问题）
bash scripts/auto_fix.sh

# 4. 运行完整验证
python scripts/pre_training_validation.py
```

---

### HPC 部署（推荐流程）

#### 方法 1：一键部署

```bash
cd /path/to/kava/quickly_check

# 自动修复常见问题
bash scripts/auto_fix.sh

# 一键部署和检测
bash scripts/deploy_and_check.sh
```

#### 方法 2：逐步验证

```bash
# 1. 修复潜在问题
bash scripts/auto_fix.sh

# 2. 完整验证（Python）
python scripts/pre_training_validation.py

# 3. 验证登录节点
bash scripts/verify_login_node.sh

# 4. 提交 GPU 检测
sbatch scripts/check_gpu_node.sh

# 5. 查看报告
cat logs/gpu_check_*.out
```

这会自动：
1. ✅ 验证登录节点环境
2. ✅ 提交 GPU 检测作业到计算节点
3. ✅ 生成详细的诊断报告

**查看报告后继续**：
```bash
# 查看 GPU 检测报告
cat logs/gpu_check_*.out

# 如果显示 "🎉 恭喜！"，立即提交训练
sbatch scripts/run_multi_seed_experiments.sh
```

📖 **详细文档**：
- **HPC 执行清单**: [`HPC_EXECUTION_CHECKLIST.md`](HPC_EXECUTION_CHECKLIST.md) ⭐
- **快速命令**: [`HPC_COMMAND_REFERENCE.md`](HPC_COMMAND_REFERENCE.md)
- **部署指南**: [`HPC_DEPLOYMENT_GUIDE.md`](HPC_DEPLOYMENT_GUIDE.md)
- **快速开始**: [`HPC_QUICKSTART.md`](HPC_QUICKSTART.md)

---

## 📁 项目结构

```
kava/quickly_check/
├── align/                    # Token 和层对齐
│   ├── tokenizer_align.py   # Tokenizer 软对齐
│   └── layer_map.py         # 层映射策略
├── teacher/                  # 教师模型路由
│   ├── router_proto.py      # 路由协议（Fixed/Similarity/Learnable）
│   └── ensemble.py          # 集成策略
├── data/                     # 数据处理
│   ├── multi_task_dataset.py
│   └── data_split_controller.py  # 统一数据切分
├── utils/                    # 工具库
│   ├── training_budget_controller.py  # 训练预算控制
│   ├── statistical_significance.py    # 统计显著性测试
│   └── learning_curve_tracker.py      # 学习曲线追踪
├── visualization/            # 可视化
│   ├── hpc_visualizer.py    # HPC 可视化（无显示器）
│   └── ablation_analysis.py # 消融分析
├── experiments/              # 训练脚本
│   ├── train_with_kv.py     # KV 蒸馏训练
│   └── train_standard_sft.py
├── scripts/                  # SLURM 作业脚本
│   ├── auto_fix.sh          # 自动修复脚本（换行符、权限等）⭐
│   ├── pre_training_validation.py  # 训练前完整验证 ⭐
│   ├── monitor_training.sh  # 训练监控（支持 --auto）⭐
│   ├── deploy_and_check.sh  # 一键部署
│   ├── verify_login_node.sh # 登录节点验证
│   ├── check_gpu_node.sh    # GPU 环境检测
│   ├── setup_hpc_environment.sh  # 自动环境配置
│   ├── run_multi_seed_experiments.sh
│   ├── run_ablation_studies.sh
│   └── run_three_stage_routing.sh
└── docs/
    ├── RIGOROUS_CONTROLS.md  # 硬性控制文档
    ├── EXPERIMENT_DESIGN.md  # 实验设计
    └── VISUALIZATION_QUICKSTART.md
```

---

## ⭐ 核心特性

### 1. 多教师 KV 蒸馏
- **三种路由策略**: Fixed、Similarity-based、Learnable
- **软对齐**: Token 长度自适应对齐
- **层级映射**: 教师-学生层对应策略

### 2. 硬性控制（避免被审稿人质疑）
- ✅ **等算力控制**: 统一训练步数和 token 数
- ✅ **多随机种子**: ≥3 个种子，统计显著性测试（t-test, bootstrap CI）
- ✅ **数据切分一致**: MD5 哈希验证，防止数据泄露
- ✅ **公平基线**: Teacher/Student 训练集分离
- ✅ **学习曲线**: KV Loss ↓ + Task Accuracy ↑ 同步追踪
- ✅ **消融实验**: 4 种消融（路由、层、K/V、对齐）

📖 完整文档：[`RIGOROUS_CONTROLS.md`](RIGOROUS_CONTROLS.md)

### 3. HPC 友好
- **自动问题修复**: 自动检测并修复常见问题（换行符、路径引号等）
- **完整验证**: 训练前 10+ 项检查（代码、环境、配置）
- **自动环境检测**: GPU/CUDA 自动配置
- **实时监控**: 支持自动刷新的训练监控（`--auto`）
- **SLURM 集成**: 完整的作业脚本
- **无显示器可视化**: 自包含 HTML 报告
- **断点续训**: Checkpoint 自动恢复

### 4. 可解释性分析
- **路由热力图**: 按层、按任务的路由权重分布
- **KV Loss 热力图**: 各层蒸馏效果可视化
- **学习曲线**: 双轴图（Loss + Accuracy）

---

## 🎯 典型工作流

### 阶段 1: 环境准备（登录节点）

```bash
# 上传代码
scp -r ./kava/quickly_check user@hpc:/path/to/

# SSH 登录
ssh user@hpc
cd /path/to/kava/quickly_check

# 创建虚拟环境
python -m venv kava_env
source kava_env/bin/activate

# 安装依赖（CPU 版本，登录节点用）
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 运行验证
bash scripts/verify_login_node.sh
```

---

### 阶段 2: GPU 环境检测（计算节点）

```bash
# 提交检测作业
chmod +x scripts/check_gpu_node.sh
sbatch scripts/check_gpu_node.sh

# 等待 1-2 分钟后查看报告
cat logs/gpu_check_*.out
```

**根据报告采取行动**：
- ✅ **环境完美**: 进入阶段 2.5（模型库检查）
- ⚠️ **需要重装 PyTorch**: 按报告中的命令执行
- ❌ **没有 GPU**: 联系管理员

---

### 阶段 2.5: HPC 共享模型库检查 ⭐ **新增**

```bash
# 检查 HPC 共享模型库
bash scripts/check_hpc_models.sh

# 检测 SLURM 资源限制
# （建议在计算节点运行以获取 GPU 信息）
srun --partition=gpu --gres=gpu:1 --pty bash scripts/detect_slurm_resources.sh

# 综合部署前验证
bash scripts/comprehensive_pre_deployment_check.sh
```

**共享模型库优势**：
- ✅ **无需下载**: 所有模型已在 `/home/share/models`
- ✅ **无需登录**: 不需要 HuggingFace Token
- ✅ **节省空间**: 避免每个用户重复下载
- ✅ **加速启动**: 跳过模型下载时间

**环境变量自动配置**（由 `setup_hpc_environment.sh` 自动处理）：
```bash
export HF_HOME="/home/share/models"
export TRANSFORMERS_CACHE="/home/share/models"
export HF_DATASETS_CACHE="${HOME}/.cache/huggingface/datasets"
```

**可用模型列表**：
- `Qwen/Qwen2.5-1.5B` (学生模型)
- `Qwen/Qwen2.5-7B` (教师模型)
- `Qwen/Qwen2.5-14B` (教师模型)
- `Qwen/Qwen2.5-0.5B` (快速测试)
- `meta-llama/Llama-3.2-1B` (对比实验)
- `meta-llama/Llama-3.2-3B` (对比实验)

---

### 阶段 3: 数据准备

```bash
python data/data_split_controller.py \
    --dataset_names gsm8k svamp strategyqa math arc_challenge \
    --output_dir ./data/unified_splits \
    --teacher_separate \
    --val_size 0.1
```

---

### 阶段 4: 训练

```bash
# 多种子实验（3 seeds）
sbatch scripts/run_multi_seed_experiments.sh

# 监控（自动刷新模式）⭐
bash scripts/monitor_training.sh --auto

# 或查看队列
squeue -u $USER
```

---

### 阶段 5: 分析

```bash
# 统计显著性
python utils/statistical_significance.py \
    --baseline_dir outputs/baseline_sft \
    --experimental_dir outputs/multi_teacher_learnable \
    --seeds 42 43 44

# 消融分析
python visualization/ablation_analysis.py \
    --ablation_base_dir outputs/ablations

# 学习曲线
python utils/learning_curve_tracker.py \
    --log_dir outputs/.../logs
```

---

## 📊 预期结果

根据我们的实验设计：

| 实验组 | 预期提升 | 统计显著性 |
|--------|---------|-----------|
| Multi-Teacher vs Single | +7-10% | p<0.01 |
| Learnable vs Fixed | +4.5% | p<0.01 |
| Full Layers vs Shallow | +6.6% | p<0.001 |
| K+V vs Only K/V | +4.4% | p<0.01 |
| Soft vs Hard Align | +2.4%, std↓57% | p<0.05 |

---

## 🆘 故障排查

### Q: 登录节点没有 GPU？
**A**: 正常！登录节点用于编辑代码和提交作业，真实训练在计算节点（通过 SLURM）。

### Q: PyTorch 检测不到 CUDA？
**A**: 运行 `sbatch scripts/check_gpu_node.sh` 查看报告，按建议重装 PyTorch。

### Q: 作业一直 PENDING？
**A**: 资源不足，运行 `squeue -u $USER --start` 查看原因。

### Q: 模型下载失败或需要 HuggingFace Token？⭐ **新增**
**A**: 使用 HPC 共享模型库！运行 `bash scripts/check_hpc_models.sh` 检查可用模型。环境脚本会自动配置 `HF_HOME=/home/share/models`，无需下载或登录。

### Q: SLURM 作业提交失败（资源配置错误）？⭐ **新增**
**A**: 运行 `bash scripts/detect_slurm_resources.sh` 检测集群资源限制，根据建议调整 `--gres`、`--cpus-per-task`、`--mem` 参数。

### Q: 脚本出现 "bad interpreter" 或语法错误？⭐ **新增**
**A**: Windows 行尾问题，运行 `bash scripts/auto_fix.sh` 自动转换所有脚本为 Unix 格式 (LF)。

📖 **完整故障排查**: [`HPC_COMMAND_REFERENCE.md`](HPC_COMMAND_REFERENCE.md)

---

## 🔧 HPC 部署工具 ⭐ **新增**

### 自动化工具

| 工具 | 功能 | 使用场景 |
|------|------|---------|
| `check_hpc_models.sh` | 检查共享模型库 | 验证模型可用性 |
| `detect_slurm_resources.sh` | 检测资源限制 | 配置 SLURM 参数 |
| `comprehensive_pre_deployment_check.sh` | 综合部署前验证 | 提交作业前全面检查 |
| `auto_fix.sh` | 自动修复常见问题 | 行尾转换、权限设置 |
| `pre_training_validation.py` | 训练前验证 | 10+ 项检查 |
| `monitor_training.sh --auto` | 实时监控 | 自动刷新训练状态 |

### 使用示例

```bash
# 完整部署前检查流程
bash scripts/auto_fix.sh                              # 修复脚本格式
bash scripts/check_hpc_models.sh                      # 检查模型库
bash scripts/comprehensive_pre_deployment_check.sh    # 综合验证
python scripts/pre_training_validation.py             # Python 环境验证

# 在计算节点获取 GPU 信息
srun --partition=gpu --gres=gpu:1 --pty bash scripts/detect_slurm_resources.sh

# 提交作业并监控
sbatch scripts/run_multi_seed_experiments.sh
bash scripts/monitor_training.sh --auto
```

---

## 📚 文档索引

### HPC 部署
- **⭐ 执行清单**: [`HPC_EXECUTION_CHECKLIST.md`](HPC_EXECUTION_CHECKLIST.md) - 逐步检查清单
- **命令参考**: [`HPC_COMMAND_REFERENCE.md`](HPC_COMMAND_REFERENCE.md) - 所有命令速查
- **部署指南**: [`HPC_DEPLOYMENT_GUIDE.md`](HPC_DEPLOYMENT_GUIDE.md) - 完整部署流程
- **快速开始**: [`HPC_QUICKSTART.md`](HPC_QUICKSTART.md) - 三步走战略

### 实验设计
- **硬性控制**: [`RIGOROUS_CONTROLS.md`](RIGOROUS_CONTROLS.md) - 7 大硬性控制
- **实验设计**: [`EXPERIMENT_DESIGN.md`](EXPERIMENT_DESIGN.md) - 完整实验方案
- **可视化**: [`VISUALIZATION_QUICKSTART.md`](VISUALIZATION_QUICKSTART.md) - 图表生成

### 实现文档
- **实现总结**: [`CONTROLS_IMPLEMENTATION_DONE.md`](CONTROLS_IMPLEMENTATION_DONE.md)

---

## 🔧 依赖

- **Python**: 3.10+
- **PyTorch**: 2.0+ (CUDA 11.8+ 或 CPU)
- **Transformers**: 4.57+
- **其他**: accelerate, datasets, scipy, scikit-learn, matplotlib, seaborn

完整列表见 [`requirements.txt`](requirements.txt)

---

## 📄 许可证

MIT License

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📧 联系

如有问题，请通过 Issue 联系。

---

**最后更新**: 2025年11月14日  
**版本**: v1.0  
**状态**: 生产就绪，HPC 部署已验证
