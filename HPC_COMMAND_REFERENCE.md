# 🚀 HPC 快速命令参考

## 一键部署（推荐）

```bash
cd /path/to/kava/quickly_check
bash scripts/deploy_and_check.sh
```

这会自动：
1. ✅ 检查所有必需文件
2. ✅ 设置脚本权限
3. ✅ 创建必需目录
4. ✅ 运行登录节点验证
5. ✅ 提交 GPU 检测作业

---

## 手动分步执行

### 准备阶段

```bash
# 进入项目目录
cd /path/to/kava/quickly_check

# 激活虚拟环境
source kava_env/bin/activate

# 给脚本执行权限
chmod +x scripts/*.sh

# 创建日志目录
mkdir -p logs
```

---

### 验证登录节点

```bash
bash scripts/verify_login_node.sh
```

预期输出最后一行：
```
✓ 登录节点验证完成！
```

---

### 检测计算节点 GPU

```bash
sbatch scripts/check_gpu_node.sh
```

预期输出：
```
Submitted batch job 12345678
```

---

### 查看作业状态

```bash
# 查看我的所有作业
squeue -u $USER

# 查看特定作业
squeue -j 12345678

# 查看作业详情
scontrol show job 12345678
```

---

### 查看 GPU 检测报告

```bash
# 等待作业完成后
cat logs/gpu_check_*.out

# 如果有错误
cat logs/gpu_check_*.err
```

---

## 根据报告采取行动

### 情况 A：环境完美 ✅

```bash
# 直接提交训练
sbatch scripts/run_multi_seed_experiments.sh
```

---

### 情况 B：需要重装 PyTorch ⚠️

```bash
source kava_env/bin/activate

# 卸载现有版本
pip uninstall torch torchvision torchaudio -y

# 安装匹配的 CUDA 版本（根据报告选择）
# CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或 CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 再次检测
sbatch scripts/check_gpu_node.sh
```

---

### 情况 C：需要加载 CUDA 模块 📋

```bash
# 编辑环境配置
vim scripts/setup_hpc_environment.sh

# 修改 CUDA_MODULES 数组为你的 HPC 提供的版本
# 例如: CUDA_MODULES=("cuda/11.8" "cuda/12.1")

# 保存后重新检测
sbatch scripts/check_gpu_node.sh
```

---

### 情况 D：没有 GPU ❌

联系 HPC 管理员，询问：
1. GPU 分区名称：`--partition=???`
2. GPU 申请格式：`--gres=gpu:???`

然后编辑：
```bash
vim scripts/check_gpu_node.sh
# 修改第 11 行: #SBATCH --partition=YOUR_GPU_PARTITION
# 修改第 14 行: #SBATCH --gres=gpu:YOUR_FORMAT
```

---

## 训练作业管理

### 提交训练

```bash
# 多种子实验
sbatch scripts/run_multi_seed_experiments.sh

# 消融实验
sbatch scripts/run_ablation_studies.sh

# 完整实验流程
sbatch scripts/run_three_stage_routing.sh
```

---

### 监控训练

```bash
# 查看作业队列
squeue -u $USER

# 实时查看日志
tail -f logs/multi_seed_*.out

# 查看最近 50 行
tail -n 50 logs/multi_seed_*.out
```

---

### 取消作业

```bash
# 取消单个作业
scancel 12345678

# 取消所有作业
scancel -u $USER

# 取消特定名称的作业
scancel --name=multi_seed_experiments
```

---

## 常用查询命令

### 查看分区信息

```bash
# 所有分区
sinfo

# GPU 分区
sinfo -p gpu

# 详细信息
sinfo -Nel
```

---

### 查看作业历史

```bash
# 最近的作业
sacct -u $USER

# 详细信息
sacct -u $USER --format=JobID,JobName,State,ExitCode,Elapsed,ReqMem

# 特定时间范围
sacct -u $USER --starttime=2025-11-01
```

---

### 查看资源配额

```bash
# 查看账户信息（如果 HPC 有配额系统）
sshare -u $USER

# 查看存储使用
du -sh /path/to/kava/quickly_check
df -h $HOME
```

---

## 调试命令

### 测试 Python 环境

```bash
source kava_env/bin/activate

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

---

### 测试模块导入

```bash
python -c "
import sys
sys.path.insert(0, '.')
from align.tokenizer_align import TokenizerAligner
print('✓ TokenizerAligner')
"
```

---

### 查看 GPU 使用（在计算节点）

```bash
# 在交互式会话或作业中
nvidia-smi

# 持续监控
watch -n 1 nvidia-smi
```

---

## 数据处理

### 创建数据切分

```bash
python data/data_split_controller.py \
    --dataset_names gsm8k svamp strategyqa math arc_challenge \
    --output_dir ./data/unified_splits \
    --teacher_separate \
    --val_size 0.1 \
    --test_size 0.1
```

---

### 验证数据切分

```bash
python data/data_split_controller.py \
    --validate_only \
    --split_dir ./data/unified_splits
```

---

## Hugging Face 登录

### 交互式登录

```bash
huggingface-cli login
# 然后粘贴你的 token
```

---

### 环境变量方式

```bash
# 在 ~/.bashrc 或作业脚本中
export HF_TOKEN="your_token_here"
```

---

## 结果分析

### 统计显著性分析

```bash
python utils/statistical_significance.py \
    --baseline_dir outputs/baseline_sft \
    --experimental_dir outputs/multi_teacher_learnable \
    --seeds 42 43 44 \
    --output_dir results/statistical_analysis
```

---

### 生成可视化

```bash
python visualization/ablation_analysis.py \
    --ablation_base_dir outputs/ablations \
    --output_dir results/ablation_visualizations
```

---

### 学习曲线

```bash
python utils/learning_curve_tracker.py \
    --log_dir outputs/multi_teacher_learnable/seed_42/logs \
    --output_dir results/learning_curves
```

---

## 文件传输

### 上传到 HPC

```bash
# 单个文件
scp file.py user@hpc:/path/to/destination/

# 整个目录
scp -r ./kava/quickly_check user@hpc:/path/to/

# 使用 rsync（推荐，支持断点续传）
rsync -avz --progress ./kava/quickly_check user@hpc:/path/to/
```

---

### 下载结果

```bash
# 下载日志
scp user@hpc:/path/to/kava/quickly_check/logs/*.out ./local_logs/

# 下载结果
scp -r user@hpc:/path/to/kava/quickly_check/results ./local_results/

# 使用 rsync
rsync -avz --progress user@hpc:/path/to/kava/quickly_check/results/ ./local_results/
```

---

## 紧急操作

### 系统资源不足

```bash
# 减少 GPU 数量（编辑脚本）
vim scripts/run_multi_seed_experiments.sh
# 修改: #SBATCH --gres=gpu:2  # 从 8 改为 2

# 减少训练时间（测试用）
# 修改: TOTAL_TOKENS=100000000  # 从 1B 改为 100M
```

---

### 作业卡住

```bash
# 查看为什么卡住
squeue -u $USER --start

# 查看节点状态
sinfo -Nel | grep gpu

# 如果队列太长，考虑换分区或减少资源
```

---

### 磁盘空间不足

```bash
# 查看使用情况
du -sh outputs/*

# 删除旧的 checkpoint（保留最后几个）
find outputs -name "checkpoint-*" -type d | head -n -3 | xargs rm -rf

# 清理缓存
rm -rf ~/.cache/huggingface/transformers/*
```

---

## 📚 文档快速索引

- **执行清单**: `HPC_EXECUTION_CHECKLIST.md`
- **部署指南**: `HPC_DEPLOYMENT_GUIDE.md`
- **快速开始**: `HPC_QUICKSTART.md`
- **硬性控制**: `RIGOROUS_CONTROLS.md`
- **实验设计**: `EXPERIMENT_DESIGN.md`

---

**最后更新**: 2025年11月14日  
**版本**: v1.0
