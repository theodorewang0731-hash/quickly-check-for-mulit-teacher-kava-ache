# KaVa Project - Quick Start Guide

## 一键部署到 HPC

### 1. 上传后立即执行
```bash
cd /path/to/uploaded/quickly\ check  # 或你上传的路径
bash setup.sh
```

`setup.sh` 会自动：
- 创建所有必要的目录（logs, outputs, cache, data）
- 检测 Python 环境
- 创建虚拟环境
- 安装所有依赖
- 检查 GPU 可用性
- 设置脚本权限

### 2. 登录 Hugging Face
```bash
hf auth login
# 粘贴你的 token
```

### 3. 测试安装
```bash
bash scripts/test_setup.sh
```

### 4. 提交训练任务

**单次实验：**
```bash
sbatch scripts/run_hpc_training.sh
```

**完整实验套件（E1-E5）：**
```bash
sbatch scripts/run_all_experiments.sh
```

### 5. 监控训练
```bash
# 查看任务状态
squeue -u $USER

# 实时查看输出
tail -f logs/train_*.out

# 查看训练日志
tail -f outputs/kava_experiment/training_log.txt
```

### 6. 训练完成后分析
```bash
python scripts/analyze_results.py --output_base outputs/
```

## 目录结构

```
quickly check/
├── setup.sh                    # 一键安装脚本
├── requirements.txt            # Python 依赖
├── HPC_SETUP.md               # 详细部署文档
├── experiments/
│   ├── train_with_kv.py       # 主训练脚本
│   ├── kv_utils.py            # KV 压缩工具
│   ├── kv_loss.py             # KV 损失函数
│   └── projector.py           # 投影层
├── scripts/
│   ├── test_setup.sh          # 安装测试脚本
│   ├── run_hpc_training.sh    # 单次训练 SLURM 脚本
│   ├── run_all_experiments.sh # E1-E5 实验套件
│   ├── export_gsm8k.py        # 数据集导出
│   └── analyze_results.py     # 结果分析
├── data/                       # 数据目录（自动创建）
├── outputs/                    # 输出目录（自动创建）
├── logs/                       # 日志目录（自动创建）
├── cache/                      # 模型缓存（自动创建）
└── venv/                       # 虚拟环境（setup.sh创建）
```

## 关键特性

### 自动路径检测
- 所有脚本使用相对路径，无需手动配置
- 缓存和日志自动保存在项目目录内

### 模块加载兼容
- SLURM 脚本中的 `module load` 已注释
- 如果你的 HPC 需要模块，取消注释即可

### 灵活配置
- 可通过修改 `.sh` 文件调整 GPU/CPU/内存资源
- 训练参数在脚本中清晰列出，易于修改

### 完整实验流程
- E1-E5 自动运行，验证 KaVa 三个目标
- 自动生成对比报告和统计分析

## 故障排查

### 如果 Python 不可用
```bash
module load python/3.11  # 或你的 HPC Python 模块
bash setup.sh
```

### 如果 GPU 不可用
检查 SLURM 脚本中的：
- `#SBATCH --gres=gpu:1`
- `#SBATCH --partition=gpu`

根据你的 HPC 配置调整。

### 如果依赖安装失败
```bash
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### 如果任务提交失败
检查 `sinfo` 查看可用分区和资源，修改 SLURM 脚本的：
- `--partition`
- `--gres`
- `--mem`
- `--time`

## 预期结果

训练完成后，每个实验目录包含：
- `checkpoint-epN/` - 每轮检查点
- `checkpoint-stepN/` - 中间检查点  
- `training_log.txt` - 详细训练日志
- `TRAINING_COMPLETED.txt` - 训练摘要

运行 `analyze_results.py` 会生成：
- `EXPERIMENT_REPORT.txt` - 对比报告
- 验证 KaVa 三个主张的统计结果

## 需要帮助？

如果遇到问题，检查：
1. `logs/train_*.err` - 错误日志
2. `logs/train_*.out` - 标准输出
3. `outputs/*/training_log.txt` - 训练日志

把具体错误信息发给我，我可以帮你调试。
