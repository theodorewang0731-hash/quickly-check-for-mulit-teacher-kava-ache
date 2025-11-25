# KaVa Training on HPC - Setup Guide

## 环境准备

### 1. 传输代码到 HPC
```bash
# 从本地上传到 HPC
scp -r "h:/kava/quickly check" username@hpc-cluster:/path/to/workspace/
```

### 2. 在 HPC 上创建虚拟环境
```bash
cd /path/to/workspace/quickly\ check
module load python/3.11
python -m venv venv
source venv/bin/activate
```

### 3. 安装依赖
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.57.1
pip install huggingface_hub==1.1.2
pip install datasets
pip install accelerate
pip install numpy
```

### 4. 登录 Hugging Face
```bash
huggingface-cli login
# 或使用新命令
hf auth login
```

### 5. 准备数据集（可选：预先下载）
```bash
python scripts/export_gsm8k.py
```

## 运行训练

### 单次训练实验
```bash
# 修改 scripts/run_hpc_training.sh 中的路径和参数
sbatch scripts/run_hpc_training.sh
```

### 完整实验套件（E1-E5）
验证 KaVa 三个目标：
- E1: 基线（无 KV 蒸馏）
- E2: Full KV 压缩
- E3: Right-crop KV 压缩
- E4: R-KV 压缩（最稳定）
- E5: Shuffled KV（阴性对照）

```bash
# 修改 scripts/run_all_experiments.sh 中的路径
sbatch scripts/run_all_experiments.sh
```

## 监控训练

### 查看任务状态
```bash
squeue -u $USER
```

### 查看输出日志
```bash
tail -f logs/train_JOBID.out
tail -f logs/train_JOBID.err
```

### 查看训练日志
```bash
tail -f outputs/kava_experiment/training_log.txt
```

## 重要参数说明

### 数据相关
- `--subset_size`: 使用的训练样本数（None=全部）
- `--streaming`: 使用流式模式处理超大数据集
- `--dataset_name`: Hugging Face 数据集名称
- `--train_file`: 本地 JSONL 文件路径

### 训练配置
- `--batch_size`: 每个 GPU 的 batch size
- `--gradient_accumulation_steps`: 梯度累积步数（有效 batch = batch_size × 累积步数）
- `--max_length`: 最大序列长度
- `--epochs`: 训练轮数
- `--lr`: 学习率

### KaVa 特定参数
- `--kv_method`: KV 压缩方法（full/right_crop/rkv）
- `--kv_weight`: KV 损失权重（0=关闭）
- `--codi_weight`: CODI 损失权重
- `--target_len`: 压缩后的 KV 长度
- `--shuffle_kv`: 启用 shuffled KV 对照

### 优化与加速
- `--fp16`: 混合精度训练（节省显存）
- `--gradient_checkpointing`: 梯度检查点（节省显存）
- `--device_map auto`: 自动分配模型到多 GPU
- `--num_workers`: DataLoader 工作进程数

### 检查点与日志
- `--save_steps`: 每 N 步保存检查点
- `--logging_steps`: 每 N 步记录日志
- `--output_dir`: 输出目录

## 超大规模数据集支持

### 使用流式模式
```bash
python experiments/train_with_kv.py \
    --streaming \
    --subset_size 100000 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_workers 8
```

### 多 GPU 并行
```bash
# 修改 SLURM 脚本
#SBATCH --gres=gpu:4

python experiments/train_with_kv.py \
    --device_map "auto" \
    --batch_size 16
```

## 结果分析

训练完成后，每个实验目录包含：
- `checkpoint-epN/`: 每轮的检查点
- `checkpoint-stepN/`: 中间检查点
- `training_log.txt`: 训练日志
- `TRAINING_COMPLETED.txt`: 训练摘要

对比 E1-E5 的最终损失和收敛曲线，验证：
1. KV 压缩保留监督信息（E2 vs E1）
2. KV 对齐补偿 latent 监督（E2/E3/E4 vs E1）
3. R-KV 最稳定（E4 vs E2/E3）
4. Shuffled KV 无效（E5 性能下降）

## 故障排除

### OOM（显存不足）
- 减小 `--batch_size`
- 增加 `--gradient_accumulation_steps`
- 启用 `--gradient_checkpointing`
- 使用 `--fp16`

### 下载超时
- 设置 `--cache_dir` 到高速存储
- 预先下载模型和数据集
- 使用 `--train_file` 加载本地数据

### 训练速度慢
- 增加 `--num_workers`
- 使用多 GPU
- 检查 I/O 瓶颈
