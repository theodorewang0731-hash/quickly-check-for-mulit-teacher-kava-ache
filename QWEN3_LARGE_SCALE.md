# Qwen3 超大规模训练配置指南

## 模型选择

### Qwen2.5 系列（推荐）
```bash
# 7B 模型 - 平衡性能与资源
--model_name "Qwen/Qwen2.5-7B"

# 14B 模型 - 更强性能，需要更多资源
--model_name "Qwen/Qwen2.5-14B"

# 32B 模型 - 顶级性能，需要多节点
--model_name "Qwen/Qwen2.5-32B"
```

## 超大数据集配置

### 1. 流式模式（无限数据集）
```bash
python experiments/train_with_kv.py \
    --streaming \
    --subset_size 1000000  # 100万样本
```

### 2. 常用大规模数据集

#### GSM8K（数学推理）
```bash
--dataset_name "openai/gsm8k" \
--dataset_config "main"
```

#### MATH（高级数学）
```bash
--dataset_name "hendrycks/competition_math" \
--dataset_config "default"
```

#### MetaMathQA（大规模数学）
```bash
--dataset_name "meta-math/MetaMathQA" \
--dataset_config "default"
```

#### OpenOrca（通用指令）
```bash
--dataset_name "Open-Orca/OpenOrca" \
--dataset_config "default"
```

#### Alpaca（大规模指令）
```bash
--dataset_name "tatsu-lab/alpaca" \
--dataset_config "default"
```

#### The Stack（代码数据集）
```bash
--dataset_name "bigcode/the-stack" \
--dataset_config "python"  # 或其他语言
```

## 资源配置建议

### 配置 1: 单 GPU（7B 模型，小规模测试）
```bash
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
--batch_size 1
--gradient_accumulation_steps 32
--max_length 1024
```

### 配置 2: 4 GPU（7B 模型，中等规模）
```bash
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
--batch_size 2
--gradient_accumulation_steps 16
--max_length 1024
```

### 配置 3: 8 GPU（7B 模型，大规模）⭐ 推荐
```bash
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
--batch_size 2
--gradient_accumulation_steps 16
--max_length 2048
```

### 配置 4: 多节点（14B+ 模型，超大规模）
```bash
#SBATCH --nodes=4
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
--batch_size 1
--gradient_accumulation_steps 32
--max_length 2048
```

## 显存优化技巧

### 1. 基础优化
```bash
--gradient_checkpointing \
--fp16 \
--torch_dtype "bfloat16"
```

### 2. 激进优化（显存不足时）
```bash
--batch_size 1 \
--gradient_accumulation_steps 64 \
--gradient_checkpointing \
--max_length 512  # 减小序列长度
```

### 3. 使用 8-bit 量化（需要 bitsandbytes）
在训练脚本中添加：
```python
model_kwargs["load_in_8bit"] = True
```

## 实际使用示例

### 示例 1: Qwen2.5-7B + GSM8K（10万样本）
```bash
sbatch scripts/run_hpc_training.sh
# 已配置为 4 GPU, 50K 样本
```

### 示例 2: 超大规模训练（100万样本）
```bash
sbatch scripts/run_ultra_large.sh
# 8 GPU, 流式模式, 7天训练时间
```

### 示例 3: 完整实验套件（E1-E5）
```bash
sbatch scripts/run_all_experiments.sh
# 自动运行所有对比实验
```

### 示例 4: 多节点训练
```bash
sbatch scripts/run_multinode.sh
# 4 节点 × 8 GPU = 32 GPU
```

## 性能估算

### Qwen2.5-7B 训练速度（单 GPU A100）
- Batch size 1, seq 1024: ~0.5 样本/秒
- Batch size 2, seq 1024: ~0.8 样本/秒
- 8 GPU 并行: ~6 样本/秒

### 训练时间估算
- 10K 样本: ~30 分钟（8 GPU）
- 100K 样本: ~5 小时（8 GPU）
- 1M 样本: ~50 小时（8 GPU）

## 监控与调试

### 查看 GPU 使用率
```bash
watch -n 1 nvidia-smi
```

### 查看训练进度
```bash
tail -f logs/train_*.out
tail -f outputs/*/training_log.txt
```

### 查看显存占用
```bash
# 在训练脚本中添加
import torch
print(torch.cuda.memory_summary())
```

## 常见问题

### OOM（显存溢出）
1. 减小 `--batch_size`
2. 增加 `--gradient_accumulation_steps`
3. 减小 `--max_length`
4. 启用 `--gradient_checkpointing`
5. 使用更多 GPU

### 训练太慢
1. 增加 `--num_workers`
2. 使用 `--streaming` 模式
3. 启用 `--fp16`
4. 增加 GPU 数量
5. 检查是否有 I/O 瓶颈

### 下载失败
1. 预先下载模型：
```bash
python scripts/download_assets.py \
    --model Qwen/Qwen2.5-7B \
    --dataset openai/gsm8k
```

2. 或使用本地数据：
```bash
--train_file data/my_dataset.jsonl
```

## 最佳实践

1. **先小规模测试**：用 `--subset_size 100` 测试代码
2. **逐步扩大**：确认无误后再用完整数据
3. **定期保存**：`--save_steps 1000`
4. **监控日志**：定期查看 loss 曲线
5. **备份检查点**：重要模型及时备份

## 示例命令

### 快速测试（100 样本）
```bash
python experiments/train_with_kv.py \
    --model_name Qwen/Qwen2.5-7B \
    --subset_size 100 \
    --epochs 1 \
    --batch_size 2 \
    --device_map auto \
    --trust_remote_code
```

### 生产级训练（完整数据）
```bash
sbatch scripts/run_ultra_large.sh
```
