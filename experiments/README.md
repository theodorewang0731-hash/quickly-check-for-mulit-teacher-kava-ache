# experiments

包含复现 KaVa 实验的最小可运行代码与说明。

主要脚本：
- `train_baseline.py`：在 GSM8k 的小子集上训练 GPT-2 的最小基线（默认 100 条）。

运行示例（PowerShell）：
```powershell
# 建议先激活 .venv
.\.venv\Scripts\Activate.ps1
python experiments/train_baseline.py --subset_size 100 --model_name gpt2 --output_dir outputs/baseline --epochs 3 --batch_size 4
```

注意：脚本会尝试下载 GSM8k 与预训练模型（gpt2），请在有网络和足够磁盘空间的环境运行。
