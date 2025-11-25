# HPC 可视化指南

## 🎯 核心功能

**在 HPC 训练完成后，自动生成一个自包含的 HTML 文件，你只需：**

1. ✅ 复制训练结束时输出的路径
2. ✅ 用 `scp` 下载到本地
3. ✅ 双击打开，所有图表都在里面！

**特点**：
- 📦 **自包含**：图片全部嵌入为 base64，单文件即可
- 🚀 **快速**：只需下载一个 HTML 文件（~2-5MB）
- 🌐 **离线可用**：下载后可在任何地方打开
- 📊 **完整信息**：训练曲线、评测结果、路由权重全包含

---

## 🚀 快速开始

### 方法 1: 自动可视化训练（最简单）

```bash
# 1. 提交训练任务（自动生成可视化）
sbatch scripts/train_with_visualization.sh

# 2. 训练结束后，你会看到类似这样的输出：
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# /home/username/kava/outputs/experiment_20250113_143022/visualizations/experiment_summary.html
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 3. 复制上面的路径，然后在你的本地电脑运行（替换 YOUR_USERNAME 和 HPC_ADDRESS）：
scp YOUR_USERNAME@HPC_ADDRESS:/home/username/kava/outputs/.../experiment_summary.html ~/Downloads/report.html

# 4. 双击打开
open ~/Downloads/report.html  # macOS
# 或
start ~/Downloads/report.html  # Windows
# 或
xdg-open ~/Downloads/report.html  # Linux
```

### 方法 2: 使用自动下载脚本（更简单）

```bash
# 训练结束后，会生成一个 download_and_open.sh 脚本

# 1. 下载脚本到本地（在本地电脑运行）
scp YOUR_USERNAME@HPC:/path/to/outputs/experiment_*/download_and_open.sh ~/

# 2. 运行脚本（自动下载并打开）
bash ~/download_and_open.sh
```

---

## 📊 生成的 HTML 报告包含什么？

打开 `experiment_summary.html`，你会看到：

### 1. 📈 训练曲线
- Train/Eval Loss 对比
- KV Distillation Loss
- Learning Rate Schedule
- Gradient Norm

### 2. 📉 评测结果表格
```
Dataset          Score
────────────────────────
GSM8K test      75.3%
MATH500         42.1%
BBH             68.5%
GPQA            35.2%
TruthfulQA      52.8%
CMMLU           63.4%
C-Eval          61.9%
────────────────────────
Average         57.0%
```

### 3. 🔀 路由权重分布（如果使用可学习路由）
- 权重随训练步数的变化
- 最终权重分布柱状图

### 4. 📍 文件路径（带一键复制按钮）
- 点击 "📋 Copy Path" 即可复制路径

---

## 🎨 HTML 文件特点

### 自包含设计
```html
<!-- 所有图片都嵌入为 base64 -->
<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...">
```

**优势**：
- ✅ 只需一个文件
- ✅ 可以通过邮件发送
- ✅ 可以上传到云盘分享
- ✅ 离线也能打开

### 响应式设计
- 自适应不同屏幕大小
- 移动设备友好
- 打印友好

### 交互功能
- 点击复制文件路径
- 图片可以右键保存
- 表格可以复制粘贴到 Excel

---

## 🎬 完整示例演示

### 训练完成后的输出示例

```bash
======================================================================
✓ Visualization Complete!
======================================================================

📊 Main Report Generated:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
/scratch/username/kava/outputs/experiment_20250113_143022/visualizations/experiment_summary.html
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📥 COPY AND RUN THIS COMMAND ON YOUR LOCAL MACHINE:
======================================================================

  scp username@hpc.university.edu:/scratch/username/.../experiment_summary.html ~/Downloads/report.html

======================================================================
Then open the file:
======================================================================

  # macOS:
  open ~/Downloads/report.html

  # Windows (PowerShell):
  start ~/Downloads/report.html

  # Linux:
  xdg-open ~/Downloads/report.html

======================================================================
💡 Tips:
======================================================================
  • The HTML file works offline (no internet needed)
  • You can share it via email or cloud storage
  • All images are embedded (no separate files needed)
  • Compatible with all modern browsers
======================================================================
```

### 实际操作流程

```bash
# 1. 在 HPC 上提交任务
[you@hpc]$ sbatch scripts/train_with_visualization.sh
Submitted batch job 123456

# 2. 等待训练完成（几小时到几天）
[you@hpc]$ squeue -u $USER
# ... training ...

# 3. 训练完成，查看输出日志
[you@hpc]$ tail -50 logs/training_with_viz_123456.log
# 看到可视化生成的路径

# 4. 复制输出的 scp 命令
# 例如：scp username@hpc:/scratch/.../experiment_summary.html ~/Downloads/report.html

# 5. 在你的本地电脑（Windows/Mac/Linux）运行：
[you@local]$ scp username@hpc:/scratch/username/kava/outputs/experiment_20250113_143022/visualizations/experiment_summary.html ~/Downloads/my_experiment.html
experiment_summary.html    100% 3.2MB   1.2MB/s   00:02

# 6. 打开文件
[you@local]$ open ~/Downloads/my_experiment.html  # macOS
# 或者直接双击文件

# 7. 在浏览器中查看所有结果！ 🎉
```

---

## 📱 多种下载方式

### 方式 1: 命令行 SCP（推荐）

```bash
# 基本用法
scp user@hpc:/full/path/to/experiment_summary.html ~/Downloads/

# 重命名下载
scp user@hpc:/path/to/report.html ~/Downloads/my_experiment_$(date +%Y%m%d).html

# 下载整个可视化文件夹
scp -r user@hpc:/path/to/visualizations/ ~/Downloads/exp_viz/
```

### 方式 2: rsync（增量同步）

```bash
# 第一次下载
rsync -avz --progress user@hpc:/path/to/visualizations/ ~/Downloads/viz/

# 后续同步（只下载新文件）
rsync -avz --progress user@hpc:/path/to/visualizations/ ~/Downloads/viz/
```

### 方式 3: 图形界面工具

**FileZilla / WinSCP**：
1. 连接到 HPC
2. 导航到可视化目录
3. 拖拽 `experiment_summary.html` 到本地
4. 双击打开

**VS Code Remote**：
1. 右键点击 `experiment_summary.html`
2. 选择 "Download"
3. 打开下载的文件

### 方式 4: 自动下载脚本

```bash
# 训练完成后会生成 download_and_open.sh

# 下载脚本
scp user@hpc:/path/to/download_and_open.sh ~/

# 运行（自动下载并打开）
bash ~/download_and_open.sh
```

---

## 🔧 本地测试（不需要 HPC）

如果你想先在本地测试可视化功能：

```bash
# 运行演示脚本
cd /path/to/kava
python demo_visualization.py

# 输出：
# ✓ Generated training curves
# ✓ Generated evaluation results
# ✓ Generated experiment summary
# 🌐 Opening in browser...

# 会自动在浏览器打开一个完整的 HTML 报告
```

---

## 方案 1: 训练自动生成（推荐）

### 使用方法

```bash
# 提交任务（自动生成可视化）
sbatch scripts/train_with_visualization.sh

# 或者指定模型
sbatch --export=STUDENT="Qwen/Qwen2.5-1.5B",TEACHERS="Qwen/Qwen2.5-7B Qwen/Qwen2.5-14B" \
       scripts/train_with_visualization.sh
```

### 训练完成后

```bash
# 1. 查看结果摘要
cd outputs/experiment_20250113_143022  # 你的实验目录
./view_results.sh

# 2. 下载可视化到本地（在你的本地电脑上运行）
scp -r your_username@hpc_address:/path/to/outputs/experiment_*/visualizations ~/Downloads/

# 3. 在本地浏览器打开
open ~/Downloads/visualizations/experiment_summary.html  # macOS
# 或
start ~/Downloads/visualizations/experiment_summary.html  # Windows
# 或
xdg-open ~/Downloads/visualizations/experiment_summary.html  # Linux
```

### 生成的内容

训练完成后，你将得到：

```
outputs/experiment_20250113_143022/
├── visualizations/
│   ├── experiment_summary.html         # 📊 综合报告（主要看这个）
│   ├── training_curves.html            # 📈 训练曲线
│   ├── evaluation_results.html         # 📉 评测结果
│   ├── routing_weights.html            # 🔀 路由权重（如果使用）
│   ├── experiment_summary.png          # 对应的 PNG 图片
│   ├── training_curves.png
│   └── ...
├── best_model/                         # 最佳模型
├── eval_results.json                   # 评测数据
├── training.log                        # 训练日志
└── view_results.sh                     # 快速查看脚本
```

---

## 方案 2: 手动生成可视化

### 场景 1: 训练已完成，想生成可视化

```bash
# 进入实验目录
cd outputs/your_experiment_dir

# 生成综合报告
python visualization/hpc_visualizer.py \
    --mode summary \
    --input . \
    --output_dir ./visualizations \
    --output_name experiment_summary

# 下载并查看
scp -r $USER@hpc:$(pwd)/visualizations ~/Downloads/
```

### 场景 2: 只想看训练曲线

```bash
# 如果有 TensorBoard 日志
python visualization/hpc_visualizer.py \
    --mode training \
    --input ./logs/events.out.tfevents.* \
    --output_dir ./visualizations

# 如果有 JSON 日志
python visualization/hpc_visualizer.py \
    --mode training \
    --input ./training_log.json \
    --output_dir ./visualizations
```

### 场景 3: 对比多个模型

```bash
# 对比三个模型的评测结果
python visualization/hpc_visualizer.py \
    --mode eval \
    --input \
        outputs/model1/eval_results.json \
        outputs/model2/eval_results.json \
        outputs/model3/eval_results.json \
    --labels "Stage1" "Stage2" "Stage3" \
    --output_dir ./comparison \
    --output_name three_stage_comparison
```

---

## 📊 HTML 报告示例

### experiment_summary.html 包含：

1. **训练曲线**
   - Train/Eval Loss
   - KV Loss
   - Learning Rate
   - Gradient Norm

2. **评测结果表格**
   - GSM8K: 75.3%
   - MATH500: 42.1%
   - BBH: 68.5%
   - GPQA: 35.2%
   - TruthfulQA: 52.8%
   - CMMLU: 63.4%
   - C-Eval: 61.9%
   - **Average: 57.0%**

3. **路由权重分布**（如果使用可学习路由）
   - 权重随训练步数变化
   - 最终权重分布

4. **元数据**
   - 生成时间
   - 实验配置
   - 模型信息

### 样式特点

- ✅ 完全独立的 HTML 文件（无需网络连接）
- ✅ 响应式设计（自适应屏幕大小）
- ✅ 高分辨率图片（150 DPI）
- ✅ 专业的表格和图表样式
- ✅ 可直接打印或导出 PDF

---

## 🔧 高级用法

### 1. 在 SLURM 脚本中集成

```bash
# 在你的训练脚本末尾添加
python visualization/hpc_visualizer.py \
    --mode summary \
    --input $OUTPUT_DIR \
    --output_dir $OUTPUT_DIR/visualizations
```

### 2. 定期生成中间可视化

```bash
# 每 1000 步生成一次可视化
for step in 1000 2000 3000 4000 5000; do
    python visualization/hpc_visualizer.py \
        --mode training \
        --input ./logs \
        --output_dir ./visualizations_step_${step}
done
```

### 3. 批量处理多个实验

```bash
# 为所有实验生成可视化
for exp_dir in outputs/experiment_*/; do
    echo "Processing $exp_dir"
    python visualization/hpc_visualizer.py \
        --mode summary \
        --input "$exp_dir" \
        --output_dir "${exp_dir}/visualizations"
done

# 创建总对比
python visualization/hpc_visualizer.py \
    --mode eval \
    --input outputs/experiment_*/eval_results.json \
    --labels "Exp1" "Exp2" "Exp3" \
    --output_dir ./all_experiments_comparison
```

---

## 📥 下载方法对比

### 方法 1: SCP（推荐）

```bash
# 单个实验
scp -r username@hpc:/path/to/outputs/experiment_*/visualizations ~/Downloads/

# 多个实验
scp -r username@hpc:/path/to/outputs/*/visualizations ~/Downloads/all_viz/
```

### 方法 2: rsync（增量同步）

```bash
# 只下载新文件
rsync -avz --progress \
    username@hpc:/path/to/outputs/experiment_*/visualizations/ \
    ~/Downloads/visualizations/
```

### 方法 3: FileZilla/WinSCP（图形界面）

1. 连接到 HPC
2. 导航到 `outputs/experiment_*/visualizations/`
3. 拖拽到本地文件夹
4. 双击 `.html` 文件打开

### 方法 4: 直接打包下载

```bash
# 在 HPC 上打包
cd outputs
tar -czf visualizations_all.tar.gz experiment_*/visualizations

# 在本地下载
scp username@hpc:/path/to/outputs/visualizations_all.tar.gz ~/Downloads/

# 解压
cd ~/Downloads
tar -xzf visualizations_all.tar.gz
```

---

## ⚠️ 注意事项

### 1. 磁盘空间

```bash
# 检查可视化文件大小
du -sh outputs/*/visualizations

# 典型大小：
# - HTML: 50-200 KB per file
# - PNG: 500 KB - 2 MB per file
# - 完整实验: 5-20 MB
```

### 2. 权限问题

```bash
# 如果下载失败，检查权限
ls -la outputs/experiment_*/visualizations

# 修复权限
chmod -R 755 outputs/experiment_*/visualizations
```

### 3. 网络传输

```bash
# 如果文件很大，先压缩
cd outputs/experiment_20250113/
zip -r visualizations.zip visualizations/

# 下载压缩文件
scp username@hpc:/path/to/visualizations.zip ~/Downloads/
```

---

## 🎨 自定义可视化

### 修改样式

编辑 `visualization/hpc_visualizer.py` 中的 `_create_html_report` 方法：

```python
# 修改颜色主题
th {{
    background-color: #4CAF50;  # 改为你喜欢的颜色
}}

# 修改字体
body {{
    font-family: 'Helvetica', 'Arial', sans-serif;
}}
```

### 添加自定义图表

```python
from visualization.hpc_visualizer import HPCVisualizer

visualizer = HPCVisualizer(output_dir="./my_viz")

# 自定义绘图
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
plt.savefig("./my_viz/custom_plot.png")

# 添加到报告
visualizer._create_html_report(
    title="My Custom Report",
    images=["./my_viz/custom_plot.png"],
    data={},
    output_name="custom_report"
)
```

---

## 💡 最佳实践

### 1. 训练时自动生成

```bash
# 总是使用 train_with_visualization.sh
sbatch scripts/train_with_visualization.sh
```

### 2. 定期备份到本地

```bash
# 每天同步一次
rsync -avz --progress username@hpc:/path/to/outputs ~/backups/hpc_outputs
```

### 3. 创建实验日志

```bash
# 在实验目录创建 README
cat > outputs/experiment_*/README.md << EOF
# Experiment: $(date)

## Configuration
- Student: Qwen/Qwen2.5-1.5B
- Teachers: Qwen2.5-7B, Qwen2.5-14B
- Dataset: GSM8K + SVAMP + StrategyQA

## Results
See visualizations/experiment_summary.html

## Notes
- Used fixed routing (0.5/0.5)
- Training time: 12 hours
- Best eval loss: 0.42
EOF
```

---

## 🆘 故障排除

### 问题 1: matplotlib 错误

```bash
# 确保使用 Agg 后端
export MPLBACKEND=Agg
python visualization/hpc_visualizer.py ...
```

### 问题 2: HTML 中图片不显示

```bash
# 检查图片路径（应该是相对路径）
ls -la visualizations/*.png

# 确保 HTML 和 PNG 在同一目录
tree visualizations/
```

### 问题 3: 日志文件格式不对

```bash
# 手动创建 JSON 格式日志
python -c "
import json
logs = {
    'step': [100, 200, 300],
    'train_loss': [1.5, 1.2, 0.9],
}
with open('training_log.json', 'w') as f:
    json.dump(logs, f)
"
```

---

## 📚 相关文档

- `visualization/hpc_visualizer.py` - 可视化工具源代码
- `scripts/train_with_visualization.sh` - 训练+可视化脚本
- `LARGE_SCALE_EXPERIMENT_GUIDE.md` - 完整实验指南

---

## ✅ 快速检查清单

训练前：
- [ ] 已创建 `visualizations/` 目录
- [ ] 已设置 `ENABLE_VISUALIZATION=true`
- [ ] 已安装 matplotlib, seaborn

训练中：
- [ ] 日志正常输出到 `training.log`
- [ ] TensorBoard 记录正常

训练后：
- [ ] 检查 `visualizations/` 目录存在
- [ ] 下载 HTML 文件到本地
- [ ] 在浏览器中打开并验证
- [ ] 备份重要的可视化结果

完美！🎉
