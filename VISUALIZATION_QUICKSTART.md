# 🚀 可视化快速参考

## 一分钟上手

### 1️⃣ 训练时启用可视化
```bash
sbatch scripts/train_with_visualization.sh
```

### 2️⃣ 训练完成后复制路径
训练日志末尾会显示：
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
/path/to/experiment_summary.html
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 3️⃣ 下载到本地（在你的电脑运行）
```bash
scp your_user@hpc:/path/to/experiment_summary.html ~/Downloads/report.html
```

### 4️⃣ 打开查看
```bash
# 双击文件，或运行：
open ~/Downloads/report.html      # macOS
start ~/Downloads/report.html     # Windows
xdg-open ~/Downloads/report.html  # Linux
```

---

## 📊 HTML 报告包含

- ✅ 训练曲线（Loss, KV Loss, Learning Rate）
- ✅ 评测结果（7 个数据集的完整分数）
- ✅ 路由权重（如果使用）
- ✅ 可复制的文件路径
- ✅ 所有图片嵌入（无需额外文件）

---

## 💡 关键特点

| 特性 | 说明 |
|------|------|
| 📦 自包含 | 单个 HTML 文件，包含所有图表 |
| 🚀 快速 | 只需下载 2-5MB |
| 🌐 离线 | 无需网络连接即可查看 |
| 📧 可分享 | 可通过邮件/云盘分享 |
| 🖨️ 可打印 | 直接打印或导出 PDF |
| 📱 响应式 | 手机/平板也能看 |

---

## 🆘 常见问题

### Q: 图片不显示？
**A**: 确保使用的是 `train_with_visualization.sh`，它会自动嵌入图片为 base64

### Q: 文件太大？
**A**: 正常！包含高清图表的 HTML 通常 2-5MB，这是正常的

### Q: 可以修改样式吗？
**A**: 可以！编辑 `visualization/hpc_visualizer.py` 中的 CSS 部分

### Q: 如何对比多个实验？
**A**: 
```bash
python visualization/hpc_visualizer.py \
    --mode eval \
    --input exp1/eval_results.json exp2/eval_results.json \
    --labels "Exp1" "Exp2"
```

### Q: 训练中途想看结果？
**A**: 在训练目录运行：
```bash
python visualization/hpc_visualizer.py --mode summary --input .
```

---

## 📞 快速命令

```bash
# 测试可视化功能
python demo_visualization.py

# 手动生成可视化（训练完成后）
python visualization/hpc_visualizer.py --mode summary --input ./outputs/exp_dir

# 显示下载信息
python visualization/show_report_info.py ./outputs/exp_dir

# 下载所有可视化
scp -r user@hpc:/path/to/outputs/*/visualizations ~/Downloads/all_viz/

# 批量生成（多个实验）
for dir in outputs/exp_*/; do
    python visualization/hpc_visualizer.py --mode summary --input "$dir"
done
```

---

## 🎯 最佳实践

1. **总是使用** `train_with_visualization.sh` 进行训练
2. **训练完成后立即下载** HTML 文件以防丢失
3. **重命名文件** 便于管理（如 `exp_qwen_20250113.html`）
4. **定期备份** 重要的可视化结果
5. **分享前检查** 确保没有敏感信息

---

## 📚 相关文档

- `HPC_VISUALIZATION_GUIDE.md` - 完整指南
- `visualization/hpc_visualizer.py` - 可视化工具
- `visualization/show_report_info.py` - 信息显示
- `demo_visualization.py` - 本地测试

---

**需要帮助？** 参考 `HPC_VISUALIZATION_GUIDE.md` 获取详细说明！
