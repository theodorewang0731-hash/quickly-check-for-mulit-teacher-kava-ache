"""
HPC 友好的可视化工具
支持在无显示器的 HPC 环境生成 HTML/PNG 报告
"""

import matplotlib
matplotlib.use('Agg')  # 无显示器后端
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime


class HPCVisualizer:
    """HPC 环境可视化器"""
    
    def __init__(self, output_dir: str = "./visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置绘图风格
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def plot_training_curves(
        self,
        log_file: str,
        output_name: str = "training_curves",
    ) -> str:
        """
        绘制训练曲线（从 TensorBoard 日志或 JSON 日志）
        包含：Loss、KV Loss、学习率、梯度范数、双轴图（KV Loss + 任务指标）
        
        Returns:
            HTML 文件路径
        """
        # 解析日志
        logs = self._parse_log_file(log_file)
        
        if not logs:
            print(f"✗ No logs found in {log_file}")
            return None
        
        # 创建图表（增加到 3x2）
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # 1. Total Loss 曲线（Train + Eval）
        if 'train_loss' in logs:
            axes[0, 0].plot(logs['step'], logs['train_loss'], label='Train Loss', linewidth=2, alpha=0.8)
        if 'eval_loss' in logs:
            axes[0, 0].plot(logs['step'], logs['eval_loss'], label='Eval Loss', linewidth=2, marker='o', markersize=3)
        axes[0, 0].set_xlabel('Steps', fontsize=11)
        axes[0, 0].set_ylabel('Loss', fontsize=11)
        axes[0, 0].set_title('Training and Evaluation Loss', fontsize=13, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. KV Loss 曲线（Train + Eval）
        if 'kv_loss' in logs:
            axes[0, 1].plot(logs['step'], logs['kv_loss'], label='Train KV Loss', color='orange', linewidth=2, alpha=0.8)
        if 'eval_kv_loss' in logs:
            axes[0, 1].plot(logs['step'], logs['eval_kv_loss'], label='Eval KV Loss', color='red', linewidth=2, marker='o', markersize=3)
        axes[0, 1].set_xlabel('Steps', fontsize=11)
        axes[0, 1].set_ylabel('KV Loss', fontsize=11)
        axes[0, 1].set_title('KV Distillation Loss', fontsize=13, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. CE Loss 曲线（如果有）
        if 'ce_loss' in logs:
            axes[1, 0].plot(logs['step'], logs['ce_loss'], label='Train CE Loss', color='blue', linewidth=2, alpha=0.8)
        if 'eval_ce_loss' in logs:
            axes[1, 0].plot(logs['step'], logs['eval_ce_loss'], label='Eval CE Loss', color='navy', linewidth=2, marker='o', markersize=3)
        axes[1, 0].set_xlabel('Steps', fontsize=11)
        axes[1, 0].set_ylabel('CE Loss', fontsize=11)
        axes[1, 0].set_title('Cross-Entropy Loss', fontsize=13, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 学习率曲线
        if 'learning_rate' in logs:
            axes[1, 1].plot(logs['step'], logs['learning_rate'], color='green', linewidth=2)
            axes[1, 1].set_xlabel('Steps', fontsize=11)
            axes[1, 1].set_ylabel('Learning Rate', fontsize=11)
            axes[1, 1].set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 5. 双轴图：KV Loss (左) + 任务准确率 (右) ⭐⭐⭐
        ax_left = axes[2, 0]
        ax_right = ax_left.twinx()
        
        # 左轴：KV Loss（红色）
        if 'eval_kv_loss' in logs:
            color = 'tab:red'
            ax_left.set_xlabel('Training Steps', fontsize=11)
            ax_left.set_ylabel('KV Loss', color=color, fontsize=11)
            ax_left.plot(logs['step'], logs['eval_kv_loss'], color=color, marker='o', 
                        markersize=4, linewidth=2, label='KV Loss')
            ax_left.tick_params(axis='y', labelcolor=color)
            ax_left.grid(True, alpha=0.3)
        
        # 右轴：任务准确率（蓝色）
        task_metrics = [k for k in logs.keys() if k.startswith('eval_') and 'acc' in k.lower()]
        if task_metrics:
            color = 'tab:blue'
            ax_right.set_ylabel('Task Accuracy (%)', color=color, fontsize=11)
            
            # 计算平均准确率（如果有多个任务）
            if len(task_metrics) > 1:
                avg_acc = np.mean([logs[m] for m in task_metrics], axis=0)
                ax_right.plot(logs['step'], avg_acc, color=color, marker='s', 
                            markersize=4, linewidth=2, label='Avg Task Accuracy')
            else:
                ax_right.plot(logs['step'], logs[task_metrics[0]], color=color, marker='s',
                            markersize=4, linewidth=2, label='Task Accuracy')
            
            ax_right.tick_params(axis='y', labelcolor=color)
        
        axes[2, 0].set_title('KV Loss vs Task Performance ⭐\n(Proves alignment improves task quality)', 
                            fontsize=13, fontweight='bold')
        
        # 图例
        lines1, labels1 = ax_left.get_legend_handles_labels()
        lines2, labels2 = ax_right.get_legend_handles_labels() if task_metrics else ([], [])
        ax_left.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        
        # 6. 梯度范数
        if 'grad_norm' in logs:
            axes[2, 1].plot(logs['step'], logs['grad_norm'], color='purple', linewidth=2)
            axes[2, 1].set_xlabel('Steps', fontsize=11)
            axes[2, 1].set_ylabel('Gradient Norm', fontsize=11)
            axes[2, 1].set_title('Gradient Norm', fontsize=13, fontweight='bold')
            axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存 PNG
        png_path = self.output_dir / f"{output_name}.png"
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 生成 HTML
        html_path = self._create_html_report(
            title="Training Curves",
            images=[str(png_path)],
            data=logs,
            output_name=output_name,
        )
        
        print(f"✓ Training curves saved to: {html_path}")
        return str(html_path)
    
    def plot_kv_loss_heatmap(
        self,
        kv_loss_by_layer_file: str,
        output_name: str = "kv_loss_heatmap",
    ) -> str:
        """
        绘制 KV Loss 热力图（按层、按时间步）
        
        Args:
            kv_loss_by_layer_file: KV loss 按层记录的文件（JSON 格式）
                格式: {"steps": [...], "layers": [...], "kv_losses": [[step1_layer_losses], [step2_layer_losses], ...]}
        """
        with open(kv_loss_by_layer_file) as f:
            data = json.load(f)
        
        steps = data.get('steps', [])
        layers = data.get('layers', [])
        kv_losses = np.array(data.get('kv_losses', []))  # [num_steps, num_layers]
        
        if kv_losses.size == 0:
            print("✗ No KV loss data found")
            return None
        
        # 创建热力图
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))
        
        # 1. KV Loss 热力图 (层 × 时间)
        im = axes[0].imshow(kv_losses.T, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        axes[0].set_xlabel('Training Steps', fontsize=12)
        axes[0].set_ylabel('Layer Index', fontsize=12)
        axes[0].set_title('KV Loss Heatmap (Layer × Time)', fontsize=14, fontweight='bold')
        
        # 设置刻度
        step_ticks = np.linspace(0, len(steps)-1, min(10, len(steps)), dtype=int)
        axes[0].set_xticks(step_ticks)
        axes[0].set_xticklabels([steps[i] for i in step_ticks])
        
        layer_ticks = np.linspace(0, len(layers)-1, min(10, len(layers)), dtype=int)
        axes[0].set_yticks(layer_ticks)
        axes[0].set_yticklabels([layers[i] for i in layer_ticks])
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=axes[0])
        cbar.set_label('KV Loss', fontsize=11)
        
        # 2. 每层的平均 KV Loss
        avg_loss_per_layer = np.mean(kv_losses, axis=0)
        axes[1].bar(range(len(layers)), avg_loss_per_layer, color='coral', alpha=0.7)
        axes[1].set_xlabel('Layer Index', fontsize=12)
        axes[1].set_ylabel('Average KV Loss', fontsize=12)
        axes[1].set_title('Average KV Loss per Layer', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # 标注最高和最低的层
        max_idx = np.argmax(avg_loss_per_layer)
        min_idx = np.argmin(avg_loss_per_layer)
        axes[1].text(max_idx, avg_loss_per_layer[max_idx], f'Max: {avg_loss_per_layer[max_idx]:.3f}',
                    ha='center', va='bottom', fontweight='bold')
        axes[1].text(min_idx, avg_loss_per_layer[min_idx], f'Min: {avg_loss_per_layer[min_idx]:.3f}',
                    ha='center', va='top', fontweight='bold')
        
        plt.tight_layout()
        
        # 保存
        png_path = self.output_dir / f"{output_name}.png"
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        html_path = self._create_html_report(
            title="KV Loss Heatmap Analysis",
            images=[str(png_path)],
            data=data,
            output_name=output_name,
        )
        
        print(f"✓ KV loss heatmap saved to: {html_path}")
        return str(html_path)
    
    def plot_routing_alpha_heatmap(
        self,
        routing_alpha_file: str,
        output_name: str = "routing_alpha_heatmap",
    ) -> str:
        """
        绘制路由权重 α 热力图（按层、按教师）
        
        Args:
            routing_alpha_file: 路由权重记录文件（JSON 格式）
                格式: {
                    "layers": [...],
                    "teachers": [...],
                    "alpha_by_layer": [[layer0_teacher_alphas], [layer1_teacher_alphas], ...],
                    "alpha_by_task": {"task1": [[layer_teacher_alphas]], "task2": ...}  # 可选
                }
        """
        with open(routing_alpha_file) as f:
            data = json.load(f)
        
        layers = data.get('layers', [])
        teachers = data.get('teachers', [])
        alpha_by_layer = np.array(data.get('alpha_by_layer', []))  # [num_layers, num_teachers]
        alpha_by_task = data.get('alpha_by_task', {})
        
        if alpha_by_layer.size == 0:
            print("✗ No routing alpha data found")
            return None
        
        # 计算需要的子图数量
        num_plots = 1 + len(alpha_by_task)
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 6*num_plots))
        if num_plots == 1:
            axes = [axes]
        
        # 1. 主热力图：层 × 教师
        im = axes[0].imshow(alpha_by_layer, aspect='auto', cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[0].set_xlabel('Teacher Index', fontsize=12)
        axes[0].set_ylabel('Layer Index', fontsize=12)
        axes[0].set_title('Routing Weights α (Layer × Teacher)\n"Shallow layers prefer small teachers, deep layers prefer large teachers"',
                         fontsize=14, fontweight='bold')
        
        # 设置刻度
        axes[0].set_xticks(range(len(teachers)))
        axes[0].set_xticklabels(teachers, rotation=45, ha='right')
        axes[0].set_yticks(range(len(layers)))
        axes[0].set_yticklabels(layers)
        
        # 添加数值标注
        for i in range(len(layers)):
            for j in range(len(teachers)):
                text = axes[0].text(j, i, f'{alpha_by_layer[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=8)
        
        # 颜色条
        cbar = plt.colorbar(im, ax=axes[0])
        cbar.set_label('Routing Weight α', fontsize=11)
        
        # 2. 按任务的路由权重热力图（如果有）
        for idx, (task_name, task_alpha) in enumerate(alpha_by_task.items(), start=1):
            task_alpha = np.array(task_alpha)
            
            im_task = axes[idx].imshow(task_alpha, aspect='auto', cmap='YlGnBu', vmin=0, vmax=1)
            axes[idx].set_xlabel('Teacher Index', fontsize=12)
            axes[idx].set_ylabel('Layer Index', fontsize=12)
            axes[idx].set_title(f'Routing Weights α for Task: {task_name}',
                               fontsize=13, fontweight='bold')
            
            axes[idx].set_xticks(range(len(teachers)))
            axes[idx].set_xticklabels(teachers, rotation=45, ha='right')
            axes[idx].set_yticks(range(len(layers)))
            axes[idx].set_yticklabels(layers)
            
            # 添加数值标注
            for i in range(task_alpha.shape[0]):
                for j in range(task_alpha.shape[1]):
                    text = axes[idx].text(j, i, f'{task_alpha[i, j]:.2f}',
                                         ha="center", va="center", color="black", fontsize=8)
            
            cbar_task = plt.colorbar(im_task, ax=axes[idx])
            cbar_task.set_label('Routing Weight α', fontsize=11)
        
        plt.tight_layout()
        
        # 保存
        png_path = self.output_dir / f"{output_name}.png"
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        html_path = self._create_html_report(
            title="Routing Alpha Heatmap Analysis",
            images=[str(png_path)],
            data=data,
            output_name=output_name,
        )
        
        print(f"✓ Routing alpha heatmap saved to: {html_path}")
        return str(html_path)
    
    def plot_routing_weights(
        self,
        weights_file: str,
        output_name: str = "routing_weights",
    ) -> str:
        """
        可视化路由权重分布
        
        Args:
            weights_file: 权重日志文件（JSON 格式）
        """
        with open(weights_file) as f:
            weights_data = json.load(f)
        
        num_teachers = len(weights_data.get('teacher_names', []))
        if num_teachers == 0:
            print("✗ No teacher weights found")
            return None
        
        # 提取权重历史
        steps = weights_data.get('steps', [])
        weights_history = weights_data.get('weights', [])  # [num_steps, num_teachers]
        
        # 创建图表
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 1. 权重随时间变化
        for i in range(num_teachers):
            teacher_weights = [w[i] for w in weights_history]
            axes[0].plot(steps, teacher_weights, label=f'Teacher {i+1}', linewidth=2)
        
        axes[0].set_xlabel('Training Steps')
        axes[0].set_ylabel('Routing Weight')
        axes[0].set_title('Routing Weights Over Training')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1])
        
        # 2. 最终权重分布
        final_weights = weights_history[-1] if weights_history else [1.0/num_teachers] * num_teachers
        colors = sns.color_palette("husl", num_teachers)
        axes[1].bar(range(num_teachers), final_weights, color=colors, alpha=0.7)
        axes[1].set_xlabel('Teacher Index')
        axes[1].set_ylabel('Final Weight')
        axes[1].set_title('Final Routing Weights Distribution')
        axes[1].set_xticks(range(num_teachers))
        axes[1].set_xticklabels([f'T{i+1}' for i in range(num_teachers)])
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # 保存
        png_path = self.output_dir / f"{output_name}.png"
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        html_path = self._create_html_report(
            title="Routing Weights Analysis",
            images=[str(png_path)],
            data=weights_data,
            output_name=output_name,
        )
        
        print(f"✓ Routing weights visualization saved to: {html_path}")
        return str(html_path)
    
    def plot_evaluation_results(
        self,
        eval_files: List[str],
        labels: List[str],
        output_name: str = "evaluation_comparison",
    ) -> str:
        """
        对比多个模型的评测结果
        
        Args:
            eval_files: 评测结果文件列表（JSON）
            labels: 每个模型的标签
        """
        # 加载所有评测结果
        all_results = {}
        for file, label in zip(eval_files, labels):
            with open(file) as f:
                all_results[label] = json.load(f)
        
        # 提取数据集名称
        datasets = set()
        for results in all_results.values():
            datasets.update(k for k in results.keys() if k != 'average')
        datasets = sorted(datasets)
        
        # 创建对比图
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(datasets))
        width = 0.8 / len(labels)
        
        for i, label in enumerate(labels):
            scores = [all_results[label].get(ds, {}).get('score', 0) for ds in datasets]
            ax.bar(x + i * width, scores, width, label=label, alpha=0.8)
        
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Score (%)')
        ax.set_title('Multi-Task Evaluation Comparison')
        ax.set_xticks(x + width * (len(labels) - 1) / 2)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # 保存
        png_path = self.output_dir / f"{output_name}.png"
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # 创建对比表格
        comparison_table = self._create_comparison_table(all_results, datasets)
        
        html_path = self._create_html_report(
            title="Evaluation Results Comparison",
            images=[str(png_path)],
            data={"comparison": comparison_table},
            output_name=output_name,
        )
        
        print(f"✓ Evaluation comparison saved to: {html_path}")
        return str(html_path)
    
    def create_experiment_summary(
        self,
        experiment_dir: str,
        output_name: str = "experiment_summary",
    ) -> str:
        """
        创建完整的实验总结报告
        
        Args:
            experiment_dir: 实验输出目录（包含日志、评测结果等）
        """
        exp_dir = Path(experiment_dir)
        
        # 收集所有相关文件
        training_log = self._find_file(exp_dir, "training_log.json")
        eval_results = self._find_file(exp_dir, "eval_results.json")
        routing_weights = self._find_file(exp_dir, "routing_weights.json")
        
        images = []
        
        # 1. 训练曲线
        if training_log:
            img = self.plot_training_curves(str(training_log), f"{output_name}_training")
            if img:
                images.append(img)
        
        # 2. 路由权重
        if routing_weights:
            img = self.plot_routing_weights(str(routing_weights), f"{output_name}_routing")
            if img:
                images.append(img)
        
        # 3. 评测结果
        if eval_results:
            with open(eval_results) as f:
                eval_data = json.load(f)
            
            # 创建评测结果表格
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.axis('tight')
            ax.axis('off')
            
            table_data = []
            for dataset, result in eval_data.items():
                if dataset != 'average':
                    table_data.append([
                        dataset,
                        result.get('metric', 'N/A'),
                        f"{result.get('score', 0):.2f}%",
                        result.get('num_examples', 'N/A'),
                    ])
            
            table = ax.table(
                cellText=table_data,
                colLabels=['Dataset', 'Metric', 'Score', 'Examples'],
                loc='center',
                cellLoc='left',
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            png_path = self.output_dir / f"{output_name}_eval_table.png"
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            images.append(str(png_path))
        
        # 生成综合 HTML 报告
        html_path = self._create_comprehensive_html(
            title=f"Experiment Summary: {exp_dir.name}",
            images=images,
            experiment_dir=str(exp_dir),
            output_name=output_name,
        )
        
        print(f"✓ Experiment summary saved to: {html_path}")
        return str(html_path)
    
    def _parse_log_file(self, log_file: str) -> Dict:
        """解析训练日志"""
        log_path = Path(log_file)
        
        if not log_path.exists():
            return {}
        
        # 尝试 JSON 格式
        if log_path.suffix == '.json':
            with open(log_path) as f:
                return json.load(f)
        
        # 尝试 TensorBoard 格式（简化）
        # 这里可以添加 TensorBoard 日志解析
        return {}
    
    def _find_file(self, directory: Path, pattern: str) -> Optional[Path]:
        """查找文件"""
        files = list(directory.glob(f"**/{pattern}"))
        return files[0] if files else None
    
    def _create_comparison_table(self, all_results: Dict, datasets: List[str]) -> str:
        """创建对比表格 HTML"""
        html = '<table border="1" style="border-collapse: collapse; width: 100%;">\n'
        html += '<tr><th>Dataset</th>'
        
        for label in all_results.keys():
            html += f'<th>{label}</th>'
        html += '</tr>\n'
        
        for dataset in datasets:
            html += f'<tr><td>{dataset}</td>'
            for label in all_results.keys():
                score = all_results[label].get(dataset, {}).get('score', 0)
                html += f'<td>{score:.2f}%</td>'
            html += '</tr>\n'
        
        html += '</table>'
        return html
    
    def _create_html_report(
        self,
        title: str,
        images: List[str],
        data: Dict,
        output_name: str,
    ) -> str:
        """创建自包含的 HTML 报告（图片嵌入为 base64）"""
        html_path = self.output_dir / f"{output_name}.html"
        
        # 将图片转换为 base64
        import base64
        embedded_images = []
        for img_path in images:
            if Path(img_path).exists():
                with open(img_path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                    img_ext = Path(img_path).suffix[1:]  # 去掉 '.'
                    embedded_images.append(f"data:image/{img_ext};base64,{img_data}")
            else:
                print(f"⚠ Image not found: {img_path}")
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        img {{
            max-width: 100%;
            height: auto;
            margin: 20px 0;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .metadata {{
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .timestamp {{
            color: #888;
            font-size: 0.9em;
        }}
        .file-path {{
            background-color: #e8f5e9;
            padding: 10px;
            border-left: 4px solid #4CAF50;
            font-family: monospace;
            word-break: break-all;
            margin: 20px 0;
        }}
        .copy-btn {{
            background-color: #4CAF50;
            color: white;
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin: 5px;
        }}
        .copy-btn:hover {{
            background-color: #45a049;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .success {{
            color: #4CAF50;
            font-weight: bold;
        }}
    </style>
    <script>
        function copyPath(elementId) {{
            const text = document.getElementById(elementId).textContent;
            navigator.clipboard.writeText(text).then(function() {{
                alert('✓ Path copied to clipboard!\\n\\nYou can now paste it in your terminal or file explorer.');
            }}, function(err) {{
                alert('Failed to copy: ' + err);
            }});
        }}
    </script>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="metadata">
            <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p class="success">✓ This is a self-contained HTML file. You can open it anywhere!</p>
        </div>
        
        <h2>📍 File Location</h2>
        <div class="file-path" id="file-path">{html_path.absolute()}</div>
        <button class="copy-btn" onclick="copyPath('file-path')">📋 Copy Path</button>
        
        <h2>📊 Visualizations</h2>
"""
        
        # 添加嵌入的图片
        for img_data in embedded_images:
            html_content += f'<div><img src="{img_data}" alt="Visualization"></div>\n'
        
        # 添加数据表格
        if 'comparison' in data:
            html_content += f'<h2>📈 Detailed Results</h2>\n{data["comparison"]}\n'
        
        html_content += """
        <h2>💡 How to Use</h2>
        <ol>
            <li><strong>On HPC:</strong> Copy the file path above</li>
            <li><strong>Download:</strong> <code>scp your_username@hpc:[paste_path] ~/Downloads/</code></li>
            <li><strong>Open:</strong> Double-click the downloaded HTML file</li>
        </ol>
        
        <p style="color: #888; font-size: 0.9em; margin-top: 40px;">
            Generated by KaVa Multi-Teacher KV Distillation Framework
        </p>
    </div>
</body>
</html>
"""
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(html_path)
    
    def _create_comprehensive_html(
        self,
        title: str,
        images: List[str],
        experiment_dir: str,
        output_name: str,
    ) -> str:
        """创建综合报告"""
        return self._create_html_report(
            title=title,
            images=images,
            data={},
            output_name=output_name,
        )


def main():
    """示例用法"""
    import argparse
    
    parser = argparse.ArgumentParser(description="HPC-friendly visualization tool")
    parser.add_argument("--mode", choices=["training", "routing", "eval", "summary"], required=True)
    parser.add_argument("--input", nargs="+", required=True, help="Input file(s)")
    parser.add_argument("--output_dir", default="./visualizations")
    parser.add_argument("--output_name", default="visualization")
    parser.add_argument("--labels", nargs="+", help="Labels for comparison (eval mode)")
    
    args = parser.parse_args()
    
    visualizer = HPCVisualizer(output_dir=args.output_dir)
    
    if args.mode == "training":
        visualizer.plot_training_curves(args.input[0], args.output_name)
    
    elif args.mode == "routing":
        visualizer.plot_routing_weights(args.input[0], args.output_name)
    
    elif args.mode == "eval":
        labels = args.labels or [f"Model {i+1}" for i in range(len(args.input))]
        visualizer.plot_evaluation_results(args.input, labels, args.output_name)
    
    elif args.mode == "summary":
        visualizer.create_experiment_summary(args.input[0], args.output_name)


if __name__ == "__main__":
    main()
