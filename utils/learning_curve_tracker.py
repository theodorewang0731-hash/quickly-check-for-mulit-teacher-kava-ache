"""
学习曲线追踪器 - KV Loss 与任务指标双曲线

硬性控制：
1. 同时追踪 KV-loss 和任务准确率
2. 分别记录 train 和 val 曲线
3. 证明不是过拟合或"只对齐不提质"

用法：
    # 训练时自动记录
    tracker = LearningCurveTracker(output_dir="./outputs/experiment")
    
    for step in training_loop:
        # 训练
        loss_dict = train_step(...)
        tracker.log_train(step, loss_dict)
        
        # 验证
        if step % eval_steps == 0:
            val_metrics = evaluate(...)
            tracker.log_val(step, val_metrics)
    
    # 生成可视化
    tracker.plot_all_curves()
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from collections import defaultdict


class LearningCurveTracker:
    """
    学习曲线追踪器
    
    同时追踪 KV-loss 和任务指标，证明模型既对齐了 KV 又提升了任务性能
    """
    
    def __init__(
        self,
        output_dir: str,
        experiment_name: str = "experiment"
    ):
        """
        Args:
            output_dir: 输出目录
            experiment_name: 实验名称
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        
        # 存储历史记录
        self.train_history = defaultdict(list)  # {metric: [(step, value), ...]}
        self.val_history = defaultdict(list)
        
        # 日志文件
        self.train_log_file = self.output_dir / "train_log.jsonl"
        self.val_log_file = self.output_dir / "val_log.jsonl"
        
    def log_train(
        self,
        step: int,
        metrics: Dict[str, float]
    ):
        """
        记录训练指标
        
        Args:
            step: 训练步数
            metrics: {
                'loss': float,
                'kv_loss': float,
                'ce_loss': float,
                'learning_rate': float,
                ...
            }
        """
        # 添加到历史
        for metric, value in metrics.items():
            self.train_history[metric].append((step, value))
        
        # 写入日志文件
        log_entry = {'step': step, 'type': 'train', **metrics}
        with open(self.train_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_val(
        self,
        step: int,
        metrics: Dict[str, float]
    ):
        """
        记录验证指标
        
        Args:
            step: 训练步数
            metrics: {
                'val_loss': float,
                'val_kv_loss': float,
                'val_ce_loss': float,
                'val_accuracy': float,
                'val_gsm8k': float,
                'val_math': float,
                ...
            }
        """
        # 添加到历史
        for metric, value in metrics.items():
            self.val_history[metric].append((step, value))
        
        # 写入日志文件
        log_entry = {'step': step, 'type': 'val', **metrics}
        with open(self.val_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def load_history(self):
        """从日志文件加载历史（用于恢复）"""
        # 加载训练历史
        if self.train_log_file.exists():
            with open(self.train_log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    step = entry.pop('step')
                    entry.pop('type', None)
                    for metric, value in entry.items():
                        self.train_history[metric].append((step, value))
        
        # 加载验证历史
        if self.val_log_file.exists():
            with open(self.val_log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    step = entry.pop('step')
                    entry.pop('type', None)
                    for metric, value in entry.items():
                        self.val_history[metric].append((step, value))
    
    def plot_kv_loss_curves(
        self,
        save_path: Optional[str] = None
    ):
        """绘制 KV Loss 曲线（train + val）"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图: KV Loss
        ax = axes[0]
        
        if 'kv_loss' in self.train_history:
            steps, values = zip(*self.train_history['kv_loss'])
            ax.plot(steps, values, label='Train KV Loss', alpha=0.7)
        
        if 'val_kv_loss' in self.val_history:
            steps, values = zip(*self.val_history['val_kv_loss'])
            ax.plot(steps, values, label='Val KV Loss', marker='o', markersize=4)
        
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('KV Loss', fontsize=12)
        ax.set_title('KV Distillation Loss (Train vs Val)', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 右图: CE Loss
        ax = axes[1]
        
        if 'ce_loss' in self.train_history:
            steps, values = zip(*self.train_history['ce_loss'])
            ax.plot(steps, values, label='Train CE Loss', alpha=0.7)
        
        if 'val_ce_loss' in self.val_history:
            steps, values = zip(*self.val_history['val_ce_loss'])
            ax.plot(steps, values, label='Val CE Loss', marker='o', markersize=4)
        
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('CE Loss', fontsize=12)
        ax.set_title('Cross-Entropy Loss (Train vs Val)', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "kv_loss_curves.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ KV Loss 曲线已保存: {save_path}")
        
        return fig
    
    def plot_task_accuracy_curves(
        self,
        task_metrics: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """
        绘制任务准确率曲线
        
        Args:
            task_metrics: 任务指标列表（如 ['val_gsm8k', 'val_math']）
        """
        if task_metrics is None:
            # 自动检测任务指标（以 val_ 开头且不是 loss）
            task_metrics = [
                k for k in self.val_history.keys()
                if k.startswith('val_') and 'loss' not in k
            ]
        
        if not task_metrics:
            print("警告: 未找到任务指标")
            return
        
        # 创建子图
        n_tasks = len(task_metrics)
        n_cols = min(3, n_tasks)
        n_rows = (n_tasks + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if n_tasks == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_tasks > 1 else [axes]
        
        for i, task_metric in enumerate(task_metrics):
            ax = axes[i]
            
            if task_metric in self.val_history:
                steps, values = zip(*self.val_history[task_metric])
                ax.plot(steps, values, marker='o', markersize=5, linewidth=2)
                
                # 添加趋势线
                z = np.polyfit(steps, values, 2)  # 二次拟合
                p = np.poly1d(z)
                ax.plot(steps, p(steps), '--', alpha=0.5, label='Trend')
            
            # 格式化任务名称
            task_name = task_metric.replace('val_', '').upper()
            ax.set_xlabel('Training Steps', fontsize=11)
            ax.set_ylabel('Accuracy (%)', fontsize=11)
            ax.set_title(f'{task_name} Performance', fontsize=13)
            ax.legend()
            ax.grid(alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_tasks, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "task_accuracy_curves.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 任务准确率曲线已保存: {save_path}")
        
        return fig
    
    def plot_dual_axis_curve(
        self,
        save_path: Optional[str] = None
    ):
        """
        绘制双轴曲线：KV Loss（左轴）+ 任务准确率（右轴）
        
        证明：KV Loss 下降的同时，任务准确率也在上升
        """
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # 左轴: KV Loss（红色）
        color = 'tab:red'
        ax1.set_xlabel('Training Steps', fontsize=13)
        ax1.set_ylabel('KV Loss', color=color, fontsize=13)
        
        if 'val_kv_loss' in self.val_history:
            steps, values = zip(*self.val_history['val_kv_loss'])
            ax1.plot(steps, values, color=color, marker='o', 
                    markersize=5, linewidth=2, label='Val KV Loss')
        
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(alpha=0.3)
        
        # 右轴: 任务准确率（蓝色）
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Task Accuracy (%)', color=color, fontsize=13)
        
        # 计算平均任务准确率
        task_metrics = [
            k for k in self.val_history.keys()
            if k.startswith('val_') and 'loss' not in k and 'accuracy' not in k.lower()
        ]
        
        if task_metrics:
            # 计算每个 step 的平均准确率
            all_steps = set()
            for metric in task_metrics:
                steps_list = [s for s, _ in self.val_history[metric]]
                all_steps.update(steps_list)
            
            all_steps = sorted(all_steps)
            avg_accuracies = []
            
            for step in all_steps:
                step_accuracies = []
                for metric in task_metrics:
                    # 找到这个 step 的值
                    for s, v in self.val_history[metric]:
                        if s == step:
                            step_accuracies.append(v)
                            break
                
                if step_accuracies:
                    avg_accuracies.append(np.mean(step_accuracies))
                else:
                    avg_accuracies.append(None)
            
            # 过滤 None
            valid_data = [(s, a) for s, a in zip(all_steps, avg_accuracies) if a is not None]
            if valid_data:
                steps, values = zip(*valid_data)
                ax2.plot(steps, values, color=color, marker='s', 
                        markersize=5, linewidth=2, label='Avg Task Accuracy')
        
        ax2.tick_params(axis='y', labelcolor=color)
        
        # 标题
        plt.title('KV Loss vs Task Performance\n(Proves alignment improves task quality)',
                 fontsize=14, pad=20)
        
        # 图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "dual_axis_curve.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 双轴曲线已保存: {save_path}")
        
        return fig
    
    def plot_overfitting_analysis(
        self,
        save_path: Optional[str] = None
    ):
        """
        绘制过拟合分析图
        
        展示 train loss 和 val loss 的差距
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图: Total Loss
        ax = axes[0]
        
        if 'loss' in self.train_history:
            steps, values = zip(*self.train_history['loss'])
            ax.plot(steps, values, label='Train Loss', alpha=0.7)
        
        if 'val_loss' in self.val_history:
            steps, values = zip(*self.val_history['val_loss'])
            ax.plot(steps, values, label='Val Loss', marker='o', markersize=4)
        
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Total Loss', fontsize=12)
        ax.set_title('Overfitting Analysis: Total Loss', fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 右图: Train-Val Gap
        ax = axes[1]
        
        if 'loss' in self.train_history and 'val_loss' in self.val_history:
            # 找到共同的 steps
            train_dict = dict(self.train_history['loss'])
            val_dict = dict(self.val_history['val_loss'])
            
            common_steps = sorted(set(train_dict.keys()) & set(val_dict.keys()))
            if common_steps:
                gaps = [val_dict[s] - train_dict[s] for s in common_steps]
                
                ax.plot(common_steps, gaps, marker='o', linewidth=2)
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='No gap')
                
                ax.set_xlabel('Training Steps', fontsize=12)
                ax.set_ylabel('Val Loss - Train Loss', fontsize=12)
                ax.set_title('Overfitting Gap (Lower is better)', fontsize=14)
                ax.legend()
                ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "overfitting_analysis.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 过拟合分析图已保存: {save_path}")
        
        return fig
    
    def plot_all_curves(self):
        """生成所有学习曲线"""
        print(f"\n{'='*60}")
        print(f"生成学习曲线: {self.experiment_name}")
        print(f"{'='*60}\n")
        
        # 1. KV Loss 曲线
        self.plot_kv_loss_curves()
        
        # 2. 任务准确率曲线
        self.plot_task_accuracy_curves()
        
        # 3. 双轴曲线（关键图：证明对齐 + 提质）
        self.plot_dual_axis_curve()
        
        # 4. 过拟合分析
        self.plot_overfitting_analysis()
        
        print(f"\n✓ 所有学习曲线已生成！保存在: {self.output_dir}")
    
    def export_summary(self) -> Dict:
        """导出摘要统计"""
        summary = {
            'experiment_name': self.experiment_name,
            'total_train_steps': max([s for s, _ in self.train_history.get('loss', [(0, 0)])]),
            'final_train_loss': None,
            'final_val_loss': None,
            'final_kv_loss': None,
            'final_task_accuracy': None,
            'best_val_loss': None,
            'best_task_accuracy': None
        }
        
        # 最终训练损失
        if 'loss' in self.train_history:
            summary['final_train_loss'] = self.train_history['loss'][-1][1]
        
        # 最终验证损失
        if 'val_loss' in self.val_history:
            summary['final_val_loss'] = self.val_history['val_loss'][-1][1]
            summary['best_val_loss'] = min(v for _, v in self.val_history['val_loss'])
        
        # 最终 KV 损失
        if 'val_kv_loss' in self.val_history:
            summary['final_kv_loss'] = self.val_history['val_kv_loss'][-1][1]
        
        # 任务准确率
        task_metrics = [
            k for k in self.val_history.keys()
            if k.startswith('val_') and 'loss' not in k
        ]
        
        if task_metrics:
            # 计算最终平均准确率
            final_accuracies = [self.val_history[m][-1][1] for m in task_metrics]
            summary['final_task_accuracy'] = np.mean(final_accuracies)
            
            # 计算最佳准确率
            best_accuracies = [max(v for _, v in self.val_history[m]) for m in task_metrics]
            summary['best_task_accuracy'] = np.mean(best_accuracies)
        
        # 保存摘要
        summary_file = self.output_dir / "learning_curve_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ 学习曲线摘要已保存: {summary_file}")
        
        return summary


if __name__ == "__main__":
    # 示例：模拟训练过程
    print("创建示例学习曲线...\n")
    
    tracker = LearningCurveTracker(
        output_dir="./demo_learning_curves",
        experiment_name="multi_teacher_demo"
    )
    
    # 模拟训练过程
    np.random.seed(42)
    
    for step in range(0, 1000, 10):
        # 训练指标（KV loss 逐渐下降）
        kv_loss = 2.0 * np.exp(-step / 500) + 0.1 + np.random.normal(0, 0.05)
        ce_loss = 1.5 * np.exp(-step / 400) + 0.2 + np.random.normal(0, 0.03)
        total_loss = kv_loss + ce_loss
        
        tracker.log_train(step, {
            'loss': total_loss,
            'kv_loss': kv_loss,
            'ce_loss': ce_loss,
            'learning_rate': 2e-5 * (1 - step / 1000)
        })
        
        # 验证指标（每 100 步）
        if step % 100 == 0:
            val_kv_loss = kv_loss + np.random.normal(0.1, 0.02)
            val_ce_loss = ce_loss + np.random.normal(0.05, 0.01)
            
            # 任务准确率逐渐上升
            base_acc = 50 + 25 * (1 - np.exp(-step / 300))
            
            tracker.log_val(step, {
                'val_loss': val_kv_loss + val_ce_loss,
                'val_kv_loss': val_kv_loss,
                'val_ce_loss': val_ce_loss,
                'val_gsm8k': base_acc + np.random.normal(0, 2),
                'val_math': base_acc - 5 + np.random.normal(0, 2),
                'val_bbh': base_acc + 3 + np.random.normal(0, 2)
            })
    
    # 生成所有曲线
    tracker.plot_all_curves()
    
    # 导出摘要
    summary = tracker.export_summary()
    
    print("\n学习曲线摘要:")
    print(json.dumps(summary, indent=2))
