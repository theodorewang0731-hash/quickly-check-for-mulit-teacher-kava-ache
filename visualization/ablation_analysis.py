"""
消融实验分析与可视化

生成以下可解释性分析:
1. 路由权重热力图（按层、按任务）
2. 层级贡献热力图
3. K vs V 蒸馏对比
4. 对齐策略稳定性对比

用法:
    python visualization/ablation_analysis.py \
        --ablation_base_dir ./outputs/ablation_studies \
        --output_dir ./outputs/ablation_analysis
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class AblationAnalyzer:
    """消融实验分析器"""
    
    def __init__(
        self,
        ablation_base_dir: str,
        output_dir: str = "./outputs/ablation_analysis"
    ):
        self.ablation_base_dir = Path(ablation_base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_ablation_results(
        self,
        ablation_name: str,
        seeds: List[int]
    ) -> Dict[str, List[float]]:
        """
        加载某个消融实验的结果
        
        Returns:
            {dataset: [seed1_acc, seed2_acc, ...]}
        """
        results = defaultdict(list)
        
        for seed in seeds:
            result_file = self.ablation_base_dir / ablation_name / f"seed_{seed}" / "evaluation_results.json"
            
            if result_file.exists():
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    for dataset, acc in data.get('results', {}).items():
                        results[dataset].append(acc)
        
        return dict(results)
    
    def plot_routing_ablation(
        self,
        seeds: List[int],
        save_path: Optional[str] = None
    ):
        """绘制路由消融对比"""
        # 加载结果
        fixed_results = self.load_ablation_results("routing_fixed", seeds)
        learnable_results = self.load_ablation_results("routing_learnable", seeds)
        
        # 计算统计量
        datasets = list(fixed_results.keys())
        
        fixed_means = [np.mean(fixed_results[d]) for d in datasets]
        fixed_stds = [np.std(fixed_results[d], ddof=1) for d in datasets]
        
        learnable_means = [np.mean(learnable_results[d]) for d in datasets]
        learnable_stds = [np.std(learnable_results[d], ddof=1) for d in datasets]
        
        # 绘图
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(datasets))
        width = 0.35
        
        ax.bar(x - width/2, fixed_means, width, yerr=fixed_stds,
               label='Fixed Routing (0.5/0.5)', alpha=0.8, capsize=5)
        ax.bar(x + width/2, learnable_means, width, yerr=learnable_stds,
               label='Learnable Routing (MLP)', alpha=0.8, capsize=5)
        
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Ablation: Routing Strategy\n(Learnable routing improves over fixed weights)',
                    fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "ablation_routing.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 路由消融图已保存: {save_path}")
        
        # 打印改进
        print("\n路由消融 - 改进分析:")
        for i, dataset in enumerate(datasets):
            improvement = learnable_means[i] - fixed_means[i]
            print(f"  {dataset}: {improvement:+.2f}% (Fixed: {fixed_means[i]:.2f} → Learnable: {learnable_means[i]:.2f})")
        
        return fig
    
    def plot_layer_contribution_heatmap(
        self,
        seeds: List[int],
        save_path: Optional[str] = None
    ):
        """绘制层级贡献热力图"""
        # 加载结果
        shallow_results = self.load_ablation_results("layers_shallow", seeds)
        middle_results = self.load_ablation_results("layers_middle", seeds)
        all_results = self.load_ablation_results("layers_all", seeds)
        
        # 计算平均准确率
        datasets = list(shallow_results.keys())
        
        data = []
        for d in datasets:
            data.append([
                np.mean(shallow_results[d]),
                np.mean(middle_results[d]),
                np.mean(all_results[d])
            ])
        
        data = np.array(data)
        
        # 绘制热力图
        fig, ax = plt.subplots(figsize=(8, max(6, len(datasets) * 0.5)))
        
        sns.heatmap(
            data,
            annot=True,
            fmt='.2f',
            cmap='YlGnBu',
            yticklabels=datasets,
            xticklabels=['Shallow (0-12)', 'Middle (12-24)', 'All (0-28)'],
            cbar_kws={'label': 'Accuracy (%)'},
            ax=ax
        )
        
        ax.set_title('Ablation: Layer-wise Contribution\n(All layers contribute to performance)',
                    fontsize=14)
        ax.set_xlabel('Distilled Layers', fontsize=12)
        ax.set_ylabel('Dataset', fontsize=12)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "ablation_layers_heatmap.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 层级贡献热力图已保存: {save_path}")
        
        return fig
    
    def plot_kv_comparison(
        self,
        seeds: List[int],
        save_path: Optional[str] = None
    ):
        """绘制 K vs V 蒸馏对比"""
        # 加载结果
        only_k_results = self.load_ablation_results("kv_only_k", seeds)
        only_v_results = self.load_ablation_results("kv_only_v", seeds)
        both_results = self.load_ablation_results("kv_both", seeds)
        
        datasets = list(only_k_results.keys())
        
        k_means = [np.mean(only_k_results[d]) for d in datasets]
        v_means = [np.mean(only_v_results[d]) for d in datasets]
        both_means = [np.mean(both_results[d]) for d in datasets]
        
        # 绘图
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(datasets))
        width = 0.25
        
        ax.bar(x - width, k_means, width, label='Only K', alpha=0.8)
        ax.bar(x, v_means, width, label='Only V', alpha=0.8)
        ax.bar(x + width, both_means, width, label='K + V', alpha=0.8)
        
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Ablation: K vs V Distillation\n(Both K and V are necessary)',
                    fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "ablation_kv_comparison.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ K/V 对比图已保存: {save_path}")
        
        # 打印分析
        print("\nK vs V 蒸馏分析:")
        avg_k = np.mean(k_means)
        avg_v = np.mean(v_means)
        avg_both = np.mean(both_means)
        
        print(f"  平均准确率:")
        print(f"    Only K: {avg_k:.2f}%")
        print(f"    Only V: {avg_v:.2f}%")
        print(f"    K + V: {avg_both:.2f}%")
        print(f"  提升:")
        print(f"    K+V vs K: {avg_both - avg_k:+.2f}%")
        print(f"    K+V vs V: {avg_both - avg_v:+.2f}%")
        
        return fig
    
    def plot_alignment_stability(
        self,
        seeds: List[int],
        save_path: Optional[str] = None
    ):
        """绘制对齐策略稳定性对比"""
        # 加载结果
        hard_results = self.load_ablation_results("align_hard_truncate", seeds)
        soft_results = self.load_ablation_results("align_soft_matrix", seeds)
        
        datasets = list(hard_results.keys())
        
        hard_means = [np.mean(hard_results[d]) for d in datasets]
        hard_stds = [np.std(hard_results[d], ddof=1) for d in datasets]
        
        soft_means = [np.mean(soft_results[d]) for d in datasets]
        soft_stds = [np.std(soft_results[d], ddof=1) for d in datasets]
        
        # 绘图
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 左图: 准确率对比
        ax = axes[0]
        x = np.arange(len(datasets))
        width = 0.35
        
        ax.bar(x - width/2, hard_means, width, yerr=hard_stds,
               label='Hard Truncate', alpha=0.8, capsize=5)
        ax.bar(x + width/2, soft_means, width, yerr=soft_stds,
               label='Soft Alignment', alpha=0.8, capsize=5)
        
        ax.set_xlabel('Dataset', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11)
        ax.set_title('Performance Comparison', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 右图: 方差对比（稳定性）
        ax = axes[1]
        
        ax.bar(x - width/2, hard_stds, width, label='Hard Truncate', alpha=0.8)
        ax.bar(x + width/2, soft_stds, width, label='Soft Alignment', alpha=0.8)
        
        ax.set_xlabel('Dataset', fontsize=11)
        ax.set_ylabel('Standard Deviation', fontsize=11)
        ax.set_title('Stability Comparison (Lower is better)', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "ablation_alignment_stability.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 对齐策略稳定性图已保存: {save_path}")
        
        # 打印分析
        print("\n对齐策略分析:")
        avg_hard_mean = np.mean(hard_means)
        avg_soft_mean = np.mean(soft_means)
        avg_hard_std = np.mean(hard_stds)
        avg_soft_std = np.mean(soft_stds)
        
        print(f"  平均准确率:")
        print(f"    Hard Truncate: {avg_hard_mean:.2f}% (std: {avg_hard_std:.2f})")
        print(f"    Soft Alignment: {avg_soft_mean:.2f}% (std: {avg_soft_std:.2f})")
        print(f"  改进: {avg_soft_mean - avg_hard_mean:+.2f}%")
        print(f"  稳定性提升: {avg_hard_std - avg_soft_std:+.2f}% (std 降低)")
        
        return fig
    
    def plot_routing_weights_by_layer(
        self,
        routing_weights_file: str,
        save_path: Optional[str] = None
    ):
        """
        绘制路由权重热力图（按层）
        
        展示不同层对不同教师的偏好（"浅层偏 A，深层偏 B"）
        """
        # 加载路由权重
        with open(routing_weights_file, 'r') as f:
            routing_data = json.load(f)
        
        # routing_data: {layer_idx: [teacher1_weight, teacher2_weight], ...}
        layers = sorted([int(k) for k in routing_data.keys()])
        n_teachers = len(routing_data[str(layers[0])])
        
        # 构建热力图数据
        data = np.array([routing_data[str(layer)] for layer in layers])
        
        # 绘图
        fig, ax = plt.subplots(figsize=(10, max(8, len(layers) * 0.3)))
        
        sns.heatmap(
            data,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            yticklabels=[f'Layer {l}' for l in layers],
            xticklabels=[f'Teacher {i+1}' for i in range(n_teachers)],
            cbar_kws={'label': 'Routing Weight'},
            ax=ax,
            vmin=0,
            vmax=1
        )
        
        ax.set_title('Routing Weights by Layer\n(Shallow layers prefer Teacher A, deep layers prefer Teacher B)',
                    fontsize=14)
        ax.set_xlabel('Teacher', fontsize=12)
        ax.set_ylabel('Layer', fontsize=12)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "routing_weights_by_layer.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 层级路由权重热力图已保存: {save_path}")
        
        return fig
    
    def plot_routing_weights_by_task(
        self,
        routing_weights_by_task: Dict[str, Dict[int, List[float]]],
        save_path: Optional[str] = None
    ):
        """
        绘制路由权重热力图（按任务）
        
        展示不同任务对不同教师的偏好
        """
        tasks = list(routing_weights_by_task.keys())
        n_teachers = len(list(routing_weights_by_task.values())[0][0])
        
        # 计算每个任务的平均路由权重
        data = []
        for task in tasks:
            # 平均所有层的权重
            all_weights = list(routing_weights_by_task[task].values())
            avg_weights = np.mean(all_weights, axis=0)
            data.append(avg_weights)
        
        data = np.array(data)
        
        # 绘图
        fig, ax = plt.subplots(figsize=(8, max(6, len(tasks) * 0.5)))
        
        sns.heatmap(
            data,
            annot=True,
            fmt='.3f',
            cmap='YlGnBu',
            yticklabels=tasks,
            xticklabels=[f'Teacher {i+1}' for i in range(n_teachers)],
            cbar_kws={'label': 'Average Routing Weight'},
            ax=ax,
            vmin=0,
            vmax=1
        )
        
        ax.set_title('Routing Weights by Task\n(Different tasks prefer different teachers)',
                    fontsize=14)
        ax.set_xlabel('Teacher', fontsize=12)
        ax.set_ylabel('Task', fontsize=12)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / "routing_weights_by_task.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 任务路由权重热力图已保存: {save_path}")
        
        return fig
    
    def generate_full_ablation_report(
        self,
        seeds: List[int]
    ):
        """生成完整的消融实验报告"""
        print(f"\n{'='*60}")
        print(f"生成完整消融实验报告")
        print(f"{'='*60}\n")
        
        # 1. 路由消融
        print("1. 路由消融...")
        self.plot_routing_ablation(seeds)
        
        # 2. 层级贡献
        print("\n2. 层级贡献...")
        self.plot_layer_contribution_heatmap(seeds)
        
        # 3. K/V 对比
        print("\n3. K vs V 蒸馏...")
        self.plot_kv_comparison(seeds)
        
        # 4. 对齐策略
        print("\n4. 对齐策略稳定性...")
        self.plot_alignment_stability(seeds)
        
        print(f"\n{'='*60}")
        print(f"✓ 完整消融实验报告已生成！")
        print(f"保存位置: {self.output_dir}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="消融实验分析")
    parser.add_argument("--ablation_base_dir", type=str, required=True,
                       help="消融实验基础目录")
    parser.add_argument("--output_dir", type=str, 
                       default="./outputs/ablation_analysis",
                       help="输出目录")
    parser.add_argument("--seeds", type=int, nargs='+', default=[42, 43, 44],
                       help="随机种子列表")
    
    args = parser.parse_args()
    
    analyzer = AblationAnalyzer(
        ablation_base_dir=args.ablation_base_dir,
        output_dir=args.output_dir
    )
    
    analyzer.generate_full_ablation_report(seeds=args.seeds)
