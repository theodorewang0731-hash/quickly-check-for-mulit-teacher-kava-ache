"""
对比所有实验结果
生成完整的对比报告（基线组 + 实验组）
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from visualization.hpc_visualizer import HPCVisualizer
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def collect_all_results(baseline_dirs, experiment_dirs):
    """收集所有实验结果"""
    all_results = {}
    
    # 收集基线结果
    print("Collecting baseline results...")
    for baseline_dir in baseline_dirs:
        baseline_path = Path(baseline_dir)
        name = baseline_path.name
        
        # 查找 eval_results.json
        eval_file = baseline_path / "eval_results.json"
        if not eval_file.exists():
            eval_file = baseline_path
            if not eval_file.suffix == '.json':
                continue
        
        if eval_file.exists():
            with open(eval_file) as f:
                all_results[name] = json.load(f)
            print(f"  ✓ {name}")
    
    # 收集实验结果
    print("\nCollecting experiment results...")
    for exp_dir in experiment_dirs:
        exp_path = Path(exp_dir)
        name = exp_path.name
        
        eval_file = exp_path / "eval_results.json"
        if eval_file.exists():
            with open(eval_file) as f:
                all_results[name] = json.load(f)
            print(f"  ✓ {name}")
    
    return all_results


def create_comprehensive_comparison(all_results, output_dir):
    """创建综合对比图"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 提取数据
    models = list(all_results.keys())
    datasets = set()
    for result in all_results.values():
        datasets.update(k for k in result.keys() if k != 'average')
    datasets = sorted(datasets)
    
    # 创建数据矩阵
    data_matrix = []
    for model in models:
        scores = [all_results[model].get(ds, {}).get('score', 0) for ds in datasets]
        data_matrix.append(scores)
    
    # 1. 柱状图对比
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(datasets))
    width = 0.8 / len(models)
    colors = sns.color_palette("husl", len(models))
    
    for i, (model, scores) in enumerate(zip(models, data_matrix)):
        ax.bar(x + i * width, scores, width, label=model, alpha=0.8, color=colors[i])
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Multi-Task Evaluation: All Models Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    bar_chart_path = output_path / "comparison_bar_chart.png"
    plt.savefig(bar_chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. 雷达图对比
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(datasets), endpoint=False).tolist()
    angles += angles[:1]  # 闭合
    
    for model, scores in zip(models, data_matrix):
        scores_plot = scores + scores[:1]  # 闭合
        ax.plot(angles, scores_plot, 'o-', linewidth=2, label=model, alpha=0.7)
        ax.fill(angles, scores_plot, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title('Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    radar_chart_path = output_path / "comparison_radar_chart.png"
    plt.savefig(radar_chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. 热力图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    df = pd.DataFrame(data_matrix, index=models, columns=datasets)
    sns.heatmap(df, annot=True, fmt='.1f', cmap='RdYlGn', center=50,
                cbar_kws={'label': 'Score (%)'}, ax=ax)
    
    ax.set_title('Performance Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    
    plt.tight_layout()
    heatmap_path = output_path / "comparison_heatmap.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. 平均分对比
    fig, ax = plt.subplots(figsize=(10, 6))
    
    avg_scores = [all_results[model].get('average', np.mean([
        all_results[model].get(ds, {}).get('score', 0) for ds in datasets
    ])) for model in models]
    
    bars = ax.barh(models, avg_scores, color=colors, alpha=0.8)
    
    # 添加数值标签
    for i, (bar, score) in enumerate(zip(bars, avg_scores)):
        ax.text(score + 1, i, f'{score:.2f}%', va='center', fontsize=10)
    
    ax.set_xlabel('Average Score (%)', fontsize=12)
    ax.set_title('Average Performance Across All Datasets', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, max(avg_scores) * 1.15)
    
    plt.tight_layout()
    avg_chart_path = output_path / "comparison_average.png"
    plt.savefig(avg_chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return [bar_chart_path, radar_chart_path, heatmap_path, avg_chart_path]


def create_improvement_analysis(all_results, output_dir):
    """分析改进幅度"""
    output_path = Path(output_dir)
    
    # 假设第一个是基线
    models = list(all_results.keys())
    if len(models) < 2:
        return None
    
    baseline = models[0]
    datasets = [k for k in all_results[baseline].keys() if k != 'average']
    
    # 计算相对于基线的改进
    improvements = {}
    for model in models[1:]:
        improvements[model] = {}
        for dataset in datasets:
            baseline_score = all_results[baseline].get(dataset, {}).get('score', 0)
            model_score = all_results[model].get(dataset, {}).get('score', 0)
            improvement = model_score - baseline_score
            improvements[model][dataset] = improvement
    
    # 绘制改进图
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(datasets))
    width = 0.8 / (len(models) - 1)
    colors = sns.color_palette("Set2", len(models) - 1)
    
    for i, model in enumerate(models[1:]):
        improvements_list = [improvements[model][ds] for ds in datasets]
        ax.bar(x + i * width, improvements_list, width, label=model, alpha=0.8, color=colors[i])
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Improvement vs Baseline (%)', fontsize=12)
    ax.set_title(f'Performance Improvement Relative to {baseline}', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * (len(models) - 2) / 2)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    improvement_path = output_path / "improvement_analysis.png"
    plt.savefig(improvement_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return improvement_path


def main():
    parser = argparse.ArgumentParser(description="Compare all experiment results")
    parser.add_argument("--baseline_dirs", nargs="+", required=True,
                       help="Baseline result directories")
    parser.add_argument("--experiment_dirs", nargs="+", required=True,
                       help="Experiment result directories")
    parser.add_argument("--output_dir", default="./comparison",
                       help="Output directory")
    parser.add_argument("--output_name", default="final_comparison",
                       help="Output HTML filename")
    
    args = parser.parse_args()
    
    print("="*80)
    print("Comprehensive Experiment Comparison")
    print("="*80)
    
    # 收集结果
    all_results = collect_all_results(args.baseline_dirs, args.experiment_dirs)
    
    if not all_results:
        print("✗ No results found!")
        return
    
    print(f"\n✓ Collected results from {len(all_results)} models")
    
    # 生成对比图表
    print("\nGenerating comparison charts...")
    chart_paths = create_comprehensive_comparison(all_results, args.output_dir)
    print(f"✓ Generated {len(chart_paths)} comparison charts")
    
    # 生成改进分析
    print("\nGenerating improvement analysis...")
    improvement_path = create_improvement_analysis(all_results, args.output_dir)
    if improvement_path:
        chart_paths.append(improvement_path)
        print(f"✓ Generated improvement analysis")
    
    # 生成 HTML 报告
    print("\nGenerating HTML report...")
    visualizer = HPCVisualizer(output_dir=args.output_dir)
    
    # 创建对比表格
    comparison_html = '<table border="1" style="border-collapse: collapse; width: 100%;">\n'
    comparison_html += '<tr><th>Model</th>'
    
    datasets = set()
    for result in all_results.values():
        datasets.update(k for k in result.keys() if k != 'average')
    datasets = sorted(datasets)
    
    for dataset in datasets:
        comparison_html += f'<th>{dataset}</th>'
    comparison_html += '<th>Average</th></tr>\n'
    
    for model, result in all_results.items():
        comparison_html += f'<tr><td><strong>{model}</strong></td>'
        for dataset in datasets:
            score = result.get(dataset, {}).get('score', 0)
            comparison_html += f'<td>{score:.2f}%</td>'
        avg = result.get('average', 0)
        comparison_html += f'<td><strong>{avg:.2f}%</strong></td></tr>\n'
    
    comparison_html += '</table>'
    
    html_path = visualizer._create_html_report(
        title="Complete Experiment Comparison",
        images=[str(p) for p in chart_paths],
        data={"comparison": comparison_html},
        output_name=args.output_name,
    )
    
    print("\n" + "="*80)
    print("✓ Comparison Report Generated!")
    print("="*80)
    print(f"HTML Report: {html_path}")
    print(f"\nTo view:")
    print(f"  scp user@hpc:{html_path} ~/Downloads/")
    print(f"  open ~/Downloads/{args.output_name}.html")
    print("="*80)


if __name__ == "__main__":
    main()
