"""
多随机种子训练与统计显著性测试

硬性控制：
1. 每个实验配置至少运行 3 个不同的随机种子
2. 计算 mean ± std
3. 进行配对 t-test 验证统计显著性
4. 提供 bootstrap 置信区间

用法：
    # 1. 运行多个随机种子
    python scripts/run_multi_seed_experiments.sh --config baseline_sft --seeds 42,43,44
    
    # 2. 计算统计显著性
    python utils/statistical_significance.py \
        --baseline_dir baselines/single_teacher \
        --experimental_dir outputs/multi_teacher \
        --output_dir stats_results
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class ExperimentResult:
    """单次实验结果"""
    seed: int
    metrics: Dict[str, float]  # {dataset: accuracy}
    metadata: Dict[str, any] = None


@dataclass
class StatisticalTestResult:
    """统计检验结果"""
    baseline_mean: float
    baseline_std: float
    experimental_mean: float
    experimental_std: float
    mean_difference: float
    t_statistic: float
    p_value: float
    is_significant: bool  # p < 0.05
    effect_size: float  # Cohen's d
    ci_lower: float  # 95% CI 下界
    ci_upper: float  # 95% CI 上界


class MultiSeedAggregator:
    """
    多随机种子结果聚合与统计分析
    """
    
    def __init__(self, output_dir: str = "./statistical_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_experiment_results(
        self, 
        experiment_dir: str,
        seeds: Optional[List[int]] = None
    ) -> List[ExperimentResult]:
        """
        加载多个随机种子的实验结果
        
        Args:
            experiment_dir: 实验目录
            seeds: 随机种子列表（如 [42, 43, 44]）
        
        Returns:
            实验结果列表
        """
        experiment_dir = Path(experiment_dir)
        results = []
        
        # 自动检测随机种子
        if seeds is None:
            seeds = []
            for seed_dir in experiment_dir.glob("seed_*"):
                if seed_dir.is_dir():
                    seed = int(seed_dir.name.split("_")[1])
                    seeds.append(seed)
            seeds.sort()
        
        print(f"加载实验结果: {experiment_dir.name}")
        print(f"  检测到 {len(seeds)} 个随机种子: {seeds}")
        
        for seed in seeds:
            seed_dir = experiment_dir / f"seed_{seed}"
            result_file = seed_dir / "evaluation_results.json"
            
            if not result_file.exists():
                print(f"  警告: 未找到 {result_file}")
                continue
            
            with open(result_file, 'r') as f:
                metrics = json.load(f)
            
            results.append(ExperimentResult(
                seed=seed,
                metrics=metrics.get("results", {}),
                metadata=metrics.get("metadata", {})
            ))
        
        print(f"  成功加载 {len(results)} 个结果\n")
        return results
    
    def compute_statistics(
        self,
        results: List[ExperimentResult],
        dataset_name: str
    ) -> Tuple[float, float, List[float]]:
        """
        计算某个数据集上的统计量
        
        Returns:
            (mean, std, values)
        """
        values = [r.metrics[dataset_name] for r in results]
        return np.mean(values), np.std(values, ddof=1), values
    
    def paired_t_test(
        self,
        baseline_results: List[ExperimentResult],
        experimental_results: List[ExperimentResult],
        dataset_name: str,
        alpha: float = 0.05
    ) -> StatisticalTestResult:
        """
        配对 t 检验
        
        Args:
            baseline_results: 基线实验结果
            experimental_results: 实验组结果
            dataset_name: 数据集名称
            alpha: 显著性水平（默认 0.05）
        
        Returns:
            统计检验结果
        """
        # 提取数值
        baseline_mean, baseline_std, baseline_values = self.compute_statistics(
            baseline_results, dataset_name
        )
        exp_mean, exp_std, exp_values = self.compute_statistics(
            experimental_results, dataset_name
        )
        
        # 确保样本数量相同
        assert len(baseline_values) == len(exp_values), \
            f"样本数量不匹配: {len(baseline_values)} vs {len(exp_values)}"
        
        # 配对 t 检验
        t_stat, p_value = stats.ttest_rel(exp_values, baseline_values)
        
        # 效应量（Cohen's d）
        pooled_std = np.sqrt((baseline_std**2 + exp_std**2) / 2)
        effect_size = (exp_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0
        
        # Bootstrap 置信区间
        ci_lower, ci_upper = self.bootstrap_ci(
            exp_values, baseline_values, n_bootstrap=10000
        )
        
        return StatisticalTestResult(
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            experimental_mean=exp_mean,
            experimental_std=exp_std,
            mean_difference=exp_mean - baseline_mean,
            t_statistic=t_stat,
            p_value=p_value,
            is_significant=(p_value < alpha),
            effect_size=effect_size,
            ci_lower=ci_lower,
            ci_upper=ci_upper
        )
    
    def bootstrap_ci(
        self,
        experimental_values: List[float],
        baseline_values: List[float],
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """
        Bootstrap 置信区间估计
        
        Returns:
            (ci_lower, ci_upper)
        """
        differences = []
        n = len(experimental_values)
        
        for _ in range(n_bootstrap):
            # 有放回抽样
            indices = np.random.choice(n, n, replace=True)
            exp_sample = [experimental_values[i] for i in indices]
            base_sample = [baseline_values[i] for i in indices]
            
            diff = np.mean(exp_sample) - np.mean(base_sample)
            differences.append(diff)
        
        # 计算置信区间
        alpha = 1 - confidence_level
        ci_lower = np.percentile(differences, alpha/2 * 100)
        ci_upper = np.percentile(differences, (1 - alpha/2) * 100)
        
        return ci_lower, ci_upper
    
    def compare_all_datasets(
        self,
        baseline_results: List[ExperimentResult],
        experimental_results: List[ExperimentResult],
        dataset_names: Optional[List[str]] = None
    ) -> Dict[str, StatisticalTestResult]:
        """
        对所有数据集进行统计检验
        
        Returns:
            {dataset_name: StatisticalTestResult}
        """
        if dataset_names is None:
            # 自动检测数据集
            dataset_names = list(baseline_results[0].metrics.keys())
        
        results = {}
        for dataset in dataset_names:
            try:
                results[dataset] = self.paired_t_test(
                    baseline_results, experimental_results, dataset
                )
            except Exception as e:
                print(f"警告: 数据集 {dataset} 统计检验失败: {e}")
        
        return results
    
    def print_statistical_report(
        self,
        comparison_results: Dict[str, StatisticalTestResult],
        baseline_name: str = "Baseline",
        experimental_name: str = "Experimental"
    ):
        """打印统计报告"""
        print(f"\n{'='*100}")
        print(f"统计显著性检验报告: {baseline_name} vs {experimental_name}")
        print(f"{'='*100}\n")
        
        # 表头
        print(f"{'数据集':<20} {'基线':<20} {'实验组':<20} {'差异':<15} "
              f"{'t统计量':<12} {'p值':<10} {'显著':<8} {'效应量':<10}")
        print(f"{'-'*100}")
        
        # 统计计数
        total_datasets = len(comparison_results)
        significant_count = 0
        positive_count = 0
        
        for dataset, result in comparison_results.items():
            # 格式化输出
            baseline_str = f"{result.baseline_mean:.2f}±{result.baseline_std:.2f}"
            exp_str = f"{result.experimental_mean:.2f}±{result.experimental_std:.2f}"
            diff_str = f"{result.mean_difference:+.2f}"
            sig_str = "✓" if result.is_significant else "✗"
            
            # 效应量解释
            if abs(result.effect_size) < 0.2:
                effect_str = f"{result.effect_size:.3f} (小)"
            elif abs(result.effect_size) < 0.5:
                effect_str = f"{result.effect_size:.3f} (中)"
            else:
                effect_str = f"{result.effect_size:.3f} (大)"
            
            print(f"{dataset:<20} {baseline_str:<20} {exp_str:<20} {diff_str:<15} "
                  f"{result.t_statistic:<12.3f} {result.p_value:<10.4f} {sig_str:<8} {effect_str:<10}")
            
            if result.is_significant:
                significant_count += 1
            if result.mean_difference > 0:
                positive_count += 1
        
        print(f"{'-'*100}\n")
        
        # 总结
        print(f"总结:")
        print(f"  • 总数据集数: {total_datasets}")
        print(f"  • 显著提升 (p<0.05): {significant_count} / {total_datasets} "
              f"({significant_count/total_datasets*100:.1f}%)")
        print(f"  • 正向改进: {positive_count} / {total_datasets} "
              f"({positive_count/total_datasets*100:.1f}%)")
        
        # 计算宏平均
        avg_diff = np.mean([r.mean_difference for r in comparison_results.values()])
        avg_p_value = np.mean([r.p_value for r in comparison_results.values()])
        
        print(f"\n宏平均:")
        print(f"  • 平均改进: {avg_diff:+.2f}%")
        print(f"  • 平均 p 值: {avg_p_value:.4f}")
        
        # 置信区间
        print(f"\n95% 置信区间:")
        for dataset, result in list(comparison_results.items())[:3]:  # 显示前3个
            print(f"  • {dataset}: [{result.ci_lower:.2f}, {result.ci_upper:.2f}]")
        
        print(f"\n{'='*100}\n")
    
    def plot_comparison_with_error_bars(
        self,
        baseline_results: List[ExperimentResult],
        experimental_results: List[ExperimentResult],
        dataset_names: Optional[List[str]] = None,
        baseline_name: str = "Baseline",
        experimental_name: str = "Experimental",
        save_path: Optional[str] = None
    ):
        """绘制带误差棒的对比图"""
        if dataset_names is None:
            dataset_names = list(baseline_results[0].metrics.keys())
        
        # 计算统计量
        baseline_means = []
        baseline_stds = []
        exp_means = []
        exp_stds = []
        
        for dataset in dataset_names:
            b_mean, b_std, _ = self.compute_statistics(baseline_results, dataset)
            e_mean, e_std, _ = self.compute_statistics(experimental_results, dataset)
            
            baseline_means.append(b_mean)
            baseline_stds.append(b_std)
            exp_means.append(e_mean)
            exp_stds.append(e_std)
        
        # 绘图
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(dataset_names))
        width = 0.35
        
        ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds,
               label=baseline_name, alpha=0.8, capsize=5)
        ax.bar(x + width/2, exp_means, width, yerr=exp_stds,
               label=experimental_name, alpha=0.8, capsize=5)
        
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Statistical Comparison with Error Bars (mean ± std)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 对比图已保存: {save_path}")
        
        return fig
    
    def plot_improvement_heatmap(
        self,
        comparison_results: Dict[str, StatisticalTestResult],
        save_path: Optional[str] = None
    ):
        """绘制改进热力图（带显著性标记）"""
        datasets = list(comparison_results.keys())
        improvements = [r.mean_difference for r in comparison_results.values()]
        p_values = [r.p_value for r in comparison_results.values()]
        
        # 创建热力图数据
        data = np.array(improvements).reshape(-1, 1)
        
        fig, ax = plt.subplots(figsize=(6, max(8, len(datasets) * 0.4)))
        
        # 绘制热力图
        sns.heatmap(
            data,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0,
            yticklabels=datasets,
            xticklabels=['Improvement (%)'],
            cbar_kws={'label': 'Performance Change (%)'},
            ax=ax
        )
        
        # 添加显著性标记
        for i, (dataset, result) in enumerate(comparison_results.items()):
            if result.is_significant:
                ax.text(0.5, i + 0.7, '***', ha='center', va='center',
                       fontsize=20, color='black', fontweight='bold')
        
        ax.set_title('Performance Improvement with Statistical Significance\n(*** p < 0.05)',
                    fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 热力图已保存: {save_path}")
        
        return fig
    
    def save_statistical_results(
        self,
        comparison_results: Dict[str, StatisticalTestResult],
        filename: str = "statistical_results.json"
    ):
        """保存统计结果到 JSON"""
        output = {}
        for dataset, result in comparison_results.items():
            output[dataset] = asdict(result)
        
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 统计结果已保存: {output_path}")


def run_statistical_analysis(
    baseline_dir: str,
    experimental_dir: str,
    output_dir: str = "./statistical_analysis",
    baseline_name: str = "Baseline",
    experimental_name: str = "Experimental",
    seeds: Optional[List[int]] = None
):
    """
    运行完整的统计分析流程
    
    Args:
        baseline_dir: 基线实验目录
        experimental_dir: 实验组目录
        output_dir: 输出目录
        baseline_name: 基线名称
        experimental_name: 实验组名称
        seeds: 随机种子列表
    """
    aggregator = MultiSeedAggregator(output_dir)
    
    # 1. 加载结果
    print("Step 1: 加载实验结果")
    baseline_results = aggregator.load_experiment_results(baseline_dir, seeds)
    experimental_results = aggregator.load_experiment_results(experimental_dir, seeds)
    
    if len(baseline_results) < 3 or len(experimental_results) < 3:
        print("警告: 每组至少需要 3 个随机种子才能进行可靠的统计检验！")
    
    # 2. 统计检验
    print("Step 2: 进行统计检验")
    comparison_results = aggregator.compare_all_datasets(
        baseline_results, experimental_results
    )
    
    # 3. 打印报告
    print("Step 3: 生成统计报告")
    aggregator.print_statistical_report(
        comparison_results, baseline_name, experimental_name
    )
    
    # 4. 可视化
    print("Step 4: 生成可视化")
    
    # 误差棒图
    aggregator.plot_comparison_with_error_bars(
        baseline_results,
        experimental_results,
        baseline_name=baseline_name,
        experimental_name=experimental_name,
        save_path=aggregator.output_dir / "comparison_with_error_bars.png"
    )
    
    # 热力图
    aggregator.plot_improvement_heatmap(
        comparison_results,
        save_path=aggregator.output_dir / "improvement_heatmap.png"
    )
    
    # 5. 保存结果
    print("Step 5: 保存统计结果")
    aggregator.save_statistical_results(comparison_results)
    
    print(f"\n✓ 统计分析完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="多随机种子统计显著性测试")
    parser.add_argument("--baseline_dir", type=str, required=True,
                       help="基线实验目录")
    parser.add_argument("--experimental_dir", type=str, required=True,
                       help="实验组目录")
    parser.add_argument("--output_dir", type=str, default="./statistical_analysis",
                       help="输出目录")
    parser.add_argument("--baseline_name", type=str, default="Baseline",
                       help="基线名称")
    parser.add_argument("--experimental_name", type=str, default="Experimental",
                       help="实验组名称")
    parser.add_argument("--seeds", type=str, default=None,
                       help="随机种子列表（逗号分隔，如 42,43,44）")
    
    args = parser.parse_args()
    
    seeds = None
    if args.seeds:
        seeds = [int(s) for s in args.seeds.split(',')]
    
    run_statistical_analysis(
        baseline_dir=args.baseline_dir,
        experimental_dir=args.experimental_dir,
        output_dir=args.output_dir,
        baseline_name=args.baseline_name,
        experimental_name=args.experimental_name,
        seeds=seeds
    )
