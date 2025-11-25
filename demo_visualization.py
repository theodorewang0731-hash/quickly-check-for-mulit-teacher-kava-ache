"""
演示可视化功能 - 快速测试
生成示例 HTML 并显示下载命令
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from visualization.hpc_visualizer import HPCVisualizer
from visualization.show_report_info import print_visualization_info, create_download_script
import json
import numpy as np


def create_demo_report():
    """创建演示报告"""
    print("="*80)
    print("KaVa Visualization Demo")
    print("="*80)
    print("\nStep 1: Creating mock experiment data...")
    
    # 创建输出目录
    output_dir = Path("./demo_experiment")
    output_dir.mkdir(exist_ok=True)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # 模拟训练日志
    training_log = {
        'step': list(range(0, 5000, 100)),
        'train_loss': [2.5 - i * 0.0003 + np.random.random() * 0.1 for i in range(50)],
        'eval_loss': [2.3 - i * 0.00025 + np.random.random() * 0.15 for i in range(50)],
        'kv_loss': [1.2 - i * 0.00015 + np.random.random() * 0.05 for i in range(50)],
        'learning_rate': [2e-5 * (1 - i/50) for i in range(50)],
        'grad_norm': [1.5 - i * 0.02 + np.random.random() * 0.3 for i in range(50)],
    }
    
    log_file = output_dir / "training_log.json"
    with open(log_file, 'w') as f:
        json.dump(training_log, f)
    
    print(f"✓ Created training log: {log_file}")
    
    # 模拟评测结果
    eval_results = {
        'gsm8k_test': {'score': 75.3, 'metric': 'exact_match', 'num_examples': 1319},
        'math500': {'score': 42.1, 'metric': 'exact_match', 'num_examples': 500},
        'bbh': {'score': 68.5, 'metric': 'exact_match', 'num_examples': 1000},
        'gpqa': {'score': 35.2, 'metric': 'accuracy', 'num_examples': 448},
        'truthfulqa': {'score': 52.8, 'metric': 'accuracy', 'num_examples': 817},
        'cmmlu_subset': {'score': 63.4, 'metric': 'accuracy', 'num_examples': 1000},
        'ceval_subset': {'score': 61.9, 'metric': 'accuracy', 'num_examples': 1000},
        'average': 57.0,
    }
    
    eval_file = output_dir / "eval_results.json"
    with open(eval_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"✓ Created eval results: {eval_file}")
    
    # 生成可视化
    print("\nStep 2: Generating visualizations...")
    visualizer = HPCVisualizer(output_dir=str(viz_dir))
    
    # 训练曲线
    visualizer.plot_training_curves(str(log_file), "training_curves")
    print("✓ Generated training curves")
    
    # 评测结果
    visualizer.plot_evaluation_results([str(eval_file)], ["Demo Model"], "evaluation_results")
    print("✓ Generated evaluation results")
    
    # 综合报告
    visualizer.create_experiment_summary(str(output_dir), "experiment_summary")
    print("✓ Generated experiment summary")
    
    print("\nStep 3: Displaying download information...")
    print("")
    
    # 显示下载信息
    print_visualization_info(str(output_dir))
    
    # 创建下载脚本
    download_script = create_download_script(str(output_dir))
    
    # 最终提示
    print("\n" + "="*80)
    print("✓ Demo Complete!")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  • Main report:     {viz_dir / 'experiment_summary.html'}")
    print(f"  • Training curves: {viz_dir / 'training_curves.html'}")
    print(f"  • Eval results:    {viz_dir / 'evaluation_results.html'}")
    print(f"  • Download script: {download_script}")
    
    print(f"\n💡 To test locally (since this is not on HPC):")
    print(f"   Simply open: {viz_dir / 'experiment_summary.html'}")
    print(f"\n   On Windows: start {viz_dir / 'experiment_summary.html'}")
    print(f"   On macOS:   open {viz_dir / 'experiment_summary.html'}")
    print(f"   On Linux:   xdg-open {viz_dir / 'experiment_summary.html'}")
    
    print("\n" + "="*80)
    
    return viz_dir / 'experiment_summary.html'


if __name__ == "__main__":
    html_path = create_demo_report()
    
    # 尝试自动打开
    import platform
    import subprocess
    
    print("\n🌐 Attempting to open in browser...")
    try:
        if platform.system() == 'Windows':
            subprocess.run(['start', str(html_path)], shell=True, check=True)
        elif platform.system() == 'Darwin':  # macOS
            subprocess.run(['open', str(html_path)], check=True)
        else:  # Linux
            subprocess.run(['xdg-open', str(html_path)], check=True)
        print("✓ Opened in browser!")
    except Exception as e:
        print(f"⚠ Could not auto-open: {e}")
        print(f"Please manually open: {html_path}")
