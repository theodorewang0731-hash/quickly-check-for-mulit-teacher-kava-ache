"""
测试 HPC 可视化工具
创建模拟数据并生成示例 HTML 报告
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from visualization.hpc_visualizer import HPCVisualizer
import json
import numpy as np

def create_mock_data():
    """创建模拟数据"""
    print("Creating mock data...")
    
    # 模拟训练日志
    training_log = {
        'step': list(range(0, 5000, 100)),
        'train_loss': [2.5 - i * 0.0003 + np.random.random() * 0.1 for i in range(50)],
        'eval_loss': [2.3 - i * 0.00025 + np.random.random() * 0.15 for i in range(50)],
        'kv_loss': [1.2 - i * 0.00015 + np.random.random() * 0.05 for i in range(50)],
        'learning_rate': [2e-5 * (1 - i/50) for i in range(50)],
        'grad_norm': [1.5 - i * 0.02 + np.random.random() * 0.3 for i in range(50)],
    }
    
    # 模拟路由权重
    routing_weights = {
        'teacher_names': ['Qwen2.5-7B', 'Qwen2.5-14B'],
        'steps': list(range(0, 5000, 100)),
        'weights': [[0.5 + (i/50) * 0.2 * np.random.random(), 
                     0.5 - (i/50) * 0.2 * np.random.random()] 
                    for i in range(50)],
    }
    
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
    
    # 保存到临时目录
    test_dir = Path('./test_visualization_data')
    test_dir.mkdir(exist_ok=True)
    
    with open(test_dir / 'training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)
    
    with open(test_dir / 'routing_weights.json', 'w') as f:
        json.dump(routing_weights, f, indent=2)
    
    with open(test_dir / 'eval_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"✓ Mock data created in: {test_dir}")
    return test_dir


def test_training_curves(visualizer, data_dir):
    """测试训练曲线可视化"""
    print("\n" + "="*60)
    print("Testing Training Curves Visualization")
    print("="*60)
    
    html_path = visualizer.plot_training_curves(
        log_file=str(data_dir / 'training_log.json'),
        output_name='test_training_curves',
    )
    
    if html_path:
        print(f"✓ Training curves HTML: {html_path}")
        return True
    return False


def test_routing_weights(visualizer, data_dir):
    """测试路由权重可视化"""
    print("\n" + "="*60)
    print("Testing Routing Weights Visualization")
    print("="*60)
    
    html_path = visualizer.plot_routing_weights(
        weights_file=str(data_dir / 'routing_weights.json'),
        output_name='test_routing_weights',
    )
    
    if html_path:
        print(f"✓ Routing weights HTML: {html_path}")
        return True
    return False


def test_evaluation_results(visualizer, data_dir):
    """测试评测结果可视化"""
    print("\n" + "="*60)
    print("Testing Evaluation Results Visualization")
    print("="*60)
    
    # 创建多个模型的评测结果（模拟对比）
    eval_file = data_dir / 'eval_results.json'
    
    html_path = visualizer.plot_evaluation_results(
        eval_files=[str(eval_file)],
        labels=['Test Model'],
        output_name='test_evaluation',
    )
    
    if html_path:
        print(f"✓ Evaluation results HTML: {html_path}")
        return True
    return False


def test_experiment_summary(visualizer, data_dir):
    """测试综合报告"""
    print("\n" + "="*60)
    print("Testing Experiment Summary")
    print("="*60)
    
    html_path = visualizer.create_experiment_summary(
        experiment_dir=str(data_dir),
        output_name='test_summary',
    )
    
    if html_path:
        print(f"✓ Experiment summary HTML: {html_path}")
        return True
    return False


def main():
    """主测试函数"""
    print("="*60)
    print("HPC Visualizer Test Suite")
    print("="*60)
    
    # 创建模拟数据
    data_dir = create_mock_data()
    
    # 创建可视化器
    output_dir = './test_visualizations'
    visualizer = HPCVisualizer(output_dir=output_dir)
    print(f"\nOutput directory: {output_dir}")
    
    # 运行测试
    results = {
        'Training Curves': test_training_curves(visualizer, data_dir),
        'Routing Weights': test_routing_weights(visualizer, data_dir),
        'Evaluation Results': test_evaluation_results(visualizer, data_dir),
        'Experiment Summary': test_experiment_summary(visualizer, data_dir),
    }
    
    # 总结
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:30s} {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✓ All tests passed!")
        print(f"\n📊 Generated visualizations in: {output_dir}/")
        print("\nTo view the results:")
        print(f"  1. Open {output_dir}/test_summary.html in your browser")
        print(f"  2. Or check individual HTML files in {output_dir}/")
        
        # 列出生成的文件
        from pathlib import Path
        html_files = list(Path(output_dir).glob("*.html"))
        png_files = list(Path(output_dir).glob("*.png"))
        
        print(f"\nGenerated files:")
        print(f"  HTML: {len(html_files)} files")
        print(f"  PNG:  {len(png_files)} files")
        
        for html_file in html_files:
            print(f"    - {html_file.name}")
        
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
