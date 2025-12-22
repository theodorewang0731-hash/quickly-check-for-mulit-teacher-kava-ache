"""r
环境设置检查脚本
运行此脚本以验证环境是否正确配置
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.environment_adapter import create_environment_adapter
from src.dynamic_kv_extractor import create_kv_extractor


def main():
    print("\n" + "="*70)
    print("  KAVA Environment Setup Checker")
    print("  检查环境配置是否正确")
    print("="*70 + "\n")
    
    # 创建环境适配器
    try:
        adapter = create_environment_adapter()
    except Exception as e:
        print(f"❌ Failed to create environment adapter: {e}")
        return 1
    
    # 检查 GPU
    print("\n" + "="*70)
    print("🖥️  GPU Check")
    print("="*70)
    
    if adapter.hardware_config['device'] == 'cuda':
        print(f"✅ GPU Available: {adapter.hardware_config['device_name']}")
        print(f"   Memory: {adapter.hardware_config['memory_gb']:.1f} GB")
        print(f"   Precision: {adapter.hardware_config['precision']}")
    else:
        print(f"⚠️  No GPU detected, using: {adapter.hardware_config['device']}")
    
    # 检查路径
    print("\n" + "="*70)
    print("📁 Path Check")
    print("="*70)
    
    all_paths_ok = True
    for path_type, path in adapter.paths.items():
        exists = path.exists()
        symbol = "✅" if exists else "⚠️"
        print(f"{symbol} {path_type}: {path}")
        if not exists:
            all_paths_ok = False
    
    if not all_paths_ok:
        print("\n⚠️  Some paths don't exist yet (will be created during training)")
    
    # 检查依赖
    print("\n" + "="*70)
    print("📦 Dependency Check")
    print("="*70)
    
    required_ok = True
    for dep_name, dep_info in adapter.dependencies.items():
        if dep_name in ['torch', 'transformers']:
            if dep_info['available']:
                print(f"✅ {dep_name} ({dep_info['version']})")
            else:
                print(f"❌ {dep_name} (REQUIRED but not found)")
                required_ok = False
        else:
            if dep_info['available']:
                print(f"✅ {dep_name} ({dep_info['version']})")
            else:
                print(f"ℹ️  {dep_name} (optional, not installed)")
    
    if not required_ok:
        print("\n❌ Some required dependencies are missing!")
        return 1
    
    # 测试 KV 提取器
    print("\n" + "="*70)
    print("🔧 KV Extractor Test")
    print("="*70)
    
    try:
        extractor = create_kv_extractor()
        print("✅ KV Extractor created successfully")
        extractor.print_extraction_info()
    except Exception as e:
        print(f"❌ Failed to create KV extractor: {e}")
        return 1
    
    # 获取推荐配置
    print("\n" + "="*70)
    print("🎯 Recommended Training Configuration")
    print("="*70)
    
    config = adapter.get_training_config()
    print(f"   Device: {config['device']}")
    print(f"   Dtype: {config['dtype']}")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   Gradient Accumulation: {config['gradient_accumulation_steps']}")
    print(f"   Effective Batch Size: {config['effective_batch_size']}")
    print(f"   Mixed Precision: {config['mixed_precision']}")
    
    # 最终结果
    print("\n" + "="*70)
    
    if adapter.hardware_config['device'] == 'cuda' and required_ok:
        print("✅ Environment is ready for training!")
        print("\nNext steps:")
        print("  1. Download models and data (if not already done)")
        print("  2. Run: python train_adaptive.py")
        print("  3. Or submit to HPC: sbatch scripts/submit_slurm.sh")
    else:
        print("⚠️  Environment check completed with warnings")
        print("\nPlease fix the issues above before training")
    
    print("="*70 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
