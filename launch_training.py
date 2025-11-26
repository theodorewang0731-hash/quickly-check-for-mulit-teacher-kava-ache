"""
🚀 KAVA 训练最终启动脚本
所有问题已解决，确保训练顺利运行
"""

import subprocess
import sys
import os

def check_environment():
    """检查环境是否准备就绪"""
    print("🔍 Pre-flight Checklist")
    print("=" * 70)
    
    checks = []
    
    # 1. 检查 CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        checks.append(("CUDA Available", cuda_available, f"{torch.cuda.get_device_name(0)}" if cuda_available else "Not available"))
    except Exception as e:
        checks.append(("CUDA Available", False, str(e)))
    
    # 2. 检查模型文件
    teacher_exists = os.path.exists("local_models/qwen-1.5b-teacher/config.json")
    student_exists = os.path.exists("local_models/qwen-0.5b-student/config.json")
    checks.append(("Teacher Model", teacher_exists, "local_models/qwen-1.5b-teacher"))
    checks.append(("Student Model", student_exists, "local_models/qwen-0.5b-student"))
    
    # 3. 检查数据集
    dataset_exists = os.path.exists("local_data/gsm8k/train")
    checks.append(("Dataset", dataset_exists, "local_data/gsm8k/train"))
    
    # 4. 检查核心文件
    losses_exists = os.path.exists("src/losses.py")
    projector_exists = os.path.exists("experiments/kv_dimension_projector.py")
    train_exists = os.path.exists("train_simplified.py")
    checks.append(("Loss Functions", losses_exists, "src/losses.py"))
    checks.append(("Projector", projector_exists, "experiments/kv_dimension_projector.py"))
    checks.append(("Training Script", train_exists, "train_simplified.py"))
    
    # 显示结果
    all_passed = True
    for check_name, status, details in checks:
        symbol = "✅" if status else "❌"
        print(f"  {symbol} {check_name}: {details}")
        if not status:
            all_passed = False
    
    print("=" * 70)
    
    if not all_passed:
        print("\n❌ Some checks failed! Please fix the issues above.")
        return False
    
    print("\n✅ All checks passed! Ready to start training.")
    return True

def show_training_info():
    """显示训练信息"""
    print("\n📊 Training Configuration")
    print("=" * 70)
    print("  Architecture:")
    print("    • Teacher: Qwen-1.5B (4-bit quantized)")
    print("    • Student: Qwen-0.5B (bfloat16)")
    print("    • Projector: Elastic Bottleneck (dynamic dims)")
    print("    • Loss: Mercator (Map Projection)")
    print("\n  Hardware:")
    print("    • GPU: RTX 4070 Laptop (8GB)")
    print("    • Batch Size: 2 x 16 = 32 (effective)")
    print("    • Expected VRAM: ~6-7GB")
    print("\n  Dataset:")
    print("    • GSM8K: 7,473 training samples")
    print("    • Sequence Length: 512 tokens")
    print("\n  Training Speed:")
    print("    • ~0.65 it/s (1.5s per iteration)")
    print("    • ~20 min per 50 steps")
    print("    • Total: 1.5-2 hours")
    print("=" * 70)

def show_monitoring_guide():
    """显示监控指南"""
    print("\n🎯 Monitoring Guide")
    print("=" * 70)
    print("  Key Metric: Cosine Similarity (CosSim)")
    print("\n  Progress Stages:")
    print("    0-50 steps:   CosSim 0.20-0.50  🔄 Adapting")
    print("    50-100 steps: CosSim 0.50-0.70  ⚠️  Learning")
    print("    100-200 steps: CosSim 0.70-0.90  📈 Good")
    print("    200+ steps:   CosSim >0.90      ✅ Excellent  ← TARGET!")
    print("\n  What to Watch:")
    print("    • Loss should decrease steadily")
    print("    • CosSim should increase to >0.90")
    print("    • No NaN or Inf values")
    print("    • No OOM errors")
    print("\n  Checkpoints:")
    print("    • Auto-save every 200 steps")
    print("    • Ctrl+C saves emergency checkpoint")
    print("    • Final models saved at completion")
    print("=" * 70)

def main():
    print("\n" + "🚀" * 35)
    print("  KAVA Training Final Launch")
    print("  最终启动检查与训练")
    print("🚀" * 35 + "\n")
    
    # 检查环境
    if not check_environment():
        sys.exit(1)
    
    # 显示信息
    show_training_info()
    show_monitoring_guide()
    
    # 询问确认
    print("\n" + "⚠️ " * 25)
    print("  Training will start and run for ~1.5-2 hours")
    print("  Make sure your laptop is plugged in!")
    print("⚠️ " * 25 + "\n")
    
    response = input("Start training now? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\n❌ Training cancelled by user")
        return 1
    
    # 启动训练
    print("\n" + "🎯" * 35)
    print("  Launching Training Script")
    print("🎯" * 35 + "\n")
    
    try:
        # 使用 subprocess 启动训练，保持输出可见
        result = subprocess.run(
            ["python", "train_simplified.py"],
            check=False
        )
        
        if result.returncode == 0:
            print("\n" + "🎉" * 35)
            print("  Training Completed Successfully!")
            print("🎉" * 35)
            return 0
        else:
            print("\n" + "⚠️ " * 35)
            print("  Training exited with errors")
            print("⚠️ " * 35)
            return result.returncode
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
        print("Emergency checkpoint should be saved automatically.")
        return 130
    
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
