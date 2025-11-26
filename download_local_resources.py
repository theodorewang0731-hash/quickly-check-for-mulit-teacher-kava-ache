"""
🔽 KAVA 本地资源下载器
下载所有模型和数据集到项目本地，实现完全离线训练
"""

import os
import sys
from huggingface_hub import snapshot_download

# 定义下载配置
DOWNLOAD_CONFIG = {
    "models": {
        "Qwen/Qwen2.5-1.5B-Instruct": "local_models/qwen-1.5b-teacher",
        "Qwen/Qwen2.5-0.5B-Instruct": "local_models/qwen-0.5b-student"
    },
    "datasets": {
        "gsm8k": "local_data/gsm8k"
    }
}

def download_models():
    """下载模型到本地"""
    print("=" * 70)
    print("📦 Step 1: Downloading Models")
    print("=" * 70)
    
    for model_id, local_path in DOWNLOAD_CONFIG["models"].items():
        print(f"\n🔽 Downloading {model_id}...")
        print(f"   Target: {local_path}")
        
        try:
            os.makedirs(local_path, exist_ok=True)
            
            snapshot_download(
                repo_id=model_id,
                local_dir=local_path,
                # 仅下载核心文件，减少不必要的下载量
                allow_patterns=[
                    "*.json",           # 配置文件
                    "*.safetensors",    # 模型权重
                    "*.py",             # 模型代码
                    "tokenizer*",       # 分词器文件
                    "*.model",          # 分词器模型
                    "*.txt",            # 其他配置
                    "generation_config.json",
                    "config.json",
                    "tokenizer_config.json"
                ],
                resume_download=True,
                local_dir_use_symlinks=False  # 避免符号链接问题
            )
            
            print(f"   ✅ {model_id} Download Complete!")
            
            # 验证关键文件
            required_files = ["config.json", "tokenizer_config.json"]
            missing_files = []
            for file in required_files:
                if not os.path.exists(os.path.join(local_path, file)):
                    missing_files.append(file)
            
            if missing_files:
                print(f"   ⚠️ Warning: Missing files: {missing_files}")
            else:
                print(f"   ✅ All required files verified")
                
        except Exception as e:
            print(f"   ❌ Error downloading {model_id}: {e}")
            return False
    
    return True

def download_dataset():
    """下载数据集到本地"""
    print("\n" + "=" * 70)
    print("📦 Step 2: Downloading Dataset (GSM8K)")
    print("=" * 70)
    
    try:
        from datasets import load_dataset
        
        dataset_path = DOWNLOAD_CONFIG["datasets"]["gsm8k"]
        os.makedirs(dataset_path, exist_ok=True)
        
        print(f"\n🔽 Downloading GSM8K dataset...")
        print(f"   Target: {dataset_path}")
        
        # 下载数据集
        dataset = load_dataset("gsm8k", "main")
        
        # 保存到本地
        dataset.save_to_disk(dataset_path)
        
        print(f"   ✅ GSM8K Download Complete!")
        print(f"   📊 Train samples: {len(dataset['train'])}")
        print(f"   📊 Test samples: {len(dataset['test'])}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error downloading dataset: {e}")
        print("\n   💡 Troubleshooting:")
        print("   1. Check network connection")
        print("   2. Try HF mirror: export HF_ENDPOINT=https://hf-mirror.com")
        print("   3. Manual download: https://huggingface.co/datasets/gsm8k")
        return False

def verify_downloads():
    """验证所有下载是否完成"""
    print("\n" + "=" * 70)
    print("🔍 Step 3: Verifying Downloads")
    print("=" * 70)
    
    all_valid = True
    
    # 验证模型
    print("\n📋 Models:")
    for model_id, local_path in DOWNLOAD_CONFIG["models"].items():
        if os.path.exists(local_path):
            # 检查关键文件
            config_file = os.path.join(local_path, "config.json")
            if os.path.exists(config_file):
                # 获取文件大小
                total_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, _, filenames in os.walk(local_path)
                    for filename in filenames
                )
                size_gb = total_size / (1024 ** 3)
                print(f"   ✅ {model_id}")
                print(f"      Path: {local_path}")
                print(f"      Size: {size_gb:.2f} GB")
            else:
                print(f"   ❌ {model_id} - Missing config.json")
                all_valid = False
        else:
            print(f"   ❌ {model_id} - Directory not found")
            all_valid = False
    
    # 验证数据集
    print("\n📋 Datasets:")
    for dataset_name, local_path in DOWNLOAD_CONFIG["datasets"].items():
        if os.path.exists(local_path):
            dataset_json = os.path.join(local_path, "dataset_info.json")
            if os.path.exists(dataset_json):
                total_size = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, _, filenames in os.walk(local_path)
                    for filename in filenames
                )
                size_mb = total_size / (1024 ** 2)
                print(f"   ✅ {dataset_name}")
                print(f"      Path: {local_path}")
                print(f"      Size: {size_mb:.2f} MB")
            else:
                print(f"   ❌ {dataset_name} - Missing dataset_info.json")
                all_valid = False
        else:
            print(f"   ❌ {dataset_name} - Directory not found")
            all_valid = False
    
    return all_valid

def main():
    print("\n" + "🎯" * 35)
    print("  KAVA Local Resource Downloader")
    print("  完全本地化训练环境搭建工具")
    print("🎯" * 35 + "\n")
    
    print("📝 Download Plan:")
    print("   Models:")
    for model_id, path in DOWNLOAD_CONFIG["models"].items():
        print(f"      • {model_id} → {path}")
    print("   Datasets:")
    for dataset_id, path in DOWNLOAD_CONFIG["datasets"].items():
        print(f"      • {dataset_id} → {path}")
    
    print("\n⚠️ Note: This may take 10-30 minutes depending on your network speed.")
    print("          Total download size: ~3-4 GB")
    
    input("\nPress Enter to start downloading...")
    
    # 执行下载
    success = True
    
    # Step 1: 下载模型
    if not download_models():
        success = False
        print("\n❌ Model download failed!")
    
    # Step 2: 下载数据集
    if not download_dataset():
        success = False
        print("\n❌ Dataset download failed!")
    
    # Step 3: 验证
    if success and verify_downloads():
        print("\n" + "=" * 70)
        print("🎉 SUCCESS! All resources downloaded successfully!")
        print("=" * 70)
        print("\n📂 Project Structure:")
        print("   .")
        print("   ├── local_models/")
        print("   │   ├── qwen-1.5b-teacher/")
        print("   │   └── qwen-0.5b-student/")
        print("   └── local_data/")
        print("       └── gsm8k/")
        print("\n✅ Ready to run: python train_local_only.py")
        return 0
    else:
        print("\n" + "=" * 70)
        print("⚠️ Download completed with errors. Please check the logs above.")
        print("=" * 70)
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️ Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
