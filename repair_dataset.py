"""
🔧 GSM8K 数据集修复脚本
确保数据集结构完整，包含 dataset_info.json 等元数据
"""

import os
import shutil
from datasets import load_dataset

DATA_DIR = "local_data/gsm8k"

def repair_dataset():
    """修复并重建完整的 GSM8K 数据集结构"""
    
    print("\n" + "=" * 70)
    print("🔧 GSM8K Dataset Repair Tool")
    print("=" * 70)
    
    # 1. 删除旧的/不完整的本地数据文件夹
    if os.path.exists(DATA_DIR):
        print(f"\n🗑️ Deleting incomplete data folder: {DATA_DIR}")
        try:
            shutil.rmtree(DATA_DIR)
            print("   ✅ Old folder removed")
        except Exception as e:
            print(f"   ⚠️ Warning: Could not remove folder: {e}")
            print("   Attempting to continue...")
    else:
        print(f"\n📁 Data folder not found (will create new): {DATA_DIR}")
    
    # 2. 从 Hugging Face Hub (或缓存) 重新加载 GSM8K
    print("\n🌍 Loading GSM8K from HuggingFace (cache or hub)...")
    print("   This may take a few minutes on first run...")
    
    try:
        # 加载完整的数据集（包含 train 和 test split）
        print("   Loading train split...")
        dataset = load_dataset("gsm8k", "main")
        
        print(f"   ✅ Dataset loaded successfully!")
        print(f"      Train samples: {len(dataset['train'])}")
        print(f"      Test samples: {len(dataset['test'])}")
        
    except Exception as e:
        print(f"\n❌ Critical Error during dataset loading: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Check network connection")
        print("   2. Try HF mirror (China users):")
        print("      PowerShell: $env:HF_ENDPOINT='https://hf-mirror.com'")
        print("      Then re-run this script")
        print("   3. Clear HuggingFace cache:")
        print("      Remove: ~/.cache/huggingface/ (Linux/Mac)")
        print("      Remove: %USERPROFILE%\\.cache\\huggingface\\ (Windows)")
        return False
    
    # 3. 创建本地目录
    print(f"\n💾 Saving complete dataset structure to {DATA_DIR}...")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    try:
        # 强制保存到本地路径，生成完整的结构文件 (包括 dataset_info.json)
        dataset.save_to_disk(DATA_DIR)
        
        print("   ✅ Dataset saved successfully!")
        
    except Exception as e:
        print(f"\n❌ Error saving dataset: {e}")
        print("\n💡 Possible causes:")
        print("   1. Insufficient disk space (need ~100 MB)")
        print("   2. Permission denied (check folder permissions)")
        print("   3. Path too long (try shorter path)")
        return False
    
    # 4. 验证关键文件
    print("\n🔍 Verifying dataset structure...")
    
    required_files = [
        "dataset_info.json",
        "state.json"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = os.path.join(DATA_DIR, file)
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"   ✅ {file} ({file_size} bytes)")
        else:
            print(f"   ❌ {file} - MISSING")
            missing_files.append(file)
    
    # 检查 train 和 test 目录
    for split in ["train", "test"]:
        split_dir = os.path.join(DATA_DIR, split)
        if os.path.exists(split_dir):
            print(f"   ✅ {split}/ directory exists")
        else:
            print(f"   ❌ {split}/ directory - MISSING")
            missing_files.append(f"{split}/")
    
    if missing_files:
        print(f"\n⚠️ Warning: Missing files/directories: {missing_files}")
        print("   Dataset may still work, but structure is incomplete.")
        return False
    
    # 5. 最终验证 - 尝试从磁盘加载
    print("\n🧪 Testing load from disk...")
    try:
        from datasets import load_from_disk
        test_dataset = load_from_disk(DATA_DIR)
        
        print(f"   ✅ Load test successful!")
        print(f"      Splits available: {list(test_dataset.keys())}")
        print(f"      Train samples: {len(test_dataset['train'])}")
        
    except Exception as e:
        print(f"   ❌ Load test failed: {e}")
        return False
    
    # 成功！
    print("\n" + "=" * 70)
    print("🎉 SUCCESS! Dataset repair completed!")
    print("=" * 70)
    print(f"\n📂 Dataset location: {DATA_DIR}")
    print(f"📊 Total samples: {len(test_dataset['train']) + len(test_dataset['test'])}")
    print(f"   - Train: {len(test_dataset['train'])}")
    print(f"   - Test: {len(test_dataset['test'])}")
    print("\n✅ Ready to run: python train_local_only.py")
    print("=" * 70 + "\n")
    
    return True

if __name__ == "__main__":
    try:
        success = repair_dataset()
        
        if success:
            print("✨ Next step: python train_local_only.py")
            exit(0)
        else:
            print("\n⚠️ Dataset repair completed with warnings.")
            print("   You may try running train_local_only.py anyway.")
            exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Repair interrupted by user")
        exit(1)
        
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
