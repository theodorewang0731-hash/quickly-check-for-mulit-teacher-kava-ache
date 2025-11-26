"""
🚀 KAVA 完全本地化训练脚本
所有模型和数据集从本地加载，实现离线训练
适配 RTX 4070 8GB VRAM
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
import os
import sys

# 引入核心组件
from experiments.kv_dimension_projector import KVDimensionProjector
from src.losses import MercatorKVLoss

# --- 🔥 全本地化配置 (4070/8GB 优化) ---
CONFIG = {
    # ✅ 本地模型路径（由 download_local_resources.py 下载）
    "teacher_path": "local_models/qwen-1.5b-teacher",          
    "student_path": "local_models/qwen-0.5b-student",
    
    # ✅ 本地数据集路径（使用原始字符串避免转义问题）
    "dataset_path": r"H:\kava\quickly check\local_data\gsm8k",
    "dataset_split": "train",
    
    # 显存优化参数 (8GB VRAM 黄金配置)
    "batch_size": 2,          
    "gradient_accumulation_steps": 16,  # 等效 Batch 32
    "max_length": 512,
    "lr_projector": 1e-3,
    "lr_student": 5e-5,
    "epochs": 1,
    "save_steps": 200,
    "device": "cuda",
    
    # 验证配置
    "verify_local_files": True  # 启动前检查本地文件
}

def verify_local_resources():
    """验证本地资源是否完整"""
    print("🔍 Verifying local resources...")
    
    errors = []
    
    # 检查模型
    for model_name, path in [("Teacher", CONFIG["teacher_path"]), 
                              ("Student", CONFIG["student_path"])]:
        if not os.path.exists(path):
            errors.append(f"❌ {model_name} not found: {path}")
        elif not os.path.exists(os.path.join(path, "config.json")):
            errors.append(f"❌ {model_name} incomplete: missing config.json")
        else:
            print(f"   ✅ {model_name}: {path}")
    
    # 检查数据集
    if not os.path.exists(CONFIG["dataset_path"]):
        errors.append(f"❌ Dataset not found: {CONFIG['dataset_path']}")
    elif not os.path.exists(os.path.join(CONFIG["dataset_path"], "dataset_info.json")):
        errors.append(f"❌ Dataset incomplete: missing dataset_info.json")
    else:
        print(f"   ✅ Dataset: {CONFIG['dataset_path']}")
    
    if errors:
        print("\n⚠️ Local resources verification failed:")
        for error in errors:
            print(f"   {error}")
        print("\n💡 Solution: Run download script first:")
        print("   python download_local_resources.py")
        return False
    
    print("   ✅ All local resources verified!\n")
    return True

def extract_flat_kv(past_key_values):
    """提取最后一层的 Key 并展平"""
    k, v = past_key_values[-1] 
    B, H, T, D_h = k.shape
    return k.permute(0, 2, 1, 3).contiguous().view(B, T, H * D_h)

def main():
    print("\n" + "🎯" * 35)
    print("  KAVA Fully Localized Training")
    print("  完全本地化训练（离线模式）")
    print("🎯" * 35 + "\n")
    
    print(f"🚀 Configuration:")
    print(f"   Teacher: {CONFIG['teacher_path']}")
    print(f"   Student: {CONFIG['student_path']}")
    print(f"   Dataset: {CONFIG['dataset_path']}")
    print(f"   Batch: {CONFIG['batch_size']} x {CONFIG['gradient_accumulation_steps']} = {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']}")
    
    # 验证本地资源
    if CONFIG["verify_local_files"]:
        if not verify_local_resources():
            sys.exit(1)
    
    # 创建检查点目录
    os.makedirs("checkpoints", exist_ok=True)
    
    # --- 1. 数据加载 (使用 Arrow 格式直接加载本地文件) ---
    print(f"📚 Loading local dataset from: {CONFIG['dataset_path']}")
    
    try:
        # 方法 1: 尝试使用 load_from_disk (完整元数据)
        try:
            print("   Attempting load_from_disk (method 1)...")
            dataset = load_from_disk(CONFIG["dataset_path"])
            
            # 智能处理数据集结构
            if isinstance(dataset, dict) and CONFIG["dataset_split"] in dataset:
                dataset_to_use = dataset[CONFIG["dataset_split"]]
                print(f"   ✅ Loaded {len(dataset_to_use)} samples from split '{CONFIG['dataset_split']}'")
            elif hasattr(dataset, '__len__'):
                dataset_to_use = dataset
                print(f"   ✅ Loaded {len(dataset_to_use)} samples (single dataset)")
            else:
                raise ValueError(f"Unrecognized dataset structure: {type(dataset)}")
            
            train_data = dataset_to_use
            
        except Exception as e1:
            # 方法 2: 回退到 Arrow 格式直接加载
            print(f"   Method 1 failed ({e1}), trying Arrow format (method 2)...")
            
            local_data_path = CONFIG['dataset_path']
            train_dir = os.path.join(local_data_path, 'train')
            
            # 检查 train 目录是否存在
            if not os.path.exists(train_dir):
                raise FileNotFoundError(f"Train directory not found: {train_dir}")
            
            # 直接读取 Arrow 文件
            print(f"   Loading Arrow files from: {train_dir}")
            dataset = load_dataset(
                "arrow",
                data_files={'train': f"{train_dir}/*.arrow"},
                split='train'
            )
            
            train_data = dataset
            print(f"   ✅ Loaded {len(train_data)} samples using Arrow format")
        
    except Exception as e:
        print(f"\n   ❌ All dataset loading methods failed!")
        print(f"   Error: {e}")
        print("\n   💡 Troubleshooting:")
        print(f"   1. Check if {CONFIG['dataset_path']} exists")
        print(f"   2. Check if {CONFIG['dataset_path']}/train/*.arrow files exist")
        print("   3. Run: python repair_dataset.py")
        print("   4. Verify dataset structure")
        
        # 列出目录内容帮助调试
        if os.path.exists(CONFIG['dataset_path']):
            print(f"\n   📂 Contents of {CONFIG['dataset_path']}:")
            for item in os.listdir(CONFIG['dataset_path']):
                item_path = os.path.join(CONFIG['dataset_path'], item)
                if os.path.isdir(item_path):
                    print(f"      [DIR]  {item}/")
                else:
                    print(f"      [FILE] {item}")
        
        sys.exit(1)
    
    # 加载分词器（从本地）
    print("\n🔤 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(CONFIG['student_path'])
        if tokenizer.pad_token is None: 
            tokenizer.pad_token = tokenizer.eos_token
        print(f"   ✅ Tokenizer loaded from {CONFIG['student_path']}")
    except Exception as e:
        print(f"   ❌ Tokenizer loading failed: {e}")
        sys.exit(1)
    
    # 数据预处理
    print("\n🔧 Processing dataset...")
    def process(examples):
        texts = [q + "\n" + a for q, a in zip(examples['question'], examples['answer'])]
        return tokenizer(texts, truncation=True, padding="max_length", max_length=CONFIG['max_length'])
    
    tokenized_data = train_data.map(process, batched=True, remove_columns=train_data.column_names)
    tokenized_data.set_format("torch", columns=["input_ids", "attention_mask"])
    dataloader = DataLoader(tokenized_data, batch_size=CONFIG['batch_size'], shuffle=True)
    print(f"   ✅ {len(dataloader)} batches ready")

    # --- 2. 模型加载 (全部从本地文件夹加载) ---
    print("\n🤖 Loading models from local disk...")
    
    try:
        # Teacher (4-bit量化，节省 VRAM)
        print("   Loading Teacher (4-bit quantized)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
        teacher = AutoModelForCausalLM.from_pretrained(
            CONFIG['teacher_path'], 
            quantization_config=bnb_config, 
            device_map="auto",
            local_files_only=True  # 强制仅使用本地文件
        )
        teacher.eval()
        print(f"      ✅ Teacher loaded: d_model={teacher.config.hidden_size}")
        
        # Student (从本地路径加载)
        print("   Loading Student (bfloat16)...")
        student = AutoModelForCausalLM.from_pretrained(
            CONFIG['student_path'], 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            local_files_only=True  # 强制仅使用本地文件
        )
        student.train()
        print(f"      ✅ Student loaded: d_model={student.config.hidden_size}")
        
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        print("\n   💡 Troubleshooting:")
        print("   1. Check if model directories exist")
        print("   2. Run: python download_local_resources.py")
        print("   3. Verify disk space (need ~3-4 GB)")
        sys.exit(1)

    # --- 3. 初始化 KAVA 组件 ---
    print("\n🗺️ Initializing Map Projection...")
    t_dim = teacher.config.hidden_size
    s_dim = student.config.hidden_size
    
    projector = KVDimensionProjector(
        teacher_configs={"local_teacher": {"d_model": t_dim}},
        student_d_model=s_dim,
        mlp_ratio=1.0,
        dropout=0.1
    ).to(CONFIG['device']).to(torch.bfloat16)
    
    loss_fn = MercatorKVLoss(alpha=1.0, beta=0.01).to(CONFIG['device'])
    
    print(f"   Projector: {t_dim} -> {s_dim}")
    print(f"   Loss: Mercator (alpha=1.0, beta=0.01)")
    
    optimizer = optim.AdamW([
        {'params': student.parameters(), 'lr': CONFIG['lr_student']},
        {'params': projector.parameters(), 'lr': CONFIG['lr_projector']}
    ])
    
    print(f"   Optimizer: Student LR={CONFIG['lr_student']}, Projector LR={CONFIG['lr_projector']}")

    # --- 4. 训练循环 ---
    print("\n" + "=" * 70)
    print("🎯 Training Start - Monitor 'CosSim' (Target: >0.90)")
    print("=" * 70)
    
    global_step = 0
    progress = tqdm(dataloader, desc="Training")
    
    try:
        for i, batch in enumerate(progress):
            input_ids = batch['input_ids'].to(CONFIG['device'])
            mask = batch['attention_mask'].to(CONFIG['device'])
            
            # Teacher Forward (No Grad)
            with torch.no_grad():
                t_out = teacher(input_ids, attention_mask=mask, use_cache=True)
                t_kv = extract_flat_kv(t_out.past_key_values)
                
            # Student Forward
            s_out = student(input_ids, attention_mask=mask, use_cache=True)
            s_kv = extract_flat_kv(s_out.past_key_values)
            
            # Projection & Loss
            t_proj, _ = projector.project_teacher_kv("local_teacher", t_kv, t_kv)
            loss, metrics = loss_fn(s_kv, t_proj)
            
            # 梯度累积
            loss = loss / CONFIG['gradient_accumulation_steps']
            loss.backward()
            
            if (i + 1) % CONFIG['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                # 核心监控
                actual_loss = loss.item() * CONFIG['gradient_accumulation_steps']
                cos_sim = metrics['cos_sim']
                
                # 状态判断
                if cos_sim >= 0.95:
                    status = "✅ Excellent"
                elif cos_sim >= 0.90:
                    status = "🎯 Great"
                elif cos_sim >= 0.70:
                    status = "📈 Good"
                elif cos_sim >= 0.50:
                    status = "⚠️ Learning"
                else:
                    status = "🔄 Adapting"
                
                progress.set_postfix({
                    "Loss": f"{actual_loss:.4f}", 
                    "CosSim": f"{cos_sim:.4f}",
                    "Status": status
                })
                
                # 每 50 步详细报告
                if global_step % 50 == 0:
                    print(f"\n[Step {global_step:04d}] Loss: {actual_loss:.4f} | CosSim: {cos_sim:.4f} {status}")
                
                # 保存检查点
                if global_step % CONFIG['save_steps'] == 0:
                    checkpoint_path = f"checkpoints/proj_step_{global_step}.pth"
                    torch.save(projector.state_dict(), checkpoint_path)
                    print(f"💾 Checkpoint saved: {checkpoint_path}")

        print("\n" + "=" * 70)
        print("✅ Training Complete!")
        print("=" * 70)
        
        # 保存最终模型
        torch.save(projector.state_dict(), "final_projector.pth")
        student.save_pretrained("final_student")
        
        print("💾 Final models saved:")
        print("   - final_projector.pth")
        print("   - final_student/")
        print("\n🎉 All Done!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        print("💾 Saving emergency checkpoint...")
        torch.save(projector.state_dict(), "checkpoints/emergency_projector.pth")
        print("✅ Emergency checkpoint saved")
    
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        print("💾 Saving emergency checkpoint...")
        torch.save(projector.state_dict(), "checkpoints/emergency_projector.pth")
        raise

if __name__ == "__main__":
    main()
