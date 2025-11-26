"""
🚀 KAVA 完全本地化训练脚本 - 简化稳定版
基于您本地已有的资源，确保 100% 能运行
适配 RTX 4070 8GB VRAM
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
import os
import sys

# 引入核心组件
from experiments.kv_dimension_projector import KVDimensionProjector
from src.losses import MercatorKVLoss

# --- 🔥 本地化配置 (基于您的实际路径) ---
CONFIG = {
    # 本地模型路径（相对路径，已验证存在）
    "teacher_path": "local_models/qwen-1.5b-teacher",          
    "student_path": "local_models/qwen-0.5b-student",
    
    # 本地数据集路径（直接指向 Arrow 文件）
    "dataset_path": "local_data/gsm8k",
    
    # 显存优化参数 (8GB VRAM)
    "batch_size": 2,          
    "gradient_accumulation_steps": 16,
    "max_length": 512,
    "lr_projector": 1e-3,
    "lr_student": 5e-5,
    "epochs": 1,
    "save_steps": 200,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def extract_flat_kv(past_key_values, debug=False, use_all_layers=False):
    """
    提取并展平 KV cache
    
    Args:
        past_key_values: HF 模型输出的 KV cache (tuple of layers)
        debug: 是否打印调试信息
        use_all_layers: 是否使用所有层（用于量化模型）
    
    Returns:
        k_flat: 展平后的 Key
    """
    if use_all_layers:
        # 量化模型：聚合所有层的 KV cache
        all_keys = []
        for layer_kv in past_key_values:
            k, v = layer_kv
            if len(k.shape) == 4:
                B, H, T, D_h = k.shape
                k_flat = k.permute(0, 2, 1, 3).contiguous().view(B, T, H * D_h)
            else:
                k_flat = k
            all_keys.append(k_flat)
        
        # 拼接所有层: [B, T, num_layers * H * D_h]
        k_combined = torch.cat(all_keys, dim=-1)
        
        if debug:
            print(f"\n[DEBUG extract_flat_kv - All Layers]")
            print(f"   Num layers: {len(past_key_values)}")
            print(f"   Per-layer K shape: {past_key_values[0][0].shape}")
            print(f"   Combined K shape: {k_combined.shape}")
        
        return k_combined
    
    else:
        # 标准模式：只用最后一层
        k, v = past_key_values[-1]
        
        if debug:
            print(f"\n[DEBUG extract_flat_kv - Last Layer]")
            print(f"   Original K shape: {k.shape}")
        
        if len(k.shape) == 4:
            B, H, T, D_h = k.shape
            k_flat = k.permute(0, 2, 1, 3).contiguous().view(B, T, H * D_h)
            
            if debug:
                print(f"   Flattened K shape: {k_flat.shape}")
                print(f"   D_model = H({H}) * D_h({D_h}) = {H * D_h}")
        elif len(k.shape) == 3:
            k_flat = k
            if debug:
                print(f"   K already flattened: {k.shape}")
        else:
            raise ValueError(f"Unexpected K shape: {k.shape}")
        
        return k_flat

def main():
    print("\n" + "🎯" * 35)
    print("  KAVA Local Training - Simplified & Stable")
    print("  完全本地化训练（简化稳定版）")
    print("🎯" * 35 + "\n")
    
    print(f"⚙️ Configuration:")
    print(f"   Teacher: {CONFIG['teacher_path']}")
    print(f"   Student: {CONFIG['student_path']}")
    print(f"   Dataset: {CONFIG['dataset_path']}")
    print(f"   Device: {CONFIG['device']}")
    print(f"   Effective Batch Size: {CONFIG['batch_size']} x {CONFIG['gradient_accumulation_steps']} = {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']}")
    
    # 创建检查点目录
    os.makedirs("checkpoints", exist_ok=True)
    
    # --- 1. 数据加载（使用最简单可靠的方法）---
    print("\n📚 Step 1: Loading Dataset")
    print(f"   Path: {CONFIG['dataset_path']}")
    
    try:
        # 直接使用 Arrow 格式加载
        train_arrow = os.path.join(CONFIG['dataset_path'], "train", "*.arrow")
        print(f"   Loading from: {train_arrow}")
        
        dataset = load_dataset(
            "arrow",
            data_files=train_arrow,
            split="train"
        )
        
        print(f"   ✅ Dataset loaded: {len(dataset)} samples")
        
    except Exception as e:
        print(f"   ❌ Dataset loading failed: {e}")
        print("\n   💡 Debug info:")
        print(f"      Checking: {CONFIG['dataset_path']}/train/")
        train_dir = os.path.join(CONFIG['dataset_path'], "train")
        if os.path.exists(train_dir):
            files = os.listdir(train_dir)
            print(f"      Files found: {files}")
        sys.exit(1)
    
    # --- 2. 分词器加载 ---
    print("\n🔤 Step 2: Loading Tokenizer")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            CONFIG['student_path'],
            local_files_only=True
        )
        if tokenizer.pad_token is None: 
            tokenizer.pad_token = tokenizer.eos_token
        print(f"   ✅ Tokenizer loaded")
    except Exception as e:
        print(f"   ❌ Tokenizer loading failed: {e}")
        sys.exit(1)
    
    # --- 3. 数据预处理 ---
    print("\n🔧 Step 3: Processing Dataset")
    def process(examples):
        texts = [q + "\n" + a for q, a in zip(examples['question'], examples['answer'])]
        return tokenizer(
            texts, 
            truncation=True, 
            padding="max_length", 
            max_length=CONFIG['max_length'],
            return_tensors=None
        )
    
    tokenized_data = dataset.map(
        process, 
        batched=True, 
        remove_columns=dataset.column_names
    )
    tokenized_data.set_format("torch", columns=["input_ids", "attention_mask"])
    
    dataloader = DataLoader(
        tokenized_data, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True
    )
    print(f"   ✅ {len(dataloader)} batches prepared")

    # --- 4. 模型加载 ---
    print("\n🤖 Step 4: Loading Models")
    
    try:
        # Teacher (4-bit量化)
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
            local_files_only=True
        )
        teacher.eval()
        t_dim = teacher.config.hidden_size
        print(f"      ✅ Teacher: d_model={t_dim}")
        
        # Student
        print("   Loading Student (bfloat16)...")
        student = AutoModelForCausalLM.from_pretrained(
            CONFIG['student_path'], 
            torch_dtype=torch.bfloat16, 
            device_map="auto",
            local_files_only=True
        )
        student.train()
        s_dim = student.config.hidden_size
        print(f"      ✅ Student: d_model={s_dim}")
        
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        print("\n   💡 Troubleshooting:")
        print("      1. Check if model folders exist")
        print("      2. Verify config.json in model folders")
        sys.exit(1)

    # --- 5. 动态检测 Teacher KV 维度 ---
    print("\n🔍 Step 5: Detecting Actual KV Dimensions")
    
    # 用一个小 batch 测试实际的 KV 维度
    test_input = torch.randint(0, 1000, (1, 32)).to(CONFIG['device'])
    
    with torch.no_grad():
        # Teacher
        t_test_out = teacher(test_input, use_cache=True)
        t_test_kv = extract_flat_kv(t_test_out.past_key_values, use_all_layers=True)
        actual_t_dim = t_test_kv.shape[-1]
        
        # Student
        s_test_out = student(test_input, use_cache=True)
        s_test_kv = extract_flat_kv(s_test_out.past_key_values, use_all_layers=True)
        actual_s_dim = s_test_kv.shape[-1]
    
    print(f"   Detected Teacher KV dim: {actual_t_dim} (config says: {t_dim})")
    print(f"   Detected Student KV dim: {actual_s_dim} (config says: {s_dim})")
    
    # 使用实际检测到的维度
    if actual_t_dim != t_dim:
        print(f"   ⚠️  Using detected dim {actual_t_dim} instead of config {t_dim}")
        t_dim = actual_t_dim
    
    if actual_s_dim != s_dim:
        print(f"   ⚠️  Using detected dim {actual_s_dim} instead of config {s_dim}")
        s_dim = actual_s_dim
    
    # --- 6. 初始化 KAVA 组件（使用实际维度）---
    print("\n🗺️ Step 6: Initializing KAVA Components")
    
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
    
    print(f"   Optimizer: AdamW")
    print(f"      Student LR: {CONFIG['lr_student']}")
    print(f"      Projector LR: {CONFIG['lr_projector']}")

    # --- 6. 训练循环 ---
    print("\n" + "=" * 70)
    print("🎯 Training Start - Monitor 'CosSim' (Target: >0.90)")
    print("=" * 70 + "\n")
    
    global_step = 0
    progress = tqdm(dataloader, desc="Training")
    first_batch = True  # 用于调试第一个 batch
    
    try:
        for i, batch in enumerate(progress):
            input_ids = batch['input_ids'].to(CONFIG['device'])
            mask = batch['attention_mask'].to(CONFIG['device'])
            
            # Teacher Forward (No Grad) - 使用所有层
            with torch.no_grad():
                t_out = teacher(input_ids, attention_mask=mask, use_cache=True)
                t_kv = extract_flat_kv(t_out.past_key_values, debug=first_batch, use_all_layers=True)
                # 转换为 bfloat16 以匹配 Projector
                t_kv = t_kv.to(torch.bfloat16)
                
            # Student Forward - 也使用所有层保持一致性
            s_out = student(input_ids, attention_mask=mask, use_cache=True)
            s_kv = extract_flat_kv(s_out.past_key_values, debug=first_batch, use_all_layers=True)
            # 确保类型一致
            s_kv = s_kv.to(torch.bfloat16)
            
            if first_batch:
                print(f"\n[First Batch Debug Info]")
                print(f"   Input shape: {input_ids.shape}")
                print(f"   Teacher KV shape: {t_kv.shape} (expected: [B, T, {t_dim}])")
                print(f"   Student KV shape: {s_kv.shape} (expected: [B, T, {s_dim}])")
                print(f"   Projector config: {t_dim} -> {s_dim}")
                first_batch = False
            
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
        
        print("\n💾 Final models saved:")
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
        print("\n💡 Full error traceback:")
        import traceback
        traceback.print_exc()
        print("\n💾 Saving emergency checkpoint...")
        torch.save(projector.state_dict(), "checkpoints/emergency_projector.pth")

if __name__ == "__main__":
    print("\n" + "🚀" * 35)
    print("Starting KAVA Training with Local Resources")
    print("🚀" * 35)
    
    # 环境检查
    print("\n📋 Environment Check:")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    main()
