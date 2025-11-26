import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
import os

# 引入核心组件
from experiments.kv_dimension_projector import KVDimensionProjector
from src.losses import MercatorKVLoss

# --- 🔥 8GB 显存黄金配置 ---
CONFIG = {
    "teacher_id": "Qwen/Qwen2.5-1.5B-Instruct", # 老师
    "student_id": "Qwen/Qwen2.5-0.5B",          # 学生
    "dataset_name": "gsm8k",
    "dataset_config": "main",
    
    # 显存优化关键点:
    "batch_size": 2,                # 极小 Batch
    "gradient_accumulation_steps": 16, # 累积 16 次 = 等效 Batch 32 (稳!)
    
    "max_length": 512,
    "lr_projector": 1e-3,           # Projector 从头学，快一点
    "lr_student": 5e-5,             # Student 微调，慢一点
    "epochs": 1,
    "save_steps": 200,
    "device": "cuda"
}

def extract_flat_kv(past_key_values):
    """提取最后一层的 Key 并展平"""
    # past_key_values[-1] 是最后一层 (Key, Value)
    k, v = past_key_values[-1] 
    B, H, T, D_h = k.shape
    # 展平: [B, H, T, D_h] -> [B, T, H*D_h]
    return k.permute(0, 2, 1, 3).contiguous().view(B, T, H * D_h)

def main():
    print(f"🚀 KAVA Training: {CONFIG['teacher_id']} -> {CONFIG['student_id']}")
    print(f"📊 Configuration:")
    print(f"   Batch Size: {CONFIG['batch_size']} x {CONFIG['gradient_accumulation_steps']} = {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']} (Effective)")
    print(f"   Max Length: {CONFIG['max_length']}")
    print(f"   Learning Rates: Projector={CONFIG['lr_projector']}, Student={CONFIG['lr_student']}")
    
    # 创建检查点目录
    os.makedirs("checkpoints", exist_ok=True)
    
    # 1. 数据准备
    print("\n📚 Loading Dataset...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['student_id'])
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset(CONFIG['dataset_name'], CONFIG['dataset_config'], split="train")
    print(f"   Total Examples: {len(dataset)}")
    
    def process(examples):
        texts = [q + "\n" + a for q, a in zip(examples['question'], examples['answer'])]
        return tokenizer(texts, truncation=True, padding="max_length", max_length=CONFIG['max_length'])
    
    tokenized_data = dataset.map(process, batched=True, remove_columns=dataset.column_names)
    tokenized_data.set_format("torch", columns=["input_ids", "attention_mask"])
    dataloader = DataLoader(tokenized_data, batch_size=CONFIG['batch_size'], shuffle=True)
    print(f"   Batches per Epoch: {len(dataloader)}")

    # 2. 模型加载 (4-bit 量化 Teacher)
    print("\n🤖 Loading Models (Quantized)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
    
    print("   Loading Teacher (4-bit Quantized)...")
    teacher = AutoModelForCausalLM.from_pretrained(
        CONFIG['teacher_id'], 
        quantization_config=bnb_config, 
        device_map="auto"
    )
    teacher.eval()
    print(f"   Teacher d_model: {teacher.config.hidden_size}")
    
    print("   Loading Student (bfloat16)...")
    student = AutoModelForCausalLM.from_pretrained(
        CONFIG['student_id'], 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    student.train()
    print(f"   Student d_model: {student.config.hidden_size}")

    # 3. 初始化 KAVA 组件
    print("\n🗺️ Initializing Map Projection...")
    t_dim = teacher.config.hidden_size
    s_dim = student.config.hidden_size
    
    projector = KVDimensionProjector(
        teacher_configs={CONFIG['teacher_id']: {"d_model": t_dim}},
        student_d_model=s_dim,
        mlp_ratio=1.0,  # 1.5B -> 0.5B 用 1.0 倍足够
        dropout=0.1
    ).to(CONFIG['device']).to(torch.bfloat16)
    
    loss_fn = MercatorKVLoss(alpha=1.0, beta=0.01).to(CONFIG['device']) # 墨卡托损失
    
    print(f"   Projector: {t_dim} -> {s_dim} (mlp_ratio=1.0)")
    print(f"   Loss Function: Mercator (alpha=1.0, beta=0.01)")
    
    optimizer = optim.AdamW([
        {'params': student.parameters(), 'lr': CONFIG['lr_student']},
        {'params': projector.parameters(), 'lr': CONFIG['lr_projector']}
    ])
    
    print(f"   Optimizer: AdamW (Student LR={CONFIG['lr_student']}, Projector LR={CONFIG['lr_projector']})")

    # 4. 训练循环
    print("\n" + "="*70)
    print("🎯 Training Start - Monitor 'CosSim' (Target: >0.90)")
    print("="*70)
    
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
            t_proj, _ = projector.project_teacher_kv(CONFIG['teacher_id'], t_kv, t_kv)
            loss, metrics = loss_fn(s_kv, t_proj)
            
            # 梯度累积
            loss = loss / CONFIG['gradient_accumulation_steps']
            loss.backward()
            
            if (i + 1) % CONFIG['gradient_accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                # 核心监控: CosSim > 0.9 即为成功
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

        print("\n" + "="*70)
        print("✅ Training Complete!")
        print("="*70)
        
        # 保存最终模型
        final_proj_path = "final_projector.pth"
        final_student_path = "final_student"
        
        torch.save(projector.state_dict(), final_proj_path)
        print(f"💾 Final Projector saved: {final_proj_path}")
        
        student.save_pretrained(final_student_path)
        print(f"💾 Final Student saved: {final_student_path}")
        
        print("\n🎉 All Done! Check your models in:")
        print(f"   - Projector: {final_proj_path}")
        print(f"   - Student: {final_student_path}/")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        print("💾 Saving emergency checkpoint...")
        torch.save(projector.state_dict(), "checkpoints/emergency_projector.pth")
        print("✅ Emergency checkpoint saved: checkpoints/emergency_projector.pth")
    
    except Exception as e:
        print(f"\n\n❌ Error occurred: {e}")
        print("💾 Saving emergency checkpoint...")
        torch.save(projector.state_dict(), "checkpoints/emergency_projector.pth")
        print("✅ Emergency checkpoint saved")
        raise

if __name__ == "__main__":
    main()
