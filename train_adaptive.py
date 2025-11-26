"""
生产级 KAVA 训练脚本（环境自适应版本）
✅ 完全环境无关，支持本地、HPC、云平台
✅ 自动检测硬件、路径、依赖
✅ 动态 KV 维度检测
✅ 跨层聚合（Cross-Layer Aggregation）
"""

import os
import sys
from pathlib import Path

# 确保可以导入项目模块
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from tqdm import tqdm
import json

# 导入环境自适应模块
from src.environment_adapter import create_environment_adapter
from src.dynamic_kv_extractor import create_kv_extractor
from src.losses import MercatorKVLoss
from experiments.kv_dimension_projector import KVDimensionProjector


# ====================================================================
# 全局配置（可通过命令行参数覆盖）
# ====================================================================
GLOBAL_CONFIG = {
    # 数据配置
    'max_length': 512,
    'max_steps': 1000,
    'eval_interval': 200,
    'save_interval': 200,
    
    # 优化器配置
    'learning_rate_student': 5e-5,
    'learning_rate_projector': 1e-3,
    'weight_decay': 0.01,
    'warmup_steps': 100,
    
    # Loss 配置
    'loss_alpha': 1.0,   # 方向损失权重
    'loss_beta': 0.01,   # 幅度损失权重
    
    # 模型配置
    'teacher_model_name': 'Qwen/Qwen2.5-1.5B-Instruct',
    'student_model_name': 'Qwen/Qwen2.5-0.5B',
    'teacher_quantization': '4bit',  # 4bit / 8bit / None
    
    # KV 提取配置
    'kv_aggregation_method': 'concat',  # concat / mean / weighted
    'use_all_layers': True,
}


def load_models_adaptive(env_adapter):
    """
    自适应加载模型
    根据环境自动选择最佳配置
    """
    print("\n" + "="*70)
    print("📦 Loading Models (Environment-Adaptive)")
    print("="*70)
    
    device = env_adapter.get_device()
    dtype = env_adapter.get_dtype()
    model_path = env_adapter.paths['models']
    
    # Teacher 模型路径
    teacher_path = model_path / "qwen-1.5b-teacher"
    student_path = model_path / "qwen-0.5b-student"
    
    # 如果本地不存在，使用 HuggingFace 名称
    if not teacher_path.exists():
        teacher_path = GLOBAL_CONFIG['teacher_model_name']
        print(f"⚠️  Local model not found, using: {teacher_path}")
    
    if not student_path.exists():
        student_path = GLOBAL_CONFIG['student_model_name']
        print(f"⚠️  Local model not found, using: {student_path}")
    
    # 量化配置
    quantization_config = None
    if GLOBAL_CONFIG['teacher_quantization'] == '4bit':
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            print("✅ Teacher quantization: 4-bit NF4")
        except ImportError:
            print("⚠️  bitsandbytes not available, loading in full precision")
    
    # 加载 Teacher
    print(f"\n🔹 Loading Teacher: {teacher_path}")
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_path,
        quantization_config=quantization_config,
        device_map="auto" if device.type == "cuda" else None,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    teacher.eval()
    print(f"   Device: {next(teacher.parameters()).device}")
    print(f"   Dtype: {next(teacher.parameters()).dtype}")
    
    # 加载 Student
    print(f"\n🔹 Loading Student: {student_path}")
    student = AutoModelForCausalLM.from_pretrained(
        student_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    student.train()
    print(f"   Device: {next(student.parameters()).device}")
    print(f"   Dtype: {next(student.parameters()).dtype}")
    
    # 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(teacher_path),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("\n✅ All models loaded successfully")
    
    return teacher, student, tokenizer


def detect_kv_dimensions_adaptive(teacher, student, env_adapter, kv_extractor):
    """
    动态检测 KV 维度（环境自适应）
    """
    print("\n" + "="*70)
    print("🔍 Dynamic KV Dimension Detection")
    print("="*70)
    
    device = env_adapter.get_device()
    
    # 创建测试输入
    test_input = torch.randint(0, 1000, (1, 32)).to(device)
    
    # Teacher KV 检测
    print("\n🔹 Analyzing Teacher KV Cache...")
    with torch.no_grad():
        t_out = teacher(test_input, use_cache=True)
        t_kv = kv_extractor.extract_kv(
            t_out.past_key_values,
            model_name="teacher",
            debug=True
        )
    
    teacher_dim = t_kv.shape[-1]
    teacher_config_dim = teacher.config.hidden_size
    
    print(f"\n   Config dimension: {teacher_config_dim}")
    print(f"   Detected dimension: {teacher_dim}")
    if teacher_dim != teacher_config_dim:
        print(f"   ⚠️  Dimension mismatch! Using detected: {teacher_dim}")
    else:
        print(f"   ✅ Dimensions match")
    
    # Student KV 检测
    print("\n🔹 Analyzing Student KV Cache...")
    with torch.no_grad():
        s_out = student(test_input, use_cache=True)
        s_kv = kv_extractor.extract_kv(
            s_out.past_key_values,
            model_name="student",
            debug=True
        )
    
    student_dim = s_kv.shape[-1]
    student_config_dim = student.config.hidden_size
    
    print(f"\n   Config dimension: {student_config_dim}")
    print(f"   Detected dimension: {student_dim}")
    if student_dim != student_config_dim:
        print(f"   ⚠️  Dimension mismatch! Using detected: {student_dim}")
    else:
        print(f"   ✅ Dimensions match")
    
    print("\n" + "="*70)
    print(f"✅ Detection Complete")
    print(f"   Teacher: {teacher_dim}D")
    print(f"   Student: {student_dim}D")
    print("="*70)
    
    return teacher_dim, student_dim


def initialize_projector_adaptive(teacher_dim, student_dim, env_adapter):
    """
    自适应初始化 Projector
    """
    print("\n" + "="*70)
    print("🔧 Initializing Adaptive Projector")
    print("="*70)
    
    device = env_adapter.get_device()
    dtype = env_adapter.get_dtype()
    
    # 使用检测到的维度初始化
    projector = KVDimensionProjector(
        teacher_configs={
            "teacher": {"d_model": teacher_dim}
        },
        student_d_model=student_dim,
        mlp_ratio=1.0,
        dropout=0.1,
    ).to(device).to(dtype)
    
    # 统计参数
    total_params = sum(p.numel() for p in projector.parameters())
    trainable_params = sum(p.numel() for p in projector.parameters() if p.requires_grad)
    
    print(f"\n✅ Projector initialized:")
    print(f"   Architecture: {teacher_dim} -> {student_dim}")
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable params: {trainable_params:,}")
    print(f"   Device: {device}")
    print(f"   Dtype: {dtype}")
    
    return projector


def load_dataset_adaptive(env_adapter, tokenizer):
    """
    自适应加载数据集
    """
    print("\n" + "="*70)
    print("📊 Loading Dataset (Environment-Adaptive)")
    print("="*70)
    
    data_path = env_adapter.paths['data']
    dataset_path = data_path / "gsm8k" / "train"
    
    if not dataset_path.exists():
        print(f"⚠️  Dataset not found at {dataset_path}")
        print("   Please download dataset first or set KAVA_DATA_PATH")
        sys.exit(1)
    
    print(f"\n📁 Loading from: {dataset_path}")
    dataset = load_from_disk(str(dataset_path))
    
    print(f"✅ Dataset loaded: {len(dataset)} samples")
    
    return dataset


def train_adaptive():
    """
    环境自适应训练主函数
    """
    print("\n" + "="*70)
    print("  KAVA Training (Environment-Adaptive)")
    print("  支持本地、HPC、云平台自动适配")
    print("="*70)
    
    # ================================================================
    # Step 1: 环境检测与配置
    # ================================================================
    env_adapter = create_environment_adapter()
    training_config = env_adapter.get_training_config()
    
    device = training_config['device']
    dtype = training_config['dtype']
    batch_size = training_config['batch_size']
    grad_accum = training_config['gradient_accumulation_steps']
    
    print(f"\n📋 Training Configuration:")
    print(f"   Batch size: {batch_size}")
    print(f"   Gradient accumulation: {grad_accum}")
    print(f"   Effective batch size: {training_config['effective_batch_size']}")
    print(f"   Mixed precision: {training_config['mixed_precision']}")
    
    # ================================================================
    # Step 2: 创建 KV 提取器
    # ================================================================
    kv_extractor = create_kv_extractor(
        aggregation_method=GLOBAL_CONFIG['kv_aggregation_method'],
        use_all_layers=GLOBAL_CONFIG['use_all_layers'],
    )
    kv_extractor.print_extraction_info()
    
    # ================================================================
    # Step 3: 加载模型
    # ================================================================
    teacher, student, tokenizer = load_models_adaptive(env_adapter)
    
    # ================================================================
    # Step 4: 动态检测 KV 维度
    # ================================================================
    teacher_dim, student_dim = detect_kv_dimensions_adaptive(
        teacher, student, env_adapter, kv_extractor
    )
    
    # ================================================================
    # Step 5: 初始化 Projector
    # ================================================================
    projector = initialize_projector_adaptive(teacher_dim, student_dim, env_adapter)
    
    # ================================================================
    # Step 6: 加载数据集
    # ================================================================
    dataset = load_dataset_adaptive(env_adapter, tokenizer)
    
    def collate_fn(batch):
        texts = [item['question'] + "\n" + item['answer'] for item in batch]
        encoded = tokenizer(
            texts,
            max_length=GLOBAL_CONFIG['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoded['input_ids'].to(device), encoded['attention_mask'].to(device)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # HPC 环境可能需要调整
    )
    
    # ================================================================
    # Step 7: 初始化优化器和损失函数
    # ================================================================
    optimizer = torch.optim.AdamW([
        {'params': student.parameters(), 'lr': GLOBAL_CONFIG['learning_rate_student']},
        {'params': projector.parameters(), 'lr': GLOBAL_CONFIG['learning_rate_projector']},
    ], weight_decay=GLOBAL_CONFIG['weight_decay'])
    
    loss_fn = MercatorKVLoss(
        alpha=GLOBAL_CONFIG['loss_alpha'],
        beta=GLOBAL_CONFIG['loss_beta'],
    )
    
    print(f"\n✅ Optimizer initialized")
    print(f"   Student LR: {GLOBAL_CONFIG['learning_rate_student']}")
    print(f"   Projector LR: {GLOBAL_CONFIG['learning_rate_projector']}")
    
    # ================================================================
    # Step 8: 训练循环
    # ================================================================
    print("\n" + "="*70)
    print("🎯 Starting Training")
    print("="*70)
    
    global_step = 0
    best_cossim = 0.0
    
    # 输出路径
    output_dir = env_adapter.paths['output']
    checkpoint_dir = env_adapter.paths['checkpoints']
    
    progress_bar = tqdm(total=GLOBAL_CONFIG['max_steps'], desc="Training")
    
    try:
        for epoch in range(100):  # 足够多的 epoch
            for input_ids, attention_mask in dataloader:
                # Teacher Forward
                with torch.no_grad():
                    t_out = teacher(input_ids, attention_mask=attention_mask, use_cache=True)
                    t_kv = kv_extractor.extract_kv(t_out.past_key_values, model_name="teacher")
                    t_kv = t_kv.to(dtype)
                
                # Student Forward
                s_out = student(input_ids, attention_mask=attention_mask, use_cache=True)
                s_kv = kv_extractor.extract_kv(s_out.past_key_values, model_name="student")
                s_kv = s_kv.to(dtype)
                
                # Projector
                t_proj, _ = projector.project_teacher_kv("teacher", t_kv, t_kv)
                
                # Loss
                loss, metrics = loss_fn(s_kv, t_proj)
                loss = loss / grad_accum
                loss.backward()
                
                # 梯度累积
                if (global_step + 1) % grad_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                # 更新进度
                global_step += 1
                cossim = metrics.get('cosine_similarity', 0.0)
                
                # 状态标记
                if cossim >= 0.90:
                    status = "✅ Excellent"
                elif cossim >= 0.70:
                    status = "📈 Good"
                elif cossim >= 0.50:
                    status = "⚠️  Learning"
                else:
                    status = "🔄 Adapting"
                
                progress_bar.update(1)
                progress_bar.set_postfix({
                    'Loss': f'{loss.item()*grad_accum:.4f}',
                    'CosSim': f'{cossim:.4f}',
                    'Status': status
                })
                
                # 保存最佳模型
                if cossim > best_cossim:
                    best_cossim = cossim
                
                # 定期保存
                if global_step % GLOBAL_CONFIG['save_interval'] == 0:
                    save_path = checkpoint_dir / f"checkpoint_step_{global_step}"
                    save_path.mkdir(parents=True, exist_ok=True)
                    
                    torch.save({
                        'step': global_step,
                        'student_state_dict': student.state_dict(),
                        'projector_state_dict': projector.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_cossim': best_cossim,
                    }, save_path / "checkpoint.pt")
                    
                    print(f"\n💾 Checkpoint saved: {save_path}")
                
                # 达到最大步数
                if global_step >= GLOBAL_CONFIG['max_steps']:
                    break
            
            if global_step >= GLOBAL_CONFIG['max_steps']:
                break
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        
        # 保存应急检查点
        emergency_path = checkpoint_dir / "emergency_checkpoint"
        emergency_path.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'step': global_step,
            'student_state_dict': student.state_dict(),
            'projector_state_dict': projector.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_cossim': best_cossim,
        }, emergency_path / "checkpoint.pt")
        
        print(f"💾 Emergency checkpoint saved: {emergency_path}")
    
    finally:
        progress_bar.close()
    
    # ================================================================
    # 训练完成
    # ================================================================
    print("\n" + "="*70)
    print("  Training Complete!")
    print("="*70)
    print(f"\n📊 Final Results:")
    print(f"   Total steps: {global_step}")
    print(f"   Best CosSim: {best_cossim:.4f}")
    print(f"   Output directory: {output_dir}")
    print(f"   Checkpoint directory: {checkpoint_dir}")
    
    # 保存最终模型
    final_path = output_dir / "final_model"
    final_path.mkdir(parents=True, exist_ok=True)
    
    student.save_pretrained(final_path / "student")
    torch.save(projector.state_dict(), final_path / "projector.pt")
    
    print(f"\n✅ Final model saved: {final_path}")


if __name__ == "__main__":
    train_adaptive()

