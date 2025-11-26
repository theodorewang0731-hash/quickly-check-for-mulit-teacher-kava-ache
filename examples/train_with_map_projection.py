"""
Training Integration Example: Elastic Bottleneck + Map Projection
==================================================================

This example shows how to integrate:
1. Elastic Bottleneck (physical dimension alignment)
2. Map Projection Loss (semantic direction alignment)

into your training loop.

Author: Quick Check Team
Date: 2025-01-18
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import components
import sys
sys.path.append(".")
from experiments.kv_dimension_projector import KVDimensionProjector
from src.losses import MercatorKVLoss, HybridKVLoss


def train_with_map_projection(
    student_model,
    teacher_model,
    dataloader,
    device="cuda",
    num_epochs=3,
    use_hybrid=False
):
    """
    Training loop with Map Projection loss.
    
    Args:
        student_model: Student LLM (e.g., Qwen2-1.5B)
        teacher_model: Teacher LLM (e.g., Qwen2-14B)
        dataloader: DataLoader with training batches
        device: cuda or cpu
        num_epochs: Number of training epochs
        use_hybrid: If True, use hybrid loss (Mercator + MSE)
    """
    
    print("\n" + "="*60)
    print("Training with Map Projection Loss")
    print("="*60)
    
    # ========================================
    # Step 1: Initialize Projector
    # ========================================
    # Elastic Bottleneck handles dimension alignment
    # Teacher d_model -> Student d_model
    
    teacher_configs = {
        "Qwen2-14B": {
            "d_model": 5120,  # Qwen2-14B hidden size
            "num_layers": 40
        }
    }
    
    student_d_model = 1536  # Qwen2-1.5B hidden size
    
    projector = KVDimensionProjector(
        teacher_configs=teacher_configs,
        student_d_model=student_d_model,
        mlp_ratio=1.0,      # Standard configuration
        dropout=0.1,        # Prevent overfitting
        init_method="xavier",
        trainable=True
    ).to(device)
    
    print(f"\n[Projector Initialized]")
    print(f"  Parameters: {projector.count_parameters():,}")
    
    # ========================================
    # Step 2: Initialize Loss Function
    # ========================================
    
    if use_hybrid:
        # Hybrid: Combine Mercator (80%) + MSE (20%)
        # Useful for gradual transition
        loss_fn = HybridKVLoss(
            mercator_weight=0.8,
            mse_weight=0.2,
            beta=0.01  # Weak magnitude constraint
        ).to(device)
        print(f"\n[Using Hybrid Loss: 80% Mercator + 20% MSE]")
    else:
        # Pure Mercator: Direction-only alignment
        loss_fn = MercatorKVLoss(
            alpha=1.0,   # Direction weight
            beta=0.01    # Weak magnitude constraint (optional)
        ).to(device)
        print(f"\n[Using Pure Mercator Loss]")
    
    # ========================================
    # Step 3: Configure Optimizer
    # ========================================
    # Use differential learning rates
    
    optimizer = optim.AdamW([
        {
            'params': student_model.parameters(),
            'lr': 5e-5,       # Student: small LR (fine-tuning)
            'weight_decay': 0.01
        },
        {
            'params': projector.parameters(),
            'lr': 1e-3,       # Projector: large LR (learning from scratch)
            'weight_decay': 0.01
        }
    ])
    
    print(f"\n[Optimizer Configured]")
    print(f"  Student LR:   5e-5")
    print(f"  Projector LR: 1e-3")
    
    # ========================================
    # Step 4: Training Loop
    # ========================================
    
    student_model.train()
    projector.train()
    teacher_model.eval()  # Teacher is frozen
    
    global_step = 0
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        
        epoch_loss = 0.0
        epoch_cos_sim = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # ========================================
            # A. Extract Teacher KV Cache
            # ========================================
            with torch.no_grad():
                teacher_outputs = teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=True
                )
                
                # Extract and flatten KV cache
                # Assuming KV cache shape: [B, num_layers, num_heads, seq_len, head_dim]
                teacher_kv = extract_and_flatten_kv(teacher_outputs)
                # Result: [B, num_layers, seq_len, d_model]
            
            # ========================================
            # B. Extract Student KV Cache
            # ========================================
            student_outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=True
            )
            
            student_kv = extract_and_flatten_kv(student_outputs)
            # Result: [B, num_layers, seq_len, d_model]
            
            # ========================================
            # C. Physical Projection (Elastic Bottleneck)
            # ========================================
            # Project teacher KV to student dimension space
            # This handles: LayerNorm + MLP + Dimension alignment
            
            teacher_kv_projected, _ = projector.project_teacher_kv(
                teacher_name="Qwen2-14B",
                teacher_K=teacher_kv,
                teacher_V=teacher_kv
            )
            # Result: [B, num_layers, seq_len, student_d_model]
            
            # ========================================
            # D. Semantic Projection (Map Loss)
            # ========================================
            # Compare semantic directions on unit sphere
            
            loss, metrics = loss_fn(student_kv, teacher_kv_projected)
            
            # ========================================
            # E. Backpropagation
            # ========================================
            
            optimizer.zero_grad()
            loss.backward()
            
            # Optional: Gradient clipping
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(projector.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # ========================================
            # F. Logging
            # ========================================
            
            epoch_loss += loss.item()
            epoch_cos_sim += metrics['cos_sim']
            
            if global_step % 10 == 0:
                print(f"  Step {global_step:04d} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Cos Sim: {metrics['cos_sim']:.4f} | "
                      f"Mag Ratio: {metrics.get('mag_ratio', 0):.2f}")
                
                # Key monitoring metrics
                if metrics['cos_sim'] < 0.5:
                    print(f"    [WARNING] Low cosine similarity - check alignment")
                elif metrics['cos_sim'] > 0.95:
                    print(f"    [EXCELLENT] High semantic alignment achieved!")
            
            global_step += 1
        
        # Epoch summary
        avg_loss = epoch_loss / len(dataloader)
        avg_cos_sim = epoch_cos_sim / len(dataloader)
        
        print(f"\n[Epoch {epoch + 1} Summary]")
        print(f"  Avg Loss:    {avg_loss:.4f}")
        print(f"  Avg Cos Sim: {avg_cos_sim:.4f}")
        print(f"  Target:      Cos Sim > 0.95")
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    
    return student_model, projector


def extract_and_flatten_kv(model_outputs):
    """
    Extract and flatten KV cache from model outputs.
    
    This is a placeholder - actual implementation depends on
    your model's output format.
    
    Args:
        model_outputs: Model output with past_key_values
    
    Returns:
        kv_flat: [B, num_layers, seq_len, d_model]
    """
    # Example implementation (adjust based on your model)
    past_key_values = model_outputs.past_key_values
    
    # past_key_values is tuple of (key, value) for each layer
    # Each is [B, num_heads, seq_len, head_dim]
    
    keys = []
    for layer_kv in past_key_values:
        key = layer_kv[0]  # [B, num_heads, seq_len, head_dim]
        B, H, T, D_h = key.shape
        
        # Flatten heads: [B, H, T, D_h] -> [B, T, H*D_h]
        key_flat = key.transpose(1, 2).contiguous()
        key_flat = key_flat.view(B, T, H * D_h)
        keys.append(key_flat)
    
    # Stack layers: List[B, T, D] -> [B, L, T, D]
    kv_flat = torch.stack(keys, dim=1)
    
    return kv_flat


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Map Projection Training Integration Example")
    print("="*60)
    print("\nThis is a template showing how to integrate:")
    print("  1. Elastic Bottleneck (physical alignment)")
    print("  2. Map Projection Loss (semantic alignment)")
    print("\nKey monitoring metric: Cosine Similarity")
    print("  - Start: ~0.3-0.5 (random)")
    print("  - Target: >0.95 (excellent alignment)")
    print("  - Perfect: 1.0 (identical direction)")
    print("\nRecommended hyperparameters:")
    print("  - MLP ratio: 1.0 (for 14B->1.5B)")
    print("  - Dropout: 0.1")
    print("  - Beta (magnitude): 0.01 (weak constraint)")
    print("  - Student LR: 5e-5")
    print("  - Projector LR: 1e-3")
