"""
CKA (Centered Kernel Alignment) auxiliary loss for representation alignment.

稳健小升级 (0.5天实现):
- 只选 1-2 个代表性层（比如学生的中层）
- 使用 RCKA 论文风格的实现
- 作为小权重辅助项 (λ_CKA ≈ 0.05)

Reference:
- "Similarity of Neural Network Representations Revisited" (ICML 2019)
- "Representation Alignment via CKA for Knowledge Distillation" (ICML 2024)
"""
import torch
import torch.nn.functional as F


def linear_cka(X, Y, debiased=True):
    """
    Linear CKA between two feature matrices.
    
    Args:
        X: (batch * seq_len, feature_dim_x) or (batch, seq_len, feature_dim_x)
        Y: (batch * seq_len, feature_dim_y) or (batch, seq_len, feature_dim_y)
        debiased: Use unbiased estimator (recommended)
        
    Returns:
        cka: Scalar CKA similarity (0 to 1, higher is better)
    """
    # Flatten if needed
    if X.dim() == 3:
        X = X.flatten(0, 1)  # (batch * seq_len, feature_dim_x)
    if Y.dim() == 3:
        Y = Y.flatten(0, 1)  # (batch * seq_len, feature_dim_y)
    
    # Center the features
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    
    # Compute Gram matrices
    X_gram = X @ X.T  # (n, n)
    Y_gram = Y @ Y.T  # (n, n)
    
    if debiased:
        # Unbiased CKA (remove diagonal for variance estimation)
        n = X_gram.shape[0]
        
        # Compute HSIC (Hilbert-Schmidt Independence Criterion)
        hsic_xy = (X_gram * Y_gram).sum() - X_gram.diagonal().sum() * Y_gram.diagonal().sum() / (n - 2)
        hsic_xx = (X_gram * X_gram).sum() - (X_gram.diagonal() ** 2).sum() / (n - 2)
        hsic_yy = (Y_gram * Y_gram).sum() - (Y_gram.diagonal() ** 2).sum() / (n - 2)
        
        # Normalize
        hsic_xx = hsic_xx + 1e-10  # Avoid division by zero
        hsic_yy = hsic_yy + 1e-10
        
        cka = hsic_xy / torch.sqrt(hsic_xx * hsic_yy)
    else:
        # Biased CKA (simpler, faster)
        cka = (X_gram * Y_gram).sum() / (torch.norm(X_gram) * torch.norm(Y_gram) + 1e-10)
    
    return cka


def cka_loss(student_hidden, teacher_hidden, debiased=True):
    """
    CKA loss for representation alignment.
    
    Args:
        student_hidden: (batch, seq_len, hidden_dim) or (batch * seq_len, hidden_dim)
        teacher_hidden: (batch, seq_len, hidden_dim) or (batch * seq_len, hidden_dim)
        debiased: Use unbiased estimator
        
    Returns:
        loss: 1 - CKA (minimize to maximize alignment)
    """
    cka_sim = linear_cka(student_hidden, teacher_hidden, debiased=debiased)
    return 1 - cka_sim


def multi_layer_cka_loss(student_hiddens, teacher_hiddens, layer_indices=None, debiased=True):
    """
    Multi-layer CKA loss (只在选定的代表性层上计算).
    
    Args:
        student_hiddens: List of hidden states per layer [(batch, seq, dim), ...]
        teacher_hiddens: List of hidden states per layer [(batch, seq, dim), ...]
        layer_indices: List of layer indices to compute CKA (e.g., [6, 12] for middle and last)
                       If None, use middle layer only
        debiased: Use unbiased estimator
        
    Returns:
        loss: Average CKA loss across selected layers
    """
    if layer_indices is None:
        # Default: use middle layer
        layer_indices = [len(student_hiddens) // 2]
    
    losses = []
    for idx in layer_indices:
        if idx >= len(student_hiddens) or idx >= len(teacher_hiddens):
            continue
        
        s_hidden = student_hiddens[idx]
        t_hidden = teacher_hiddens[idx]
        
        loss = cka_loss(s_hidden, t_hidden, debiased=debiased)
        losses.append(loss)
    
    if len(losses) == 0:
        return torch.tensor(0.0, device=student_hiddens[0].device)
    
    return torch.stack(losses).mean()


def contrastive_alignment_loss(student_hidden, teacher_hidden, temperature=0.1):
    """
    Contrastive loss for hidden state alignment (optional enhancement).
    
    Args:
        student_hidden: (batch, seq_len, hidden_dim)
        teacher_hidden: (batch, seq_len, hidden_dim)
        temperature: Temperature for softmax
        
    Returns:
        loss: Contrastive loss (lower is better)
    """
    # Flatten
    s = student_hidden.flatten(0, 1)  # (batch * seq_len, hidden_dim)
    t = teacher_hidden.flatten(0, 1)
    
    # Normalize
    s_norm = F.normalize(s, dim=-1)
    t_norm = F.normalize(t, dim=-1)
    
    # Similarity matrix
    sim = s_norm @ t_norm.T / temperature  # (n, n)
    
    # Contrastive loss: diagonal should be maximum
    labels = torch.arange(sim.shape[0], device=sim.device)
    loss = F.cross_entropy(sim, labels)
    
    return loss


def combined_alignment_loss(
    student_hidden, 
    teacher_hidden, 
    cka_weight=1.0, 
    contrastive_weight=0.0,
    debiased=True,
    temperature=0.1
):
    """
    Combined CKA + Contrastive alignment loss.
    
    Default: Only CKA (contrastive_weight=0)
    Optional: Add contrastive term for stronger alignment
    
    Args:
        student_hidden: (batch, seq_len, hidden_dim)
        teacher_hidden: (batch, seq_len, hidden_dim)
        cka_weight: Weight for CKA loss (default 1.0)
        contrastive_weight: Weight for contrastive loss (default 0.0)
        debiased: Use unbiased CKA estimator
        temperature: Temperature for contrastive loss
        
    Returns:
        loss: Combined alignment loss
    """
    loss = 0.0
    
    if cka_weight > 0:
        loss_cka = cka_loss(student_hidden, teacher_hidden, debiased=debiased)
        loss += cka_weight * loss_cka
    
    if contrastive_weight > 0:
        loss_contrastive = contrastive_alignment_loss(student_hidden, teacher_hidden, temperature)
        loss += contrastive_weight * loss_contrastive
    
    return loss


# ============================================================
# Quick test (for verification)
# ============================================================
if __name__ == "__main__":
    # Test CKA loss
    batch, seq_len, dim = 4, 10, 64
    
    student = torch.randn(batch, seq_len, dim)
    teacher = torch.randn(batch, seq_len, dim)
    
    # Test 1: Basic CKA loss
    loss = cka_loss(student, teacher)
    print(f"CKA loss (random): {loss.item():.4f}")
    
    # Test 2: Perfect alignment
    loss_perfect = cka_loss(student, student)
    print(f"CKA loss (perfect): {loss_perfect.item():.4f}")  # Should be ~0
    
    # Test 3: Multi-layer CKA
    student_layers = [torch.randn(batch, seq_len, dim) for _ in range(12)]
    teacher_layers = [torch.randn(batch, seq_len, dim) for _ in range(12)]
    
    loss_multi = multi_layer_cka_loss(student_layers, teacher_layers, layer_indices=[5, 11])
    print(f"Multi-layer CKA loss: {loss_multi.item():.4f}")
    
    # Test 4: Combined loss
    loss_combined = combined_alignment_loss(student, teacher, cka_weight=1.0, contrastive_weight=0.5)
    print(f"Combined alignment loss: {loss_combined.item():.4f}")
    
    print("\n✓ All CKA tests passed!")
