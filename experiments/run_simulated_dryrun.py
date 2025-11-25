"""
Run a fully-local simulated dry-run of the KV training pipeline.

This script does NOT load Hugging Face models or datasets. Instead it:
- creates small random input_ids and attention_mask
- creates a tiny student model (nn.GRU + linear) that returns logits and hidden states
- simulates teacher past_key_values as random tensors
- runs one training epoch with CE + KV loss using existing experiments.kv_loss utilities

Purpose: fast CI-like check that the pipeline (loss composition, shapes, backward) works end-to-end.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from experiments.kv_loss import align_teacher_kv_to_student, compute_kv_loss, shuffled_kv

class TinyStudent(nn.Module):
    def __init__(self, vocab_size=100, hidden=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        x = self.embed(input_ids)
        out, _ = self.rnn(x)
        logits = self.out(out)
        if output_hidden_states:
            return type('O', (), {'logits': logits, 'hidden_states': [out]})
        return type('O', (), {'logits': logits})

def make_simulated_teacher_pkv(batch, n_layer=2, n_head=2, seq_len=12, head_dim=8):
    layers = []
    for _ in range(n_layer):
        k = np.random.randn(batch, n_head, seq_len, head_dim).astype(np.float32)
        v = np.random.randn(batch, n_head, seq_len, head_dim).astype(np.float32)
        layers.append((k, v))
    return tuple(layers)

def main():
    device = torch.device('cpu')
    batch = 2
    seq_len = 12
    vocab = 100

    # tiny synthetic batch
    input_ids = torch.randint(0, vocab, (batch, seq_len), dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    labels = input_ids.clone()

    student = TinyStudent(vocab_size=vocab, hidden=32).to(device)
    student.train()
    optim_ = optim.Adam(student.parameters(), lr=1e-3)

    teacher_pkv = make_simulated_teacher_pkv(batch=batch, n_layer=2, n_head=2, seq_len=8, head_dim=16)

    # run one training step
    out = student(input_ids, attention_mask=attention_mask, output_hidden_states=True)
    logits = out.logits  # (batch, seq_len, vocab)
    student_hidden = out.hidden_states[-1]  # (batch, seq_len, hidden)

    # CE loss
    ce_f = torch.nn.CrossEntropyLoss()
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    ce = ce_f(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # compress/align teacher kv via simple full->align
    # use experiments.kv_loss.align_teacher_kv_to_student
    kv_losses = []
    for layer in teacher_pkv:
        tk, sv = layer
        # align to student last tokens
        tk_t, student_seg = align_teacher_kv_to_student((tk, sv), student_hidden, method='right_crop')
        l = compute_kv_loss(student_seg, tk_t, loss_type='smooth_l1')
        kv_losses.append(l)
    kv_loss = torch.stack(kv_losses).mean()

    total = ce + 1.0 * kv_loss
    optim_.zero_grad()
    total.backward()
    optim_.step()

    print(f"Simulated dry-run finished. CE={ce.item():.6f}, KV={kv_loss.item():.6f}")

if __name__ == '__main__':
    main()
