"""
run_kv_demo.py

演示如何从 GPT-2 提取 past_key_values（对单条样本），并对比 Full / RightCrop / R-KV 压缩的输出形状与重构误差（diagnostic）。

用法示例：
  python experiments/run_kv_demo.py --model_name gpt2 --sample_index 0 --subset_size 5 --target_len 8

"""
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from experiments.kv_utils import full_kv, right_crop_kv, rkv_greedy, reconstruct_from_compressed

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="gpt2")
    p.add_argument("--subset_size", type=int, default=5)
    p.add_argument("--sample_index", type=int, default=0)
    p.add_argument("--target_len", type=int, default=8)
    return p.parse_args()

def extract_past_kv(model, tokenizer, text, device='cpu'):
    inputs = tokenizer(text, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    pkv = getattr(outputs, 'past_key_values', None)
    if pkv is None:
        # build a simulated pkv using hidden states shape
        # We'll return a tuple of layers of (k, v) with small random numbers
        n_layer = model.config.n_layer
        n_head = model.config.n_head
        head_dim = model.config.hidden_size // n_head
        seq_len = inputs['input_ids'].shape[-1]
        batch = inputs['input_ids'].shape[0]
        simulated = []
        for _ in range(n_layer):
            k = np.random.randn(batch, n_head, seq_len, head_dim).astype(np.float32)
            v = np.random.randn(batch, n_head, seq_len, head_dim).astype(np.float32)
            simulated.append((k, v))
        return tuple(simulated)
    # convert tensors to numpy
    pkv_np = []
    for (k, v) in pkv:
        pkv_np.append((k.cpu().numpy(), v.cpu().numpy()))
    return tuple(pkv_np)

def mse(a, b):
    return float(((a - b) ** 2).mean())

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

    ds = load_dataset('gsm8k', 'main', split='train')
    sample = ds[args.sample_index]
    text = sample['question'] if 'question' in sample else sample.get('problem', '')

    print('Extracting past_key_values for sample:', text[:200])
    pkv = extract_past_kv(model, tokenizer, text, device=device)

    print('Original per-layer shapes:')
    for i, (k, v) in enumerate(pkv):
        print(f' layer {i}: k {np.array(k).shape}, v {np.array(v).shape}')

    full = full_kv(pkv)
    rc = right_crop_kv(pkv, args.target_len)
    rkv = rkv_greedy(pkv, args.target_len, lambda_param=0.1)

    print('\nCompressed shapes (target_len=%d):' % args.target_len)
    for i, ((fk, fv), (rk, rv), (gk, gv)) in enumerate(zip(full, rc, rkv)):
        print(f' layer {i}: full k {np.array(fk).shape}, rightcrop k {np.array(rk).shape}, rkv k {np.array(gk).shape}')

    # compute simple reconstruction MSE for keys per layer
    print('\nReconstruction MSE (keys) diagnostics:')
    for i, (fk, _) in enumerate(full):
        fk = np.array(fk)
        rc_k = reconstruct_from_compressed(fk, rc[i][0], method='right_crop')
        rkv_k = reconstruct_from_compressed(fk, rkv[i][0], method='rkv')
        mse_rc = mse(fk, rc_k)
        mse_rkv = mse(fk, rkv_k)
        print(f' layer {i}: right_crop_mse={mse_rc:.6f}, rkv_mse={mse_rkv:.6f}')

    print('\nDemo complete.')

if __name__ == '__main__':
    main()
