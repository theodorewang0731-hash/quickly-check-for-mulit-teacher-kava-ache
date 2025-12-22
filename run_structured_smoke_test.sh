#!/bin/bash
# Structured mode smoke test (10 steps)
# 避免使用 rkv_official，使用 full kv + 高级对齐功能

python experiments/train_with_kv.py \
  --model_name /home/share/models/Qwen2.5-1.5B \
  --teacher_name /home/share/models/Qwen2.5-7B \
  --kv_method full \
  --use_cka_layer_mapping \
  --use_segment_resampling \
  --kv_weight 1.0 \
  --cka_weight 0.1 \
  --subset_size 100 \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_length 512 \
  --epochs 1 \
  --output_dir outputs/structured_smoke_v4 \
  --logging_steps 1 \
  --save_steps 10 \
  --device cuda \
  --trust_remote_code
