# Shape Verification Test Results
## Mock Data Testing (No Real Models)

**Date**: 2025-01-26  
**Test File**: `tests/check_shapes.py`  
**Status**: ✅ **ALL TESTS PASSED**

---

## 测试目的

验证完整的四维对齐流程在**不加载真实大模型**的情况下，shape 转换是否正确。

使用 `torch.randn` 生成假数据模拟真实场景：
- Teacher: 70B 模型 (28层, 32头, 4096维)
- Student: 7B 模型 (12层, 2048维)

---

## 测试结果

### ✅ Test 1: Head Dimension Flattening
```
Input:  [B=2, L=28, H=32, T=80, d_head=128]
Output: [B=2, L=28, T=80, d_model=4096]
```
- ✓ Head 展平正确
- ✓ Flatten/Unflatten 可逆
- ✓ 数据无损

### ✅ Test 2: Layer Alignment (CKA Top-k)
```
Input:  Teacher KV per layer [B=2, T=80, d_model=4096]
Output: Aligned KV [B=2, T=80, d_model=4096]
```
- ✓ CKA 层映射构建成功
- ✓ Top-2 加权组合正确
- ✓ Shape 保持不变

### ✅ Test 3: Time Dimension Resampling
```
Input:  [B=2, T_teacher=80, d_model=4096]
Output: [B=2, T_student=50, d_model=4096]
```
- ✓ Global 重采样工作正常
- ✓ Segment-aware 重采样工作正常
- ✓ 线性插值平滑

### ✅ Test 4: Dimension Projection
```
Input:  [B=2, L=1, T=50, d_teacher=4096]
Output: [B=2, L=1, T=50, d_student=2048]
```
- ✓ 可学习投影正确
- ✓ 参数量：16,777,216 (1677万)
- ✓ Xavier 初始化成功

### ✅ Test 5: Complete Pipeline (End-to-End)
```
Input:  Teacher KV [B=2, L=28, H=32, T=80, d_head=128]
        Format: [Batch, Layers, Heads, SeqLen, HeadDim]

Step 1: Flatten Heads
        → [B=2, L=28, T=80, d_model=4096]

Step 2: Layer Alignment (Student L5 → Teacher L25+L9)
        → [B=2, T=80, d_model=4096]

Step 3: Time Resampling (80 → 50 tokens)
        → [B=2, T=50, d_model=4096]

Step 4: Dimension Projection (4096 → 2048)
        → [B=2, T=50, d_student=2048]

Output: Student KV [B=2, T=50, d_student=2048] ✓✓✓
```

### ✅ Test 6: Common Pitfalls & Edge Cases
- ✓ Non-contiguous tensor 处理 (transpose 后)
- ✓ Dimension order 验证 (最后一维是 d_model)
- ✓ Batch dimension 保持不变
- ✓ Single token (T=1) 边界情况
- ✓ Very long sequence (T=2048) 边界情况

---

## 关键发现

### 1. View vs Reshape 陷阱已避免
- 使用 `.contiguous()` 确保 tensor 连续性
- Transpose 操作后正确处理 memory layout

### 2. Dimension Order 正确
```python
# 正确的维度顺序
[Batch, Layers, SeqLen, d_model]  # 在层处理时
[Batch, SeqLen, d_model]           # 单层处理时
```

### 3. Shape 转换流程
```
[B, L, H, T, d_head]  # 原始 KV (with heads)
    ↓ flatten_kv_heads()
[B, L, T, H*d_head]   # 展平后
    ↓ layer_alignment()
[B, T, d_teacher]     # 层对齐后
    ↓ time_resampling()
[B, T_student, d_teacher]  # 时间对齐后
    ↓ dimension_projection()
[B, T_student, d_student]  # 最终输出 ✓
```

---

## 性能数据

### 参数量统计
- **Dimension Projector**: 16,777,216 参数 (单教师 70B→7B)
  - W_K: 4096 × 2048 = 8,388,608
  - W_V: 4096 × 2048 = 8,388,608

### 内存占用估算 (Float32)
```
Teacher KV (single layer):
  [B=8, H=32, T=100, d_head=128]
  = 8 × 32 × 100 × 128 × 4 bytes
  = 13.1 MB per layer
  = 366.8 MB for 28 layers

Student KV (single layer):
  [B=8, T=100, d_student=2048]
  = 8 × 100 × 2048 × 4 bytes
  = 6.6 MB per layer
  = 79.2 MB for 12 layers
```

---

## 结论

✅ **所有 Shape 验证通过！**

完整的四维对齐流程（Head Flatten → Layer Alignment → Time Resampling → Dimension Projection）在 mock data 测试下运行正确，可以安全集成到真实模型训练中。

### 下一步
1. ✅ Mock data 测试完成
2. ⏭️ 集成到 `train_with_kv.py`
3. ⏭️ 使用真实模型进行小规模验证 (1个 batch)
4. ⏭️ 运行完整训练实验

---

**测试命令**:
```bash
python tests/check_shapes.py
```

**输出**: 所有测试通过，无错误 ✓
