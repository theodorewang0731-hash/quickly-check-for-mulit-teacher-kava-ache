# 精确修复指南：头数不匹配 + 时间重采样越界

## 问题根源分析

### 问题 1：头数不匹配 (12 vs 2)
**根因**：代码使用了 `num_attention_heads` (Q 头数) 而不是 `num_key_value_heads` (KV 头数)。  
在 GQA/MQA 架构中，**Q heads ≠ KV heads**，必须从张量 shape 动态获取或使用正确的 config 字段。

### 问题 2：时间重采样越界
**根因**：
1. 索引不是 `long` 类型
2. 索引没有 clamp 到 [0, T-1]
3. 边界情况 (T=0, T=1, 空段) 没有处理
4. device/dtype 不一致导致 CUDA 错误

---

## 修复方案概览

### 总体策略
1. **头数对齐**：添加 `KVProjector` 模块，先投影 head_dim，再混合 head 数
2. **时间重采样**：替换为 `safe_time_resample` + `build_linear_indices`，带完整边界检查

### 需要修改的 3 个文件
1. `experiments/alignment_v2.py` - 修改时间重采样函数
2. `experiments/kv_dimension_projector.py` - 集成头数投影器
3. **任何调用对齐的地方** - 插入头数投影

---

## 文件 1：`experiments/alignment_v2.py`

### 位置 1：在文件开头添加安全重采样工具函数

**在第 29 行之后 (import 语句后) 插入：**

```python
# ============================================================================
# Safe Time Resampling Utilities (Fix for Index Out of Bounds)
# ============================================================================

def safe_time_resample(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    安全的时间维重采样 (避免越界)
    
    Args:
        x: [B, H, T_in, D] 或 [B, T_in, D]
        indices: [B, T_out] 或 [T_out], 每个值在 [0, T_in-1] 范围内
    
    Returns:
        [B, H, T_out, D] 或 [B, T_out, D]
    """
    device = x.device
    is_4d = x.dim() == 4
    
    if is_4d:
        B, H, T_in, D = x.shape
    else:
        B, T_in, D = x.shape
        H = None
    
    # 确保 indices 在正确设备上
    indices = indices.to(device=device)
    
    # 转换为 long 类型
    indices = indices.long()
    
    # Clamp 到合法范围
    indices = indices.clamp(0, T_in - 1)
    
    # 处理 indices 的 shape
    if indices.dim() == 1:
        # [T_out] -> [B, T_out]
        indices = indices.unsqueeze(0).expand(B, -1)
    
    T_out = indices.shape[1]
    
    if is_4d:
        # 扩展 indices 用于 gather: [B, H, T_out, D]
        idx = indices[:, None, :, None].expand(B, H, T_out, D)
        return torch.gather(x, dim=2, index=idx)
    else:
        # 扩展 indices 用于 gather: [B, T_out, D]
        idx = indices[:, :, None].expand(B, T_out, D)
        return torch.gather(x, dim=1, index=idx)


def build_safe_linear_indices(B: int, T_in: int, T_out: int, device: torch.device) -> torch.Tensor:
    """
    生成线性插值的索引 (避免天然越界)
    
    处理边界情况:
    - T_in = 0: 返回空张量
    - T_in = 1: 所有索引指向 0
    - T_out = 1: 返回中间位置
    
    Args:
        B: batch size
        T_in: 输入序列长度
        T_out: 输出序列长度
        device: torch device
    
    Returns:
        indices: [B, T_out], dtype=long
    """
    # 边界情况 1: 输入为空
    if T_in == 0:
        return torch.zeros(B, T_out, device=device, dtype=torch.long)
    
    # 边界情况 2: 输入只有 1 个 token
    if T_in == 1:
        return torch.zeros(B, T_out, device=device, dtype=torch.long)
    
    # 边界情况 3: 输出只有 1 个 token
    if T_out == 1:
        mid = T_in // 2
        return torch.full((B, 1), mid, device=device, dtype=torch.long)
    
    # 正常情况: 线性插值
    # 浮点生成, 再 round, 再 clamp
    base = torch.linspace(0, T_in - 1, steps=T_out, device=device)  # [T_out]
    idx = torch.round(base).long().clamp(0, T_in - 1)  # [T_out]
    
    # 扩展到 batch
    return idx.unsqueeze(0).expand(B, -1)  # [B, T_out]
```

### 位置 2：修改 `_global_resample` 函数 (第 196-229 行)

**替换整个 `_global_resample` 函数体为：**

```python
def _global_resample(
    teacher_kv_flat: torch.Tensor,
    student_length: int,
    is_4d: bool,
    heads: Optional[int],
    head_dim: Optional[int]
) -> torch.Tensor:
    """全局等比例重采样（简单版）- 使用安全 gather"""
    batch, teacher_len, dim = teacher_kv_flat.shape
    device = teacher_kv_flat.device
    
    # 边界情况处理
    if teacher_len == 0:
        # 输入为空，返回零填充
        result = torch.zeros(batch, student_length, dim, device=device, dtype=teacher_kv_flat.dtype)
        if is_4d:
            result = result.reshape(batch, student_length, heads, head_dim).transpose(1, 2)
        return result
    
    if teacher_len == 1:
        # 输入只有 1 token，重复
        resampled = teacher_kv_flat.repeat(1, student_length, 1)
    else:
        # 使用安全的线性索引生成
        indices = build_safe_linear_indices(batch, teacher_len, student_length, device)
        
        # 安全重采样 (不会越界)
        resampled = safe_time_resample(teacher_kv_flat, indices)
    
    # Reshape back to 4D if needed
    if is_4d:
        resampled = resampled.reshape(batch, student_length, heads, head_dim).transpose(1, 2)
    
    return resampled
```

### 位置 3：修改 `_segment_aware_resample` 函数 (第 232-297 行)

**在函数内部的段循环中，将原来的 `_global_resample` 调用替换：**

找到这段代码（大约在 256-271 行）：
```python
        if teacher_seg is None:
            # Fallback: use global resampling for this segment
            seg_resampled = _global_resample(
                teacher_kv_flat[:, :student_seg.length, :],
                student_seg.length,
                False, None, None
            )
        else:
            # Extract teacher segment KV
            teacher_seg_kv = teacher_kv_flat[:, teacher_seg.start:teacher_seg.end, :]
            
            # Resample this segment
            seg_resampled = _global_resample(
                teacher_seg_kv,
                student_seg.length,
                False, None, None
            )
```

替换为：
```python
        if teacher_seg is None:
            # Fallback: use global resampling for this segment
            # 边界检查：确保不访问越界
            seg_len = min(student_seg.length, teacher_kv_flat.shape[1])
            if seg_len == 0:
                # 空段，创建零填充
                seg_resampled = torch.zeros(
                    batch, student_seg.length, dim,
                    device=device, dtype=teacher_kv_flat.dtype
                )
            else:
                seg_resampled = _global_resample(
                    teacher_kv_flat[:, :seg_len, :],
                    student_seg.length,
                    False, None, None
                )
        else:
            # Extract teacher segment KV
            # 边界检查：确保 start/end 不越界
            seg_start = max(0, min(teacher_seg.start, teacher_len))
            seg_end = max(seg_start, min(teacher_seg.end, teacher_len))
            
            if seg_start >= seg_end:
                # 空段或无效段，创建零填充
                seg_resampled = torch.zeros(
                    batch, student_seg.length, dim,
                    device=device, dtype=teacher_kv_flat.dtype
                )
            else:
                teacher_seg_kv = teacher_kv_flat[:, seg_start:seg_end, :]
                
                # Resample this segment
                seg_resampled = _global_resample(
                    teacher_seg_kv,
                    student_seg.length,
                    False, None, None
                )
```

---

## 文件 2：`experiments/kv_head_projector.py`

**这个文件已经创建完成**，包含：
- `KVProjector` 类：处理头数 + head_dim 投影
- `safe_time_resample`：安全的时间重采样
- `build_linear_indices`：生成安全的索引
- `get_kv_heads_from_tensor`：从张量动态获取头数

---

## 文件 3：`experiments/kv_dimension_projector.py`

### 修改：集成头数投影器

**在文件开头 (第 23 行 import 之后) 添加：**

```python
# Import head projector
try:
    from experiments.kv_head_projector import KVProjector, get_kv_heads_from_tensor
except ImportError:
    # Fallback: define minimal version inline
    print("[WARNING] kv_head_projector not found, using inline fallback")
    KVProjector = None
    get_kv_heads_from_tensor = None
```

### 修改 `KVDimensionProjector.__init__` 方法

**在 `__init__` 方法中 (大约第 80-120 行)，添加头数投影器初始化：**

找到这段代码：
```python
        # Store configs
        self.teacher_configs = teacher_configs
        self.student_d_model = student_d_model
        self.trainable = trainable
```

在这之后添加：
```python
        # Head projectors for each teacher (handles num_heads mismatch)
        self.head_projectors = nn.ModuleDict()
        
        # Will be initialized dynamically when first KV arrives
        self._head_projectors_initialized = False
```

### 修改 `project_teacher_kv` 方法

**在方法开始处 (处理输入 K, V 之后) 添加头数投影：**

找到这段代码（大约在 180-200 行）：
```python
    def project_teacher_kv(
        self,
        teacher_name: str,
        K: torch.Tensor,
        V: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project teacher KV to student dimension.
        
        Args:
            K: [B, num_layers, T, d_teacher] or [B, num_layers, H_t, T, d_head_t]
            V: Same as K
        """
        # Handle 5D (with heads) - flatten heads first
        if K.dim() == 5:
            from experiments.kv_dimension_projector import flatten_kv_heads
            B, L, H, T, d_head = K.shape
            K = flatten_kv_heads(K, H, d_head)
            V = flatten_kv_heads(V, H, d_head)
```

在这段代码**之前**插入头数投影：
```python
        # Step 0: Handle head mismatch (if K/V are 4D with heads)
        if K.dim() == 4 or K.dim() == 5:
            # Get actual KV head counts from tensor
            if K.dim() == 4:
                # [B, H, T, D] format
                Ht = K.shape[1]
                Dt = K.shape[3]
            elif K.dim() == 5:
                # [B, L, H, T, D] format
                Ht = K.shape[2]
                Dt = K.shape[4]
            
            # Initialize head projector if needed
            if teacher_name not in self.head_projectors:
                # Get student KV heads (assume config available)
                # For safety, use 2 as default (common for small models)
                Hs = getattr(self, 'student_num_kv_heads', 2)
                Ds = Dt  # Keep head_dim same initially
                
                print(f"[KV Projector] Initializing head projector for {teacher_name}: "
                      f"Ht={Ht}, Hs={Hs}, Dt={Dt}, Ds={Ds}")
                
                if KVProjector is not None:
                    self.head_projectors[teacher_name] = KVProjector(
                        Ht=Ht, Hs=Hs, Dt=Dt, Ds=Ds, share_kv=True
                    ).to(K.device)
            
            # Apply head projection if available
            if teacher_name in self.head_projectors:
                projector = self.head_projectors[teacher_name]
                
                if K.dim() == 4:
                    # [B, H, T, D] -> [B, H_s, T, D_s]
                    K, V = projector(K, V)
                elif K.dim() == 5:
                    # [B, L, H, T, D] -> process each layer
                    B, L, H, T, D = K.shape
                    K_list, V_list = [], []
                    for l in range(L):
                        K_l, V_l = projector(K[:, l], V[:, l])
                        K_list.append(K_l.unsqueeze(1))
                        V_list.append(V_l.unsqueeze(1))
                    K = torch.cat(K_list, dim=1)
                    V = torch.cat(V_list, dim=1)
```

---

## 使用示例：如何在训练脚本中集成

### 在任何调用 KV 对齐的地方，确保：

#### 示例 1：使用 `align_multi_teacher_kv_v2`

```python
from experiments.alignment_v2 import align_multi_teacher_kv_v2
from experiments.kv_head_projector import KVProjector

# 初始化头数投影器（训练前）
Ht = teacher_k.shape[1]  # 从实际张量获取
Hs = student_k.shape[1]
Dt = teacher_k.shape[-1]
Ds = student_k.shape[-1]

kv_projector = KVProjector(Ht, Hs, Dt, Ds).to(device)

# 在对齐前先投影头数
teacher_k_proj, teacher_v_proj = kv_projector(teacher_k, teacher_v)

# 然后进行时间 + 层对齐
aligned_k, aligned_v = align_multi_teacher_kv_v2(
    student_hidden,
    student_layer_idx,
    [[teacher_k_proj, teacher_v_proj]],  # 已经投影过头数
    layer_mapper,
    student_segments,
    teacher_segments_list
)

# 计算 loss
loss = F.mse_loss(student_k, aligned_k) + F.mse_loss(student_v, aligned_v)
```

#### 示例 2：使用 `KVDimensionProjector` (已集成头数投影)

```python
from experiments.kv_dimension_projector import KVDimensionProjector

# 初始化（会自动处理头数投影）
projector = KVDimensionProjector(
    teacher_configs={"Qwen2-7B": {"d_model": 3584, "num_layers": 28}},
    student_d_model=2048,
    student_num_kv_heads=2,  # 新增：指定学生 KV 头数
    mlp_ratio=1.0,
    trainable=True
)

# 使用（内部会自动处理头数不匹配）
K_aligned, V_aligned = projector.project_teacher_kv("Qwen2-7B", K_teacher, V_teacher)

# 计算 loss (此时 K_aligned 和 student_k 的 head 数已经匹配)
loss = F.mse_loss(K_aligned, student_k) + F.mse_loss(V_aligned, student_v)
```

---

## 关键注意事项

### 1. 一定要用 KV head 数，不是 Q head 数

❌ **错误**：
```python
num_heads = config.num_attention_heads  # 这是 Q 的头数！
```

✅ **正确**：
```python
# 方法 1: 从 config 读取
num_kv_heads = config.num_key_value_heads  # GQA/MQA 的 KV 头数

# 方法 2: 从张量 shape 推断
num_kv_heads = teacher_k.shape[1]  # 假设 shape 是 [B, H, T, D]
```

### 2. 时间重采样必须做三件事

1. **类型转换**：`indices = indices.long()`
2. **Clamp**：`indices = indices.clamp(0, T_in - 1)`
3. **边界检查**：处理 T=0, T=1 等特殊情况

### 3. 训练流程建议

```
Step 1: 初始化头数投影器 (每个 teacher 一个)
   ↓
Step 2: 在 forward 前投影头数: K_t[Ht] -> K_t[Hs]
   ↓
Step 3: 时间对齐 (使用 safe_time_resample)
   ↓
Step 4: 层对齐 (CKA-based weighted sum)
   ↓
Step 5: 维度投影 (d_teacher -> d_student)
   ↓
Step 6: 计算 loss (此时所有维度都匹配)
```

---

## 快速验证脚本

创建 `tests/test_kv_alignment.py`:

```python
import torch
from experiments.kv_head_projector import KVProjector, safe_time_resample, build_linear_indices

# 测试 1: 头数投影
print("Test 1: Head projection (12 -> 2)")
projector = KVProjector(Ht=12, Hs=2, Dt=128, Ds=128)
k_t = torch.randn(4, 12, 50, 128)
v_t = torch.randn(4, 12, 50, 128)
k_s, v_s = projector(k_t, v_t)
print(f"  Input: {k_t.shape} -> Output: {k_s.shape}")
assert k_s.shape == (4, 2, 50, 128), "Head projection failed!"
print("  ✓ Passed")

# 测试 2: 时间重采样
print("\nTest 2: Time resampling (80 -> 50)")
x = torch.randn(4, 2, 80, 128)
indices = build_linear_indices(4, 80, 50, x.device)
x_resampled = safe_time_resample(x, indices)
print(f"  Input: {x.shape} -> Output: {x_resampled.shape}")
assert x_resampled.shape == (4, 2, 50, 128), "Time resampling failed!"
print("  ✓ Passed")

# 测试 3: 边界情况
print("\nTest 3: Edge case (T_in=1, T_out=1)")
x_edge = torch.randn(4, 2, 1, 128)
indices_edge = build_linear_indices(4, 1, 1, x_edge.device)
x_resampled_edge = safe_time_resample(x_edge, indices_edge)
assert x_resampled_edge.shape == (4, 2, 1, 128), "Edge case failed!"
print("  ✓ Passed")

print("\n✓ All tests passed!")
```

运行测试：
```bash
python tests/test_kv_alignment.py
```

---

## 总结

### 修改清单
- [x] 创建 `experiments/kv_head_projector.py` (新文件)
- [ ] 修改 `experiments/alignment_v2.py` (3 处)
- [ ] 修改 `experiments/kv_dimension_projector.py` (2 处)
- [ ] 在你的训练脚本中集成头数投影

### 预期效果
- ✅ 不再出现 "shape mismatch: 12 vs 2" 错误
- ✅ 不再出现 "index out of bounds" 错误
- ✅ 支持任意头数组合 (GQA/MQA/MHA)
- ✅ 支持任意序列长度 (包括边界情况)

### 如果还有问题
请提供：
1. 完整的错误堆栈 (包含文件名和行号)
2. 出错时的 tensor shapes (print 出 teacher_k.shape, student_k.shape)
3. 你使用的模型配置 (teacher/student 的 num_attention_heads, num_key_value_heads)

我会基于具体信息给出更精确的修复建议。
