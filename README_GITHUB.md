# Multi-Teacher KV Distillation with 4D Alignment

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

多教师 KV Cache 蒸馏用于推理能力迁移，采用创新的四维对齐策略。

## 🎯 项目目标

让小模型（Qwen2-1.5B）通过学习多个大模型教师的 KV cache 来获得更强的数学推理能力。

## ✨ 核心创新：四维对齐系统

### 1. 时间维对齐 (Time Alignment v2)
**问题**：多教师 CoT 长度不同（50步 vs 30步），硬 index 对齐导致语义错位

**方案**：Segment-aware 等比例重采样 + 线性插值
- 自动识别 Prompt/Reasoning/Answer 段
- 段内等比例映射：确保语义对应
- 线性插值平滑过渡

```python
u_i = i/(T_s-1) * (T_t-1)  # 等比例映射
KV_i = (1-λ) * KV_j + λ * KV_{j+1}  # 线性插值
```

### 2. 层维对齐 (Layer Alignment via CKA)
**问题**：固定比例映射不考虑表征相似性

**方案**：CKA 相似度 + Top-k 加权组合
- 预计算 CKA 相似度矩阵（100样本）
- 每个学生层选择最相似的 k 个教师层
- 按相似度权重加权融合

```python
S[k,l] = CKA(student_layer_k, teacher_layer_l)
KV_k = Σ_i β_i * KV_{teacher_layer_i}
```

### 3. Hidden 维度对齐 (Dimension Projection)
**问题**：教师 d_model (3584/4096) ≠ 学生 d_model (1536/2048)

**方案**：可学习线性投影（按教师粒度共享）
- 每个教师独立的 W_K, W_V 投影矩阵
- 所有层共享，减少参数量（~2100万参数）
- 与学生模型联合训练

```python
K_aligned = K_teacher · W_K
V_aligned = V_teacher · W_V
```

### 4. Head 维度对齐 (Head Flattening)
**问题**：教师 num_heads (28) ≠ 学生 num_heads (12)

**方案**：展平处理，暂不做细粒度 head 映射
- 展平：[B, L, H, T, d_head] → [B, L, T, H*d_head]
- 在 d_model 维度上做投影对齐
- 避免 head-to-head mapping 的复杂度爆炸

## 🔄 完整对齐流程

```
Teacher KV: [B, L_t=28, H_t=28, T_t=80, d_head=128]
    ↓
【Step 1: Head 展平】
    → [B, L_t=28, T_t=80, d_t=3584]
    ↓
【Step 2: 层对齐 - CKA Top-k】
    → [B, T_t=80, d_t=3584]
    ↓
【Step 3: 时间对齐 - Segment Resampling】
    → [B, T_s=50, d_t=3584]
    ↓
【Step 4: 维度投影 - Learnable Linear】
    → [B, T_s=50, d_s=2048]
    ↓
Student Target: [B, T_s=50, d_s=2048] ✓
```

## 🤖 模型与数据集

### 模型配置
- **学生**：Qwen/Qwen2-1.5B (1.5B参数, d_model=1536, 28层)
- **教师1**：Qwen/Qwen2-7B (7B参数, d_model=3584, 28层)
- **教师2**：Qwen/Qwen2-1.5B (辅助教师)

### 数据集
- **主训练**：openai/gsm8k (8,500+ 小学数学应用题)
- **补充**：SVAMP, StrategyQA, Math23K, MATH, ARC-Challenge, HotpotQA
- **评估**：GSM8K test, MATH500, BBH, GPQA, TruthfulQA, CMMLU, C-Eval

## 📦 安装

```bash
# 克隆仓库
git clone https://github.com/theodorewang0731-hash/quickly-check-for-mulit-teacher-kava-ache.git
cd quickly-check-for-mulit-teacher-kava-ache

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## 🚀 快速开始

### 1. 预计算 CKA 层映射

```bash
python experiments/precompute_layer_mapping.py \
    --student_model Qwen/Qwen2-1.5B \
    --teacher_model Qwen/Qwen2-7B \
    --dataset_name openai/gsm8k \
    --num_samples 100 \
    --output layer_mapping_qwen15b_7b.json
```

### 2. 训练（完整四维对齐）

```bash
python experiments/train_with_kv.py \
    --model_name Qwen/Qwen2-1.5B \
    --teacher_model Qwen/Qwen2-7B \
    --dataset_name openai/gsm8k \
    --use_cka_layer_mapping \
    --layer_mapping_path layer_mapping_qwen15b_7b.json \
    --use_segment_resampling \
    --kv_weight 1.0 \
    --codi_weight 0.5 \
    --epochs 2 \
    --batch_size 8 \
    --fp16 \
    --output_dir outputs/alignment_v2_full
```

### 3. 对比实验（Baseline vs v2）

```powershell
# Windows PowerShell
.\scripts\compare_alignment_methods.ps1

# Linux/Mac
bash scripts/compare_alignment_methods.sh
```

## 📊 实验设计

对比四组配置：
1. **Baseline**: 硬 index 对齐 + 等比例层映射
2. **+CKA Layer**: 硬 index 对齐 + CKA 层映射
3. **+Segment Time**: Segment 重采样 + 等比例层映射
4. **Alignment v2 (Full)**: 完整四维对齐 ⭐

**预期提升**（根据文献和老师反馈）：
- 时间对齐改进：+1-2%
- 层对齐改进：+2-3%
- **组合效果：+3-5%**

## 📁 项目结构

```
├── experiments/
│   ├── alignment_v2.py              # 时间+层对齐核心逻辑 (630行)
│   ├── kv_dimension_projector.py    # 维度投影+Head展平 (450行)
│   ├── precompute_layer_mapping.py  # CKA预计算脚本 (180行)
│   ├── train_with_kv.py             # 主训练脚本
│   ├── train_multi_teacher_kv.py    # 多教师训练
│   └── cka_loss.py                  # CKA损失计算
├── tests/
│   └── test_complete_alignment.py   # 完整对齐流程测试 (6/6通过)
├── scripts/
│   ├── compare_alignment_methods.ps1  # 对比实验脚本 (PowerShell)
│   ├── compare_alignment_methods.sh   # 对比实验脚本 (Bash)
│   └── validate_stable_upgrades.ps1   # 稳健升级验证
├── align/                           # 对齐模块
├── teacher/                         # 教师KV提取
├── fuse/                            # KV融合
├── visualization/                   # 可视化工具
├── ALIGNMENT_V2_GUIDE.md           # 完整技术文档 (500行)
├── STABLE_UPGRADES_GUIDE.md        # 稳健升级指南
└── requirements.txt                # 依赖列表
```

## 🔧 核心文件说明

### `experiments/alignment_v2.py`
- `SegmentIdentifier`: 自动识别 Prompt/Reasoning/Answer 段
- `resample_kv_with_interpolation()`: 时间维重采样
- `CKALayerMapper`: CKA 层相似度映射
- `align_multi_teacher_kv_with_projection()`: 完整三阶段对齐
- `fuse_multi_teacher_kv()`: 多教师融合

### `experiments/kv_dimension_projector.py`
- `KVDimensionProjector`: 可学习维度投影
- `flatten_kv_heads()`: Head 维度展平
- 支持 Xavier/Orthogonal/Identity-scale 初始化
- Save/Load 权重管理

### `experiments/train_with_kv.py`
- 主训练循环
- 稳健小升级：Detach + Warmup + Teacher Attention + Loss Diagnostics
- CLI 参数：`--use_cka_layer_mapping`, `--use_segment_resampling`
- 多教师权重配置

## ✅ 测试状态

| 测试模块 | 状态 | 说明 |
|---------|------|------|
| KV Dimension Projector | ✅ 5/5 | 维度投影、Head展平、Save/Load |
| Alignment v2 | ✅ 4/4 | 时间重采样、CKA映射、层对齐 |
| Complete Pipeline | ✅ 6/6 | 完整三阶段对齐+多教师融合 |

## 📈 性能优化

### 已实现的稳健升级
1. ✅ **Detach 修复**：防止教师梯度污染
2. ✅ **Warmup 机制**：1000步渐进式权重增长
3. ✅ **Teacher Attention 可选**：减少计算开销
4. ✅ **Loss 诊断工具**：实时监控各组件损失

### 内存优化
- Gradient checkpointing 支持
- FP16 混合精度训练
- 按需加载教师模型

## 📚 文档

- [Alignment v2 完整指南](ALIGNMENT_V2_GUIDE.md) - 技术细节、公式推导、使用示例
- [稳健升级指南](STABLE_UPGRADES_GUIDE.md) - Bug 修复和训练稳定性
- [HPC 部署指南](HPC_DEPLOYMENT_GUIDE.md) - 高性能计算集群使用
- [实验设计文档](EXPERIMENT_DESIGN.md) - 对比实验和消融研究

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 License

MIT License

## 🙏 致谢

感谢老师对多教师 KV 蒸馏和对齐策略的宝贵反馈！

## 📧 联系

- GitHub: [@theodorewang0731-hash](https://github.com/theodorewang0731-hash)
- 仓库: [quickly-check-for-mulit-teacher-kava-ache](https://github.com/theodorewang0731-hash/quickly-check-for-mulit-teacher-kava-ache)

---

**更新日期**: 2025-01-25  
**版本**: v2.0 - 完整四维对齐实现
