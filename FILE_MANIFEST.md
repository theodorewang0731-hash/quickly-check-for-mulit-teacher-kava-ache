# 项目文件清单

## 已优化用于 HPC 部署

### 核心训练脚本
- `experiments/train_with_kv.py` - 单教师 KV 蒸馏训练脚本（支持大规模数据、多GPU、混合精度）
- `experiments/train_multi_teacher_kv.py` ⭐ - **多教师 KV 蒸馏训练脚本**（支持 5 阶段、异构教师）
- `experiments/kv_utils.py` - KV 压缩工具（full/right_crop/rkv）
- `experiments/kv_loss.py` - KV 损失函数和对齐工具
- `experiments/projector.py` - Student→Teacher 投影层

### 多教师模块（新增）

#### 对齐层 (`align/`)
- `align/tokenizer_align.py` - 异构 tokenizer 对齐（字符级 IoU）
- `align/time_align.py` - 时间维度对齐（padding、masking、软对齐）
- `align/layer_map.py` - 层映射（ratio-based + 插值）
- `align/head_dim_adapter.py` - Head/Dim 适配器（线性投影 + head 聚合）
- `align/rope_scale.py` - RoPE 缩放（NTK-aware scaling）
- `align/__init__.py` - 对齐模块统一接口

#### 教师层 (`teacher/`)
- `teacher/extract_teacher_kv.py` - 教师 KV 离线提取（支持批量、SLURM array）
- `teacher/router_proto.py` - 教师原型计算（mean/cls/kmeans）
- `teacher/__init__.py` - 教师模块统一接口

#### 融合层 (`fuse/`)
- `fuse/fuse_kv.py` - KV 融合策略（fixed/similarity/learnable）
- `fuse/__init__.py` - 融合模块统一接口

### HPC 部署脚本
- `setup.sh` ⭐ - **一键安装脚本**（创建环境、安装依赖、检查GPU）
- `scripts/run_hpc_training.sh` - 单教师训练 SLURM 脚本
- `scripts/run_multi_teacher.sh` ⭐ - **多教师训练 SLURM 脚本**（支持 Phase 1-5）
- `scripts/run_all_experiments.sh` - E1-E5 完整实验套件
- `scripts/run_ultra_large.sh` - 超大规模训练（8 GPU）
- `scripts/run_multinode.sh` - 多节点分布式训练（32 GPU）
- `scripts/test_setup.sh` - 安装测试脚本
- `scripts/test_multi_teacher.py` ⭐ - **多教师模块测试脚本**

### 数据处理脚本
- `scripts/export_gsm8k.py` - 导出 GSM8K 到 JSONL
- `scripts/download_assets.py` ⭐ - **预下载模型和数据集**

### 结果分析
- `scripts/analyze_results.py` - 自动分析和对比实验结果

### 文档
- `QUICKSTART.md` ⭐ - **快速开始指南**（上传后立即可用）
- `HPC_SETUP.md` - 详细部署文档
- `QWEN3_LARGE_SCALE.md` - Qwen3 大规模训练指南
- `MULTI_TEACHER_KV_PLAN.md` ⭐ - **多教师 KV 蒸馏完整规范**（5 阶段、对齐策略、融合方法）
- `MULTI_TEACHER_README.md` ⭐ - **多教师 KV 蒸馏使用指南**
- `FILE_MANIFEST.md` - 本文件
- `requirements.txt` - Python 依赖列表

### 数据文件
- `data/sample_train.jsonl` - 示例训练数据

## 使用流程

### 单教师 KV 蒸馏（原始 KaVa 方法）

#### 1. 上传到 HPC
使用 FileZilla 或 scp 上传整个 `quickly check` 目录

#### 2. 一键安装
```bash
cd /path/to/quickly\ check
bash setup.sh
```

#### 3. 登录 Hugging Face
```bash
hf auth login
```

#### 4. 预下载资源（推荐）
```bash
python scripts/download_assets.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset openai/gsm8k \
    --export_jsonl
```

#### 5. 提交训练
```bash
sbatch scripts/run_hpc_training.sh        # 单次实验
sbatch scripts/run_all_experiments.sh     # 完整实验套件（E1-E5）
```

#### 6. 分析结果
```bash
python scripts/analyze_results.py --output_base outputs/
```

### 多教师 KV 蒸馏（新增功能）⭐

#### 1. 测试所有模块
```bash
python scripts/test_multi_teacher.py
```

#### 2. HPC 大规模训练
```bash
# 修改配置：vim scripts/run_multi_teacher.sh
sbatch scripts/run_multi_teacher.sh
```

#### 3. 查看日志
```bash
tail -f logs/multi_teacher_*.out
```

## 关键特性

### 单教师 KV 蒸馏
✅ 自动路径检测（无需手动配置）  
✅ 兼容各种 HPC 环境（module load 已注释）  
✅ 支持超大数据集（流式模式）  
✅ 多 GPU 并行训练  
✅ 混合精度训练（节省显存）  
✅ 梯度检查点（大幅降低显存）  
✅ 完整的日志和检查点  
✅ 自动生成实验报告  

### 多教师 KV 蒸馏（新增）⭐
✅ **异构教师支持**（不同 tokenizer、层数、维度、注意力头）  
✅ **5 阶段训练**（dual-prompt → multi-sample → real-multi → routing → z-space）  
✅ **3 种融合策略**（固定权重、相似度路由、可学习路由）  
✅ **5 种对齐方法**（tokenizer、time、layer、head/dim、RoPE）  
✅ **3 种路由器**（MLP、Gate、Attention）  
✅ **熵正则化**（鼓励专业化或多样化）  
✅ **完整测试套件**（test_multi_teacher.py 验证所有模块）  
✅ **详细文档**（MULTI_TEACHER_README.md + MULTI_TEACHER_KV_PLAN.md）  

## 三个验证目标

### 单教师 KV 蒸馏（E1-E5 实验）

通过 E1-E5 实验自动验证：

1. **KV 压缩保留监督** - 对比 E1(baseline) vs E2(full KV)
2. **KV 对齐补充监督** - 对比 E1 vs E4(R-KV)
3. **R-KV 最稳定** - 对比 E2/E3/E4 的收敛性

### 多教师 KV 蒸馏（Phase 1-5 验证）⭐

1. **Phase 1-2**：验证多教师可行性（loss 应下降）
2. **Phase 3**：真正多教师融合（性能提升 5-10%）
3. **Phase 4**：动态路由优化（进一步提升 3-5%）
4. **Phase 5**：跨架构对齐（支持异构教师组合）

## 输出结果

每个实验生成：
- `checkpoint-epN/` - 模型检查点
- `training_log.txt` - 训练日志
- `TRAINING_COMPLETED.txt` - 训练摘要

最终生成：
- `outputs/EXPERIMENT_REPORT.txt` - 对比报告
- 验证三个主张的统计分析
