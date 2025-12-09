# 🎯 v4.0 快速参考卡片

## 📍 当前位置
✅ Phase 2 集成完成 | ⏳ Phase 2.5 测试验证

---

## 🧪 三关通关命令速查

### ⏳ 第一关：集成测试
```bash
# 在项目根目录执行
python experiments/test_v4_integration.py
```
**通过标准**: 所有测试通过，无错误日志

### ⏳ 第二关：10-Step 冒烟测试

**Flat Mode**:
```bash
python experiments/train_with_kv.py \
    --model_name gpt2 \
    --subset_size 10 \
    --batch_size 2 \
    --epochs 1 \
    --alignment_mode flat \
    --kv_method rkv \
    --output_dir outputs/smoke_flat
```

**Structured Mode**:
```bash
python experiments/train_with_kv.py \
    --model_name gpt2 \
    --subset_size 10 \
    --batch_size 2 \
    --epochs 1 \
    --alignment_mode structured \
    --map_proj_share_dim \
    --map_proj_init_uniform \
    --kv_method rkv \
    --output_dir outputs/smoke_structured
```
**通过标准**: 两个模式都能跑完，Loss 正常，无 NaN

### ⏳ 第三关：A/B 实验

**Baseline (Flat)**:
```bash
python experiments/train_with_kv.py \
    --model_name gpt2 \
    --subset_size 1000 \
    --batch_size 8 \
    --epochs 3 \
    --alignment_mode flat \
    --kv_method rkv \
    --lr 5e-5 \
    --output_dir outputs/ab_flat \
    --logging_steps 10 \
    --save_steps 200
```

**v4.0 (Structured)**:
```bash
python experiments/train_with_kv.py \
    --model_name gpt2 \
    --subset_size 1000 \
    --batch_size 8 \
    --epochs 3 \
    --alignment_mode structured \
    --map_proj_share_dim \
    --map_proj_init_uniform \
    --kv_method rkv \
    --lr 5e-5 \
    --output_dir outputs/ab_structured_uniform \
    --logging_steps 10 \
    --save_steps 200
```
**通过标准**: 收集结果，对比 loss 曲线和稳定性

---

## 🚀 自动化测试脚本

### 完整测试套件（包含上述所有测试）
```bash
bash scripts/test_v4_quick.sh
```

---

## 📚 关键文档（四层体系）

| 文档 | 用途 | 何时使用 |
|------|------|---------|
| [`V4_INTEGRATION_COMPLETE.md`](V4_INTEGRATION_COMPLETE.md) | 📋 **我都改了啥** | 溯源代码变更 |
| [`V4_EXECUTION_ROADMAP.md`](V4_EXECUTION_ROADMAP.md) | 🗺️ **整体怎么走** | Phase 2/2.5/3 时间线 |
| [`PHASE2_STATUS_REPORT.md`](PHASE2_STATUS_REPORT.md) | 📍 **现在在哪** | 当前进度盘点 |
| [`V4_QUICK_REFERENCE.md`](V4_QUICK_REFERENCE.md) | ⚡ **命令怎么敲** | 实战速查（本文件） |

---

## 🔧 新增功能

### 命令行参数
```bash
--alignment_mode {flat,structured}  # 对齐模式（核心开关）
--map_proj_share_dim               # 共享维度投影（推荐开启）
--map_proj_init_uniform            # 均匀初始化（推荐开启）
```

### 训练日志示例
```
Step 10: loss=2.5, CE=2.0, KV=0.5, CODI=0.3 [Mode: structured]
Step 20: loss=2.3, CE=1.8, KV=0.4, CODI=0.25 [Mode: structured]
```

---

## 📊 代码统计
- 新增文件: 6 个
- 修改文件: 2 个
- 总代码量: +960 lines
- Git Commit: fb36290

---

## ✅ 检查清单（总控面板）

### Phase 2 (完成 ✅)
- [x] MapProjectionAligner 集成
- [x] 双模式训练循环
- [x] 命令行参数
- [x] 测试工具
- [x] 完整文档
- [x] Git 推送成功

### Phase 2.5 (待执行 ⏳)
- [ ] ⏳ 第一关：test_v4_integration.py
- [ ] ⏳ 第二关：10-step 冒烟（flat & structured）
- [ ] ⏳ 日志验证：模式标记正确
- [ ] ⏳ 形状验证：无维度错配

### Phase 3 (待执行 ⏳)
- [ ] ⏳ Baseline 实验（flat, 1000 samples）
- [ ] ⏳ v4.0 实验（structured, 1000 samples）
- [ ] ⏳ 结果分析（loss 曲线对比）
- [ ] ⏳ 统计显著性测试

---

## 🎯 成功标准
- ✅ 代码集成无语法错误
- ✅ Git 推送成功
- ⏳ test_v4_integration.py 全部通过
- ⏳ 10-step 冒烟测试成功
- ⏳ A/B 实验完成
- ⏳ structured ≥ flat 性能

---

## 💡 关键创新
1. **Anti-Flatten 结构化设计** - 全程保持 5D 形状
2. **Uniform Initialization** - Teacher heads 均分到 Student heads
3. **双模式控制变量实验** - 同一脚本，唯一差异是对齐方式

---

## 📞 快速链接
- GitHub: https://github.com/theodorewang0731-hash/quickly-check-for-mulit-teacher-kava-ache
- Commit: fb36290

---

**准备开始测试！** 🚀
