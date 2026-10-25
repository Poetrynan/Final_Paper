# Sync-X 超参数搜索指南

## 1. 参数分类

### A. 来自文献的参数（固定，不建议调）

| 参数 | 值 | 来源 | 说明 |
|------|-----|------|------|
| `infonce_tau_init` | 0.07 | CLIP (Radford 2021) | InfoNCE 温度初始值 |
| `infonce_tau_learnable` | True | CLIP 风格 | 可学习温度 |
| `infonce_tau_min` | 0.01 | CLIP | 防止 τ→0 崩溃 |
| `lambda_align` | 0.1 | GenD (Yermakov 2026) | 超球面 alignment 权重（默认=论文契约 Ch3:97；**可搜**，见 B 表） |
| `lambda_uniform` | 0.5 | GenD (Yermakov 2026) | 超球面 uniformity 权重（默认=论文契约 Ch3:97；**可搜**，见 B 表） |
| `aam_margin` | 0.30 | ArcFace default=0.5 | binary task 用小 margin |
| `aam_scale` | 30 | ArcFace default=64 | frozen backbone 用小 scale |
| `kin_order` | 3 | BioLip (Chen 2026) | 运动学差分阶数 |
| `kin_window` | 5 | BioLip (Chen 2026) | S-G 窗长 |
| `kin_poly` | 3 | BioLip (Chen 2026) | S-G 多项式阶 |
| `dwf_freeze_epochs` | 5 | 打破训练对称性 | 前 N 轮冻结门控 |
| `early_stop_patience` | 8 | 常规设置 | early stopping |
| `warmup_steps` | 500 | 常规设置 | 线性 warmup |
| `optimizer` | "adamw" | AdamW 论文 | 优化器选择 |
| `weight_decay` | 0.01 | AdamW 论文 | 权重衰减 |
| `grad_clip` | 1.0 | 常规设置 | 梯度裁剪 |
| `amp` | True | Mixed Precision | 混合精度 |

### B. 需要搜索的参数（按优先级排序）

> ⚠️ 2026-08-11 修订（对齐论文契约）：λ_corr_lcma=0.3 / λ_shared=0.1 / lr=1e-4 /
> epochs 40/80 / batch 1024(Stage1)/512(Stage2) 等已被论文锁定（Ch3:97 + Ch4:169-173），**不做主搜索**（搜了偏离即无法写论文）。
> 真正值得搜的是论文留白项：λ_align、λ_uniform（Ch4:171-172 "selected on val split"）。

#### 高优先级（论文留白项，值得搜）

| 参数 | 当前值 | 搜索范围 | 建议值 |
|------|--------|----------|--------|
| `lambda_align` | 0.1 | [0.05, 0.3] | 0.05, 0.1, 0.2 |
| `lambda_uniform` | 0.5 | [0.2, 1.0] | 0.2, 0.5, 1.0 |

> 可选敏感性佐证（非主搜索）：λ_corr 单变量 {0.2,0.3,0.4}、λ_shared 单变量 {0.05,0.1,0.15}，
> 用曲线佐证锁定值 0.3/0.1 合理，可写进消融表。

#### 中优先级

| 参数 | 当前值 | 搜索范围 | 建议值 |
|------|--------|----------|--------|
| `stage1_epochs` | 40 | [20, 60] | 20, 40, 60 |
| `stage2_epochs` | 80 | [40, 120] | 40, 80, 120 |

> `lambda_kin`（0.3）**仅 Stage-1 使用**，Stage-2 损失不含 L_kin——不要放进 Stage-2 搜索，否则是死代码。

#### 低优先级

| 参数 | 当前值 | 搜索范围 | 建议值 |
|------|--------|----------|--------|
| `kin_jitter_sigma` | 0.015 | [0.005, 0.03] | 0.01, 0.015, 0.02 |

> `batch_size_stage2`（512）论文已锁定，不搜。

### C. 架构参数（可选调优）

| 参数 | 当前值 | 搜索范围 | 说明 |
|------|--------|----------|------|
| `d_model` | 256 | 128, 256, 512 | 隐藏维度 |
| `n_layers` | 2 | 1, 2, 3 | LCMA 层数 |
| `n_heads` | 4 | 2, 4, 8 | 注意力头数 |
| `T` | 50 | 25, 50, 100 | 时序长度 |
| `laca_lip_dim` | 64 | 32, 64, 128 | 唇特征维度 |
| `dropout` | 0.1 | 0.0, 0.1, 0.2 | Dropout 率 |

---

## 2. 搜索策略

### 策略 1: 顺序网格搜索（推荐，对齐论文契约）

只搜论文留白项，逐步收敛；`--epochs` 控制搜索速度：

```
Step A: 搜索论文留白损失权重（高优先级）— 快速筛选
  - λ_align ∈ {0.05, 0.1, 0.2}
  - λ_uniform ∈ {0.2, 0.5, 1.0}
  - 固定: lr=1e-4（论文契约）
  - --epochs 20（快速筛选，~15min/实验）
  - 共 9 个组合，~2.3 GPU 小时
  - 选优：summary.csv 的 val_auc 列（脚本已自动打印最佳）

Step A': 最优组合放大确认
  - 1 个组合 × 80 epoch，~1h

Step B: 最优配置 × ≥3 seeds 聚合（论文 Ch4:250 要求 mean±std）
  - --seeds "0 1 2"，3 × 80 epoch，~3h

Step C(可选): λ_corr / λ_shared 单变量邻域敏感性佐证（不选优，仅验证锁定值）
  - corr {0.2,0.3,0.4} 3 组合 + shared {0.05,0.1,0.15} 3 组合，20 epoch，~1.5h

最低可落地（A+A'+B）: 13 个实验 ≈ 6.3 GPU 小时
含 C: 19 个实验 ≈ 7.8 GPU 小时
```

> ⚠️ 运行 `--stage training_epochs` 时【不要】传 `--epochs`——搜索值就是
> `stage2_epochs ∈ {40,80,120}`，传了会被覆盖成同一个数，三组合全部空转。

### 策略 2: 消融替代（最省算力）

不搜索，直接用消融实验展示参数影响：

```
- 在消融表中加一行不同 lr 的对比
- 在消融表中加一行不同 λ 的对比
- 用消融结果说明参数选择合理
```

---

## 3. 代码使用

### 配置文件

`config.py` 中已定义搜索配置：

```python
HYPERPARAMETER_SEARCH_CONFIG = {
    'lambda_corr_lcma': {
        'type': 'float',
        'range': [0.1, 1.0],
        'values': [0.1, 0.3, 0.5, 1.0],
        'priority': 'high',
    },
    # ... 其他参数
}
```

### 运行搜索

```bash
# Step A: 搜索论文留白损失权重 λ_align × λ_uniform（20 轮快速筛选，9 组合 ≈ 2.3h）
python hyperparameter_search.py --stage loss_weights --epochs 20

# Step A': 最优组合放大到 80 轮（用 summary.csv 选出的 best_lambda_align/best_lambda_uniform）
python hyperparameter_search.py --stage loss_weights --epochs 80 \
    --best_lambda_align 0.1 --best_lambda_uniform 0.5

# Step B: 最优配置 × 3 seeds 聚合（论文 mean±std；也可对最优跑单独实验）
python hyperparameter_search.py --stage loss_weights --seeds "0 1 2" \
    --best_lambda_align 0.1 --best_lambda_uniform 0.5

# Step C(可选): λ_corr 邻域敏感性佐证
python hyperparameter_search.py --stage learning_rate --epochs 20 --best_lambda_corr 0.2
# （把 --best_lambda_corr 换成 0.3/0.4 各跑一次；λ_shared 同理 0.05/0.1/0.15）
```

> `--stage training_epochs` 阶段（若需）勿传 `--epochs`；`--stage1_ckpt` 指向
> 你的 Stage-1 预训练 ckpt（缺失时 train_stage2 会静默跳过预训练，结果不可比）。

### 查看结果

```bash
# 查看汇总
cat logs/hyperparameter_search/loss_weights/summary.csv
cat logs/hyperparameter_search/learning_rate/summary.csv
cat logs/hyperparameter_search/training_epochs/summary.csv
```

---

## 4. 实验注意事项

### 评估指标

- **主要指标**: DeepSpeak val split AUC
- **次要指标**: AP, F1, EER
- **跨域指标**: 在 5 个跨域集上评测（Deepfake-Eval-2024, Celeb-DF-v2, THB, MVAD, HiFi-AVDF）

### 控制变量

- 每个搜索阶段只变一个参数族
- 其他参数固定为默认值或上一阶段最佳值（论文契约值不动：λ_corr=0.3/λ_shared=0.1/lr=1e-4/epochs 40/80/batch 1024/512）
- **筛选阶段**用同一 seed（默认 0）保证可比性；**最终阶段用 ≥3 seeds 报 mean±std**（论文 Ch4:250 要求），用 `--seeds "0 1 2"`

### 时间预算

| 阶段 | 实验数 | 单实验时间 | 总时间 |
|------|--------|-----------|--------|
| λ_align×λ_uniform（20 轮） | 9 | ~15-20min | ~2.3h |
| 最优放大（80 轮） | 1 | ~1-2h | ~1h |
| ≥3 seeds 聚合（80 轮） | 3 | ~1-2h | ~3h |
| 敏感性佐证（20 轮，可选） | 6 | ~15-20min | ~1.5h |
| **最低可落地（A+A'+B）** | **13** | - | **~6.3h** |

### 单 GPU 优化

- 使用 `amp=True` 加速训练
- 减少 `num_workers` 如果内存不足
- 考虑减少 `stage2_epochs` 进行初步筛选

---

## 5. 推荐方案

### 方案 A: 完整搜索（推荐，约 1 天 GPU 时间）

1. 按"策略 1"执行 Step A（λ_align×λ_uniform 筛选）→ A'（放大）→ B（3 seeds 聚合）
2. 用 summary.csv 的 val_auc 列自动选优（脚本已打印最佳）
3. 用最优配置训练最终模型，报 mean±std

### 方案 B: 快速搜索（只有几小时 GPU 时间）

1. 只搜 λ_align×λ_uniform（Step A，9 组合 20 轮 ≈ 2.3h）
2. 其余全部用论文契约默认值
3. 在消融中展示参数敏感性

### 方案 C: 消融替代（无额外 GPU 时间）

1. 不运行超参数搜索
2. 在消融实验中加入 λ_align/λ_uniform 对比
3. 用消融结果论证参数选择（λ_corr=0.3/λ_shared=0.1 已有论文契约背书）

---

## 6. 结果记录模板

```markdown
## 超参数搜索结果

### Step A: λ_align × λ_uniform（论文留白项）

| λ_align | λ_uniform | Val AUC | AP | EER |
|---------|-----------|---------|----|-----|
| 0.05 | 0.2 | - | - | - |
| 0.1  | 0.5 | - | - | - |
| ... | ... | ... | ... | ... |
| **0.1** | **0.5** | **-** | **-** | **-** |

最佳: λ_align=0.1, λ_uniform=0.5（与论文契约一致）

### Step B: 最优配置 × seeds

| seed | Val AUC |
|------|---------|
| 0 | - |
| 1 | - |
| 2 | - |
| mean±std | - |

### 最终配置

- λ_corr_lcma: 0.3（论文锁定）
- λ_shared: 0.1（论文锁定）
- λ_align: 0.1（搜索确认）
- λ_uniform: 0.5（搜索确认）
- lr: 1e-4（论文锁定）
- stage2_epochs: 80（论文锁定）
- seeds: 3 × mean±std
```

---

## 7. 常见问题

**Q: 为什么要先搜损失权重？**
A: 损失权重直接影响多任务学习的平衡，是最敏感的参数。但注意：λ_corr/λ_shared 已被论文锁定（0.3/0.1），主搜索只做论文留白的 λ_align×λ_uniform。

**Q: 可以用随机搜索代替网格搜索吗？**
A: 可以，但网格搜索更系统，适合单 GPU 顺序执行；当前参数域小（9 组合），网格即可。

**Q: 搜索时要用全部训练数据吗？**
A: 可以用子集加速初步筛选，但最终配置应在完整数据上验证。

**Q: 如果搜索结果显示默认值最好怎么办？**
A: 说明默认值（论文契约 0.1/0.5）已经是好的选择，论文中可以直接报告消融结果作为论证。

**Q: 为什么 training_epochs 阶段不能传 --epochs？**
A: 该阶段的搜索值就是 stage2_epochs ∈ {40,80,120}，传 --epochs 会把三者全部覆盖成同一个数，导致搜索空转（已在代码与本文档双处修复/警示）。
