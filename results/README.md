# Sync-X 结果 — v1（第一版）

> 本目录收纳 Sync-X **第一版**完整复现链路的实验结果。
> **参数真相源**：本 README 下方参数与 `../code_cloud/config.py` 完全一致（经逐项核对）；结果数字直接引自云端 `persist/logs` 与各 `*_with_pretrain.jsonl`/status 日志，非估计值。
> **项目定位**：提交答辩/论文评审的初始证据链版本。跨域泛化为本版软肋（详见"结果概览"），后续 v2 将据此调整。

---

## 目录结构约定

```
result/v1/
├── README.md        ← 本次结果对应的全部代码参数 + 结果清单（本文件）
├── figure/          ← 论文用图：ROC 曲线、训练曲线、柱状对比等（待导出）
└── data/            ← 原始数据：各评测集预测分数、AUC/AP/EER 明细、训练 loss 序列（待导出）
```

**约定**：`figure/` 只放可供论文直接引用的最终图（命名含版本，如 `v1_roc_fakeavceleb.png`）；`data/` 放可复算的原始 csv/json。每份文件在 `## 3 文件清单` 登记。

---

## 1. 本次结果概览（seed=0，来自云端实测日志）

### 1.1 Stage-2 训练（seed=0）
- **最佳验证 AUC** = **0.9758**（epoch 26），早停触发于 epoch 34（patience=8）。
- Stage-2 配置路径：`persist/logs/syncx_bs512_lam0.3_lacaboth_seed0/`
- 注意：此高 AUC 是**训练域内（DeepSpeak 验证划分）**，不代表跨域泛化。

### 1.2 跨域评测（seed=0）
| 评测集 | n | AUC | AP | ACC | EER | 判定 |
|--------|-----|------|------|------|------|------|
| FakeAVCeleb | 19725 | 0.6187 | 0.9947 | 0.971 | 0.4494 | 弱（远超随机但偏低） |
| DFE2024 | 728 | 0.535 | 0.4955 | 0.4876 | 0.4835 | 接近随机 |
| HiFi-AVDF | 3787 | 0.4735 | 0.4668 | 0.5073 | 0.5147 | 低于随机（反向） |

> ⚠️ **解读警示**：FakeAVCeleb 的 AP/ACC 因类别不平衡严重虚高，**判定泛化只信 AUC/EER**（实测 real=178 / fake=19547，real 仅 0.9%——全判多数类即可 AP≈1/ACC≈0.97，故这两个指标无信息量）。
> 训练域 0.98 对跨域 ~0.54 的巨大缺口 = **结构性过拟合**（模型依赖 DeepSpeak 域指纹而非通用伪造线索）。此为 v1 的关键结论与 v2 的改进起点。

### 1.3 阶段产物（供复算/绘图）
- 权重：`syncx_best_seed0.pt`、Stage-1 `stage1.pt`、Stage-2 续跑检查点。
- 评测记录：各 `*_with_pretrain.jsonl`（含逐样本分数 → 可重画 ROC）。
- 训练序列：`stage1_train.jsonl` / `stage2_train.jsonl`（loss 曲线用）。

---

## 2. 本次结果对应的全部代码参数

> 参数为**实测运行值**，与 `config.py` 逐项一致。标注来源：`[config]`=config.py 硬编码默认值；`[env]`=tmux 启动时环境变量；`[shell]`=pipeline 运行时由脚本传入/覆盖。

### 2.1 运行环境（`[env]`）
| 项 | 值 |
|----|----|
| 设备 | `SYNCX_DEVICE=cuda`（RTX 4090D, 24GB） |
| 特征抽取并行分片 | `SHARDS=8`（⚠️ 16 会 OOM） |
| MediaPipe 后端 | `SYNCX_MEDIAPIPE_GPU=0`（CPU XNNPACK，防多进程 delegate 冲突） |
| torch.compile | `SYNCX_COMPILE=0`（**必须关闭**，4090D 上编译 AAM/DWF 会启动即崩） |
| 显存碎片 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| HuggingFace | `HF_HOME`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` |

### 2.2 特征抽取（`[config]` DATA_ROOT=/root/autodl-tmp/SyncX/Datasets）
| 特征 | 维度 | 源头 |
|------|------|------|
| audio_seq | 768 | wav2vec2-base-960h `last_hidden_state` |
| v_clip | 512 | CLIP ViT-B/16 `visual_projection` |
| v_lip | 192 | MediaPipe 口周点 64 点 × 3 (x,y,z) |
| 时序窗口 | 4s 中心窗 @25fps，视觉重采样到 T=50 | 固定窗（修复 P1/P2/P5 对齐三陷） |
| 冻结推理精度 | FP16（`mean cos>0.999`，零 NaN 已验） | `SYNCX_INFER_FP16` 默认开 |

### 2.3 模型架构（`[config]`）
| 组件 | 参数 |
|------|------|
| LCMA | d_model=256, n_heads=4, n_layers=2, dropout=0.1 |
| LACA | mode=`both`（corr+biomech）, lip_dim=64, kin_order=3, shared_dim=256, axes=`xyz` |
| LACA 运动学前端 | `savgol`, window=5, poly=3, fps=25（削 MediaPipe 高频量化噪声） |
| DWF | 门控前 5 epoch 冻结为只用 LCMA |
| AAM 头 | n_classes=2, margin=0.30, scale=30, fp32 |
| 融合 | DWF 门控融合 LCMA×LACA；`same_source_sampling=True`（Rec C 同源配对） |

### 2.4 训练超参（`[config]` / 默认 λ）
| 项 | 值 |
|----|----|
| 优化器 | AdamW, lr=1e-4, weight_decay=0.01, warmup=500, grad_clip=1.0, AMP=fp16 |
| Stage-1 | 60 epochs, batch=1024（InfoNCE τ_init=0.07 可学习, min=0.01；kin_jitter_sigma=0.015, λ_kin=0.3） |
| Stage-2 | 80 epochs, batch=512, early_stop_patience=8 |
| 损失权重 | λ_corr=0.3, λ_shared=0.1, λ_align=0.1, λ_uniform=0.5 |
| Dropout / weight decay | dropout=0.1 / wd=0.01 |
| 超参搜索 | MVP 默认 **跳过**（`SKIP_SEARCH=1`），用论文默认 λ |

### 2.5 运行命令要点
- 一键流程：`bash cloud_settle_all_mvp.sh`（tmux 内，幂等 + Epoch 级续跑）。
- 数据集完成：DeepSpeak(v1.1 主训练) / VoxCeleb2_subset(SSL) / FakeAVCeleb / DFE2024 / HiFi-AVDF。
- 跨域评测集：FakeAVCeleb + DFE2024 + HiFi-AVDF（Celeb-DF-v2 无音轨已砍）。

---

## 3. 文件清单（导出后登记）

> 待结果全部导出后填写：文件名 / 对应内容 / 用途 / 论文引用位置。

| 文件 | 内容 | 用途 |
|------|------|------|
| `figure/…` | （待导出） | |
| `data/…` | （待导出） | |

---

## 4. 待办 / 备注
- [ ] pipeline 全流程结束后，用导出脚本把 `figure/`、`data/` 填满并回填 §3 清单。
- [ ] 跨域泛化问题（§1.2）是 v1 的软肋：待全 seed 结果后统一诊断（见 SyncX_Progress_and_Pitfalls §12）。

---

## 5. 跨域崩塌根因诊断记录（2026-08-21）

> 工具：`code_cloud/diag_crossdomain.py`（`meta` 纯元数据 / `model` 逐样本前向按身份分组）。
> 方法：训练集为 DeepSpeak；评测集 FakeAVCeleb / DFE2024 / HiFi-AVDF 零样本。用 ckpt 实测量化，无估计值。

### 5.1 meta 实测（不占 GPU，秒级）

**重叠/泄漏检查（应为 0）**
| 评测集 | 身份交集(与训练) | 样本路径交集 | 判定 |
|--------|------------------|--------------|------|
| FakeAVCeleb | 0 | 0 | ✅ 无泄漏 |
| DFE2024 | 0 | 0 | ✅ 无泄漏 |
| HiFi-AVDF | 0 | 0 | ✅ 无泄漏 |
- 预处理一致性：三集首样本 `audio_seq=[199,768]`、`v_clip=[100,512]`、`v_lip=[100,192]`（audio 略短于 200，三集一致，`__getitem__` 会 pad，非根因）。

**类别分布**
| 评测集 | real | fake | real 占比 | 含义 |
|--------|------|------|-----------|------|
| FakeAVCeleb | 178 | 19547 | 0.9% | 严重不平衡 → AP/ACC 无信息量，只信 AUC（0.6187） |
| DFE2024 | 393 | 335 | 54% | 大致平衡 |
| HiFi-AVDF | 1891 | 1896 | 50% | 平衡 |

**训练集身份捷径嫌疑（否定）**
- DeepSpeak 训练：n=13463，real=6667 / fake=6796，身份=220。
- 身份→标签：单标签身份仅 **3** 个，同身份含 real+fake（混合）**217** 个 → **粗"背身份"捷径被否定**（认出一个身份不能直接给标签）。

### 5.2 model 实测（HiFi-AVDF，逐样本前向，min_per_id=4）

HiFi 的 6 个身份 = 6 家视频生成器（seedance1.0 / sora2 / veo3.1 / pixverseV5 / wan2.5 / kling2.5）。固定同说话人 + 同生成器内部真/伪排序 AUC：

| 生成器 | n_real | n_fake | 组内 AUC | 判定 |
|--------|--------|--------|----------|------|
| seedance1.0 | 394 | 394 | 0.387 | 随机偏反向 |
| sora2 | 364 | 364 | 0.571 | 接近随机 |
| veo3.1 | 314 | 314 | 0.454 | 随机偏反向 |
| pixverseV5 | 302 | 308 | 0.600 | 接近随机 |
| wan2.5 | 290 | 290 | 0.404 | 随机偏反向 |
| kling2.5 | 227 | 226 | 0.449 | 随机偏反向 |

### 5.3 诊断结论（据实收敛）

1. **数据层三嫌疑全部洗清**：无训练/评测重叠泄漏；FakeAVCeleb 虚高指标实锤为不平衡陷阱；"背身份"粗捷径被否定。
2. **瓶颈在模型学到的表征本身**：固定说话人 + 固定生成器内，模型对真/伪排序 ≈ 随机（多为 0.39–0.45 反向）。即使排除"换说话人"因素，模型也**没有可泛化的音画同步判别力**。
3. **跨域崩塌是"换生成器/换合成风格即失效"**，而非"换说话人即失效"——HiFi 各生成器内部即崩。

### 5.4 v2 方向分叉（待定，附下一步检验）
- **假设 A（监督目标未教会可迁移表征）** → v2 重设计同步任务/负样本。验证：在 HiFi 上跑 `--mask audio` vs `--mask visual` 消融，判定模型被哪路信息主导、对假信号是否敏感。
- **假设 B（方法表达力够但缺域泛化机制）** → v2 加跨域增强/多域训练/规约。
- 目前实证（组内≈随机）更倾向 **A**，但需单变量实验锁定后再定。

### 5.5 方向 0 验证（训练域 mask 消融，2026-08-21 补充）—— 病根收敛为两处

| 条件 | DeepSpeak AUC | HiFi AUC |
|------|:---:|:---:|
| 全模态 | 0.9758 | 0.4735 |
| 保留视觉（置零音频） | **0.9873** | 0.4548 |
| 保留听觉（置零视觉） | 0.5197 | 0.4706 |

收敛结论（综合 §5.1–§5.4 + 本表）：
1. **训练域 0.9758 主要靠视觉单模态**（0.9873 甚至更高）→ 存在强 visual 单模态捷径，但不可迁移（跨域 0.45）。
2. **听觉分支从未学会判别力**：训练域 0.52、跨域 0.47，任何条件都 ≈ 随机 → 属**监督缺失**，非泛化问题。
3. 病根两块、彼此独立：**听觉监督断裂** + **视觉表征绑定训练域**。v2 主路（见 `诊断与优化方向_v1.md` §4.1）：先救听觉监督（让 audio-only 在训练域 >0.8），再做视觉去域。

### 5.6 L_corr 文献溯源与监督冲突（2026-08-21 补充，见 `诊断与优化方向_v1.md` §3.6）

**思想有据、具体形式是实例化**：
- "用音-视失配/不和谐判伪"有强文献支持：Chugh et al.(ACM MM 2020) 的 MDS 是 L_corr"fake 段去相关"思想的直接来源 [$TRAE_REF](https://arxiv.org/abs/2005.14405)；AV-Lip-Sync+(2025) 是 LACA 共享空间损失的引用源；LipFD、AV-sync 综述亦佐证该方向。
- 但 **`target=(1-y)` + MSE 严格写死的具体形式是本项目实例化**，且与 L_AAM 分类目标在 `audio_emb` 上**反向拉扯**（L_corr 要 fake 音频"去相关"，L_AAM 要"判对"）——这是可论证的监督冲突，是听觉 0.52 的一个结构性成因。

**对 v2 的落点**：模块 1 补 **1d**——解除 L_corr 对 fake 音频的"去相关"强制（只约束 real 段），避免与 1a 音频头冲突。此偏离原 (1-y) 形式属**修正自相矛盾的监督**，应作为论文 design rationale 写明。详见 `诊断与优化方向_v1.md` §3.6 / §4.1。

### 5.7 v2 模块 1 执行结果（2026-08-21，详见 `诊断与优化方向_v1.md` §4.1）

**采用 1a + 1d 同开**（音频独立头 `audio_aam` + L_corr real_only），Stage-2 从零重训。

| 项 | v1 | v2 | 判定 |
|----|----|----|----|
| train 域 best val_auc（融合头） | 0.9758 | 0.9744 | 同级（非模块1验收口） |
| FakeAVCeleb AUC | 0.6187 | 0.6171 | ≈持平 |
| DFE2024 AUC | 0.535 | 0.5506 | 微升 |
| HiFi AUC | 0.4735 | 0.499 | 回升至≈随机 |
| train 域 audio-only AUC（`--head audio --mask visual`） | 0.52* | **0.5734** | ⚠️ **未达标**：仅 +0.05，离 >0.8 目标很远 |
| train 域 audio-only AUC（融合头 `--mask visual`，同口径） | 0.52 | **0.5252** | ≈不变 → 音频路径硬瓶颈 |

> \* 口径说明：v1 的 0.52 ="音频输入 + 融合头"打分；v2 的 0.5734 = "音频输入 + 专门训练的独立音频头"打分。**对照 A**（融合头同口径）v2=0.5252 ≈ v1 0.52 → 三数齐证明 **audio_emb 本身判别信息有限**：1a 独立头只榨出 +0.05、1b 放大权重也救不了（放大的是弱信号）。**音频线降级为低成本保留**（音频头 0.5734 不拖累融合分，可作论文诚实 counterfactual），**主攻转模块2 视觉去域**。

- **融合跨域分基本不动 = 符合预期**：模块1 只治"听觉断裂"，未动"视觉域绑定"，恰成为模块 2 立项证据。
- **audio-only 0.52→0.5734 的诚实结论**：1a 独立音频头 + 1d 解除 L_corr 冲突只带来 ~+0.05 提升，远低于 >0.8 目标。印证"听觉断裂"比单纯监督接线更底层（≤0.57 属能判定却判别力弱）；音频监督回归暂不可用，v2 需转向强化监督权重（1b）/或并入模块2 一起重设计。
- 训练日志佐证 1a/1d 生效：`audio=` 损失 9.63→7.54、`corr=` 0.212→0.030 持续下降。
- 复现防踩坑（已入 project_memory）：换架构须删 `<out_ckpt>.resume`；`utils:strip_compile_prefix` 已修 int 键崩溃。