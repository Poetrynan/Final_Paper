# Sync-X 代码 (`code/`)

Sync-X 轻量音视频深伪检测框架的完整可运行实现，与论文 Ch3/Ch4 一致。

## 架构

```
离线特征 (extract_features.py, 冻结)          轻量适配器 (本目录, 可训练 ~3-8M)
┌────────────────────────────┐         ┌─────────────────────────────────────┐
│ CLIP-ViT-B/16  → v_clip(512)│   ┌──►  LCMA  双向跨模态注意力(audio↔visual) ┐
│ wav2vec2-base  → audio(768) │   │    LACA  唇运动学一致性(仅 v_lip, BioLip式)│
│ MediaPipe 唇    → v_lip(192)│   ├──►  LACA: 运动学(S-G 平滑导数: 位置/速度/加速度/加加速度)│
│ (v_crops 已移除)│   │         →MLP→GRU→pool→2d (无音频)        │
│ 仅抽 audio/v_clip/v_lip │   │    DWF   动态门控融合                    │
└────────────────────────────┘   │    AAM   Additive Angular Margin 分类头  │
                                 └─────────────────────────────────────┘
训练: Stage-1 InfoNCE SSL(真实) → Stage-2 AAM + sync(DeepSpeak)
```

## 文件

| 文件 | 作用 |
|---|---|
| `config.py` | 全局超参数（经文献调研，见下表）+ **数据集根路径真相源** `DATASET_PATHS` |
| `models.py` | LCMA / LACA / DWF / AAMHead / SyncX |
| `losses.py` | AAM-Softmax / InfoNCE / CorrConsistencyLoss / Alignment / Uniformity |
| `dataset.py` | 消费 `.pt` 特征缓存；支持 real_only / B6 trim / B7 排除身份 / PairedBatchSampler |
| `utils.py` | 优化器/调度器/AMP/指标(AUC,AP,EER,ACC,F1) |
| `train_stage1.py` | Stage-1 自监督预训练(LCMA InfoNCE + LACA 伪抖动 + lip↔audio 对应) |
| `train_stage2.py` | Stage-2 监督训练(完整 SyncX, AAM + corr + align + uniform) |
| `evaluate.py` | 跨域零样本评测 + B6 trim / B7 身份 / A1 模态遮蔽 / 组件消融 |
| `smoke_test.py` | 一键形状/反向/损失/指标自检(dummy 数据) |
| `prep_celebdf.py` | Celeb-DF-v2 官方测试 518 段(178 real / 340 fake, 读 List_of_testing_videos.txt) |
| `prep_eval2024.py` | DFE2024 标签从 video-metadata CSV 读(绝不靠文件名), 默认 test 划分 815 段 |
| `prep_fakeavceleb_b6.py` | FakeAVCeleb B6 探针：RVRA(500 真,对照)+RVFA(500)+FVFA(10835) **fake-audio 子集**（偏置所在）；产出扁平 `b6_raw/` 与 `b6_trimmed/` 缓存（Smeu 逐片段动态 trim） |
| `prep_thb.py` | TalkingHeadBench(real FF++ 704 split + fake 7 引擎含 unseen Hallo3/MAGI-1) |
| `prep_mvad.py` | MVAD(mvad_eval_clean.csv, eval_include=1 的 4432 human-only) |
| `prep_hifi.py` | HiFi-AVDF(6 个 per-gen CSV 读标签, 不碰 jsonl) |
| `prep_voxceleb2.py` | VoxCeleb2 子集(Stage-1 SSL, 500 说话人) |
| `prep_deepspeak.py` | DeepSpeak v1.1 主训练集(**待授权下载**; 标签/source_id 待接官方 metadata) |
| `prep_diffusion.py` | **[OBSOLETE]** 早期自建 diffusion 基准方案, 已被 THB 取代, 勿运行 |
| `prep_wilddeepfake.py` | **[OBSOLETE]** WildDeepfake 不接入最终 lineup, 勿运行 |

## 完整运行流程

```bash
# 0) 离线抽特征 → .pt 缓存（冻结骨干 FP16 推理为默认，省显存+提速；
#    SYNCX_INFER_FP16=0 可回退 fp32）。DeepSpeak 主训练集用 4 窗口切片并行：
python prep_deepspeak.py --num-shards 4 --shard-id 0   # 另开 3 个终端分别 shard-id 1/2/3
#    （其余 7 个 prep_* + B6 单进程跑即可，无需切片）

# 1) 验证模型
python smoke_test.py

# 2) Stage-1: 真实片段(主源 VoxCeleb2 子集,论文 Ch4:70)上预训练
#    - LCMA: 音-视对齐 InfoNCE
#    - LACA: 唇部运动学"伪抖动"自监督(纯真实+自生成抖动负样本, BioLip/AVPF 思路)
python train_stage1.py \
    --real_cache ./feature_cache_voxceleb2_real \
    --out_ckpt ./ckpt/stage1.pt --epochs 40 --batch 256 --lr 1e-4

#    - B7 身份泄漏控制(论文 Ch4:337): 从 Stage-1 SSL 子集剔除 FakeAVCeleb 身份
python train_stage1.py \
    --real_cache ./feature_cache_voxceleb2_real \
    --exclude_ids ./configs/exclude_ids.json \
    --out_ckpt ./ckpt/stage1_b7.pt --epochs 40 --batch 256 --lr 1e-4

# 3) Stage-2: DeepSpeak(真实类用 DeepSpeak 自带 real clip)上训完整模型
python train_stage2.py \
    --train_cache ./feature_cache_deepspeak/train \
    --val_cache   ./feature_cache_deepspeak/val \
    --stage1_ckpt ./ckpt/stage1.pt \
    --out_ckpt    ./ckpt/syncx_best.pt --epochs 80 --batch 256 --lr 1e-4

# 4) 跨域零样本评测
python evaluate.py --ckpt ./ckpt/syncx_best.pt --eval_cache ./feature_cache_eval2024_test
python evaluate.py --ckpt ./ckpt/syncx_best.pt --eval_cache ./feature_cache_fakeavceleb_test

# 5) 消融
#  B6 静音捷径探针（Smeu 逐片段动态 trim；扁平缓存由 prep_fakeavceleb_b6.py 产出）
python evaluate.py --ckpt ./ckpt/syncx_best.pt --eval_cache ./feature_cache_fakeavceleb_b6_trimmed --b6
python evaluate.py --ckpt ./ckpt/syncx_best.pt --eval_cache ./feature_cache_fakeavceleb_b6_raw --b6_untrimmed
#  B7 身份泄漏控制(重训时剔除 FakeAVCeleb 身份):
python train_stage2.py --train_cache ... --exclude_ids fakeavceleb_identities.txt --out_ckpt ./ckpt/syncx_b7.pt
#  A1 模态遮蔽:
python evaluate.py --ckpt ./ckpt/syncx_best.pt --eval_cache ... --mask visual
python evaluate.py --ckpt ./ckpt/syncx_best.pt --eval_cache ... --mask audio
#  C 组件消融(w/o LCMA / w/o DWF)—— 重训 + 评测两条线，日志自动隔离到 *_nolcma / *_nodwf
python train_stage2.py --train_cache ./feature_cache_deepspeak/train --val_cache ./feature_cache_deepspeak/val \
    --stage1_ckpt ./ckpt/stage1.pt --out_ckpt ./ckpt/syncx_nolcma.pt --disable_lcma
python train_stage2.py --train_cache ./feature_cache_deepspeak/train --val_cache ./feature_cache_deepspeak/val \
    --stage1_ckpt ./ckpt/stage1.pt --out_ckpt ./ckpt/syncx_nodwf.pt  --disable_dwf
#  评测组件消融(variant 自动标 wo_lcma / wo_dwf；也可省略 --disable_* 由 ckpt 内 cfg 推导)
python evaluate.py --ckpt ./ckpt/syncx_nolcma.pt --eval_cache ./feature_cache_eval2024_test --disable_lcma
python evaluate.py --ckpt ./ckpt/syncx_nodwf.pt  --eval_cache ./feature_cache_eval2024_test --disable_dwf
```

> ⚠️ **B6 评测标识（2026-08-02 对齐论文）**：B6 走精确路径，不再用固定 N 特征层零化。
> `evaluate.py` 用 `--b6`（加载 `b6_trimmed` 缓存，variant=`b6_trimmed`）与
> `--b6_untrimmed`（加载 `b6_raw` 缓存，variant=`b6_untrimmed`，供 `plot_figures.plot_f9`
> 读取）两个显式标记；`FeatureCacheDataset` 已移除 `b6_trim_ms` 固定 N 零化分支。
> 组件消融(C) 与本修复相互独立。

## 参数表（经文献调研确定）

| 参数 | 值 | 依据 |
|---|---|---|
| AAM margin m | 0.30 | CosFace/AM-Softmax 测 0.1–0.5；说话人 AAM 用 0.2；0.3 居中 [Wang CVPR18, arXiv:1801.09414] |
| AAM scale s | 30 | 说话人验证 AAM 标准 s=30 [Chung Interspeech20, arXiv:2010.11558] |
| InfoNCE τ | 0.07(可学习,≥0.01) | CLIP 默认 [Radford ICML21, arXiv:2103.00020] |
| 优化器 | AdamW | [Loshchilov ICLR19, arXiv:1711.05101] |
| lr / wd | 1e-4 / 0.01 | 冻结骨干下仅训适配器 |
| warmup → cosine | 500 步 | 标准对比学习调度 |
| batch | 256 | 冻结特征下 8GB 可行 |
| λ_aam | 1.0 | 主分类损失 |
| λ_corr_lcma | 0.3 | 跨模态对应正则(低于主损失；与论文 Ch4 训练表/Ch3 公式一致) |
| **λ_ce** | **0(去掉)** | AAM 即分类头，CE 冗余(调研结论) |
| Stage-1 / Stage-2 epochs | 40 / 80(+early stop) | SSL / 监督 |
| T | 50 | LCMA 内部时序重采样长度 |
| CLIP / wav2vec | ViT-B/16(512) / base-960h(768) | [Radford21, Baevski NeurIPS20 arXiv:2006.11477] |
| LCMA hidden/heads/layers/dropout | 256 / 4 / 2 / 0.1 | 轻量 Transformer（与论文 Ch3 公式/Ch4 训练表一致）[Vaswani17] |
| LACA 结构 | 仅 v_lip(192) → 解析运动学(Savitzky-Golay 平滑导数: 位置/速度/加速度/加加速度=768-d; `kin_smooth='none'` 退回朴素差分) → MLP(→64) → 双向 GRU(d) → pool → out_proj(2d) | 论文 Ch3 唇运动学一致性分支（BioLip 式，纯视觉、无音频） |
| LACA 运动学前端 | `savgol` (window=5, poly=3, use_fps=False, axes=xyz, norm=False)；`kin_smooth='none'` 退回朴素差分 | 抑制 MediaPipe 量化噪声在 jerk(3 阶)中的放大(~8×)；window=5 @25fps 保 1–8Hz 生理带、砍 >~8Hz 量化噪声；不比朴素更过平滑(保住 Stage-1 伪抖动检测信号) [BioLip arXiv:2604.16808] |
| DWF 门控 | softmax 两标量门 α_ℓ,α_p；前 5 轮冻结为只用 LCMA（偏置 favour LCMA） | 论文 Ch3 §Dynamic Weight Fusion |
| LACA 唇 landmark | 192-d(64 口周点×3) | MediaPipe 口周点：40 唇轮廓(FACEMESH_LIPS) + 24 周围点，见 common._PERIORAL_64_IDS [Lugaresi arXiv:1906.08172] |
| 唇 ROI | 112×112 灰度 | **[已废弃]** v_crops 不再抽取/缓存（见 §1.2 修复计划）；`lip_roi=112` 仅 config 弱引用保留 [SyncNet, Chung ACMM16 arXiv:1612.05678] |
| 混合精度 | AMP fp16；**AAM 头强制 fp32** | s=30 下 softmax 防饱和 |
| grad clip | 1.0 | 标准 |

## 关键方法学决策（调研得出）

1. **不用独立 CE**：AAM-Softmax 本身是 margin 增强的分类头，与 CE 并列冗余且可能梯度冲突。Stage-2 损失为 `L = L_AAM + λ_corr_lcma·L_corr_lcma + λ_shared·L_shared + λ_align·L_align + λ_uniform·L_uniform`（5 项，与论文 Ch3 一致）；同源 real/fake 配对按 route-A 改为**数据层采样**(dataset.PairedBatchSampler)，非损失项。
2. **Stage-2 真实类用 DeepSpeak 自带 real clip**（与 fake 同源采集、同设备），避免"real=访谈、fake=webcam"设备捷径；VoxCeleb2 仅用于 Stage-1 SSL（论文 Ch4:70）。
3. **AAM 头 fp32**：s=30 下 fp16 安全，但 softmax 强制 fp32 防饱和。
4. **B6 移除 leading silence（Smeu 文档化的 cue），并作为保守扩展同时移除 trailing**：偏置只存在于 **fake-audio 子集**（RVFA+FVFA，由 TTS 引入），real-audio 子集（RVRA 全真 / FVRA=wavtolip 真音频）无此偏置，故 B6 只用 fake-audio 子集 + RVRA 对照，FVRA 显式排除。trim 走**精确路径**（b6_std_trim.py 逐片段动态检测 → prep_fakeavceleb_b6.py 重提特征 → 扁平 b6_trimmed/b6_raw 缓存），运行时不再做固定 N 零化。
5. **B7 身份泄漏控制**：VoxCeleb2⟷FakeAVCeleb 身份重叠 → 重训时剔除 FakeAVCeleb 身份，比 ΔAUC。
6. **LACA 运动学用 Savitzky-Golay 平滑导数**：朴素有限差分会把 MediaPipe 量化噪声在 jerk(3 阶)中放大 ~8×，探针易学到噪声而非生物力学（报告1 工程风险）；S-G(window=5, poly=3) 是最小二乘平滑微分器，在 1–8Hz 生理带保真、对宽带白噪衰减，jerk 噪声泄漏从 ~1.4× 压回 ~1.0；window=5 温和不抹掉 Stage-1 注入的伪抖动(检测信号存活如基线)。`kin_smooth='none'` 可退回朴素差分做消融。

## 注意

- B6 走**精确路径**：`b6_std_trim.py` 按 Smeu 协议（τ=5e-4，1/FPS=40ms 向上取整）对 fake-audio 子集逐片段动态 trim，再经 `prep_fakeavceleb_b6.py` 重提特征 → 扁平 `prep/b6_trimmed/`（raw 基线在 `prep/b6_raw/`）。`evaluate.py --b6/--b6_untrimmed` 直接加载这两份缓存，运行时不再做固定 N 零化。原始目录 `D:/Datasets/FakeAVCeleb_v1.2` 只读、不修改。
- 特征缓存维度必须与 `common.py` 的 extract_* 输出一致（audio 200×768、v_clip 100×512、v_lip 100×192；**v_crops 已移除**）。`smoke_test.py` 用此形状验证。
- GPU：单卡 RTX 4060(8GB)，冻结骨干 + AMP 可跑 batch 256。
- 推理精度：冻结骨干默认 **FP16**（省显存+提速，实测 v_clip/audio_seq cos>0.999）；`SYNCX_INFER_FP16=0` 回退 fp32。落盘 `.half()`、训练 `.float()` 不变。
