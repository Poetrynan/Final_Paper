# Sync-X 模块2 · 方向B 重定义与改动方案（LTCA：Long-Temporal Content Consistency Adapter）

> 面向评审。基于 `evidence_chain_crossdomain.md` 的**最新归因**（训练配方轴三连负 + 结构谱代理全证伪）
> 重定义方向B。本文件是**设计文档，未改任何代码**；改动方案供审，批准后再动。
> 最后更新：2026-08-21（v2：从"结构谱 FTA"重定义为"内容特征长时一致性 LTCA"）

---

## 0. 为什么必须重定义方向B（立项依据·一句话 + 归因收紧）

**原方案 §3 的 FTA 分支用的是"子带能量谱 + 音画交叉谱相干(MSC)"这套结构谱代理，
已在 `evidence_chain §6B` 实测四集全 ≈0.5（含训练域）被证伪——直接照搬 = 重烧死路。**

而三条已证事实把归因牢牢钉在**内容特征层**：

1. **可分信号在"内容"特征**：`audio_emb` 线性探针 AUC=0.827，|Δ质心|=0.044。
   可分信息在 **wav2vec2 深层音频内容特征**，**不在任何结构代理**（运动学 / 能量包络 / 频域谱 / 交叉谱相干）。
2. **训练配方轴三连负**（全量 / 全冻结 / 仅冻视觉）跨域全 ≈随机（§8B/§8C）→ 在**同一组冻结特征**上换监督、
   换冻结、换 loss，怎么都突破不了 0.6。
3. **瓶颈上移到离线特征表征层本身**：三条配方都在 `CLIP-ViT-B/16 + wav2vec2-base-960h` 同一根棍子上打转，
   跨生成器不可迁移。

**→ 方向B 的落点不是"再缝一个结构谱分支"，而是在内容特征轴上做长时域增强：
让已经可分的音频/视觉内容信号，在更长的时序上下文里被稳定性利用。** 这仍严守在 Cross-Modal 红线内。

---

## 1. 红线与硬约束（不可违背，沿用 + 收紧）

1. **Cross-Modal / 音画一致性主线**：LTCA **必须**同时消费音频内容 + 视觉内容，并含一个显式的**音画时序相干**项；
   **不得退化为纯视觉 / 纯单模态刷分**（绝不做 v1 的 `--mask audio` 式作弊分）。
2. **诚实计量**：凡"跨生成器可分"结论必须有探针/实测支撑，不 extrapolate。结构谱代理已证伪 → **本方案彻底删除频域谱/交叉谱MSC**。
3. **断点/权重安全**：换架构 → 启动前 `rm -f $PERSIST/ckpt/*.resume $PERSIST/ckpt/syncx_best*.pt`
   及对应 `.markers/{4_stage2,5_eval,5b,5c}.done`（memory 红线）。`SYNCX_COMPILE=0` / `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 不变。
4. **诚实约束（长时域边界）**：现有缓存特征单片段 4s（audio 200@50Hz / video 100@25fps，T=50）。
   "长时域"= **在现有 4s 窗口内建模时序演化/规律**，不引入多片段拼接或重新抽取（成本可控）。

---

## 2. 前置门（重定义后，依据现行证据收紧）

- **结构谱门已判死**：§6B 把原 FTA 的两类结构谱输入（子带能量、MSC 交叉谱相干）全判 ≈随机。
  因此本方案**不再设"结构谱探针门"**（那门已关），避免再做无用功。
- **内容口径门（改为验证性、非阻塞）**：可分性本身已由 `audio_emb 0.827` 实证。落地前若想再保险，
  可加一个秒级脚本测"现有 audio_emb 内容特征在跨生成器上的长时一致性（跨 200 帧的自相关/跨段相关性）是否≥0.6"——
  但**不作为阻塞门**（不满足也不回退，因为内容可分已是既有事实，只是增强时序利用）。
- **真正的门 = 架构落地后的验收（§6）**：跨域 DFE/HiFi 必须 >0.55 且较 v2 单调改善，才是方向B 成立。

---

## 3. 架构设计：LTCA 分支（Long-Temporal Content Consistency Adapter）

第三个线索分支 `h_fta ∈ ℝ^{2d}`，与 LCMA、LACA 并列，经 DWF 门控融合。
**输入复用 LCMA pre-attention 的内容嵌入**（注意：是"内容"，不是结构代理）：
`a_seq=(B,50,256)`、`v_seq=(B,50,256)`——LCMA 已返回的对齐时序嵌入，可分信号所在地。

### 3.1 子系统 A：长时域内容骨干（必选，长时主线）

对 `a_seq` / `v_seq` 各建一条轻量长时建模：
- **Audio 长时**：dilated temporal conv（膨胀 1,2,4，感受野覆盖 ~全 50 帧）→ 池化 → `a_ctx`
- **Visual 长时**：同 dilated conv → 池化 → `v_ctx`
- 合并 `concat(a_ctx; v_ctx)` → MLP → `h_long (2d)`

目的：把"只在单帧/短窗上可分的音频内容"提升为"跨 4s 时长的累计/演化证据"，增强跨生成器鲁棒性。
（用 conv 而非 transformer 起步，轻量、可复现；`fta_temp_layers/scale` 可切 attention，留后门。）

### 3.2 子系统 B：音画时序相干（必选，红线核心）

显式 AV 时序一致性，杜绝单模态回潮：
- 单层 **cross-attention**：visual 序列为 query，audio 序列为 key/value（或镜像），得时序对齐特征 `h_av`
- 加一个**全局时序相干标量** `sync_t`（两序列沿时序维的相关性），作为跨模态一致性的显式信号
- `h_av` → MLP → `h_sync (2d)`，与 `h_long` 相加得 `h_fta`（或用 gate，见权衡）

这一项确保 LTCA **绝非单模态**：即使长度为 0（只留子系统 A 的 conv），A 也是 audio+visual 双路拼接。

### 3.3 汇总与融合（沿用 §3 选项 A：DWF 扩 3 路）

`h = Σα_i h_i`（i ∈ lcma / laca / fta），偏置 favour 前两路，FTA 渐进接管。
`--disable_fta` 做 w/o-FTA 消融。FTA 内部 `--fta_no_sync` 可关子系统 B（只留长时骨干），单独归因。

---

## 4. 具体改动点

### 4.1 `models.py`
- 新增 `class LTCA(nn.Module)`：`aud_conv` / `vis_conv`（dilated temporal conv）+ `cross_attn` + `sync_scalar`(可选) + 融合 MLP → `(B, 2d)`。
- `class DWF`：`gate` 支持 3 路（`in_features = fused_dim*n_branches`，`n_branches=3`），偏置 favour LCMA/LACA。
- `class SyncX`：依 `cfg.fta_enabled` 构造 `self.fta`；`forward` 内用 `a_seq/v_seq` 算 `h_fta`，
  并入 `fused = self.dwf(h_lcma, h_laca, h_fta)`；`--mask audio/visual` 时对应模态的 LTCA 输入置零（口径可比）。

### 4.2 `config.py`
```python
fta_enabled: bool = False        # 主开关（SYNCX_FTA=1）
fta_temp_layers: int = 1         # dilated conv 层数（0=停用说明异常）
fta_dilation: list = (1, 2, 4)   # 膨胀序列
fta_use_cross_attn: bool = True  # 子系统 B：音画时序相干（红线核心）
fta_hidden: int = 256            # LTCA 内部隐藏维
```

### 4.3 `train_stage2.py`
- 读 `SYNCX_FTA` 覆写 cfg；可选把 `lambda_fta·L_avsync`（子系统 B 的相干监督，light）加入 loss，
  日志列加 `L_sync`（默认关，先靠分类隐式学）。

### 4.4 `evaluate.py`
- `--mask audio/visual` 传导到 `SyncX.forward` 置零对应模态 LTCA 输入。
- 保留 `--head audio`（audio_aam）路径不变。

---

## 5. 断点续跑 / 权重 / 重训红线（换架构必做，沿用原 §5）

```bash
rm -f $PERSIST/ckpt/*.resume $PERSIST/ckpt/syncx_best*.pt
rm -f $PERSIST/.markers/4_stage2.done $PERSIST/.markers/5_eval.done \
      $PERSIST/.markers/5b_b6.done $PERSIST/.markers/5c_ablations.done
export SYNCX_COMPILE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 落地时 export SYNCX_FTA=1
```
断点续跑/原子写/评测缓存幂等机制不变。重训一并修 `savgol_coeffs` 地板除 bug（Kinematics 8→7 窗）。

---

## 6. 验收标准（实测，非估算）

1. 训练域：全模态 val_auc 保持 0.95+，且 **audio-only（`--mask visual`）较 0.5734 单调升**（证明内容可分信号被逼用上）。
2. 跨域：FakeAVCeleb/DFE2024/HiFi 融合头 AUC 较 v2(0.6171/0.5506/0.499) 单调改善，**至少 DFE/HiFi >0.55**。
3. 消融：`w/o FTA vs w/ FTA` 跨域差值 >0 且可复现；`--mask audio`(保留视觉) 不依赖 FTA 过度（防纯视觉回潮）。
4. 红线自检：`fta_no_sync`（只留长时骨干）与全开的对比——全开应**不降**，证明子系统 B 没拖累。

---

## 7. 实施步序

1. 按 §4 落代码（先 LTCA 单分支 + 三路 DWF + `--disable_fta`；`--fta_no_sync` 消融开关）。
2. 本地无 torch 托盘验证 forward 形状，或云端小 batch 冒烟 + `SYNCX_COMPILE=0` 看首 epoch `L_sync` 列。
3. 跑完整 stage2 + eval，对照 §6 验收；并行跑 `--disable_fta` / `--fta_no_sync` 消融。

---

## 8. 风险与权衡表

| 项 | 取舍 | 理由 / 缓解 |
|----|------|-----------|
| 长时建模用 dilated conv vs temporal transformer | 先 conv | 轻量、可复现、感受野覆盖全窗；`fta_temp_layers` 留 attention 后门 |
| 子系统 B 用 cross-attn + 显式 sync 标量 | 必选 | 红线核心，杜绝单模态；标量可选便于消融归因 |
| AV 相干监督 `L_sync` 是否加入 loss | 默认关 | 先靠分类隐式学，避免与 L_corr/L_shared 再叠加监督冲突（§8B 教训） |
| 3 路 DWF 门控把 FTA 权重学成近 0 | 中风险 | 偏置 favour 前两路；报告 gate 分布；必要时 `--max_fta_weight` 兜底 |
| 内容长时在 4s 内收益有限 | 中/低 | 先只上 LTCA 主线，视验收再决定是否重抽更长特征（打破诚实约束需另行批准） |
| modality dropout 是否需要伴开 | 待定 | 仍是防单模态域捷径的有效伴项；单变量验证，勿与冻结同开 |
| 换架构误用旧权重（漏删 step） | 高 | §5 的 rm 写进启动脚本；marker 语义已在 memory 存档 |