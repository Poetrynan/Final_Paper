# Sync-X 数据集系统性体检报告（2026-07-25）

> 触发：用户要求重新调研"已修改/已划分"的所有数据集，系统性检查**划分(split)、使用(usage)、角色(role)** 是否有问题，重点排查是否还有其他类似 MVAD 的"错放/标签污染"。
> 方法：逐数据集**只读上盘核查**（不动任何原文件），与 `DATASET_CAVEATS.md` 既有结论交叉比对。
> 结论：**未再发现新的 MVAD 式错放**；唯一真实数据问题是 MVAD（已用元数据 CSV 修好，且根因为源仓库自身错放，非我方整理错误）；其余数据集结构/标签/角色均健康。

---

## 0. 阵容与角色（审计基准）

| 角色 | 数据集 | 是否训练 |
|---|---|---|
| 主训练源 | DeepSpeak v1.1 | ✅ 训练 |
| Stage-1 SSL | VoxCeleb2 subset（real） | ✅ 预训练（须 disjoint 剔 37 泄漏 id） |
| 零样本跨域评测 | DFE2024 / Celeb-DF-v2 / THB / MVAD / HiFi-AVDF | ❌ 仅评测 |
| 参考 | FF++ | ❌ 仅参考行 |
| 探针(B6 捷径) | FakeAVCeleb | ❌ 仅 B6 探针，never trained |
| 不接入 | WildDeepfake / AV-Deepfake1M++ | ❌ 驻留但不读 |

---

## 1. 逐项体检结果

### 1.1 DeepSpeak v1.1（主训练源）— ⏳ 待审批
- **在盘**：否（`D:/Datasets/DeepSpeak*` 均不存在）。
- **状态**：等 HuggingFace manual-gated 审批（符合预期，外部依赖）。
- **风险**：下完后须实测 identity-disjoint 是否真成立（下载后跑 `verify_disjoint` 类检查）。
- **结论**：无数据问题，卡在审批。

### 1.2 VoxCeleb2 subset（SSL）— ✅ 健康
- **在盘**：`D:/Datasets/VoxCeleb2_subset`，500 说话人目录（id00020…）。
- **37-id 泄漏 blacklist**：`configs/exclude_ids.json` 存在（37 个 leak_id），**全部 37 个都在 VoxCeleb2 的 500 内**（已校验）。
- **配套脚本**：`build_disjoint_vox2.py` / `verify_disjoint.py` / `download_voxceleb2_subset.py` 齐全。
- **使用**：SSL 只用 real 片段，disjoint 模式剔 37 个（防 B6 探针"认脸不认伪造"）。
- **结论**：结构/blacklist 正确；blacklist 尚未接入 dataloader（无训练 pipeline，待 DeepSpeak 训练时接）。

### 1.3 THB（零样本评测）— ✅ 健康（一处良性重叠已记录）
- **两份 manifest 均在盘**：
  - `real_dataset_split.json`（**704**：FF++ 470/84/150 + CVHQ 1083/191/334）
  - `real_dataset_split_official_ff++.json`（**1000**：FF++ 720/140/140 + CVHQ 1083/191/334）
- **两版 FF++ 部分均为身份不相交**（源视频级重叠 = 0）。
- **CVHQ 三切源视频级重叠**（记忆原值，本次重验**精确复现**）：Train∩Val=**55**、Train∩Test=**100**、Val∩Test=**31**。
  - ⚠️ 注意粒度：clip 级重叠 = 0（每个 `_N.mp4` 唯一）；源视频级重叠 = 55/100/31（同一 YouTube 视频贡献多 clip 到不同切）。
  - **为何良性**：THB 纯评测、从不训练 → 该重叠不泄漏进训练；论文已写明 "internal identity overlap does not leak into our training pipeline"。
- **结论**：划分/角色正确，无需动作。

### 1.4 FF++（参考）— ✅ 完整无缺
- **original c23/c40 各 1000 mp4**；4 法（Deepfakes/Face2Face/FaceSwap/NeuralTextures）× c23/c40 各 **1000 mp4**。
- 1000 canonical（720/140/140）对齐文献惯例。
- **结论**：结构完整，角色（参考行）正确。

### 1.5 HiFi-AVDF（零样本评测）— ✅ 重验通过
- **6 生成器目录** mp4 计数：kling2.5 460 / pixverseV5 616 / seedance1.0 788 / sora2 728 / veo3.1 638 / wan2.5 580 = **3810**（与文档一致）。
- **6 份根层 CSV**（`{gen}.csv`）齐全；**master 人脸 jsonl 3810 行 intact**（pass 3627 / fail_haar 183，dataloader 按 `decision!=pass` 过滤）。
- **overall_label 重算**：6 gen 精确 **1:1（1905:1905）**；**seedance1.0 音频 100% real（788/788，`audio_label=0`）**。
- **无错放**：各 gen 视频在各自目录，计数与 CSV 对齐；评测须 `join {gen}.csv` 的 `overall_label`，严禁"文件夹=全 fake"（已在 caveats 写明）。
- **结论**：标签/结构正确，无 MVAD 式问题。

### 1.6 DFE2024（零样本评测）— ✅ 完整
- 媒体目录实际名为 `video-data/`(2036) / `audio-data/`(1980) / `image-data/`(1976)，与三份 metadata CSV 行数吻合。
- 三份 metadata CSV 均在（含 `*-with-links` 版）：video 含 **Video Ground Truth + Audio Ground Truth 双独立标签**。
- `Finetuning Set`：video train=1221 / **test=815**；**AV 一致（V==A 且同 Real/Fake）= 163**（严格跨模态探针子集）。
- README（许可 CC-BY-SA-4.0 + gated ToU，**仅评估禁止训练**）在盘。
- **结论**：结构/标签/许可正确；评测须分模态报告（video 视觉 + audio 分开），163 AV 一致子集作严格跨模态探针。

### 1.7 Celeb-DF-v2（零样本评测）— ✅ 标准
- Celeb-real 590 / Celeb-synthesis 5639 / YouTube-real 300。
- real/synthesis **同人配对**为官方设计，非身份污染。
- **结论**：无问题。

### 1.8 MVAD（零样本评测）— 🔧 已修复（本次审计核心）
- **真实问题**：源仓库把 `seedance1_5.zip` 嵌套在 `jimeng3.5pro/` 目录 → 我们忠实解压后 394 个 seedance 视频落入 jimeng 目录；且 `human_manifest.txt` 是 0 字节空文件。
- **根因**：**数据集作者自身错放，非我方整理错误**（证据：`download_mvad_v2.py` TARGETS 第 24 行；`screen_mvad_human.py` 用 zip 父目录名当生成器名）。重解压会原样复现。
- **修复**：`build_mvad_clean.py` 解析文件名 → `D:/Datasets/MVAD/mvad_eval_clean.csv`（4731 行，true_generator + content_type + eval_include）；`mvad_dataset.py` 为消费入口（默认剔 299 非 human，按 true_generator 分组）。**原盘零改动**。
- **结论**：修复正确且唯一；MVAD 是本次审计中唯一发现真实数据问题的数据集，已闭环。

### 1.9 FakeAVCeleb（B6 探针）— ✅ 健康
- 结构：A6(90) / B6(6882) / real(500)；`B6_trim_report.jsonl` 4925 行（全 B6 分析日志），磁盘实际 `*_trimmed.mp4` = **1957** 段（记忆数字精确正确）。
- **37-id 重叠**：FakeAVCeleb real 500 身份中有 37 个与 VoxCeleb2 重叠 → 已由 `configs/exclude_ids.json` 在 VoxCeleb2 SSL 阶段剔除以保 B6 探针有效。
- **结论**：探针设置正确；FakeAVCeleb never trained。

### 1.10 不接入集 — ✅ 符合策略
- WildDeepfake：在盘 632 文件，**不读**（与 FakeAVCeleb 同源捷径伪迹，占位）。
- AV-Deepfake1M++：当前不在盘，不接入。

---

## 2. 系统性结论

| 维度 | 结论 |
|---|---|
| **划分(split)** | THB 704/1000 两版、FF++ 1000 canonical、VoxCeleb2 500、FakeAVCeleb 500 均正确；THB CVHQ 55/100/31 源视频级重叠为良性（评测专用） |
| **使用(usage)** | 训练仅 DeepSpeak v1.1 + VoxCeleb2(real, disjoint)；其余一律评测，无越界训练 |
| **角色(role)** | 9 数据集角色与论文一致；DFE2024 许可护栏(eval-only)已记录 |
| **错放/污染** | **仅 MVAD 一处**，且为源数据集自身错误，已用元数据 CSV 修好；其余无类似问题 |
| **完整性** | FF++(1000×c23/c40 + 4法)、DFE2024(2036/1980/1976)、HiFi(3810)、THB/MVAD 均计数吻合，无缺失 |

## 3. 待办（均为外部依赖或接入动作，非数据错误）

1. **DeepSpeak v1.1 审批**：获批后下载 → 实测 identity-disjoint → 训练。
2. **训练 pipeline 接入**：当前无 dataloader/eval pipeline 代码；MVAD（`mvad_dataset.py`）、VoxCeleb2（blacklist）、HiFi（overall_label join）的修正逻辑均已就绪，待训练跑通后接入。
3. **评测 dataloader 补齐**：HiFi / DFE2024 / Celeb-DF-v2 / THB 的评测读取逻辑（join 各自标签文件）尚未写成独立 dataloader 模块（同 MVAD 模式），可在训练前预置。

## 4. 一句话总结
> 除 MVAD（源数据集自身错放，已用 CSV 元数据修好、原盘未动）外，所有数据集的划分、使用、角色均健康，未发现新的错放或标签污染。体检通过。
