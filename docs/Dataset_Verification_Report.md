# Sync-X 数据集核对报告（划分 / 数量 / 内容）
**日期：** 2026-08-01
**范围：** `D:\Datasets` 下全部数据集（含 `D:\Datasets\Deepfake-Eval-2024`）
**方法：** 磁盘实测文件计数 + 权威清单（CSV/JSONL/manifest）行数核对，对照论文与项目记忆中的期望数值。

---

## 一、总览表

| 数据集 | 论文角色 | 期望（记忆/论文） | 实测 | 结论 |
|---|---|---|---|---|
| VoxCeleb2_subset | Stage-1 SSL（仅真实） | 500 说话人 | 500 id / 89,135 clips | ✅ |
| Celeb-DF-v2 | 跨域评测 | 官方测试 518（178 real + 340 fake） | List = 518（178+340）；原始文件更多 | ✅（须用 List 过滤） |
| TalkingHeadBench | 跨域评测 | CVHQ=1608 / FF++默认=704 / FF++参考=1000 | manifest 实测 = 1608 / 704 / 1000 | ✅ |
| HiFi-AVDF | 跨域评测 | 3810，6 gen，1:1（1905:1905），seedance 音频 100% 真 | CSV = 1905:1905；seedance 394 fake 音频全 0 | ✅ |
| MVAD | 跨域评测 | 4731 总(1000 real+3731 fake) / 4432 human子集(eval_include=1剔299非人类) | 论文未硬编码任一数字(仅"human subset"+`--`占位) | ✅ 已对齐 |
| FakeAVCeleb B6 | 捷径探针 | 6882 trimmed mp4 | 6882 mp4 | ✅ |
| FF++ | 参考 / THB 源 | c23 & c40 | c23: 1000 orig + 4000 manip；c40: 同 | ✅ |
| Deepfake-Eval-2024 | 跨域评测（仅评估） | 2036 vid / 1980 aud / 1975 img，CC-BY-SA，禁训练 | 实测一致；README 明确 eval-only | ✅ |
| FF++_mirror_c23 | （镜像，已删除） | 应含 c23 视频 | **0 mp4（仅 HF 缓存）** | ✅ 已移入回收站（2026-08-01） |

---

## 二、逐项核对详情

### 1. VoxCeleb2_subset ✅
- 500 个身份目录，89,135 个片段（min 21 / max 500 / avg 178.3）。
- 全部为真实片段，用于 Stage-1 自监督。数量健康，身份数符合记忆（500 spk）。

### 2. Celeb-DF-v2 ✅（注意过滤口径）
- `List_of_testing_videos.txt` = 518 行 = 178 real + 340 fake，与论文引用的官方测试集完全一致。
- 磁盘原始文件远多于 518（Celeb-real 590、YouTube-real 300、Celeb-synthesis 5639）。**评测必须用 List 过滤**，不能 glob 全目录，否则会混入非测试视频。

### 3. TalkingHeadBench ✅
两个 manifest 精确复现了论文三个关键数：
- CelebV-HQ 真实片段 = 1083 + 191 + 334 = **1608** ✓
- FF++ 默认子集（`real_dataset_split.json`）= 470 + 84 + 150 = **704** ✓
- FF++ 官方参考（`real_dataset_split_official_ff++.json`）= 720 + 140 + 140 = **1000** ✓
- 注：raw 目录计数（real 2611 / fake 1219 / audio 2927）比 manifest 大，因为含 manifest 文件及派生文件；评测以 manifest 为准。

### 4. HiFi-AVDF ✅
- 6 个生成器 per-gen CSV 总和 = 3810 = 1905 real + 1905 fake，**精确 1:1** ✓
  - kling2.5 460（230/230）、pixverseV5 616（308/308）、seedance1.0 788（394/394）、sora2 728（364/364）、veo3.1 638（319/319）、wan2.5 580（290/290）
- **seedance 音频 100% 真实**：394 个 fake 行 `audio_label` 全部 = 0（real audio），即“画面伪 + 音频真” ✓
- `face_verify_hifi_master.jsonl`（3810 行）是**人脸验证门控清单**（字段：`path / pass_haar / pass_retina / decision / gen`），**不含标签字段**；真实/伪造标签在 per-gen CSV 的 `overall_label`。代码必须 join CSV 取标签，不能从 master 读标签。
- **per-gen JSONL 缺 4 个（已确认无影响）**：官方只发布了 `face_verify_hifi_veo3.1.jsonl` / `face_verify_hifi_wan2.5.jsonl` 这 2 个；`kling2.5 / pixverseV5 / sora2 / seedance1.0` 这 4 个 per-gen JSONL **官方未发布**（磁盘仅存 CSV + master）。但 6 个 per-gen CSV 全齐，标签（`overall_label`/`audio_label`/`visual_label`）全部可从中读取，**评测零依赖缺失 JSONL**。已落地 `code/prep_hifi.py`，用 `_HIFI_GENS` + `_load_labels()` 把"只从 6 个 CSV 读标签、不碰任何 `.jsonl`"在代码与注释双重锁死。✔️ 无需补文件、无需改论文。

### 5. MVAD ✅ 已对齐（原"⚠️ 数量变动"为误报，特此更正）
- `mvad_eval_clean.csv` 原始总行数 = **4731** = 1000 real + 3731 fake（fake_fake 3291、fake_real 440）。
- `eval_include` 列：4432 行 = `1`（纳入评测的 human 子集），299 行 = `0`（剔除，全部为 `fake_fake` 里的 `content_type=non_human`，即当初"MVAD 三坑"源仓库错放的 scene/object/animal）。
- 即 **4731（CSV 总行数）= 4432（human 评测子集）+ 299（非人类剔除）**，两者都对、阶段不同。论文/记忆里记的 **4432 正是"human subset"口径**，与论文正文"Only the human subset is used"完全一致。
- **论文正文未硬编码任一数字**：Ch4/Ch5 仅写 "MVAD (human subset)" + 结果表用 `--` 占位（既定回填约定），grep 证实所有 `.tex` 均无 4432/4731 硬编码。故**无需改论文**。

### 6. FakeAVCeleb B6 ✅
- `FakeAVCeleb_v1.2_split/B6/` = **6882** 个 trimmed mp4（5 个种族子目录）。与记忆一致（此前报告的 4925 是误报，磁盘实际 6882）。
- A6 / real 同存 5 种族子目录，结构正常。

### 7. FF++ ✅
- c23：original 1000 + manipulated 4000；c40：original 1000 + manipulated 4000。标准 FF++ 全量，两种压缩均在。

### 8. Deepfake-Eval-2024 ✅
- video-data 2036 / audio-data 1980 / image-data 1975，与元数据 CSV 行数完全一致。
- 许可证 CC-BY-SA-4.0；README 明确：**“Users may only use this dataset for evaluation. Use … for training goes against the terms.”** → 严格遵守仅评估、禁训练。

---

## 三、需要你处理的问题（行动项）

1. **✅ FF++_mirror_c23 已清理（2026-08-01）**：该空壳（仅 `.cache/huggingface` 下载缓存、0 个 mp4）已按用户要求**移入 D 盘回收站**（`D:\$Recycle.Bin\S-1-5-21-...\$RJJJ1TL\`，可恢复），原位置已消失。真实 c23 不受影响，仍在 `FF++/original_sequences/youtube/c23`（1000 个 original c23）。无任何代码/论文依赖该镜像，删除零影响。

2. **✅ MVAD 数字 4432 / 4731（已核实无需处理）**：4731 = `mvad_eval_clean.csv` 总行数，4432 = `eval_include=1` 的 human 评测子集（剔除 299 非人类）。论文正文未硬编码任一数字（仅"human subset"+`--`占位），故无需改论文。详见 §5。

3. **✅ HiFi-AVDF 缺 4 个 per-gen JSONL（已核实无影响）**：仅 veo3.1/wan2.5 的 per-gen JSONL 存在，其余 4 个官方未发布；但 6 个 per-gen CSV 全齐，标签全部可从中读取，评测零依赖缺失 JSONL。`code/prep_hifi.py` 已用 `_HIFI_GENS`+`_load_labels()` 双重锁死"只走 CSV"。无需补文件、无需改论文。详见 §4。

---

## 四、结论
10 个数据集中，9 个的核心划分 / 数量 / 内容与论文及记忆完全一致。关键声明全部属实：HiFi 精确 1:1、seedance 音频 100% 真、THB 1608/704/1000、Celeb-DF-v2 官方 518、FakeAVCeleb B6 6882、DFE2024 仅评估。
**0 个待办**：`FF++_mirror_c23` 空壳已于 2026-08-01 移入回收站；MVAD / HiFi 经核验均为误报，无需处理。全部数据集核对闭环。
