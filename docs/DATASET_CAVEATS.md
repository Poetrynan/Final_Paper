# 数据集评测与训练注意事项（Sync-X Thesis）

> 用途：跑评测 / 训练前先翻这篇，避免踩已发现的数据坑。
> 维护：每次发现新的数据集/划分/标签坑，追加到对应章节 + 更新底部「变更记录」。
> 配套内部记忆：`C:\Users\ZhuanZ1\WorkBuddy\2026-07-08-16-37-49\.workbuddy\memory\MEMORY.md`（更细，但本文件是人读版）。
> **镜像同步**：本文件在 `monash_thesis/DATASET_CAVEATS.md` 有一份镜像，改任何一处请同步另一处。

---

## 0. 总原则（zero-shot 协议）

- 训练源只有 **DeepSpeak v1.1**（主）+ **VoxCeleb2 real**（Stage-1 SSL）。
- 其余 5 个（DFE2024 / Celeb-DF-v2 / THB / MVAD / HiFi-AVDF）+ FF++（参考）+ FakeAVCeleb（B6 探针）**一律只评测、不训练**。
- 所有外部集走同一套 `extract_features.py` 离线特征缓存，训练好的 `syncx_best.pt` 直接 zero-shot 评估。

---

## 1. THB / FaceForensics++ —— manifest 怎么读（最重要，最容易读混）

THB 目录 `D:/Datasets/TalkingHeadBench/real/` 下有两份 real 划分 manifest，**不是同一份的不同名字，是两套重排过的划分**：

| 文件 | FF++ 段数 | 三切 | 用途 |
|---|---|---|---|
| `real_dataset_split.json` | **704**（470/84/150） | THB 自己重排 | **THB 评测用这份**（THB 论文 baseline 全部按此报） |
| `real_dataset_split_official_ff++.json` | **1000**（720/140/140） | FF++ canonical | **独立 FF++ 参考行用这份**（跟 FF++ 文献惯例对齐） |

**规则（写进 dataloader / eval 脚本）：**
- 取 **THB 的 real 类** → 读 `real_dataset_split.json`（704 FF++ 子集）。
- 取 **独立 FF++ 参考集** → 读 `real_dataset_split_official_ff++.json`（1000，标准 720/140/140）。
- **两套不可直接比较**：THB 把 train/val/test 重新打乱过，THB Test(150) ∩ FF++ 官方 Test(140) 仅 19 段。混用会产出"假一致"数字。

**为什么是 704 不是 1000：** THB 是多模态 benchmark，需要从原始 FF++ YouTube 视频抽原声音轨当真实音频；296 段原始音频已被 YouTube 下架，只能留 704。THB 作者原话："The official FF++ split is provided for compatibility only; results in this work are based on the default split. Results across different split definitions are not directly comparable."

论文已写死（Chapter 4）：FF++ 参考行标 "canonical 720/140/140 over 1,000"；THB 协议段写明用 THB 704、与 canonical 不可比。

---

## 2. 各数据集评测坑位速查表

| 数据集 | 角色 | 坑 | 处理动作 |
|---|---|---|---|
| **VoxCeleb2 ↔ FakeAVCeleb** | SSL / 探针 | 两集真实身份重叠 **37 个 id**（同 `id00xxx` 命名空间）→ 若不隔离，B6 探针"认脸不认伪造" | 代码层 blacklist（`configs/exclude_ids.json`，由 `build_disjoint_vox2.py` 生成）；dataloader / SSL feature-cache 开 `disjoint` 模式跳过 37 个 |
| **THB real CelebV-HQ** | 评测 | 三切**非身份不相交**（Train∩Val=55 / Train∩Test=100 / Val∩Test=31） | 纯评测、不训练 → 不污染我们训练；论文已加句说明"internal identity overlap does not leak into training" |
| **MVAD** | 评测 | ① `human_manifest.txt` 是 0 字节空文件（实测 size=0）；② `jimeng3.5pro/human/` 混了 298 个非 human（scene/animail/animal/object，文件名关键词精确定位）；③ **生成器标注错位最致命**：目录结构里无正经 seedance 文件夹，394 个 `seedance1_5_pro-*` 全错放在 jimeng 目录 → 按目录评测 seedance 近乎消失、jimeng 虚增一倍 | **已修复（元数据-only，原盘零改动）**：`build_mvad_clean.py` 扫 `test_human/**/human/*.mp4` 仅解析文件名 → 输出 `D:/Datasets/MVAD/mvad_eval_clean.csv`（clip_path, split, dir_generator, true_generator, content_type, label, eval_include）。dataloader 改用此 CSV：按 `true_generator` 算 per-generator 指标、剔除 299 个 non_human（`eval_include=0`，含 298 seedance + 1 kling3_0）、彻底弃用空 manifest。**修正后计数**：seedance1.5=399（394 从 jimeng 捞回 +5）、jimeng3.5pro=377、总计 4731 行。消费入口已落地 `mvad_dataset.py`（`MVADDataset` 类：默认剔 non_human、`groups()` 按 `true_generator` 分组、只读不改原盘），eval 代码直接 `from mvad_dataset import MVADDataset` 即可。（根因：源仓库把 `seedance1_5.zip` 嵌套在 `jimeng3.5pro/` 目录下，是数据集作者自己的错放，非我方整理出错；重解压会原样复现，故 CSV 元数据纠偏是唯一正确修法，D 盘 test_human 是源仓库忠实提取、无需重下/重解） |
| **HiFi-AVDF** | 评测 | **不是全 fake（已核实 2026-07-25）**：6 生成器 overall real/fake **精确 1:1（1905:1905）**；除 seedance 外 5 个 gen 的 audio/video 标签完美对齐（fake=音画皆伪），**seedance1.0 音频 100% real（788/788），其 394 段 fake 全是"画面伪+音频真"** | 评测**必须 join 各 gen `{gen}.csv` 的 `overall_label`**（=max(audio,visual)，0/1 编码）取标签，严禁"文件夹=全 fake"；**seedance 已在 Chapter 5 主结果表 `tab:main-results` 与跨数据集表 `tab:cross-dataset` 单独拉一行**（`HiFi-AVDF -- Seedance 1.0 (visually-fabricated, audio-real)`），是音视一致性检测器的最硬压力测试 |
| **DFE2024** | 评测 | ① 许可 CC-BY-SA-4.0 + gated ToU → **仅评估，禁止训练**；② video 双标签（Video/Audio Ground Truth）独立标注，AV 一致仅 163/815 | 代码层护栏拒绝进 SSL/train；分模态报告（video 视觉 + audio 分开）；163 AV-一致子集作严格跨模态探针 |
| **Celeb-DF-v2** | 评测 | real/synthesis **同人**（idN）配对 | 官方设计，非泄漏；别当身份污染处理 |
| **FF++** | 参考 | 见 §1 | 锁 1000 canonical（720/140/140） |
| **DeepSpeak v1.1** | 训练 | 等 manual-gated 审批；下完需实测 identity-disjoint 是否真成立 | 下完跑 `verify_disjoint` 类检查；若用 v2.0 备选：统一重编码 H.264 CRF18 控 codec、自建 ~10–20% val |
| **AV-Deepfake1M++ / WildDeepfake** | 不接入 | 与 FakeAVCeleb 同源（同样首尾静音捷径伪迹）；占位但不进阵容 | 仅驻留磁盘，不读 |

---

## 3. 训练红线

- **绝不**在任何一个评测集 / 探针集上做参数更新（FakeAVCeleb 仅 B6 探针，never trained）。
- VoxCeleb2 SSL 只用 **real** 片段，且必须开 `disjoint`（剔 37 泄漏 id）。
- DeepSpeak v1.1 是唯一训练源；v1.1 无 avatar 引擎 → THB 的 LivePortrait 保持真零样本（论文 pin v1.1 的原因）。
- 若将来 fallback 用 DeepSpeak v2.0：所有 train+eval 片段**统一重编码 H.264 CRF18** 控 codec 指纹泄漏；训练前 shuffle；自建 val。

---

## 4. 关键文件位置速查

- THB manifests：`D:/Datasets/TalkingHeadBench/real/real_dataset_split.json` (704) / `real_dataset_split_official_ff++.json` (1000)
- HiFi 人脸过滤 master：`D:/Datasets/HiFi-AVDF/face_verify_hifi_master.jsonl`（3810 行，dataloader 按 `decision!=pass` 过滤 183 段 fail_haar；stale 默认 `face_verify_hifi.jsonl` 已删，备份 `.bak` 在）
- B6 静音裁剪报告：`D:/Datasets/FakeAVCeleb_v1.2_split/B6_trim_report.jsonl`（1957 段 `*_trimmed.mp4`）
- VoxCeleb2 排除清单：`configs/exclude_ids.json`（37 个 leak id）
- 划分审计完整报告：`split_correctness_audit.md`
- 论文数据集角色 / 协议：Chapter 4（`Chapters/Chapter4.tex`）

---

## 5. 变更记录

- 2026-07-25：建此文件。锁定 THB/FF++ manifest 决策（A 方案：THB 内 704 / FF++ 参考 1000）；汇总 9 数据集评测坑 + 训练红线 + 文件位置。
- 2026-07-25：HiFi `verify_hifi_faces.py` 默认输出改指 master，删 stale 默认 jsonl（备份 `.bak`）。
- 2026-07-25：Chapter 5 两张表（主结果 `tab:main-results` + 跨数据集 `tab:cross-dataset`）在 HiFi-AVDF 联合行后各加 Seedance 1.0 单独行（visually-fabricated, audio-real）；md HiFi 行同步指向该论文行。
- 2026-07-25：MVAD 三坑修复闭环——`build_mvad_clean.py` 生成 `D:/Datasets/MVAD/mvad_eval_clean.csv`（4731 行，true_generator+content_type+eval_include），`mvad_dataset.py` 为消费入口（剔 299 非 human、按 true_generator 分组）；根因=源仓库把 seedance1_5.zip 嵌套在 jimeng3.5pro/ 下（非我方错误，重解压原样复现）。
- 2026-07-25：**系统性数据集体检**完成 → `DATASET_AUDIT.md`（工作区+thesis 双镜像）。逐数据集只读上盘核查划分/使用/角色：除 MVAD（已修）外未再发现错放/标签污染；THB CVHQ 源视频级重叠 55/100/31 为良性（评测专用，已记录）；FF++/DFE2024/HiFi/Celeb-DF-v2/THB 结构与标签全部吻合计数；DeepSpeak v1.1 等审批、VoxCeleb2 37-id blacklist 正确。
