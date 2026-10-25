# Sync-X 消融补齐结果汇总（2026-10-24 实测）

> 本文档汇总本轮为回应审稿/评审意见而补齐的三项实验：**模块消融**、**多种子统计**、**载体探针扩样本**。
> 所有数字均来自云端实测日志，未经推断或外推。原始记录见文末「原始产物路径」。

---

## 0. 三项状态总览

| # | 项目 | 状态 | 关键结论 |
|---|---|---|---|
| ① | 模块消融（LCMA / DWF / LACA 两支路） | ✅ 完成 | LCMA 是唯一显著承载模块；其余三者差异在种子噪声内 |
| ② | 单种子 → 三种子统计 | ✅ 完成（只读既有日志，未重训） | FAC 0.6072±0.0222、DFE 0.5212±0.0181、HiFi 0.4833±0.0086 |
| ③ | 载体探针扩样本（600/400 → 3000/3000） | ✅ 完成 | 数字逐位不变；样本量受数据集规模限制，非抽样截断 |

---

## 1. 模块消融（①）

### 1.1 实验协议（与主线 v2 严格对齐）

| 项 | 值 |
|---|---|
| 训练集 | DeepSpeak v1.1 `clips/prep/train`，**10,550** 片段 |
| 验证集 | `clips/prep/test`，**2,913** 片段 |
| 初始化 | `persist/ckpt/stage1.pt`（Stage-1 自监督权重） |
| batch / lr / epochs | 512 / 1e-4 / 80（早停） |
| 采样 | same-source paired sampler ON（28 batches，Rec C 生效） |
| 损失权重 | λ_audio=1.00、λ_corr=0.30、λ_shared=0.10、λ_align=0.10、λ_uniform=0.50 |
| 冻结 | 无（全量微调），与主线一致 |
| LTCA / FTA | OFF，DWF 两分支 |
| 编译 | `SYNCX_COMPILE=0`（必须，见 §5.1） |
| 评测缓存 | FAC `prep_normal` 19,725；DFE `prep/test` 728；HiFi `prep` 3,787 |
| 种子 | seed 0 |

变体定义：

- `wo_lcma`：`--disable_lcma`，去掉跨模态注意力适配器，fused = h_laca
- `wo_dwf`：`--disable_dwf`，可学习门控改为固定 0.5/0.5
- `laca_corr`：`laca_mode=corr`，仅保留唇–音频失配支路
- `laca_biomech`：`laca_mode=biomech`，仅保留生物力学抖动支路（**同时失去 L_shared**，因为该损失定义在 corr 支路上）

### 1.2 结果

| 变体 | 训练域 best val AUC | FakeAVCeleb | DFE-2024 | HiFi-AVDF |
|---|---|---|---|---|
| **v2 全模型（论文主线）** | 0.9758 | 0.6187 | 0.5350 | 0.4735 |
| **w/o LCMA** | **0.7457** | **0.4909** | **0.4757** | **0.4570** |
| w/o DWF（固定 0.5/0.5） | 0.9827 | 0.6287 | 0.5145 | 0.4777 |
| LACA-corr（仅唇–音频） | 0.9839 | 0.6194 | 0.5171 | 0.4870 |
| LACA-biomech（仅生物力学） | 0.9845 | 0.6229 | 0.5207 | 0.4784 |

与基线的差值（跨域）：

| 变体 | FakeAV | DFE | HiFi |
|---|---|---|---|
| w/o LCMA | −0.1278 | −0.0593 | −0.0165 |
| w/o DWF | +0.0100 | −0.0205 | +0.0042 |
| LACA-corr | +0.0007 | −0.0179 | +0.0135 |
| LACA-biomech | +0.0042 | −0.0143 | +0.0049 |

### 1.3 判读（严格口径）

1. **LCMA 是唯一显著承载判别能力的模块。**
   训练域 0.9758 → 0.7457（−0.230）；三个跨域集全部 **低于 0.5**
   （0.4909 / 0.4757 / 0.4570）。低于 0.5 不是「效果变差」，而是**排序系统性反转**：
   DeepSpeak 上学到的判别方向在每个未见生成器上都反向。这与 §16 载体探针的「符号不稳定」
   在行为层完全对应，是负结果论文的一条独立佐证。

2. **DWF 与 LACA 两支路的单体贡献无法在本实验分辨率下分辨。**
   三者与基线差距均 ≤0.021，而主线三种子的标准差为
   FAC ±0.0222 / DFE ±0.0181 / HiFi ±0.0086。因此**不得写成「有效」或「无效」**，
   只能写「本分辨率下无法分辨」。这与 GenD 组件消融的 "trend-level, not significant" 口径一致。

3. **必须披露的口径**：`laca_mode=biomech` 时 corr 支路关闭，`L_shared` 无定义（置 0），
   该行同时损失了支路与其对齐项，不能等同于「仅去掉 biomech 支路」。

4. **FAC 指标陷阱**：real=178 / fake=19,547 极度不平衡，AP≈0.99、
   ACC 0.08–0.90、F1 0.29–0.95 均无信息量（biomech 的 ACC 0.8979 实为「全判假」）。
   **消融表只报 AUC。**

5. **可复现性**：同一配置独立重训两次，wo_lcma / wo_dwf / laca_corr 的 best AUC
   逐位相同（0.7457 / 0.9827 / 0.9839）。因此在固定 seed + 配对采样器下训练是确定性的，
   观察到的差异不是运行噪声。

---

## 2. 三种子统计（②）

数据来源：既有评测日志 `logs/syncx_bs512_lam0.3_lacaboth_seed{0,1,2}_seed{0,1,2}/eval/*.jsonl`，
**只读，未重训、未重评**。

| 数据集 | n seeds | mean ± std | 逐种子 |
|---|---|---|---|
| FakeAVCeleb | 3 | **0.6072 ± 0.0222** | 0.6187 / 0.5816 / 0.6214 |
| DFE-2024 | 3 | **0.5212 ± 0.0181** | 0.5350 / 0.5279 / 0.5007 |
| HiFi-AVDF | 3 | **0.4833 ± 0.0086** | 0.4735 / 0.4873 / 0.4892 |

标准差为样本标准差（ddof=1）。论文主表仍报 seed0 单点（0.6187 / 0.5350 / 0.4735），
与三种子均值一致；上述 σ 用于判断消融差异是否显著。

---

## 3. 载体探针扩样本（③）

- 原设置：`--max-per-real 600 --max-per-fake 400`
- 复测：`--max-per-real 3000 --max-per-fake 3000`
- 结论：**逐生成器统计与论文 `tab:carrier` 逐位相同**

| 生成器 | real | fake | auc | strength | cohen_d |
|---|---|---|---|---|---|
| kling2.5 | 227 | 226 | 0.551 | 0.551 | +0.196 |
| pixverseV5 | 302 | 308 | 0.591 | 0.591 | +0.369 |
| seedance1.0 | 394 | 394 | 0.478 | 0.522 | −0.029 |
| sora2 | 364 | 364 | 0.521 | 0.521 | +0.046 |
| veo3.1 | 314 | 314 | 0.378 | **0.622** | −0.394 |
| wan2.5 | 290 | 290 | 0.521 | 0.521 | +0.097 |

闸门判定：组数 6、强可分（≥0.6）1（veo3.1，且 **sign-reversed**）、符号方向 +4/−2 →
**NOT_SEPARABLE**。

**判读**：每组实际可用片段为 **226–394**，全部低于原上限 600/400 →
**原上限从未生效**，样本量受数据集规模限制，不是抽样截断。上限提高 5 倍后数字零变化，
说明扩大采样无法改变结论。

排除项：DFE-2024 的 `identity` 是逐视频 ID，产生大量 `real=0/fake=1` 单例组，
均因 `<R_MIN` 被跳过，不参与闸门（闸门实际只用 HiFi-AVDF 的 6 个生成器组）。

---

## 4. 代码改动清单

### 4.1 修改的既有文件（2）

| 文件 | 改动 | 原因 |
|---|---|---|
| `models.py` | `SyncX.forward` 增加 `audio_emb_head` | `--disable_lcma` 时 `audio_emb` 为 2d=512，而 `audio_aam` 按 d_model=256 建 → 第一步即崩 |
| `train_stage2.py` | `lshared` 增加 `k_shared/a_shared is None` 判断 | `laca_mode=biomech` 时 corr 支路关闭，返回 None，原代码无条件 `.float()` 崩 |

两处修复都**不新增参数、不改模型结构、不影响已有 ckpt 加载**。

### 4.2 新增脚本（4）

| 文件 | 用途 |
|---|---|
| `train_laca.py` | 给 `train_stage2.py` 补 `laca_mode` 入口（训练端原本无 CLI），通过 monkey-patch `get_config` + 环境变量 `SYNCX_LACA_MODE` |
| `run_module_ablation.py` | 一键跑 4 变体：训练 → 三集评测 → 汇总 CSV |
| `report_seed_stats.py` | 只读日志算 mean±std（对应第②项） |
| `run_carrier_probe_expanded.py` | 先列各组可用条数，再按大上限重跑探针（对应第③项） |

### 4.3 云端产物

| 产物 | 路径 |
|---|---|
| 4 个消融权重 | `persist/ckpt/abl_{wo_lcma,wo_dwf,laca_corr,laca_biomech}_seed0.pt` |
| 模块消融汇总 | `persist/ablation/module_ablation_summary.csv`（12 行） |
| 三种子统计 | `persist/ablation/seed_stats_syncx_bs512_lam0.3_lacaboth_full.csv` |
| 训练日志 | `/tmp/abl_train.log` |
| 12 份评测日志 | `persist/logs/abl_*/eval/*.jsonl` |
| 探针报告 | `persist/diag/carrier_probe_expanded.log`（182 KB） |

---

## 5. 复现命令

### 5.1 训练四个消融变体

```bash
cd /root/autodl-tmp/SyncX/code_cloud
export SYNCX_DATA_ROOT=/root/autodl-tmp/SyncX/Datasets
export SYNCX_PERSIST=/root/autodl-tmp/SyncX/persist
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SYNCX_COMPILE=0          # 必须，否则 cudagraph 必崩

PYTHONUNBUFFERED=1 nohup python run_module_ablation.py --stage train \
  --only wo_lcma,wo_dwf,laca_corr,laca_biomech --batch 512 \
  --stage1_ckpt $SYNCX_PERSIST/ckpt/stage1.pt \
  > /tmp/abl_train.log 2>&1 &
```

### 5.2 评测（12 次）

```bash
P=/root/autodl-tmp/SyncX/persist; LOG=$P/logs
FAC=/root/autodl-tmp/SyncX/Datasets/FakeAVCeleb_v1.2/prep_normal
DFE=/root/autodl-tmp/SyncX/Datasets/Deepfake-Eval-2024/prep/test
HIFI=/root/autodl-tmp/SyncX/Datasets/HiFi-AVDF/prep
for v in wo_lcma wo_dwf laca_corr laca_biomech; do
  case $v in
    wo_lcma) F="--disable_lcma";; wo_dwf) F="--disable_dwf";;
    laca_corr) F="--laca_mode corr";; laca_biomech) F="--laca_mode biomech";;
  esac
  for pair in "FakeAVCeleb:$FAC" "DFE2024:$DFE" "HiFiAVDF:$HIFI"; do
    ds=${pair%%:*}; cache=${pair#*:}
    python evaluate.py --ckpt $P/ckpt/abl_${v}_seed0.pt --eval_cache $cache \
      --log_dir $LOG --exp_id abl_${v}_seed0 --dataset $ds $F
  done
done
```

### 5.3 汇总与两种子统计 / 探针

```bash
python run_module_ablation.py --stage summary --only wo_lcma,wo_dwf,laca_corr,laca_biomech
python report_seed_stats.py --exp_group syncx_bs512_lam0.3_lacaboth
python run_carrier_probe_expanded.py --max_per_real 3000 --max_per_fake 3000
```

---

## 6. 论文改动位置

| 文件 | 位置 | 内容 |
|---|---|---|
| `ResearchPaper.tex` | `\subsection{Module Ablation: LCMA Carries the Discriminative Load}` | 新小节 + 表 `tab:module` |
| `ResearchPaper.tex` | Limitations「Evaluation scale and statistical conventions」 | 226–394、上限从未生效、扩样逐位复现；GenD 与模块消融的非 LCMA 差异落在种子偏差内 |
| `ResearchPaper.tex` | 载体探针段 | `407--712 clips` → `226--394 clips` |
| `GraduationThesis.tex` | 标题页 | Part 2 词数 **7103**（strict count） |
| `ResearchPaper.tex` 顶部注释 | — | `% target 7103 words` |

编译状态：`ResearchPaper.pdf` 11 页、`GraduationThesis.pdf` 42 页、提交件
`36310085-ChrisFeng-AsadMalik-Thesis.pdf` 已覆盖更新；三处词数声明一致，无 LaTeX 错误。

---

## 7. 工程坑（重跑前必读）

1. **必须 `SYNCX_COMPILE=0`**：容器内 `torch.compile(mode=max-autotune)` 在 cudagraph
   内存池处必崩（`_cuda_setCheckpointPoolState` → `Expected curr_block->next == nullptr ...`），
   发生在编译图首次运行时，栈顶落在 `LACA.lip_repr` 属表象。编译只影响速度，关掉无副作用。
2. **禁止 `nohup bash -c '...'` 套娃启动**：曾出现 3 条训练链写同一 `--out_ckpt` / `--exp_id` /
   日志，互相覆盖 `.resume`。应直接 `PYTHONUNBUFFERED=1 nohup python ... &`。
3. **`append_jsonl_dedup` 去重键为 `("epoch","phase")`**：旧 jsonl 未删就重跑，新 run 的 val 行
   会被静默跳过（进度永不更新）。重跑前删 `logs/<eid>/stage2_train.jsonl`。
4. **`run_module_ablation.py` 的跳过逻辑**：`<out_ckpt>` 存在即跳过该变体；若要重训需先删 `.pt`。
5. **进程判断用 `pgrep -af`**：`ps aux | grep` 会截断命令行导致假阴性。
6. **容器内 `dmesg` 无权限**，OOM 只能看 traceback 与 `nvidia-smi`。
7. **FAC 评测必须用 `prep_normal`**（19,725 条）；直接用 `prep` 会混入 17,849 条 B6 探针缓存，AUC 全错。

---

## 8. 未做 / 边界

- 消融为 **单 seed**；结论已按「落在种子噪声内」保守表述，未做多 seed 扩充。
- 第②项使用既有日志，**未重训**；因此三种子统计对应的是主线配置，不含消融变体。
- 会议投稿版：已同步 4 个自包含版本（isdfs2027_negresult / online isdfs2027 / acit2027 /
  ifip_aiai2027）的载体探针样本量（407–712 → 226–394）；四版均已回到各自页数上限
  （6 / 6 / 10 / 14）。按方案 A，模块消融内容只保留在学位论文，不进入会议短版，
  因为四版均已顶满页数，加入 45 词即多 1 页。原 8 个 Chapter 结构版本不含 carrier probe，未改。
- 原 `result/v1/evidence_chain_crossdomain.md` 已被删除；本文件替代其汇总职能，
  但不替代其逐节证据链结构。
