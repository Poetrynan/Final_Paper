# Sync-X 图件流水线（F1–F16 零重训重画）

所有图只读取训练/评测/推理留下的 **JSONL 日志**，绝不触发 train/eval/infer。
改样式、补图、换配色都无需重跑实验。

## 1. exp_id / exp_group 命名规则

- **exp_id**（编码超参 + seed）：`syncx_bs{bs}_lam{lambda}_laca{mode}[_tag]_seed{seed}`
  - 例：`syncx_bs256_lam0.3_lacaboth_seed0`（默认）
  - A2 from-scratch：`..._scratch_seed0`；A3 λ 扫描：`..._lam1.0_lacaboth_seed0`
  - 实现：`logging_utils.exp_id(cfg, seed, tag="")`；`exp_group(eid)` 去掉 `_seedN` 用于跨 seed 聚合。
- **目录**：`logs/<exp_id>/{stage1_train.jsonl, stage2_train.jsonl, eval/<dataset>__<variant>.jsonl,
  interpret/<case_id>.jsonl, features/clip_features.jsonl, features/clip_embeddings.jsonl, profile.jsonl}`
- **基线/文献值**写到种子无关的组目录 `logs/<exp_group>/eval/...`（与所有 seed 的 Sync-X 聚合）；
  `plot_figures --exp_group` 会 glob `logs/<exp_group>*` 合并 Sync-X + 该组基线。

## 2. 生产日志（只需跑一次）

```bash
# 训练（每 K 步 + epoch 末 val 落盘）
python train_stage1.py --real_cache ... --out_ckpt ckpt/stage1.pt --val_cache ... \
    --seed 0 --log_dir logs --exp_id syncx_bs256_lam0.3_lacaboth_seed0 --log_every 100
python train_stage2.py --train_cache ... --val_cache ... --stage1_ckpt ckpt/stage1.pt \
    --out_ckpt ckpt/syncx_best.pt --seed 0 --log_dir logs --exp_id syncx_bs256_lam0.3_lacaboth_seed0 --log_every 100

# 跨域评测（主 + 分组）
python evaluate.py --ckpt ckpt/syncx_best.pt --eval_cache ... --log_dir logs \
    --exp_id syncx_bs256_lam0.3_lacaboth_seed0 --dataset DeepfakeEval2024
python evaluate.py ... --per_group_regex "(ARVR|AFVR|ARVF|AFVF)"   # F4
python evaluate.py ... --mask audio                               # F5 (audio-only)
python evaluate.py ... --laca_mode corr                            # F8
python evaluate.py ... --b6                                      # F9 (B6 trimmed)
python evaluate.py ... --b6_untrimmed                             # F9 (B6 raw baseline)
python evaluate.py ... --exclude_ids configs/exclude_ids.json     # F10 (B7)

# 基线 / 文献值注入（如 MDS，F9）
python log_external.py --log_dir logs --exp_group syncx_bs256_lam0.3_lacaboth \
    --dataset FakeAVCeleb --method mds --variant b6_trimmed --auc 73.8

# 精选案例 / 特征 / 嵌入 / 剖析（F11/F12/F15/F13）
python dump_interpret.py --ckpt ckpt/syncx_best.pt \
    --clip "real01|.../real.pt|0" --clip "fake01|.../fake.pt|1|12-28" \
    --log_dir logs --exp_id syncx_bs256_lam0.3_lacaboth_seed0
python dump_features.py   --ckpt ckpt/syncx_best.pt --cache ... --dataset DeepSpeak --sample_n 400 \
    --log_dir logs --exp_id syncx_bs256_lam0.3_lacaboth_seed0
python dump_embeddings.py --ckpt ckpt/syncx_best.pt --cache ... --dataset DeepSpeak --sample_n 400 \
    --log_dir logs --exp_id syncx_bs256_lam0.3_lacaboth_seed0
python bench_profile.py --ckpt ckpt/syncx_best.pt --exp_id syncx_bs256_lam0.3_lacaboth_seed0 --steps_per_epoch 1000
```

## 3. 出图（只用日志，CPU 即可）

```bash
python plot_figures.py \
    --log_dir logs \
    --exp_id syncx_bs256_lam0.3_lacaboth_seed0 \
    --exp_group syncx_bs256_lam0.3_lacaboth \
    --exp_group_b syncx_bs256_lam0.3_lacaboth_scratch \          # F6 对照
    --lam_scan syncx_bs256_lam0.1_lacaboth,syncx_bs256_lam0.3_lacaboth,syncx_bs256_lam1.0_lacaboth \  # F7
    --out out/figures
```

- 输出 `out/figures/F{1..16}.pdf`（矢量）+ `.png`（600 dpi）。
- 任一所需日志缺失 → 该图 SKIP 并打印缺哪个文件（不静默重跑）。
- 样式：默认 IEEE（`plot_style.VENUE="ieee"`）；改投 Nature/CVPR 仅翻该常量。
- 时间轴纪律：F1 训练 loss 按 **global_step**（每 K 步落盘，EMA 平滑）；val 按各自 step 位置叠加；epoch 由 `global_step/steps_per_epoch` 反推。
