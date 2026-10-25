# 组件消融运行清单（w/o LCMA / w/o DWF）

> 本清单对应 `tab:component-ablation`（Ch5 §component-ablation）。
> 开关已接入：`config.py`(字段) / `models.py`(forward 短路) / `train_stage2.py`(训练) /
> `evaluate.py`(评测 variant) / `logging_utils.exp_id`(日志隔离) / `plot_figures.plot_component_ablation`(F17)。

## 1. 训练两个消融模型（在 DeepSpeak v1.1 上，需 GPU + 已提取特征缓存）

```bash
# 完整基线（对照）
python train_stage2.py \
    --train_cache ./feature_cache_deepspeak/train \
    --val_cache   ./feature_cache_deepspeak/val \
    --stage1_ckpt ./ckpt/stage1.pt \
    --out_ckpt    ./ckpt/syncx_best.pt --epochs 80 --batch 256 --lr 1e-4

# w/o LCMA：仅 LACA 分支（fused = h_laca）
python train_stage2.py \
    --train_cache ./feature_cache_deepspeak/train \
    --val_cache   ./feature_cache_deepspeak/val \
    --stage1_ckpt ./ckpt/stage1.pt \
    --out_ckpt    ./ckpt/syncx_nolcma.pt --epochs 80 --batch 256 --lr 1e-4 \
    --disable_lcma

# w/o DWF：固定 0.5/0.5 平均融合（关闭可学习门控）
python train_stage2.py \
    --train_cache ./feature_cache_deepspeak/train \
    --val_cache   ./feature_cache_deepspeak/val \
    --stage1_ckpt ./ckpt/stage1.pt \
    --out_ckpt    ./ckpt/syncx_nodwf.pt --epochs 80 --batch 256 --lr 1e-4 \
    --disable_dwf
```

日志自动落到互不覆盖的目录（exp_id 带 `_nolcma` / `_nodwf`）：
`logs/syncx_bs256_lam0.3_lacaboth[_nolcma|_nodwf]_seed0/stage2_train.jsonl`

## 2. 跨域评测（为每个评测集生成 wo_lcma / wo_dwf / full 三条 variant 日志）

```bash
for DSET in eval2024 fakeavceleb celebdf thb mvad hifi; do
  # 完整基线
  python evaluate.py --ckpt ./ckpt/syncx_best.pt  --eval_cache ./feature_cache_${DSET}_test
  # w/o LCMA
  python evaluate.py --ckpt ./ckpt/syncx_nolcma.pt --eval_cache ./feature_cache_${DSET}_test --disable_lcma
  # w/o DWF
  python evaluate.py --ckpt ./ckpt/syncx_nodwf.pt  --eval_cache ./feature_cache_${DSET}_test --disable_dwf
done
```

> `--disable_*` 也可省略：评测时会从 ckpt 内 cfg 自动推导 wo_* variant，避免标错。

## 3. 出图（含新增 F17 组件消融图）

```bash
python plot_figures.py --log_dir logs \
    --exp_id syncx_bs256_lam0.3_lacaboth_seed0 \
    --exp_group syncx_bs256_lam0.3_lacaboth \
    --out out/figures
```

`out/figures/F17.pdf` = 组件消融（full vs w/o LCMA vs w/o DWF）跨数据集 AUC。

## 4. 回填论文

把各评测集上 `full / wo_lcma / wo_dwf` 的 AUC 填入 `Chapters/Chapter5.tex` 的
`tab:component-ablation`（当前全 `--` 占位）。

## 注意

- 本沙箱未安装 torch，且 DeepSpeak v1.1 主源数据集尚未审批/下载，故上机运行需在
  **本机 GPU 环境**（已装 torch + 已提取特征缓存）执行。代码改动已通过 `py_compile` 语法校验。
- B6 命名 bug（`_variant_from_args` 末行 `return "full"` 应为 `return "b6_untrimmed"`）**未修**，
  与本次组件消融相互独立，待你授权后再改。
