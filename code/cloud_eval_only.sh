#!/bin/bash
# ============================================================================
# cloud_eval_only.sh — Sync-X 评测（解耦独立跑，不碰训练）
# ============================================================================
# 用途：训练跑完后，单独启动这一步回填 Chapter5 的结果表（主结果 / ROC / 消融 / B6）。
#       与 cloud_run_all.sh 完全解耦——可以隔几天、换机器、甚至本地跑，只要
#       $PERSIST/ckpt 和评测集 prep 缓存就位即可。所有结果写 logs/eval/<exp_id>/*.jsonl，
#       出图 (figures/plot_*.py) 只读这些日志，零重训。
#
# 前置：
#   1) 训练已产出 $PERSIST/ckpt/syncx_best_seed*.pt
#   2) 评测集已抽取：DFE2024 / HiFi-AVDF / FakeAVCeleb（prep_* 已跑过）
#   3) B6 通路（如要跑 B6）：b6_std_trim.py（需原始视频，本地跑）→ prep_fakeavceleb --mode b6
#
# 用法：
#   export SYNCX_DATA_ROOT=/root/autodl-tmp/SyncX/Datasets
#   bash cloud_eval_only.sh
#   MVP_SEEDS="0 1 2" bash cloud_eval_only.sh     # 多 seed 聚合时
#
# 云端挂机：
#   tmux new -d -s syncx_eval "bash cloud_eval_only.sh"
# ============================================================================
set -uo pipefail

SYNCX_DATA_ROOT="${SYNCX_DATA_ROOT:-/root/autodl-tmp/SyncX/Datasets}"
PERSIST="${PERSIST:-/root/autodl-tmp/SyncX/persist}"
DEVICE="${DEVICE:-cuda}"
MVP_SEEDS="${MVP_SEEDS:-0}"
CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
STATUS_FILE="$PERSIST/status_eval.log"
STAGE2_CKPT="$PERSIST/ckpt/syncx_best.pt"

mkdir -p "$PERSIST/logs/eval" "$PERSIST/ckpt"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')]  $*" | tee -a "$STATUS_FILE"; }
log "=========================================================="
log "Sync-X 评测（解耦独立跑）  种子=$MVP_SEEDS"
log "=========================================================="

# ── STEP 5 — 评测 3 个跨域集 ──────────────────────────────────────────
log "=== STEP 5: 评测 3 跨域集 × seeds($MVP_SEEDS) ==="
for s in $MVP_SEEDS; do
  CKPT="$PERSIST/ckpt/syncx_best_seed${s}.pt"
  [ -f "$CKPT" ] || CKPT="$STAGE2_CKPT"
  for pair in "FakeAVCeleb_v1.2:fakeavceleb" "Deepfake-Eval-2024:DFE2024" "HiFi-AVDF:hifi"; do
    ds_dir="${pair%%:*}"; ds_key="${pair##*:}"
    cache="${SYNCX_DATA_ROOT}/${ds_dir}/prep"
    log "  Evaluating $ds_key seed=$s ($cache)..."
    python "$CODE_DIR/evaluate.py" \
        --ckpt "$CKPT" --eval_cache "$cache" \
        --log_dir "$PERSIST/logs" \
        --dataset "$ds_key" --seed "$s" --device "$DEVICE" --tag "seed${s}" \
        || log "WARN: eval $ds_key seed=$s failed (continuing)"
  done
done

# ── STEP 5a — 论文 tab:forgery-category (F4) + tab:ablation (F5 模态) + 组件消融 b/c/e ──
log "=== STEP 5a: F4 FakeAV 4 类别 + F5 模态消融 + 组件消融 ==="
for s in $MVP_SEEDS; do
  CKPT="$PERSIST/ckpt/syncx_best_seed${s}.pt"
  [ -f "$CKPT" ] || CKPT="$STAGE2_CKPT"
  # F4: FakeAV 按 4 类别分组（ARVR/AFVR/ARVF/AFVF，写 group 字段）
  python "$CODE_DIR/evaluate.py" --ckpt "$CKPT" \
      --eval_cache "$SYNCX_DATA_ROOT/FakeAVCeleb_v1.2/prep" \
      --dataset fakeavceleb --per_category --log_dir "$PERSIST/logs" \
      --seed "$s" --device "$DEVICE" --tag "per_category" \
      || log "WARN: F4 per_category eval failed"
  # F5: 模态消融（HiFi 宿主，与组件消融同源）
  for m in audio visual; do
    python "$CODE_DIR/evaluate.py" --ckpt "$CKPT" \
        --eval_cache "$SYNCX_DATA_ROOT/HiFi-AVDF/prep" \
        --dataset hifi --mask "$m" --log_dir "$PERSIST/logs" \
        --seed "$s" --device "$DEVICE" --tag "mask_${m}" \
        || log "WARN: F5 mask=$m eval failed"
  done
  # 组件消融 (b): w/o DWF；(e): w/o LCMA（用 HiFi 宿主，与 F17 同源）
  python "$CODE_DIR/evaluate.py" --ckpt "$CKPT" \
      --eval_cache "$SYNCX_DATA_ROOT/HiFi-AVDF/prep" \
      --dataset hifi --disable_dwf --log_dir "$PERSIST/logs" \
      --seed "$s" --device "$DEVICE" --tag "wo_dwf" \
      || log "WARN: wo_dwf eval failed"
  python "$CODE_DIR/evaluate.py" --ckpt "$CKPT" \
      --eval_cache "$SYNCX_DATA_ROOT/HiFi-AVDF/prep" \
      --dataset hifi --disable_lcma --log_dir "$PERSIST/logs" \
      --seed "$s" --device "$DEVICE" --tag "wo_lcma" \
      || log "WARN: wo_lcma eval failed"
done

# ── STEP 5b — B6 静音捷径探针（需 b6_std_trim 已跑 + prep_fakeavceleb --mode b6）──
log "=== STEP 5b: B6 静音捷径探针（FakeAVCeleb）==="
for s in $MVP_SEEDS; do
  CKPT="$PERSIST/ckpt/syncx_best_seed${s}.pt"
  [ -f "$CKPT" ] || CKPT="$STAGE2_CKPT"
  python "$CODE_DIR/evaluate.py" --ckpt "$CKPT" \
      --eval_cache "$SYNCX_DATA_ROOT/FakeAVCeleb_v1.2_split/prep/b6_trimmed" \
      --dataset fakeavceleb --b6 --log_dir "$PERSIST/logs" \
      --seed "$s" --device "$DEVICE" --tag "b6_trimmed" \
      || log "WARN: B6 trimmed eval failed"
  python "$CODE_DIR/evaluate.py" --ckpt "$CKPT" \
      --eval_cache "$SYNCX_DATA_ROOT/FakeAVCeleb_v1.2_split/prep/b6_raw" \
      --dataset fakeavceleb --b6_untrimmed --log_dir "$PERSIST/logs" \
      --seed "$s" --device "$DEVICE" --tag "b6_raw" \
      || log "WARN: B6 raw eval failed"
done

# ── STEP 5c — 评测期消融（复用 ckpt，零重训；宿主 HiFi-AVDF）──
log "=== STEP 5c: 评测期消融变体（HiFi-AVDF 宿主）==="
ABL_CACHE="$SYNCX_DATA_ROOT/HiFi-AVDF/prep"
eval_var() {
  python "$CODE_DIR/evaluate.py" --ckpt "$STAGE2_CKPT" --eval_cache "$ABL_CACHE" \
      --dataset hifi --log_dir "$PERSIST/logs" --device "$DEVICE" \
      --tag "$2" $1 || log "WARN: ablation $2 failed"
}
eval_var "--mask audio"   mask_audio_masked
eval_var "--mask visual"  mask_visual_masked
eval_var "--laca_mode corr"     laca_corr
eval_var "--laca_mode biomech"  laca_biomech
eval_var "--disable_lcma" wo_lcma
eval_var "--disable_dwf"  wo_dwf
eval_var "--disable_xs"   wo_xs

log "=========================================================="
log "✅ 评测完成。结果：logs/eval/<exp_id>/*.jsonl"
log "   出图：python figures/plot_figures.py --log_dir $PERSIST/logs"
log "=========================================================="
