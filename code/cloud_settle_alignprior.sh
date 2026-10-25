#!/bin/bash
# ============================================================================
# cloud_settle_alignprior.sh — §13 端到端对齐试点一键挂机（B / A / C 三组）
# ============================================================================
# 目的：验证探针证伪的是「离线冻结特征框架」而非「一致先验方向」，用一次廉价
#       Stage-2 重训判定"端到端对齐能否把真实一致性压成跨生成器可迁移"。
#
# 三组（ALIGN_GROUP 控制，默认 B）：
#   B = 全量微调(不冻视觉投影) + 显式逐帧对齐监督 L_align_prior(SYNCX_ALIGN_LOSS=1)
#       ——§13 主试组：特征随"真实一致性要高"一起学
#   A = 保留 §8D 基线配方（全量微调、无对齐、SYNCX_ALIGN_LOSS=0，仅 FTA 语义）
#       ——归零参照，复用现有 syncx_best_seed0 亦等价，但自跑一版保证同源对照
#   C = 只放开视觉投影、不加对齐（SYNCX_ALIGN_LOSS=0 + 不冻结）
#       ——解耦"放开投影" vs "加对齐"，区分哪一环在起作用
#
# 依赖（复用现有产物，绝不重抽/不重训 Stage-1，见注释下方"不重做"清单）：
#   - 已抽特征: DeepSpeak train/test prep cache（本脚本不触发 prep）
#   - Stage-1 权重: $STAGE1_CKPT（LCMACA 未改，直接复用）
#   - 三个跨域评测集 cache 已存在（FakeAV / DFE2024 / HiFi）
#
# 不重做（单变量守卫核心）：特征抽取、Stage-1 SSL 预训练全部跳过；
#   本脚本只重训 Stage-2 并自动评测，换的是"监督配方"这一环。
#
# 用法（云端）：
#   cd /root/autodl-tmp/SyncX/code_cloud
#   unzip -o stage2_alignprior_beta.zip
#   # B 组（主试，推荐）：
#   tmux new -d -s b "bash cloud_settle_alignprior.sh"
#   # 若要跑 A/C 对照：
#   tmux new -d -s a "ALIGN_GROUP=A bash cloud_settle_alignprior.sh"
#   tmux new -d -s c "ALIGN_GROUP=C bash cloud_settle_alignprior.sh"
#   tmux a -t b
#
# 环境变量：
#   ALIGN_GROUP   B|A|C（默认 B）
#   ALIGN_WEIGHT  对齐损失权重（默认 0.1；B 组用）
#   SEEDS         "0"（默认）；多 seed 用 "0 1"
#   EPOCHS        Stage-2 轮数（默认 40，试点加速）
#   SYNCX_COMPILE 0（硬性默认；Stage-2 在 4090D 上 torch.compile 启动即崩）
#
# 幂等：每步 marker 防重跑；FRESH=1 强制清旧 marker+旧 ckpt 重新开始。
# ============================================================================
set -euo pipefail

SYNCX_DATA_ROOT="${SYNCX_DATA_ROOT:-/root/autodl-tmp/SyncX/Datasets}"
HF_HOME="${HF_HOME:-/root/autodl-tmp/SyncX/hfcache}"
PERSIST="${PERSIST:-/root/autodl-tmp/SyncX/persist}"
DEVICE="${DEVICE:-cuda}"
GROUP="${ALIGN_GROUP:-B}"
WEIGHT="${ALIGN_WEIGHT:-0.1}"
SEEDS="${SEEDS:-0}"
EPOCHS="${EPOCHS:-40}"

# ── 硬性环境（内联，防丢失）：Stage-2 在 4090D 上 torch.compile 启动即崩 → 必须关 ──
export SYNCX_COMPILE=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SYNCX_MEDIAPIPE_GPU=0                       # 防御：评测集若触发抽取也走 CPU 防 OOM
# 方向B 语义保持一致（LTCA 第三分支在 B/A/C 三组统一开启，单变量只在"对齐监督"）
export SYNCX_FTA=1

CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
STATUS_FILE="$PERSIST/status_alignprior.log"
MARKER_DIR="$PERSIST/.markers_alignprior"
mkdir -p "$MARKER_DIR"
log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')]  $*" | tee -a "$STATUS_FILE"; }
done_marker() { touch "$MARKER_DIR/$1.done"; }
is_done()    { [ -f "$MARKER_DIR/$1.done" ]; }

STAGE1_CKPT="$PERSIST/ckpt/stage1.pt"
TAG="group${GROUP}"

# ── FRESH=1：强制清空本试点旧产物，重新开始（换 λ / 换组 / 修正后必用）──
if [ "${FRESH:-0}" = "1" ]; then
  log "[FRESH] 清空 §13 试点(group=$TAG)旧 ckpt/resume/marker"
  rm -f "$PERSIST/ckpt"/alignprior_${TAG}*.pt "$PERSIST/ckpt"/alignprior_${TAG}*.resume
  rm -f "$MARKER_DIR"/${TAG}_*.done
fi

# ── 试运行依赖自检：特征与 Stage-1 必须已在，否则直接报错（不重抽是纪律）──
TRAIN_CACHE="$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/prep/train"
VAL_CACHE="$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/prep/test"
for _d in "$TRAIN_CACHE" "$VAL_CACHE" "$STAGE1_CKPT" \
          "$SYNCX_DATA_ROOT/FakeAVCeleb_v1.2/prep" "$SYNCX_DATA_ROOT/Deepfake-Eval-2024/prep" "$SYNCX_DATA_ROOT/HiFi-AVDF/prep"; do
  [ -e "$_d" ] || { log "FATAL: 缺失依赖 $_d —— 拒绝重抽，补齐后再跑。"; exit 1; }
done
log "=========================================================="
log "§13 端到端对齐试点启动  group=$GROUP(λ_align_prior=$WEIGHT)  seeds=[$SEEDS]  epochs=$EPOCHS  compile=$SYNCX_COMPILE"
log "  数据根: $SYNCX_DATA_ROOT   持久卷: $PERSIST"
log "  输出 ckpt: $PERSIST/ckpt/alignprior_${TAG}_best_seed<seed>.pt"
log "  不重抽特征 / 不重训 Stage-1（复用 $STAGE1_CKPT）"
log "=========================================================="

# ============================================================================
# STEP 1 — Stage-2 对齐试点重训（换监督配方，单变量）
# ============================================================================
for s in $SEEDS; do
  CSP=$PERSIST/ckpt/alignprior_${TAG}_best_seed${s}.pt
  STEP="train_stage2_${TAG}_seed${s}"
  if is_done "$STEP"; then
    log "SKIP: $STEP (marker 已存在)"; continue
  fi
  log "=== STEP: Stage-2 重训 group=$GROUP seed=$s (λ_align_prior=$WEIGHT, epochs=$EPOCHS) ==="
  # 清该 seed 旧 resume，确保从新配方干净起训（新增监督不改架构，但防旧优化器状态污染）
  rm -f "$CSP.resume"
  # 组配方：B=对齐监督；A=基线(无对齐);C=无对齐但等价全量微调
  export SYNCX_ALIGN_LOSS=0    # 默认关
  if [ "$GROUP" = "B" ]; then
    export SYNCX_ALIGN_LOSS=1 SYNCX_ALIGN_WEIGHT="$WEIGHT"
  fi
  python "$CODE_DIR/train_stage2.py" \
      --train_cache "$TRAIN_CACHE" \
      --val_cache   "$VAL_CACHE" \
      --stage1_ckpt "$STAGE1_CKPT" \
      --out_ckpt    "$CSP" \
      --log_dir     "$PERSIST/logs" \
      --epochs "$EPOCHS" --batch 512 --lr 1e-4 --seed "$s" --device "$DEVICE" \
      --num_workers 4 --empty_cache_every 1 --tag "alignprior_$TAG" \
      || { log "FAIL: Stage-2 group=$GROUP seed=$s"; exit 1; }
  done_marker "$STEP"; log "$STEP DONE"
done

# ============================================================================
# STEP 2 — 评测 3 个跨域集（自动；复用训练产物 ckpt）
# ============================================================================
for s in $SEEDS; do
  CKPT="$PERSIST/ckpt/alignprior_${TAG}_best_seed${s}.pt"
  for pair in "FakeAVCeleb_v1.2:fakeavceleb" "Deepfake-Eval-2024:DFE2024" "HiFi-AVDF:hifi"; do
    ds_dir="${pair%%:*}"; ds_key="${pair##*:}"
    cache="${SYNCX_DATA_ROOT}/${ds_dir}/prep"
    log "  Evaluating group=$TAG $ds_key seed=$s ($cache)..."
    python "$CODE_DIR/evaluate.py" \
        --ckpt "$CKPT" --eval_cache "$cache" \
        --log_dir "$PERSIST/logs" \
        --dataset "$ds_key" --seed "$s" --device "$DEVICE" --tag "alignprior_${TAG}" \
        || log "WARN: eval $ds_key seed=$s failed (continuing)"
  done
done

log "=========================================================="
log "🎉 §13 试点 group=$TAG 完成。"
log "   判死/复活闸门：在 DFE2024 与 HiFi 两个跨域集上，用 diag_real_prior.py"
log "   重测本组 ckpt（--ckpt $PERSIST/ckpt/alignprior_${TAG}_best_seed0.pt），"
log "   corr 类统计量 within-set auc≥0.65 且真实跨源漂移<差距 → 方向复活；否则收笔。"
log "=========================================================="