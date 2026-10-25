#!/bin/bash
# ============================================================================
# cloud_abl_full.sh — 满 GenD 主配置 runner（严格 component-ablation 基线）
# ============================================================================
# 目的：显式复现"主配置 = 满 GenD"（λ_align=0.1 + λ_uniform=0.5 + 同源配对 ON、
#       FTA/LTCA ON、λ_audio=1.0、corr_real_only=1、80ep/batch512/lr1e-4、early-stop），
#       输出到**独立的 ckpt** syncx_gend_seed{s}.pt，与 abl_nob(=Rec B off) / abl_noc(=Rec C off)
#       同代码、同架构、同训练配方进行严格单变量对照。
#   不碰 syncx_best_seed*.pt，不依赖 cloud_settle_all_mvp.sh 的 marker/STEP5d。
#
# 用法（云端，先上传本脚本）：
#   bash cloud_abl_full.sh "0"           # seed0 快速探针
#   bash cloud_abl_full.sh "0 1 2"       # 完整三 seed
#   后台（setsid，勿丢）：
#   setsid bash -c 'bash cloud_abl_full.sh "0 1 2" > $PERSIST/logs/full_rerun.out 2>&1' < /dev/null & disown
#   进度：tail -f $PERSIST/logs/full_rerun.out
# ============================================================================
set -euo pipefail

ISEEDS="${1:-0}"
SYNCX_DATA_ROOT="${SYNCX_DATA_ROOT:-/root/autodl-tmp/SyncX/Datasets}"
PERSIST="${PERSIST:-/root/autodl-tmp/SyncX/persist}"
DEVICE="${DEVICE:-cuda}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')]  $*"; }

# ── 与消融完全一致的训练配方（唯一差异 = 目标开关全部 ON）────────────────────
export SYNCX_COMPILE=0
export SYNCX_FTA=1                       # LTCA 三支 ON（与主模型/消融一致）
export DIRECT_B=0                        # 绝不清主模型权重/续跑点
export SYNCX_LAMBDA_AUDIO=1.0
export SYNCX_CORR_REAL_ONLY=1
export SYNCX_LAMBDA_ALIGN=0.1            # Rec B ON（超球面 metric）
export SYNCX_LAMBDA_UNIFORM=0.5
export SYNCX_NO_SAME_SOURCE=0            # Rec C ON（同源配对，同 config 默认）

STAGE1="$PERSIST/ckpt/stage1.pt"
TRAIN="$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/prep/train"
VAL="$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/prep/test"
LOG="$PERSIST/logs"
TAG="full_gend"

log "满 GenD 主配置复现 seeds=$ISEEDS (λ_align=0.1 λ_uniform=0.5, Rec C ON, FTA=1)"

for s in $ISEEDS; do
  CKPT="$PERSIST/ckpt/syncx_gend_seed${s}.pt"
  log "── [full_gend] Stage-2 训练 seed=$s → $CKPT ──"
  python "$(dirname "$0")/train_stage2.py" \
      --train_cache "$TRAIN" --val_cache "$VAL" \
      --stage1_ckpt "$STAGE1" \
      --out_ckpt "$CKPT" --log_dir "$LOG" \
      --epochs 80 --batch 512 --lr 1e-4 --seed "$s" --device "$DEVICE" \
      --num_workers 4 --empty_cache_every 1 --tag "$TAG" \
      || { log "FAIL train full_gend seed=$s"; exit 1; }
  for pair in "FakeAVCeleb_v1.2:fakeavceleb" "Deepfake-Eval-2024:DFE2024" "HiFi-AVDF:hifi"; do
    ds="${pair%%:*}"; key="${pair##*:}"
    log "── [full_gend] 评测 $key seed=$s ──"
    python "$(dirname "$0")/evaluate.py" \
        --ckpt "$CKPT" --eval_cache "$SYNCX_DATA_ROOT/${ds}/prep" \
        --dataset "$key" --seed "$s" --device "$DEVICE" \
        --log_dir "$LOG" --tag "${TAG}_seed${s}" \
        || log "WARN: eval $key seed=$s failed"
  done
done
log "[full_gend] all done."