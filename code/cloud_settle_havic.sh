#!/bin/bash
# ============================================================================
# cloud_settle_havic.sh — §14 HAVIC 式一致性重建先验一键挂机（补全三层，验证方向）
# ============================================================================
# 动机（§14）：前半程把 HAVIC 三层一致性先验退化成一层 InfoNCE（仅对齐、不重建），
#   且一致性监督被训练集 fake 样本掺杂——这是"文献支持却实测失效"最可能漏掉的细节。
#   本脚本把结构层(MaskedCrossRecon)与宏层(CrossSemanticRecon)补回 **Stage-1 纯真实**
#   自监督（avoid fake 掺杂），然后走与 baseline 完全相同的 Stage-2 配方，
#   单变量判定：补全 HAVIC 先验是否让"真实一致性"跨生成器可迁移。
#
# 关键设计（单变量守卫）：
#   ✅ Stage-1 带 §14 重建自监督重训（SYNCX_HAVIC_RECON=1）→ 写【新】stage1_havic.pt
#   ✅ Stage-2 走 baseline 配方（无对齐损失 SYNCX_ALIGN_LOSS=0；LTCA/FTA=1 与基线一致）
#   ✅ 输出全部用【新】路径，绝不覆盖已有 stage1.pt / syncx_best_seed0.pt
#   ✅ 复用已有 prep cache（VoxCeleb2 特征 / DeepSpeak prep / 3 个跨域 prep），零重抽
#
# 判据（与 §13 探针一致，方向复活闸门）：
#   对 DFE2024 与 HiFi-AVDF 两跨域集，比较 HAVIC 模型(syncx_havic_best_seed0)与
#   baseline(syncx_best_seed0) 的 AUC 是否提升 且 diag_real_prior 的 corr 类统计量
#   within-set auc≥0.65 且真实跨源漂移<类间差距 → 方向复活；否则收笔为如实否定。
#
# 用法（云端）：
#   cd /root/autodl-tmp/SyncX/code_cloud
#   unzip -o code_cloud_havic.zip
#   tmux new -d -s havic "bash cloud_settle_havic.sh"     # 推荐，可重连
#   tmux a -t havic
# 可选环境变量：
#   EPOCHS_S1    Stage-1 轮数（默认 40，与 baseline 一致）
#   EPOCHS_S2    Stage-2 轮数（默认 80，与 baseline 一致）
#   BATCH_S1     Stage-1 batch（默认 1024，与 baseline 一致）
#   SEED         seed（默认 0）
#   FRESH=1      强制清本 pilot 旧 ckpt/resume/marker 重新起
# ============================================================================
set -euo pipefail

SYNCX_DATA_ROOT="${SYNCX_DATA_ROOT:-/root/autodl-tmp/SyncX/Datasets}"
HF_HOME="${HF_HOME:-/root/autodl-tmp/SyncX/hfcache}"
PERSIST="${PERSIST:-/root/autodl-tmp/SyncX/persist}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
EPOCHS_S1="${EPOCHS_S1:-40}"
EPOCHS_S2="${EPOCHS_S2:-80}"
BATCH_S1="${BATCH_S1:-1024}"
LAMBDA_MASK="${LAMBDA_MASK:-0.5}"
LAMBDA_SEM="${LAMBDA_SEM:-0.5}"
MASK_RATIO="${MASK_RATIO:-0.25}"

# ── 硬性环境（内联，防丢失）──────────────────────────────────────────────
export SYNCX_COMPILE=0                       # 4090D 上 torch.compile 启动即崩（红线）
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SYNCX_MEDIAPIPE_GPU=0                 # 防御：评测若触抽取也走 CPU 防 OOM
export SYNCX_FTA=1                           # LTCA 开，与 baseline(方向B) 语义一致，单变量
export SYNCX_HAVIC_RECON=1                   # §14：开启 HAVIC 三层重建先验
export SYNCX_HAV_LAMBDA_MASK="$LAMBDA_MASK"
export SYNCX_HAV_LAMBDA_SEM="$LAMBDA_SEM"
export SYNCX_HAV_MASK_RATIO="$MASK_RATIO"

CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
STATUS_FILE="$PERSIST/status_havic.log"
MARKER_DIR="$PERSIST/.markers_havic"
mkdir -p "$MARKER_DIR"
log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')]  $*" | tee -a "$STATUS_FILE"; }
done_marker() { touch "$MARKER_DIR/$1.done"; }
is_done()    { [ -f "$MARKER_DIR/$1.done" ]; }

S1_CKPT="$PERSIST/ckpt/stage1_havic.pt"          # 【新】不覆盖 stage1.pt
S2_CKPT="$PERSIST/ckpt/syncx_havic_best_seed${SEED}.pt"   # 【新】不覆盖 baseline
BASE_CKPT="$PERSIST/ckpt/syncx_best_seed${SEED}.pt"       # baseline 参照，只读

# ── FRESH=1：清本 pilot 旧产物重新开始（换 λ / 修正后必用，架构变更需删 .resume）──
if [ "${FRESH:-0}" = "1" ]; then
  log "[FRESH] 清空 §14 试点旧产物"
  rm -f "$S1_CKPT" "$S1_CKPT.resume"
  rm -f "$S2_CKPT" "$S2_CKPT.resume"
  rm -f "$MARKER_DIR"/*.done
fi

# ── 依赖自检（复用已有 prep，绝不重抽）────────────────────────────────────
for _d in "$SYNCX_DATA_ROOT/VoxCeleb2_subset/prep" \
          "$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/prep/train" \
          "$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/prep/test" \
          "$SYNCX_DATA_ROOT/FakeAVCeleb_v1.2/prep" \
          "$SYNCX_DATA_ROOT/Deepfake-Eval-2024/prep" \
          "$SYNCX_DATA_ROOT/HiFi-AVDF/prep" \
          "$BASE_CKPT"; do
  [ -e "$_d" ] || { log "FATAL: 缺失依赖 $_d —— 拒绝重抽，补齐后再跑。"; exit 1; }
done

log "=========================================================="
log "§14 HAVIC 重建先验试点启动 seed=$SEED EPOCHS_S1=$EPOCHS_S1 EPOCHS_S2=$EPOCHS_S2"
log "  λ_mask=$LAMBDA_MASK λ_sem=$LAMBDA_SEM mask_ratio=$MASK_RATIO compile=$SYNCX_COMPILE"
log "  数据根: $SYNCX_DATA_ROOT   持久卷: $PERSIST"
log "  输出: $S1_CKPT → $S2_CKPT"
log "  参照 baseline(只读): $BASE_CKPT"
log "=========================================================="

# ============================================================================
# STEP 1 — Stage-1 带 HAVIC 重建先验重训（仅真实 VoxCeleb2，40 epoch 同 baseline）
# ============================================================================
STEP="s1_havic"
if is_done "$STEP"; then
  log "SKIP: $STEP (marker 已存在)"
else
  if [ -f "$S1_CKPT.resume" ]; then
    log " 清旧 resume（架构新增重建头，旧 resume 不兼容严格加载）"
    rm -f "$S1_CKPT.resume"
  fi
  log "=== STEP: Stage-1 HAVIC 重训（λ_mask=$LAMBDA_MASK λ_sem=$LAMBDA_SEM）==="
  python "$CODE_DIR/train_stage1.py" \
      --real_cache "$SYNCX_DATA_ROOT/VoxCeleb2_subset/prep" \
      --out_ckpt   "$S1_CKPT" \
      --log_dir    "$PERSIST/logs" \
      --epochs "$EPOCHS_S1" --batch "$BATCH_S1" --lr 1e-4 --seed "$SEED" --device "$DEVICE" \
      --num_workers 8 --prefetch_factor 4 --empty_cache_every 1 \
      || { log "FAIL: Stage-1 HAVIC seed=$SEED"; exit 1; }
  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
# STEP 2 — Stage-2 用 HAVIC Stage-1 重训（标准配方：无对齐、LTCA on）
# ============================================================================
STEP="s2_havic"
if is_done "$STEP"; then
  log "SKIP: $STEP (marker 已存在)"
else
  log "=== STEP: Stage-2 HAVIC 重训（从 $S1_CKPT，baseline 配方）==="
  export SYNCX_ALIGN_LOSS=0    # 与 baseline 一致：本 pilot 换的是 Stage-1 先验，不是对齐监督
  python "$CODE_DIR/train_stage2.py" \
      --train_cache "$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/prep/train" \
      --val_cache   "$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/prep/test" \
      --stage1_ckpt "$S1_CKPT" \
      --out_ckpt    "$S2_CKPT" \
      --log_dir     "$PERSIST/logs" \
      --epochs "$EPOCHS_S2" --batch 512 --lr 1e-4 --seed "$SEED" --device "$DEVICE" \
      --num_workers 8 --prefetch_factor 4 --empty_cache_every 1 \
      --tag "havic_s2" \
      || { log "FAIL: Stage-2 HAVIC seed=$SEED"; exit 1; }
  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
# STEP 3 — 3 个跨域集评测（HAVIC 模型 + baseline 对照）
# ============================================================================
eval_ckpt() {  # $1=ckpt  $2=tag
  CK="$1"; TAG="$2"
  for pair in "FakeAVCeleb_v1.2:fakeavceleb" "Deepfake-Eval-2024:DFE2024" "HiFi-AVDF:hifi"; do
    ds_dir="${pair%%:*}"; ds_key="${pair##*:}"
    cache="${SYNCX_DATA_ROOT}/${ds_dir}/prep"
    log "  Eval $TAG $ds_key ($cache)..."
    python "$CODE_DIR/evaluate.py" \
        --ckpt "$CK" --eval_cache "$cache" \
        --log_dir "$PERSIST/logs" \
        --dataset "$ds_key" --seed "$SEED" --device "$DEVICE" --tag "$TAG" \
        || log "WARN: eval $ds_key $TAG failed (continuing)"
  done
}

STEP="eval_havic_vs_base"
if is_done "$STEP"; then
  log "SKIP: $STEP (marker 已存在)"
else
  log "=== STEP: 3 跨域集评测（HAVIC vs baseline 同配方对照）==="
  eval_ckpt "$S2_CKPT"     "havic_vs_base"
  eval_ckpt "$BASE_CKPT"   "baseline_ref"
  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
# STEP 4 — 真实一致先验探针：HAVIC 模型是否让"真实一致性"跨生成器可分离
# ============================================================================
STEP="probe_real_prior"
if is_done "$STEP"; then
  log "SKIP: $STEP (marker 已存在)"
else
  log "=== STEP: diag_real_prior 探针（HAVIC 模型）==="
  python "$CODE_DIR/diag_real_prior.py" \
      --ckpt "$S2_CKPT" --device "$DEVICE" --max-per-class 600 \
      --cache "$SYNCX_DATA_ROOT/VoxCeleb2_subset/prep" \
      --cache "$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/prep/test" \
      --cache "$SYNCX_DATA_ROOT/HiFi-AVDF/prep" \
      || log "WARN: diag_real_prior 探针失败"
  done_marker "$STEP"; log "$STEP DONE"
fi

log "=========================================================="
log "✅ §14 HAVIC 试点完成。"
log "   对照：syncx_havic_best_seed${SEED}.pt  vs  baseline syncx_best_seed${SEED}.pt"
log "   判定：DFE2024 / HiFi-AVDF 两集 AUC 相对 baseline 提升且探针 corr within-set≥0.65"
log "         且真实跨源漂移<类间差距 → 方向复活；否则收笔为如实否定。"
log "=========================================================="