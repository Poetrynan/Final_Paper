#!/bin/bash
# ============================================================================
# cloud_settle_all.sh — Sync-X 全流程一键挂机（AutoDL / 任意 Linux GPU 云）
# ============================================================================
# 用法：
#   export SYNCX_DATA_ROOT=/root/autodl-tmp/SyncX/Datasets      # 数据集挂载点
#   export HF_TOKEN=hf_xxx                     # DeepSpeak 门控（可选但建议）
#   export SKIP_PREP=0   SKIP_SEARCH=0   SEED=0
#   bash cloud_settle_all.sh
#
# 特点：
#   - 幂等：中断后重跑同一命令 → 从断点继续，不重头开始
#   - 每步落 checkpoint + resume（Stage-1/2 每 epoch 末原子写 .resume）
#   - 已完成阶段写 marker 文件，重跑时自动跳过
#   - 超参搜索按论文留白项策略（λ_align×λ_uniform，可 SKIP_SEARCH=1 跳过）
#   - 架构决定：Epoch-based 训练（非 step-based）—— resume 粒度 = 1 epoch
#     原因：每个 epoch 末自动原子落 .resume，最坏丢 1 epoch（T4≈15-20min），
#           step 粒度在大实验中保存开销过高、日志膨胀、且当前代码就是 epoch 设计
# ============================================================================
set -euo pipefail

# ── 环境变量（支持外部覆盖）────────────────────────────────────────────
SYNCX_DATA_ROOT="${SYNCX_DATA_ROOT:-/root/autodl-tmp/SyncX/Datasets}"
HF_HOME="${HF_HOME:-/root/autodl-tmp/SyncX/hfcache}"
HF_TOKEN="${HF_TOKEN:-}"
PERSIST="${PERSIST:-/root/autodl-tmp/SyncX/persist}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
SKIP_PREP="${SKIP_PREP:-0}"
SKIP_SEARCH="${SKIP_SEARCH:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

# ── 路径 ───────────────────────────────────────────────────────────────
CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
STATUS_FILE="$PERSIST/status.log"
MARKER_DIR="$PERSIST/.markers"
mkdir -p "$PERSIST/ckpt" "$PERSIST/logs" "$MARKER_DIR"

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')]  $*" | tee -a "$STATUS_FILE"; }
done_marker() { touch "$MARKER_DIR/$1.done"; }
is_done()    { [ -f "$MARKER_DIR/$1.done" ]; }

STAGE1_CKPT="$PERSIST/ckpt/stage1.pt"
STAGE2_CKPT="$PERSIST/ckpt/syncx_best.pt"

# ============================================================================
log "=========================================================="
log "Sync-X 全流程启动"
log "  数据根: $SYNCX_DATA_ROOT"
log "  持久卷: $PERSIST"
log "  设备:   $DEVICE  种子: $SEED"
log "  跳过:   PREP=$SKIP_PREP  SEARCH=$SKIP_SEARCH  EVAL=$SKIP_EVAL"
log "=========================================================="

# ============================================================================
# STEP 0 — Cloud environment check（必须跑通）
# ============================================================================
STEP="0_cloud_check"
if is_done "$STEP"; then
  log "SKIP: $STEP (marker 已存在)"
else
  log "=== STEP 0: 云端环境自检 ==="
  python "$CODE_DIR/_verify_cloud.py" "$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/real/14-5467.mp4"
  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
# STEP 1 — 8 个数据集特征抽取（每 clip 已抽则跳过，幂等）
# ============================================================================
STEP="1_extraction"
if [ "$SKIP_PREP" = "1" ]; then
  log "SKIP: $STEP (SKIP_PREP=1)"
elif is_done "$STEP"; then
  log "SKIP: $STEP (marker 已存在)"
else
  log "=== STEP 1: 特征抽取（串行 ≈35h；PREP_PARALLEL=1 时用多核并行）==="
  # B6 前置: prep_fakeavceleb_b6.py 依赖 b6_std_trim.py 产出的 b6_std/trimmed；缺则先生成
  if [ ! -d "$SYNCX_DATA_ROOT/FakeAVCeleb_v1.2_split/b6_std/trimmed" ]; then
    log "  > 预生成 B6 Smeu trim（缺 b6_std/trimmed）"
    python "$CODE_DIR/b6_std_trim.py" || { log "FAIL: b6_std_trim"; exit 1; }
  fi
  PREP_LIST="prep_deepspeak.py prep_voxceleb2.py prep_celebdf.py prep_eval2024.py \
             prep_thb.py prep_mvad.py prep_hifi.py prep_fakeavceleb_b6.py"
  if [ "${PREP_PARALLEL:-0}" = "1" ]; then
    # 并发数不硬编码：优先取 PREP_WORKERS(环境/启动参数)；否则按机器自动推导
    #   CPU 侧: 每并发进程约吃 1-2 核(ffmpeg 解码 + MediaPipe) → cores/2
    #   显存侧: 每并发进程独立加载 CLIP+wav2vec2+MediaPipe ≈4GB,
    #           只取 85% 显存可用(留头寸给 CUDA context/激活峰值) → vram*0.85/4000
    #   取两者较小者，硬上限 8(防自爆)
    if [ -n "${PREP_WORKERS:-}" ]; then
      W="$PREP_WORKERS"
      log "  并行模式: 用户指定 PREP_WORKERS=${W}"
    else
      _CORES=$(nproc 2>/dev/null || echo 4)
      _VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 || true)
      _VRAM=$(printf '%s' "${_VRAM:-0}" | tr -cd '0-9')
      _BY_VRAM=999
      [ "${_VRAM:-0}" -gt 0 ] && _BY_VRAM=$(( _VRAM * 85 / 100 / 4000 ))
      _BY_CPU=$(( _CORES / 2 )); [ "$_BY_CPU" -lt 1 ] && _BY_CPU=1
      [ "$_BY_VRAM" -lt 1 ] && _BY_VRAM=1
      W=$(( _BY_VRAM < _BY_CPU ? _BY_VRAM : _BY_CPU ))
      [ "$W" -gt 8 ] && W=8
      log "  并行模式: 自动推导 cores=${_CORES} vram=${_VRAM}MB → PREP_WORKERS=${W}（可用 env PREP_WORKERS=N 覆盖）"
    fi
    i=0
    for prep in $PREP_LIST; do
      ( python "$CODE_DIR/$prep" 2>&1 | tee -a "$PERSIST/logs/${prep%.py}.log" ) &
      i=$((i+1))
      if [ $((i % W)) -eq 0 ]; then wait; fi
    done
    wait || { log "FAIL: 并行 prep 存在失败，重跑同一命令会从失败处继续"; exit 1; }
  else
    for prep in $PREP_LIST; do
      log "  > $prep"
      python "$CODE_DIR/$prep" || { log "FAIL: $prep"; exit 1; }
    done
  fi
  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
# STEP 2 — Stage-1 SSL 自监督预训练（40 epoch）
# ============================================================================
STEP="2_stage1"
if is_done "$STEP"; then
  log "SKIP: $STEP (marker 已存在)"
else
  log "=== STEP 2: Stage-1 SSL（40 epoch）==="
  # --resume: 中断后相同命令重跑即自动续跑
  python "$CODE_DIR/train_stage1.py" \
      --real_cache "$SYNCX_DATA_ROOT/VoxCeleb2_subset/prep" \
      --out_ckpt   "$STAGE1_CKPT" \
      --log_dir    "$PERSIST/logs" \
      --epochs 40 --batch 1024 --lr 1e-4 --seed "$SEED" --device "$DEVICE" \
      --num_workers 4 --empty_cache_every 1 || { log "FAIL: Stage-1"; exit 1; }
  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
# STEP 3 — 超参搜索（可选：SKIP_SEARCH=1 跳过）
# 论文留白项 λ_align×λ_uniform，corr/shared/lr/epochs 论文锁定不搜
# ============================================================================
STEP="3_search"
if [ "$SKIP_SEARCH" = "1" ]; then
  log "SKIP: $STEP (SKIP_SEARCH=1，直接用论文默认 λ_align=0.1 λ_uniform=0.5)"
elif is_done "$STEP"; then
  log "SKIP: $STEP (marker 已存在)"
else
  log "=== STEP 3: 超参搜索 λ_align×λ_uniform（9 组合，≈6.3h）==="
  SEARCH_LOG="$PERSIST/logs/hyperparameter_search"
  mkdir -p "$SEARCH_LOG"

  # Step A: 筛选（9 组合 × 20ep ≈ 2.3h）
  log "  Step A: 筛选 λ_align×λ_uniform"
  python "$CODE_DIR/hyperparameter_search.py" --stage loss_weights --epochs 20 \
      --stage1_ckpt "$STAGE1_CKPT" --seeds "$SEED" || { log "FAIL: search Step A"; exit 1; }

  # 自动选优：summary.csv 里 val_auc 最高的组合
  BEST_LINE=$(grep "True" "$SEARCH_LOG/loss_weights/summary.csv" 2>/dev/null \
    | tail -n +1 | sort -t',' -k6 -nr 2>/dev/null | head -1)

  if [ -z "$BEST_LINE" ]; then
    log "  WARN: 搜索全失败，退回论文默认 λ_align=0.1 λ_uniform=0.5"
    BEST_ALIGN="0.1"
    BEST_UNIFORM="0.5"
  else
    BEST_ALIGN=$(echo "$BEST_LINE" | grep -oP "'lambda_align': \K[\d.]+" || echo "0.1")
    BEST_UNIFORM=$(echo "$BEST_LINE" | grep -oP "'lambda_uniform': \K[\d.]+" || echo "0.5")
    log "  Best: λ_align=$BEST_ALIGN  λ_uniform=$BEST_UNIFORM"

    # Step A': 最优组合放大到 80ep（≈1h）
    log "  Step A': 最优放大"
    python "$CODE_DIR/hyperparameter_search.py" --stage loss_weights --epochs 80 \
        --best_lambda_align "$BEST_ALIGN" --best_lambda_uniform "$BEST_UNIFORM" \
        --stage1_ckpt "$STAGE1_CKPT" --seeds "$SEED" || { log "FAIL: search Step A'"; exit 1; }

    # Step B: 最优 × 3 seeds 聚合（论文 ≥3 seeds mean±std，≈3h）
    log "  Step B: 3 seeds 聚合"
    python "$CODE_DIR/hyperparameter_search.py" --stage loss_weights --seeds "0 1 2" \
        --best_lambda_align "$BEST_ALIGN" --best_lambda_uniform "$BEST_UNIFORM" \
        --stage1_ckpt "$STAGE1_CKPT" || { log "FAIL: search Step B"; exit 1; }
  fi

  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
# λ 衔接 — 把 STEP 3 搜索出的最优 λ 灌入最终训练（未搜/失败则用论文默认 0.1/0.5）
# ============================================================================
if [ -n "${BEST_ALIGN:-}" ] && [ -n "${BEST_UNIFORM:-}" ]; then
  export SYNCX_LAMBDA_ALIGN="$BEST_ALIGN" SYNCX_LAMBDA_UNIFORM="$BEST_UNIFORM"
  log "[λ] 最终训练使用搜索最优 λ_align=$BEST_ALIGN λ_uniform=$BEST_UNIFORM"
else
  export SYNCX_LAMBDA_ALIGN="0.1" SYNCX_LAMBDA_UNIFORM="0.5"
  log "[λ] 最终训练使用论文默认 λ_align=0.1 λ_uniform=0.5"
fi

# ============================================================================
# STEP 4 — Stage-2 最终训练（论文契约：80 epoch，5 项损失，3 seeds mean±std）
# ============================================================================
STEP="4_stage2"
if is_done "$STEP"; then
  log "SKIP: $STEP (marker 已存在)"
else
  log "=== STEP 4: Stage-2 最终训练 × 3 seeds（80 epoch/seed，论文 ≥3 seeds mean±std）==="
  for s in 0 1 2; do
    log "  Stage-2 seed=$s（λ_align=$SYNCX_LAMBDA_ALIGN λ_uniform=$SYNCX_LAMBDA_UNIFORM）"
    python "$CODE_DIR/train_stage2.py" \
        --train_cache "$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/prep/train" \
        --val_cache   "$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/prep/test" \
        --stage1_ckpt "$STAGE1_CKPT" \
        --out_ckpt    "$PERSIST/ckpt/syncx_best_seed${s}.pt" \
        --log_dir     "$PERSIST/logs" \
        --epochs 80 --batch 512 --lr 1e-4 --seed "$s" --device "$DEVICE" \
        --num_workers 4 --empty_cache_every 1 || { log "FAIL: Stage-2 seed=$s"; exit 1; }
  done
  # 兼容下游(plot/单测)对 syncx_best.pt 的引用: seed0 副本
  cp "$PERSIST/ckpt/syncx_best_seed0.pt" "$STAGE2_CKPT"
  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
# STEP 5 — 评测 5 个跨域集
# ============================================================================
STEP="5_eval"
if [ "$SKIP_EVAL" = "1" ]; then
  log "SKIP: $STEP (SKIP_EVAL=1)"
elif is_done "$STEP"; then
  log "SKIP: $STEP (marker 已存在)"
else
  log "=== STEP 5: 评测 6 数据集 × 3 seeds（主结果表 mean±std）==="
  for s in 0 1 2; do
    CKPT="$PERSIST/ckpt/syncx_best_seed${s}.pt"
    for pair in "Celeb-DF-v2:celebdf" "Deepfake-Eval-2024:DFE2024" \
                "TalkingHeadBench:thb" "MVAD:mvad" "HiFi-AVDF:hifi" \
                "FakeAVCeleb_v1.2_split:fakeavceleb"; do
      ds_dir="${pair%%:*}"
      ds_key="${pair##*:}"
      cache="${SYNCX_DATA_ROOT}/${ds_dir}/prep"
      log "  Evaluating $ds_key seed=$s ($cache)..."
      python "$CODE_DIR/evaluate.py" \
          --ckpt "$CKPT" \
          --eval_cache "$cache" \
          --log_dir "$PERSIST/logs" \
          --dataset "$ds_key" --seed "$s" --device "$DEVICE" --tag "seed${s}" \
          || log "WARN: eval $ds_key seed=$s failed (continuing)"
    done
  done
  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
# STEP 5b — B6 捷径探针（Smeu trim, §5.6）
# ============================================================================
STEP="5b_b6"
if [ "$SKIP_EVAL" = "1" ]; then
  log "SKIP: $STEP (SKIP_EVAL=1)"
elif is_done "$STEP"; then
  log "SKIP: $STEP (marker 已存在)"
else
  log "=== STEP 5b: B6 捷径探针（Smeu 动态 trim 协议）==="
  B6_PREP="$SYNCX_DATA_ROOT/FakeAVCeleb_v1.2_split/prep"
  # 若 STEP 1 未产出 b6 特征(如 SKIP_PREP=1), 补生成
  if [ ! -d "$B6_PREP/b6_trimmed" ] || [ ! -d "$B6_PREP/b6_raw" ]; then
    log "  > 补跑 b6_std_trim + prep_fakeavceleb_b6"
    python "$CODE_DIR/b6_std_trim.py" || { log "FAIL: b6_std_trim"; exit 1; }
    python "$CODE_DIR/prep_fakeavceleb_b6.py" || { log "FAIL: prep_fakeavceleb_b6"; exit 1; }
  fi
  log "  B6 trimmed 条件评测（prep/b6_trimmed）"
  python "$CODE_DIR/evaluate.py" --ckpt "$STAGE2_CKPT" --eval_cache "$B6_PREP" \
      --b6 --dataset fakeavceleb --log_dir "$PERSIST/logs" --device "$DEVICE" \
      --tag b6_trimmed || log "WARN: B6 trimmed failed"
  log "  B6 raw 未裁条件评测（prep/b6_raw, 对照）"
  python "$CODE_DIR/evaluate.py" --ckpt "$STAGE2_CKPT" --eval_cache "$B6_PREP" \
      --b6_untrimmed --dataset fakeavceleb --log_dir "$PERSIST/logs" --device "$DEVICE" \
      --tag b6_raw || log "WARN: B6 raw failed"
  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
# STEP 5c — 论文消融评测变体（复用 ckpt, 零重训; FakeAVCeleb 上跑）
#   §5.3 per-category / §5.4 模态 masking / §5.5 组件消融(eval-time 部分)
# ============================================================================
STEP="5c_ablations"
if [ "$SKIP_EVAL" = "1" ]; then
  log "SKIP: $STEP (SKIP_EVAL=1)"
elif is_done "$STEP"; then
  log "SKIP: $STEP (marker 已存在)"
else
  log "=== STEP 5c: 消融评测变体（FakeAVCeleb, 复用 $STAGE2_CKPT）==="
  ABL_CACHE="$SYNCX_DATA_ROOT/FakeAVCeleb_v1.2_split/prep"
  eval_var() {  # $1=额外参数 $2=tag
    python "$CODE_DIR/evaluate.py" --ckpt "$STAGE2_CKPT" --eval_cache "$ABL_CACHE" \
        --dataset fakeavceleb --log_dir "$PERSIST/logs" --device "$DEVICE" \
        --tag "$2" $1 || log "WARN: ablation $2 failed"
  }
  log "  §5.3 per-category（4 模态组合）"
  eval_var "--per_category" per_category
  log "  §5.4 Visual-Only（Audio masked）"
  eval_var "--mask audio" mask_audio_masked
  log "  §5.4 Audio-Only（Visual masked）"
  eval_var "--mask visual" mask_visual_masked
  log "  §5.5 w/o LCMA（concatenation only）"
  eval_var "--disable_lcma" wo_lcma
  log "  §5.5 w/o DWF"
  eval_var "--disable_dwf" wo_dwf
  log "  §5.5 w/o BioLip x_s（时序 std 分支）"
  eval_var "--disable_xs" wo_xs
  log "  §5.5 LACA 仅 corr 子分支"
  eval_var "--laca_mode corr" laca_corr
  log "  §5.5 LACA 仅 biomech 子分支"
  eval_var "--laca_mode biomech" laca_biomech
  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
# STEP 5d — 训练期消融（可选, RUN_TRAIN_ABLATIONS=1 开启; 每次 ≈4-6h）
#   论文 §5.5 中需要重训的变体里, 能用现有机制实现的:
#   (a) w/o hyperspherical: λ_align=λ_uniform=0
#   (b) w/o Stage-1 pre-training: 不传 stage1_ckpt
#   其余行(无AAM/无corr损失/无同源采样)需代码改动, 不在编排器覆盖范围
# ============================================================================
STEP="5d_train_ablations"
if [ "${RUN_TRAIN_ABLATIONS:-0}" != "1" ]; then
  log "SKIP: $STEP (RUN_TRAIN_ABLATIONS != 1，如需论文全部训练期消融请置 1)"
elif is_done "$STEP"; then
  log "SKIP: $STEP (marker 已存在)"
else
  log "=== STEP 5d: 训练期消融（RUN_TRAIN_ABLATIONS=1）==="
  # (a) w/o hyperspherical regularisation
  log "  > train: w/o hyperspherical (λ_align=λ_uniform=0)"
  SYNCX_LAMBDA_ALIGN=0 SYNCX_LAMBDA_UNIFORM=0 \
  python "$CODE_DIR/train_stage2.py" \
      --train_cache "$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/prep/train" \
      --val_cache   "$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/prep/test" \
      --stage1_ckpt "$STAGE1_CKPT" \
      --out_ckpt    "$PERSIST/ckpt/abl_wo_hypersph.pt" \
      --log_dir     "$PERSIST/logs" \
      --epochs 80 --batch 512 --lr 1e-4 --seed "$SEED" --device "$DEVICE" \
      --num_workers 4 --empty_cache_every 1 || { log "FAIL: abl wo_hypersph"; exit 1; }
  # (b) w/o Stage-1 pre-training
  log "  > train: w/o Stage-1 pre-training（from scratch）"
  python "$CODE_DIR/train_stage2.py" \
      --train_cache "$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/prep/train" \
      --val_cache   "$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/prep/test" \
      --stage1_ckpt "" \
      --out_ckpt    "$PERSIST/ckpt/abl_wo_stage1.pt" \
      --log_dir     "$PERSIST/logs" \
      --epochs 80 --batch 512 --lr 1e-4 --seed "$SEED" --device "$DEVICE" \
      --num_workers 4 --empty_cache_every 1 || { log "FAIL: abl wo_stage1"; exit 1; }
  # 评测两个消融 ckpt（FakeAVCeleb 为论文口径）
  for abl in wo_hypersph wo_stage1; do
    log "  > eval $abl"
    python "$CODE_DIR/evaluate.py" --ckpt "$PERSIST/ckpt/abl_${abl}.pt" \
        --eval_cache "$SYNCX_DATA_ROOT/FakeAVCeleb_v1.2_split/prep" \
        --dataset fakeavceleb --log_dir "$PERSIST/logs" --device "$DEVICE" \
        --tag "abl_${abl}" || log "WARN: abl $abl eval failed"
  done
  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
# STEP 5e — B7 身份泄漏控制探针（§5.7）
#   前置: configs/exclude_ids.json（37 个 FakeAVCeleb∩VoxCeleb2 身份）必须存在
# ============================================================================
STEP="5e_b7"
if [ "$SKIP_EVAL" = "1" ]; then
  log "SKIP: $STEP (SKIP_EVAL=1)"
elif is_done "$STEP"; then
  log "SKIP: $STEP (marker 已存在)"
else
  EXCLUDE_IDS="$CODE_DIR/configs/exclude_ids.json"
  if [ ! -f "$EXCLUDE_IDS" ]; then
    log "SKIP: B7 缺前置文件 $EXCLUDE_IDS（37 个 FakeAVCeleb 身份列表, 仓库中不存在）"
    log "  生成: 取 FakeAVCeleb 真实片段身份 ∩ VoxCeleb2_subset 说话人 的交集写入 json 后重跑"
  else
    log "=== STEP 5e: B7 身份泄漏控制（exclude_ids=$EXCLUDE_IDS）==="
    log "  > B7 Stage-1（剔除 37 身份）"
    python "$CODE_DIR/train_stage1.py" \
        --real_cache "$SYNCX_DATA_ROOT/VoxCeleb2_subset/prep" \
        --exclude_ids "$EXCLUDE_IDS" \
        --out_ckpt "$PERSIST/ckpt/stage1_b7.pt" \
        --log_dir  "$PERSIST/logs" \
        --epochs 40 --batch 1024 --lr 1e-4 --seed "$SEED" --device "$DEVICE" \
        --num_workers 4 --empty_cache_every 1 || { log "FAIL: B7 Stage-1"; exit 1; }
    log "  > B7 Stage-2"
    python "$CODE_DIR/train_stage2.py" \
        --train_cache "$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/prep/train" \
        --val_cache   "$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/prep/test" \
        --stage1_ckpt "$PERSIST/ckpt/stage1_b7.pt" \
        --exclude_ids "$EXCLUDE_IDS" \
        --out_ckpt    "$PERSIST/ckpt/syncx_b7.pt" \
        --log_dir     "$PERSIST/logs" \
        --epochs 80 --batch 512 --lr 1e-4 --seed "$SEED" --device "$DEVICE" \
        --num_workers 4 --empty_cache_every 1 || { log "FAIL: B7 Stage-2"; exit 1; }
    log "  > B7 评测（leaky = 主模型 syncx_best.pt；disjoint = syncx_b7.pt）"
    python "$CODE_DIR/evaluate.py" --ckpt "$STAGE2_CKPT" \
        --eval_cache "$SYNCX_DATA_ROOT/FakeAVCeleb_v1.2_split/prep" \
        --dataset fakeavceleb --log_dir "$PERSIST/logs" --device "$DEVICE" \
        --tag b7_leaky || log "WARN: B7 leaky eval failed"
    python "$CODE_DIR/evaluate.py" --ckpt "$PERSIST/ckpt/syncx_b7.pt" \
        --eval_cache "$SYNCX_DATA_ROOT/FakeAVCeleb_v1.2_split/prep" \
        --dataset fakeavceleb --log_dir "$PERSIST/logs" --device "$DEVICE" \
        --tag b7_disjoint || log "WARN: B7 disjoint eval failed"
    done_marker "$STEP"; log "$STEP DONE"
  fi
fi

# ============================================================================
# STEP 6 — 出图（CPU，分钟级）
# ============================================================================
STEP="6_plot"
if is_done "$STEP"; then
  log "SKIP: $STEP (marker 已存在)"
else
  log "=== STEP 6: 出图 ==="
  python "$CODE_DIR/plot_figures.py" --log_dir "$PERSIST/logs"
  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
log "=========================================================="
log "🎉 Sync-X 全流程完成。"
log "   模型: $STAGE2_CKPT"
log "   日志: $PERSIST/logs"
log "   图件: plot_figures 已生成"
log "=========================================================="
