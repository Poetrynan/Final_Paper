#!/bin/bash
# ============================================================================
# cloud_settle_all_mvp.sh — Sync-X 最小可行性版本（MVP）一键挂机
# ============================================================================
# 设计目标：以最小 GPU 时间代价，产出"既能投会议、又能满足毕业"的核心证据链。
# 相对 cloud_settle_all.sh 的裁剪（详见下方各 STEP 注释）：
#   - 复活 B6 静音捷径探针（STEP 5b）      → FakeAVCeleb 有音轨，Smeu 动态裁剪正面回应"虚假捷径"关切
#   - 砍掉 B7 身份泄漏探针（STEP 5e）      → 省 1 次 Stage-1/2 重训
#   - 砍掉 λ_corr 敏感性扫描（F7）          → 论文 λ_corr=0.3 已锁定，不搜
#   - 超参搜索 SKIP（STEP 3）              → 直接用论文默认 λ_align=0.1/λ_uniform=0.5
#   - 跨域评测 5→3 集（STEP 5）            → 留 FakeAVCeleb / DFE2024 / HiFi-AVDF（Celeb-DF-v2 无音轨已砍）
#   - prep 8→5 集（STEP 1）                → 砍 THB / MVAD / Celeb-DF-v2(无音轨)；FakeAVCeleb 作第 3 评测集+B6 宿主
#   - 评测期消融宿主改 HiFi（STEP 5c）      → FakeAVCeleb 改作正常跨域评测集
#   - 训练期消融仅 from-scratch（STEP 5d）  → 证明 SSL 预训练增益，最具性价比
#
# 保留的核心证据链（MVP 必备）：
#   F1 训练动力学 | F2/F3 跨域 ROC+AUC(3集) | F5 多模态融合 | F6 from-scratch
#   F8 LACA 三模式 | F11/F12/F13/F15 可解释性/效率(零/低重训) | F17 组件消融
#
# 用法：
#   export SYNCX_DATA_ROOT=/root/autodl-tmp/SyncX/Datasets
#   export HF_TOKEN=hf_xxx
#   export MVP_SEEDS="0"            # 先 1 seed 快速出结果；确认 pipeline 跑通后改 "0 1 2" 补 3 seed
#   export SHARDS=8                 # 抽取并行分片数（默认 8；16 vCPU 吃满。GPU 显存紧张时降到 4 或设 SYNCX_MEDIAPIPE_GPU=0）
#
# ── 云端"关页面 / 断网 / 关本地电脑也在跑"的启动方式（务必套一层！否则 SSH 一断进程连带被杀）──
#   cd code_cloud
#   # 方式 A（推荐，可随时重连查看进度）：
#   tmux new -d -s syncx "bash cloud_settle_all_mvp.sh"
#   tmux a -t syncx            # 回来看；断网 / 关页面 / 关本地电脑都不杀进程
#   # 方式 B（无 tmux 时）：
#   setsid bash cloud_settle_all_mvp.sh > /root/autodl-tmp/SyncX/persist/run.out 2>&1 < /dev/null &
#   disown
#   tail -f /root/autodl-tmp/SyncX/persist/run.out
#   ⚠️ 严禁裸跑 `bash cloud_settle_all_mvp.sh`：SSH 断开时内核向"整个前台进程组"发 SIGHUP，
#      脚本内用 `&` 后台的分片同属该进程组，会一并被杀（已抽的 .pt 不丢，重跑续传即可）。
#
# 幂等：中断后重跑同一命令 → 从断点继续；每阶段 marker 文件防重跑。
# 架构：Epoch-based resume（粒度=1 epoch，最坏丢 1 epoch）。
# ============================================================================
set -euo pipefail

SYNCX_DATA_ROOT="${SYNCX_DATA_ROOT:-/root/autodl-tmp/SyncX/Datasets}"
HF_HOME="${HF_HOME:-/root/autodl-tmp/SyncX/hfcache}"
HF_TOKEN="${HF_TOKEN:-}"
PERSIST="${PERSIST:-/root/autodl-tmp/SyncX/persist}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
SKIP_PREP="${SKIP_PREP:-0}"
SKIP_SEARCH="${SKIP_SEARCH:-1}"   # MVP：默认跳过超参搜索，用论文默认 λ
SKIP_EVAL="${SKIP_EVAL:-0}"
MVP_SEEDS="${MVP_SEEDS:-0}"        # MVP：默认 1 seed；补 3 seed 时置 "0 1 2"
SHARDS="${SHARDS:-8}"               # MVP：抽取并行分片数（8 进程并发，吃满 16 vCPU；显存约 5GB，远低于 24GB）
# ⚠️ 硬性开关（2026-08-20 教训）：Stage-2 在 RTX 4090D 上用 torch.compile 会启动即崩
#    （max-aututune 编译 AAM/DWF 组合 crash，导致 Stage-2 FAIL 多次）。已实测 SYNCX_COMPILE=0
#    完全正常（仅略慢，精度不变）。默认关闭编译；如需加速可在 tmux 命令里覆写 SYNCX_COMPILE=1。
export SYNCX_COMPILE="${SYNCX_COMPILE:-0}"

# [v2-1a/1d] 模块1（强化音频监督）默认开启：
#   - SYNCX_LAMBDA_AUDIO：音频独立分类头(loss_cls_audio)权重，给 audio_emb 显式 real/fake 监督（治"听觉监督断裂"）。
#   - SYNCX_CORR_REAL_ONLY：L_corr 只约束 real 段(→1)、fake 段不再强制去相关（解除与 AAM 分类头在 audio_emb 上的反向拉扯）。
#   均可用 tmux 启动命令覆写（如 SYNCX_CORR_REAL_ONLY=0 回退原论文行为作对照实验）。
export SYNCX_LAMBDA_AUDIO="${SYNCX_LAMBDA_AUDIO:-1.0}"
export SYNCX_CORR_REAL_ONLY="${SYNCX_CORR_REAL_ONLY:-1}"

# [方向B重定义] LTCA（内容特征长时一致性，第三分支）主开关 + 一键切换"方向B重训模式"：
#   - SYNCX_FTA=1：接入 LTCA 分支（DWF 扩三路）。默认 0=原行为，防污染论文基线。
#   - DIRECT_B=1：进入方向B重训模式 = 在 STEP 4 前自动清掉 Stage-2 旧权重/续跑点 + 后续 marker
#     （换架构必清，否则旧权重/续跑点 stride 失配或 marker 误 SKIP）。Stage-1(LCMACA 未改)权重保留复用。
#   仅当两者都 =1 才触发清理；DIRECT_B=1 但 SYNCX_FTA=0 也只在 STEP4 前清（用于对照重跑原架构）。
export SYNCX_FTA="${SYNCX_FTA:-0}"
export DIRECT_B="${DIRECT_B:-0}"

# Pipeline 宏观进度分母（按当前 env 预估会实际执行的阶段；断点续跑时个别阶段已 marker 跳过则不计入 n，N 仍含其占位）
PIPELINE_TOTAL=1                                                  # STEP 0
[ "$SKIP_PREP"   != "1" ] && PIPELINE_TOTAL=$((PIPELINE_TOTAL+1)) # STEP 1
PIPELINE_TOTAL=$((PIPELINE_TOTAL+1))                              # STEP 2 (stage1)
[ "$SKIP_SEARCH" != "1" ] && PIPELINE_TOTAL=$((PIPELINE_TOTAL+1)) # STEP 3
PIPELINE_TOTAL=$((PIPELINE_TOTAL+1))                              # STEP 4 (stage2)
if [ "$SKIP_EVAL" != "1" ]; then PIPELINE_TOTAL=$((PIPELINE_TOTAL+3)); fi  # STEP 5/5b/5c
[ "${RUN_TRAIN_ABLATIONS:-0}" = "1" ] && PIPELINE_TOTAL=$((PIPELINE_TOTAL+1)) # STEP 5d
PIPELINE_TOTAL=$((PIPELINE_TOTAL+1))                              # STEP 6 (plot)
STEP_NUM=0

CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
STATUS_FILE="$PERSIST/status.log"
MARKER_DIR="$PERSIST/.markers"
mkdir -p "$PERSIST/ckpt" "$PERSIST/logs" "$MARKER_DIR"

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')]  $*" | tee -a "$STATUS_FILE"; }
done_marker() { touch "$MARKER_DIR/$1.done"; }
is_done()    { [ -f "$MARKER_DIR/$1.done" ]; }

# 宏观进度头：每个实际执行的 STEP 进入时打印 [n/N]；跳过(SKIP/marker)不计入 n
step_start() {
  STEP_NUM=$((STEP_NUM+1))
  log "=== [${STEP_NUM}/${PIPELINE_TOTAL}] $1 ==="
}

# 并行分片抽取：启动 SHARDS 个独立进程（每片一个），各自按 --shard-id 处理分配到的片段，
# 互不写同一文件、各自断点续跑。全部完成后返回；任一分片非零退出则 return 1（调用方决定 fatal/warn）。
# 显存：每进程约 0.5-1GB（CLIP+wav2vec2+MediaPipe GPU），8 进程并发 ≈5GB，远低于 24GB 显存。
run_sharded() {
  local prep="$1"; shift
  log "  > $prep (并行分片 x$SHARDS)"
  log "    各分片进度/日志: $PERSIST/logs/prep_${prep%.py}_shard*.log"
  # 线程纪律（防超发）：每 shard 限 16/SHARDS 线程，覆盖 torch/ffmpeg/MediaPipe CPU 部分
  local nt=$((16 / SHARDS)); [ "$nt" -lt 1 ] && nt=1
  export OMP_NUM_THREADS=$nt MKL_NUM_THREADS=$nt OPENBLAS_NUM_THREADS=$nt
  local k pid rc=0 pids=()
  for k in $(seq 0 $((SHARDS-1))); do
    python "$CODE_DIR/$prep" --num-shards "$SHARDS" --shard-id "$k" "$@" \
      > "$PERSIST/logs/prep_${prep%.py}_shard${k}.log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
  done
  if [ "$rc" -ne 0 ]; then log "FAIL: $prep (存在非零退出分片)"; return 1; fi
  return 0
}

STAGE1_CKPT="$PERSIST/ckpt/stage1.pt"
STAGE2_CKPT="$PERSIST/ckpt/syncx_best.pt"

log "=========================================================="
log "Sync-X MVP 全流程启动（最小可行性版本）"
log "  数据根: $SYNCX_DATA_ROOT   持久卷: $PERSIST"
log "  种子: $MVP_SEEDS   SKIP_SEARCH=$SKIP_SEARCH"
log "=========================================================="

# [方向B重训] 架构切换硬清理（仅 DIRECT_B=1 触发，看 SYNCX_FTA 决定删法），放在 STEP 0 前：
#   换了 Stage-2 架构（LTCA/DWF 三路）→ 必须删旧权重 + 续跑点 + 后续 marker，
#   否则旧权重 stride 失配或 marker 误 SKIP 会用被删/旧权重续跑（见 evidence_chain 教训）。
#   Stage-1(LCMACA 未改) 权重保留复用，故不删 2_stage1.done。
if [ "${DIRECT_B:-0}" = "1" ]; then
  log "[方向B重训] 架构切换：清理 Stage-2 旧权重/续跑点 + 后置 marker"
  log "            SYNCX_FTA=${SYNCX_FTA}（1=接入 LTCA 第三分支；0=对照重跑原架构）"
  rm -f "$PERSIST/ckpt"/syncx_best*.pt "$PERSIST/ckpt"/*.resume
  for m in 4_stage2 5_eval 5b_b6 5c_ablations 5d_train_ablations 6_plot; do
    rm -f "$MARKER_DIR/$m.done"
    log "            rm .markers/$m.done"
  done
fi

# ============================================================================
# STEP 0 — 云端环境自检
# ============================================================================
STEP="0_cloud_check"
if is_done "$STEP"; then
  log "SKIP: $STEP (marker 已存在)"
else
  step_start "STEP 0: 云端环境自检"
  python "$CODE_DIR/_verify_cloud.py" "$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/real/14-5467.mp4"
  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
# STEP 1 — 特征抽取（MVP：5 集，砍 THB/MVAD/Celeb-DF-v2；FakeAVCeleb 作第 3 评测集+B6 宿主）
# ============================================================================
STEP="1_extraction"
if [ "$SKIP_PREP" = "1" ]; then
  log "SKIP: $STEP (SKIP_PREP=1)"
elif is_done "$STEP"; then
  log "SKIP: $STEP (marker 已存在)"
else
  step_start "STEP 1: 特征抽取（MVP 5 集；deepspeak/vox/fakeavceleb 走 $SHARDS 路并行分片，eval2024/hiifi 顺序）"
  # 三大体量集并行分片（每集 $SHARDS 进程并发，GPU 始终饱和）：墙钟 ≈ 单进程/$SHARDS
  run_sharded prep_deepspeak.py                 || { log "FAIL: prep_deepspeak";          exit 1; }
  run_sharded prep_voxceleb2.py                 || { log "FAIL: prep_voxceleb2";          exit 1; }
  run_sharded prep_fakeavceleb.py --mode normal || { log "FAIL: prep_fakeavceleb(normal)"; exit 1; }
  # 小体量集并行分片（已支持分片参数）
run_sharded prep_eval2024.py || { log "FAIL: prep_eval2024"; exit 1; }
run_sharded prep_hifi.py      || { log "FAIL: prep_hifi";      exit 1; }
  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
# STEP 2 — Stage-1 SSL 预训练（40 epoch）
# ============================================================================
STEP="2_stage1"
if is_done "$STEP"; then
  log "SKIP: $STEP (marker 已存在)"
else
  step_start "STEP 2: Stage-1 SSL（60 epoch）"
  python "$CODE_DIR/train_stage1.py" \
      --real_cache "$SYNCX_DATA_ROOT/VoxCeleb2_subset/prep" \
      --out_ckpt   "$STAGE1_CKPT" \
      --log_dir    "$PERSIST/logs" \
      --epochs 60 --batch 1024 --lr 1e-4 --seed "$SEED" --device "$DEVICE" \
      --num_workers 4 --empty_cache_every 1 || { log "FAIL: Stage-1"; exit 1; }
  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
# STEP 3 — 超参搜索（MVP：SKIP，用论文默认 λ_align=0.1 λ_uniform=0.5；未 SKIP 则跑第一轮损失权重搜索）
# ============================================================================
STEP="3_search"
if [ "$SKIP_SEARCH" = "1" ]; then
  log "SKIP: $STEP (MVP：直接用论文默认 λ_align=0.1 λ_uniform=0.5)"
  # λ 衔接：MVP 固定论文默认
  export SYNCX_LAMBDA_ALIGN="0.1" SYNCX_LAMBDA_UNIFORM="0.5"
  export SYNCX_COMPILE="${SYNCX_COMPILE:-0}"   # 硬性：默认关 torch.compile（Stage-2 在其下启动即崩）
  log "[λ] MVP 使用论文默认 λ_align=0.1 λ_uniform=0.5"
else
  step_start "STEP 3: 超参搜索 → 第一轮 loss_weights（λ_align × λ_uniform，4 组合，≈2.3h）"
  # 超参搜索入口配置：第一轮 loss_weights，20 epoch/组合加速筛选
  # 输出落盘到 $PERSIST/logs/hyperparameter_search/ 持久卷，避免实例重启丢失
  python "$CODE_DIR/hyperparameter_search.py" --stage loss_weights --epochs 20 \
    --seeds "0" \
    --stage1_ckpt "$STAGE1_CKPT" \
    --log_root "$PERSIST/logs" \
    || { log "FAIL: hyperparameter_search (loss_weights)"; exit 1; }
  # 自动从 summary.csv 读最佳组合，写入环境变量供 Stage-2 最终训练用
  # （若读失败，仍用论文默认，不硬错）
  SUMMARY_CSV="$PERSIST/logs/hyperparameter_search/loss_weights/summary.csv"
  if [ -f "$SUMMARY_CSV" ]; then
    # 取 val_auc 最大的成功组合，输出 params 列（如 "{'lambda_align': 0.1, 'lambda_uniform': 0.5}"）
    BEST=$(python - "$SUMMARY_CSV" <<'PY'
import sys, csv
best, best_val = None, -1.0
with open(sys.argv[1], newline='') as f:
    for row in csv.DictReader(f):
        if row['success'] != 'True' or row['val_auc'] == 'None':
            continue
        v = float(row['val_auc'])
        if v > best_val:
            best_val, best = v, row
if best:
    print(best['params'])
PY
)
    if [ -n "$BEST" ]; then
      # 格式: "{'lambda_align': 0.1, 'lambda_uniform': 0.5}" → 提取值
      LAMBDA_ALIGN=$(echo "$BEST" | grep -oE "lambda_align': [0-9.]+([eE][+-]?[0-9]+)?" | cut -d' ' -f2)
      LAMBDA_UNIFORM=$(echo "$BEST" | grep -oE "lambda_uniform': [0-9.]+([eE][+-]?[0-9]+)?" | cut -d' ' -f2)
      [ -n "$LAMBDA_ALIGN" ] && export SYNCX_LAMBDA_ALIGN="$LAMBDA_ALIGN"
      [ -n "$LAMBDA_UNIFORM" ] && export SYNCX_LAMBDA_UNIFORM="$LAMBDA_UNIFORM"
      log "[SEARCH] 最佳λ: λ_align=$SYNCX_LAMBDA_ALIGN λ_uniform=$SYNCX_LAMBDA_UNIFORM (来自 $SUMMARY_CSV)"
    else
      log "[WARN] 找不到有效最佳组合，回退论文默认 λ_align=0.1 λ_uniform=0.5"
    fi
  else
    log "[WARN] $SUMMARY_CSV 不存在，回退论文默认 λ_align=0.1 λ_uniform=0.5"
  fi
  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
# STEP 4 — Stage-2 最终训练（MVP：MVP_SEEDS 控制，默认 1 seed 先出结果）
# ============================================================================
STEP="4_stage2"
if is_done "$STEP"; then
  log "SKIP: $STEP (marker 已存在)"
else
  step_start "STEP 4: Stage-2 最终训练 × seeds($MVP_SEEDS)（80 epoch/seed）"
  for s in $MVP_SEEDS; do
    log "  Stage-2 seed=$s（λ_align=0.1 λ_uniform=0.5）"
    python "$CODE_DIR/train_stage2.py" \
        --train_cache "$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/prep/train" \
        --val_cache   "$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/prep/test" \
        --stage1_ckpt "$STAGE1_CKPT" \
        --out_ckpt    "$PERSIST/ckpt/syncx_best_seed${s}.pt" \
        --log_dir     "$PERSIST/logs" \
        --epochs 80 --batch 512 --lr 1e-4 --seed "$s" --device "$DEVICE" \
        --num_workers 4 --empty_cache_every 1 || { log "FAIL: Stage-2 seed=$s"; exit 1; }
  done
  cp "$PERSIST/ckpt/syncx_best_seed${MVP_SEEDS%% *}.pt" "$STAGE2_CKPT"
  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
# STEP 5 — 评测 3 个跨域集（MVP：FakeAVCeleb / DFE2024 / HiFi-AVDF；Celeb-DF-v2 无音轨已砍）
# ============================================================================
STEP="5_eval"
if [ "$SKIP_EVAL" = "1" ]; then
  log "SKIP: $STEP (SKIP_EVAL=1)"
elif is_done "$STEP"; then
  log "SKIP: $STEP (marker 已存在)"
else
  step_start "STEP 5: 评测 3 跨域集 × seeds($MVP_SEEDS)"
  for s in $MVP_SEEDS; do
    CKPT="$PERSIST/ckpt/syncx_best_seed${s}.pt"
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
  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
# STEP 5b — B6 静音捷径探针（FakeAVCeleb；MVP 复活）
#   管线: b6_std_trim.py (Smeu 动态 leading-silence 裁剪, τ=5e-4, 1/FPS 对齐)
#         -> prep_fakeavceleb.py --mode b6 (抽 raw/trimmed 两批缓存)
#         -> evaluate --b6(trimmed) / --b6_untrimmed(raw) 复用最终 ckpt
#   零重训；正面回应"方法是否依赖虚假静音捷径"的审稿关切。
# ============================================================================
STEP="5b_b6"
if [ "$SKIP_EVAL" = "1" ]; then
  log "SKIP: $STEP (SKIP_EVAL=1)"
elif is_done "$STEP"; then
  log "SKIP: $STEP (marker 已存在)"
else
  step_start "STEP 5b: B6 静音捷径探针（FakeAVCeleb）"
  # 1) Smeu 动态裁剪（仅裁 leading silence；FUNC: b6_std_trim.py，写 b6_std/{raw,trimmed}）
  python "$CODE_DIR/b6_std_trim.py" || log "WARN: b6_std_trim failed"
  # 2) 抽 raw / trimmed 两批缓存（FUNC: prep_fakeavceleb.py --mode b6）
  run_sharded prep_fakeavceleb.py --mode b6 || log "WARN: prep_fakeavceleb b6 failed (某分片)"
  # 3) 评测：trimmed(Smeu 裁后) vs raw(未裁基线)
  for s in $MVP_SEEDS; do
    CKPT="$PERSIST/ckpt/syncx_best_seed${s}.pt"
    log "  B6 seed=$s: trimmed(Smeu) vs raw(untrimmed)"
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
  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
# STEP 5c — 评测期消融（复用 ckpt，零重训；MVP 宿主改 HiFi-AVDF）
# ============================================================================
STEP="5c_ablations"
if [ "$SKIP_EVAL" = "1" ]; then
  log "SKIP: $STEP (SKIP_EVAL=1)"
elif is_done "$STEP"; then
  log "SKIP: $STEP (marker 已存在)"
else
  step_start "STEP 5c: 评测期消融变体（HiFi-AVDF 宿主，复用 $STAGE2_CKPT）"
  ABL_CACHE="$SYNCX_DATA_ROOT/HiFi-AVDF/prep"
  eval_var() {
    python "$CODE_DIR/evaluate.py" --ckpt "$STAGE2_CKPT" --eval_cache "$ABL_CACHE" \
        --dataset hifi --log_dir "$PERSIST/logs" --device "$DEVICE" \
        --tag "$2" $1 || log "WARN: ablation $2 failed"
  }
  log "  A1 融合：full / audio-only / video-only"
  eval_var "--mask audio"   mask_audio_masked
  eval_var "--mask visual"  mask_visual_masked
  log "  A4 LACA 三模式：both / corr / biomech"
  eval_var "--laca_mode corr"     laca_corr
  eval_var "--laca_mode biomech"  laca_biomech
  log "  §5.5 组件消融：w/o LCMA / w/o DWF / w/o BioLip x_s"
  eval_var "--disable_lcma" wo_lcma
  eval_var "--disable_dwf"  wo_dwf
  eval_var "--disable_xs"   wo_xs
  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
# STEP 5d — 训练期消融（MVP：仅 from-scratch，RUN_TRAIN_ABLATIONS=1 开启）
#   砍掉 wo_hypersph（超球面是核心贡献，砍它无意义）；只留 from-scratch 证明 SSL 增益。
# ============================================================================
STEP="5d_train_ablations"
if [ "${RUN_TRAIN_ABLATIONS:-0}" != "1" ]; then
  log "SKIP: $STEP (RUN_TRAIN_ABLATIONS != 1)"
elif is_done "$STEP"; then
  log "SKIP: $STEP (marker 已存在)"
else
  step_start "STEP 5d: 训练期消融（仅 from-scratch）"
  log "  > train: w/o Stage-1 pre-training（from scratch）"
  python "$CODE_DIR/train_stage2.py" \
      --train_cache "$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/prep/train" \
      --val_cache   "$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/prep/test" \
      --stage1_ckpt "" \
      --out_ckpt    "$PERSIST/ckpt/abl_wo_stage1.pt" \
      --log_dir     "$PERSIST/logs" \
      --epochs 80 --batch 512 --lr 1e-4 --seed "$SEED" --device "$DEVICE" \
      --num_workers 4 --empty_cache_every 1 || { log "FAIL: abl wo_stage1"; exit 1; }
  for s in $MVP_SEEDS; do
    log "  > eval abl_wo_stage1 (HiFi)"
    python "$CODE_DIR/evaluate.py" --ckpt "$PERSIST/ckpt/abl_wo_stage1.pt" \
        --eval_cache "$SYNCX_DATA_ROOT/HiFi-AVDF/prep" \
        --dataset hifi --log_dir "$PERSIST/logs" --device "$DEVICE" \
        --tag "abl_wo_stage1" || log "WARN: abl wo_stage1 eval failed"
  done
  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
# STEP 6 — 出图（CPU，分钟级；只读日志，不触发训练/评测）
# ============================================================================
STEP="6_plot"
if is_done "$STEP"; then
  log "SKIP: $STEP (marker 已存在)"
else
  step_start "STEP 6: 出图"
  python "$CODE_DIR/plot_figures.py" --log_dir "$PERSIST/logs"
  done_marker "$STEP"; log "$STEP DONE"
fi

log "=========================================================="
log "🎉 Sync-X MVP 全流程完成。"
log "   模型: $STAGE2_CKPT"
log "   日志: $PERSIST/logs  （F1–F17 中 MVP 覆盖的图由这些日志再生）"
log "=========================================================="
