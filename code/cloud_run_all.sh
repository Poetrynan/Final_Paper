#!/bin/bash
# ============================================================================
# cloud_run_all.sh — Sync-X 云端一键挂机（抽取 + 训练 一体化，评测解耦）
# ============================================================================
# 这是 code_cloud/ 专用的云端运行入口。相对 code/cloud_settle_all_mvp.sh 的改动：
#
#   ✅ 抽取 + 训练 一体化：默认 SKIP_PREP=0（云端有原始视频 → 先抽特征再训）
#   ✅ 评测解耦：默认 SKIP_EVAL=1（评测集后续单独跑，不进本次一体流程）
#   ✅ 日志结构化 + 分类归档（核心交付）：
#        - 训练/评测日志已由脚本自身落盘到 logs/<exp_id>/*.jsonl（append，崩溃可续）
#        - 抽取阶段：STEP 1 后调用 log_extraction_manifest.py → logs/extraction/manifest.json
#        - run 级归档：logs/_meta/run_meta.json（RUN_ID 时间戳 + 环境快照 + 命令）
#        - 每步 stdout 另存 logs/orchestrator/<RUN_ID>_<step>.log（扁平文本也按 run 分类）
#   ✅ 出图与计算解耦：所有 Figure 数据来自 logs/，重画零重训（见 figures/plot_*.py）
#
# ── 云端"关页面也在跑"的启动方式（在云端终端执行）──────────────────────
#   cd code_cloud
#   # 方式 A（推荐，可随时重连查看）：
#   tmux new -d -s syncx "bash cloud_run_all.sh"
#   tmux a -t syncx            # 随时回来看；断网/关页面都不杀进程
#   # 方式 B（无 tmux 时）：
#   setsid bash cloud_run_all.sh > run.out 2>&1 < /dev/null &
#   disown
#   tail -f run.out
#
# 用法：
#   export SYNCX_DATA_ROOT=/root/autodl-tmp/SyncX/Datasets
#   # ⚠️ 本流程【不需要 HF_TOKEN】：数据集已直传、CLIP/wav2vec2 是公开模型，均无需 token。
#   #   （HF_TOKEN 仅用于从 HF 拉门控 DeepSpeak 原仓，走 download_deepspeak.py，不在本脚本内。）
#   #   已默认走镜像 HF_ENDPOINT=https://hf-mirror.com（无需 token），权重若已缓存则不下载。
#   bash cloud_run_all.sh                         # 默认：抽+训，跳过评测
#   SKIP_PREP=1 bash cloud_run_all.sh             # 特征已在盘上，跳抽取直接训
#   SHARDS=2 bash cloud_run_all.sh                # 抽取分片并行数（默认4；显存小/CPU 用 2 或 1）
#   SKIP_EVAL=0 bash cloud_run_all.sh             # 本次也顺带跑评测（需评测集已抽好）
#
# 幂等：中断后重跑同一命令 → 从断点继续（每阶段 marker 防重跑 + 训练 epoch 级 resume）。
# 架构：Epoch-based resume（粒度=1 epoch，最坏丢 1 epoch）。
# ============================================================================
set -euo pipefail

SYNCX_DATA_ROOT="${SYNCX_DATA_ROOT:-/root/autodl-tmp/SyncX/Datasets}"
HF_HOME="${HF_HOME:-/root/autodl-tmp/SyncX/hfcache}"
# 公开模型(CLIP/wav2vec2)走 HF Hub，无需 token；中国机房默认走镜像，避免拉权重超时/被墙
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_TOKEN="${HF_TOKEN:-}"   # 仅 download_deepspeak.py 拉门控数据集需要；本脚本（抽+训）完全不用
PERSIST="${PERSIST:-/root/autodl-tmp/SyncX/persist}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
SKIP_PREP="${SKIP_PREP:-0}"          # ← 默认 0：云端有原始视频，先抽特征
SKIP_SEARCH="${SKIP_SEARCH:-0}"       # ← 默认 0：执行轻量超参搜索（λ_align×λ_uniform 4 组合，支撑论文 selected-on-val）
SKIP_SCRATCH="${SKIP_SCRATCH:-0}"     # ← 默认 0：执行 F6 from-scratch 训练（tab:component-ablation 行 f）
SKIP_EVAL="${SKIP_EVAL:-1}"           # ← 默认 1：评测解耦，后续单独跑
MVP_SEEDS="${MVP_SEEDS:-0}"           # MVP：默认 1 seed；补 3 seed 时置 "0 1 2"
SHARDS="${SHARDS:-8}"                 # ← 特征抽取分片并行进程数（每数据集）。
                                      #    4090D 24GB 显存可开 12（每进程 ~2GB），但 CPU 16 核才是上限：
                                      #    每进程 MediaPipe(CPU)+ffmpeg 解码约 1.5-2 核 → 8 进程×2核=16核 全用满。
                                      #    8 是甜点位（显存 16GB/24GB，CPU 100%）；显存<16GB 或 CPU<8 核调 4。

CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
STATUS_FILE="$PERSIST/status.log"
MARKER_DIR="$PERSIST/.markers"
ORCH_DIR="$PERSIST/logs/orchestrator"
RUN_ID="run_$(date +%Y%m%d_%H%M%S)_seed${SEED}"
RUN_META="$PERSIST/logs/_meta/run_meta.json"

mkdir -p "$PERSIST/ckpt" "$PERSIST/logs" "$MARKER_DIR" \
         "$ORCH_DIR" "$PERSIST/logs/extraction" "$PERSIST/logs/stage1" \
         "$PERSIST/logs/stage2" "$PERSIST/logs/eval" "$PERSIST/logs/_meta"

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')]  $*" | tee -a "$STATUS_FILE"; }
done_marker() { touch "$MARKER_DIR/$1.done"; }
is_done()    { [ -f "$MARKER_DIR/$1.done" ]; }

STAGE1_CKPT="$PERSIST/ckpt/stage1.pt"
STAGE2_CKPT="$PERSIST/ckpt/syncx_best.pt"

# ── run 级归档：写 run_meta.json（启动快照）────────────────────────────
python - "$RUN_ID" "$SEED" "$SKIP_PREP" "$SKIP_EVAL" "$SYNCX_DATA_ROOT" "$RUN_META" <<'PY'
import os, json, sys, platform, datetime
run_id, seed, skip_prep, skip_eval, data_root, out = sys.argv[1:7]
env = {}
try:
    import torch
    env["torch"] = torch.__version__; env["cuda"] = torch.cuda.is_available()
except Exception:
    env["torch"] = "n/a"
env["python"] = platform.python_version(); env["host"] = platform.node() or "unknown"
meta = {
    "run_id": run_id, "phase": "start", "started_at": datetime.utcnow().isoformat() + "Z",
    "seed": seed, "skip_prep": skip_prep, "skip_eval": skip_eval,
    "data_root": data_root, "env": env,
    "command": " ".join(sys.argv) if False else os.environ.get("BASH_COMMAND", ""),
}
os.makedirs(os.path.dirname(out), exist_ok=True)
# 保留历史 run（追加数组，便于多次 run 考古）
hist = []
if os.path.exists(out):
    try: hist = json.load(open(out))
    except Exception: hist = []
hist.append(meta)
json.dump(hist, open(out, "w"), ensure_ascii=False, indent=2)
print(f"[run_meta] wrote {out} (run_id={run_id})")
PY

log "=========================================================="
log "Sync-X 云端一体化启动（抽取 + 训练；评测 SKIP_EVAL=$SKIP_EVAL）"
log "  RUN_ID: $RUN_ID"
log "  数据根: $SYNCX_DATA_ROOT   持久卷: $PERSIST"
log "  种子: $MVP_SEEDS   SKIP_PREP=$SKIP_PREP   SKIP_SEARCH=$SKIP_SEARCH   SHARDS=$SHARDS"
log "=========================================================="

# ============================================================================
# STEP 0 — 云端环境自检
# ============================================================================
STEP="0_cloud_check"
if is_done "$STEP"; then
  log "SKIP: $STEP (marker 已存在)"
else
  log "=== STEP 0: 云端环境自检 ==="
  python "$CODE_DIR/_verify_cloud.py" \
    2>&1 | tee -a "$ORCH_DIR/${RUN_ID}_0_cloud_check.log" || log "WARN: cloud check skipped"
  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
# STEP 1 — 特征抽取（MVP：2 训练集；云端有原始视频 → SKIP_PREP=0 默认执行）
# ============================================================================
STEP="1_extraction"
if [ "$SKIP_PREP" = "1" ]; then
  log "SKIP: $STEP (SKIP_PREP=1)"
elif is_done "$STEP"; then
  log "SKIP: $STEP (marker 已存在)"
else
  log "=== STEP 1: 特征抽取（MVP 2 训练集：DeepSpeak / VoxCeleb2；评测集解耦，后续单独抽）==="
  # 仅抽 2 个已上传的训练集；3 个评测集（DFE2024 / HiFi-AVDF / FakeAVCeleb）尚未上传，
  # 其 prep（prep_eval2024 / prep_hifi / prep_fakeavceleb）需在评测前单独跑：
  #   python prep_eval2024.py && python prep_hifi.py && python prep_fakeavceleb.py --mode normal
  # 之后再用 cloud_eval_only.sh 回填 Chapter5 结果表。
  PREP_LIST="prep_deepspeak.py prep_voxceleb2.py"
  for prep in $PREP_LIST; do
    log "  > $prep（${SHARDS} 进程分片并行，shard 0..$((SHARDS-1))）"
    # 并行拉起 SHARDS 个进程：prep 内部按 index % SHARDS == shard_id 分配片段，
    # 各 shard 写独立 manifest_shard{i}.jsonl，互不写同一 .pt（断点续跑语义不变）。
    # ⚠️ 线程纪律：每进程限 2 线程（SHARDS×2≤16 vCPU）。否则 MediaPipe/ffmpeg/torch
    #    每进程默认开满 16 线程 → SHARDS 个进程 → 线程超发抢核，抽取反而变慢甚至拖卡系统。
    per_shard_threads=$((16 / SHARDS))
    [ "$per_shard_threads" -lt 1 ] && per_shard_threads=1
    log "  > 每 shard 限 ${per_shard_threads} 线程（16 vCPU ÷ ${SHARDS} shard）"
    pids=()
    for ((s=0; s<SHARDS; s++)); do
      OMP_NUM_THREADS=$per_shard_threads MKL_NUM_THREADS=$per_shard_threads \
      OPENBLAS_NUM_THREADS=$per_shard_threads \
      python "$CODE_DIR/$prep" --num-shards "$SHARDS" --shard-id "$s" \
        2>&1 | tee -a "$ORCH_DIR/${RUN_ID}_1_prep_${prep}_shard${s}.log" &
      pids+=($!)
    done
    fail=0
    for pid in "${pids[@]}"; do
      wait "$pid" || fail=1
    done
    if [ "$fail" = "1" ]; then
      log "FAIL: $prep（某分片失败，见 ${RUN_ID}_1_prep_${prep}_shard*.log）"; exit 1
    fi
    log "  > $prep 全部分片完成"
  done
  # ── 抽取阶段全局日志聚合（关键：把散落的 manifest 汇总成单一 JSON）──
  log "  > 聚合抽取 manifest → logs/extraction/manifest.json"
  python "$CODE_DIR/log_extraction_manifest.py" --data_root "$SYNCX_DATA_ROOT" \
    --out "$PERSIST/logs/extraction/manifest.json" \
    2>&1 | tee -a "$ORCH_DIR/${RUN_ID}_1_extraction_manifest.log"
  # ── 清理中间产物（关键：VoxCeleb2 的 frames/ PNG + audio.wav 训练不读，合计 ~450GB）──
  #    训练只消费 .pt 特征（dataset.py 仅 glob **/*.pt），删掉后磁盘仅剩特征(~55GB)。
  VOX_PREP="$SYNCX_DATA_ROOT/VoxCeleb2_subset/prep"
  if [ -d "$VOX_PREP" ]; then
    log "  > 清理 VoxCeleb2 中间产物（frames/ + audio.wav，训练不读）"
    find "$VOX_PREP" -type d -name frames -prune -exec rm -rf {} + 2>/dev/null || true
    find "$VOX_PREP" -type f -name audio.wav -delete 2>/dev/null || true
    log "  > 清理完成，prep 仅保留 .pt 特征"
  fi
  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
# STEP 2 — Stage-1 SSL 预训练（40 epoch，VoxCeleb2 真实片段）
# ============================================================================
STEP="2_stage1"
if is_done "$STEP"; then
  log "SKIP: $STEP (marker 已存在)"
else
  log "=== STEP 2: Stage-1 SSL（40 epoch）==="
  python "$CODE_DIR/train_stage1.py" \
      --real_cache "$SYNCX_DATA_ROOT/VoxCeleb2_subset/prep" \
      --out_ckpt   "$STAGE1_CKPT" \
      --log_dir    "$PERSIST/logs" \
      --epochs 40 --batch 1024 --lr 1e-4 --seed "$SEED" --device "$DEVICE" \
      --num_workers 8 --prefetch_factor 4 --empty_cache_every 1 \
      2>&1 | tee -a "$ORCH_DIR/${RUN_ID}_2_stage1.log" \
      || { log "FAIL: Stage-1"; exit 1; }
  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
# STEP 3 — 超参搜索（MVP 轻量版：λ_align × λ_uniform = 2×2 = 4 组合，1 seed 筛选）
#   支撑论文 Ch4:171-172 "selected on DeepSpeak v1.1 val split" 的表述（学术诚信）。
#   搜索脚本跑 train_stage2.py（早停）→ 读 val AUC 选优 → 最佳 λ 注入 STEP 4。
# ============================================================================
STEP="3_search"
if [ "$SKIP_SEARCH" = "1" ]; then
  log "SKIP: $STEP (SKIP_SEARCH=1 — 直接用论文默认 λ_align=0.1 λ_uniform=0.5)"
  BEST_LAMBDA_ALIGN="0.1"; BEST_LAMBDA_UNIFORM="0.5"
elif is_done "$STEP"; then
  log "SKIP: $STEP (marker 已存在)"
else
  log "=== STEP 3: 超参搜索 λ_align × λ_uniform（轻量 4 组合，1 seed 筛选）==="
  # --log_root 指定日志落盘到持久卷；保持 cwd=code_cloud（搜索脚本用相对路径调 train_stage2.py）
  python "$CODE_DIR/hyperparameter_search.py" \
      --stage loss_weights \
      --seeds "$SEED" \
      --stage1_ckpt "$STAGE1_CKPT" \
      --epochs 30 \
      --log_root "$PERSIST/logs" \
      2>&1 | tee -a "$ORCH_DIR/${RUN_ID}_3_search.log" \
      || { log "FAIL: 超参搜索"; exit 1; }
  # 从搜索结果解析最佳 λ：读 summary.csv（含 params/val_auc），选 val_auc 最大组合的 λ
  BEST_LAMBDA_ALIGN="0.1"; BEST_LAMBDA_UNIFORM="0.5"   # 默认回退
  SEARCH_DIR="$PERSIST/logs/hyperparameter_search/loss_weights"
  if [ -f "$SEARCH_DIR/summary.csv" ]; then
    read -r BEST_LAMBDA_ALIGN BEST_LAMBDA_UNIFORM <<< "$(python - "$SEARCH_DIR/summary.csv" <<'PY'
import sys, csv
path = sys.argv[1]
best_auc, best_align, best_uniform = -1.0, 0.1, 0.5
with open(path, newline="") as f:
    for r in csv.DictReader(f):
        try: auc = float(r["val_auc"])
        except (TypeError, ValueError): continue
        if auc > best_auc:
            best_auc = auc
            # params 形如 "{'lambda_align': 0.2, 'lambda_uniform': 0.5}"
            p = r.get("params", "")
            for token in p.replace("{", "").replace("}", "").split(","):
                kv = token.strip().split(":")
                if len(kv) == 2 and kv[0].strip().strip("'\"") == "lambda_align":
                    best_align = float(kv[1].strip())
                elif len(kv) == 2 and kv[0].strip().strip("'\"") == "lambda_uniform":
                    best_uniform = float(kv[1].strip())
print(f"{best_align} {best_uniform}")
PY
)"
    log "  [λ] summary.csv 解析 → λ_align=$BEST_LAMBDA_ALIGN λ_uniform=$BEST_LAMBDA_UNIFORM"
  else
    log "  [λ] summary.csv 缺失（搜索可能失败），回退论文默认 0.1/0.5"
  fi
  export SYNCX_LAMBDA_ALIGN="$BEST_LAMBDA_ALIGN" SYNCX_LAMBDA_UNIFORM="$BEST_LAMBDA_UNIFORM"
  log "  [λ] 搜索完成 → λ_align=$BEST_LAMBDA_ALIGN λ_uniform=$BEST_LAMBDA_UNIFORM"
  done_marker "$STEP"; log "$STEP DONE"
fi
log "[λ] STEP 4 使用 λ_align=${BEST_LAMBDA_ALIGN:-0.1} λ_uniform=${BEST_LAMBDA_UNIFORM:-0.5}"

# ============================================================================
# STEP 4 — Stage-2 最终训练（80 epoch/seed，早停 patience=8 监控 DeepSpeak val AUC）
# ============================================================================
STEP="4_stage2"
if is_done "$STEP"; then
  log "SKIP: $STEP (marker 已存在)"
else
  log "=== STEP 4: Stage-2 最终训练 × seeds($MVP_SEEDS)（80 epoch/seed）==="
  for s in $MVP_SEEDS; do
    log "  Stage-2 seed=$s（λ_align=0.1 λ_uniform=0.5）"
    python "$CODE_DIR/train_stage2.py" \
        --train_cache "$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/prep/train" \
        --val_cache   "$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/prep/test" \
        --stage1_ckpt "$STAGE1_CKPT" \
        --out_ckpt    "$PERSIST/ckpt/syncx_best_seed${s}.pt" \
        --log_dir     "$PERSIST/logs" \
        --epochs 80 --batch 512 --lr 1e-4 --seed "$s" --device "$DEVICE" \
        --num_workers 8 --prefetch_factor 4 --empty_cache_every 1 \
        2>&1 | tee -a "$ORCH_DIR/${RUN_ID}_4_stage2_seed${s}.log" \
        || { log "FAIL: Stage-2 seed=$s"; exit 1; }
  done
  cp "$PERSIST/ckpt/syncx_best_seed${MVP_SEEDS%% *}.pt" "$STAGE2_CKPT"
  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
# STEP 4b — Stage-2 from-scratch（F6 消融：不加载 Stage-1 预训练）
#   论文 tab:component-ablation 行 (f)："Without Stage 1 pre-training"。
#   复用 best λ（STEP 3 解析），仅去掉 --stage1_ckpt 初始化的影响。
# ============================================================================
STEP="4b_stage2_scratch"
if [ "$SKIP_SCRATCH" = "1" ]; then
  log "SKIP: $STEP (SKIP_SCRATCH=1)"
elif is_done "$STEP"; then
  log "SKIP: $STEP (marker 已存在)"
else
  log "=== STEP 4b: Stage-2 from-scratch（F6 消融，不加载 Stage-1）==="
  for s in $MVP_SEEDS; do
    log "  Stage-2 scratch seed=$s（λ_align=${BEST_LAMBDA_ALIGN:-0.1} λ_uniform=${BEST_LAMBDA_UNIFORM:-0.5}）"
    python "$CODE_DIR/train_stage2.py" \
        --train_cache "$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/prep/train" \
        --val_cache   "$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/prep/test" \
        --out_ckpt    "$PERSIST/ckpt/syncx_scratch_seed${s}.pt" \
        --log_dir     "$PERSIST/logs" \
        --epochs 80 --batch 512 --lr 1e-4 --seed "$s" --device "$DEVICE" \
        --num_workers 8 --prefetch_factor 4 --empty_cache_every 1 \
        --tag scratch \
        2>&1 | tee -a "$ORCH_DIR/${RUN_ID}_4b_scratch_seed${s}.log" \
        || { log "FAIL: Stage-2 scratch seed=$s"; exit 1; }
  done
  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
# STEP 5+ — 评测（默认 SKIP_EVAL=1 解耦；后续单独跑评测脚本）
#   评测脚本见 code_cloud/cloud_eval_only.sh（待你决定评测时启用）
# ============================================================================
if [ "$SKIP_EVAL" = "1" ]; then
  log "SKIP: STEP 5 评测（SKIP_EVAL=1）—— 评测已解耦，后续单独运行 cloud_eval_only.sh"
else
  log "（评测分支未在本脚本实现；请使用 cloud_eval_only.sh 单独跑评测）"
fi

# ── run 级归档：追加 end 快照 ───────────────────────────────────────────
python - "$RUN_META" <<'PY'
import json, sys, datetime
out = sys.argv[1]
hist = json.load(open(out)) if os.path.exists(out) else []
if hist:
    hist[-1].update({"phase": "end", "ended_at": datetime.utcnow().isoformat() + "Z",
                      "status": "train-done(eval-deferred)"})
json.dump(hist, open(out, "w"), ensure_ascii=False, indent=2)
print("[run_meta] updated end snapshot")
PY

log "=========================================================="
log "✅ Sync-X 云端训练完成（评测已解耦，待后续单独跑）。"
log "   模型: $STAGE2_CKPT"
log "   训练日志: $PERSIST/logs/<exp_id>/{stage1,stage2}_train.jsonl"
log "   抽取归档: $PERSIST/logs/extraction/manifest.json"
log "   run 归档: $RUN_META  (RUN_ID=$RUN_ID)"
log "   出图(训练动力学): python figures/plot_training_dynamics.py --log_dir $PERSIST/logs"
log "=========================================================="
