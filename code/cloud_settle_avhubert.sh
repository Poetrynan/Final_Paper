#!/bin/bash
# ============================================================================
# cloud_settle_avhubert.sh — 换载体实验 B：AV-HuBERT 联合 token 替代 wav2vec2
# ============================================================================
# 动机（§16）：特征载体探针判定分离载体(CLIP视觉+wav2vec2 音频) NOT_SEPARABLE，
#   分离特征不带「跨生成器一致的音画一致性」→ 方向 B：把 audio_seq 的载体换成
#   AV-HuBERT（音画联合自监督预训练 token）。若 B 提升跨域 AUC，则一致性信息
#   确实在"联合载体"里、只是分离载体没装它；若仍不提升，则根因收敛到
#   "判别头监督"而非载体，整条线按 §11.3 收稿。

# ⚠️ 硬门槛：AV-HuBERT 是 fairseq-only + 权重需网络下载。本脚本第一步 verify，
#   通过前**绝不**进入 prep/训练（防盲烧 GPU）。权重 URL 用 verify 探测可达者。

# 关键设计（单变量守卫）：
#   ✅ 视觉前端(v_clip CLIP / v_lip MediaPipe)与 A 组**完全一致**，仅换音频上行
#      wav2vec2 → AV-HuBERT 联合 token(→768，经 post_extract_proj)
#   ✅ 输出全部用【新 path】(avh_*)；绝不覆盖 baseline(/stem prep)
#   ✅ prep 输出到 <data_root>/prep_avh 独立树，原 prep 零改动、可随时回退
#   ✅ Stage-1/2 用与 baseline 相同配方，唯一变量 = 载体

# 用法（云端）：
#   cd /root/autodl-tmp/SyncX/code_cloud
#   tmux new -d -s avh "bash cloud_settle_avhubert.sh"     # 推荐
#   tmux a -t avh
# 环境变量：
#   EPOCHS_S1/EPOCHS_S2/BATCH_S1/SEED  同 baseline 默认
#   PREP_SHARDS   每个数据集 prep 分片数（默认 8；显存紧则调小，AV-HuBERT 较吃显存）
#   SKIP_VERIFY=1 已跑过 verify 且名校验通过（默认强制验）
#   FRESH=1       清本 pilot 旧产物
#
# ⚠️ 运行环境（2026-08-29 已打通）：必须用 conda 环境 py39，且 av_hubert 已就绪——
#   • fairseq 1.0.0a0 = av_hubert 官方 fairseq 子模块(commit afc77bd)，editable 安装
#   • avhubert 源码目录挂载（PYTHONPATH 注入，见 runpy）
#   • omegaconf 2.0.6 / hydra-core 1.0.7 / numpy 1.23.5 / scipy 1.10 /
#     opencv-python 4.5.4.60 / sentencepiece 0.1.96 / editdistance / python_speech_features
#   本脚本已把一切 python 调用固定到 `conda run -n py39 python`，勿改。
# ============================================================================
set -euo pipefail

SYNCX_DATA_ROOT="${SYNCX_DATA_ROOT:-/root/autodl-tmp/SyncX/Datasets}"
HF_HOME="${HF_HOME:-/root/autodl-tmp/SyncX/hfcache}"
PERSIST="${PERSIST:-/root/autodl-tmp/SyncX/persist}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-0}"
EPOCHS_S1="${EPOCHS_S1:-60}"
EPOCHS_S2="${EPOCHS_S2:-80}"
BATCH_S1="${BATCH_S1:-1024}"
PREP_SHARDS="${PREP_SHARDS:-8}"

export SYNCX_COMPILE=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export SYNCX_MEDIAPIPE_GPU=0
export SYNCX_FTA=1                 # 与 baseline(方向B) 一致
export HF_HOME="$HF_HOME"

CODE_DIR="$(cd "$(dirname "$0")" && pwd)"
STATUS_FILE="$PERSIST/status_avhubert.log"
MARKER_DIR="$PERSIST/.markers_avhubert"
mkdir -p "$MARKER_DIR"
log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')]  $*" | tee -a "$STATUS_FILE"; }
done_marker() { touch "$MARKER_DIR/$1.done"; }
is_done()    { [ -f "$MARKER_DIR/$1.done" ]; }

# ── AV-HuBERT 运行环境（2026-08-29 已验证）──────────────────────────────
# 官方 av_hubert 用 fairseq 子模块(commit afc77bd)，依赖公平验证组合：
#   fairseq 1.0.0a0 (editable, av_hubert/fairseq) + avhubert(源码挂载)
#   omegaconf 2.0.6 / hydra-core 1.0.7 / numpy 1.23.5 / scipy 1.10 / 
#   opencv 4.5.4.60 / sentencepiece 0.1.96 / editdistance / python_speech_features
# 全部位于 conda 环境 py39。AV-HuBERT 权重由 verify 下载写 $AVH_CKPT_DIR。
# 所有 python 一律走：PY39 python ...（避免主环境 3.12 的 dataclasses/omegaconf 不兼容）。
# 所有 python 一律强制走 py39（主环境 3.12 的 dataclasses/omegaconf 与 fairseq 不兼容）
PY="conda run -n py39 python"
AVH_SRC="${AVH_SRC:-$CODE_DIR/av_hubert}"        # av_hubert 根目录
AVH_PKG="$AVH_SRC/avhubert"                       # 包本体目录（含 hubert.py / utils.py）
# prep/训练/评测脚本均通过 CLI 参数进入（sys.argv 长度>1 → avhubert DBG=False，
# 走相对导入，规避 DEBUG 顶层导入分支）。此处仅导出路径供子进程 import。
export AVH_PKG
# 执行 python（注入 av_hubert 的 PYTHONPATH，供 avhubert/fairseq 顶层 import）
# 修正：PYTHONPATH 用 export 提到环境层，而非 runpy 行内临时前缀。
#   conda run 生成的子进程不继承行内 `VAR=...` 临时环境前缀，导致 import avhubert
#   失败(No module named 'avhubert')。改为 export 后所有 python 子进程均能搜到。
export PYTHONPATH="$AVH_SRC:$AVH_PKG:${PYTHONPATH:-}"
runpy() {
  $PY "$@"
}
# 无 PYTHONPATH 注入的普通 python（纯项目脚本，不触 avhubert）
runpy_plain() { $PY "$@"; }

S1_CKPT="$PERSIST/ckpt/stage1_avh.pt"
S2_CKPT="$PERSIST/ckpt/syncx_avh_best_seed${SEED}.pt"
BASE_CKPT="$PERSIST/ckpt/syncx_best_seed${SEED}.pt"    # baseline 参照，只读

# AV-HuBERT 权重（verify 会下载候选可达者并写 /root/avhubert_ckpt）
AVH_CKPT_DIR="/root/avhubert_ckpt"

# ── 数据集 / prep_avh 缓存根 ─────────────────────────────────────────────
VOCAB_PREP="$SYNCX_DATA_ROOT/VoxCeleb2_subset/prep_avh"           # Stage-1 real_SSL
DS_TRAIN_AVH="$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/prep_avh_train" # Stage-2 训练
DS_VAL_AVH="$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/prep_avh_test"
# 跨域评测（独立 prep_avh 树）
declare -A EVAL=(
  [fakeavceleb]="$SYNCX_DATA_ROOT/FakeAVCeleb_v1.2/prep_avh"
  [DFE2024]="$SYNCX_DATA_ROOT/Deepfake-Eval-2024/prep_avh_test"
  [hifi]="$SYNCX_DATA_ROOT/HiFi-AVDF/prep_avh"
)

if [ "${FRESH:-0}" = "1" ]; then
  log "[FRESH] 清空 §B(avhubert) 试点旧产物"
  rm -f "$S1_CKPT" "$S1_CKPT.resume"
  rm -f "$S2_CKPT" "$S2_CKPT.resume"
  rm -f "$MARKER_DIR"/*.done
fi

log "============================================================"
log "换载体实验 B (AV-HuBERT) 启动 seed=$SEED S1=$EPOCHS_S1 S2=$EPOCHS_S2"
log "  方案：audio_seq 换 AV-HuBERT 联合 token(768)，视觉前端与 A 组一致"
log "  baseline(只读参照): $BASE_CKPT"
log "  新输出: $S1_CKPT -> $S2_CKPT"
log "============================================================"

prep_one() {  # $1=ref_prep  $2=src_root  $3=out_root
  local ref="$1" src="$2" out="$3"
  # ✅ 修正 SKIP 判断：cache 可能位于 0-XXXX/ 子目录（非顶层），须用 find 递归计数。
  #    旧 `ls "$out"/cache.pt` 只查顶层 → 对子目录结构的已完树(如 DeepSpeak)判断失效，
  #    会误对一个已抽完的树重新全量跑，触发 AV-HuBERT 加载内存峰值 → 62G 容器 OOM-killer 砍掉。
  local have
  have="$(find "$out" -name cache.pt 2>/dev/null | wc -l)"
  if [ "${have:-0}" -gt 0 ]; then
    log "  SKIP prep $out (已有 ${have} 个 cache, 幂等跳过)"; return
  fi
  log "  prep $out  (shards=$PREP_SHARDS)"
  : > "$PERSIST/status_prep_avh.log"
  for ((sh=0; sh<PREP_SHARDS; sh++)); do
    runpy "$CODE_DIR/prep_avhubert.py" \
        --ref-prep "$ref" --src-root "$src" --out-root "$out" \
        --num-shards "$PREP_SHARDS" --shard-id "$sh" --device "$DEVICE" \
        >> "$PERSIST/status_prep_avh.log" 2>&1 &
  done
  wait   # 等本数据集全部分片
  log "  done prep $out (cache.pt count: $(find "$out" -name cache.pt | wc -l))"
}

# ============================================================================
# STEP 00 — avh_clone：确保 avhubert 包可得（pypi import / 现有 clone / 自动 git clone）
# ============================================================================
# 用户红线：AV-HuBERT 是 fairseq-only，必须先拿到官方仓库（`import avhubert` 注册
# model/task class 的前提）。这里自动完成；也可 SKIP_CLONE=1 跳过（若已手动 clone）。
STEP="avh_clone"
if is_done "$STEP" && [ "${SKIP_CLONE:-0}" = "1" ]; then
  log "SKIP: $STEP (marker + SKIP_CLONE=1)"
else
  log "=== STEP: 检查/准备 avhubert 包 ==="
  if $PY -c "import sys; sys.argv=['-c','x']; import avhubert" 2>/dev/null; then
    log "  avhubert 已可 import（py39 已挂载），跳过 clone"
  elif [ -d "$CODE_DIR/av_hubert/avhubert" ]; then
    log "  复用现有 clone: $CODE_DIR/av_hubert（确认 avhubert 可导）"
    $PY -c "import sys; sys.argv=['-c','x']; import avhubert" 2>/dev/null \
      || { log "FATAL: avhubert 在 py39 里 import 失败——请按环境说明部署（见文档 §依赖）。"; exit 1; }
  else
    log "  未发现 avhubert。注意：官方 av_hubert 依赖 fairseq 子模块(pin commit afc77bd)。"
    log "  已在 py39 手动部署者应有 $CODE_DIR/av_hubert/avhubert。若缺失，请手动："
    log "    curl -L -o /tmp/avh.tgz https://codeload.github.com/facebookresearch/av_hubert/tar.gz/HEAD"
    log "    tar 解包并按 README 同步 fairseq 子模块 commit afc77bd 后放回 $CODE_DIR/av_hubert"
    exit 1
  fi
  [ -d "$CODE_DIR/av_hubert/avhubert" ] \
    || { log "FATAL: 找不到 $CODE_DIR/av_hubert/avhubert/（包目录缺失，clone 不完整）"; exit 1; }
  log "  avhubert 包目录确认: $CODE_DIR/av_hubert/avhubert"
  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
# STEP 0 — verify 硬门槛（通过才允许后续所有步骤）
# ============================================================================
STEP="avh_verify"
if is_done "$STEP" && [ "${SKIP_VERIFY:-0}" = "1" ]; then
  log "SKIP: $STEP (marker + SKIP_VERIFY=1)"
else
  log "=== STEP: AV-HuBERT verify 硬门槛 ==="
  # 幂等：已有权重（prior 下载/手动放）则直接用它，避免每次重下 1.2G 触发网络不稳。
  CAND_CKPT="$(ls "$AVH_CKPT_DIR"/*.pt 2>/dev/null | head -1)"
  if ! runpy "$CODE_DIR/verify_avhubert.py" --ckpt "${CAND_CKPT:-}" ; then
    log "FATAL: AV-HuBERT verify 未通过。先解决（网络/权重/fairseq/avhubert 包），"
    log "       再重跑: bash cloud_settle_avhubert.sh"
    exit 1
  fi
  # verify 成功会写 /root/avhubert_ckpt；导出路径供后续
  export SYNCX_AVH_CKPT="$([ -n "$CAND_CKPT" ] && echo "$CAND_CKPT" || ls "$AVH_CKPT_DIR"/*.pt 2>/dev/null | head -1)"
  [ -n "$SYNCX_AVH_CKPT" ] || { log "FATAL: verify 通过但没找到权重路径"; exit 1; }
  done_marker "$STEP"; log "$STEP DONE (ckpt=$SYNCX_AVH_CKPT)"
fi

# ============================================================================
# STEP 0b — avh_smoke：用 HiFi 真实 clip 小批量端到端联合前向（通过才允许全量 prep）
# ============================================================================
STEP="avh_smoke"
if ! is_done "$STEP"; then
  log "=== STEP: AV-HuBERT smoke（HiFi-AVDF 头部真实 clip 端到端联合前向）==="
  SMOKE_OUT="$PERSIST/.smoke_avh"
  rm -rf "$SMOKE_OUT"
  if ! runpy "$CODE_DIR/prep_avhubert.py" \
        --ref-prep "${EVAL[hifi]}" --src-root "$SYNCX_DATA_ROOT/HiFi-AVDF" \
        --out-root "$SMOKE_OUT" --num-shards 1 --shard-id 0 \
        --limit "${SMOKE_N:-8}" --strict --device "$DEVICE"; then
    log "FATAL: smoke 未通过 —— AV-HuBERT 联合前向不可用（logfbank audio_dim、唇 ROI、extract_features 任一环节），"
    log "       先修（见状态日志），勿进全量 prep/训练。"
    exit 1
  fi
  _cnt=$(find "$SMOKE_OUT" -name cache.pt | wc -l)
  if [ "$_cnt" -lt 1 ]; then
    log "FATAL: smoke 产出 ${_cnt} 个 cache.pt（需≥1）——真实视频 AV-HuBERT 前向失败。"; exit 1
  fi
  log "smoke OK：${_cnt} 个真实 clip 端到端联合前向通过（cache 维度由 --strict 保证 200×768）"
  rm -rf "$SMOKE_OUT"
  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
# STEP 1 — 全套数据 prep 换载体（Vox real + DeepSpeak train/test + 3 跨域）
# ============================================================================
STEP="avh_prep"
if ! is_done "$STEP"; then
  log "=== STEP: 全数据集 AV-HuBERT prep（独立 prep_avh 树）==="
  status_prep="$PERSIST/status_prep_avh.log"; : > "$status_prep"

  # Stage-1 真实域（Vox）：ref=Vox prep, src=Vox 根
  prep_one "$SYNCX_DATA_ROOT/VoxCeleb2_subset/prep" \
           "$SYNCX_DATA_ROOT/VoxCeleb2_subset" "$VOCAB_PREP"

  # Stage-2 训练域（DeepSpeak）：ref=DeepSpeak prep/train, src=DeepSpeak 根
  prep_one "$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/prep/train" \
           "$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips" "$DS_TRAIN_AVH"
  prep_one "$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/prep/test" \
           "$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips" "$DS_VAL_AVH"

  # 3 跨域评测集
  prep_one "$SYNCX_DATA_ROOT/FakeAVCeleb_v1.2/prep" \
           "$SYNCX_DATA_ROOT/FakeAVCeleb_v1.2" "${EVAL[fakeavceleb]}"
  prep_one "$SYNCX_DATA_ROOT/Deepfake-Eval-2024/prep/test" \
           "$SYNCX_DATA_ROOT/Deepfake-Eval-2024" "${EVAL[DFE2024]}"
  prep_one "$SYNCX_DATA_ROOT/HiFi-AVDF/prep" \
           "$SYNCX_DATA_ROOT/HiFi-AVDF" "${EVAL[hifi]}"

  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
# STEP 2 — Stage-1（AV-HuBERT 真实域 SSL，60 ep：与 HAVIC 单变量对齐）
# ============================================================================
STEP="avh_s1"
if is_done "$STEP"; then
  log "SKIP: $STEP"
else
  log "=== STEP: Stage-1 AV-HuBERT 真实 SSL (Vox prep_avh) ==="
  runpy_plain "$CODE_DIR/train_stage1.py" \
      --real_cache "$VOCAB_PREP" \
      --out_ckpt   "$S1_CKPT" \
      --log_dir    "$PERSIST/logs" \
      --epochs "$EPOCHS_S1" --batch "$BATCH_S1" --lr 1e-4 --seed "$SEED" --device "$DEVICE" \
      --num_workers 8 --prefetch_factor 4 --empty_cache_every 1 \
      || { log "FAIL: Stage-1 AVH"; exit 1; }
  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
# STEP 3 — Stage-2（从 AV-HuBERT Stage-1，baseline 配方，用 DeepSpeak prep_avh）
# ============================================================================
STEP="avh_s2"
if is_done "$STEP"; then
  log "SKIP: $STEP"
else
  log "=== STEP: Stage-2 AV-HuBERT 重训（DeepSpeak prep_avh）==="
  export SYNCX_ALIGN_LOSS=0
  runpy_plain "$CODE_DIR/train_stage2.py" \
      --train_cache "$DS_TRAIN_AVH" --val_cache "$DS_VAL_AVH" \
      --stage1_ckpt "$S1_CKPT" \
      --out_ckpt    "$S2_CKPT" \
      --log_dir     "$PERSIST/logs" \
      --epochs "$EPOCHS_S2" --batch 512 --lr 1e-4 --seed "$SEED" --device "$DEVICE" \
      --num_workers 8 --prefetch_factor 4 --empty_cache_every 1 \
      --tag "avh_s2" \
      || { log "FAIL: Stage-2 AVH"; exit 1; }
  done_marker "$STEP"; log "$STEP DONE"
fi

# ============================================================================
# STEP 4 — 3 跨域评测（AV-HuBERT 模型 vs baseline 对比）
# ============================================================================
STEP="avh_eval"
if is_done "$STEP"; then
  log "SKIP: $STEP"
else
  log "=== STEP: 3 跨域集评测 (avh vs baseline) ==="
  for key in fakeavceleb DFE2024 hifi; do
    cache="${EVAL[$key]}"
    [ -e "$cache" ] || { log "WARN: $key cache 缺失 $cache"; continue; }
    log "  Eval avh $key ($cache)..."
    runpy_plain "$CODE_DIR/evaluate.py" \
        --ckpt "$S2_CKPT" --eval_cache "$cache" \
        --log_dir "$PERSIST/logs" --dataset "$key" --seed "$SEED" --device "$DEVICE" \
        --tag "avh_s2" || log "WARN: eval avh $key failed"
  done
  log "  Eval baseline 参照（用原 prep 缓存，只读对照）..."
  for pair in "FakeAVCeleb_v1.2/prep:fakeavceleb" \
              "Deepfake-Eval-2024/prep:DFE2024" "HiFi-AVDF/prep:hifi"; do
    bdir="${pair%%:*}"; bkey="${pair##*:}"
    bcache="$SYNCX_DATA_ROOT/$bdir"
    [ -e "$bcache" ] || { log "WARN: baseline cache 缺失 $bcache"; continue; }
    runpy_plain "$CODE_DIR/evaluate.py" \
        --ckpt "$BASE_CKPT" --eval_cache "$bcache" \
        --log_dir "$PERSIST/logs" --dataset "$bkey" --seed "$SEED" --device "$DEVICE" \
        --tag baseline_ref >/dev/null 2>&1 || true
  done
  done_marker "$STEP"; log "$STEP DONE"
fi

log "============================================================"
log "✅ 换载体实验 B (AV-HuBERT) 完成。"
log "   新 ckpt: $S2_CKPT   baseline: $BASE_CKPT"
log "   查看指标: grep -h auc $PERSIST/logs/*/eval/*.jsonl  或看 status 文件按提示"
log "============================================================"
echo
echo "# ── 查看进度的命令（可随时自查）────────────────────────"
echo "  # 当前跑到哪一步 / 失败原因:"
echo "  tail -40 $STATUS_FILE"
echo "  ls -1 $MARKER_DIR"
echo "  # Stage-1 / Stage-2 训练实时日志:"
echo "  ls -t $PERSIST/logs/*/train*.jsonl | head -1  # 最新训练日志"
echo "  # 跨域评测结果:"
echo "  grep -rh auc $PERSIST/logs/*/eval/*.jsonl 2>/dev/null | tail -20"
echo "  # prep 进度（cache.pt 数量应持续增长至参考树规模):"
echo "  find ${EVAL[hifi]} -name cache.pt | wc -l"
echo "----------------------------------------------------------"