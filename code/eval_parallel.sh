#!/bin/bash
# ============================================================================
# eval_parallel.sh — 多条 evaluate.py 消融评测并行执行器（加速串行多次）
# ============================================================================
# 作用：把 "同一 ckpt × 多组消融参数（--mask/--disable_dwf/--laca_mode...)" 的
#       多次 evaluate.py 调用的"串行"压成"多进程并发"，共抢 GPU 算力加速。
# ⚠️ 若并发导致 CUDA OOM：把 MAX_CONCURRENCY 调小（如 1 即串行），
#    或删掉命令串末尾的 " & " 手动逐条跑。
#
# 用法：
#   bash eval_parallel.sh                 # 用脚本内默认清单（DeepSpeak 关键消融）
#   bash eval_parallel.sh <cmds_file>     # 自定义清单文件，每行一条完整 evaluate.py 命令
#   可选环境变量：MAX_CONCURRENCY（并行上限）
# ============================================================================
set -u

CMDS_FILE="${1:-}"
MAX_CONC="${MAX_CONCURRENCY:-4}"
LOG_DIR="${LOG_DIR:-persist/logs/eval_parallel}"
mkdir -p "$LOG_DIR"

write_default_cmds() {
  cat <<CMD
python evaluate.py --ckpt /root/autodl-tmp/SyncX/persist/ckpt/syncx_best_seed0.pt --eval_cache /root/autodl-tmp/SyncX/Datasets/DeepSpeak_v1_1/clips/prep --mask audio  --batch_size 1024 --num_workers 8 --device cuda --store_scores
python evaluate.py --ckpt /root/autodl-tmp/SyncX/persist/ckpt/syncx_best_seed0.pt --eval_cache /root/autodl-tmp/SyncX/Datasets/DeepSpeak_v1_1/clips/prep --mask visual --batch_size 1024 --num_workers 8 --device cuda --store_scores
python evaluate.py --ckpt /root/autodl-tmp/SyncX/persist/ckpt/syncx_best_seed0.pt --eval_cache /root/autodl-tmp/SyncX/Datasets/DeepSpeak_v1_1/clips/prep --disable_dwf --batch_size 1024 --num_workers 8 --device cuda --store_scores
CMD
}

if [ -n "$CMDS_FILE" ]; then
  mapfile -t CMDS < "$CMDS_FILE"
else
  mapfile -t CMDS < <(write_default_cmds)
fi

echo "[eval_parallel] 并行上限=$MAX_CONC  日志=$LOG_DIR  共 ${#CMDS[@]} 条"
PIDS=()
i=0
for c in "${CMDS[@]}"; do
  c="$(echo "$c" | xargs)"
  [ -z "$c" ] && continue
  # 控制并发上限
  while [ ${#PIDS[@]} -ge "$MAX_CONC" ]; do
    live=()
    for p in "${PIDS[@]}"; do
      if kill -0 "$p" 2>/dev/null; then live+=("$p"); fi
    done
    PIDS=("${live[@]}")
    [ ${#PIDS[@]} -ge "$MAX_CONC" ] && sleep 3
  done
  echo "  [job#$i] $c"
  eval "$c" > "$LOG_DIR/job_${i}.log" 2>&1 &
  PIDS+=($!)
  i=$((i+1))
done
for p in "${PIDS[@]}"; do
  wait "$p" || echo "  [job $p] FAILED（见对应日志）"
done

echo "===== 汇总（overall 行） ====="
for f in "$LOG_DIR"/job_*.log; do
  echo "--- $(basename "$f") ---"
  grep "\[eval\] overall" "$f" || { echo "   (无 overall，尾部:)"; tail -n 3 "$f"; }
done