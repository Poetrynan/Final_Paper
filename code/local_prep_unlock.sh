#!/bin/bash
# 本地 RTX 4060 抽取 Sync-X MVP —— 并行编排版（解锁集优先）
# ---------------------------------------------------------------------------
# 目的：先抽 VoxCeleb2 + DeepSpeak（这是云端训练的唯一前置），抽完打印横幅，
#       提示你立刻在第二终端起云端训练（SKIP_PREP=1）；与此同时本脚本继续抽
#       剩下 3 个跨域评测集，与云端 GPU 训练真正并行，最大化压低墙钟。
#
# 用法（终端 1，本地免费）：
#   nohup bash local_prep_unlock.sh >/dev/null 2>&1 &
#   tail -f D:/Datasets/prep_local_unlock.log
#
# 云端（看到 "UNLOCK READY" 横幅后，终端 2）：
#   SKIP_PREP=1 bash cloud_settle_all_mvp.sh
#   （SKIP_PREP=1 让云端跳过 STEP 1 抽取，直接读本地抽好的 prep）
#
# 依赖：本地与云端必须能看到同一份 SYNCX_DATA_ROOT 下的 prep/ 目录
#       （本地写 D:/Datasets/.../prep，云端需挂载/同步到同一路径）。
#       本地抽的 3 个跨域集通常几小时内完成，远早于云端 Stage-1+Stage-2 结束，
#       因此云端 STEP 5 评测开始时这些 prep 已就绪。
# ---------------------------------------------------------------------------
set -u
PY="D:/Anaconda/envs/thesis/python.exe"
CODE="C:/Users/ZhuanZ1/Desktop/my_proposal_papers/GraduationThesis/code"
LOG="D:/Datasets/prep_local_unlock.log"

run() {
  local name="$1"; local script="$2"; shift 2
  echo "==== [$(date)] >>> 开始 $name ($script $*) ====" | tee -a "$LOG"
  "$PY" "$CODE/$script" "$@" >> "$LOG" 2>&1
  local rc=$?
  echo "==== [$(date)] <<< $name 结束 退出码=$rc ====" | tee -a "$LOG"
  return $rc
}

echo "==== [$(date)] 本地 MVP 抽取启动（解锁集优先 / RTX 4060）====" | tee -a "$LOG"

# ── 解锁集：VoxCeleb2(Stage-1 SSL) + DeepSpeak(Stage-2 监督) ──
#    这两个抽完，云端才能开始训练；因此前置、优先。
run "VoxCeleb2_subset" prep_voxceleb2.py
run "DeepSpeak_v1_1"  prep_deepspeak.py

echo "############################################################" | tee -a "$LOG"
echo "#### [$(date)] >>> UNLOCK READY：现在可起云端训练 <<< ####" | tee -a "$LOG"
echo "####   云端命令（第二终端）：SKIP_PREP=1 bash cloud_settle_all_mvp.sh ####" | tee -a "$LOG"
echo "####   本脚本继续抽剩下 3 个跨域集（与云端并行）... ####" | tee -a "$LOG"
echo "############################################################" | tee -a "$LOG"

# ── 剩余跨域评测集：与云端训练并行抽取（不影响云端启动）──
run "Deepfake-Eval-2024" prep_eval2024.py
run "HiFi-AVDF"       prep_hifi.py
run "FakeAVCeleb_v1.2" prep_fakeavceleb.py --mode normal

echo "==== [$(date)] ★ 全部 MVP 集抽取完成（本地）====" | tee -a "$LOG"
