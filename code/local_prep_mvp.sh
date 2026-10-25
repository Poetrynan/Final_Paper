#!/bin/bash
# 本地 RTX 4060 抽取 Sync-X MVP 5 集（串行 + 每集断点续跑 + 关机续跑）
# 数据集：DeepSpeak_v1_1(训) / VoxCeleb2(Stage-1) / DFE2024(评) / HiFi-AVDF(评) / FakeAVCeleb_v1.2(评+B6)
# 说明：Celeb-DF-v2 无音轨已从 MVP 砍掉；FakeAVCeleb 作第 3 跨域评测集 + B6 宿主。
# 用法（任意时刻重跑本脚本即可续跑，已抽片段自动跳过）：
#   nohup bash local_prep_mvp.sh >/dev/null 2>&1 &
#   tail -f D:/Datasets/prep_local_mvp.log
set -u
PY="D:/Anaconda/envs/thesis/python.exe"
CODE="C:/Users/ZhuanZ1/Desktop/my_proposal_papers/GraduationThesis/code"
LOG="D:/Datasets/prep_local_mvp.log"

run() {
  local name="$1"; local script="$2"; shift 2
  echo "==== [$(date)] >>> 开始 $name ($script $*) ====" | tee -a "$LOG"
  "$PY" "$CODE/$script" "$@" >> "$LOG" 2>&1
  local rc=$?
  echo "==== [$(date)] <<< $name 结束 退出码=$rc ====" | tee -a "$LOG"
  return $rc
}

echo "==== [$(date)] 本地 MVP 抽取启动（环境 thesis / RTX 4060）====" | tee -a "$LOG"
run "DeepSpeak_v1_1"  prep_deepspeak.py
run "VoxCeleb2_subset" prep_voxceleb2.py
run "Deepfake-Eval-2024" prep_eval2024.py
run "HiFi-AVDF"       prep_hifi.py
run "FakeAVCeleb_v1.2" prep_fakeavceleb.py --mode normal
echo "==== [$(date)] ★ 全部 MVP 集抽取完成 ====" | tee -a "$LOG"
