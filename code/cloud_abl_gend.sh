#!/bin/bash
# ============================================================================
# cloud_abl_gend.sh — GenD 消融（Rec B / Rec C）专用 runner
# ============================================================================
# 目的：按「GenD_role_and_empirical_gate.md §4」的决策门，把「w/o Rec B」和
#       「w/o Rec C」两个单变量消融跑出来，ΔAUC 对比主配置（syncx_best_seed*）。
# 设计要点：
#   1) 输出到**独立的 ckpt**（${TAG}_seed<s>.pt），绝不覆盖主模型 syncx_best_seed*.pt。
#   2) 不依赖 cloud_settle_all_mvp.sh（其 STEP5d 把超球面消融砍掉了，且会写主 ckpt）。
#   3) 训练配方全部锁成与主配置一致（FTA=1 / batch512 / lr1e-4 / 80ep /
#      λ_audio=1.0 / corr_real_only=1），唯一变化是目标 GenD 开关 → 单变量对照。
#   4) 只需已有特征缓存，不额外抽特征（云端 prep 已就绪）。
#
# 用法（云端，先上传本脚本 + 新增 SYNCX_NO_SAME_SOURCE 钩子的 train_stage2.py）：
#   # 消融 nob：w/o Rec B（超球面 metric 关闭，Rec C 同源配对保持开启）
#   bash cloud_abl_gend.sh nob                 # seed0 探针
#   bash cloud_abl_gend.sh nob "0 1 2"         # 补全三 seed
#   # 消融 noc：w/o Rec C（同源配对关闭，Rec B 超球面保持主配置 0.1/0.5）
#   bash cloud_abl_gend.sh noc
#   # 云端后台（用不再丢环境变量的方式启动，勿用 `VAR=1 tmux ...` 前缀）：
#   tmux new -d -s abl_nob "bash /root/autodl-tmp/SyncX/code_cloud/cloud_abl_gend.sh nob"
#   tmux new -d -s abl_noc "bash /root/autodl-tmp/SyncX/code_cloud/cloud_abl_gend.sh noc"
#   # 进度：tmux a -t abl_nob / tail -f $PERSIST/logs/*
# ============================================================================
set -euo pipefail

# 所有 SYNCX_* 与路径均在本脚本内显式 export；这里只用位置参数传 MODE/ISEEDS，
# 避免 tmux 新窗口不继承环境变量前缀的问题（详见项目记忆「tmux 吞环境变量」教训）。
MODE="${1:?用法: bash cloud_abl_gend.sh nob|noc [iseeds]}"
ISEEDS="${2:-0}"                            # 默认 seed0 探针；严谨可 "0 1 2"
SYNCX_DATA_ROOT="${SYNCX_DATA_ROOT:-/root/autodl-tmp/SyncX/Datasets}"
PERSIST="${PERSIST:-/root/autodl-tmp/SyncX/persist}"
DEVICE="${DEVICE:-cuda}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')]  $*"; }

# ── 与主配置一致的训练配方（单变量：只变 GenD 开关）────────────────────────
export SYNCX_COMPILE=0                      # 硬性：4090D 上 torch.compile 启动即崩
export SYNCX_FTA=1                          # 主模型=LTCA 三支；消融必须同架构
export DIRECT_B=0                           # 绝不清主模型权重/续跑点
export SYNCX_LAMBDA_AUDIO=1.0               # 锁主配置
export SYNCX_CORR_REAL_ONLY=1               # 锁主配置

STAGE1="$PERSIST/ckpt/stage1.pt"
TRAIN="$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/prep/train"
VAL="$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/prep/test"
LOG="$PERSIST/logs"

# ── 目标开关：按 MODE 设置唯一变化项 ───────────────────────────────────────
if [ "$MODE" = "nob" ]; then
  export SYNCX_LAMBDA_ALIGN=0  SYNCX_LAMBDA_UNIFORM=0   # Rec B OFF
  export SYNCX_NO_SAME_SOURCE=0                          # Rec C keep ON
  TAG="abl_nob"; DESC="w/o Rec B(超球面 off) · Rec C on"
elif [ "$MODE" = "noc" ]; then
  export SYNCX_LAMBDA_ALIGN=0.1  SYNCX_LAMBDA_UNIFORM=0.5  # Rec B keep主配置
  export SYNCX_NO_SAME_SOURCE=1                             # Rec C OFF
  TAG="abl_noc"; DESC="w/o Rec C(同源配对 off) · Rec B on(0.1/0.5)"
else
  log "MODE must be nob|noc"; exit 2
fi

log "=========================================================="
log "GenD 消融：$DESC   seeds=$ISEEDS"
log "  输出 ckpt: $PERSIST/ckpt/${TAG}_seed*.pt (独立，不影响 syncx_best_seed*)"
log "=========================================================="

for s in $ISEEDS; do
  CKPT="$PERSIST/ckpt/${TAG}_seed${s}.pt"
  log "── [${TAG}] Stage-2 训练 seed=$s ──"
  python "$(dirname "$0")/train_stage2.py" \
      --train_cache "$TRAIN" --val_cache "$VAL" \
      --stage1_ckpt "$STAGE1" \
      --out_ckpt "$CKPT" --log_dir "$LOG" \
      --epochs 80 --batch 512 --lr 1e-4 --seed "$s" --device "$DEVICE" \
      --num_workers 4 --empty_cache_every 1 --tag "$TAG" \
      || { log "FAIL train ${TAG} seed=$s"; exit 1; }

  for pair in "FakeAVCeleb_v1.2:fakeavceleb" "Deepfake-Eval-2024:DFE2024" "HiFi-AVDF:hifi"; do
    ds="${pair%%:*}"; key="${pair##*:}"
    log "── [${TAG}] 评测 $key seed=$s ──"
    python "$(dirname "$0")/evaluate.py" \
        --ckpt "$CKPT" --eval_cache "$SYNCX_DATA_ROOT/${ds}/prep" \
        --dataset "$key" --seed "$s" --device "$DEVICE" \
        --log_dir "$LOG" --tag "${TAG}_seed${s}" \
        || log "WARN: eval $key seed=$s failed (continuing)"
  done
done
log "[${TAG}] all done."