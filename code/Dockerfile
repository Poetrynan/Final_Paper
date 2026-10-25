# =============================================================================
# Sync-X 训练/抽取镜像
# -----------------------------------------------------------------------------
# 用法：
#   docker build -t syncx:latest .
#   docker run --gpus all --shm-size=16g \
#     -e SYNCX_DATA_ROOT=/root/autodl-tmp/SyncX/Datasets \
#     -e HF_HOME=/root/autodl-tmp/SyncX/hfcache -e HF_TOKEN=<your_hf_token> \
#     -e SYNCX_FACE_LANDMARKER_MODEL=/models/face_landmarker.task \
#     -v /your/datasets:/root/autodl-tmp/SyncX/Datasets:ro \
#     -v /your/persist:/syncx/persist \
#     -w /syncx syncx:latest \
#     python code/train_stage2.py --train_cache /data/.../prep/train \
#       --out_ckpt /syncx/persist/ckpt/syncx_best.pt --log_dir /syncx/persist/logs
#
# 要点：
#   * CUDA 12.1 + cuDNN8 基础镜像，匹配 requirements.txt 的 cu121 wheel。
#   * ffmpeg 系统安装（prep 抽音频必需）。
#   * 模型权重离线预烘焙：构建时把 face_landmarker.task 放进 /models，
#     运行时用 SYNCX_FACE_LANDMARKER_MODEL 指向；CLIP/wav2vec2 走 HF_HOME 缓存。
#   * /dev/shm 默认在容器里很小，跑命令务必 --shm-size（或 num_workers 调小）。
# =============================================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # 防显存碎片 OOM：让 CUDA caching allocator 用可扩展段而非预保留大块，
    # 长训练/大 batch 下显著降低“显存明明够却炸”的碎片问题（对性能几乎无影响）。
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 系统依赖：ffmpeg（抽音频）+ python + 编译基础
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        ffmpeg \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# 把代码拷进镜像并装依赖（先装 torch cu121 wheel，再装其余）
WORKDIR /syncx
COPY code/requirements.txt /syncx/code/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /syncx/code/requirements.txt

# 代码本体
COPY code/ /syncx/code/
WORKDIR /syncx/code

# 离线烘焙 MediaPipe FaceLandmarker 模型（避免首次运行联网下载）
# 若构建机无外网，可事先把 face_landmarker.task 放到 build 上下文再 COPY。
RUN mkdir -p /models && \
    curl -fsSL -o /models/face_landmarker.task \
        https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task || \
    echo "[WARN] 无法下载 FaceLandmarker 模型，运行时需联网或手动挂载"

# HF 缓存与数据挂载点（运行时由 -v 提供）
ENV HF_HOME=/root/autodl-tmp/SyncX/hfcache \
    SYNCX_FACE_LANDMARKER_MODEL=/models/face_landmarker.task

CMD ["bash"]
