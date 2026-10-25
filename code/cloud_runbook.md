# Sync-X 云端运行手册（cloud_runbook）

> 目标：把 Sync-X 的特征抽取 + 两阶段训练 + 评测搬到云 GPU 实例上跑，
> 且不丢训练成果（支持竞价实例中断续跑）、环境可复现。

---

## 0. 选择平台

### 算力需求
- GPU 显存：**16 GB 以上**（CLIP-ViT-B/16 + wav2vec2-base-960h 冻结，仅训小 head；峰值 ~8-12 GB，T4 即够）
- 总 GPU 时间：**抽取 ~35h + 训练 ~20h ≈ 55 GPU-hours**
- 不需要 A100/H100——T4/P100/RTX3090 就够

### 免费档（零成本，靠分片 + 断点续跑）

| 平台 | 免费额度 | 单次会话上限 | GPU | 持久盘 | 需卡？ | 适用 |
|---|---|---|---|---|---|---|
| **Lightning AI** ⭐ | 80 GPU-h/月 | 4h 后自动重启 | T4/L4/A10 | 50GB | 手机验证 | 训练（持久 IDE + `--resume` 自动续） |
| **Kaggle** | 30 GPU-h/周 | 9h | P100/T4×2 | 20GB | 手机（+86 支持） | 抽取分片（配额最稳） |
| **Modal** | $30/月≈50 T4-h | Serverless（无硬上限） | T4–A100 | Volume 挂载 | 无 | 抽取分片（按秒计费，缩到零） |

**免费组合拳（三平台并行 ≈ ¥0）**：
- 抽取 35h 拆给 Kaggle (9h/段) + Lightning (4h/段) + Modal (per-clip 函数) → **墙钟 3-7 天**
- 训练 20h 放 Lightning，`--resume` 每 4h 自动续 → **墙钟 1-2 天**
- ⚠️ 真正瓶颈是**数据上传**（免费档上传大模型较慢），非 GPU

**不推荐/已死的免费档**：
- ❌ **Colab 免费**：GPU 不保证给、90min 闲置踢、最不稳
- ❌ **Hugging Face ZeroGPU**：仅 5 min/天(免费)、只能做 demo、不能训练
- ❌ **Paperspace Gradient**：被收购后免费缩水为 M4000 8GB（太弱）、需信用卡
- ❌ **SageMaker Studio Lab**：2026-07-30 关闭新注册

### 便宜付费档（真·不中断，一键上传、连续跑完）

| 平台 | 价格 (RTX3090) | 价格 (A100) | 持久盘 | 说明 |
|---|---|---|---|---|
| **AutoDL** ⭐ | ¥0.78-1.98/h（学生认证 -33%≈¥1.2） | ¥3.98-6/h | ✅ 关机停计费 | 国内首推，数据直传 |
| **Vast.ai** | $0.07-0.14/h (spot) | $0.79-0.98 (spot) | 选 ≥97% 评分主机 | 全球最便宜，可靠性参差 |
| **RunPod** | 4090 $0.24-0.44/h | A100 $1.39/h | ✅ 持久盘 | 稳定、UI 友好 |
| **Thunder Compute** | RTX A6000 $0.35/h | A100 $2.19/h | ✅ 按分钟 | 新晋、低延迟 |

**推荐 AutoDL RTX3090 学生价 ¥1.2/h**：55h 全包 ≈ **¥66**，一条命令从头跑到尾、不用分片、不用切平台。

---

## 1. 前置条件
- 云 GPU 实例（NVIDIA + CUDA 12.1 驱动）
- **持久卷/云盘**：存数据集、`.pt` 缓存、ckpt、logs。⚠️ 绝不放容器临时盘（实例回收即丢）
- 竞价/抢占实例：**必须**启用 §4 断点续跑（免费档同理——靠 `--resume` 跨会话拼进度）

---

## 2. 数据集上传

```bash
# 云端 Linux 挂载一律用环境变量覆盖，无需改代码：
export SYNCX_DATA_ROOT=/root/autodl-tmp/SyncX/Datasets      # 对应本机 D:/Datasets
```

| 数据集 | 角色 | 帧数 | 大小(约) | 上传方式 |
|---|---|---|---|---|
| DeepSpeak v1.1 | 训练 | 100 | ~GB | HF Hub（门控需 `HF_TOKEN`）或本机直传 |
| VoxCeleb2 subset | Stage-1 SSL | 100 | ~GB | 本机直传 |
| Celeb-DF-v2 | 评测 | 100 | ~GB | 直传 |
| DFE2024 | 评测 | 100 | ~GB | 直传 |
| MVAD | 评测 | 100 | ~GB | 直传 |
| HiFi-AVDF | 评测 | 100 | ~GB | 直传 |
| THB | 评测 | 100 | ~GB | 直传 |
| FakeAVCeleb B6 | 捷径探针 | 100 | ~GB | 直传 |

> **AutoDL 直传**：网页上传 / `scp` 到持久数据盘，一次传完永久用。
> **免费档上传**：大模型建议先 tar.gz 打包，用平台各自的上传接口（Kaggle Dataset / Modal Volume / Lightning 文件管理）。

---

## 3. 环境

### 3.1 模型权重

| 权重 | 来源 | 云端处理 |
|---|---|---|
| MediaPipe FaceLandmarker | Google Storage | `SYNCX_FACE_LANDMARKER_MODEL` 指向 `.task`；首次运行自动下 |
| CLIP ViT-B/16 | HF Hub | 自动下；设 `HF_HOME` 到持久盘 |
| wav2vec2-base-960h | HF Hub | 同上 |

```bash
export HF_HOME=/root/autodl-tmp/SyncX/hfcache
# ⚠️ 若数据集是【直传云端】（非从 HF 拉门控 DeepSpeak 原仓），则【不需要 HF_TOKEN】。
#    CLIP/wav2vec2 是公开模型，无需 token 即可下载；若拉取慢/被墙可设镜像：
export HF_ENDPOINT=https://hf-mirror.com
# export HF_TOKEN=hf_xxx   # 仅当你要从 HF 拉【门控数据集】时才需要；本流程（直传+抽+训）不用
```

### 3.2 Docker（推荐，一键起环境）

```bash
docker build -t syncx:latest code/
docker run --gpus all --shm-size=16g \
  -e SYNCX_DATA_ROOT=/root/autodl-tmp/SyncX/Datasets \
  -e HF_HOME=/root/autodl-tmp/SyncX/hfcache \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -e SYNCX_FACE_LANDMARKER_MODEL=/models/face_landmarker.task \
  -v /your/datasets:/root/autodl-tmp/SyncX/Datasets:ro \
  -v /your/persist:/root/autodl-tmp/SyncX/persist \
  -w /syncx/code syncx:latest \
  python train_stage2.py --train_cache /root/autodl-tmp/SyncX/Datasets/.../prep/train \
    --out_ckpt /root/autodl-tmp/SyncX/persist/ckpt/syncx_best.pt --log_dir /root/autodl-tmp/SyncX/persist/logs
```

镜像含：CUDA12.1+cuDNN8、ffmpeg、torch(cu121)、transformers、mediapipe、opencv-headless、numpy/scipy，已预烘焙 FaceLandmarker 模型 + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

### 3.3 裸机 (pip)

```bash
pip install -r code/requirements.txt      # 已锁版本；云端 pip 需 CUDA 版 torch
apt-get install -y ffmpeg                 # 系统依赖
```

---

## 4. 特征抽取（~35h，全部 100 帧/4s）

```bash
export SYNCX_DATA_ROOT=/root/autodl-tmp/SyncX/Datasets
cd /syncx/code

# 训练集
python prep_deepspeak.py           # DeepSpeak v1.1, --source_id 用于 Rec C
python prep_voxceleb2.py           # Stage-1 SSL（仅真实片段）

# 评测集（全 5 个）
python prep_celebdf.py
python prep_eval2024.py
python prep_thb.py
python prep_mvad.py
python prep_hifi.py

# B6 捷径探针（评测用，不影响训练；不抽也可）
python prep_fakeavceleb_b6.py
```

- 产出：各 `<ROOT>/prep/` 下 `.pt` 扁平缓存，**192-D v_lip (64 口周点×3)**
- 旧缓存（40 点/120-D 或 240-D）**不兼容**，必须先清再用本版重抽
- ⚠️ 免费档建议**按数据集拆到不同平台会话**，每段 ≤ 会话上限、下一段随缘补

### 4.1 云上批跑前：先跑一次单视频端到端验证（必做）

`_smoke_ok.py` 只验证**代码**；若 35h 抽取在云上跑，还需验证**云环境本身**：
ffmpeg 是否装好、HF 门控 token 是否放行、CLIP/wav2vec2/MediaPipe 能否下载、
`SYNCX_DATA_ROOT` 挂载路径是否正确、持久卷是否可写。用**一个真实视频**试通，再批跑：

```bash
# 云上（进入容器 / 激活环境，先传一个视频上来）
python _verify_cloud.py "$SYNCX_DATA_ROOT/DeepSpeak_v1_1/clips/real/14-5467.mp4"
# 预期：环境变量/ffmpeg/GPU/持久卷/真实全链路（CLIP+MediaPipe+wav2vec2+pack_cache）全 PASS
```

只要这个 PASS，云环境就绪，直接跑 §4 的 8 个 prep 批抽；若 FAIL，按报错定位
（缺 ffmpeg → `apt-get install -y ffmpeg`；HF 下载失败 → 查 `HF_TOKEN`/`HF_HOME`；
门控拒绝 → 检查 DeepSpeak 数据集访问权限；落盘失败 → 查持久卷挂载/权限）。

---

## 5. 训练

```bash
# Stage-1 自监督（VoxCeleb2 真实片段）
python train_stage1.py \
    --real_cache /root/autodl-tmp/SyncX/Datasets/VoxCeleb2_subset/prep \
    --out_ckpt   /root/autodl-tmp/SyncX/persist/ckpt/stage1.pt \
    --log_dir    /root/autodl-tmp/SyncX/persist/logs \
    --epochs 40 --batch 256 --lr 1e-4 --seed 0

# Stage-2 监督（DeepSpeak）
python train_stage2.py \
    --train_cache /root/autodl-tmp/SyncX/Datasets/DeepSpeak_v1_1/clips/prep/train \
    --val_cache   /root/autodl-tmp/SyncX/Datasets/DeepSpeak_v1_1/clips/prep/test \
    --stage1_ckpt /root/autodl-tmp/SyncX/persist/ckpt/stage1.pt \
    --out_ckpt    /root/autodl-tmp/SyncX/persist/ckpt/syncx_best.pt \
    --log_dir     /root/autodl-tmp/SyncX/persist/logs \
    --epochs 80 --batch 256 --lr 1e-4 --seed 0
```

### 5.1 断点续跑（竞价实例 / 免费档会话中断救星）

每 epoch 末**原子写** `<out_ckpt>.resume`（含 模型/优化器/调度器/缩放器/epoch/步数/RNG/best_auc）。
中断后用**完全相同**的 `--out_ckpt` 重跑即自动续，最多丢 1 个 epoch。

```bash
# 中断恢复：无需额外参数，同一条命令重跑即可
python train_stage2.py ... --out_ckpt /root/autodl-tmp/SyncX/persist/ckpt/syncx_best.pt
# → "[stage2] 断点续跑：从 .../.resume 恢复，start_epoch=N"
```
- 免费档（Lightning 4h 重启 / Kaggle 9h 断连）=> 写个循环脚本自动重拉
- 强制从头：`rm <out_ckpt>.resume`

### 5.2 /dev/shm 太小（DataLoader "Bus error"）

```bash
docker run --shm-size=16g ...          # Docker 直接放大
python train_stage2.py ... --num_workers 0   # 裸机降 workers（略慢）
```

### 5.3 GPU 显存防炸

三重防线（均不影响训练精度）：
1. `--empty_cache_every 1`（默认）：每 epoch 末释放 CUDA 空闲缓存，不进热路径
2. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`：分配器防碎片 OOM
3. 每步 `del` 中间张量，缩短显存占用窗口

若仍 OOM：`--empty_cache_every 1 --batch 128`。

### 5.4 超参数搜索（可选，论文留白项）

> **原则**：论文已锁定 λ_corr=0.3 / λ_shared=0.1 / lr=1e-4 / epochs 40/80 / batch 256
> （Ch3:97 + Ch4:169-173），**不搜**；只搜论文留白项 λ_align × λ_uniform（Ch4:171-172
> "selected on val split"）+ ≥3 seeds 平均（Ch4:250）。
> 详细策略见 `HYPERPARAMETER_SEARCH_GUIDE.md`。

**前置**：Stage-1 ckpt 已训好 + DeepSpeak `prep/train`、`prep/test` 已抽取（搜索脚本调 `train_stage2.py`）。

```bash
cd /syncx/code
# Step A: 筛选 λ_align×λ_uniform（9 组合 × 20ep ≈ 2.3h）
python hyperparameter_search.py --stage loss_weights --epochs 20 \
    --stage1_ckpt /root/autodl-tmp/SyncX/persist/ckpt/stage1.pt

# Step A': 最优组合放大 80ep（用 summary.csv 的 val_auc 列选优，脚本已自动打印最佳）
python hyperparameter_search.py --stage loss_weights --epochs 80 \
    --best_lambda_align <最佳> --best_lambda_uniform <最佳> \
    --stage1_ckpt /root/autodl-tmp/SyncX/persist/ckpt/stage1.pt

# Step B: 最优配置 × 3 seeds（论文 mean±std，≈3h）
python hyperparameter_search.py --stage loss_weights --seeds "0 1 2" \
    --best_lambda_align <最佳> --best_lambda_uniform <最佳> \
    --stage1_ckpt /root/autodl-tmp/SyncX/persist/ckpt/stage1.pt
```

云上要点：
- ⚠️ 运行 `--stage training_epochs` 时**勿传 `--epochs`**（会覆盖搜索值 {40,80,120} 空转）
- 每个 (实验,seed) 独立 ckpt + `--resume` 兼容 → 竞价实例中断后重跑同命令即续
- 结果：`logs/hyperparameter_search/<stage>/summary.csv`（含 `val_auc` 列，**自动选优**）
- `--stage1_ckpt` 必须传真实路径，否则 train_stage2 静默跳过预训练、结果不可比

---

## 6. 评测与出图

```bash
# 对每个评测集跑一次 evaluate.py
python evaluate.py --ckpt /root/autodl-tmp/SyncX/persist/ckpt/syncx_best.pt \
    --eval_cache /root/autodl-tmp/SyncX/Datasets/Celeb-DF-v2/prep --log_dir /root/autodl-tmp/SyncX/persist/logs
python evaluate.py --ckpt /root/autodl-tmp/SyncX/persist/ckpt/syncx_best.pt \
    --eval_cache /root/autodl-tmp/SyncX/Datasets/Deepfake-Eval-2024/prep --log_dir /root/autodl-tmp/SyncX/persist/logs
# ... THB, MVAD, HiFi-AVDF 同理

# 出图：CPU 即可，不需 GPU / 重训 / 重评
pip install matplotlib scikit-learn
python plot_figures.py --log_dir /root/autodl-tmp/SyncX/persist/logs
```

---

## 7. 多卡（可选）

当前为**单卡**。加速方案：
- 调大 batch + 单卡（A100 80G）
- 多卡 `DistributedDataParallel`：需代码改动，按需再做
- 抽取可**按数据集分片并行到多机**，互不依赖

---

## 8. 代码改动清单

| 文件 | 改动 |
|---|---|
| `config.py` | `import os` + `DATA_ROOT = os.environ.get("SYNCX_DATA_ROOT", r"D:/Datasets")`（缺 `import os` 会启动即 NameError，已修） |
| `train_stage1.py` | `--resume` / `--num_workers` / `--prefetch_factor` / `--empty_cache_every`；原子 `.resume`；`loader_kwargs` 定义在 DataLoader 前；**删除 `build_optimizer` 死引用 import**（utils 无此符号，实际用本地 `build_optimizer_typed`；smoke 测试抓出） |
| `train_stage2.py` | 同上 + `del out,loss,...` 步末释放 |
| `utils.py` | `capture_rng` / `restore_rng` / `save_checkpoint_atomic` |
| `prep_*.py` (8 个) | `TARGET_FRAMES = 100` 统一（训练/评测一致） |
| `prep_fakeavceleb_b6.py` / `b6_std_trim.py` / `b6_leadsilence_audit.py` | 硬编码路径 → `os.path.join(DATA_ROOT, ...)` |
| `make_figs.py` / `make_extra_figs.py` | `OUTDIR` 改为 `SYNCX_FIG_OUTDIR` 环境变量可覆盖 |
| `evaluate.py` / `dump_*.py` | `torch.load(..., weights_only=False)` |
| `extract_features.py` | 已废弃，`VisualBackbone.__call__` 顶部 `raise RuntimeError` 硬拒绝（防误触 120-D） |
| `requirements.txt` | 完整训练/抽取栈 + 锁版本（含 ffmpeg 系统依赖说明） |
| `Dockerfile` | CUDA12.1+cuDNN8 + ffmpeg + FaceLandmarker 离线 + env 变量 |

**契约验证**（本机运行，跑通再上云）：
```bash
python _smoke_ok.py             # ⭐ 一键体检（最快，7 组检查，无需 GPU/模型）
python _smoke_train.py          # ⭐ 训练链路冒烟（CPU：SyncX 5 项损失 + Stage-1 SSL，各 2 步反向）
python _smoke_search.py         # ⭐ 超参搜索冒烟（组合生成/指标读取/env 透传/CSV 列，不启动训练）
python _verify_contract.py      # 192-D 维度 + 点序 + fail-fast 守卫
python _verify_extraction.py "D:/Datasets/DeepSpeak_v1_1/clips/real/14-5467.mp4"
python _verify_cloud.py "$SYNCX_DATA_ROOT/.../14-5467.mp4"   # ⭐ 云上必跑：环境+真实全链路（§4.1）
python _verify_resume.py        # 原子写 + RNG 恢复 + 续跑接续
```
> `_smoke_ok.py` 覆盖：config 可 import（曾缺 import os）、192-D 契约、
> 废弃抽取器锁死、weights_only、loader_kwargs 顺序、TARGET_FRAMES=100、
> LACA 守卫行为（120-D 抛错 / 192-D 通过）。全 PASS 即代码健康。
> `_smoke_train.py` 造 6 个契约正确的 .pt → 真实 FeatureCacheDataset →
> Stage-2（5 项损失）与 Stage-1（InfoNCE+kin+shared）各 2 步反向优化，
> 断言 loss 有限、梯度流通——证明模型装配/前向/损失/优化器全链路可跑。
