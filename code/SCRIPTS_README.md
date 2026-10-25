# Sync-X 数据预处理脚本（standby）

> 这些脚本**已实现并可运行**——编码器（CLIP ViT-B/16 / wav2vec2-base-960h / MediaPipe FaceMesh）已接入 `common.py`，无需再接线。
> 等对应数据集下载完成 / 授权批准后再正式跑全量抽取；当前已通过 `smoke_test.py` 与 `verify_fp16.py` 验证。

## 依赖（跑特征前才需要）
```bash
pip install opencv-python torch torchaudio transformers mediapipe
# 系统还需 ffmpeg（抽音频/trim 静音用到）
```

## 文件
| 脚本 | 对应数据集 | 职责 | 是否需全量下载 |
|---|---|---|---|
| `common.py` | — | 共享：解包/抽帧/音频/缓存打包；特征编码器已落地（CLIP/wav2vec2/MediaPipe） | — |
| `prep_wilddeepfake.py` | WildDeepfake (test) | 解包 tar → 抽帧 → 特征 → 缓存 | ❌ 只 test 划分 (~9.25GB) |
| `prep_voxceleb2.py` | VoxCeleb2 (500 子集) | 抽帧 → 特征 → 缓存 | ❌ 只 500 说话人 (~25GB 落盘) |
| `prep_fakeavceleb_b6.py` | FakeAVCeleb | B6 leading-silence trim（严格 Smeu：τ=5e-4 动态检测，仅裁 leading，1/FPS 对齐） | ❌ 只需 B6 所需 clips（全集也不大） |
| `prep_deepspeak.py` | DeepSpeak v1.1 | 主训练集预处理（train/val/test 官方划分） | ✅ **需全量** |
| `prep_eval2024.py` | Deepfake-Eval-2024 | 跨域主评测预处理 | ❌ 只 test 划分 |
| `prep_celebdf.py` | Celeb-DF-v2 | 跨域评测预处理 | ❌ 只 test 划分 |
| `prep_diffusion.py` | 扩散基准 (LivePortrait/Hallo2) | 授权后预处理 | ⚠️ 按授权范围/子集 |

## 缓存格式（与指南 §B.4 一致）
每个视频 → `cache.pt`，含：
- `audio_seq`: wav2vec2 → R^{T_a×768}
- `v_clip`: CLIP ViT-B/16 (224×224) → 512-D/帧
- `v_lip`: MediaPipe 口周点(64 点: FACEMESH_LIPS 40 唇轮廓 + 24 周围) → 展平 R^192/帧
- `label` (0 真 / 1 伪), `identity`

## 运行顺序建议
1. 先跑 `common.py` 确认导入 OK。
2. 下载完成的集（如 WildDeepfake test / VoxCeleb2 子集）即可跑对应 prep。
3. 特征提取已可用；数据集就位后直接跑对应 prep，产出 `.pt` 缓存（续跑逻辑跳过已完成项）。

## 编码器（已落地）
`common.py` 已实现 `encode_clip` / `_AudioEncoder.encode` / MediaPipe 唇点抽取，**无需再接线**。冻结骨干默认 FP16 推理（`SYNCX_INFER_FP16=0` 回退 fp32）；音频后端缺失时回退标准库 `wave` 读 PCM WAV。
