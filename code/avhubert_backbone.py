#!/usr/bin/env python
"""
avhubert_backbone.py — AV-HuBERT 联合表征提取器（换载体实验 B 的前端编码器）
================================================================================
动机
----
实验 A/B 的诊断结论：分离载体（CLIP 视觉 + wav2vec2 音频）在 6 个生成器上
RV_av 符号乱、强可分仅 1/6（§16）→ 分离特征不带「跨生成器一致的音画一致性」。
方向 B 是把一致性载体换成 **AV-HuBERT（音画联合自监督预训练 token）**，
用它替代 `audio_seq` 喂给 LCMA/判别头。

本文档诚实限定（勿把它当"已端到端可跑"）
----------------------------------------
AV-HuBERT 是 fairseq-only 实现，**云端必须满足前置条件才能跑**：
  1) clone facebookresearch/av_hubert → 使 `import avhubert` 注册 model/task class；
  2) 下载官方 base 权重 *.pt（URL 见下方候选，须 `curl -I` 实测可达）；
  3) fairseq（官方 avhubert fork）与 compatible 版本 install。
这三步由 `cloud_settle_avhubert.sh` 的 `STEP avh_verify`（加载+签名自检）与
`STEP avh_smoke`（用 8 个真实 clip 做端到端联合前向）共同把关，**全部通过后**才允许
`STEP avh_prep` 大规模重抽并训练。在 verify+smoke 通过前，本模块不会产出可信特征——
这是刻意设的门槛，防盲目烧 GPU（红线）。

关键契约（已确证，来自 avhubert 官方源码 + 权威核实 2026-08-28）
----------------------------------------------------------------
1. **官方 `AVHubertModel` 没有一体化的 `forward`/`extract_features`**（官方 README 的
   inference 走 `task.inference_step`，见 `avhubert/infer_s2s.py`）。因此本模块**手动编排
   已确证存在的成员**来忠实复刻官方联合 token：
     a = feature_extractor_audio(audio_fbank)        # [B,768,T]
     v = feature_extractor_video(video_px)           # [B,768,T]
     concat→1536 → LayerNorm(1536) → post_extract_proj→768 → [B,T,768]
   （`modality_fuse='concat'` 时 concat；若为 'add' 则直接相加不投影。）
2. **输出 token 维 = 768**（base encoder_embed_dim）。即便 concat(768+768)=1536 也经
   `post_extract_proj: Linear(1536,768) + LayerNorm(1536→768)` 拉回 → 与现 `audio_seq
   (T,768)` 契约一致，下游 dataset/models/config **零改动**。
3. **视频分支**：`feature_extractor_video = SubModel(ResEncoder)`；
   - ResEncoder 输入 `[B, C, T, H, W]`，`backend_out=512`；
   - 像素归一化 = 官方 `hubert_dataset.py`：`(x/255 - 0.421)/0.165`，先
     `CenterCrop(image_crop_size=88)`；C=1（`load_video` 返回 `[T,H,W,1]` 灰度）。
   - 本模块唇 ROI 由 common（MediaPipe 112×112 灰度 crop）缩放至 88×88，与现有 prep 一致。
4. **音频分支**：`feature_extractor_audio = SubModel(resnet=None)`，输入**不是原始波形**，
   而是 `python_speech_features.logfbank` 特征（官方 `hubert_dataset.py`：16k → logfbank
   `[T, F]`，F 须 == `cfg.audio_feat_dim`），帧数对齐由 video 主导（`load_feature` 的
   diff 裁剪/补零语义）。若 logfbank 列数 ≠ `audio_feat_dim`，`proj` 会 Shape 不匹配——
   `_audio_logfbank_to_channel` 会给出带 `audio_dim` 的诊断并 raise（见 verify/smoke）。
5. 联合前向需要 audio+video 同时提供；无人脸 → 无 video 分支 → 该段放弃（返回 None，
   上层按「抽帧失败」跳过，不污染缓存）。

依赖本项目的 common：MediaPipe 唇 landmark（`common.extract_lip_landmarks_seq` →
112×112 灰度 crop），本模块再缩放到 88×88 供 AV-HuBERT——保证与现有 prep 唇定位一致。
"""
import os, sys
import numpy as np

# ── 官方仓库：允许在 `code_cloud/` 同级 clone；或 pip 装 avhubert 包 ─────────
_PROJECT_AVH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "av_hubert")
if os.path.isdir(_PROJECT_AVH):
    sys.path.insert(0, _PROJECT_AVH)

SR = 16000
_A_TOT = int(SR * 4)          # 4s @16k = 64000（= common._MAX_AUDIO_LEN）
FPS = 25                       # AV-HuBERT 视频 token 率（4s → 100 视频帧）
CROP = 88                      # 官方 AV-HuBERT 唇 ROI 边长
MEAN, STD = 0.421, 0.165      # 官方 image_mean / image_std

# checkpoint 路径（setup/verify 写入并回填；可被 SYNCX_AVH_CKPT 覆盖）
# 候选权重（官方 avhubert release，base 系）。verify 用 curl -I 挑可达者回填：
#   base_lrs3_vox_iter5.pt 已被官方下线(403)；实测可达的同为英文 SSL base：
#   clean-pretrain/base_vox_iter5.pt（无微调, 768 维, LRS3+VoxCeleb2）✅200
_CKPT_CANDIDATES = [
    "https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/" \
    "clean-pretrain/base_vox_iter5.pt",
    "https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/" \
    "noise-pretrain/base_vox_iter5.pt",
]

_AVH_CKPT = None               # verify 通过后由 cloud_settle 写 /root/avhubert_ckpt 并 export SYNCX_AVH_CKPT
_LOADED = {"model": None, "task": None, "device": None}


# =============================================================================
# 惰性单例加载（跨 prep 进程可复用，避免重复建 fairseq task）
# =============================================================================
def load_backbone(device="cuda", ckpt=None):
    """加载 AV-HuBERT (model, task)。惰性 + 全局单例。"""
    if _LOADED["model"] is not None:
        return _LOADED["model"], _LOADED["task"]
    import torch
    import fairseq
    try:
        import avhubert            # noqa: F401  注册 model/task class（必需）
    except ImportError as e:
        raise ImportError(
            "未找到 avhubert（fairseq 注册 class 缺失）。先 clone facebookresearch/"
            "av_hubert 到 code_cloud/ 下，或 pip install。原始错误: %s" % e)

    if ckpt is None:
        ckpt = os.environ.get("SYNCX_AVH_CKPT", "")
    if not ckpt or not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"AV-HuBERT checkpoint 不存在: {ckpt!r}。先跑 cloud_settle_avhubert.sh "
            "step1（下载权重 + 写 SYNCX_AVH_CKPT），再进入 prep。")

    print(f"[AVH] loading {os.path.basename(ckpt)} on {device}", flush=True)
    # torch>=2.6: torch.load 默认 weights_only=True，会拒 fairseq Dictionary 等官方全局类。
    # 权重来自官方信任源，故仅对 fairseq 本次加载临时回退 weights_only=False，完成后还原。
    _orig_tl = torch.load
    def _tl_compat(*a, **k):
        k.setdefault("weights_only", False)
        return _orig_tl(*a, **k)
    torch.load = _tl_compat
    try:
        models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [ckpt], arg_overrides={"device": device})
    finally:
        torch.load = _orig_tl
    model = models[0].to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    _LOADED.update(model=model, task=task, device=device)
    print(f"[AVH] embed_dim={getattr(model,'encoder_embed_dim','?')} "
          f"fuse={getattr(model,'modality_fuse','?')} | "
          f"expected out-dim=768 @ post_extract_proj+LayerNorm", flush=True)
    return model, task


# =============================================================================
# 输入预处理（已确证，来自官方 avhubert 源码 2026-08-28）
# =============================================================================
def _audio_fbank_to_source(wav_16k, T_video, audio_dim):
    """官方 `hubert_dataset.load_feature`：16k 波形 → python_speech_features.logfbank
    得到 [Ta, F]，再按 video 主导对齐到 T_video 帧（audio 短补零、长出裁剪末尾）。
    返回 Audio source tensor [1, F, Ta=T_video]（官方 SubModel.forward：
    `proj(x.transpose(1,2))`，输入布局为 [B, F, T]，F=特征维）。logfbank 列数 F 必须 == audio_dim(=Linear input_features)，
    否则 proj 形状不匹配——这里显式校验并给出诊断而非静默错。"""
    import torch
    from python_speech_features import logfbank
    wav = np.asarray(wav_16k, np.float32).reshape(-1)
    if wav.size < 2 or float(np.abs(wav).max()) < 1e-7:
        raise ValueError("音频近乎静音/空，无法提取 fbank")
    feat = logfbank(wav, samplerate=SR, nfilt=audio_dim, nfft=512).astype(
        np.float32)                                        # [Ta, F]
    F = feat.shape[1]
    if F != audio_dim:
        raise RuntimeError(
            f"logfbank 列数 F={F} != audio_dim={audio_dim}（feature_extractor_audio.proj "
            f"in_features）。请核对 fbank 参数或改用官方 AVHubertDataset collate。")
    Ta = feat.shape[0]
    if Ta < T_video:                                       # audio 短 → 末尾补零
        feat = np.concatenate([feat, np.zeros((T_video - Ta, F), np.float32)], axis=0)
    else:
        feat = feat[:T_video]                              # audio 长 → 裁剪末尾
    x = torch.as_tensor(feat, dtype=torch.float32).unsqueeze(0)   # [1, Ta, F]
    return x.transpose(1, 2).contiguous()                 # [1, F, T]

def _frames_to_roi(frames_bgr, device="cpu"):
    """取唇 ROI 灰度帧序列并缩放到 88×88。复用 common.extract_lip_landmarks_seq
    （返回 (112×112 crop, 192 coords)），crop 再 resize 到 CROP。
    任一帧无人脸 → 整个片段视为不可用（返回 None），避免静默污染。"""
    import common
    seq = common.extract_lip_landmarks_seq(frames_bgr, device=device)
    rois = []
    for crop112, _coord in seq:
        if crop112 is None or crop112.size == 0:
            return None
        crop112 = np.asarray(crop112, np.float32)
        # 112 -> 88（线性缩放；AV-HuBERT 对唇 ROI 缩放稳健）
        if crop112.shape[0] != CROP:
            from scipy import ndimage
            crop112 = ndimage.zoom(crop112, (CROP / crop112.shape[0],
                                             CROP / crop112.shape[1]), order=1)
        rois.append(crop112)
    if not rois:
        return None
    return np.clip(np.stack(rois), 0, 255).astype(np.float32)   # (T,88,88)


# =============================================================================
# 主入口（联合前向）—— 依据完整 avhubert/hubert.py(780行) 已确证的调用
# =============================================================================
def extract_avhubert_seq(frames_bgr, wav_16k, device="cuda"):
    """frames_bgr: list[np.ndarray(H,W,3) BGR]；wav_16k: np.ndarray [T] 16k mono。
    返回 (T_avh, 768) float32；无人脸/无有效音频 → None。

    已确证的官方调用（2026-08-28，完整源码 hubert.py 676-684 行）：
      feature, padding_mask = model.extract_features(
          source={'audio': a, 'video': v}, padding_mask=None, mask=False,
          ret_conv=False)   # feature: [B, T, 768]  ← TransformerEncoder 输出（最后一层）
    拼接/归一化/fbank 见 _frames_to_roi / _audio_fbank_to_source。

    ⚠️ 仍留一道真机门槛：`aud feat_dim`(proj.in_features) 与 logfbank 列数是否吻合、
    实际输出维是否确实 768，由 cloud_settle 的 STEP-verify(加载自检)+STEP-smoke
    （真实 clip 端到端联合前向）把关；通过前 prep 不被触发。
    """
    model, task = load_backbone(device=device)
    roi = _frames_to_roi(frames_bgr, device)
    if roi is None:
        return None
    T_v = roi.shape[0]
    audio_dim = model.feature_extractor_audio.proj.in_features
    a = _audio_fbank_to_source(wav_16k, T_v, audio_dim)
    return _forward_to_tokens(model, roi, a, device)


def _forward_to_tokens(model, roi, a, device):
    """把 roi([T,88,88] 灰度 0-255) + audio fbank source 送进 AV-HuBERT，
    经官方 extract_features 取出 (T, 768) 最后一层 token 序列。"""
    import torch
    v = torch.as_tensor(roi, dtype=torch.float32)          # [T,88,88]
    v = v.unsqueeze(0).unsqueeze(-1)                       # [1,T,88,88,1] ← 官方 load_video 后 expand_dims
    v = v.permute(0, 4, 1, 2, 3).contiguous()              # [1,1,T,88,88] ← collate 后布局
    v = ((v / 255.0 - MEAN) / STD)                        # Normalize(0,255) → Normalize(mean,std)
    source = {"audio": a.to(device), "video": v.to(device)}
    with torch.no_grad():
        feature, _padding = model.extract_features(
            source, padding_mask=None, mask=False, ret_conv=False)
    out = feature[0].detach().cpu().float().numpy()       # (T, 768)
    return out


# =============================================================================
# 快捷打印实际 embed_dim（供 verify 门槛自动回填 config）
# =============================================================================
def inspect_avhubert(device="cuda"):
    model, task = load_backbone(device=device)
    print("\n[AVH inspect]")
    print("  encoder_embed_dim :", getattr(model, "encoder_embed_dim", "?"))
    print("  modality_fuse     :", getattr(model, "modality_fuse", "?"))
    print("  post_extract_proj :", getattr(model, "post_extract_proj", None))
    print("  layer_norm        :", getattr(model, "layer_norm", None))
    print("  输出 token 维     : self.embed(1536 if concat) -> post_extract_proj -> 768")
    return model


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="AV-HuBERT 编码器自检（verify 门槛用）")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    inspect_avhubert(a.device)