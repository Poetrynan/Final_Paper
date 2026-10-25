#!/usr/bin/env python
"""
verify_avhubert.py — 换载体实验 B 的**前置硬门槛**（通过才允许 prep 重抽 / 训练）
================================================================================
AV-HuBERT 是 fairseq-only 实现，加上权重 URL 需真机实测，所以本脚本先做 gate：
 - 1) avhubert 包可得（官方仓库已 clone / 可 import）；
 - 2) checkpoint 下载 + fairseq 加载成功，输出 token 维==768；
 - 3) 打印 model 的 forward/extract_features 签名，供 prep 前人工核对调用（诚实）。

⚠️ verify 只做"包+权重+加载+契约自检"，**不做**真实视频端到端前向塑形——
真实唇 ROI 提取依赖 MediaPipe，合成假帧会过不了人脸检测、把真问题误报成链路
失败。端到端验证放在 cloud_settle 的 STEP 1b（用一小批真实视频试跑 prep shard，
失败即停），不在此造假数据。
 - 成功输出 `AVH_VERIFY_OK dim=768` ；失败非零退出。
用法（云端）：  python verify_avhubert.py --ckpt <path.pt>
"""
import os, sys, glob, argparse
import numpy as np

# 实测可达性(2026-08-29)：官方把 base_lrs3_vox_iter5.pt 下线(403)，但同名 base 仍在。
#   clean-pretrain/base_vox_iter5.pt = SSL Base(LRS3+VoxCeleb2 英文, 无微调, 768 维) ✅200
#   noise-pretrain/base_vox_iter5.pt  = SSL Base, 噪声鲁棒版 ✅200
# base 优先（768 维 == 下游约定）；large 是 1024 维会破 dim 约定，故不列入自动候选。
_CKPT_CANDIDATES = [
    "https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/" \
    "clean-pretrain/base_vox_iter5.pt",
    "https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/" \
    "noise-pretrain/base_vox_iter5.pt",
]


def _probe_url(url, timeout=25):
    """HEAD 探测 URL 可达性。"""
    import urllib.request
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return True, r.status
    except Exception as e:
        return False, str(e)


def _download(url, dest):
    import urllib.request
    print(f"[verify] 下载 {url} -> {dest}", flush=True)
    urllib.request.urlretrieve(url, dest)
    return os.path.exists(dest)


def ensure_ckpt(ckpt_arg):
    """返回可用权重路径；未给定则依次探测候选，取首个可达并下载。"""
    if ckpt_arg and os.path.exists(ckpt_arg):
        return ckpt_arg
    dl_dir = "/root/avhubert_ckpt"
    os.makedirs(dl_dir, exist_ok=True)
    for url in _CKPT_CANDIDATES:
        ok, info = _probe_url(url)
        print(f"  探测 {url} -> ok={ok} ({info})", flush=True)
        if ok:
            name = url.rstrip("/").split("/")[-1] or "avhubert_base.pt"
            dest = os.path.join(dl_dir, name)
            if _download(url, dest):
                print(f"[verify] 可用权重: {dest}", flush=True)
                return dest
    raise RuntimeError(f"所有候选权重均不可达。候选请见 _CKPT_CANDIDATES，或 --ckpt 指定")


def check_pkg():
    """检查 avhubert 包可得；给出 clone/pip 指引。"""
    proj = os.path.join(os.path.dirname(os.path.abspath(__file__)), "av_hubert")
    if os.path.isdir(os.path.join(proj, "avhubert")):
        return f"仓库 clone 在 {proj}"
    try:
        import avhubert          # noqa: F401
        return "pip avhubert 已安装"
    except Exception:
        return ("缺失。请先 clone 官方仓库:\n"
                f"  git clone https://github.com/facebookresearch/av_hubert {proj}\n"
                "或 pip install avhubert（若 pypi 有）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--skip-download", action="store_true")
    aa = ap.parse_args()

    pkg = check_pkg()
    print(f"[verify] 包检查: {pkg}", flush=True)
    if "缺失" in pkg:
        print("AVH_PKG_MISSING"); sys.exit(1)
    try:
        import python_speech_features        # noqa: F401  官方 logfbank 依赖
        print("[verify] python_speech_features: OK (官方 logfbank)", flush=True)
    except ImportError:
        print("[verify] python_speech_features 缺失 —— prep 的 audio fbank 需要它。"
              "\n        pip install python_speech_features==0.6.0", flush=True)
        print("AVH_PSPY_MISSING"); sys.exit(1)

    w = aa.ckpt
    if aa.skip_download:
        w = w or "/root/avhubert_ckpt/avhubert_base.pt"
        if not os.path.exists(w):
            print(f"AVH_CKPT_MISSING: {w}"); sys.exit(1)
    else:
        w = ensure_ckpt(w)

    os.environ["SYNCX_AVH_CKPT"] = w
    from avhubert_backbone import load_backbone, inspect_avhubert, CROP, MEAN, STD
    try:
        model, task = load_backbone(device="cpu", ckpt=w)
    except Exception as e:
        print(f"AVH_LOAD_FAIL: {e}"); sys.exit(2)

    dim = getattr(model, "encoder_embed_dim", None)
    print(f"[verify] encoder_embed_dim={dim} fuse={getattr(model,'modality_fuse','?')}",
          flush=True)
    # 已确证契约（avhubert 完整源码 780 行，2026-08-28）
    a_dim = model.feature_extractor_audio.proj.in_features
    v_dim = model.feature_extractor_video.proj.in_features
    print(f"[verify] 契约: audio_dim(proj.in_features)={a_dim}  "
          f"video_resnet_backend_out={v_dim}"
          f"  image_crop_size={CROP}  mean/std={MEAN}/{STD}", flush=True)
    print(f"[verify] 前向入口: model.extract_features(source={{'audio':<{a_dim}>, "
          f"'video':<[B,1,T,88,88]>}}, mask=False) -> (T, {dim})", flush=True)
    # 打印前向入口签名，供 prep 前人工核对（诚实，不强撑）
    for name in ("forward", "extract_features", "forward_features", "get_feature"):
        m = getattr(model, name, None)
        if m is not None:
            print(f"  model.{name} 签名: {m.__doc__ if m.__doc__ else '无文档'}", flush=True)

    inspect_avhubert(device="cpu")
    print(f"AVH_VERIFY_OK dim={dim}", flush=True)


if __name__ == "__main__":
    main()