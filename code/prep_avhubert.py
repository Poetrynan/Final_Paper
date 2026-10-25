#!/usr/bin/env python
"""
prep_avhubert.py — 换载体实验 B：把现有 prep 缓存的 `audio_seq` 替换为
                    AV-HuBERT 联合 token（硬标签/identity/source_id 从参考树继承）
================================================================================
背景/动机
--------
实验 A/B 已诊断：分离载体（CLIP视觉 + wav2vec2 音频）在 6 个商用生成器上
RV_av 符号乱、强可分仅 1/6（§16「特征载体探针 NOT_SEPARABLE」）→ 分离特征
不带「跨生成器一致的音画一致性」。方向 B：把一致性载体换成 AV-HuBERT。

工程要点（与 A 组干净可比的唯一正确做法）
  A 组视觉前端 = CLIP(v_clip) + MediaPipe(v_lip)。B 组**保留同一视觉前端**，
  仅切换音频上行：wav2vec2(audio_seq 768) → AV-HuBERT(audio_seq 768，联合 token)。
  因此 <data_root>/prep/ 的 .pt 缓存里，只有 audio_seq 是错的（分离开），
  其余 v_clip/v_lip/label/identity/source_id **全部复用** → 本脚本按参考树
  （旧 prep）逐 clip 重算 audio_seq，其余字段原样继承，写全新 <data_root>/prep_avh/
  整棵树。标签/身份零损失、零改动，训练/评测 pipeline 完全不用改。
  换完出问题可随时用旧 prep 树回退。

用法（云端，GPU；需先过 verify 门槛，见 cloud_settle_avhubert.sh step1）
  # 跨域评测三件套 + 训练域，各自换载体：
  python prep_avhubert.py --ref-prep $Datasets/HiFi-AVDF/prep \
      --src-root $Datasets/HiFi-AVDF --out-root $Datasets/HiFi-AVDF/prep_avh \
      --num-shards 8 --shard-id 0
  python prep_avhubert.py --ref-prep $Datasets/Deepfake-Eval-2024/prep/test \
      --src-root $Datasets/Deepfake-Eval-2024 \
      --out-root $Datasets/Deepfake-Eval-2024/prep_avh_test --num-shards 8 --shard-id 0
  ...
  训练域（DeepSpeak）与 VoxCeleb2（Stage-1 real_SSL 真实域）同理。

定位源视频的约定（各数据集其目录布局，下列规则已覆盖 HiFi/DFE/DeepSpeak/Vox）：
  1) 参考缓存路径形如  <ref>/<gen>/<stem>/cache.pt（HiFi 风格：identity=生成器）
       → 源视频 = <src-root>/<gen>/<stem>.mp4
  2) 或形如            <ref>/<stem>.pt （扁平原声 + train/val/test 仍采 prep 约定）
       → 源视频 = 在 <src-root> 下按 <stem>.mp4 全局递归定位（含 train/val/test 子目录）
  找不到源视频 → 打印并跳过该 clip（manifest 记录），不假装成功。
"""
import os, sys, glob, shutil, argparse, time, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from common import (
    sample_frames, clip_center_window, video_to_wav, load_audio_tensor,
    write_manifest, log_progress,
)

SR = 16000
TARGET_FRAMES = 100
WINDOW_SEC = 4.0
WINDOW_FPS = 25.0
AVH_DIM = 768   # AV-HuBERT base 经 post_extract_proj 后的 token 维


def np_stack(xs):
    return np.stack([np.asarray(x) for x in xs], axis=0)


def _upsample_tokens(x, Lout):
    """把 token 序列 (T, D) 线性内插到 (Lout, D)。用 numpy 逐列插值，稳定无依赖。"""
    x = np.asarray(x, np.float32)
    T, D = x.shape
    if T == Lout:
        return x
    if T < 1:
        raise ValueError("token 序列为空，无法内插")
    if Lout <= 0:
        raise ValueError(f"无效目标长度 {Lout}")
    # 归一化采样位置 idx = linspace(0, T-1, Lout)，逐列插值
    old_x = np.linspace(0, T - 1, T)
    new_x = np.linspace(0, T - 1, Lout)
    out = np.empty((Lout, D), dtype=np.float32)
    for d in range(D):
        out[:, d] = np.interp(new_x, old_x, x[:, d])
    return out


def _ref_items(ref_prep):
    """遍历参考 prep 树的所有 .pt 缓存，返回 [{path, stem, identity(目录), kind}]。
    kind='nested' 表示 <ref>/<gen>/<stem>/cache.pt；'flat' 表示 <ref>/<stem>.pt。"""
    items = []
    for p in sorted(glob.glob(os.path.join(ref_prep, "**", "cache.pt"), recursive=True)):
        parent = os.path.basename(os.path.dirname(p))       # = stem
        gen = os.path.basename(os.path.dirname(os.path.dirname(p)))  # = identity/生成器
        items.append({"path": p, "stem": parent, "identity": gen, "kind": "nested"})
    for p in sorted(glob.glob(os.path.join(ref_prep, "*.pt"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        if os.path.dirname(p) == ref_prep:
            items.append({"path": p, "stem": stem, "identity": None, "kind": "flat",
                          "parent_dir": os.path.dirname(p)})
    return items


def _locate_source(src_root, item):
    """按参考缓存约定反查源视频绝对路径；找不到返回 None。"""
    exts = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    if item["kind"] == "nested" and item.get("identity"):
        for e in exts:
            cand = os.path.join(src_root, item["identity"], item["stem"] + e)
            if os.path.exists(cand):
                return cand
    # flat / 兜底：全局按 stem 找
    hits = [p for p in glob.glob(os.path.join(src_root, "**", item["stem"] + "*.mp4"),
                                 recursive=True)
            if p.endswith(tuple(exts)) and os.path.basename(p).startswith(item["stem"] + ".")]
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser(description="Prep AV-HuBERT 换载体（audio_seq 替换）")
    ap.add_argument("--ref-prep", required=True, help="旧 prep 参考树（含 label/identity）")
    ap.add_argument("--src-root", required=True, help="源视频根目录")
    ap.add_argument("--out-root", required=True, help="输出 prep_avh 树")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--device", default=None, help="默认取 SYNCX_DEVICE，未设则 cuda")
    ap.add_argument("--limit", type=int, default=0,
                    help=">0 时仅处理前 N 个归属本分片的 clip（用于 smoke 门槛验证）")
    ap.add_argument("--strict", action="store_true",
                    help="门槛模式：前端/联合前向任何异常直接 raise 退出，不吞成 skip")
    args = ap.parse_args()
    if not (args.num_shards >= 1 and 0 <= args.shard_id < args.num_shards):
        ap.error("--shard-id 必须在 [0,num-shards)")

    device = args.device or os.environ.get("SYNCX_DEVICE", "cuda")
    from avhubert_backbone import extract_avhubert_seq as extract_avh

    items = _ref_items(args.ref_prep)
    manifest = []
    t0 = time.time(); done = skipped = used = 0
    owned = sum(1 for gi in range(len(items)) if gi % args.num_shards == args.shard_id)
    n_processed = 0
    for gi, item in enumerate(items):
        if gi % args.num_shards != args.shard_id:
            continue
        if args.limit and n_processed >= args.limit:
            break                                   # smoke：只处理前 N 个
        n_processed += 1
        # 读参考缓存：继承 label/identity/source_id + 旧 v_clip/v_lip
        try:
            ref = torch.load(item["path"], map_location="cpu", weights_only=True)
        except Exception as e:
            if args.strict:
                raise
            print(f"  [skip] 参考缓存读取失败 {item['path']}: {e}", flush=True)
            skipped += 1
            continue
        if ref.get("label", -2) < 0:
            skipped += 1
            continue                      # 参考树里本就是坏样本，跳过

        src = _locate_source(args.src_root, item)
        if src is None:
            if args.strict:
                raise FileNotFoundError(f"找不到源视频 {item['stem']}（{item['path']}）")
            print(f"  [skip] 找不到源视频 {item['stem']}（{item['path']}）", flush=True)
            skipped += 1
            continue

        out_dir = os.path.join(args.out_root, item["stem"])
        if os.path.exists(os.path.join(out_dir, "cache.pt")):
            skipped += 1
            continue                      # 断点续跑

        # 重新抽帧 + 音频（与 prep_hifi 同流程），再换 audio_seq
        ss, win_dur = clip_center_window(src, WINDOW_SEC, WINDOW_FPS)
        frames = sample_frames(src, target_count=TARGET_FRAMES,
                               window_sec=WINDOW_SEC, window_fps=WINDOW_FPS)
        os.makedirs(out_dir, exist_ok=True)
        wav = os.path.join(out_dir, "_avh_audio.wav")
        ok_wav = video_to_wav(src, wav, sr=SR, ss=ss, td=win_dur)
        if not frames or not ok_wav:
            skipped += 1
            continue
        wav_1t = np.squeeze(np.asarray(load_audio_tensor(wav, sr=SR), np.float32))

        try:
            avh = extract_avh(frames, wav_1t, device=device)      # (T_avh,768)|None
        except Exception as e:
            # 前端/联合前向异常：strict(门槛) 时直接暴露出错，否则跳过该样本
            if args.strict:
                raise
            print(f"  [AVH] fail {item['stem']}: {e}", flush=True)
            skipped += 1
            continue
        if avh is None or avh.ndim != 2 or avh.shape[1] != AVH_DIM:
            if args.strict:
                raise RuntimeError(f"AV-HuBERT 输出形状异常 {item['stem']}: "
                                   f"shape={None if avh is None else avh.shape}")
            print(f"  [AVH] 无人脸/维度不符 {item['stem']} shape={None if avh is None else avh.shape}", flush=True)
            skipped += 1
            continue

        # AV-HuBERT 输出约 100 帧(25fps×4s)，而 dataset 固定 AUDIO_LEN=200；
        # 线性内插到 200 帧保持 50Hz 对齐，避免 _fix_len padding 出的静音段
        # 污染 LCMA 的时序注意力（resample_temporal 到 T=50）。
        avh = _upsample_tokens(avh, 200)          # (200, 768)

        # 写新缓存：audio_seq=AV-HuBERT token，其余字段从参考树原样继承
        payload = {
            "audio_seq": torch.as_tensor(avh).half(),
            "v_clip": ref["v_clip"],
            "v_lip": ref["v_lip"],
            "label": int(ref["label"]),
            "identity": str(ref.get("identity", item.get("identity", ""))),
            "source_id": str(ref.get("source_id", "")),
        }
        cpath = os.path.join(out_dir, "cache.pt")
        torch.save(payload, cpath)
        hascrops = ref.get("v_crops") is not None
        if os.path.exists(wav):
            try: os.remove(wav)
            except OSError: pass
        manifest.append({"stem": item["stem"], "identity": payload["identity"],
                         "label": payload["label"]})
        used += 1
        done += 1
        log_progress(f"[avh][shard{args.shard_id}]", done, owned, t0)

    mname = f"manifest_shard{args.shard_id}.jsonl" if args.num_shards > 1 else "manifest.jsonl"
    write_manifest(manifest, args.out_root, mname)
    print(f"[avh] rebuilt-ready={used}  skipped={skipped} -> {args.out_root}")


if __name__ == "__main__":
    main()