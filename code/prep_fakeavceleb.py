"""
prep_fakeavceleb.py — 为 Sync-X 评测准备 FakeAVCeleb_v1.2 特征。
============================================================================
两种模式：
  --mode normal (默认)
      从原始 FakeAVCeleb_v1.2 (4 子集) 抽特征
        -> <FakeAVCeleb_v1.2>/prep/<subset>/<stem>/cache.pt
      用于 STEP 5 跨域评测（会议 MVP 的第 3 个评测集）。
      标签：含 "Fake" 的子集(fake video 或 fake audio) -> 1(伪造)；
            RealVideo-RealAudio -> 0(真实)。
      identity：从路径中的 idXXXX 文件夹取。

  --mode b6
      读取 b6_std_trim.py 的产出（需先跑 b6_std_trim.py）：
        raw     -> <FakeAVCeleb_v1.2_split>/prep/b6_raw/<...>/cache.pt
        trimmed -> <FakeAVCeleb_v1.2_split>/prep/b6_trimmed/<...>/cache.pt
      用于 STEP 5b B6 静音捷径探针（evaluate --b6 / --b6_untrimmed）。
      标签：RVRA(全真) -> 0；RVFA/FVFA(假音频) -> 1。
      ⚠️ 若 b6_std/raw 或 b6_std/trimmed 不存在，该分支自动 SKIP（先跑 b6_std_trim.py）。

特征维度契约（与 common.py 一致，64 点唇部）：
    audio_seq (T_a, 768) | v_clip (T_v, 512) | v_lip (T_v, 192)
断点续跑：每片段产出 cache.pt 后，重启会跳过已存在的 cache.pt（os.path.exists 检查）。
分片：--num-shards N --shard-id K 可多窗口并行，各片独立续跑。
"""
import os, sys, glob, argparse, shutil, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from common import (
    iter_videos, sample_frames, clip_center_window, save_frames, video_to_wav, load_audio_tensor,
    extract_clip_frames, extract_lip_landmarks_seq, extract_audio_seq,
    pack_cache, write_manifest, log_progress,
)
from config import DATA_ROOT

ORIG = os.path.join(DATA_ROOT, "FakeAVCeleb_v1.2")
SPLIT = os.path.join(DATA_ROOT, "FakeAVCeleb_v1.2_split")
TARGET_FRAMES = 100
WINDOW_SEC = 4.0         # 中心窗时长（秒），对齐训练域窗口；短于窗的片段用全部帧
WINDOW_FPS = 25.0        # 窗内采样率
SR = 16000

# 4 原始子集
_SUBSETS = ["FakeVideo-FakeAudio", "FakeVideo-RealAudio",
            "RealVideo-FakeAudio", "RealVideo-RealAudio"]


def _label_from_subset(subset):
    return 0 if subset == "RealVideo-RealAudio" else 1


def _shard(items, num_shards, shard_id):
    if num_shards <= 1:
        return items
    return [it for i, it in enumerate(items) if i % num_shards == shard_id]


def _identity_from_rel(rel):
    """从相对路径取 idXXXX 身份（无则 unknown）。"""
    for p in rel.split(os.sep):
        if p.startswith("id"):
            return p
    return "unknown"


def _prep_one(vid, out_base, label, identity):
    """抽 1 片段特征并落盘。返回 True=新抽, False=已存在跳过, None=抽帧/抽音频失败。"""
    stem = os.path.splitext(os.path.basename(vid))[0]
    cache = os.path.join(out_base, stem, "cache.pt")
    if os.path.exists(cache):
        return False  # resume 跳过
    ss, win_dur = clip_center_window(vid, WINDOW_SEC, WINDOW_FPS)
    frames = sample_frames(vid, target_count=TARGET_FRAMES,
                           window_sec=WINDOW_SEC, window_fps=WINDOW_FPS)
    if not frames:
        return None
    save_frames(frames, os.path.join(out_base, stem, "frames"))
    wav = os.path.join(out_base, stem, "audio.wav")
    if not video_to_wav(vid, wav, sr=SR, ss=ss, td=win_dur):
        return None
    wav_t = load_audio_tensor(wav, sr=SR)
    v_clip = extract_clip_frames(frames, device=os.environ.get("SYNCX_DEVICE", "cuda"))
    _, v_lip = zip(*extract_lip_landmarks_seq(frames, device=os.environ.get("SYNCX_DEVICE", "cuda")))
    audio_seq = extract_audio_seq(wav_t, device=os.environ.get("SYNCX_DEVICE", "cuda"))
    pack_cache(stem, out_base,
               audio_seq=audio_seq, v_clip=v_clip,
               v_lip=np.stack([np.asarray(x) for x in v_lip], axis=0),
               label=label, identity=identity)
    # 边抽边删：cache.pt 已落盘，frames/ + audio.wav 为中间产物（评测只读 .pt）
    _fd = os.path.join(out_base, stem, "frames")
    if os.path.isdir(_fd):
        shutil.rmtree(_fd, ignore_errors=True)
    if os.path.exists(wav):
        try:
            os.remove(wav)
        except OSError:
            pass
    return True


def _run_normal(args):
    PREP = os.path.join(ORIG, "prep")
    manifest = []
    for subset in _SUBSETS:
        d = os.path.join(ORIG, subset)
        vids = sorted(iter_videos(d))
        vids = _shard(vids, args.num_shards, args.shard_id)
        label = _label_from_subset(subset)
        print(f"[fac-normal] {subset}: {len(vids)} clips (label={label})")
        t0 = time.time(); done = 0
        for vid in vids:
            if args.dry_run:
                continue
            rel = os.path.relpath(vid, d)
            identity = _identity_from_rel(rel)
            out_base = os.path.join(PREP, subset)
            os.makedirs(out_base, exist_ok=True)
            r = _prep_one(vid, out_base, label, identity)
            if r is True:
                manifest.append({
                    "stem": os.path.splitext(os.path.basename(vid))[0],
                    "subset": subset, "label": label, "identity": identity,
                })
                done += 1
                log_progress(f"[fac-normal][shard{args.shard_id}] {subset}", done, len(vids), t0)
    if not args.dry_run:
        write_manifest(manifest, PREP, "manifest.jsonl")
        print(f"[fac-normal] prepped {len(manifest)} clips -> {PREP}")


def _run_b6(args):
    for src_name, out_name in [("raw", "b6_raw"), ("trimmed", "b6_trimmed")]:
        src = os.path.join(SPLIT, "b6_std", src_name)
        out_base = os.path.join(SPLIT, "prep", out_name)
        if not os.path.isdir(src):
            print(f"[fac-b6] SKIP {src_name}: {src} 不存在 (请先跑 b6_std_trim.py)")
            continue
        vids = sorted(iter_videos(src))
        vids = _shard(vids, args.num_shards, args.shard_id)
        print(f"[fac-b6] {src_name}: {len(vids)} clips -> {out_base}")
        t0 = time.time(); done = 0
        manifest = []
        for vid in vids:
            if args.dry_run:
                continue
            rel = os.path.relpath(vid, src)
            identity = _identity_from_rel(rel)
            mod = rel.split(os.sep)[0] if rel else "unknown"
            label = 0 if mod == "RVRA" else 1   # RVRA 全真对照；RVFA/FVFA 假音频
            os.makedirs(out_base, exist_ok=True)
            r = _prep_one(vid, out_base, label, identity)
            if r is True:
                manifest.append({
                    "stem": os.path.splitext(os.path.basename(vid))[0],
                    "mod": mod, "label": label, "identity": identity,
                })
                done += 1
                log_progress(f"[fac-b6][shard{args.shard_id}] {src_name}", done, len(vids), t0)
        if not args.dry_run:
            write_manifest(manifest, out_base, "manifest.jsonl")
            print(f"[fac-b6] {src_name}: prepped {len(manifest)} clips -> {out_base}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="normal", choices=["normal", "b6"])
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="只统计片段数，不抽特征")
    args = ap.parse_args()
    if args.mode == "normal":
        _run_normal(args)
    else:
        _run_b6(args)


if __name__ == "__main__":
    main()
