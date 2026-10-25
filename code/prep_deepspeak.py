"""
Prep DeepSpeak v1.1（主训练集）for Sync-X 训练 —— 适配官方 real/fake/test 解压布局。

来源: HuggingFace `faridlab/deepspeak_v1_1`（gated，25 个 zip，46 GB）。解压后：
    <ROOT>/real/          训练真实 (label=0)
    <ROOT>/fake/          训练伪造 (label=1)
    <ROOT>/test/real/     测试真实 (label=0)
    <ROOT>/test/fake/     测试伪造 (label=1)

标签 / 身份 / 同源信息一律从官方 annotations CSV 读取（不靠文件名猜）：
  annotations-real.csv : video-file / identity / recording / ...
  annotations-fake.csv : video-file / kind / engine / identity-source /
                         identity-target / recording-source / recording-target / ...
  annotations-split-def.csv : identity -> train(175) / test(46)

Rec C 同源配对键（dataset.PairedBatchSampler 消费）：
  real.source_id = video-file 去扩展名（文件名 "<identity>-<recording>.mp4" 即权威键）
  fake.source_id = "<identity-target>-<recording-target>"
  注意：annotations-real.csv 的 recording 列是 identity 内的 0 基索引（非文件名里的
  录音号，如 0-4095.mp4 的 recording=0），不能用来拼 source_id；fake 的
  recording-target 则与真实录音文件名一致（已实测 6,796/6,796 精确命中）。

已知数据集缺口/垃圾：
  - annotations-real.csv 里 identity=220 的 29 段在 25 个 zip 中不存在
    （官方 loader 同样以 "a data point is missing" 跳过）——本脚本遍历磁盘，
    天然不产出这 29 条。
  - zip 内混入 1 个 macOS AppleDouble 垃圾文件 `._0-4438.mp4`（不在任何 CSV）——
    本脚本按"不在 annotations 即跳过"过滤，与官方 annotation-driven 语义一致。
  - 上述两点均不影响 manifest 与训练；论文数据描述可加脚注。

输出（扁平 *.pt，与 dataset.FeatureCacheDataset 的 glob(cache_dir/*.pt) 一致）：
    <ROOT>/prep/train/*.pt     训练 real+fake（label 0/1）
    <ROOT>/prep/test/*.pt      测试 real+fake（label 0/1）
    <ROOT>/prep/manifest.jsonl

用法：
    python prep_deepspeak.py --dry-run     # 只校验布局/标签/同源映射，不抽特征（无需重型依赖）
    python prep_deepspeak.py               # 正式（单进程；需先接线 common.py 的 TODO(WIRE) 编码器）
    python prep_deepspeak.py --num-shards 4 --shard-id 0   # 并行分片 0/4（另开窗口跑 1/4 2/4 3/4）
    # 每个分片按视频索引 i % num_shards == shard_id 分配，互不写同一文件；
    # 崩了/关了窗口重跑同参数即可，已抽的 .pt 会被跳过（断点续跑）。
Stage-2 : --train-cache <ROOT>/prep/train --val-cache <ROOT>/prep/test
Stage-1 : --real-cache <ROOT>/prep/train   # real_only=True 自动滤掉 fake（主源仍为 VoxCeleb2）
"""
import os, csv, sys, argparse, tempfile, time
sys.path.insert(0, os.path.dirname(__file__))
from common import (
    iter_videos, sample_frames, clip_center_window, video_to_wav, load_audio_tensor,
    extract_clip_frame, extract_clip_frames, extract_lip_landmarks, extract_lip_landmarks_seq, extract_audio_seq,
    pack_cache, write_manifest, log_progress,
)
from config import DATASET_PATHS
import numpy as np

ROOT = DATASET_PATHS["deepspeak"]        # clip 目录（D:/Datasets/DeepSpeak_v1_1/clips）
DATASET_ROOT = os.path.dirname(ROOT)     # 数据集根（annotations CSV 所在）
PREP = os.path.join(ROOT, "prep")
TARGET_FRAMES = 100      # 4s @ 25fps = 100 帧（固定中心窗，修 P1 零填充 + 采样混叠）
WINDOW_SEC = 4.0         # 中心窗时长（秒）；短于窗的片段用全部帧
WINDOW_FPS = 25.0        # 窗内采样率（满足 1–8Hz 运动学 Nyquist）
SR = 16000

# (split, 磁盘子目录, label)
SPLIT_DIRS = [
    ("train", "real", 0),
    ("train", "fake", 1),
    ("test",  "real", 0),
    ("test",  "fake", 1),
]


def np_stack(xs):
    return np.stack([np.asarray(x) for x in xs], axis=0)


def _read_csv(name):
    with open(os.path.join(DATASET_ROOT, name), encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def load_annotations():
    """返回 (real_lookup, fake_lookup)：video-file -> 行 dict。"""
    real = {r["video-file"]: r for r in _read_csv("annotations-real.csv")
            if (r.get("video-file") or "").strip()}
    fake = {r["video-file"]: r for r in _read_csv("annotations-fake.csv")
            if (r.get("video-file") or "").strip()}
    return real, fake


def clip_meta(sub, vf, real_lookup, fake_lookup):
    """返回 (identity, source_id)。

    real: 文件名 "<identity>-<recording>.mp4" 即权威键
          -> identity=前缀, source_id=stem（CSV 的 recording 列是 identity 内索引，不可用）。
    fake: identity=identity-target, source_id="<identity-target>-<recording-target>"
          （recording-target 与真实录音文件名一致，已实测 100% 命中）。
    """
    if "fake" in sub:
        r = fake_lookup.get(vf, {})
        return (r.get("identity-target", "?"),
                f"{r.get('identity-target', '?')}-{r.get('recording-target', '?')}")
    stem = vf[:-4] if vf.lower().endswith(".mp4") else os.path.splitext(vf)[0]
    return (stem.split("-")[0] if "-" in stem else "?", stem)


def main():
    ap = argparse.ArgumentParser(description="Prep DeepSpeak v1.1（real/fake/test 布局）")
    ap.add_argument("--dry-run", action="store_true", help="只统计布局/标签/同源命中，不抽特征")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="并行分片总数：每片一个独立进程，按索引分配片段，互不写同一文件")
    ap.add_argument("--shard-id", type=int, default=0,
                    help="当前进程负责的分片编号，范围 0..num-shards-1")
    args = ap.parse_args()
    if not (args.num_shards >= 1 and 0 <= args.shard_id < args.num_shards):
        ap.error("--shard-id 必须在 [0, num-shards) 且 num-shards>=1")

    real_lookup, fake_lookup = load_annotations()
    manifest = []
    stat = {}
    os.makedirs(PREP, exist_ok=True)

    for split, sub, label in SPLIT_DIRS:
        src = os.path.join(ROOT, sub) if split == "train" else os.path.join(ROOT, "test", sub)
        all_videos = list(iter_videos(src))
        owned = [v for i, v in enumerate(all_videos)
                 if i % args.num_shards == args.shard_id]
        out_root = os.path.join(PREP, split)
        os.makedirs(out_root, exist_ok=True)
        print(f"[shard] {args.shard_id}/{args.num_shards}: {sub} 本片 {len(owned)}/{len(all_videos)} 个片段")
        t0 = time.time(); seen = 0
        for vid in owned:
            vf = os.path.basename(vid)
            stem = os.path.splitext(vf)[0]
            # 仅处理官方 annotations 覆盖的片段（垃圾/孤儿文件如 ._ 前缀一律跳过）
            lookup = real_lookup if "real" in sub else fake_lookup
            if vf not in lookup:
                print(f"[skip] 不在官方 annotations 中: {vf}")
                continue
            identity, sid = clip_meta(sub, vf, real_lookup, fake_lookup)
            key = f"{split}/{sub}"
            stat[key] = stat.get(key, 0) + 1
            if args.dry_run:
                manifest.append({"stem": stem, "identity": identity, "split": split,
                                 "label": label, "source_id": sid})
                continue
            out_pt = os.path.join(out_root, f"{stem}.pt")
            if os.path.exists(out_pt):
                continue            # 断点续跑：已抽则跳过
            ss, win_dur = clip_center_window(vid, WINDOW_SEC, WINDOW_FPS)
            frames = sample_frames(vid, target_count=TARGET_FRAMES,
                                   window_sec=WINDOW_SEC, window_fps=WINDOW_FPS)
            if not frames:
                print(f"[skip] 抽帧失败: {vf}")
                continue
            with tempfile.TemporaryDirectory(prefix="ds_wav_") as td:
                wav = os.path.join(td, "a.wav")
                if not video_to_wav(vid, wav, sr=SR, ss=ss, td=win_dur):
                    print(f"[skip] 音频抽取失败: {vf}")
                    continue
                wav_t = load_audio_tensor(wav, sr=SR)
                v_clip = extract_clip_frames(frames, device=os.environ.get("SYNCX_DEVICE", "cuda"))              # (N,512) 批量 CLIP 前向
                _, v_lip = zip(*extract_lip_landmarks_seq(frames, device=os.environ.get("SYNCX_DEVICE", "cuda")))
                audio_seq = extract_audio_seq(wav_t, device=os.environ.get("SYNCX_DEVICE", "cuda"))
                pack_cache(stem, out_root, flat=True, label=label, identity=identity,
                           source_id=sid, audio_seq=audio_seq, v_clip=v_clip,
                           v_lip=np_stack(v_lip))
            manifest.append({"stem": stem, "identity": identity, "split": split,
                             "label": label, "source_id": sid})
            seen += 1
            log_progress(f"[deepspeak][shard{args.shard_id}] {sub}", seen, len(owned), t0)

    # 分片并行时各写独立 manifest，避免多进程并发覆盖同一文件；
    # 训练侧用 glob(prep/*.pt) 读取，不依赖 manifest。
    mname = f"manifest_shard{args.shard_id}.jsonl" if args.num_shards > 1 else "manifest.jsonl"
    write_manifest(manifest, PREP, mname)
    print("[deepspeak] 分布:", dict(stat))
    if args.dry_run:
        real_sids = {m["source_id"] for m in manifest if m["label"] == 0}
        fake_total = sum(1 for m in manifest if m["label"] == 1)
        fake_hit = sum(1 for m in manifest if m["label"] == 1 and m["source_id"] in real_sids)
        print(f"[deepspeak] dry-run: fake 同源命中 {fake_hit}/{fake_total}（未写缓存）")
    else:
        print(f"[deepspeak] prepped {len(manifest)} clips -> {PREP}")


if __name__ == "__main__":
    main()
