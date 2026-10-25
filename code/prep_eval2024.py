"""
Prep Deepfake-Eval-2024 (跨域主评测) for Sync-X 评测。
来源 : Chandra et al., 2025 (16_Deepfake-Eval-2024)。大规模 in-the-wild 多模态基准。
不参训，只评测。

磁盘结构（2026-08-02 核实，D:/Datasets/Deepfake-Eval-2024/）：
    video-data/                          : 2035 个视频平铺（文件名即视频 ID）
    video-metadata-publish-with-links.csv: 权威标签源，字段：
        Filename, Date, Video Ground Truth (Real/Fake), Audio Ground Truth,
        Public Comments, Finetuning Set (train/test), Media
    共 2036 行；2035 个文件名与 video-data/ 内 mp4 一一对应。

**标签铁律**：真实/伪造标签只能从 CSV 的 `Video Ground Truth` 读取，
              绝不能靠文件名猜（DFE2024 文件名是随机 YouTube ID，不含 fake 字样）。

输出 : D:/Datasets/Deepfake-Eval-2024/prep/<split>/<stem>/cache.pt
       split 取 CSV 的 Finetuning Set（test=815 / train=1221；默认只跑 test）。
流程 : 抽帧 -> 抽音频 -> (WIRE) 特征 -> 半精度缓存
"""
import os, sys, csv, glob, shutil, argparse, time
sys.path.insert(0, os.path.dirname(__file__))
from common import (
    sample_frames, clip_center_window, save_frames,
    video_to_wav, load_audio_tensor,
    extract_clip_frames, extract_lip_landmarks, extract_lip_landmarks_seq, extract_audio_seq,
    pack_cache, write_manifest, log_progress,
)
from config import DATASET_PATHS
import numpy as np

ROOT = DATASET_PATHS["dfe2024"]
PREP = os.path.join(ROOT, "prep")
META = os.path.join(ROOT, "video-metadata-publish-with-links.csv")
SPLITS = ["test"]                  # 只评测划分；改 ["train","test"] 可含训练划分
TARGET_FRAMES = 100
WINDOW_SEC = 4.0         # 中心窗时长（秒），对齐训练域窗口；短于窗的片段用全部帧
WINDOW_FPS = 25.0        # 窗内采样率
SR = 16000


def np_stack(xs):
    return np.stack([np.asarray(x) for x in xs], axis=0)


def _load_metadata():
    """读权威标签 CSV -> {filename: (video_label, finetune_set)}。"""
    meta = {}
    with open(META, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            fn = row["Filename"].strip()
            vgt = row["Video Ground Truth"].strip().lower()   # "real" / "fake"
            if vgt not in ("real", "fake"):
                print(f"  [warn] 未知 Video Ground Truth: {row['Video Ground Truth']!r} ({fn})")
                continue
            meta[fn] = (1 if vgt == "fake" else 0, row["Finetuning Set"].strip())
    return meta


def main():
    ap = argparse.ArgumentParser(description="Prep Deepfake-Eval-2024 for Sync-X 评测")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="并行分片总数：每片一个独立进程，按全局索引分配片段，互不写同一文件")
    ap.add_argument("--shard-id", type=int, default=0,
                    help="当前进程负责的分片编号，范围 0..num-shards-1")
    args = ap.parse_args()
    if not (args.num_shards >= 1 and 0 <= args.shard_id < args.num_shards):
        ap.error("--shard-id 必须在 [0, num-shards) 且 num-shards>=1")

    meta = _load_metadata()
    print(f"[eval2024] metadata: {len(meta)} clips (Video Ground Truth)")
    # 先平铺全部 (split, vid, label) 以便按全局索引分片（与 prep_voxceleb2 行为一致）
    all_items = []
    for split in SPLITS:
        vids = sorted(glob.glob(os.path.join(ROOT, "video-data", "*.mp4")))
        print(f"[eval2024] {split}: {len(vids)} videos on disk")
        for vid in vids:
            fn = os.path.basename(vid)
            if fn not in meta:
                print(f"  [skip] 不在 metadata CSV: {fn}")
                continue
            label, finetune = meta[fn]
            if finetune != split:
                continue
            all_items.append((split, vid, label))

    manifest = []
    total = 0
    skipped = 0
    owned = sum(1 for gi in range(len(all_items)) if gi % args.num_shards == args.shard_id)
    t0 = time.time(); done = 0
    for gi, (split, vid, label) in enumerate(all_items):
        if gi % args.num_shards != args.shard_id:
            continue
        stem = os.path.splitext(os.path.basename(vid))[0]
        out_dir = os.path.join(PREP, split, stem)
        cache_pt = os.path.join(out_dir, "cache.pt")
        if os.path.exists(cache_pt):
            skipped += 1
            continue            # 断点续跑：已抽则跳过
        ss, win_dur = clip_center_window(vid, WINDOW_SEC, WINDOW_FPS)
        frames = sample_frames(vid, target_count=TARGET_FRAMES,
                               window_sec=WINDOW_SEC, window_fps=WINDOW_FPS)
        if not frames:
            continue
        save_frames(frames, os.path.join(out_dir, "frames"))
        wav = os.path.join(out_dir, "audio.wav")
        if not video_to_wav(vid, wav, sr=SR, ss=ss, td=win_dur):
            continue
        wav_t = load_audio_tensor(wav, sr=SR)
        v_clip = extract_clip_frames(frames, device=os.environ.get("SYNCX_DEVICE", "cuda"))
        _, v_lip = zip(*extract_lip_landmarks_seq(frames, device=os.environ.get("SYNCX_DEVICE", "cuda")))
        audio_seq = extract_audio_seq(wav_t, device=os.environ.get("SYNCX_DEVICE", "cuda"))
        pack_cache(stem, os.path.join(PREP, split),
                   audio_seq=audio_seq, v_clip=v_clip,
                   v_lip=np_stack(v_lip),
                   label=label, identity=stem)
        # 边抽边删：cache.pt 已落盘，frames/ + audio.wav 为中间产物（训练/评测只读 .pt）
        _fd = os.path.join(out_dir, "frames")
        if os.path.isdir(_fd):
            shutil.rmtree(_fd, ignore_errors=True)
        if os.path.exists(wav):
            try:
                os.remove(wav)
            except OSError:
                pass
        manifest.append({"stem": stem, "split": split, "label": label})
        total += 1
        done += 1
        log_progress(f"[eval2024][shard{args.shard_id}]", done, owned, t0)

    mname = f"manifest_shard{args.shard_id}.jsonl" if args.num_shards > 1 else "manifest.jsonl"
    write_manifest(manifest, PREP, mname)
    n_real = sum(1 for m in manifest if m["label"] == 0)
    print(f"[eval2024] prepped {len(manifest)} clips "
          f"({n_real} real / {len(manifest)-n_real} fake) -> {PREP}")


if __name__ == "__main__":
    main()
