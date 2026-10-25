"""
Prep HiFi-AVDF (跨域主评测 / 商用级原生联合 AV 生成基准) for Sync-X 评测。
来源 : Peng et al., 2026 (HAVIC 配套数据集, 30_Leave-no-stone)。商用生成器 (Veo~3.1 /
       Kling~2.5 / Seedance~1.0 / PixVerse~V5 / WAN~2.5 / Sora~2)。
下载 : 官方 release (HAVIC 仓库, 开源代码 + 划分)。只下【评测划分】；不参训，只评测。
输入 : <HIFI_ROOT>/<gen>/<clip>.mp4
输出 : <HIFI_ROOT>/prep/<gen>/<stem>/cache.pt
流程 : 抽帧 -> 抽音频 -> (WIRE) 特征 -> 半精度缓存

============================================================================
  ⚠️ 标签来源铁律（方案 ① 的核心约束，请勿改动）
============================================================================
  HiFi-AVDF 的真实/伪造标签 **只能** 从 6 个 per-gen CSV 读取：
      kling2.5.csv / pixverseV5.csv / seedance1.0.csv /
      sora2.csv / veo3.1.csv / wan2.5.csv
  字段: video_path, audio_label, visual_label, overall_label
        (video_path 为相对 HIFI_ROOT 的路径, 如 "kling2.5/1.mp4")

  **绝对禁止** 从以下文件读标签:
    - face_verify_hifi_master.jsonl  : 仅是"人脸验证门控清单"
                                        (path/pass_haar/pass_retina/decision/gen),
                                        **不含任何标签字段**。
    - face_verify_hifi_<gen>.jsonl    : **仅 veo3.1 / wan2.5 存在**；
                                        kling2.5 / pixverseV5 / sora2 / seedance1.0
                                        这 4 个 per-gen JSONL **官方未发布** -> 读会 FileNotFound。

  结论: 评测 dataloader 必须 (1) 只读 6 个 per-gen CSV 取 overall_label,
         (2) 可选 join master.jsonl 做"人脸是否通过验证"的质检门控 (非标签)。
        本脚本已用 _HIFI_GENS + _load_labels() 把"只走 CSV"锁死, 不依赖任何 .jsonl。
============================================================================
"""
import os, sys, csv, glob, shutil, argparse, time
sys.path.insert(0, os.path.dirname(__file__))
from common import (
    iter_videos, sample_frames, clip_center_window, save_frames,
    video_to_wav, load_audio_tensor,
    extract_clip_frames, extract_lip_landmarks, extract_lip_landmarks_seq, extract_audio_seq,
    pack_cache, write_manifest, log_progress,
)
from config import DATASET_PATHS
import numpy as np

ROOT = DATASET_PATHS["hifi"]
PREP = os.path.join(ROOT, "prep")
TARGET_FRAMES = 100
WINDOW_SEC = 4.0         # 中心窗时长（秒），对齐训练域窗口；短于窗的片段用全部帧
WINDOW_FPS = 25.0        # 窗内采样率
SR = 16000

# 6 个生成器 = 6 个 CSV (全齐)。per-gen JSONL 缺失 4 个, 故本脚本只依赖 CSV。
_HIFI_GENS = ["kling2.5", "pixverseV5", "seedance1.0", "sora2", "veo3.1", "wan2.5"]


def _load_labels():
    """读取 6 个 per-gen CSV, 返回 {video_path(相对ROOT): (overall, audio, visual)}。
    唯一权威标签来源 — 不碰任何 .jsonl。"""
    labels = {}
    for gen in _HIFI_GENS:
        csv_path = os.path.join(ROOT, f"{gen}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"HiFi 标签 CSV 缺失: {csv_path} (6 个 per-gen CSV 必须全在)")
        with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                vp = row["video_path"].strip()
                labels[vp] = (
                    int(row["overall_label"]),
                    int(row["audio_label"]),
                    int(row["visual_label"]),
                )
    return labels


def np_stack(xs):
    return np.stack([np.asarray(x) for x in xs], axis=0)


def main():
    ap = argparse.ArgumentParser(description="Prep HiFi-AVDF for Sync-X 评测")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="并行分片总数：每片一个独立进程，按全局索引分配片段，互不写同一文件")
    ap.add_argument("--shard-id", type=int, default=0,
                    help="当前进程负责的分片编号，范围 0..num-shards-1")
    args = ap.parse_args()
    if not (args.num_shards >= 1 and 0 <= args.shard_id < args.num_shards):
        ap.error("--shard-id 必须在 [0, num-shards) 且 num-shards>=1")

    labels = _load_labels()
    print(f"[hifi] loaded {len(labels)} clip labels from {len(_HIFI_GENS)} per-gen CSVs (no JSONL used)")
    # 先平铺全部 (gen, vid, rel) 以便按全局索引分片（与 prep_voxceleb2 行为一致）
    all_items = []
    for gen in _HIFI_GENS:
        vids = iter_videos(os.path.join(ROOT, gen))
        print(f"[hifi] {gen}: {len(vids)} videos")
        for vid in vids:
            rel = os.path.relpath(vid, ROOT).replace("\\", "/")   # e.g. "kling2.5/1.mp4"
            if rel not in labels:
                # 不在 CSV 中 -> 跳过 (如 macOS ._ 元数据或质检未通过样本)
                print(f"  [skip] not in CSV label map: {rel}")
                continue
            overall, audio, visual = labels[rel]
            all_items.append((gen, vid, overall, audio, visual))

    manifest = []
    total = 0
    skipped = 0
    owned = sum(1 for gi in range(len(all_items)) if gi % args.num_shards == args.shard_id)
    t0 = time.time(); done = 0
    for gi, (gen, vid, overall, audio, visual) in enumerate(all_items):
        if gi % args.num_shards != args.shard_id:
            continue
        stem = os.path.splitext(os.path.basename(vid))[0]
        out_dir = os.path.join(PREP, gen, stem)
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
        pack_cache(stem, os.path.join(PREP, gen),
                   audio_seq=audio_seq, v_clip=v_clip,
                   v_lip=np_stack(v_lip),
                   label=overall, identity=gen)
        # 边抽边删：cache.pt 已落盘，frames/ + audio.wav 为中间产物（评测只读 .pt）
        _fd = os.path.join(out_dir, "frames")
        if os.path.isdir(_fd):
            shutil.rmtree(_fd, ignore_errors=True)
        if os.path.exists(wav):
            try:
                os.remove(wav)
            except OSError:
                pass
        manifest.append({
            "stem": stem, "generator": gen, "split": "eval",
            "label": overall, "audio_label": audio, "visual_label": visual,
        })
        total += 1
        done += 1
        log_progress(f"[hifi][shard{args.shard_id}]", done, owned, t0)

    mname = f"manifest_shard{args.shard_id}.jsonl" if args.num_shards > 1 else "manifest.jsonl"
    write_manifest(manifest, PREP, mname)
    print(f"[hifi] prepped {len(manifest)} clips  (real/fake from per-gen CSV only) -> {PREP}")


if __name__ == "__main__":
    main()
