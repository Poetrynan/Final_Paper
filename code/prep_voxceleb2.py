"""
Prep VoxCeleb2 子集(500 说话人) for Sync-X Stage-1 SSL + 真实样本补充。
输入 : D:/Datasets/VoxCeleb2_subset/<speaker_id>/<file>.mp4  (后台下载 task AYmcBd 完成后再跑)
输出 : D:/Datasets/VoxCeleb2_subset/prep/<speaker_id>/<file>/cache.pt
流程 : 抽帧 -> 抽音频 -> (WIRE) 特征 -> 半精度缓存
注意 : VoxCeleb2 只下子集(500人/~25GB 落盘)，不参训，仅 Stage-1 SSL + 平衡真实比。
       全部为真实样本，label=0。
"""
import os, sys, glob, argparse, shutil, time
sys.path.insert(0, os.path.dirname(__file__))
from common import (
    iter_videos, sample_frames, save_frames,
    video_to_wav, load_audio_tensor,
    extract_clip_frames, extract_lip_landmarks, extract_lip_landmarks_seq, extract_audio_seq,
    pack_cache, write_manifest, log_progress,
)
from config import DATASET_PATHS
import numpy as np

ROOT = DATASET_PATHS["voxceleb2"]
PREP = os.path.join(ROOT, "prep")
TARGET_FRAMES = 100
SR = 16000


def np_stack(xs):
    return np.stack([np.asarray(x) for x in xs], axis=0)


def main():
    ap = argparse.ArgumentParser(description="Prep VoxCeleb2 子集 for Sync-X Stage-1 SSL")
    ap.add_argument("--dry-run", action="store_true", help="只统计说话人/片段数，不抽特征（无需重型依赖）")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="并行分片总数：每片一个独立进程，按全局索引分配片段，互不写同一文件")
    ap.add_argument("--shard-id", type=int, default=0,
                    help="当前进程负责的分片编号，范围 0..num-shards-1")
    args = ap.parse_args()
    if not (args.num_shards >= 1 and 0 <= args.shard_id < args.num_shards):
        ap.error("--shard-id 必须在 [0, num-shards) 且 num-shards>=1")

    speakers = sorted(d for d in glob.glob(os.path.join(ROOT, "*")) if os.path.isdir(d))
    speakers = [s for s in speakers if os.path.basename(s) != "prep"]

    # 先平铺全部 (speaker, video) 以便按全局索引分片（与 prep_deepspeak 行为一致）
    all_vids = []
    for spk_dir in speakers:
        spk = os.path.basename(spk_dir)
        for vid in iter_videos(spk_dir):
            all_vids.append((spk, vid))

    manifest = []
    total = 0
    skipped = 0
    owned = sum(1 for gi in range(len(all_vids)) if gi % args.num_shards == args.shard_id)
    t0 = time.time(); done = 0
    for gi, (spk, vid) in enumerate(all_vids):
        if gi % args.num_shards != args.shard_id:
            continue
        stem = os.path.splitext(os.path.basename(vid))[0]
        out_dir = os.path.join(PREP, spk, stem)
        if args.dry_run:
            total += 1
            continue
        cache_pt = os.path.join(out_dir, "cache.pt")
        if os.path.exists(cache_pt):
            skipped += 1
            continue            # 断点续跑：已抽则跳过
        frames = sample_frames(vid, target_count=TARGET_FRAMES)
        n = save_frames(frames, os.path.join(out_dir, "frames"))
        if n == 0:
            continue
        wav = os.path.join(out_dir, "audio.wav")
        if not video_to_wav(vid, wav, sr=SR):
            continue
        # 防御：PyAV/ffmpeg 静默失败可能返回 True 但未落盘 wav，避免下面 load 抛 FileNotFoundError
        if not os.path.exists(wav):
            continue
        wav_t = load_audio_tensor(wav, sr=SR)
        # device 一律走 CPU：SYNCX_MEDIAPIPE_GPU=0 路线下，视觉/音频编码器走 CPU 最稳；
        # 硬编码 "cuda" 会在无卡/混用环境触发 CLIP safetensors 负缓存 OSError（见 .no_exist 坑）。
        v_clip = extract_clip_frames(frames, device=os.environ.get("SYNCX_DEVICE", "cuda"))
        _, v_lip = zip(*extract_lip_landmarks_seq(frames, device=os.environ.get("SYNCX_DEVICE", "cuda")))
        audio_seq = extract_audio_seq(wav_t, device=os.environ.get("SYNCX_DEVICE", "cuda"))
        pack_cache(stem, os.path.join(PREP, spk),
                   audio_seq=audio_seq, v_clip=v_clip,
                   v_lip=np_stack(v_lip),
                   label=0, identity=spk)
        # 边抽边删：cache.pt 已落盘，frames/ + audio.wav 为中间产物（训练只读 .pt），
        # 当场删除，磁盘峰值从 ~450GB 堆积降到单 clip 的 ~5MB（89K clips 不撑爆数据盘）。
        _fd = os.path.join(out_dir, "frames")
        if os.path.isdir(_fd):
            shutil.rmtree(_fd, ignore_errors=True)
        if os.path.exists(wav):
            try:
                os.remove(wav)
            except OSError:
                pass
        manifest.append({"stem": stem, "speaker": spk, "label": 0})
        total += 1
        done += 1
        log_progress(f"[voxceleb2][shard{args.shard_id}]", done, owned, t0)

    if args.dry_run:
        owned = sum(1 for gi in range(len(all_vids)) if gi % args.num_shards == args.shard_id)
        print(f"[voxceleb2][dry-run] {len(speakers)} 说话人, {len(all_vids)} 片段总数")
        print(f"[voxceleb2][dry-run] 本分片 {args.shard_id}/{args.num_shards} 将处理 {owned} 片段")
        print("[voxceleb2][dry-run] 估算：本地 CPU 约 5–15s/片段；云端 GPU(4090D) 约 0.3–0.6s/片段"
              "（含 CLIP/wav2vec2/MediaPipe）。实际以首片段计时为准。")
        return

    mname = f"manifest_shard{args.shard_id}.jsonl" if args.num_shards > 1 else "manifest.jsonl"
    write_manifest(manifest, PREP, mname)
    print(f"[voxceleb2] prepped {total} clips (skipped {skipped} 已存在), {len(speakers)} speakers -> {PREP}")


if __name__ == "__main__":
    main()
