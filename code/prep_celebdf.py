"""
Prep Celeb-DF-v2 (跨域评测 / OOD 视觉探针) for Sync-X 评测。
来源 : Li et al., 2020 (11_Celeb-DF)。纯视觉 reenactment 伪造。
只下【test 划分】；不参训，只评测 -> 不需要全量。

磁盘结构（2026-08-02 核实，D:/Datasets/Celeb-DF-v2/）：
    Celeb-real/          : 真实视频 (590 段, 按 idN 分)
    Celeb-synthesis/     : 伪造视频 (5639 段, idN_idM)
    YouTube-real/        : 真实视频 (300 段)
    List_of_testing_videos.txt : 官方测试清单, 每行 "<label> <path>"
                                 label: 0=fake(340) / 1=real(178), 共 518
                                 (178 real = 70 YouTube-real + 108 Celeb-real)

输出 : D:/Datasets/Celeb-DF-v2/prep/test/<stem>/cache.pt
流程 : 抽帧 -> 抽音频 -> (WIRE) 特征 -> 半精度缓存
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import (
    iter_videos, sample_frames, save_frames,
    video_to_wav, load_audio_tensor,
    extract_clip_frames, extract_lip_landmarks, extract_lip_landmarks_seq, extract_audio_seq,
    pack_cache, write_manifest,
)
from config import DATASET_PATHS
import numpy as np

ROOT = DATASET_PATHS["celebdf"]
PREP = os.path.join(ROOT, "prep")
TEST_LIST = os.path.join(ROOT, "List_of_testing_videos.txt")
TARGET_FRAMES = 100
SR = 16000


def np_stack(xs):
    return np.stack([np.asarray(x) for x in xs], axis=0)


def main():
    # 官方测试清单: "label path" (0=fake, 1=real)
    test_entries = []
    with open(TEST_LIST, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            label = int(parts[0])
            rel = parts[1].replace("\\", "/")
            test_entries.append((rel, label))
    print(f"[celebdf] official test list: {len(test_entries)} entries "
          f"({sum(1 for _, l in test_entries if l==1)} real / "
          f"{sum(1 for _, l in test_entries if l==0)} fake)")

    manifest = []
    prepped = 0
    for rel, label in test_entries:
        vid = os.path.join(ROOT, rel)
        if not os.path.exists(vid):
            print(f"  [skip] missing: {rel}")
            continue
        stem = os.path.splitext(os.path.basename(vid))[0]
        identity = os.path.basename(os.path.dirname(vid))   # idN / YouTube-real
        out_dir = os.path.join(PREP, "test", identity, stem)
        frames = sample_frames(vid, target_count=TARGET_FRAMES)
        if not frames:
            continue
        save_frames(frames, os.path.join(out_dir, "frames"))
        wav = os.path.join(out_dir, "audio.wav")
        if not video_to_wav(vid, wav, sr=SR):
            continue
        wav_t = load_audio_tensor(wav, sr=SR)
        v_clip = extract_clip_frames(frames)
        _, v_lip = zip(*extract_lip_landmarks_seq(frames))
        audio_seq = extract_audio_seq(wav_t)
        pack_cache(stem, os.path.join(PREP, "test", identity),
                   audio_seq=audio_seq, v_clip=v_clip,
                   v_lip=np_stack(v_lip),
                   label=label, identity=identity)
        manifest.append({"stem": stem, "identity": identity, "split": "test", "label": label})
        prepped += 1
    write_manifest(manifest, PREP, "manifest.jsonl")
    print(f"[celebdf] prepped {prepped}/{len(test_entries)} clips")


if __name__ == "__main__":
    main()
