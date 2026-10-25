"""
Prep TalkingHeadBench (THB, 跨域主评测) for Sync-X 评测。
来源 : Xiong et al., WACV 2026 (xiongTalkingHeadBenchMultimodalBenchmark2026, arXiv:2505.24866)。
不参训，只评测。

磁盘结构（2026-08-02 核实，D:/Datasets/TalkingHeadBench/）：
    real/
        CelebV-HQ/                : 1608 段真实视频（真人口播）
        ff++/                     : 1000 段 FF++ 真实视频
        real_dataset_split.json   : THB 作者划分 {Train,Val,Test} × {CelebV-HQ, FaceForensics++}
                                    FF++: 470/84/150 = 704（论文 Ch4:252 采用此 704-video split）
        real_dataset_split_official_ff++.json : 官方 FF++ 720/140/140 + 同 CelebV-HQ 划分
    fake/
        additional_dataset/{Hallo3,MAGI-1}   : 未见生成器（zero-shot 核心）
        AniPortraitAudio/ AniPortraitVideo/ EmoPortrait/ Hallo/ Hallo2/ LivePortrait/
        各引擎下按 train/test/val 划分
    audio/fake/                   : 2927 段 wav（音频分离存储，评测用视频自带音轨时忽略）

协议（论文 Ch4:252）：
    - THB 的 real 类采用 THB 作者划分的 704-video FF++ split（470+84+150），
      而非 canonical 1000 段 FF++（296 段原始音频已不可公开获取被排除）。
    - fake 全部 1219 段（含 unseen Hallo3 / MAGI-1）。

输出 : D:/Datasets/TalkingHeadBench/prep/{real,fake}/<cond>/<stem>/cache.pt
       cond: real -> ff++_704（仅 704 段）；fake -> <engine>（按引擎分）。
流程 : 抽帧 -> 抽音频 -> (WIRE) 特征 -> 半精度缓存
"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(__file__))
from common import (
    sample_frames, save_frames,
    video_to_wav, load_audio_tensor,
    extract_clip_frames, extract_lip_landmarks, extract_lip_landmarks_seq, extract_audio_seq,
    pack_cache, write_manifest,
)
from config import DATASET_PATHS
import numpy as np

ROOT = DATASET_PATHS["thb"]
REAL_DIR = os.path.join(ROOT, "real")
FAKE_DIR = os.path.join(ROOT, "fake")
PREP = os.path.join(ROOT, "prep")
SPLIT_JSON = os.path.join(ROOT, "real", "real_dataset_split.json")
TARGET_FRAMES = 100
SR = 16000

# fake 引擎目录（additional_dataset 内含 unseen Hallo3 / MAGI-1）
FAKE_ENGINES = ["additional_dataset", "AniPortraitAudio", "AniPortraitVideo",
                "EmoPortrait", "Hallo", "Hallo2", "LivePortrait"]


def np_stack(xs):
    return np.stack([np.asarray(x) for x in xs], axis=0)


def prep_one(vid, out_root, label, identity, cond):
    stem = os.path.splitext(os.path.basename(vid))[0]
    out_dir = os.path.join(out_root, cond, stem)
    frames = sample_frames(vid, target_count=TARGET_FRAMES)
    if not frames:
        return None
    save_frames(frames, os.path.join(out_dir, "frames"))
    wav = os.path.join(out_dir, "audio.wav")
    if not video_to_wav(vid, wav, sr=SR):
        return None
    wav_t = load_audio_tensor(wav, sr=SR)
    v_clip = extract_clip_frames(frames)
    _, v_lip = zip(*extract_lip_landmarks_seq(frames))
    audio_seq = extract_audio_seq(wav_t)
    pack_cache(stem, os.path.join(out_root, cond),
               audio_seq=audio_seq, v_clip=v_clip,
               v_lip=np_stack(v_lip),
               label=label, identity=identity)
    return (stem, label, cond)


def main():
    manifest = []

    # ── real: THB 作者划分的 FF++ 704 (470+84+150) ──
    split = json.load(open(SPLIT_JSON, encoding="utf-8"))
    ffpp_files = set()
    for phase in ("Train", "Val", "Test"):
        for fn in split[phase].get("FaceForensics++", []):
            ffpp_files.add(os.path.basename(fn))
    print(f"[thb] real FF++ 704 split: {len(ffpp_files)} unique files")
    real_vids = []
    for fn in sorted(ffpp_files):
        p = os.path.join(REAL_DIR, "ff++", fn)
        if os.path.exists(p):
            real_vids.append(p)
        else:
            print(f"  [skip] 缺 FF++ real: {fn}")
    for vid in real_vids:
        r = prep_one(vid, os.path.join(PREP, "real"), 0, "ff++", "ff++_704")
        if r:
            manifest.append({"stem": r[0], "split": "real", "engine": "ff++", "label": r[1], "cond": r[2]})
    print(f"[thb] real: prepped {sum(1 for m in manifest if m['split']=='real')}")

    # ── fake: 7 引擎全部（含 unseen Hallo3 / MAGI-1）──
    n_fake = 0
    for eng in FAKE_ENGINES:
        eng_dir = os.path.join(FAKE_DIR, eng)
        if not os.path.isdir(eng_dir):
            continue
        vids = sorted(glob.glob(os.path.join(eng_dir, "**", "*.mp4"), recursive=True))
        for vid in vids:
            # identity: additional_dataset/Hallo3 -> "Hallo3"; 其余 -> 引擎名
            rel = os.path.relpath(vid, FAKE_DIR)
            identity = eng if eng != "additional_dataset" else \
                os.path.basename(os.path.dirname(vid))
            r = prep_one(vid, os.path.join(PREP, "fake"), 1, identity, eng)
            if r:
                manifest.append({"stem": r[0], "split": "fake", "engine": eng,
                                 "label": r[1], "cond": r[2]})
                n_fake += 1
    print(f"[thb] fake: prepped {n_fake}")

    write_manifest(manifest, PREP, "manifest.jsonl")
    print(f"[thb] done: {len(manifest)} clips "
          f"({sum(1 for m in manifest if m['label']==0)} real / "
          f"{sum(1 for m in manifest if m['label']==1)} fake)")


if __name__ == "__main__":
    main()
