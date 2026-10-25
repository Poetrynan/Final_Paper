"""
Prep MVAD (跨域主评测 / 原生联合 AV 生成基准) for Sync-X 评测。
来源 : Hu et al., 2025 (huMVADBenchmark2025)。直接针对二代原生联合 AV 生成器
       (Sora2 / Seedance 2.0 / Veo3 / Kling 家族)。不参训，只评测。

磁盘结构（2026-08-02 核实，D:/Datasets/MVAD/）：
    test_human/
        fake_fake/  {jimeng3.5pro,kling2_5_Turbo,kling3_0,seedance2_0,Sora2,veo3,ViduQ2,wan2_6}/human/*.mp4
        fake_real/  {hunyuan_avater,kling_avatar,wan2_1}/human/*.mp4
        real_real/  ugc/human/*.mp4
    mvad_eval_clean.csv : 权威清单（4432 行 eval_include=1, 全部 content_type=human），字段：
        clip_path, split, dir_generator, true_generator, content_type, label, eval_include
        clip_path 前缀为旧路径 D:\\MVAD\\，需替换为 D:/Datasets/MVAD。

**标签铁律**：只取 eval_include=1 的行（4432, 纯 human-only）；content_type=non_human
              全部排除。label: fake/real -> 1/0。只跑 test 划分（全部都在 test_human 下）。

输出 : D:/Datasets/MVAD/prep/<split>/<stem>/cache.pt
流程 : 抽帧 -> 抽音频 -> (WIRE) 特征 -> 半精度缓存
"""
import os, sys, csv
sys.path.insert(0, os.path.dirname(__file__))
from common import (
    sample_frames, save_frames,
    video_to_wav, load_audio_tensor,
    extract_clip_frames, extract_lip_landmarks, extract_lip_landmarks_seq, extract_audio_seq,
    pack_cache, write_manifest,
)
from config import DATASET_PATHS
import numpy as np

ROOT = DATASET_PATHS["mvad"]
PREP = os.path.join(ROOT, "prep")
CLEAN_CSV = os.path.join(ROOT, "mvad_eval_clean.csv")
TARGET_FRAMES = 100
SR = 16000


def np_stack(xs):
    return np.stack([np.asarray(x) for x in xs], axis=0)


def _fix_path(p):
    """把 CSV 中的旧前缀 D:\\MVAD\\ 换成实际根目录 D:/Datasets/MVAD/。"""
    return p.replace("D:\\MVAD", ROOT).replace("D:/MVAD", ROOT)


def main():
    rows = []
    with open(CLEAN_CSV, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["eval_include"] != "1":
                continue          # 只保留 human-only（4432）
            rows.append(r)
    print(f"[mvad] clean CSV: {len(rows)} eval_include=1 (human-only)")

    manifest = []
    prepped = 0
    n_real = n_fake = 0
    for r in rows:
        vid = _fix_path(r["clip_path"])
        if not os.path.exists(vid):
            print(f"  [skip] 缺视频: {vid}")
            continue
        label = 1 if r["label"].strip().lower() == "fake" else 0
        split = r["split"].strip()
        stem = os.path.splitext(os.path.basename(vid))[0]
        out_dir = os.path.join(PREP, split, stem)
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
        identity = r["dir_generator"].strip() or r["true_generator"].strip()
        pack_cache(stem, os.path.join(PREP, split),
                   audio_seq=audio_seq, v_clip=v_clip,
                   v_lip=np_stack(v_lip),
                   label=label, identity=identity)
        manifest.append({"stem": stem, "split": split, "label": label,
                         "dir_generator": r["dir_generator"].strip(),
                         "true_generator": r["true_generator"].strip()})
        prepped += 1
        n_real += (label == 0)
        n_fake += (label == 1)
    write_manifest(manifest, PREP, "manifest.jsonl")
    print(f"[mvad] prepped {prepped} clips ({n_real} real / {n_fake} fake)")


if __name__ == "__main__":
    main()
