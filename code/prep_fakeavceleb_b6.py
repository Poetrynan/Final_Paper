"""
Prep FakeAVCeleb for B6 (leading-silence 捷径消融) —— 仅分析，不参训。

修正（2026-08-02）：B6 捷径只存在于 **fake-audio 子集**（RVFA=RealVideo-FakeAudio,
FVFA=FakeVideo-FakeAudio，由 TTS 引入前导静音偏置）。real-audio 子集（RVRA 全真、
FVRA=wavtolip 假画+真音频）无此偏置，FVRA 显式排除。

磁盘来源（只读，原始目录由 SYNCX_DATA_ROOT 覆盖，默认 D:/Datasets/FakeAVCeleb_v1.2 完全不动）：
    RealVideo-RealAudio/   (RVRA, 500, 全真 -> label 0, 对照, 无 trim)
    RealVideo-FakeAudio/   (RVFA, 500, 假音频 -> label 1, 有偏置)
    FakeVideo-RealAudio/   (FVRA, 9709, wavtolip 真音频 -> 排除, 无偏置)
    FakeVideo-FakeAudio/   (FVFA, 10835, 假音频 -> label 1, 有偏置)
    D:/Datasets/FakeAVCeleb_v1.2_split/b6_std/trimmed/{RVFA,FVFA}/...
         (由 b6_std_trim.py 按 Smeu 协议逐片段动态 trim 产出; RVRA 不 trim)

B6 条件（供 evaluate.py 选择；扁平 *.pt 缓存，匹配 FeatureCacheDataset）：
    prep/b6_raw/      : RVRA(real) + RVFA(fake) + FVFA(fake) 的未裁原片
    prep/b6_trimmed/  : RVRA(real, 等同 raw) + RVFA(fake) + FVFA(fake) 的 Smeu 裁片
    （real 无前导静音, trimmed==raw, 故 real 在两目录内容相同）

输出 : <SYNCX_DATA_ROOT>/FakeAVCeleb_v1.2_split/prep/b6_{raw,trimmed}/<stem>.pt
流程 : 抽帧 -> 抽音频 -> (WIRE) 特征 -> 半精度扁平缓存
"""
import os, sys, glob, json
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from common import (
    sample_frames, save_frames,
    video_to_wav, load_audio_tensor,
    extract_clip_frames, extract_lip_landmarks, extract_lip_landmarks_seq, extract_audio_seq,
)
from config import DATA_ROOT, DATASET_PATHS

ROOT = DATASET_PATHS["fakeavceleb"]                 # FakeAVCeleb_v1.2_split
SRC = os.path.join(DATA_ROOT, "FakeAVCeleb_v1.2")              # 原始（raw 只读，不动）；云端经 SYNCX_DATA_ROOT 覆盖
B6_STD_TRIM = os.path.join(ROOT, "b6_std", "trimmed")   # RVFA/, FVFA/
PREP = os.path.join(ROOT, "prep")
RAW_OUT = os.path.join(PREP, "b6_raw")
TRIM_OUT = os.path.join(PREP, "b6_trimmed")
TARGET_FRAMES = 100
SR = 16000

# fake-audio 子集（带偏置）；RAW 从原始目录按全名读取
FAKE_MODS = ["RealVideo-FakeAudio", "FakeVideo-FakeAudio"]
# TRIMMED 子目录名对齐 b6_std_trim.py 写出的短键（RVFA/FVFA）
TRIM_KEYS = ["RVFA", "FVFA"]
REAL_MOD = "RealVideo-RealAudio"                   # RVRA 对照


def np_stack(xs):
    return np.stack([np.asarray(x) for x in xs], axis=0)


def identity_of(vid):
    """idN：文件两级之上的目录名（.../men/id00076/xxx.mp4 -> id00076）。"""
    return os.path.basename(os.path.dirname(os.path.dirname(vid)))


def extract_features(vid):
    """抽帧+音频+特征（依赖 common.py 的 extract_*，192-D 唇点契约 / 64 口周点），返回 dict 或 None。"""
    frames = sample_frames(vid, target_count=TARGET_FRAMES)
    if not frames:
        return None
    wav = os.path.join(PREP, "_tmp_audio.wav")
    if not video_to_wav(vid, wav, sr=SR):
        return None
    wav_t = load_audio_tensor(wav, sr=SR)
    v_clip = extract_clip_frames(frames)
    _, v_lip = zip(*extract_lip_landmarks_seq(frames))
    audio_seq = extract_audio_seq(wav_t)
    return {
        "audio_seq": np.asarray(audio_seq),
        "v_clip": v_clip,
        "v_lip": np_stack(v_lip),
    }


def save_flat(feats, out_path, label, identity):
    d = {
        "audio_seq": torch.as_tensor(np.asarray(feats["audio_seq"])).half(),
        "v_clip": torch.as_tensor(np.asarray(feats["v_clip"])).half(),
        "v_lip": torch.as_tensor(np.asarray(feats["v_lip"])).half(),
        "label": int(label),
        "identity": str(identity),
        "source_id": "",
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(d, out_path)
    return out_path


def main():
    os.makedirs(RAW_OUT, exist_ok=True)
    os.makedirs(TRIM_OUT, exist_ok=True)
    manifest = []

    # ── real (RVRA) : label 0；raw 与 trimmed 内容相同（无静音）──
    real_vids = sorted(glob.glob(os.path.join(SRC, REAL_MOD, "**", "*.mp4"), recursive=True))
    print(f"[b6] real (RVRA): {len(real_vids)}")
    for vid in real_vids:
        stem = os.path.splitext(os.path.basename(vid))[0]
        feats = extract_features(vid)
        if feats is None:
            continue
        save_flat(feats, os.path.join(RAW_OUT, stem + ".pt"), 0, identity_of(vid))
        save_flat(feats, os.path.join(TRIM_OUT, stem + ".pt"), 0, identity_of(vid))
        manifest.append({"stem": stem, "split": "real", "label": 0, "cond": "raw"})
    n_real = len(real_vids)

    # ── fake raw (RVFA+FFVA) : label 1, cond=raw ──
    n_raw = 0
    for mod in FAKE_MODS:
        vids = sorted(glob.glob(os.path.join(SRC, mod, "**", "*.mp4"), recursive=True))
        for vid in vids:
            stem = os.path.splitext(os.path.basename(vid))[0]
            feats = extract_features(vid)
            if feats is None:
                continue
            save_flat(feats, os.path.join(RAW_OUT, stem + ".pt"), 1, identity_of(vid))
            manifest.append({"stem": stem, "split": "fake", "label": 1, "cond": "raw"})
            n_raw += 1
    print(f"[b6] fake raw: {n_raw}")

    # ── fake trimmed (RVFA+FVFA 的 Smeu 裁片) : label 1, cond=trimmed ──
    #    子目录名对齐 b6_std_trim.py 输出的短键（b6_std/trimmed/RVFA, /FVFA）
    n_trim = 0
    for key in TRIM_KEYS:
        vids = sorted(glob.glob(os.path.join(B6_STD_TRIM, key, "**", "*.mp4"), recursive=True))
        for vid in vids:
            stem = os.path.splitext(os.path.basename(vid))[0]
            feats = extract_features(vid)
            if feats is None:
                continue
            save_flat(feats, os.path.join(TRIM_OUT, stem + ".pt"), 1, identity_of(vid))
            manifest.append({"stem": stem, "split": "fake", "label": 1, "cond": "trimmed"})
            n_trim += 1
    print(f"[b6] fake trimmed: {n_trim}")

    with open(os.path.join(PREP, "manifest_b6.jsonl"), "w", encoding="utf-8") as f:
        for r in manifest:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[b6] done: {len(manifest)} clips "
          f"({n_real} real / {n_raw} fake-raw / {n_trim} fake-trimmed)")


if __name__ == "__main__":
    main()
