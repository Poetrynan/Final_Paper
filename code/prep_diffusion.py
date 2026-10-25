"""
[OBSOLETE — 2026-08-02 标记废弃] Prep 扩散基准 (LivePortrait / Hallo2) —— 不再使用。
================================================================
本脚本对应早期"自建 diffusion benchmark"方案，已被 **TalkingHeadBench (THB)** 取代
（论文 Ch4 §dataset-crossdomain；脚本见 prep_thb.py）。THB 提供 Hallo/Hallo2/
LivePortrait/AniPortrait/EMOPortraits + unseen Hallo3/MAGI-1，并开源全部划分与协议，
无需自建。保留本文件仅为历史记录，**勿运行、勿引用**。
来源 : 最新 diffusion-based avatar / lip-sync 合成范式。
状态 : draft 注明 subject to licence approval；授权批准后才纳入。
下载 : TODO 视授权范围而定（可能是官方 demo 视频子集，不是完整集）。
       按【授权范围/子集】取，不下载未授权内容。
输入 : <DIFFUSION_ROOT>/<engine>/<clip>.mp4
输出 : <DIFFUSION_ROOT>/prep/<engine>/<stem>/cache.pt
流程 : 抽帧 -> 抽音频 -> (WIRE) 特征 -> 半精度缓存  (label=1, 全部伪造)
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(__file__))
from common import (
    iter_videos, sample_frames, clip_center_window, save_frames,
    video_to_wav, load_audio_tensor,
    extract_clip_frame, extract_lip_landmarks, extract_lip_landmarks_seq, extract_audio_seq,
    pack_cache, write_manifest,
)
import numpy as np

ROOT = r"D:/DiffusionBenchmark"   # TODO: 授权后建目录
PREP = r"D:/DiffusionBenchmark/prep"
TARGET_FRAMES = 50
WINDOW_SEC = 4.0         # 中心窗时长（秒），对齐训练域窗口；短于窗的片段用全部帧
WINDOW_FPS = 25.0        # 窗内采样率
SR = 16000


def np_stack(xs):
    return np.stack([np.asarray(x) for x in xs], axis=0)


def main():
    engines = sorted(d for d in glob.glob(os.path.join(ROOT, "*"))
                   if os.path.isdir(d) and os.path.basename(d) != "prep")
    if not engines:
        print("[diffusion] 未找到任何 engine 目录 —— 授权未批或 ROOT 未建。跳过。")
        return
    manifest = []
    for eng_dir in engines:
        engine = os.path.basename(eng_dir)
        vids = iter_videos(eng_dir)
        for vid in vids:
            stem = os.path.splitext(os.path.basename(vid))[0]
            out_dir = os.path.join(PREP, engine, stem)
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
            v_clip = np_stack([extract_clip_frame(f) for f in frames])
            _, v_lip = zip(*extract_lip_landmarks_seq(frames))
            audio_seq = extract_audio_seq(wav_t)
            pack_cache(stem, os.path.join(PREP, engine),
                       audio_seq=audio_seq, v_clip=v_clip,
                       v_lip=np_stack(v_lip),
                       label=1, identity=f"{engine}/{stem}")
            manifest.append({"stem": stem, "engine": engine, "label": 1})
    write_manifest(manifest, PREP, "manifest.jsonl")
    print(f"[diffusion] prepped {len(manifest)} clips")


if __name__ == "__main__":
    main()
