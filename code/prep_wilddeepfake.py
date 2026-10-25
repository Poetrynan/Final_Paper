"""
[OBSOLETE — 2026-08-02 标记废弃] Prep WildDeepfake —— 不再接入评测 lineup。
================================================================
WildDeepfake **不在** Sync-X 最终数据集阵容中（论文 Ch4 无此集；工作区已决定不接入）。
保留本文件仅为历史记录，**勿运行、勿引用**。原始数据驻留 D:/WildDeepfake（未迁移）。
输入 : D:/Datasets/WildDeepfake/deepfake_in_the_wild/{fake_test,real_test}/*.tar.gz  (后台下载 task El9JRB 完成后再跑)
输出 : D:/Datasets/WildDeepfake/prep/{fake,real}/<stem>/cache.pt + frames/
流程 : 解包 -> 抽帧 -> 抽音频 -> (WIRE) 特征 -> 半精度缓存
注意 : WildDeepfake 只下 test 划分(~9.25GB)，不参训，只用于评测。
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from common import (
    unpack_targz, iter_videos, sample_frames, save_frames,
    video_to_wav, load_audio_tensor,
    extract_clip_frame, extract_lip_landmarks, extract_lip_landmarks_seq, extract_audio_seq,
    pack_cache, write_manifest,
)

RAW = r"D:/Datasets/WildDeepfake"
UNPACKED = r"D:/Datasets/WildDeepfake/unpacked"
PREP = r"D:/Datasets/WildDeepfake/prep"
TARGET_FRAMES = 50
SR = 16000


def main():
    # 1) 解包 test 划分
    for split in ["fake_test", "real_test"]:
        unpack_targz(os.path.join(RAW, "deepfake_in_the_wild", split),
                     os.path.join(UNPACKED, split), remove_after=False)

    manifest = []
    for split, label in [("fake_test", 1), ("real_test", 0)]:
        vids = iter_videos(os.path.join(UNPACKED, split))
        print(f"[prep] {split}: {len(vids)} videos, label={label}")
        for vid in vids:
            stem = os.path.splitext(os.path.basename(vid))[0]
            out_dir = os.path.join(PREP, split, stem)
            # 2) 抽帧
            frames = sample_frames(vid, target_count=TARGET_FRAMES)
            n = save_frames(frames, os.path.join(out_dir, "frames"))
            if n == 0:
                print(f"  skip (no frames): {vid}")
                continue
            # 3) 音频
            wav = os.path.join(out_dir, "audio.wav")
            if not video_to_wav(vid, wav, sr=SR):
                print(f"  skip (no audio): {vid}")
                continue
            wav_t = load_audio_tensor(wav, sr=SR)
            # 4) WIRE 特征
            v_clip = np_stack([extract_clip_frame(f) for f in frames])
            _, v_lip = zip(*extract_lip_landmarks_seq(frames))
            audio_seq = extract_audio_seq(wav_t)
            # 5) 缓存
            pack_cache(stem, os.path.join(PREP, split),
                       audio_seq=audio_seq, v_clip=v_clip,
                       v_lip=np_stack(v_lip),
                       label=label, identity=stem)
            manifest.append({"stem": stem, "split": split, "label": label})


def np_stack(xs):
    import numpy as np
    return np.stack([np.asarray(x) for x in xs], axis=0)


if __name__ == "__main__":
    main()
