"""
diag_clip_duration.py — 统计各数据集视频片段时长分布
================================================================
背景：evidence_chain §9 发现 pipeline 级不一致——训练域 prep_deepspeak 对音频/视频
做"中心 4s 窗 + 音画对齐"(clip_center_window + sample_frames(window_sec=4) +
video_to_wav(ss,td))，而跨域评测集 prep_hifi/fakeavceleb/eval2024/diffusion 均未对齐：
  - 视频 sample_frames(target_count=100, window_sec=None) → 整段均匀抽 100 帧
  - 音频 video_to_wav(无 ss/td)          → 取【前 4 秒】(load_audio_tensor 卷到 64000=4s)

对片段时长 >4s 的视频，音频只覆盖前 4s、视频覆盖全程 → 音画时间轴错位，
sync_t / cross-attn / LTCA 在音画序列间算一致性时信号被破坏。本脚本量化该错位影响面。

判读口径：
  时长 <= WINDOW_SEC(4.0)：音频整段 / 视频整段 → 基本对齐，错位可忽略。
  时长 >  WINDOW_SEC(4.0)：视频取全程、音频只取前 4s → 错位，一致性特征受损。
于是 ">4s 视频占比" 即为错位影响面（越高越需要重抽固定）。

用法：
  python diag_clip_duration.py \
      --data-root /root/autodl-tmp/SyncX/Datasets \
      --sets hifi fakeavceleb dfe2024 deepspeak \
      [--max-per-set N] [--window 4.0]
若 --data-root 缺省，默认取 config.DATA_ROOT 绝对路径（云端）。
输出逐集：总视频数 / ≤window 数 / >window 数 与占比 / 时长 min-p25-med-p75-max / 均值。
"""
import os, sys, argparse, glob, time
import numpy as np

DATA_ROOT_DEFAULT = None
try:
    from config import DATA_ROOT
    DATA_ROOT_DEFAULT = DATA_ROOT
except Exception:
    pass

# 各数据集在 DATA_ROOT 下对应的根目录 + 是否含 prep 缓存子目录（需剔除，只走原始视频）
#   deepspeak: DISP 的是 v1.1 clips（训练域，已对齐，作对照）
#   其余 = 跨域评测集（本脚本要测的错位对象）
SET_DIRS = {
    "deepspeak":   "DeepSpeak_v1_1/clips",
    "fakeavceleb": "FakeAVCeleb_v1.2_split",
    "dfe2024":     "Deepfake-Eval-2024",
    "hifi":        "HiFi-AVDF",
}
# 这些子目录名是 prep 输出（cache.pt / frames / audio.wav / prep / manifest*），须跳过
PREP_SUBDIRS = {"prep", "frames"}


def probe_durations(root, window):
    """遍历 root 下所有 mp4/mov/mkv/webm/avi 的原始视频，读时长(秒)。跳过 prep/缓存目录。"""
    import cv2
    durations = []
    n_skip = 0
    for dirpath, dirnames, filenames in os.walk(root):
        # 剔除 prep 输出子目录（不进递归）
        dirnames[:] = [d for d in dirnames if d not in PREP_SUBDIRS and not d.startswith("manifest")]
        for fn in filenames:
            if not fn.lower().endswith((".mp4", ".mov", ".mkv", ".webm", ".avi")):
                continue
            p = os.path.join(dirpath, fn)
            if not os.path.isfile(p):
                continue
            cap = cv2.VideoCapture(p)
            if not cap.isOpened():
                n_skip += 1
                cap.release()
                continue
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps and fps > 0 and n > 0:
                durations.append(n / fps)
            else:
                n_skip += 1
    return np.array(durations), n_skip


def summarize(name, root, window, max_per):
    if not os.path.isdir(root):
        print(f"[{name}] ❌ 目录不存在: {root}  （跳过）")
        return
    t0 = time.time()
    d, n_skip = probe_durations(root, window)
    if max_per and len(d) > max_per:
        rng = np.random.default_rng(0)
        d = rng.choice(d, size=max_per, replace=False)
    if len(d) == 0:
        print(f"[{name}] 未找到可读视频（skip={n_skip}）  root={root}")
        return
    over = d > window
    n_over = int(over.sum())
    n_total = len(d)
    pct = 100.0 * n_over / n_total
    q = np.percentile(d, [0, 25, 50, 75, 100])
    print(f"[{name}] ── root={root}")
    print(f"  总视频 = {n_total}   可读失败 = {n_skip}")
    print(f"  >{window:.1f}s(错位) : {n_over} / {n_total}  = {pct:.1f}%   ← 错位影响面")
    print(f"  时长s: min={q[0]:.2f}  p25={q[1]:.2f}  med={q[2]:.2f}  p75={q[3]:.2f}  max={q[4]:.2f}  mean={d.mean():.2f}  ({time.time()-t0:.0f}s)")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DATA_ROOT_DEFAULT,
                    help="数据集根目录（缺省 = config.DATA_ROOT）")
    ap.add_argument("--sets", nargs="+", default=["hifi", "fakeavceleb", "dfe2024", "deepspeak"],
                    help="要统计的数据集名（keys of SET_DIRS）")
    ap.add_argument("--max-per-set", type=int, default=None, help="每集随机采样上限(调试快跑)")
    ap.add_argument("--window", type=float, default=4.0, help="时间窗秒(默认4.0，应=训练域对齐窗)")
    args = ap.parse_args()

    if not args.data_root:
        print("!! 未传入 --data-root 且 config.DATA_ROOT 不可用，请显式指定")
        sys.exit(1)
    print(f"数据根 = {args.data_root}   时间窗 = {args.window:.1f}s")
    print("=" * 60)
    for name in args.sets:
        sub = SET_DIRS.get(name)
        if sub is None:
            print(f"[{name}] 未知集合名（可选: {list(SET_DIRS.keys())}）")
            continue
        summarize(name, os.path.join(args.data_root, sub), args.window, args.max_per_set)
    print("=" * 60)
    print("判读(注意): >4s占比高 ≠ 一定错位。真正决定错位的是【prep 是否对原始视频")
    print("          做窗口截断】：训练域 deepspeak 传了 window_sec=4.0(中心4s窗)与")
    print("          video_to_wav(ss,td) → 音频视频都取中心4s、对齐；评测集 hifi/dfe/fakeav")
    print("          的 prep 用 window_sec=None(整段抽帧) + 无 ss/td(音频取前4s) → 长视频上")
    print("          音频只覆盖前4s、视频覆盖全程 → 错位。故"多长>window"是对错位需要的输入，")
    print("          最终判据看对应 prep 是否传窗口参数(见证据链 §9)。")


if __name__ == "__main__":
    main()