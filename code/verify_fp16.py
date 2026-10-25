"""Sync-X — FP16 推理对照验证脚本。

目的：确认冻结编码器（CLIP ViT-B/16 / wav2vec2-base-960h）从 fp32 切换到 fp16
推理后，抽出的特征与 fp32 版「数值一致」（余弦相似度 >0.999）且「无 NaN」，
从而可以安全把默认推理精度改为 fp16（提速 ~30–50%、显存↓50%）。

做法：
  - 复用 common.sample_frames(4s@25fps=100 帧) + video_to_wav + load_audio_tensor，
    得到与正式 prep 管线完全一致的帧序列与音频（同一中心窗）。
  - 同一份帧/音频，分别用 fp32 与 fp16 编码器抽 v_clip / audio_seq。
  - v_lip 来自 MediaPipe（CPU，与 torch dtype 无关），作参照（预期 cos=1.0）。
  - 逐模态打印 fp32-vs-fp16 的 mean/min 余弦相似度 + NaN 计数。
  - 闸门：三模态 mean cos > 0.999 且 NaN 数 == 0 → PASS。

用法：
  python verify_fp16.py --n 8
  python verify_fp16.py --root D:/Datasets/DeepSpeak_v1_1/clips --n 8 --device cuda
"""
from __future__ import annotations
import argparse
import os
import sys
import tempfile

import numpy as np

# 让脚本能 import 同目录的 common / config
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import common  # noqa: E402
import torch    # noqa: E402


def _collect_videos(root: str, n: int):
    """递归收集 root 下前 n 个 .mp4（跳过 macOS AppleDouble `._*`）。"""
    out = []
    for dp, _, fns in os.walk(root):
        for fn in sorted(fns):
            if fn.lower().endswith(".mp4") and not fn.startswith("._"):
                out.append(os.path.join(dp, fn))
                if len(out) >= n:
                    return out
    return out


def _cosine_rows(a: np.ndarray, b: np.ndarray):
    """逐行余弦相似度，返回 (mean, min)。a,b: (M, D)。"""
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    sim = (a * b).sum(1)
    return float(sim.mean()), float(sim.min())


def _nan_count(arr: np.ndarray) -> int:
    return int(np.isnan(arr).sum())


def main():
    ap = argparse.ArgumentParser(description="Sync-X FP16 推理对照验证")
    ap.add_argument("--root", default=None,
                    help="视频根目录；默认 = config.DATA_ROOT/DeepSpeak_v1_1/clips")
    ap.add_argument("--n", type=int, default=8, help="抽取视频数量")
    ap.add_argument("--device", default="cuda", help="推理设备 cuda / cpu")
    ap.add_argument("--window-sec", type=float, default=4.0, help="中心窗秒数")
    ap.add_argument("--window-fps", type=float, default=25.0, help="抽帧率")
    ap.add_argument("--target-count", type=int, default=100, help="目标帧数 (VIS_LEN)")
    args = ap.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    print(f"[verify] device={device}  window={args.window_sec}s@{args.window_fps}fps"
          f"  target_count={args.target_count}", flush=True)

    # 解析默认 root
    if args.root is None:
        try:
            import config
            # DATA_ROOT / DATASET_PATHS 是模块级常量（非 Config 属性）
            args.root = config.DATASET_PATHS.get("deepspeak") or os.path.join(
                config.DATA_ROOT, "DeepSpeak_v1_1", "clips")
        except Exception as e:
            print(f"[verify] 无法从 config 解析默认 root ({e})；请用 --root 指定视频目录", flush=True)
            sys.exit(2)
    print(f"[verify] video root = {args.root}", flush=True)

    videos = _collect_videos(args.root, args.n)
    if not videos:
        print(f"[verify] 在 {args.root} 下未找到 .mp4，退出", flush=True)
        sys.exit(2)
    print(f"[verify] 选中 {len(videos)} 个视频做 fp32 vs fp16 对照", flush=True)

    # 构造两组编码器（显式设置全局推理精度后直接实例化，绕过单例缓存）
    common.set_infer_dtype("float32")
    enc_v32 = common._VisualEncoder(device)
    enc_a32 = common._AudioEncoder(device)
    common.set_infer_dtype("float16")
    enc_v16 = common._VisualEncoder(device)
    enc_a16 = common._AudioEncoder(device)
    print(f"[verify] 已构造 fp32 / fp16 编码器对；fp16 模式 "
          f"clip.dtype={enc_v16.dtype}  wav2vec2.dtype={enc_a16.dtype}", flush=True)

    tmpd = tempfile.mkdtemp(prefix="syncx_verify_")

    agg = {"v_clip": [], "audio_seq": [], "v_lip": []}
    nan_agg = {"v_clip": 0, "audio_seq": 0, "v_lip": 0}

    for vi, vpath in enumerate(videos):
        try:
            frames = common.sample_frames(
                vpath, target_count=args.target_count,
                window_sec=args.window_sec, window_fps=args.window_fps)
            if not frames:
                print(f"  [{vi+1}/{len(videos)}] SKIP（无帧）{os.path.basename(vpath)}", flush=True)
                continue

            # v_lip：MediaPipe CPU，抽 fp32 版（无需音频）
            v_lip32_list = [enc_v32.encode_lip(f)[1] for f in frames]  # (N,192)
            v_lip32 = np.stack(v_lip32_list, 0)

            # v_clip：fp32 vs fp16 同一帧
            v_clip32 = enc_v32.encode_clip_batch(frames)
            v_clip16 = enc_v16.encode_clip_batch(frames)

            # audio_seq：fp32 vs fp16 同一波形（独立 try，音频失败不中断视觉对比）
            a32 = a16 = None
            try:
                wav_path = os.path.join(tmpd, f"v{vi}.wav")
                ss, td = common.clip_center_window(vpath, args.window_sec, args.window_fps)
                ok = common.video_to_wav(vpath, wav_path, sr=16000, ss=ss, td=td)
                if ok:
                    wav_tensor = common.load_audio_tensor(wav_path)
                    a32 = enc_a32.encode(wav_tensor)
                    a16 = enc_a16.encode(wav_tensor)
            except Exception as e:
                print(f"    [audio] {os.path.basename(vpath)} 抽取失败：{e}", flush=True)

            # v_lip：fp32 vs fp16（MediaPipe CPU，预期 cos≈1.0，证明与 dtype 无关）
            v_lip16_list = [enc_v16.encode_lip(f)[1] for f in frames]
            v_lip16 = np.stack(v_lip16_list, 0)
            c_vl_mean, c_vl_min = _cosine_rows(v_lip32, v_lip16)
            n_vl = _nan_count(v_lip16)
            agg["v_lip"].append((c_vl_mean, c_vl_min))
            nan_agg["v_lip"] += n_vl

            # v_clip：剔除「无人脸检测→零向量」的退化帧后再比
            # （退化帧两边输入都是零裁图，CLIP 输出近似偏置向量，对比无意义，会污染闸门）
            valid = ~np.all(v_clip32 == 0, axis=1)
            n_valid = int(valid.sum())
            if n_valid < 10:
                print(f"  [{vi+1}/{len(videos)}] {os.path.basename(vpath)} | "
                      f"v_clip 有效帧仅 {n_valid}/{len(frames)}（大量无人脸），跳过该视频对比", flush=True)
                continue
            c_vc_mean, c_vc_min = _cosine_rows(v_clip32[valid], v_clip16[valid])
            n_vc = _nan_count(v_clip16[valid])
            agg["v_clip"].append((c_vc_mean, c_vc_min))
            nan_agg["v_clip"] += n_vc

            if a32 is not None and a16 is not None:
                c_a_mean, c_a_min = _cosine_rows(a32, a16)
                n_a = _nan_count(a16)
                agg["audio_seq"].append((c_a_mean, c_a_min))
                nan_agg["audio_seq"] += n_a
                astr = f"audio mean={c_a_mean:.5f} min={c_a_min:.5f} nan={n_a}"
            else:
                astr = "audio SKIPPED（无音频/ffmpeg 失败）"

            print(f"  [{vi+1}/{len(videos)}] {os.path.basename(vpath)} | "
                  f"v_clip valid={n_valid} mean={c_vc_mean:.5f} min={c_vc_min:.5f} nan={n_vc} | "
                  f"v_lip cos={c_vl_mean:.5f} nan={n_vl} | {astr}", flush=True)
        except Exception as e:
            print(f"  [{vi+1}/{len(videos)}] ERROR {os.path.basename(vpath)}: {e}", flush=True)
            continue

    # 汇总
    print("\n==================== FP16 对照汇总 ====================", flush=True)
    def _report(name):
        rows = agg[name]
        if not rows:
            print(f"  {name:10s}: 无数据", flush=True)
            return None
        means = [r[0] for r in rows]
        mins = [r[1] for r in rows]
        print(f"  {name:10s}: 视频数={len(rows)} | mean余弦 "
              f"avg={np.mean(means):.6f} min={np.min(means):.6f} | "
              f"worst单帧min={np.min(mins):.6f} | NaN={nan_agg[name]}", flush=True)
        return np.mean(means), np.min(mins), nan_agg[name]

    r_vc = _report("v_clip")
    r_a = _report("audio_seq")
    r_vl = _report("v_lip")

    # 闸门
    pass_vc = r_vc is not None and r_vc[0] > 0.999 and r_vc[2] == 0
    pass_a = r_a is None or (r_a[0] > 0.999 and r_a[2] == 0)   # 音频缺失不判 FAIL
    pass_vl = r_vl is None or (r_vl[2] == 0)
    passed = pass_vc and pass_a and pass_vl

    print("------------------------------------------------------", flush=True)
    print(f"  闸门：v_clip>0.999且无NaN={pass_vc} | "
          f"audio_seq>0.999且无NaN={pass_a} | v_lip无NaN={pass_vl}", flush=True)
    if passed:
        print("  ✅ PASS：FP16 推理与 fp32 数值一致（cos>0.999 且无 NaN），"
              "可将默认推理精度改为 FP16。", flush=True)
        print("  操作：将 common.py 的 _INFER_DTYPE_STR 默认改为 'float16'，"
              "或运行 prep 时设 SYNCX_INFER_FP16=1。", flush=True)
    else:
        print("  ❌ FAIL：FP16 与 fp32 偏差过大或出现 NaN，暂不要默认开启 FP16。", flush=True)
        print("  排查：检查 wav2vec2 fp16 是否溢出（NaN）；或保持 fp32。", flush=True)
    print("======================================================", flush=True)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
