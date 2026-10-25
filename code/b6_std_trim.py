"""
b6_std_trim.py — 按 Smeu et al. (CVPR 2025) 原文协议重做 FakeAVCeleb B6 标准 trim
================================================================================
出处核实（2026-08-02，`papers/_smeu_full.txt` §3.2–3.3）：
  Smeu 的 trim 协议是**动态检测每段自己的静音**，不是固定 N ms：
    "For the trimmed version, we discard the beginning of videos such that each
     of them starts with the first non-silent segment. We find the duration to
     discard by first computing the leading silence (using the same 5e-4 threshold
     on the magnitude) and then rounding it up to the nearest multiple of 1/FPS
     (the reciprocal of the video's frame rate); this rounding ensures that the
     audio and video channels remain synchronized."
  → 幅度阈值 τ = 5e-4；裁到第一段非静音；按 1/FPS 向上取整（25fps → 40ms）。

**关键修正（2026-08-02）**：本机 FakeAVCeleb 的前导静音偏置**只存在于 fake-audio
子集**（RVFA=RealVideo-FakeAudio, FVFA=FakeVideo-FakeAudio），由 TTS 音频管线引入；
real-audio 子集（RVRA=RealVideo-RealAudio, FVRA=FakeVideo-RealAudio=wavtolip）音频是
真人原声，**无偏置**。先前建的 FakeAVCeleb_v1.2_split/B6 误用了 wavtolip(FVRA，
real audio) → 无偏置，故 B6 子集选错。本脚本改用正确子集（见下）。

输入（只读，绝不修改原始目录）：
  D:/Datasets/FakeAVCeleb_v1.2/RealVideo-FakeAudio/   (RVFA, 500, 假音频->有偏置)
  D:/Datasets/FakeAVCeleb_v1.2/FakeVideo-FakeAudio/   (FVFA, 10835, 假音频->有偏置)
  D:/Datasets/FakeAVCeleb_v1.2/RealVideo-RealAudio/  (RVRA, 500, 全真->对照, 无偏置)
  （FVRA=FakeVideo-RealAudio 显式排除：其音频为 wavtolip 真人原声，无此 shortcut）

输出（新目录，不覆盖任何现有数据；原始目录 D:/Datasets/FakeAVCeleb_v1.2 完全不动）：
  D:/Datasets/FakeAVCeleb_v1.2_split/b6_std/
      raw/      <MOD>/<rel>.mp4   (fake raw 的副本；real 直接登记原路径)
      trimmed/  <MOD>/<rel>_smeu_trimmed.mp4  (按 Smeu 协议裁好的 fake)
      report_smeu_trim.jsonl (每段的 lead/trail/决策/输出，含 1/FPS 对齐)

说明：
  - 论文 Ch4 (B6 protocol) 描述：严格按 Smeu 协议仅移除 leading silence（Smeu 文档化的 cue），
    不裁 trailing（Smeu 原法只去开头）；裁掉的长度按 1/FPS 向上取整保证音画同步。
  - RVRA(全真) 不作 trim，仅登记为 untrimmed control（Smeu: real 开头即非静音）。
  - all_silence（整段低于阈值）丢弃。
  - 检测方式：一次 ffmpeg 解码整段为 16k 单声道 f32，numpy 找首个/末个 >= τ 样本。
"""
import os, sys, json, glob, subprocess
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_ROOT

SRC = os.path.join(DATA_ROOT, "FakeAVCeleb_v1.2")          # 只读原始目录；云端经 SYNCX_DATA_ROOT 覆盖
OUT_ROOT = os.path.join(DATA_ROOT, "FakeAVCeleb_v1.2_split", "b6_std")   # 输出到 split 目录，不动原始

# fake-audio 子集（带前导静音偏置）
FAKE_MODS = {
    "RVFA": "RealVideo-FakeAudio",
    "FVFA": "FakeVideo-FakeAudio",
}
# 全真对照（无偏置，不 trim）
REAL_MOD = ("RVRA", "RealVideo-RealAudio")

RAW_OUT = os.path.join(OUT_ROOT, "raw")
TRIM_OUT = os.path.join(OUT_ROOT, "trimmed")
REPORT = os.path.join(OUT_ROOT, "report_smeu_trim.jsonl")

SR = 16000
THRESHOLD = 5e-4          # Smeu τ
FPS = 25.0                # FakeAVCeleb 25fps → 40ms 对齐
FRAME_MS = int(round(1000.0 / FPS))   # 40
WIN = int(SR * 0.02)      # 20ms 窗


def decode_audio(path):
    """解码整段为 16k 单声道 f32 numpy。返回 (x, duration_sec)。"""
    cmd = ["ffmpeg", "-v", "quiet", "-i", path, "-ar", str(SR), "-ac", "1",
           "-f", "f32le", "-"]
    r = subprocess.run(cmd, capture_output=True)
    x = np.frombuffer(r.stdout, dtype=np.float32)
    return x, len(x) / SR


def detect_silences(x, sr=SR, threshold=THRESHOLD):
    """返回 (lead_sec, trail_sec)：开头/结尾静音时长（连续低于阈值的部分）。"""
    amp = np.abs(x)
    non_sil = np.nonzero(amp >= threshold)[0]
    if len(non_sil) == 0:
        return len(x) / sr, 0.0          # 整段静音
    first = int(non_sil[0])
    last = int(non_sil[-1])
    lead = first / sr
    trail = (len(x) - 1 - last) / sr
    return lead, trail


def _duration(path):
    r = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries",
                        "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def trim_video(path, out_path, lead_align):
    """严格 Smeu：仅裁掉开头 lead_align 秒（1/FPS 向上取整），不裁结尾 trailing。"""
    dur = _duration(path)
    keep_dur = max(0.0, dur - lead_align)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "quiet", "-ss", f"{lead_align:.6f}",
           "-i", path, "-t", f"{keep_dur:.6f}",
           "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
           "-c:a", "aac", "-b:a", "128k", out_path]
    subprocess.run(cmd, check=True)
    return lead_align


def main():
    os.makedirs(RAW_OUT, exist_ok=True)
    os.makedirs(TRIM_OUT, exist_ok=True)
    report = []

    # ── fake-audio 子集：RVFA + FVFA（带偏置，需 trim）──
    n_trimmed = n_nosil = n_all_sil = n_fail = 0
    for mod, d in FAKE_MODS.items():
        dpath = os.path.join(SRC, d)
        fake_raw = sorted(glob.glob(os.path.join(dpath, "**", "*.mp4"), recursive=True))
        print(f"[b6_std][{mod}] fake raw: {len(fake_raw)}")
        for i, vid in enumerate(fake_raw, 1):
            stem = os.path.splitext(os.path.basename(vid))[0]
            rel = os.path.relpath(vid, dpath)   # African/men/id00076/xxx.mp4
            rec = {"mod": mod, "path": vid, "stem": stem, "rel": rel,
                   "lead": None, "trail": None, "decision": None, "out_path": None}
            try:
                x, _ = decode_audio(vid)
                if len(x) == 0:
                    rec["decision"] = "fail"; rec["err"] = "no audio"
                    n_fail += 1; report.append(rec); continue
                lead, trail = detect_silences(x)
                rec["lead"] = round(lead, 4); rec["trail"] = round(trail, 4)
                if len(np.nonzero(np.abs(x) >= THRESHOLD)[0]) == 0:
                    rec["decision"] = "all_silence"; n_all_sil += 1
                    report.append(rec); continue
                # 严格 Smeu：仅按 leading silence 裁；trailing 不裁
                lead_align = int(np.ceil(lead * 1000 / FRAME_MS)) * FRAME_MS / 1000.0
                rec["lead_align_ms"] = int(lead_align * 1000)
                out_stem = stem + "_smeu_trimmed"
                out_rel = os.path.join(TRIM_OUT, mod, os.path.dirname(rel), out_stem + ".mp4")
                if lead_align > 0:
                    trim_video(vid, out_rel, lead_align)
                    rec["decision"] = "trimmed"; n_trimmed += 1
                else:
                    rec["decision"] = "no_silence"; n_nosil += 1
                rec["out_path"] = out_rel if rec["decision"] == "trimmed" else None
            except Exception as e:
                rec["decision"] = "fail"; rec["err"] = str(e); n_fail += 1
            report.append(rec)
            if i % 500 == 0 or i == len(fake_raw):
                print(f"[b6_std][{mod}] {i}/{len(fake_raw)}  "
                      f"trimmed={n_trimmed} no_silence={n_nosil} "
                      f"all_sil={n_all_sil} fail={n_fail}")

    # ── real 对照：RVRA（全真，无 leading silence，不 trim，仅登记）──
    mod, d = REAL_MOD
    real_vids = sorted(glob.glob(os.path.join(SRC, d, "**", "*.mp4"), recursive=True))
    for vid in real_vids:
        stem = os.path.splitext(os.path.basename(vid))[0]
        rel = os.path.relpath(vid, os.path.join(SRC, d))
        report.append({"mod": mod, "path": vid, "stem": stem, "rel": rel,
                       "lead": 0.0, "trail": 0.0, "decision": "real_no_trim",
                       "out_path": None})
    print(f"[b6_std][{mod}] real control: {len(real_vids)} (no trim, Smeu protocol)")

    with open(REPORT, "w", encoding="utf-8") as f:
        for rec in report:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[b6_std] done: {len(report)} rows  "
          f"trimmed={n_trimmed} no_silence={n_nosil} all_silence={n_all_sil} fail={n_fail}")
    print(f"[b6_std] report -> {REPORT}")


if __name__ == "__main__":
    main()
