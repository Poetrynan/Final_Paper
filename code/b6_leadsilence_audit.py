"""
b6_leadsilence_audit.py — 复测 FakeAVCeleb_v1.2 四模态前导静音（leading silence）
================================================================================
目的：用**正确指标**（首个 >= tau 样本的位置 = 前导静音时长）验证 Smeu 报告的
      "前导静音偏置只存在于 fake-audio 子集 (RVFA + FVFA)" 是否成立。

指标说明（此前误用"前100ms最大幅度"导致漏判短促前导静音，本次修正）：
    leading_silence_ms = (第一个 |x| >= tau 的样本索引) / SR * 1000
    tau = 5e-4 (Smeu 阈值)

判定偏置：lead_ms 落在 (5ms, 100ms] 视为存在可学习的前导静音偏置
          （real 音频通常 lead==0；fake-audio 由 TTS 生成，常在开头留 25-30ms 静音）。

输入（只读）：
    D:/Datasets/FakeAVCeleb_v1.2/{RealVideo-RealAudio,RealVideo-FakeAudio,
                                  FakeVideo-RealAudio,FakeVideo-FakeAudio}
输出：
    D:/Datasets/FakeAVCeleb_v1.2/b6_leadsilence_audit.jsonl   (每 clip 一行)
    stdout 汇总表
"""
import os, glob, subprocess, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_ROOT

SRC = os.path.join(DATA_ROOT, "FakeAVCeleb_v1.2")
MODS = {
    "RVRA": "RealVideo-RealAudio",   # 全真 (control)
    "RVFA": "RealVideo-FakeAudio",   # 真画面 + SV2TTS 假音频 -> 应有偏置
    "FVRA": "FakeVideo-RealAudio",   # wavtolip 假画面 + 真音频 -> 应无偏置
    "FVFA": "FakeVideo-FakeAudio",   # wavtolip 假画面 + 假音频 -> 应有偏置
}
SR = 16000
TAU = 5e-4
N_PER = {"RVRA": 60, "RVFA": 60, "FVRA": 80, "FVFA": 150}
# 证据报告写到 split 目录（原始 FakeAVCeleb_v1.2 只读、不新增任何文件）
OUT = os.path.join(DATA_ROOT, "FakeAVCeleb_v1.2_split", "b6_leadsilence_audit.jsonl")

BUCKETS = [(0, 5), (5, 20), (20, 40), (40, 60), (60, 100), (100, 1e9)]


def decode_audio(path):
    cmd = ["ffmpeg", "-v", "quiet", "-i", path, "-ar", str(SR), "-ac", "1",
           "-f", "f32le", "-"]
    r = subprocess.run(cmd, capture_output=True)
    return np.frombuffer(r.stdout, dtype=np.float32)


def main():
    rows = []
    summary = {}
    for tag, d in MODS.items():
        dpath = os.path.join(SRC, d)
        vids = sorted(glob.glob(os.path.join(dpath, "**", "*.mp4"), recursive=True))
        vids = vids[: N_PER[tag]]
        leads = []
        for vid in vids:
            try:
                x = decode_audio(vid)
            except Exception:
                continue
            if len(x) == 0:
                continue
            amp = np.abs(x)
            ns = np.nonzero(amp >= TAU)[0]
            lead_ms = (int(ns[0]) / SR * 1000) if len(ns) else float("inf")
            leads.append(lead_ms)
            rows.append({"mod": tag, "stem": os.path.splitext(os.path.basename(vid))[0],
                         "lead_ms": None if lead_ms == float("inf") else round(lead_ms, 2)})
        leads = np.array(leads)
        n = len(leads)
        biased = int(np.sum((leads > 5) & (leads <= 100)))
        hist = {}
        for lo, hi in BUCKETS:
            c = int(np.sum((leads >= lo) & (leads < hi))) if hi < 1e9 else int(np.sum(leads >= lo))
            hist[f"[{lo},{hi if hi<1e9 else 'inf'})"] = c
        summary[tag] = {
            "n": n, "median_ms": round(float(np.median(leads)), 2),
            "mean_ms": round(float(np.mean(leads)), 2),
            "pct_biased_5_100ms": round(100.0 * biased / n, 1) if n else 0.0,
            "hist": hist,
        }
        print(f"{tag:5s} n={n:4d}  median={summary[tag]['median_ms']:8.2f}ms  "
              f"mean={summary[tag]['mean_ms']:8.2f}ms  biased%={summary[tag]['pct_biased_5_100ms']:5.1f}  "
              f"hist={hist}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"summary": summary}, f, ensure_ascii=False, indent=2)
        f.write("\n")
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[audit] wrote {OUT}  ({len(rows)} clip rows)")


if __name__ == "__main__":
    main()
