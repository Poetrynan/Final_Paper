#!/usr/bin/env python
"""
diag_shortcut.py — 嫌疑① 训练域捷径诊断（Scan DeepSpeak train for non-content shortcuts）
=============================================================================================
目的：复刻 18_Circumventing_Shortcuts (Oneata) 的思路，但作用在**训练域**上——检查
      DeepSpeak **训练集**里 real vs fake 是否在"非内容统计特征"上可分。若一个极简规则
      （例如"前半段音频能量""首帧特征""帧间运动量""landmark 抖动"）就能把两类别分开，
      说明模型完全可能只靠这个捷径达到同域 0.97，从而解释跨域全崩。

两种运行模式（都用 prep 缓存，无需原始数据集，秒级出结果）：
  cache — 复用 prep 的 .pt 特征，对每 clip 算若干"捷径候选标量"，real vs fake 算 AUC+Cohen's d。
         推荐先跑这个。默认按标签抽样子采样（耗时可接受）。
  audio — 对原始视频解码音频，核对前导静音 / 全程响度（最贴 18 论文指标，但慢、需原始数据集）。

用法示例（云端 / 本地一致）：
  # 训练集（训练看到的部分）：
  python diag_shortcut.py cache \
      --cache /root/autodl-tmp/SyncX/Datasets/DeepSpeak_v1_1/clips/prep/train \
      --max-per-class 800

  # 若想看验证拆分也一并扫：
  python diag_shortcut.py cache \
      --cache /root/autodl-tmp/SyncX/Datasets/DeepSpeak_v1_1/clips/prep \
      --max-per-class 800

  # 原始音频前导静音核对（可选，需要 DeepSpeak 原始 mp4 根路径）：
  python diag_shortcut.py audio \
      --video-root /root/autodl-tmp/SyncX/Datasets/DeepSpeak_v1_1 \
      --max-per-label 300
"""
import os, sys, glob, json, argparse, subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------- 指标 ----------------
def roc_auc_y_score(y, s):
    """AUC：s 越大 → 判 fake。若方向相反(实际是 real 更大)，返回 <0.5。"""
    y = np.asarray(y, dtype=bool)
    s = np.asarray(s, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")            # 升序
    rank = np.empty_like(order, dtype=np.float64)
    rank[order] = np.arange(1, len(s) + 1)
    sum_pos = rank[y].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def cohens_d(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    d = (a.mean() - b.mean()) / np.sqrt(var + 1e-12)
    return float(d)


# ---------------- 捷径候选特征（每 clip -> 标量） ----------------
# 约定：特征得分"越大"方向不预设；报告时同时给 d（方向带着）与 strength=max(auc,1-auc)。

def feat_audio_head_silence(meta):
    """音频前 8 帧平均范数 / 全段平均范数 —— 开头若有 TTS 前导静音，此比值显著偏小。"""
    a = meta.get("audio_seq")
    if a is None or a.ndim != 2 or a.shape[0] < 16:
        return None
    k = min(8, a.shape[0])
    head = a[:k].mean(axis=0)
    whole = a.mean(axis=0)
    return float(np.linalg.norm(head) / (np.linalg.norm(whole) + 1e-8))


def feat_audio_rms(meta):
    """音频序列整体能量（wav2vec 特征 L2 均值）—— 响度 / 能量级代理。"""
    a = meta.get("audio_seq")
    if a is None or a.ndim != 2:
        return None
    return float(np.linalg.norm(a) / max(1, a.shape[0]))


def feat_audio_firsthalf_energy(meta):
    """前段(前 1/3 帧)相对整体能量 —— 开头是否偏静/偏响。"""
    a = meta.get("audio_seq")
    if a is None or a.ndim != 2 or a.shape[0] < 9:
        return None
    L = a.shape[0]
    head = a[: L // 3]
    whole = a.mean(axis=0)
    hmean = np.linalg.norm(np.abs(head).mean(axis=0))
    wmean = np.linalg.norm(np.abs(whole)) + 1e-8
    return float(hmean / wmean)


def feat_visual_motion(meta):
    """视觉帧间运动量：||v_clip[t+1]-v_clip[t]|| 均值 —— 生成视频运动模式异常代理。"""
    v = meta.get("v_clip")
    if v is None or v.ndim != 2 or v.shape[0] < 3:
        return None
    diff = np.linalg.norm(v[1:] - v[:-1], axis=1)
    return float(diff.mean())


def feat_visual_motion_std(meta):
    """帧间运动量的标准差 —— 运动节奏是否"过稳/过抖"。"""
    v = meta.get("v_clip")
    if v is None or v.ndim != 2 or v.shape[0] < 3:
        return None
    diff = np.linalg.norm(v[1:] - v[:-1], axis=1)
    return float(diff.std())


def feat_lip_jitter(meta):
    """landmark 时域抖动(逐帧位移均值) —— BioLip 信号，MediaPipe 对生成脸可能系统性不稳。"""
    l = meta.get("v_lip")
    if l is None or l.ndim != 2 or l.shape[0] < 3:
        return None
    diff = np.linalg.norm(l[1:] - l[:-1], axis=1)
    return float(diff.mean())


def feat_lip_rms(meta):
    """landmark 整体 RMS —— 嘴部运动幅度代理。"""
    l = meta.get("v_lip")
    if l is None or l.ndim != 2:
        return None
    return float(np.linalg.norm(l) / max(1, l.shape[0]))


FEATURES = [
    ("audio_head_silence", "音频开头8帧能量比(<<1=开头静音)", feat_audio_head_silence),
    ("audio_rms", "音频序列整体能量", feat_audio_rms),
    ("audio_firsthalf_energy", "音频前1/3段能量比", feat_audio_firsthalf_energy),
    ("visual_motion", "视觉帧间运动量", feat_visual_motion),
    ("visual_motion_std", "视觉帧间运动量标准差", feat_visual_motion_std),
    ("lip_jitter", "landmark 时域抖动", feat_lip_jitter),
    ("lip_rms", "landmark 幅度RMS", feat_lip_rms),
]


# ---------------- cache 模式 ----------------
def load_index(cache_dir):
    from dataset import _load_or_build_index
    files = sorted(glob.glob(os.path.join(cache_dir, "**", "*.pt"), recursive=True))
    return _load_or_build_index(files, cache_dir)


def run_cache(cache, max_per_class):
    print(f"[cache] 扫描 {cache}  每类上限 {max_per_class}")
    index = load_index(cache)
    real = [ix for ix in index if ix["label"] == 0]
    fake = [ix for ix in index if ix["label"] == 1]
    print(f"[cache] real={len(real)}  fake={len(fake)}")
    if max_per_class:
        rng = np.random.default_rng(0)
        real = list(rng.choice(real, size=min(max_per_class, len(real)), replace=False)) if real else []
        fake = list(rng.choice(fake, size=min(max_per_class, len(fake)), replace=False)) if fake else []
    rows = [{"label": 0, "path": ix["path"]} for ix in real] + \
           [{"label": 1, "path": ix["path"]} for ix in fake]
    y = np.array([r["label"] for r in rows], dtype=np.int64)
    vals = {name: [None] * len(rows) for name, _, _ in FEATURES}

    for i, r in enumerate(rows):
        try:
            meta = torch.load(r["path"], map_location="cpu", weights_only=True)
        except Exception:
            continue
        for name, _, fn in FEATURES:
            vals[name][i] = fn(meta)
    # 需要 torch；顶部不 import 以便 --help 也能跑
    # （实际 torch.load 时已用到）

    print("\n%-24s %8s %8s %10s  %s" % ("feature", "auc(实0)", "strength", "cohen_d", "判定"))
    hits = []
    for name, desc, _ in FEATURES:
        s = np.array([v if v is not None else np.nan for v in vals[name]], dtype=np.float64)
        ok = ~np.isnan(s) & (y == 0) | ~np.isnan(s) & (y == 1)
        m_real = s[(y == 0) & ~np.isnan(s)]
        m_fake = s[(y == 1) & ~np.isnan(s)]
        if len(m_real) < 10 or len(m_fake) < 10:
            print("%-24s %8s %8s %10s  %s" % (desc, "-", "-", "-", "数据不足"))
            continue
        auc = roc_auc_y_score(y[~np.isnan(s)], s[~np.isnan(s)])
        strength = max(auc, 1 - auc)
        d = cohens_d(m_real, m_fake)
        mark = "★强捷径" if (strength >= 0.8) else ("▲中等" if strength >= 0.65 else ".")
        print("%-24s %8.3f %8.3f %+10.3f  %s" % (desc, auc, strength, d, mark))
        if strength >= 0.65:
            hits.append((desc, name, auc, strength, d))
    print("\n说明：auc(实0) 以 real=0 计分，>0.5 表示 real 特征更高、<0.5 表示 fake 更高；"
          "strength=max(auc,1-auc)，≥0.8 即强捷径(模型很可能只靠它)。cohen_d 带方向（+ = real 更大）。")
    if hits:
        print("\n⚠️ 命中可学习捷径(≥0.65)：")
        for desc, name, auc, st, d in hits:
            print(f"   - {desc}  [{name}]  strength={st:.3f}  cohen_d={d:+.3f}")
    else:
        print("\n✅ 未发现明显捷径(所有特征 strength<0.65) → 嫌疑①初步排除。")


# ---------------- audio 模式（可选，最贴 18 论文） ----------------
def decode_audio(path, sr=16000):
    cmd = ["ffmpeg", "-v", "quiet", "-i", path, "-ar", str(sr), "-ac", "1", "-f", "f32le", "-"]
    r = subprocess.run(cmd, capture_output=True)
    return np.frombuffer(r.stdout, dtype=np.float32)


def run_audio(video_root, max_per_label):
    print(f"[audio] 扫描 {video_root}  每类上限 {max_per_label}")
    fake_dirs = ["fake", "Fake", "fakes", "fraud"]
    real_dirs = ["real", "Real"]
    # 递归找全部分层的音频/视频文件（DeepSpeak 目录结构含 train/test × real/fake）
    vids = sorted(glob.glob(os.path.join(video_root, "**", "*.mp4"), recursive=True)) + \
           sorted(glob.glob(os.path.join(video_root, "**", "*.wav"), recursive=True))
    import random
    random.seed(0)
    random.shuffle(vids)
    buckets = {"real": [], "fake": []}
    for p in vids:
        low = p.lower()
        if any(f"/{fd}/" in f"/{low}/" or low.replace("\\", "/").endswith(f"/{fd}") or
               (f"_{fd}" in low.replace("\\", "/")) for fd in fake_dirs):
            buckets["fake"].append(p)
        elif any(fd in low for fd in real_dirs):
            buckets["real"].append(p)
        if len(buckets["real"]) and len(buckets["fake"]) and \
           len(buckets["real"]) >= max_per_label and len(buckets["fake"]) >= max_per_label >= 1:
            buckets["real"] = random.sample(buckets["real"], max_per_label)
            buckets["fake"] = random.sample(buckets["fake"], max_per_label)
            break
    if max_per_label:
        buckets["real"] = (buckets["real"] or [])[:max_per_label]
        buckets["fake"] = (buckets["fake"] or [])[:max_per_label]
    print(f"[audio] 找到 real={len(buckets['real'])}  fake={len(buckets['fake'])}")
    rows = []
    for lab, key in (("real", "real"), ("fake", "fake")):
        for p in buckets[key]:
            try:
                x = decode_audio(p)
            except Exception:
                continue
            if len(x) == 0:
                continue
            amp = np.abs(x)
            nz = np.nonzero(amp >= 5e-4)[0]
            lead_ms = (int(nz[0]) / 16000 * 1000) if len(nz) else float("inf")
            rows.append({"label": 0 if key == "real" else 1,
                         "path": p,
                         "lead_ms": None if lead_ms == float("inf") else round(lead_ms, 3),
                         "rms": round(float(np.sqrt((x ** 2).mean())), 5)})
    y = np.array([r["label"] for r in rows])
    for meta_key, key in (("前导静音ms(实0)", "lead_ms"), ("全程RMS", "rms")):
        s = np.array([(r[key] if r[key] is not None else np.nan) for r in rows], dtype=np.float64)
        ok = ~np.isnan(s)
        m_real = s[(y == 0) & ok]
        m_fake = s[(y == 1) & ok]
        if len(m_real) < 10 or len(m_fake) < 10:
            print(f"{meta_key}: 数据不足")
            continue
        auc = roc_auc_y_score(y[ok], s[ok])
        st = max(auc, 1 - auc)
        d = cohens_d(m_real, m_fake)
        print(f"{meta_key}: auc(实0)={auc:.3f}  strength={st:.3f}  cohen_d={d:+.3f}  "
              f"real_med={np.nanmedian(m_real):.4f}  fake_med={np.nanmedian(m_fake):.4f}")
    print("\n若 前导静音 被识别为强捷径(≥0.8) → 锁死嫌疑①(存在 TTS 前导静音类捷径)。")


def main():
    ap = argparse.ArgumentParser(description="训练域捷径诊断 (嫌疑①)")
    sub = ap.add_subparsers(dest="mode", required=True)

    pc = sub.add_parser("cache")
    pc.add_argument("--cache", required=True, help="prep 缓存目录(建议 train 划分)")
    pc.add_argument("--max-per-class", type=int, default=800)
    pc.set_defaults(func=lambda a: run_cache(a.cache, a.max_per_class))

    pa = sub.add_parser("audio")
    pa.add_argument("--video-root", required=True, help="原始视频根目录(DeepSpeak)")
    pa.add_argument("--max-per-label", type=int, default=300)
    pa.set_defaults(func=lambda a: run_audio(a.video_root, a.max_per_label))

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

# 延迟 import torch（--help 时不至于卡环境）
import torch  # noqa: E402