#!/usr/bin/env python
"""
diag_freq_probe.py — 方向B 前置门：频域 / 时序 / 音画交叉谱相干 跨生成器可分性
====================================================================================
背景：方向A(短程高阶运动学)实测跨生成器不可迁移(见 evidence_chain §6)。方向B 主张
“长程频谱结构 / 时序演化 / 音画交叉谱相干”才是跨生成器不变信号。本脚本复用 prep 缓存，
在训练域 + 三跨域集上测这些代理的可分性(秒级,仅 numpy,不占 GPU)，为 FTA 立项做“门”。

门条件(与项目口径一致)：
  - DFE2024 / HiFi 上「sync 音画相干」或任一时域/频域代理 strength≥0.6 → 立项方向B。
  - 全 <0.6 → 方向B 也关，转“监督侧”(Stage-2 冻结编码器 / pseudo-fake / modality dropout)。

口径：
  - auc(实0) 以 real=0 计分，>0.5 表示 real 特征更高；
  - strength = max(auc, 1-auc)；
  - cohen_d(+ = real 更大)；跨集 cohen_d 符号翻转 ⇒ 生成器专属、不立项(=方向A 铁证)。

简化的物理口径(非标准 MSC，但可解释)：
  垂直颌动 m(t)（v_lip 每帧 64 点头均值 y）与音频包络 e(t)（wav2vec2 每帧 L2 范数）
  各自去均值+带通(1–8Hz 语音带)后，唱上げ采样对齐，算 Pearson 相关作为“同步相干”代理；
  另算单模态低频能量占比 / 自相关(AC1、AC2)等长时平滑/规律特征。
"""
import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_shortcut import roc_auc_y_score, cohens_d, load_index

FS_V, FS_A = 25.0, 50.0        # 视觉帧率 / wav2vec2 帧率(20ms)
SYNC_BAND = (1.0, 8.0)         # 语音带(Hz)：真实唇音同步最强相干区


# ── 频域基础工具 ──────────────────────────────────────────────────
def _power(sig, fs):
    """去均值+汉宁窗后 rfft 功率谱。返回 (freqs, P)。"""
    sig = np.asarray(sig, dtype=np.float64)
    sig = sig - sig.mean()
    if sig.std() < 1e-12:
        return np.zeros(1), np.zeros(1)
    w = np.hanning(len(sig))
    P = np.abs(np.fft.rfft(sig * w)) ** 2
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    return freqs, P


def lowband_ratio(sig, fs, cutoff=2.0):
    f, P = _power(sig, fs)
    if P.sum() < 1e-12:
        return 0.0
    return float(P[f <= cutoff].sum() / P.sum())


def acf_lag(sig, lag):
    s = np.asarray(sig, dtype=np.float64)
    s = s - s.mean()
    if s.std() < 1e-12 or len(s) <= lag:
        return 0.0
    return float(np.corrcoef(s[:-lag], s[lag:])[0, 1])


def bandpass(sig, fs, lo, hi):
    """FFT 理想带通（软边缘），返回同长信号。"""
    s = np.asarray(sig, dtype=np.float64) - np.asarray(sig).mean()
    n = len(s)
    S = np.fft.fft(s)
    fe = np.fft.fftfreq(n, d=1.0 / fs)
    keep = (np.abs(fe) >= lo) & (np.abs(fe) <= hi)
    S[~keep] = 0.0
    return np.real(np.fft.ifft(S))


def sync_corr(a_env, m, fs_a=FS_A, fs_v=FS_V, band=SYNC_BAND):
    """垂直颌动 m 上采样到音频包络长，各自带通语音带后算 Pearson 相关(=同步相干代理)。"""
    a = bandpass(a_env, fs_a, band[0], band[1])
    idx = np.linspace(0, len(m) - 1, len(a_env))
    m_up = bandpass(np.interp(idx, np.arange(len(m)), m), fs_a, band[0], band[1])
    a = a - a.mean(); m_up = m_up - m_up.mean()
    if a.std() < 1e-12 or m_up.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, m_up)[0, 1])


def vertical_jaw(traj):
    """v_lip (T,192=64×3) → 每帧 64 点头的 y(索引1) 均值 → (T,) 垂直颌动轨迹。"""
    P = np.asarray(traj, dtype=np.float64)
    P = P.reshape(P.shape[0], P.shape[1] // 3, 3)
    return P[..., 1].mean(axis=1)


def features_from(meta):
    """从单条 .pt 缓存算方向B 候选特征字典。meta 为 dict，需含 v_lip 与 audio_seq。"""
    v_lip = meta["v_lip"] if isinstance(meta, dict) else meta["v_lip"]
    a_seq = meta["audio_seq"] if isinstance(meta, dict) else meta["audio_seq"]
    if v_lip.ndim != 2 or v_lip.shape[1] not in (192, 64) or a_seq.ndim != 2:
        return None
    m = vertical_jaw(v_lip)                                   # (T_v,)
    a_env = np.linalg.norm(np.asarray(a_seq, dtype=np.float64), axis=1)   # (T_a,)
    f = {
        "motion_lowband_ratio": lowband_ratio(m, FS_V, 2.0),
        "motion_AC1": acf_lag(m, 1),
        "motion_AC2": acf_lag(m, 2),
        "audio_env_lowband_ratio": lowband_ratio(a_env, FS_A, 2.0),
        "audio_env_AC1": acf_lag(a_env, 1),
        "sync_corr_1to8Hz": sync_corr(a_env, m),             # 音画交叉谱相干(红线核心)
        "motion_AC10": acf_lag(m, 10),                        # 更长程平滑(≈0.4s)
    }
    return f


FEATURES = ["motion_lowband_ratio", "motion_AC1", "motion_AC2", "motion_AC10",
            "audio_env_lowband_ratio", "audio_env_AC1", "sync_corr_1to8Hz"]


def run(cache, max_per_class):
    print(f"[freq] 扫描 {cache}   每类上限 {max_per_class}")
    index = load_index(cache)
    real = [ix for ix in index if ix["label"] == 0]
    fake = [ix for ix in index if ix["label"] == 1]
    print(f"[freq] real={len(real)}  fake={len(fake)}")
    if max_per_class:
        rng = np.random.default_rng(0)
        real = list(rng.choice(real, size=min(max_per_class, len(real)), replace=False)) if real else []
        fake = list(rng.choice(fake, size=min(max_per_class, len(fake)), replace=False)) if fake else []
    rows = ([{"label": 0, "path": ix["path"]} for ix in real] +
            [{"label": 1, "path": ix["path"]} for ix in fake])
    y = np.array([r["label"] for r in rows], dtype=np.int64)
    vals = {name: [None] * len(rows) for name in FEATURES}
    n_skip = 0
    for i, r in enumerate(rows):
        try:
            import torch
            meta = torch.load(r["path"], map_location="cpu", weights_only=True)
        except Exception:
            n_skip += 1
            continue
        feats = features_from(meta)
        if feats is None:
            n_skip += 1
            continue
        for name in FEATURES:
            vals[name][i] = feats[name]
    if n_skip:
        print(f"[freq] 跳过无法解析样本 {n_skip}/{len(rows)}")
    print("\n%-24s %8s %8s %10s  %s" % ("feature", "auc(实0)", "strength", "cohen_d", "判定"))
    best = []
    for name in FEATURES:
        s = np.array([(v if v is not None else np.nan) for v in vals[name]], dtype=np.float64)
        ok = ~np.isnan(s)
        m_real = s[(y == 0) & ok]; m_fake = s[(y == 1) & ok]
        if len(m_real) < 10 or len(m_fake) < 10:
            print("%-24s %8s %8s %10s  %s" % (name, "-", "-", "-", "数据不足")); continue
        auc = roc_auc_y_score(y[ok], s[ok]); st = max(auc, 1 - auc); d = cohens_d(m_real, m_fake)
        mark = "★强可分" if st >= 0.8 else ("▲可分" if st >= 0.65 else ("●中" if st >= 0.6 else "."))
        print("%-24s %8.3f %8.3f %+10.3f  %s" % (name, auc, st, d, mark))
        if st >= 0.6:
            best.append((name, st, d))
    if best:
        print("\n⚠️ 跨生成器可分(≥0.6)：")
        for name, st, d in best:
            print(f"   - {name}  strength={st:.3f}  cohen_d={d:+.3f}  → 方向B 信号存在，可立项")
    else:
        print("\n✅ 所有频域/时序代理 <0.6 → 方向B 也不迁移，转监督侧(Stage-2 冻结编码器 / pseudo-fake / modality dropout)。")


def main():
    ap = argparse.ArgumentParser(description="方向B 前置门：频域/时序/音画相干跨生成器可分性")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--max-per-class", type=int, default=800)
    a = ap.parse_args()
    run(a.cache, a.max_per_class)


if __name__ == "__main__":
    main()