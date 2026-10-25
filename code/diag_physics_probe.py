#!/usr/bin/env python
"""
diag_physics_probe.py — 秒级验证「方向A：高阶/物理运动学代理是否跨生成器可分」
====================================================================================
背景：diag_shortcut 的 lip_jitter(1阶差分均值=统计抖动) 在 train 域 strength=0.738、
但跨生成器 HiFi/DFE 退化到 ~0.52。本脚本换成「物理合理性」代理——用 Savitzky-Golay
高阶导数(尤其 jerk/加加速度)的 分布尾部 / 峰度 / 能量比，在三跨域集上测可分性。

判据：
  - 若某物理代理在 HiFi/DFE 上 strength ≥ 0.6（明显优于 lip_jitter 的 ~0.52）
    → 「物理边界」线索跨生成器可迁移，值得造完整 2A-2。
  - 若全部仍 ≈ 0.5 → 短程运动学 (无论统计还是高阶尾部) 都不可迁移，转向方向B。

口径与 models.Kinematics 严格一致：S-G window=5, poly=3, order=3, xyz=192, kin_norm=False。
运动学核复用已验证的 models.savgol_coeffs，避免重复实现的系数长度/数值漂移。
仅 numpy，不占 GPU，秒级。
"""
import os, sys, argparse
from math import factorial
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_shortcut import roc_auc_y_score, cohens_d, load_index

WINDOW, POLY, ORDER = 5, 3, 3          # 与 cfg.kin_* 一致


def _sg_coeffs(window, poly, deriv, delta=1.0):
    """自包含 S-G 平滑导数核，恒返回长度 window（等值同 models.savgol_coeffs）。

    不再 import models，避免因云端 models.py 版本差异导致系数长度漂移。
    """
    assert window % 2 == 1 and window > poly >= deriv
    x = np.arange(window, dtype=np.float64) - (window // 2)          # 长度恒=window，居中索引，无地板除歧义
    A = np.vander(x, poly + 1, increasing=True)                      # (W, poly+1)
    pinv = np.linalg.pinv(A)                                         # (poly+1, W)
    c = pinv[deriv] * factorial(deriv) / (delta ** deriv)            # (W,)
    return c.astype(np.float64)


def kin_savgol(P, window=WINDOW, poly=POLY, order=ORDER):
    """P:(T, C) → (T, C*(order+1)) 位置 + 1..order 阶 S-G 平滑导数（边缘 replicate pad）。"""
    T, C = P.shape
    half = window // 2
    out = []
    K = (P / (P.std() + 1e-8)) if P.std() > 1e-9 else P    # 批内尺度归一化(抑制绝对尺度跨集漂移)
    Ppad = np.pad(K, ((half, half), (0, 0)), mode="edge")
    for d in range(order + 1):
        c = _sg_coeffs(window, poly, deriv=d, delta=1.0)
        assert len(c) == window, f"系数长度 {len(c)} != 窗口 {window}"
        dv = np.stack([Ppad[i:i + window].T @ c for i in range(T)])   # (T, C)
        out.append(dv)
    return np.concatenate(out, axis=-1)                   # (T, C*(order+1))


def per_order_norms(K, order=ORDER):
    Ck = K.shape[1] // (order + 1)
    return [np.linalg.norm(K[:, o * Ck:(o + 1) * Ck], axis=1) for o in range(order + 1)]


def phys_features(v_lip):
    """v_lip:(T,192)→ 若干"物理代理"标量特征。"""
    K = kin_savgol(v_lip)
    pos, vel, acc, jerk = per_order_norms(K)
    j = jerk[np.isfinite(jerk)]
    j = j[np.abs(j) > 1e-12]
    f = {}
    # 对照：现有 diag 的 1 阶统计抖动（证明它跨生成器退化）
    f["lip_jitter_1d(对照)"] = float(np.linalg.norm(v_lip[1:] - v_lip[:-1], axis=1).mean())
    if len(j) == 0:
        for k in ["jerk_mean", "jerk_p95", "jerk_max", "jerk_kurtosis", "jerk_pos_energy_ratio"]:
            f[k] = 0.0
        return f
    f["jerk_mean"] = float(np.mean(j))
    f["jerk_p95"] = float(np.quantile(j, 0.95))
    f["jerk_max"] = float(np.max(j))
    f["jerk_kurtosis"] = float(((j - j.mean()) ** 4).mean() / (j.std() ** 2 + 1e-12)) if j.std() > 1e-9 else 0.0
    f["jerk_pos_energy_ratio"] = float((j ** 2).sum() / (pos ** 2).sum() + 1e-12)
    return f


FEATURES = ["lip_jitter_1d(对照)", "jerk_mean", "jerk_p95", "jerk_max", "jerk_kurtosis", "jerk_pos_energy_ratio"]


def run(cache, max_per_class):
    print(f"[phys] 扫描 {cache}   每类上限 {max_per_class}")
    index = load_index(cache)
    real = [ix for ix in index if ix["label"] == 0]
    fake = [ix for ix in index if ix["label"] == 1]
    print(f"[phys] real={len(real)}  fake={len(fake)}")
    if max_per_class:
        rng = np.random.default_rng(0)
        real = list(rng.choice(real, size=min(max_per_class, len(real)), replace=False)) if real else []
        fake = list(rng.choice(fake, size=min(max_per_class, len(fake)), replace=False)) if fake else []
    rows = ([{"label": 0, "path": ix["path"]} for ix in real] +
            [{"label": 1, "path": ix["path"]} for ix in fake])
    y = np.array([r["label"] for r in rows], dtype=np.int64)
    vals = {name: [None] * len(rows) for name in FEATURES}
    for i, r in enumerate(rows):
        try:
            import torch
            meta = torch.load(r["path"], map_location="cpu", weights_only=True)
        except Exception:
            continue
        v_lip = (meta.get("v_lip") if isinstance(meta, dict) else meta["v_lip"])
        if v_lip.ndim != 2 or v_lip.shape[1] != 192:
            continue
        for name in FEATURES:
            vals[name][i] = phys_features(v_lip)[name]
    print("\n%-28s %8s %8s %10s  %s" % ("feature", "auc(实0)", "strength", "cohen_d", "判定"))
    best = []
    for name in FEATURES:
        s = np.array([(v if v is not None else np.nan) for v in vals[name]], dtype=np.float64)
        ok = ~np.isnan(s)
        m_real = s[(y == 0) & ok]; m_fake = s[(y == 1) & ok]
        if len(m_real) < 10 or len(m_fake) < 10:
            print("%-28s %8s %8s %10s  %s" % (name, "-", "-", "-", "数据不足")); continue
        auc = roc_auc_y_score(y[ok], s[ok]); st = max(auc, 1 - auc); d = cohens_d(m_real, m_fake)
        mark = "★强可分" if st >= 0.8 else ("▲可分" if st >= 0.65 else ("●中" if st >= 0.6 else "."))
        print("%-28s %8.3f %8.3f %+10.3f  %s" % (name, auc, st, d, mark))
        if st >= 0.6:
            best.append((name, st, d))
    if best:
        print("\n⚠️ 跨生成器可分(≥0.6)：")
        for name, st, d in best:
            print(f"   - {name}  strength={st:.3f}  cohen_d={d:+.3f}  → 物理代理可迁移，方向A有戏")
    else:
        print("\n✅ 所有物理代理 <0.6 → 短程运动学跨生成器不可迁移，转方向B。")


def main():
    ap = argparse.ArgumentParser(description="秒级验证 方向A 高阶/物理运动学代理跨生成器可分性")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--max-per-class", type=int, default=800)
    a = ap.parse_args()
    run(a.cache, a.max_per_class)


if __name__ == "__main__":
    main()