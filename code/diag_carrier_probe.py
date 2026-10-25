#!/usr/bin/env python
"""
diag_carrier_probe.py — 特征载体探针：分离预训练特征是否承载可跨生成器分离的音画一致性
=====================================================================================
目的：给「换骨干最后一赌」(AV-HuBERT 音画联合预训练) 提供**换之前**的事实判据，不烧 GPU、不重训。

背景（实测/调研锚点）：
  - 现载体 = 分离预训练：CLIP-ViT-B/16(视觉) + wav2vec2-base-960h(音频)。
    wav2vec2 是纯音频自监督，**没有把音画对齐结构编入特征**（§8F 已调查转向 AV-HuBERT 的相关性）。
  - AV-HuBERT 是自监督「音画联合预训练」（facebookresearch/av_hubert，LRS3，SOTA 于 AVSR/唇读），
    理论载荷上包含真实音画匹配结构 → 这正是 HAVIC 配方假设成立所需的载体类型。
  - 因此一个问题值得先廉价回答：**当前分离特征，到底承载了多少「可跨生成器分离」的音画一致性？**

统计量（非拟合，绕开训练好的模型，直接读缓存特征）：
  RV_av = RV coefficient(audio_seq, v_clip)
        = tr(X Xᵀ Y Yᵀ) / sqrt( tr((XXᵀ)²) tr((YYᵀ)²) )
    其中 X: (T,p)=audio(时间×768 中心化)，Y: (T,q)=visual(时间×512 中心化)，行按时间对齐、重采样到同长。
    RV ∈ [0,1]，衡量两路时间过程整体一致性（0=正交无关，1=线性一致）；对称、无量纲、无需公共投影。
    ★ 刻意不 cos(audio_pool, visual_pool) 或 CCA：那是「拟合」到判别方向；RV 是纯协变结构，非学习。

判据（决策导向，非玄学）：
  - 按生成器分组（复用 diag_multigen_prior 的分组 + 跨生成器符号一致性闸门，HiFi identity=生成器权威）。
  - 对每个生成器组：real vs 该生成器 fake 的 RV_av 是否可分（strength≥0.6），且跨所有生成器方向一致。
  - 可分(≥2 生成器组同号，且大多组一致) → 分离特征已承载一致性 → **不宜仅因载体而换骨干**，先修训练；
  - 不可分(强组不足或符号乱) → 分离载体不见一致性 → **换 AV-HuBERT 联合载体是有据的一赌**（非充分保证）。

用法（无需 --ckpt，无需 GPU，秒级）：
  1) 核对分组（HiFi 用 identity）：
     python diag_carrier_probe.py --cache $Datasets/HiFi-AVDF/prep --group-by identity --list-groups
  2) 正式统计（可多集）：
     python diag_carrier_probe.py --cache $Datasets/HiFi-AVDF/prep \
        --cache $Datasets/Deepfake-Eval-2024/prep/test --group-by identity
     # DFE2024 无生成器标签 → 只能按"real vs 全部fake"当总池看（见 GLOBAL_ONLY）

局限（诚实声明）：
  - 探针测的是「分离特征」承载的一致性；NEGATIVE 支持换骨干，但不能**证明** AV-HuBERT 必可分
    （AV-HuBERT 在真实语音上预训练，对「伪造」无直接监督，统计上未必对 fake 敏感）。
  - 因此本探针是「换骨干的必要前置判断」，不是「换骨干成败的预言」。
"""
import os, sys, argparse, re, glob
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_shortcut import load_index, roc_auc_y_score, cohens_d

STATS = ["RV_av"]
R_MIN = 10            # 每组最少样本数
MIN_GROUP_STRENGTH = 0.60
MIN_GROUPS_POSITIVE = 2


def _resample_t(x, T):
    """把 x (n,p) 线性重采样到 (T,p)。n 与 T 差的 factor 用整数步长做，避免反卷积噪音。"""
    n, p = x.shape
    if n == 0 or p == 0:
        return np.zeros((T, p), np.float32)
    if n == T:
        return x.astype(np.float32)
    idx = np.round(np.linspace(0, n - 1, T)).astype(int)
    return x[idx].astype(np.float32)


def rv_av(d):
    """AV 特征一致性 = 线性 CKA（中心核对齐），非拟合、兼容 audio(768)/visual(512) 不同维。

    线性 CKA 是 RV coefficient 的等价物，但允许两组变量维数不同：
        CKA(A,V) = ||AᵀV||_F² / ( ||AᵀA||_F * ||VᵀV||_F )
    其中 A:(T,p_a), V:(T,p_v)，仅要求时间维行对齐（已重采样到同 T），
    AᵀV ∈ R^{p_a×p_v} 不要求 p_a=p_v → 与 768/512 天然兼容。
    CKA ∈ [0,1]，0=正交无关，1=线性一致；对称、无量纲、无须公共投影，符合"不拟合"红线。
    """
    a = d.get("audio_seq")
    v = d.get("v_clip")
    if a is None or v is None:
        return None
    a = torch_load_np(a)
    v = torch_load_np(v)
    if a.ndim != 2 or v.ndim != 2 or a.shape[1] == 0 or v.shape[1] == 0:
        return None
    T = min(a.shape[0], v.shape[0])
    if T < 8:
        return None
    A = _resample_t(a, T)                      # (T,p_a)
    V = _resample_t(v, T)                      # (T,p_v)
    A = A - A.mean(0, keepdims=True)           # 时间中心化
    V = V - V.mean(0, keepdims=True)
    A = A / (np.linalg.norm(A, "fro") + 1e-12)
    V = V / (np.linalg.norm(V, "fro") + 1e-12)
    # 线性 CKA：||AᵀV||_F² / (||AᵀA||_F · ||VᵀV||_F)
    Avt = A.T @ V                              # (p_a, p_v)
    vaa = float(np.linalg.norm(A.T @ A, "fro"))  # ||AᵀA||_F
    vvv = float(np.linalg.norm(V.T @ V, "fro"))  # ||VᵀV||_F
    if vaa < 1e-12 or vvv < 1e-12:
        return None
    return float(np.sum(Avt * Avt) / (vaa * vvv))


def torch_load_np(x):
    """tensor -> numpy float32（容忍 list/numpy）。"""
    if hasattr(x, "dtype") and hasattr(x, "cpu"):   # torch.Tensor（含 fp16）：先 .float() 再 .numpy()
        return x.float().cpu().numpy().astype(np.float32)
    if hasattr(x, "numpy"):                          # 其他 numpy-like
        return x.numpy().astype(np.float32)
    return np.asarray(x, np.float32)


def _realpath_under(path, root):
    try:
        return os.path.relpath(path, root)
    except Exception:
        return path


def group_index(cache, group_by, group_regex):
    index = load_index(cache)
    rgx = re.compile(group_regex) if group_regex else None
    buckets = {}
    for ix in index:
        if ix["label"] not in (0, 1):
            continue
        if group_by == "identity":
            g = ix.get("identity") or "NOID"
        else:
            if rgx is None:
                raise SystemExit("--group-by path 需要 --group-regex")
            rel = _realpath_under(ix["path"], cache)
            m = rgx.search(rel)
            g = m.group(1) if (m and m.groups()) else (m.group(0) if m else "NOGROUP")
        buckets.setdefault(g, {0: [], 1: []})[ix["label"]].append(ix)
    return {g: d for g, d in buckets.items() if (d[0] or d[1])}


def list_groups(caches, group_by, group_regex):
    print("=" * 72)
    print("分组核对（可信性前置：先确认分组正确再跑统计）")
    print("=" * 72)
    for cache in caches:
        print(f"\n# 缓存: {cache}")
        buckets = group_index(cache, group_by, group_regex)
        if not buckets:
            print("   (无可分组样本)")
            continue
        for g in sorted(buckets):
            nr, nf = len(buckets[g][0]), len(buckets[g][1])
            ex = []
            for lab in (0, 1):
                if buckets[g][lab]:
                    p = buckets[g][lab][0]["path"]
                    ex.append(os.path.basename(os.path.dirname(p)))
            flag = "  ⚠️ 样本过少，不入统计" if (nr < R_MIN or nf < R_MIN) else ""
            print(f"  [{g}]  real={nr}  fake={nf}  {flag}")
            print(f"        示例: {ex}")
    print("\n请核对：分组是否对应单一生成器。若不对，改用 --group-regex / --group-by path。")


def run_probe(cache, group_by, group_regex, max_real, max_fake):
    print(f"\n=== 载体探针集: {cache} ===")
    buckets = group_index(cache, group_by, group_regex)
    rng = np.random.default_rng(0)
    acc = {}
    for g in sorted(buckets):
        real_ix, fake_ix = buckets[g][0], buckets[g][1]
        if len(real_ix) < R_MIN or len(fake_ix) < R_MIN:
            print(f"  [组 {g}] real={len(real_ix)} fake={len(fake_ix)} <{R_MIN}，跳过")
            continue
        real_ix = list(rng.choice(real_ix, size=min(max_real, len(real_ix)), replace=False))
        fake_ix = list(rng.choice(fake_ix, size=min(max_fake, len(fake_ix)), replace=False))
        vals = {lab: [] for lab in ("real", "fake")}
        n_ok = 0
        for lab, ixs in (("real", real_ix), ("fake", fake_ix)):
            for ix in ixs:
                d = torch.load(ix["path"], map_location="cpu", weights_only=True)
                rv = rv_av(d)
                if rv is None:
                    continue
                n_ok += 1
                vals[lab].append(rv)
        print(f"  [组 {g}] 有效片段 {n_ok}/{len(real_ix)+len(fake_ix)}")
        if len(vals["real"]) >= R_MIN and len(vals["fake"]) >= R_MIN:
            acc[g] = vals
    return cache, acc


def stats_metrics(s, mr, mf):
    sign = 1.0  # RV 越大越一致；real 一致性更高 → + 方向
    mr_a, mf_a = sign * np.asarray(mr, np.float64), sign * np.asarray(mf, np.float64)
    if len(mr_a) >= R_MIN and len(mf_a) >= R_MIN:
        y = np.concatenate([np.zeros(len(mr_a)), np.ones(len(mf_a))])
        auc = roc_auc_y_score(y, np.concatenate([mr_a, mf_a]))
        auc_real_hi = 1.0 - auc if not np.isnan(auc) else float("nan")
        st = max(auc_real_hi, 1 - auc_real_hi) if not np.isnan(auc_real_hi) else float("nan")
        d = cohens_d(mr_a, mf_a)
    else:
        auc_real_hi = st = d = float("nan")
    return auc_real_hi, st, d


def print_group_table(name, acc):
    print(f"\n── 集: {name} ── (RV_av: real vs 该生成器 fake；+ = real 一致性更高)")
    print("%-6s %-12s %10s %8s %+9s  判定" % ("stat", "generator", "auc(实0)", "strength", "cohen_d"))
    for g in sorted(acc):
        auc_r, st, d = stats_metrics("RV_av", acc[g]["real"], acc[g]["fake"])
        mark = "★强" if st >= 0.8 else ("▲" if st >= 0.65 else ("●" if st >= 0.6 else "≈随机"))
        print("%-6s %-12s %10.3f %8.3f %+9.3f  %s" %
              ("RV_av", str(g)[:12], auc_r, st, (d if not np.isnan(d) else 0.0), mark))


def gate_report(acc_all):
    print("\n" + "=" * 72)
    print("特征载体闸门（跨生成器符号一致性 → “分离特征是否承载一致性”的操作定义）")
    print("=" * 72)
    groups = []
    for _, acc in acc_all:
        for g, d in acc.items():
            auc_r, st, dd = stats_metrics("RV_av", d["real"], d["fake"])
            groups.append((f"{g}", auc_r, st, dd))
    usable = [gr for gr in groups if not np.isnan(gr[3])]
    if not usable:
        print("   无可统计生成器组")
        return
    signs = [np.sign(gr[3]) for gr in usable]
    positive_groups = [gr for gr in usable if gr[2] >= MIN_GROUP_STRENGTH and gr[3] > 0]
    n_strong = sum(1 for gr in usable if gr[2] >= MIN_GROUP_STRENGTH)
    n_pos_strong = len(positive_groups)
    n_pos_sign = sum(1 for s in signs if s > 0)
    n_neg_sign = sum(1 for s in signs if s < 0)
    n_sign = len(signs)
    sign_ok = (n_strong >= MIN_GROUPS_POSITIVE and n_pos_strong >= MIN_GROUPS_POSITIVE and
               max(n_pos_sign, n_neg_sign) > n_sign / 2)
    print(f"\n[RV_av] 生成器组数={n_sign}  强可分(≥{MIN_GROUP_STRENGTH})={n_strong} "
          f"(其中 real更强={n_pos_strong})  符号方向 +:{n_pos_sign} / -:{n_neg_sign}")
    for gr in usable:
        print("    %-12s auc(实0=%+.3f) strength=%.3f cohen_d=%+.3f" % (
            str(gr[0])[:12], gr[1], gr[2], gr[3] if not np.isnan(gr[3]) else 0.0))
    if sign_ok:
        print("\n    ✅ SEPARABLE：分离特征已承载跨生成器可分离的一致性 → 换骨干非首要，先考虑从现有特征榨取/修训练")
    else:
        print("\n    ❌ NOT_SEPARABLE：分离载体不呈现跨生成器一致的一致性 → 换 AV-HuBERT 联合载体是有据的一赌"
              "（非充分保证，需全量重训验证）")
    print("\n判读：NOT_SEPARABLE 支持『换骨干最后一赌』；SEPARABLE 则建议先别换骨干、先诊疗训练。"
          "RV 是无监督协变强度，不偏袒任何判别方向，故符号 +/− 均需跨生成器一致才算可分。")


def main():
    ap = argparse.ArgumentParser(description="特征载体探针：分离预训练特征是否承载跨生成器音画一致性（无需 ckpt/GPU）")
    ap.add_argument("--cache", action="append", required=True)
    ap.add_argument("--group-by", choices=["identity", "path"], default="identity")
    ap.add_argument("--group-regex", default="")
    ap.add_argument("--list-groups", action="store_true")
    ap.add_argument("--max-per-real", type=int, default=600)
    ap.add_argument("--max-per-fake", type=int, default=400)
    args = ap.parse_args()

    if args.list_groups:
        list_groups(args.cache, args.group_by, args.group_regex)
        return
    acc_all = []
    for c in args.cache:
        name, acc = run_probe(c, args.group_by, args.group_regex, args.max_per_real, args.max_per_fake)
        print_group_table(name, acc)
        acc_all.append((name, acc))
    gate_report(acc_all)


if __name__ == "__main__":
    main()