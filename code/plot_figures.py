"""
plot_figures.py — 只读日志出 17 图（F1–F17，零重训重画）
================================================================
所有 load_* 只读 JSONL 日志，绝不调用 train/eval/infer（§17.10）。任一所需日志
缺失 → 抛 FileNotFoundError 指明缺哪个文件（不静默重跑）。

CLI：
    python plot_figures.py --log_dir logs \
        --exp_id syncx_bs256_lam0.3_lacaboth_seed0 \
        --exp_group syncx_bs256_lam0.3_lacaboth \
        --exp_group_b syncx_bs256_lam0.3_lacaboth_scratch \
        --lam_scan syncx_bs256_lam0.1_lacaboth,syncx_bs256_lam0.3_lacaboth,syncx_bs256_lam1.0_lacaboth \
        --out out/figures
"""
import os, sys, glob, argparse, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from logging_utils import read_jsonl, aggregate_seeds, ema
import plot_style as ps


# ---------------- load 层（只读日志） ----------------
def _gather_exp_group(log_dir, exp_group):
    dirs = [os.path.join(log_dir, exp_group)] + sorted(glob.glob(os.path.join(log_dir, exp_group + "_*")))
    out = {}
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for p in glob.glob(os.path.join(d, "eval", "*.jsonl")):
            out.setdefault(os.path.basename(p), []).extend(read_jsonl(p))
    return out

def load_train_stage1(log_dir, eid):
    p = os.path.join(log_dir, eid, "stage1_train.jsonl")
    if not os.path.exists(p): raise FileNotFoundError(p)
    return read_jsonl(p)

def load_train_stage2(log_dir, eid):
    p = os.path.join(log_dir, eid, "stage2_train.jsonl")
    if not os.path.exists(p): raise FileNotFoundError(p)
    return read_jsonl(p)

def load_interpret(log_dir, eid, case_id):
    p = os.path.join(log_dir, eid, "interpret", f"{case_id}.jsonl")
    if not os.path.exists(p): raise FileNotFoundError(p)
    return read_jsonl(p)

def load_features(log_dir, eid):
    p = os.path.join(log_dir, eid, "features", "clip_features.jsonl")
    if not os.path.exists(p): raise FileNotFoundError(p)
    return read_jsonl(p)

def load_embeddings(log_dir, eid):
    p = os.path.join(log_dir, eid, "features", "clip_embeddings.jsonl")
    if not os.path.exists(p): raise FileNotFoundError(p)
    return read_jsonl(p)

def load_profile(log_dir, eid):
    p = os.path.join(log_dir, eid, "profile.jsonl")
    if not os.path.exists(p): raise FileNotFoundError(p)
    rows = read_jsonl(p)
    return rows[-1] if rows else {}


def _flatten(eval_map):
    rows = []
    for recs in eval_map.values():
        rows.extend(recs)
    return rows

def _auc(r):
    m = r.get("metrics") or {}
    return m.get("AUC")

def _filter(rows, **kw):
    out = []
    for r in rows:
        ok = True
        for k, v in kw.items():
            if isinstance(v, (list, tuple)):
                if r.get(k) not in v: ok = False
            else:
                if r.get(k) != v: ok = False
        if ok: out.append(r)
    return out


# ---------------- 通用原语 ----------------
def grouped_bar(ax, categories, series, errors=None, colors=None, hatches=None,
                ylabel="AUC (%)", ylim=(40, 102), legend=True):
    """series: {name: [val per category]}；errors: {name: [std per category]}。"""
    names = list(series.keys())
    x = np.arange(len(categories)); w = 0.8 / max(len(names), 1)
    for i, name in enumerate(names):
        vals = series[name]
        errs = errors.get(name) if errors else None
        ax.bar(x + (i - (len(names)-1)/2) * w, vals, w,
               label=name, color=(colors or {}).get(name, None),
               hatch=(hatches or [None]*len(names))[i % len(hatches)] if hatches else None,
               edgecolor="black", linewidth=0.6,
               yerr=errs, capsize=3 if errs is not None else 0, error_kw={"elinewidth":0.8})
    ax.set_xticks(x); ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylabel(ylabel); ax.set_ylim(*ylim)
    if legend and names: ax.legend(fontsize=7, ncol=min(len(names), 3))


# ================= 图函数 =================
def plot_f1(s1, s2, out):
    tr1 = [r for r in s1 if r.get("phase") == "train"]
    va1 = [r for r in s1 if r.get("phase") == "val"]
    tr2 = [r for r in s2 if r.get("phase") == "train"]
    va2 = [r for r in s2 if r.get("phase") == "val"]
    fig, axes = plt.subplots(2, 2, figsize=(ps.fig_width(True), ps.fig_width(True)*0.62), constrained_layout=True)
    (aL, aR), (bT, bB) = axes

    # (a) 左 InfoNCE EMA vs global_step；右 val acc vs global_step
    if tr1:
        xs = [r["global_step"] for r in tr1]
        aL.plot(xs, ema([r["train_infoNCE_loss"] for r in tr1]),
                color=ps.METHOD_COLORS["syncx"], lw=ps.SYNCX_LW, label="InfoNCE (EMA)")
        aL.set_xlabel("Training step"); aL.set_ylabel("InfoNCE loss"); aL.legend(fontsize=7)
    if va1:
        aR.plot([r["global_step"] for r in va1], [r["val_contrastive_acc"] for r in va1],
                color=ps.DATASET_COLORS["THB"], lw=ps.SYNCX_LW, marker="o", ms=3, label="Val retrieval acc")
        aR.set_xlabel("Training step"); aR.set_ylabel("Validation accuracy"); aR.legend(fontsize=7)
    aL.set_title("Stage-1 (a)"); aR.set_title("Stage-1 (a) val")

    # (b) 顶三条损失 EMA vs global_step；底 val_auc vs global_step + is_best 竖线
    if tr2:
        xs = [r["global_step"] for r in tr2]
        bT.plot(xs, ema([r["L_total"] for r in tr2]), color="#0072B2", lw=ps.SYNCX_LW, label="L_total (EMA)")
        bT.plot(xs, ema([r["L_AAM"] for r in tr2]), color="#E69F00", lw=ps.BASE_LW, label="L_AAM")
        bT.plot(xs, ema([r["L_corr_weighted"] for r in tr2]), color="#009E73", lw=ps.BASE_LW, label=r"$\lambda\cdot L_{corr}$ (EMA)")
        bT.set_xlabel("Training step"); bT.set_ylabel("Loss"); bT.legend(fontsize=7)
    if va2:
        xs = [r["global_step"] for r in va2]
        aucs = [r.get("val_auc") for r in va2]
        bB.plot(xs, aucs, color=ps.METHOD_COLORS["syncx"], lw=ps.SYNCX_LW, marker="o", ms=3, label="Val AUC")
        best = [r for r in va2 if r.get("is_best")]
        for r in best:
            bB.axvline(r["global_step"], color="black", lw=1.0, ls="--")
            bB.text(r["global_step"], min(v for v in aucs if v is not None), "syncx_best.pt",
                    rotation=90, fontsize=6, va="bottom", ha="right")
        bB.set_xlabel("Training step"); bB.set_ylabel("Validation AUC"); bB.legend(fontsize=7)
    bT.set_title("Stage-2 (b)"); bB.set_title("Stage-2 (b) val")
    ps.savefig(fig, out)


def plot_f2(eval_map, out):
    rows = _flatten(eval_map)
    rows = [r for r in rows if r.get("fpr") and r.get("tpr")]
    if not rows: raise FileNotFoundError("no ROC records (need fpr/tpr in eval jsonl)")
    # (a) 叠加：每个 (dataset, method) 一条曲线
    #    曲线多时 legend 撑爆 constrained_layout（24+ 项塌缩）。
    #    改用普通布局 + 图例放图外 + savefig 的 bbox_inches="tight" 自动收边，
    #    任意曲线数都不塌（不再用 constrained_layout，避免与其冲突）。
    fig, axA = plt.subplots(figsize=(ps.fig_width(True), ps.fig_width(True)*0.62))
    seen = set()
    for r in rows:
        key = (r["dataset"], r.get("method"), r.get("group") or "ALL")
        if key in seen: continue
        seen.add(key)
        c = ps.METHOD_COLORS.get(r.get("method"), "#333333")
        lbl = f"{r['dataset']}/{r.get('method')}" + (f"/{r['group']}" if r.get("group") else "")
        axA.plot(r["fpr"], r["tpr"], color=c, lw=ps.BASE_LW, label=lbl, alpha=0.85)
    axA.plot([0,1],[0,1], color="black", lw=ps.REF_LW, ls="--", label="random")
    axA.set_xlim(0,1); axA.set_ylim(0,1); axA.set_xlabel("False positive rate")
    axA.set_ylabel("True positive rate"); axA.set_title("ROC overlay")
    n_lbl = len(seen) + 1
    ncol = min(4, max(1, int(math.ceil(n_lbl / 8))))
    axA.legend(fontsize=5.5, ncol=ncol, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    # (b) 3x2 网格，每数据集一张（Sync-X + 对角线）
    #    原 2x3 每格仅 1.16in 宽 → 塌缩；改 3x2 每格 ~3.3in 宽。
    #    不用 constrained_layout（与 savefig 的 bbox_inches="tight" 兼容，避免塌缩）。
    dsets = sorted({r["dataset"] for r in rows})
    dsets = dsets[:6]
    fig2, axs = plt.subplots(3, 2, figsize=(ps.fig_width(True), ps.fig_width(True)*0.88))
    axs = axs.ravel()
    for i, ds in enumerate(dsets):
        ax = axs[i]
        for r in rows:
            if r["dataset"] != ds: continue
            c = ps.METHOD_COLORS.get(r.get("method"), "#333333")
            ax.plot(r["fpr"], r["tpr"], color=c, lw=ps.BASE_LW,
                    label=r.get("method") + (f"/{r['group']}" if r.get("group") else ""))
        ax.plot([0,1],[0,1], color="black", lw=ps.REF_LW, ls="--")
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_title(ds, fontsize=8)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(fontsize=6)
    for j in range(len(dsets), len(axs)):
        axs[j].axis("off")
    ps.savefig(fig2, out.replace("F2", "F2b"))
    ps.savefig(fig, out)


def plot_f3(eval_map, out):
    rows = _filter(_flatten(eval_map), variant="full", group=[None, ""])
    if not rows: raise FileNotFoundError("no variant==full eval records")
    agg = aggregate_seeds(rows, ["dataset", "method"])
    dsets = sorted({r["dataset"] for r in agg})
    methods = sorted({r["method"] for r in agg})
    series = {m: [next((a["mean"] for a in agg if a["dataset"]==d and a["method"]==m), 0.0) for d in dsets] for m in methods}
    errors = {m: [next((a["std"] for a in agg if a["dataset"]==d and a["method"]==m), 0.0) for d in dsets] for m in methods}
    fig, ax = plt.subplots(figsize=(ps.fig_width(True), ps.fig_width(True)*0.5), constrained_layout=True)
    grouped_bar(ax, dsets, series, errors, colors=ps.METHOD_COLORS)
    ax.set_title("Cross-dataset detection (full)")
    ps.savefig(fig, out)


def plot_f4(eval_map, out):
    rows = _filter(_flatten(eval_map), dataset="FakeAVCeleb")
    rows = [r for r in rows if r.get("group")]   # 仅分组（ARVR/AFVR/ARVF/AFVF）
    if not rows: raise FileNotFoundError("no FakeAVCeleb group records")
    groups = sorted({r["group"] for r in rows})
    methods = sorted({r["method"] for r in rows})
    # 2×2 布局（4 个伪造类别），每格宽度 ≈ 全宽/2，标签可横排；
    # 修复原 1×4 布局 constrained_layout 把子图压缩到零的 warning。
    fig, axes = plt.subplots(2, 2, figsize=(ps.fig_width(True), ps.fig_width(True)*0.62), constrained_layout=True)
    for i, g in enumerate(groups):
        ax = axes.ravel()[i]
        gr = _filter(rows, group=g)
        agg = aggregate_seeds(gr, ["method"])
        vals = [next((a["mean"] for a in agg if a["method"]==m), 0.0) for m in methods]
        ax.bar(range(len(methods)), vals, color=[ps.METHOD_COLORS.get(m, "#999") for m in methods],
               edgecolor="black", linewidth=0.6)
        ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=6.5)
        ax.set_title(g + ("  (highlight)" if g == "AFVF" else ""), fontsize=8,
                     color=("red" if g == "AFVF" else "black"))
        ax.set_ylabel("AUC (%)"); ax.set_ylim(40, 102)
    ps.savefig(fig, out)


def plot_f14(eval_map, out):
    rows = _filter(_flatten(eval_map), dataset="HiFiAVDF")
    rows = [r for r in rows if r.get("group")]   # 仅分组（各 generator）
    if not rows: raise FileNotFoundError("no HiFiAVDF group records")
    groups = sorted({r["group"] for r in rows})
    methods = sorted({r["method"] for r in rows})
    fig, ax = plt.subplots(figsize=(ps.fig_width(True), ps.fig_width(True)*0.5), constrained_layout=True)
    agg = aggregate_seeds(rows, ["group", "method"])
    x = np.arange(len(groups)); w = 0.8 / max(len(methods), 1)
    for i, m in enumerate(methods):
        vals = [next((a["mean"] for a in agg if a["group"]==g and a["method"]==m), 0.0) for g in groups]
        errs = [next((a["std"] for a in agg if a["group"]==g and a["method"]==m), 0.0) for g in groups]
        ax.bar(x + (i - (len(methods)-1)/2) * w, vals, w, label=m,
               color=ps.METHOD_COLORS.get(m, None), edgecolor="black", linewidth=0.6,
               yerr=errs, capsize=3 if any(errs) else 0, error_kw={"elinewidth":0.8})
    ax.set_xticks(x); ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.set_ylabel("AUC (%)"); ax.set_ylim(40, 102); ax.set_title("HiFi-AVDF per-generator AUC (F14)")
    ax.legend(fontsize=7)
    ps.savefig(fig, out)


def plot_f5(eval_map, out):
    rows = _filter(_flatten(eval_map), variant=["full", "audio-only", "video-only"], group=[None, ""])
    if not rows: raise FileNotFoundError("no modality-ablation eval records")
    agg = aggregate_seeds(rows, ["dataset", "variant"])
    dsets = sorted({r["dataset"] for r in agg})
    variants = ["full", "audio-only", "video-only"]
    series = {v: [next((a["mean"] for a in agg if a["dataset"]==d and a["variant"]==v), 0.0) for d in dsets] for v in variants}
    errors = {v: [next((a["std"] for a in agg if a["dataset"]==d and a["variant"]==v), 0.0) for d in dsets] for v in variants}
    fig, ax = plt.subplots(figsize=(ps.fig_width(True), ps.fig_width(True)*0.5), constrained_layout=True)
    grouped_bar(ax, dsets, series, errors, colors=ps.VARIANT_COLORS)
    ax.set_title("Modality ablation (A1)")
    ps.savefig(fig, out)


def plot_f6(eval_main, eval_b, out):
    if eval_b is None: raise FileNotFoundError("need --exp_group_b for Stage-1 ablation (F6)")
    ra = _filter(_flatten(eval_main), variant="full", method="syncx", group=[None, ""])
    rb = _filter(_flatten(eval_b), variant="full", method="syncx", group=[None, ""])
    if not ra or not rb: raise FileNotFoundError("no syncx full records in both groups")
    ag = aggregate_seeds(ra, ["dataset"]); bg = aggregate_seeds(rb, ["dataset"])
    dsets = sorted({r["dataset"] for r in ag} | {r["dataset"] for r in bg})
    sa = [next((a["mean"] for a in ag if a["dataset"]==d), 0.0) for d in dsets]
    sb = [next((a["mean"] for a in bg if a["dataset"]==d), 0.0) for d in dsets]
    ea = [next((a["std"] for a in ag if a["dataset"]==d), 0.0) for d in dsets]
    eb = [next((a["std"] for a in bg if a["dataset"]==d), 0.0) for d in dsets]
    fig, ax = plt.subplots(figsize=(ps.fig_width(True), ps.fig_width(True)*0.5), constrained_layout=True)
    grouped_bar(ax, dsets, {"with Stage-1": sa, "from scratch": sb},
                {"with Stage-1": ea, "from scratch": eb},
                colors={"with Stage-1": ps.METHOD_COLORS["syncx"], "from scratch": ps.METHOD_COLORS["baseline_avff"]})
    ax.set_title("Stage-1 pretraining contribution (A2)")
    ps.savefig(fig, out)


def plot_f7(lam_maps, out):
    if not lam_maps: raise FileNotFoundError("need --lam_scan for lambda sweep (F7)")
    xs, means, stds = [], [], []
    for g, mp in lam_maps:
        ra = _filter(_flatten(mp), variant="full", method="syncx", group=[None, ""])
        if not ra: continue
        ag = aggregate_seeds(ra, ["dataset"])
        # 跨数据集平均 AUC 作为该 λ 的标量
        m = float(np.mean([a["mean"] for a in ag])) if ag else float("nan")
        s = float(np.mean([a["std"] for a in ag])) if ag else 0.0
        lam = float(g.split("lam")[-1].split("_")[0]) if "lam" in g else len(xs)
        xs.append(lam); means.append(m); stds.append(s)
    if not xs: raise FileNotFoundError("no syncx full records in lam_scan groups")
    order = np.argsort(xs); xs = [xs[i] for i in order]; means = [means[i] for i in order]; stds = [stds[i] for i in order]
    best_i = int(np.argmax(means))
    fig, ax = plt.subplots(figsize=(ps.fig_width(True), ps.fig_width(True)*0.5), constrained_layout=True)
    ax.plot(xs, means, color=ps.METHOD_COLORS["syncx"], lw=ps.SYNCX_LW, marker="o", ms=4, label=r"Sync-X AUC")
    ax.fill_between(xs, np.array(means)-np.array(stds), np.array(means)+np.array(stds),
                    color=ps.METHOD_COLORS["syncx"], alpha=0.15)
    ax.axvline(xs[best_i], color="black", lw=1.0, ls="--")
    ax.text(xs[best_i], min(means), f"best $\\lambda$={xs[best_i]:g}", rotation=90, fontsize=7, va="bottom", ha="right")
    ax.set_xlabel(r"$\lambda_{corr}$"); ax.set_ylabel("Mean AUC (%)"); ax.set_ylim(40, 102)
    ax.set_title("Cross-modal correspondence weight sweep (A3)")
    ps.savefig(fig, out)


def plot_f8(eval_map, out):
    rows = _filter(_flatten(eval_map), variant=["laca_both", "laca_corr", "laca_biomech"], group=[None, ""])
    if not rows: raise FileNotFoundError("no LACA-mode eval records")
    agg = aggregate_seeds(rows, ["dataset", "variant"])
    dsets = sorted({r["dataset"] for r in agg})
    variants = ["laca_both", "laca_corr", "laca_biomech"]
    series = {v: [next((a["mean"] for a in agg if a["dataset"]==d and a["variant"]==v), 0.0) for d in dsets] for v in variants}
    errors = {v: [next((a["std"] for a in agg if a["dataset"]==d and a["variant"]==v), 0.0) for d in dsets] for v in variants}
    fig, ax = plt.subplots(figsize=(ps.fig_width(True), ps.fig_width(True)*0.5), constrained_layout=True)
    grouped_bar(ax, dsets, series, errors, colors=ps.VARIANT_COLORS, hatches=ps.HATCHES)
    ax.set_title("LACA branch ablation (B)")
    ps.savefig(fig, out)


def plot_f9(eval_map, out):
    rows = _filter(_flatten(eval_map), variant=["b6_untrimmed", "b6_trimmed"], method=["syncx", "mds"])
    if not rows: raise FileNotFoundError("no B6 (silence trim) eval records")
    agg = aggregate_seeds(rows, ["variant", "method"])
    conds = ["b6_untrimmed", "b6_trimmed"]
    methods = ["syncx", "mds"]
    series = {m: [next((a["mean"] for a in agg if a["variant"]==c and a["method"]==m), 0.0) for c in conds] for m in methods}
    errors = {m: [next((a["std"] for a in agg if a["variant"]==c and a["method"]==m), 0.0) for c in conds] for m in methods}
    fig, ax = plt.subplots(figsize=(ps.fig_width(True), ps.fig_width(True)*0.5), constrained_layout=True)
    grouped_bar(ax, ["untrimmed", "trimmed"], series, errors,
                colors={"syncx": ps.METHOD_COLORS["syncx"], "mds": ps.METHOD_COLORS["mds"]})
    ax.set_title("Silence-trim robustness (B6)")
    ps.savefig(fig, out)


def plot_f10(eval_map, out):
    rows = _filter(_flatten(eval_map), variant=["b7_with_exclude_ids", "b7_without_exclude_ids"], group=[None, ""])
    if not rows: raise FileNotFoundError("no B7 (identity exclusion) eval records")
    agg = aggregate_seeds(rows, ["dataset", "variant"])
    dsets = sorted({r["dataset"] for r in agg})
    variants = ["b7_with_exclude_ids", "b7_without_exclude_ids"]
    series = {v: [next((a["mean"] for a in agg if a["dataset"]==d and a["variant"]==v), 0.0) for d in dsets] for v in variants}
    errors = {v: [next((a["std"] for a in agg if a["dataset"]==d and a["variant"]==v), 0.0) for d in dsets] for v in variants}
    fig, ax = plt.subplots(figsize=(ps.fig_width(True), ps.fig_width(True)*0.5), constrained_layout=True)
    grouped_bar(ax, dsets, series, errors,
                colors={"b7_with_exclude_ids": ps.METHOD_COLORS["syncx"], "b7_without_exclude_ids": ps.METHOD_COLORS["baseline_avff"]})
    ax.set_title("Identity-leakage exclusion (B7)")
    ps.savefig(fig, out)


def plot_f11(rec_real, rec_fake, out):
    fig, axes = plt.subplots(2, 2, figsize=(ps.fig_width(True), ps.fig_width(True)*0.5), constrained_layout=True)
    for row, (rec, title) in enumerate([(rec_real[0], "Real"), (rec_fake[0], "Fake")]):
        attn = np.array(rec["attention"]); anom = np.array(rec["anomaly_scores"])
        axH, axC = axes[row]
        im = axH.imshow(attn, cmap="cividis", aspect="auto")
        axH.set_title(f"{title}: cross-modal attention"); axH.set_xlabel("key frame"); axH.set_ylabel("query frame")
        fig.colorbar(im, ax=axH, fraction=0.046, pad=0.04)
        axC.plot(anom, color=("#009E73" if title == "Real" else "#D55E00"), lw=ps.SYNCX_LW, label="frame anomaly")
        fs = rec.get("forgery_segment")
        if fs:
            axC.axvspan(fs[0], fs[1], color="#D55E00", alpha=0.15, label="forgery segment")
        axC.set_title(f"{title}: per-frame A-V distance"); axC.set_xlabel("frame"); axC.set_ylabel("anomaly (norm.)")
        axC.legend(fontsize=6)
    ps.savefig(fig, out)


def plot_f12(feat, out):
    if not feat: raise FileNotFoundError("no clip_features.jsonl")
    jerk = np.array([r["biomech_jerk"] for r in feat])
    corr = np.array([r["corr_score"] for r in feat])
    labels = np.array([int(r["label"]) for r in feat])
    fig, ax = plt.subplots(figsize=(ps.fig_width(True), ps.fig_width(True)*0.5), constrained_layout=True)
    ax.scatter(jerk[labels == 0], corr[labels == 0], c=ps.VARIANT_COLORS["Real"], s=14, alpha=0.6, label="Real", edgecolor="none")
    ax.scatter(jerk[labels == 1], corr[labels == 1], c=ps.VARIANT_COLORS["Fake"], s=14, alpha=0.6, label="Fake", marker="x")
    ax.set_xlabel("Biomechanical jerk (norm.)"); ax.set_ylabel("Corr-score"); ax.legend(fontsize=8)
    ax.set_title("Dual-cue separation (F12)")
    ps.savefig(fig, out)


def plot_f13(prof, out):
    if not prof: raise FileNotFoundError("no profile.jsonl")
    cats = ["Peak VRAM (GB)", "Epoch (min)", "Infer (ms)"]
    vals = [prof.get("peak_vram_gb"), prof.get("stage2_epoch_min"), prof.get("inference_latency_ms")]
    if any(v is None for v in vals): raise FileNotFoundError("profile.jsonl missing fields")
    vals = [float(v) for v in vals]
    fig, ax = plt.subplots(figsize=(ps.fig_width(True)*0.6, ps.fig_width(True)*0.5), constrained_layout=True)
    ax.bar(cats, vals, color=[ps.METHOD_COLORS["syncx"], ps.METHOD_COLORS["baseline_avff"], ps.METHOD_COLORS["baseline_mmmsba"]],
           edgecolor="black", linewidth=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Compute profile (F13)")
    ps.savefig(fig, out)


def _pca_np(X, n=2):
    """纯 NumPy PCA（不依赖 sklearn）：中心化后取前 n 个主成分。"""
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = (Xc @ Vt[:n].T) * (S[:n] / np.sqrt(max(X.shape[0] - 1, 1)))
    return Z

def plot_f15(emb, out):
    if not emb: raise FileNotFoundError("no clip_embeddings.jsonl")
    X = np.array([r["embedding"] for r in emb], dtype=float)
    labels = np.array([int(r["label"]) for r in emb])
    # 离线降维：优先 t-SNE（需 sklearn），否则纯 NumPy PCA 兜底（F15 不硬依赖 sklearn）
    try:
        from sklearn.manifold import TSNE
        Z = TSNE(n_components=2, init="pca", random_state=0).fit_transform(X)
        method = "t-SNE"
    except Exception:
        Z = _pca_np(X, 2)
        method = "PCA (numpy fallback)"
    fig, ax = plt.subplots(figsize=(ps.fig_width(True), ps.fig_width(True)*0.62), constrained_layout=True)
    ax.scatter(Z[labels == 0, 0], Z[labels == 0, 1], c=ps.VARIANT_COLORS["Real"], s=12, alpha=0.6, label="Real", edgecolor="none")
    ax.scatter(Z[labels == 1, 0], Z[labels == 1, 1], c=ps.VARIANT_COLORS["Fake"], s=12, alpha=0.6, label="Fake", marker="x")
    ax.set_title(f"Representation separability ({method}, F15)"); ax.set_xlabel("dim-1"); ax.set_ylabel("dim-2"); ax.legend(fontsize=8)
    ps.savefig(fig, out)


def plot_f16(eval_map, out):
    rows = [r for r in _flatten(eval_map) if str(r.get("variant", "")).startswith("perturb")]
    if not rows: raise FileNotFoundError("no robustness/perturbation eval records (variant startswith 'perturb')")
    agg = aggregate_seeds(rows, ["variant", "method"])
    conds = sorted({r["variant"] for r in agg})
    methods = sorted({r["method"] for r in agg})
    series = {m: [next((a["mean"] for a in agg if a["variant"]==c and a["method"]==m), 0.0) for c in conds] for m in methods}
    errors = {m: [next((a["std"] for a in agg if a["variant"]==c and a["method"]==m), 0.0) for c in conds] for m in methods}
    fig, ax = plt.subplots(figsize=(ps.fig_width(True), ps.fig_width(True)*0.5), constrained_layout=True)
    grouped_bar(ax, conds, series, errors, colors=ps.METHOD_COLORS)
    ax.set_title("Robustness to perturbations (F16, optional)")
    ps.savefig(fig, out)


def plot_component_ablation(eval_map, out):
    """组件消融 (C)：full vs w/o LCMA vs w/o DWF，对应论文 tab:component-ablation。
    读取 variant ∈ {full, wo_lcma, wo_dwf} 的评测日志（跨数据集分组聚合）。
    """
    rows = _filter(_flatten(eval_map), variant=["full", "wo_lcma", "wo_dwf"], group=[None, ""])
    if not rows: raise FileNotFoundError("no component-ablation eval records (full/wo_lcma/wo_dwf)")
    agg = aggregate_seeds(rows, ["dataset", "variant"])
    dsets = sorted({r["dataset"] for r in agg})
    variants = ["full", "wo_lcma", "wo_dwf"]
    series = {v: [next((a["mean"] for a in agg if a["dataset"]==d and a["variant"]==v), 0.0) for d in dsets] for v in variants}
    errors = {v: [next((a["std"] for a in agg if a["dataset"]==d and a["variant"]==v), 0.0) for d in dsets] for v in variants}
    fig, ax = plt.subplots(figsize=(ps.fig_width(True), ps.fig_width(True)*0.5), constrained_layout=True)
    grouped_bar(ax, dsets, series, errors,
                colors={"full": ps.METHOD_COLORS["syncx"],
                        "wo_lcma": "#7F7F7F",
                        "wo_dwf": "#BFBFBF"})
    ax.set_title("Component ablation (C): w/o LCMA / w/o DWF")
    ps.savefig(fig, out)


# ================= main =================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default="logs")
    ap.add_argument("--exp_id", default="syncx_bs256_lam0.3_lacaboth_seed0")
    ap.add_argument("--exp_id_stage1", default=None,
                    help="F1 的 Stage-1 log 目录 exp_id（缺省从 --exp_id 派生 bs{batch_size_stage1}）")
    ap.add_argument("--exp_id_stage2", default=None,
                    help="F1 的 Stage-2 log 目录 exp_id（缺省从 --exp_id 派生 bs{batch_size_stage2}）")
    ap.add_argument("--exp_group", default="syncx_bs256_lam0.3_lacaboth")
    ap.add_argument("--exp_group_b", default=None, help="F6 对照（from-scratch）exp_group")
    ap.add_argument("--lam_scan", default=None, help="F7 逗号分隔的 exp_group 列表")
    ap.add_argument("--case_real", default="real01")
    ap.add_argument("--case_fake", default="fake01")
    ap.add_argument("--out", default="out/figures")
    args = ap.parse_args()

    ps.setup_style()
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    lam_maps = None
    if args.lam_scan:
        lam_maps = [(g, _gather_exp_group(args.log_dir, g)) for g in args.lam_scan.split(",") if g]

    jobs = []  # (num, fn, kwargs, path)

    def add(num, fn, **kw):
        jobs.append((num, fn, kw, os.path.join(out_dir, f"F{num}")))

    # 预加载（缺则对应图抛 FileNotFoundError）
    # F1 需要 stage1 和 stage2 两个 log，而二者 exp_id 不同
    # （stage1 用 cfg.batch_size_stage1 → bs1024；stage2 用 cfg.batch_size_stage2 → bs512）。
    # 因此分开探测：优先显式 --exp_id_stage1/--exp_id_stage2，否则从 --exp_id 派生（兼容旧用法）。
    try:
        from config import get_config
        _cfg = get_config()
    except Exception:
        _cfg = None
    eid_s1 = args.exp_id_stage1 or (args.exp_id.replace("bs512", f"bs{_cfg.batch_size_stage1}")
                                    if _cfg and args.exp_id and f"bs512" in args.exp_id else args.exp_id)
    eid_s2 = args.exp_id_stage2 or (args.exp_id.replace(f"bs{_cfg.batch_size_stage1}", f"bs{_cfg.batch_size_stage2}")
                                    if _cfg and args.exp_id else args.exp_id)
    try: s1 = load_train_stage1(args.log_dir, eid_s1)
    except FileNotFoundError as e: s1 = None
    try: s2 = load_train_stage2(args.log_dir, eid_s2)
    except FileNotFoundError as e: s2 = None
    eval_main = _gather_exp_group(args.log_dir, args.exp_group)
    eval_b = _gather_exp_group(args.log_dir, args.exp_group_b) if args.exp_group_b else None
    try: ir = load_interpret(args.log_dir, args.exp_id, args.case_real)
    except FileNotFoundError: ir = None
    try: iff = load_interpret(args.log_dir, args.exp_id, args.case_fake)
    except FileNotFoundError: iff = None
    try: feat = load_features(args.log_dir, args.exp_id)
    except FileNotFoundError: feat = None
    try: emb = load_embeddings(args.log_dir, args.exp_id)
    except FileNotFoundError: emb = None
    try: prof = load_profile(args.log_dir, args.exp_id)
    except FileNotFoundError: prof = {}

    if s1 and s2: add(1, plot_f1, s1=s1, s2=s2)
    add(2, plot_f2, eval_map=eval_main)
    add(3, plot_f3, eval_map=eval_main)
    add(4, plot_f4, eval_map=eval_main)
    add(5, plot_f5, eval_map=eval_main)
    add(6, plot_f6, eval_main=eval_main, eval_b=eval_b)
    add(7, plot_f7, lam_maps=lam_maps)
    add(8, plot_f8, eval_map=eval_main)
    add(9, plot_f9, eval_map=eval_main)
    add(10, plot_f10, eval_map=eval_main)
    if ir and iff: add(11, plot_f11, rec_real=ir, rec_fake=iff)
    if feat is not None: add(12, plot_f12, feat=feat)
    if prof: add(13, plot_f13, prof=prof)
    add(14, plot_f14, eval_map=eval_main)
    if emb is not None: add(15, plot_f15, emb=emb)
    add(16, plot_f16, eval_map=eval_main)
    add(17, plot_component_ablation, eval_map=eval_main)   # 组件消融 (C) — 对应 tab:component-ablation；图计划需 +1

    failed = []
    for num, fn, kw, path in jobs:
        try:
            fn(out=path, **kw)
            print(f"[fig] F{num} -> {path}.pdf")
        except FileNotFoundError as e:
            failed.append((num, str(e)))
            print(f"[fig] F{num} SKIPPED (missing: {e})")
        except Exception as e:
            failed.append((num, f"{type(e).__name__}: {e}"))
            print(f"[fig] F{num} ERROR: {e}")
    print(f"\n=== rendered {len(jobs)-len(failed)}/{len(jobs)} figures; skipped/errored: {[f[0] for f in failed]} ===")
    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
