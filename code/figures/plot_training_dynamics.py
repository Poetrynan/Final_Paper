"""
plot_training_dynamics.py — 训练动力学出图（与计算彻底解耦）
================================================================
只读 logs/<exp_id>/{stage1,stage2}_train.jsonl，绝不触发训练/抽取/评测。
跑一次训练后，这里就能零重算地画出论文需要的"训练动态"图：

  figures/stage2_total_loss.png      总损失随 epoch（EMA 平滑）
  figures/stage2_components.png      5 项分量损失（AAM/corr/shared/align/uniform）
  figures/stage2_val_auc.png         DeepSpeak val AUC + 早停(best)标记
  figures/stage1_losses.png          Stage-1 三项损失 + val 检索 acc

用法：
  python figures/plot_training_dynamics.py --log_dir logs
  python figures/plot_training_dynamics.py --log_dir logs --exp_id syncx_bs512_..._seed0

依赖：matplotlib（CPU only）。若无：pip install matplotlib。
"""
import os
import sys
import glob
import json
import argparse
from collections import defaultdict

import numpy as np

# matplotlib 延迟导入：模块在没有 matplotlib 的环境也能 import（便于本地校验聚合逻辑），
# 真正出图时才需要它（云端已装）。
_PLT = None
def _plt():
    global _PLT
    if _PLT is None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _PLT = plt
    return _PLT

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logging_utils import read_jsonl


def _aggregate_per_epoch(records, phase="train"):
    """按 epoch 聚合 phase 记录，返回 {epoch: {field: [vals]}}。"""
    by_ep = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.get("phase") != phase:
            continue
        ep = r.get("epoch")
        if ep is None:
            continue
        for k, v in r.items():
            if k in ("epoch", "phase", "global_step", "timestamp", "lr"):
                continue
            if isinstance(v, (int, float)) and v == v:  # 跳过 nan/None
                by_ep[ep][k].append(v)
    return by_ep


def _epoch_means(by_ep, fields):
    eps = sorted(by_ep.keys())
    out = {f: [] for f in fields}
    out["epoch"] = eps
    for ep in eps:
        for f in fields:
            vals = by_ep[ep].get(f, [])
            out[f].append(float(np.mean(vals)) if vals else float("nan"))
    return out


def _find_best_val(val_by_ep):
    best_ep, best_auc = None, -1.0
    for ep in sorted(val_by_ep.keys()):
        aucs = val_by_ep[ep].get("val_auc", [])
        if aucs:
            a = float(np.mean(aucs))
            if a == a and a > best_auc:
                best_auc, best_ep = a, ep
    return best_ep, best_auc


def plot_stage2(path, out_dir):
    recs = read_jsonl(path)
    if not recs:
        print(f"[plot] 空或无文件：{path}")
        return
    train_ep = _aggregate_per_epoch(recs, "train")
    val_ep = _aggregate_per_epoch(recs, "val")
    comp_fields = ["L_total", "L_AAM", "L_corr_weighted", "L_shared_weighted", "L_align", "L_uniform"]
    tm = _epoch_means(train_ep, comp_fields)
    epochs = tm["epoch"]

    # 1) 总损失
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, tm["L_total"], color="#1f77b4", lw=1.6, label="total loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("total loss"); ax.set_title("Stage-2 total loss")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "stage2_total_loss.png"), dpi=130); plt.close(fig)

    # 2) 分量损失
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"L_AAM": "#1f77b4", "L_corr_weighted": "#ff7f0e", "L_shared_weighted": "#2ca02c",
              "L_align": "#d62728", "L_uniform": "#9467bd"}
    for f in ["L_AAM", "L_corr_weighted", "L_shared_weighted", "L_align", "L_uniform"]:
        ax.plot(epochs, tm[f], lw=1.4, label=f, color=colors.get(f))
    ax.set_xlabel("epoch"); ax.set_ylabel("weighted loss"); ax.set_title("Stage-2 component losses (λ-weighted)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "stage2_components.png"), dpi=130); plt.close(fig)

    # 3) val AUC + best 标记
    veps = sorted(val_ep.keys())
    if veps:
        aucs = [float(np.mean(val_ep[e].get("val_auc", [float("nan")]))) for e in veps]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(veps, aucs, color="#2ca02c", lw=1.8, marker="o", ms=3, label="val AUC (DeepSpeak)")
        best_ep, best_auc = _find_best_val(val_ep)
        if best_ep is not None:
            ax.axvline(best_ep, color="#d62728", ls="--", lw=1.2,
                       label=f"best @ ep{best_ep} ({best_auc:.3f})")
        ax.set_xlabel("epoch"); ax.set_ylabel("AUC"); ax.set_title("Stage-2 validation AUC (early-stop monitor)")
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "stage2_val_auc.png"), dpi=130); plt.close(fig)
        print(f"[plot] stage2: best val AUC = {best_auc:.4f} @ epoch {best_ep}")
    print(f"[plot] stage2 动力学图已写出 -> {out_dir}/stage2_*.png")


def plot_stage1(path, out_dir):
    recs = read_jsonl(path)
    if not recs:
        print(f"[plot] 空或无文件：{path}")
        return
    train_ep = _aggregate_per_epoch(recs, "train")
    val_ep = _aggregate_per_epoch(recs, "val")
    fields = ["train_infoNCE_loss", "train_kin_loss", "train_shared_loss"]
    tm = _epoch_means(train_ep, fields)
    epochs = tm["epoch"]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"train_infoNCE_loss": "#1f77b4", "train_kin_loss": "#ff7f0e", "train_shared_loss": "#2ca02c"}
    for f in fields:
        ax.plot(epochs, tm[f], lw=1.4, label=f, color=colors.get(f))
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title("Stage-1 SSL losses")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "stage1_losses.png"), dpi=130); plt.close(fig)

    veps = sorted(val_ep.keys())
    if veps:
        accs = [float(np.mean(val_ep[e].get("val_contrastive_acc", [float("nan")]))) for e in veps]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(veps, accs, color="#9467bd", lw=1.8, marker="o", ms=3, label="val contrastive acc")
        ax.set_xlabel("epoch"); ax.set_ylabel("acc"); ax.set_title("Stage-1 validation retrieval acc")
        ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "stage1_val_acc.png"), dpi=130); plt.close(fig)
    print(f"[plot] stage1 动力学图已写出 -> {out_dir}/stage1_*.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default="logs")
    ap.add_argument("--exp_id", default=None, help="只画某个 exp_id；缺省画所有")
    ap.add_argument("--out_dir", default=None, help="缺省 = <log_dir>/../figures")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.log_dir)), "figures")
    os.makedirs(out_dir, exist_ok=True)

    targets = []
    if args.exp_id:
        targets = [(args.exp_id, os.path.join(args.log_dir, args.exp_id))]
    else:
        for d in sorted(glob.glob(os.path.join(args.log_dir, "*"))):
            if os.path.isdir(d):
                targets.append((os.path.basename(d), d))

    found = False
    for eid, d in targets:
        s2 = os.path.join(d, "stage2_train.jsonl")
        s1 = os.path.join(d, "stage1_train.jsonl")
        if os.path.exists(s2):
            found = True; print(f"[plot] Stage-2 exp={eid}"); plot_stage2(s2, out_dir)
        if os.path.exists(s1):
            found = True; print(f"[plot] Stage-1 exp={eid}"); plot_stage1(s1, out_dir)
    if not found:
        print(f"[plot] 在 {args.log_dir} 下未找到任何 stage{{1,2}}_train.jsonl。"
              f"请先跑训练（cloud_run_all.sh）。")
        sys.exit(0)
    print(f"[plot] 全部完成 -> {out_dir}")


if __name__ == "__main__":
    main()
