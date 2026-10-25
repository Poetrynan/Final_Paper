"""
logging_utils.py — 训练/评测/推理日志持久化工具
================================================================
Sync-X 出图零重训方案的核心：所有 F1–F16 的数据都来自这里落盘的 JSONL 日志。
出图脚本 (plot_figures.py) 只读这些日志，绝不触发 train/eval/infer。

提供：
  - exp_id / exp_group : 由超参+seed 构造实验 ID（用于目录与跨 seed 聚合）
  - append_jsonl / read_jsonl : 安全的 JSONL 追加/读取
  - roc_curve : 由 y_true/y_score 生成降采样 (fpr,tpr)，供 F2 重画无需重评
  - ema : 指数滑动平均，去 mini-batch 噪声（F1 loss 曲线用）
  - aggregate_seeds : 同 key 跨 seed 求 mean±std
"""
import os
import json
import glob
import re
from datetime import datetime


# ---------------------------------------------------------------------------
# 实验 ID
# ---------------------------------------------------------------------------
def exp_id(cfg, seed, tag=""):
    """构造 exp_id：syncx_bs{bs}_lam{lambda}_laca{mode}[_nolcma|_nodwf][_tag]_seed{seed}。"""
    eid = f"syncx_bs{cfg.batch_size_stage2}_lam{cfg.lambda_corr_lcma}_laca{cfg.laca_mode}"
    if getattr(cfg, "disable_lcma", False):
        eid += "_nolcma"
    if getattr(cfg, "disable_dwf", False):
        eid += "_nodwf"
    if tag:
        eid += f"_{tag}"
    return f"{eid}_seed{seed}"


def exp_group(eid):
    """去掉 _seedN，得到种子无关的组 ID（用于跨 seed 聚合 + 基线/文献值注入）。"""
    return re.sub(r"_seed\d+$", "", eid)


# ---------------------------------------------------------------------------
# JSONL 读写
# ---------------------------------------------------------------------------
def _default(o):
    """json 默认序列化：numpy/torch 标量与张量 → 原生类型。"""
    import numpy as np
    import torch
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def append_jsonl(path, record):
    """追加一行 JSONL（自动建目录、flush，防崩溃丢尾）。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    record.setdefault("timestamp", datetime.utcnow().isoformat())
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=_default) + "\n")
        f.flush()


def read_jsonl(path):
    """读取 JSONL，文件不存在返回 []；跳过空行。"""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def append_jsonl_dedup(path, record, key_fields):
    """幂等追加 JSONL：已有相同 key_fields 的记录则跳过。

    用于"评测/验证记录"这类重跑会重复的落盘点（中断发生在记录写入后、
    marker 创建前时，重跑会追加重复行，污染出图数据）。
    返回 True=已写入，False=因重复被跳过。
    """
    if os.path.exists(path):
        for r in read_jsonl(path):
            if all(r.get(k) == record.get(k) for k in key_fields):
                return False
    append_jsonl(path, record)
    return True


# ---------------------------------------------------------------------------
# ROC / 指标
# ---------------------------------------------------------------------------
def roc_curve(y_true, y_score, n=2000):
    """由标签/分数生成降采样 (fpr, tpr) 数组，供 F2 重画无需重评。

    与 utils.auc_score 同样的逻辑：按分数降序排，累计 TP/FP 得 TPR/FPR。
    """
    import numpy as np
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score).astype(float)
    order = np.argsort(-s)
    y = y[order]
    P = y.sum()
    N = len(y) - P
    if P == 0 or N == 0:
        return [0.0, 1.0], [0.0, 1.0]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = np.r_[0, tp / P]
    fpr = np.r_[0, fp / N]
    idx = np.linspace(0, len(fpr) - 1, n).astype(int)
    return fpr[idx].tolist(), tpr[idx].tolist()


def ema(ys, alpha=0.05):
    """指数滑动平均，去 mini-batch 噪声（F1 loss 曲线用）。"""
    out = []
    s = ys[0]
    for y in ys:
        s = alpha * y + (1 - alpha) * s
        out.append(s)
    return out


def aggregate_seeds(rows, key_cols, val_col="AUC"):
    """同 key_cols 跨 seed 求 mean±std；返回 [{**key, mean, std}]。"""
    import numpy as np
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in rows:
        k = tuple(r[c] for c in key_cols)
        v = r["metrics"][val_col] if isinstance(r.get("metrics"), dict) else r.get(val_col)
        if v is not None and v == v:  # 跳过 nan
            buckets[k].append(v)
    out = []
    for k, vs in buckets.items():
        vs = np.array(vs, dtype=float)
        out.append({
            **dict(zip(key_cols, k)),
            "mean": float(vs.mean()),
            "std": float(vs.std(ddof=0) if len(vs) > 1 else 0.0),
        })
    return out
