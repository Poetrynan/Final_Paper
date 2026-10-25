"""
utils.py — 训练/评测共用工具
================================================================
  优化器、调度器、混合精度、评测指标（AUC/AP/EER/Acc/F1）。
"""
import math
import os
import random
import numpy as np
import torch
from torch.optim import AdamW


# ── 断点续跑（云端竞价实例/意外中断必备）─────────────────────────────────
def capture_rng():
    """捕获 CPU/CUDA/numpy/random 全部随机状态，用于训练续跑后复现。"""
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "random": random.getstate(),
    }


def restore_rng(d):
    """恢复 capture_rng() 捕获的随机状态。带类型校验：旧格式/损坏的 resume 状态跳过，
    避免 torch.set_rng_state 因非 ByteTensor 崩溃（云端续跑健壮性）。"""
    if d is None:
        return
    try:
        t = d.get("torch")
        if t is not None and isinstance(t, torch.Tensor) and t.dtype == torch.uint8:
            torch.set_rng_state(t)
    except Exception:
        pass
    try:
        c = d.get("cuda")
        if c is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(c)
    except Exception:
        pass
    try:
        if d.get("numpy") is not None:
            np.random.set_state(d["numpy"])
    except Exception:
        pass
    try:
        if d.get("random") is not None:
            random.setstate(d["random"])
    except Exception:
        pass


def strip_compile_prefix(sd):
    """清洗 torch.compile 产生的 _orig_mod. 前缀。

    torch.compile 会把模块包装成 _orig_mod.*，导致 state_dict 的键带上前缀。
    Stage-1 编译过、Stage-2 加载时不编译，键不一致会让所有权重 load 失败（白训）。
    此函数递归清理 state_dict 里任意层级的 _orig_mod. 前缀，使键恢复裸模块格式。
    """
    if isinstance(sd, dict):
        # 仅对字符串键去前缀；非字符串键（如 optimizer state 的 int 参数索引）原样透传，
        # 避免 AttributeError('int' has no 'removeprefix')。值递归处理。
        return {k.removeprefix("_orig_mod.") if isinstance(k, str) else k: strip_compile_prefix(v)
                for k, v in sd.items()}
    return sd


def save_checkpoint_atomic(path, state):
    """原子保存检查点：先写 .tmp 再 os.replace，避免中断写出半个文件导致无法续跑。
    保存前自动 strip torch.compile 的 _orig_mod. 前缀，保证跨脚本（Stage1 编译 / Stage2 不编译）加载兼容。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    wrapped = {k: strip_compile_prefix(v) for k, v in state.items()}
    tmp = f"{path}.tmp"
    torch.save(wrapped, tmp)
    os.replace(tmp, path)


def build_scheduler(optimizer, warmup_steps, total_steps):
    """warmup (线性) -> cosine 衰减。"""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def autocast_ctx(amp=True):
    return torch.autocast(device_type="cuda", enabled=amp)


# ── 评测指标 ─────────────────────────────────────────────────────────
def _to_numpy(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def auc_score(y_true, y_score):
    y_true = _to_numpy(y_true).astype(int)
    y_score = _to_numpy(y_score).astype(float)
    # ROC-AUC (trapezoid on TPR-FPR)
    order = np.argsort(-y_score)
    y = y_true[order]
    P = y.sum(); N = len(y) - P
    if P == 0 or N == 0:
        return float("nan")
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    tpr = tp / P; fpr = fp / N
    tpr = np.r_[0, tpr]; fpr = np.r_[0, fpr]
    _trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2.0 renamed trapz->trapezoid
    return float(_trapz(tpr, fpr))


def ap_score(y_true, y_score):
    y_true = _to_numpy(y_true).astype(int)
    y_score = _to_numpy(y_score).astype(float)
    order = np.argsort(-y_score)
    y = y_true[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    recall = tp / max(1, y_true.sum())
    precision = tp / np.maximum(1, tp + fp)
    # AP = sum (R_n - R_{n-1}) * P_n
    recall = np.r_[0, recall]; precision = np.r_[1, precision]
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def eer_score(y_true, y_score):
    y_true = _to_numpy(y_true).astype(int)
    y_score = _to_numpy(y_score).astype(float)
    order = np.argsort(-y_score)
    y = y_true[order]
    P = y.sum(); N = len(y) - P
    if P == 0 or N == 0:
        return float("nan")
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    fpr = fp / N; fnr = (P - tp) / P
    # EER = FPR == FNR 交点
    diff = fpr - fnr
    idx = np.argmin(np.abs(diff))
    return float((fpr[idx] + fnr[idx]) / 2)


def accuracy(y_true, y_pred):
    y_true = _to_numpy(y_true).astype(int)
    y_pred = _to_numpy(y_pred).astype(int)
    return float((y_true == y_pred).mean()) if len(y_true) else float("nan")


def f1_score(y_true, y_pred):
    y_true = _to_numpy(y_true).astype(int)
    y_pred = _to_numpy(y_pred).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    return float(2 * prec * rec / max(1e-9, prec + rec))


def all_metrics(y_true, y_score, threshold=0.5):
    y_true = _to_numpy(y_true).astype(int)
    y_score = _to_numpy(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    return {
        "AUC": auc_score(y_true, y_score),
        "AP": ap_score(y_true, y_score),
        "EER": eer_score(y_true, y_score),
        "ACC": accuracy(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
    }
