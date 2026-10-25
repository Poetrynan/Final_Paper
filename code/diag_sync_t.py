#!/usr/bin/env python
"""
diag_sync_t.py — 方向B「LTCA sync 标量」跨生成器可分性实测（诚实计量，探针）
====================================================================================
背景：方向B(LTCA) 实测 DFT2024 首顶 0.55、但 HiFi-AVDF 反降(0.499→0.4834)。
待验证假设（见 evidence_chain §8D 待办 + 调研）：HiFi 假视频是 T2V/I2V **一体生成**、
lip-sync 高度成熟 → LTCA 子系统B 的显式时序相干标量
      sync_t = mean_t cos( a_seq[t], v_seq[t] )   (a_seq/v_seq = LCMA post-attn 内容嵌入, T=50)
在 HiFi 上 real vs fake **无可分信号** → sync 项不带来增益、甚至负迁移。

本脚本加载「方向B 训好的 ckpt（含 fta.* 权重）」，对训练域 + 三跨域集，
复用模型真实前向，抽取**模型真正使用的 sync_t** 并测可分性：
  - 若 HiFi 上 sync_t 可分(≥0.65) → sync 项无责，负迁移在别处(长时骨干/门控)。
  - 若 HiFi 上 sync_t ≈随机(<0.6) → 坐实"sync 在 HiFi 无信号"，方向B 需关 sync 或改监督。

口径（与 evidence_chain §5/§6 一致）：
  - auc(实0)：real=0 计分，>0.5 表示 real 的 sync_t 更高
  - strength=max(auc,1-auc)；cohen_d(+ = real 更大)
  - 跨集 cohen_d 符号翻转 ⇒ 生成器专属、不迁移（方向B sync 线判死）
  - 只信 DFE2024/HiFi（干净平衡），FakeAVCeleb 仅参考

用法（云端，需 GPU；跑一次约几百样本 × 单次 lcma 前向）：

    python diag_sync_t.py \
        --ckpt $PERSIST/ckpt/syncx_best_seed0.pt \
        --cache $Datasets/DeepSpeak_v1_1/clips/prep/train \
        --cache $Datasets/Deepfake-Eval-2024/prep \
        --cache $Datasets/HiFi-AVDF/prep \
        --cache $Datasets/FakeAVCeleb_v1.2/prep \
        --max-per-class 600

附带输出：alsp 打印各集 sync_t 的 mean/std（real/fake），供看一眼分布尺度。
"""
import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_config
from models import SyncX
from diag_shortcut import roc_auc_y_score, cohens_d, load_index


def batch_sync_t(model, cfg, items, dev, max_per_class):
    """逐 mini-batch 前向 lcma，返回 (sync_t 数组, label 数组, 输出列嵌套 dict)。"""
    model.eval()
    syncs, labels = [], []
    with torch.inference_mode():
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        real = [ix for ix in items if ix["label"] == 0]
        fake = [ix for ix in items if ix["label"] == 1]
        real = list(rng.choice(real, size=min(max_per_class, len(real)), replace=False)) if real else []
        fake = list(rng.choice(fake, size=min(max_per_class, len(fake)), replace=False)) if fake else []
        rows = real + fake
        for i, r in enumerate(rows):
            try:
                d = torch.load(r["path"], map_location="cpu", weights_only=True)
            except Exception:
                continue
            a_seq = d["audio_seq"].float()          # (200,768)
            v_clip = d["v_clip"].float()            # (100,512)
            # 对齐 evaluate.py 的 _fix_len 语义：直接截断/补零到 (200)/(100)
            a_seq = a_seq[:200].clone().to(dev, non_blocking=True)
            v_clip = v_clip[:100].clone().to(dev, non_blocking=True)
            # 长 200 但不足时补零
            a_seq = torch.nn.functional.pad(a_seq, (0, 0, 0, max(0, 200 - a_seq.size(0))))
            v_clip = torch.nn.functional.pad(v_clip, (0, 0, 0, max(0, 100 - v_clip.size(0))))
            a_seq, v_clip = a_seq[:200], v_clip[:100]
            # LCMA post-attn 内容嵌入 = LTCA 的输入 a_lcma/v_lcma
            try:
                h_lcma, attn_map, a_lcma, v_lcma = model.lcma(a_seq.unsqueeze(0), v_clip.unsqueeze(0))
            except Exception:
                continue
            # 与 LTCA.forward 完全一致：sync_t = mean_t cos(a_seq,v_seq)
            sync_t = torch.nn.functional.cosine_similarity(a_lcma, v_lcma, dim=-1).mean(dim=1).item()
            syncs.append(sync_t); labels.append(int(r["label"]))
            if (i + 1) % 200 == 0:
                print(f"   ... {i+1}/{len(rows)}")
    return np.array(syncs), np.array(labels, dtype=np.int64)


def run_one(cfg, ckpt, cache, max_per_class, dev):
    print(f"\n=== 探针集: {cache}  max_per_class={max_per_class} ===")
    index = load_index(cache)
    real_n = sum(1 for ix in index if ix["label"] == 0)
    fake_n = sum(1 for ix in index if ix["label"] == 1)
    print(f"[sync_t] real(总采样池)={real_n}  fake={fake_n}")
    if min(real_n, fake_n) < 20:
        print("   !! 样本过少，跳过"); return
    s, y = batch_sync_t(model, cfg, index, dev, max_per_class)
    if len(s) < 40:
        print(f"   有效样本 {len(s)} 过少，跳过"); return
    m_real, m_fake = s[y == 0], s[y == 1]
    auc = roc_auc_y_score(y, s); st = max(auc, 1 - auc); d = cohens_d(m_real, m_fake)
    mark = "★强可分" if st >= 0.8 else ("▲可分" if st >= 0.65 else ("●中" if st >= 0.6 else "≈随机"))
    print(f"[sync_t] auc(实0)={auc:.3f}  strength={st:.3f}  cohen_d={d:+.3f}  {mark}")
    print(f"          real sync_t mean={m_real.mean():+.4f}±{m_real.std():.4f} | fake={m_fake.mean():+.4f}±{m_fake.std():.4f}  (n_real={len(m_real)}, n_fake={len(m_fake)})")


def main():
    ap = argparse.ArgumentParser(description="方向B LTCA sync_t 跨生成器可分性探针")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", action="append", required=True, help="可多次：每个评测/训练缓存目录")
    ap.add_argument("--max-per-class", type=int, default=600)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    global model
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = get_config(**{k: v for k, v in (ck.get("cfg") or {}).items()
                        if k in get_config().__dict__})
    if not getattr(cfg, "fta_enabled", False):
        print("!! ckpt cfg.fta_enabled=False —— 这不是方向B(LTCA)的权重，探针无意义，退出")
        sys.exit(1)
    model = SyncX(cfg).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[sync_t] 已加载方向B ckpt；fta_enabled=True，LTCA 权重在位："
          f"{sum(p.numel() for p in model.fta.parameters() if model.fta is not None)} params")
    for c in args.cache:
        run_one(cfg, args.ckpt, c, args.max_per_class, dev)
    print("\n判读：HiFi sync_t strength≥0.65→sync 无辜(负迁移在别处)；<0.6→坐实 HiFi 上 sync 无信号。")


if __name__ == "__main__":
    main()