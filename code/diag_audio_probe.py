#!/usr/bin/env python
"""
diag_audio_probe.py — 听觉监督断裂的线性探针（秒级，不重训）
================================================================
目的：判定"听觉监督断裂"断在哪一环。
方法：用已训模型前向，取 attention 之前的纯音频投影嵌入 `audio_emb` 与
纯视觉投影嵌入 `visual_emb`（两者正是监督损失使用的嵌入，见
models.SyncX.forward → a_raw.mean/v_raw.mean）。对每个嵌入做
"类质心线性探针"，算 AUROC（权重无关，纯 torch，不依赖 sklearn）。

判读：
  - audio_emb 探针 AUC 高（>0.8）→ 音频特征已携带可分的 real/fake 信息，
    断点在更下游的监督/损失/融合，而非特征。
  - audio_emb 探针 AUC≈0.5 → 音频投影学出的表征本身不含可分信息，
    断点在特征/投影环节。
  - visual_emb 作对照：预期高分，用于验证探针方法本身有效。

用法（云端，在 code_cloud 下）：
  python diag_audio_probe.py --ckpt /root/autodl-tmp/SyncX/persist/ckpt/syncx_best_seed0.pt \
      --cache /root/autodl-tmp/SyncX/Datasets/DeepSpeak_v1_1/clips/prep [--limit 3000]
"""
import os, sys, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from torch.utils.data import DataLoader


def auroc(y, s):
    """两分类 AUROC，0.5=随机，1=正类完全排序在前。纯 torch，无 sklearn。"""
    y = torch.as_tensor(y, dtype=torch.long)
    s = torch.as_tensor(s, dtype=torch.float32)
    n0 = int((y == 0).sum()); n1 = int((y == 1).sum())
    if n0 == 0 or n1 == 0:
        return None, n0, n1
    order = torch.argsort(s, descending=False)
    ranks = torch.arange(1, len(y) + 1, dtype=torch.float32)
    s1 = ranks[order[y[order] == 1]].sum().item()
    auc = (s1 - n1 * (n1 + 1) / 2.0) / (n0 * n1)
    return auc, n0, n1


def centroid_probe(emb, y):
    """类质心线性探针：score = emb · (ctr1 − ctr0)。返回 (AUC, n0, n1)。"""
    E = emb.detach().float()
    y = torch.as_tensor(y, dtype=torch.long)
    ctr0 = E[y == 0].mean(dim=0)
    ctr1 = E[y == 1].mean(dim=0)
    score = (E @ (ctr1 - ctr0)).tolist()
    return auroc(y.tolist(), score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--limit", type=int, default=0, help=">0 时仅取前 N 样本（加速）")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    from config import get_config
    from models import SyncX
    from dataset import FeatureCacheDataset

    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = get_config(**{k: v for k, v in (ck.get("cfg") or {}).items()
                        if k in get_config().__dict__})
    model = SyncX(cfg).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    ds = FeatureCacheDataset(args.cache, T=cfg.T, real_only=False)
    if args.limit > 0:
        rng = __import__("random").Random(0)
        total = min(args.limit, len(ds))
        idx = sorted(rng.sample(range(len(ds)), total))
        ds.files = [ds.files[i] for i in idx]
        ds.source_ids = [ds.source_ids[i] for i in idx]
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)
    print(f"[probe] cache={args.cache}  n={len(ds)}  device={dev}")

    audio_emb, visual_emb, labels = [], [], []
    with torch.no_grad():
        for batch in dl:
            for k in ("audio_seq", "v_clip", "v_lip"):
                batch[k] = batch[k].to(dev, non_blocking=True)
            out = model(batch)
            audio_emb.append(out["audio_emb"].cpu())
            visual_emb.append(out["visual_emb"].cpu())
            labels.append(batch["label"].cpu())
    A = torch.cat(audio_emb); V = torch.cat(visual_emb); y = torch.cat(labels).tolist()
    if len(set(y)) < 2:
        print("!! 该缓存只有单一标签，无法判可分性，请换数据集。"); return

    for name, E in (("audio_emb(纯音频投影)", A), ("visual_emb(纯视觉投影)", V)):
        auc, n0, n1 = centroid_probe(E, y)
        # 特征退化检查：两分类质心距离 vs 全局 std
        yt = torch.as_tensor(y, dtype=torch.long)
        d0 = E[yt == 0].mean(0); d1 = E[yt == 1].mean(0)
        sep = (d1 - d0).norm().item()
        stdg = E.std().item()
        txt = (f"{name:<26} 探针AUC={auc:<7.3f}  n0={n0} n1={n1}  "
               f"|Δ质心|={sep:.3f} 全局std={stdg:.3f}")
        print(txt)
        if auc is not None:
            if auc >= 0.8:
                print("   → 该嵌入线性可分隔 → 特征已含可分信息，断点在监督/下游。")
            elif auc <= 0.6:
                print("   → 该嵌入≈不可分 → 特征本身未携带可分信息（听觉断裂指向特征/投影环节）。")
            else:
                print("   → 中弱可分，需结合其它信号判定。")
    print("\n判读：若 audio_emb 探针 AUC≈0.5 而 visual_emb 高 → 听觉断裂在特征/投影层；"
          "\n若 audio_emb 高但模型 audio-only 评测仍随机 → 断点在监督/下游 head。")


if __name__ == "__main__":
    main()