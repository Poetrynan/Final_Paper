"""
dump_embeddings.py — F15 表示可分性嵌入抽取（零重训重画）
================================================================
对一批片段（--sample_n，复用 F12 的精选集）抽 pooled 音视频嵌入向量，落盘
features/clip_embeddings.jsonl，供出图时离线做 t-SNE/UMAP（对标 HAVIC 附录
D.3 / MSCT Fig5 / FoVB Fig5 的表示可分性图）。

运行：
    python dump_embeddings.py --ckpt ./ckpt/syncx_best.pt \
        --cache D:/Datasets/DeepSpeak_v1_1/clips/prep/test --dataset DeepSpeak \
        --sample_n 400 --log_dir logs --exp_id syncx_bs256_lam0.3_lacaboth_seed0
重画 F15 仅重抽嵌入（廉价，不训练/不全量评测）；t-SNE/UMAP 在 plot_figures 内离线计算。
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from config import get_config
from models import SyncX
from dataset import FeatureCacheDataset
from logging_utils import append_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--dataset", default="DeepSpeak")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--sample_n", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log_dir", default="logs")
    ap.add_argument("--exp_id", required=True)
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = get_config(**{k: v for k, v in (ck.get("cfg") or {}).items() if k in get_config().__dict__})
    model = SyncX(cfg).to(dev)
    model.load_state_dict(ck["model"]); model.eval()

    ds = FeatureCacheDataset(args.cache, T=cfg.T, real_only=False)
    n = min(args.sample_n, len(ds))
    out = os.path.join(args.log_dir, args.exp_id, "features", "clip_embeddings.jsonl")
    for i in range(n):
        item = ds[i]
        label = int(item["label"])
        batch = {k: v.unsqueeze(0).to(dev) for k, v in item.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            emb = model.pooled_embedding(batch)[0].detach().cpu()   # (2d,)
        rec = {"clip_id": i, "dataset": args.dataset, "label": label,
               "embedding": emb.tolist(), "variant": args.variant}
        append_jsonl(out, rec)
        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"[dump_embeddings] {i+1}/{n}  dim={emb.shape[-1]}  label={label}")


if __name__ == "__main__":
    main()
