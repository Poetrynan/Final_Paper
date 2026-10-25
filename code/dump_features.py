"""
dump_features.py — F12 双线索特征抽取（零重训重画）
================================================================
对一批片段（--sample_n）逐条前向，落盘 features/clip_features.jsonl：
  - corr_score : -L_corr(audio_emb, visual_emb, label)，真实高 / 伪造低
  - biomech_jerk : 唇部运动学最高阶（jerk）范数均值，暴露生成器非生理抖动
重画 F12 仅重抽特征（廉价，不训练/不全量评测）。

运行：
    python dump_features.py --ckpt ./ckpt/syncx_best.pt \
        --cache D:/Datasets/DeepSpeak_v1_1/clips/prep/test --dataset DeepSpeak \
        --sample_n 400 --log_dir logs --exp_id syncx_bs256_lam0.3_lacaboth_seed0
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from config import get_config
from models import SyncX, compute_kinematics
from losses import CorrConsistencyLoss
from dataset import FeatureCacheDataset
from logging_utils import append_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True, help="片段特征缓存目录（train/test 皆可）")
    ap.add_argument("--dataset", default="DeepSpeak")
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
    corr_loss = CorrConsistencyLoss().to(dev)

    ds = FeatureCacheDataset(args.cache, T=cfg.T, real_only=False)
    n = min(args.sample_n, len(ds))
    idxs = list(range(n))   # 顺序前 N 条；如需随机可在此 shuffle
    out = os.path.join(args.log_dir, args.exp_id, "features", "clip_features.jsonl")
    kin_channels = model.laca.kin_channels
    for i, idx in enumerate(idxs):
        item = ds[idx]
        label = int(item["label"])
        batch = {k: v.unsqueeze(0).to(dev) for k, v in item.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            out_d = model(batch)
            audio_emb = out_d["audio_emb"].float(); visual_emb = out_d["visual_emb"].float()
            loss = corr_loss(audio_emb, visual_emb, torch.tensor([label], device=dev))
        corr_score = -float(loss)
        # biomech_jerk：唇部运动学最高阶（jerk）范数均值
        P = item["v_lip"].unsqueeze(0).to(dev)        # (1, T, 192)：kinematics 要求 3 维入参且须与模型同设备(cuda)
        kin = model.laca.kinematics(P) if model.laca.kinematics is not None \
            else compute_kinematics(P, cfg.kin_order)
        jerk = kin[..., kin_channels * cfg.kin_order:].norm(dim=-1).mean().item()
        rec = {"clip_id": i, "dataset": args.dataset, "label": label,
               "corr_score": corr_score, "biomech_jerk": jerk}
        append_jsonl(out, rec)
        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"[dump_features] {i+1}/{n}  corr={corr_score:.3f} jerk={jerk:.3f}")


if __name__ == "__main__":
    main()
