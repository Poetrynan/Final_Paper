"""
dump_interpret.py — F11 精选案例前向（零重训重画）
================================================================
载入 syncx_best.pt，对精选 1 real + 1 fake 片段跑 model.forward_interpret，
落盘 interpret/<case_id>.jsonl（注意力热图 + 逐帧音视距离异常分）。
重画 F11 只需对这两段重跑（极廉价，无训练/全量评测）。

运行：
    python dump_interpret.py --ckpt ./ckpt/syncx_best.pt \
        --clip "real01|D:/Datasets/THB/real/clip001.pt|0" \
        --clip "fake01|D:/Datasets/THB/fake/clip077.pt|1|12-28" \
        --log_dir logs --exp_id syncx_bs256_lam0.3_lacaboth_seed0
每个 --clip 格式： case_id|clip_path|label[|forgery_start-forgery_end]
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from config import get_config
from models import SyncX
from dataset import FeatureCacheDataset
from logging_utils import append_jsonl


def parse_clip(spec):
    parts = spec.split("|")
    case_id, path, label = parts[0], parts[1], int(parts[2])
    forgery = None
    if len(parts) > 3 and parts[3] and parts[3] != "none":
        a, b = parts[3].split("-")
        forgery = [int(a), int(b)]
    return case_id, path, label, forgery


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--clip", action="append", default=[], help="case_id|path|label[|start-end]，可重复")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log_dir", default="logs")
    ap.add_argument("--exp_id", required=True)
    args = ap.parse_args()

    if not args.clip:
        print("[dump_interpret] 未提供 --clip，退出。"); return

    dev = args.device if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = get_config(**{k: v for k, v in (ck.get("cfg") or {}).items() if k in get_config().__dict__})
    model = SyncX(cfg).to(dev)
    model.load_state_dict(ck["model"]); model.eval()

    out_dir = os.path.join(args.log_dir, args.exp_id, "interpret")
    for spec in args.clip:
        case_id, path, label, forgery = parse_clip(spec)
        ds = FeatureCacheDataset("", T=cfg.T, real_only=False, file_list=[path])
        item = ds[0]
        batch = {k: v.unsqueeze(0).to(dev) for k, v in item.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            out = model.forward_interpret(batch)        # attn:(1,T,T), a_seq/v_seq:(1,T,d)
        attn = out["attn"][0].detach().cpu().numpy()    # (T,T)
        a = out["a_seq"][0]; v = out["v_seq"][0]
        dist = torch.norm(a - v, dim=-1)                # (T,) 逐帧音视距离
        anomaly = (dist - dist.min()) / (dist.max() - dist.min() + 1e-8)
        rec = {"case_id": case_id, "label": label, "source_clip": path,
               "attention": attn.tolist(), "anomaly_scores": anomaly.cpu().numpy().tolist(),
               "forgery_segment": forgery}
        append_jsonl(os.path.join(out_dir, f"{case_id}.jsonl"), rec)
        print(f"[dump_interpret] {case_id}  label={label}  T={attn.shape[0]}  attn_sum={attn.sum():.2f}")


if __name__ == "__main__":
    main()
