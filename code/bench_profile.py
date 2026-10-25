"""
bench_profile.py — F13 计算画像（零重训重画）
================================================================
对存档 syncx_best.pt 计时剖析，落盘 profile.jsonl：
  - peak_vram_gb        : 单步前向+反向峰值显存
  - stage2_epoch_min    : 由 N 步计时外推的单 epoch 训练时长（分钟）
  - inference_latency_ms: 固定 batch 前向平均延迟
  - train_budget_gpu_hours : 全程训练 GPU 预算（epoch_min * epochs * n_gpus / 60）
  - offline_extraction_time_s : 留空（离线特征抽取另测）

重画 F13 对存档权重重跑剖析，无需重训。CPU 环境下 GPU 字段记 None。

运行：
    python bench_profile.py --ckpt ./ckpt/syncx_best.pt \
        --steps_per_epoch 1000 --stage2_epochs 80 --device cuda
"""
import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from config import get_config
from models import SyncX
from logging_utils import append_jsonl


def _synth_batch(cfg, B, dev):
    """构造与 model.forward 输入形状一致的随机 batch（仅用于计时，不代表真实数据）。"""
    T_a, T_v = cfg.T, cfg.T
    return {
        "audio_seq": torch.randn(B, T_a, cfg.audio_dim, device=dev),
        "v_clip": torch.randn(B, T_v, cfg.clip_dim, device=dev),
        "v_lip": torch.randn(B, T_v, cfg.lip_lm_dim, device=dev),
    }, torch.randint(0, 2, (B,), device=dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps_per_epoch", type=int, default=1000, help="用于外推 epoch 时长的步数")
    ap.add_argument("--stage2_epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n_gpus", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--log_dir", default="logs")
    ap.add_argument("--exp_id", required=True)
    args = ap.parse_args()

    dev = args.device if (torch.cuda.is_available() and args.device == "cuda") else "cpu"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = get_config(**{k: v for k, v in (ck.get("cfg") or {}).items() if k in get_config().__dict__})
    model = SyncX(cfg).to(dev); model.eval()
    if args.stage2_epochs is None:
        args.stage2_epochs = getattr(cfg, "stage2_epochs", 80)

    rec = {"exp_id": args.exp_id, "device": dev,
           "peak_vram_gb": None, "stage2_epoch_min": None,
           "inference_latency_ms": None, "train_budget_gpu_hours": None,
           "offline_extraction_time_s": None}

    B = args.batch
    # 1) 峰值 VRAM + 训练步计时（N 步 forward+backward）
    N = 20
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    batch, labels = _synth_batch(cfg, B, dev)
    if dev == "cuda":
        batch = {k: v.to(dev) for k, v in batch.items()}
    t0 = time.perf_counter()
    for _ in range(N):
        model.train()
        out = model(batch, labels, gate_enabled=True)
        loss = out["loss_cls"]
        loss.backward()
        model.zero_grad()
    dt = (time.perf_counter() - t0) / N
    if dev == "cuda":
        rec["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
    step_min = dt / 60.0
    rec["stage2_epoch_min"] = step_min * args.steps_per_epoch
    rec["train_budget_gpu_hours"] = rec["stage2_epoch_min"] * args.stage2_epochs * args.n_gpus / 60.0

    # 2) 推理延迟（M 次前向取均值）
    model.eval()
    M = 30
    with torch.no_grad():
        for _ in range(5):   # warmup
            model(_synth_batch(cfg, B, dev)[0])
        t0 = time.perf_counter()
        for _ in range(M):
            with torch.no_grad():
                model(_synth_batch(cfg, B, dev)[0])
        lat = (time.perf_counter() - t0) / M * 1000.0
    rec["inference_latency_ms"] = lat

    out = os.path.join(args.log_dir, args.exp_id, "profile.jsonl")
    append_jsonl(out, rec)
    print(f"[profile] peak_vram_gb={rec['peak_vram_gb']}  stage2_epoch_min={rec['stage2_epoch_min']:.2f}  "
          f"latency_ms={rec['inference_latency_ms']:.2f}  budget_h={rec['train_budget_gpu_hours']:.2f}")


if __name__ == "__main__":
    main()
