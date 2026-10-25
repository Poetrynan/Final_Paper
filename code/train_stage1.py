r"""
train_stage1.py — Stage-1 自监督预训练（真实片段）
================================================================
在真实片段（主源 VoxCeleb2 子集；可选追加 DeepSpeak reals）上同时预训练 LACA 的双信号唇探针：

  1) LCMA  —— 对称 InfoNCE，学音-视语义对齐先验（与旧版一致）。
  2) LACA 支路(ii) 唇部运动学"伪抖动"自监督（BioLip 式 / AVPF 思路）：
             仅用真实片段，对 perioral landmark 轨迹加平滑高频噪声生成
             "伪抖动"负样本，训练 LACA 把真实平滑运动与合成抖动分开。
  3) LACA 支路(i) 共享空间唇–音频对齐：
             通过两个投影头把 lip 表征 K_t 与 audio 表征 a_t 映射到同一空间，
             再约束真实片段上二者余弦相似度趋近 +1，使 Stage-2 的 Sync-Check
             |\hat{K}_t − \hat{a}_t| 在公共流形上有判别基础。

（kin_head 仅 Stage-1 用，Stage-2 丢弃，只保留 LACA 编码器权重。）

运行：
    python train_stage1.py \
        --real_cache ./feature_cache_voxceleb2_real \
        --out_ckpt ./ckpt/stage1.pt \
        --epochs 40 --batch 256 --lr 1e-4

    # B7 身份泄漏控制（论文 Ch4:337）：从 Stage-1 SSL 预训练子集剔除 FakeAVCeleb 身份
    python train_stage1.py \
        --real_cache ./feature_cache_voxceleb2_real \
        --exclude_ids ./configs/exclude_ids.json \
        --out_ckpt ./ckpt/stage1_b7.pt --epochs 40 --batch 256 --lr 1e-4

日志（供 F1 零重训重画）：
    每 --log_every 步写 logs/<exp_id>/stage1_train.jsonl（phase==train，x=global_step）；
    若给 --val_cache，则每 epoch 末追加 phase==val 的 val_contrastive_acc（x=global_step）。
"""
import os, sys, argparse, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from config import get_config
from models import LCMA, LACA, CrossReconNeck
from losses import InfoNCELoss, SharedSpaceLoss, MaskedCrossReconLoss, CrossSemanticReconLoss
from dataset import FeatureCacheDataset, load_identity_set
from utils import build_scheduler, autocast_ctx, capture_rng, restore_rng, save_checkpoint_atomic
from logging_utils import exp_id, append_jsonl


def jitter_landmarks(P, sigma):
    """对 landmark 轨迹加平滑高频噪声，模拟生成器残留的唇部运动抖动（伪伪造负样本）。

    P: (B, T, 192) 归一化 landmark 坐标（64 口周点×3）。返回 P + 时间平滑噪声。
    """
    noise = torch.randn_like(P) * sigma
    B, T, D = noise.shape
    noise = noise.transpose(1, 2)                              # (B, D, T)
    kernel = torch.tensor([1.0, 2.0, 1.0], device=P.device).view(1, 1, 3) / 4.0
    noise = F.conv1d(noise, kernel.expand(D, 1, 3), groups=D, padding=1)
    noise = noise.transpose(1, 2)                             # (B, T, D)
    return P + noise


class KinHead(nn.Module):
    """Stage-1 专用：把 LACA 的 2d 运动学表征映射到 真实/伪抖动 二分类。
    Stage-2 训练时丢弃，仅 LACA 编码器权重被复用。"""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 2))
    def forward(self, x):
        return self.net(x)


def val_retrieval_acc(lcma, laca, vdl, dev, cfg):
    """真实片段上 mean-pooled 音+视嵌入的 leave-one-out 1-NN 检索准确率（F1(a) 右轴）。

    取 LCMA pooled 音/视嵌入拼接、L2 归一化后做余弦近邻检索；命中同 identity 记正确。
    """
    lcma.eval(); laca.eval()
    embs, ids = [], []
    with torch.no_grad():
        for batch in vdl:
            for k in ("audio_seq", "v_clip"):
                batch[k] = batch[k].to(dev, non_blocking=True)
            pooled, attn, a, v = lcma(batch["audio_seq"], batch["v_clip"])
            a_pool = a.mean(1); v_pool = v.mean(1)
            e = F.normalize(torch.cat([a_pool, v_pool], dim=-1), dim=-1)
            embs.append(e.cpu())
            ids.extend(list(batch.get("identity", [""] * e.size(0))))
    if not embs:
        return float("nan")
    E = torch.cat(embs, dim=0).numpy()          # (N, d)
    ids = [str(x) for x in ids]
    N = E.shape[0]
    S = E @ E.T                                  # 已归一化，点积即余弦
    correct = 0
    for i in range(N):
        sims = S[i].copy(); sims[i] = -1e9
        j = int(sims.argmax())
        if ids[i] and ids[j] and ids[i] == ids[j]:
            correct += 1
    return correct / N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_cache", required=True, help="真实片段特征缓存目录（可多个，逗号分隔）")
    ap.add_argument("--out_ckpt", default="ckpt/stage1.pt")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default="cuda")
    # —— 新增：日志 / 复现 ——
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_dir", default="logs")
    ap.add_argument("--exp_id", default=None, help="缺省由 exp_id(cfg, seed) 派生")
    ap.add_argument("--val_cache", default="", help="真实片段验证缓存；给定则每轮末写 val_contrastive_acc")
    ap.add_argument("--exclude_ids", default="", help="B7: 剔除身份列表文件（如 FakeAVCeleb 身份），"
                                                       "从 Stage-1 SSL 真实预训练子集剔除（论文 Ch4:337）")
    ap.add_argument("--disable_xs", action="store_true", help="w/o BioLip 统计分支 x_s（时序 std 汇总）")
    ap.add_argument("--log_every", type=int, default=100, help="每 K 步落盘一次训练损失")
    # —— 云端适配 ——
    ap.add_argument("--num_workers", type=int, default=None,
                    help="DataLoader workers（云端容器 /dev/shm 较小时调小，如 0 或 2）")
    ap.add_argument("--prefetch_factor", type=int, default=2,
                    help="DataLoader prefetch_factor（仅 num_workers>0 生效）")
    ap.add_argument("--resume", default="",
                    help="断点续跑：显式指定 resume 检查点；留空则自动尝试 <out_ckpt>.resume"
                         "（云竞价实例中断后，用相同 --out_ckpt 重跑命令即自动续跑）")
    ap.add_argument("--empty_cache_every", type=int, default=1,
                    help="每 N 个 epoch 末释放一次 GPU 缓存(torch.cuda.empty_cache)，"
                         "防长训练显存碎片导致 OOM；0=不主动释放（默认 1，对训练几乎无影响）")
    args = ap.parse_args()

    cfg = get_config()
    if args.epochs: cfg.stage1_epochs = args.epochs
    if args.batch: cfg.batch_size_stage1 = args.batch
    if args.lr: cfg.lr = args.lr
    cfg.use_xs = not args.disable_xs
    dev = args.device if torch.cuda.is_available() else "cpu"

    # DataLoader workers / prefetch（云端 /dev/shm 较小时用 --num_workers 0/2 调小）
    nw = args.num_workers if args.num_workers is not None else cfg.num_workers
    pf = args.prefetch_factor
    loader_kwargs = dict(num_workers=nw, pin_memory=True)
    if nw > 0:
        loader_kwargs["prefetch_factor"] = pf

    # 复现 + 日志目录
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    eid = args.exp_id or exp_id(cfg, args.seed)
    path = os.path.join(args.log_dir, eid, "stage1_train.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[stage1] exp_id={eid}  seed={args.seed}  log->{path}")

    # B7: 从 Stage-1 SSL 真实预训练子集剔除指定身份（论文 Ch4:337）
    exclude_ids = load_identity_set(args.exclude_ids) if args.exclude_ids else None
    if exclude_ids:
        print(f"[stage1] B7 exclude {len(exclude_ids)} ids from SSL pre-training subset")

    # 多缓存目录合并
    import glob
    files = []
    for d in args.real_cache.split(","):
        d = d.strip()
        if d:
            # [FIX 2026-08-15] 递归 glob：VoxCeleb2 prep 嵌套 <spk>/<stem>/cache.pt
            files += sorted(glob.glob(os.path.join(d, "**", "*.pt"), recursive=True))
    ds = FeatureCacheDataset("", T=cfg.T, real_only=True, file_list=files,
                             exclude_ids=exclude_ids, temporal_jitter=True)
    print(f"[stage1] real clips = {len(ds)}")
    if len(ds) == 0:
        print("[stage1] no real clips, exit."); return
    dl = DataLoader(ds, batch_size=cfg.batch_size_stage1, shuffle=True,
                   drop_last=True, **loader_kwargs)
    steps_per_epoch = len(dl)

    # 验证集（可选）
    vdl = None
    if args.val_cache:
        vfiles = []
        for d in args.val_cache.split(","):
            d = d.strip()
            if d:
                # [FIX 2026-08-15] 递归 glob
                vfiles += sorted(glob.glob(os.path.join(d, "**", "*.pt"), recursive=True))
        vds = FeatureCacheDataset("", T=cfg.T, real_only=True, file_list=vfiles,
                                  exclude_ids=exclude_ids)
        if len(vds) > 0:
            vdl = DataLoader(vds, batch_size=cfg.batch_size_stage1, shuffle=False,
                             drop_last=False, **loader_kwargs)
            print(f"[stage1] val clips = {len(vds)}")
        else:
            print("[stage1] val_cache 无片段，跳过验证。")

    lcma = LCMA(cfg).to(dev)
    laca = LACA(cfg).to(dev)
    kin_head = KinHead(cfg.d_model * 2).to(dev)
    infonce = InfoNCELoss(cfg.infonce_tau_init, cfg.infonce_tau_learnable, cfg.infonce_tau_min).to(dev)
    shared_loss = SharedSpaceLoss().to(dev)
    params = list(lcma.parameters()) + list(laca.parameters()) + \
             list(kin_head.parameters()) + list(infonce.parameters())
    opt = build_optimizer_typed(params, cfg.lr, cfg.weight_decay)
    # [§14-①] HAVIC 式跨模态一致性重建（Stage-1 纯真实，SYNCX_HAVIC_RECON=1 开启）
    hav_rec_on = os.environ.get("SYNCX_HAVIC_RECON") in ("1", "true", "True")
    if hav_rec_on:
        # λ / mask 占比也支持 env 覆写（默退回 config 默认），便于云端一键调窗而不改代码
        try:
            cfg.lambda_hav_mask  = float(os.environ.get("SYNCX_HAV_LAMBDA_MASK", cfg.lambda_hav_mask))
        except ValueError:
            pass
        try:
            cfg.lambda_hav_sem   = float(os.environ.get("SYNCX_HAV_LAMBDA_SEM",  cfg.lambda_hav_sem))
        except ValueError:
            pass
        try:
            cfg.hav_rec_mask_ratio = float(os.environ.get("SYNCX_HAV_MASK_RATIO", cfg.hav_rec_mask_ratio))
        except ValueError:
            pass
    else:
        print("[stage1] §14 HAVIC 跨模态重建 OFF (SYNCX_HAVIC_RECON=1 开启)")
    hav_av_neck = hav_va_neck = None
    hav_sem_av = hav_sem_va = None
    hav_mask_loss = hav_sem_loss = None
    if hav_rec_on:
        d = cfg.d_model
        hav_av_neck = CrossReconNeck(d).to(dev)
        hav_va_neck = CrossReconNeck(d).to(dev)
        hav_sem_av = CrossReconNeck(d).to(dev)   # audio→visual 全局语义重建
        hav_sem_va = CrossReconNeck(d).to(dev)   # visual→audio 全局语义重建
        hav_mask_loss = MaskedCrossReconLoss(cfg.hav_rec_mask_ratio).to(dev)
        hav_sem_loss = CrossSemanticReconLoss().to(dev)
        params += list(hav_av_neck.parameters()) + list(hav_va_neck.parameters()) + \
                  list(hav_sem_av.parameters()) + list(hav_sem_va.parameters())
        print(f"[stage1] §14 HAVIC 跨模态重建 ON: mask_ratio={cfg.hav_rec_mask_ratio} "
              f"λ_mask={cfg.lambda_hav_mask} λ_sem={cfg.lambda_hav_sem}")
    else:
        print("[stage1] §14 HAVIC 跨模态重建 OFF (SYNCX_HAVIC_RECON=1 开启)")
    # 注意：opt 已建，重建头参数已并入 params —— 需在 opt 建好后融合
    opt = build_optimizer_typed(params, cfg.lr, cfg.weight_decay)
    total_steps = cfg.stage1_epochs * len(dl)
    sched = build_scheduler(opt, cfg.warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    os.makedirs(os.path.dirname(args.out_ckpt) or ".", exist_ok=True)

    # ── 断点续跑：云端竞价实例/意外中断后从 resume 检查点恢复 ──────────
    resume_in = args.resume if args.resume else (args.out_ckpt + ".resume")
    start_epoch = 0
    global_step = 0
    if os.path.exists(resume_in):
        ck = torch.load(resume_in, map_location=dev, weights_only=False)
        lcma.load_state_dict(ck["lcma"]); laca.load_state_dict(ck["laca"])
        kin_head.load_state_dict(ck["kin_head"])
        infonce.log_tau.data.copy_(ck["infonce_tau"])
        opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = int(ck["epoch"]) + 1
        global_step = int(ck.get("global_step", 0))
        restore_rng(ck.get("rng"))
        # 重建头：仅当启用了 HAVIC 且 checkpoint 里存了对应键才加载（兼容无重建头的旧 resume）
        if hav_mask_loss is not None and "hav_av_neck" in ck:
            hav_av_neck.load_state_dict(ck["hav_av_neck"]); hav_va_neck.load_state_dict(ck["hav_va_neck"])
            hav_sem_av.load_state_dict(ck["hav_sem_av"]);  hav_sem_va.load_state_dict(ck["hav_sem_va"])
        print(f"[stage1] 断点续跑：从 {resume_in} 恢复，start_epoch={start_epoch} global_step={global_step}")
    else:
        print(f"[stage1] 从头训练（未找到 resume：{resume_in}）")

    # ── 训练加速（零精度影响）：torch.compile 图编译 ──
    #    必须在所有 load_state_dict（含 resume）之后。三级回退，绝不崩：
    #    SYNCX_COMPILE_MODE=max-autotune → default → 不编译（env 可调）
    #    默认开；SYNCX_COMPILE=0 直接关。
    if os.environ.get("SYNCX_COMPILE", "1") == "1" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        mode = os.environ.get("SYNCX_COMPILE_MODE", "max-autotune")
        def _try_compile(mod, name):
            for m in ([mode, "default"] if mode != "default" else ["default"]):
                try:
                    return torch.compile(mod, mode=m), m
                except Exception as e:
                    print(f"[stage1] {name} compile mode={m} 失败: {e}")
            return mod, None
        lcma, m1 = _try_compile(lcma, "LCMA")
        laca, m2 = _try_compile(laca, "LACA")
        kin_head, m3 = _try_compile(kin_head, "KinHead")
        if m1 or m2 or m3:
            print(f"[stage1] torch.compile ENABLED (modes: lcma={m1 or 'off'}, laca={m2 or 'off'}, kin={m3 or 'off'}, cudnn.benchmark=True)")
            # 关键兜底：lazy-compile 的 triton 等错误发生在首次前向，不在 torch.compile()；
            # 官方 suppress_errors 让首个 forward 失败自动回退 eager，绝不中断训练。
            try:
                import importlib
                _dyn = importlib.import_module("torch._dynamo")
                _dyn.config.suppress_errors = True
                print("[stage1] dynamo suppress_errors=True (compile 失败自动回退 eager)")
            except Exception:
                pass
        else:
            print("[stage1] torch.compile 全部模式失败，回退未编译（训练正常继续）")
    else:
        print("[stage1] torch.compile disabled (SYNCX_COMPILE=0 或 CPU)")

    run = {"info": 0.0, "kin": 0.0, "shared": 0.0, "hav_mask": 0.0, "hav_sem": 0.0, "n": 0}
    for ep in range(start_epoch, cfg.stage1_epochs):
        lcma.train(); laca.train(); kin_head.train()
        run = {"info": 0.0, "kin": 0.0, "shared": 0.0, "hav_mask": 0.0, "hav_sem": 0.0, "n": 0}
        for batch in dl:
            for k in ("audio_seq", "v_clip", "v_lip"):
                batch[k] = batch[k].to(dev, non_blocking=True)
            v_lip = batch["v_lip"]
            with autocast_ctx(cfg.amp):
                # (1) LCMA 音-视对齐 InfoNCE
                pooled, attn, a, v = lcma(batch["audio_seq"], batch["v_clip"])
                a_pool = a.mean(1); v_pool = v.mean(1)
                loss_infonce = infonce(a_pool, v_pool)
                # (2) LACA 支路(ii) 唇部运动学"伪抖动"自监督
                # NOTE: 用 laca 返回的 h_bii（纯 biomech 2d），不是 h_laca（融合 2d），
                #       与 Ch3:270 "on the LACA biomechanical sub-branch" 承诺一致。
                _, _, _, h_bii_real, _, _ = laca(v_lip, batch["audio_seq"])
                _, _, _, h_bii_jit, _, _  = laca(jitter_landmarks(v_lip, cfg.kin_jitter_sigma), batch["audio_seq"])
                logits = kin_head(torch.cat([h_bii_real, h_bii_jit], dim=0))     # (2B, 2)
                B = v_lip.size(0)
                target = torch.cat([torch.zeros(B), torch.ones(B)]).long().to(dev)
                loss_kin = F.cross_entropy(logits, target)
                # (3) LACA 支路(i) 共享空间唇–音频对齐（真实片段上 \hat{K}↔\hat{a} 对应）
                #       投影头把 K_t/a_t 显式映射到同一空间，再约束 real 上 cos→+1。
                #       注意：Stage-1 只用真实片段，故 labels=None 默认目标 +1。
                _, _, _, _, K_s, a_s = laca(v_lip, batch["audio_seq"])
                loss_shared = shared_loss(K_s, a_s)
                # [§14-①] HAVIC 式跨模态一致性重建：逐帧掩码重建 + 全局语义交叉重建。
                #   在 LCMA post-attn 内容流 (a, v) 上，用另一模态还原被 mask 的逐帧时间步、
                #   并用跨模态全局语义互建，把 HAVIC 三层中的结构层与宏层补进 Stage-1 纯真实自监督。
                loss_hav_mask = loss_hav_sem = 0.0
                if hav_mask_loss is not None:
                    loss_hav_mask = hav_mask_loss(a, v, hav_av_neck, hav_va_neck)
                    loss_hav_sem  = hav_sem_loss(a_pool, v_pool, hav_sem_av, hav_sem_va)
                loss = loss_infonce + cfg.lambda_kin * loss_kin + cfg.lambda_shared * loss_shared + \
                       cfg.lambda_hav_mask * loss_hav_mask + cfg.lambda_hav_sem * loss_hav_sem
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            scaler.step(opt); scaler.update(); sched.step()
            # —— 日志：每 K 步落盘（x=global_step）——
            global_step += 1
            run["info"] += float(loss_infonce.detach()); run["kin"] += float(loss_kin.detach()); run["shared"] += float(loss_shared.detach())
            run["hav_mask"] += float(loss_hav_mask.detach()); run["hav_sem"] += float(loss_hav_sem.detach()); run["n"] += 1
            if global_step % args.log_every == 0:
                append_jsonl(path, {"global_step": global_step, "epoch": ep + 1, "phase": "train",
                    "train_infoNCE_loss": run["info"] / run["n"],
                    "train_kin_loss": run["kin"] / run["n"],
                    "train_shared_loss": run["shared"] / run["n"],
                    "train_hav_mask_loss": run["hav_mask"] / run["n"],
                    "train_hav_sem_loss": run["hav_sem"] / run["n"],
                    "tau": infonce.tau.item(), "lr": sched.get_last_lr()[0]})
                run = {"info": 0.0, "kin": 0.0, "shared": 0.0, "hav_mask": 0.0, "hav_sem": 0.0, "n": 0}
        # epoch 末可选 val（F1(a) 右轴）
        if vdl is not None:
            append_jsonl(path, {"global_step": global_step, "epoch": ep + 1, "phase": "val",
                                "val_contrastive_acc": val_retrieval_acc(lcma, laca, vdl, dev, cfg)})
        print(f"[stage1] epoch {ep+1}/{cfg.stage1_epochs}  infonce={run['info']/max(run['n'],1):.4f}  "
              f"kin={run['kin']/max(run['n'],1):.4f}  shared={run['shared']/max(run['n'],1):.4f}  "
              f"hav_mask={run['hav_mask']/max(run['n'],1):.4f}  hav_sem={run['hav_sem']/max(run['n'],1):.4f}  tau={infonce.tau.item():.4f}")

        # ── 每 epoch 末落 resume 检查点（云端中断可续跑，最多丢 1 epoch）──
        _resume = {
            "lcma": lcma.state_dict(), "laca": laca.state_dict(),
            "kin_head": kin_head.state_dict(), "infonce_tau": infonce.log_tau.detach().cpu(),
            "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": ep, "global_step": global_step, "rng": capture_rng(),
        }
        if hav_mask_loss is not None:
            _resume["hav_av_neck"] = hav_av_neck.state_dict(); _resume["hav_va_neck"] = hav_va_neck.state_dict()
            _resume["hav_sem_av"] = hav_sem_av.state_dict(); _resume["hav_sem_va"] = hav_sem_va.state_dict()
        save_checkpoint_atomic(args.out_ckpt + ".resume", _resume)

        # ── 及时释放 GPU 缓存：每 N epoch 末把 CUDA caching allocator 的空闲块
        #    归还驱动，防止长训练显存碎片累积导致 OOM。仅在 epoch 边界调用，
        #    不进训练热循环，对训练速度/精度零影响。──
        if torch.cuda.is_available() and args.empty_cache_every > 0 and (ep + 1) % args.empty_cache_every == 0:
            torch.cuda.empty_cache()

    _final = {"lcma": lcma.state_dict(), "laca": laca.state_dict(),
              "kin_head": kin_head.state_dict(), "infonce_tau": infonce.log_tau.detach().cpu(),
              "cfg": cfg.__dict__}
    if hav_mask_loss is not None:
        _final["hav_av_neck"] = hav_av_neck.state_dict(); _final["hav_va_neck"] = hav_va_neck.state_dict()
        _final["hav_sem_av"] = hav_sem_av.state_dict(); _final["hav_sem_va"] = hav_sem_va.state_dict()
        _final["hav_rec_mask_ratio"] = cfg.hav_rec_mask_ratio
        _final["lambda_hav_mask"] = cfg.lambda_hav_mask; _final["lambda_hav_sem"] = cfg.lambda_hav_sem
    save_checkpoint_atomic(args.out_ckpt, _final)
    print(f"[stage1] saved -> {args.out_ckpt}")


def build_optimizer_typed(params, lr, wd):
    from torch.optim import AdamW
    return AdamW(params, lr=lr, weight_decay=wd)


if __name__ == "__main__":
    main()
