"""
train_stage2.py — Stage-2 监督训练（AAM + corr consistency + GenD hyperspherical）
================================================================
在 DeepSpeak v1.1 上训练完整 Sync-X（LCMA+LACA+DWF+AAM）。
损失：L = L_AAM + λ_corr_lcma·L_corr_lcma + λ_shared·L_shared + λ_align·L_align + λ_uniform·L_uniform
      （AAM 即分类头，不用独立 CE —— 调研结论；L_shared 为 LACA corr 投影头共享空间对齐；
       GenD Rec B 超球面正则默认激活；同源 real/fake 配对 = 数据层采样，非损失项）
真实类用 DeepSpeak 自带 real clip（与 fake 同源采集，避免 real=访谈/fake=webcam 设备捷径）。
可载入 Stage-1 预训练的 LCMA/LACA 权重。

运行：
    python train_stage2.py \
        --train_cache D:/Datasets/DeepSpeak_v1_1/clips/prep/train \
        --val_cache   D:/Datasets/DeepSpeak_v1_1/clips/prep/test \
        --stage1_ckpt ./ckpt/stage1_lcma.pt \
        --out_ckpt    ./ckpt/syncx_best.pt \
        --epochs 80 --batch 256 --lr 1e-4

日志（供 F1 零重训重画）：
    每 --log_every 步写 logs/<exp_id>/stage2_train.jsonl（phase==train，x=global_step）；
    每 epoch 末若给 --val_cache，追加 phase==val 的 val_{auc,ap,eer,acc,f1} 及 is_best 标志。
"""
import os, sys, argparse, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader
from config import get_config
from models import SyncX
from losses import CorrConsistencyLoss, AlignmentLoss, UniformityLoss, SharedSpaceLoss, FrameAlignmentPriorLoss
from dataset import FeatureCacheDataset, collect_source_ids, PairedBatchSampler, load_identity_set
from utils import build_scheduler, autocast_ctx, all_metrics, capture_rng, restore_rng, save_checkpoint_atomic
from torch.optim import AdamW
from logging_utils import exp_id, append_jsonl, append_jsonl_dedup


def evaluate(model, dl, dev, cfg):
    model.eval()
    ys, ss = [], []
    with torch.no_grad():
        for batch in dl:
            for k in ("audio_seq", "v_clip", "v_lip"):
                batch[k] = batch[k].to(dev, non_blocking=True)
            with autocast_ctx(cfg.amp):
                out = model(batch)
            prob = torch.softmax(out["logits"].float(), dim=1)[:, 1]
            ys.append(batch["label"]); ss.append(prob.cpu())
    if not ys: return {}
    y = torch.cat([t.view(-1) for t in ys]); s = torch.cat(ss)
    return all_metrics(y, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_cache", required=True)
    ap.add_argument("--val_cache", default="")
    ap.add_argument("--stage1_ckpt", default="")
    ap.add_argument("--out_ckpt", default="ckpt/syncx_best.pt")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--exclude_ids", default="", help="B7: 剔除身份列表文件")
    ap.add_argument("--device", default="cuda")
    # —— 新增：日志 / 复现 ——
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_dir", default="logs")
    ap.add_argument("--exp_id", default=None, help="缺省由 exp_id(cfg, seed, tag) 派生")
    ap.add_argument("--tag", default="", help="附加到 exp_id 的标签（如 scratch / lam1.0）")
    ap.add_argument("--log_every", type=int, default=100, help="每 K 步落盘一次训练损失")
    # —— 组件消融开关（Gap-2 修复）——
    ap.add_argument("--disable_lcma", action="store_true", help="w/o LCMA：仅 LACA 分支（fused=h_laca）")
    ap.add_argument("--disable_dwf", action="store_true", help="w/o DWF：固定 0.5/0.5 平均融合，关闭可学习门控")
    ap.add_argument("--disable_xs", action="store_true", help="w/o BioLip 统计分支 x_s（时序 std 汇总）")
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
    # 超参搜索透传：hyperparameter_search.py 通过环境变量注入 λ 覆盖（仅搜索时存在）
    # 论文契约：λ_corr=0.3, λ_shared=0.1, λ_align=0.1, λ_uniform=0.5（Ch3:97）；λ_align/λ_uniform
    # 为论文留白项（Ch4:171-172 "selected on val split"），经 SYNCX_LAMBDA_ALIGN/UNIFORM 可搜。
    for _env, _attr in (("SYNCX_LAMBDA_CORR_LCMA", "lambda_corr_lcma"),
                        ("SYNCX_LAMBDA_SHARED", "lambda_shared"),
                        ("SYNCX_LAMBDA_ALIGN", "lambda_align"),
                        ("SYNCX_LAMBDA_UNIFORM", "lambda_uniform"),
                        ("SYNCX_LAMBDA_AUDIO", "lambda_audio")):
        if os.environ.get(_env) is not None:
            try:
                setattr(cfg, _attr, float(os.environ[_env]))
            except ValueError:
                pass
    # [v2-1d] L_corr/L_shared 是否只约束 real 段（解监督冲突）：布尔，可用 SYNCX_CORR_REAL_ONLY=1/0 覆写
    if os.environ.get("SYNCX_CORR_REAL_ONLY") is not None:
        try:
            cfg.corr_real_only = bool(int(os.environ["SYNCX_CORR_REAL_ONLY"]))
        except ValueError:
            pass
    # [GenD 消融] Rec C 同源配对数据层采样开关：SYNCX_NO_SAME_SOURCE=1 关闭（回退标准 i.i.d.）
    #   principal config 默认启用（config.same_source_sampling）。对应 Ch5 component ablation
    #   "Without same-source pair sampling"。0.1/0.5 的 Rec B 权重经 SYNCX_LAMBDA_ALIGN/UNIFORM 覆写。
    if os.environ.get("SYNCX_NO_SAME_SOURCE") is not None:
        try:
            cfg.same_source_sampling = not bool(int(os.environ["SYNCX_NO_SAME_SOURCE"]))
        except ValueError:
            pass
    # [module2-监督侧] env 覆写：冻结编码器 + paired modality dropout（默认关闭，仅重训时打开）
    if os.environ.get("SYNCX_FREEZE_ENCODER") is not None:
        cfg.freeze_encoder_stage2 = os.environ["SYNCX_FREEZE_ENCODER"] in ("1", "true", "True")
    if os.environ.get("SYNCX_FREEZE_VISUAL") is not None:
        cfg.freeze_visual_only = os.environ["SYNCX_FREEZE_VISUAL"] in ("1", "true", "True")
    if os.environ.get("SYNCX_MODAL_DROPOUT") is not None:
        try:
            cfg.modal_dropout_prob = float(os.environ["SYNCX_MODAL_DROPOUT"])
        except ValueError:
            pass
    # [方向B] LTCA 第三分支主开关：SYNCX_FTA=1 接入（默认关，防污染现有行为）
    if os.environ.get("SYNCX_FTA") is not None:
        cfg.fta_enabled = os.environ["SYNCX_FTA"] in ("1", "true", "True")
    # [§13 试点] 端到端对齐先验主开关：SYNCX_ALIGN_LOSS=1 开启，SYNCX_ALIGN_WEIGHT 覆写权重。
    #   只用于 B 组试点（与 freeze encoder 互斥：要特征随一致性目标学须放开编码器）。
    if os.environ.get("SYNCX_ALIGN_LOSS") is not None:
        cfg.align_prior_loss = os.environ["SYNCX_ALIGN_LOSS"] in ("1", "true", "True")
    if os.environ.get("SYNCX_ALIGN_WEIGHT") is not None:
        try:
            cfg.lambda_align_prior = float(os.environ["SYNCX_ALIGN_WEIGHT"])
        except ValueError:
            pass
    if args.epochs: cfg.stage2_epochs = args.epochs
    if args.batch: cfg.batch_size_stage2 = args.batch
    if args.lr: cfg.lr = args.lr
    cfg.disable_lcma = args.disable_lcma
    cfg.disable_dwf = args.disable_dwf
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
    eid = args.exp_id or exp_id(cfg, args.seed, args.tag)
    path = os.path.join(args.log_dir, eid, "stage2_train.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[stage2] exp_id={eid}  seed={args.seed}  log->{path}"
          + (f"  [ablation: {'w/o LCMA' if cfg.disable_lcma else ''}{' + ' if cfg.disable_lcma and cfg.disable_dwf else ''}{'w/o DWF' if cfg.disable_dwf else ''}]" if (cfg.disable_lcma or cfg.disable_dwf) else ""))

    exclude_ids = load_identity_set(args.exclude_ids) if args.exclude_ids else None
    ds = FeatureCacheDataset(args.train_cache, T=cfg.T, real_only=False,
                             exclude_ids=exclude_ids, temporal_jitter=True)
    print(f"[stage2] train clips = {len(ds)}" + (f" (B7 exclude {len(exclude_ids)} ids)" if exclude_ids else ""))
    # Rec C 同源配对采样器（数据层，非损失）：principal config 默认启用，数据含
    # source_id 时把同源 real/fake 片段放入同一 mini-batch；关闭则标准 i.i.d.
    # （对应 Ch5 component ablation "Without same-source pair sampling"）。
    pair_sampler = None
    if cfg.same_source_sampling:
        sids = collect_source_ids(ds)
        if any(sids):
            pair_sampler = PairedBatchSampler(sids, cfg.batch_size_stage2, shuffle=True, drop_last=True)
            print(f"[stage2] same-source paired sampler ON ({len(pair_sampler)} batches)")
        else:
            print("[stage2] same_source_sampling=True but no source_id found; falling back to i.i.d. sampling")
    if pair_sampler is not None:
        dl = DataLoader(ds, batch_sampler=pair_sampler, **loader_kwargs)
    else:
        dl = DataLoader(ds, batch_size=cfg.batch_size_stage2, shuffle=True,
                        drop_last=True, **loader_kwargs)
    vds = vdl = None
    if args.val_cache:
        vds = FeatureCacheDataset(args.val_cache, T=cfg.T, real_only=False,
                                  exclude_ids=exclude_ids)
        vdl = DataLoader(vds, batch_size=cfg.batch_size_stage2, shuffle=False,
                         **loader_kwargs)

    model = SyncX(cfg).to(dev)
    if args.stage1_ckpt and os.path.exists(args.stage1_ckpt):
        ck = torch.load(args.stage1_ckpt, map_location=dev, weights_only=False)
        msg1 = model.lcma.load_state_dict(ck["lcma"], strict=False)
        msg2 = model.laca.load_state_dict(ck["laca"], strict=False)
        print(f"[stage2] loaded stage1 LCMA: {msg1}")
        print(f"[stage2] loaded stage1 LACA (kinematic prior): {msg2}")
    corr_loss = CorrConsistencyLoss(real_only=cfg.corr_real_only).to(dev)   # [v2-1d]
    shared_loss = SharedSpaceLoss().to(dev)
    align_loss = AlignmentLoss().to(dev)
    uniform_loss = UniformityLoss().to(dev)
    align_prior_loss = FrameAlignmentPriorLoss().to(dev)   # [§13 试点] 端到端逐帧对齐先验
    print(f"[stage2] loss wiring: λ_audio={cfg.lambda_audio:.2f} (audio AAM head ON)  corr_real_only={cfg.corr_real_only}")
    print(f"[stage2] end-to-end align prior: {'ON' if cfg.align_prior_loss else 'OFF'}  λ_align_prior={cfg.lambda_align_prior:.2f}  (SYNCX_ALIGN_LOSS/WEIGHT)")
    # [module2-监督侧] SYNCX_FREEZE_ENCODER=1：冻结 LCMA/LACA 骨干，只训 DWF/AAM/audio_aam 头。
    #   保住 Stage-1 在 VoxCeleb2 学到的 AV 表征，不被 DeepSpeak 全量微调盖成域专属（对齐 39 ICS/12 AVFF）。
    #   冻结参数仍留在 state_dict，resume/续跑兼容；可训练子集自动变为 DWF/AAM/audio_aam。
    if cfg.freeze_visual_only:
        # 仅冻视觉支路：lcma.visual_proj + laca 唇部运动学(kin_mlp/gru/biomech/xs/lip_shared_proj)。
        # 放开 audio_proj/双 shared_proj/corr/fuse/blocks(DWF/AAM/audio_aam 本就放开)，
        # 逼训练真正用上 audio_emb 的可分信号(证据链§8B)。与 freeze_encoder_stage2 互斥(单变量)。
        _vis_pre = ("lcma.visual_proj.", "laca.kin_mlp.", "laca.kinematics.",
                    "laca.gru.", "laca.gru_proj.", "laca.out_norm.",
                    "laca.biomech_proj.", "laca.xs_mlp.", "laca.lip_shared_proj.")
        for name, p in model.named_parameters():
            if name.startswith(_vis_pre):
                p.requires_grad = False
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        print(f"[stage2] FREEZE VISUAL-ONLY：冻结视觉支路，放开 audio/fusion；可训练 {n_train} / {n_all} 参数")
    elif cfg.freeze_encoder_stage2:
        for name, p in model.named_parameters():
            if name.startswith("lcma.") or name.startswith("laca."):
                p.requires_grad = False
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        print(f"[stage2] FREEZE encoder ON：可训练 {n_train} / {n_all} 参数（仅 DWF/AAM/audio_aam 头）")
    else:
        print("[stage2] 无冻结：全量微调（逐原行为）")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.stage2_epochs * len(dl)
    sched = build_scheduler(opt, cfg.warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)
    # [方向B] 打印 LTCA 接入状态 + 三路 DWF 门偏置（供排障：偏置应 favour LCMA）
    n_fta = sum(p.numel() for p in model.fta.parameters()) if model.fta is not None else 0
    print(f"[stage2] FTA/LTCA={'ON' if cfg.fta_enabled else 'OFF'}  fta_params={n_fta}  "
          f"dwf_n_branches={model.dwf.n_branches}  dwf_gate_bias={model.dwf.gate.bias.data.tolist() if model.dwf.gate.bias is not None else None}")

    os.makedirs(os.path.dirname(args.out_ckpt) or ".", exist_ok=True)

    # ── 断点续跑：云端竞价实例/意外中断后从 resume 检查点恢复 ──────────
    resume_in = args.resume if args.resume else (args.out_ckpt + ".resume")
    start_epoch = 0
    global_step = 0
    best_auc = -1.0
    patience = 0
    if os.path.exists(resume_in):
        ck = torch.load(resume_in, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch = int(ck["epoch"]) + 1
        global_step = int(ck.get("global_step", 0))
        best_auc = float(ck.get("best_auc", -1.0))
        patience = int(ck.get("patience", 0))
        restore_rng(ck.get("rng"))
        print(f"[stage2] 断点续跑：从 {resume_in} 恢复，start_epoch={start_epoch} "
              f"global_step={global_step} best_auc={best_auc:.4f} patience={patience}")
    else:
        print(f"[stage2] 从头训练（未找到 resume：{resume_in}）")

    # ── 训练加速（零精度影响）：torch.compile 图编译 ──
    #    必须在所有 load_state_dict（含 resume）之后。三级回退，绝不崩：
    #    SYNCX_COMPILE_MODE=max-autotune → default → 不编译（env 可调）
    #    默认开；SYNCX_COMPILE=0 直接关。
    if os.environ.get("SYNCX_COMPILE", "1") == "1" and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        mode = os.environ.get("SYNCX_COMPILE_MODE", "max-autotune")
        compiled = False
        for m in ([mode, "default"] if mode != "default" else ["default"]):
            try:
                model = torch.compile(model, mode=m)
                print(f"[stage2] torch.compile ENABLED (mode={m}, cudnn.benchmark=True)")
                compiled = True
                break
            except Exception as e:
                print(f"[stage2] torch.compile mode={m} 失败: {e}")
        if not compiled:
            print("[stage2] torch.compile 全部模式失败，回退未编译（训练正常继续）")
        else:
            # 关键兜底：torch.compile 的 triton 等错误发生在"首次前向"（lazy compile），
            # 不在 torch.compile() 调用时。开启官方 suppress_errors，让首个 forward
            # 编译失败时自动回退 eager 模式，绝不中断训练（精度不变，只是少了编译加速）。
            try:
                import importlib
                _dyn = importlib.import_module("torch._dynamo")
                _dyn.config.suppress_errors = True
                print("[stage2] dynamo suppress_errors=True (compile 失败自动回退 eager)")
            except Exception:
                pass
    else:
        print("[stage2] torch.compile disabled (SYNCX_COMPILE=0 或 CPU)")

    run = {"L_total": 0.0, "L_AAM": 0.0, "L_audio": 0.0, "L_corr_weighted": 0.0, "L_shared_weighted": 0.0, "L_align": 0.0, "L_uniform": 0.0, "L_align_prior_weighted": 0.0, "n": 0}
    for ep in range(start_epoch, cfg.stage2_epochs):
        model.train(); run = {k: 0.0 for k in run}; run["n"] = 0
        for batch in dl:
            for k in ("audio_seq", "v_clip", "v_lip"):
                batch[k] = batch[k].to(dev, non_blocking=True)
            labels = batch["label"].to(dev)
            # [module2-监督侧] paired modality dropout（嫌疑③）：达起始 epoch 后按概率整体遮蔽
            #    audio 或 visual 支路，逼融合头在单模态缺失下也能判，削弱对单模态域捷径的依赖
            #    （文献 61：跨 5 种未见伪造唯一稳健 = late-fusion + paired modality dropout）。
            if cfg.modal_dropout_prob > 0 and (ep + 1) >= cfg.modal_dropout_start_epoch:
                if torch.rand(1).item() < cfg.modal_dropout_prob:
                    if torch.rand(1).item() < 0.5:
                        batch["audio_seq"].zero_()
                    else:
                        batch["v_clip"].zero_(); batch["v_lip"].zero_()
            with autocast_ctx(cfg.amp):
                out = model(batch, labels, gate_enabled=(ep >= cfg.dwf_freeze_epochs))
                loss = out["loss_cls"]
                # [v2-1a] 音频独立分类头：显式把 audio_emb 绑到 real/fake，修复"听觉监督断裂"。
                loss = loss + cfg.lambda_audio * out["loss_cls_audio"]
                lcorr = corr_loss(out["audio_emb"].float(), out["visual_emb"].float(), labels)
                loss = loss + cfg.lambda_corr_lcma * lcorr
                # [§13 试点] 端到端逐帧对齐先验：仅在 a_lcma/v_lcma 存在（disable_lcma 时为空）
                # 且开关开启时接入。B 组试点 = 放开编码器 + L_align_prior，让特征随"真实
                # 一致性要高"一起学；此处不发 freeze 日志（freeze 互斥在 config 层由配方保证）。
                lalign_prior = None
                if cfg.align_prior_loss and out.get("a_lcma") is not None and out.get("v_lcma") is not None:
                    lalign_prior = align_prior_loss(out["a_lcma"].float(), out["v_lcma"].float(), labels)
                    loss = loss + cfg.lambda_align_prior * lalign_prior
                # LACA corr 共享空间对齐（投影后的 \hat{K}↔\hat{a}，与 thesis Ch3 对齐）
                # [FIX 2026-10-24] laca_mode="biomech"（仅生物力学支路）时 corr 支路关闭，
                #   LACA 返回 k_shared=a_shared=None（无音频对应可投影），直接 .float() 会
                #   AttributeError: 'NoneType' object has no attribute 'float'，训练第 1 步即崩。
                #   该模式下共享空间对齐项本就无定义，按 0 计入（λ_shared 不生效）；
                #   不新增参数、不改模型结构与 ckpt，仅让 biomech 消融能正常训练。
                if out.get("k_shared") is not None and out.get("a_shared") is not None:
                    lshared = shared_loss(out["k_shared"].float(), out["a_shared"].float(), labels)
                else:
                    lshared = torch.zeros((), device=loss.device, dtype=torch.float32)
                loss = loss + cfg.lambda_shared * lshared
                # Rec B：超球面 alignment + uniformity（Wang & Isola 2020，principal config 激活）
                lalign = align_loss(out["emb"].float(), labels)
                loss = loss + cfg.lambda_align * lalign
                luniform = uniform_loss(out["emb"].float())
                loss = loss + cfg.lambda_uniform * luniform
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            scaler.step(opt); scaler.update(); sched.step()
            # —— 日志：每 K 步落盘（x=global_step）——
            global_step += 1
            _lap = cfg.lambda_align_prior * lalign_prior.item() if lalign_prior is not None else 0.0
            run["L_total"] += float(out["loss_cls"].item()
                                    + cfg.lambda_audio * out["loss_cls_audio"].item()
                                    + cfg.lambda_corr_lcma * lcorr.item()
                                    + cfg.lambda_shared * lshared.item()
                                    + cfg.lambda_align * lalign.item()
                                    + cfg.lambda_uniform * luniform.item()
                                    + _lap)
            run["L_AAM"] += float(out["loss_cls"].item())
            run["L_audio"] += float(out["loss_cls_audio"].item())
            run["L_corr_weighted"] += float(cfg.lambda_corr_lcma * lcorr.item())
            run["L_shared_weighted"] += float(cfg.lambda_shared * lshared.item())
            run["L_align"] += float(cfg.lambda_align * lalign.item())
            run["L_uniform"] += float(cfg.lambda_uniform * luniform.item())
            run["L_align_prior_weighted"] += _lap
            run["n"] += 1
            if global_step % args.log_every == 0:
                rec = {"global_step": global_step, "epoch": ep + 1, "phase": "train",
                       "L_total": run["L_total"] / run["n"], "L_AAM": run["L_AAM"] / run["n"],
                       "L_audio": run["L_audio"] / run["n"],
                       "L_corr_weighted": run["L_corr_weighted"] / run["n"],
                       "L_shared_weighted": run["L_shared_weighted"] / run["n"],
                       "L_align": run["L_align"] / run["n"],
                       "L_uniform": run["L_uniform"] / run["n"],
                       "L_align_prior_weighted": run["L_align_prior_weighted"] / run["n"],
                       "lr": sched.get_last_lr()[0]}
                append_jsonl(path, rec)
                run = {k: 0.0 for k in run}; run["n"] = 0
            # ── 及时释放本步张量（含模型输出嵌入/各损失图），缩短显存占用窗口 ──
            del out, loss, lcorr, lshared, lalign, luniform
            del lalign_prior
        # epoch 末 eval（标 is_best，F1(b) 底轴 + best 竖线）
        log = f"[stage2] epoch {ep+1}/{cfg.stage2_epochs}  cls={run['L_AAM']/max(run['n'],1):.4f}  audio={run['L_audio']/max(run['n'],1):.4f}  corr={run['L_corr_weighted']/max(run['n'],1):.4f}  shared={run['L_shared_weighted']/max(run['n'],1):.4f}  align={run['L_align']/max(run['n'],1):.4f}  uniform={run['L_uniform']/max(run['n'],1):.4f}  align_prior={run['L_align_prior_weighted']/max(run['n'],1):.4f}"
        if vdl is not None:
            m = evaluate(model, vdl, dev, cfg)
            auc = m.get("AUC", float("nan"))
            is_best = bool(vdl is not None and auc == auc and auc > best_auc)
            append_jsonl_dedup(path, {"global_step": global_step, "epoch": ep + 1, "phase": "val",
                "val_auc": m.get("AUC"), "val_ap": m.get("AP"), "val_eer": m.get("EER"),
                "val_acc": m.get("ACC"), "val_f1": m.get("F1"), "is_best": is_best,
                "ckpt_path": args.out_ckpt if is_best else None}, ("epoch", "phase"))
            log += (f"  val_auc={auc:.4f}" if auc == auc else "  val_auc=nan")
            if is_best:
                best_auc = auc; patience = 0
                save_checkpoint_atomic(args.out_ckpt, {"model": model.state_dict(),
                                                       "cfg": cfg.__dict__, "best_auc": best_auc})
            else:
                patience += 1
        print(log)
        if vdl is not None and patience >= cfg.early_stop_patience:
            print(f"[stage2] early stop @ {ep+1}"); break

        # ── 每 epoch 末落 resume 检查点（云端中断可续跑，最多丢 1 epoch）──
        save_checkpoint_atomic(args.out_ckpt + ".resume", {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": ep, "global_step": global_step,
            "best_auc": best_auc, "patience": patience, "rng": capture_rng(),
        })

        # ── 及时释放 GPU 缓存：每 N epoch 末把 CUDA caching allocator 的空闲块
        #    归还驱动，防止长训练显存碎片累积导致 OOM。仅在 epoch 边界调用，
        #    不进训练热循环，对训练速度/精度零影响。──
        if torch.cuda.is_available() and args.empty_cache_every > 0 and (ep + 1) % args.empty_cache_every == 0:
            torch.cuda.empty_cache()

    if vdl is None:
        save_checkpoint_atomic(args.out_ckpt, {"model": model.state_dict(), "cfg": cfg.__dict__})
    print(f"[stage2] best AUC={best_auc:.4f}  saved -> {args.out_ckpt}")


if __name__ == "__main__":
    main()
