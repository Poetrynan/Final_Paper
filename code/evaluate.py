"""
evaluate.py — 跨域零样本评测 + 消融（B6 静音 trim / B7 身份 / A1 模态遮蔽）
================================================================
载入训练好的 syncx_best.pt，在各评测集上零样本评测。
支持：
  --b6            : B6 Smeu 逐片段动态 trim 条件（加载 prep/b6_trimmed 缓存）
  --b6_untrimmed   : B6 未裁剪基线（加载 prep/b6_raw 缓存，供 plot_figures.plot_f9）
  --mask audio|visual : A1 模态遮蔽（推理时把某模态特征置零，无需重训）
  --laca_mode both|corr|biomech : LACA 双信号消融（biomech=仅运动学，corr=仅唇–音频，
                      both=双支路融合；对应论文 B 表 LACA 支路贡献拆解）
  --disable_lcma  : 组件消融 w/o LCMA（仅 LACA 分支；variant=wo_lcma，对应 tab:component-ablation）
  --disable_dwf   : 组件消融 w/o DWF（固定 0.5/0.5 平均融合；variant=wo_dwf，对应 tab:component-ablation）
  --per_category   : 按 FakeAVCeleb ARVR/AFVR/ARVF/AFVF 细分（需 .pt 文件名含类别）
  --per_group_regex: 通用分组正则（F4/F14），对每条样本文件名 re.search 取 group

日志（供 F2/F3/F4/F5/F8/F9/F14 零重训重画）：
    主评测写 logs/<exp_id>/eval/<dataset>__<variant>.jsonl（含 fpr/tpr）；
    --per_category / --per_group_regex 对每个分组追加带 group 字段的同一 jsonl。
"""
import os, sys, argparse, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from collections import defaultdict
from torch.utils.data import DataLoader
from config import get_config
from models import SyncX
from dataset import FeatureCacheDataset
from utils import autocast_ctx, all_metrics
from logging_utils import append_jsonl, append_jsonl_dedup, roc_curve, exp_id


def run_eval(model, dl, dev, cfg, mask=None, perturb=None, head="fused"):
    model.eval()
    ys, ss, cats = [], [], []
    with torch.inference_mode():
        for batch in dl:
            for k in ("audio_seq", "v_clip", "v_lip"):
                batch[k] = batch[k].to(dev, non_blocking=True)
            if mask == "audio":
                batch["audio_seq"] = torch.zeros_like(batch["audio_seq"])
            elif mask == "visual":
                batch["v_clip"] = torch.zeros_like(batch["v_clip"])
                batch["v_lip"] = torch.zeros_like(batch["v_lip"])   # LACA 双信号支路(ii)仅用 landmark
            if perturb:
                # F16 扰动鲁棒：轻度加性退化（确定性 seed），模拟后处理/对抗噪声
                # 注意：torch.randn_like 不接受 generator 参数，改用 torch.randn(shape, generator=, device=)
                g = torch.Generator(device=dev).manual_seed(1234)
                batch["audio_seq"] = batch["audio_seq"] + torch.randn(batch["audio_seq"].shape, generator=g, device=dev) * 0.05
                batch["v_clip"] = batch["v_clip"] + torch.randn(batch["v_clip"].shape, generator=g, device=dev) * 0.05
            with autocast_ctx(cfg.amp):
                out = model(batch)
            logits = out["audio_logits"] if head == "audio" else out["logits"]
            prob = torch.softmax(logits.float(), dim=1)[:, 1]
            ys.append(batch["label"]); ss.append(prob.cpu())
    y = torch.cat([t.view(-1) for t in ys]); s = torch.cat(ss)
    return all_metrics(y, s), y, s


def _ds_name(eval_cache, override=None):
    canonical = {
        "fakeavceleb": "FakeAVCeleb", "dfe": "DFE2024", "deepfake-eval": "DFE2024",
        "celebdf": "CelebDFv2", "thb": "THB", "talkinghead": "THB",
        "mvad": "MVAD", "hifi": "HiFiAVDF", "ff++": "FF++", "ffp": "FF++",
    }
    if override:
        # 归一化已知小写别名 → canonical；已是规范名（如 DFE2024/DeepSpeak）原样返回
        # 这是 F4/F14 SKIP 的根因修复：smoke 脚本传小写 "fakeavceleb"/"hifi"，
        # 原实现直接 return override → plot_figures 过滤 "FakeAVCeleb"/"HiFiAVDF" 匹配不到
        return canonical.get(override.lower(), override)
    s = os.path.basename(eval_cache.rstrip("/\\")).lower()
    for k, v in canonical.items():
        if k in s:
            return v
    return os.path.basename(eval_cache).split("_")[0]


def _variant_from_args(args):
    # 组件消融优先级最高：w/o LCMA、w/o DWF（对应论文 tab:component-ablation）
    if getattr(args, "variant_override", ""):
        v = args.variant_override
    elif getattr(args, "disable_lcma", False):
        v = "wo_lcma"
    elif getattr(args, "disable_dwf", False):
        v = "wo_dwf"
    elif getattr(args, "disable_xs", False):
        v = "wo_xs"
    elif getattr(args, "exclude_ids", ""):
        v = "b7_with_exclude_ids"
    elif getattr(args, "tag", "") == "b7":
        v = "b7_without_exclude_ids"
    elif getattr(args, "b6_untrimmed", False):
        v = "b6_untrimmed"        # B6 未裁剪基线（plot_figures.plot_f9 需要）
    elif getattr(args, "b6", False):
        v = "b6_trimmed"          # B6 Smeu 逐片段动态 trim 缓存
    elif args.mask == "audio":
        v = "audio-only"
    elif args.mask == "visual":
        v = "video-only"
    elif args.laca_mode != "both":
        v = f"laca_{args.laca_mode}"
    else:
        v = "full"
    if getattr(args, "head", "fused") == "audio":
        v = v + "_head_audio"     # [v2-1a] 用音频独立头打分，记录与融合头区分
    if getattr(args, "fta_no_sync", False):
        v = v + "_fta_no_sync"    # [方向B] 关闭 LTCA 子系统B sync（只留长时骨干）
    return v


def _group_fn(args):
    """返回 (group_name_or_None) 的函数；无分组时返回 None。"""
    if args.per_category:
        cats = ("ARVR", "AFVR", "ARVF", "AFVF")
        return lambda p: next((c for c in cats if c in p), None)
    if args.per_group_regex:
        pat = re.compile(args.per_group_regex)
        def fn(p):
            m = pat.search(p)
            return m.group(1) if (m and m.groups()) else (m.group(0) if m else None)
        return fn
    return None


def _per_sample_eval(model, ds, dev, cfg, mask, group_fn, head="fused"):
    """逐样本前向，按 group 分桶；同时返回整体 (y, s)。"""
    model.eval()
    groups = defaultdict(lambda: ([], []))   # group -> (labels, scores)
    all_y, all_s = [], []
    with torch.inference_mode():
        for i in range(len(ds)):
            item = ds[i]
            p = os.path.basename(item["path"])
            g = group_fn(p) if group_fn else "ALL"
            batch_i = {k: v.unsqueeze(0).to(dev) for k, v in item.items() if isinstance(v, torch.Tensor)}
            if mask == "audio":
                batch_i["audio_seq"] = torch.zeros_like(batch_i["audio_seq"])
            elif mask == "visual":
                batch_i["v_clip"] = torch.zeros_like(batch_i["v_clip"])
                batch_i["v_lip"] = torch.zeros_like(batch_i["v_lip"])
            with autocast_ctx(cfg.amp):
                out = model(batch_i)
            logits = out["audio_logits"] if head == "audio" else out["logits"]
            prob = torch.softmax(logits.float(), dim=1)[0, 1].item()
            all_y.append(item["label"]); all_s.append(prob)
            if g is not None:
                groups[g][0].append(item["label"]); groups[g][1].append(prob)
    y = torch.tensor(all_y); s = torch.tensor(all_s)
    return groups, y, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval_cache", required=True)
    ap.add_argument("--b6", action="store_true",
                    help="B6 Smeu 逐片段动态 trim 条件：加载 prep/b6_trimmed 缓存")
    ap.add_argument("--b6_untrimmed", action="store_true",
                    help="B6 未裁剪基线标记（加载 prep/b6_raw；variant=b6_untrimmed，供 plot_figures.plot_f9 读取）")
    ap.add_argument("--mask", default="none", choices=["none", "audio", "visual"])
    ap.add_argument("--head", default="fused", choices=["fused", "audio"],
                    help="[v2-1a] 打分用哪个分类头：fused=融合AAM头(默认)；audio=音频独立头(udio_aam)。"
                         "测音频-only 改进用 --mask visual --head audio")
    ap.add_argument("--laca_mode", default="both", choices=["both", "corr", "biomech"])
    ap.add_argument("--per_category", action="store_true")
    ap.add_argument("--device", default="cuda")
    # —— 评测提速可覆盖参数（缺省沿用 ckpt 内 cfg，不改动任何语义）——
    ap.add_argument("--batch_size", type=int, default=None,
                    help="覆盖评测 batch（缺省用 cfg.batch_size_stage2=512）；纯推理可加大提速")
    ap.add_argument("--num_workers", type=int, default=None,
                    help="覆盖 DataLoader workers（缺省用 cfg.num_workers=4）")
    # —— 新增：日志 / 分组 / 复现 ——
    ap.add_argument("--log_dir", default="logs")
    ap.add_argument("--exp_id", default=None, help="缺省由 exp_id(cfg, seed, tag) 派生")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default=None, help="覆盖数据集名（否则由路径推断）")
    ap.add_argument("--exclude_ids", default="", help="仅用于 variant 标注（b7_with_exclude_ids）")
    ap.add_argument("--tag", default="", help="附加到 exp_id / variant 标注（如 b7）")
    ap.add_argument("--per_group_regex", default="", help="F4/F14 通用分组正则")
    ap.add_argument("--perturb", default="", help="F16 扰动鲁棒：非空则施加轻度退化并写入 perturb_<tag> 变体")
    ap.add_argument("--variant_override", default="", help="强制覆盖 variant 标签（如 laca_both），优先于自动判定")
    ap.add_argument("--store_scores", action="store_true", help="在日志中保存 real/fake 分数数组")
    # —— 组件消融开关（Gap-2 修复）：优先于 args 的是 ckpt 内 cfg，二者任一为真即标注 wo_* ——
    ap.add_argument("--disable_lcma", action="store_true", help="w/o LCMA 评测（variant=wo_lcma）")
    ap.add_argument("--disable_dwf", action="store_true", help="w/o DWF 评测（variant=wo_dwf）")
    ap.add_argument("--disable_xs", action="store_true",
                    help="w/o BioLip 统计分支 x_s 评测（variant=wo_xs；同时真正关闭模型 x_s 支路）")
    # [方向B] 单变量归因：评测时关闭 LTCA 子系统B(sync 标量)，只留长时骨干 h_long。
    #   不改训练、用已训 ckpt 直接测"去 sync 后 HiFi 是否回升"（见 diag_sync_t.py / evidence_chain §8D 待办）。
    ap.add_argument("--fta_no_sync", action="store_true",
                    help="w/o FTA-sync：LTCA 只保留长时骨干 h_long，跳过子系统B sync（仅推理，不改权重）")
    args = ap.parse_args()

    dev = args.device if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = get_config(**{k: v for k, v in (ck.get("cfg") or {}).items()
                        if k in get_config().__dict__})
    # 从 ckpt 恢复消融开关，确保忘记传 --disable_* 时 variant 仍正确
    args.disable_lcma = bool(args.disable_lcma or getattr(cfg, "disable_lcma", False))
    args.disable_dwf = bool(args.disable_dwf or getattr(cfg, "disable_dwf", False))
    args.disable_xs = bool(args.disable_xs or (not getattr(cfg, "use_xs", True)))
    model = SyncX(cfg).to(dev)
    model.load_state_dict(ck["model"])
    model.cfg.laca_mode = args.laca_mode
    model.cfg.use_xs = not args.disable_xs   # [FIX 2026-08-15] 真正关闭 x_s 支路，使 wo_xs 消融生效（不仅是标签）
    model.cfg.disable_lcma = args.disable_lcma   # [FIX 2026-08-15] 真正关闭 LCMA 模块（原仅改标签 → 现与 x_s 同构）
    model.cfg.disable_dwf = args.disable_dwf     # [FIX 2026-08-15] 真正关闭 DWF 可学习门控（原仅改标签 → 现与 x_s 同构）
    model.cfg.fta_no_sync = bool(getattr(args, "fta_no_sync", False))  # [方向B] 真跳子系统B sync
    model.eval()

    # B6: 加载 prep 产出的扁平缓存。trimmed 条件 = Smeu 逐片段动态 trim 后重提
    # （论文 Ch4:334 B6 protocol）；raw 条件 = 未裁基线。运行时不再做固定 N 零化。
    ds = FeatureCacheDataset(args.eval_cache, T=cfg.T, real_only=False)
    eval_bs = args.batch_size if args.batch_size else cfg.batch_size_stage2
    eval_nw = args.num_workers if args.num_workers is not None else cfg.num_workers
    dl = DataLoader(ds, batch_size=eval_bs, shuffle=False,
                    num_workers=eval_nw, pin_memory=True)
    print(f"[eval] {args.eval_cache}  n={len(ds)}  b6={getattr(args,'b6',False)} "
          f"b6_untrimmed={getattr(args,'b6_untrimmed',False)}  mask={args.mask}  laca_mode={args.laca_mode}")

    eid = args.exp_id or exp_id(cfg, args.seed, args.tag)
    dataset = _ds_name(args.eval_cache, args.dataset)
    variant = _variant_from_args(args)
    out_jsonl = os.path.join(args.log_dir, eid, "eval", f"{dataset}__{variant}.jsonl")

    # 主评测
    metrics, y, s = run_eval(model, dl, dev, cfg, mask=None if args.mask == "none" else args.mask, head=args.head)
    print("[eval] overall:", {k: round(v, 4) for k, v in metrics.items()})

    fpr, tpr = roc_curve(y.numpy(), s.numpy())
    rec = {"exp_id": eid, "dataset": dataset, "split": "test",
           "method": "syncx", "variant": variant, "seed": args.seed,
           "global_step": None,   # 评测是独立过程，不绑训练 step
           "metrics": {k: float(metrics[k]) for k in ("AUC", "AP", "EER", "ACC", "F1")},
           "fpr": fpr, "tpr": tpr, "group": None}
    if args.store_scores:
        rec["scores_real"] = s[y == 0].numpy().tolist()
        rec["scores_fake"] = s[y == 1].numpy().tolist()
    append_jsonl_dedup(out_jsonl, rec, ("dataset", "variant", "seed", "group"))

    # F16 扰动鲁棒：同 ckpt 同数据，施加轻度退化后重评，落 perturb_<tag> 变体
    if args.perturb:
        metrics_p, yp, sp = run_eval(model, dl, dev, cfg, mask=None, perturb=args.perturb, head=args.head)
        fp, tp = roc_curve(yp.numpy(), sp.numpy())
        rec_p = dict(rec)
        rec_p["variant"] = f"perturb_{args.perturb}"
        rec_p["metrics"] = {k: float(metrics_p[k]) for k in ("AUC", "AP", "EER", "ACC", "F1")}
        rec_p["fpr"], rec_p["tpr"] = fp, tp
        rec_p["group"] = None
        if args.store_scores:
            rec_p["scores_real"] = sp[yp == 0].numpy().tolist()
            rec_p["scores_fake"] = sp[yp == 1].numpy().tolist()
        append_jsonl_dedup(out_jsonl, rec_p, ("dataset", "variant", "seed", "group"))
        print(f"[eval] perturb_{args.perturb}:", {k: round(v, 4) for k, v in metrics_p.items()})

    # 分组（F4/F14）
    gfn = _group_fn(args)
    if gfn is not None:
        groups, gy, gs = _per_sample_eval(model, ds, dev, cfg,
                                          mask=None if args.mask == "none" else args.mask, group_fn=gfn, head=args.head)
        # 分组时保留整体记录已写入；这里只为每个 group 追加
        for g, (yy, ss_) in sorted(groups.items()):
            if len(yy) == 0:
                continue
            gm = all_metrics(torch.tensor(yy), torch.tensor(ss_))
            print(f"[eval] {g}: { {k: round(v,4) for k,v in gm.items()} }  n={len(yy)}")
            gfpr, gtpr = roc_curve(torch.tensor(yy).numpy(), torch.tensor(ss_).numpy())
            grec = dict(rec)
            grec["group"] = g
            grec["metrics"] = {k: float(gm[k]) for k in ("AUC", "AP", "EER", "ACC", "F1")}
            grec["fpr"], grec["tpr"] = gfpr, gtpr
            if args.store_scores:
                yt = torch.tensor(yy); st = torch.tensor(ss_)
                grec["scores_real"] = st[yt == 0].numpy().tolist()
                grec["scores_fake"] = st[yt == 1].numpy().tolist()
            append_jsonl_dedup(out_jsonl, grec, ("dataset", "variant", "seed", "group"))


if __name__ == "__main__":
    main()
