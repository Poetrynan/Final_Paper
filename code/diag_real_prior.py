#!/usr/bin/env python
"""
diag_real_prior.py — 「真实先验」方向可行性探针：真实 vs 生成的音画一致性统计特性是否跨生成器可分离
====================================================================================================
假设（HAVIC / X-AVDT 式一致性先验，见 evidence_chain §10 调研）：
  真实视频有"来源无关"的高音画一致性（无论数据集/生成器），生成视频即便 lip-sync 高度成熟
  也会在**音画一致性的统计特性**上系统偏离。若真 → 值得加"真实一致性先验"正则/预训练；
  若真视频自身跨集漂移 ≥ real/fake 差距 → 真实先验不迁移，方向判死，不值得重训。

本脚本把「真实域(VoxCeleb2) + 训练域(DeepSpeak) + 跨域评测(HiFi)」一起送入**现有训练好的模型**，
抽取三类**模型自己就在用**的音画一致性统计标量，测"真实 vs 生成"可否分离：

  统计量（都取自现有模型真实前向，非新造特征）：
    corr_pre  = cos(audio_emb, visual_emb)        attention-前 池化投影余弦 —— 模型的 L_corr 一致性目标
    corr_sync = mean_t cos(a_lcma[t], v_lcma[t])  attention-后 逐帧余弦均值 —— LTCA 子系统B 的 sync_t
    sync_std  = std_t  cos(a_lcma[t], v_lcma[t])  逐帧余弦的时间稳定性 —— 一致性能否跨时长保持

判读口径（与 evidence_chain §5/§6 一致，诚实计量）：
  - auc(实0)：real=0 计分；>0.5 表示 real 一致性更高（先验方向合理）；strength=max(auc,1-auc)
  - 每集给 real-mean / fake-mean + within-set AUC → 看方向是否跨集一致（cohen_d 符号不翻转）
  - 全局「真实池 vs 生成池」AUC：一致性先验分类器理论可达水平（⚠️ 附数据集混淆说明）
  - 跨集实列漂移检查：若 real-mean 跨各真实源的极差 > |real/fake 差距| → 真实先验不迁移，判死
  - 只信双类干净集（DeepSpeak/HiFi）+ Vox 全真作真实基准列；FakeAVCeleb 仅参考

用法（云端，需 GPU；每 clip 一次 lcma 前向，几百样本秒级）：
    python diag_real_prior.py \
        --ckpt $PERSIST/ckpt/syncx_best_seed0.pt \
        --cache $Datasets/VoxCeleb2_subset/prep            # 真实域（全 real，作真实基准列）
        --cache $Datasets/DeepSpeak_v1_1/clips/prep        # 训练域（real+fake）
        --cache $Datasets/HiFi-AVDF/prep                   # 跨域评测（real+fake）
        --max-per-class 600

输出：每集一致性统计简表 + 全局可分性 + 跨集实列漂移判定。
"""
import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import get_config
from models import SyncX
from diag_shortcut import roc_auc_y_score, cohens_d, load_index

STATS = ["corr_pre", "corr_sync", "sync_std"]


def per_clip_stats(model, cfg, d, dev):
    """给定单个 clip 特征 dict，返回三个一致性统计标量（模型真实前向，无新参数）。

    corr_pre:  attention-前池化投影余弦 == 模型 L_corr 的 audio_emb/visual_emb 一致方式。
    corr_sync: attention-后逐帧余弦均值 == LTCA sync_t（见 diag_sync_t.py 同款口径）。
    sync_std:  attention-后逐帧余弦标准差（>0=时序上忽高忽低，一致性不稳定）。
    """
    a_seq = d["audio_seq"].float()               # (200,768)
    v_clip = d["v_clip"].float()                 # (100,512)
    a_seq = torch.nn.functional.pad(a_seq, (0, 0, 0, max(0, 200 - a_seq.size(0))))[:200]
    v_clip = torch.nn.functional.pad(v_clip, (0, 0, 0, max(0, 100 - v_clip.size(0))))[:100]
    a_seq = a_seq.to(dev, non_blocking=True).unsqueeze(0)
    v_clip = v_clip.to(dev, non_blocking=True).unsqueeze(0)
    # attention-后 内容嵌入 a_lcma/v_lcma (1,T,d)
    _, _, a_lcma, v_lcma = model.lcma(a_seq, v_clip)
    corr_sync = torch.nn.functional.cosine_similarity(a_lcma, v_lcma, dim=-1)   # (1,T)
    # attention-前 池化投影余弦（复刻 SyncX.forward 的 L_corr 口径）
    a_raw = model.lcma.out_norm(model.lcma.audio_proj(a_seq))
    v_raw = model.lcma.out_norm(model.lcma.visual_proj(v_clip))
    audio_emb = a_raw.mean(dim=1)
    visual_emb = v_raw.mean(dim=1)
    corr_pre = torch.nn.functional.cosine_similarity(audio_emb, visual_emb).item()
    return {
        "corr_pre":  corr_pre,
        "corr_sync": corr_sync.mean(dim=1).item(),
        "sync_std":  corr_sync.std(dim=1).item(),
    }


def run_one(model, cfg, cache, max_per_class, dev):
    print(f"\n=== 探针集: {cache}  max_per_class={max_per_class} ===")
    index = load_index(cache)
    real = [ix for ix in index if ix["label"] == 0]
    fake = [ix for ix in index if ix["label"] == 1]
    print(f"[real_prior] real(池)={len(real)}  fake(池)={len(fake)}  样本池合计 {len(index)}")
    rng = np.random.default_rng(0)
    real = list(rng.choice(real, size=min(max_per_class, len(real)), replace=False)) if real else []
    fake = list(rng.choice(fake, size=min(max_per_class, len(fake)), replace=False)) if fake else []
    rows = [{"label": 0, "path": ix["path"]} for ix in real] + \
           [{"label": 1, "path": ix["path"]} for ix in fake]
    n_ok = 0
    acc = {s: {"real": [], "fake": []} for s in STATS}
    for i, r in enumerate(rows):
        try:
            d = torch.load(r["path"], map_location="cpu", weights_only=True)
        except Exception:
            continue
        try:
            st = per_clip_stats(model, cfg, d, dev)
        except Exception:
            continue
        n_ok += 1
        for s in STATS:
            acc[s]["real" if r["label"] == 0 else "fake"].append(st[s])
        if (i + 1) % 300 == 0:
            print(f"   ... {i+1}/{len(rows)} (ok={n_ok})")
    print(f"[real_prior] 有效前向 {n_ok}/{len(rows)}")
    return cache, acc


def stats_metrics(s, mr, mf):
    """统一指标口径：符号对齐后“越高=越一致”，并让 AUC 与 cohen_d 同方向。

    roc_auc_y_score(y, s) 的正类是 fake(更高分=越像 fake)；故先按 sign 对齐分数
    (corr_* : +1, sync_std : -1)，再取 auc_real_hi = 1 - auc → >0.5 表示
    “real 一致性更强”；cohen_d 用对齐分数，+ 同为 real 更强。两指标严格同向。
    """
    sign = -1.0 if s == "sync_std" else 1.0
    mr_a, mf_a = sign * np.asarray(mr, np.float64), sign * np.asarray(mf, np.float64)
    if len(mr_a) >= 10 and len(mf_a) >= 10:
        y = np.concatenate([np.zeros(len(mr_a)), np.ones(len(mf_a))])
        auc = roc_auc_y_score(y, np.concatenate([mr_a, mf_a]))
        auc_real_hi = 1.0 - auc if not np.isnan(auc) else float("nan")
        st = max(auc_real_hi, 1 - auc_real_hi) if not np.isnan(auc_real_hi) else float("nan")
        d = cohens_d(mr_a, mf_a)
    else:
        auc_real_hi = st = d = float("nan")
    return auc_real_hi, st, d


def print_set_table(name, acc):
    """打印单集 real/fake 均值(原始) + within-set AUC(对齐口径)/cohen_d。"""
    print(f"\n── 集: {name} ──")
    print("%-12s %10s %10s %10s %8s %7s  判定" % ("stat", "real_mean", "fake_mean", "auc(实0更高)", "strength", "cohen_d"))
    for s in STATS:
        mr = np.array(acc[s]["real"]); mf = np.array(acc[s]["fake"])
        if len(mr) < 10:
            print("%-12s %10s  (真实样本 <10，跳过)" % (s, f"{mr.mean():+.4f}" if len(mr) else "NA"))
            continue
        if len(mf) < 10:
            print("%-12s %+10.4f  (无/少 fake —— 纯真实基准列，仅作 real 水平参照)" % (s, mr.mean()))
            continue
        auc_real_hi, st, d = stats_metrics(s, mr, mf)
        mark = "★强" if not np.isnan(st) and st >= 0.8 else ("▲" if not np.isnan(st) and st >= 0.65 else ("●" if not np.isnan(st) and st >= 0.6 else "≈随机"))
        print("%-12s %+10.4f %+10.4f %10.3f %8.3f %+7.3f  %s" %
              (s, mr.mean(), (mf.mean() if len(mf) else np.nan),
               auc_real_hi, st, d, mark))
    print("  说明：auc(实0更高)>0.5 → real 一致性更强（corr_* 更高 / sync_std 更稳）；"
          "cohen_d 同向(+ = real 更强)。符号跨集翻转 ⇒ 不迁移。")


def transferred(pool_real, pool_fake, sets_real_means):
    """跨集实列漂移判定：real-mean 跨源极差 vs 全局 real/fake 差距。

    返回 (verdict, reason)。若全局可分离但 real 跨源漂移 >= 差距 → 真实先验不迁移（判死）。
    """
    gap = float(np.mean(pool_fake) - np.mean(pool_real))     # 高-real 方向，负=False
    spread = float(np.max(sets_real_means) - np.min(sets_real_means)) if len(sets_real_means) >= 2 else 0.0
    if np.mean(pool_real) <= np.mean(pool_fake):  # 方向已反：先验方向错误
        return False, f"real 一致性均值({np.mean(pool_real):+.4f}) < fake({np.mean(pool_fake):+.4f})，先验方向反"
    if spread >= abs(gap):
        return False, f"real 跨源漂移({spread:+.4f}) >= real/fake 差距({gap:+.4f})，真实先验不迁移"
    return True, f"real 跨源漂移({spread:+.4f}) < 差距({gap:+.4f})，真实先验可迁移(方向成立)"


def global_report(all_acc):
    """所有集合并：真实池 vs 生成池 的全局可分性（含数据集混淆提示）+ 跨集漂移判定。"""
    print("\n" + "=" * 70)
    print("全局「真实池 vs 生成池」可分性 (一致性先验分类器理论可达)")
    print("=" * 70)
    for s in STATS:
        real_all = np.concatenate([np.asarray(acc[s]["real"]) for _, acc in all_acc])
        fake_all = np.concatenate([np.asarray(acc[s]["fake"]) for _, acc in all_acc])
        if len(real_all) < 10 or len(fake_all) < 10:
            print(f"[{s}] 数据不足(real_n={len(real_all)}, fake_n={len(fake_all)})，跳过")
            continue
        auc_real_hi, st, d = stats_metrics(s, real_all, fake_all)
        # 漂移判定须符号对齐：corr_* 真实一致性更高(+1)；sync_std 真实更低=更稳(-1)。
        sign = -1.0 if s == "sync_std" else 1.0
        aligned_real, aligned_fake = sign * real_all, sign * fake_all
        sets_real_aligned = [sign * np.mean(acc[s]["real"])
                             for _, acc in all_acc if len(acc[s]["real"]) >= 10]
        ok, reason = transferred(aligned_real, aligned_fake, sets_real_aligned)
        mark = "★强可分" if st >= 0.8 else ("▲可分" if st >= 0.65 else ("●中" if st >= 0.6 else "≈随机"))
        print(f"\n[{s}] auc(实0更高)={auc_real_hi:.3f}  strength={st:.3f}  cohen_d={d:+.3f}  {mark}")
        print(f"   real-mean={np.mean(real_all):+.4f}±{np.std(real_all):.4f} | fake-mean={np.mean(fake_all):+.4f}±{np.std(fake_all):.4f}  (n_real={len(real_all)}, n_fake={len(fake_all)})")
        if ok is not None:
            print(f"   跨集漂移判定: {'✅' if ok else '❌'} {reason}")
        # 每源 real-mean（对齐后，看漂移）
        if len(sets_real_aligned) >= 2:
            print("   real-mean跨源(对齐): " + "; ".join(f"{np.mean(acc[s]['real']):+.4f}"
                  for _, acc in all_acc if len(acc[s]['real']) >= 10))
    print("\n判读：corr_pre/corr_sync 为强可分且跨集漂移<差距 →「真实一致性先验」值得重训；否则方向判死。")


def main():
    ap = argparse.ArgumentParser(description="真实先验方向可行性探针（音画一致性统计特性跨生成器可分性）")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", action="append", required=True, help="可多次：真实域(Vox/全真) + 训练域(DeepSpeak) + 跨域(HiFi) 缓标目录")
    ap.add_argument("--max-per-class", type=int, default=600)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = get_config(**{k: v for k, v in (ck.get("cfg") or {}).items()
                        if k in get_config().__dict__})
    model = SyncX(cfg).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[real_prior] 已加载 ckpt={args.ckpt}")
    print(f"  cfg: fta_enabled={getattr(cfg,'fta_enabled',False)}  laca_mode={cfg.laca_mode}  kin_smooth={cfg.kin_smooth}")
    all_acc = []
    for c in args.cache:
        name, acc = run_one(model, cfg, c, args.max_per_class, dev)
        print_set_table(name, acc)
        all_acc.append((name, acc))
    global_report(all_acc)


if __name__ == "__main__":
    main()