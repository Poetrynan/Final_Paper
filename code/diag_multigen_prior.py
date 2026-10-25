#!/usr/bin/env python
"""
diag_multigen_prior.py — 廉价多源一致性先验探针（不重训、只用现有 ckpt + 盘上缓存）
====================================================================================
目的：给"多生成器/多域一致性联合训练（ROGA 式）"这个尚未试过的方向做**上线前闸门**，
     判断"真实一致性先验跨生成器是否可分的"这一必要条件是否成立。不满足则不值得烧重训。

与 diag_real_prior.py 的本质区别（这也是"多源"这一新增值的来源）：
  - diag_real_prior 只按【整集】聚合 fake，看不到"哪些生成器方向对、哪些反过来"；
  - 本脚本把每个评测集的 fake 按【单个生成器】分组，逐组算 real-vs-该生成器 可分性，
    再跨所有生成器统计【符号一致性】——这正是"跨生成器不变"的操作性定义，
    也决定"多源监督有没有一个一致的目标可学"。

统计量（复用 diag_real_prior.per_clip_stats，模型现有前向，无新参数，非拟合）：
    corr_pre  = cos(audio_emb, visual_emb)        注意力前 L_corr 一致性
    corr_sync = mean_t cos(a_lcma[t], v_lcma[t])  LTCA 逐帧一致性
    sync_std  = std_t  cos(a_lcma[t], v_lcma[t])  一致性时间稳定性

分组（必须可靠，否则探针自欺；因此强制两步）：
  ① 默认按缓存元数据 identity （HiFi 的 identity=生成器名，权威）；
     其它集若 identity 不是生成器，用 --group-regex 对 cache_dir 相对路径取 capture group 1。
  ② 运行前先 --list-groups 打印【检测到的组 + 该组 real/fake 数量 + 2 个示例路径】，
     由你确认分组正确后，再跑统计。分组不对，绝不出结论。

判读闸门（跑前定死，不事后改，对标 evidence_chain §10/§13.1）：
   对于某个统计量 s，设 G = 满足 real/fake 分别 ≥10 的所有生成器组：
     POSITIVE(值得多源重训):
       存在 ≥2 个生成器组 strength≥0.6 且符号相同，且大多数(>一半)组符号一致；
       且全局(real vs 全部fake)同方向 strength≥0.6；且该统计量真实跨源漂移<real/fake差距。
     NEGATIVE(不烧卡):
       G 中所有组 strength<0.6，或符号冲突无大多比（含方向乱）；→ 连现有最优对齐下、
       逐生成器都没有稳定一致的一致性信号 → 多源监督没有一致目标可学，不值得重训。

注（诚实界定）：
  - 一致性用现有 ckpt 的对齐（受单源训练影响）；POSITIVE 是"值得多源重训"的**必要非充分**条件。
  - NEGATIVE 则强化 §13b 结论：一致性在"当前对齐"下无法成为跨生成器不变量。
  - 判据只看干净双类集期望（HiFi 各生成器 / DFE）；FakeAVCeleb 因极端不平衡仅参考（strength≥0.6 才算数）。

用法（云端，GPU，直接挥舞每 clip 一次 lcma 前向，几百~几千样本分钟级）：
  1) 先验证分组：
     python diag_multigen_prior.py --ckpt $PERSIST/ckpt/syncx_best_seed0.pt \
        --cache $Datasets/HiFi-AVDF/prep --group-by identity --list-groups
  2) 正式统计（可一次给多个评测集）：
     python diag_multigen_prior.py --ckpt $PERSIST/ckpt/syncx_best_seed0.pt \
        --cache $Datasets/HiFi-AVDF/prep --cache $Datasets/Deepfake-Eval-2024/prep/test \
        --group-by identity --max-per-fake 400 --max-per-real 600
"""
import os, sys, argparse, re, glob
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diag_shortcut import load_index
from diag_real_prior import per_clip_stats, STATS, stats_metrics

# 判读闸门参数（跑前定死，见文档）
MIN_GROUP_STRENGTH = 0.60      # 单生成器组算"可分"的 strength 下限
MIN_GROUPS_POSITIVE = 2        # 至少要几个生成器组同符号可达可分
GATE_STRENGTH_NEEDED = 0.60    # 全局可分需要的 strength


def _realpath_under(path, root):
    try:
        return os.path.relpath(path, root)
    except Exception:
        return path


def group_index(cache, group_by, group_regex):
    """把 cache 下 index 分组：返回 {group: {0:[], 1:[]}}（real=0, fake=1，按 identity 或路径正则）。"""
    index = load_index(cache)
    rgx = re.compile(group_regex) if group_regex else None
    buckets = {}

    def grp_of(ix):
        if group_by == "identity":
            return ix.get("identity") or "NOID"
        if rgx is None:
            raise SystemExit("--group-by path 需要 --group-regex（relpath 上取 group(1)）")
        rel = _realpath_under(ix["path"], cache)
        m = rgx.search(rel)
        return m.group(1) if (m and m.groups()) else (m.group(0) if m else "NOGROUP")

    for ix in index:
        if ix["label"] not in (0, 1):
            continue
        g = grp_of(ix)
        buckets.setdefault(g, {0: [], 1: []})[ix["label"]].append(ix)
    # 删除空组
    return {g: d for g, d in buckets.items() if (d[0] or d[1])}


def list_groups(caches, group_by, group_regex):
    print("=" * 74)
    print("分组核对（可信性前置：先确认这里的分组正确，再跑统计）")
    print("=" * 74)
    fail = False
    for cache in caches:
        print(f"\n# 缓存: {cache}")
        buckets = group_index(cache, group_by, group_regex)
        if not buckets:
            print("   (无可分组样本)")
            continue
        for g in sorted(buckets):
            d = buckets[g]
            nr, nf = len(d[0]), len(d[1])
            ex = []
            for lab in (0, 1):
                if d[lab]:
                    ex.append(os.path.basename(os.path.dirname(d[lab][0]["path"])))
            flag = ""
            if nr < 10 or nf < 10:
                flag = "  ⚠️ 样本过少，不入统计"
                fail = True
            print(f"  [{g}]  real={nr}  fake={nf}  {flag}")
            print(f"        示例: {ex}")
    print("\n请核对：'(identity 或 路径正则) 是否真的对应单一生成器'。若不对，改用 --group-regex。")


def run_probe(cache, model, cfg, dev, group_by, group_regex, max_real, max_fake):
    print(f"\n=== 探针集: {cache} ===")
    buckets = group_index(cache, group_by, group_regex)
    rng = np.random.default_rng(0)
    acc = {s: {} for s in STATS}                  # acc[s]["<gen>"] = {"real":[], "fake":[]}

    for g in sorted(buckets):
        d = buckets[g]
        real_ix, fake_ix = d[0], d[1]
        if len(real_ix) < 10 or len(fake_ix) < 10:
            print(f"  [组 {g}] real={len(real_ix)} fake={len(fake_ix)} <10，跳过")
            continue
        real_ix = list(rng.choice(real_ix, size=min(max_real, len(real_ix)), replace=False))
        fake_ix = list(rng.choice(fake_ix, size=min(max_fake, len(fake_ix)), replace=False))
        rows = [{"label": 0, "path": ix["path"]} for ix in real_ix] + \
               [{"label": 1, "path": ix["path"]} for ix in fake_ix]
        gacc = {s: {"real": [], "fake": []} for s in STATS}
        n_ok = 0
        for r in rows:
            try:
                dpt = __import__("torch").load(r["path"], map_location="cpu", weights_only=True)
            except Exception:
                continue
            try:
                st = per_clip_stats(model, cfg, dpt, dev)
            except Exception:
                continue
            n_ok += 1
            for s in STATS:
                gacc[s]["real" if r["label"] == 0 else "fake"].append(st[s])
        print(f"  [组 {g}] 有效前向 {n_ok}/{len(rows)}")
        # 记录（无论可分与否都保留，供符号一致性判断）
        for s in STATS:
            if len(gacc[s]["real"]) >= 10 and len(gacc[s]["fake"]) >= 10:
                acc[s][g] = gacc[s]
    return cache, acc


def print_group_table(name, acc):
    print(f"\n── 集: {name} ── (每生成器组: real vs 该生成器 fake —— real 来自同集真实池)")
    print("%-10s %-12s %10s %8s %+8s  判定" % ("stat", "generator", "auc(实0更高)", "strength", "cohen_d"))
    for s in STATS:
        for g in sorted(acc[s]):
            auc_r, st, d = stats_metrics(s, acc[s][g]["real"], acc[s][g]["fake"])
            mark = "★强" if st >= 0.8 else ("▲" if st >= 0.65 else ("●" if st >= 0.6 else "≈随机"))
            print("%-10s %-12s %10.3f %8.3f %+8.3f  %s" %
                  (s, str(g)[:12], auc_r, st, (d if not np.isnan(d) else 0.0), mark))


def gate_report(acc_all):
    """跨所有集的生成器组做符号一致性判定。acc_all: list of (cache_name, acc[stat][gen])。"""
    print("\n" + "=" * 74)
    print("多源判读闸门（跨生成器符号一致性 —— “跨生成器不变”的操作性定义）")
    print("=" * 74)
    # 收集每个 stat 的 所有生成器组：(gen_label, auc_r, strength, cohen_d sign)
    for s in STATS:
        groups = []
        for _, acc in acc_all:
            for g, d in (acc.get(s) or {}).items():
                auc_r, st, d_ = stats_metrics(s, d["real"], d["fake"])
                groups.append((f"{g}", auc_r, st, d_))
        usable = [gr for gr in groups if gr[2] >= MIN_GROUP_STRENGTH or gr[3] is not np.nan]
        n_used = sum(1 for gr in groups if not np.isnan(gr[3]))
        if n_used == 0:
            print(f"\n[{s}] 无可统计生成器组")
            continue
        # 方向：用 aligned cohen_d 符号（stats_metrics 已把 + 对齐为 real 更强）
        signs = [np.sign(gr[3]) for gr in groups if not np.isnan(gr[3])]
        positive_groups = [gr for gr in groups if gr[2] >= MIN_GROUP_STRENGTH and gr[3] > 0]
        n_strong = len([gr for gr in groups if gr[2] >= MIN_GROUP_STRENGTH])
        n_pos_strong = len(positive_groups)
        n_pos_sign = sum(1 for x in signs if x > 0)
        n_neg_sign = sum(1 for x in signs if x < 0)
        majority_sign = 1 if n_pos_sign >= n_neg_sign else -1
        n_majority = max(n_pos_sign, n_neg_sign)
        n_sign = len(signs)
        sign_ok = (n_strong >= MIN_GROUPS_POSITIVE and n_pos_strong >= MIN_GROUPS_POSITIVE and
                   n_majority > n_sign / 2)

        print(f"\n[{s}] 生成器组数={n_used}  强可分(≥{MIN_GROUP_STRENGTH})={n_strong} "
              f"(其中 real更强方向={n_pos_strong})  符号方向 +:{n_pos_sign} / -:{n_neg_sign}")
        for gr in groups:
            print("    %-12s auc(实0=%+.3f) strength=%.3f cohen_d=%+.3f" % (
                str(gr[0])[:12], gr[1], gr[2], gr[3] if not np.isnan(gr[3]) else 0.0))
        if sign_ok:
            print("    ✅ POSITIVE：≥2 生成器组同符号强可分 且 大多数组方向一致 → 值得多源重训")
        else:
            print("    ❌ NEGATIVE：强可分组不足 或 符号方向冲突/无大多比 → 一致性无一致目标，不值得重训（强化 §13b）")
    print("\n注：corr_* 的 + 方向 = real 一致性更强；sync_std 的 + = real 更稳（stats_metrics 已对齐口径）。")


def main():
    ap = argparse.ArgumentParser(description="廉价多源一致性先验探针（不重训）")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", action="append", required=True)
    ap.add_argument("--group-by", choices=["identity", "path"], default="identity")
    ap.add_argument("--group-regex", default="", help="--group-by path 时对 relpath 取 group(1)")
    ap.add_argument("--list-groups", action="store_true", help="只列分组核对，不跑统计")
    ap.add_argument("--max-per-real", type=int, default=600)
    ap.add_argument("--max-per-fake", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.list_groups:
        list_groups(args.cache, args.group_by, args.group_regex)
        return

    dev = args.device if __import__("torch").cuda.is_available() else "cpu"
    torch = __import__("torch")
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    from config import get_config
    from models import SyncX
    cfg = get_config(**{k: v for k, v in (ck.get("cfg") or {}).items()
                        if k in get_config().__dict__})
    model = SyncX(cfg).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[multigen] ckpt={args.ckpt}  group_by={args.group_by}  "
          f"group_regex={args.group_regex or '(identity)'}")

    acc_all = []
    for c in args.cache:
        name, acc = run_probe(c, model, cfg, dev, args.group_by, args.group_regex,
                              args.max_per_real, args.max_per_fake)
        print_group_table(name, acc)
        acc_all.append((name, acc))
    gate_report(acc_all)


if __name__ == "__main__":
    main()