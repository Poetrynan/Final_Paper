#!/usr/bin/env python
"""
report_seed_stats.py — 把已有的多种子评测日志汇总成 mean ± std
================================================================
用途：论文里 v2 主结果原来只报了一个种子的单点值（FakeAVCeleb 0.6187、
DFE-2024 0.5350、HiFi-AVDF 0.4735），这是审稿最常打的「单种子不可靠」。
好消息是：三种子的评测日志其实早就跑过了，只是没汇总 —— 本脚本只读日志、
不训练、不评测、不碰任何 ckpt，所以不会有任何数字被改动或产生。

它会做三件事：
  1) 按 exp_group 前缀找到 seed0 / seed1 / seed2 三个实验目录；
  2) 每个目录的 eval/<Dataset>__<variant>.jsonl 里取整体那行（group=None）的 AUC；
  3) 输出 | 数据集 | n seeds | mean | std | 每个种子原始值 | 的表格（Markdown + CSV）。

注意 std 用样本标准差（ddof=1）。论文里如果写 ±，请和这里保持一致；
若审稿要求 ddof=0，加 --ddof 0 即可。

用法：
    python report_seed_stats.py                       # 自动扫 logs 下所有 syncx_bs512_lam0.3_lacaboth_seed*
    python report_seed_stats.py --exp_group syncx_bs512_lam0.3_lacaboth
    python report_seed_stats.py --variant full --datasets FakeAVCeleb,DFE2024,HiFiAVDF
"""
import argparse
import csv
import glob
import json
import os
import re

PERSIST = os.environ.get("SYNCX_PERSIST", "/root/autodl-tmp/SyncX/persist")
CANON = {  # 日志里的 dataset 名 → 论文表格里的写法
    "FakeAVCeleb": "FakeAVCeleb",
    "DFE2024": "DFE2024",
    "HiFiAVDF": "HiFiAVDF",
}


def _seed_of(dirname):
    """从目录名尾部抓种子号：..._seed2 或 ..._seed2_seed2 都认。"""
    m = re.search(r"_seed(\d+)$", dirname)
    return int(m.group(1)) if m else None


def _overall_auc(path):
    """取 jsonl 里 group=None 的那一行（整体指标），返回 AUC；没有就 None。"""
    auc = None
    # 用 utf-8-sig：容忍 Windows 上被记事本/重定向写过的 BOM（云端原生 jsonl 没有 BOM）
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("group") is None and isinstance(r.get("metrics"), dict):
                auc = r["metrics"].get("AUC")
    return auc


def main():
    ap = argparse.ArgumentParser(description="多种子评测日志 → mean ± std")
    ap.add_argument("--log_root", default=os.path.join(PERSIST, "logs"))
    ap.add_argument("--exp_group", default="syncx_bs512_lam0.3_lacaboth",
                    help="实验目录前缀（去掉 _seedN 的组名）")
    ap.add_argument("--variant", default="full", help="评测变体标签（默认 full）")
    ap.add_argument("--datasets", default="FakeAVCeleb,DFE2024,HiFiAVDF")
    ap.add_argument("--ddof", type=int, default=1, help="标准差自由度：1=样本std(默认)，0=总体std")
    ap.add_argument("--out_dir", default=os.path.join(PERSIST, "ablation"))
    args = ap.parse_args()

    ds_list = [x.strip() for x in args.datasets.split(",") if x.strip()]
    dirs = sorted(glob.glob(os.path.join(args.log_root, args.exp_group + "_seed*")))
    if not dirs:
        raise SystemExit(f"没找到实验目录：{os.path.join(args.log_root, args.exp_group + '_seed*')}")

    # 收集 {dataset: {seed: auc}}
    table = {ds: {} for ds in ds_list}
    for d in dirs:
        seed = _seed_of(os.path.basename(d))
        if seed is None:
            continue
        for ds in ds_list:
            files = glob.glob(os.path.join(d, "eval", f"{ds}__{args.variant}.jsonl"))
            if not files:
                # 有些日志的 variant 标签带后缀（如 full_head_fused），退一步模糊匹配
                files = glob.glob(os.path.join(d, "eval", f"{ds}__*.jsonl"))
            if not files:
                continue
            auc = _overall_auc(sorted(files)[-1])
            if auc is not None:
                table[ds][seed] = float(auc)

    rows = []
    print("\n| dataset | n seeds | mean | std | per-seed AUC |")
    print("|---|---|---|---|---|")
    for ds in ds_list:
        vals = table[ds]
        if not vals:
            print(f"| {ds} | 0 | - | - | 日志缺失 |")
            continue
        xs = [vals[s] for s in sorted(vals)]
        n = len(xs)
        mean = sum(xs) / n
        if n > 1:
            var = sum((x - mean) ** 2 for x in xs) / (n - args.ddof)
            std = var ** 0.5
        else:
            std = 0.0
        per = ", ".join(f"s{s}={vals[s]:.4f}" for s in sorted(vals))
        print(f"| {ds} | {n} | {mean:.4f} | {std:.4f} | {per} |")
        rows.append({"dataset": ds, "n_seeds": n, "mean": round(mean, 4),
                     "std": round(std, 4), "ddof": args.ddof, "per_seed": per})

    if rows:
        os.makedirs(args.out_dir, exist_ok=True)
        out = os.path.join(args.out_dir, f"seed_stats_{args.exp_group}_{args.variant}.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[seed_stats] 已写入 {out}")

    # 提醒：只有 1 个种子时 std 恒为 0，别把 0 写进论文当成“稳定”
    if all(len(v) <= 1 for v in table.values()):
        print("\n⚠️ 每个数据集只找到 1 个种子，std 无意义。请确认日志目录命名，"
              "或先用 --seeds \"0 1 2\" 跑 run_module_ablation.py 补齐训练。")


if __name__ == "__main__":
    main()
