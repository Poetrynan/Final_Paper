#!/usr/bin/env python
"""
run_module_ablation.py — 一键补齐论文「组件 / 支路消融」表
================================================================
论文 Table（component ablation）目前只报了 v2 全模型；审稿会问：
  「LCMA 到底有没有用？DWF 的可学习门控比固定 0.5/0.5 好多少？
    LACA 的两条支路分别贡献多少？」
本脚本就把这 4 个缺的格子跑出来（+1 个可选 full 复跑，用来排除环境漂移）。

四个变体（每个 = 一次训练 + 三个跨域集评测）：
  wo_lcma        : --disable_lcma      仅 LACA 分支，LCMA 跨模态注意力整块拿掉
  wo_dwf         : --disable_dwf       融合固定 0.5/0.5，关掉可学习门控
  laca_corr      : laca_mode=corr      只用「唇-音频失配」支路（支路 i）
  laca_biomech   : laca_mode=biomech   只用「生物力学抖动」支路（支路 ii）+ x_s
  full (可选)    : 什么都不关，在当前代码/环境下重训一遍做对照

设计要点（为什么不直接在命令行拼一串 python）：
  1) laca_mode 在训练端没有 CLI，需要套一层 train_laca.py；这里统一处理。
  2) 竞价实例会断，train_stage2.py 自带 <out_ckpt>.resume 自动续跑，
     所以本脚本「发现 ckpt 已存在就跳过训练」不是偷懒，是为了不覆盖已完成的训练。
  3) 每个变体用固定的 --exp_id，评测日志路径可预测，最后能自动汇总成 CSV，
     不需要人肉去 logs 目录里翻。

用法（云端 bash，先 export SYNCX_DATA_ROOT / SYNCX_PERSIST）：
    python run_module_ablation.py --stage all          # 训练 + 评测 + 汇总
    python run_module_ablation.py --stage train        # 只训练
    python run_module_ablation.py --stage eval         # 只评测（ckpt 已训好）
    python run_module_ablation.py --stage summary      # 只汇总现有日志
    python run_module_ablation.py --stage all --only wo_lcma,laca_corr
    python run_module_ablation.py --stage all --seeds "0 1 2"     # 三种子（更贵）

断点续跑：训练中断后，用完全相同的命令重跑即可（train_stage2 自动读 .resume）。
进度自查：
    tail -f $SYNCX_PERSIST/logs/abl_wo_lcma_seed0/stage2_train.jsonl
    ls -lh $SYNCX_PERSIST/ckpt/abl_*.pt
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("SYNCX_DATA_ROOT", "/root/autodl-tmp/SyncX/Datasets")
PERSIST = os.environ.get("SYNCX_PERSIST", "/root/autodl-tmp/SyncX/persist")

# 三个跨域评测集的缓存目录（与 evidence_chain §8 用的路径保持一致）
EVAL_CACHES = {
    "FakeAVCeleb": os.path.join(DATA_ROOT, "FakeAVCeleb_v1.2", "prep"),
    "DFE2024": os.path.join(DATA_ROOT, "Deepfake-Eval-2024", "prep", "test"),
    "HiFiAVDF": os.path.join(DATA_ROOT, "HiFi-AVDF", "prep"),
}

# 每个变体的定义：训练端用什么脚本/参数，评测端用什么参数
VARIANTS = {
    "wo_lcma": {
        "train_flags": ["--disable_lcma"],
        "eval_flags": ["--disable_lcma"],
        "desc": "w/o LCMA（仅 LACA 分支）",
    },
    "wo_dwf": {
        "train_flags": ["--disable_dwf"],
        "eval_flags": ["--disable_dwf"],
        "desc": "w/o DWF（固定 0.5/0.5 融合）",
    },
    "laca_corr": {
        "train_flags": [],
        "eval_flags": ["--laca_mode", "corr"],
        "env": {"SYNCX_LACA_MODE": "corr"},
        "train_entry": "train_laca.py",
        "desc": "仅 LACA 支路 i（唇-音频失配）",
    },
    "laca_biomech": {
        "train_flags": [],
        "eval_flags": ["--laca_mode", "biomech"],
        "env": {"SYNCX_LACA_MODE": "biomech"},
        "train_entry": "train_laca.py",
        "desc": "仅 LACA 支路 ii（生物力学抖动）",
    },
    "full": {
        "train_flags": [],
        "eval_flags": [],
        "desc": "全模型（复跑对照，可不要）",
    },
}


def _run(cmd, env_extra=None):
    """跑一条子命令；失败就抛异常，避免“报错了还继续往下跑”造成半套结果。"""
    env = os.environ.copy()
    env.update(env_extra or {})
    print("\n$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=HERE, env=env)
    if r.returncode != 0:
        raise SystemExit(f"[ablation] 子命令失败 (code={r.returncode})：{cmd[0]}")


def train_one(name, cfg, seed, args):
    entry = os.path.join(HERE, cfg.get("train_entry", "train_stage2.py"))
    out_ckpt = os.path.join(PERSIST, "ckpt", f"abl_{name}_seed{seed}.pt")
    eid = f"abl_{name}_seed{seed}"
    if os.path.exists(out_ckpt) and not args.force:
        print(f"[ablation] {name} seed{seed}: 已有 ckpt，跳过训练 -> {out_ckpt}")
        return out_ckpt, eid
    if not os.path.exists(args.stage1_ckpt):
        print(f"[ablation] ⚠️ stage1 ckpt 不存在: {args.stage1_ckpt}（不加预训练也能跑，"
              f"但会和论文主线不可比，建议先确认）")
    cmd = [
        sys.executable, entry,
        "--train_cache", args.train_cache,
        "--val_cache", args.val_cache,
        "--stage1_ckpt", args.stage1_ckpt,
        "--out_ckpt", out_ckpt,
        "--log_dir", os.path.join(PERSIST, "logs"),
        "--exp_id", eid,
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
        "--lr", str(args.lr),
        "--seed", str(seed),
    ] + list(cfg.get("train_flags", []))
    _run(cmd, cfg.get("env"))
    return out_ckpt, eid


def eval_one(name, cfg, ckpt, eid, seed, args):
    """在三个跨域集上各评一次；开关必须与训练严格配对（否则测的是没训过的支路）。"""
    for ds, cache in EVAL_CACHES.items():
        if not os.path.isdir(cache):
            print(f"[ablation] ⚠️ 跳过 {ds}：缓存不存在 {cache}（先跑 prep 抽特征）")
            continue
        cmd = [
            sys.executable, os.path.join(HERE, "evaluate.py"),
            "--ckpt", ckpt,
            "--eval_cache", cache,
            "--log_dir", os.path.join(PERSIST, "logs"),
            "--exp_id", eid,
            "--dataset", ds,
            "--seed", str(seed),
        ] + list(cfg.get("eval_flags", []))
        _run(cmd, cfg.get("env"))


def _read_overall(path):
    """从评测 jsonl 里取整体那一行（group=None），返回 AUC 等指标。"""
    best = None
    # utf-8-sig：容忍 BOM（云端原生 jsonl 无 BOM，此处只是让脚本在 Windows 上也能自测）
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("group") is None and isinstance(r.get("metrics"), dict):
                best = r
    return best


def build_summary(names, seeds):
    """把所有变体 × 数据集的结果汇总成一张 CSV（论文表格直接抄这一列）。"""
    rows = []
    for name in names:
        for seed in seeds:
            eid = f"abl_{name}_seed{seed}"
            for ds in EVAL_CACHES:
                files = glob.glob(os.path.join(PERSIST, "logs", eid, "eval", f"{ds}__*.jsonl"))
                if not files:
                    continue
                rec = _read_overall(sorted(files)[-1])
                if not rec:
                    continue
                m = rec["metrics"]
                rows.append({
                    "variant": name,
                    "desc": VARIANTS[name]["desc"],
                    "seed": seed,
                    "dataset": rec.get("dataset", ds),
                    "variant_tag": rec.get("variant", ""),
                    "AUC": round(float(m["AUC"]), 4),
                    "AP": round(float(m["AP"]), 4),
                    "EER": round(float(m["EER"]), 4),
                    "ACC": round(float(m["ACC"]), 4),
                })
    out_dir = os.path.join(PERSIST, "ablation")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "module_ablation_summary.csv")
    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[ablation] 汇总已写入 {out_csv}（{len(rows)} 行）")
        # 顺手打一张人看的透视表：行=变体，列=数据集，格=AUC
        print("\n%-14s %s" % ("variant", "  ".join(f"{d:>11s}" for d in EVAL_CACHES)))
        for name in names:
            cells = []
            for ds in EVAL_CACHES:
                vals = [r["AUC"] for r in rows if r["variant"] == name and r["dataset"] == ds]
                cells.append("%11.4f" % (sum(vals) / len(vals)) if vals else "%11s" % "-")
            print("%-14s %s" % (name, "  ".join(cells)))
    else:
        print("[ablation] 没有找到任何评测日志，先跑 --stage eval")
    return out_csv


def main():
    ap = argparse.ArgumentParser(description="组件/支路消融：训练 + 跨域评测 + 汇总")
    ap.add_argument("--stage", default="all", choices=["all", "train", "eval", "summary"])
    ap.add_argument("--only", default="", help="只跑这些变体，逗号分隔，如 wo_lcma,laca_corr")
    ap.add_argument("--with_full", action="store_true", help="额外把全模型也重训一遍做对照")
    ap.add_argument("--seeds", default="0", help="种子列表，空格分隔的字符串，如 \"0 1 2\"")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--stage1_ckpt", default=os.path.join(PERSIST, "ckpt", "stage1.pt"))
    ap.add_argument("--train_cache", default=os.path.join(DATA_ROOT, "DeepSpeak_v1_1", "clips", "prep", "train"))
    ap.add_argument("--val_cache", default=os.path.join(DATA_ROOT, "DeepSpeak_v1_1", "clips", "prep", "test"))
    ap.add_argument("--force", action="store_true", help="ckpt 已存在也重训（会覆盖，慎用）")
    # 三个评测缓存可单独覆写。
    # 为什么必须给 --fac_cache：FakeAVCeleb 的 prep 根目录同时躺着正常四子集和 B6 探针缓存
    # （label 全是 1），整目录当评测集会混入 B6、AUC 全错。正常评测要用只含四子集的 prep_normal。
    ap.add_argument("--fac_cache", default="")
    ap.add_argument("--dfe_cache", default="")
    ap.add_argument("--hifi_cache", default="")
    args = ap.parse_args()

    # 未显式指定时：若存在 prep_normal 就用它，否则退回 prep（并明确警告）
    global EVAL_CACHES
    fac_norm = os.path.join(DATA_ROOT, "FakeAVCeleb_v1.2", "prep_normal")
    if args.fac_cache:
        EVAL_CACHES["FakeAVCeleb"] = args.fac_cache
    elif os.path.isdir(fac_norm):
        EVAL_CACHES["FakeAVCeleb"] = fac_norm
    elif os.path.isdir(EVAL_CACHES["FakeAVCeleb"]):
        print("[ablation] ⚠️ 未发现 prep_normal，FakeAVCeleb 将使用整个 prep（含 B6，结果会失真）")
    if args.dfe_cache:
        EVAL_CACHES["DFE2024"] = args.dfe_cache
    if args.hifi_cache:
        EVAL_CACHES["HiFiAVDF"] = args.hifi_cache
    print("[ablation] 评测缓存：")
    for _k, _v in EVAL_CACHES.items():
        print(f"  {_k:12s} {_v}")

    names = list(VARIANTS) if args.with_full else ["wo_lcma", "wo_dwf", "laca_corr", "laca_biomech"]
    if args.only:
        pick = [x.strip() for x in args.only.split(",") if x.strip()]
        unknown = [x for x in pick if x not in VARIANTS]
        if unknown:
            raise SystemExit(f"未知变体: {unknown}；可选: {list(VARIANTS)}")
        names = pick
    seeds = [int(s) for s in args.seeds.split()]
    print(f"[ablation] 变体={names}  seeds={seeds}  stage={args.stage}")
    print(f"[ablation] DATA_ROOT={DATA_ROOT}\n[ablation] PERSIST={PERSIST}")

    if args.stage in ("all", "train"):
        for name in names:
            for seed in seeds:
                train_one(name, VARIANTS[name], seed, args)

    if args.stage in ("all", "eval"):
        for name in names:
            for seed in seeds:
                ckpt = os.path.join(PERSIST, "ckpt", f"abl_{name}_seed{seed}.pt")
                if not os.path.exists(ckpt):
                    print(f"[ablation] ⚠️ {name} seed{seed} 没有 ckpt，跳过评测")
                    continue
                eval_one(name, VARIANTS[name], ckpt, f"abl_{name}_seed{seed}", seed, args)

    if args.stage in ("all", "summary"):
        build_summary(names, seeds)


if __name__ == "__main__":
    main()
