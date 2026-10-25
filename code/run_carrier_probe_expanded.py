#!/usr/bin/env python
"""
run_carrier_probe_expanded.py — 载体探针「扩样本」版一键脚本
================================================================
背景：论文里的载体探针（CKA / RV）结论是 NOT_SEPARABLE，但每组实际只统计了
407–712 个片段 —— 原因是 diag_carrier_probe.py 的默认上限
    --max-per-real 600  --max-per-fake 400
先把每个生成器组截断了。审稿会说「样本这么少，NOT_SEPARABLE 不可靠」。

本脚本做两件事把这件事说清楚：
  1) 先跑 --list-groups，把每个缓存里每个生成器组「真实可用」的 real/fake 条数打出来。
     → 这一步决定上限到底是不是卡在数据量上。如果组里本来就只有 500 条，
       再怎么调 --max-per-* 也扩不了，必须在论文里如实写“受数据集规模限制”，
       而不是继续暗示"我们抽样不够"。
  2) 再按一个大得多的上限（默认 3000/3000）重跑统计。
     若 (1) 显示每组都够，那么样本量提升是真实的；否则打印警告。

它不改 diag_carrier_probe.py 的任何逻辑，只是把参数开大 + 存一份文本报告，
方便把「样本量」这一条直接从 Limitations 里删掉。

用法（CPU 即可，不需要 GPU，不需要 ckpt）：
    python run_carrier_probe_expanded.py
    python run_carrier_probe_expanded.py --max_per_real 5000 --max_per_fake 5000
    python run_carrier_probe_expanded.py --caches HiFi-AVDF/prep Deepfake-Eval-2024/prep/test

进度自查：
    tail -f $SYNCX_PERSIST/diag/carrier_probe_expanded.log
"""
import argparse
import glob
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("SYNCX_DATA_ROOT", "/root/autodl-tmp/SyncX/Datasets")
PERSIST = os.environ.get("SYNCX_PERSIST", "/root/autodl-tmp/SyncX/persist")


def _run(argv):
    """跑探测脚本，把 stdout 原样返回（调用方负责打印 + 落盘）。"""
    r = subprocess.run(argv, cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise SystemExit(f"[carrier] 子命令失败: {' '.join(argv)}")
    return r.stdout


def _parse_pool_sizes(text):
    """从 --list-groups 输出里抓 [组名] real=N fake=M，返回 {组: (N,M)}。

    输出格式固定为：  [veo3.1]  real=1234  fake=567
    """
    out = {}
    for line in text.splitlines():
        m = re.search(r"\[(.+?)\]\s+real=(\d+)\s+fake=(\d+)", line)
        if m:
            out[m.group(1).strip()] = (int(m.group(2)), int(m.group(3)))
    return out


def main():
    ap = argparse.ArgumentParser(description="载体探针扩样本")
    ap.add_argument("--caches", nargs="*", default=None,
                    help="缓存相对路径（相对 SYNCX_DATA_ROOT）；默认 HiFi + DFE")
    ap.add_argument("--group_by", default="identity", choices=["identity", "path"])
    ap.add_argument("--group_regex", default="")
    ap.add_argument("--max_per_real", type=int, default=3000)
    ap.add_argument("--max_per_fake", type=int, default=3000)
    args = ap.parse_args()

    rel_caches = args.caches or ["HiFi-AVDF/prep", "Deepfake-Eval-2024/prep/test"]
    caches = [os.path.join(DATA_ROOT, c) for c in rel_caches]
    for c in caches:
        if not os.path.isdir(c):
            raise SystemExit(f"缓存不存在: {c}（需要先跑 prep 抽特征）")

    out_dir = os.path.join(PERSIST, "diag")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "carrier_probe_expanded.log")

    cache_args = []
    for c in caches:
        cache_args += ["--cache", c]
    base = [sys.executable, os.path.join(HERE, "diag_carrier_probe.py")] + cache_args

    lines = []

    # ---- 第一步：先看池子有多大（决定扩样本是不是被数据量卡住）----
    print("[carrier] 第一步：列出每个生成器的可用条数 ...", flush=True)
    listing = _run(base + ["--group-by", args.group_by, "--list-groups"]
                   + (["--group-regex", args.group_regex] if args.group_regex else []))
    lines += ["", "=" * 72, "STEP 1: 池子大小（--list-groups）", "=" * 72, listing]

    sizes = _parse_pool_sizes(listing)
    if sizes:
        small = {g: n for g, n in sizes.items()
                 if n[0] < args.max_per_real or n[1] < args.max_per_fake}
        print(f"[carrier] 组数={len(sizes)}；受池子上限约束的组={len(small)}")
        if small:
            lines.append("\n⚠️ 以下组的可用条数小于本次上限，扩样本到此为止（非抽样截断）:")
            for g, (nr, nf) in sorted(small.items()):
                lines.append(f"   {g}: real={nr} fake={nf}")

    # ---- 第二步：按大上限重跑统计 ----
    print(f"[carrier] 第二步：重跑统计 (max_real={args.max_per_real}, "
          f"max_fake={args.max_per_fake}) ...", flush=True)
    stat = _run(base + ["--group-by", args.group_by,
                        "--max-per-real", str(args.max_per_real),
                        "--max-per-fake", str(args.max_per_fake)]
                + (["--group-regex", args.group_regex] if args.group_regex else []))
    lines += ["", "=" * 72, "STEP 2: 扩大样本后的统计", "=" * 72, stat]

    text = "\n".join(lines)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(stat)
    print(f"\n[carrier] 完整报告已写入 {log_path}")
    print("[carrier] 论文里要引用的数字：看 STEP 2 的『有效片段 N/M』和最后 NOT_SEPARABLE/SEPARABLE 判定。")


if __name__ == "__main__":
    main()
