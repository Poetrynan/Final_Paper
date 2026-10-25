#!/usr/bin/env python3
"""
organize_deepspeak.py — 校验并按官方 split-def 整理 DeepSpeak v1.1 目录。

prep_deepspeak.py 期望布局:
    <ROOT>/clips/real/        train 真实
    <ROOT>/clips/fake/        train 伪造
    <ROOT>/clips/test/real/   test  真实
    <ROOT>/clips/test/fake/   test  伪造

本脚本:
    1. 读 annotations-split-def.csv -> train_ids / test_ids
    2. 扫描上述 4 目录，按文件名前缀推断 identity
       - real:  "<identity>-<rec>.mp4"          -> identity = 前缀
       - fake:  "<identity-target>-<rec>.mp4"   -> identity = 前缀
    3. 校验每个视频的 identity 是否落在正确 split
       (train 身份应在 real/ 或 fake/；test 身份应在 test/real/ 或 test/fake/)
    4. 默认只报告各 split/label 数量 + 错位清单；--fix 才移动错位文件

用法:
    python organize_deepspeak.py --root /root/autodl-tmp/SyncX/Datasets/DeepSpeak_v1_1
    python organize_deepspeak.py --root <本地路径>          # 先看报告
    python organize_deepspeak.py --root <路径> --fix         # 确认无误后整理
"""
import os
import csv
import argparse
import shutil


def load_split_def(root):
    path = os.path.join(root, "annotations-split-def.csv")
    train_ids, test_ids = set(), set()
    with open(path, encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            ident = (r.get("identity") or "").strip()
            split = (r.get("split") or "").strip()
            if not ident:
                continue
            if split == "train":
                train_ids.add(ident)
            elif split == "test":
                test_ids.add(ident)
    return train_ids, test_ids


def identity_of(vf: str, is_fake: bool) -> str:
    stem = vf[:-4] if vf.lower().endswith(".mp4") else os.path.splitext(vf)[0]
    # real:  "<identity>-<rec>.mp4"                       -> identity = 第一段
    # fake:  "<engine>--<src>-<tgt>--<recsrc>-<rectgt>--<k>.mp4"
    #         (如 facefusion--0-1--4095-4931--0.mp4)      -> identity = 第二段(identity-target)
    if is_fake and "--" in stem:
        parts = stem.split("--")
        tgt = parts[1] if len(parts) > 1 else stem
        return tgt.split("-")[0] if "-" in tgt else tgt
    return stem.split("-")[0] if "-" in stem else stem


def main() -> None:
    ap = argparse.ArgumentParser(description="校验/整理 DeepSpeak v1.1 目录")
    ap.add_argument("--root", default="/root/autodl-tmp/SyncX/Datasets/DeepSpeak_v1_1",
                    help="DeepSpeak_v1_1 根目录（含 clips/ 与 annotations-*.csv）")
    ap.add_argument("--fix", action="store_true", help="移动错位文件（默认仅报告）")
    args = ap.parse_args()

    root = os.path.expanduser(args.root)
    clips = os.path.join(root, "clips")
    if not os.path.isdir(clips):
        print(f"[error] 找不到 {clips}，请先下载并解压")
        return

    train_ids, test_ids = load_split_def(root)
    print(f"[split-def] train={len(train_ids)} identities, test={len(test_ids)} identities\n")

    plan = [
        ("train", "real", 0, train_ids),
        ("train", "fake", 1, train_ids),
        ("test",  "real", 0, test_ids),
        ("test",  "fake", 1, test_ids),
    ]

    stat = {}
    mismatches = []
    for split, sub, _label, expected_ids in plan:
        src = os.path.join(clips, sub) if split == "train" else os.path.join(clips, "test", sub)
        if not os.path.isdir(src):
            print(f"[warn] 目录不存在: {src}")
            continue
        for vf in sorted(os.listdir(src)):
            if not vf.lower().endswith(".mp4"):
                continue
            if vf.startswith("._"):        # macOS AppleDouble 垃圾，官方 loader 也跳过
                continue
            is_fake = (sub == "fake")
            ident = identity_of(vf, is_fake)
            stat[(split, sub)] = stat.get((split, sub), 0) + 1
            if ident not in expected_ids:
                mismatches.append((split, sub, vf, ident))

    # 报告
    print(f"{'split':6} {'sub':6} {'count':>7}")
    print("-" * 22)
    for split, sub, _l, _e in plan:
        c = stat.get((split, sub), 0)
        print(f"{split:6} {sub:6} {c:>7}")
    total = sum(stat.values())
    print("-" * 22)
    print(f"{'TOTAL':6} {'':6} {total:>7}")

    if mismatches:
        print(f"\n[!] 发现 {len(mismatches)} 个身份错位文件:")
        for split, sub, vf, ident in mismatches[:20]:
            print(f"    {split}/{sub}: {vf}  (identity={ident} 不在该 split)")
        if len(mismatches) > 20:
            print(f"    ... 其余 {len(mismatches) - 20} 个省略")
        if args.fix:
            print("\n[fix] 按身份重新归位 ...")
            # 把错位文件移到 clips/_misplaced/ 待人工检查，避免误删
            dest = os.path.join(clips, "_misplaced")
            os.makedirs(dest, exist_ok=True)
            for split, sub, vf, _ident in mismatches:
                src = os.path.join(clips, sub if split == "train" else os.path.join("test", sub), vf)
                shutil.move(src, os.path.join(dest, f"{split}_{sub}_{vf}"))
            print(f"[fix] 已移 {len(mismatches)} 个文件到 {dest}/（请人工核对官方 split-def）")
    else:
        print("\n[ok] 所有视频身份均与官方 split-def 一致，目录无需整理。")


if __name__ == "__main__":
    main()
