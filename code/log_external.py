"""
log_external.py — 注入基线 / 文献值到统一 schema（F2/F3/F9）
================================================================
把外部方法（AVFF / Seedance / MMMS-BA / AV-Lip-Sync+ / MDS 等）的已发表
或复现指标写进 logs/<exp_group>/eval/<dataset>__<variant>.jsonl，与 Sync-X
的评测日志同 schema，供 plot_figures 直接合并。

注意：
  - 基线/文献值写在**种子无关**的组目录 logs/<exp_group>/eval/（与所有 seed 的 Sync-X 聚合）。
  - 这些方法不进 Sync-X 训练日志；其 AUC 通过 --auc（必要时 --ap）注入。
  - MDS 的 F9 双条件（untrimmed / trimmed）需各跑一次：见 §7 示例。

示例：
    python log_external.py --log_dir logs --exp_group syncx_bs256_lam0.3_lacaboth \
        --dataset FakeAVCeleb --method mds --variant b6_trimmed --auc 73.8
    python log_external.py --log_dir logs --exp_group syncx_bs256_lam0.3_lacaboth \
        --dataset FakeAVCeleb --method mds --variant b6_untrimmed --auc 90.4
"""
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logging_utils import append_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", default="logs")
    ap.add_argument("--exp_group", required=True, help="Sync-X 的 exp_group（不含 _seedN）")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", required=True, help="e.g. mds / avff / seedance / mmmsba / avlipsync")
    ap.add_argument("--variant", default="full")
    ap.add_argument("--auc", type=float, required=True)
    ap.add_argument("--ap", type=float, default=None)
    ap.add_argument("--eer", type=float, default=None)
    ap.add_argument("--acc", type=float, default=None)
    ap.add_argument("--f1", type=float, default=None)
    ap.add_argument("--group", default=None, help="分组（F4/F14），如 ARVF / 各 generator")
    ap.add_argument("--fpr", default="", help="JSON 数组字符串（可选 ROC 曲线）")
    ap.add_argument("--tpr", default="", help="JSON 数组字符串（可选 ROC 曲线）")
    args = ap.parse_args()

    metrics = {"AUC": args.auc, "AP": args.ap, "EER": args.eer,
               "ACC": args.acc, "F1": args.f1}
    metrics = {k: v for k, v in metrics.items() if v is not None}

    rec = {"exp_id": args.exp_group, "dataset": args.dataset, "split": "test",
           "method": args.method, "variant": args.variant, "seed": None,
           "metrics": metrics, "fpr": json.loads(args.fpr or "[]"),
           "tpr": json.loads(args.tpr or "[]"), "group": args.group}
    out = os.path.join(args.log_dir, args.exp_group, "eval", f"{args.dataset}__{args.variant}.jsonl")
    append_jsonl(out, rec)
    print(f"[log_external] wrote {out}  method={args.method} auc={args.auc} variant={args.variant}")


if __name__ == "__main__":
    main()
