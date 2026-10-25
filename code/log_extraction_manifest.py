"""
log_extraction_manifest.py — 抽取阶段全局日志聚合（MVP 解耦架构的一部分）
===============================================================================
问题：每个 prep_*.py 只把各自数据集的 clip 元数据写到自己的 PREP 目录
      （manifest.jsonl），散落多处，且没有"跨数据集总览"。训练/评测的日志
      已经结构化（logs/<exp_id>/{stage1,stage2}_train.jsonl、eval/*.jsonl），
      唯独"抽取"这一步没有统一的归档入口——但这恰恰是论文 Figure（数据规模、
      维度契约 64 点→192-D、各集 clip 数）的数据来源。

本脚本在抽取完成后（cloud_run_all.sh STEP 1 之后调用）扫描所有已知数据集的
PREP 目录，聚合成一份 **单一、结构化、可重画** 的全局 manifest：

    logs/extraction/manifest.json
        {
          "run_id": ..., "generated_at": ..., "host": ...,
          "python": ..., "torch": ..., "cuda_available": ...,
          "data_root": ...,
          "contract": {"lip_lm_dim": 192, "T_frames": ..., "v_clip_dim": 512, ...},
          "total_clips": N,
          "datasets": [
             {"name": "DeepSpeak_train", "prep_root": "...", "n_clips": ...,
              "n_real": ..., "n_fake": ..., "n_identities": ...,
              "shapes": {"audio_seq": [T_a,768], "v_clip":[T,512], "v_lip":[T,192]},
              "manifest_jsonl": "exists|missing", "exists": true},
             ...
          ]
        }

特点：
  - 只读缓存，绝不触发任何训练/抽取；纯 CPU、秒级。
  - 幂等：每次抽取后重跑即刷新（覆盖写）。
  - 容错：某个数据集 PREP 不存在则标记 exists=false，不报错中断。
  - 维度探测：peek 第一个 .pt 的真实 tensor shape，与论文契约(192-D)对账，
    不一致会在 JSON 里标 red_flag，便于"维度契约被破坏"这类问题早发现。

用法：
    python log_extraction_manifest.py --data_root $SYNCX_DATA_ROOT \
        --out logs/extraction/manifest.json
"""
import os
import sys
import json
import glob
import platform
from datetime import datetime

# 已知数据集 -> PREP 相对子路径（与 cloud_run_all.sh / prep_*.py 严格对齐）
KNOWN_PREP = [
    ("DeepSpeak_train",        "DeepSpeak_v1_1/clips/prep/train"),
    ("DeepSpeak_test",         "DeepSpeak_v1_1/clips/prep/test"),
    ("VoxCeleb2",              "VoxCeleb2_subset/prep"),
    ("DFE2024",                "Deepfake-Eval-2024/prep"),
    ("HiFi-AVDF",              "HiFi-AVDF/prep"),
    ("FakeAVCeleb",            "FakeAVCeleb_v1.2/prep"),
    ("FakeAVCeleb_b6_trimmed", "FakeAVCeleb_v1.2_split/prep/b6_trimmed"),
    ("FakeAVCeleb_b6_raw",     "FakeAVCeleb_v1.2_split/prep/b6_raw"),
]


def _peek_shapes(pt_path):
    """读第一个 .pt 的真实 tensor shape（audio_seq / v_clip / v_lip）。"""
    try:
        import torch
        d = torch.load(pt_path, map_location="cpu", weights_only=False)
        shapes = {}
        for k in ("audio_seq", "v_clip", "v_lip"):
            if k in d and hasattr(d[k], "shape"):
                shapes[k] = list(d[k].shape)
        return shapes
    except Exception as e:  # 读坏/缺依赖都不应中断聚合
        return {"_error": str(e)[:120]}


def _read_manifest_stats(jsonl_path):
    """读 prep 自带的 manifest.jsonl，统计 real/fake/identity 数。"""
    n_real = n_fake = 0
    ids = set()
    try:
        with open(jsonl_path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                r = json.loads(ln)
                lab = r.get("label", 0)
                if lab == 0:
                    n_real += 1
                else:
                    n_fake += 1
                sid = r.get("identity")
                if sid:
                    ids.add(sid)
    except Exception:
        pass
    return n_real, n_fake, len(ids)


def scan_dataset(name, prep_root):
    entry = {
        "name": name, "prep_root": prep_root, "exists": False,
        "n_clips": 0, "n_real": None, "n_fake": None, "n_identities": None,
        "shapes": None, "manifest_jsonl": "missing", "red_flag": None,
    }
    if not os.path.isdir(prep_root):
        entry["red_flag"] = "prep dir not found"
        return entry
    entry["exists"] = True
    pts = sorted(glob.glob(os.path.join(prep_root, "**", "*.pt"), recursive=True))
    entry["n_clips"] = len(pts)
    # 维度契约对账（论文锁定 64 点 → v_lip 192-D）
    if pts:
        shapes = _peek_shapes(pts[0])
        entry["shapes"] = shapes
        vlip = shapes.get("v_lip")
        if vlip is not None and vlip[-1] != 192:
            entry["red_flag"] = f"v_lip last-dim={vlip[-1]} != 192 (契约破坏!)"
    # 读 prep 自带 manifest（若存在）
    mj = os.path.join(prep_root, "manifest.jsonl")
    if os.path.exists(mj):
        entry["manifest_jsonl"] = "exists"
        nr, nf, ni = _read_manifest_stats(mj)
        entry["n_real"], entry["n_fake"], entry["n_identities"] = nr, nf, ni
    return entry


def main():
    ap = argparse.ArgumentParser(description="Aggregate extraction manifests into one JSON")
    ap.add_argument("--data_root", default=os.environ.get("SYNCX_DATA_ROOT", "/root/autodl-tmp/SyncX/Datasets"))
    ap.add_argument("--out", default="logs/extraction/manifest.json")
    ap.add_argument("--extra", nargs="*", default=[],
                    help="额外 name=path 对，补充 KNOWN_PREP（如新数据集）")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    datasets = []
    total = 0
    for name, rel in KNOWN_PREP:
        entry = scan_dataset(name, os.path.join(args.data_root, rel))
        datasets.append(entry)
        if entry["exists"]:
            total += entry["n_clips"]
    for kv in args.extra:
        if "=" not in kv:
            continue
        name, path = kv.split("=", 1)
        entry = scan_dataset(name, path)
        datasets.append(entry)
        if entry["exists"]:
            total += entry["n_clips"]

    # 环境快照（复现用）
    env = {}
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
    except Exception:
        env["torch"] = "import-failed"
        env["cuda_available"] = False
    env["python"] = platform.python_version()
    env["host"] = platform.node() or "unknown"

    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "data_root": args.data_root,
        "env": env,
        "contract": {
            "lip_lm_dim": 192,            # 64 口周点 × 3 (xyz)
            "v_lip_frames_dim_note": "v_lip shape [-1,192] per frame",
            "v_clip_dim": 512,           # CLIP ViT-B/16 per frame
            "audio_seq_dim": 768,        # wav2vec2-base per frame
        },
        "total_clips": total,
        "datasets": datasets,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 控制台摘要
    flags = [d["name"] for d in datasets if d.get("red_flag")]
    print(f"[extraction-manifest] wrote {args.out}")
    print(f"[extraction-manifest] total clips across present datasets = {total}")
    for d in datasets:
        if d["exists"]:
            print(f"    {d['name']:24s} clips={d['n_clips']:<7d} "
                  f"real={str(d['n_real']):>5} fake={str(d['n_fake']):>5} "
                  f"ids={str(d['n_identities']):>5} shapes={d['shapes']}")
        else:
            print(f"    {d['name']:24s} [ABSENT] {d.get('red_flag')}")
    if flags:
        print(f"[extraction-manifest] ⚠ RED FLAGS: {flags}")


if __name__ == "__main__":
    main()
