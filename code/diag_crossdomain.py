#!/usr/bin/env python
"""
diag_crossdomain.py — Sync-X 跨域崩塌「先诊断」脚本
================================================================
目的：在修改任何代码/超参之前，用**可实测**的手段回答三个问题：
  1. 数据重叠/泄漏：评测集与训练集是否有身份 or 样本重叠，预处理是否一致？
  2. 类别不平衡：各评测集 real/fake 分布（解释 FakeAVCeleb AP/ACC 虚高）。
  3. 身份捷径嫌疑：训练集里「一个身份是否只对应一种标签」——若身份几乎等价于
     标签，模型完全可以"背身份"而非学通用伪造线索，这正是跨域崩溃的根因之一。

两种运行模式：
  meta  — 纯元数据，不加载模型、不占 GPU、秒级出结果（读 .syncx_index.jsonl 缓存）。
         推荐先跑这个。
  model — 加载已训权重前向，按 identity 分组在"同身份内含 real+fake"的子集上
         算组内 AUC。较慢（逐样本单卡前向），建议先在 HiFi-AVDF / DFE2024 上跑。

用法示例（云端）：
  python diag_crossdomain.py meta \
      --train-cache /root/autodl-tmp/SyncX/Datasets/DeepSpeak_v1_1/clips/prep \
      --eval-caches "FakeAVCeleb=/root/autodl-tmp/SyncX/Datasets/FakeAVCeleb_v1.2/prep" \
                     "DFE2024=/root/autodl-tmp/SyncX/Datasets/Deepfake-Eval-2024/prep" \
                     "HiFi=/root/autodl-tmp/SyncX/Datasets/HiFi-AVDF/prep"

  python diag_crossdomain.py model \
      --ckpt /root/autodl-tmp/SyncX/persist/ckpt/syncx_best_seed0.pt \
      --eval-cache /root/autodl-tmp/SyncX/Datasets/HiFi-AVDF/prep \
      --min-per-id 4
"""
import os, sys, glob, json, argparse
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch


def _load_index(cache_dir):
    """复用 dataset 的索引缓存；必要时重建（仅首次较慢）。"""
    from dataset import _load_or_build_index
    files = sorted(glob.glob(os.path.join(cache_dir, "**", "*.pt"), recursive=True))
    index = _load_or_build_index(files, cache_dir)
    return index


def _summarize(index, label="cache"):
    """对一个数据集的 idx 数组做统计。返回 dict（纯计数，不算模型）。"""
    from collections import Counter
    idc = Counter(ix["identity"] for ix in index)
    src = Counter(ix["source_id"] for ix in index)
    lab = Counter(ix["label"] for ix in index)
    n_bad = sum(1 for ix in index if ix["label"] < 0)
    return {
        "n_total": len(index),
        "n_bad_label": n_bad,
        "n_real": lab.get(0, 0),
        "n_fake": lab.get(1, 0),
        "n_unique_identity": len(idc),
        "n_empty_identity": idc.get("", 0),
        "n_unique_source": len(src),
        "n_empty_source": src.get("", 0),
    }


def _identity_label_mapping(index):
    """身份→该身份出现的标签集合。判断身份捷径的关键。"""
    m = {}
    for ix in index:
        if ix["label"] < 0:
            continue
        m.setdefault(ix["identity"], set()).add(ix["label"])
    return m


def run_meta(train_cache, eval_caches):
    result = {}

    print("===== 训练集（DeepSpeak）=====")
    tri = _load_index(train_cache)
    s = _summarize(tri)
    result["train"] = s
    print(json.dumps(s, ensure_ascii=False, indent=2))
    # 身份→标签 独态分析（身份捷径核心判据）
    train_map = _identity_label_mapping(tri)
    monosome = {k for k, v in train_map.items() if len(v) == 1}
    mixed = {k for k, v in train_map.items() if len(v) > 1}
    print(f"身份总数={len(train_map)}  单一标签身份(身份≈标签)={len(monosome)}  "
          f"同身份含real+fake(混合)={len(mixed)}")
    result["train"]["n_mono_identity_label"] = len(monosome)
    result["train"]["n_mixed_identity"] = len(mixed)
    print("  → 若「单一标签身份」占比极高（如 >90%），模型存在强身份捷径嫌疑："
          "把它认出来就近乎等于给出标签。\n")

    train_ids = set(train_map.keys())
    train_paths = set(ix["path"] for ix in tri)

    for name, cp in eval_caches.items():
        print(f"===== 评测集 {name} =====")
        ei = _load_index(cp)
        es = _summarize(ei)
        result[name] = es
        print(json.dumps(es, ensure_ascii=False, indent=2))
        e_ids = set(ix["identity"] for ix in ei)
        e_paths = set(ix["path"] for ix in ei)
        id_overlap = train_ids & e_ids
        path_overlap = train_paths & e_paths
        print(f"与训练集身份交集 size={len(id_overlap)}")
        print(f"与训练集样本路径交集 size={len(path_overlap)}")
        if len(id_overlap) or len(path_overlap):
            print("  ⚠️ 发现重叠 → 疑似数据泄漏/划分重叠，需深入核查！")
        else:
            print("  ✅ 无身份/路径重叠（预期，说明不是训练集混入导致。）")
        # 预处理一致性核查：样本维度（该集首样本）
        first = next((ix for ix in ei), None)
        if first:
            try:
                meta = torch.load(first["path"], map_location="cpu", weights_only=True)
                dims = {k: list(meta[k].shape) for k in ("audio_seq", "v_clip", "v_lip")}
                print(f"  首样本形状(应=T_a200/T_v100): {dims}")
                result[name]["sample_dims"] = dims
            except Exception as ex:
                print(f"  首样本读取失败: {ex}")
        result[name]["identity_overlap_with_train"] = len(id_overlap)
        result[name]["path_overlap_with_train"] = len(path_overlap)
        print()

    print("===== 小结（决策依据）=====")
    for k in ("train", *eval_caches.keys()):
        r = result[k]
        print(f"{k}: n={r['n_total']} real={r['n_real']} fake={r['n_fake']} "
              f"身份={r['n_unique_identity']} (身份双向={r.get('n_mixed_identity','-')})")
    return result


def run_model(ckpt, eval_cache, min_per_id):
    import argparse as _a
    from config import get_config
    from models import SyncX
    from dataset import FeatureCacheDataset
    from utils import all_metrics

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  ckpt={ckpt}")
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    cfg = get_config(**{k: v for k, v in (ck.get("cfg") or {}).items()
                        if k in get_config().__dict__})
    model = SyncX(cfg).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    ds = FeatureCacheDataset(eval_cache, T=cfg.T, real_only=False)
    print(f"eval_cache={eval_cache}  n={len(ds)}")
    if len(ds) == 0:
        print("!! 无有效样本，退出。"); return
    if len(ds) > 20000:
        print("!! 样本 >2 万，逐样本单卡前向会很久；建议换小集(HiFi/DFE)或缩减后重跑。")

    groups = {}  # identity -> {"y": [], "s": []}
    with torch.no_grad():
        import tqdm as _tqdm
        for i in _tqdm.tqdm(range(len(ds)), desc="per-sample"):
            item = ds[i]
            x = {k: v.unsqueeze(0).to(dev) for k, v in item.items()
                 if isinstance(v, torch.Tensor)}
            out = model(x)
            prob = torch.softmax(out["logits"].float(), dim=1)[0, 1].item()
            ident = item["identity"] or "(empty)"
            groups.setdefault(ident, {"y": [], "s": []})
            groups[ident]["y"].append(int(item["label"]))
            groups[ident]["s"].append(prob)

    print(f"\n身份总数={len(groups)}  阈值 min_per_id≥{min_per_id}\n")
    print(f"{'identity':<24}{'n_real':>6}{'n_fake':>6}{'组内AUC':>10}  判定")
    usable = 0
    for ident, g in sorted(groups.items(), key=lambda kv: - (sum(1 for y in kv[1]['y'] if y == 1) + sum(1 for y in kv[1]['y'] if y == 0))):
        y = g["y"]; s = g["s"]
        n0 = sum(1 for v in y if v == 0); n1 = sum(1 for v in y if v == 1)
        if n0 >= min_per_id and n1 >= min_per_id:
            m = all_metrics(torch.tensor(y), torch.tensor(s))
            auc = m["AUC"]
            usable += 1
            verdict = "组内可分(仍学得到lowe伪造克制)" if auc >= 0.6 else "组内接近随机(学不到泛化线索)"
            print(f"{ident[:24]:<24}{n0:>6}{n1:>6}{auc:>10.3f}  {verdict}")
        else:
            print(f"{ident[:24]:<24}{n0:>6}{n1:>6}{'--':>10}  样本不足")
    print(f"\n可用组(双标签且≥{min_per_id})={usable}")
    print("解读：若训练同身份内 AUC ≈ 随机，说明在排除了'背身份'后，模型区分真假的能力")
    print("本身就很弱——即它靠的确实是身份/域指纹，而非通用伪造信号。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("meta")
    p.add_argument("--train-cache", required=True)
    p.add_argument("--eval-caches", nargs="+", required=True,
                   help='形如 "Name=/path/prep" 的键值对，可多个')

    m = sub.add_parser("model")
    m.add_argument("--ckpt", required=True)
    m.add_argument("--eval-cache", required=True)
    m.add_argument("--min-per-id", type=int, default=4)
    args = ap.parse_args()

    if args.mode == "meta":
        ev = {}
        for kv in args.eval_caches:
            name, _, path = kv.partition("=")
            if not name or not path:
                raise SystemExit(f"eval-caches 项格式应为 Name=/path，收到: {kv!r}")
            ev[name] = path
        run_meta(args.train_cache, ev)
    else:
        run_model(args.ckpt, args.eval_cache, args.min_per_id)