"""diag_train_cache.py — 云端：健康检查(cache) 维度契约 + 时间对齐 + 数值健康 + 掉检率
========================================================================================
目的：回答"特征是不是也抽错了"。不改模型、不重训、只读 cache 统计。支持单集深查或多集汇总。

查询维度（每集）：
  1. 维度契约：audio_seq (T_a,768)、v_clip (T_v,512)、v_lip (T_v,192)，type 半精度。
  2. 时序：T_a / T_v 分布；短视频(<100帧)/短音频(<200帧)占比 = 补零样本（短 clip 会被 _fix_len pad）。
  3. 数值健康：NaN/Inf、全零占比（MediaPipe 掉人脸→v_lip 全零；音频无声→audio 全零）。
  4. real/fake 平衡（label 0/1）。

用法（云端，只读 CPU，不占大显存）：
  单集：
    env HF_HOME=/root/autodl-tmp/SyncX/hfcache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        SYNCX_MEDIAPIPE_GPU=0 SYNCX_DEVICE=cpu \
        python diag_train_cache.py \
          --cache-dir /root/autodl-tmp/SyncX/Datasets/DeepSpeak_v1_1/clips/prep/train --max-n 500
  多集列表：
    python diag_train_cache.py \\
        --data-root /root/autodl-tmp/SyncX/Datasets \\
        --sets deepspeak/train hifi dfe2024 fakeavceleb
    
数据根默认 = config.DATA_ROOT(SYNCX_DATA_ROOT)。各集 prep 子路径：
  deepspeak/train : DeepSpeak_v1_1/clips/prep/train
  deepspeak/test  : DeepSpeak_v1_1/clips/prep/test
  hifi            : HiFi-AVDF/prep
  dfe2024         : Deepfake-Eval-2024/prep
  fakeavceleb     : FakeAVCeleb_v1.2_split/prep
"""
import os, sys, argparse, glob
import numpy as np

CONTRACT = {"audio_dim": 768, "clip_dim": 512, "lip_dim": 192}
VIS_LEN, AUDIO_LEN = 100, 200

# 预设集/clip 子路径（相对 data-root）
SET_PATHS = {
    "deepspeak/train": "DeepSpeak_v1_1/clips/prep/train",
    "deepspeak/test":  "DeepSpeak_v1_1/clips/prep/test",
    "hifi":            "HiFi-AVDF/prep",
    "dfe2024":         "Deepfake-Eval-2024/prep",
    "fakeavceleb":     "FakeAVCeleb_v1.2_split/prep",
}


def scan_one(cache_dir, max_n):
    import torch
    flat = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
    nested = sorted(glob.glob(os.path.join(cache_dir, "**", "cache.pt"), recursive=True))
    files = sorted(set([f for f in flat + nested if not f.endswith(".tmp")]))
    if max_n and max_n > 0:
        files = files[:max_n]
    n = len(files)
    print(f"\n----- {cache_dir} -----")
    print(f"  cache 文件 = {n}  (flat .pt={len(flat)} / nested cache.pt={len(nested)})")
    if not files:
        print("  无 cache 文件（路径不对或尚未抽取）。")
        return

    bad_contract = 0; n_short_video = 0; n_short_audio = 0; n_nan_or_inf = 0
    labels = []; Ta_all = []; Tv_all = []
    agg = {"audio": [0, 0, 0, 0.0], "v_clip": [0, 0, 0, 0.0], "v_lip": [0, 0, 0, 0.0]}
    examples = []
    for f in files:
        try:
            d = torch.load(f, map_location="cpu", weights_only=True)
        except Exception as e:
            examples.append((os.path.basename(f), f"load fail: {str(e)[:60]}")); bad_contract += 1; continue
        if "audio_seq" not in d or "v_clip" not in d or "v_lip" not in d:
            examples.append((os.path.basename(f), "missing key")); bad_contract += 1; continue
        a = d["audio_seq"]; vc = d["v_clip"]; vl = d["v_lip"]
        if (a.size(-1) != CONTRACT["audio_dim"] or vc.size(-1) != CONTRACT["clip_dim"]
                or vl.size(-1) != CONTRACT["lip_dim"]):
            bad_contract += 1
            examples.append((os.path.basename(f), f"DIM a{tuple(a.shape)} vc{tuple(vc.shape)} vl{tuple(vl.shape)}"))
        Ta = a.size(0); Tv = vc.size(0)
        Ta_all.append(Ta); Tv_all.append(Tv)
        if Tv < VIS_LEN: n_short_video += 1
        if Ta < AUDIO_LEN: n_short_audio += 1
        if "label" in d: labels.append(int(d["label"]))
        an = np.asarray(a, dtype=np.float32); vn = np.asarray(vc, dtype=np.float32); ln = np.asarray(vl, dtype=np.float32)
        if np.isnan(an).any() or np.isinf(an).any() or np.isnan(vn).any() or np.isinf(vn).any() \
                or np.isnan(ln).any() or np.isinf(ln).any():
            n_nan_or_inf += 1; examples.append((os.path.basename(f), "NaN/Inf"))
        ag = agg["audio"]; ag[0] += int(np.isnan(an).sum()); ag[1] += int(np.isinf(an).sum()); ag[2] += int((an == 0).sum()); ag[3] += an.size
        ag = agg["v_clip"]; ag[0] += int(np.isnan(vn).sum()); ag[1] += int(np.isinf(vn).sum()); ag[2] += int((vn == 0).sum()); ag[3] += vn.size
        ag = agg["v_lip"];  ag[0] += int(np.isnan(ln).sum()); ag[1] += int(np.isinf(ln).sum()); ag[2] += int((ln == 0).sum()); ag[3] += ln.size

    Ta = np.array(Ta_all); Tv = np.array(Tv_all)
    print(f"  [契约] 维度不匹配 = {bad_contract}")
    print(f"  [时序] T_a med/max = {int(np.median(Ta))}/{Ta.max()}   T_v med/max = {int(np.median(Tv))}/{Tv.max()}")
    print(f"  [补零] 短视频(<{VIS_LEN}f)={n_short_video} ({100.0*n_short_video/n:.1f}%)  "
          f"短音频(<{AUDIO_LEN}f)={n_short_audio} ({100.0*n_short_audio/n:.1f}%)")
    print(f"  [数值] NaN/Inf 样本={n_nan_or_inf}")
    for k in ("audio", "v_clip", "v_lip"):
        nan, inf, zero, tot = agg[k]
        print(f"    {k:8s}: NaN={nan} Inf={inf} 全零占比={zero/tot if tot else 0:.4f}")
    if labels:
        l = np.array(labels)
        print(f"  [标签] real(0)={int((l==0).sum())}  fake(1)={int((l==1).sum())}  n={len(l)}")
    if examples:
        print(f"  ⚠️ 异常样本示例({len(examples)}):")
        for p, why in examples[:8]:
            print(f"    - {p}: {why}")
    # 结论速签
    ok = bad_contract == 0 and n_nan_or_inf == 0
    print(f"  => {'✅ 健康' if ok else '⚠️ 有异常（见上）'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="", help="单集模式：直接指定 cache 目录")
    ap.add_argument("--data-root", default=None, help="多集模式：数据集根（缺省=config.DATA_ROOT）")
    ap.add_argument("--sets", nargs="+", default=None, help="多集模式：要测的集名(见 SET_PATHS)")
    ap.add_argument("--max-n", type=int, default=0, help="每集抽样上限（0=全扫，先 500 快出数）")
    args = ap.parse_args()

    if args.cache_dir:
        scan_one(args.cache_dir, args.max_n)
        return
    if args.data_root is None:
        try:
            from config import DATA_ROOT
            args.data_root = DATA_ROOT
        except Exception:
            pass
    if not args.data_root:
        print("需指定 --cache-dir 或 --data-root(多集)")
        return
    sets = args.sets or list(SET_PATHS.keys())
    for s in sets:
        if s not in SET_PATHS:
            print(f"\n未知集名: {s}（可选: {list(SET_PATHS.keys())}）")
            continue
        scan_one(os.path.join(args.data_root, SET_PATHS[s]), args.max_n)


if __name__ == "__main__":
    main()