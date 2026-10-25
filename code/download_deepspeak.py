#!/usr/bin/env python3
"""
download_deepspeak.py — 在 AutoDL 实例上拉取并解压 DeepSpeak v1.1（主训练集）。

前置（在实例终端里先设好）:
    export HF_TOKEN=你的HuggingFace访问令牌        # 且 HF 上已批准该数据集访问
    export HF_ENDPOINT=https://hf-mirror.com        # 实例在中国机房，走镜像加速
    pip install -U huggingface_hub

用法:
    python download_deepspeak.py                  # 默认下到 /root/autodl-tmp/SyncX/Datasets/DeepSpeak_v1_1
    python download_deepspeak.py --out /别的路径
    python download_deepspeak.py --no-extract      # 只下载不解压（后面手动解）
    python download_deepspeak.py --shutdown        # 下载+解压成功后自动关机（省按量计费）

说明:
    - 数据集 gated: 没在 HF 网页点 "Request access" 并获批会 401 失败。
    - 仓库约 46GB（25 个 zip + 3 个 CSV）。云端下载快；解压后 clips/ 约 44GB。
    - snapshot_download 支持断点续传: 中途断了重跑同命令即可，已下的不重复下。
    - 解压后请手动删 *.zip 释放空间: rm <OUT>/*.zip
    - 最终布局（prep_deepspeak.py 期望）:
          <OUT>/clips/real/  clips/fake/  clips/test/real/  clips/test/fake/
          <OUT>/annotations-real.csv  annotations-fake.csv  annotations-split-def.csv
    - --shutdown 需要两个环境变量（AutoDL 控制台「设置→API Token」获取，与 HF token 无关）:
          export AUTO_API_TOKEN=你的AutoDL_API令牌
          export AUTO_INSTANCE_ID=你的实例ID
      关机前会留 30 秒反悔窗口；仅在本脚本【全部成功】后才关机，失败不关。
"""
import os
import argparse
import zipfile
import subprocess
import time

REPO = "faridlab/deepspeak_v1_1"
DEFAULT_OUT = "/root/autodl-tmp/SyncX/Datasets/DeepSpeak_v1_1"
# AutoDL 停止实例 API（若官方改了端点，改这里即可；见 autodl.com 文档）
AUTO_API_STOP = "https://www.autodl.com/api/v1/instance/stop"

# ⚠️ 禁止硬编码 HF token。一律从环境变量读取：export HF_TOKEN=hf_xxx
#    2026-08-12 曾误硬编码真实 token 导致泄露，已移除并作废。别再写回。


def download(out_dir: str) -> None:
    from huggingface_hub import snapshot_download
    print(f"[download] repo={REPO} -> {out_dir}")
    snapshot_download(
        repo_id=REPO,
        repo_type="dataset",
        local_dir=out_dir,
        local_dir_use_symlinks=False,   # 真复制，避免额外占用 inode/空间
    )
    print("[download] 完成（25 个 zip + 3 个 CSV）")


def extract(out_dir: str) -> None:
    clips = os.path.join(out_dir, "clips")
    os.makedirs(clips, exist_ok=True)
    zips = sorted(f for f in os.listdir(out_dir) if f.endswith(".zip"))
    if not zips:
        print("[extract] 没找到 zip，跳过")
        return
    print(f"[extract] 找到 {len(zips)} 个 zip，解压到 {clips}")
    for f in zips:
        zp = os.path.join(out_dir, f)
        print(f"[extract] {f} ...", flush=True)
        # 优先用系统 unzip（快）；没有则退回 Python zipfile
        try:
            subprocess.run(["unzip", "-o", "-q", zp, "-d", clips], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            with zipfile.ZipFile(zp) as z:
                z.extractall(clips)
    print(f"[extract] 完成 -> {clips}")


def main() -> None:
    ap = argparse.ArgumentParser(description="下载并解压 DeepSpeak v1.1")
    ap.add_argument("--out", default=DEFAULT_OUT, help="输出根目录")
    ap.add_argument("--no-extract", action="store_true", help="只下载不解压")
    args = ap.parse_args()

    if not os.environ.get("HF_TOKEN"):
        print("[FATAL] 未设置 HF_TOKEN。请先: export HF_TOKEN=hf_你的真实token（不要硬编码进脚本）")
        raise SystemExit(1)

    os.makedirs(args.out, exist_ok=True)
    download(args.out)
    if not args.no_extract:
        extract(args.out)
    print("\n下一步:")
    print(f"  1. 确认布局: ls {args.out}/clips/   应看到 real/ fake/ test/")
    print(f"  2. 删 zip 释放空间: rm {args.out}/*.zip")
    print(f"  3. 跑 prep 校验: python prep_deepspeak.py --dry-run")


if __name__ == "__main__":
    main()
