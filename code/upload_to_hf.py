#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地 -> 私有 Hugging Face 上传器（断点续传）

为什么需要它：
    云端实例（中国 AutoDL）直接下载某些数据集要么被墙（如 VoxCeleb2 的 YouTube 源），
    要么公开镜像的结构和 prep_*.py 期望的不一致。最稳的做法是：
        1) 你在本机把“本地 D:\\Datasets 下、云端拉不到的库”上传到【自己的私有 HF repo】；
        2) 云端用 cloud_fetch_datasets.py 从你的私有 repo 拉取。
    这样彻底绕开“本地->中国机房”的国际慢链路（本地->HF 是全球 CDN，快）。

安全 / 版权：
    - 默认建【私有】repo，绝不公开。部分数据集（FF++ 等）许可禁止再分发，私有自拉通常可被容忍，
      但请勿公开分享 repo。
    - token 通过环境变量 HF_TOKEN 传入，不要写死在脚本里。

用法：
    pip install -U huggingface_hub hf_transfer
    set HF_TOKEN=hf_你的token
    python upload_to_hf.py --local-dir "D:/Datasets/VoxCeleb2_subset" \
                           --repo-id "你的用户名/syncx-voxceleb2-subset"

中断后续传：重跑同一条命令即可（hf_transfer 后端支持分块续传）。
"""
import argparse
import os
import sys

# 启用 hf_transfer 后端：大文件分块、可断点续传
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import HfApi, upload_folder  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="本地数据集 -> 私有 HF（断点续传）")
    ap.add_argument("--local-dir", required=True, help="本地数据集目录，如 D:/Datasets/VoxCeleb2_subset")
    ap.add_argument("--repo-id", required=True, help="目标 HF dataset repo，如 用户名/syncx-voxceleb2-subset")
    ap.add_argument("--private", action="store_true", default=True, help="建为私有 repo（默认，推荐）")
    ap.add_argument("--public", dest="private", action="store_false", help="改为公开（不推荐，版权风险）")
    ap.add_argument("--endpoint", default="https://huggingface.co", help="HF endpoint（默认官方）")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token（默认读 $HF_TOKEN）")
    args = ap.parse_args()

    if not args.token:
        sys.exit("✗ 未找到 HF_TOKEN，请先：set HF_TOKEN=hf_xxx")
    if not os.path.isdir(args.local_dir):
        sys.exit(f"✗ 本地目录不存在：{args.local_dir}")

    api = HfApi(token=args.token, endpoint=args.endpoint)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    print(f"[upload] {args.local_dir}\n        -> {args.repo_id}  (private={args.private})")
    upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=args.local_dir,
        token=args.token,
        endpoint=args.endpoint,
    )
    print("[upload] 完成 ✓  云端拉取见 cloud_fetch_datasets.py（--only 对应库名）")


if __name__ == "__main__":
    main()
