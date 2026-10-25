#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
云端测评集一键拉取 + 解压（AutoDL 专用）

从 HF 私有仓 Lucifer696/Thesis 按文件名下载 3 个测评集 zip，
解压到 /root/autodl-tmp/SyncX/Datasets/ 下，与训练集目录并列：
    Deepfake-Eval-2024
    FakeAVCeleb_v1.2
    HiFi-AVDF

用法（在 AutoDL 实例里运行，建议无卡模式开机省钱）：
    source /etc/network_turbo                 # 境外源提速，只对当前终端生效
    export HF_TOKEN=hf_你的token              # 私有仓需要 Write/Read token
    pip install -U huggingface_hub hf_transfer
    python cloud_download_eval_datasets.py    # 下载 + 解压 + 删 zip
    python cloud_download_eval_datasets.py --keep-zip   # 保留压缩包

前置条件：
    本机需先把 3 个 zip 上传到 HF 私有仓 Lucifer696/Thesis：
        Deepfake-Eval-2024.zip
        FakeAVCeleb_v1.2.zip
        HiFi-AVDF.zip
    上传脚本见同目录 HF_数据集上传流程.md §0。
"""
import argparse
import os
import subprocess
import sys
import zipfile

REPO_ID = "Lucifer696/Thesis"
REPO_TYPE = "dataset"

# 测评集文件名 -> 解压后目录名
EVAL_FILES = {
    "Deepfake-Eval-2024.zip": "Deepfake-Eval-2024",
    "FakeAVCeleb_v1.2.zip":   "FakeAVCeleb_v1.2",
    "HiFi-AVDF.zip":          "HiFi-AVDF",
}

DATA_ROOT = os.environ.get("SYNCX_DATA_ROOT", "/root/autodl-tmp/SyncX/Datasets")


def _check_space(path, need_gb=80):
    """粗略检查目标盘剩余空间（GB）。"""
    try:
        st = os.statvfs(path)
        free_gb = st.f_bavail * st.f_frsize / (1024 ** 3)
        print(f"[space] {path} 剩余 {free_gb:.1f} GB")
        if free_gb < need_gb:
            print(f"[warn] 剩余空间不足 {need_gb} GB，请清理数据盘！")
    except Exception as e:
        print(f"[warn] 无法检查磁盘空间: {e}")


def _download(zip_name: str, out_dir: str):
    from huggingface_hub import hf_hub_download
    print(f"[download] {zip_name}  ->  {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    # hf_hub_download 会把文件放到 HF 缓存；我们显式指定 local_dir 直接落到目标目录
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename=zip_name,
        local_dir=out_dir,
        local_dir_use_symlinks=False,
    )
    print(f"[download] ✓ 已下载 {local_path}")
    return local_path


def _extract(zip_path: str, out_dir: str, keep_zip: bool):
    print(f"[extract] {os.path.basename(zip_path)}  ->  {out_dir}")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(out_dir)
    print(f"[extract] ✓ 完成")
    if not keep_zip:
        os.remove(zip_path)
        print(f"[extract] 已删除 {os.path.basename(zip_path)} 释放空间")


def main():
    ap = argparse.ArgumentParser(description="云端测评集一键拉取 + 解压")
    ap.add_argument("--keep-zip", action="store_true", help="解压后保留压缩包")
    ap.add_argument("--dry-run", action="store_true", help="只检查/列文件，不下载")
    args = ap.parse_args()

    if "HF_TOKEN" not in os.environ:
        sys.exit("✗ 请先 export HF_TOKEN=hf_你的token（拉私有仓需要）")

    os.makedirs(DATA_ROOT, exist_ok=True)
    _check_space(DATA_ROOT)

    if args.dry_run:
        from huggingface_hub import list_repo_files
        print(f"[dry-run] 检查 {REPO_ID} 中的文件：")
        files = list_repo_files(REPO_ID, repo_type=REPO_TYPE)
        for zf in EVAL_FILES:
            status = "✓ 存在" if zf in files else "✗ 缺失"
            print(f"  {status}: {zf}")
        print(f"[dry-run] 目标目录: {DATA_ROOT}")
        missing = [z for z in EVAL_FILES if z not in files]
        if missing:
            print(f"[dry-run] 缺失文件需本机补传到 {REPO_ID}: {missing}")
        return

    ok, fail = [], []
    for zip_name, dir_name in EVAL_FILES.items():
        out_dir = os.path.join(DATA_ROOT, dir_name)
        try:
            zip_path = _download(zip_name, out_dir)
            _extract(zip_path, out_dir, args.keep_zip)
            ok.append(dir_name)
        except Exception as e:
            print(f"[!] {zip_name} 失败: {e}")
            fail.append(zip_name)

    print("\n==== 拉取汇总 ====")
    print(f"成功({len(ok)}): {ok}")
    print(f"失败({len(fail)}): {fail}")
    if fail:
        print("失败项请排查：1) HF_TOKEN 权限；2) 文件是否已上传到 Lucifer696/Thesis；3) 磁盘空间。")


if __name__ == "__main__":
    main()
