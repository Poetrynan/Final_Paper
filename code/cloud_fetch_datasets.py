#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
云端数据集拉取 + 自动解压编排器（在 AutoDL 实例内运行）

把所有数据落到 SYNCX_DATA_ROOT（默认 /root/autodl-tmp/SyncX/Datasets），
目录结构与本地 D:\\Datasets 完全一致 → prep_*.py 无需改动即可直接跑。

权限分类（access 字段）：
  open            : 公开数据集（MIT/Apache 等），云端直接拉，零上传、零 HF 存储
  gated_approved  : HF gated 数据集，你已在网页 Request access 并获批（需 HF_TOKEN）
  relay           : 官方需填表/审批才能下载，但你【本地 D:\\Datasets 已有副本】
                    → 用本机 upload_to_hf.py 传到你的【私有】HF repo，云端再拉
                      （无需重新向第三方申请；FF++/FakeAVCeleb 绝不可公开）

用法（无卡模式开机，省钱）：
    export SYNCX_DATA_ROOT=/root/autodl-tmp/SyncX/Datasets
    export HF_TOKEN=hf_你的token          # 拉 gated/私有 repo 时需要
    pip install -U huggingface_hub modelscope hf_transfer
    python cloud_fetch_datasets.py --dry-run          # 先预览：将拉什么 + 权限状态
    python cloud_fetch_datasets.py --all              # 下载 + 自动解压，一步到位
    python cloud_fetch_datasets.py --only VoxCeleb2_subset FF++   # 只拉指定库
    python cloud_fetch_datasets.py --all --shutdown   # 拉完解压后自动关机（省按量计费）

默认下载后自动解压（zip/tar/tar.gz/tgz）并删除压缩包以释放 550G 数据盘；
若想保留压缩包加 --keep-zip。
"""
import argparse
import os
import subprocess
import sys
import glob

HF_MIRROR = "https://hf-mirror.com"
DATA_ROOT = os.environ.get("SYNCX_DATA_ROOT", "/root/autodl-tmp/SyncX/Datasets")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# 把 <你的用户名> 替换成你的 HF 账号。relay 类库需先用本机 upload_to_hf.py 上传到对应私有 repo。
# 注意：Deepfake-Eval-2024 / FakeAVCeleb_v1.2 / HiFi-AVDF 这 3 个测评集已统一塞进
#       Lucifer696/Thesis 一个私有仓（按文件名上传了 zip）。
#       它们应改用同目录的 cloud_download_eval_datasets.py 拉取 + 解压，
#       本脚本用 snapshot_download 拉整个仓不适合这种混合仓。
PLAN = {
    # —— 公开直拉（无需申请、不占 HF 存储）——
    "TalkingHeadBench":   dict(kind="hf_public",  repo="luchaoqi/TalkingHeadBench", access="open"),
    "MVAD":               dict(kind="modelscope", repo="MengxueBoBo/MVAD",         access="open"),

    # —— 官方需申请，但你本地已有副本 → 走私有 HF 中转（无需重新申请第三方）——
    "VoxCeleb2_subset":   dict(kind="hf_private", repo="Lucifer696/syncx-voxceleb2-subset", access="relay"),
    "Celeb-DF-v2":        dict(kind="hf_private", repo="Lucifer696/syncx-celebdfv2",       access="relay"),
    "FF++":               dict(kind="hf_private", repo="Lucifer696/syncx-ffpp",            access="relay"),

    # 以下 3 个测评集已合并到 Lucifer696/Thesis，请用 cloud_download_eval_datasets.py 下载：
    # "Deepfake-Eval-2024"  -> Lucifer696/Thesis/Deepfake-Eval-2024.zip
    # "FakeAVCeleb_v1.2"    -> Lucifer696/Thesis/FakeAVCeleb_v1.2.zip
    # "HiFi-AVDF"           -> Lucifer696/Thesis/HiFi-AVDF.zip
}


def _pull_hf(name, repo, private, mirror):
    from huggingface_hub import snapshot_download
    endpoint = HF_MIRROR if mirror else "https://huggingface.co"
    local_dir = os.path.join(DATA_ROOT, name)
    print(f"[hf] {name}: {repo}  ->  {local_dir}  (endpoint={endpoint})")
    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        token=HF_TOKEN if private else None,
        endpoint=endpoint,
    )
    print(f"[hf] {name} 下载完成 ✓")


def _pull_modelscope(name, repo):
    try:
        from modelscope import snapshot_download
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "modelscope"])
        from modelscope import snapshot_download
    local_dir = os.path.join(DATA_ROOT, name)
    print(f"[modelscope] {name}: {repo}  ->  {local_dir}")
    snapshot_download(repo_id=repo, local_dir=local_dir)
    print(f"[modelscope] {name} 下载完成 ✓")


def _extract_after_pull(name, keep_zip):
    """下载后扫描压缩包并解压（best-effort），可选删原包释放空间。"""
    base = os.path.join(DATA_ROOT, name)
    if not os.path.isdir(base):
        return
    archives = []
    for pat in ("*.zip", "*.tar", "*.tar.gz", "*.tgz"):
        archives += glob.glob(os.path.join(base, "**", pat), recursive=True)
    if not archives:
        print(f"[extract] {name}: 无压缩包，跳过（已是解压态）")
        return
    print(f"[extract] {name}: 发现 {len(archives)} 个压缩包，开始解压...")
    for a in sorted(set(archives)):
        d = os.path.dirname(a)
        try:
            if a.endswith(".zip"):
                subprocess.run(["unzip", "-o", "-q", a, "-d", d], check=True)
            else:
                subprocess.run(["tar", "-xf", a, "-C", d], check=True)
            print(f"[extract]   ✓ {os.path.basename(a)}")
            if not keep_zip:
                os.remove(a)
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"[extract]   ✗ {os.path.basename(a)} 失败: {e}")
    print(f"[extract] {name} 解压完成" + ("" if keep_zip else "（压缩包已删以释放空间）"))


def pull(name, keep_zip):
    spec = PLAN[name]
    kind = spec["kind"]
    if kind == "hf_private":
        if not HF_TOKEN:
            sys.exit(f"✗ 拉私有 repo {name} 需要 HF_TOKEN，先 export HF_TOKEN=hf_xxx")
        _pull_hf(name, spec["repo"], private=True, mirror=spec.get("mirror", False))
    elif kind == "hf_public":
        _pull_hf(name, spec["repo"], private=False, mirror=spec.get("mirror", True))
    elif kind == "modelscope":
        _pull_modelscope(name, spec["repo"])
    else:
        sys.exit(f"✗ 未知 kind: {kind}")
    _extract_after_pull(name, keep_zip)


def main():
    ap = argparse.ArgumentParser(description="云端数据集拉取 + 自动解压编排器")
    ap.add_argument("--all", action="store_true", help="拉取 PLAN 中全部")
    ap.add_argument("--only", nargs="+", help="只拉指定库名，如 --only VoxCeleb2_subset FF++")
    ap.add_argument("--dry-run", action="store_true", help="只预览，不下载")
    ap.add_argument("--keep-zip", action="store_true", help="解压后保留压缩包")
    ap.add_argument("--shutdown", action="store_true", help="全部完成后自动关机（省按量计费）")
    args = ap.parse_args()

    if not args.all and not args.only:
        ap.error("必须指定 --all 或 --only")

    targets = list(PLAN.keys()) if args.all else args.only
    unknown = [t for t in targets if t not in PLAN]
    if unknown:
        sys.exit(f"✗ PLAN 中不存在：{unknown}，可选项：{list(PLAN.keys())}")

    placeholder = [t for t in targets if "<你的用户名>" in PLAN[t].get("repo", "")]
    if placeholder:
        sys.exit(
            "✗ 以下库的 repo 还含 <你的用户名> 占位，请先在脚本里替换成你的 HF 账号：\n"
            + "  " + "\n  ".join(f"{t} -> {PLAN[t]['repo']}" for t in placeholder)
        )

    print(f"数据根目录 DATA_ROOT = {DATA_ROOT}\n")
    if args.dry_run:
        print("==== DRY-RUN：将拉取以下库（不实际下载）====")
        for t in targets:
            s = PLAN[t]
            print(f"  - {t:24} access={s['access']:14} kind={s['kind']:11} repo={s['repo']}")
        print("\nrelay 类库需确保本机已用 upload_to_hf.py 上传到对应私有 repo。")
        return

    os.makedirs(DATA_ROOT, exist_ok=True)
    ok, fail = [], []
    for name in targets:
        try:
            pull(name, args.keep_zip)
            ok.append(name)
        except Exception as e:  # noqa: BLE001
            print(f"[!] {name} 失败: {e}")
            fail.append(name)
    print("\n==== 拉取汇总 ====")
    print(f"成功({len(ok)}): {ok}")
    print(f"失败({len(fail)}): {fail}")
    if fail:
        print("失败项可单独重跑：python cloud_fetch_datasets.py --only " + " ".join(fail))

    if args.shutdown and not fail:
        print("\n[shutdown] 全部成功，执行 /usr/bin/shutdown ...")
        subprocess.run(["/usr/bin/shutdown"])
    elif args.shutdown and fail:
        print("\n[shutdown] 存在失败项，未自动关机，请人工排查。")


if __name__ == "__main__":
    main()
