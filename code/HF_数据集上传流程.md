# Sync-X 数据集上传 Hugging Face 流程（已验证可用）

> 用途：把本机 `D:\MVP\*.zip`（或 `D:\Datasets\*` 下需要中转的数据集）传到 HF 私有仓，
> 让中国机房的 AutoDL 实例用 `cloud_fetch_datasets.py` 拉取（绕过本地→中国机房的国际慢链路）。
> 本流程于 2026-08-16 实测跑通。

---

## 0. TL;DR（直接抄这条）

### 0.1 上传 3 个测评集 zip 到 HF（本机 Windows cmd / Git Bash）

```bash
export HF_TOKEN=hf_你的WriteToken
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()                       # 自动读环境变量 HF_TOKEN（优先级高于本地缓存）
repo = "Lucifer696/Thesis"         # 已存在的私有仓，直接传进去，不用新建
files = {
    r"D:/MVP/Deepfake-Eval-2024.zip": "Deepfake-Eval-2024.zip",
    r"D:/MVP/FakeAVCeleb_v1.2.zip":   "FakeAVCeleb_v1.2.zip",
    r"D:/MVP/HiFi-AVDF.zip":          "HiFi-AVDF.zip",
}
for local, remote in files.items():
    api.upload_file(path_or_fileobj=local, path_in_repo=remote,
                    repo_id=repo, repo_type="dataset")
    print("✓ 已上传", remote, "->", repo)
PY
```

### 0.2 云端下载 + 解压（AutoDL 实例内运行）

> ⚠️ **私有仓只能走官方站 + token，hf-mirror 镜像拉不到（404）。**
> ⚠️ **大文件（>5GB）别用本节 `hf_hub_download` 单次长连接下**——AutoDL 代理会掐断长连接（反复 `peer closed connection`），直连官方站又 `Cannot assign requested address` 连不上。大 zip 一律走 §0.3 的 `wget -c` 断点续传。本节写法只对**小文件/模型权重**可靠（已联网验证 2026-08-17）。
> - **去掉已废弃的 `HF_HUB_ENABLE_HF_TRANSFER=1`**（HF 已全面切 XET，hf_transfer 不再用于下载）。
> - **必须 `HF_HUB_DISABLE_XET=1`**：XET 后端对 >2GB 大文件有卡死 bug，关掉降级到标准 HTTPS 最稳。
> - 加 `HF_HUB_DOWNLOAD_TIMEOUT=120` + `HF_HUB_ETAG_TIMEOUT=900` 扛抖动。

```bash
tmux new -s hf                                   # 长任务必须进 tmux，关窗不死
# —— 进入 tmux 后粘贴下面 ——
source /etc/network_turbo                        # 仅当前终端生效；私有仓只能靠它/直连加速
export HF_HUB_DISABLE_XET=1                      # 关 XET，避免大文件卡死
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_ETAG_TIMEOUT=900
export HF_TOKEN=hf_你的token                     # 读私有仓需要
export SYNCX_DATA_ROOT=/root/autodl-tmp/SyncX/Datasets
mkdir -p "$SYNCX_DATA_ROOT"
pip install -U huggingface_hub
python - <<'PY'
import os, zipfile, shutil
from huggingface_hub import hf_hub_download
REPO="Lucifer696/Thesis"; ROOT=os.environ["SYNCX_DATA_ROOT"]
files={"Deepfake-Eval-2024.zip":"Deepfake-Eval-2024",
       "FakeAVCeleb_v1.2.zip":"FakeAVCeleb_v1.2",
       "HiFi-AVDF.zip":"HiFi-AVDF"}
def extract_flat(zp, td):
    tmp=zp+".ex"; os.makedirs(tmp, exist_ok=True)
    with zipfile.ZipFile(zp) as z: z.extractall(tmp)
    items=os.listdir(tmp)
    if len(items)==1 and os.path.isdir(os.path.join(tmp,items[0])):
        inner=os.path.join(tmp,items[0])
        for f in os.listdir(inner): shutil.move(os.path.join(inner,f),os.path.join(td,f))
        os.rmdir(inner)
    else:
        for f in items: shutil.move(os.path.join(tmp,f),os.path.join(td,f))
    os.rmdir(tmp)
for zn,dn in files.items():
    td=os.path.join(ROOT,dn); os.makedirs(td,exist_ok=True)
    print(">>> 下载",zn)
    zp=hf_hub_download(repo_id=REPO, repo_type="dataset", filename=zn, local_dir=td)
    print(">>> 解压",zn)
    extract_flat(zp,td); os.remove(zp); print(">>> ✓",dn)
print("=== 全部完成 ===")
print("目录结构：")
for d in sorted(os.listdir(ROOT)): print("  "+os.path.join(ROOT,d))
PY
# —— 跑起来后 Ctrl+B 再按 D 脱离，窗口随便关 ——
```

解压后目录结构：
```
/root/autodl-tmp/SyncX/Datasets/
├── DeepSpeak_v1_1
├── VoxCeleb2_subset
├── Deepfake-Eval-2024
├── FakeAVCeleb_v1.2
└── HiFi-AVDF
```

### 0.3 大文件保底：`wget -c` 断点续传（私有仓用，会很慢）

> **7GB 以上的 zip 不要用 `hf_hub_download` 单次长连接下**——AutoDL 代理会掐断长连接，反复 `peer closed connection`；而直连官方站又经常无响应。私有仓只能用 **保留代理 + `wget -c`（断点续传）**：`wget` 被掐断后会自动重连、从已下部分继续，最终能磨完。
>
> ⚠️ **实测 2026-08-17：私有仓这样下非常慢**（Deepfake-Eval-2024.zip 被反复掐断，速度 9–80 KB/s，FakeAVCeleb ETA 10h+）。原因是官方站已全面切 XET CDN，resolve 链接会被 302 到 `us.aws.cdn.hf.co/xet-bridge-us/...`，`HF_HUB_DISABLE_XET=1` 对 `wget` 直链无效；加上 AutoDL 代理长连接不稳定，导致频繁 `Connection closed ... Retrying`。所以 §0.3 是**保底方案**，能下完但体验差；**想快见 §0.4（临时转公开 + hf-mirror）**。

```bash
source /etc/network_turbo
export HF_TOKEN=hf_你的token
export SYNCX_DATA_ROOT=/root/autodl-tmp/SyncX/Datasets
mkdir -p "$SYNCX_DATA_ROOT" && cd "$SYNCX_DATA_ROOT"
set -euo pipefail

download_one() {
  local fn=$1
  local dn=$2
  mkdir -p "$dn" && cd "$dn"
  echo ">>> 下载 $fn"
  wget -c --header="Authorization: Bearer $HF_TOKEN" \
    "https://huggingface.co/datasets/Lucifer696/Thesis/resolve/main/$fn"
  echo ">>> 校验 $fn"
  python - <<PY
import zipfile, sys
try:
    with zipfile.ZipFile('$fn') as z:
        bad = z.testzip()
        if bad:
            print(f"ERROR: zip corrupt, first bad file: {bad}"); sys.exit(1)
except Exception as e:
    print(f"ERROR: not a valid zip: {e}"); sys.exit(1)
print("OK: zip valid")
PY
  echo ">>> 解压 $fn"
  python - <<PY
import zipfile, os, shutil
zp='$fn'; td='.'; tmp='extract_tmp'
with zipfile.ZipFile(zp) as z: z.extractall(tmp)
items=os.listdir(tmp)
if len(items)==1 and os.path.isdir(os.path.join(tmp,items[0])):
    for f in os.listdir(os.path.join(tmp,items[0])):
        shutil.move(os.path.join(tmp,items[0],f), td)
else:
    for f in items: shutil.move(os.path.join(tmp,f), td)
shutil.rmtree(tmp); os.remove(zp)
PY
  cd ..
}

download_one "Deepfake-Eval-2024.zip" "Deepfake-Eval-2024"
download_one "FakeAVCeleb_v1.2.zip"   "FakeAVCeleb_v1.2"
download_one "HiFi-AVDF.zip"          "HiFi-AVDF"

echo "=== 全部完成 ==="
ls -d "$SYNCX_DATA_ROOT"/*
```

要点：
- `wget -c` 会在当前目录生成 `.zip` 文件，断线后**重跑同一条命令即可续传**。
- 代理掐断时 `wget` 会打印 `Read error ... Retrying` / `Connection closed ... Retrying`，等几秒自动继续，不用管。
- 下完自动解压并删除 zip，最终目录结构和 §0.2 一致。
- 若速度实在无法忍受，**临时把仓库转 Public 用 §0.4**，下完立刻转回 Private。

### 0.4 大文件推荐：临时转公开 + hf-mirror + `wget -c`（AutoDL 实例内运行）

> 私有仓走官方站 + XET CDN 就是慢；**临时把 `Lucifer696/Thesis` 设为 Public**，即可走国内镜像 `hf-mirror.com`，无需 token、无需 `source /etc/network_turbo`，速度通常快一个数量级。下完**立刻**转回 Private 保护版权。

```bash
# 1. 先在 HF 网页把 Lucifer696/Thesis 设为 Public，等 3–5 分钟镜像同步。
# 2. 进入 tmux（可选但推荐，窗口关闭不中断）
tmux new -s hf

# 3. 直接下，不用代理、不用 token
export SYNCX_DATA_ROOT=/root/autodl-tmp/SyncX/Datasets
mkdir -p "$SYNCX_DATA_ROOT" && cd "$SYNCX_DATA_ROOT"
set -euo pipefail

download_one() {
  local fn=$1
  local dn=$2
  mkdir -p "$dn" && cd "$dn"
  echo ">>> 下载 $fn"
  wget -c "https://hf-mirror.com/datasets/Lucifer696/Thesis/resolve/main/$fn" -O "$fn"
  echo ">>> 校验 $fn"
  python - <<PY
import zipfile, sys
try:
    with zipfile.ZipFile('$fn') as z:
        bad = z.testzip()
        if bad:
            print(f"ERROR: zip corrupt, first bad file: {bad}"); sys.exit(1)
except Exception as e:
    print(f"ERROR: not a valid zip: {e}"); sys.exit(1)
print("OK: zip valid")
PY
  echo ">>> 解压 $fn"
  python - <<PY
import zipfile, os, shutil
zp='$fn'; td='.'; tmp='extract_tmp'
with zipfile.ZipFile(zp) as z: z.extractall(tmp)
items=os.listdir(tmp)
if len(items)==1 and os.path.isdir(os.path.join(tmp,items[0])):
    for f in os.listdir(os.path.join(tmp,items[0])):
        shutil.move(os.path.join(tmp,items[0],f), td)
else:
    for f in items: shutil.move(os.path.join(tmp,f), td)
shutil.rmtree(tmp); os.remove(zp)
PY
  cd ..
}

download_one "Deepfake-Eval-2024.zip" "Deepfake-Eval-2024"
download_one "FakeAVCeleb_v1.2.zip"   "FakeAVCeleb_v1.2"
download_one "HiFi-AVDF.zip"          "HiFi-AVDF"

echo "=== 全部完成 ==="
ls -d "$SYNCX_DATA_ROOT"/*
# Ctrl+B 再按 D 脱离
```

关键注意事项：
- **同步延迟**：HF 刚转 Public，hf-mirror 要几分钟同步，此时 wget 可能 404 → 等 3–5 分钟再试。
- **转私有时机**：必须等 `=== 全部完成 ===` 且五个目录都出现，再去 HF 网页设回 Private。中途转私有，镜像 404，剩余文件失败；已下完的不受影响。
- **版权**：公开期间任何人能下，下完立刻转回 Private。部分数据集（FF++ 等）禁止再分发，不要长期公开。
- 如果 mirror 也慢或 404，回退到 §0.3（私有仓 + 代理 + wget -c）保底。
- ⚠️ **脚本必须带 `set -euo pipefail` + zip 校验（见上面修正版）**：早期版本没有错误检查，XET 断连导致 zip 只下一半时 `wget` 误判完成、python 解压抛异常但 bash 不中断，会**悄悄跳过失败文件去下第二个**（2026-08-17 实测 Deepfake-Eval-2024.zip 只下 2.93GB/应 6.6GB 就被判完成）。修正版加了 `zipfile.testzip()` 校验 + 退出码检查，任一环节失败立即 `exit 1` 停止。
- 💡 **错误处理 ≠ 提速**：`set -e` 和 zip 校验只保证正确性，不改变下载速度（仍由 XET CDN→AutoDL 链路决定，~100–300KB/s）。真正提速只能换非 HF 源（如 ModelScope `Lucifer999/Thesis`）。`network_turbo` 代理实测对 >7GB 反而更慢。

### 0.5 HiFi-AVDF 捷径：公开仓 JielunPeng + 核对脚本（2026-08-18）

> HiFi-AVDF **不再走** `Lucifer696/Thesis` 整包（私有仓 XET 极慢；且本机 `D:\Datasets\HiFi-AVDF` 才是已划分好的 canonical）。改用第三方公开重打包仓 `JielunPeng/HiFi-AVDF`（6.51GB，6 生成器 × (zip 视频 + csv 标签)，列名 `video_path,audio_label,visual_label,overall_label` 与本机一致），走 hf-mirror 快且免 token。
> ⚠️ 该公开仓**缺官方 `face_verify_hifi_master.jsonl`（人脸校验层）**，但每个 gen csv 已含完整 `overall_label`，评测直接吃 per-gen csv 即可（本机评测本就是 join 各 gen csv 的 overall_label），无需 master。
> ✅ **核对脚本 `code_cloud/verify_hifi_split.sh` 已在本机 oracle 跑通 PASS**：公开仓解压后划分与本机 100% 一致。

**下载（AutoDL 前台，带进度条，断线重贴 `wget -c` 续传）**：
```bash
export SYNCX_DATA_ROOT=/root/autodl-tmp/SyncX/Datasets
mkdir -p "$SYNCX_DATA_ROOT/HiFi-AVDF" && cd "$SYNCX_DATA_ROOT/HiFi-AVDF"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
for g in kling2.5 pixverseV5 seedance1.0 sora2 veo3.1 wan2.5; do
  wget -c "https://hf-mirror.com/datasets/JielunPeng/HiFi-AVDF/resolve/main/$g.zip" -O $g.zip
  wget -c "https://hf-mirror.com/datasets/JielunPeng/HiFi-AVDF/resolve/main/$g.csv" -O $g.csv
done
echo "=== 全部完成 ==="
```

**核对（解压 + 三项严格校验 + 重建等价 master）**：
```bash
BASE=$SYNCX_DATA_ROOT/HiFi-AVDF bash /path/to/code_cloud/verify_hifi_split.sh
```
脚本会：① 防嵌套解压（zip 内顶层若是 gen 名则拉平，保证视频落在 `$g/` 与 csv 相对路径命中）；② C1 核对表头 + 准确数据行（`awk 'END{print NR-1}'` 规避 `wc -l` 末行无换行少算 1 行的坑）；③ C2 视频数==数据行；④ C3 csv 每个 `video_path` 文件存在；⑤ 重建 6-gen union（合计 3810 = 官方 master）。全部 PASS 即划分与本机等价。

**本机基准（awk 准确统计，已 oracle 验证）**：
| gen | 数据行 |
|---|---|
| kling2.5 | 460 |
| pixverseV5 | 616 |
| seedance1.0 | 788 |
| sora2 | 728 |
| veo3.1 | 638 |
| wan2.5 | 580 |
| **合计** | **3810 = face_verify_hifi_master.jsonl** |

> ⚠️ 早期记的 kling2.5=459 是 `wc -l` 末行无换行少算 1 的 bug，真实为 **460**。脚本一律用 awk，勿用 `wc -l - 1`。

---

## 1. 关键事实（踩坑后确认）

| 项 | 值 |
|---|---|
| **HF 用户名** | `Lucifer696`（公开 handle，非秘密） |
| **中转私有仓** | `Lucifer696/Thesis`（已存在，直接传，不要新建） |
| **endpoint** | 官方 `https://huggingface.co`（默认；**勿用 hf-mirror**，镜像拉不到私有库） |
| **token 权限** | 必须 **Write**（或对该仓有写权限）；纯 Read-only 会 403 |
| **token 传递** | 走环境变量 `HF_TOKEN`，**绝不硬编码 / 绝不贴进聊天** |

---

## 2. 前置条件

1. 装包：`pip install -U huggingface_hub`（`hf_transfer` 已废弃，不要装）。
2. 去 https://huggingface.co/settings/tokens 建一个 **Role = Write** 的 token（细粒度要勾数据集写权限）。
3. 设环境变量：`export HF_TOKEN=hf_xxx`（Windows cmd 用 `set HF_TOKEN=hf_xxx`）。

---

## 3. 为什么不用 `upload_to_hf.py` / `create_repo`

- `upload_to_hf.py` 会 `create_repo` 新建 `syncx-*` 仓库。若 token 没有「创建数据集」权限，
  会报 `403 Forbidden: You don't have the rights to create a dataset under the namespace "Lucifer696"`。
- 直接传进**已存在的** `Lucifer696/Thesis` 仓（用 `upload_file`），不触发建仓，绕开该 403。
- 若日后想恢复「每数据集一个独立仓 (`syncx-dfe2024` 等)」，需确保 token 有 create 权限，
  且云端 `cloud_fetch_datasets.py` 的 PLAN repo 字符串要与之对应。

---

## 4. 踩过的坑（避免重蹈）

1. **`huggingface-cli: command not found`**
   → 包装了但没进 PATH。改用 `python -c "from huggingface_hub import HfApi; ..."` 或
   `python -m huggingface_hub.commands.huggingface_cli whoami`，不要依赖 `huggingface-cli` 命令。

2. **`login(token=True)` 报错 `AttributeError: 'bool' object has no attribute 'startswith'`**
   → `login(token=True)` 是把布尔 `True` 当 token 传，是错的。**不需要 `login()`**。
   `HfApi()` 会自动读 `HF_TOKEN` 环境变量（且环境变量优先级高于本地缓存 `~/.cache/huggingface/token`）。

3. **`403 Forbidden` on `api/repos/create`**
   → token 无建仓权限（只读 / 细粒度未勾创建）。换 Write token，或直接传已有仓（见 §0）。

4. **`404 Repository Not Found` on `.../preupload/main`**
   → 仓库不存在却去 `upload_file`（之前的 `try/except` 把 `create_repo` 的 403 吞了，导致仓没建、上传 404）。
   → 用已有仓 `Lucifer696/Thesis` 就不会有这问题；若坚持新建仓，别吞 `create_repo` 异常。

5. **`404` 报 repo type 不对**
   → `Lucifer696/Thesis` 若建的是 model 仓而非 dataset 仓，把命令里 `repo_type="dataset"` 改成 `"model"`。

---

## 5. 云端拉取对应改动（重要，待办）

> **云端拉取前置加速**：在 AutoDL 实例上拉取前先 `source /etc/network_turbo`（仅当前终端生效，
> 只对境外 HF 源快；国内 pip 源反而更慢）。这条是 HF 拉取提速的关键，务必先 source 再跑下载。

当前 `cloud_fetch_datasets.py` 的 PLAN 里 DFE2024 / FakeAVCeleb 对应的是**独立仓库**
`syncx-dfe2024` / `syncx-fakeavceleb`，现在改成塞进 `Lucifer696/Thesis` 一个仓了，
云端直接 `--only` 会找不到。上传成功后需把云端拉取改为：
- 从 `Lucifer696/Thesis` 用 `snapshot_download`（或按文件名 `upload_file` 反向的 `hf_hub_download`）拉这两个 zip，
- 解压到对应 `DATA_ROOT/Deepfake-Eval-2024` 和 `DATA_ROOT/FakeAVCeleb_v1.2`。
- 同时把 PLAN 里的 `<你的用户名>` 替换成 `Lucifer696`。

> 这条改动没在本文件自动完成；需要时让 AI 改 `cloud_fetch_datasets.py` 或直接给按文件名下载的片段。

---

## 7. 私有仓下载实测复盘（2026-08-17，已联网验证）

### 7.1 第一次踩坑：临时转 Public 后走 `HF_ENDPOINT=https://hf-mirror.com` 卡 0%

**现象**：把 `Lucifer696/Thesis` 临时转 Public 后，用 `hf_hub_download` + `HF_ENDPOINT=https://hf-mirror.com` 下载，卡在 `0%` 不动。

**根因**：
1. **hf-mirror 不支持私有仓**——它是公开库只读镜像，私有仓拉取会 404。
2. 当时用的是 Python SDK 的 `hf_hub_download`，`HF_HUB_DISABLE_XET=1` 与镜像 endpoint 叠加后行为不稳定；且刚转 Public 时镜像同步有延迟。

### 7.2 第二次踩坑：私有仓 + `wget -c` + 代理，能下但极慢、反复断

**现象（见用户截图 2026-08-17）**：私有仓用 `wget -c` + `Authorization: Bearer $HF_TOKEN` + `source /etc/network_turbo` 下载 6–7G zip，速度 9–220 KB/s，频繁出现 `Connection closed at byte ... Retrying`，Deepfake-Eval-2024 反复在 18% 附近断连，FakeAVCeleb ETA 10h+。

**根因**：
1. **官方站已全面切 XET CDN**：`https://huggingface.co/datasets/.../resolve/main/xxx.zip` 会 302 到 `https://us.aws.cdn.hf.co/xet-bridge-us/...`。`HF_HUB_DISABLE_XET=1` 只对 Python SDK 生效，**对 `wget -c` 直链无效**。
2. **AutoDL 代理长连接不稳定**：XET CDN 的 6–7G 大文件下载会被 network_turbo 代理反复掐断，每次 `Connection closed` 后 `wget -c` 重新握手、速度归零再爬升。
3. 直连官方站 `unset http_proxy https_proxy` 在 AutoDL 机房经常 `Cannot assign requested address` / 长时间无响应（无代理可达性差）。

**结论**：私有仓 `wget -c` + 代理是**保底方案**（能下完，靠断点续传磨），但**不是推荐方案**。想快见 §0.4：**临时转 Public + hf-mirror + wget -c**。

### 7.3 当前推荐策略

| 仓库状态 | 推荐命令 | 速度 | 备注 |
|---|---|---|---|
| **私有仓（默认）** | §0.3：`wget -c` + token + 代理 | 慢，9–220 KB/s，易断 | 保底，最终能下完 |
| **临时 Public** | §0.4：`wget -c` + hf-mirror，无 token 无代理 | 快（通常 MB/s 级） | 下完立刻转回 Private |
| 小文件/模型权重 | §0.2：`hf_hub_download` + `HF_HUB_DISABLE_XET=1` | 中等 | 仅适合 <2GB |

**不要再做的事**：
- 不要用 `HF_HUB_ENABLE_HF_TRANSFER=1`（已废弃）。
- 不要对 `wget -c` 直链指望 `HF_HUB_DISABLE_XET=1` 绕过 XET CDN（只对 Python SDK 有效）。
- 不要在 AutoDL 上 `unset http_proxy https_proxy` 直连官方站（经常连不上）。

---

## 6. 安全

- token 一旦贴进聊天/截图就视为泄露，用完去 HF 后台 **revoke** 旧 token、换新。
- Write token 权限大，只走环境变量，不要写进任何 `.py` / `.sh` / 文档源码。
- `Lucifer696/Thesis` 是私有仓；部分数据集（FF++ 等）许可禁止再分发，私有自拉通常被容忍，勿公开分享。
