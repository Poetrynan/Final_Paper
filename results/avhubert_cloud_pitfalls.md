# AV-HuBERT 换载体（方向 B）云端排坑全档案

日期：2026-08-29（cloud exec /root/autodl-tmp/SyncX/code_cloud，实例 42bc42b309-4dadcbd2，RTX 4090D）
补齐：2026-08-30（坑 14–18，restart 轮）
当前状态：avh_smoke DONE，avh_prep 全量重抽中 →
  VoxCeleb2_subset 500 ✔ · DeepSpeak train 10548 ✔ · DeepSpeak test 2913 ✔ ·
  FakeAVCeleb_v1.2/prep_avh 进行中（目标 ~22044）→ 之后 DFE2024 → HiFi-AVDF → avh_s1。
**git 命令：`tmux new -d -s avh "SYNCX_AVH_CKPT=/root/avhubert_ckpt/base_vox_iter5.pt PREP_SHARDS=8 bash cloud_settle_avhubert.sh"`**
  （监控：`tail -3 /root/autodl-tmp/SyncX/persist/status_avhubert.log` +
   `find /root/autodl-tmp/SyncX/Datasets/FakeAVCeleb_v1.2/prep_avh -name cache.pt | wc -l`）

---

## 二·二、8-30 restart 轮的坑（坑 14–18）

### 坑 14 · tmux 命令套 `bash -c` 把顶层 env 又吞了
- 现象：restart 后用 `tmux new -d -s avh "bash -c 'export ...' && bash cloud_settle_avhubert.sh"`，
  verify 又报 `FATAL`，但 `ls /root/avhubert_ckpt/*.pt` 明明有权重。
- 根因：多一层 `bash -c` 子 shell 后，`SYNCX_AVH_CKPT` 没传到最内层脚本；verify 读到空串 = 同坑 6。
- 解决：**不要再用 `bash -c` 包裹**，环境变量直接平铺在 tmux 命令里：
  ```bash
  tmux new -d -s avh "SYNCX_AVH_CKPT=/root/avhubert_ckpt/base_vox_iter5.pt PREP_SHARDS=8 bash cloud_settle_avhubert.sh"
  ```

### 坑 15 · prep_one 的 SKIP 只查顶层 cache.pt → 已完成数据集被重跑 → OOM Killed
- 现象：接通 verify 后 avh_prep 跑 DeepSpeak 时报 `Killed`（cgroup 62GB 内存被打爆）。
- 根因：`prep_one` 只 `[ -e "$out/cache.pt" ]` 判断顶层单个文件，而 cache 实际**按子目录**存
  （`<prep_avh_test>/<子目录>/cache.pt`），已完成的 DeepSpeak train/test 也因此被重抽，内存峰值爆 62GB。
- 解决：SKIP 判定改为**递归数 cache.pt**：
  ```bash
  if [ "$(find "$out" -name cache.pt | wc -l)" -gt 0 ]; then echo "skip完全($cached)"; return 0; fi
  ```

### 坑 16 · 8 个 shard 并发内存叠加超 cgroup 上限 → 再 OOM
- 现象：修好 SKIP 后 Trainer/前向仍偶发 `Killed`。
- 根因：容器 cgroup 内存上限 62GB，`PREP_SHARDS=8` 时单 shard 若吃 8G+，并发达峰即被杀。
- 解决：**给每个 shard 加内存墙**：`ulimit -v 8000000`（8GB/进程），确保 8 × 8G=64G 依旧逼近，
  经验上按 shard 数除 max 并留余量；重启命令里前置 `ulimit -v 8000000`。

### 坑 17 · PYTHONPATH 内联前缀不被 conda run 继承 → `No module named avhubert`
- 现象：`conda run -n py39 python prep_avhubert.py` 报 `No module named 'avhubert'`；
  改 `PYTHONPATH=... conda run ...` 内联注入也无效。
- 根因：`conda run` 会起新的子进程，内联的 `PYTHONPATH` 只作用在包裹它的那层 shell，
  进不到 python 进程。
- 解决：**在脚本顶层 `export PYTHONPATH`**（全局环境变量），让所有下层子进程继承：
  ```bash
  export PYTHONPATH="$AVH_PKG:$AVH_SRC:${PYTHONPATH:-}"
  runpy() { $PY "$@"; }   # PY="conda run -n py39 python"
  ```
  同理坑 6 顶层的 `export SYNCX_AVH_CKPT/HF_HOME/...` 是同一原则：**env 一律在脚本顶层 export**。

### 坑 18（非故障） · ps 用 `grep prep_avhubert.py` 数不到"轮转的新 shard"
- 现象：`ps|grep '[p]rep_avhubert.py'` 计数忽 0 忽有，误以为进程又没了。
- 根因：shard 会轮转（旧的退出、新的再起），新起的命令行不含 `prep_avhubert.py` 字样。
- 判定：**进程数别当唯一健康信号**，以 cache.pt **持续增长**为准 + `status_avhubert.log` 无报错。

### 坑 19（FAQ） · 为什么 avh_prep 阶段 nvidia-smi 几乎不占 GPU（1 MiB）？
- 现象：16 个 shard 全在 100%+ CPU 跑，`nvidia-smi` 只看到 1 MiB 显存占用，疑惑"没用到 GPU"。
- 根因（设计如此，不是故障）：prep 阶段的主要开销是
  **MediaPipe 唇部关键点（强制 CPU·XNNPACK，`SYNCX_MEDIAPIPE_GPU=0`）+ 音频 fbank + 大文件 IO**，
  这些占去 pipeline 绝大部分时间；AV-HuBERT 的联合前向只是每段末尾一小步，GPU 被占用很短暂，
  且受 `SYNCX_DEVICE` 控制。所以整段 prep 表现为 CPU 密集 = **正常**。
- 判定：GPU 的大头占用在**后续 Stage-1 / Stage-2 训练**才会出现；prep 阶段盯 CPU % + cache.pt 增长即可，
  别用 GPU 占用判断 prep 是否健康。

---

## 一、一句话结论
**所有坑 = 三件事：跑错解释器（3.12 vs py39）、缺/错依赖版本（numpy/torchvision/opencv）、脚本环境变量/参考路径接线不到位。**
当这三类全部对齐到"手动验证通过"的状态后，脚本就自然跑通了——因为**手动命令早就证明链路本身（AV-HuBERT 联合前向 8/8 成功）是好的**，脚本只是没把同样的 env/路径喂进去。

排查原则：**status 日志只记 STEP 摘要，真报错要手动以完全相同方式跑 + 抓 stderr；别被 MediaPipe 退出噪音骗。**

---

## 二、逐坑清单（按时间序）

### 坑 0 · 权重下不动（HTTP 403）
- 现象：verify 探测 3 个候选 URL 全 `403 Forbidden`。
- 根因：候选里 `base_lrs3_vox_iter5.pt`（带 lrs3 字样的那几款）在 fbaipublicfiles 上 403。
- 解决：换成实测可达的 **`clean-pretrain/base_vox_iter5.pt`**（英文 SSL base，无微调，out-dim 768）。
- 关键 URL：`https://dl.fbaipublicfiles.com/avhubert/model/lrs3_vox/clean-pretrain/base_vox_iter5.pt`（1.2G）。

### 坑 1 · torch.load weights_only 拦截
- 现象：`WeightsUnpickler error: Unsupported global: GLOBAL fairseq.data.dictionary.Dictionary`。
- 根因：torch 2.6+ 默认 `torch.load weights_only=True`，禁止 unpickler 加载 fairseq 的 Dictionary 类。
- 解决：`avhubert_backbone.py` 里在 fairseq 加载前临时垫片：
  ```python
  _orig_tl = torch.load
  def _tl_compat(*a, **k): k.setdefault("weights_only", False); return _orig_tl(*a, **k)
  torch.load = _tl_compat
  try: models, ... = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt], ...)
  finally: torch.load = _orig_tl
  ```

### 坑 2 · 脚本用了裸 python（3.12）而不是 py39
- 现象：`FATAL: AV-HuBERT verify 未通过`，秒退。
- 根因：verify 那行是 `python ...`（= 系统主环境 3.12），`import fairseq/avhubert` 因 dataclasses/omegaconf 不兼容崩。
- 解决：整份脚本统一 `runpy`/`runpy_plain`（=`conda run -n py39 python`），verify/smoke/prep 用 `runpy`（注入 AVH_PYTHONPATH），训练/评测用 `runpy_plain`。

### 坑 3 · avhubert import 的 DBG 顶层导入分支
- 现象：`python -c "import avhubert"` 时 `ModuleNotFoundError: hubert_pretraining` / `compute_mask_indices from utils` 冲突。
- 根因：`hubert.py` 里 `if len(sys.argv)==1: DBG=True` 走顶层 `from .hubert import`；带参数进入才走相对导入。`-c` 时 sys.argv 长度 1 → 触发 DBG 坏分支。
- 解决：脚本里**不要用 `python -c "import avhubert"` 当门槛**，改验目录存在；把真正 import 交给 `verify_avhubert.py`（脚本入口，argv>1）。

### 坑 4 · PYTHONPATH 顺序 → 项目 utils.py 污染 avhubert 的 utils
- 现象：`ImportError: cannot import compute_mask_indices from 'utils' (/root/autodl-tmp/SyncX/code_cloud/utils.py)`。
- 根因：`PYTHONPATH=".../av_hubert:.../av_hubert/avhubert"` 时 `av_hubert`(SRC) 在前，`from utils import compute_mask_indices` 命中了项目根 `code_cloud/utils.py`。
- 解决：**`avhubert` 目录必须放最前**：`PYTHONPATH="$AVH_PKG:$AVH_SRC:${PYTHONPATH:-}"`（runpy 内同款）。

### 坑 5 · bash 语法被补丁改崩（重复 else）
- 现象：`line 148: syntax error near unexpected token 'else'`，脚本启动即退、连第一个 STEP 都不打。
- 根因：多次 `python` replace 补丁把 `if/elif/else` 改出两个连续 `else`。
- 解决：`bash -n` 狠检 + 删掉多余 else/重复 log 行。**教训：改 shell 脚本后一定先 `bash -n`。**

### 坑 6 · smoke 拿不到权重路径（SYNCX_AVH_CKPT 空字符串）
- 现象：`FileNotFoundError: AV-HuBERT checkpoint 不存在: ''`。
- 根因：`SYNCX_AVH_CKPT` 只在 verify 子进程块内 export，没回流主进程；smoke/prep 子进程继承不到。
- 解决：**脚本顶层** export：`export SYNCX_AVH_CKPT="${SYNCX_AVH_CKPT:-$(ls /root/avhubert_ckpt/*.pt ...|head -1)}"`，顺带同补 `HF_HOME`/`HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE=1`。

### 坑 7 · py39 缺 mediapipe
- 现象：`ModuleNotFoundError: No module named 'mediapipe'`（common.py _VisualEncoder 需它做唇关键点）。
- 解决：`conda run -n py39 pip install mediapipe`（CPU 版即可，SYNCX_MEDIAPIPE_GPU=0 已设）。

### 坑 8 · cv2 与 numpy 版本冲突
- 现象：`ImportError: numpy.core.multiarray failed to import` / `AttributeError: module 'cv2.dnn' has no attribute 'DictValue'`。
- 根因：装成了新版 opencv-python（带 typing），与 numpy 1.23.5 不兼容。
- 解决：**卸载干净后装** `opencv-python-headless==4.5.4.60`（官方与 numpy 1.23 兼容组合）。

### 坑 9 · torchvision 与 torch 二进制不匹配
- 现象：`RuntimeError: operator torchvision::nms does not exist`。
- 根因：torch 是 **2.8.0+cu128**，装了不配对的 torchvision。
- 解决：配对重装 **`torchvision==0.23.0+cu128`**（torch 2.8 ↔ tv 0.23）。

### 坑 10 · scipy 引用 numpy.Inf
- 现象：`ImportError: cannot import name 'Inf' from 'numpy'`（transformers→CLIP→scipy 链）。
- 解决：升级 `scipy==1.11.4`（不再碰已删的 numpy.Inf；与 numpy 1.23 兼容）。

### 坑 11 · fairseq 需 numpy 旧别名 np.float（numpy 被拉成 1.26/2.x）
- 现象：`AttributeError: module 'numpy' has no attribute 'float'`。
- 根因：实际生效 numpy 是 1.26.4（被某依赖拉高），fairseq 老代码要 `np.float`。
- 解决：**`--force-reinstall numpy==1.23.5`**，并全套复验 cv2 4.5.4 / scipy 1.11.4 / torchvision 0.23 / transformers 全兼容。

### 坑 12 · EVAL 参考树误指不存在的 prep_avh
- 现象：smoke 仍 FATAL，但手动用 `--ref-prep .../prep` 却成功。
- 根因：脚本 `EVAL[hifi]=".../HiFi-AVDF/prep_avh"`（输出树，不存在），`--ref-prep` 空白。
- 解决：EVAL 三个键改回**存在的参考树**：`fakeavceleb→.../prep`、`DFE2024→.../prep/test`、`hifi→.../prep`。

### 坑 13（非故障） · MediaPipe 退出噪音
- 现象：`Exception ignored in: <function FaceLandmarker.__del__> ... TypeError: 'NoneType' object is not callable`。
- 判定：**无害**。发生在 `__del__` 析构、主链路完成后，`Exception ignored` 不中断；其后紧接 `smoke OK` / `rebuilt-ready=8` 证明成功。
- 教训：判断脚本真伪看 **marker 文件 + `rebuilt-ready=X / skipped=Y`**，别被终端滚屏里的 traceback 吓到。

---

## 三、为什么现在能跑通（本质）
1. **链路本身一直是好的**：手动 cpu/cuda 都 `rebuilt-ready=8 / skipped=0`，AV-HuBERT 联合前向、CLIP、MediaPipe 唇定位、fbank、200×768 token 全部正确。
2. **把"手动命令凑齐的 env/路径"完整喂给脚本**：
   - 解释器：全走 py39（`runpy`）
   - PYTHONPATH：`AVH_PKG:$AVH_SRC`（avhubert 在前）
   - 环境变量：`SYNCX_AVH_CKPT`/`HF_HOME`/offline 顶层 export
   - 参考路径：`EVAL` 指向真实 `prep` 树
   - 依赖版本：numpy 1.23.5 + cv2 4.5.4.60 + scipy 1.11.4 + torchvision 0.23.0+cu128 + mediapipe
3. **门槛逻辑合理**：clone 只验目录、verify 幂等（复用已有权重不外下）、avhubert 真伪由 verify 把关。

一旦这三类（解释器 / 依赖版本 / env+路径）与"手动验证通过"的状态一致，脚本就复现了手动的成功。