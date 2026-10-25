"""
Sync-X — 共享预处理工具（standby 脚本，未执行）。

可立即运行的部分：解包、均匀抽帧、音频抽取、半精度 tensor 缓存打包。
WIRE 接线点：特征编码器（CLIP / wav2vec2 / MediaPipe）在 extract_* 函数内
「懒加载」，所以即使没装重型依赖，解包+抽帧步骤也能跑。
把标记了 `TODO(WIRE)` 的块替换成你冻结/训练好的 Sync-X 编码器即可。

路径约定（与 Sync-X 指南 §B.4 一致）：
  audio_seq : wav2vec2-base-960h -> R^{T_a x 768}
  v_clip    : 松框 224x224 -> CLIP ViT-B/16 -> 512-D / 帧
  v_crops   : 紧唇 ROI 112x112 灰度（可选；当前 Sync-X 不消费，pack_cache 默认不写）
  v_lip     : FaceLandmarker 唇点 64 点 -> 展平 R^192 / 帧
"""

from __future__ import annotations
import os, glob, json, math, tarfile, subprocess, shutil, time
from pathlib import Path

# ── 线程纪律（2026-08-16 加入，防云端 CPU 超发）──
# cloud_run_all.sh 以 SHARDS 进程并行抽取时，每进程注入 OMP/MKL/OPENBLAS 线程数。
# 此处再兜底一道：若环境变量已设则尊重，未设则不干预（单进程默认全核 OK）。
# 必须在 numpy/torch 初始化前设置才生效。
for _env_k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _env_v = os.environ.get(_env_k)
    if _env_v:
        try:
            import os as _os
            _os.environ[_env_k] = str(int(_env_v))
        except Exception:
            pass

import numpy as np


# ----------------------------- 解包 -----------------------------
def unpack_targz(root: str, out_root: str, remove_after: bool = False) -> int:
    """递归找 root 下所有 *.tar.gz，解到 out_root/<stem>/。返回解包个数。"""
    root = Path(root)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    tars = sorted(glob.glob(str(root / "**" / "*.tar.gz"), recursive=True))
    print(f"[unpack] found {len(tars)} .tar.gz under {root}")
    n = 0
    for t in tars:
        stem = Path(t).stem
        if stem.endswith(".tar"):
            stem = stem[:-4]
        dest = out_root / stem
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(t, "r:gz") as tf:
            tf.extractall(dest)
        if remove_after:
            os.remove(t)
        n += 1
    print(f"[unpack] done, {n} archives extracted -> {out_root}")
    return n


def iter_videos(root: str, exts=(".mp4", ".avi", ".mov", ".mkv", ".webm")) -> list[str]:
    """递归收集 root 下所有视频文件路径。"""
    root = Path(root)
    out = []
    for ext in exts:
        out.extend(glob.glob(str(root / "**" / f"*{ext}"), recursive=True))
    return sorted(out)


# ----------------------------- 抽帧 -----------------------------
def sample_frames(video_path: str, target_count: int = 50, src_fps_hint: float = 25.0,
                  window_sec: float | None = None, window_fps: float = 25.0):
    """均匀抽最多 target_count 帧；返回 list[np.ndarray BGR]。

    window_sec=None（默认）：横跨整段均匀抽（旧行为，向后兼容）。
    window_sec>0：仅在视频中心 window_sec 秒窗口内、按 window_fps 抽帧
      （如 4s@25fps=100 帧），保证原生采样率满足运动学 Nyquist（1–8Hz 带）。
      短于窗的片段退化为「用全部可用帧」（不重复），由下游 _fix_len 补零。
    详见 SyncX_Fix_Plan_4s_Window.md。
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total == 0:
        cap.release()
        return []
    if window_sec and window_sec > 0:
        fps = cap.get(cv2.CAP_PROP_FPS) or src_fps_hint
        dur_frames = total
        win_frames = int(round(window_sec * fps))
        if win_frames >= dur_frames:        # 短片段：用全部帧
            start, n = 0, dur_frames
        else:                               # 长片段：中心窗
            start = max(0, int(round((dur_frames - win_frames) / 2)))
            n = win_frames
        idxs = sorted(set(int(round(start + i * (n - 1) / max(1, target_count - 1)))
                          for i in range(target_count)))
    else:
        idxs = sorted(set(int(round(i * (total - 1) / max(1, target_count - 1))) for i in range(target_count)))
    idxs = [max(0, min(i, total - 1)) for i in idxs]
    # 顺序解码一次，只取目标帧：cv2 的 CAP_PROP_POS_FRAMES 对 mp4 是「从头解码到目标帧」，
    # 100 个分散索引 = 100 次全片解码（云端实测 ~23s/clip）。改为 seek 一次到首帧 + 顺序读窗内，
    # 解码量从 O(seek数 × 全片) 降到 O(窗起点 + 窗长)。
    frames = []
    if idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idxs[0])
    want = set(idxs)
    cur = idxs[0] if idxs else 0
    last = idxs[-1] if idxs else -1
    while cur <= last:
        ok, fr = cap.read()
        if not ok:
            break
        if cur in want:
            frames.append(fr)
        cur += 1
    cap.release()
    return frames


def clip_center_window(video_path: str, window_sec: float, fps_hint: float = 25.0):
    """返回 (start_sec, dur_sec) 中心窗；视频短于窗则 (0.0, 全长秒)。

    供 prep 把音频抽取（video_to_wav 的 ss/td）对齐到与 sample_frames
    同一个中心窗，避免「音频前 4s / 视频跨整段」的时间轴错位。
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return (0.0, 0.0)
    fps = cap.get(cv2.CAP_PROP_FPS) or fps_hint
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cap.release()
    if total == 0 or fps <= 0:
        return (0.0, 0.0)
    dur_frames = total
    win_frames = int(round(window_sec * fps))
    if win_frames >= dur_frames:
        return (0.0, dur_frames / fps)
    start = max(0, int(round((dur_frames - win_frames) / 2)))
    return (start / fps, win_frames / fps)


def save_frames(frames, out_dir: str) -> int:
    import cv2
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(str(out / f"f{i:04d}.png"), f)
    return len(frames)


# ----------------------------- 音频 -----------------------------
def _video_to_wav_ffmpeg(video_path: str, wav_path: str, sr: int = 16000,
                         ss: float | None = None, td: float | None = None) -> bool:
    """ffmpeg subprocess 抽取（PyAV 不可用时的回退路径）。"""
    wav_path = str(wav_path)
    # 多进程分片抽取时限制 ffmpeg 解码线程（读 OMP_NUM_THREADS，默认自动）。
    # 避免 8 shard 各自 ffmpeg 全核解码 → 线程超发抢核。
    cmd = ["ffmpeg", "-y"]
    _nt = os.environ.get("OMP_NUM_THREADS")
    if _nt:
        cmd += ["-threads", str(int(_nt))]
    if ss is not None:
        cmd += ["-ss", f"{float(ss):.3f}"]
    cmd += ["-i", video_path]
    if td is not None:
        cmd += ["-t", f"{float(td):.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(sr), wav_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return Path(wav_path).exists()
    except Exception as e:
        print(f"[audio] ffmpeg failed for {video_path}: {e}")
        return False


def _video_to_wav_pyav(video_path: str, wav_path: str, sr: int = 16000,
                       ss: float | None = None, td: float | None = None) -> bool:
    """PyAV（进程内 ffmpeg 绑定）抽单声道 16k PCM WAV。~0.03s/clip，比 subprocess 快 50x+。"""
    import av
    import numpy as np
    import wave as _wave
    container = av.open(str(video_path))
    astream = next((s for s in container.streams if s.type == "audio"), None)
    if astream is None:
        container.close()
        return False
    if ss:
        container.seek(int(float(ss) / astream.time_base), stream=astream)
    resampler = av.AudioResampler(format="s16", layout="mono", rate=sr)
    deadline = (float(ss) if ss else 0.0) + (float(td) if td else 0.0) + 0.05
    pcm = []
    for frame in container.decode(astream):
        if frame.time > deadline:
            break
        if ss and frame.time < float(ss) - 0.05:
            continue
        for rf in resampler.resample(frame):
            pcm.append(rf.to_ndarray())
    container.close()
    if not pcm:
        return False
    data = np.concatenate(pcm, axis=1).reshape(-1).astype(np.int16)
    with _wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(data.tobytes())
    return True


def _video_to_wav_torchaudio(video_path, wav_path, sr, ss, td):
    """兜底：PyAV/ffmpeg 都失败时，用 torchaudio（能解 DFE 一批 stereo aac/h264）抽 mono 16k。
    仅在主流路径全部失败时触发，不影响已验证的 HiFi/Vox/DeepSpeak 主干。本函数不 import torchaudio 于模块级避免重依赖。"""
    import torch, torchaudio, os
    w, orig_sr = torchaudio.load(video_path)
    if w.shape[0] > 1:
        w = w.mean(0, keepdim=True)          # stereo -> mono
    if orig_sr != sr:
        w = torchaudio.functional.resample(w, orig_sr, sr)
    if ss is not None and td is not None:
        start = int(round(ss * sr)); n = int(round(td * sr))
        end = min(start + n, w.shape[1]); start = min(start, end)
        w = w[:, start:end]
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    torchaudio.save(wav_path, w, sr)
    return os.path.exists(wav_path) and w.shape[1] > 0

def video_to_wav(video_path: str, wav_path: str, sr: int = 16000,
                 ss: float | None = None, td: float | None = None) -> bool:
    """抽单声道 16k wav。优先 PyAV（进程内解码，快），失败回退 ffmpeg subprocess，再兜底 torchaudio。"""
    try:
        return _video_to_wav_pyav(video_path, wav_path, sr, ss, td)
    except Exception as e:
        print(f"[audio] PyAV 不可用/失败，回退 ffmpeg: {e}")
        try:
            return _video_to_wav_ffmpeg(video_path, wav_path, sr, ss, td)
        except Exception as e2:
            print(f"[audio] ffmpeg 也失败({e2})，兜底 torchaudio 抽取")
            try:
                return _video_to_wav_torchaudio(video_path, wav_path, sr, ss, td)
            except Exception as e3:
                print(f"[audio] torchaudio 兜底也失败: {e3}")
                return False


def load_audio_tensor(wav_path: str, sr: int = 16000):
    """返回 [1, T] 波形 tensor（float32，单声道，重采样到 sr）。

    优先用 torchaudio；若其无可用后端（常见于未装 soundfile 的环境，
    torchaudio.list_audio_backends() 返回空），回退到标准库 `wave` 直接读
    PCM WAV——video_to_wav 用 ffmpeg 产出的正是 PCM WAV，无需任何额外依赖。
    这保证 prep 抽特征在缺 soundfile 的 conda 环境也能正常取音频。
    """
    import torch
    try:
        import torchaudio
        if torchaudio.list_audio_backends():
            wav, _ = torchaudio.load(wav_path)
            if wav.shape[0] > 1:
                wav = wav.mean(0, keepdim=True)
            return wav
    except Exception as e:
        print(f"[audio] torchaudio 不可用（{e}），回退 wave 读 PCM WAV", flush=True)
    # ── 回退：标准库 wave（仅支持 PCM WAV）──
    import wave
    with wave.open(wav_path, "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()          # 字节 / 样本
        fr = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)
    if sw == 2:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 1:
        samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 4:
        samples = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width {sw}")
    if nch > 1:
        samples = samples.reshape(-1, nch).mean(1)
    if fr != sr and fr > 0:
        # 简单线性重采样（video_to_wav 已强制 -ar 16000，通常 fr==sr 不触发）
        x_old = np.arange(len(samples))
        x_new = np.linspace(0, len(samples), int(round(len(samples) * sr / fr)), endpoint=False)
        samples = np.interp(x_new, x_old, samples).astype(np.float32)
    return torch.from_numpy(samples).unsqueeze(0).float()


# ----------------------------- WIRE: 特征编码器（已实现） -----------------------------
# 懒加载单例：重型依赖（torch/transformers/mediapipe）只在首次调用时 import，
# 因此未安装这些包时，common.py 的其它函数（解包/抽帧/打包）仍可独立运行。
#
# 维度契约（与 Sync-X 指南 §B.4 / dataset.py 对齐）：
#   extract_clip_frame     -> (512,)        CLIP ViT-B/16 CLS（vision_model+projection），每帧
#   extract_clip_frames    -> (N,512)       批量版（单次 CLIP 批量前向，GPU 并行，更快）
#   extract_lip_landmarks  -> ((112,112), (192,))   MediaPipe FaceLandmarker 唇点（64×3，_PERIORAL_64_IDS 唯一索引）
#   extract_audio_seq      -> (T_a, 768)    wav2vec2-base-960h last_hidden_state
#
# 模型（冻结，仅推理）：
#   CLIP  : openai/clip-vit-base-patch16        -> 512-D
#   wav2vec2: facebook/wav2vec2-base-960h       -> 768-D
#   MediaPipe: tasks.vision.FaceLandmarker（现代 Tasks API，mediapipe>=0.10，前向兼容，无需降级）
#              -> 478 关键点中含 64 个口周点（40 FACEMESH_LIPS + 24 周围口周，见 _PERIORAL_64_IDS），与 legacy FaceMesh 索引一致

_CLIP_NAME     = "openai/clip-vit-base-patch16"
_WAV2VEC_NAME  = "facebook/wav2vec2-base-960h"
_LIP_ROI_SIZE  = 112

# ── 推理精度控制（FP16 提速，不影响维度契约 / 落盘 / 训练）──────────────────
# Sync-X 离线抽特征只用冻结模型做前向（无梯度）。FP16 推理可提速 ~30–50%、显存↓50%；
# 落盘本就 .half()（pack_cache）、训练侧 .float() 读取（dataset.py）→ 不改动任何下游契约。
# 默认 "float16"（verify_fp16.py 已验证：三模态 cos>0.999 且零 NaN）；
# 若需回退 fp32，设环境变量 SYNCX_INFER_FP16=0（或 false/no/off）。
# 亦可在代码内用 common.set_infer_dtype("float16"/"float32") 覆盖（必须在构造编码器前调用）。
_INFER_DTYPE_STR = "float16"
if os.environ.get("SYNCX_INFER_FP16", "").lower() in ("0", "false", "no", "off"):
    _INFER_DTYPE_STR = "float32"

def set_infer_dtype(dtype_str: str) -> None:
    """覆盖全局推理精度：'float32'（默认）或 'float16'。必须在构造编码器前调用。"""
    global _INFER_DTYPE_STR
    assert dtype_str in ("float32", "float16"), f"unknown dtype {dtype_str!r}"
    _INFER_DTYPE_STR = dtype_str

def get_infer_dtype() -> str:
    """返回当前全局推理精度字符串（'float32' / 'float16'）。"""
    return _INFER_DTYPE_STR
_MAX_AUDIO_LEN = 16000 * 4                       # 4s @16k = 64000，T_a≈200

# ── MediaPipe FaceLandmarker（现代 Tasks API）唇点索引 ──────────────────
# mediapipe>=0.10 起删除了 legacy `solutions.face_mesh_connections.FACEMESH_LIPS`，
# 故此处硬编码其官方定义（来源：google/mediapipe v0.10.21 同文件），保证版本无关且精确。
# 40 条连接 → 40 个唯一唇点索引（与 legacy FaceMesh 的 FACEMESH_LIPS 索引完全一致），
# 此为「唇轮廓子集」；Sync-X 当前采用更密的 64 口周点（见 _PERIORAL_64_IDS）。
_FACEMESH_LIPS = frozenset([
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
    (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267),
    (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
    (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13),
    (13, 312), (312, 311), (311, 310), (310, 415), (415, 308),
])
_LIP_IDS_LEGACY40 = sorted({i for pair in _FACEMESH_LIPS for i in pair})   # 40 个唯一唇点索引（legacy/A-B 用）
assert len(_LIP_IDS_LEGACY40) == 40

# ── 64 口周点（BioLip 区域定义：9 下唇内 + 9 下唇外 + 22 上唇 + 24 周围口周）────
# 在 40 FACEMESH_LIPS 唇轮廓基础上，叠加 24 个周围口周点（取自同一 MediaPipe 478 点网格），
# 共 64 个唯一索引。更密的口周运动场 → 更细的伪造唇部动态捕捉（对齐 BioLip 的 64 点设计）。
# 注意：MediaPipe FaceLandmarker 每帧仍输出全部 478 点，64 仅是「我们挑选的子集」，无需新模型。
_PERIORAL_64_SURROUND = (1, 2, 19, 58, 94, 98, 120, 123, 132, 147, 149,
                          175, 176, 192, 199, 200, 201, 205, 206, 207,
                          213, 214, 216, 234)
_PERIORAL_64_IDS = sorted(set(_LIP_IDS_LEGACY40) | set(_PERIORAL_64_SURROUND))   # 64 个唯一口周点索引
assert len(_PERIORAL_64_IDS) == 64

# 当前默认唇点集（Sync-X v2 契约）：64 口周点 × 3 = 192-D v_lip。
_LIP_IDS = _PERIORAL_64_IDS

# FaceLandmarker 模型（Google 官方，float16，~3.6MB），首次调用自动下载到本地。
_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task")
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
_FACE_LANDMARKER_MODEL = os.environ.get(
    "SYNCX_FACE_LANDMARKER_MODEL",
    os.path.join(_CODE_DIR, "weights", "face_landmarker.task"))


def _ensure_face_landmarker_model() -> str:
    """返回 face_landmarker.task 路径；缺失则自动从 Google 官方源下载。"""
    import urllib.request
    path = _FACE_LANDMARKER_MODEL
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[WIRE] 下载 FaceLandmarker 模型 -> {path}", flush=True)
    urllib.request.urlretrieve(_FACE_LANDMARKER_URL, path)
    print("[WIRE] FaceLandmarker 模型下载完成", flush=True)
    return path


_visual_singleton: object | None = None
_audio_singleton: object | None = None


class _VisualEncoder:
    """冻结 CLIP ViT-B/16（仅 vision_model + visual_projection）+ MediaPipe FaceMesh。"""

    def __init__(self, device: str):
        import torch
        import mediapipe as mp
        from transformers import CLIPModel, CLIPImageProcessor
        self.device = device
        print(f"[WIRE] loading CLIP {_CLIP_NAME} on {device}", flush=True)
        # 2026-08-16 修复：hf-mirror 上 openai/clip-vit-base-patch16 仅有 pytorch_model.bin，
        # 无 safetensors；use_safetensors=True 会导致 from_pretrained 直接 OSError。
        # 改为 False（加载 .bin），不影响特征契约（同为 fp32 权重）。
        clip_full = CLIPModel.from_pretrained(_CLIP_NAME, use_safetensors=True)
        self.clip = clip_full.vision_model.to(device).eval()        # ViT only
        self.cproj = clip_full.visual_projection.to(device).eval()  # 768 -> 512
        for p in self.clip.parameters():  p.requires_grad = False
        for p in self.cproj.parameters(): p.requires_grad = False
        # ── FP16 推理（可选，提速 + 省显存；落盘/训练契约不变）──
        self.dtype = torch.float16 if _INFER_DTYPE_STR == "float16" else torch.float32
        if self.dtype == torch.float16:
            self.clip = self.clip.half()
            self.cproj = self.cproj.half()
        # CLIP 归一化常数（输入需先 /255 到 [0,1] 再减均值除标准差）
        proc = CLIPImageProcessor.from_pretrained(_CLIP_NAME)
        self._mean = torch.tensor(proc.image_mean).view(1, 3, 1, 1).to(device)
        self._std = torch.tensor(proc.image_std).view(1, 3, 1, 1).to(device)
        if self.dtype == torch.float16:
            self._mean = self._mean.half()
            self._std = self._std.half()
        # MediaPipe FaceLandmarker（现代 Tasks API；SYNCX_MEDIAPIPE_GPU=1 时尝试 GPU，失败自动回退 CPU）
        self._detector = _build_face_landmarker("IMAGE")
        self._lip_ids = _LIP_IDS

    def _detect(self, frame_bgr):
        import cv2
        import mediapipe as mp
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_img)
        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]   # 该人脸的 478 个 NormalizedLandmark（.x/.y/.z）

    @staticmethod
    def _face_crop(frame_bgr, landmarks):
        import cv2
        h, w = frame_bgr.shape[:2]
        xs = [int(lm.x * w) for lm in landmarks]
        ys = [int(lm.y * h) for lm in landmarks]
        x1, x2 = max(min(xs) - 20, 0), min(max(xs) + 20, w)
        y1, y2 = max(min(ys) - 30, 0), min(max(ys) + 20, h)
        face = frame_bgr[y1:y2, x1:x2]
        if face.size == 0:
            face = frame_bgr
        face = cv2.resize(face, (224, 224))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        return face

    @staticmethod
    def _lip_crop(frame_bgr, landmarks, lip_ids):
        import cv2
        h, w = frame_bgr.shape[:2]
        xs = [int(landmarks[i].x * w) for i in lip_ids]
        ys = [int(landmarks[i].y * h) for i in lip_ids]
        x1, x2 = max(min(xs) - 10, 0), min(max(xs) + 10, w)
        y1, y2 = max(min(ys) - 10, 0), min(max(ys) + 10, h)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((_LIP_ROI_SIZE, _LIP_ROI_SIZE), dtype=np.float32)
        crop = cv2.resize(crop, (_LIP_ROI_SIZE, _LIP_ROI_SIZE))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        return crop

    def encode_clip(self, frame_bgr) -> np.ndarray:
        import torch
        lms = self._detect(frame_bgr)
        if lms is None:
            return np.zeros(512, dtype=np.float32)
        face = self._face_crop(frame_bgr, lms)
        with torch.no_grad():
            x = torch.from_numpy(face).permute(2, 0, 1).float().unsqueeze(0) / 255.0  # (1,3,224,224)
            x = x.to(self.device)
            if self.dtype == torch.float16:
                x = x.half()
            x = (x - self._mean) / self._std
            out = self.clip(pixel_values=x).pooler_output        # (1,768)
            out = self.cproj(out)                                # (1,512)
            feat = out.squeeze(0).float().cpu().numpy().astype(np.float32)  # (512,)
        return feat

    def encode_clip_batch(self, frames_bgr) -> np.ndarray:
        """批量 BGR 帧 -> 冻结 CLIP ViT-B/16 -> (N,512)。

        与 encode_clip 语义一致，但把逐帧裁好的脸一次性喂 CLIP，省掉逐帧 kernel
        launch 并利用 GPU 并行，显著提速（prep_deepspeak 每片段 50 帧场景）。
        MediaPipe 人脸检测仍逐帧（CPU，Tasks API 不支持批量）。

        frames_bgr: list[np.ndarray(H,W,3)]。返回 np.ndarray (N,512)。
        _detect 失败的帧对应行返回零向量（与单帧 encode_clip 行为一致）。"""
        import torch
        # 1) 逐帧 MediaPipe 检测 + 裁剪（CPU，必须逐帧）
        crops = []
        for f in frames_bgr:
            lms = self._detect(f)
            crops.append(self._face_crop(f, lms) if lms is not None else None)
        # 2) 堆叠为 batch 张量 (N,3,224,224)，失败帧填零
        tensors = []
        for c in crops:
            if c is None:
                tensors.append(torch.zeros(3, 224, 224))
            else:
                tensors.append(torch.from_numpy(c).permute(2, 0, 1).float() / 255.0)
        x = torch.stack(tensors, 0).to(self.device)              # (N,3,224,224)
        if self.dtype == torch.float16:
            x = x.half()
        x = (x - self._mean) / self._std                         # 广播 (1,3,1,1)
        with torch.no_grad():
            out = self.clip(pixel_values=x).pooler_output        # (N,768)
            out = self.cproj(out)                                # (N,512)
            feats = out.float().cpu().numpy().astype(np.float32) # (N,512)
        # 3) 失败帧置零（与单帧行为一致）
        for i, c in enumerate(crops):
            if c is None:
                feats[i] = 0.0
        return feats

    def encode_lip(self, frame_bgr):
        lms = self._detect(frame_bgr)
        if lms is None:
            return (np.zeros((_LIP_ROI_SIZE, _LIP_ROI_SIZE), dtype=np.float32),
                    np.zeros(len(self._lip_ids) * 3, dtype=np.float32))
        crop = self._lip_crop(frame_bgr, lms, self._lip_ids)
        coord = np.array([v for i in self._lip_ids for v in (lms[i].x, lms[i].y, lms[i].z)],
                         dtype=np.float32)  # (192,)
        return crop, coord


class _AudioEncoder:
    """冻结 wav2vec2-base-960h。"""

    def __init__(self, device: str):
        import torch
        from transformers import Wav2Vec2Model
        self.device = device
        print(f"[WIRE] loading wav2vec2 {_WAV2VEC_NAME} on {device}", flush=True)
        self.model = Wav2Vec2Model.from_pretrained(_WAV2VEC_NAME, use_safetensors=True).to(device).eval()
        self.dtype = torch.float16 if _INFER_DTYPE_STR == "float16" else torch.float32
        if self.dtype == torch.float16:
            self.model = self.model.half()
        for p in self.model.parameters():
            p.requires_grad = False

    def encode(self, wav_tensor) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        wav = wav_tensor
        if not hasattr(wav, "dim"):
            wav = torch.as_tensor(np.asarray(wav, dtype=np.float32))
        if wav.dim() > 1:
            wav = wav.mean(0, keepdim=True)          # 多声道 -> 单声道
        wav = wav.squeeze(0) if wav.dim() > 1 else wav
        if wav.numel() < _MAX_AUDIO_LEN:
            wav = F.pad(wav, (0, _MAX_AUDIO_LEN - wav.numel()))
        else:
            wav = wav[:_MAX_AUDIO_LEN]
        wav = (wav - wav.mean()) / (wav.std() + 1e-7)   # per-utterance z-score
        wav = wav.unsqueeze(0).to(self.device)
        if self.dtype == torch.float16:
            wav = wav.half()
        with torch.no_grad():
            feat = self.model(wav).last_hidden_state.squeeze(0)  # (T_a, 768)
        return feat.float().cpu().numpy().astype(np.float32)


def _resolve_device(device: str) -> str:
    """默认 'cpu' 时若有 CUDA 则自动升 'cuda'（prep 脚本未显式传 device）。"""
    if device in (None, "cpu", ""):
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"
    return device


def _get_visual(device: str):
    global _visual_singleton
    device = _resolve_device(device)
    if _visual_singleton is None:
        _visual_singleton = _VisualEncoder(device)
    return _visual_singleton


def _get_audio(device: str):
    global _audio_singleton
    device = _resolve_device(device)
    if _audio_singleton is None:
        _audio_singleton = _AudioEncoder(device)
    return _audio_singleton


def extract_clip_frame(frame_bgr, device="cpu") -> np.ndarray:
    """单帧 BGR -> 冻结 CLIP ViT-B/16 -> 512-D。返回 np.ndarray (512,)。"""
    return _get_visual(device).encode_clip(frame_bgr)


def extract_clip_frames(frames_bgr, device="cpu") -> np.ndarray:
    """批量 BGR 帧 -> 冻结 CLIP ViT-B/16 -> (N,512)。
    比逐帧 extract_clip_frame 更快（单次 CLIP 批量前向，GPU 并行）。
    frames_bgr: list[np.ndarray(H,W,3)]。返回 np.ndarray (N,512)。"""
    return _get_visual(device).encode_clip_batch(frames_bgr)


def extract_lip_landmarks(frame_bgr, device="cpu"):
    """MediaPipe FaceLandmarker -> _PERIORAL_64_IDS 64 点 -> 紧唇 ROI 112x112 + 展平 R^192。
    返回 (v_crops: np.ndarray (112,112), v_lip: np.ndarray (192,))。"""
    return _get_visual(device).encode_lip(frame_bgr)


# ───────────────────────── 序列级唇 landmark（VIDEO 模式 tracking） ─────────────────────────
# 单帧 extract_lip_landmarks 用 IMAGE 模式，逐帧独立 detect（慢）。对一段连续视频，
# 改用 VIDEO 模式 FaceLandmarker：相邻帧靠 tracking 初始化，比逐帧重检测快；首帧/检测
# 失败/人脸短暂丢失时回退上一帧（静止假设），保证输出连续且 shape 不变
# （crop (112,112) + coord (192,)）。每个片段用独立 detector 实例（ts 从 0 递增），
# 杜绝跨片段 tracking 污染。coord 拼装逻辑与 encode_lip 逐字一致，零契约回归。
class _LipVideoProcessor:
    """VIDEO 模式 FaceLandmarker 跨帧 tracking 封装（detector / _detect 可注入，便于无模型单测）。"""

    def __init__(self, lip_ids):
        self._lip_ids = lip_ids
        self._last_lms = None

    def process(self, frames_bgr, detector, _detect=None, _ts0=0):
        """frames_bgr: list[np.ndarray(H,W,3)] -> list[landmarks|None]。
        _detect(detector, frame_bgr, ts_ms) -> landmarks|None 可注入（测试用）；
        默认走 mediapipe VIDEO 模式 detect_for_video（需模型，本机运行）。
        _ts0：起始时间戳（复用跨片段 detector 时保证 ts 严格递增）。"""
        if _detect is None:
            import mediapipe as mp
            def _detect(d, frame_bgr, ts_ms):
                import cv2
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = d.detect_for_video(mp_img, ts_ms)
                return res.face_landmarks[0] if res.face_landmarks else None
        out = []
        ts_ms = _ts0
        for f in frames_bgr:
            ts_ms += 33  # ~30fps 步长；VIDEO 模式要求严格递增时间戳
            lms = _detect(detector, f, ts_ms)
            if lms is None:
                lms = self._last_lms  # 人脸丢失 → 沿用上一帧（静止假设）
            self._last_lms = lms
            out.append(lms)
        return out


def _build_face_landmarker(mode: str):
    """构造 FaceLandmarker（现代 Tasks API）。
    SYNCX_MEDIAPIPE_GPU=1（或 true/yes/on）时优先 GPU delegate，创建失败自动回退 CPU。
    mode: "IMAGE" 或 "VIDEO"；输出 478 点（含 _PERIORAL_64_IDS 64 口周点），GPU/CPU 同一模型，零契约差异。"""
    from mediapipe.tasks import python as tasks_python
    from mediapipe.tasks.python import vision
    model_path = _ensure_face_landmarker_model()
    running_mode = vision.RunningMode.IMAGE if mode == "IMAGE" else vision.RunningMode.VIDEO

    def _create(delegate=None):
        # MediaPipe 线程控制：BaseOptions 的 num_threads 参数在不同版本 API 有差异
        # （mediapipe 1.0.0 无此参数，新版有）。先探测签名，支持才传，兼容两种版本。
        # 多进程分片时 cloud_run_all.sh 已注入 OMP_NUM_THREADS 限制 torch/ffmpeg；
        # MediaPipe 内部 XNNPACK 线程池若支持 num_threads 则一并限制，不支持则忽略。
        _base_kw = {"model_asset_path": model_path}
        if delegate:
            _base_kw["delegate"] = delegate
        _nt = os.environ.get("OMP_NUM_THREADS", "")
        if _nt:
            try:
                import inspect as _inspect
                _params = _inspect.signature(tasks_python.BaseOptions.__init__).parameters
                if "num_threads" in _params:
                    _base_kw["num_threads"] = int(_nt)
            except Exception:
                pass
        base = tasks_python.BaseOptions(**_base_kw)
        opts = vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=running_mode,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        return vision.FaceLandmarker.create_from_options(opts)

    use_gpu = os.environ.get("SYNCX_MEDIAPIPE_GPU", "0").lower() in ("1", "true", "yes", "on")
    if use_gpu:
        try:
            lm = _create(tasks_python.BaseOptions.Delegate.GPU)
            print("[WIRE] MediaPipe FaceLandmarker 使用 GPU delegate", flush=True)
            return lm
        except Exception as e:
            print(f"[WIRE] MediaPipe GPU delegate 不可用，回退 CPU: {e}", flush=True)
    return _create(None)


def _create_video_landmarker():
    """创建 VIDEO 模式 FaceLandmarker（每片段独立实例，ts 从 0 递增）。"""
    return _build_face_landmarker("VIDEO")


def _lip_seq_postprocess(frames_bgr, lms_list, enc):
    """把 landmark 列表转成 (crop(112,112), coord(192,)) 对；无人脸帧返回零向量。"""
    out = []
    for f, lms in zip(frames_bgr, lms_list):
        if lms is None:
            out.append((np.zeros((_LIP_ROI_SIZE, _LIP_ROI_SIZE), dtype=np.float32),
                        np.zeros(len(enc._lip_ids) * 3, dtype=np.float32)))
        else:
            crop = enc._lip_crop(f, lms, enc._lip_ids)
            coord = np.array([v for i in enc._lip_ids
                              for v in (lms[i].x, lms[i].y, lms[i].z)], dtype=np.float32)
            out.append((crop, coord))
    return out


def extract_lip_landmarks_seq(frames_bgr, device="cpu", detector=None, _detect=None):
    """序列级唇 landmark 抽取（VIDEO 模式 tracking，显著快于逐帧 IMAGE）。

    frames_bgr: list[np.ndarray(H,W,3)]。返回 list[(v_crop(112,112), v_lip(192,))]。
    第 i 个元素与单帧 extract_lip_landmarks(frames_bgr[i]) 语义一致（同模型、同 64 点、
    同 coord 拼装逻辑），仅检测策略不同（tracking vs 每帧重检测）。失败/无人脸帧返回零向量。
    detector / _detect 可注入（测试用）；生产默认每片段新建 VIDEO 实例（ts 从 0 递增），
    杜绝跨片段 tracking 污染（VIDEO 模式检测器跨不连续 clip 复用会泄漏时序状态 → 改 192-D
    数值且顺序依赖、不可复现）。

    此 API 只改变抽特征速度，不改动任何下游维度契约（仍为 (112,112)+(192,)）。"""
    enc = _get_visual(device)
    if detector is None:
        detector = _create_video_landmarker()   # 每 clip 新建：跨 clip 零污染、结果可复现
    proc = _LipVideoProcessor(enc._lip_ids)
    lms_list = proc.process(frames_bgr, detector, _detect=_detect, _ts0=0)
    return _lip_seq_postprocess(frames_bgr, lms_list, enc)


def extract_audio_seq(wav_tensor, device="cpu") -> np.ndarray:
    """wav2vec2-base-960h -> R^{T_a x 768}。输入 wav_tensor 为 [1,T] 波形（torch.Tensor）。
    返回 np.ndarray (T_a, 768)。"""
    return _get_audio(device).encode(wav_tensor)


# ----------------------------- 缓存打包 -----------------------------
def pack_cache(stem: str, out_root: str, *, audio_seq, v_clip, v_lip, label, identity,
               source_id: str = "", v_crops=None, flat: bool = False):
    """把单视频特征存为半精度 tensor 缓存 + meta.json。

    - flat=False（默认，旧行为）：写 <out_root>/<stem>/cache.pt + meta.json。
    - flat=True：写 <out_root>/<stem>.pt（扁平），供 dataset.FeatureCacheDataset
      的 glob(cache_dir/*.pt) 直接消费（train_stage1/2 均用扁平扫描）。
    - source_id：Rec C 同源配对键（dataset.py 从 cache.pt 读取；缺省空串=配对退化 i.i.d.）。
    - v_crops：可选。Sync-X 模型不消费唇 ROI（仅用 64×3 唇点坐标 v_lip），
      故默认不写，省 ~33GB/数据集。如需旧 PCA 流可显式传入。
    """
    import torch
    payload = {
        "audio_seq": torch.as_tensor(np.asarray(audio_seq)).half(),
        "v_clip": torch.as_tensor(np.asarray(v_clip)).half(),
        "v_lip": torch.as_tensor(np.asarray(v_lip)).half(),
        "label": int(label),
        "identity": str(identity),
        "source_id": str(source_id),
    }
    if v_crops is not None:
        payload["v_crops"] = torch.as_tensor(np.asarray(v_crops)).half()
    meta = {"stem": stem, "label": int(label), "identity": str(identity), "source_id": str(source_id)}
    if flat:
        out_root = Path(out_root)
        out_root.mkdir(parents=True, exist_ok=True)
        final = out_root / f"{stem}.pt"
        tmp = out_root / f"{stem}.pt.tmp"          # 先写临时文件
        torch.save(payload, tmp)
        os.replace(tmp, final)                     # 原子改名：中断只留 .tmp（被 resume 忽略），绝不产生半截 .pt
        return final
    d = Path(out_root) / stem
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "cache.pt.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, d / "cache.pt")                # 同上，原子写
    (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False))
    return d / "cache.pt"


def write_manifest(rows: list[dict], out_root: str, name: str = "manifest.jsonl"):
    p = Path(out_root) / name
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[manifest] wrote {len(rows)} rows -> {p}")
    return p


# ── 轻量进度打印（无外部依赖；各 prep 抽取循环调用）──
def log_progress(prefix, done, total, t0, report_every=None):
    """打印一行进度（ASCII 条 + 百分比 + ETA）。prefix 区分进程/分片。
    done=已完成数, total=本分片总数, t0=起始 time.time()。
    每 report_every 个或收尾打印一次；默认 total//50（约每 2%）。"""
    if total <= 0:
        return
    if report_every is None:
        report_every = max(1, total // 50)
    if done > 0 and (done % report_every == 0 or done >= total):
        el = time.time() - t0
        rate = done / el if el > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else 0.0
        filled = done * 20 // total
        bar = "#" * filled + "-" * (20 - filled)
        print(f"{prefix} [{bar}] {done}/{total} ({100.0*done/total:.0f}%) eta={eta:.0f}s",
              flush=True)


if __name__ == "__main__":
    print("common.py loaded OK — import from prep_*.py scripts.")
