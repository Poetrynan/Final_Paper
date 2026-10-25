"""
==============================================================================
extract_features.py  —  Offline feature extraction for Sync-X v4.0
==============================================================================
-------------------------------------------------------------------------------
⚠️ DEPRECATED（2026-08-08 起）：本脚本是 WIRE 改造前的旧版离线提取器，已被
`common.py` 的 extract_clip_frame / extract_lip_landmarks / extract_audio_seq
（经 prep_*.py 调用）取代。两点不兼容当前代码契约：
  1) 它使用已废弃的 `mediapipe.solutions` API（mediapipe>=1.0 已删除），在当前
     thesis 环境（mediapipe 1.0.0）会直接报错；
  2) 它输出 **v_lip = 120-D（40 唇点×3，旧 FACEMESH_LIPS 子集）**（其内部注释曾误写 240-D），
     而当前维度契约为 **192-D（64 口周点×3）**，与 dataset.py / models.py / config.py 的 lip_lm_dim=192 不一致。
新训练/评测请改用 prep_*.py（依赖 common.py）。本文件仅作历史参考，请勿再用于生产缓存。
-------------------------------------------------------------------------------
离线特征提取脚本（解耦策略的核心）
将 Wav2Vec2 和 CLIP-ViT 这两个大模型冻结，对每个视频只跑一次推理，
把提取好的特征以 .pt 文件形式缓存起来，后续训练时 GPU 只需加载轻量级适配器。

For each video in the metadata CSV, this script saves a single .pt file with:
   audio_seq : (T_a, 768)  float16   wav2vec2-base last hidden state
   v_clip    : (T_v, 512)  float16   CLIP ViT-B/16 image embedding (CLS)
   v_lip     : (T_v, 192)  float16   MediaPipe lip landmark coords (64×3)
   v_crops   : (T_v,112,112) float16 grayscale lip ROI (used for PCA stream)
   label     : int (0=real, 1=fake)
   identity  : str

──────────────────────────────────────────────────────────
 Colab / Kaggle 使用说明：
──────────────────────────────────────────────────────────
 1. 在 Notebook 第一个 cell 中挂载 Google Drive：
        from google.colab import drive
        drive.mount('/content/drive')

 2. 运行本脚本（只需运行一次！生成的 .pt 会永久保存在 Drive 中）：
        !python extract_features.py \
            --metadata_csv /content/drive/MyDrive/SyncX/data/FakeAVCeleb/meta_data.csv \
            --data_root    /content/drive/MyDrive/SyncX/data/FakeAVCeleb \
            --output_dir   /content/drive/MyDrive/SyncX/feature_cache

 3. 下次重新连接 Colab 时，不需要重新跑特征提取，
    因为 .pt 文件已经安全保存在 Google Drive 中了。
==============================================================================
"""
import os, argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as AT
import cv2
import mediapipe as mp
from moviepy.editor import VideoFileClip
from transformers import Wav2Vec2Model, CLIPModel, CLIPImageProcessor

# ─────────────────────────────────────────────────────────────────────────────
# Google Drive 根路径配置
# ─────────────────────────────────────────────────────────────────────────────
# 如果在 Colab/Kaggle 上运行，请将 DRIVE_ROOT 改为你的 Google Drive 挂载路径。
# 如果在本地运行，保持 "" 即可（会使用相对路径）。
#
# 示例（Colab）：DRIVE_ROOT = "/content/drive/MyDrive/SyncX"
# 示例（本地）：  DRIVE_ROOT = ""
# ─────────────────────────────────────────────────────────────────────────────
DRIVE_ROOT = ""  # ← 在 Colab 上改成 "/content/drive/MyDrive/SyncX"

# ------------------------- Config (must match Sync-X.py) ---------------------
AUDIO_SR        = 16000
CLIP_SECONDS    = 4.0
MAX_AUDIO_LEN   = int(AUDIO_SR * CLIP_SECONDS)        # 64 000
VIDEO_FPS       = 25
MAX_VID_FRAMES  = int(VIDEO_FPS * CLIP_SECONDS)       # 100
LIP_ROI_SIZE    = 112
WAV2VEC2_NAME   = "facebook/wav2vec2-base-960h"
CLIP_NAME       = "openai/clip-vit-base-patch16"      # 512-D output


# =============================================================================
# Audio backbone  (wav2vec 2.0 — frozen)
# =============================================================================
class AudioBackbone:
    def __init__(self, device):
        self.device = device
        print(f"[AudioBackbone] loading {WAV2VEC2_NAME} on {device}")
        self.model = Wav2Vec2Model.from_pretrained(WAV2VEC2_NAME).to(device).eval()
        for p in self.model.parameters(): p.requires_grad = False

    @torch.no_grad()
    def __call__(self, wav_path):
        wav, sr = torchaudio.load(str(wav_path))
        if wav.size(0) > 1: wav = wav.mean(0, keepdim=True)
        if sr != AUDIO_SR:  wav = AT.Resample(sr, AUDIO_SR)(wav)
        wav = wav.squeeze(0)
        if wav.numel() < MAX_AUDIO_LEN:
            wav = F.pad(wav, (0, MAX_AUDIO_LEN - wav.numel()))
        else:
            wav = wav[:MAX_AUDIO_LEN]
        # zero-mean / unit-var per-utterance (wav2vec2 convention)
        wav = (wav - wav.mean()) / (wav.std() + 1e-7)
        wav = wav.unsqueeze(0).to(self.device)
        feat = self.model(wav).last_hidden_state.squeeze(0).cpu()  # (T_a, 768)
        return feat


# =============================================================================
# Visual backbone  (CLIP ViT-B/16 — frozen) + MediaPipe lip stream
# =============================================================================
class VisualBackbone:
    LIP_IDS = sorted({i for pair in mp.solutions.face_mesh.FACEMESH_LIPS for i in pair})

    def __init__(self, device):
        self.device = device
        print(f"[VisualBackbone] loading {CLIP_NAME} on {device}")
        clip_full   = CLIPModel.from_pretrained(CLIP_NAME)
        self.clip   = clip_full.vision_model.to(device).eval()        # ViT only
        self.cproj  = clip_full.visual_projection.to(device).eval()   # 768→512
        for p in self.clip.parameters():  p.requires_grad = False
        for p in self.cproj.parameters(): p.requires_grad = False
        self.processor = CLIPImageProcessor.from_pretrained(CLIP_NAME)
        # CLIP normalization constants
        self.mean = torch.tensor(self.processor.image_mean
                                 ).view(1,3,1,1).to(device)
        self.std  = torch.tensor(self.processor.image_std
                                 ).view(1,3,1,1).to(device)
        # MediaPipe FaceMesh
        print("[VisualBackbone] loading MediaPipe FaceMesh")
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    # ---- per-frame helpers --------------------------------------------------
    def _face_crop_for_clip(self, frame_bgr, landmarks):
        h, w = frame_bgr.shape[:2]
        xs = [int(lm.x * w) for lm in landmarks]
        ys = [int(lm.y * h) for lm in landmarks]
        x1, x2 = max(min(xs)-20,0), min(max(xs)+20,w)
        y1, y2 = max(min(ys)-30,0), min(max(ys)+20,h)
        face = frame_bgr[y1:y2, x1:x2]
        if face.size == 0:
            face = frame_bgr
        face = cv2.resize(face, (224, 224))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        return face

    def _lip_crop(self, frame_bgr, landmarks):
        h, w = frame_bgr.shape[:2]
        xs = [int(landmarks[i].x*w) for i in self.LIP_IDS]
        ys = [int(landmarks[i].y*h) for i in self.LIP_IDS]
        x1, x2 = max(min(xs)-10,0), min(max(xs)+10,w)
        y1, y2 = max(min(ys)-10,0), min(max(ys)+10,h)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((LIP_ROI_SIZE, LIP_ROI_SIZE), dtype=np.float32)
        crop = cv2.resize(crop, (LIP_ROI_SIZE, LIP_ROI_SIZE))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        return crop

    # ---- main entry ---------------------------------------------------------
    @torch.no_grad()
    def __call__(self, video_path):
        # ── 硬拒绝：本脚本已废弃，且会产出 120-D v_lip（40 唇点），与当前 192-D（64 口周点）
        #    训练契约不兼容。若被误用，会静默生成错误维度的缓存、浪费数小时抽取。
        #    正确的抽取入口是 prep_*.py（调用 common.extract_lip_landmarks_seq，单一真相源）。
        raise RuntimeError(
            "extract_features.py 已废弃，拒绝执行。它产出 120-D v_lip（40 唇点），"
            "与当前 192-D（64 口周点, common._PERIORAL_64_IDS）训练契约不兼容。"
            "请用 prep_*.py（依赖 common.extract_lip_landmarks_seq）重新抽取特征缓存。")
        cap = cv2.VideoCapture(str(video_path))
        face_imgs, lip_crops, lip_coords = [], [], []
        n = 0
        while cap.isOpened() and n < MAX_VID_FRAMES:
            ok, frame = cap.read()
            if not ok: break
            res = self.mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if res.multi_face_landmarks:
                lms = res.multi_face_landmarks[0].landmark
                face_imgs.append(self._face_crop_for_clip(frame, lms))
                lip_crops.append(self._lip_crop(frame, lms))
                coord = []
                for i in self.LIP_IDS:
                    coord.extend([lms[i].x, lms[i].y, lms[i].z])
                lip_coords.append(np.array(coord, dtype=np.float32))
            else:
                face_imgs.append(np.zeros((224,224,3), dtype=np.uint8))
                lip_crops.append(np.zeros((LIP_ROI_SIZE,LIP_ROI_SIZE), dtype=np.float32))
                lip_coords.append(np.zeros(len(self.LIP_IDS)*3, dtype=np.float32))
            n += 1
        cap.release()
        if n == 0:
            raise RuntimeError("no frames decoded")

        # pad to MAX_VID_FRAMES
        while len(face_imgs) < MAX_VID_FRAMES:
            face_imgs.append(face_imgs[-1])
            lip_crops.append(lip_crops[-1])
            lip_coords.append(lip_coords[-1])

        # ---- CLIP ViT in batched mode (T,3,224,224) ------------------------
        x = torch.from_numpy(np.stack(face_imgs)).permute(0,3,1,2).float() / 255.0
        x = x.to(self.device)
        x = (x - self.mean) / self.std
        # process in mini-chunks to fit T4 memory
        feats = []
        for i in range(0, x.size(0), 25):
            chunk = x[i:i+25]
            out = self.clip(pixel_values=chunk).pooler_output      # (b,768)
            out = self.cproj(out)                                  # (b,512)
            feats.append(out.cpu())
        v_clip = torch.cat(feats, dim=0)                           # (T_v,512)

        v_lip   = torch.from_numpy(np.stack(lip_coords))           # (T_v,120)  ← 错误维度！见 __call__ 拒绝说明
        v_crops = torch.from_numpy(np.stack(lip_crops))            # (T_v,112,112)
        return v_clip, v_lip, v_crops


# =============================================================================
# Audio extraction helper
# =============================================================================
def extract_wav(video_path, wav_path):
    try:
        clip = VideoFileClip(str(video_path))
        if clip.audio is None:
            clip.close(); return False
        clip.audio.write_audiofile(str(wav_path), fps=AUDIO_SR,
                                   codec='pcm_s16le', verbose=False, logger=None)
        clip.close()
        return True
    except Exception as e:
        print(f"  audio fail: {e}")
        return False


# =============================================================================
# Main
# =============================================================================
def main():
    # ── 构建默认路径（自动适配本地 / Google Drive） ──
    _join = lambda *p: os.path.join(DRIVE_ROOT, *p) if DRIVE_ROOT else os.path.join(*p)

    ap = argparse.ArgumentParser(description="Sync-X 离线特征提取（只需运行一次）")
    ap.add_argument('--metadata_csv', default=_join('data', 'FakeAVCeleb', 'meta_data.csv'),
                    help='数据集的 CSV 元数据文件路径')
    ap.add_argument('--data_root',    default=_join('data', 'FakeAVCeleb'),
                    help='视频文件的根目录')
    ap.add_argument('--output_dir',   default=_join('feature_cache'),
                    help='生成的 .pt 特征缓存保存目录（建议存 Google Drive，防止丢失）')
    ap.add_argument('--device',       default='cuda',
                    help='推理设备：cuda 或 cpu')
    ap.add_argument('--max_videos',   type=int, default=None,
                    help='只处理前 N 个视频（调试用）')
    args = ap.parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA unavailable, fallback to CPU"); args.device = 'cpu'

    # 创建输出目录（如果指向 Google Drive，会自动在 Drive 中创建）
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_tmp_audio"; tmp.mkdir(exist_ok=True)    # 临时 WAV 文件（提取完会自动删除）

    df = pd.read_csv(args.metadata_csv)
    if 'identity' not in df.columns:
        df['identity'] = df['video_path'].apply(
            lambda p: Path(p).parent.name if '/' in str(p) else 'unknown')
    print(f"[main] {len(df)} videos in metadata")

    audio_bb  = AudioBackbone(args.device)
    visual_bb = VisualBackbone(args.device)

    n_ok = n_skip = n_fail = 0
    pbar = tqdm(df.iterrows(), total=len(df) if not args.max_videos else args.max_videos)
    for idx, row in pbar:
        if args.max_videos and idx >= args.max_videos: break
        stem = Path(row['video_path']).stem
        cache_path = out / f"{stem}.pt"
        if cache_path.exists(): n_skip += 1; continue

        vp = Path(args.data_root) / row['video_path']
        if not vp.exists(): vp = Path(row['video_path'])
        if not vp.exists():
            n_fail += 1; continue

        wav_path = tmp / f"{stem}.wav"
        if not wav_path.exists() and not extract_wav(vp, wav_path):
            n_fail += 1; continue

        try:
            audio_seq                = audio_bb(wav_path)               # (T_a,768)
            v_clip, v_lip, v_crops   = visual_bb(vp)                    # see above
            torch.save({
                'audio_seq': audio_seq.half(),
                'v_clip'   : v_clip.half(),
                'v_lip'    : v_lip.half(),
                'v_crops'  : v_crops.half(),
                'label'    : int(row['label']),
                'identity' : str(row['identity']),
            }, cache_path)
            n_ok += 1
            wav_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"  fail {stem}: {e}"); n_fail += 1

    try: tmp.rmdir()
    except: pass
    print(f"\n[done] ok={n_ok}  skip={n_skip}  fail={n_fail}")
    print(f"[done] cache → {out}")


if __name__ == "__main__":
    main()