"""
dataset.py — Sync-X 特征缓存数据集
================================================================
消费 common.py 的 extract_clip_frame / extract_lip_landmarks / extract_audio_seq
（经 prep_*.py 调用）产出的 .pt 缓存：
    audio_seq : (T_a, 768)   wav2vec2
    v_clip    : (T_v, 512)   CLIP
    v_lip     : (T_v, 192)   唇 landmark (64×3)
    label     : int (0=real, 1=fake)
    identity  : str

支持：
  - real_only        : 仅真实片段（Stage-1 SSL）
  - exclude_ids      : 剔除指定身份集合（B7 身份泄漏控制）
  - B6 静音捷径       : 加载 prep 产出的 b6_trimmed（Smeu 逐片段动态 trim 后重提）
                       或 b6_raw（未裁基线）扁平缓存；运行时不再做固定 N 零化
  - split/identity 文件: 划分与身份去重
"""
import os
import glob
import json
import random
import torch
from torch.utils.data import Dataset, Sampler

# 必须与 common.py 的 extract_* 输出一致（4s @ 25fps=100 帧；wav2vec2 50Hz=200 帧）
AUDIO_LEN = 200
VIS_LEN = 100

# 元数据索引文件（cache_dir 下）：预提取 label/identity/source_id，
# 避免初始化时为读几个标量而全量反序列化每个 .pt 的张量（Stage-1 曾因此 ~15min）。
INDEX_NAME = ".syncx_index.jsonl"


def _build_index(files):
    """全量扫描 .pt 提取元数据（label/identity/source_id）。

    加载失败的文件也占位（label=-1），保证 index 长度与 files 对齐，
    使 index 可复用、不会因个别损坏文件而每次重建。
    """
    index = []
    for f in files:
        try:
            meta = torch.load(f, map_location="cpu", weights_only=True)
            index.append({
                "path": f,
                "label": int(meta.get("label", -1)),
                "identity": str(meta.get("identity", "")),
                "source_id": str(meta.get("source_id", "")),
            })
        except Exception:
            index.append({"path": f, "label": -1, "identity": "", "source_id": ""})
    return index


def _atomic_write_index(idx_path, index):
    tmp = idx_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for ix in index:
            f.write(json.dumps(ix, ensure_ascii=False) + "\n")
    os.replace(tmp, idx_path)


def _load_or_build_index(files, cache_dir):
    """读现成 index（路径对齐则复用），否则构建并原子落盘。"""
    idx_path = os.path.join(cache_dir, INDEX_NAME)
    if os.path.exists(idx_path):
        try:
            with open(idx_path, "r", encoding="utf-8") as f:
                lines = [l for l in f if l.strip()]
            if len(lines) == len(files):
                index = [json.loads(l) for l in lines]
                if all(ix["path"] == fp for ix, fp in zip(index, files)):
                    return index
        except Exception:
            pass
    index = _build_index(files)
    _atomic_write_index(idx_path, index)
    return index


def _fix_len(x, L, dim=0):
    """沿 dim 把张量 pad/truncate 到长度 L。"""
    n = x.size(dim)
    if n == L:
        return x
    if n > L:
        idx = [slice(None)] * x.dim()
        idx[dim] = slice(0, L)
        return x[tuple(idx)]
    pad = [0] * (2 * x.dim())
    pad[2 * (x.dim() - 1 - dim) + 1] = L - n
    return torch.nn.functional.pad(x, pad)


class FeatureCacheDataset(Dataset):
    def __init__(self, cache_dir, T=50, real_only=False, exclude_ids=None,
                 file_list=None, temporal_jitter=False, jitter_crop_ratio=0.85):
        self.T = T
        self.real_only = real_only
        self.exclude_ids = set(exclude_ids) if exclude_ids else set()
        # [FIX 2026-08-09] 时间抖动（DeepFake Doctor arXiv:2506.05851）：训练时随机错开
        # 时间窗起点，打破"总是正中心 4s"的位置依赖捷径。仅训练集开启，验证/评测保持
        # 确定性。不影响已抽 .pt 缓存（裁剪在加载阶段做，缓存本身不变）。
        self.temporal_jitter = temporal_jitter
        self.jitter_crop_ratio = jitter_crop_ratio
        if file_list is not None:
            self.files = list(file_list)
            # 自定义文件列表（评测/子集）：直接全量扫描（列表通常小）
            index = _build_index(self.files)
        else:
            # [FIX 2026-08-15] 改为递归 glob：prep 端 pack_cache(flat=False) 写嵌套
            # <stem>/cache.pt，非递归 glob 会漏掉 4/5 数据集（仅 DeepSpeak flat=True 可读到）。
            self.files = sorted(glob.glob(os.path.join(cache_dir, "**", "*.pt"), recursive=True))
            index = _load_or_build_index(self.files, cache_dir)
        # 基于元数据索引过滤（不反序列化张量）
        self._index = index
        self.source_ids = []
        kept = []
        for ix in index:
            if ix["label"] < 0:
                continue            # 损坏/加载失败
            if real_only and ix["label"] != 0:
                continue
            if ix["identity"] in self.exclude_ids:
                continue
            kept.append(ix["path"])
            self.source_ids.append(ix["source_id"])
        self.files = kept

    def __len__(self):
        return len(self.files)

    def _jitter_temporal(self, audio_seq, v_clip, v_lip):
        """[FIX 2026-08-09] 随机时间裁剪（同一相对窗），保音/视/唇对齐。

        取 crop_ratio 比例的时间子窗，起点随机；三模态用同一相对起止，
        保证唇-音-视时间对齐。裁剪后由 __getitem__ 的 _fix_len 补零回固定长度。
        每个 epoch 因 DataLoader shuffle 重新调用而获得不同抖动。
        """
        crop = self.jitter_crop_ratio
        La, Lv = audio_seq.size(0), v_clip.size(0)
        if Lv <= 1 or La <= 1:
            return audio_seq, v_clip, v_lip
        new_Lv = max(1, int(round(crop * Lv)))
        start_v = random.randint(0, Lv - new_Lv)
        new_La = max(1, int(round(crop * La)))
        start_a = int(round(start_v / Lv * La))
        start_a = min(start_a, max(0, La - new_La))
        return (audio_seq[start_a:start_a + new_La],
                v_clip[start_v:start_v + new_Lv],
                v_lip[start_v:start_v + new_Lv])

    def __getitem__(self, idx):
        f = self.files[idx]
        d = torch.load(f, map_location="cpu", weights_only=True)
        audio_seq = d["audio_seq"].float()                 # (T_a, 768)
        v_clip = d["v_clip"].float()                       # (T_v, 512)
        v_lip = d["v_lip"].float()                         # (T_v, 192)
        label = int(d.get("label", 0))
        identity = str(d.get("identity", ""))
        source_id = str(d.get("source_id", ""))   # Rec C 同源配对用；缺省空串

        # [FIX 2026-08-09] 时间抖动：训练时随机裁剪同一相对时间窗，打散"总是正中心"
        # 的位置依赖。音/视/唇三模态用同一相对起止比例，保持时间对齐。验证/评测不开启。
        if self.temporal_jitter:
            audio_seq, v_clip, v_lip = self._jitter_temporal(audio_seq, v_clip, v_lip)

        # 固定长度（pad/truncate）
        audio_seq = _fix_len(audio_seq, AUDIO_LEN, dim=0)
        v_clip = _fix_len(v_clip, VIS_LEN, dim=0)
        v_lip = _fix_len(v_lip, VIS_LEN, dim=0)

        # B6 静音 trim 在 prep 阶段完成：trimmed 缓存已逐片段按检测时长
        # 裁掉开头/结尾静音并重提特征（论文 Ch4 B6 protocol）；raw 缓存为未裁基线。
        # 此处不再做运行时固定 N 零化，避免与逐片段动态 trim 冲突。

        return {
            "audio_seq": audio_seq,      # (200, 768)
            "v_clip": v_clip,            # (100, 512)
            "v_lip": v_lip,              # (100, 192)
            "label": label,
            "identity": identity,
            "source_id": source_id,
            "path": f,
        }


def load_identity_set(identity_file):
    """从文本文件（每行一个 identity）加载身份集合，供 B7 排除。"""
    if not identity_file or not os.path.exists(identity_file):
        return set()
    with open(identity_file, "r", encoding="utf-8") as f:
        return {ln.strip() for ln in f if ln.strip()}


def collect_source_ids(dataset):
    """返回与索引对齐的 source_id 列表（Rec C 配对用）。

    直接读 dataset 构造期维护的 source_ids（元数据索引），不再逐样本全量 load。
    仅当 cfg.same_source_sampling=True 时调用；数据无 source_id 时返回空串列表，
    此时 PairedBatchSampler 退化为无操作（调用方回退到标准 i.i.d. 采样）。
    """
    return dataset.source_ids


class PairedBatchSampler(Sampler):
    """把同 source_id 的片段尽量放进同一 mini-batch（Rec C 同源配对）。

    构造期按 source_id 对索引分组；__iter__ 贪心装桶，保证每个 batch
    至少含一个完整 source 组（同源 real/fake 共现）。若数据全无 source_id
    （空串），__len__ 返回 0、__iter__ 返回空，调用方应回退到标准 shuffle。
    """
    def __init__(self, source_ids, batch_size, shuffle=True, drop_last=True):
        self.source_ids = list(source_ids)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        groups = {}
        for i, s in enumerate(self.source_ids):
            if s:
                groups.setdefault(s, []).append(i)
        self.groups = list(groups.values())     # 仅含有效 source 组
        self.batches = self._pack()

    def _pack(self):
        groups = [list(g) for g in self.groups]
        if self.shuffle:
            random.shuffle(groups)
            for g in groups:
                random.shuffle(g)
        batches, cur, cur_n = [], [], 0
        for g in groups:
            if cur_n + len(g) <= self.batch_size:
                cur.extend(g); cur_n += len(g)
                if cur_n == self.batch_size:
                    batches.append(cur); cur, cur_n = [], 0
            else:
                if cur:
                    batches.append(cur); cur, cur_n = [], 0
                for st in range(0, len(g), self.batch_size):
                    batches.append(g[st:st + self.batch_size])
        if cur and not self.drop_last:
            batches.append(cur)
        return batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)
