r"""
models.py — Sync-X 模型（LCMA + LACA + DWF + AAM 分类头）
================================================================
架构（与论文 Ch3 一致，方向 B：对齐 thesis 叙述）：
  - 离线冻结特征：CLIP-ViT-B/16 (512-d) + wav2vec2-base-960h (768-d)
  - LCMA  : 双向跨模态注意力适配器（audio↔visual CLIP），d=256, h=4, L=2
            pooling = concat(pool(A); pool(V)) ∈ ℝ^{2d}
  - LACA   : 双信号唇探针（B′ 设计）—— 两条并行唇支路，拼接后经 laca_fuse 映射到 2d：
             支路(i) 唇–音频对应/失配（用音频，类 AV-Lip-Sync+ 的 Sync-Check）：
                     lip 表征 K 与 audio 表征先经投影头到共享空间 \hat{K}, \hat{a}，
                     再逐帧比对 |\hat{K}_t − \hat{a}_t|，暴露"唇形与音频对不对得上"；
                     满足 thesis 核心——LACA 重新消费音频，与 LCMA 形成不同粒度的 AV 一致性信号。
             支路(ii) biomech 抖动（无音频，BioLip 式）：仅用 perioral landmark 轨迹 P 的高阶
                     运动学(位置/速度/加速度/加加速度)，对生成器残留的非生理运动抖动(jitter)敏感；
                     该线索与音画同步无关、不被联合生成消掉，补 LCMA 在二代威胁上的盲区，
                     且与支路(i) 正交。cfg.laca_mode ∈ {both,corr,biomech} 控制消融。
  - DWF   : softmax 两标量门 (α_ℓ, α_p)，h = α_ℓ h_lcma + α_p h_laca；
            前 dwf_freeze_epochs 轮冻结为 LCMA（偏置初始化已 favour LCMA）
  - AAM   : Additive Angular Margin 分类头（替代 CE），输入 2d
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import factorial


def resample_temporal(x, T):
    """把时序插值到固定长度 T。x: (B, L, D) -> (B, T, D)。"""
    if x.size(1) == T:
        return x
    x = x.transpose(1, 2)                     # (B, D, L)
    x = F.interpolate(x, size=T, mode="linear", align_corners=False)
    return x.transpose(1, 2)                  # (B, T, D)


def savgol_coeffs(window, poly, deriv, delta):
    """Savitzky-Golay 系数（纯 numpy，等价 scipy.signal.savgol_coeffs(use='dot')）。

    返回长度 window 的向量；与滑窗点积即得中心点第 deriv 阶「平滑导数」。
    deriv=0 → 平滑值；deriv=1/2/3 → 平滑速度/加速度/加加速度(jerk)。
    delta 为采样间隔(1/fps)，使导数具有真实时间量纲(/s, /s², /s³)。
    """
    assert window % 2 == 1 and window > poly >= deriv, "window 须为奇数且 window>poly>=deriv"
    x = np.arange(window, dtype=np.float64) - (window // 2)          # 长度恒=window（修复 -window//2 地板除致 window+1；Kinematics 窗长回正）
    A = np.vander(x, poly + 1, increasing=True)          # (W, poly+1) Vandermonde
    pinv = np.linalg.pinv(A)                             # (poly+1, W) 最小二乘投影
    c = pinv[deriv] * factorial(deriv) / (delta ** deriv)
    return torch.from_numpy(c).float()                   # (W,)


def normalize_landmarks(P):
    """逐帧去质心 + RMS 尺度归一化（去平移/尺度，抑制头部运动泄漏进微分）。

    P: (B, T, C)，C = N×3（每个 landmark 的 x,y,z 展平）。索引无关：用全部 N 点
    的质心与均方根尺度，不依赖特定口角索引。返回同形状。
    """
    B, T, C = P.shape
    assert C % 3 == 0, "landmark 维须为 3 的倍数 (x,y,z)"
    Pm = P.view(B, T, C // 3, 3)                         # (B,T,N,3)
    P0 = Pm - Pm.mean(dim=-2, keepdim=True)              # 去每帧质心
    s = P0.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-6)  # 每帧 RMS 尺度
    return (P0 / s).view(B, T, C)


class Kinematics(nn.Module):
    """确定性运动学前端（无可学习参数）：Savitzky-Golay 平滑导数。

    对每条 landmark 轨迹先做 S-G 局部多项式拟合，再取 0..order 阶平滑导数，
    替代朴素有限差分，避免把 MediaPipe 高频量化噪声放大进 jerk（报告指出的问题）。

    设计要点（对齐 BioLip arXiv:2604.16808）：
      - 小窗(window≈5)+低阶(poly≈3)：只削 >信号带 的量化噪声，保住 1–8Hz 生理抖动带；
      - fps 感知：delta=1/fps，导数量纲为 /s、/s²、/s³，跨帧率可比；
      - real/jittered 走同一前端 → Stage-1 伪抖动自监督不受影响；
      - 全部系数为 conv1d 固定权重(buffer)，确定、与梯度无关、GPU 上廉价。

    输入 P: (B, T, C)；输出 (B, T, C*(order+1))，沿时间等长。
    """
    def __init__(self, channels, order=3, window=7, poly=3, fps=25.0, use_fps=True):
        super().__init__()
        self.order, self.channels, self.half = order, channels, window // 2
        delta = (1.0 / fps) if use_fps else 1.0
        for d in range(order + 1):
            c = savgol_coeffs(window, poly, deriv=d, delta=delta)
            # conv1d 权重 (C, 1, W)，groups=C → 每个通道独立做同一 S-G
            self.register_buffer(f"w{d}", c.view(1, 1, -1).repeat(channels, 1, 1))

    def forward(self, P):                                # P: (B, T, C)
        x = F.pad(P.transpose(1, 2), (self.half, self.half), mode="replicate")
        outs = [F.conv1d(x, getattr(self, f"w{d}"), groups=self.channels)
                for d in range(self.order + 1)]
        return torch.cat([o.transpose(1, 2) for o in outs], dim=-1)   # (B,T,C*(order+1))


class CrossAttentionBlock(nn.Module):
    """一层双向跨模态注意力：V←A 与 A←V。"""
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.v_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.a_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.v_norm1 = nn.LayerNorm(d_model); self.v_norm2 = nn.LayerNorm(d_model)
        self.a_norm1 = nn.LayerNorm(d_model); self.a_norm2 = nn.LayerNorm(d_model)
        self.v_ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
        self.a_ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
        self.attn_weights = None  # 缓存最后一层 V←A 注意力（供时序热图）

    def forward(self, v, a):
        # v, a: (B, T, D)
        v2, w_va = self.v_attn(v, a, a, need_weights=True, average_attn_weights=False)
        self.attn_weights = w_va
        v = self.v_norm1(v + v2)
        v = self.v_norm2(v + self.v_ffn(v))
        a2, _ = self.a_attn(a, v, v, need_weights=False)
        a = self.a_norm1(a + a2)
        a = self.a_norm2(a + self.a_ffn(a))
        return v, a


class LCMA(nn.Module):
    """Lightweight Cross-Modal Adapter（双向跨模态注意力）。"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.audio_proj = nn.Linear(cfg.audio_dim, cfg.d_model)
        self.visual_proj = nn.Linear(cfg.clip_dim, cfg.d_model)
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(cfg.d_model, cfg.n_heads, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.out_norm = nn.LayerNorm(cfg.d_model)

    def forward(self, audio_seq, v_clip):
        # audio_seq: (B, T_a, 768); v_clip: (B, T_v, 512)
        a = self.audio_proj(audio_seq)
        v = self.visual_proj(v_clip)
        a = resample_temporal(a, self.cfg.T)
        v = resample_temporal(v, self.cfg.T)
        last_w = None
        for blk in self.blocks:
            v, a = blk(v, a)
            last_w = blk.attn_weights
        a = self.out_norm(a)
        v = self.out_norm(v)
        a_pool = a.mean(dim=1)                          # (B, d)
        v_pool = v.mean(dim=1)                          # (B, d)
        pooled = torch.cat([a_pool, v_pool], dim=-1)    # (B, 2d)
        attn_map = last_w.mean(dim=1) if last_w is not None else None   # (B, T, T)
        return pooled, attn_map, a, v


def compute_kinematics(P, order=3):
    """解析唇部运动学特征（不引入可学习参数）。

    P: (B, T, 192) = 64 个口周点的 (x,y,z)。
    返回位置 + 1~order 阶时序差分（速度/加速度/加加速度...），
    形状 (B, T, 192*(order+1))。各阶差分首帧用自身补齐，保持长度 T。
    BioLip 的关键观察：真实说话的唇部运动满足软组织力学约束，
    高阶运动学统计量（尤其 jerk）平滑、低能量；生成器即便音画同步完美，
    仍残留非生理的运动抖动——这正是 LACA 要暴露的信号。
    """
    feats = [P]
    cur = P
    for _ in range(order):
        d = cur[:, 1:] - cur[:, :-1]
        d = torch.cat([d[:, :1], d], dim=1)      # pad 首帧
        feats.append(d)
        cur = d
    return torch.cat(feats, dim=-1)


class LACA(nn.Module):
    """Lip Articulatory Consistency Adapter —— 双信号唇探针（B′ 设计）。

    两条并行唇支路，最后拼接经 laca_fuse 映射到 2d（与 h_lcma 同维供 DWF 融合）：

      支路(i) 唇–音频对应 / 失配（用音频，类 AV-Lip-Sync+ 的 Sync-Check）：
          lip 表征 K（来自运动学 GRU）与 audio 表征逐帧比对 |K_t − a_t|，
          得到"唇形与音频对不对得上"的差异序列 → corr_mlp → 池化 → corr_proj(2d)。
          满足 thesis 核心：LACA 重新消费音频，与 LCMA 形成不同粒度的 AV 一致性信号。

      支路(ii) biomech 抖动（无音频，BioLip 式）：
          仅用 perioral landmark 轨迹 P 的高阶运动学（位置/速度/加速度/加加速度），
          对生成器残留的非生理运动抖动(jitter)敏感；该线索与音频无关、不被联合生成消掉，
          专门补 LCMA 在二代威胁上的盲区，且与支路(i) 正交。
          该支路的输出与 BioLip 统计分支 x_s（时序 std 汇总各阶方差）残差融合，
          增强跨域稳定性（cfg.use_xs 控制）。

    通过 cfg.laca_mode ∈ {both, corr, biomech} 可单独关闭某支路（消融用）。
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # 运动学前端通道数：xyz=192(论文 768-d) / y=64(BioLip 式,改则论文文本需同步)
        self.kin_channels = cfg.lip_lm_dim if cfg.kin_axes == "xyz" else cfg.lip_lm_dim // 3
        kin_dim = self.kin_channels * (cfg.kin_order + 1)   # 192*4=768 或 64*4=256
        # 确定性运动学前端：S-G 平滑导数（抑制 MediaPipe 高频量化噪声放大进 jerk）。
        # cfg.kin_smooth=="none" 时回退到朴素差分(compute_kinematics，旧行为)。
        if cfg.kin_smooth == "savgol":
            self.kinematics = Kinematics(self.kin_channels, order=cfg.kin_order,
                                         window=cfg.kin_window, poly=cfg.kin_poly,
                                         fps=cfg.fps, use_fps=cfg.kin_use_fps)
        else:
            self.kinematics = None
        # 共享唇部 backbone：运动学 → MLP → 双向 GRU → 投影/LN = lip 表征 K
        self.kin_mlp = nn.Sequential(
            nn.Linear(kin_dim, 256), nn.GELU(),
            nn.Linear(256, cfg.laca_lip_dim),
        )
        self.gru = nn.GRU(cfg.laca_lip_dim, cfg.d_model, batch_first=True, bidirectional=True)
        self.gru_proj = nn.Linear(cfg.d_model * 2, cfg.d_model)
        self.out_norm = nn.LayerNorm(cfg.d_model)
        # 支路(ii)：biomech 抖动 → out_proj(2d)
        self.biomech_proj = nn.Linear(cfg.d_model, cfg.d_model * 2)
        # BioLip 式统计分支 x_s：时序 std 汇总各阶运动学方差（对跨域泛化有效）
        if self.cfg.use_xs:
            self.xs_mlp = nn.Sequential(
                nn.Linear(kin_dim, 256), nn.GELU(),
                nn.Linear(256, cfg.d_model), nn.GELU(),
                nn.Linear(cfg.d_model, cfg.d_model * 2),
            )
        else:
            self.xs_mlp = None
        # 支路(i)：唇–音频对应 / 失配
        self.audio_proj = nn.Linear(cfg.audio_dim, cfg.d_model)
        # 共享空间投影头：把 lip 表征 K_t 与 audio 表征 a_t 显式映射到同一空间，
        # 使 |K_t - a_t| 在公共流形上有意义（对齐 AV-Lip-Sync+ 的共享 AV-HuBERT 思路）。
        self.lip_shared_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, cfg.laca_shared_dim),
        )
        self.audio_shared_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, cfg.laca_shared_dim),
        )
        self.corr_mlp = nn.Sequential(
            nn.Linear(cfg.laca_shared_dim, cfg.laca_shared_dim), nn.GELU(),
            nn.Linear(cfg.laca_shared_dim, cfg.d_model),
        )
        self.corr_proj = nn.Linear(cfg.d_model, cfg.d_model * 2)
        # 双支路拼接 → 2d（与 h_lcma 对齐，供 DWF）
        self.laca_fuse = nn.Linear(cfg.d_model * 4, cfg.d_model * 2)

    def lip_repr(self, v_lip):
        """返回 lip 表征序列 K 与池化 k_pool（支路(ii) 与对应对比共用）。

        前端流水线：可选逐帧归一化(去头部位姿/尺度) → 可选 y 轴切片 → 运动学展开
        (S-G 平滑导数 / 朴素差分)。real 与 Stage-1 jittered 走同一前端，自监督不受影响。
        同时返回原始运动学特征 K_raw（供 BioLip 式统计分支 x_s 使用）。
        """
        P = v_lip
        # ── 维度契约守卫：抽取端 v_lip 末维必须 == cfg.lip_lm_dim（xyz=192）──
        #    若 .pt 缓存是用错误契约（旧 120-D / 240-D）抽取的，这里立即报错并给出
        #    重抽指引，避免静默训练出错误结果（抽取一次代价极大）。
        if P.shape[-1] != self.cfg.lip_lm_dim:
            raise ValueError(
                f"[CONTRACT-MISMATCH] v_lip 末维={int(P.shape[-1])} 但 cfg.lip_lm_dim="
                f"{self.cfg.lip_lm_dim}。这说明特征缓存是用错误契约抽取的，"
                f"请用 prep_*.py（调用 common.extract_lip_landmarks_seq，_PERIORAL_64_IDS 192-D）重新抽取。")
        if self.cfg.kin_norm:
            P = normalize_landmarks(P)                    # 先归一化(须在切片前，C=192)
        if self.cfg.kin_axes == "y":
            P = P.view(*P.shape[:-1], P.shape[-1] // 3, 3)[..., 1]   # (B,T,64) 仅垂直颌动（N_points）
        if self.kinematics is not None:
            K_raw = self.kinematics(P)                    # S-G 平滑导数（抑噪）
        else:
            K_raw = compute_kinematics(P, order=self.cfg.kin_order)   # 朴素差分(旧行为)
        K = self.kin_mlp(K_raw)
        K = resample_temporal(K, self.cfg.T)
        K, _ = self.gru(K)
        K = self.gru_proj(K)
        K = self.out_norm(K)                       # (B, T, d_model)
        return K, K.mean(dim=1), K_raw            # (seq, pool, raw_kin)

    def audio_repr(self, audio_seq):
        """返回 audio 表征序列 a 与池化 a_pool（与 lip 表征同维）。"""
        a = self.audio_proj(audio_seq)             # (B, T_a, d_model)
        a = resample_temporal(a, self.cfg.T)
        return a, a.mean(dim=1)

    def forward(self, v_lip, audio_seq=None):
        K, k_pool, K_raw = self.lip_repr(v_lip)  # (B,T,d_model), (B,d_model), (B,T,kin_dim)
        h_bii = self.biomech_proj(k_pool)          # (B, 2d) 支路(ii) biomech

        # BioLip 式统计分支 x_s：时序 std 汇总各阶运动学方差
        h_xs = torch.zeros_like(h_bii)
        if self.cfg.use_xs and self.xs_mlp is not None:
            # K_raw: (B, T, kin_dim) → std across time → (B, kin_dim)
            xs_feat = K_raw.std(dim=1)             # BioLip: σ^(k)_i = std_t(Δ^k p_i)
            h_xs = self.xs_mlp(xs_feat)            # (B, 2d)

        h_bi = torch.zeros_like(h_bii)             # 支路(i) 默认关闭（无音频 / 模式禁用）
        K_s = a_s = None
        if audio_seq is not None and self.cfg.laca_mode in ("both", "corr"):
            a, a_pool = self.audio_repr(audio_seq)  # (B,T,d_model), (B,d_model)
            # 投影到共享空间后再算帧级失配（对齐 AV-Lip-Sync+ 的同空间 Sync-Check）
            K_s = self.lip_shared_proj(K)           # (B,T,shared_dim)
            a_s = self.audio_shared_proj(a)         # (B,T,shared_dim)
            diff = torch.abs(K_s - a_s)             # (B,T,shared_dim) 逐帧唇–音频失配
            corr_seq = self.corr_mlp(diff)          # (B,T,d_model)
            corr_pool = corr_seq.mean(dim=1)        # (B,d_model)
            h_bi = self.corr_proj(corr_pool)        # (B, 2d) 支路(i)

        if self.cfg.laca_mode == "biomech":
            h_laca = h_bii + h_xs                   # biomech + 统计分支（残差融合）
        elif self.cfg.laca_mode == "corr":
            h_laca = h_bi                           # 仅 corr 支路（唇-音频失配）
        else:
            h_laca = self.laca_fuse(torch.cat([h_bi, h_bii + h_xs], dim=-1))  # (B, 2d)
        return h_laca, K, h_bi, h_bii, K_s, a_s


class LTCA(nn.Module):
    """Long-Temporal Content Consistency Adapter（方向B 重定义：内容特征长时一致性）。

    取代已被 `evidence_chain §6B` 证伪的结构谱 FTA。核心：可分信号在音频"内容"特征
    （audio_emb 线性可分 0.827），**不在任何结构代理**（运动学/能量谱/交叉谱相干）——
    故不再造结构谱，而是在 LCMA 输出的"内容"嵌入 a_seq/v_seq (B, T, d_model) 上做长时一致性。

    红线（Cross-Modal，不可违背）：LTCA **必须**同时消费 audio+visual，含显式音画时序相干，
    绝无"纯单模态"路径——即便时序骨干（子系统B）关闭，子系统A 也是 audio+visual 双路拼接。

      子系统A 长时骨干：audio 与 visual 各一条 dilated temporal conv(膨胀1,2,4，感受野~全窗)
                        → 池化 → MLP → h_long (B, 2d)。把"单帧可分"提升为"跨 4s 时长的演化证据"。
      子系统B 音画时序相干（红线核心）：visual 序列 attend audio（cross-attn）得时序对齐特征，
                        再乘以显式时序相干标量 sync_t=<cos>（沿时间维，∈[-1,1]）→ h_sync。
      h_fta = h_long + h_sync   (B, 2d)

    掩码口径：输入即 LCMA post-attention 嵌入，评测 `--mask audio/visual` 在 forward 前置零
    对应输入，a_lcma/v_lcma 随动 → LTCA 掩码自动正确，无需改 evaluate.py。
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model; H = cfg.fta_hidden
        # 子系统A：audio/visual 各一条 dilated temporal conv（膨胀 1,2,4，覆盖全 50 窗）
        self.aud_conv = self._conv_stack(d, cfg.fta_dilation)
        self.vis_conv = self._conv_stack(d, cfg.fta_dilation)
        self.pool_proj = nn.Linear(2 * d, 2 * d)
        self.ff = nn.Sequential(nn.Linear(2 * d, H), nn.GELU(), nn.Linear(H, 2 * d))
        # 子系统B：音画时序相干（visual seq attend audio seq）
        self.cross_attn = nn.MultiheadAttention(d, cfg.n_heads, cfg.dropout, batch_first=True)
        self.sync_proj = nn.Linear(2 * d, 2 * d)

    @staticmethod
    def _conv_stack(d, dilations):
        layers = []
        for k in dilations:
            layers.append(nn.Conv1d(d, d, kernel_size=3, padding=k, dilation=k))
            layers.append(nn.GELU())
        return nn.Sequential(*layers)

    def forward(self, a_seq, v_seq, no_sync=False):
        # a_seq, v_seq: (B, T, d_model) —— LCMA post-attention 内容嵌入（T=50）
        aT = self.aud_conv(a_seq.transpose(1, 2)).transpose(1, 2)   # (B,T,d)
        vT = self.vis_conv(v_seq.transpose(1, 2)).transpose(1, 2)
        a_p, v_p = aT.mean(1), vT.mean(1)                           # 池化
        h_long = self.ff(self.pool_proj(torch.cat([a_p, v_p], dim=-1)))   # (B,2d)
        # 子系统B：visual attend audio → 时序对齐特征；显式时序相干标量缩放。
        # 输入拼成 2d（cross-attn 音画特征 d + audio 池化 d），保持与 h_long 同宽可相加。
        # [方向B 单变量归因] no_sync=True：跳过子系统B，只留长时骨干 h_long（评测/消融用，不改训练）。
        if no_sync:
            return h_long                                            # (B,2d)
        v_sync, _ = self.cross_attn(v_seq, a_seq, a_seq)            # (B,T,d)
        sync_t = F.cosine_similarity(a_seq, v_seq, dim=-1).mean(dim=1)    # (B,) ∈[-1,1]
        h_av = v_sync.mean(1)                                       # (B,d)
        h_sync = self.sync_proj(torch.cat([h_av, a_p], dim=-1)) * (0.5 + 0.5 * sync_t.unsqueeze(-1))
        return h_long + h_sync                                       # (B,2d)


class CrossReconNeck(nn.Module):
    """HAVIC 式跨模态重建头（§14-①）：逐帧 MLP，(B,T,d)->(B,T,d)。

    在 LCMA post-attention 内容流上做 MAE 式重建用；另有全局语义重建 MLP 复用本类(d->d)。
    """
    def __init__(self, dim, hidden=None, num_layers=2):
        super().__init__()
        h = hidden or dim * 2
        layers = []
        for i in range(num_layers):
            in_d = dim if i == 0 else h
            layers.append(nn.Linear(in_d, h)); layers.append(nn.GELU())
        layers.append(nn.Linear(h, dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


class DWF(nn.Module):
    """Dynamic Weight Fusion：softmax 门控融合 LCMA / LACA（/ 可扩 FTA）。

    n_branches=2：concat(h_lcma, h_laca) (4d) → 2 logits → softmax(α_ℓ, α_p)。
    n_branches=3：concat(h_lcma, h_laca, h_fta) (6d) → 3 logits（方向B 第三分支）。
    偏置初始化 favour LCMA（α_ℓ 初始最大）；前 freeze_epochs 轮可冻结为只用 LCMA。
    """
    def __init__(self, fused_dim, freeze_epochs=5, n_branches=2):
        super().__init__()
        self.freeze_epochs = freeze_epochs
        self.n_branches = n_branches
        self.gate = nn.Linear(fused_dim * n_branches, n_branches)
        nn.init.zeros_(self.gate.weight)
        if n_branches == 3:
            self.gate.bias.data = torch.tensor([3.0, 0.0, -1.0])   # favour LCMA，LACA 次之，FTA 低起步
        else:
            self.gate.bias.data = torch.tensor([3.0, -3.0])

    def forward(self, h_lcma, h_laca, h_fta=None, enabled=True):
        if h_lcma is None and h_fta is None:
            return h_laca                             # 全组件消融：仅 LACA 分支
        if not enabled:
            return h_lcma if h_lcma is not None else (h_laca + h_fta) if h_fta is not None else h_laca
        if h_fta is None:
            # 两路（n_branches=2，原语义）
            logits = self.gate(torch.cat([h_lcma, h_laca], dim=-1))   # (B,2)
            w = F.softmax(logits, dim=-1)
            return w[:, 0:1] * h_lcma + w[:, 1:2] * h_laca
        # 三路（n_branches=3，方向B）；w/o LCMA 时用零占位保持维度对齐
        z = h_lcma if h_lcma is not None else torch.zeros_like(h_laca)
        logits = self.gate(torch.cat([z, h_laca, h_fta], dim=-1))   # (B,3)
        w = F.softmax(logits, dim=-1)
        return w[:, 0:1] * z + w[:, 1:2] * h_laca + w[:, 2:3] * h_fta


class SyncX(nn.Module):
    """完整 Sync-X：LCMA + LACA + DWF + AAM 分类头。"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lcma = LCMA(cfg)
        self.laca = LACA(cfg)
        self.dwf = DWF(cfg.d_model * 2, cfg.dwf_freeze_epochs, n_branches=3 if cfg.fta_enabled else 2)
        # [方向B重定义] 内容特征长时一致性适配器（LTCA，第三分支；cfg.fta_enabled 默认关）
        self.fta = LTCA(cfg) if cfg.fta_enabled else None
        self.aam = AAMHead(cfg.d_model * 2, cfg.n_classes, cfg.aam_margin, cfg.aam_scale, cfg.aam_fp32)
        # [v2-1a] 音频独立分类头：单独监督 audio_emb（attention 前纯音频投影，d=d_model）。
        #   针对"听觉监督断裂"——给音频独立 real/fake 目标，避免其可分信号在下游被丢弃。
        self.audio_aam = AAMHead(cfg.d_model, cfg.n_classes, cfg.aam_margin, cfg.aam_scale, cfg.aam_fp32)

    def forward(self, batch, labels=None, gate_enabled=True):
        audio_seq = batch["audio_seq"]
        v_clip = batch["v_clip"]
        v_lip = batch["v_lip"]

        if self.cfg.disable_lcma:
            # w/o LCMA：跳过跨模态适配器，仅 LACA 分支（fused = h_laca）。
            h_lcma = None
            attn_map = None
            a_lcma = v_lcma = None
        else:
            h_lcma, attn_map, a_lcma, v_lcma = self.lcma(audio_seq, v_clip)

        h_laca, k_seq, h_laca_bi, h_laca_bii, k_shared, a_shared = self.laca(v_lip, audio_seq)  # 双信号唇探针

        # [方向B] LTCA 第三分支：在 LCMA 内容嵌入上做长时一致性（输入即 a_lcma/v_lcma，
        #   post-attn 内容嵌入；disable_lcma 时现算 pre-attn 投影过 out_norm 对齐分布）。
        #   掩码口径：评测 --mask 在 forward 前置零 audio_seq/v_clip，a_lcma/v_lcma 随动 → 自动正确。
        h_fta = None
        if self.cfg.fta_enabled and self.fta is not None:
            if a_lcma is not None:
                a_seq_fta, v_seq_fta = a_lcma, v_lcma
            else:
                a_seq_fta = self.lcma.out_norm(resample_temporal(self.lcma.audio_proj(audio_seq), self.cfg.T))
                v_seq_fta = self.lcma.out_norm(resample_temporal(self.lcma.visual_proj(v_clip), self.cfg.T))
            h_fta = self.fta(a_seq_fta, v_seq_fta,
                             no_sync=getattr(self.cfg, "fta_no_sync", False))   # (B, 2d)

        if self.cfg.disable_dwf:
            # w/o DWF：固定平均替代可学习门控（fused 已 2d 对齐；含 FTA 时按分支数平均）
            if h_fta is None:
                fused = h_laca if h_lcma is None else 0.5 * h_lcma + 0.5 * h_laca
            else:
                fused = (h_lcma + h_laca + h_fta) / 3 if h_lcma is not None else (h_laca + h_fta) / 2
        else:
            fused = self.dwf(h_lcma, h_laca, h_fta, enabled=gate_enabled)   # (B, 2d)

        # 同步一致性用的音/视嵌入：
        # L_corr 必须用 LCMA 跨模态注意力**之前**的原始池化嵌入测对应，
        # 否则双向注意力已将音/视结构性地拉近（无论 real/fake），
        # corr_score 退化为常数 → 损失梯度≈0 → 正则失效。
        # LCMA 关时仍用 LACA 双信号代理（同原逻辑）。
        if a_lcma is None:
            audio_emb = h_laca_bi       # 唇–音频对应支路（含音频），(B, 2d)
            visual_emb = h_laca_bii     # biomech 支路（仅 landmark/视觉），(B, 2d)
            # [FIX 2026-10-24] 音频独立分类头 audio_aam 是按 d_model 维建的，而 h_laca_bi 是 2d，
            # 直接喂进去会触发 512×(256×2) 维度不匹配（wo_lcma 变体因此在第一步就崩）。
            # 这里改用 LACA 内部的音频投影池化：与正常路径的 a_raw.mean(dim=1) 同源同维，
            # 不新增参数、不改 ckpt 结构，只让「w/o LCMA」这条支路能正常训练。
            if audio_seq is not None:
                audio_emb_head = resample_temporal(self.laca.audio_proj(audio_seq),
                                                   self.cfg.T).mean(dim=1)
            else:
                audio_emb_head = audio_emb[..., : self.cfg.d_model]
        else:
            # [FIX 2026-08-09] 用 pre-attention 原始投影池化，不用 post-attention 的 a_lcma/v_lcma
            a_raw = self.lcma.audio_proj(audio_seq)
            v_raw = self.lcma.visual_proj(v_clip)
            a_raw = resample_temporal(a_raw, self.cfg.T)
            v_raw = resample_temporal(v_raw, self.cfg.T)
            audio_emb = a_raw.mean(dim=1)    # (B, d) — attention 前的纯音频投影
            visual_emb = v_raw.mean(dim=1)   # (B, d) — attention 前的纯视觉投影
            audio_emb_head = audio_emb       # 正常路径本来就是 d_model 维，直接复用
        # 音频独立分类头输出（供 1a 监督 / 评测单模态用）。推理时也返回 logits。
        # 注意：这里喂的是 audio_emb_head（d_model 维），不是给 L_corr 用的 audio_emb。
        if labels is None:
            audio_logits = self.audio_aam(audio_emb_head)        # 纯推理
            return {"logits": self.aam(fused), "emb": fused, "attn": attn_map,
                    "audio_emb": audio_emb, "visual_emb": visual_emb,
                    "audio_logits": audio_logits, "h_fta": h_fta,
                    "h_laca_bi": h_laca_bi, "h_laca_bii": h_laca_bii,
                    "k_shared": k_shared, "a_shared": a_shared}
        loss_cls, logits = self.aam(fused, labels)
        loss_cls_audio, audio_logits = self.audio_aam(audio_emb_head, labels)   # [v2-1a]
        return {"loss_cls": loss_cls, "logits": logits, "emb": fused, "attn": attn_map,
                "audio_emb": audio_emb, "visual_emb": visual_emb,
                "loss_cls_audio": loss_cls_audio, "audio_logits": audio_logits,
                "h_fta": h_fta,
                "a_lcma": a_lcma, "v_lcma": v_lcma,   # [§13] 注意力后逐帧内容嵌入，供 end-to-end align prior
                "h_laca_bi": h_laca_bi, "h_laca_bii": h_laca_bii,
                "k_shared": k_shared, "a_shared": a_shared}

    def forward_interpret(self, batch):
        """返回逐帧信号供 F11：注意力热图 + 音/视序列嵌入（计算逐帧异常分）。

        复用现有 self.lcma 返回的 attn_map 与 a/v 序列，不引入新参数。
        """
        audio_seq = batch["audio_seq"]
        v_clip = batch["v_clip"]
        pooled, attn_map, a_seq, v_seq = self.lcma(audio_seq, v_clip)  # a_seq,v_seq:(B,T,d)
        return {"attn": attn_map, "a_seq": a_seq, "v_seq": v_seq}

    def pooled_embedding(self, batch):
        """供 F15：返回 mean-pooled 音+视拼接嵌入 (B, 2d)，用于 t-SNE/UMAP 可分性。"""
        audio_seq = batch["audio_seq"]
        v_clip = batch["v_clip"]
        pooled, attn, a, v = self.lcma(audio_seq, v_clip)
        a_pool = a.mean(1); v_pool = v.mean(1)
        return torch.cat([a_pool, v_pool], dim=-1)   # (B, 2d)


class AAMHead(nn.Module):
    """AAM-Softmax 分类头的薄封装（保持与 losses.AAMSoftmax 一致）。"""
    def __init__(self, in_features, n_classes, margin, scale, fp32):
        super().__init__()
        from losses import AAMSoftmax
        self.aam = AAMSoftmax(in_features, n_classes, margin, scale, fp32)

    def forward(self, emb, labels=None):
        if labels is None:
            return self.aam(emb)              # logits
        return self.aam(emb, labels)          # (loss, logits)
