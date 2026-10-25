"""
losses.py — Sync-X 损失函数
================================================================
  AAM-Softmax : margin 增强分类头（替代 CE）。CosFace / AM-Softmax 形式
                （cosine margin, 与 ArcFace 的 angular margin 不同）。
  InfoNCE     : Stage-1 SSL，真实片段上学音-视对齐先验。
  CorrConsistencyLoss : Stage-2 跨模态对应正则（真实片段上拉近音/视嵌入；论文 L_corr）。
  AlignmentLoss / UniformityLoss : Stage-2 超球面结构正则（Wang & Isola 2020）。

引用：
  ArcFace/AAM (angular margin)  Deng et al., CVPR 2019, arXiv:1801.07698
  CosFace/AAM-Softmax (cosine margin)  Wang et al., CVPR 2018, arXiv:1801.09414
  CLIP InfoNCE Radford et al., ICML 2021, arXiv:2103.00020 (τ=0.07)
  Wang & Isola 2020  alignment/uniformity loss
  SyncNet      Chung & Zisserman, ACMM 2016, arXiv:1612.05678
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AAMSoftmax(nn.Module):
    """Additive Margin Softmax 分类头（CosFace / AM-Softmax 形式：cosine margin）。

    在 cosine 空间施加 margin（target 类 cos 减 m），而非 ArcFace 的 angular margin
    （target 类 arccos 内加 m）。这两种形式在数学上不同，但同属 AAM-Softmax 家族。
    本实现与论文 Ch3 §AAM 完全一致：`cos_margin = cos - onehot * m`，
    `logits = s * cos_margin`，再走 cross_entropy。
    二分类即 real/fake。强制 fp32 计算（s=30 下 softmax 防饱和）。
    """
    def __init__(self, in_features, n_classes=2, margin=0.30, scale=30.0, fp32=True):
        super().__init__()
        self.in_features = in_features
        self.n_classes = n_classes
        self.margin = margin
        self.scale = scale
        self.fp32 = fp32
        self.weight = nn.Parameter(torch.empty(n_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels=None):
        if self.fp32:
            emb = F.normalize(embeddings.float(), p=2, dim=1)
            w = F.normalize(self.weight.float(), p=2, dim=1)
        else:
            emb = F.normalize(embeddings, p=2, dim=1)
            w = F.normalize(self.weight, p=2, dim=1)
        cos = emb @ w.t()                                  # (B, C)
        cos = cos.clamp(-1 + 1e-7, 1 - 1e-7)
        if labels is None:
            return self.scale * cos                        # 推理 logits
        onehot = F.one_hot(labels.long(), self.n_classes).float()
        cos_margin = cos - onehot * self.margin            # 仅目标类减 margin
        logits = self.scale * cos_margin
        loss = F.cross_entropy(logits, labels)
        return loss, self.scale * cos                      # (loss, raw_logits)


class InfoNCELoss(nn.Module):
    """对称 InfoNCE：audio↔visual 对比学习（Stage-1 SSL）。

    正样本 = 同一片段的 (audio, visual)；负样本 = batch 内其他片段。
    τ 可学习（CLIP 风格），clamp ≥ tau_min 防崩溃。
    """
    def __init__(self, tau_init=0.07, tau_learnable=True, tau_min=0.01):
        super().__init__()
        self.tau_min = tau_min
        self.tau_learnable = tau_learnable
        log_tau = torch.tensor(math.log(tau_init))
        if tau_learnable:
            self.log_tau = nn.Parameter(log_tau)
        else:
            self.register_buffer("log_tau", log_tau)

    @property
    def tau(self):
        return torch.exp(self.log_tau.clamp(min=math.log(self.tau_min)))

    def forward(self, audio_emb, visual_emb):
        a = F.normalize(audio_emb, dim=1)
        v = F.normalize(visual_emb, dim=1)
        logits = a @ v.t()                                 # (B, B)
        logits = logits / self.tau
        labels = torch.arange(a.size(0), device=a.device)
        loss_av = F.cross_entropy(logits, labels)
        loss_va = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_av + loss_va)


class CorrConsistencyLoss(nn.Module):
    """跨模态对应正则（与论文 Ch3 公式一致）：

    L_corr = || corr_score - (1-y) ||_2^2, 对所有样本平均。
    corr_score = cos(audio_emb, visual_emb)；y=1(fake)->目标0(低对应), y=0(real)->目标1(高对应)。

    [v2-1d] corr_real_only=True 时：只对 real 段约束 corr→1（保跨模态对准初衷），
    fake 段不再强制去相关（目标改用一个宽松浮点阈值 0.5）。根因：原 (1-y) 形式
    要求 fake 音频去相关，与 AAM 分类目标在同一 audio_emb 上反向拉扯（see 诊断 §3.6）。
    """
    def __init__(self, real_only=False, fake_target=0.5):
        super().__init__()
        self.real_only = real_only
        self.fake_target = fake_target

    def forward(self, audio_emb, visual_emb, labels=None):
        a = F.normalize(audio_emb, dim=1)
        v = F.normalize(visual_emb, dim=1)
        corr = (a * v).sum(dim=1)                 # (B,) 余弦相似度
        if labels is None:
            return (1.0 - corr).mean()             # 退化：仅拉近
        target = (1.0 - labels.float())            # real(0)->1, fake(1)->0
        if self.real_only:
            # 只约束 real(目标=1)：fake 段目标放宽到 self.fake_target（不强制去相关）。
            # 用真实标签选 real 段，避免把"模型判伪"错当监督。
            t = torch.where(labels == 0, torch.ones_like(corr), torch.full_like(corr, self.fake_target))
            return ((corr - t) ** 2).mean()
        return ((corr - target) ** 2).mean()


class SharedSpaceLoss(nn.Module):
    r"""LACA corr 共享空间对齐损失。

    在 LACA 的唇–音频对应支路中，K_t 来自 lip-GRU（基于 MediaPipe 运动学），
    a_t 来自冻结 wav2vec-2.0，二者原属不同预训练流形。通过两个投影头把它们映射到
    共享空间 \hat{K}_t, \hat{a}_t \in \mathbb{R}^{d_{shared}} 后，本损失要求真实片段上
    二者余弦相似度趋近 +1，伪造片段趋近 0。这使得 |\hat{K}_t - \hat{a}_t| 的 Sync-Check
    在公共流形上有明确几何意义，对齐 AV-Lip-Sync+ 的共享 AV-HuBERT 设计思路
    [Shahzad et al., THMS 2025]。

    输入 : lip_emb, audio_emb: (B, T, D_shared) 或 (B, D_shared)
           labels (可选): (B,) 0=real, 1=fake；缺省为 None，按真实目标 +1 训练
    输出 : scalar MSE  between cosine similarity and target
    """
    def __init__(self):
        super().__init__()

    def forward(self, lip_emb, audio_emb, labels=None):
        if lip_emb is None or audio_emb is None:
            # biomech-only 模式无音频输入时安全退化
            return torch.tensor(0.0, device=(lip_emb.device if lip_emb is not None else audio_emb.device),
                                requires_grad=True)
        if lip_emb.dim() == 3:
            lip_emb = lip_emb.mean(dim=1)
            audio_emb = audio_emb.mean(dim=1)
        lip_emb = F.normalize(lip_emb, p=2, dim=1)
        audio_emb = F.normalize(audio_emb, p=2, dim=1)
        cos = (lip_emb * audio_emb).sum(dim=1)          # (B,)
        if labels is None:
            target = torch.ones_like(cos)
        else:
            target = 1.0 - labels.float()               # real(0)->1, fake(1)->0
        return ((cos - target) ** 2).mean()


class AlignmentLoss(nn.Module):
    """超球面同类对齐损失（Rec B / GenD finding #2，对应论文 Ch3 Eq.align-loss）。

    作用于 DWF 输出、已 L2 归一化的 fused 表征 z_i = h_fused / ||h_fused||_2：
    在 *单位超球面* 上把 **同类**（real/real 或 fake/fake）样本对拉近，使 real / fake
    两类各自收成紧致球冠，与逐样本的 AAM 角边距头互补，提供 batch 级度量结构。
    对应论文公式 L_align = E_{(i,j)~P_pos} ||z_i - z_j||_2^2（P_pos 为同类的 same-source 样本对）。

    输入 : embeddings (B, D) 任意尺度（内部 L2 归一化）, labels (B,) 0/1
    输出 : scalar loss（无同类对时返回 0，避免 NaN）
    """
    def __init__(self):
        super().__init__()

    def forward(self, embeddings, labels):
        emb = F.normalize(embeddings, p=2, dim=1)        # (B, D) 单位球
        B = emb.size(0)
        if B < 2:
            return emb.sum() * 0.0
        dist = (emb.unsqueeze(1) - emb.unsqueeze(0)).pow(2).sum(-1)  # (B, B) 平方欧氏
        labels = labels.view(-1, 1)
        same = (labels == labels.t()).float()              # (B, B) 同类=1
        mask = same - torch.eye(B, device=emb.device)    # 去掉自身
        pos_per = mask.sum(dim=1)
        valid = pos_per > 0
        if not valid.any():
            return emb.sum() * 0.0
        loss = (mask * dist).sum(dim=1) / pos_per.clamp(min=1.0)
        return loss[valid].mean()


class FrameAlignmentPriorLoss(nn.Module):
    """端到端跨模态对齐一致性先验（§13 试点 · L_align_prior）。

    与显式跨模态对齐不同的是：不在"池化后"的 pre-attn audio_emb/visual_emb 上做
    （那是 CorrConsistencyLoss），而是在 LCMA 跨模态注意力**之后**的逐帧内容嵌入
    a_lcma/v_lcma (B, T, d_model) 上，按时间维逐帧求 `cos(a_lcma[t], v_lcma[t])`，
    把"真实段 → 拉向 +1"，让特征**在注意力对齐后的内容流里**也保持跨模态一致。

    动机（§13 / evidence_chain §6B）：离线冻结特征框架下，diag_real_prior 实测
    真实/生成的音画一致性不可分离。这是对"一致性先验"的一次在线化——让特征随
    "真实一致性要高"这个目标一起被学习，验证换成对齐约束压出的表征能否跨生成器
    迁移（B 组试点核心监督项）。纯真实域 Real-Vox 数据无法参与 DeepSpeak 分类
    Stage-2，故此处只用训练 batch 内的真实段（label==0）约束对齐，不另加真实域
    样本——单变量守卫：只动"加了逐帧对齐监督"这一环，其余不变。

    输入 : a_seq, v_seq: (B, T, d) 已对齐长度 T 的逐帧内容嵌入
           labels: (B,) 0=real, 1=fake —— 只对真实段约束，fake 段不参与（避免把
                  "generator 也自洽"误当监督，与 L_corr real_only 同口径）
    输出 : 标量 mse：mean_t((cos - target)^2)，仅对真实段平均；无真实段返回 0
    """
    def __init__(self):
        super().__init__()

    def forward(self, a_seq, v_seq, labels):
        a = F.normalize(a_seq, p=2, dim=-1)
        v = F.normalize(v_seq, p=2, dim=-1)
        cos_t = (a * v).sum(dim=-1)                    # (B, T) ∈[-1,1] 逐帧
        # 只约束真实段（real=0），fake 段不参与对齐监督
        if labels is not None and labels.numel() > 0:
            real = (labels == 0).float().unsqueeze(-1)  # (B,1)
            if real.sum() < 1:
                return torch.tensor(0.0, device=a_seq.device, requires_grad=True)
            err = ((cos_t - 1.0) ** 2) * real          # (B,T) 真实段压向 +1
            return err.sum() / real.sum()
        return ((cos_t - 1.0) ** 2).mean()


class MaskedCrossReconLoss(nn.Module):
    """MAE 式跨模态 token 重建（HAVIC modality-specific reconstruction 的轻量在线版，§14-①）。

    在 LCMA post-attention 的逐帧内容流上：随机 mask 掉 audio 序列一部分时间 token，
    用**视觉**序列去重建被 mask 的 audio token；反之 mask visual、用 audio 重建。
    这一"用另一模态还原被遮挡时间步"的目标，强制模型把'什么内容在什么时刻该发生'
    编码成跨模态结构一致性——正是 HAVIC 靠深层重建才捕获、而我们前半程只有 InfoNCE
    （仅对齐、不重建）漏掉的"结构层"成分。

    只作正样本监督（输入为纯真实片段的 a/v 流；Stage-1 天然如此，avoid fake 掺杂 §14-②）。

    输入 a_seq, v_seq: (B, T, d)（同长、LCMA post-attn 内容嵌入）
      mask_ratio ∈[0,1]  每样本独立 mask
      necks : 重建头，(B,T,d)->(B,T,d)，可用简单 MLP
    输出 scalar (MSE in normalized space)。
    """
    def __init__(self, mask_ratio=0.25):
        super().__init__()
        self.mask_ratio = mask_ratio

    def _mask_recon(self, src, ctx, neck):
        B, T, d = src.shape
        n_mask = max(1, int(round(self.mask_ratio * T)))
        rand = torch.rand(B, T, device=src.device)
        _, keep_idx = torch.topk(rand, T - n_mask, dim=1)   # 保留索引
        src_masked = src.clone()
        mask = torch.ones(B, T, dtype=torch.bool, device=src.device)
        mask.scatter_(1, keep_idx, False)
        src_masked[mask] = 0.0
        # masked src + 另一模态全局语义，喂给重建头还原完整内容
        ctx_agg = ctx.mean(dim=1, keepdim=True)             # (B,1,d)
        inp = src_masked + ctx_agg.expand(B, T, d).clone()
        rec = neck(inp)                                     # (B,T,d)
        loss = ((rec - src) ** 2).mean()
        return loss

    def forward(self, a_seq, v_seq, av_neck, va_neck):
        a_seq = F.normalize(a_seq, p=2, dim=-1)
        v_seq = F.normalize(v_seq, p=2, dim=-1)
        # mask audio、用 visual 重建；反之亦然
        l_av = self._mask_recon(a_seq, v_seq, av_neck)
        l_va = self._mask_recon(v_seq, a_seq, va_neck)
        return 0.5 * (l_av + l_va)


class CrossSemanticReconLoss(nn.Module):
    """跨模态语义交叉重建（HAVIC cross-modal semantic reconstruction 的轻量版，§14-①）。

    用一模态的*全局语义*(池化)解码重建另一模态的全局语义，约束 macro-coherence：
      visual 全局 → 重建 audio 全局（audio 该长什么内容）；audio 全局 → 重建 visual 全局。
    比 MaskedRecon 更粗粒度，但最贴近 HAVIC 的"语义级交叉重建"，与逐帧区分开。

    输入 a_pool, v_pool: (B, d) 已池化；av_mlp/v_a_mlp: 重建头 (d->d)。
    输出 scalar (MSE in normalized space)。
    """
    def __init__(self):
        super().__init__()

    def forward(self, a_pool, v_pool, av_mlp, va_mlp):
        a_pool = F.normalize(a_pool, p=2, dim=-1)
        v_pool = F.normalize(v_pool, p=2, dim=-1)
        l_av = ((av_mlp(a_pool) - v_pool) ** 2).mean()       # audio→视觉
        l_va = ((va_mlp(v_pool) - a_pool) ** 2).mean()       # visual→audio
        return 0.5 * (l_av + l_va)


class UniformityLoss(nn.Module):
    """超球面均匀性损失（Rec B / GenD finding #2，对应论文 Ch3 Eq.uniform-loss, Wang & Isola 2020）。

    在 *单位超球面* 上把全部样本对均匀铺开，防止表征塌缩到一点，
    保留 angular 决策边界。对应论文公式
    L_uniform = log E_{i,j} [ exp(-2 ||z_i - z_j||_2^2) ]（系数 2 取自 GenD 对 Wang & Isola 的实例化）。

    输入 : embeddings (B, D) 任意尺度（内部 L2 归一化）
    输出 : scalar loss
    """
    def __init__(self):
        super().__init__()

    def forward(self, embeddings):
        emb = F.normalize(embeddings, p=2, dim=1)        # (B, D) 单位球
        B = emb.size(0)
        if B < 1:
            return emb.sum() * 0.0
        dist = (emb.unsqueeze(1) - emb.unsqueeze(0)).pow(2).sum(-1)  # (B, B) 平方欧氏
        # Wang & Isola 均匀性：log 平均 exp(-2 * d)；对角线 d=0 贡献 exp(0)=1，梯度为 0，不影响优化
        return torch.log(torch.exp(-2.0 * dist).mean() + 1e-8)


# NOTE: 同源 real/fake 配对按 route-A 改为 **数据层采样**（dataset.PairedBatchSampler），
#       而非损失项（论文 Ch3 §rec-c / Ch4：同源配对通过 batch 采样实现，非专门损失；
#       且作为损失会在球面上与超球面度量项产生梯度冲突，见 Liu et al. 2025）。
#       故此处不再定义 SameSourcePairLoss。
