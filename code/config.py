"""
config.py — Sync-X 全局超参数（经文献调研确定）
================================================================
所有参数都有调研依据（见各注释的引用）。修改前请先读 README 的"参数依据"一节。

参数分类标记：
  [LIT]  = 来自文献，固定不建议调
  [TUNE] = 需要超参数搜索
  [ARCH] = 架构参数，可选调优
  [ABL]  = 消融实验开关

核心引用：
  ArcFace          Deng et al., CVPR 2019, arXiv:1801.07698   (AAM margin/scale)
  AAM-Softmax(spk) Chung et al., Interspeech 2020, arXiv:2010.11558  (s=30 惯例)
  CLIP             Radford et al., ICML 2021, arXiv:2103.00020   (τ=0.07)
  SimCLR           Chen et al., ICML 2020, arXiv:2002.05709      (InfoNCE/对比学习)
  wav2vec2         Baevski et al., NeurIPS 2020, arXiv:2006.11477
  SyncNet          Chung & Zisserman, ACMM 2016, arXiv:1612.05678 (唇音同步)
  AdamW            Loshchilov & Hutter, ICLR 2019, arXiv:1711.05101
  Mixed Precision  Micikevicius et al., ICLR 2018, arXiv:1710.03740
  GenD             Yermakov et al., WACV 2026, arXiv:2508.06248   (hyperspherical)
  BioLip           Chen & Xu, arXiv:2604.16808, 2026             (kinematics)
  Wang & Isola     ICML 2020, arXiv:2005.10242                    (alignment/uniformity)
"""
from dataclasses import dataclass, field
import os


# ── 数据集根路径（路径真相源，2026-08-02 核实，与磁盘实况一致）────────────
#   已迁移数据集全部在 D:/Datasets/ 下；WildDeepfake / AV-Deepfake1M-PlusPlus
#   未迁移（驻留 D:/），且不接入最终 lineup，故不在此登记。
#   云端适配：本 code_cloud 副本默认指向云端数据盘；如需用本机 D:/Datasets，
#   显式 export SYNCX_DATA_ROOT=D:/Datasets 即可覆盖（无需改代码）。
DATA_ROOT = os.environ.get("SYNCX_DATA_ROOT", "/root/autodl-tmp/SyncX/Datasets")
DATASET_PATHS = {
    "deepspeak":     f"{DATA_ROOT}/DeepSpeak_v1_1/clips", # v1.1 主训练集（已落盘, 25 zip 解压目录）
    "voxceleb2":     f"{DATA_ROOT}/VoxCeleb2_subset",     # Stage-1 SSL（已落盘）
    "fakeavceleb":   f"{DATA_ROOT}/FakeAVCeleb_v1.2_split",  # B6 探针（已落盘）
    "dfe2024":       f"{DATA_ROOT}/Deepfake-Eval-2024",   # 跨域主评测（已落盘）
    "celebdf":       f"{DATA_ROOT}/Celeb-DF-v2",          # 跨域评测（已落盘）
    "thb":           f"{DATA_ROOT}/TalkingHeadBench",     # 跨域评测（已落盘）
    "mvad":          f"{DATA_ROOT}/MVAD",                 # 跨域评测（已落盘）
    "hifi":          f"{DATA_ROOT}/HiFi-AVDF",            # 跨域评测（已落盘）
    "ffpp":          f"{DATA_ROOT}/FF++",                 # 参考基准（已落盘）
}


@dataclass
class Config:
    # ── 特征维度（必须与 common.py 的 extract_* 输出一致；口周点=64×3=192）────
    audio_dim:   int = 768          # wav2vec2-base-960h last_hidden_state
    clip_dim:    int = 512          # CLIP ViT-B/16 visual_projection 输出
    lip_lm_dim:  int = 192         # MediaPipe 口周点: 64 个唯一口周点 × 3 (x,y,z)，见 common._PERIORAL_64_IDS
    lip_roi:     int = 112         # 灰度唇 ROI 边长

    # ── 时序 ────────────────────────────────────────────────────────────
    T:           int = 50          # [ARCH] LCMA 内部把音/视时序重采样到的固定长度
                                   #   视觉 100 帧@25fps / 音频 200 帧@50Hz(20ms 窗) → 重采样到 T

    # ── LCMA（双向跨模态注意力适配器）──────────────────────────────────
    d_model:     int = 256         # [ARCH] 隐藏维（与论文 Ch3 数学公式/Ch4 训练表一致）
    n_heads:     int = 4           # [ARCH] 注意力头数
    n_layers:    int = 2           # [ARCH] 轻量，避免过参数
    dropout:     float = 0.1       # [ARCH] Dropout 率

    # ── LACA（双信号唇探针）─────────────────────────────────────────────
    #   支路(i) 唇–音频对应/失配（用音频，类 AV-Lip-Sync+）：|lip_emb − audio_emb|
    #   支路(ii) biomech 抖动（无音频，BioLip 式）：位置/速度/加速度/加加速度
    #   两支路正交：biomech 与音画同步无关、不被联合生成消掉；corr 补回 LCMA 的
    #   局部唇–音频粒度。laca_mode 控制消融：both / corr / biomech。
    laca_lip_dim: int = 64          # [ARCH] 每帧运动学特征经 MLP 压缩到的维（论文：→64-D）
    kin_order:    int = 3           # [LIT] 运动学差分阶数：1=速度,2=加速度,3=加加速度(jerk) — BioLip
    laca_mode:    str = "both"      # [ABL] both | corr | biomech（消融用；biomech = 旧 7-14 版）
    laca_shared_dim: int = 256      # [ARCH] LACA corr 支路：K_t/a_t 投影到的共享空间维

    # ── LACA 运动学前端（抑制 MediaPipe 高频量化噪声，防探针学"噪声"而非生物力学）──
    #   问题：朴素有限差分 ×3 求 jerk 会把 landmark 高频噪声放大 ~2^3 倍。
    #   注意：支路(ii) 的检测信号本身就是高频非生理抖动，且 Stage-1 会注入 jitter，
    #   故只做「轻量去噪」、绝不过度平滑（否则连信号一起抹掉、自监督坍塌）。
    #   方案（对齐 BioLip arXiv:2604.16808）：Savitzky-Golay 小窗低阶平滑导数 +
    #   fps 感知量纲；真实唇动信号带 1–8Hz，仅削 >信号带 的量化噪声。
    kin_smooth:  str  = "savgol"    # [ABL] 运动学前端: "savgol"(平滑导数) | "none"(=朴素差分,旧行为)
    kin_window:  int  = 5           # [LIT] S-G 窗长(奇数)；25fps 下 window=5 保住 1–8Hz 抖动信号带
    kin_poly:    int  = 3           # [LIT] S-G 多项式阶(须 >= kin_order, < kin_window)
    kin_use_fps: bool = False       # [ABL] 导数是否×fps 换算真实量纲（默认 False）
    fps:         float = 25.0       # [LIT] 视频帧率(供 kin_use_fps=True 时的 delta)
    kin_axes:    str  = "xyz"       # [ABL] "xyz"(768-d: 64点×3×4阶) | "y"(256-d, BioLip 式: 64点×4阶)
    kin_norm:    bool = False       # [ABL] 逐帧去质心/尺度归一化
    use_xs:      bool = True        # [ABL] BioLip 式统计分支：时序 std 汇总各阶运动学方差

    # ── DWF（动态权重门控）─────────────────────────────────────────────
    dwf_freeze_epochs: int = 5     # [LIT] 前 N 轮冻结门控为只用 LCMA（打破训练对称性）

    # ── 组件消融开关（Gap-2 修复：支持 w/o LCMA 与 w/o DWF 实验）────────────
    disable_lcma: bool = False      # [ABL] 跳过 LCMA 跨模态适配器
    disable_dwf:  bool = False      # [ABL] 关闭可学习门控，改用固定 0.5/0.5 平均融合

    # ── AAM-Softmax（分类头；替代 CE）────────────────────────────────────
    #   调研结论：AAM 本身就是 margin 增强的分类头，与 CE 并列会冗余/冲突
    n_classes:   int = 2          # real / fake
    aam_margin:  float = 0.30     # [LIT] CosFace/AM-Softmax；ArcFace default=0.5，binary task 用 0.3
    aam_scale:   float = 30.0     # [LIT] 说话人验证 AAM 标准 s=30；ArcFace default=64
    aam_fp32:    bool = True      # s=30 下 fp16 安全，但 AAM softmax 强制 fp32 防饱和

    # ── Stage-1 InfoNCE SSL（真实片段上学 AV 对齐先验）─────────────────
    infonce_tau_init: float = 0.07   # [LIT] CLIP 默认初始化 (Radford 2021)
    infonce_tau_learnable: bool = True  # [LIT] CLIP 风格可学习温度
    infonce_tau_min: float = 0.01     # [LIT] clamp 下界，防止 τ→0 崩溃

    # ── Stage-1 LACA 运动学伪抖动自监督（真实片段 + 自生成抖动负样本）────
    kin_jitter_sigma: float = 0.015  # [TUNE] 伪抖动噪声强度（归一化 landmark 坐标尺度）
    lambda_kin:  float = 0.3        # [TUNE] LACA 运动学自监督损失权重

    # ── 损失权重 ────────────────────────────────────────────────────────
    lambda_corr_lcma: float = 0.3  # [TUNE] 跨模态对应(LCMA 池化嵌入余弦)正则（调研区间 0.1–1.0）
    lambda_shared: float = 0.1    # [TUNE] LACA corr 共享空间对齐损失权重（调研区间 0.05–0.5）
    # [v2-1a] 音频独立分类头损失权重。针对"听觉监督断裂"（audio_emb 弱可分 0.827 但
    # audio-only 0.52）：给 audio_emb 加独立 AAM 头显式监督 real/fake，λ_audio 控制其力度。
    lambda_audio: float = 1.0      # [TUNE] 音频独立分类头(loss_cls_audio)权重（v2 新增；可从 SYNCX_LAMBDA_AUDIO 覆写）
    # [v2-1d] 是否放宽 L_corr/L_shared 对 fake 音频的"去相关"强制。
    #   根因（see 诊断文件 §3.6）：L_corr 要 fake 音频去相关、L_AAM 要判对，同一 audio_emb 上
    #   反向拉扯 → 音频被带去学"不相关"而非"判别"。True=只约束 real 段(目标=1)，fake 段目标放宽
    #   到浮点区间(不强制 0)，消除与分类头的冲突。默认关闭以保原论文行为；v2 训练需开启。
    corr_real_only: bool = False   # [v2-1d] L_corr/L_shared 是否只约束 real 段（解监督冲突）
    # [module2-监督侧] Stage-2 冻结编码器 + paired modality dropout（训练配方，非新分支）
    #   依据 evidence_chain §6B：可分信号在音频"内容"特征(audio_emb 0.827)却未被训练用上；
    #   嫌疑④=Stage-2 全量微调污了 Stage-1 AV 表征；嫌疑③=无 modality dropout 致依赖单模态域捷径。
    #   两者均默认关闭，仅重训时经 env 打开，不污染现 pipeline。
    freeze_encoder_stage2: bool = False   # True=冻结 LCMA/LACA 骨干(只训 DWF/AAM/audio_aam 头)，保住 Stage-1 AV 表征
    freeze_visual_only: bool = False      # True=仅冻结视觉支路(lcma.visual_proj + laca 唇部/biomech)，放开音频与融合层
                                          #   → 逼训练用上 audio_emb 的可分信号(证据链§8B: 全冻结 train 平台死0.874+跨域反向，故只冻视觉)；
                                          #   单变量使用，勿与 freeze_encoder_stage2 同开。MODAL_DROPOUT 建议关(=0)。
    modal_dropout_prob: float = 0.0       # paired modality dropout 概率(0=关)；训练中按概率整体遮蔽 audio 或 visual 支路
    modal_dropout_start_epoch: int = 10   # 到达该 epoch 后才启用 dropout（避免训练初期强制单模态）

    # ── [方向B重定义] LTCA：内容特征长时一致性适配器（第三分支，非结构谱）────────
    #   依据 evidence_chain §6B/§8C：可分信号在音频"内容"特征(audio_emb 0.827)，不在任何结构代理
    #   （运动学/能量谱/交叉谱相干，已全实测证伪）。方向B = 在 LCMA 内容嵌入(a_seq/v_seq)上做
    #   长时一致性（长时骨干 dilated-conv + 音画时序相干 cross-attn），守 Cross-Modal 红线。
    #   默认关闭防污染现有行为；仅重训时 SYNCX_FTA=1 打开。
    fta_enabled: bool = False             # [ABL] 主开关（SYNCX_FTA=1 覆写）；开=接入 LTCA 第三分支
    fta_hidden: int = 256                 # [ARCH] LTCA 内部隐藏维
    fta_dilation: tuple = (1, 2, 4)       # [ARCH] 长时 dilated temporal conv 膨胀序列（感受野~全 50 窗）
    fta_no_sync: bool = False             # [ABL] 评测时关闭 LTCA 子系统B(sync 标量)，只留长时骨干 h_long。
                                          #   方向B 实测 HiFi 反降，假设 sync 在 HiFi 无信号(负迁移)——此开关不改训练、
                                          #   用已训 ckpt 直接测"去 sync 后 HiFi 是否回升"，单变量归因(见 diag_sync_t.py)。

    # ── GenD 启发的两项泛化升级（principal config 默认激活）────────────
    #   对应论文 Ch3 §gen-upgrades / Ch4 训练表：
    #   Rec B: 超球面 alignment + uniformity（Wang & Isola 2020），作用于已 L2
    #          归一化的 fused 表征 z_i，与 AAM 头互补提供 batch 级度量结构；
    #   Rec C: 同源 real/fake 配对 = **数据层采样**（dataset.PairedBatchSampler），
    #          非损失项，故此处无 lambda_pair。两项按 Ch4 val-split 选定值默认激活。
    lambda_align:   float = 0.1   # [LIT] Rec B: 超球面 alignment 权重（GenD 原文 = 0.1）
    lambda_uniform: float = 0.5   # [LIT] Rec B: 超球面 uniformity 权重（GenD 原文 = 0.5）
    same_source_sampling: bool = True  # [ABL] Rec C: 同源配对数据层采样

    # [§13 试点] 端到端跨模态对齐一致性先验（B 组核心监督项）。
    #   默认关闭（对齐 §8 现有行为）；SYNCX_ALIGN_LOSS=1 开启，SYNCX_ALIGN_WEIGHT 覆写权重。
    #   在 LCMA 注意力后逐帧内容嵌入 a_lcma/v_lcma 上，把真实段 cos→+1（见 losses.FrameAlignmentPriorLoss）。
    #   与 freeze_encoder_stage2 互斥使用（要"特征随一致性目标学"就必须放开编码器）。
    align_prior_loss: bool = False
    lambda_align_prior: float = 0.1   # [§13] 逐帧对齐先验权重（start 保守，防与 AAM/L_corr 同嵌拉扯）

    # [§14] Stage-1 HAVIC 式跨模态一致性重建先验（在线化"做全 HAVIC 三层"）。
    #       默认关闭（§8 现有行为）；SYNCX_HAVIC_RECON=1 开启，权重由下列 lambda 控制。
    #       在 LCMA post-attn 内容流上补两条：MaskedCrossRecon（逐帧跨模态重建）+
    #       CrossSemanticRecon（全局语义交叉重建）。仅 Stage-1 纯真实数据训，
    #       avoid fake 掺杂（对应 validation=验证该假设：弱实现害了方向，非方向失效）。
    #       头部为一个 shared CrossReconNeck（逐帧）+ 两个 semantic MLP（全局）。
    hav_rec_mask_ratio: float = 0.25
    lambda_hav_mask:   float = 0.5    # [§14] 逐帧跨模态重建权重
    lambda_hav_sem:    float = 0.5    # [§14] 全局语义交叉重建权重

    # ── 优化器 ──────────────────────────────────────────────────────────
    optimizer:   str = "adamw"     # [LIT] AdamW (Loshchilov 2019)
    lr:          float = 1e-4     # [TUNE] 学习率（冻结骨干下仅训适配器）
    weight_decay: float = 0.01    # [LIT] AdamW weight decay
    warmup_steps: int = 500       # [LIT] 线性 warmup 步数
    grad_clip:   float = 1.0      # [LIT] 梯度裁剪
    amp:         bool = True     # [LIT] 混合精度（fp16）

    # ── 训练时长 ────────────────────────────────────────────────────────
    stage1_epochs: int = 60      # [TUNE] Stage 1 SSL 轮数（CAV-MAE 调研：40 合理、60 更稳；>80 边际递减）
    stage2_epochs: int = 80      # [TUNE] Stage 2 监督轮数
    early_stop_patience: int = 8  # [LIT] early stopping 耐心值

    # ── 批次 ────────────────────────────────────────────────────────────
    batch_size_stage1: int = 1024   # [TUNE] Stage 1 批次（InfoNCE 负样本数量）
    batch_size_stage2: int = 512   # [TUNE] Stage 2 批次
    num_workers: int = 4          # [ABL] DataLoader workers

    # ── 数据集身份（B7 身份泄漏控制用）──────────────────────────────────
    fakeavceleb_identity_file: str = ""   # 可选：FakeAVCeleb 身份列表文件
    exclude_fakeavceleb_ids: bool = False # [ABL] B7 开关


def get_config(**overrides) -> Config:
    """构造 Config 实例（兼容各脚本的 `cfg = get_config()` / `get_config(**saved)` 调用）。

    所有训练/评测/导出脚本通过 `from config import get_config` 获取配置；本函数是
    Config dataclass 的工厂，支持用保存的配置字典覆盖默认值（dump_*/evaluate 据此
    把 ckpt 中存回的 cfg 字段回灌）。
    """
    return Config(**overrides)


# ── 超参数搜索配置 ──────────────────────────────────────────────────
# 用于 hyperparameter_search.py。⚠️ 注意：以下部分条目（lambda_kin / stage1_epochs /
# batch_size_stage2 / kin_jitter_sigma）仅在 config 中登记，SEARCH_STRATEGY 未引用
# （Stage-1 或手动使用），避免误导"可搜"。
HYPERPARAMETER_SEARCH_CONFIG = {
    # 损失权重（高优先级）
    'lambda_corr_lcma': {
        'type': 'float',
        'range': [0.1, 1.0],
        'values': [0.1, 0.3, 0.5, 1.0],
        'description': '跨模态对应损失权重',
        'priority': 'high',
    },
    'lambda_shared': {
        'type': 'float',
        'range': [0.05, 0.5],
        'values': [0.05, 0.1, 0.3, 0.5],
        'description': 'LACA 共享空间对齐权重',
        'priority': 'high',
    },
    'lambda_align': {
        'type': 'float',
        'range': [0.05, 0.3],
        'values': [0.1, 0.2],   # MVP 轻量搜索（4 组合）：跳过 0.05 极端值，约 2-3h
        'description': 'GenD 超球面 alignment 权重（论文 Ch4:171-172 "selected on val split" 留白可搜；默认 0.1 为契约值）',
        'priority': 'high',
    },
    'lambda_uniform': {
        'type': 'float',
        'range': [0.2, 1.0],
        'values': [0.5, 1.0],   # MVP 轻量搜索（4 组合）：跳过 0.2 极端值
        'description': 'GenD 超球面 uniformity 权重（论文 Ch4:171-172 留白可搜；默认 0.5 为契约值）',
        'priority': 'high',
    },
    'lambda_kin': {
        'type': 'float',
        'range': [0.1, 1.0],
        'values': [0.1, 0.3, 0.5, 1.0],
        'description': '运动学自监督损失权重（⚠️ 仅 Stage-1 用；Stage-2 损失不含 L_kin，hyperparameter_search 不传）',
        'priority': 'medium',
    },
    # 训练超参数（中优先级）
    'lr': {
        'type': 'float',
        'range': [5e-5, 5e-4],
        'values': [5e-5, 1e-4, 3e-4, 5e-4],
        'description': '学习率',
        'priority': 'high',
    },
    'stage2_epochs': {
        'type': 'int',
        'range': [40, 120],
        'values': [40, 80, 120],
        'description': 'Stage 2 训练轮数',
        'priority': 'medium',
    },
    'stage1_epochs': {
        'type': 'int',
        'range': [20, 60],
        'values': [20, 40, 60],
        'description': 'Stage 1 训练轮数',
        'priority': 'medium',
    },
    'batch_size_stage2': {
        'type': 'int',
        'range': [128, 512],
        'values': [128, 256, 512],
        'description': 'Stage 2 批次大小',
        'priority': 'low',
    },
    # 运动学参数（低优先级）
    'kin_jitter_sigma': {
        'type': 'float',
        'range': [0.005, 0.03],
        'values': [0.005, 0.01, 0.015, 0.03],
        'description': '伪抖动噪声强度',
        'priority': 'low',
    },
}


# ── 超参数搜索策略 ──────────────────────────────────────────────────
SEARCH_STRATEGY = {
    'name': 'sequential_grid',  # 顺序网格搜索
    'description': '按优先级顺序搜索，固定其他参数，逐步收敛',
    'steps': [
        {
            'name': 'loss_weights',
            'description': '搜索论文留白损失权重：λ_align × λ_uniform（λ_corr=0.3/λ_shared=0.1 论文已锁定，不做主搜索；如需敏感性佐证可手动单变量跑）',
            'params': ['lambda_align', 'lambda_uniform'],
            'fixed': {'lr': 1e-4, 'stage2_epochs': 80},
        },
        {
            'name': 'learning_rate',
            'description': '搜索学习率（固定最佳损失权重；lr 论文锁定 1e-4，此步通常只作确认）',
            'params': ['lr'],
            'fixed': {'stage2_epochs': 80},  # lambda_align/lambda_uniform 用最佳值
        },
        {
            'name': 'training_epochs',
            'description': '搜索训练轮数（注意：运行本阶段时【不要】传 --epochs，否则会覆盖 stage2_epochs 的搜索值导致空转）',
            'params': ['stage2_epochs'],
            'fixed': {},  # lr 和损失权重用最佳值
        },
    ],
}
