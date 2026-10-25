"""
plot_style.py — 全局出版级样式（IEEE 默认 + Nature/CVPR 一键切换）
================================================================
所有出图脚本统一 import 本模块，保证全文视觉一致。默认 IEEE 预设，与 thesis 的
IEEEtran / newtxtext+newtxmath 对齐；改投 Nature/CVPR 仅翻 VENUE 常量即可整套换肤。
§0.3 Okabe-Ito 色盲安全调色板；矢量 PDF（pdf.fonttype=42 避免 Type-3 拒稿）；
误差棒 capsize=3；灰度影线；Sync-X 2.4pt 强调线。
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

VENUE = "ieee"   # "ieee"(默认) | "nature" | "cvpr"

_PRESETS = {
    "ieee": {
        "w_full": 6.97, "w_col": 3.39,          # inch
        "family": "serif", "serif": ["Times New Roman", "DejaVu Serif"],
        "math": "stix", "fs": 9, "lab": 9, "title": 10, "tick": 8, "leg": 8,
        "grid": True, "gcol": "#DDDDDD", "glw": 0.6,
    },
    "nature": {
        "w_full": 180/25.4, "w_col": 89/25.4,
        "family": "sans-serif", "serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "math": "dejavusans", "fs": 8, "lab": 8, "title": 9, "tick": 7, "leg": 7,
        "grid": False, "gcol": "#E0E0E0", "glw": 0.5,
    },
    "cvpr": {
        "w_full": 8.0, "w_col": 3.33,
        "family": "serif", "serif": ["Times New Roman", "DejaVu Serif"],
        "math": "stix", "fs": 10, "lab": 9, "title": 10, "tick": 8, "leg": 8,
        "grid": False, "gcol": "#DDDDDD", "glw": 0.5,
    },
}

def setup_style():
    """出版级默认。scienceplots 有则叠加 IEEE/Nature 网格抛光，最后用本字典全覆盖。"""
    p = _PRESETS[VENUE]
    try:
        import scienceplots
        style = ["science"] + (["ieee"] if VENUE == "ieee" else ["nature"] if VENUE == "nature" else [])
        plt.style.use(style)
    except Exception:
        pass
    rcParams.update({
        "font.family": p["family"], "font.serif": p["serif"], "mathtext.fontset": p["math"],
        "font.size": p["fs"], "axes.labelsize": p["lab"], "axes.titlesize": p["title"],
        "xtick.labelsize": p["tick"], "ytick.labelsize": p["tick"], "legend.fontsize": p["leg"],
        "axes.grid": p["grid"], "grid.color": p["gcol"], "grid.linewidth": p["glw"],
        "axes.linewidth": 0.8, "lines.linewidth": 1.6, "lines.markersize": 4.0,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.format": "pdf",
        "pdf.fonttype": 42, "ps.fonttype": 42,   # 嵌入 TrueType，禁用 Type-3（IEEE/商业社拒稿红线）
        "legend.frameon": False, "legend.handlelength": 1.4,
    })

def fig_width(full=True): return _PRESETS[VENUE]["w_full" if full else "w_col"]
def newfig(full=True, aspect=0.62, ncols=1):
    w = fig_width(full)
    fig, ax = plt.subplots(ncols, figsize=(w, w*aspect), constrained_layout=True)
    return fig, (ax if ncols == 1 else ax)

SYNCX_LW, BASE_LW, REF_LW = 2.4, 1.6, 1.0
SYNCX_LS = "-"
MARKERS = {"syncx": "o", "baseline_avff": "s", "baseline_seedance": "^",
           "baseline_mmmsba": "D", "baseline_avlipsync": "v", "mds": "P", "Real": "o", "Fake": "x"}
HATCHES = ["", "////", "\\\\\\", "||", "--", "xx", "..", "oo"]  # 柱状图 B/W 印刷区分

METHOD_COLORS = {"syncx": "#0072B2", "baseline_avff": "#E69F00", "baseline_seedance": "#D55E00",
                 "baseline_mmmsba": "#009E73", "baseline_avlipsync": "#CC79A7", "mds": "#56B4E9"}
DATASET_COLORS = {"DFE2024": "#0072B2", "CelebDFv2": "#E69F00", "THB": "#009E73",
                  "MVAD": "#D55E00", "HiFiAVDF": "#CC79A7", "FF++": "#56B4E9"}
VARIANT_COLORS = {"full": "#0072B2", "audio-only": "#E69F00", "video-only": "#D55E00",
                  "laca_both": "#0072B2", "laca_corr": "#009E73", "laca_biomech": "#CC79A7",
                  "Real": "#009E73", "Fake": "#D55E00"}

def savefig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stem = path[:-4] if path.endswith(".pdf") else path
    fig.savefig(stem + ".pdf", bbox_inches="tight")              # 矢量，投稿用
    fig.savefig(stem + ".png", dpi=600, bbox_inches="tight")     # 位图预览
    # fig.savefig(stem + ".pgf", bbox_inches="tight")            # 如需 LaTeX 原生存档，取消注释
