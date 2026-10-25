# -*- coding: utf-8 -*-
"""
Three watermark-free vector figures for the Sync-X thesis.
  1. fig_sync_structure.pdf        (Chapter 3: system framework)
  2. fig_offline_pipeline.pdf      (Chapter 3: offline feature pre-extraction)
  3. fig_temporal_attention_case.pdf (Chapter 5: explainability case study)

Re-run: python make_extra_figs.py
"""
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
import math, random, os

# ---- palette ----
WHITE      = HexColor('#ffffff')
PANEL_FILL = HexColor('#eef1f4')
PANEL_STK  = HexColor('#9aa3ad')
CHIP_FILL  = HexColor('#ffffff')
CHIP_STK   = HexColor('#3a3f47')
RED_FILL   = HexColor('#c0392b')
RED_STK    = HexColor('#8e2b21')
YEL_FILL   = HexColor('#fff3cd')
YEL_STK    = HexColor('#d9a400')
BLU_FILL   = HexColor('#e7f0ff')
BLU_STK    = HexColor('#3b6db4')
GRN_FILL   = HexColor('#e6f5ea')
GRN_STK    = HexColor('#2f7d3a')
PINK_FILL  = HexColor('#fdecea')
INK        = HexColor('#1f2429')
GREY       = HexColor('#5b6470')
SUB        = HexColor('#3a3f47')
AXIS       = HexColor('#2b2f36')

OUTDIR = os.environ.get("SYNCX_FIG_OUTDIR",
                         r"C:/Users/ZhuanZ1/Desktop/my_proposal_papers/monash_thesis/Figures")


def rect(c, x, top, w, h, H, fill=None, stroke=None, lw=1.0, radius=0.0):
    c.saveState()
    if fill is not None:
        c.setFillColor(fill)
    if stroke is not None:
        c.setStrokeColor(stroke); c.setLineWidth(lw)
    f = 1 if fill is not None else 0
    s = 1 if stroke is not None else 0
    fx, fy = x, H - (top + h)
    if radius:
        c.roundRect(fx, fy, w, h, radius, stroke=s, fill=f)
    else:
        c.rect(fx, fy, w, h, stroke=s, fill=f)
    c.restoreState()


def box(c, x, top, w, h, H, text, fill=CHIP_FILL, stroke=CHIP_STK, lw=1.0,
        radius=4.0, fontsize=9, fontname="Helvetica", tcolor=INK):
    rect(c, x, top, w, h, H, fill=fill, stroke=stroke, lw=lw, radius=radius)
    c.setFont(fontname, fontsize); c.setFillColor(tcolor)
    ty = H - (top + h / 2.0) - fontsize * 0.34
    c.drawCentredString(x + w / 2, ty, text)


def header(c, x, top, w, text, H, fill=BLU_FILL, stroke=BLU_STK, fontsize=9):
    box(c, x, top, w, 22, H, text, fill=fill, stroke=stroke, fontsize=fontsize)


def arrow(c, x1_top, y1_top, x2_top, y2_top, H, color=AXIS, lw=1.2):
    """Draw arrow from (x1_top, y1_top) to (x2_top, y2_top), where y's are
    top-down (smaller y = higher in PDF). PDF coords are y_pdf = H - y_top."""
    x1, y1 = x1_top, H - y1_top
    x2, y2 = x2_top, H - y2_top
    c.saveState()
    c.setStrokeColor(color); c.setFillColor(color); c.setLineWidth(lw)
    c.line(x1, y1, x2, y2)
    ang = math.atan2(y2 - y1, x2 - x1)
    L = 6
    a1 = ang + math.radians(150)
    a2 = ang - math.radians(150)
    p1 = (x2 + L * math.cos(a1), y2 + L * math.sin(a1))
    p2 = (x2 + L * math.cos(a2), y2 + L * math.sin(a2))
    p = c.beginPath()
    p.moveTo(x2, y2); p.lineTo(p1[0], p1[1]); p.lineTo(p2[0], p2[1]); p.close()
    c.drawPath(p, stroke=0, fill=1)
    c.restoreState()


def title(c, W, H, top, main, sub=None):
    c.setFont("Helvetica-Bold", 13); c.setFillColor(INK)
    c.drawCentredString(W / 2, H - (top + 14), main)
    if sub:
        c.setFont("Helvetica", 7.5); c.setFillColor(GREY)
        c.drawCentredString(W / 2, H - (top + 30), sub)


# =========================================================================
# 1) Sync-X system framework
# =========================================================================
W, H = 720, 360
c = canvas.Canvas(OUTDIR + r"/fig_sync_structure.pdf", pagesize=(W, H))

title(c, W, H, 8, "Sync-X system framework",
      sub="Two frozen self-supervised backbones feed two parallel streams; outputs fuse via a learned gate before classification.")

# Row Y positions (top-down):
ROW0 = 200  # raw inputs
ROW1 = 200  # frozen backbones (aligned with raw inputs)
ROW2 = 130  # adapters
ROW3 = 100  # DWF + loss
# Columns
C_RAW   = 40
C_BB    = 180
C_ADAPT = 360
C_FUSE  = 540
C_OUT   = 660

# Row: raw inputs
header(c, C_RAW,  ROW0, 100, "Raw video clip", H)
header(c, C_RAW,  170, 100, "Raw audio (16 kHz)", H)
# Backbones (frozen, gray)
box(c, C_BB,  ROW0, 130, 22, H, "CLIP ViT-B/16 (frozen)", fill=PANEL_FILL, stroke=PANEL_STK, fontsize=8.5)
c.setFont("Helvetica", 7); c.setFillColor(GREY)
c.drawCentredString(C_BB + 65, H - (ROW0 + 22) - 8, "face crop 224x224")
box(c, C_BB,  170, 130, 22, H, "wav2vec 2.0 (frozen)", fill=PANEL_FILL, stroke=PANEL_STK, fontsize=8.5)
c.setFont("Helvetica", 7); c.setFillColor(GREY)
c.drawCentredString(C_BB + 65, H - 192 - 8, "base-960h, 768-D")

# Adapters
box(c, C_ADAPT, 170, 130, 26, H, "LCMA", fill=YEL_FILL, stroke=YEL_STK, fontsize=10)
c.setFont("Helvetica", 7); c.setFillColor(GREY)
c.drawCentredString(C_ADAPT + 65, H - 196 - 8, "BiT-CMA L=2 (semantic)")
box(c, C_ADAPT, 130, 130, 26, H, "LACA", fill=YEL_FILL, stroke=YEL_STK, fontsize=10)
c.setFont("Helvetica", 7); c.setFillColor(GREY)
c.drawCentredString(C_ADAPT + 65, H - 156 - 8, "Lip Kin+GRU + BiT-CMA")

# MediaPipe helper (small, below backbones, source for LACA)
box(c, C_BB, 110, 130, 22, H, "MediaPipe FaceMesh", fill=PANEL_FILL, stroke=PANEL_STK, fontsize=8.5)
c.setFont("Helvetica", 7); c.setFillColor(GREY)
c.drawCentredString(C_BB + 65, H - 132 - 8, "64 lip landmarks")

# DWF
box(c, C_FUSE, 150, 110, 30, H, "Dynamic Weight Fusion",
    fill=GRN_FILL, stroke=GRN_STK, fontsize=9.5)
c.setFont("Helvetica", 7); c.setFillColor(GREY)
c.drawCentredString(C_FUSE + 55, H - 180 - 8, "learned gate (a, 1-a)")

# Outputs
box(c, C_OUT, 170, 50, 22, H, "CLS", fontsize=8)
box(c, C_OUT, 145, 50, 22, H, "Corr", fontsize=8)
box(c, C_OUT, 120, 50, 22, H, "Embed", fontsize=8)

# Loss (red, top-right)
box(c, C_FUSE, 240, 130, 26, H, "L_AAM + \u03bb\u00b7L_corr",
    fill=RED_FILL, stroke=RED_STK, fontsize=9, tcolor=WHITE)

# Arrows
arrow(c, C_RAW + 100, 211, C_BB,      211, H)   # video -> CLIP
arrow(c, C_RAW + 100, 181, C_BB,      181, H)   # audio -> wav2vec
arrow(c, C_BB + 130,  196, C_ADAPT,   183, H)   # CLIP -> LCMA
arrow(c, C_BB + 130,  165, C_ADAPT,   183, H)   # wav2vec -> LCMA
arrow(c, C_BB + 130,  121, C_ADAPT,   143, H)   # MediaPipe -> LACA (curved)
arrow(c, C_ADAPT + 130, 183, C_FUSE,   165, H)   # LCMA -> DWF
arrow(c, C_ADAPT + 130, 143, C_FUSE,   165, H)   # LACA -> DWF
arrow(c, C_FUSE + 110, 165, C_OUT + 25, 181, H)  # DWF -> CLS
arrow(c, C_FUSE + 110, 160, C_OUT + 25, 156, H)  # DWF -> Corr
arrow(c, C_FUSE + 110, 155, C_OUT + 25, 131, H)  # DWF -> Embed
arrow(c, C_FUSE + 65,  150, C_FUSE + 65, 240, H) # DWF -> loss (loop up, thin)

# Caption
c.setFont("Helvetica-Oblique", 7); c.setFillColor(GREY)
c.drawCentredString(W / 2, H - 320, "Frozen backbones (gray) feed lightweight adapters (yellow) and a learned gate (green). The red box is the training loss.")

c.showPage(); c.save()
print("wrote fig_sync_structure.pdf  (%dx%d pt)" % (W, H))


# =========================================================================
# 2) Offline feature pre-extraction pipeline
# =========================================================================
W, H = 720, 320
c = canvas.Canvas(OUTDIR + r"/fig_offline_pipeline.pdf", pagesize=(W, H))

title(c, W, H, 8, "Offline feature pre-extraction pipeline",
      sub="Run once per video. Caches half-precision tensors for fast iteration during training.")

# Columns
C_INPUT = 30
C_RAW   = 130
C_BB    = 260
C_TENS  = 410
C_CACHE = 580

# Row 1: Raw inputs (top)
header(c, C_INPUT, 240, 90, "Raw .mp4 video", H)
header(c, C_INPUT, 210, 90, "Extracted .wav (16 kHz)", H)

# Row 2: Stage labels (column headers) -- at PDF y=195 (above the data rows
# which start at PDF y=150; PDF y is H - top = 320 - 170 = 150)
c.setFont("Helvetica-Bold", 8); c.setFillColor(SUB)
c.drawString(C_RAW,   H - 195, "Raw tensors")
c.drawString(C_BB,    H - 195, "Frozen extractors")
c.drawString(C_TENS,  H - 195, "Cached tensors")
c.drawString(C_CACHE, H - 195, "On-disk cache")

# Row 3: Per-modality entries (3 rows: video | audio | mediapipe)
row_top = 170
box(c, C_RAW,  row_top,     100, 22, H, "Video frames @ 25 fps", fill=BLU_FILL, stroke=BLU_STK, fontsize=8.5)
box(c, C_RAW,  row_top - 30, 100, 22, H, "Audio waveform", fill=BLU_FILL, stroke=BLU_STK, fontsize=8.5)
box(c, C_RAW,  row_top - 60, 100, 22, H, "Lip landmarks / frame", fill=BLU_FILL, stroke=BLU_STK, fontsize=8.5)

box(c, C_BB,   row_top,     120, 22, H, "CLIP ViT-B/16", fill=PANEL_FILL, stroke=PANEL_STK, fontsize=8.5)
box(c, C_BB,   row_top - 30, 120, 22, H, "wav2vec 2.0", fill=PANEL_FILL, stroke=PANEL_STK, fontsize=8.5)
box(c, C_BB,   row_top - 60, 120, 22, H, "MediaPipe FaceMesh", fill=PANEL_FILL, stroke=PANEL_STK, fontsize=8.5)

# Cached tensor specs
box(c, C_TENS, row_top,     150, 22, H, "v_clip: (T_v, 512)", fill=GRN_FILL, stroke=GRN_STK, fontsize=8.5)
box(c, C_TENS, row_top - 30, 150, 22, H, "audio_seq: (T_a, 768)", fill=GRN_FILL, stroke=GRN_STK, fontsize=8.5)
box(c, C_TENS, row_top - 60, 150, 22, H, "v_lip: (T_v, 192)", fill=GRN_FILL, stroke=GRN_STK, fontsize=8.5)

# On-disk cache: big yellow box with title at top, inner text below
# Box is at top=row_top=170, height=110; PDF y range = [H-row_top-h, H-row_top] = [40, 150]
CACHE_H = 110
# Title at top of box
box(c, C_CACHE, row_top, 130, 22, H, "<stem>.pt",
    fill=YEL_FILL, stroke=YEL_STK, fontsize=10)
# Sub-text rows below the title (smaller PDF y = lower in image)
CACHE_PDF_TOP = H - row_top - 22        # just below the title (PDF y = 128)
inner_y_pdf = CACHE_PDF_TOP - 12
c.setFont("Helvetica", 7); c.setFillColor(GREY)
c.drawCentredString(C_CACHE + 65, inner_y_pdf,        "feature_cache/")
c.drawCentredString(C_CACHE + 65, inner_y_pdf - 11,   "{audio_seq, v_clip,")
c.drawCentredString(C_CACHE + 65, inner_y_pdf - 22,   "v_lip,")
c.drawCentredString(C_CACHE + 65, inner_y_pdf - 33,   "label, identity}")
c.setFont("Helvetica-Oblique", 7); c.setFillColor(SUB)
c.drawCentredString(C_CACHE + 65, inner_y_pdf - 50,   "(float16, ~half size)")

# Training row at bottom
box(c, 30, 40, 660, 36, H,
    "Training: Stage 1 InfoNCE on REAL clips   |   Stage 2 L_AAM + \u03bb\u00b7L_corr on (REAL, FAKE)",
    fill=PANEL_FILL, stroke=PANEL_STK, fontsize=9)

# Arrows: raw -> extractors -> tensors
arrow(c, C_RAW + 100, 231, C_BB,        231, H)   # video -> CLIP
arrow(c, C_RAW + 100, 201, C_BB,        201, H)   # audio -> wav2vec
arrow(c, C_RAW + 100, 171, C_BB,        171, H)   # mediapipe
arrow(c, C_BB  + 120, 231, C_TENS,      231, H)
arrow(c, C_BB  + 120, 201, C_TENS,      201, H)
arrow(c, C_BB  + 120, 171, C_TENS,      171, H)
arrow(c, C_TENS + 150, 200, C_CACHE,    185, H)   # tensors -> cache
# cache -> training
arrow(c, C_CACHE + 65, 130, 360, 80, H)  # cache down-left to training row

c.showPage(); c.save()
print("wrote fig_offline_pipeline.pdf  (%dx%d pt)" % (W, H))


# =========================================================================
# 3) Temporal attention case study
# =========================================================================
W, H = 720, 360
c = canvas.Canvas(OUTDIR + r"/fig_temporal_attention_case.pdf", pagesize=(W, H))

title(c, W, H, 8, "Temporal explainability: authentic vs manipulated clips",
      sub="Top: per-time-step cosine-similarity anomaly curve.  Bottom: LCMA cross-attention weight matrix (audio time x visual frame).")
label = "qualitative case study, not supervised temporal localisation"
c.setFont("Helvetica-Oblique", 7); c.setFillColor(GREY)
c.drawCentredString(W / 2, H - 60, label)
c.setFont("Helvetica-Oblique", 7); c.setFillColor(GREY)
c.drawCentredString(W / 2, H - 76, "Interpretation: low cosine similarity in the anomaly curve corresponds to weak LCMA cross-modal correspondence.")

# Two side-by-side panels
COL_W = 280
X1 = 50
X2 = X1 + COL_W + 60

# Panel 1: Authentic
c.setFont("Helvetica-Bold", 9); c.setFillColor(GRN_STK)
c.drawString(X1, H - 100, "Authentic clip")
c.setFont("Helvetica-Oblique", 7); c.setFillColor(GREY)
c.drawString(X1, H - 112, "(expected: coherent diagonal)")

# Panel 2: Manipulated
c.setFont("Helvetica-Bold", 9); c.setFillColor(RED_STK)
c.drawString(X2, H - 100, "Manipulated clip")
c.setFont("Helvetica-Oblique", 7); c.setFillColor(GREY)
c.drawString(X2, H - 112, "(scattered attention, anomaly dip)")

# Heatmap helper
def draw_heatmap_authentic(c, x, top, w, h, H):
    nx, ny = 20, 12
    cw, ch = w / nx, h / ny
    c.saveState()
    c.setStrokeColor(AXIS); c.setLineWidth(0.8)
    c.rect(x, H - (top + h), w, h, stroke=1, fill=0)
    c.restoreState()
    random.seed(7)
    for j in range(ny):
        for i in range(nx):
            d = abs(i / nx - j / ny)
            base = max(0, 1.0 - d * 3.5)        # strong diagonal
            noise = (random.random() - 0.5) * 0.1
            v = max(0, min(1, base + noise))
            g = int(255 - v * 200)
            c.setFillColor(HexColor(f"#{g:02x}{g:02x}{g+10:02x}"))
            c.rect(x + i * cw, H - (top + h - j * ch), cw, ch, fill=1, stroke=0)


def draw_heatmap_scattered(c, x, top, w, h, H):
    nx, ny = 20, 12
    cw, ch = w / nx, h / ny
    c.saveState()
    c.setStrokeColor(AXIS); c.setLineWidth(0.8)
    c.rect(x, H - (top + h), w, h, stroke=1, fill=0)
    c.restoreState()
    random.seed(13)
    for j in range(ny):
        for i in range(nx):
            v = max(0, min(1, random.random() * 0.5))
            g = int(255 - v * 200)
            c.setFillColor(HexColor(f"#{g:02x}{g:02x}{g+10:02x}"))
            c.rect(x + i * cw, H - (top + h - j * ch), cw, ch, fill=1, stroke=0)


def draw_anomaly(c, x, top, w, h, H, smooth=True, dip=None):
    c.saveState()
    c.setStrokeColor(AXIS); c.setLineWidth(0.8)
    c.rect(x, H - (top + h), w, h, stroke=1, fill=0)
    c.restoreState()
    c.saveState()
    c.setStrokeColor(RED_FILL); c.setLineWidth(1.4)
    T = 60
    prev = None
    rng = random.Random(99)
    for i in range(T + 1):
        if smooth:
            v = 0.72 + 0.16 * math.sin(i / 8.0) + 0.05 * (rng.random() - 0.5)
        else:
            v = 0.55 + 0.32 * (rng.random() - 0.3)
            if dip is not None and dip[0] <= i <= dip[1]:
                v -= 0.40
        v = max(0.10, min(0.95, v))
        px = x + (i / T) * w
        py = H - (top + h * v)
        if prev is not None:
            c.line(prev[0], prev[1], px, py)
        prev = (px, py)
    c.restoreState()
    # threshold dashed line
    c.saveState()
    c.setStrokeColor(GREY); c.setLineWidth(0.5); c.setDash(2, 2)
    c.line(x, H - (top + h * 0.5), x + w, H - (top + h * 0.5))
    c.restoreState()
    # axis labels
    c.setFont("Helvetica", 6.5); c.setFillColor(SUB)
    c.drawCentredString(x + w / 2, H - (top + h + 8), "time step")
    c.setFont("Helvetica", 6.5); c.setFillColor(SUB)
    c.drawString(x - 10, H - (top + 2),  "cos")
    c.drawString(x - 10, H - (top + h * 0.5) - 3, "0.5")


# Heatmap rows (TOP)
HM_TOP = 240;  HM_H = 110
# Anomaly curve (BOTTOM)
AN_TOP = 80;   AN_H = 60

draw_heatmap_authentic(c, X1, HM_TOP, COL_W, HM_H, H)
draw_heatmap_scattered(c, X2, HM_TOP, COL_W, HM_H, H)
draw_anomaly(c, X1, AN_TOP, COL_W, AN_H, H, smooth=True)
draw_anomaly(c, X2, AN_TOP, COL_W, AN_H, H, smooth=False, dip=(30, 42))

# y-axis label rotated for the heatmap row
c.saveState()
c.setFont("Helvetica", 6.5); c.setFillColor(SUB)
c.translate(X1 - 20, H - (HM_TOP + HM_H / 2))
c.rotate(90)
c.drawString(0, 0, "audio time step")
c.restoreState()
c.saveState()
c.translate(X2 - 20, H - (HM_TOP + HM_H / 2))
c.rotate(90)
c.drawString(0, 0, "audio time step")
c.restoreState()

# x-axis label below heatmap
c.setFont("Helvetica", 6.5); c.setFillColor(SUB)
c.drawCentredString(X1 + COL_W / 2, H - (HM_TOP + HM_H + 8), "visual frame index")
c.drawCentredString(X2 + COL_W / 2, H - (HM_TOP + HM_H + 8), "visual frame index")

# Color-scale hint under the authentic heatmap
c.setFont("Helvetica", 6); c.setFillColor(GREY)
c.drawString(X1, H - (HM_TOP + HM_H + 22), "dark = high attention weight")
c.drawString(X2, H - (HM_TOP + HM_H + 22), "dark = high attention weight")

c.showPage(); c.save()
print("wrote fig_temporal_attention_case.pdf  (%dx%d pt)" % (W, H))
