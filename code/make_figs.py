# -*- coding: utf-8 -*-
"""
Generate watermark-free vector figures for the Sync-X thesis using reportlab's
native canvas API. v3 -- layout fixes vs the previous version:

  Taxonomy:
    * page widened so the four columns have breathing room
    * dashed B->C connector now anchored to the actual C-group box (not
      dangling into empty space)
    * uniform vertical spacing within each column so the bottom is no longer
      dominated by dead space

  Timeline:
    * page widened significantly to fit per-year columns without overlap
    * each year gets a dedicated x-coordinate; chips are auto-width but
      forced to stay within a per-year column
    * "2026" axis label is rendered ABOVE the 2026 cluster box
    * detector cluster multi-line label uses 4 lines (one per paper) with
      larger height to avoid clipping
    * 2026 cluster box width also auto-fitted to its widest line

Re-run any time:  python make_figs.py
Outputs: fig_av_taxonomy.pdf, fig_timeline.pdf
"""
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
import os

# ---- palette -------------------------------------------------------------
WHITE      = HexColor('#ffffff')
PANEL_FILL = HexColor('#eef1f4')
PANEL_STK  = HexColor('#9aa3ad')
CHIP_FILL  = HexColor('#ffffff')
CHIP_STK   = HexColor('#3a3f47')
RED_FILL   = HexColor('#c0392b')
RED_STK    = HexColor('#8e2b21')
YEL_FILL   = HexColor('#fff3cd')
YEL_STK    = HexColor('#d9a400')
INK        = HexColor('#1f2429')
GREY       = HexColor('#5b6470')
SUB        = HexColor('#3a3f47')
AXIS       = HexColor('#2b2f36')

OUTDIR = os.environ.get("SYNCX_FIG_OUTDIR",
                         r"C:/Users/ZhuanZ1/Desktop/my_proposal_papers/monash_thesis/Figures")


def fit_size(c, text, font, max_w, start=10, lo=5.0, step=0.5):
    s = start
    while s > lo and c.stringWidth(text, font, s) > max_w:
        s -= step
    return max(s, lo)


def vcenter(top, h, size, H_local):
    """Baseline y for a box whose TOP is `top` (top-down) and height h, on a
    page of height H_local."""
    return H_local - (top + h / 2.0) - size * 0.34


def rect(c, x, top, w, h, H_local, fill=None, stroke=None, lw=1.0, radius=0.0):
    c.saveState()
    if fill is not None:
        c.setFillColor(fill)
    if stroke is not None:
        c.setStrokeColor(stroke); c.setLineWidth(lw)
    f = 1 if fill is not None else 0
    s = 1 if stroke is not None else 0
    fx, fy = x, H_local - (top + h)
    if radius:
        c.roundRect(fx, fy, w, h, radius, stroke=s, fill=f)
    else:
        c.rect(fx, fy, w, h, stroke=s, fill=f)
    c.restoreState()


def chip(c, cx, top, h, label, H_local, fill=CHIP_FILL, stroke=CHIP_STK,
         font="Helvetica", size=9, tcolor=INK, pad=6, lw=1.0, radius=3.0,
         max_w=None):
    if max_w:
        size = fit_size(c, label, font, max_w - 2 * pad, start=size, lo=5.0)
    tw = c.stringWidth(label, font, size)
    w = tw + 2 * pad
    x = cx - w / 2.0
    rect(c, x, top, w, h, H_local, fill=fill, stroke=stroke, lw=lw, radius=radius)
    c.setFont(font, size); c.setFillColor(tcolor)
    c.drawCentredString(cx, vcenter(top, h, size, H_local), label)
    return x, w


def label(c, cx, top, h, text, H_local, font="Helvetica", size=9, color=INK):
    c.setFont(font, size); c.setFillColor(color)
    c.drawCentredString(cx, vcenter(top, h, size, H_local), text)


def leftlabel(c, x, top, h, text, H_local, font="Helvetica", size=9, color=INK):
    c.setFont(font, size); c.setFillColor(color)
    c.drawString(x, vcenter(top, h, size, H_local), text)


def rightlabel(c, x, top, h, text, H_local, font="Helvetica", size=9, color=INK):
    c.setFont(font, size); c.setFillColor(color)
    c.drawRightString(x, vcenter(top, h, size, H_local), text)


# =========================================================================
# FIGURE 1 -- Taxonomy  (720x520, fully balanced grid)
# =========================================================================
W, H = 720, 520
c = canvas.Canvas(OUTDIR + r"/fig_av_taxonomy.pdf", pagesize=(W, H))

# title block
label(c, W / 2, 8, 18, "Taxonomy of Audio-Visual Deepfake Detection",
      H, font="Helvetica-Bold", size=14, color=INK)
label(c, W / 2, 28, 12, "Methods released in 2026 are highlighted in red; the shaded branch is the direction taken in this thesis.",
      H, font="Helvetica", size=8, color=GREY)

# 4 column panels -- all extending to the SAME bottom for a clean baseline
TOP, BOT = 56, 460
# A: Unimodal,   B: Multimodal Fusion (widest),   C: Consistency-first (shaded),   D: LMM
A_x, A_w = 18,  120
B_x, B_w = 152, 270
C_x, C_w = 436, 170
D_x, D_w = 620,  82
A_cx = A_x + A_w / 2; B_cx = B_x + B_w / 2
C_cx = C_x + C_w / 2; D_cx = D_x + D_w / 2

rect(c, A_x, TOP, A_w, BOT - TOP, H, fill=PANEL_FILL, stroke=PANEL_STK, lw=1.2, radius=6)
rect(c, B_x, TOP, B_w, BOT - TOP, H, fill=PANEL_FILL, stroke=PANEL_STK, lw=1.2, radius=6)
# C is a *short* column (its 2 methods are the thesis direction); pin bottom
C_BOT = 200
rect(c, C_x, TOP, C_w, C_BOT - TOP, H, fill=YEL_FILL, stroke=YEL_STK, lw=1.2, radius=6)
# D is also short (1 method)
D_BOT = 165
rect(c, D_x, TOP, D_w, D_BOT - TOP, H, fill=PANEL_FILL, stroke=PANEL_STK, lw=1.2, radius=6)

# column headers
label(c, A_cx, 60, 16, "UNIMODAL", H, font="Helvetica-Bold", size=10, color=INK)
label(c, B_cx, 60, 16, "MULTIMODAL FUSION", H, font="Helvetica-Bold", size=10, color=INK)
label(c, C_cx, 60, 14, "CONSISTENCY-FIRST", H, font="Helvetica-Bold", size=8.5, color=INK)
label(c, C_cx, 74, 10, "(this thesis)", H, font="Helvetica-Oblique", size=7, color=GREY)
label(c, D_cx, 60, 16, "LMM-BASED", H, font="Helvetica-Bold", size=8.5, color=INK)

# sub-header + chip helper
def sub_header(x_col, top, text):
    leftlabel(c, x_col + 6, top, 12, text, H,
              font="Helvetica-Bold", size=7.5, color=SUB)

CH = 18  # chip height (smaller; column space is tight)
GAP = 6  # vertical gap between chips

# ===== Group A: UNIMODAL =====
# 3 sub-groups, each with 1 method (no need to spread to match B -- column
# widths differ; balance columns by row count not height)
sub_header(A_x, 90, "Visual-only")
chip(c, A_cx, 102, CH, "XceptionNet (2017)", H, max_w=A_w - 8, size=8)
sub_header(A_x, 132, "Audio-only")
chip(c, A_cx, 144, CH, "LipForensics (2021)", H, max_w=A_w - 8, size=8)
sub_header(A_x, 174, "Audio backbone")
chip(c, A_cx, 186, CH, "wav2vec 2.0 (2020)", H, max_w=A_w - 8, size=8)
# right-side note to mirror B's "AV consistency" emphasis
rightlabel(c, A_x + A_w - 6, 230, 12,
           "(no AV consistency\nmodelled here)", H,
           font="Helvetica-Oblique", size=6.5, color=GREY)

# ===== Group B: MULTIMODAL FUSION =====
# 3 sub-groups, vertically distributed, last sub-group near bottom of panel
y = 90
sub_header(B_x, y, "Early / late fusion"); y += 12
chip(c, B_cx, y, CH, "Zhou & Lim (2021)", H, max_w=B_w - 8, size=9); y += CH + GAP
y += 4
sub_header(B_x, y, "Cross-modal attention"); y += 12
chip(c, B_cx, y, CH, "AVT-DWF (2024)",    H, max_w=B_w - 8, size=9); y += CH + GAP
chip(c, B_cx, y, CH, "AVFF (2024)",       H, max_w=B_w - 8, size=9); y += CH + GAP
chip(c, B_cx, y, CH, "AV-Lip-Sync+ (2025)", H, max_w=B_w - 8, size=9); y += CH + GAP
chip(c, B_cx, y, CH, "MMMS-BA (2024)",    H, max_w=B_w - 8, size=9); y += CH + GAP
y += 4
sub_header(B_x, y, "Contrastive consistency"); y += 12
chip(c, B_cx, y, CH, "Modality Dissonance (2021)", H, max_w=B_w - 8, size=9); y += CH + GAP
chip(c, B_cx, y, CH, "AV-Contrastive (2025)",     H, max_w=B_w - 8, size=9); y += CH + GAP
chip(c, B_cx, y, CH, "AV attention w/ contrastive learning (2024)",
     H, max_w=B_w - 8, size=9); y += CH + GAP
chip(c, B_cx, y, CH, "MCL (2024)",       H, max_w=B_w - 8, size=9); y += CH + GAP

# Thin curved arrow from bottom-of-B to the C column -- conveys "B -> C"
# without drawing a red box.
c.saveState()
c.setStrokeColor(GREY); c.setLineWidth(0.8); c.setDash(2, 2)
arrow_y_top = y - 8
arrow_y_bot = y + 8
c.line(B_x + B_w - 4, arrow_y_top, C_x + 4, C_BOT - 6)
# Arrowhead at C side
c.line(C_x + 4, C_BOT - 6, C_x + 10, C_BOT - 10)
c.line(C_x + 4, C_BOT - 6, C_x + 8, C_BOT - 2)
c.restoreState()
rightlabel(c, B_x + B_w - 4, y - 4, 10,
           "trend: towards AV\nconsistency signals",
           H, font="Helvetica-Oblique", size=6.5, color=GREY)

# ===== Group C: CONSISTENCY-FIRST (shaded) -- 2026 red chips =====
# Place both 2026 chips INSIDE the C box, no floating "(this thesis)" outside
C_CH = 26
chip(c, C_cx, 90, C_CH, "X-AVDT (2026)", H, fill=RED_FILL, stroke=RED_STK,
     tcolor=WHITE, font="Helvetica-Bold", size=9, max_w=C_w - 8)
chip(c, C_cx, 122, C_CH, "LHCC (2026)", H, fill=RED_FILL, stroke=RED_STK,
     tcolor=WHITE, font="Helvetica-Bold", size=9, max_w=C_w - 8)
# "this thesis" tag, drawn inside the C box, beneath the chips
label(c, C_cx, 158, 10, "this thesis", H,
      font="Helvetica-Oblique", size=7, color=INK)
# Subtle separator above the C box indicating the unifying idea
rightlabel(c, C_x + 6, C_BOT - 12, 10, "bi-directional audio-visual consistency",
           H, font="Helvetica-Oblique", size=6, color=GREY)

# ===== Group D: LMM-BASED =====
chip(c, D_cx, 90, 28, "AV-LMMDetect (2026)", H, fill=RED_FILL, stroke=RED_STK,
     tcolor=WHITE, font="Helvetica-Bold", size=6.5, max_w=D_w - 4)
rightlabel(c, D_x + 4, 130, 10, "(not the\nfocus here)", H,
           font="Helvetica-Oblique", size=6, color=GREY)

# legend (bottom of page, below the panel baseline)
LEG_Y = 478
rect(c, 20, LEG_Y, 14, 14, H, fill=RED_FILL, stroke=RED_STK, lw=1, radius=2)
leftlabel(c, 40, LEG_Y - 2, 18, "2026", H, font="Helvetica", size=9, color=INK)
rect(c, 90, LEG_Y, 14, 14, H, fill=CHIP_FILL, stroke=CHIP_STK, lw=1, radius=2)
leftlabel(c, 110, LEG_Y - 2, 18, "earlier work", H, font="Helvetica", size=9, color=INK)
rect(c, 200, LEG_Y, 14, 14, H, fill=YEL_FILL, stroke=YEL_STK, lw=1, radius=2)
leftlabel(c, 220, LEG_Y - 2, 18, "direction of this thesis", H, font="Helvetica", size=9, color=INK)

c.showPage(); c.save()
print("wrote fig_av_taxonomy.pdf  (%dx%d pt)" % (W, H))


# =========================================================================
# FIGURE 2 -- Timeline  (widened to 920pt, per-year column discipline)
# =========================================================================
W, H = 920, 300
c = canvas.Canvas(OUTDIR + r"/fig_timeline.pdf", pagesize=(W, H))

label(c, W / 2, 8, 18, "Timeline of AV Deepfake Generation & Detection (2017-2026)",
      H, font="Helvetica-Bold", size=14, color=INK)
label(c, W / 2, 30, 12, "Top lane: generative technology. Bottom lane: detection methods. 2026 entries are highlighted in red.",
      H, font="Helvetica", size=8.5, color=GREY)

# year x-coordinates (per-year columns; padded out so 7 years span the page)
years = [(2017, 70), (2019, 195), (2021, 320), (2023, 445),
         (2024, 540), (2025, 635), (2026, 760)]
COL_W = 95   # half-width of chip's allowed column on each side of year tick

# axis
AXIS_Y = 175
c.saveState(); c.setStrokeColor(AXIS); c.setLineWidth(1.6)
c.line(40, H - AXIS_Y, W - 30, H - AXIS_Y); c.restoreState()
for yr, x in years:
    c.saveState(); c.setStrokeColor(AXIS); c.setLineWidth(1)
    c.line(x, H - (AXIS_Y - 4), x, H - (AXIS_Y + 4)); c.restoreState()
    col = RED_FILL if yr == 2026 else INK
    fnt = "Helvetica-Bold" if yr == 2026 else "Helvetica"
    label(c, x, AXIS_Y + 6, 12, str(yr), H, font=fnt, size=10, color=col)

# lane labels (left margin must be wide enough not to clip "DETECTORS")
leftlabel(c, 8, 65, 12, "GENERATORS", H, font="Helvetica-Bold", size=8.5, color=SUB)
leftlabel(c, 8, 220, 12, "DETECTORS", H, font="Helvetica-Bold", size=8.5, color=SUB)


def column_chip(year_x, top, h, text, H_local, max_w=COL_W * 2,
                fill=CHIP_FILL, stroke=CHIP_STK, tcolor=INK, size=8, pad=5):
    """Place a chip horizontally CENTERED on its year column, constrained to
    [year_x - max_w/2, year_x + max_w/2] so it never collides with neighbours.
    Text auto-shrinks to fit. Returns (x, w)."""
    fw = fit_size(c, text, "Helvetica", max_w - 2 * pad, start=size, lo=5.0)
    tw = c.stringWidth(text, "Helvetica", fw)
    w = tw + 2 * pad
    cx = year_x
    # guard: if width exceeds max_w, shrink further
    if w > max_w:
        w = max_w
        cx = max(w / 2 + 2, min(year_x, W - w / 2 - 2))
    x = cx - w / 2.0
    rect(c, x, top, w, h, H_local, fill=fill, stroke=stroke, lw=1, radius=3)
    c.setFont("Helvetica", fw); c.setFillColor(tcolor)
    c.drawCentredString(cx, vcenter(top, h, fw, H_local), text)
    return x, w


# dashed connectors
def tick_to(year_x, top, bot):
    c.saveState()
    c.setStrokeColor(GREY); c.setLineWidth(0.7); c.setDash(2, 2)
    c.line(year_x, H - top, year_x, H - bot)
    c.restoreState()

# ---- generators (top lane) ----
GEN_TOP = 78; GEN_H = 22
# Note: 2021 has NO generator chip; skip its tick.
gen_years = [2017, 2019, 2023, 2024, 2025, 2026]
for yr in gen_years:
    x = dict([(y, x) for y, x in years])[yr]
    tick_to(x, 102, AXIS_Y - 4)
chip_data_gen = [
    (2017, "FaceSwap / DeepFaceLab", False),
    (2019, "FaceForensics++", False),
    (2023, "Deepfake-Eval-2024 (host)", False),
    (2024, "Seedance 1.0", False),
    (2025, "Seedance 1.5 Pro", False),
    (2026, "Seedance 2.0 / SkyReels-V4", True),
]
for yr, lbl, red in chip_data_gen:
    x = dict([(y, x) for y, x in years])[yr]
    if red:
        column_chip(x, GEN_TOP, GEN_H + 2, lbl, H,
                    fill=RED_FILL, stroke=RED_STK, tcolor=WHITE, size=7.5)
    else:
        column_chip(x, GEN_TOP, GEN_H, lbl, H, size=8)

# ---- detectors (bottom lane) ----
DET_TOP = 215; DET_H = 22
# 2019 and 2026 have NO standalone detector chip; skip those ticks.
# (2026 has the red cluster which is positioned slightly higher.)
det_years = [2017, 2021, 2023, 2024, 2025]
for yr in det_years:
    x = dict([(y, x) for y, x in years])[yr]
    tick_to(x, AXIS_Y + 4, 213)

# primary row
primary = [(2017, "XceptionNet", False),
           (2021, "MDS / Zhou & Lim", False),
           (2023, "LipForensics", False),
           (2025, "AV-Lip-Sync+", False)]
for yr, lbl, red in primary:
    x = dict([(y, x) for y, x in years])[yr]
    column_chip(x, DET_TOP, DET_H, lbl, H, size=8)

# 2024 column: two stacked chips (LAV-DF on top, AVFF/AVT-DWF below)
x24 = dict([(y, x) for y, x in years])[2024]
column_chip(x24, 215, DET_H, "LAV-DF", H, size=8)
column_chip(x24, 240, DET_H, "AVFF / AVT-DWF", H, size=8)

# 2026 cluster (3 lines). Order: most thesis-relevant first.
# X-AVDT is the closest 2026 baseline to this thesis; LHCC is consistency-first;
# AV-LMMDetect is LMM; DeepfakeBench-MM is the benchmark; NTIRE is the challenge.
red_lines = ["X-AVDT / LHCC", "AV-LMMDetect / DeepfakeBench-MM", "NTIRE 2026"]
rsize = 7.0
while rsize > 5.5 and max(c.stringWidth(l, "Helvetica-Bold", rsize) for l in red_lines) > COL_W * 2 - 10:
    rsize -= 0.2
red_w = max(c.stringWidth(l, "Helvetica-Bold", rsize) for l in red_lines) + 12
red_h = 11 * len(red_lines) + 6
x26 = dict([(y, x) for y, x in years])[2026]
# shift left if it would overflow the right page margin
if x26 + red_w / 2 > W - 20:
    x26 = W - 20 - red_w / 2
# move cluster slightly above the detector row so the 2026 label is visible
rect(c, x26 - red_w / 2, DET_TOP, red_w, red_h + 4, H,
     fill=RED_FILL, stroke=RED_STK, lw=1.2, radius=3)
c.setFont("Helvetica-Bold", rsize); c.setFillColor(WHITE)
# First line at TOP of box (top-down yy = DET_TOP + small offset);
# subsequent lines go DOWN (yy += 11).
yy = DET_TOP + 11
for ln in red_lines:
    c.drawCentredString(x26, H - (yy + 3), ln)
    yy += 11

# 2026 year label -- place it ABOVE the axis, at a slightly different y
# than the regular years, so it doesn't overlap the red cluster.
# (axis labels live at y=181, red cluster top is at y=215+red_h+4 ~ y=251;
# the label at y=181 is comfortably above.)
# Also, draw a small white halo behind the "2026" text so it pops.
c.saveState()
c.setFillColor(WHITE)
c.rect(dict([(y, x) for y, x in years])[2026] - 14, H - (AXIS_Y + 20),
       28, 14, fill=1, stroke=0)
c.restoreState()
label(c, dict([(y, x) for y, x in years])[2026], AXIS_Y + 6, 12, "2026",
      H, font="Helvetica-Bold", size=10, color=RED_FILL)

# legend (bottom-left)
rect(c, 20, 275, 14, 14, H, fill=RED_FILL, stroke=RED_STK, lw=1, radius=2)
leftlabel(c, 40, 273, 18, "2026", H, font="Helvetica", size=9, color=INK)
rect(c, 90, 275, 14, 14, H, fill=CHIP_FILL, stroke=CHIP_STK, lw=1, radius=2)
leftlabel(c, 110, 273, 18, "earlier work", H, font="Helvetica", size=9, color=INK)

c.showPage(); c.save()
print("wrote fig_timeline.pdf   (%dx%d pt)" % (W, H))
