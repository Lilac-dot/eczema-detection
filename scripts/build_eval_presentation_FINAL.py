# -*- coding: utf-8 -*-
"""Rebuilds the Honors evaluation presentation from scratch so it matches the CURRENT
paper (eczema_compression_deployment_paper_2026-09-20.docx), replacing the outdated
edge_ai_eval_presentation_2026-09-19.pptx (built for the superseded "Compression
Robustness Is Architecture-Dependent" draft, before the backend-dependent INT8 finding
existed). Every number below is read from this project's own JSON result files or
copied from values already verified against those files in this session -- nothing is
invented. See papers/edge-ai-lightweight-deployment/PRESENTATION_CHANGELOG_2026-09-20.md
for the slide-by-slide list of what changed and why.
"""
import json
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.oxml.ns import qn

from paths import ROOT

PAPER_DIR = ROOT / "papers" / "edge-ai-lightweight-deployment"
OUT_PATH = PAPER_DIR / "edge_ai_eval_presentation_FINAL.pptx"

# ---------- palette (contrast-checked against the light background in this session) ----
DARK = RGBColor(0x14, 0x21, 0x3D)      # navy -- headings, dark panels
LIGHT_BG = RGBColor(0xFB, 0xFB, 0xF8)  # off-white slide background
CARD_BG = RGBColor(0xFF, 0xFF, 0xFF)
CARD_BORDER = RGBColor(0xE3, 0xE1, 0xD8)
ROW_ALT = RGBColor(0xF1, 0xEF, 0xE6)
BODY = RGBColor(0x3A, 0x3F, 0x44)
MUTED = RGBColor(0x5B, 0x61, 0x69)
FAINT = RGBColor(0x8A, 0x8F, 0x87)
GOLD = RGBColor(0xC9, 0xA2, 0x27)          # decorative fills only
GOLD_TEXT_ON_DARK = RGBColor(0xE8, 0xC5, 0x47)
LIGHT_TEXT_ON_DARK = RGBColor(0xD9, 0xD6, 0xC9)
RED = RGBColor(0xA3, 0x2C, 0x2C)           # collapsed
AMBER = RGBColor(0x96, 0x60, 0x0E)         # severely degraded
GREEN = RGBColor(0x2F, 0x6B, 0x4F)         # backend-stable
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

HEAD_FONT = "Georgia"
BODY_FONT = "Calibri"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
TOTAL_SLIDES = 14

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
BLANK = prs.slide_layouts[6]


def new_slide(bg_color=LIGHT_BG):
    s = prs.slides.add_slide(BLANK)
    rect = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    rect.fill.solid()
    rect.fill.fore_color.rgb = bg_color
    rect.line.fill.background()
    rect.shadow.inherit = False
    # send to back
    sp = rect._element
    sp.getparent().remove(sp)
    s.shapes._spTree.insert(2, sp)
    return s


def _set_run(run, text, size=18, bold=False, italic=False, color=BODY, font=BODY_FONT):
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = font


def textbox(slide, left, top, width, height, text, size=18, bold=False, italic=False,
            color=BODY, font=BODY_FONT, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
            line_spacing=1.0, wrap=True):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = align
    if line_spacing:
        p.line_spacing = line_spacing
    r = p.add_run()
    _set_run(r, text, size, bold, italic, color, font)
    return box


def multiline(slide, left, top, width, height, lines, anchor=MSO_ANCHOR.TOP):
    """lines: list of dicts {text, size, bold, italic, color, font, space_after, align, bullet}"""
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = ln.get("align", PP_ALIGN.LEFT)
        p.line_spacing = ln.get("line_spacing", 1.15)
        p.space_after = Pt(ln.get("space_after", 8))
        if ln.get("bullet"):
            _enable_bullet(p)
        r = p.add_run()
        _set_run(r, ln["text"], ln.get("size", 16), ln.get("bold", False),
                  ln.get("italic", False), ln.get("color", BODY), ln.get("font", BODY_FONT))
    return box


def _enable_bullet(paragraph):
    pPr = paragraph._p.get_or_add_pPr()
    pPr.set("indent", "-228600")
    pPr.set("marL", "228600")
    buFont = pPr.makeelement(qn("a:buFont"), {"typeface": "Arial"})
    buChar = pPr.makeelement(qn("a:buChar"), {"char": "•"})
    pPr.append(buFont)
    pPr.append(buChar)


def rounded_card(slide, left, top, width, height, fill=CARD_BG, line_color=CARD_BORDER,
                  radius=0.06):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shp.adjustments[0] = radius
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line_color:
        shp.line.color.rgb = line_color
        shp.line.width = Pt(1)
    else:
        shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def header(slide, kicker, title, title_size=34, top=Inches(0.45)):
    textbox(slide, Inches(0.7), top, Inches(11.9), Inches(0.4), kicker.upper(),
            size=15, bold=True, color=DARK, font=BODY_FONT)
    textbox(slide, Inches(0.7), top + Inches(0.38), Inches(11.9), Inches(0.9), title,
            size=title_size, bold=True, color=DARK, font=HEAD_FONT, line_spacing=1.05)


def footer(slide, page_num, source_text):
    textbox(slide, Inches(0.7), Inches(7.08), Inches(3), Inches(0.35),
            f"{page_num} / {TOTAL_SLIDES}", size=12, color=FAINT)
    textbox(slide, Inches(7.5), Inches(7.08), Inches(5.15), Inches(0.35), source_text,
            size=12, color=FAINT, align=PP_ALIGN.RIGHT)


def make_table(slide, left, top, width, height, headers, rows, col_widths_frac,
               font_size=14, header_fill=DARK, header_color=WHITE, alt_row=ROW_ALT,
               cell_colors=None):
    """cell_colors: optional dict {(row_idx, col_idx): RGBColor} for data-row text (0-indexed data rows)."""
    n_rows = len(rows) + 1
    n_cols = len(headers)
    gshape = slide.shapes.add_table(n_rows, n_cols, left, top, width, height)
    tbl = gshape.table
    for c, frac in enumerate(col_widths_frac):
        tbl.columns[c].width = Emu(int(width * frac))
    tbl.rows[0].height = Emu(int(height / n_rows))
    for c, h in enumerate(headers):
        cell = tbl.cell(0, c)
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_fill
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = Pt(6); cell.margin_right = Pt(6)
        cell.margin_top = Pt(2); cell.margin_bottom = Pt(2)
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT
        r = p.add_run()
        _set_run(r, h, font_size, True, False, header_color, BODY_FONT)
    for ri, row in enumerate(rows):
        tbl.rows[ri + 1].height = Emu(int(height / n_rows))
        for c, val in enumerate(row):
            cell = tbl.cell(ri + 1, c)
            cell.fill.solid()
            cell.fill.fore_color.rgb = alt_row if ri % 2 == 1 else CARD_BG
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = Pt(6); cell.margin_right = Pt(6)
            cell.margin_top = Pt(2); cell.margin_bottom = Pt(2)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT
            r = p.add_run()
            col = BODY
            bold = False
            if cell_colors and (ri, c) in cell_colors:
                col = cell_colors[(ri, c)]
                bold = True
            _set_run(r, str(val), font_size, bold, False, col, BODY_FONT)
    return gshape


STATUS_COLOR = {"backend-stable": GREEN, "severely degraded": AMBER, "collapsed": RED,
                 "degraded": AMBER}

print("Building slides...")

# ================= SLIDE 1 — TITLE =================
s = new_slide(bg_color=DARK)
textbox(s, Inches(0.9), Inches(0.9), Inches(11.5), Inches(0.4),
        "HONORS PROJECT — FLEXIBLE ELECTRONICS FOR HUMAN HEALTHCARE",
        size=15, bold=True, color=GOLD_TEXT_ON_DARK, font=BODY_FONT)
textbox(s, Inches(0.9), Inches(1.9), Inches(11.5), Inches(2.0),
        "Backend-Dependent INT8 Quantization\nBehavior in an Eczema Image Classifier",
        size=40, bold=True, color=WHITE, font=HEAD_FONT, line_spacing=1.08)
textbox(s, Inches(0.9), Inches(4.05), Inches(11.5), Inches(0.6),
        "A 9-Architecture Study for Edge and Wearable Healthcare Deployment",
        size=20, color=LIGHT_TEXT_ON_DARK, font=BODY_FONT)
multiline(s, Inches(0.9), Inches(5.6), Inches(8), Inches(1.2), [
    {"text": "Tishya Yadlapalli", "size": 18, "color": LIGHT_TEXT_ON_DARK, "space_after": 4},
    {"text": "Mentor: Dr. Priyanka Dwivedi", "size": 18, "color": LIGHT_TEXT_ON_DARK, "space_after": 4},
])
textbox(s, Inches(0.9), Inches(6.85), Inches(8), Inches(0.4),
        "github.com/Lilac-dot/eczema-detection", size=13, color=RGBColor(0x85, 0x93, 0xA8))

# ================= SLIDE 2 — MOTIVATION + RESEARCH QUESTIONS =================
s = new_slide()
header(s, "Motivation", "Why compression, and does it cost the same for every architecture?")
multiline(s, Inches(0.7), Inches(1.55), Inches(5.6), Inches(2.3), [
    {"text": "Wearable / edge healthcare deployment needs:", "size": 17, "bold": True, "color": DARK, "space_after": 8},
    {"text": "Small memory footprint", "size": 16, "bullet": True, "space_after": 6},
    {"text": "Low latency", "size": 16, "bullet": True, "space_after": 6},
    {"text": "Low power draw", "size": 16, "bullet": True, "space_after": 6},
    {"text": "Compression (pruning + quantization) is the mechanism that gets a trained classifier onto that hardware.", "size": 16, "space_after": 0, "color": MUTED},
])
rounded_card(s, Inches(6.6), Inches(1.55), Inches(6.0), Inches(1.5), fill=DARK, line_color=None)
textbox(s, Inches(6.9), Inches(1.75), Inches(5.4), Inches(0.4), "CORE QUESTION",
        size=14, bold=True, color=GOLD_TEXT_ON_DARK)
textbox(s, Inches(6.9), Inches(2.1), Inches(5.4), Inches(0.9),
        "Does compression affect all 9 architectures the same way?",
        size=21, bold=True, color=WHITE, font=HEAD_FONT, line_spacing=1.1)
rq_lines = [
    "1. Does compression affect architectures differently?",
    "2. Can quantization-aware training (QAT) recover PTQ losses?",
    "3. Does the quantized CPU backend change INT8 behavior?",
    "4. Which architectures remain feasible after compression?",
    "5. Do findings reproduce under corruption and held-out test evaluation?",
]
multiline(s, Inches(6.6), Inches(3.35), Inches(6.0), Inches(3.2),
          [{"text": t, "size": 16, "space_after": 12, "color": BODY} for t in rq_lines])
footer(s, 2, "Section 1 · Research questions")

# ================= SLIDE 3 — DATASET + ARCHITECTURES =================
s = new_slide()
header(s, "Dataset & Architectures", "3,330 curated images, 9 candidate architectures", title_size=30)
textbox(s, Inches(0.7), Inches(1.35), Inches(11.9), Inches(0.5),
        "Binary task: eczema-labeled vs. other skin-disease classes. “Eczema” is this "
        "project's own dataset label, not a dermatologist-verified diagnosis.",
        size=15, color=MUTED, italic=True)
arch_rows = [
    ["ResNet18 (baseline)", "No", "80.24%", "42.72"],
    ["EfficientNet-B0", "Yes (SE + SiLU)", "80.65%", "16.20"],
    ["EfficientNet-Lite0", "No", "76.41%", "13.75"],
    ["MobileNetV3-Small", "Yes (SE + Hardswish)", "76.01%", "3.95"],
    ["MobileNetV2-1.0x", "No", "78.83%", "9.35"],
    ["ShuffleNetV2-0.5x", "No", "79.84%", "1.95"],
    ["ShuffleNetV2-1.0x", "No", "79.23%", "5.45"],
    ["SqueezeNet1.1", "No", "78.02%", "3.03"],
    ["RepGhostNet-0.5x", "Yes (Squeeze-Excite)", "76.81%", "4.87"],
]
make_table(s, Inches(0.7), Inches(2.0), Inches(9.0), Inches(4.35),
           ["Architecture", "SE / swish?", "FP32 Acc.", "Size (MB)"], arch_rows,
           [0.42, 0.28, 0.16, 0.14], font_size=14)
rounded_card(s, Inches(9.9), Inches(2.0), Inches(2.7), Inches(4.35), fill=DARK, line_color=None)
multiline(s, Inches(10.15), Inches(2.25), Inches(2.25), Inches(4.0), [
    {"text": "SPLIT", "size": 13, "bold": True, "color": GOLD_TEXT_ON_DARK, "space_after": 10},
    {"text": "2,327", "size": 26, "bold": True, "color": WHITE, "space_after": 0},
    {"text": "train images", "size": 13, "color": LIGHT_TEXT_ON_DARK, "space_after": 16},
    {"text": "496", "size": 26, "bold": True, "color": WHITE, "space_after": 0},
    {"text": "validation images", "size": 13, "color": LIGHT_TEXT_ON_DARK, "space_after": 16},
    {"text": "507", "size": 26, "bold": True, "color": WHITE, "space_after": 0},
    {"text": "held-out test images", "size": 13, "color": LIGHT_TEXT_ON_DARK, "space_after": 16},
    {"text": "No patient identifiers in the manifest.", "size": 12, "italic": True, "color": RGBColor(0x85,0x93,0xA8)},
])
footer(s, 3, "Section 3 · Dataset and Task")

# ================= SLIDE 4 — EXPERIMENTAL DESIGN =================
s = new_slide()
header(s, "Experimental Design", "One pipeline, every compression condition", title_size=30)

def pipe_box(x, y, w, h, title, sub):
    shp = rounded_card(s, x, y, w, h, fill=DARK, line_color=None, radius=0.12)
    textbox(s, x + Inches(0.12), y + Inches(0.12), w - Inches(0.24), Inches(0.35), title,
            size=15, bold=True, color=GOLD_TEXT_ON_DARK, align=PP_ALIGN.CENTER)
    textbox(s, x + Inches(0.12), y + Inches(0.5), w - Inches(0.24), h - Inches(0.6), sub,
            size=12.5, color=LIGHT_TEXT_ON_DARK, align=PP_ALIGN.CENTER, line_spacing=1.1)

def arrow(x, y, w=Inches(0.27), h=Inches(0.4)):
    ar = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x, y, w, h)
    ar.fill.solid(); ar.fill.fore_color.rgb = GOLD
    ar.line.fill.background()
    ar.shadow.inherit = False

bx_y = Inches(1.65)
bx_h = Inches(1.55)
bx_w = Inches(2.05)
gap = Inches(0.33)
xs = [Inches(0.7) + i * (bx_w + gap) for i in range(5)]
pipe_box(xs[0], bx_y, bx_w, bx_h, "TRAINING DATA", "9 FP32 architectures trained identically on the training split only")
arrow(xs[0] + bx_w + Inches(0.03), bx_y + bx_h/2 - Inches(0.2))
pipe_box(xs[1], bx_y, bx_w, bx_h, "PRUNING", "Unstructured L1 · structured / channel · pruning + fine-tune")
arrow(xs[1] + bx_w + Inches(0.03), bx_y + bx_h/2 - Inches(0.2))
pipe_box(xs[2], bx_y, bx_w, bx_h, "QUANTIZATION", "Static INT8 PTQ · QAT · x86/fbgemm vs. ARM/qnnpack")
arrow(xs[2] + bx_w + Inches(0.03), bx_y + bx_h/2 - Inches(0.2))
pipe_box(xs[3], bx_y, bx_w, bx_h, "ROBUSTNESS", "7 image corruptions · SkinDisNet zero-shot check")
arrow(xs[3] + bx_w + Inches(0.03), bx_y + bx_h/2 - Inches(0.2))
pipe_box(xs[4], bx_y, bx_w, bx_h, "FINAL CONFIRMATION", "Frozen protocol · 507-image held-out test, evaluated once")

multiline(s, Inches(0.7), Inches(3.75), Inches(11.9), Inches(2.6), [
    {"text": "Test-set discipline", "size": 18, "bold": True, "color": DARK, "space_after": 10},
    {"text": "INT8 calibration uses the training split only — never validation or test.", "size": 16, "bullet": True, "space_after": 8},
    {"text": "The validation split (496 images) is used for every exploratory result in this deck (Slides 5–11).", "size": 16, "bullet": True, "space_after": 8},
    {"text": "The test split (507 images) was opened exactly once, after the evaluation protocol was written to disk — not “never opened.”", "size": 16, "bullet": True, "space_after": 8},
    {"text": "Two CPU quantized backends are compared throughout: x86/fbgemm and ARM/qnnpack — not one.", "size": 16, "bullet": True, "space_after": 8},
    {"text": "Both unstructured AND structured (channel) pruning were tested — structured pruning ran successfully for 7 of 9 architectures.", "size": 16, "bullet": True, "space_after": 0},
])
footer(s, 4, "Section 4 · Methods")

# ================= SLIDE 5 — FINDING 1: PRUNING KNEES =================
s = new_slide()
header(s, "Finding 1", "Pruning sensitivity is architecture-dependent", title_size=32)
knee_labels = ["ResNet18", "EfficientNet-B0", "EfficientNet-\nLite0", "MobileNetV3-\nSmall",
               "MobileNetV2-\n1.0x", "ShuffleNetV2-\n0.5x", "ShuffleNetV2-\n1.0x", "SqueezeNet1.1",
               "RepGhostNet-\n0.5x"]
knee_values = [60, 50, 50, 40, 20, 40, 50, 30, 40]
chart_data = CategoryChartData()
chart_data.categories = knee_labels
chart_data.add_series("Pruning knee (%)", knee_values)
gframe = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.7), Inches(1.5),
                             Inches(8.1), Inches(4.6), chart_data)
chart = gframe.chart
chart.has_legend = False
plot = chart.plots[0]
plot.has_data_labels = True
plot.data_labels.number_format = "0\\%"
plot.data_labels.number_format_is_linked = False
plot.data_labels.font.size = Pt(12)
series = plot.series[0]
series.format.fill.solid()
series.format.fill.fore_color.rgb = DARK
# highlight MobileNetV2-1.0x (index 4) in the status-amber color
pt4 = series.points[4]
pt4.format.fill.solid()
pt4.format.fill.fore_color.rgb = AMBER
cat_ax = chart.category_axis
cat_ax.tick_labels.font.size = Pt(10.5)
val_ax = chart.value_axis
val_ax.tick_labels.font.size = Pt(11)
val_ax.has_major_gridlines = False

rounded_card(s, Inches(9.1), Inches(1.5), Inches(3.5), Inches(4.6), fill=DARK, line_color=None)
multiline(s, Inches(9.35), Inches(1.75), Inches(3.0), Inches(4.1), [
    {"text": "“Knee” definition", "size": 14, "bold": True, "color": GOLD_TEXT_ON_DARK, "space_after": 8},
    {"text": "First sparsity level where accuracy drops more than 2 points below that architecture's own 0%-sparsity baseline.", "size": 14, "color": LIGHT_TEXT_ON_DARK, "space_after": 18},
    {"text": "No single sparsity target is uniformly safe.", "size": 19, "bold": True, "color": WHITE, "space_after": 14},
    {"text": "MobileNetV2-1.0x has the earliest knee, at 20% — half of the next-earliest.", "size": 14, "color": LIGHT_TEXT_ON_DARK, "space_after": 10},
    {"text": "A knee marks where accuracy starts falling — not the best achievable sparsity for that architecture.", "size": 12.5, "italic": True, "color": RGBColor(0x85,0x93,0xA8)},
])
footer(s, 5, "Paper Table 1 · Section 5.1")

# ================= SLIDE 6 — SE/SWISH ASSOCIATION =================
s = new_slide()
header(s, "Finding 2", "SE / swish architectures show far larger PTQ degradation", title_size=28)
seswish_rows = [
    ["EfficientNet-B0", "80.65%", "53.0%", "−27.6 pt"],
    ["MobileNetV3-Small", "76.01%", "50.4%", "−25.6 pt"],
    ["RepGhostNet-0.5x", "76.81%", "48.4%", "−28.4 pt"],
    ["EfficientNet-Lite0", "76.41%", "66.7%", "−9.7 pt"],
    ["MobileNetV2-1.0x", "78.83%", "72.0%", "−6.9 pt"],
    ["ShuffleNetV2-0.5x", "79.84%", "76.6%", "−3.2 pt"],
]
make_table(s, Inches(0.7), Inches(1.6), Inches(7.1), Inches(3.5),
           ["Architecture", "FP32", "INT8 (fbgemm)", "ΔAcc"], seswish_rows,
           [0.42, 0.19, 0.22, 0.17], font_size=14.5,
           cell_colors={(0,3): RED, (1,3): RED, (2,3): RED})
textbox(s, Inches(0.7), Inches(5.25), Inches(7.1), Inches(0.4),
        "+ ResNet18, ShuffleNetV2-1.0x, SqueezeNet1.1: all −1.4 pt or better (not shown)",
        size=13, italic=True, color=MUTED)

rounded_card(s, Inches(8.1), Inches(1.6), Inches(4.5), Inches(3.9), fill=DARK, line_color=None)
textbox(s, Inches(8.35), Inches(1.8), Inches(4.0), Inches(0.35), "THE MATH",
        size=13, bold=True, color=GOLD_TEXT_ON_DARK)
textbox(s, Inches(8.35), Inches(2.15), Inches(4.0), Inches(0.9), "27.2%  vs.  3.9%",
        size=34, bold=True, color=WHITE, font=HEAD_FONT)
textbox(s, Inches(8.35), Inches(2.95), Inches(4.0), Inches(0.5),
        "mean drop: SE/swish (n=3) vs. other 6 architectures", size=14, color=LIGHT_TEXT_ON_DARK)
textbox(s, Inches(8.35), Inches(3.5), Inches(4.0), Inches(0.5),
        "= 23.3 pt difference ≈ 7.0×", size=20, bold=True, color=GOLD_TEXT_ON_DARK)
textbox(s, Inches(8.35), Inches(4.15), Inches(4.0), Inches(1.2),
        "Association, not demonstrated causation.", size=17, bold=True, italic=True, color=WHITE)
textbox(s, Inches(0.7), Inches(5.85), Inches(11.9), Inches(1.0),
        "A component-removal ablation (removing SE blocks / replacing swish, without retraining) was "
        "attempted but is confounded: it collapses FP32 accuracy to ~50% on its own, so it cannot "
        "isolate a causal quantization effect. This finding is a measured architecture-family "
        "correlation, not a proven mechanism.",
        size=14.5, color=MUTED, line_spacing=1.2)
footer(s, 6, "Paper Table 2 · Section 5.2")

# ================= SLIDE 7 — PRIMARY FINDING: BACKEND-DEPENDENT COLLAPSE =================
s = new_slide()
header(s, "Primary finding", "Identical weights, different INT8 backend — very different outcomes", title_size=26)
backend_rows = [
    ["ResNet18", "79.6%", "79.6%", "backend-stable"],
    ["EfficientNet-B0", "53.0%", "48.8%", "severely degraded"],
    ["EfficientNet-Lite0", "66.7%", "49.8%", "collapsed"],
    ["MobileNetV3-Small", "50.4%", "48.2%", "severely degraded"],
    ["MobileNetV2-1.0x", "72.0%", "49.8%", "collapsed"],
    ["ShuffleNetV2-0.5x", "76.6%", "73.6%", "backend-stable"],
    ["ShuffleNetV2-1.0x", "77.8%", "74.4%", "backend-stable"],
    ["SqueezeNet1.1", "76.6%", "76.4%", "backend-stable"],
    ["RepGhostNet-0.5x", "48.4%", "49.8%", "collapsed"],
]
cell_colors = {(i, 3): STATUS_COLOR[row[3]] for i, row in enumerate(backend_rows)}
make_table(s, Inches(0.7), Inches(1.45), Inches(7.7), Inches(4.55),
           ["Architecture", "fbgemm INT8", "qnnpack INT8", "Status"], backend_rows,
           [0.34, 0.22, 0.22, 0.22], font_size=14, cell_colors=cell_colors)

rounded_card(s, Inches(8.7), Inches(1.45), Inches(3.9), Inches(4.55), fill=DARK, line_color=None)
big_stats = [
    ("3 / 9", "collapsed", RED),
    ("2 / 9", "severely degraded", AMBER),
    ("4 / 9", "backend-stable", GREEN),
]
yy = Inches(1.65)
for label, sub, col in big_stats:
    textbox(s, Inches(8.95), yy, Inches(3.4), Inches(0.7), label, size=36, bold=True, color=col, font=HEAD_FONT)
    textbox(s, Inches(8.95), yy + Inches(0.62), Inches(3.4), Inches(0.35), sub.upper(), size=13, bold=True, color=LIGHT_TEXT_ON_DARK)
    yy += Inches(1.12)
textbox(s, Inches(8.95), Inches(5.15), Inches(3.4), Inches(0.5),
        "5 / 9 (55.6%) fail this study's backend-stability criterion",
        size=15, bold=True, color=WHITE, line_spacing=1.1)
textbox(s, Inches(8.95), Inches(5.65), Inches(3.4), Inches(0.3), "on the ARM/qnnpack backend tested",
        size=12, italic=True, color=RGBColor(0x85,0x93,0xA8))

textbox(s, Inches(0.7), Inches(6.2), Inches(11.9), Inches(0.7),
        "Backend-dependent INT8 behavior under the default recipes tested on this one ARM/qnnpack "
        "machine — not a claim about ARM devices, or PyTorch quantization, in general.",
        size=14, italic=True, color=MUTED, line_spacing=1.15)
footer(s, 7, "Paper Table 5 · Section 5.5")

# ================= SLIDE 8 — WHAT EXPLAINS THE BACKEND GAP =================
s = new_slide()
header(s, "What explains the backend gap?", "Weight-quantization granularity: part, not all, of the story", title_size=27)
diag_rows = [
    ["EfficientNet-B0", "0.080", "0.649"],
    ["MobileNetV3-Small", "0.065", "0.165"],
    ["EfficientNet-Lite0", "0.000", "0.000"],
    ["MobileNetV2-1.0x", "0.000", "0.000"],
    ["RepGhostNet-0.5x", "0.000", "0.000"],
]
diag_colors = {(0,2): GREEN, (1,2): AMBER, (2,2): RED, (3,2): RED, (4,2): RED}
make_table(s, Inches(0.7), Inches(1.55), Inches(7.9), Inches(2.9),
           ["Architecture", "qnnpack default F1\n(per-tensor)", "Forced per-channel F1"],
           diag_rows, [0.42, 0.29, 0.29], font_size=15, cell_colors=diag_colors)
textbox(s, Inches(0.7), Inches(4.7), Inches(7.9), Inches(1.4),
        "qnnpack defaults to per-tensor weight quantization; fbgemm defaults to per-channel. "
        "Forcing per-channel quantization on qnnpack isolates that one variable — recovering most "
        "of EfficientNet-B0's F1, part of MobileNetV3-Small's, and none of the other three, which "
        "stay fully collapsed regardless.",
        size=15, color=BODY, line_spacing=1.25)
rounded_card(s, Inches(8.9), Inches(1.55), Inches(3.7), Inches(4.55), fill=DARK, line_color=None)
textbox(s, Inches(9.15), Inches(1.85), Inches(3.2), Inches(2.0),
        "Quantization recipe differences explain part — not all — of the backend effect.",
        size=20, bold=True, color=WHITE, font=HEAD_FONT, line_spacing=1.15)
textbox(s, Inches(9.15), Inches(4.1), Inches(3.2), Inches(1.8),
        "A partial, architecture-specific contributor — not a single mechanism that accounts for "
        "every architecture in Slide 7's table.",
        size=14, italic=True, color=LIGHT_TEXT_ON_DARK, line_spacing=1.2)
footer(s, 8, "Paper Table 5a · Section 5.5.1")

# ================= SLIDE 9 — QAT + PRUNING RECOVERY =================
s = new_slide()
header(s, "Recovery attempts", "QAT did not rescue the collapsed models; fine-tuning after pruning mostly worked", title_size=24)
# left panel: QAT
textbox(s, Inches(0.7), Inches(1.5), Inches(5.7), Inches(0.4), "QUANTIZATION-AWARE TRAINING (QAT, on qnnpack)",
        size=14, bold=True, color=DARK)
qat_rows = [
    ["EfficientNet-B0", "48.8%", "51.4%"],
    ["MobileNetV3-Small", "48.2%", "49.8%"],
    ["RepGhostNet-0.5x", "49.8%", "55.0%"],
]
make_table(s, Inches(0.7), Inches(1.95), Inches(5.7), Inches(1.75),
           ["Architecture", "PTQ (qnnpack)", "After 3-epoch QAT"], qat_rows,
           [0.44, 0.28, 0.28], font_size=13.5,
           cell_colors={(0,2): AMBER, (1,2): AMBER, (2,2): AMBER})
textbox(s, Inches(0.7), Inches(3.8), Inches(5.7), Inches(1.1),
        "3-epoch QAT did not restore usable accuracy after the qnnpack PTQ collapse — every result "
        "stays within a few points of the ~50% constant-classifier floor.",
        size=14.5, color=MUTED, line_spacing=1.2)
# right panel: pruning + fine-tune
textbox(s, Inches(6.9), Inches(1.5), Inches(5.7), Inches(0.4), "PRUNING + FINE-TUNING",
        size=14, bold=True, color=DARK)
ft_rows = [
    ["ResNet18", "30/50/70%", "84.1 / 82.3 / 81.5%", "recovers fully"],
    ["ShuffleNetV2-1.0x", "30/50/70%", "80.0 / 79.8 / 75.8%", "recovers fully"],
    ["MobileNetV3-Small", "30/50/70%", "79.2 / 74.2 / 49.8%", "partial at 70%"],
]
ft_colors = {(0,3): GREEN, (1,3): GREEN, (2,3): AMBER}
make_table(s, Inches(6.9), Inches(1.95), Inches(5.7), Inches(1.75),
           ["Architecture", "Sparsity", "Acc. after fine-tune", "Result"], ft_rows,
           [0.28, 0.16, 0.32, 0.24], font_size=12.5, cell_colors=ft_colors)
textbox(s, Inches(6.9), Inches(3.8), Inches(5.7), Inches(1.3),
        "Fine-tuning recovered most or all of the pruning-induced loss for ResNet18 and "
        "ShuffleNetV2-1.0x, at every sparsity tested — but MobileNetV3-Small does not fully "
        "recover at 70%. Tested on only 3 of 9 architectures; not generalized to the other 6.",
        size=14.5, color=MUTED, line_spacing=1.2)
footer(s, 9, "Paper Tables 4, 7 · Sections 5.4, 5.7")

# ================= SLIDE 10 — DEPLOYMENT CANDIDATE SET =================
s = new_slide()
header(s, "Deployment candidates", "A candidate set, not a single winner", title_size=30)
rounded_card(s, Inches(0.7), Inches(1.6), Inches(5.7), Inches(4.4), fill=CARD_BG)
textbox(s, Inches(1.0), Inches(1.85), Inches(5.1), Inches(0.5), "qnnpack backend-stable — 4 / 9",
        size=19, bold=True, color=DARK, font=HEAD_FONT)
multiline(s, Inches(1.0), Inches(2.5), Inches(5.1), Inches(3.3), [
    {"text": "ResNet18", "size": 20, "bullet": True, "space_after": 14, "color": BODY},
    {"text": "ShuffleNetV2-0.5x", "size": 20, "bullet": True, "space_after": 14, "color": BODY},
    {"text": "ShuffleNetV2-1.0x", "size": 20, "bullet": True, "space_after": 14, "color": BODY},
    {"text": "SqueezeNet1.1", "size": 20, "bullet": True, "space_after": 14, "color": BODY},
])
rounded_card(s, Inches(6.9), Inches(1.6), Inches(5.7), Inches(4.4), fill=DARK, line_color=None)
textbox(s, Inches(7.2), Inches(1.85), Inches(5.1), Inches(0.5), "Lightweight candidate set",
        size=19, bold=True, color=WHITE, font=HEAD_FONT)
multiline(s, Inches(7.2), Inches(2.5), Inches(5.1), Inches(2.6), [
    {"text": "ShuffleNetV2-0.5x", "size": 20, "bullet": True, "space_after": 14, "color": LIGHT_TEXT_ON_DARK},
    {"text": "ShuffleNetV2-1.0x", "size": 20, "bullet": True, "space_after": 14, "color": LIGHT_TEXT_ON_DARK},
    {"text": "SqueezeNet1.1", "size": 20, "bullet": True, "space_after": 14, "color": LIGHT_TEXT_ON_DARK},
])
textbox(s, Inches(7.2), Inches(5.05), Inches(5.1), Inches(0.4),
        "(excludes the ResNet18 baseline)", size=13, italic=True, color=RGBColor(0x85,0x93,0xA8))
textbox(s, Inches(0.7), Inches(6.25), Inches(11.9), Inches(0.6),
        "Candidate set, NOT a single winner — Slide 11 shows why one cannot be declared within it.",
        size=18, bold=True, color=DARK, align=PP_ALIGN.CENTER)
footer(s, 10, "Paper Section 7.2")

# ================= SLIDE 11 — SHUFFLENETV2 0.5x vs 1.0x =================
s = new_slide()
header(s, "ShuffleNetV2: 0.5x vs. 1.0x", "A formal Pareto non-dominance — no single winner declared", title_size=27)
sf_rows = [
    ["Accuracy (qnnpack INT8)", "73.6%", "74.4%", "1.0x"],
    ["Sensitivity", "70.7%", "88.0%", "1.0x"],
    ["Specificity", "76.5%", "60.7%", "0.5x"],
    ["Worst-case margin", "70.7%", "60.7%", "0.5x"],
    ["Dense checkpoint size", "1.95 MB", "5.45 MB", "0.5x"],
]
win_colors = {(i, 3): (GREEN if row[3] == "0.5x" else DARK) for i, row in enumerate(sf_rows)}
make_table(s, Inches(0.7), Inches(1.55), Inches(8.4), Inches(3.3),
           ["Metric", "0.5x", "1.0x", "Wins"], sf_rows, [0.42, 0.19, 0.19, 0.20],
           font_size=16, cell_colors=win_colors)

rounded_card(s, Inches(9.4), Inches(1.55), Inches(3.2), Inches(3.3), fill=CARD_BG)
multiline(s, Inches(9.6), Inches(1.75), Inches(2.85), Inches(3.0), [
    {"text": "0.5x wins:", "size": 14, "bold": True, "color": GREEN, "space_after": 4},
    {"text": "✓ Smaller", "size": 13.5, "space_after": 3},
    {"text": "✓ Higher specificity", "size": 13.5, "space_after": 3},
    {"text": "✓ Higher worst-case margin", "size": 13.5, "space_after": 12},
    {"text": "1.0x wins:", "size": 14, "bold": True, "color": DARK, "space_after": 4},
    {"text": "✓ Higher accuracy", "size": 13.5, "space_after": 3},
    {"text": "✓ Higher sensitivity", "size": 13.5, "space_after": 3},
])
textbox(s, Inches(0.7), Inches(5.15), Inches(11.9), Inches(0.55),
        "Pareto non-dominant — no single winner declared.",
        size=24, bold=True, color=DARK, font=HEAD_FONT)
textbox(s, Inches(0.7), Inches(5.85), Inches(11.9), Inches(1.0),
        "Resolving this needs an explicit sensitivity/specificity weighting for THIS task — and a "
        "skin-cancer-screening sensitivity-priority argument does not transfer, because this task is "
        "eczema vs. other diseases (not eczema vs. healthy skin).",
        size=14.5, color=MUTED, line_spacing=1.2)
footer(s, 11, "Paper Table 8 · Section 7.3")

# ================= SLIDE 12 — ROBUSTNESS CHECKS =================
s = new_slide()
header(s, "Robustness checks", "Corruption, an external dataset, and a frozen held-out test", title_size=26)
rounded_card(s, Inches(0.7), Inches(1.5), Inches(3.75), Inches(4.5), fill=CARD_BG)
textbox(s, Inches(0.95), Inches(1.7), Inches(3.25), Inches(0.4), "A. IMAGE CORRUPTION", size=14, bold=True, color=DARK)
multiline(s, Inches(0.95), Inches(2.2), Inches(3.25), Inches(3.6), [
    {"text": "Gaussian blur and JPEG compression are the most damaging of 7 corruptions tested.", "size": 14, "space_after": 12},
    {"text": "Exploratory / hypothesis-generating", "size": 14, "bold": True, "italic": True, "color": AMBER, "space_after": 12},
    {"text": "Not proof of real-world camera robustness, and not evidence that focus matters more than illumination.", "size": 13, "italic": True, "color": MUTED},
])
rounded_card(s, Inches(4.65), Inches(1.5), Inches(3.75), Inches(4.5), fill=CARD_BG)
textbox(s, Inches(4.9), Inches(1.7), Inches(3.25), Inches(0.4), "B. SKINDISNET (external)", size=14, bold=True, color=DARK)
multiline(s, Inches(4.9), Inches(2.2), Inches(3.25), Inches(3.6), [
    {"text": "Zero-shot check on a second, hospital-sourced dataset.", "size": 14, "space_after": 12},
    {"text": "Collapse signature reproduces for the already-collapsed architectures.", "size": 14, "space_after": 12},
    {"text": "NOT a general external-validity claim", "size": 14, "bold": True, "italic": True, "color": AMBER, "space_after": 12},
    {"text": "All 9 models already show substantial FP32 accuracy loss on SkinDisNet before any quantization.", "size": 13, "italic": True, "color": MUTED},
])
rounded_card(s, Inches(8.6), Inches(1.5), Inches(4.0), Inches(4.5), fill=DARK, line_color=None)
textbox(s, Inches(8.85), Inches(1.7), Inches(3.5), Inches(0.4), "C. FROZEN HELD-OUT TEST", size=14, bold=True, color=GOLD_TEXT_ON_DARK)
textbox(s, Inches(8.85), Inches(2.25), Inches(3.5), Inches(0.9), "507", size=48, bold=True, color=WHITE, font=HEAD_FONT)
textbox(s, Inches(8.85), Inches(3.0), Inches(3.5), Inches(0.4), "images, evaluated once", size=14, color=LIGHT_TEXT_ON_DARK)
textbox(s, Inches(8.85), Inches(3.55), Inches(3.5), Inches(0.4), "Protocol frozen before the split was opened.", size=13.5, color=LIGHT_TEXT_ON_DARK, line_spacing=1.15)
textbox(s, Inches(8.85), Inches(4.35), Inches(3.5), Inches(0.9), "8 / 9", size=38, bold=True, color=GREEN, font=HEAD_FONT)
textbox(s, Inches(8.85), Inches(5.1), Inches(3.5), Inches(0.6), "architectures match their validation-split status", size=13.5, color=LIGHT_TEXT_ON_DARK, line_spacing=1.1)
footer(s, 12, "Paper Sections 6, 5.5.2, 9")

# ================= SLIDE 13 — LIMITATIONS + CONCLUSION =================
s = new_slide()
header(s, "Limitations & Conclusion", "What to keep in mind before citing this work", title_size=27)
textbox(s, Inches(0.7), Inches(1.5), Inches(5.7), Inches(0.4), "LIMITATIONS", size=15, bold=True, color=DARK)
lim_lines = [
    "Backend collapse characterized on one ARM machine's qnnpack build",
    "No clinical validation — no diagnostic-accuracy claim is made",
    "No patient identifiers; image-level split, not subject-level",
    "No real target-device latency or power measurement",
    "Structured pruning excluded for ShuffleNetV2 — tooling limitation",
    "SkinDisNet check has a pre-existing FP32 distribution shift",
    "SE/swish causality unresolved — association only",
    "Corruption tests use synthetic transformations, not camera captures",
]
multiline(s, Inches(0.7), Inches(1.95), Inches(5.7), Inches(4.9),
          [{"text": t, "size": 14.5, "bullet": True, "space_after": 10} for t in lim_lines])

textbox(s, Inches(6.9), Inches(1.5), Inches(5.7), Inches(0.4), "CONCLUSION", size=15, bold=True, color=DARK)
concl_lines = [
    "INT8 behavior is strongly architecture-dependent under the tested backend recipes.",
    "5 / 9 architectures fail this study's backend-stability criterion on qnnpack.",
    "Backend default quantization differences explain only part of the effect.",
    "Deployment selection should be treated as a multi-objective problem.",
    "ShuffleNetV2-0.5x and 1.0x are Pareto non-dominant; no single winner is declared.",
]
multiline(s, Inches(6.9), Inches(1.95), Inches(5.7), Inches(4.9),
          [{"text": f"{i+1}.  {t}", "size": 14.5, "space_after": 16, "line_spacing": 1.2}
           for i, t in enumerate(concl_lines)])
footer(s, 13, "Paper Sections 8, 10")

# ================= SLIDE 14 — THANK YOU =================
s = new_slide(bg_color=DARK)
textbox(s, Inches(0.9), Inches(0.9), Inches(11.5), Inches(0.4), "THANK YOU",
        size=18, bold=True, color=GOLD_TEXT_ON_DARK)
textbox(s, Inches(0.9), Inches(2.6), Inches(11.5), Inches(1.2), "Questions?",
        size=54, bold=True, color=WHITE, font=HEAD_FONT)
multiline(s, Inches(0.9), Inches(4.6), Inches(11.0), Inches(1.8), [
    {"text": "Paper: eczema_compression_deployment_paper_2026-09-20.docx", "size": 16, "color": LIGHT_TEXT_ON_DARK, "space_after": 8},
    {"text": "Reproducibility: REPRODUCIBILITY_ENVIRONMENT_2026-09-20.md · FROZEN_TEST_PROTOCOL_2026-09-20.md", "size": 16, "color": LIGHT_TEXT_ON_DARK, "space_after": 8},
    {"text": "github.com/Lilac-dot/eczema-detection", "size": 16, "color": LIGHT_TEXT_ON_DARK, "space_after": 8},
])
footer(s, 14, "Honors Project — Flexible Electronics for Human Healthcare")

prs.save(OUT_PATH)
print(f"Saved: {OUT_PATH}")
print(f"Slides: {len(prs.slides)}")

