# -*- coding: utf-8 -*-
"""v2 clarity pass over edge_ai_eval_presentation_FINAL.pptx (which was itself a full
rebuild matching the current paper, eczema_compression_deployment_paper_2026-09-20.docx).
This pass does NOT change any number, conclusion, or scientific claim -- it restructures
the results slides (5-13) around a "what was tested -> what happened -> what it means"
pattern, adds speaker notes to every results slide, fixes "pt" vs "%" terminology for
point differences, replaces "wins"/"hospital-sourced" with paper-scoped language, and
turns Slide 10 into an explicit elimination funnel. Every number is unchanged from v1
and re-verified against the same JSON result files. See
papers/edge-ai-lightweight-deployment/PRESENTATION_CHANGELOG_v2_2026-09-20.md for the
slide-by-slide list of what changed and why in this pass specifically.
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
OUT_PATH = PAPER_DIR / "edge_ai_eval_presentation_FINAL_v2.pptx"

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
TOTAL_SLIDES = 15

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


def speaker_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def takeaway_bar(slide, text, top=Inches(6.35), left=Inches(0.7), width=Inches(11.9),
                  height=Inches(0.62), size=20, fill=DARK, color=GOLD_TEXT_ON_DARK):
    """The one-line, large 'big takeaway' banner every results slide ends on."""
    rounded_card(slide, left, top, width, height, fill=fill, line_color=None, radius=0.18)
    textbox(slide, left + Inches(0.3), top, width - Inches(0.6), height, text,
            size=size, bold=True, color=color, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


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
header(s, "Motivation", "Why compression, and does it cost the same for every architecture?", title_size=27)
multiline(s, Inches(0.7), Inches(2.0), Inches(5.6), Inches(2.3), [
    {"text": "Wearable / edge healthcare deployment needs:", "size": 17, "bold": True, "color": DARK, "space_after": 8},
    {"text": "Small memory footprint", "size": 16, "bullet": True, "space_after": 6},
    {"text": "Low latency", "size": 16, "bullet": True, "space_after": 6},
    {"text": "Low power draw", "size": 16, "bullet": True, "space_after": 6},
    {"text": "Compression (pruning + quantization) is the mechanism that gets a trained classifier onto that hardware.", "size": 16, "space_after": 0, "color": MUTED},
])
rounded_card(s, Inches(6.6), Inches(2.0), Inches(6.0), Inches(1.5), fill=DARK, line_color=None)
textbox(s, Inches(6.9), Inches(2.2), Inches(5.4), Inches(0.4), "CORE QUESTION",
        size=14, bold=True, color=GOLD_TEXT_ON_DARK)
textbox(s, Inches(6.9), Inches(2.55), Inches(5.4), Inches(0.9),
        "Does compression affect all 9 architectures the same way?",
        size=21, bold=True, color=WHITE, font=HEAD_FONT, line_spacing=1.1)
rq_lines = [
    "1. Does compression affect architectures differently?",
    "2. Can quantization-aware training (QAT) recover PTQ losses?",
    "3. Does the quantized CPU backend change INT8 behavior?",
    "4. Which architectures remain feasible after compression?",
    "5. Do findings reproduce under corruption and held-out test evaluation?",
]
multiline(s, Inches(6.6), Inches(3.75), Inches(6.0), Inches(2.9),
          [{"text": t, "size": 16, "space_after": 12, "color": BODY} for t in rq_lines])
footer(s, 2, "Section 1 · Research questions")

# ================= SLIDE 3 — DATASET + ARCHITECTURES =================
s = new_slide()
header(s, "Dataset & Architectures", "3,330 curated images, 9 candidate architectures", title_size=25)
textbox(s, Inches(0.7), Inches(1.6), Inches(11.9), Inches(0.5),
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
make_table(s, Inches(0.7), Inches(2.2), Inches(9.0), Inches(4.15),
           ["Architecture", "SE / swish?", "FP32 Acc.", "Size (MB)"], arch_rows,
           [0.42, 0.28, 0.16, 0.14], font_size=14)
rounded_card(s, Inches(9.9), Inches(2.2), Inches(2.7), Inches(4.15), fill=DARK, line_color=None)
multiline(s, Inches(10.15), Inches(2.4), Inches(2.25), Inches(3.9), [
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

# ================= SLIDE 5 — KEY DEFINITIONS (added so the deck is self-contained) =====
s = new_slide()
header(s, "Reading the results", "Key definitions used on every following slide", title_size=28)
textbox(s, Inches(0.7), Inches(1.55), Inches(11.9), Inches(0.35),
        "The exact formulas and thresholds behind every table and number in this deck.",
        size=15, italic=True, color=MUTED)

def_cards = [
    ("Sensitivity", "TP / (TP + FN)"),
    ("Specificity", "TN / (TN + FP)"),
    ("F1", "2 · Precision · Sens. / (Precision + Sens.)"),
    ("Balanced accuracy", "(Sensitivity + Specificity) / 2"),
    ("Margin M", "min(Sensitivity, Specificity)"),
    ("ΔAcc (percentage points)", "Acc(condition A) − Acc(condition B)"),
    ("Accuracy retention R", "Acc(compressed) / Acc(FP32)"),
    ("Relative degradation", "(Acc_fbgemm − Acc_qnnpack) / Acc_fbgemm"),
    ("Fraction of drop recovered", "(Acc_after − Acc_before) / (Acc_FP32 − Acc_before)"),
]
col_w = Inches(3.83)
row_h = Inches(1.05)
gap_x = Inches(0.2)
gap_y = Inches(0.15)
for idx, (label, formula) in enumerate(def_cards):
    r, c = divmod(idx, 3)
    left = Inches(0.7) + c * (col_w + gap_x)
    top = Inches(2.0) + r * (row_h + gap_y)
    rounded_card(s, left, top, col_w, row_h, fill=CARD_BG)
    textbox(s, left + Inches(0.18), top + Inches(0.12), col_w - Inches(0.36), Inches(0.32),
            label, size=13.5, bold=True, color=DARK)
    textbox(s, left + Inches(0.18), top + Inches(0.5), col_w - Inches(0.36), Inches(0.5),
            formula, size=13.5, color=BODY, line_spacing=1.1)

rounded_card(s, Inches(0.7), Inches(5.7), Inches(11.9), Inches(0.85), fill=DARK, line_color=None)
textbox(s, Inches(0.95), Inches(5.82), Inches(11.4), Inches(0.3),
        "Status thresholds: collapsed (F1 < 0.05) · severely degraded (Sensitivity or Specificity < 20%) · else backend-stable",
        size=13.5, bold=True, color=GOLD_TEXT_ON_DARK)
textbox(s, Inches(0.95), Inches(6.13), Inches(11.4), Inches(0.35),
        "95% CIs: nonparametric bootstrap, 2,000 resamples of the validation split. No parametric "
        "significance test was run anywhere in this study unless explicitly stated.",
        size=12.5, italic=True, color=LIGHT_TEXT_ON_DARK, line_spacing=1.15)
footer(s, 5, "Paper Section 4.8")
speaker_notes(s,
    "This slide is here so the deck is self-contained -- every term used from this point on "
    "(margin, balanced accuracy, the collapsed / severely degraded / backend-stable thresholds, "
    "the bootstrap confidence intervals) is defined here exactly as the paper defines it, so I don't "
    "have to stop and define terms mid-result. One thing worth saying out loud: these status "
    "thresholds are this study's own operational choice for reading the results, not a clinical "
    "safety classification, and no parametric significance test was run anywhere in this study.")

# ================= SLIDE 6 — FINDING 1: PRUNING KNEES =================
s = new_slide()
header(s, "Finding 1", "No universally safe pruning level", title_size=32)
textbox(s, Inches(0.7), Inches(1.32), Inches(11.9), Inches(0.4),
        "Each architecture reaches its degradation threshold at a different sparsity.",
        size=16, italic=True, color=MUTED)
knee_labels = ["ResNet18", "EfficientNet-B0", "EfficientNet-\nLite0", "MobileNetV3-\nSmall",
               "MobileNetV2-\n1.0x", "ShuffleNetV2-\n0.5x", "ShuffleNetV2-\n1.0x", "SqueezeNet1.1",
               "RepGhostNet-\n0.5x"]
knee_values = [60, 50, 50, 40, 20, 40, 50, 30, 40]
chart_data = CategoryChartData()
chart_data.categories = knee_labels
chart_data.add_series("Pruning knee (%)", knee_values)
gframe = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.7), Inches(1.85),
                             Inches(8.1), Inches(4.35), chart_data)
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

rounded_card(s, Inches(9.1), Inches(1.85), Inches(3.5), Inches(4.35), fill=DARK, line_color=None)
multiline(s, Inches(9.35), Inches(2.05), Inches(3.0), Inches(4.0), [
    {"text": "20% – 60%", "size": 34, "bold": True, "color": WHITE, "space_after": 2, "font": HEAD_FONT},
    {"text": "range of knees across all 9 architectures", "size": 13, "color": LIGHT_TEXT_ON_DARK, "space_after": 16},
    {"text": "“Knee” = first sparsity level where accuracy drops more than 2 points below that architecture's own 0%-sparsity baseline.", "size": 13, "color": LIGHT_TEXT_ON_DARK, "space_after": 16},
    {"text": "MobileNetV2-1.0x: earliest knee, at 20%", "size": 15, "bold": True, "color": GOLD_TEXT_ON_DARK, "space_after": 14},
    {"text": "A knee marks where degradation begins — not an optimal deployment sparsity.", "size": 12, "italic": True, "color": RGBColor(0x85,0x93,0xA8)},
])
takeaway_bar(s, "Pruning tolerance is architecture-specific.")
footer(s, 6, "Paper Table 1 · Section 5.1")
speaker_notes(s,
    "What I tested: one-shot magnitude pruning at increasing sparsity, for all 9 architectures, "
    "each measured against its own 0%-sparsity baseline. "
    "What happened: the 'knee' -- the point where accuracy starts dropping more than 2 points -- "
    "ranges from 20% for MobileNetV2-1.0x up to 60% for ResNet18. "
    "What it means: there is no single sparsity target that is safe for every architecture; pruning "
    "tolerance has to be measured per architecture, not assumed. "
    "What this does NOT show: the knee is not the best or final deployment sparsity -- it is only "
    "where degradation begins.")

# ================= SLIDE 6 — SE/SWISH ASSOCIATION =================
s = new_slide()
header(s, "Finding 2", "SE / swish architectures are more PTQ-fragile", title_size=26)
textbox(s, Inches(0.7), Inches(1.5), Inches(11.9), Inches(0.4),
        "Architectures using squeeze-and-excitation blocks or swish-family activations lose far more "
        "accuracy to post-training quantization.", size=15, italic=True, color=MUTED)

rounded_card(s, Inches(0.7), Inches(2.05), Inches(5.55), Inches(3.0), fill=DARK, line_color=None)
textbox(s, Inches(0.95), Inches(2.22), Inches(5.05), Inches(0.35), "SE / SWISH (n=3)", size=14, bold=True, color=GOLD_TEXT_ON_DARK)
textbox(s, Inches(0.95), Inches(2.55), Inches(5.05), Inches(0.9), "27.2 pt", size=48, bold=True, color=WHITE, font=HEAD_FONT)
textbox(s, Inches(0.95), Inches(3.35), Inches(5.05), Inches(0.4), "mean FP32 → INT8 (fbgemm) accuracy drop", size=13, color=LIGHT_TEXT_ON_DARK)
textbox(s, Inches(0.95), Inches(3.8), Inches(5.05), Inches(0.4), "EfficientNet-B0 · MobileNetV3-Small · RepGhostNet-0.5x", size=12, italic=True, color=RGBColor(0x85,0x93,0xA8))

rounded_card(s, Inches(6.45), Inches(2.05), Inches(5.55), Inches(3.0), fill=CARD_BG)
textbox(s, Inches(6.7), Inches(2.22), Inches(5.05), Inches(0.35), "OTHER ARCHITECTURES (n=6)", size=14, bold=True, color=DARK)
textbox(s, Inches(6.7), Inches(2.55), Inches(5.05), Inches(0.9), "3.9 pt", size=48, bold=True, color=DARK, font=HEAD_FONT)
textbox(s, Inches(6.7), Inches(3.35), Inches(5.05), Inches(0.4), "mean FP32 → INT8 (fbgemm) accuracy drop", size=13, color=MUTED)
textbox(s, Inches(6.7), Inches(3.8), Inches(5.05), Inches(0.4), "ResNet18, EfficientNet-Lite0, MobileNetV2-1.0x, both ShuffleNetV2, SqueezeNet1.1", size=12, italic=True, color=MUTED)

textbox(s, Inches(0.7), Inches(5.3), Inches(11.9), Inches(0.5),
        "≈ 23.4 pt difference ≈ 7.0× larger", size=28, bold=True, color=DARK,
        font=HEAD_FONT, align=PP_ALIGN.CENTER)
textbox(s, Inches(0.7), Inches(5.9), Inches(11.9), Inches(0.4),
        "Association — not demonstrated causation. A component-removal ablation was confounded: "
        "removing SE / swish from trained weights collapsed FP32 accuracy on its own.",
        size=13.5, italic=True, color=MUTED, align=PP_ALIGN.CENTER)
footer(s, 7, "Paper Table 2 · Section 5.2")
speaker_notes(s,
    "What I tested: FP32-to-INT8 accuracy drop on the x86/fbgemm backend, split by whether an "
    "architecture uses squeeze-and-excitation blocks or swish-family activations. "
    "What happened: the three SE/swish architectures dropped 27.2 percentage points on average, "
    "versus 3.9 points for the other six -- about 7 times larger. "
    "What it means: this is a strong architecture-family association worth flagging before "
    "deployment. "
    "What this does NOT prove: that SE or swish itself causes the fragility. I tried removing those "
    "components directly, but that ablation is confounded -- removing them from already-trained "
    "weights collapses FP32 accuracy on its own, so it can't isolate a causal effect. I report this "
    "as a correlation, not a mechanism.")

# ================= SLIDE 7 — PRIMARY RESULT =================
s = new_slide()
header(s, "Primary finding", "INT8 behavior depends on the backend", title_size=30)
textbox(s, Inches(0.7), Inches(1.32), Inches(11.9), Inches(0.42),
        "Same trained weights + same task + INT8 quantization → different backend → very different outcomes.",
        size=15.5, italic=True, color=MUTED)
backend_rows = [
    ["ResNet18", "79.6%", "79.6%", "stable"],
    ["EfficientNet-B0", "53.0%", "48.8%", "severe"],
    ["EfficientNet-Lite0", "66.7%", "49.8%", "collapse"],
    ["MobileNetV3-Small", "50.4%", "48.2%", "severe"],
    ["MobileNetV2-1.0x", "72.0%", "49.8%", "collapse"],
    ["ShuffleNetV2-0.5x", "76.6%", "73.6%", "stable"],
    ["ShuffleNetV2-1.0x", "77.8%", "74.4%", "stable"],
    ["SqueezeNet1.1", "76.6%", "76.4%", "stable"],
    ["RepGhostNet-0.5x", "48.4%", "49.8%", "collapse"],
]
_status_map = {"stable": ("backend-stable", GREEN), "severe": ("severely degraded", AMBER),
               "collapse": ("collapsed", RED)}
cell_colors = {(i, 3): _status_map[row[3]][1] for i, row in enumerate(backend_rows)}
disp_rows = [[r[0], r[1], r[2], _status_map[r[3]][0]] for r in backend_rows]
make_table(s, Inches(0.7), Inches(1.85), Inches(7.2), Inches(4.05),
           ["Architecture", "fbgemm INT8", "qnnpack INT8", "Status"], disp_rows,
           [0.36, 0.21, 0.21, 0.22], font_size=13, cell_colors=cell_colors)

rounded_card(s, Inches(8.15), Inches(1.85), Inches(4.45), Inches(4.05), fill=DARK, line_color=None)
big_stats = [("3/9", "COLLAPSED", RED), ("2/9", "SEVERELY DEGRADED", AMBER), ("4/9", "BACKEND-STABLE", GREEN)]
xx = Inches(8.4)
for label, sub, col in big_stats:
    textbox(s, xx, Inches(2.05), Inches(1.35), Inches(0.75), label, size=34, bold=True, color=col, font=HEAD_FONT, align=PP_ALIGN.CENTER)
    textbox(s, xx, Inches(2.78), Inches(1.35), Inches(0.7), sub, size=10.5, bold=True, color=LIGHT_TEXT_ON_DARK, align=PP_ALIGN.CENTER)
    xx += Inches(1.42)
textbox(s, Inches(8.4), Inches(3.65), Inches(4.0), Inches(0.6),
        "5 / 9 = 55.6%", size=32, bold=True, color=WHITE, font=HEAD_FONT)
textbox(s, Inches(8.4), Inches(4.3), Inches(4.0), Inches(0.4),
        "fail the study's backend-stability criterion", size=13, color=LIGHT_TEXT_ON_DARK)
textbox(s, Inches(8.4), Inches(4.85), Inches(4.0), Inches(0.9),
        "MobileNetV2-1.0x on qnnpack: Sensitivity = 0%, Specificity = 100% — ≈ constant-classifier behavior, not “slightly worse.”",
        size=12, italic=True, color=GOLD_TEXT_ON_DARK, line_spacing=1.2)
textbox(s, Inches(8.4), Inches(5.75), Inches(4.0), Inches(0.5),
        "Default recipes tested on one ARM/qnnpack machine — not a universal ARM or PyTorch claim.",
        size=11, italic=True, color=RGBColor(0x85,0x93,0xA8), line_spacing=1.2)
footer(s, 8, "Paper Table 5 · Section 5.5")
speaker_notes(s,
    "This is the primary finding of the study. Same trained weights, same nominal quantization "
    "procedure -- the only thing that changes is the CPU quantized backend, x86/fbgemm versus "
    "ARM/qnnpack. What happened: 3 of 9 architectures collapsed, 2 more were severely degraded, and "
    "4 stayed backend-stable -- so 5 of 9, about 56%, fail this study's backend-stability criterion. "
    "The important point is this is not just an accuracy drop. MobileNetV2-1.0x goes from 78.8% FP32 "
    "and 72.0% on fbgemm to 49.8% on qnnpack, with 0% sensitivity and 100% specificity -- the model "
    "has effectively collapsed into predicting one class every time. "
    "What this does NOT prove: that ARM devices in general, or PyTorch quantization in general, "
    "behave this way -- this is the default recipe on one tested machine.")

# ================= SLIDE 8 — MECHANISM INVESTIGATION =================
s = new_slide()
header(s, "Why does the backend gap occur?", "Weight-quantization granularity explains part — not all — of the effect", title_size=25)
diag_rows = [
    ["EfficientNet-B0", "0.080", "0.649"],
    ["MobileNetV3-Small", "0.065", "0.165"],
    ["EfficientNet-Lite0", "0.000", "0.000"],
    ["MobileNetV2-1.0x", "0.000", "0.000"],
    ["RepGhostNet-0.5x", "0.000", "0.000"],
]
diag_colors = {(0,2): GREEN, (1,2): AMBER, (2,2): RED, (3,2): RED, (4,2): RED}
make_table(s, Inches(0.7), Inches(1.75), Inches(7.5), Inches(2.7),
           ["Architecture", "qnnpack default F1\n(per-tensor)", "Forced per-channel F1"],
           diag_rows, [0.42, 0.29, 0.29], font_size=15, cell_colors=diag_colors)
textbox(s, Inches(0.7), Inches(4.65), Inches(7.5), Inches(1.5),
        "qnnpack defaults to per-tensor weight quantization; fbgemm defaults to per-channel. Forcing "
        "per-channel quantization on qnnpack isolates that one variable: it substantially recovers "
        "EfficientNet-B0, partially recovers MobileNetV3-Small, and does not rescue the other three, "
        "which stay fully collapsed regardless.",
        size=15, color=BODY, line_spacing=1.3)
rounded_card(s, Inches(8.5), Inches(1.75), Inches(4.1), Inches(4.4), fill=DARK, line_color=None)
textbox(s, Inches(8.75), Inches(2.0), Inches(3.6), Inches(1.9),
        "Quantization recipe differences explain PART — not all — of the backend effect.",
        size=21, bold=True, color=WHITE, font=HEAD_FONT, line_spacing=1.15)
textbox(s, Inches(8.75), Inches(4.15), Inches(3.6), Inches(1.9),
        "A partial, architecture-specific contributor — not a single mechanism that accounts for "
        "every architecture in the backend table.",
        size=14, italic=True, color=LIGHT_TEXT_ON_DARK, line_spacing=1.25)
footer(s, 9, "Paper Table 5a · Section 5.5.1")
speaker_notes(s,
    "What I tested: qnnpack's default weight quantization is per-tensor, while fbgemm's default is "
    "per-channel. I forced qnnpack to use per-channel quantization instead, to isolate that one "
    "variable, for the 5 architectures affected in the previous slide. "
    "What happened: EfficientNet-B0 recovered dramatically, MobileNetV3-Small recovered partially, "
    "but EfficientNet-Lite0, MobileNetV2-1.0x, and RepGhostNet-0.5x stayed fully collapsed. "
    "What it means: quantization recipe differences are a real, partial, architecture-specific "
    "contributor to the backend gap. "
    "What this does NOT prove: that granularity is THE mechanism -- it explains 2 of 5 affected "
    "architectures, not all of them, so something else is still at play for the other three.")

# ================= SLIDE 9 — RECOVERY =================
s = new_slide()
header(s, "Can the compression damage be recovered?", "Two recovery methods, tested separately", title_size=26)
textbox(s, Inches(0.7), Inches(1.5), Inches(5.7), Inches(0.35), "QUANTIZATION-AWARE TRAINING (QAT, qnnpack)",
        size=13.5, bold=True, color=DARK)
qat_rows = [
    ["EfficientNet-B0", "48.8%", "51.4%"],
    ["MobileNetV3-Small", "48.2%", "49.8%"],
    ["RepGhostNet-0.5x", "49.8%", "55.0%"],
]
make_table(s, Inches(0.7), Inches(1.9), Inches(5.7), Inches(1.65),
           ["Architecture", "PTQ (qnnpack)", "After 3-epoch QAT"], qat_rows,
           [0.44, 0.28, 0.28], font_size=13,
           cell_colors={(0,2): AMBER, (1,2): AMBER, (2,2): AMBER})
takeaway_bar(s, "QAT DID NOT RESTORE USABLE ACCURACY", top=Inches(3.65), left=Inches(0.7),
             width=Inches(5.7), height=Inches(0.55), size=16, fill=RED)
textbox(s, Inches(0.7), Inches(4.35), Inches(5.7), Inches(0.7),
        "3-epoch QAT under the tested qnnpack setup — not a general claim that QAT does not work.",
        size=12.5, italic=True, color=MUTED, line_spacing=1.2)

textbox(s, Inches(6.9), Inches(1.5), Inches(5.7), Inches(0.35), "PRUNING + FINE-TUNING (3 of 9 architectures)",
        size=13.5, bold=True, color=DARK)
ft_rows = [
    ["ResNet18", "30/50/70%", "84.1/82.3/81.5%", "recovers fully"],
    ["ShuffleNetV2-1.0x", "30/50/70%", "80.0/79.8/75.8%", "recovers fully"],
    ["MobileNetV3-Small", "30/50/70%", "79.2/74.2/49.8%", "partial at 70%"],
]
ft_colors = {(0,3): GREEN, (1,3): GREEN, (2,3): AMBER}
make_table(s, Inches(6.9), Inches(1.9), Inches(5.7), Inches(1.65),
           ["Architecture", "Sparsity", "Acc. after fine-tune", "Result"], ft_rows,
           [0.27, 0.17, 0.31, 0.25], font_size=11.5, cell_colors=ft_colors)
takeaway_bar(s, "RECOVERED MOST/ALL PRUNING LOSS FOR 2 / 3 TESTED", top=Inches(3.65),
             left=Inches(6.9), width=Inches(5.7), height=Inches(0.55), size=16, fill=GREEN)
textbox(s, Inches(6.9), Inches(4.35), Inches(5.7), Inches(0.7),
        "MobileNetV3-Small did not fully recover at 70%; only 3 of 9 architectures were tested.",
        size=12.5, italic=True, color=MUTED, line_spacing=1.2)
textbox(s, Inches(0.7), Inches(5.35), Inches(11.9), Inches(0.5),
        "Recovery depends on the compression method: QAT did not rescue the qnnpack collapse; fine-tuning after pruning mostly did.",
        size=15, bold=True, color=DARK, align=PP_ALIGN.CENTER)
footer(s, 10, "Paper Tables 4, 7 · Sections 5.4, 5.7")
speaker_notes(s,
    "I tested two different ways to recover compression losses. "
    "First, quantization-aware training on the 3 SE/swish architectures that collapsed under "
    "qnnpack PTQ: 3 epochs of QAT did not restore usable accuracy -- every result stayed within a "
    "few points of the roughly 50% constant-classifier floor. "
    "Second, fine-tuning after pruning, on 3 different architectures at 3 sparsity levels each: "
    "ResNet18 and ShuffleNetV2-1.0x recovered most or all of the pruning-induced loss at every "
    "sparsity tested, but MobileNetV3-Small did not fully recover at 70%. "
    "What it means: recovery success depends heavily on which compression method and which "
    "architecture -- it is not a blanket 'fine-tuning fixes it' story. "
    "What this does NOT prove: that QAT never works, or that fine-tuning would work the same way "
    "for the other 6 architectures not tested here.")

# ================= SLIDE 10 — DEPLOYMENT CANDIDATES (FUNNEL) =================
s = new_slide()
header(s, "From 9 architectures to a deployment candidate set", "An elimination funnel, not a single winner", title_size=24)

def funnel_stage(top, w, label, sub=None, fill=CARD_BG, text_color=DARK, sub_color=MUTED):
    left = Inches(0.7) + (Inches(11.9) - w) / 2
    card = rounded_card(s, left, top, w, Inches(0.62), fill=fill, line_color=None if fill != CARD_BG else CARD_BORDER, radius=0.25)
    textbox(s, left, top, w, Inches(0.62), label, size=17, bold=True, color=text_color,
            align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    if sub:
        textbox(s, left, top + Inches(0.64), w, Inches(0.3), sub, size=12, italic=True,
                color=sub_color, align=PP_ALIGN.CENTER)

def funnel_arrow_down(top):
    ar = s.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(6.42), top, Inches(0.5), Inches(0.28))
    ar.fill.solid(); ar.fill.fore_color.rgb = GOLD
    ar.line.fill.background()
    ar.shadow.inherit = False

funnel_stage(Inches(1.65), Inches(6.0), "9 trained architectures", "the full set tested in this study")
funnel_arrow_down(Inches(2.33))
funnel_stage(Inches(2.66), Inches(7.6), "Apply qnnpack backend-stability criterion", "Slide 8's collapsed / severely degraded / backend-stable classification", fill=DARK, text_color=WHITE, sub_color=LIGHT_TEXT_ON_DARK)
funnel_arrow_down(Inches(3.34))
funnel_stage(Inches(3.67), Inches(9.2), "4 backend-stable: ResNet18, ShuffleNetV2-0.5x, ShuffleNetV2-1.0x, SqueezeNet1.1", fill=GREEN, text_color=WHITE)
funnel_arrow_down(Inches(4.35))
funnel_stage(Inches(4.68), Inches(6.4), "Exclude the ResNet18 baseline", "not a lightweight edge candidate", fill=CARD_BG)
funnel_arrow_down(Inches(5.36))
funnel_stage(Inches(5.69), Inches(9.2), "3 lightweight candidates: ShuffleNetV2-0.5x, ShuffleNetV2-1.0x, SqueezeNet1.1", fill=DARK, text_color=GOLD_TEXT_ON_DARK)

takeaway_bar(s, "CANDIDATE SET — NOT A SINGLE WINNER", top=Inches(6.45), height=Inches(0.55), size=17)
footer(s, 11, "Paper Section 7.2")
speaker_notes(s,
    "I am not choosing one model based on one metric -- I eliminate by criterion instead. "
    "Starting from all 9 trained architectures, I first apply the qnnpack backend-stability "
    "criterion from the primary finding: only 4 survive -- ResNet18, ShuffleNetV2-0.5x, "
    "ShuffleNetV2-1.0x, and SqueezeNet1.1. I then exclude ResNet18 as the baseline, since it isn't a "
    "lightweight edge candidate, leaving 3 lightweight candidates. "
    "What it means: this is a principled shortlist, not a ranking. "
    "What this does NOT do: declare any single architecture the deployment choice -- the next slide "
    "shows why that can't be done within this set either.")

# ================= SLIDE 11 — SHUFFLENETV2 TRADE-OFF =================
s = new_slide()
header(s, "ShuffleNetV2 — a multi-objective trade-off", "Neither size variant dominates the other", title_size=27)
sf_rows = [
    ["Accuracy (qnnpack INT8)", "73.6%", "74.4%", "1.0x"],
    ["Sensitivity", "70.7%", "88.0%", "1.0x"],
    ["Specificity", "76.5%", "60.7%", "0.5x"],
    ["Worst-case margin", "70.7%", "60.7%", "0.5x"],
    ["Dense checkpoint size", "1.95 MB", "5.45 MB", "0.5x"],
]
win_colors = {(i, 3): (GREEN if row[3] == "0.5x" else DARK) for i, row in enumerate(sf_rows)}
disp_sf = [[r[0], r[1], r[2], f"better: {r[3]}"] for r in sf_rows]
make_table(s, Inches(0.7), Inches(1.55), Inches(8.4), Inches(3.3),
           ["Metric", "0.5x", "1.0x", ""], disp_sf, [0.40, 0.19, 0.19, 0.22],
           font_size=15.5, cell_colors=win_colors)

rounded_card(s, Inches(9.4), Inches(1.55), Inches(3.2), Inches(3.3), fill=CARD_BG)
multiline(s, Inches(9.6), Inches(1.75), Inches(2.85), Inches(3.0), [
    {"text": "0.5x is better on:", "size": 13.5, "bold": True, "color": GREEN, "space_after": 4},
    {"text": "Size", "size": 13, "space_after": 3},
    {"text": "Specificity", "size": 13, "space_after": 3},
    {"text": "Worst-case margin", "size": 13, "space_after": 12},
    {"text": "1.0x is better on:", "size": 13.5, "bold": True, "color": DARK, "space_after": 4},
    {"text": "Accuracy", "size": 13, "space_after": 3},
    {"text": "Sensitivity", "size": 13, "space_after": 3},
])
takeaway_bar(s, "PARETO NON-DOMINANT", top=Inches(5.1), height=Inches(0.6), size=24)
textbox(s, Inches(0.7), Inches(5.85), Inches(11.9), Inches(0.9),
        "No single winner is mathematically justified without task-specific metric weighting. "
        "For one variant to dominate, it would need to be at least as good on every metric and "
        "better on at least one — that does not happen here.",
        size=14, color=MUTED, line_spacing=1.2, align=PP_ALIGN.CENTER)
footer(s, 12, "Paper Table 8 · Section 7.3")
speaker_notes(s,
    "This compares the two ShuffleNetV2 size variants directly, on qnnpack INT8. "
    "0.5x is better on size, specificity, and worst-case margin; 1.0x is better on accuracy and "
    "sensitivity. Formally, for one to dominate the other it would need to be at least as good on "
    "every metric and strictly better on at least one -- that does not hold here, so they are "
    "Pareto non-dominant. "
    "What it means: choosing between them requires an explicit, task-specific weighting of "
    "sensitivity versus specificity that this study does not have. "
    "What this does NOT do: declare either variant the winner -- I report a candidate set rather "
    "than force a choice.")

# ================= SLIDE 12 — ROBUSTNESS: WHAT REPRODUCES =================
s = new_slide()
header(s, "Robustness checks", "What reproduces, and what doesn't?", title_size=30)
rounded_card(s, Inches(0.7), Inches(1.6), Inches(3.75), Inches(4.4), fill=CARD_BG)
textbox(s, Inches(0.95), Inches(1.8), Inches(3.25), Inches(0.4), "A. CORRUPTION", size=14, bold=True, color=DARK)
multiline(s, Inches(0.95), Inches(2.3), Inches(3.25), Inches(3.5), [
    {"text": "Gaussian blur + JPEG", "size": 17, "bold": True, "color": DARK, "space_after": 2},
    {"text": "most damaging of 7 corruptions tested", "size": 12.5, "color": MUTED, "space_after": 14},
    {"text": "EXPLORATORY", "size": 13, "bold": True, "color": AMBER, "space_after": 8},
    {"text": "Not proof of real-world camera robustness.", "size": 12.5, "italic": True, "color": MUTED},
])
rounded_card(s, Inches(4.65), Inches(1.6), Inches(3.75), Inches(4.4), fill=CARD_BG)
textbox(s, Inches(4.9), Inches(1.8), Inches(3.25), Inches(0.4), "B. SKINDISNET", size=14, bold=True, color=DARK)
multiline(s, Inches(4.9), Inches(2.3), Inches(3.25), Inches(3.5), [
    {"text": "Zero-shot check on a second, external dermatology dataset.", "size": 14, "space_after": 12},
    {"text": "Collapse signature reproduces for the already-collapsed architectures.", "size": 13, "color": MUTED, "space_after": 14},
    {"text": "NOT GENERAL VALIDATION", "size": 13, "bold": True, "color": AMBER, "space_after": 8},
    {"text": "All 9 models already show substantial FP32 distribution shift before any quantization.", "size": 12, "italic": True, "color": MUTED},
])
rounded_card(s, Inches(8.6), Inches(1.6), Inches(4.0), Inches(4.4), fill=DARK, line_color=None)
textbox(s, Inches(8.85), Inches(1.8), Inches(3.5), Inches(0.4), "C. FROZEN HELD-OUT TEST — STRONGEST", size=13, bold=True, color=GOLD_TEXT_ON_DARK)
textbox(s, Inches(8.85), Inches(2.3), Inches(3.5), Inches(0.85), "507", size=46, bold=True, color=WHITE, font=HEAD_FONT)
textbox(s, Inches(8.85), Inches(3.05), Inches(3.5), Inches(0.4), "images, evaluated once, protocol frozen first", size=13, color=LIGHT_TEXT_ON_DARK, line_spacing=1.1)
textbox(s, Inches(8.85), Inches(3.75), Inches(3.5), Inches(0.85), "8 / 9", size=42, bold=True, color=GREEN, font=HEAD_FONT)
textbox(s, Inches(8.85), Inches(4.55), Inches(3.5), Inches(0.6), "architectures match their validation-split operational status", size=13, color=LIGHT_TEXT_ON_DARK, line_spacing=1.15)
textbox(s, Inches(8.85), Inches(5.35), Inches(3.5), Inches(0.5), "Not “all results replicated” — operational-status agreement only.", size=11, italic=True, color=RGBColor(0x85,0x93,0xA8), line_spacing=1.15)
footer(s, 13, "Paper Sections 6, 5.5.2, 9")
speaker_notes(s,
    "Three robustness checks, with different strength. "
    "The corruption benchmark and the SkinDisNet zero-shot check are both exploratory: Gaussian "
    "blur and JPEG compression were the most damaging corruptions tested, but that's not proof of "
    "real-world camera robustness; and the SkinDisNet collapse signature reproduces for the "
    "already-collapsed architectures, but all 9 models already show substantial FP32 distribution "
    "shift on that dataset, so it is not a general external-validation claim. "
    "The strongest confirmation is the held-out test set: after freezing the evaluation protocol, I "
    "evaluated the 507 test images exactly once, and 8 of 9 architectures kept the same operational "
    "status they had on validation. "
    "What this does NOT say: that every result in the study replicated -- only that the "
    "collapsed/severely-degraded/backend-stable status largely did.")

# ================= SLIDE 13 — RESULTS SYNTHESIS =================
s = new_slide()
header(s, "What did the experiments show?", "Five results, in order of what they establish", title_size=28)
synth = [
    ("1", "PRUNING IS ARCHITECTURE-DEPENDENT", "20-60% pruning-knee range across 9 architectures"),
    ("2", "INT8 IS STRONGLY BACKEND/ARCHITECTURE-DEPENDENT", "5/9 fail the qnnpack backend-stability criterion"),
    ("3", "NO SINGLE MECHANISM EXPLAINS THE GAP", "per-channel quantization rescues some models, not others"),
    ("4", "RECOVERY DEPENDS ON COMPRESSION TYPE", "QAT insufficient for the tested qnnpack collapse; fine-tuning effective for pruning in 2/3 tested"),
    ("5", "DEPLOYMENT IS MULTI-OBJECTIVE", "ShuffleNetV2-0.5x and 1.0x are Pareto non-dominant"),
]
yy = Inches(1.55)
for num, head, sub in synth:
    rounded_card(s, Inches(0.7), yy, Inches(0.55), Inches(0.78), fill=DARK, line_color=None, radius=0.3)
    textbox(s, Inches(0.7), yy, Inches(0.55), Inches(0.78), num, size=26, bold=True, color=GOLD_TEXT_ON_DARK, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    textbox(s, Inches(1.45), yy + Inches(0.02), Inches(11.1), Inches(0.4), head, size=18, bold=True, color=DARK)
    textbox(s, Inches(1.45), yy + Inches(0.42), Inches(11.1), Inches(0.35), sub, size=13.5, color=MUTED)
    yy += Inches(0.9)
takeaway_bar(s, "Compression should be evaluated jointly with architecture and deployment "
                "backend, not as a hardware-agnostic post-processing step.",
             top=Inches(6.15), height=Inches(0.72), size=15.5)
footer(s, 14, "Paper Sections 8, 10")
speaker_notes(s,
    "This slide pulls the five results together. Pruning tolerance is architecture-dependent "
    "(20-60% knee range). INT8 behavior is strongly backend- and architecture-dependent, with 5 of 9 "
    "architectures failing the qnnpack stability criterion. No single mechanism explains the "
    "backend gap -- the per-channel diagnostic only rescues some models. Recovery depends on which "
    "compression method: QAT did not rescue the qnnpack collapse, but fine-tuning after pruning "
    "worked for 2 of 3 tested architectures. And deployment selection is inherently multi-objective, "
    "since ShuffleNetV2-0.5x and 1.0x are Pareto non-dominant. "
    "The one-line takeaway: compression has to be evaluated jointly with architecture and "
    "deployment backend -- treating it as a hardware-agnostic post-processing step, as is common "
    "practice, would have missed the central finding of this study. "
    "Limitations (one ARM machine, no clinical validation, no real target-device latency, and the "
    "others already discussed) still apply to every number on this slide.")

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
footer(s, 15, "Honors Project — Flexible Electronics for Human Healthcare")

prs.save(OUT_PATH)
print(f"Saved: {OUT_PATH}")
print(f"Slides: {len(prs.slides)}")

