# -*- coding: utf-8 -*-
"""Builds the current-state honors evaluation deck, in the style of the earlier
dead_ends/multimodal_system_paper/Multi_Sensor_Wearable_AD_Monitoring_Eval_Detailed.pptx
(dense, data-forward, section-divider structure) but covering the project's CURRENT
state (2026-09-19): the per-modality architecture-selection report, the new edge-AI
compression paper, the corruption-robustness benchmark, the SkinDisNet patient-leakage
correction, the Widiawaty fusion workstream, and firmware status -- none of which existed
when the reference deck was built. All numbers below are transcribed from the project's
own docs/results (cited per-slide in the deck itself), not recomputed here.

Built with python-pptx (pptxgenjs's Node.js runtime is not installed on this machine).
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.oxml.ns import qn
from PIL import Image

from paths import ROOT

OUT_DIR = ROOT / "presentations" / "2026-09-19_honors_eval"
OUT_PATH = OUT_DIR / "honors_eval_presentation_2026-09-19.pptx"

# ---------- palette: "Midnight Executive" ----------
NAVY = RGBColor(0x1E, 0x27, 0x61)
NAVY_DARK = RGBColor(0x13, 0x18, 0x40)
ICE = RGBColor(0xCA, 0xDC, 0xFC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
OFFWHITE = RGBColor(0xF7, 0xF9, 0xFD)
INK = RGBColor(0x1F, 0x29, 0x37)
MUTED = RGBColor(0x5B, 0x66, 0x77)
GREEN = RGBColor(0x1F, 0x7A, 0x4D)
AMBER = RGBColor(0xB0, 0x6A, 0x0C)
RED = RGBColor(0xB0, 0x30, 0x4A)
CARD = RGBColor(0xEE, 0xF2, 0xFB)

BODY_FONT = "Calibri"
HEAD_FONT = "Cambria"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
BLANK = prs.slide_layouts[6]

PAGE = [0]  # mutable page counter


def new_slide(bg=WHITE):
    s = prs.slides.add_slide(BLANK)
    rect = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    rect.fill.solid()
    rect.fill.fore_color.rgb = bg
    rect.line.fill.background()
    rect.shadow.inherit = False
    # push to back
    sp = rect._element
    sp.getparent().remove(sp)
    s.shapes._spTree.insert(2, sp)
    PAGE[0] += 1
    return s


def textbox(slide, x, y, w, h, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    return tb, tf


def set_run(run, text, size=14, bold=False, italic=False, color=INK, font=BODY_FONT):
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.name = font
    run.font.color.rgb = color


def add_para(tf, text, size=14, bold=False, italic=False, color=INK, font=BODY_FONT,
             align=PP_ALIGN.LEFT, space_after=6, first=False, bullet=False, level=0):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_after = Pt(space_after)
    p.level = level
    r = p.add_run()
    prefix = "•  " if bullet else ""
    set_run(r, prefix + text, size=size, bold=bold, italic=italic, color=color, font=font)
    return p


def kicker_title(slide, kicker, title, kicker_color=NAVY, title_color=INK, x=Inches(0.6),
                  y=Inches(0.4), w=Inches(12.1)):
    _, tf = textbox(slide, x, y, w, Inches(1.15))
    add_para(tf, kicker.upper(), size=13, bold=True, color=kicker_color, font=BODY_FONT,
             space_after=2, first=True)
    add_para(tf, title, size=28, bold=True, color=title_color, font=HEAD_FONT, space_after=0)


def footer(slide, note):
    _, tf = textbox(slide, Inches(0.6), Inches(7.12), Inches(10.5), Inches(0.32))
    add_para(tf, note, size=9, italic=True, color=MUTED, space_after=0, first=True)
    _, tf2 = textbox(slide, Inches(11.9), Inches(7.12), Inches(0.9), Inches(0.32))
    add_para(tf2, f"Page {PAGE[0]:02d}", size=9, italic=True, color=MUTED,
             align=PP_ALIGN.RIGHT, space_after=0, first=True)


FOOT = "Multi-Sensor AD Monitoring — Honors Evaluation (2026-09-19)"


def card(slide, x, y, w, h, fill=CARD, line=None):
    r = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    r.adjustments[0] = 0.06
    r.fill.solid()
    r.fill.fore_color.rgb = fill
    if line:
        r.line.color.rgb = line
        r.line.width = Pt(0.75)
    else:
        r.line.fill.background()
    r.shadow.inherit = False
    return r


def stat_callout(slide, x, y, w, value, label, value_color=NAVY, value_size=40, label_size=11.5):
    _, tf = textbox(slide, x, y, w, Inches(1.15))
    add_para(tf, value, size=value_size, bold=True, color=value_color, font=HEAD_FONT,
             align=PP_ALIGN.LEFT, space_after=2, first=True)
    add_para(tf, label, size=label_size, color=MUTED, align=PP_ALIGN.LEFT, space_after=0)


def bullets_block(slide, x, y, w, h, items, size=13.5, color=INK, title=None, title_color=NAVY):
    _, tf = textbox(slide, x, y, w, h)
    first = True
    if title:
        add_para(tf, title, size=14, bold=True, color=title_color, space_after=6, first=True)
        first = False
    for it in items:
        if isinstance(it, tuple) and len(it) == 2:
            text, kwargs = it
        elif isinstance(it, tuple):
            text, kwargs = it[0], {}
        else:
            text, kwargs = it, {}
        add_para(tf, text, size=kwargs.get("size", size), bold=kwargs.get("bold", False),
                 color=kwargs.get("color", color), space_after=kwargs.get("space_after", 7),
                 bullet=kwargs.get("bullet", True), first=first)
        first = False


def add_table(slide, x, y, w, h, headers, rows, col_widths=None, font_size=10.5,
              header_fill=NAVY, header_color=WHITE, body_size=None, zebra=True):
    n_rows = len(rows) + 1
    n_cols = len(headers)
    gtable = slide.shapes.add_table(n_rows, n_cols, x, y, w, h)
    table = gtable.table
    if col_widths:
        total = sum(col_widths)
        for i, cw in enumerate(col_widths):
            table.columns[i].width = Emu(int(w * (cw / total)))
    for i, htext in enumerate(headers):
        cell = table.cell(0, i)
        cell.text = ""
        p = cell.text_frame.paragraphs[0]
        r = p.add_run()
        set_run(r, htext, size=font_size, bold=True, color=header_color, font=BODY_FONT)
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_fill
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = Pt(4); cell.margin_right = Pt(4)
        cell.margin_top = Pt(2); cell.margin_bottom = Pt(2)
    bsize = body_size or font_size
    for ri, row in enumerate(rows, start=1):
        for ci, val in enumerate(row):
            cell = table.cell(ri, ci)
            cell.text = ""
            text = str(val)
            bold = text.startswith("**") and text.endswith("**")
            if bold:
                text = text[2:-2]
            p = cell.text_frame.paragraphs[0]
            r = p.add_run()
            set_run(r, text, size=bsize, bold=bold, color=INK, font=BODY_FONT)
            cell.fill.solid()
            cell.fill.fore_color.rgb = OFFWHITE if (zebra and ri % 2 == 0) else WHITE
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = Pt(4); cell.margin_right = Pt(4)
            cell.margin_top = Pt(1); cell.margin_bottom = Pt(1)
    table.first_row = False
    table.horz_banding = False
    return table


def add_picture_fit(slide, path, x, y, max_w, max_h):
    im = Image.open(path)
    ar = im.size[0] / im.size[1]
    w, h = max_w, int(max_w / ar)
    if h > max_h:
        h = max_h
        w = int(max_h * ar)
    slide.shapes.add_picture(str(path), x, y, width=w, height=h)
    return w, h


def status_pill(slide, x, y, text, color):
    w, h = Inches(1.55), Inches(0.34)
    r = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    r.adjustments[0] = 0.5
    r.fill.solid()
    r.fill.fore_color.rgb = color
    r.line.fill.background()
    r.shadow.inherit = False
    tf = r.text_frame
    tf.margin_left = 0; tf.margin_right = 0; tf.margin_top = 0; tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r2 = p.add_run()
    set_run(r2, text, size=10.5, bold=True, color=WHITE)
    return r


def section_divider(kicker, title, subtitle):
    s = new_slide(bg=NAVY)
    grad_accent = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(10.6), Inches(-2.2), Inches(6), Inches(6))
    grad_accent.fill.solid()
    grad_accent.fill.fore_color.rgb = NAVY_DARK
    grad_accent.line.fill.background()
    grad_accent.shadow.inherit = False
    _, tf = textbox(s, Inches(0.9), Inches(2.9), Inches(10.5), Inches(2.2))
    add_para(tf, kicker.upper(), size=16, bold=True, color=ICE, space_after=10, first=True)
    add_para(tf, title, size=40, bold=True, color=WHITE, font=HEAD_FONT, space_after=10)
    add_para(tf, subtitle, size=16, italic=True, color=ICE, space_after=0)
    footer(s, FOOT)
    return s


# ================= 1. TITLE =================
s = new_slide(bg=NAVY)
circ = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(-2), Inches(3.8), Inches(7), Inches(7))
circ.fill.solid(); circ.fill.fore_color.rgb = NAVY_DARK; circ.line.fill.background()
circ.shadow.inherit = False
_, tf = textbox(s, Inches(0.9), Inches(0.6), Inches(8), Inches(0.4))
add_para(tf, "S20240020346", size=13, color=ICE, first=True, space_after=0)
_, tf = textbox(s, Inches(0.9), Inches(2.3), Inches(11), Inches(2.4))
add_para(tf, "Multi-Sensor Wearable System", size=38, bold=True, color=WHITE, font=HEAD_FONT,
         first=True, space_after=2)
add_para(tf, "for Monitoring Eczema Severity", size=38, bold=True, color=WHITE, font=HEAD_FONT,
         space_after=14)
add_para(tf, "HONORS EVALUATION — CURRENT PROGRESS (2026-09-19)", size=16, bold=True,
         color=ICE, space_after=0)
_, tf = textbox(s, Inches(0.9), Inches(6.3), Inches(10), Inches(1.0))
add_para(tf, "Tishya Yadlapalli", size=15, bold=True, color=WHITE, first=True, space_after=2)
add_para(tf, "Mentor: Dr. Priyanka Dwivedi", size=13, color=ICE, space_after=2)
add_para(tf, "github.com/Lilac-dot/eczema-detection", size=13, color=ICE, space_after=0)

# ================= 2. MOTIVATION =================
s = new_slide()
kicker_title(s, "Motivation", "From One System to a Validated, Modality-by-Modality Pipeline")
bullets_block(s, Inches(0.6), Inches(1.7), Inches(11.9), Inches(3.6), [
    "Atopic dermatitis affects 15–20% of children and 1–3% of adults worldwide; severity "
    "flares and settles over days, but clinical scoring (SCORAD, EASI) only happens episodically, "
    "in person.",
    "A prior version of this project (archived, dead_ends/multimodal_system_paper/) attempted a "
    "full joint multimodal system in one pass. That framing is superseded: the current phase "
    "validates each modality (image, stress, moisture) on its own controlled comparison before any "
    "system-level claim is made — a slower but more defensible path.",
    "Research question, current phase: for each modality this system will use, is there a "
    "systematically chosen, statistically evaluated model — not one picked by convention — and "
    "what are that model's honestly measured limits (external generalization, image-corruption "
    "robustness, compression cost)?",
], size=15.5)
card(s, Inches(0.6), Inches(5.55), Inches(11.9), Inches(1.35))
_, tf = textbox(s, Inches(0.85), Inches(5.72), Inches(11.4), Inches(1.0))
add_para(tf, "What's new since the last evaluation", size=13.5, bold=True, color=NAVY,
         first=True, space_after=4)
add_para(tf, "9-architecture image comparison → ShuffleNetV2-1.0x selected  •  edge-AI "
             "compression study across all 9  •  corruption-robustness benchmark  •  a caught "
             "and corrected data-leakage result  •  ESP32-S3 firmware, builds clean", size=12.5,
         color=MUTED, space_after=0)
footer(s, FOOT)

# ================= 3. DATASETS =================
s = new_slide()
kicker_title(s, "Data", "Datasets Used")
headers = ["Dataset", "Used for", "Size", "Link"]
rows = [
    ["WESAD", "Stress detection (Sec. 6)", "15 subjects", "archive.ics.uci.edu/dataset/465"],
    ["AAUWSS", "Sleep trigger (excluded, chance-level)", "13 subjects", "zenodo.org/records/16919071"],
    ["Curated eczema archive (3 merged Kaggle sources)", "Image channel, all 9 architectures", "3,330 images", "kaggle.com (Eczema2 + 20-Skin-Diseases + DermNet)"],
    ["SCIN (Google)", "External zero-shot validation", "976 images", "github.com/google-research-datasets/scin"],
    ["SkinDisNet", "External zero-shot validation + leakage audit", "1,710 → 1,558 cleaned", "data.mendeley.com/datasets/yj3md44hxg"],
    ["Widiawaty et al. Figshare (new, 2026-09-19)", "New image+text joint-fusion workstream", "2,811 patients", "doi.org/10.6084/m9.figshare.29925533.v4"],
    ["Southampton e-textile capacitive sensor data", "Moisture channel, exploratory only", "13 patients", "cited in moisture comparison doc"],
]
add_table(s, Inches(0.6), Inches(1.65), Inches(12.1), Inches(3.7), headers, rows,
          col_widths=[2.6, 2.6, 1.5, 3.2], font_size=10.5)
_, tf = textbox(s, Inches(0.6), Inches(5.55), Inches(12.1), Inches(1.35))
add_para(tf, "Not used (ruled out earlier)", size=13, bold=True, color=RED, first=True, space_after=4)
add_para(tf, "WISDM (phone accelerometer, 20Hz) — cannot capture the 100–800Hz signal real "
             "scratch detection needs (Chun et al. 2021). Dropped entirely.  SkinDisease 20-class "
             "Kaggle set's own “Normal” class — badly contaminated with unrelated stock photos, "
             "excluded.", size=12.5, color=MUTED, space_after=0)
footer(s, FOOT)

# ================= 4. ARCHITECTURE =================
s = new_slide()
kicker_title(s, "System Design", "Hardware Architecture v3: Two-Device Split")
# phone
phone = card(s, Inches(5.15), Inches(1.75), Inches(3.0), Inches(1.5), fill=NAVY)
tf = phone.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
r = p.add_run(); set_run(r, "PHONE", size=14, bold=True, color=WHITE)
p2 = tf.add_paragraph(); p2.alignment = PP_ALIGN.CENTER
r2 = p2.add_run(); set_run(r2, "Camera → image CNN → fusion → dashboard", size=11, color=ICE)
# wristband
wb = card(s, Inches(1.2), Inches(3.9), Inches(3.6), Inches(1.9), fill=CARD, line=NAVY)
tf = wb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
r = p.add_run(); set_run(r, "WRISTBAND", size=13, bold=True, color=NAVY)
for line in ["BMI270 (accel + gyro)", "PPG (MAX30101-class)", "ESP32-S3"]:
    pp = tf.add_paragraph(); pp.alignment = PP_ALIGN.CENTER
    rr = pp.add_run(); set_run(rr, line, size=11, color=INK)
# skin patch
sp = card(s, Inches(8.5), Inches(3.9), Inches(3.6), Inches(1.9), fill=CARD, line=NAVY)
tf = sp.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
r = p.add_run(); set_run(r, "SKIN PATCH", size=13, bold=True, color=NAVY)
for line in ["Moisture + Temperature", "EDA (exploratory)", "ESP32-S3 — firmware builds clean"]:
    pp = tf.add_paragraph(); pp.alignment = PP_ALIGN.CENTER
    rr = pp.add_run(); set_run(rr, line, size=11, color=INK)
# connector lines
for (x1, y1, x2, y2) in [(Inches(5.6), Inches(3.05), Inches(3.0), Inches(3.9)),
                          (Inches(7.6), Inches(3.05), Inches(10.3), Inches(3.9))]:
    ln = s.shapes.add_connector(1, x1, y1, x2, y2)
    ln.line.color.rgb = NAVY
    ln.line.width = Pt(2)
_, tf = textbox(s, Inches(4.4), Inches(3.3), Inches(4.5), Inches(0.5))
add_para(tf, "BLE", size=11, bold=True, color=MUTED, align=PP_ALIGN.CENTER, first=True, space_after=0)
bullets_block(s, Inches(0.6), Inches(6.1), Inches(12.1), Inches(0.9), [
    ("Deployed today: image (ShuffleNetV2-1.0x) + stress (personal-baseline LightGBM). Sleep "
     "trigger excluded (chance-level). Moisture: planned methodology only.", {"size": 12.5}),
], size=12.5)
footer(s, FOOT)

# ================= SECTION: IMAGE CHANNEL =================
section_divider("Image Channel", "Architecture Selection for On-Device Diagnosis",
                 "9 lightweight CNNs, one controlled comparison, one selected deployment architecture")

# ---- image methodology ----
s = new_slide()
kicker_title(s, "Image — Methodology", "Nine Architectures, One Fixed Protocol")
bullets_block(s, Inches(0.6), Inches(1.7), Inches(11.9), Inches(3.0), [
    "Task: Eczema vs. 7 clinically similar conditions (Psoriasis, Tinea, Candidiasis, "
    "Infestations/Bites, Lichen, Drug Eruption, Rosacea) — 3,330 images, 70/15/15 split, seed 42.",
    "8 candidates (EfficientNet-B0/Lite0, MobileNetV3-Small, MobileNetV2-1.0x, ShuffleNetV2-0.5x/"
    "1.0x, SqueezeNet1.1, RepGhostNet-0.5x) vs. the previously used ResNet18 baseline — identical "
    "2-stage transfer-learning protocol, identical head, only the backbone varies.",
    "Statistics: McNemar's test + paired bootstrap + Holm-Bonferroni correction across all "
    "C(9,2)=36 pairwise comparisons — one official test-set read, only after all 9 finished "
    "training.",
], size=15)
footer(s, FOOT)

# ---- image results table ----
s = new_slide()
kicker_title(s, "Image — Results", "Held-Out Test Performance, All 9 Architectures")
headers = ["Model", "Size (MB)", "Latency (ms)", "Test Acc", "F1", "AUC"]
rows = [
    ["ResNet18 (baseline)", "42.72", "33.05", "81.07%", "81.18%", "—"],
    ["EfficientNet-B0", "16.20", "37.62", "77.51%", "75.54%", "0.876"],
    ["EfficientNet-Lite0", "13.74", "32.99", "78.90%", "77.66%", "0.874"],
    ["MobileNetV2-1.0x", "9.34", "30.01", "78.90%", "77.94%", "0.865"],
    ["**ShuffleNetV2-1.0x (selected)**", "**5.45**", "**25.98**", "**79.68%**", "**79.68%**", "**0.867**"],
    ["RepGhostNet-0.5x", "4.85", "38.95", "76.53%", "76.34%", "0.855"],
    ["MobileNetV3-Small", "3.94", "18.09", "76.73%", "76.21%", "0.855"],
    ["SqueezeNet1.1", "3.03", "32.69", "75.54%", "75.00%", "0.838"],
    ["ShuffleNetV2-0.5x", "1.94", "21.48", "74.95%", "76.17%", "0.836"],
]
add_table(s, Inches(0.6), Inches(1.65), Inches(12.1), Inches(4.3), headers, rows,
          col_widths=[3.0, 1.3, 1.5, 1.3, 1.1, 1.1], font_size=11)
_, tf = textbox(s, Inches(0.6), Inches(6.15), Inches(12.1), Inches(0.85))
add_para(tf, "Zero of 36 pairwise comparisons significant after Holm correction (n=507). "
             "ShuffleNetV2-1.0x selected on size/speed (7.8x smaller, 21% faster than ResNet18) "
             "with the closest raw accuracy to baseline of any candidate — not a significance "
             "claim.", size=12, italic=True, color=MUTED, first=True, space_after=0)
footer(s, FOOT)

# ---- image chart ----
s = new_slide()
kicker_title(s, "Image — Selection", "Size vs. Accuracy, All 9 Candidates")
img_path = ROOT / "docs" / "architecture_comparison_size_vs_accuracy_2026-09-18.png"
add_picture_fit(s, img_path, Inches(3.3), Inches(1.55), Inches(6.8), Inches(4.9))
_, tf = textbox(s, Inches(0.6), Inches(6.55), Inches(12.1), Inches(0.55))
add_para(tf, "Source: docs/architecture_comparison_size_vs_accuracy_2026-09-18.png — "
             "papers/architecture-selection-report/honors-paper-report.docx, Section 5.6",
         size=10.5, italic=True, color=MUTED, first=True, space_after=0)
footer(s, FOOT)

# ---- external validation ----
s = new_slide()
kicker_title(s, "Image — External Validation", "Generalization Gap Is Architecture-Independent")
headers = ["Test set", "n", "Model", "Accuracy", "AUC (95% CI)"]
rows = [
    ["Internal (baseline)", "507", "ResNet18", "81.07%", "0.864 (0.831–0.897)"],
    ["SCIN (external)", "976", "ResNet18", "50.92%", "0.535 (0.500–0.569)"],
    ["SkinDisNet (external, cleaned)", "1,558", "ResNet18", "67.65%", "0.475 (0.445–0.505)"],
    ["**SkinDisNet (external, cleaned)**", "**1,558**", "**ShuffleNetV2-1.0x (selected)**", "**69.64%**", "**0.481 (0.446–0.511)**"],
]
add_table(s, Inches(0.6), Inches(1.65), Inches(12.1), Inches(2.3), headers, rows,
          col_widths=[2.6, 0.8, 2.4, 1.5, 2.4], font_size=11)
bullets_block(s, Inches(0.6), Inches(4.2), Inches(11.9), Inches(2.5), [
    "Re-tested the actually-selected architecture (ShuffleNetV2-1.0x), not just the old baseline: "
    "paired bootstrap ShuffleNetV2 vs. ResNet18 AUC on the same 1,558 images — observed diff "
    "+0.0061, 95% CI [−0.029, 0.038], p = 0.77, not significant.",
    "Both models sit at chance (AUC ≈ 0.48) on an independent clinical photo source — confirms "
    "the generalization gap is a property of the training data, not the backbone choice.",
    "Strong internal test performance tells you nothing about performance on an independent "
    "source — stated as a scope limitation before this was tested, now confirmed rather than "
    "assumed.",
], size=13.5)
footer(s, FOOT)

# ---- robustness benchmark ----
s = new_slide()
kicker_title(s, "Image — Corruption Robustness", "New (2026-09-19): Blur and JPEG Dominate Degradation")
headers = ["Corruption", "Acc @ sev1", "Acc @ sev5", "Drop"]
rows = [
    ["**Gaussian blur**", "78.23%", "**54.64%**", "**−24.6 pt**"],
    ["**JPEG compression**", "78.63%", "**54.64%**", "**−24.0 pt**"],
    ["Contrast down", "78.83%", "65.12%", "−13.7 pt"],
    ["Brightness up", "79.03%", "68.35%", "−10.7 pt"],
    ["Color shift", "77.62%", "72.38%", "−5.2 pt"],
    ["Gaussian noise", "76.81%", "75.40%", "−1.4 pt"],
    ["Brightness down", "77.82%", "75.60%", "−2.2 pt"],
]
add_table(s, Inches(0.6), Inches(1.6), Inches(6.2), Inches(4.2), headers, rows,
          col_widths=[2.4, 1.3, 1.3, 1.3], font_size=10.5)
bullets_block(s, Inches(7.1), Inches(1.6), Inches(5.6), Inches(4.6), [
    "36 conditions (7 corruption types × 5 severities + clean), validation set, ShuffleNetV2-1.0x.",
    "Blur/JPEG collapse accuracy to near-chance (~55%); lighting, noise, and color errors are far "
    "more forgiving — the opposite of the original hardware plan's equal emphasis on both.",
    "Reframes the VL53L1X distance sensor: correct capture distance (which prevents blur) may "
    "matter more than illumination control.",
    ("Caveat: synthetic corruption — not yet linked to real physical defocus blur.", {"color": MUTED, "italic": True}),
], size=13)
footer(s, FOOT)

# ---- deferral ----
s = new_slide()
kicker_title(s, "Image — Uncertainty-Aware Deferral", "Confidence Recovers Accuracy Under Corruption")
add_picture_fit(s, ROOT / "docs" / "robustness_risk_coverage_shufflenet_v2_x1_0_2026-09-19.png",
                 Inches(0.6), Inches(1.6), Inches(6.3), Inches(4.6))
headers = ["Coverage retained", "Clean acc", "Blur sev3 acc", "Blur sev5 acc"]
rows = [
    ["100% (no deferral)", "79.23%", "69.35%", "54.64%"],
    ["50%", "90.32%", "83.87%", "68.15%"],
    ["**20%**", "**98.99%**", "**92.93%**", "**81.82%**"],
]
add_table(s, Inches(7.1), Inches(1.9), Inches(5.6), Inches(1.9), headers, rows,
          col_widths=[2.0, 1.2, 1.2, 1.2], font_size=10.5)
_, tf = textbox(s, Inches(7.1), Inches(4.1), Inches(5.6), Inches(2.9))
add_para(tf, "Deferring the least-confident 80% of predictions and trusting only the top 20% "
             "recovers accuracy to 81.8% even at the worst corruption tested — no extra "
             "uncertainty machinery (no MC-dropout, no ensembling), just the model's own softmax "
             "confidence.", size=13, first=True, space_after=8)
add_para(tf, "Caveat: tested per known corruption type/severity, not yet on a mixture of unknown "
             "real-world degradations.", size=11.5, italic=True, color=MUTED, space_after=0)
footer(s, FOOT)

# ================= SECTION: STRESS CHANNEL =================
section_divider("Stress Channel", "Wearable Physiological Stress Detection",
                 "A calibration fix, an architecture comparison, and one honest negative result")

# ---- stress methodology ----
s = new_slide()
kicker_title(s, "Stress — Methodology", "LSTM vs. Classical Models on WESAD")
bullets_block(s, Inches(0.6), Inches(1.7), Inches(11.9), Inches(3.3), [
    "Deployed model: personal-baseline-calibrated LightGBM (52 hand-engineered features) — mean "
    "AUC 0.9405, up from 0.871 pooled-normalized. The calibration effect is real: p = 0.028 after "
    "correcting a leakage bug found during the significance audit (raw effect +0.069 AUC → "
    "corrected +0.054 AUC).",
    "New comparison: a CNN-downsampling-into-LSTM model trained directly on raw, bandpass-filtered "
    "BVP waveforms — the argument being that physiological signals are sequential, not a set of "
    "independent observations the way hand-engineered features treat them.",
    "Also compared: XGBoost, CatBoost, Random Forest (same 52 features). 15 WESAD subjects split "
    "by subject — 11 train/val, 4 held out, never touched until the single official test read.",
], size=14.5)
footer(s, FOOT)

# ---- stress results ----
s = new_slide()
kicker_title(s, "Stress — Results", "Held-Out Test Set (4 Subjects, Single Official Read)")
headers = ["Model", "Test Accuracy", "Test AUC (95% CI)", "Test F1"]
rows = [
    ["Random Forest", "85.6%", "0.883 [0.805, 0.950]", "0.697"],
    ["CatBoost", "82.0%", "0.876 [0.792, 0.950]", "0.603"],
    ["CNN-LSTM", "85.5%", "0.868 [0.831, 0.902]", "0.723"],
    ["XGBoost", "80.6%", "0.873 [0.801, 0.938]", "0.526"],
    ["LightGBM (uncalibrated, bonus ref.)", "84.2%", "0.802 [0.699, 0.891]", "0.645"],
]
add_table(s, Inches(0.6), Inches(1.65), Inches(12.1), Inches(2.4), headers, rows,
          col_widths=[3.4, 1.6, 2.6, 1.2], font_size=11.5)
bullets_block(s, Inches(0.6), Inches(4.3), Inches(11.9), Inches(2.2), [
    "No pairwise comparison among the 4 tabular-feature models is significant after Holm "
    "correction (best raw p = 0.0455, corrected p = 0.273).",
    "CNN-LSTM's standing as the primary direction rests on architectural fit to sequential "
    "physiological data (Sec. 6.1), not a statistical win — the comparison's job was confirming "
    "this choice isn't contradicted by the data, which it isn't.",
    "The deployed model in production remains the calibrated LightGBM (AUC 0.9405) — this "
    "comparison did not replace it.",
], size=13.5)
footer(s, FOOT)

# ---- stress calibration chart + sleep negative result ----
s = new_slide()
kicker_title(s, "Stress — Calibration, and a Negative Result", "What Worked, and What Didn't")
add_picture_fit(s, ROOT / "docs" / "wesad_calibration_delta_auc_2026-09-16.png",
                 Inches(0.6), Inches(1.6), Inches(5.7), Inches(4.3))
_, tf = textbox(s, Inches(6.7), Inches(1.6), Inches(6.0), Inches(4.9))
add_para(tf, "Personal-baseline calibration fix", size=15, bold=True, color=NAVY, first=True, space_after=6)
add_para(tf, "Normalizing each subject against their OWN resting baseline (not a pooled "
             "population statistic) fixed a severe per-subject decision-threshold spread "
             "(88x → 2.8x range) that a pooled-normalized model hid behind a strong mean AUC.",
         size=12.5, space_after=14)
add_para(tf, "Sleep trigger: excluded (dead_ends/negative_results/)", size=15, bold=True,
         color=RED, space_after=6)
add_para(tf, "Same architecture applied to AAUWSS (13 subjects, overnight PSG-scored sleep) "
             "scored at chance across every method tried — corroborated by two independent, "
             "literature-standard actigraphy formulas (Cole-Kripke, Sadeh) on the same data. "
             "Checked for a bug first (feature ablation, manual inspection); found none. "
             "Excluded from the deployed fusion pipeline on this evidence, not silently dropped.",
         size=12.5, space_after=0)
footer(s, FOOT)

# ================= SECTION: MOISTURE =================
section_divider("Moisture Channel", "Skin-Hydration Proxy — Planned, Not Yet Trained",
                 "Sample size, not model choice, is the blocking constraint")

s = new_slide()
kicker_title(s, "Moisture — Status", "Exploratory Only; Real Data Collection Is the Next Step")
bullets_block(s, Inches(0.6), Inches(1.7), Inches(11.9), Inches(3.3), [
    "No dataset collected on this project's own hardware yet — no trained model exists for this "
    "channel.",
    "Exploratory comparison run on external reference data only: Southampton e-textile capacitive "
    "sensor dataset, 13 patients, lesional/non-lesional readings vs. Corneometer/TEWL.",
    "TabICL (pretrained tabular foundation model), logistic regression, and LightGBM all tied or "
    "underperformed a plain classical baseline — with only 13 patients, model choice was not the "
    "limiting factor; sample size was (same conclusion independently reached for the stress-model "
    "comparison).",
    "Planned methodology: personal-baseline calibration (mirroring the stress-channel fix), "
    "drift/environmental compensation, then a Random Forest/XGBoost/LSTM comparison — once real "
    "data exists.",
], size=14.5)
footer(s, FOOT)

# ================= SECTION: EDGE-AI =================
section_divider("New: Edge-AI Compression Study", "Compression Robustness Is Architecture-Dependent",
                 "A standalone paper, papers/edge-ai-lightweight-deployment/")

# ---- edge methodology ----
s = new_slide()
kicker_title(s, "Edge-AI — Methodology", "Pruning + Quantization, All 9 Architectures")
bullets_block(s, Inches(0.6), Inches(1.7), Inches(11.9), Inches(3.3), [
    "Question: does compression cost (pruning, INT8 quantization) apply uniformly across "
    "architectures, or does it vary enough to matter for a deployment decision made before "
    "compression is applied?",
    "Pruning: unstructured global L1 magnitude pruning, 10 sparsity levels (0–90%), one-shot, no "
    "fine-tuning — a sensitivity sweep.",
    "Quantization: dynamic INT8 (Linear layers only) and static INT8 (full network, FX graph "
    "mode, x86/onednn backend, calibrated on 256 training images).",
    "No real Pi/Android hardware available — CPU-only proxy latency on a shared dev laptop, "
    "validation set only (test set never opened for this exploratory sweep).",
], size=14.5)
footer(s, FOOT)

# ---- edge headline finding ----
s = new_slide()
kicker_title(s, "Edge-AI — Headline Finding", "Quantization Fragility Tracks SE-Blocks / Swish, Not Family")
headers = ["Model", "SE/swish?", "FP32 Acc", "Static INT8 Acc", "Drop"]
rows = [
    ["EfficientNet-B0", "Yes", "80.65%", "56.65%", "**−24.0 pt**"],
    ["MobileNetV3-Small", "Yes", "76.01%", "50.00%", "**−26.0 pt**"],
    ["RepGhostNet-0.5x", "Yes", "76.81%", "47.98%", "**−28.8 pt**"],
    ["ResNet18 (baseline)", "No", "80.24%", "79.44%", "−0.8 pt"],
    ["**ShuffleNetV2-1.0x (selected)**", "**No**", "**79.23%**", "**78.43%**", "**−0.8 pt**"],
]
add_table(s, Inches(0.6), Inches(1.65), Inches(12.1), Inches(2.6), headers, rows,
          col_widths=[3.1, 1.4, 1.5, 1.9, 1.4], font_size=11)
chart_data = CategoryChartData()
chart_data.categories = ["EfficientNet-B0", "MobileNetV3-Small", "RepGhostNet-0.5x",
                          "ResNet18", "ShuffleNetV2-1.0x", "SqueezeNet1.1"]
chart_data.add_series("Static INT8 accuracy drop (pt)", (24.0, 26.0, 28.8, 0.8, 0.8, 2.4))
gframe = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(4.5),
                             Inches(7.6), Inches(2.7), chart_data)
chart = gframe.chart
chart.has_legend = False
plot = chart.plots[0]
plot.has_data_labels = True
plot.series[0].format.fill.solid()
plot.series[0].format.fill.fore_color.rgb = NAVY
_, tf = textbox(s, Inches(8.4), Inches(4.6), Inches(4.3), Inches(2.6))
add_para(tf, "Mean drop, SE/swish group (n=3): 26.3 pt", size=13, bold=True, color=RED,
         first=True, space_after=8)
add_para(tf, "Mean drop, plain-ReLU group (n=6): 2.8 pt", size=13, bold=True, color=GREEN,
         space_after=8)
add_para(tf, "9.3x difference, verified by scanning each loaded model's actual modules for "
             "SqueezeExcitation/SiLU/Hardswish — not assumed from architecture family name.",
         size=11.5, italic=True, color=MUTED, space_after=0)
footer(s, FOOT)

# ---- pruning sensitivity ----
s = new_slide()
kicker_title(s, "Edge-AI — Pruning Sensitivity", "The Knee Doesn't Track Model Size")
headers = ["Model", "Baseline Acc", "Knee (sparsity)", "Acc at Knee"]
rows = [
    ["ResNet18 (42.72 MB, largest)", "80.24%", "60%", "72.98%"],
    ["**ShuffleNetV2-1.0x (5.45 MB, selected)**", "**79.23%**", "**50%**", "**69.35%**"],
    ["EfficientNet-B0", "80.65%", "50%", "63.91%"],
    ["SqueezeNet1.1", "78.02%", "30%", "72.58%"],
    ["MobileNetV2-1.0x (9.35 MB)", "78.83%", "20% (earliest)", "73.59%"],
]
add_table(s, Inches(0.6), Inches(1.65), Inches(12.1), Inches(2.5), headers, rows,
          col_widths=[3.6, 1.6, 1.8, 1.6], font_size=11)
bullets_block(s, Inches(0.6), Inches(4.5), Inches(11.9), Inches(2.3), [
    "“Knee” = first sparsity level with >5pt accuracy drop from that model's own baseline, or "
    "F1 < 0.05. Ranges from 20% (MobileNetV2-1.0x, earliest) to 60% (ResNet18, most tolerant).",
    "ShuffleNetV2-1.0x (7.8x smaller than ResNet18) reaches its knee at nearly the same sparsity "
    "— pruning tolerance is an architecture property, not simply a function of parameter count.",
    "One-shot, no fine-tuning after pruning — a sensitivity sweep, not the best achievable "
    "accuracy at a given sparsity.",
], size=13.5)
footer(s, FOOT)

# ---- edge honest limitation ----
s = new_slide()
kicker_title(s, "Edge-AI — Honest Limitation", "CPU Laptop Timing Is Not a Valid Latency Proxy")
card(s, Inches(0.6), Inches(1.75), Inches(11.9), Inches(2.0), fill=CARD)
_, tf = textbox(s, Inches(0.9), Inches(1.95), Inches(11.3), Inches(1.6))
add_para(tf, "If CPU timing were just noisy-but-unbiased, quantized models would measure faster "
             "than FP32 on average, even if imprecise. That's not what happened.", size=14.5,
         first=True, space_after=8)
add_para(tf, "ResNet18: static INT8 measured FASTER (10.6ms vs. 40.4ms FP32).  EfficientNet-B0: "
             "static INT8 measured over 2x SLOWER (89.8ms vs. 35.5ms FP32). No consistent "
             "direction across the remaining 7.", size=13.5, bold=True, color=NAVY, space_after=0)
bullets_block(s, Inches(0.6), Inches(4.1), Inches(11.9), Inches(2.5), [
    "Reported plainly rather than hidden — this is the reliable finding, not a caveat to gloss "
    "over: a single dev-machine CPU benchmark cannot even predict the DIRECTION of a latency "
    "change from quantizing a given architecture.",
    "A real deployment decision needs measurement on the actual target runtime/hardware "
    "(Core ML, TFLite, ONNX Runtime Mobile) — not available in this project yet.",
    "Practical conclusion: ShuffleNetV2-1.0x's earlier selection (size/speed grounds) is "
    "reinforced, not reversed — it is also among the most compression-robust of the 9 candidates "
    "on both pruning and quantization.",
], size=13.5)
footer(s, FOOT)

# ================= SECTION: INTEGRITY =================
s = new_slide()
kicker_title(s, "Rigor Check", "A Number Was Caught and Corrected: SkinDisNet Fine-Tuning")
headers = ["Version", "AUC", "F1", "vs. patient-clean split"]
rows = [
    ["No fine-tuning (baseline)", "0.450", "0.063", "diff +0.038, **p = 0.13 (n.s.)**"],
    ["Fine-tuned, LEAKY split (old, retracted number)", "**0.613**", "0.313", "diff −0.125, **p < 0.001**"],
    ["Fine-tuned, patient-clean split (current)", "0.488", "0.244", "—"],
]
add_table(s, Inches(0.6), Inches(1.65), Inches(12.1), Inches(2.1), headers, rows,
          col_widths=[3.6, 1.2, 1.2, 3.0], font_size=11)
bullets_block(s, Inches(0.6), Inches(4.0), Inches(11.9), Inches(2.6), [
    "The earlier-reported “fine-tuning recovers generalization” result (AUC 0.6515) was measured "
    "on a split where the same patient's photos could appear in both fine-tuning data and the "
    "eval set — patient identity leakage, not a real fix.",
    "A genuine patient-clean re-split retrain shows the “recovery” is NOT statistically "
    "significant (p = 0.13) — the leaky version differs from the clean version at p < 0.001.",
    "This number is retired from this project's claims. Caught by design (a re-split audit before "
    "trusting the result), not by accident — the same test-set-isolation discipline used "
    "everywhere else in this project.",
], size=13.5)
footer(s, FOOT)

# ================= NEW WORKSTREAM =================
s = new_slide()
kicker_title(s, "New Workstream (Started 2026-09-19)", "Widiawaty et al. — Genuine Image+Text Fusion")
bullets_block(s, Inches(0.6), Inches(1.7), Inches(11.9), Inches(3.6), [
    "The wearable+image fusion (Stage C) is a genuine dead end for joint training — no dataset "
    "pairs wearable sensor data with images from the same patient. This does NOT fix that; it's a "
    "separate, real opportunity.",
    "2,811 AD/non-AD patient records with real, paired lesion photo + clinical text (chief "
    "complaint, trigger factors, Hanifin-Rajka criteria) from the same patient/visit — the first "
    "dataset in this project enabling genuinely jointly-trained (not decision-level) fusion.",
    "Status: manifest + patient-wise split done (1,915 patient groups via a clinical-text-based "
    "grouping heuristic, after a first grouping-key attempt was caught as wrong and fixed). "
    "1,968/422/419 train/val/test rows. Images not yet bulk-downloaded; 5-model comparison "
    "(text-only, image-only, decision-level fusion, jointly-trained fusion, gated fusion) planned "
    "but not run.",
    ("Not yet folded into the main report — new work outside its current scope.", {"italic": True, "color": MUTED}),
], size=14)
footer(s, FOOT)

# ================= FIRMWARE =================
s = new_slide()
kicker_title(s, "Hardware & Firmware", "ESP32-S3 Skin-Patch Firmware: Builds Clean, Not Yet Run")
status_pill(s, Inches(0.6), Inches(1.65), "BUILDS CLEAN", GREEN)
status_pill(s, Inches(2.3), Inches(1.65), "NEVER RUN", AMBER)
status_pill(s, Inches(4.0), Inches(1.65), "BLE UNTESTED", RED)
bullets_block(s, Inches(0.6), Inches(2.3), Inches(11.9), Inches(3.2), [
    "Skin-patch role: samples EDA/temperature/moisture at 4Hz, streams over BLE (NimBLE GATT) to "
    "the phone. Simulated-sensor HAL (no physical sensor yet) validates the firmware pipeline "
    "itself, not real sensor behavior.",
    "Zero warnings on a clean rebuild against ESP-IDF 5.4's NimBLE headers — real evidence of "
    "API correctness, not evidence it runs correctly.",
    "Attempted today: Espressif's QEMU emulator for ESP32-S3. Installed, but the official fork "
    "does not emulate Bluetooth/BLE hardware at all — this firmware's entire job — so QEMU can "
    "only ever verify boot/timers, never the BLE streaming itself.",
    "Most realistic next step identified: Wokwi (browser-based ESP32-S3 simulator with BLE "
    "support) or a real ESP32-S3 dev board (a few dollars) flashed and scanned from a phone.",
], size=13.5)
footer(s, FOOT)

# ================= SUMMARY TABLE =================
s = new_slide()
kicker_title(s, "Summary", "Current Status by Component")
headers = ["Component", "Dataset", "Metric", "Result", "Status"]
rows = [
    ["Stress (LightGBM, deployed)", "WESAD", "Mean AUC", "0.9405", "**Deployed**"],
    ["Stress (CNN-LSTM, comparison)", "WESAD", "Test AUC", "0.868 (n.s. vs. others)", "Documented alt."],
    ["Sleep trigger", "AAUWSS", "Mean LOSO AUC", "0.464 (chance)", "**Excluded**"],
    ["Image (9-arch. comparison)", "Curated archive", "Test Acc / F1", "79.68% / 79.68%", "**Deployed (ShuffleNetV2-1.0x)**"],
    ["Image (external, selected arch.)", "SkinDisNet", "AUC (zero-shot)", "0.481 (chance)", "Reported limitation"],
    ["Image (corruption robustness)", "Curated archive (synthetic)", "Acc @ blur sev5", "54.64%", "New finding"],
    ["Edge-AI compression (9-arch.)", "Curated archive", "Static INT8 drop (SE/swish)", "26.3 pt vs. 2.8 pt", "New standalone paper"],
    ["Moisture", "Southampton (13 pts, external)", "—", "No usable model (n too small)", "Planned only"],
    ["Widiawaty fusion", "Figshare (2,811 patients)", "—", "Split done, no model yet", "In progress"],
    ["Firmware (skin patch)", "N/A", "—", "Builds clean, unverified at runtime", "In progress"],
]
add_table(s, Inches(0.4), Inches(1.6), Inches(12.5), Inches(5.3), headers, rows,
          col_widths=[2.6, 2.2, 2.2, 2.4, 2.1], font_size=9.8)
footer(s, FOOT)

# ================= SYNTHESIS =================
s = new_slide()
kicker_title(s, "Synthesis", "Observations Across Every Component")
bullets_block(s, Inches(0.6), Inches(1.7), Inches(11.9), Inches(5.2), [
    "Statistical significance testing, applied consistently, keeps producing honest null results "
    "(image architecture choice, stress model choice) — selection then has to rest on other "
    "grounds (size, speed, architectural fit), stated explicitly rather than dressed up as an "
    "accuracy win.",
    "Two integrity catches this cycle: a premature test-set read on the image comparison "
    "(sequencing fixed with a code-enforced gate) and a patient-leakage-inflated SkinDisNet number "
    "(caught by a re-split audit before being trusted further).",
    "Compression cost is not architecture-agnostic — a model picked on accuracy/size alone could "
    "collapse to chance the moment it's quantized, depending on whether it uses SE-attention/"
    "swish components.",
    "Corruption robustness is corruption-specific, not uniform — blur/JPEG matter far more than "
    "lighting for this model, a testable, falsifiable claim for the real hardware build.",
    "Small-sample modalities (moisture: 13 patients, stress test set: 4 subjects) keep hitting the "
    "same wall — at this scale, model choice stops moving the number; sample size is the real "
    "constraint.",
], size=14)
footer(s, FOOT)

# ================= LIMITATIONS =================
s = new_slide()
kicker_title(s, "Scope", "Limitations")
bullets_block(s, Inches(0.6), Inches(1.7), Inches(11.9), Inches(5.2), [
    "Stress detection is validated as general physiological stress, not an eczema-flare "
    "predictor — no dataset links the two in the same patients.",
    "Image generalization to independent clinical photo sources is at chance (AUC ≈ 0.48–0.53), "
    "confirmed for the actually-selected architecture, not just the old baseline.",
    "All edge-AI latency numbers are a CPU-only laptop proxy, shown directly (not just assumed) to "
    "be unreliable even in direction — no real phone/Pi hardware available.",
    "No physical wearable prototype exists; the skin-patch firmware builds but has never run, on "
    "hardware or in emulation (QEMU cannot cover its core BLE function).",
    "Moisture channel and Widiawaty fusion have no trained model yet — both are planned "
    "methodology, not results.",
    "Patient-level independence between image train/val/test splits cannot be fully guaranteed — "
    "the merged curated archive provides no patient identifier.",
], size=14)
footer(s, FOOT)

# ================= NEXT STEPS =================
s = new_slide()
kicker_title(s, "Next Steps", "Priority Order")
bullets_block(s, Inches(0.6), Inches(1.7), Inches(11.9), Inches(5.2), [
    ("Priority 1: Get the skin-patch firmware genuinely running (Wokwi or a real ESP32-S3 board) "
     "— confirming BLE streaming works end-to-end before writing more firmware.", {"bold": True}),
    ("Priority 2: Bulk-download the Widiawaty images and run the planned 5-model fusion "
     "comparison, starting with the text-only baseline (fastest sanity check on whether text "
     "carries real signal).", {"bold": True}),
    ("Priority 3: Extend the corruption-robustness benchmark to the other 8 trained "
     "architectures — check whether “blur/JPEG matter most” is model-specific or general.",),
    ("Priority 4: Real sensor drivers for the wristband role and skin-patch hardware, once "
     "a physical device is built — validate against the Empatica E4 reference.",),
    ("Only once paired patient data exists (IRB-approved collection): train and validate Stage C "
     "fusion as a real predictor, not a computational proof-of-concept.", {"italic": True, "color": MUTED}),
], size=15)
footer(s, FOOT)

# ================= REFERENCES =================
s = new_slide()
kicker_title(s, "References", "Selected Citations")
refs = [
    "[1] Au, C.-Y. et al. “Development of objective measurements of scratching as a proxy of "
    "atopic dermatitis: a review.” Sensors 25(14), 2025.",
    "[2] Bawany, F. et al. “Sleep disturbances and atopic dermatitis.” J. Allergy Clin. Immunol. "
    "Pract. 9(4), 2021.",
    "[3] Chun, K. S. et al. “A skin-conformable wireless sensor to objectively quantify symptoms "
    "of pruritus.” Sci. Adv. 7(35), 2021.",
    "[4] Howard, A. et al. “Searching for MobileNetV3.” ICCV, 2019.",
    "[5] Ma, N. et al. “ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture "
    "Design.” ECCV, 2018.",
    "[6] Nagel, M. et al. “A White Paper on Neural Network Quantization.” arXiv:2106.08295, 2021.",
    "[7] Rothman, K. J., Greenland, S., & Walker, A. M. “Concepts of interaction.” Am. J. "
    "Epidemiol. 112(4), 1980.",
    "[8] Schmidt, P. et al. “Introducing WESAD, a multimodal dataset for wearable stress and "
    "affect detection.” ICMI, 2018.",
    "[9] Sultana, M. et al. “SkinDisNet: A comprehensive dataset of clinical images...” Data in "
    "Brief 63, 2025.",
    "[10] TensorFlow Blog. “Higher accuracy on vision models with EfficientNet-Lite.” 2020.",
    "[11] Ward, A. et al. “Creating an empirical dermatology dataset through crowdsourcing "
    "(SCIN).” JAMA Netw. Open 7(11), 2024.",
    "[12] Widiawaty et al. “Multimodal Machine Learning Approach for Diagnosing Atopic "
    "Dermatitis.” 2025. Dataset: doi.org/10.6084/m9.figshare.29925533.v4.",
]
_, tf = textbox(s, Inches(0.6), Inches(1.65), Inches(12.1), Inches(5.3))
first = True
for r in refs:
    add_para(tf, r, size=11.5, color=INK, space_after=8, first=first)
    first = False
footer(s, FOOT)

# ================= THANK YOU =================
s = new_slide(bg=NAVY)
_, tf = textbox(s, Inches(0), Inches(3.0), Inches(13.333), Inches(1.2))
add_para(tf, "Thank You", size=44, bold=True, color=WHITE, font=HEAD_FONT,
         align=PP_ALIGN.CENTER, first=True, space_after=8)
_, tf = textbox(s, Inches(0), Inches(4.1), Inches(13.333), Inches(0.6))
add_para(tf, "Questions?", size=18, color=ICE, align=PP_ALIGN.CENTER, first=True, space_after=8)
_, tf = textbox(s, Inches(0), Inches(4.8), Inches(13.333), Inches(0.5))
add_para(tf, "github.com/Lilac-dot/eczema-detection", size=13, color=ICE,
         align=PP_ALIGN.CENTER, first=True, space_after=0)

OUT_DIR.mkdir(parents=True, exist_ok=True)
prs.save(str(OUT_PATH))
print(f"Saved to {OUT_PATH} -- {len(prs.slides.__iter__.__self__._sldIdLst)} slides")
