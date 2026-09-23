# -*- coding: utf-8 -*-
"""Builds a dedicated eval deck for the edge-AI compression paper only (the "new idea"
from this session) -- not the full multi-topic project overview deck
(presentations/2026-09-19_honors_eval/). Same visual system (helpers copied from
build_eval_presentation_2026-09-19.py) but scoped entirely to
papers/edge-ai-lightweight-deployment/edge_ai_compression_paper_2026-09-19.docx's
content. All numbers transcribed from that paper / its source JSON, not recomputed here.

Built with python-pptx (pptxgenjs's Node.js runtime is not installed on this machine).
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE
from PIL import Image

from paths import ROOT

OUT_DIR = ROOT / "papers" / "edge-ai-lightweight-deployment"
OUT_PATH = OUT_DIR / "edge_ai_eval_presentation_2026-09-19.pptx"

# ---------- palette: "Midnight Executive" (same system as the project-overview deck) ----------
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
PAGE = [0]


def new_slide(bg=WHITE):
    s = prs.slides.add_slide(BLANK)
    rect = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    rect.fill.solid(); rect.fill.fore_color.rgb = bg
    rect.line.fill.background(); rect.shadow.inherit = False
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
    tf.margin_left = 0; tf.margin_right = 0; tf.margin_top = 0; tf.margin_bottom = 0
    return tb, tf


def set_run(run, text, size=14, bold=False, italic=False, color=INK, font=BODY_FONT):
    run.text = text
    run.font.size = Pt(size); run.font.bold = bold; run.font.italic = italic
    run.font.name = font; run.font.color.rgb = color


def add_para(tf, text, size=14, bold=False, italic=False, color=INK, font=BODY_FONT,
             align=PP_ALIGN.LEFT, space_after=6, first=False, bullet=False, level=0):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align; p.space_after = Pt(space_after); p.level = level
    r = p.add_run()
    set_run(r, ("•  " if bullet else "") + text, size=size, bold=bold, italic=italic,
            color=color, font=font)
    return p


def kicker_title(slide, kicker, title, kicker_color=NAVY, title_color=INK, x=Inches(0.6),
                  y=Inches(0.4), w=Inches(12.1)):
    _, tf = textbox(slide, x, y, w, Inches(1.15))
    add_para(tf, kicker.upper(), size=13, bold=True, color=kicker_color, space_after=2, first=True)
    add_para(tf, title, size=27, bold=True, color=title_color, font=HEAD_FONT, space_after=0)


def footer(slide, note):
    _, tf = textbox(slide, Inches(0.6), Inches(7.12), Inches(10.5), Inches(0.32))
    add_para(tf, note, size=9, italic=True, color=MUTED, space_after=0, first=True)
    _, tf2 = textbox(slide, Inches(11.9), Inches(7.12), Inches(0.9), Inches(0.32))
    add_para(tf2, f"Page {PAGE[0]:02d}", size=9, italic=True, color=MUTED,
              align=PP_ALIGN.RIGHT, space_after=0, first=True)


FOOT = "Compression Robustness Is Architecture-Dependent — Honors Evaluation Supplement (2026-09-19)"


def card(slide, x, y, w, h, fill=CARD, line=None):
    r = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    r.adjustments[0] = 0.06
    r.fill.solid(); r.fill.fore_color.rgb = fill
    if line:
        r.line.color.rgb = line; r.line.width = Pt(0.75)
    else:
        r.line.fill.background()
    r.shadow.inherit = False
    return r


def bullets_block(slide, x, y, w, h, items, size=14, color=INK, title=None, title_color=NAVY):
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
                 color=kwargs.get("color", color), space_after=kwargs.get("space_after", 8),
                 bullet=kwargs.get("bullet", True), first=first)
        first = False


def add_table(slide, x, y, w, h, headers, rows, col_widths=None, font_size=11,
              header_fill=NAVY, header_color=WHITE, zebra=True):
    n_rows = len(rows) + 1; n_cols = len(headers)
    gtable = slide.shapes.add_table(n_rows, n_cols, x, y, w, h)
    table = gtable.table
    if col_widths:
        total = sum(col_widths)
        for i, cw in enumerate(col_widths):
            table.columns[i].width = Emu(int(w * (cw / total)))
    for i, htext in enumerate(headers):
        cell = table.cell(0, i); cell.text = ""
        p = cell.text_frame.paragraphs[0]; r = p.add_run()
        set_run(r, htext, size=font_size, bold=True, color=header_color)
        cell.fill.solid(); cell.fill.fore_color.rgb = header_fill
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = Pt(4); cell.margin_right = Pt(4)
        cell.margin_top = Pt(2); cell.margin_bottom = Pt(2)
    for ri, row in enumerate(rows, start=1):
        for ci, val in enumerate(row):
            cell = table.cell(ri, ci); cell.text = ""
            text = str(val); bold = text.startswith("**") and text.endswith("**")
            if bold:
                text = text[2:-2]
            p = cell.text_frame.paragraphs[0]; r = p.add_run()
            set_run(r, text, size=font_size, bold=bold, color=INK)
            cell.fill.solid()
            cell.fill.fore_color.rgb = OFFWHITE if (zebra and ri % 2 == 0) else WHITE
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = Pt(4); cell.margin_right = Pt(4)
            cell.margin_top = Pt(1); cell.margin_bottom = Pt(1)
    table.first_row = False
    table.horz_banding = False
    return table


def stat_callout(slide, x, y, w, value, label, value_color=NAVY, value_size=42, label_size=12):
    _, tf = textbox(slide, x, y, w, Inches(1.2))
    add_para(tf, value, size=value_size, bold=True, color=value_color, font=HEAD_FONT,
             space_after=2, first=True)
    add_para(tf, label, size=label_size, color=MUTED, space_after=0)


def add_picture_fit(slide, path, x, y, max_w, max_h):
    im = Image.open(path)
    ar = im.size[0] / im.size[1]
    w, h = max_w, int(max_w / ar)
    if h > max_h:
        h = max_h; w = int(max_h * ar)
    slide.shapes.add_picture(str(path), x, y, width=w, height=h)
    return w, h


def section_divider(kicker, title, subtitle):
    s = new_slide(bg=NAVY)
    ov = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(10.6), Inches(-2.2), Inches(6), Inches(6))
    ov.fill.solid(); ov.fill.fore_color.rgb = NAVY_DARK; ov.line.fill.background()
    ov.shadow.inherit = False
    _, tf = textbox(s, Inches(0.9), Inches(2.9), Inches(10.7), Inches(2.3))
    add_para(tf, kicker.upper(), size=16, bold=True, color=ICE, space_after=10, first=True)
    add_para(tf, title, size=36, bold=True, color=WHITE, font=HEAD_FONT, space_after=10)
    add_para(tf, subtitle, size=15, italic=True, color=ICE, space_after=0)
    footer(s, FOOT)
    return s


# ================= 1. TITLE =================
s = new_slide(bg=NAVY)
circ = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(-2), Inches(3.6), Inches(7), Inches(7))
circ.fill.solid(); circ.fill.fore_color.rgb = NAVY_DARK; circ.line.fill.background()
circ.shadow.inherit = False
_, tf = textbox(s, Inches(0.9), Inches(0.6), Inches(11), Inches(0.4))
add_para(tf, "S20240020346  —  SUPPLEMENTARY EVALUATION", size=13, color=ICE, first=True, space_after=0)
_, tf = textbox(s, Inches(0.9), Inches(1.9), Inches(11.5), Inches(3.0))
add_para(tf, "Compression Robustness Is", size=34, bold=True, color=WHITE, font=HEAD_FONT,
         first=True, space_after=2)
add_para(tf, "Architecture-Dependent", size=34, bold=True, color=WHITE, font=HEAD_FONT, space_after=16)
add_para(tf, "A Pruning and Post-Training Quantization Study Across Nine", size=16.5,
         italic=True, color=ICE, space_after=2)
add_para(tf, "Lightweight CNNs for On-Device Skin-Lesion Classification", size=16.5,
         italic=True, color=ICE, space_after=0)
_, tf = textbox(s, Inches(0.9), Inches(6.3), Inches(10), Inches(1.0))
add_para(tf, "Tishya Yadlapalli", size=15, bold=True, color=WHITE, first=True, space_after=2)
add_para(tf, "Mentor: Dr. Priyanka Dwivedi", size=13, color=ICE, space_after=2)
add_para(tf, "github.com/Lilac-dot/eczema-detection", size=13, color=ICE, space_after=0)

# ================= 2. MOTIVATION / RQ =================
s = new_slide()
kicker_title(s, "Motivation", "Does Compression Cost the Same Across Architectures?")
bullets_block(s, Inches(0.6), Inches(1.7), Inches(11.9), Inches(3.1), [
    "A companion report already compared 9 CNN architectures for this project's image channel "
    "and selected ShuffleNetV2-1.0x on checkpoint size and CPU latency — accuracy alone couldn't "
    "decide it, since no candidate was statistically distinguishable from any other.",
    "That comparison stopped at architecture selection. It never tested what happens to any "
    "candidate under the compression (pruning, quantization) a real on-device deployment applies "
    "on top of whichever architecture wins.",
    "The implicit assumption in treating those as separate steps: compression cost is roughly "
    "architecture-agnostic. This study tests that assumption directly — and finds it false.",
], size=15.5)
card(s, Inches(0.6), Inches(5.2), Inches(11.9), Inches(1.6))
_, tf = textbox(s, Inches(0.85), Inches(5.38), Inches(11.4), Inches(1.3))
add_para(tf, "Research questions", size=13.5, bold=True, color=NAVY, first=True, space_after=6)
add_para(tf, "1. Is compression cost architecture-agnostic, or does it vary enough to matter for "
             "deployment decisions made before compression is applied?", size=12, space_after=3)
add_para(tf, "2. Does a known ImageNet-domain quantization-fragility finding generalize to a "
             "clinical dermatology task?   3. Is CPU-only dev-machine latency a usable proxy for "
             "edge latency?", size=12, space_after=0)
footer(s, FOOT)

# ================= 3. RELATED WORK =================
s = new_slide()
kicker_title(s, "Related Work", "A Known, Narrow Finding — Not Yet Tested Outside ImageNet")
bullets_block(s, Inches(0.6), Inches(1.7), Inches(11.9), Inches(4.6), [
    ("EfficientNet + naive post-training quantization: reported top-1 accuracy collapse from "
     "~75% to ~46% on ImageNet, caused by squeeze-excitation/swish layers producing a quantized "
     "output range too wide for a simple uniform quantizer.", {}),
    ("EfficientNet-Lite was designed specifically to fix this — removing squeeze-excitation "
     "blocks and replacing swish with ReLU6.", {}),
    ("MobileNetV3's hard-swish activation is separately documented to cause post-training "
     "quantization to fail outright, requiring quantization-aware training to recover.", {}),
    ("General finding: ReLU-family activations are more quantization-friendly than swish-family "
     "activations — simpler, piecewise-linear form is easier for a fixed-point quantizer to "
     "model.", {}),
    ("This study's contribution: testing whether this holds, at comparable magnitude, on a small "
     "(3,330-image) clinical dermatology task — and extending it to RepGhostNet, a family not "
     "covered by these sources.", {"bold": True, "color": NAVY}),
], size=14)
footer(s, FOOT)

# ================= 4. DATASET & MODELS =================
s = new_slide()
kicker_title(s, "Dataset & Models", "Same Task, Same 9 Architectures as the Source Comparison")
headers = ["Model", "Family", "SE-block / swish?", "Baseline val. acc.", "Size (MB)"]
rows = [
    ["ResNet18 (baseline)", "Residual", "No (plain ReLU)", "80.24%", "42.72"],
    ["EfficientNet-B0", "EfficientNet", "**Yes** (SE + SiLU)", "80.65%", "16.20"],
    ["EfficientNet-Lite0", "EfficientNet", "No (plain ReLU6)", "76.41%", "13.75"],
    ["MobileNetV3-Small", "MobileNet", "**Yes** (SE + Hardswish)", "76.01%", "3.95"],
    ["MobileNetV2-1.0x", "MobileNet", "No (plain ReLU6)", "78.83%", "9.35"],
    ["ShuffleNetV2-0.5x", "ShuffleNet", "No (plain ReLU)", "79.84%", "1.95"],
    ["**ShuffleNetV2-1.0x (deployed)**", "ShuffleNet", "**No** (plain ReLU)", "**79.23%**", "**5.45**"],
    ["SqueezeNet1.1", "SqueezeNet", "No (plain ReLU)", "78.02%", "3.03"],
    ["RepGhostNet-0.5x", "RepGhost", "**Yes** (SqueezeExcite)", "76.81%", "4.87"],
]
add_table(s, Inches(0.6), Inches(1.65), Inches(12.1), Inches(4.5), headers, rows,
          col_widths=[2.6, 1.4, 2.2, 1.6, 1.2], font_size=11)
_, tf = textbox(s, Inches(0.6), Inches(6.35), Inches(12.1), Inches(0.55))
add_para(tf, "3,330 curated clinical images, Eczema vs. 7 similar diseases. Validation split "
             "only (496 images) — test set never opened for this exploratory sweep.",
         size=11, italic=True, color=MUTED, first=True, space_after=0)
footer(s, FOOT)

# ================= 5. METHODOLOGY =================
s = new_slide()
kicker_title(s, "Methodology", "Pruning, Quantization, and a Verification Step")
bullets_block(s, Inches(0.6), Inches(1.7), Inches(11.9), Inches(4.9), [
    ("Pruning", {"bold": True, "color": NAVY, "size": 15, "space_after": 4}),
    ("Unstructured, global, L1 magnitude pruning across every Conv2d/Linear weight, 10 sparsity "
     "levels (0–90%), one-shot — no fine-tuning afterward. A sensitivity sweep, not a claim "
     "about the best achievable accuracy at a given sparsity.", {"size": 13}),
    ("Quantization", {"bold": True, "color": NAVY, "size": 15, "space_after": 4}),
    ("Dynamic INT8 (Linear layers only, low-effort reference point) and static INT8 (full "
     "network, PyTorch FX graph mode, x86/onednn backend, calibrated on 256 training images, "
     "never validation/test).", {"size": 13}),
    ("Architecture composition check", {"bold": True, "color": NAVY, "size": 15, "space_after": 4}),
    ("Before attributing any pattern to architecture properties, every loaded model was scanned "
     "programmatically for SE-attention/swish module types — not assumed from family name. "
     "Confirmed exactly 3 of 9 contain either component.", {"size": 13}),
], size=13)
footer(s, FOOT)

# ================= 6. PRUNING RESULTS =================
s = new_slide()
kicker_title(s, "Result 1 — Pruning", "Sensitivity Varies by Architecture, Not by Size")
headers = ["Model", "Baseline Acc.", "Knee (sparsity)", "Acc. at Knee"]
rows = [
    ["ResNet18 (42.72 MB, largest)", "80.24%", "60% (most tolerant)", "72.98%"],
    ["**ShuffleNetV2-1.0x (5.45 MB, deployed)**", "**79.23%**", "**50%**", "**69.35%**"],
    ["EfficientNet-B0", "80.65%", "50%", "63.91%"],
    ["MobileNetV3-Small", "76.01%", "40%", "60.08%"],
    ["SqueezeNet1.1", "78.02%", "30%", "72.58%"],
    ["MobileNetV2-1.0x (9.35 MB)", "78.83%", "20% (earliest)", "73.59%"],
]
add_table(s, Inches(0.6), Inches(1.65), Inches(12.1), Inches(2.9), headers, rows,
          col_widths=[3.6, 1.6, 1.8, 1.6], font_size=11)
bullets_block(s, Inches(0.6), Inches(4.85), Inches(11.9), Inches(2.0), [
    "“Knee” = first sparsity level with a >5pt accuracy drop from that model's own baseline, "
    "or F1 < 0.05. Ranges 20%–60% across the 9 candidates — no simple relationship to "
    "parameter count.",
    "ResNet18 (largest) and ShuffleNetV2-1.0x (7.8x smaller) reach their knee at nearly the same "
    "sparsity; MobileNetV2-1.0x (roughly ShuffleNetV2-1.0x's own size) degrades earliest of all "
    "nine.",
], size=13)
footer(s, FOOT)

# ================= 7. QUANTIZATION HEADLINE =================
s = new_slide()
kicker_title(s, "Result 2 — Quantization", "The Headline Finding")
headers = ["Model", "SE/swish?", "FP32 Acc", "Static INT8 Acc", "Drop"]
rows = [
    ["EfficientNet-B0", "Yes", "80.65%", "56.65%", "**−24.0 pt**"],
    ["MobileNetV3-Small", "Yes", "76.01%", "50.00%", "**−26.0 pt**"],
    ["RepGhostNet-0.5x", "Yes", "76.81%", "47.98%", "**−28.8 pt**"],
    ["ResNet18 (baseline)", "No", "80.24%", "79.44%", "−0.8 pt"],
    ["**ShuffleNetV2-1.0x (deployed)**", "**No**", "**79.23%**", "**78.43%**", "**−0.8 pt**"],
]
add_table(s, Inches(0.6), Inches(1.6), Inches(12.1), Inches(2.5), headers, rows,
          col_widths=[3.1, 1.4, 1.5, 1.9, 1.4], font_size=11)
chart_data = CategoryChartData()
chart_data.categories = ["EfficientNet-B0", "MobileNetV3-Small", "RepGhostNet-0.5x",
                          "ResNet18", "ShuffleNetV2-1.0x", "SqueezeNet1.1"]
chart_data.add_series("Static INT8 accuracy drop (pt)", (24.0, 26.0, 28.8, 0.8, 0.8, 2.4))
gframe = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.6), Inches(4.35),
                             Inches(7.5), Inches(2.75), chart_data)
chart = gframe.chart
chart.has_legend = False
plot = chart.plots[0]
plot.has_data_labels = True
plot.series[0].format.fill.solid()
plot.series[0].format.fill.fore_color.rgb = NAVY
stat_callout(s, Inches(8.4), Inches(4.5), Inches(4.3), "9.3x", "difference in mean accuracy drop",
             value_color=RED, value_size=46)
_, tf = textbox(s, Inches(8.4), Inches(5.7), Inches(4.3), Inches(1.3))
add_para(tf, "Mean drop: 26.3 pt (SE/swish, n=3) vs. 2.8 pt (plain-ReLU, n=6) — zero overlap "
             "between groups. Verified against actual model modules, not assumed from family "
             "name.", size=12, italic=True, color=MUTED, first=True, space_after=0)
footer(s, FOOT)

# ================= 8. WHY: LITERATURE MATCH =================
s = new_slide()
kicker_title(s, "Interpretation", "This Generalizes a Known Finding to a New Domain")
bullets_block(s, Inches(0.6), Inches(1.7), Inches(11.9), Inches(4.8), [
    "The EfficientNet/MobileNetV3 quantization fragility was previously documented on "
    "ImageNet-scale natural images. This study shows the same fragility, at comparable relative "
    "magnitude (~20–30pt collapse in both cases), on a 3,330-image clinical dermatology task — "
    "suggesting the mechanism is a property of the architecture + quantizer, not an artifact of "
    "ImageNet's scale or content.",
    "Extending the same test to RepGhostNet-0.5x (a family not covered in the literature found "
    "for this study) and finding the same collapse (28.8pt, the largest of the nine) supports "
    "that the SE-block mechanism itself — not something specific to EfficientNet or MobileNetV3 "
    "individually — is the operative cause.",
    ("Practical stakes: had this project selected on size alone, MobileNetV3-Small (the smallest "
     "candidate, 3.95 MB) was a defensible pick — only to collapse to exact chance accuracy "
     "(50.00%) the moment a standard quantization step was applied.", {"bold": True, "color": NAVY}),
], size=14.5)
footer(s, FOOT)

# ================= 9. HONEST LIMITATION: LATENCY =================
s = new_slide()
kicker_title(s, "Result 3 — Honest Limitation", "CPU Laptop Timing Is Not a Valid Latency Proxy")
card(s, Inches(0.6), Inches(1.75), Inches(11.9), Inches(2.1), fill=CARD)
_, tf = textbox(s, Inches(0.9), Inches(1.95), Inches(11.3), Inches(1.7))
add_para(tf, "If CPU timing were just noisy-but-unbiased, quantized models would measure faster "
             "than FP32 on average, even if imprecise. That's not what happened.", size=14.5,
         first=True, space_after=8)
add_para(tf, "ResNet18: static INT8 measured FASTER (10.6ms vs. 40.4ms FP32).  EfficientNet-B0: "
             "static INT8 measured over 2x SLOWER (89.8ms vs. 35.5ms FP32). No consistent "
             "direction across the remaining 7 architectures.", size=13.5, bold=True, color=NAVY,
         space_after=0)
bullets_block(s, Inches(0.6), Inches(4.2), Inches(11.9), Inches(2.5), [
    "Reported directly, not softened — a single dev-machine CPU benchmark cannot even predict "
    "the DIRECTION of a latency change from quantizing a given architecture.",
    "A real deployment decision needs measurement on the actual target runtime/hardware "
    "(Core ML, TFLite, ONNX Runtime Mobile) — not available in this project yet.",
    "No real Raspberry Pi or Android device was available for this study — every latency number "
    "is this CPU-only proxy, stated as such throughout.",
], size=13.5)
footer(s, FOOT)

# ================= 10. DISCUSSION: REINFORCEMENT =================
s = new_slide()
kicker_title(s, "Discussion", "Reinforcement, Not Reversal, of the Existing Choice")
bullets_block(s, Inches(0.6), Inches(1.7), Inches(11.9), Inches(4.8), [
    "ShuffleNetV2-1.0x was selected in the source comparison on checkpoint size and CPU latency, "
    "with no accuracy claim — none of the 9 candidates was statistically distinguishable on "
    "accuracy.",
    "This study adds two more reasons that selection holds up: ShuffleNetV2-1.0x has among the "
    "highest pruning knees (50% sparsity) AND among the smallest quantization costs (0.81pt) of "
    "the nine candidates.",
    "Its earlier selection didn't know about either property — they weren't tested at the time — "
    "but neither result contradicts it, and the quantization result specifically shows the "
    "selection avoided a real risk (the SE/swish collapse pattern) invisible from the original "
    "accuracy/size/speed comparison alone.",
], size=15)
footer(s, FOOT)

# ================= 11. LIMITATIONS =================
s = new_slide()
kicker_title(s, "Limitations", "Stated Plainly")
bullets_block(s, Inches(0.6), Inches(1.7), Inches(11.9), Inches(5.0), [
    "No real edge/mobile hardware — every latency number is a CPU-only proxy, shown directly to "
    "be unreliable even in direction.",
    "No fine-tuning after pruning — the knees describe one-shot pruning with no chance to "
    "recover accuracy; a real pipeline would typically fine-tune, which could push knees higher.",
    "Only one quantization backend available (x86/onednn) — the SE/swish fragility could be "
    "specific to this backend's kernel support, not a universal property of INT8 quantization on "
    "all hardware.",
    "Only unstructured pruning tested — structured (channel-level) pruning, the more likely "
    "route to a real measured speedup, needs a library (torch-pruning) not installed here.",
    "Validation-set-only evaluation (496 images), consistent with this project's test-set "
    "discipline, but not directly comparable to the 507-image test-set numbers used for the "
    "original architecture selection.",
], size=14)
footer(s, FOOT)

# ================= 12. CONCLUSION =================
s = new_slide(bg=NAVY)
_, tf = textbox(s, Inches(0.9), Inches(1.2), Inches(11.5), Inches(0.9))
add_para(tf, "CONCLUSION", size=15, bold=True, color=ICE, first=True, space_after=0)
_, tf = textbox(s, Inches(0.9), Inches(1.9), Inches(11.5), Inches(3.5))
add_para(tf, "Compression robustness is not a property that can be assumed constant across "
             "architectures, or predicted from baseline accuracy, size, or family name alone.",
         size=19, bold=True, color=WHITE, font=HEAD_FONT, first=True, space_after=18)
add_para(tf, "A 9.3x difference in mean quantization cost between architectures with and "
             "without squeeze-excitation/swish components — verified against actual model code "
             "— generalizes a previously narrow, ImageNet-domain finding to a new clinical "
             "imaging task and a wider set of architecture families than it had been tested on "
             "before.", size=15, color=ICE, space_after=18)
add_para(tf, "Practical outcome: ShuffleNetV2-1.0x, already selected on size and speed grounds, "
             "turns out to also be among the most compression-robust of the nine candidates — a "
             "property the original selection didn't know about but happened not to contradict.",
         size=15, bold=True, color=WHITE, space_after=0)
footer(s, FOOT)

# ================= 13. REFERENCES =================
s = new_slide()
kicker_title(s, "References", "Selected Citations")
refs = [
    "[1] Howard, A. et al. “Searching for MobileNetV3.” ICCV, 2019.",
    "[2] Ma, N. et al. “ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture "
    "Design.” ECCV, 2018.",
    "[3] Nagel, M. et al. “A White Paper on Neural Network Quantization.” arXiv:2106.08295, 2021.",
    "[4] TensorFlow Blog. “Higher accuracy on vision models with EfficientNet-Lite.” 2020.",
    "[5] Companion report: “Architecture Selection for On-Device Eczema Image Classification.” "
    "papers/architecture-selection-report/honors-paper-report.docx, Section 5.",
    "[6] Full paper: papers/edge-ai-lightweight-deployment/edge_ai_compression_paper_2026-09-19.docx",
    "[7] Raw results: papers/edge-ai-lightweight-deployment/"
    "edge_simulation_all_architectures_2026-09-19.json",
]
_, tf = textbox(s, Inches(0.6), Inches(1.75), Inches(12.1), Inches(4.5))
first = True
for r in refs:
    add_para(tf, r, size=13, color=INK, space_after=12, first=first)
    first = False
footer(s, FOOT)

# ================= 14. THANK YOU =================
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
print(f"Saved to {OUT_PATH}")
