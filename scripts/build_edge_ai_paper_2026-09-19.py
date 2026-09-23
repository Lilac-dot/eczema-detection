# -*- coding: utf-8 -*-
"""Builds the standalone edge-AI compression paper as a .docx using python-docx,
matching the style of build_architecture_comparison_report_2026-09-18.py (same helper
functions, same visual conventions) so the two documents read as part of one project.

Source data: papers/edge-ai-lightweight-deployment/edge_simulation_all_architectures_
2026-09-19.json (all 9 architectures) and edge_simulation_shufflenet_2026-09-19.json
(single-model deep dive that motivated running the full sweep). All numbers in this
script are transcribed from those two files, verified against the printed sweep output
in the same session that produced them -- not recomputed here.
"""
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

from paths import ROOT

OUT_DIR = ROOT / "papers" / "edge-ai-lightweight-deployment"
BODY_FONT = "Calibri"
HEAD_FONT = "Calibri"

doc = Document()
section = doc.sections[0]
section.page_width = Inches(8.5)
section.page_height = Inches(11)
section.top_margin = Inches(1)
section.bottom_margin = Inches(1)
section.left_margin = Inches(1)
section.right_margin = Inches(1)

normal = doc.styles["Normal"]
normal.font.name = BODY_FONT
normal.font.size = Pt(11)
normal.paragraph_format.space_after = Pt(8)
normal.paragraph_format.line_spacing = 1.15

for lvl, size, bold in [(1, 16, True), (2, 13, True), (3, 11.5, True)]:
    st = doc.styles[f"Heading {lvl}"]
    st.font.name = HEAD_FONT
    st.font.size = Pt(size)
    st.font.bold = bold
    st.font.color.rgb = RGBColor(0x1F, 0x1F, 0x1F)
    st.paragraph_format.space_before = Pt(18 if lvl == 1 else 12)
    st.paragraph_format.space_after = Pt(6)


def para(text, style=None, bold=False, italic=False, size=None, align=None, space_after=None):
    p = doc.add_paragraph(style=style)
    r = p.add_run(text)
    r.bold = bold
    r.italic = italic
    if size:
        r.font.size = Pt(size)
    if align:
        p.alignment = align
    if space_after is not None:
        p.paragraph_format.space_after = Pt(space_after)
    return p


def heading(text, level=1):
    return doc.add_heading(text, level=level)


def bullet(text):
    doc.add_paragraph(text, style="List Bullet")


def set_cell_shading(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def make_table(headers, rows, col_widths_in, header_fill="D9E2F3"):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    hdr_cells = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].width = Inches(col_widths_in[i])
        p = hdr_cells[i].paragraphs[0]
        p.text = ""
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(9.5)
        set_cell_shading(hdr_cells[i], header_fill)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].width = Inches(col_widths_in[i])
            p = cells[i].paragraphs[0]
            p.text = ""
            r = p.add_run(str(val))
            r.font.size = Pt(9.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


def caption(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(9)
    p.paragraph_format.space_after = Pt(14)


# ================= TITLE PAGE =================
para("", space_after=40)
para("Compression Robustness Is Architecture-Dependent", bold=True, size=22,
     align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)
para("A Pruning and Post-Training Quantization Study Across Nine Lightweight CNNs "
     "for On-Device Skin-Lesion Classification", italic=True, size=14,
     align=WD_ALIGN_PARAGRAPH.CENTER, space_after=30)
para("Tishya", size=12, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
para("September 19, 2026", size=11, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=40)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run(
    "Status: complete simulation study. All 9 architectures from the prior architecture-"
    "selection comparison (papers/architecture-selection-report/honors-paper-report.docx, "
    "Section 5) were pruned (10 sparsity levels, one-shot, no fine-tuning) and "
    "post-training quantized (dynamic and static INT8) on the same held-out validation "
    "set. Headline finding: static quantization collapses accuracy near or below chance "
    "specifically in the 3 architectures using squeeze-and-excitation blocks and/or "
    "swish-family activations, while the 6 plain-ReLU architectures lose under 3 points "
    "on average -- a 9.3x difference (Section 5.2). No real edge/mobile hardware was "
    "available; all latency numbers are a CPU-only proxy on a shared development "
    "laptop, and are shown in Section 5.3 to be actively misleading, not just imprecise."
)
r.italic = True
r.font.size = Pt(10)
r.font.color.rgb = RGBColor(0x60, 0x60, 0x60)
doc.add_page_break()

# ================= ABSTRACT =================
heading("Abstract", level=1)
para(
    "Selecting a neural network architecture for on-device deployment is usually followed "
    "by a compression step -- pruning, quantization, or both -- to fit real storage and "
    "latency budgets. This step is often treated as architecture-agnostic: pick a "
    "compression recipe, apply it, report the size/speed win. This paper tests that "
    "assumption directly by applying an identical pruning sweep (unstructured global L1 "
    "magnitude pruning, 0-90% sparsity, one-shot) and an identical post-training "
    "quantization procedure (dynamic and static INT8) to all nine convolutional "
    "architectures already compared for a binary eczema-vs-similar-disease image "
    "classification task in a prior report, without changing any architecture-specific "
    "training. The result contradicts architecture-agnostic treatment of compression: "
    "post-training static quantization drops accuracy by a mean of 26.3 percentage points "
    "in the three architectures containing squeeze-and-excitation attention blocks and/or "
    "swish-family activations (EfficientNet-B0, MobileNetV3-Small, RepGhostNet-0.5x) -- two "
    "of the three falling to at or below chance-level accuracy -- versus a mean drop of "
    "only 2.8 points in the six architectures built from plain ReLU/ReLU6 convolutions "
    "(ResNet18, EfficientNet-Lite0, MobileNetV2-1.0x, both ShuffleNetV2 variants, "
    "SqueezeNet1.1), a 9.3x difference. This reproduces, on a new clinical imaging task "
    "outside the ImageNet domain the effect was previously documented on, a known but "
    "narrowly-reported failure mode of naive post-training quantization on "
    "attention/swish-based mobile architectures. Pruning tolerance is separately "
    "architecture-dependent and does not track baseline model size: the accuracy-vs-"
    "sparsity knee ranges from 20% (MobileNetV2-1.0x) to 60% (ResNet18) across the nine "
    "candidates, with no simple relationship to parameter count. A third, purely "
    "methodological finding is reported alongside the two substantive ones: CPU-only "
    "latency measured on this project's shared development laptop is not merely imprecise "
    "but actively misleading as an edge-latency proxy -- quantized models measured faster "
    "than FP32 for some architectures and over twice as slow for others, with no "
    "consistent direction, undermining any latency claim not measured on representative "
    "target hardware. No real Raspberry Pi or Android device was available for this study; "
    "every latency number is a CPU-only simulation, stated as such throughout. All "
    "experiments used the validation split only, on the code-enforced test-set-isolation "
    "discipline already established in the source project (the test set was never "
    "opened). The practical conclusion is a reinforcement, not a reversal, of the prior "
    "report's deployment choice: ShuffleNetV2-1.0x, already selected on size/speed "
    "grounds, is also among the most compression-robust candidates on both axes tested "
    "here (a 0.8-point quantization drop and a pruning knee at 50% sparsity), making its "
    "earlier selection more, not less, defensible in light of this study."
)

# ================= 1. INTRODUCTION =================
heading("1. Introduction", level=1)
para(
    "A companion report (\"Architecture Selection for On-Device Eczema Image "
    "Classification\") compared nine convolutional architectures -- a ResNet18 baseline "
    "plus eight lightweight candidates spanning the MobileNet, EfficientNet, ShuffleNet, "
    "SqueezeNet, and RepGhost families -- for a binary eczema-vs-seven-similar-diseases "
    "image classification task, and selected ShuffleNetV2-1.0x for deployment on "
    "checkpoint size and CPU inference latency, since no candidate's accuracy was "
    "statistically distinguishable from any other after correction for multiple "
    "comparisons. That comparison stopped at architecture selection -- it did not test "
    "what happens to any of the nine candidates under the compression techniques "
    "(pruning, quantization) that a real on-device deployment would typically apply on "
    "top of whichever architecture is chosen. This paper picks up exactly there."
)
para(
    "The implicit assumption behind treating architecture selection and compression as "
    "separate, sequential decisions is that compression cost is roughly architecture-"
    "agnostic -- that whichever candidate wins the accuracy/size/speed comparison can then "
    "be pruned or quantized with a broadly similar, predictable cost. This paper tests "
    "that assumption by running the same compression procedures across all nine "
    "candidates rather than only the one already selected, and finds it does not hold: "
    "compression robustness varies so sharply by architecture family that it would have "
    "been possible to select a candidate on accuracy/size/speed grounds that then "
    "collapsed to near-chance accuracy the moment a standard compression step was "
    "applied."
)
heading("1.1 Contribution", level=2)
bullet("A systematic pruning-sensitivity sweep (10 sparsity levels, one-shot magnitude "
       "pruning, no fine-tuning) across all nine architectures from the source "
       "comparison, showing the accuracy-vs-sparsity knee varies from 20% to 60% "
       "sparsity with no simple relationship to baseline model size.")
bullet("A systematic post-training quantization comparison (dynamic and static INT8) "
       "across the same nine architectures, identifying a 9.3x difference in mean "
       "accuracy cost between architectures with squeeze-and-excitation/swish "
       "components and those without -- verified directly against each model's actual "
       "module composition (Section 4.3), not inferred from architecture family names.")
bullet("A demonstration, not just a caveat, that CPU-only development-machine latency is "
       "an unreliable edge-deployment proxy -- shown concretely to reverse direction "
       "(faster for some architectures, over 2x slower for others under identical "
       "quantization) rather than simply being imprecise.")
heading("1.2 Research questions", level=2)
bullet("Is compression cost (pruning, quantization) architecture-agnostic, or does it "
       "vary enough across architectures to matter for deployment decisions made before "
       "compression is applied?")
bullet("Does a known ImageNet-domain finding about quantization fragility in "
       "attention/swish-based architectures generalize to a clinical dermatology imaging "
       "task in a different domain?")
bullet("Is CPU-only development-machine latency a usable, if imprecise, stand-in for "
       "edge/mobile latency, or does it fail more fundamentally than that?")

# ================= 2. RELATED WORK =================
heading("2. Related Work", level=1)
heading("2.1 Post-training quantization fragility in efficient CNN architectures", level=2)
para(
    "Two specific, previously documented findings motivate the architecture grouping "
    "used in this paper. First, applying naive post-training quantization to the "
    "original EfficientNet family causes severe accuracy collapse (reported informally as "
    "a drop from roughly 75% to 46% top-1 accuracy on ImageNet), attributed to the "
    "quantized output range of its squeeze-and-excitation and swish-activation layers "
    "being too wide for a simple uniform quantizer; the EfficientNet-Lite variant was "
    "designed specifically to fix this, by removing squeeze-and-excitation blocks and "
    "replacing swish activations with ReLU6 [TensorFlow Blog, \"Higher accuracy on vision "
    "models with EfficientNet-Lite\"]. Second, MobileNetV3's hard-swish activation is "
    "separately documented to cause post-training quantization to fail, with "
    "quantization-aware training required to recover accuracy where post-training "
    "quantization alone does not [general PTQ/QAT literature on hard-swish, surfaced via "
    "literature search 2026-09-19]. More broadly, ReLU-family activations are considered "
    "more quantization-friendly than swish-family activations because their simple, "
    "piecewise-linear form is easier for a fixed-point quantizer's requantization step to "
    "model, whereas activation quantization causes disproportionately larger degradation "
    "for swish-based networks such as EfficientNet."
)
para(
    "This paper does not claim to discover this mechanism -- it is documented, "
    "narrowly, for EfficientNet and MobileNetV3 specifically, usually on ImageNet-scale "
    "natural-image classification. This paper's contribution on this point is empirical "
    "generalization: testing whether the same fragility appears, at comparable magnitude, "
    "on a clinical dermatology image classification task outside the domain and dataset "
    "scale where it was previously reported, and extending the comparison to two "
    "architecture families (ShuffleNetV2, RepGhostNet) not covered by the sources above."
)
heading("2.2 Relationship to the source architecture comparison", level=2)
para(
    "This paper reuses the nine architectures, the training protocol, the dataset "
    "(3,330 curated clinical photographs, Eczema vs. seven clinically similar "
    "conditions), and the validation/test split exactly as established in the companion "
    "report (its Sections 3-4), changing only what happens to each already-trained "
    "checkpoint after training: pruning and quantization are applied post-hoc to the "
    "existing best-validation checkpoints, with no retraining of the base models "
    "themselves."
)

# ================= 3. DATASET AND MODELS =================
heading("3. Dataset and Models", level=1)
para(
    "Task and dataset are unchanged from the source comparison: binary classification, "
    "Eczema vs. seven clinically similar conditions (Psoriasis, Tinea, Candidiasis, "
    "Infestations/Bites, Lichen, Drug Eruption, Rosacea), 3,330 curated images, "
    "70/15/15 train/val/test split (2,327/496/507), fixed seed 42, image-level (no "
    "patient identifier exists in the three merged source archives). This paper uses "
    "only the validation split (496 images) -- the test split was never opened for any "
    "experiment reported here, for the reason given in Section 4.4."
)
make_table(
    ["Model", "Family", "SE-block / swish?", "Baseline val. accuracy", "Baseline size (MB)"],
    [
        ["ResNet18 (baseline)", "Residual", "No (plain ReLU)", "80.24%", "42.72"],
        ["EfficientNet-B0", "EfficientNet", "Yes (SE + SiLU)", "80.65%", "16.20"],
        ["EfficientNet-Lite0", "EfficientNet", "No (plain ReLU6)", "76.41%", "13.75"],
        ["MobileNetV3-Small", "MobileNet", "Yes (SE + Hardswish)", "76.01%", "3.95"],
        ["MobileNetV2-1.0x", "MobileNet", "No (plain ReLU6)", "78.83%", "9.35"],
        ["ShuffleNetV2-0.5x", "ShuffleNet", "No (plain ReLU)", "79.84%", "1.95"],
        ["ShuffleNetV2-1.0x", "ShuffleNet", "No (plain ReLU)", "79.23%", "5.45"],
        ["SqueezeNet1.1", "SqueezeNet", "No (plain ReLU)", "78.02%", "3.03"],
        ["RepGhostNet-0.5x", "RepGhost", "Yes (SqueezeExcite)", "76.81%", "4.87"],
    ],
    [1.7, 1.1, 1.7, 1.4, 1.1],
)
caption(
    "Table 1. The nine architectures and their SE-block/swish status, verified directly "
    "by scanning each loaded model's modules for SqueezeExcitation/SqueezeExcite, SiLU, "
    "and Hardswish module types (Section 4.3) -- not inferred from architecture family "
    "name alone. Baseline accuracy figures are this paper's own validation-set numbers "
    "(reloaded checkpoints, re-evaluated), not copied from the source report's separate "
    "test-set numbers -- the two are not directly comparable and are not conflated here."
)

# ================= 4. METHODOLOGY =================
heading("4. Methodology", level=1)
heading("4.1 Pruning", level=2)
para(
    "Unstructured, global, L1 magnitude pruning across every Conv2d and Linear weight "
    "tensor in a model simultaneously (not per-layer), at ten target sparsity levels "
    "(0%, 10%, ..., 90%), one-shot -- weights are masked once and evaluated immediately, "
    "with no fine-tuning afterward. This is a sensitivity sweep (how much accuracy does "
    "removing weights cost, with no opportunity to recover it), not a claim about the "
    "best achievable accuracy at a given sparsity; fine-tuning after pruning typically "
    "recovers some accuracy and was not attempted in this study (Section 6, Future Work)."
)
heading("4.2 Quantization", level=2)
para(
    "Two post-training INT8 quantization procedures, both applied to the FP32 checkpoint "
    "with no retraining: (1) dynamic quantization, which quantizes only Linear-layer "
    "weights at inference time and leaves all convolutional layers in FP32 -- included as "
    "a low-effort reference point, not expected to meaningfully compress a convolution-"
    "heavy backbone; (2) static quantization via PyTorch's FX graph-mode quantizer "
    "(prepare_fx/convert_fx), which quantizes the full network (weights and activations) "
    "using the x86/onednn backend (the only backend torch.backends.quantized."
    "supported_engines reports on this project's torch build), calibrated on 8 batches "
    "(256 images) drawn from the training split only -- never validation or test."
)
heading("4.3 Architecture composition check", level=2)
para(
    "Before attributing any quantization-fragility pattern to architecture properties, "
    "each of the nine loaded models was scanned programmatically for module types "
    "associated with squeeze-and-excitation attention (SqueezeExcitation, SqueezeExcite, "
    "SEModule) and swish-family activations (SiLU, Hardswish), rather than relying on "
    "which architecture family typically includes these components. This confirmed "
    "exactly three of the nine models contain either component -- EfficientNet-B0 (SiLU "
    "+ SqueezeExcitation), MobileNetV3-Small (Hardswish + SqueezeExcitation), and "
    "RepGhostNet-0.5x (SqueezeExcite) -- and the remaining six (ResNet18, "
    "EfficientNet-Lite0, MobileNetV2-1.0x, both ShuffleNetV2 variants, SqueezeNet1.1) "
    "contain neither, confirming the grouping used in Section 5.2 is a property of the "
    "actual loaded models, not an assumption from architecture naming."
)
heading("4.4 Evaluation protocol", level=2)
para(
    "All experiments in this paper use the validation split (496 images) exclusively. "
    "The test split (507 images) was never opened. This follows the same test-set-"
    "isolation discipline already established in the source project "
    "(docs/transfer_cnn_test_set_access_2026-09-18.md): the source architecture "
    "comparison treats the test set as a resource opened exactly once, for a finalized "
    "candidate list. The 90 pruning/quantization variants evaluated in this paper (9 "
    "architectures x 10 pruning levels, plus 9 x 3 quantization variants) are an "
    "exploratory sweep, not a finalized candidate list, and belong on validation for "
    "the same reason every other sweep in this project's history does."
)
heading("4.5 Hardware and latency measurement", level=2)
para(
    "All experiments ran on the same CPU-only, 8 GB RAM development machine used "
    "throughout this project -- no GPU, and no real edge/mobile device (Raspberry Pi, "
    "Android phone) was available. Latency was measured as the mean of 20 single-image "
    "(batch=1) forward passes after 4 warmup passes, on a machine shared with normal "
    "background load (browser, editor, and other applications actively running, not a "
    "dedicated or isolated benchmarking environment). Every latency number in this paper "
    "is reported as a CPU-only proxy under this caveat -- Section 5.3 shows this caveat "
    "is load-bearing, not a formality, since the proxy is shown to give the wrong answer "
    "in a specific, checkable way."
)

# ================= 5. RESULTS =================
heading("5. Results", level=1)
heading("5.1 Pruning sensitivity varies by architecture, not by size", level=2)
make_table(
    ["Model", "Baseline val. acc.", "Accuracy-drop knee (sparsity)", "Accuracy at knee"],
    [
        ["ResNet18 (baseline)", "80.24%", "60%", "72.98%"],
        ["EfficientNet-B0", "80.65%", "50%", "63.91%"],
        ["EfficientNet-Lite0", "76.41%", "50%", "59.88%"],
        ["ShuffleNetV2-1.0x", "79.23%", "50%", "69.35%"],
        ["MobileNetV3-Small", "76.01%", "40%", "60.08%"],
        ["ShuffleNetV2-0.5x", "79.84%", "40%", "74.19%"],
        ["RepGhostNet-0.5x", "76.81%", "40%", "57.86%"],
        ["SqueezeNet1.1", "78.02%", "30%", "72.58%"],
        ["MobileNetV2-1.0x", "78.83%", "20%", "73.59%"],
    ],
    [1.9, 1.5, 1.9, 1.3],
)
caption(
    "Table 2. \"Knee\" is defined operationally as the first sparsity level (of the ten "
    "tested) at which validation accuracy has fallen more than 5 points from that "
    "model's own 0%-sparsity baseline, or F1 has fallen below 0.05 (complete collapse of "
    "positive-class prediction), whichever comes first. Sorted by knee sparsity, "
    "highest (most pruning-tolerant) to lowest."
)
para(
    "The knee ranges from 20% (MobileNetV2-1.0x, the earliest to degrade) to 60% "
    "(ResNet18, the most tolerant), and does not track baseline parameter count or "
    "checkpoint size in any simple way -- ResNet18 (42.72 MB, the largest candidate) and "
    "ShuffleNetV2-1.0x (5.45 MB, 7.8x smaller) both reach their knee around 50-60% "
    "sparsity, while MobileNetV2-1.0x (9.35 MB, roughly ShuffleNetV2-1.0x's size) "
    "degrades earliest of all nine candidates at just 20%. Beyond each model's own knee, "
    "every candidate continues to degrade toward chance-level accuracy by 70-90% "
    "sparsity, several with F1 collapsing to exactly 0.0000 (the model stops predicting "
    "the positive class at all) -- consistent across architectures even though the point "
    "at which this collapse begins is not."
)
heading("5.2 Quantization fragility tracks squeeze-excitation/swish components, not "
        "architecture family broadly", level=2)
make_table(
    ["Model", "SE/swish?", "FP32 acc.", "Static INT8 acc.", "Accuracy drop", "Static size (MB)"],
    [
        ["EfficientNet-B0", "Yes", "80.65%", "56.65%", "23.99 pt", "4.72"],
        ["MobileNetV3-Small", "Yes", "76.01%", "50.00%", "26.01 pt", "1.24"],
        ["RepGhostNet-0.5x", "Yes", "76.81%", "47.98%", "28.83 pt", "1.63"],
        ["MobileNetV2-1.0x", "No", "78.83%", "71.37%", "7.46 pt", "2.68"],
        ["EfficientNet-Lite0", "No", "76.41%", "72.38%", "4.03 pt", "4.16"],
        ["SqueezeNet1.1", "No", "78.02%", "75.60%", "2.42 pt", "0.86"],
        ["ShuffleNetV2-0.5x", "No", "79.84%", "78.43%", "1.41 pt", "0.64"],
        ["ResNet18 (baseline)", "No", "80.24%", "79.44%", "0.81 pt", "10.79"],
        ["ShuffleNetV2-1.0x", "No", "79.23%", "78.43%", "0.81 pt", "1.58"],
    ],
    [1.7, 0.8, 1.0, 1.2, 1.1, 1.1],
)
caption(
    "Table 3. Sorted by accuracy drop under static (full-network) INT8 quantization, "
    "worst to best. All three SE/swish architectures (verified per Section 4.3, not "
    "assumed) are the three worst -- two falling to at or below chance-level accuracy "
    "for a roughly balanced dataset (50.00% and 47.98%) -- with a clean separation from "
    "the six plain-ReLU/ReLU6 architectures, none of which drops more than 7.5 points."
)
para(
    "Mean accuracy drop for the three SE/swish architectures: 26.28 percentage points. "
    "Mean accuracy drop for the six plain-ReLU/ReLU6 architectures: 2.82 percentage "
    "points. Ratio: 9.3x. This is not a marginal or borderline pattern -- there is no "
    "overlap between the two groups in Table 3 (the SE/swish group's smallest drop, "
    "23.99 points, is still over 3x the plain-ReLU group's largest drop, 7.46 points). "
    "Dynamic quantization (Linear-layer-only, full results in the accompanying JSON) "
    "shows no comparable pattern in either group, consistent with Section 4.2's "
    "expectation that it barely touches a convolution-heavy backbone -- the fragility "
    "here is specific to quantizing the SE/swish convolutional layers themselves, not a "
    "general property of these architectures' classification heads."
)
heading("5.3 CPU-only latency is not merely imprecise -- it is directionally unreliable",
        level=2)
para(
    "If CPU-only development-machine timing were simply a noisy but unbiased proxy for "
    "real edge latency, quantized models would be expected to measure faster than FP32 "
    "on average, even if the exact numbers were not trustworthy. That is not what was "
    "observed. Static INT8 measured faster than FP32 for ResNet18 (10.6 ms vs. 40.4 ms) "
    "and SqueezeNet1.1 (13.2 ms vs. 26.2 ms), but over twice as slow for EfficientNet-B0 "
    "(89.8 ms vs. 35.5 ms) and EfficientNet-Lite0 (71.2 ms vs. 27.7 ms), with no "
    "consistent direction across the remaining five architectures either. The most "
    "plausible explanation is that this torch build's onednn CPU backend has fast fused "
    "INT8 kernels for some op patterns (plain conv/BN/ReLU stacks) but not others "
    "(depthwise-separable and SE-block-heavy patterns common in the mobile-oriented "
    "candidates) -- but this was not root-caused further, and is offered as a plausible "
    "explanation, not a verified one. The one thing this result reliably establishes: a "
    "single dev-machine CPU benchmark cannot be used to infer whether quantizing any "
    "given architecture will make it faster or slower, even in direction, let alone by "
    "how much -- a real deployment decision needs a measurement on the actual target "
    "runtime and hardware, which this study does not have access to."
)

# ================= 6. DISCUSSION =================
heading("6. Discussion", level=1)
heading("6.1 Compression cost is not architecture-agnostic", level=2)
para(
    "The central finding of this paper is that a compression step applied uniformly "
    "across architectures does not produce a uniform cost. Had this project selected an "
    "architecture on accuracy/size/speed grounds alone (as the source comparison did, "
    "necessarily, since accuracy differences were not statistically significant) without "
    "also checking compression robustness, it could plausibly have selected "
    "MobileNetV3-Small -- the smallest checkpoint among the eight candidates (3.95 MB) "
    "and a reasonable choice on size grounds alone -- and only discovered afterward, at "
    "the compression step of a real deployment pipeline, that it collapses to exactly "
    "chance-level accuracy (50.00%) under a standard post-training INT8 quantization "
    "procedure. This is the practical argument for treating compression robustness as "
    "part of architecture selection, not a separate step performed after the fact."
)
heading("6.2 This generalizes a known finding to a new domain and extends it to two more "
        "architecture families", level=2)
para(
    "The EfficientNet/MobileNetV3 post-training-quantization fragility documented in "
    "Section 2.1 was previously reported on ImageNet-scale natural-image classification. "
    "This paper's contribution is showing the same fragility, at comparable relative "
    "magnitude (roughly a 20-30 point collapse in both this study and the previously "
    "reported EfficientNet case), on a substantially smaller (3,330-image), clinically "
    "specific (dermatology lesion classification) dataset -- suggesting the underlying "
    "mechanism (quantization error in SE-attention and swish-activation layers) is a "
    "property of the architecture and the quantization procedure, not an artifact of "
    "ImageNet's scale or content. Extending the same test to RepGhostNet-0.5x, an "
    "architecture family not covered in the sources found for Section 2.1, and finding "
    "the same collapse (28.83-point drop, the largest of the nine candidates) is "
    "additional support that the SE-block mechanism, not something specific to "
    "EfficientNet or MobileNetV3 individually, is the operative cause."
)
heading("6.3 Reinforcement, not reversal, of the prior deployment choice", level=2)
para(
    "ShuffleNetV2-1.0x was selected in the source report on checkpoint size and CPU "
    "inference latency, with no accuracy claim (none of the nine candidates was "
    "statistically distinguishable on accuracy). This paper's results add two more "
    "reasons that selection holds up: ShuffleNetV2-1.0x has among the highest pruning "
    "knees of the nine candidates (50% sparsity, tied for second-most-tolerant) and among "
    "the smallest quantization accuracy costs (0.81 points, tied for least-fragile). Its "
    "earlier selection was not informed by either property -- they were not tested at the "
    "time -- but neither result contradicts it, and the quantization result in particular "
    "(Section 5.2) makes it clear the selection avoided a real risk (the SE/swish "
    "collapse pattern) that was not visible from the source report's accuracy/size/speed "
    "comparison alone."
)

# ================= 7. LIMITATIONS =================
heading("7. Limitations", level=1)
bullet("No real edge/mobile hardware. Every latency number is a CPU-only proxy on a "
       "shared development laptop, shown directly in Section 5.3 to be an unreliable, "
       "not merely imprecise, stand-in for real deployment latency.")
bullet("No fine-tuning after pruning. The pruning knees in Table 2 describe one-shot "
       "magnitude pruning with no chance to recover accuracy; a real deployment pipeline "
       "would typically fine-tune after pruning, which could shift some or all of these "
       "knees to higher sparsity.")
bullet("Only one quantization backend (x86/onednn) was available on this project's torch "
       "build. The SE/swish fragility finding (Section 5.2) may be specific to this "
       "backend's kernel support rather than a universal property of INT8 quantization "
       "on all hardware/runtime combinations -- a mobile-native runtime (Core ML, TFLite, "
       "ONNX Runtime Mobile) with dedicated INT8 kernels for these op patterns could "
       "behave differently, and this was not tested.")
bullet("Validation-set-only evaluation (496 images), consistent with this project's "
       "test-set-isolation discipline (Section 4.4), but a smaller sample than the "
       "507-image test set used for the source report's headline accuracy numbers -- "
       "not directly comparable to those numbers for that reason, and not claimed to be.")
bullet("Only unstructured pruning was tested. Structured (channel-level) pruning, which "
       "would need a library such as torch-pruning (not installed in this project) to "
       "physically shrink the model, was not attempted -- it is the more likely route to "
       "a real measured speedup, since unstructured pruning is shown here (Table 2's "
       "companion latency data, in the accompanying JSON) not to reduce measured latency "
       "at all, consistent with dense-tensor CPU computation not benefiting from "
       "unstructured zeros.")

# ================= 8. CONCLUSION =================
heading("8. Conclusion", level=1)
para(
    "Applying an identical pruning and quantization procedure to nine architectures "
    "already compared for the same clinical image classification task shows that "
    "compression robustness is not a property that can be assumed constant across "
    "architectures, or predicted from baseline accuracy, size, or family name alone. The "
    "clearest single result -- a 9.3x difference in mean quantization accuracy cost "
    "between architectures with and without squeeze-excitation/swish components, "
    "verified against each model's actual module composition -- generalizes a "
    "previously narrow, ImageNet-domain finding to a new clinical imaging task and a "
    "wider set of architecture families than it had been tested on before. The practical "
    "outcome for this project is a reinforcement of an already-made decision: "
    "ShuffleNetV2-1.0x, selected before this study on size and speed grounds, turns out "
    "to also be among the most compression-robust of the nine candidates tested -- a "
    "property the original selection did not know about but happened not to contradict."
)

doc.save(str(OUT_DIR / "edge_ai_compression_paper_2026-09-19.docx"))
print(f"Saved to {OUT_DIR / 'edge_ai_compression_paper_2026-09-19.docx'}")
