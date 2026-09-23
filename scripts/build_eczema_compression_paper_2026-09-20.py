# -*- coding: utf-8 -*-
"""Builds the eczema image-classification compression/deployment manuscript as a .docx,
using the same python-docx helper conventions as build_edge_ai_paper_2026-09-19.py.

Reframing note (2026-09-19/20): this manuscript supersedes
edge_ai_compression_paper_2026-09-19.docx ("Compression Robustness Is Architecture-
Dependent"), which framed this work as a generic lightweight-CNN/edge-AI compression
study. Per explicit instruction, this is instead an eczema image-classification study
for an Honors Project on flexible electronics for human healthcare, in which compression
is the deployment MECHANISM (getting a classifier onto resource-constrained edge/wearable
hardware), not the end goal. The eczema application stays central throughout; claims are
scoped to this dataset and deployment objective, not stated as general lightweight-CNN
findings, and no clinical-diagnosis claim is made anywhere in this document.

All numeric tables in this script are read directly from the JSON result files on disk
at build time (not hand-transcribed), specifically to avoid repeating the kind of
transcription error already found and documented in the superseded draft (a "90 vs 117
variants" counting mistake, papers/edge-ai-lightweight-deployment/
PAPER_REVISION_STATUS_2026-09-19.md Section 1.1).

Publication-readiness revision (this file, 2026-09-20 pass 2): implements a forensic
self-audit (A. critical errors, B. overclaims, C. missing math, D. claims to preserve,
E. corrected central-contribution statement, F. corrected conclusion outline) and
compresses the manuscript toward a 15-20 page target. Table count reduced from 16 to 10
by merging same-shape sub-tables (SkinDisNet, pruning+fine-tune, final test) into single
multi-row tables; repeated qnnpack-proxy caveats stated once (Section 4.2) and
cross-referenced rather than restated; the "Novelty Audit" section is consolidated from
four historical sub-sections into two. The "robust" backend-stability label is renamed
"backend-stable" throughout (verdict3() and all display sites) since "robust" reads as a
stronger claim than a three-bucket, study-defined, non-clinical classification supports.
"""
import json
from pathlib import Path

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

from paths import ROOT

PAPER_DIR = ROOT / "papers" / "edge-ai-lightweight-deployment"
RESULTS_DIR = ROOT / "results"
DOCS_DIR = ROOT / "docs"
BODY_FONT = "Calibri"
HEAD_FONT = "Calibri"


def load(path):
    with open(path) as f:
        return json.load(f)


EXT = load(PAPER_DIR / "edge_ai_extended_analysis_2026-09-19.json")
SIM = load(PAPER_DIR / "edge_simulation_all_architectures_2026-09-19.json")
QAT = load(PAPER_DIR / "edge_ai_qat_2026-09-19.json")
STRUCT = load(PAPER_DIR / "edge_ai_structured_pruning_2026-09-19.json")
PF = load(PAPER_DIR / "edge_ai_pruning_finetune_2026-09-19.json")
PARETO = load(PAPER_DIR / "edge_ai_pareto_analysis_2026-09-19.json")
BACKEND = load(PAPER_DIR / "edge_ai_qnnpack_backend_check_2026-09-19.json")
CORR = load(RESULTS_DIR / "robustness_cross_architecture_summary.json")
CORR_BY_MODEL = {r["model"]: r for r in CORR["rows"]}
UC_SHUFFLE_X86 = load(RESULTS_DIR / "robustness_under_compression_shufflenet_v2_x1_0_x86backend_original.json")
UC_SHUFFLE_QNN = load(RESULTS_DIR / "robustness_under_compression_shufflenet_v2_x1_0.json")
UC_EFFB0 = load(RESULTS_DIR / "robustness_under_compression_efficientnet_b0.json")
UC_MBV2 = load(RESULTS_DIR / "robustness_under_compression_mobilenetv2_100.json")
SDN9 = load(PAPER_DIR / "skindisnet_all9_external_validation_2026-09-20.json")
PERCHAN = load(PAPER_DIR / "qnnpack_perchannel_diagnostic_2026-09-20.json")
TEST = load(PAPER_DIR / "final_test_frozen_results_2026-09-20.json")

ARCH_LABELS = {
    "resnet18": "ResNet18 (baseline)", "efficientnet_b0": "EfficientNet-B0",
    "efficientnet_lite0": "EfficientNet-Lite0", "mobilenetv3_small": "MobileNetV3-Small",
    "mobilenetv2_100": "MobileNetV2-1.0x", "shufflenet_v2_x0_5": "ShuffleNetV2-0.5x",
    "shufflenet_v2_x1_0": "ShuffleNetV2-1.0x", "squeezenet1_1": "SqueezeNet1.1",
    "repghostnet_050": "RepGhostNet-0.5x",
}
SE_SWISH = {"efficientnet_b0", "mobilenetv3_small", "repghostnet_050"}
ALL9 = ["resnet18", "efficientnet_b0", "efficientnet_lite0", "mobilenetv3_small",
        "mobilenetv2_100", "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "squeezenet1_1",
        "repghostnet_050"]
QNNPACK_AFFECTED = ["efficientnet_b0", "efficientnet_lite0", "mobilenetv3_small",
                    "mobilenetv2_100", "repghostnet_050"]


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def pt_(a, b, d=1):
    return f"{(a - b) * 100:+.{d}f} pt"


# Computed once, here, and reused everywhere this statistic is quoted (abstract,
# Section 5.2) so the two can never drift apart the way they did in an earlier draft.
SE_MEAN_DROP = sum(
    EXT["per_model"][a]["fp32"]["point"]["accuracy"] - EXT["per_model"][a]["static_int8"]["point"]["accuracy"]
    for a in SE_SWISH
) / len(SE_SWISH)
NON_SE = [a for a in ALL9 if a not in SE_SWISH]
NON_SE_MEAN_DROP = sum(
    EXT["per_model"][a]["fp32"]["point"]["accuracy"] - EXT["per_model"][a]["static_int8"]["point"]["accuracy"]
    for a in NON_SE
) / len(NON_SE)
SE_RATIO = SE_MEAN_DROP / NON_SE_MEAN_DROP
SE_DIFF_PT = SE_MEAN_DROP - NON_SE_MEAN_DROP


def balanced_acc(point):
    return (point["sensitivity_recall"] + point["specificity"]) / 2


# Per-architecture derived quantities computed once from the raw JSON, used throughout
# the paper (tables, discussion, decision framework) so every derived number traces to
# exactly one computation, not a hand-typed value that could drift from its source.
MATH = {}
for a in ALL9:
    fp32 = EXT["per_model"][a]["fp32"]["point"]
    fbg = BACKEND[a]["x86_fbgemm_backend_original"]
    qnn = BACKEND[a]["qnnpack_backend_this_mac"]["point"]
    MATH[a] = {
        "fp32": fp32, "fbgemm": fbg, "qnnpack": qnn,
        "delta_fp32_to_fbgemm": fp32["accuracy"] - fbg["accuracy"],
        "delta_fp32_to_qnnpack": fp32["accuracy"] - qnn["accuracy"],
        "delta_fbgemm_to_qnnpack": fbg["accuracy"] - qnn["accuracy"],
        "r_acc_fbgemm": fbg["accuracy"] / fp32["accuracy"],
        "r_acc_qnnpack": qnn["accuracy"] / fp32["accuracy"],
        "r_backend": qnn["accuracy"] / fbg["accuracy"] if fbg["accuracy"] else float("nan"),
        "rel_degradation_backend": (fbg["accuracy"] - qnn["accuracy"]) / fbg["accuracy"] if fbg["accuracy"] else float("nan"),
        "r_sens_backend": (qnn["sensitivity_recall"] / fbg["sensitivity_recall"]) if fbg["sensitivity_recall"] else float("nan"),
        "r_spec_backend": (qnn["specificity"] / fbg["specificity"]) if fbg["specificity"] else float("nan"),
        "balanced_acc_fp32": balanced_acc(fp32),
        "balanced_acc_fbgemm": balanced_acc(fbg),
        "balanced_acc_qnnpack": balanced_acc(qnn),
        "margin_qnnpack": min(qnn["sensitivity_recall"], qnn["specificity"]),
    }

# Fix B3: "robust" renamed "backend-stable" -- a three-bucket, study-defined,
# non-clinical classification of INT8-vs-FP32 behavior on this project's own held-out
# split, not a general robustness or safety claim.
def verdict3(qnn_point):
    if qnn_point["f1"] < 0.05:
        return "collapsed"
    if qnn_point["sensitivity_recall"] < 0.20 or qnn_point["specificity"] < 0.20:
        return "severely degraded"
    return "backend-stable"


def disp_status(raw):
    """Normalizes verdict strings coming from older JSON files (which still say
    'robust' / 'severely_degraded' internally) to this document's current three-bucket
    labels, without editing the underlying result files."""
    s = raw.replace("_", " ")
    return "backend-stable" if s == "robust" else s


N_COLLAPSED = sum(1 for a in ALL9 if verdict3(MATH[a]["qnnpack"]) == "collapsed")
N_DEGRADED = sum(1 for a in ALL9 if verdict3(MATH[a]["qnnpack"]) == "severely degraded")
N_STABLE = sum(1 for a in ALL9 if verdict3(MATH[a]["qnnpack"]) == "backend-stable")
N_SE_SWISH = len(SE_SWISH)
N_NON_SE = len(NON_SE)


def frac(n, d=9):
    return f"{n}/{d} ({pct(n / d, 1)} of the 9 architectures tested here)"


doc = Document()
section = doc.sections[0]
section.page_width = Inches(8.5)
section.page_height = Inches(11)
section.top_margin = Inches(0.9)
section.bottom_margin = Inches(0.9)
section.left_margin = Inches(0.95)
section.right_margin = Inches(0.95)

normal = doc.styles["Normal"]
normal.font.name = BODY_FONT
normal.font.size = Pt(10.5)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.08

for lvl, size, bold in [(1, 15, True), (2, 12.5, True), (3, 11, True)]:
    st = doc.styles[f"Heading {lvl}"]
    st.font.name = HEAD_FONT
    st.font.size = Pt(size)
    st.font.bold = bold
    st.font.color.rgb = RGBColor(0x1F, 0x1F, 0x1F)
    st.paragraph_format.space_before = Pt(14 if lvl == 1 else 10)
    st.paragraph_format.space_after = Pt(4)


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
        r.font.size = Pt(9)
        set_cell_shading(hdr_cells[i], header_fill)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].width = Inches(col_widths_in[i])
            p = cells[i].paragraphs[0]
            p.text = ""
            r = p.add_run(str(val))
            r.font.size = Pt(9)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def caption(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(8.5)
    p.paragraph_format.space_after = Pt(10)


def figure(path, width_in=6.1):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(path), width=Inches(width_in))


# ================= TITLE PAGE =================
para("", space_after=30)
para("Backend-Dependent INT8 Quantization Behavior in a Compressed Eczema Image "
     "Classifier: A 9-Architecture Study for Edge and Wearable Healthcare Deployment",
     bold=True, size=17, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)
para("(Working title)", italic=True, size=10, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=24)
para("Honors Project -- Flexible Electronics for Human Healthcare", size=12,
     align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
para("September 2026", size=11, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=40)
para(
    "Scope statement: this manuscript studies model compression (pruning, INT8 "
    "quantization, quantization-aware training) as a MECHANISM for deploying an eczema "
    "image classifier on resource-constrained edge and wearable hardware, in the context "
    "of a flexible-electronics healthcare platform. It does not claim diagnostic "
    "accuracy, clinical validity, or regulatory readiness for any model described here. "
    "\"Eczema\" refers to the label used in this project's own curated dataset, not a "
    "verified clinical diagnosis. All findings are scoped to the 9 architectures, the "
    "dataset, and the deployment objective described in this paper -- not to lightweight "
    "CNNs, ARM devices, or PyTorch quantization in general.",
    italic=True, size=9.5, align=WD_ALIGN_PARAGRAPH.CENTER)
doc.add_page_break()

# ================= ABSTRACT =================
heading("Abstract", level=1)
para(
    f"Deploying an image classifier on wearable or edge healthcare hardware usually "
    f"requires compressing it -- through pruning, INT8 quantization, or both -- to fit "
    f"memory, latency, and power budgets. This paper studies what that compression does "
    f"to an eczema-vs-other-skin-condition image classifier across nine convolutional "
    f"architectures (eight lightweight candidates plus a ResNet18 baseline; nine "
    f"architectures total). We measure not only accuracy but sensitivity, specificity, "
    f"balanced accuracy, and F1 under one-shot magnitude pruning, post-training static "
    f"INT8 quantization (PTQ), quantization-aware training (QAT), structured (channel) "
    f"pruning, and pruning-with-fine-tuning. The central, unplanned finding is that "
    f"static INT8 quantization behaves very differently depending on the CPU quantized "
    f"backend used to run it, even with the same trained weights and the same nominal "
    f"quantization procedure: on the ARM/qnnpack backend available on this project's "
    f"Apple Silicon machine, {N_COLLAPSED} of 9 architectures collapse to a "
    f"near-constant classifier ({frac(N_COLLAPSED)}) and {N_DEGRADED} more become "
    f"severely imbalanced between sensitivity and specificity ({frac(N_DEGRADED)}), "
    f"while only {N_STABLE} remain within this study's own backend-stability criterion "
    f"({frac(N_STABLE)}); on the x86/fbgemm backend used for this project's original "
    f"development machine, all 9 architectures quantize with only small accuracy "
    f"changes. Architectures using squeeze-and-excitation blocks and swish-family "
    f"activations show a mean FP32-to-INT8 accuracy drop of "
    f"{pct(SE_MEAN_DROP)} versus {pct(NON_SE_MEAN_DROP)} for the other "
    f"{N_NON_SE} architectures ({SE_DIFF_PT * 100:+.1f} percentage points, "
    f"approximately {SE_RATIO:.1f}x) -- an association, not a demonstrated causal "
    f"mechanism, since an ablation removing these components from already-trained "
    f"weights collapses FP32 accuracy on its own and so cannot isolate the effect. A "
    f"forced per-channel-quantization diagnostic explains most of this gap for one "
    f"architecture, a smaller part for a second, and none of it for three others that "
    f"remain fully collapsed regardless -- so backend default recipe differences are a "
    f"partial, architecture-specific contributor, not a single unifying mechanism. Using "
    f"a decision framework based on accuracy retention, sensitivity/specificity margins, "
    f"and Pareto dominance over model size, we identify a feasible candidate set for "
    f"this deployment objective and show that two size variants of one architecture "
    f"(ShuffleNetV2, 0.5x vs 1.0x) are Pareto non-dominant under our own metrics -- "
    f"the smaller variant wins on size, specificity, and worst-case margin; the larger "
    f"wins on accuracy and sensitivity -- so no single winner is declared. Corruption "
    f"robustness (Gaussian blur, JPEG compression, noise, brightness, contrast, color "
    f"shift) and its interaction with compression are reported as an exploratory, "
    f"hypothesis-generating analysis, not a confirmatory one. A zero-shot check on a "
    f"second, external dataset (SkinDisNet) reproduces the backend-collapse signature "
    f"for the three fully-collapsed architectures, but cannot establish broader external "
    f"generalization, because even the FP32 models already show substantial accuracy "
    f"loss on that dataset before any quantization is applied. All comparisons reported "
    f"as the paper's central and secondary findings were reproduced on a final, "
    f"previously untouched held-out test split, evaluated once, after the evaluation "
    f"protocol was committed to disk. Every claim in this paper is scoped to the 9 "
    f"architectures, the qnnpack default recipe on this machine, and this dataset -- not "
    f"to lightweight CNNs, ARM hardware, or PyTorch quantization in general.",
    space_after=10)

# ================= 1. INTRODUCTION =================
heading("1. Introduction", level=1)
para(
    "Wearable and edge devices for chronic-condition monitoring -- including flexible "
    "electronics platforms aimed at continuous skin-health tracking -- must run "
    "inference under tight memory, latency, and power constraints. A natural approach "
    "is to train a convolutional classifier on curated clinical-style images and then "
    "compress it (pruning, INT8 quantization, or both) for on-device or near-device "
    "execution. This paper asks what compression actually does to an eczema-vs-other "
    "skin-condition image classifier, across nine architectures chosen to span the "
    "lightweight-CNN design space (depthwise separable convolutions, squeeze-and-"
    "excitation attention, channel shuffle/grouped convolutions, Fire modules, and "
    "reparameterized ghost modules), plus a ResNet18 baseline. Compression is treated "
    "throughout as the deployment mechanism for an eventual wearable/edge eczema-"
    "monitoring system, not as the paper's subject in its own right.")
heading("1.1 Research questions", level=2)
bullet("RQ1: For each of 9 architectures, how much accuracy, sensitivity, and specificity "
       "are lost to one-shot magnitude pruning, static INT8 quantization, and their "
       "combination, and where is each architecture's own pruning \"knee\"?")
bullet("RQ2: Does quantization-aware training (QAT) recover accuracy lost to PTQ for the "
       "architectures most affected by it?")
bullet("RQ3: Does the CPU quantized backend (x86/fbgemm vs. ARM/qnnpack) change INT8 "
       "behavior for the same trained weights and the same nominal procedure -- and if "
       "so, by how much, and for which architectures?")
bullet("RQ4: Given accuracy, sensitivity/specificity balance, and model size, which of "
       "the 9 architectures form a defensible candidate set for this deployment "
       "objective, and can a single architecture be declared the best choice?")
bullet("RQ5: Are the compression findings robust to common image corruptions, and do "
       "they reproduce on a final, previously unopened held-out test split and on a "
       "second, external dataset?")
heading("1.2 Clinical scope and terminology", level=2)
para(
    "This paper makes no diagnostic claim. \"Eczema\" denotes the label in this "
    "project's own curated dataset (Section 3), not a dermatologist-verified diagnosis "
    "for every image, and the task is binary (eczema-labeled vs. all other classes in "
    "the dataset), not a clinical differential-diagnosis task with graded severity. "
    "Terms such as \"collapsed,\" \"severely degraded,\" and \"backend-stable\" "
    "(Section 4.7) are this study's own operational thresholds on this project's "
    "held-out split, not clinical safety classifications, and \"sensitivity\"/"
    "\"specificity\" here describe binary-classifier behavior on a curated dataset, not "
    "diagnostic performance in clinical use.")

# ================= 2. RELATED WORK =================
heading("2. Related Work", level=1)
heading("2.1 Lightweight architectures and dermatology image classification", level=2)
para(
    "Depthwise-separable and grouped-convolution architectures (MobileNetV2 [1], "
    "MobileNetV3 [2], ShuffleNetV2 [3], SqueezeNet [4], EfficientNet and its "
    "\"Lite\" variants [5], RepGhostNet [6]) were designed to trade accuracy for "
    "parameter count and FLOPs on mobile hardware, motivating their use here as "
    "compression candidates rather than as a single recommended default. Convolutional "
    "classifiers have been applied to dermatology images across several tasks: "
    "skin-lesion/melanoma screening with deep CNNs [7], and, closer to this paper's "
    "application, at least one MobileNet-based atopic-dermatitis-vs-psoriasis "
    "classifier deployed on a Raspberry Pi [8] and an EfficientNet-based mobile "
    "eczema/acne classifier [9]; a 2025 review surveys deep-learning approaches to "
    "eczema image detection generally [10]. None of these directly studies "
    "compression (pruning, PTQ, QAT) as this paper does; they motivate the deployment "
    "objective, not the compression methodology.")
heading("2.2 Pruning and quantization methodology", level=2)
para(
    "Unstructured magnitude pruning [11] and structured/channel pruning [12] are "
    "well-established compression techniques; this paper uses global unstructured L1 "
    "pruning as its primary method and torch-pruning's magnitude-based structured "
    "pruning [13] as a secondary check (Section 4.4-4.5). Post-training static "
    "quantization and quantization-aware training [14,15] are likewise standard; this "
    "paper uses PyTorch's FX graph-mode quantization API [16] rather than a custom "
    "implementation.")
heading("2.3 Squeeze-and-excitation, swish activations, and quantization fragility", level=2)
para(
    "Squeeze-and-excitation (SE) blocks [17] and swish-family activations [18] are used "
    "in EfficientNet-B0, MobileNetV3-Small, and RepGhostNet-0.5x among this paper's nine "
    "architectures. Prior work has noted that swish's smooth, non-piecewise-linear "
    "shape is harder to represent under low-bit, fixed-point mobile inference than "
    "ReLU-family activations -- the motivation MobileNetV3 itself gives for "
    "introducing \"hard-swish\" as a quantization- and efficiency-friendly "
    "approximation [2] -- and quantization practitioners more broadly note that "
    "smooth, non-piecewise-linear activations are harder to calibrate than "
    "piecewise-linear ones [15]. This is consistent with this paper's own PTQ results "
    "(Section 5.2), but neither source establishes SE specifically (as opposed to "
    "swish) as harder to quantize. We attempted an "
    "ablation to test this more directly -- removing SE blocks or replacing swish with "
    "ReLU on the already-trained weights, without retraining -- but this collapsed "
    "FP32 accuracy to approximately 49.8-50% (chance level) for all three architectures "
    "before any quantization was applied, because the components are load-bearing for "
    "the model's learned function and cannot be removed post-hoc as a clean causal "
    "test. The association reported in Section 5.2 is therefore a structural "
    "correlation across architecture families, not an isolated causal mechanism, and no "
    "formal significance test was run on the two-group comparison (Section 5.2).")
heading("2.4 Backend-dependent INT8 behavior", level=2)
para(
    "PyTorch's quantized backends (fbgemm for x86, qnnpack for ARM) use different "
    "default quantization recipes -- notably per-channel vs. per-tensor weight "
    "quantization and different histogram-observer settings (Section 4.2) -- and "
    "engineering reports have already documented accuracy differences between them for "
    "some architectures [19,20]. This paper's contribution on this point is not "
    "discovering that the backends differ, but showing, with a controlled same-weights "
    "comparison across 9 architectures on an applied clinical-image task, how severe "
    "that difference can be (complete classifier collapse, not a modest accuracy gap) "
    "and how architecture-dependent it is (Section 5.5).")

# ================= 3. DATASET AND TASK =================
heading("3. Dataset and Task", level=1)
para(
    "The task is binary classification of dermatology-style images into an "
    "eczema-labeled class versus all other classes in this project's curated dataset "
    "(SkinDisease manifest, train/val/test splits). All 9 architectures are trained "
    "identically on the training split and evaluated on the same validation split for "
    "every result in Sections 5-8; the test split (507 images) is evaluated exactly "
    "once, in Section 9, using a protocol committed to disk before the split was opened "
    "for this compression study. Sections 3-8 of this paper describe validation-split "
    "results; where a result is later reproduced on the held-out test split, this is "
    "stated explicitly (Section 9.3).")
heading("3.1 No patient-level identifiers", level=2)
para(
    "The dataset manifest carries no patient identifier, so a subject-level train/"
    "test leakage check was not possible and is not claimed; the split is at the "
    "image level.")
heading("3.2 Other limitations of this dataset and task", level=2)
bullet("The task is binary (eczema-labeled vs. other), not the multi-class differential "
       "diagnosis a clinician performs; a classifier tuned for this binary task may not "
       "transfer to that harder problem.")
bullet("Image quality, lighting, and framing in this dataset may not match images "
       "captured by an eventual wearable/edge camera sensor; no such sensor-captured "
       "images were evaluated in this study.")
bullet("One zero-shot check against a second, external dataset (SkinDisNet) was "
       "performed (Section 5.5.2); it is limited by substantial FP32 accuracy loss on "
       "that dataset before any quantization is applied, so it does not by itself "
       "establish that this project's models generalize beyond their own training "
       "distribution.")
bullet("Class balance, demographic composition, and skin-tone representation in this "
       "dataset were not separately audited for this paper; any disparity present in "
       "the source dataset is inherited without correction.")

# ================= 4. METHODS =================
heading("4. Methods", level=1)
heading("4.1 Architectures and training", level=2)
para(
    "Nine architectures are trained identically on the training split: ResNet18 "
    "(baseline) and eight lightweight candidates -- EfficientNet-B0, EfficientNet-"
    "Lite0, MobileNetV3-Small, MobileNetV2-1.0x, ShuffleNetV2-0.5x, ShuffleNetV2-1.0x, "
    "SqueezeNet1.1, and RepGhostNet-0.5x. Per-architecture FP32 accuracy, sensitivity, "
    "specificity, and AUROC on the validation split are reported once, in Table 1, and "
    "reused (not recomputed) as the FP32 reference point for every later table.")
heading("4.2 Post-training static INT8 quantization (PTQ)", level=2)
para(
    "Quantization uses PyTorch's FX graph-mode static quantization API "
    "(prepare_fx/convert_fx) with the default per-backend qconfig mapping "
    "(get_default_qconfig_mapping), calibrated on the training split only. Two CPU "
    "quantized backends are used in this paper: x86/fbgemm (this project's original "
    "development machine) and ARM/qnnpack (this paper's new-experiment machine, whose "
    "PyTorch build supports only qnnpack -- torch.backends.quantized.supported_engines "
    "== ['qnnpack', 'none']). These backends' DEFAULT recipes are not identical: fbgemm "
    "defaults to per-channel symmetric weight quantization with a reduced-range "
    "histogram activation observer, while qnnpack defaults to per-tensor symmetric "
    "weight quantization with a full-range histogram observer. This is a genuine "
    "methodology difference between the two conditions compared in Section 5.5, not a "
    "controlled ablation of \"the backend\" alone; Section 5.5.1 reports a targeted "
    "diagnostic that re-quantizes on qnnpack with weight granularity forced to "
    "per-channel, to separate the contribution of granularity from whatever remains "
    "backend-specific.")
heading("4.3 Quantization-aware training (QAT)", level=2)
para(
    "For the three squeeze-and-excitation/swish architectures most affected by PTQ "
    "(Section 5.2), fake-quantization modules are inserted via prepare_qat_fx and the "
    "model is fine-tuned for 3 epochs on the training split before conversion to a true "
    "INT8 model (Section 5.4).")
heading("4.4 Unstructured pruning", level=2)
para(
    "Global unstructured L1 magnitude pruning (torch.nn.utils.prune.global_unstructured) "
    "removes the smallest-magnitude weights across all prunable layers jointly, at "
    "sparsity targets from 0% to 60-70% depending on architecture. Each architecture's "
    "\"pruning knee\" (Table 1) is defined as the lowest sparsity target at which "
    "validation accuracy first drops by more than 2 percentage points from that "
    "architecture's own 0%-sparsity accuracy -- an architecture-relative threshold, not "
    "an absolute one, so knee values are not comparable in absolute sparsity across "
    "architectures with different parameter redundancy.")
heading("4.5 Structured (channel) pruning", level=2)
para(
    "Structured pruning uses torch-pruning's magnitude-based MetaPruner to remove whole "
    "output channels, tested at 20%, 30%, and 50% channel-sparsity targets for 7 of the "
    "9 architectures. Both ShuffleNetV2 variants are excluded: torch-pruning raises a "
    "ZeroDivisionError on ShuffleNetV2's channel-shuffle/grouped-convolution structure, "
    "including for ShuffleNetV2-1.0x, this project's own leading deployment candidate "
    "(Section 7). This is reported as a tooling limitation, not a result.")
heading("4.6 Pruning followed by fine-tuning", level=2)
para(
    "For 3 architectures (ResNet18, MobileNetV3-Small, ShuffleNetV2-1.0x) at 3 "
    "sparsity levels (30%, 50%, 70%), the model is pruned, then fine-tuned for several "
    "epochs on the training split with the pruning mask held fixed, to test whether "
    "fine-tuning recovers accuracy lost immediately after pruning (Section 5.7). A "
    "known data-quality issue affects a small number of recorded epoch wall-clock times "
    "in this experiment (probably host throttling, not a real compute-time difference); "
    "accuracy and loss values are unaffected and used normally, but no latency claim in "
    "this paper is based on those specific timings.")
heading("4.7 Corruption benchmark", level=2)
para(
    "Seven image corruptions (Gaussian blur, brightness up/down, contrast down, "
    "Gaussian noise, JPEG compression, warm color shift) are applied at up to 5 "
    "severity levels to the validation split, for 8 of 9 architectures (corruption "
    "results predate the addition of RepGhostNet-0.5x's full sweep at this project's "
    "outset and were not re-run retroactively) and, for the compression-x-corruption "
    "interaction (Section 6.2), across compression variants (FP32, pruned-at-knee, "
    "static INT8, pruned+INT8) for 3 architectures. This is reported as an exploratory, "
    "hypothesis-generating analysis (Section 6), not a confirmatory benchmark.")
heading("4.8 Metrics", level=2)
para(
    "Let TP, FP, TN, FN denote true/false positives and negatives on the eczema-"
    "labeled-positive binary task. All metrics below are computed on point predictions "
    "at a fixed 0.5 decision threshold unless noted.")
bullet("Accuracy = (TP + TN) / (TP + FP + TN + FN)")
bullet("Precision = TP / (TP + FP)")
bullet("Sensitivity (recall) = TP / (TP + FN) -- the fraction of true eczema-labeled "
       "images correctly flagged positive")
bullet("Specificity = TN / (TN + FP) -- the fraction of true non-eczema images "
       "correctly flagged negative")
bullet("F1 = 2 x Precision x Sensitivity / (Precision + Sensitivity)")
bullet("AUROC: area under the ROC curve over the model's continuous output "
       "probability, threshold-independent")
bullet("Balanced accuracy = (Sensitivity + Specificity) / 2")
bullet("Margin M = min(Sensitivity, Specificity) -- the worse of the two class-wise "
       "rates; used as a single worst-case number for the collapsed/severely-degraded/"
       "backend-stable classification below")
bullet("Delta-Accuracy (ΔAcc) = Acc(condition A) − Acc(condition B), reported "
       "in percentage points (pt)")
bullet("Accuracy retention ratio R = Acc(compressed) / Acc(FP32) -- e.g. "
       "R_qnnpack = Acc(qnnpack INT8) / Acc(FP32)")
bullet("Relative degradation = (Acc(fbgemm) − Acc(qnnpack)) / Acc(fbgemm) -- the "
       "backend gap expressed as a fraction of the fbgemm accuracy, so gaps are "
       "comparable across architectures with different baseline accuracy")
bullet("Fraction of the pruning-induced drop recovered by fine-tuning = "
       "(Acc(after fine-tune) − Acc(immediately after pruning)) / "
       "(Acc(FP32) − Acc(immediately after pruning))")
bullet("95% confidence intervals on all point metrics use a nonparametric bootstrap "
       "(2000 resamples of the validation set with replacement); no parametric "
       "significance test (e.g. a two-proportion z-test) was run anywhere in this paper "
       "unless explicitly stated, and none is stated for the SE/swish group comparison "
       "(Section 2.3, 5.2) or the corruption-ranking comparisons (Section 6.1).")
para(
    "Operational classification used in Tables 5, 5b, and 9: \"collapsed\" if F1 < "
    "0.05; else \"severely degraded\" if Sensitivity < 20% or Specificity < 20%; else "
    "\"backend-stable.\" These thresholds are this study's own choice, fixed before "
    "inspecting most of the qnnpack results, and are not clinical safety thresholds.")
heading("4.9 Deployment characteristics: theoretical vs. measured", level=2)
para(
    "Parameter count and dense checkpoint size (MB) are measured directly from each "
    "trained model. Wall-clock CPU inference time is measured on this project's own "
    "development machines, not on any wearable or embedded target, and is reported as "
    "a theoretical proxy for edge latency, not a measured deployment characteristic; no "
    "figure in this paper should be read as an on-device measurement.")
heading("4.10 Environment, reproducibility, and provenance", level=2)
para(
    "This paper's new experiments (Sections 5.4-5.7, 5.5, 5.5.2, 9) ran on macOS "
    "(Darwin), Apple M5 (ARM64, 10 cores), 16 GB RAM, PyTorch 2.8.0, torchvision 0.23.0, "
    "timm 1.0.29, torch_pruning 1.6.0, with torch.backends.quantized.engine set to "
    "qnnpack (the only engine this build supports). Earlier results reused from this "
    "project's original development machine (Table 1's FP32 sweep, the original PTQ/"
    "pruning simulation, and the fbgemm side of Table 5) ran on Windows/x86_64 with the "
    "fbgemm backend; no trained weight, dataset split, or evaluation procedure was "
    "altered to reuse them, only source code (backend-name detection, dataset path "
    "resolution, a Windows-only memory-guard call). A companion document "
    "(REPRODUCIBILITY_ENVIRONMENT_2026-09-20.md) lists every result file used in this "
    "paper against the machine and script that produced it. The held-out test "
    "evaluation (Section 9) followed a protocol written to disk "
    "(FROZEN_TEST_PROTOCOL_2026-09-20.md) before the test split was opened for this "
    "compression study.")

# ================= 5. RESULTS =================
heading("5. Results", level=1)
heading("5.1 Baseline accuracy and pruning knees", level=2)
para(
    "Table 1 reports each architecture's FP32 accuracy, sensitivity, specificity, and "
    "AUROC on the validation split, plus its own pruning knee (Section 4.4). Knees "
    "range from 20% (MobileNetV2-1.0x) to 60% (ResNet18), confirming that a single "
    "fixed sparsity target cannot be assumed \"safe\" across architectures -- a point "
    "that matters directly for Section 5.3.")
rows1 = []
for a in ALL9:
    fp32 = MATH[a]["fp32"]
    knee = PARETO["pruning_knees"].get(a)
    rows1.append([ARCH_LABELS[a], pct(fp32["accuracy"]), pct(fp32["sensitivity_recall"]),
                  pct(fp32["specificity"]), pct(fp32["auroc"]),
                  f"{int(knee * 100)}%" if knee is not None else "n/a"])
make_table(["Architecture", "Accuracy", "Sensitivity", "Specificity", "AUROC", "Pruning knee"],
           rows1, [1.6, 0.9, 0.95, 0.95, 0.8, 0.95])
caption("Table 1. FP32 baseline (validation split) and each architecture's own pruning "
        "knee -- the lowest one-shot unstructured-pruning sparsity at which its "
        "validation accuracy first drops more than 2 points below its own 0%-sparsity "
        "value. Knees are architecture-relative, not comparable in absolute terms.")

heading("5.2 Post-training INT8 quantization (x86/fbgemm): an SE/swish association", level=2)
para(
    f"On the x86/fbgemm backend, static INT8 PTQ drops accuracy by a mean of "
    f"{pct(SE_MEAN_DROP)} for the {N_SE_SWISH} squeeze-and-excitation/swish "
    f"architectures (EfficientNet-B0, MobileNetV3-Small, RepGhostNet-0.5x) versus a "
    f"mean of {pct(NON_SE_MEAN_DROP)} for the other {N_NON_SE} architectures -- a "
    f"raw difference of {SE_DIFF_PT * 100:.1f} percentage points, or approximately "
    f"{SE_RATIO:.1f}x. As noted in Section 2.3, this is an association across "
    f"architecture families on this dataset, not an isolated causal effect of SE or "
    f"swish specifically, and no significance test was run on this two-group "
    f"comparison.")
rows2 = []
for a in ALL9:
    fp32 = EXT["per_model"][a]["fp32"]["point"]
    i8 = EXT["per_model"][a]["static_int8"]["point"]
    rows2.append([ARCH_LABELS[a], pct(fp32["accuracy"]), pct(i8["accuracy"]),
                  pt_(i8["accuracy"], fp32["accuracy"]), "Yes" if a in SE_SWISH else "No"])
make_table(["Architecture", "FP32 Acc.", "INT8 Acc. (x86/fbgemm)", "ΔAcc.", "SE/swish?"],
           rows2, [1.6, 0.95, 1.35, 0.9, 0.85])
caption("Table 2. FP32-to-INT8 accuracy change on the x86/fbgemm backend, all 9 "
        "architectures, ranked by architecture family (Section 2.3).")

heading("5.3 Combined pruning (30%) and INT8 quantization", level=2)
para(
    "Table 3 evaluates a fixed 30% one-shot unstructured sparsity target combined with "
    "x86/fbgemm INT8 quantization. A 30% target is at or above the pruning knee "
    "(Table 1) for 7 of 9 architectures, but is exactly at SqueezeNet1.1's own knee "
    "(30%) and, importantly, ABOVE MobileNetV2-1.0x's own knee (20% < 30%) -- so "
    "MobileNetV2-1.0x's row in Table 3 reflects an architecture already past its own "
    "pruning-induced degradation point, not a clean isolation of quantization's "
    "marginal contribution. Consistent with this, MobileNetV2-1.0x drops from "
    f"{pct(EXT['per_model']['mobilenetv2_100']['fp32']['point']['accuracy'])} (FP32) to "
    f"{pct(EXT['per_model']['mobilenetv2_100']['static_int8']['point']['accuracy'])} "
    "with INT8 alone, but to "
    f"{pct(EXT['per_model']['mobilenetv2_100']['pruned30_plus_int8']['point']['accuracy'])} "
    "combined -- a substantially larger drop than INT8 alone accounts for, consistent "
    "with (not proof of) an added pruning-specific contribution once past its own "
    "knee.")
rows3 = []
for a in ALL9:
    fp32 = EXT["per_model"][a]["fp32"]["point"]["accuracy"]
    i8 = EXT["per_model"][a]["static_int8"]["point"]["accuracy"]
    comb = EXT["per_model"][a]["pruned30_plus_int8"]["point"]["accuracy"]
    knee = PARETO["pruning_knees"].get(a)
    flag = "past own knee" if (knee is not None and 0.30 > knee) else ("at own knee" if knee == 0.30 else "below own knee")
    rows3.append([ARCH_LABELS[a], pct(fp32), pct(i8), pct(comb), pt_(comb, fp32), flag])
make_table(["Architecture", "FP32 Acc.", "INT8-alone Acc.", "30%-pruned+INT8 Acc.", "ΔAcc. (total)", "30% vs. own knee"],
           rows3, [1.35, 0.8, 0.95, 1.15, 0.85, 1.0])
caption("Table 3. Combined 30% unstructured pruning + x86/fbgemm INT8, all 9 "
        "architectures, with each row flagged against that architecture's own knee "
        "(Table 1) rather than assumed uniformly safe.")

heading("5.4 Quantization-aware training (QAT)", level=2)
para(
    "QAT was run for the 3 SE/swish architectures on this paper's ARM/qnnpack machine "
    "(3 epochs, fake-quantization during fine-tuning, then true INT8 conversion). "
    "Table 4 compares against qnnpack PTQ (no QAT) on the same backend, since QAT here "
    "was never run on x86/fbgemm. QAT does not recover usable accuracy for any of the "
    "3 architectures on this backend -- all 3 remain within a few points of the "
    "49.8-50% constant-classifier floor.")
rows4 = []
for a in sorted(SE_SWISH):
    fp32 = MATH[a]["fp32"]["accuracy"]
    ptq = MATH[a]["qnnpack"]["accuracy"]
    qat = QAT[a]["val_metrics"]["accuracy"]
    rows4.append([ARCH_LABELS[a], pct(fp32), pct(ptq), pct(qat), pt_(qat, ptq)])
make_table(["Architecture", "FP32 Acc.", "qnnpack PTQ Acc. (no QAT)", "qnnpack QAT Acc.", "ΔAcc. (QAT vs. PTQ)"],
           rows4, [1.6, 0.9, 1.5, 1.05, 1.1])
caption("Table 4. Quantization-aware training vs. plain PTQ, both on qnnpack, for the "
        "3 SE/swish architectures.")

heading("5.5 Backend-dependent INT8 behavior: the paper's central finding", level=2)
para(
    f"With identical trained weights and the same nominal static-quantization "
    f"procedure, INT8 accuracy differs sharply between the x86/fbgemm and ARM/qnnpack "
    f"backends for {N_COLLAPSED + N_DEGRADED} of 9 architectures. Table 5 reports, per "
    f"architecture: FP32 accuracy; fbgemm and qnnpack INT8 accuracy; the backend delta "
    f"(Δ = Acc(fbgemm) − Acc(qnnpack)); qnnpack sensitivity, specificity, and "
    f"balanced accuracy; and this study's collapsed/severely-degraded/backend-stable "
    f"classification (Section 4.8). {N_COLLAPSED} architectures collapse to F1 < 0.05 "
    f"({frac(N_COLLAPSED)}); {N_DEGRADED} more show severely imbalanced sensitivity/"
    f"specificity ({frac(N_DEGRADED)}); {N_STABLE} remain backend-stable "
    f"({frac(N_STABLE)}). For ResNet18 specifically, accuracy is identical to four "
    f"decimal places between backends "
    f"({pct(MATH['resnet18']['fbgemm']['accuracy'], 2)}), yet sensitivity and "
    f"specificity individually shift in opposite directions "
    f"({pt_(MATH['resnet18']['qnnpack']['sensitivity_recall'], MATH['resnet18']['fbgemm']['sensitivity_recall'])} "
    f"sensitivity, "
    f"{pt_(MATH['resnet18']['qnnpack']['specificity'], MATH['resnet18']['fbgemm']['specificity'])} "
    f"specificity) -- a reminder that accuracy alone can hide a backend effect even "
    f"when it is present.")
rows5 = []
for a in ALL9:
    m = MATH[a]
    rows5.append([ARCH_LABELS[a], pct(m["fp32"]["accuracy"]), pct(m["fbgemm"]["accuracy"]),
                  pct(m["qnnpack"]["accuracy"]),
                  pt_(m["fbgemm"]["accuracy"], m["qnnpack"]["accuracy"]),
                  pct(m["qnnpack"]["sensitivity_recall"]), pct(m["qnnpack"]["specificity"]),
                  pct(m["balanced_acc_qnnpack"]), verdict3(m["qnnpack"])])
make_table(["Architecture", "FP32 Acc.", "fbgemm INT8", "qnnpack INT8", "Δ Backend",
            "qnnpack Sens.", "qnnpack Spec.", "qnnpack Bal.Acc.", "Status"],
           rows5, [1.25, 0.75, 0.8, 0.8, 0.75, 0.85, 0.85, 0.85, 1.0], header_fill="F3D9D9")
caption("Table 5. Backend-dependent INT8 behavior, all 9 architectures. \"Status\" uses "
        "this study's own thresholds (Section 4.8): collapsed (F1<0.05), severely "
        "degraded (Sens. or Spec. <20%), or backend-stable.")

heading("5.5.1 What was, and was not, held constant between backends", level=2)
perchan_rows = []
for a in QNNPACK_AFFECTED:
    base_f1 = MATH[a]["qnnpack"]["f1"]
    forced = PERCHAN[a]["point"]
    perchan_rows.append([ARCH_LABELS[a], f"{base_f1:.3f}", f"{forced['accuracy']*100:.1f}%",
                          f"{forced['f1']:.3f}"])
make_table(["Architecture", "qnnpack default F1 (per-tensor)", "Forced per-channel Acc.", "Forced per-channel F1"],
           perchan_rows, [1.6, 1.7, 1.5, 1.4])
caption("Table 5a. Diagnostic: re-quantizing on qnnpack with weight quantization "
        "forced to per-channel (qnnpack's own default is per-tensor; Section 4.2), for "
        "the 5 qnnpack-affected architectures.")
para(
    "As stated in Section 4.2, the fbgemm-vs-qnnpack comparison in Table 5 is not a "
    "controlled ablation of \"the backend\" alone -- the two backends' default recipes "
    "differ in weight-quantization granularity (per-channel vs. per-tensor) and "
    "activation-observer range, and both differences are present simultaneously in "
    "Table 5. The forced-per-channel diagnostic (Table 5a) separates granularity's "
    "contribution: it is associated with most of EfficientNet-B0's recovery (F1 rises "
    f"from {MATH['efficientnet_b0']['qnnpack']['f1']:.3f} to "
    f"{PERCHAN['efficientnet_b0']['point']['f1']:.3f}), a smaller part of MobileNetV3-"
    f"Small's (F1 to {PERCHAN['mobilenetv3_small']['point']['f1']:.3f}), and none of "
    "EfficientNet-Lite0's, MobileNetV2-1.0x's, or RepGhostNet-0.5x's, which remain "
    "fully collapsed (F1 = 0.000) under forced per-channel quantization too. Weight-"
    "quantization granularity is therefore associated with backend-dependent INT8 "
    "behavior for some architectures and not others; it is not a single mechanism that "
    "explains all of Table 5, and this paper does not claim it does.")

heading("5.5.2 A zero-shot check on a second dataset (SkinDisNet) -- not a "
        "generalization validation", level=2)
para(
    "To check whether the collapse signature in Table 5 reproduces outside this "
    "project's own dataset, all 9 architectures were evaluated zero-shot (no "
    "fine-tuning) on SkinDisNet, a separate dermatology image dataset, with Eczema and "
    "Atopic Dermatitis merged into the positive class to match this project's own "
    "binary task definition. INT8 calibration used only this project's own training "
    "split, never SkinDisNet. x86/fbgemm INT8 was not attempted on SkinDisNet -- "
    "infeasible on this machine, not fabricated. Table 5b reports FP32 and qnnpack "
    "INT8 accuracy and F1 on SkinDisNet for all 9 architectures.")
rows5b = []
for a in ALL9:
    v = SDN9["per_model"][a]
    fp = v["fp32"]["point"]; qint8 = v["qnnpack_int8"]["point"]
    rows5b.append([ARCH_LABELS[a], pct(fp["accuracy"]), f"{fp['f1']:.3f}",
                   pct(qint8["accuracy"]), f"{qint8['f1']:.3f}",
                   disp_status(v["qnnpack_int8"]["verdict"])])
make_table(["Architecture", "FP32 Acc. (SkinDisNet)", "FP32 F1", "qnnpack Acc.", "qnnpack F1", "Status"],
           rows5b, [1.4, 1.2, 0.7, 1.0, 0.7, 1.15])
caption("Table 5b. Zero-shot check on SkinDisNet, all 9 architectures. FP32 F1 is well "
        "below this project's own validation-split F1 for every architecture -- a "
        "pre-existing generalization gap that exists before any quantization is "
        "applied.")
para(
    "The 3 architectures that fully collapse on this project's own validation split "
    "(EfficientNet-Lite0, MobileNetV2-1.0x, RepGhostNet-0.5x) reach the same F1 = 0.000 "
    "or near-0.000 signature on SkinDisNet, consistent with (not additional proof "
    "beyond) the Table 5 finding for those architectures. The other 6 architectures "
    "cannot be validated as backend-stable on SkinDisNet specifically, because their "
    "own FP32 zero-shot F1 on SkinDisNet is already low (0.03-0.11) before any "
    "quantization -- a separate, pre-existing distribution-shift limitation that this "
    "single check cannot disentangle from a quantization effect. This check therefore "
    "reproduces the collapse signature for the already-collapsed architectures; it "
    "does not establish, and this paper does not claim, that the backend-stable "
    "architectures generalize to a second dataset.")

heading("5.6 Structured (channel) pruning", level=2)
para(
    "Table 6 reports one comparison point (30% channel sparsity, matching Section "
    "5.3's unstructured-pruning target) for the 7 architectures torch-pruning could "
    "process; both ShuffleNetV2 variants are excluded for the tooling reason given in "
    "Section 4.5. Across the 20%/30%/50% sweep actually run (not fully tabulated here "
    "for space), structured pruning degrades accuracy faster than the equivalent "
    "unstructured sparsity for every one of these 7 architectures, consistent with "
    "removing whole channels being a coarser, less targeted operation than removing "
    "individual weights.")
rows6 = []
for a in ["resnet18", "mobilenetv2_100", "mobilenetv3_small", "squeezenet1_1",
          "efficientnet_lite0", "efficientnet_b0", "repghostnet_050"]:
    fp32 = STRUCT["per_model"][a]["fp32"]["point"]["accuracy"]
    s30 = STRUCT["per_model"][a]["structured_pruning"]["0.3"]["point"]["accuracy"]
    rows6.append([ARCH_LABELS[a], pct(fp32), pct(s30), pt_(s30, fp32)])
make_table(["Architecture", "FP32 Acc.", "30%-structured Acc.", "ΔAcc."],
           rows6, [1.7, 1.0, 1.3, 0.9])
caption("Table 6. Structured (channel) pruning at 30% target, 7 of 9 architectures "
        "(ShuffleNetV2-0.5x and -1.0x excluded -- Section 4.5).")

heading("5.7 Pruning followed by fine-tuning", level=2)
para(
    "Table 7 reports, for 3 architectures at 3 sparsity levels each, accuracy "
    "immediately after pruning (mask applied, no fine-tuning), accuracy after "
    "fine-tuning with the mask held fixed, the accuracy change from fine-tuning, and "
    "the fraction of the pruning-induced drop that fine-tuning recovers (Section 4.8). "
    "Fine-tuning recovers most or all of the pruning-induced drop for ResNet18 and "
    "ShuffleNetV2-1.0x at every sparsity tested, including cases where the "
    "fine-tuned model exceeds its own FP32 baseline (a fraction-recovered value above "
    "100%). MobileNetV3-Small -- one of the 3 SE/swish architectures -- recovers less "
    "consistently, though this experiment does not include quantization and is "
    "therefore separate from the SE/swish PTQ association in Section 5.2.")
rows7 = []
for a in ["resnet18", "mobilenetv3_small", "shufflenet_v2_x1_0"]:
    fp32 = PF["per_model"][a]["fp32_baseline"]["point"]["accuracy"]
    for s in ["0.3", "0.5", "0.7"]:
        lvl = PF["per_model"][a]["sparsity_levels"][s]
        before = lvl["immediately_after_pruning_no_finetune"]["accuracy"]
        after = lvl["after_finetune"]["point"]["accuracy"]
        drop = fp32 - before
        recovered = after - before
        frac_rec = (recovered / drop * 100) if drop > 1e-9 else float("nan")
        frac_str = f"{frac_rec:.0f}%" if drop > 1e-9 else "n/a (no drop)"
        rows7.append([ARCH_LABELS[a], f"{int(float(s)*100)}%", pct(before), pct(after),
                      pt_(after, before), frac_str])
make_table(["Architecture", "Sparsity", "Immediately after pruning", "After fine-tuning",
            "Accuracy change after fine-tuning", "Fraction of drop recovered"],
           rows7, [1.35, 0.65, 1.1, 1.0, 1.15, 1.05])
caption("Table 7. Pruning + fine-tuning, 3 architectures x 3 sparsity levels. "
        "\"Fraction of drop recovered\" = (Acc(after) − Acc(before)) / "
        "(Acc(FP32) − Acc(before)); values above 100% indicate the fine-tuned "
        "model exceeded its own FP32 baseline.")

# ================= 6. ROBUSTNESS =================
heading("6. Robustness to Image Corruption", level=1)
para(
    "This section is exploratory and hypothesis-generating, not a confirmatory "
    "benchmark (Section 4.7); no significance test was run on any ranking below.")
heading("6.1 Corruption ranking (FP32, 8 architectures)", level=2)
ranked = CORR["mean_drop_ranked"]
top2 = ranked[:2]
para(
    f"Averaged over 8 architectures' FP32 models, Gaussian blur "
    f"({top2[0][1]*100:.1f} pt mean accuracy drop) and JPEG compression "
    f"({top2[1][1]*100:.1f} pt) are the two most damaging corruptions tested, each "
    f"roughly 2.5-3x larger than the next-largest ({ranked[2][0].replace('_', ' ')}, "
    f"{ranked[2][1]*100:.1f} pt). Both are corruptions that a wearable/edge camera "
    f"sensor could plausibly introduce (motion blur, lossy on-device compression), "
    f"which is why they are highlighted here -- as a hypothesis worth targeted testing "
    f"in any future work using real sensor-captured images, not as a demonstrated "
    f"deployment risk.")
heading("6.2 Interaction between compression and corruption", level=2)
def mean_corruption_drop(variants_dict):
    out = {}
    for variant, v in variants_dict.items():
        clean = v["clean"]["accuracy"]
        keys = [k for k in v if k != "clean"]
        mean_corr = sum(v[k]["accuracy"] for k in keys) / len(keys)
        out[variant] = (clean, mean_corr, clean - mean_corr)
    return out

sh_x86 = mean_corruption_drop(UC_SHUFFLE_X86["variants"])
sh_qnn = mean_corruption_drop(UC_SHUFFLE_QNN["variants"])
eff_qnn = mean_corruption_drop(UC_EFFB0["variants"])
mbv2_qnn = mean_corruption_drop(UC_MBV2["variants"])
para(
    "For ShuffleNetV2-1.0x (backend-stable on both backends -- Table 5), the mean "
    "accuracy drop under corruption stays within a narrow band across FP32, "
    "pruned-at-knee, INT8-alone, and pruned+INT8 variants, on EITHER backend "
    f"(roughly {min(v[2] for v in sh_x86.values())*100:.1f}-"
    f"{max(v[2] for v in sh_x86.values())*100:.1f} pt on x86/fbgemm, "
    f"{min(v[2] for v in sh_qnn.values())*100:.1f}-"
    f"{max(v[2] for v in sh_qnn.values())*100:.1f} pt on qnnpack) -- consistent with "
    "(not proof of) compression not meaningfully changing this architecture's relative "
    "corruption sensitivity. For EfficientNet-B0 and MobileNetV2-1.0x on qnnpack, the "
    "same mean-drop calculation collapses toward zero once INT8 is applied "
    f"({eff_qnn['static_int8'][2]*100:+.1f} pt, "
    f"{mbv2_qnn['static_int8'][2]*100:+.1f} pt) -- but this is an artifact of Table "
    "5's backend collapse, not evidence of corruption robustness: once a model is "
    "already a near-constant classifier at its collapsed accuracy, corrupting the "
    "input cannot make its output much worse, because it was not responding to the "
    "clean input's content in the first place. Reading a near-zero corruption drop as "
    "\"robustness\" for a collapsed model would be a misinterpretation of Table 5's "
    "own finding.")

# ================= 7. DISCUSSION =================
heading("7. Discussion", level=1)
heading("7.1 Hierarchy of findings", level=2)
para(
    "PRIMARY: backend-dependent INT8 behavior (Section 5.5) -- with identical trained "
    f"weights, {N_COLLAPSED + N_DEGRADED} of 9 architectures behave severely worse on "
    "ARM/qnnpack than on x86/fbgemm, scoped to these 9 architectures and this "
    "machine's default recipes, not a general ARM claim.")
bullet("SECONDARY: SE/swish architectures show a larger FP32-to-INT8 accuracy drop on "
       "x86/fbgemm (Section 5.2) -- an architecture-family association, not an "
       "isolated causal mechanism.")
bullet("SECONDARY: each architecture has its own pruning knee (Table 1); a single "
       "fixed sparsity target is not uniformly safe across architectures (Section "
       "5.3).")
bullet("SECONDARY: fine-tuning after pruning recovers most or all of the "
       "pruning-induced drop for 2 of 3 architectures tested, including cases "
       "exceeding the FP32 baseline (Table 7).")
bullet("SECONDARY: weight-quantization granularity (per-channel vs. per-tensor) "
       "explains most of one architecture's backend gap, part of a second's, and none "
       "of three others' (Section 5.5.1) -- backend defaults are a partial, "
       "architecture-specific contributor to the PRIMARY finding, not a full "
       "explanation of it.")
bullet("SUPPORTING: Gaussian blur and JPEG compression are the most damaging tested "
       "corruptions on average (Section 6.1); the compression-corruption interaction "
       "is exploratory (Section 6.2).")
bullet("DEPLOYMENT: model size, parameter count, and CPU inference time are measured "
       "on development hardware, not a wearable/edge target, and are reported as "
       "theoretical proxies only (Section 4.9).")
heading("7.2 A feasible candidate set for this deployment objective", level=2)
para(
    f"Restricting to architectures that are backend-stable on qnnpack (Table 5, "
    f"{N_STABLE} of 9: ResNet18, ShuffleNetV2-0.5x, ShuffleNetV2-1.0x, SqueezeNet1.1) "
    "and cross-referencing size and accuracy (Table 1, Table 5) gives a feasible "
    "candidate set for a wearable/edge deployment target using this project's own "
    "quantized-model requirement: the 3 lightweight, backend-stable architectures "
    "(excluding the ResNet18 baseline, which is neither the smallest nor the most "
    "accurate of the four). This is a candidate set, not a single recommendation -- "
    "Section 7.3 shows why a single winner is not declared within it.")
heading("7.3 ShuffleNetV2-0.5x vs. 1.0x: a Pareto non-dominance", level=2)
sh5 = MATH["shufflenet_v2_x0_5"]["qnnpack"]; sh5m = min(sh5["sensitivity_recall"], sh5["specificity"])
sh10 = MATH["shufflenet_v2_x1_0"]["qnnpack"]; sh10m = min(sh10["sensitivity_recall"], sh10["specificity"])
sz5 = SIM["shufflenet_v2_x0_5"]["pruning_sweep"][0]["state_dict_disk_mb_dense"]
sz10 = SIM["shufflenet_v2_x1_0"]["pruning_sweep"][0]["state_dict_disk_mb_dense"]
make_table(["Metric (qnnpack INT8)", "ShuffleNetV2-0.5x", "ShuffleNetV2-1.0x", "Which wins"],
           [["Accuracy", pct(sh5["accuracy"]), pct(sh10["accuracy"]), "1.0x"],
            ["Sensitivity", pct(sh5["sensitivity_recall"]), pct(sh10["sensitivity_recall"]), "1.0x"],
            ["Specificity", pct(sh5["specificity"]), pct(sh10["specificity"]), "0.5x"],
            ["Margin M = min(Sens., Spec.)", pct(sh5m), pct(sh10m), "0.5x"],
            ["Dense checkpoint size", f"{sz5:.2f} MB", f"{sz10:.2f} MB", "0.5x"]],
           [1.9, 1.35, 1.35, 1.0])
caption("Table 8. ShuffleNetV2-0.5x vs. 1.0x on qnnpack: neither Pareto-dominates the "
        "other under this study's own metrics.")
para(
    "1.0x wins on accuracy and sensitivity; 0.5x wins on specificity, worst-case "
    "margin, and size (roughly 2.8x smaller). Neither vector weakly dominates the "
    "other on every metric, so this is a formal Pareto non-dominance, not an "
    "unresolved tie broken by preference alone. Resolving it requires an explicit "
    "weighting of sensitivity vs. specificity for THIS task -- and that weighting "
    "does not transfer from unrelated literature. Sensitivity-first framings common in "
    "skin-cancer screening (e.g. [7]) assume a screening task, eczema-labeled vs. "
    "healthy skin, where a missed positive risks a missed malignancy. This task is "
    "different: eczema vs. OTHER SKIN CONDITIONS that can look similar, several of "
    "which (e.g. tinea corporis) can be worsened by a treatment appropriate for "
    "eczema but not for them -- a phenomenon documented in dermatology as \"tinea "
    "incognito,\" where topical corticosteroids used to treat presumed eczema mask or "
    "worsen an underlying fungal infection [21]. A false positive (predicting eczema "
    "for a different condition) is not obviously lower-cost than a false negative "
    "here, so a sensitivity-priority argument imported from screening literature does "
    "not transfer to this differential-diagnosis-shaped task, and this paper does not "
    "make it. Without a task-specific cost model for this differential (which this "
    "paper does not have), Table 8's non-dominance is left unresolved rather than "
    "forced.")
heading("7.3.1 A parametric consideration: size as a tie-breaker", level=2)
para(
    "One narrower argument that does transfer: if the deployment target is memory- or "
    "power-constrained enough that a 2.8x size difference (Table 8) matters "
    "operationally -- plausible for a flexible-electronics wearable sensor, where "
    "on-board memory is a documented hard constraint in the microcontroller-ML "
    "literature [22] -- then 0.5x's size advantage is a legitimate, narrowly-scoped "
    "reason to prefer it over 1.0x specifically for that constrained target, "
    "independent of the sensitivity/specificity question above. This is offered as one "
    "possible resolution under a stated hardware assumption, not as this paper's "
    "general recommendation.")
heading("7.4 Theoretical vs. measured deployment characteristics", level=2)
para(
    "Every size, parameter-count, and latency number in this paper is measured on "
    "development hardware or computed analytically (Section 4.9), not measured on a "
    "wearable or embedded target. Real on-device behavior depends on factors this "
    "paper does not test: the target CPU's actual quantized-kernel support, memory "
    "bandwidth, thermal throttling, and power budget. The candidate set in Section 7.2 "
    "should be read as a theoretically-motivated shortlist for further, hardware-in-"
    "the-loop testing, not as a validated deployment recommendation.")

# ================= 8. LIMITATIONS =================
heading("8. Limitations", level=1)
bullet("No clinical validity, diagnostic accuracy, or regulatory claim is made for any "
       "model in this paper (Section 1.2).")
bullet("The backend-dependent collapse (Section 5.5) is characterized on one ARM "
       "machine's qnnpack build; no second ARM device or PyTorch build was tested, so "
       "this paper cannot say how far the finding generalizes across ARM hardware or "
       "PyTorch versions.")
bullet("The SE/swish PTQ association (Section 5.2) and the granularity diagnostic "
       "(Section 5.5.1) are correlational; the ablation that could more directly test "
       "causation is confounded by collapsing FP32 accuracy on its own (Section 2.3).")
bullet("No parametric significance test was run anywhere in this paper (bootstrap CIs "
       "only); group comparisons (SE/swish, corruption ranking) are descriptive.")
bullet("The SkinDisNet check (Section 5.5.2) is zero-shot and confounded by a "
       "pre-existing FP32 distribution-shift gap; it cannot establish external "
       "generalization for the 6 non-collapsed architectures.")
bullet("Corruption severities and types (Section 4.7) are synthetic transformations "
       "of curated images, not captures from a real camera sensor or wearable device.")
bullet("Structured pruning could not be evaluated for ShuffleNetV2-0.5x or -1.0x "
       "(Section 4.5), so Table 8's comparison excludes structured-pruning behavior "
       "entirely.")
bullet("All deployment-characteristic numbers (size, parameters, latency) are "
       "theoretical proxies, not measurements on a wearable or edge target (Section "
       "7.4).")

# ================= 9. FINAL HELD-OUT TEST EVALUATION =================
heading("9. Final Held-Out Test Evaluation", level=1)
heading("9.1 Protocol", level=2)
para(
    "FROZEN_TEST_PROTOCOL_2026-09-20.md specifies exactly the conditions evaluated on "
    "the test split (507 images, opened for the first time in this compression study "
    "only after that document was written): FP32 and qnnpack INT8, all 9 "
    "architectures, INT8 calibrated on the training split only. x86/fbgemm INT8 on "
    "test was not attempted -- the same infeasibility as Section 5.5.2, not a "
    "fabricated omission.")
heading("9.2 Results", level=2)
rows9 = []
for a in ALL9:
    v = TEST["per_model"][a]
    fp = v["fp32"]["point"]; qint8 = v["qnnpack_int8"]["point"]
    rows9.append([ARCH_LABELS[a], pct(fp["accuracy"]), pct(qint8["accuracy"]),
                  f"{qint8['f1']:.3f}", verdict3(qint8)])
make_table(["Architecture", "FP32 Acc. (test)", "qnnpack Acc. (test)", "qnnpack F1 (test)", "Status (test)"],
           rows9, [1.6, 1.15, 1.15, 1.0, 1.15])
caption("Table 9. Final held-out test-split evaluation, all 9 architectures, "
        "evaluated once under the protocol in Section 9.1.")
heading("9.3 Relationship to the validation-split findings", level=2)
val_status = {a: verdict3(MATH[a]["qnnpack"]) for a in ALL9}
test_status = {a: verdict3(TEST["per_model"][a]["qnnpack_int8"]["point"]) for a in ALL9}
n_match = sum(1 for a in ALL9 if val_status[a] == test_status[a])
para(
    f"The collapsed/severely-degraded/backend-stable status (Section 4.8) matches "
    f"between validation (Table 5) and test (Table 9) for {n_match} of 9 "
    f"architectures. This reproduction is reported as support for the PRIMARY finding "
    f"(Section 7.1) specifically -- backend-dependent INT8 collapse -- not as "
    f"validation of every secondary or exploratory claim in this paper, several of "
    f"which (SE/swish causation, corruption-compression interaction, SkinDisNet "
    f"generalization) were explicitly not re-tested on this split and remain "
    f"correlational or exploratory as stated in Sections 2.3, 5.5.1, 5.5.2, and 6.")

# ================= 10. CONCLUSION =================
heading("10. Conclusion", level=1)
para(
    f"Across the 9 architectures tested in this paper, INT8 quantization behavior on "
    f"the ARM/qnnpack backend is strongly associated with architecture family -- "
    f"{frac(N_COLLAPSED + N_DEGRADED)} degrade severely or collapse outright on qnnpack "
    f"despite behaving normally on x86/fbgemm with the same trained weights -- but this "
    f"study does not isolate a single causal mechanism for that association. A forced-"
    f"granularity diagnostic (Section 5.5.1) accounts for most of one architecture's "
    f"gap, part of a second's, and none of three others', so backend default-recipe "
    f"differences are a partial, architecture-specific contributor, not a complete "
    f"explanation, and the SE/swish PTQ pattern (Section 5.2) remains a correlational "
    f"observation across architecture families rather than a demonstrated causal "
    f"effect of any single component.")
para(
    "For deployment, this means: a candidate set of backend-stable, appropriately-"
    "sized architectures can be identified for this task on this machine's default "
    f"recipes ({N_STABLE} of 9 -- Section 7.2), but no single architecture is declared "
    "the winner within that set, because ShuffleNetV2-0.5x and -1.0x are formally "
    "Pareto non-dominant under this study's own sensitivity/specificity/size metrics "
    "(Section 7.3), and this task's differential-diagnosis shape means a sensitivity-"
    "priority tie-break borrowed from screening literature does not transfer. This "
    "study did NOT determine: whether the backend-collapse finding generalizes beyond "
    "this one ARM machine and PyTorch build; whether SE/swish components are causally "
    "responsible for their architectures' quantization fragility, as opposed to "
    "merely correlated with it; whether any of the 9 models generalize to images "
    "outside this project's own dataset (the SkinDisNet check is inconclusive for 6 of "
    "9 architectures -- Section 5.5.2); or any measure of real-world diagnostic "
    "performance, clinical utility, or on-device latency and power draw. Every finding "
    "in this paper is scoped to the 9 architectures, the qnnpack and fbgemm default "
    "recipes on the two machines used, and this project's own curated dataset -- not "
    "to lightweight CNNs, ARM devices, or PyTorch quantization in general.")

# ================= CLAIMS AUDIT =================
heading("Claims Audit", level=1)
para(
    "This section states, in one place, which claims in this paper are strongly "
    "supported by this paper's own measurements and which are limited, correlational, "
    "or exploratory -- consolidating what earlier drafts of this manuscript tracked "
    "as a multi-part revision history into a single current statement.")
heading("Strongly supported by this paper's own measurements", level=2)
bullet("Backend-dependent INT8 collapse for 5 of 9 architectures on qnnpack (Table 5), "
       "reproduced on a final, previously untouched held-out test split for the "
       "collapsed/severely-degraded/backend-stable status of most architectures "
       "(Table 9, Section 9.3).")
bullet("Architecture-specific pruning knees (Table 1) and the risk of applying a "
       "single fixed sparsity target across architectures (Table 3).")
bullet("Fine-tuning after pruning recovering most or all of the pruning-induced drop "
       "for 2 of 3 architectures tested (Table 7).")
bullet("Formal Pareto non-dominance between ShuffleNetV2-0.5x and -1.0x under this "
       "study's own sensitivity/specificity/size metrics (Table 8) -- a mathematical "
       "result, not a judgment call.")
bullet("Weight-quantization granularity accounting for most of one architecture's "
       "backend gap and part of a second's, while not explaining three others' "
       "(Table 5a) -- i.e., that backend defaults are a partial, not universal, "
       "contributor to the PRIMARY finding.")
heading("Limited, correlational, exploratory, or unresolved", level=2)
bullet("The SE/swish PTQ association (Section 5.2): a real, measured pattern across "
       "architecture families, but not demonstrated as a causal effect of SE or swish "
       "specifically -- the direct ablation test is confounded (Section 2.3), and no "
       "significance test was run on the two-group comparison.")
bullet("The SkinDisNet zero-shot check (Section 5.5.2): reproduces the collapse "
       "signature for the 3 already-collapsed architectures, but is inconclusive for "
       "the other 6 because of a pre-existing FP32 distribution-shift gap -- it does "
       "not establish external generalization for this paper's models generally.")
bullet("The corruption ranking (Section 6.1) and the compression-corruption "
       "interaction (Section 6.2) are exploratory and hypothesis-generating; the "
       "near-zero corruption drop for backend-collapsed architectures is an artifact "
       "of those models already being near-constant classifiers, not a robustness "
       "finding, and this paper states that explicitly rather than reporting the raw "
       "number without context.")
bullet("Whether the backend-collapse finding generalizes beyond this one ARM machine "
       "and PyTorch build (2.8.0, qnnpack) was not tested and is not claimed.")
bullet("No clinical validity, diagnostic accuracy, differential-diagnosis performance, "
       "or on-device (as opposed to development-hardware) latency/power claim is made "
       "anywhere in this paper (Sections 1.2, 4.9, 7.4, 8).")
bullet("Superlative novelty language (\"first,\" \"novel,\" \"previously unreported\") "
       "is avoided throughout this paper for the backend-dependence finding "
       "specifically, because backend-level fbgemm/qnnpack accuracy discrepancies are "
       "already documented at the PyTorch engineering level [19,20]; this paper's "
       "contribution is demonstrating the severity of that discrepancy (complete "
       "classifier collapse) on an applied clinical-image task across 9 architectures, "
       "not discovering that backends can differ.")

# ================= REFERENCES =================
heading("References", level=1)
refs = [
    "[1] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). "
    "MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR 2018.",
    "[2] Howard, A., Sandler, M., Chu, G., et al. (2019). Searching for MobileNetV3. "
    "ICCV 2019.",
    "[3] Ma, N., Zhang, X., Zheng, H.-T., & Sun, J. (2018). ShuffleNet V2: Practical "
    "Guidelines for Efficient CNN Architecture Design. ECCV 2018.",
    "[4] Iandola, F. N., Moskewicz, M. W., Ashraf, K., Han, S., Dally, W. J., & "
    "Keutzer, K. (2016). SqueezeNet: AlexNet-level accuracy with 50x fewer "
    "parameters and <0.5MB model size. arXiv:1602.07360.",
    "[5] Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for "
    "Convolutional Neural Networks. ICML 2019.",
    "[6] Chen, C., Guo, Z., Zeng, H., Xiong, P., & Dong, J. (2022). RepGhost: A "
    "Hardware-Efficient Ghost Module via Re-parameterization. arXiv:2211.06088.",
    "[7] Esteva, A., Kuprel, B., Novoa, R. A., et al. (2017). Dermatologist-level "
    "classification of skin cancer with deep neural networks. Nature, 542, 115-118.",
    "[8] Padilla, D. A., Yumang, A. N., et al. (2020). Differentiating Atopic "
    "Dermatitis and Psoriasis Chronic Plaque using Convolutional Neural Network "
    "MobileNet Architecture. IEEE conference publication.",
    "[9] Juwairi, K. P., Fudholi, D. H., Arifin, A., & Muhimmah, I. (2023). An "
    "EfficientNet-based mobile model for classifying eczema and acne. AIP "
    "Conference Proceedings, 2508(1), 020041.",
    "[10] Safi, I., Safi, F. M., & Pashay, R. (2025). Systematic Review of Deep "
    "Learning and Machine Learning Models for Eczema (Atopic Dermatitis) Image "
    "Detection. Preprint (ResearchGate); covers studies published 2020-2025.",
    "[11] Han, S., Pool, J., Tran, J., & Dally, W. J. (2015). Learning both Weights "
    "and Connections for Efficient Neural Networks. NeurIPS 2015.",
    "[12] Li, H., Kadav, A., Durdanovic, I., Samet, H., & Graf, H. P. (2017). "
    "Pruning Filters for Efficient ConvNets. ICLR 2017.",
    "[13] Fang, G., Ma, X., Song, M., Mi, M. B., & Wang, X. (2023). DepGraph: "
    "Towards Any Structural Pruning. CVPR 2023. (torch-pruning library.)",
    "[14] Jacob, B., Kligys, S., Chen, B., et al. (2018). Quantization and Training "
    "of Neural Networks for Efficient Integer-Arithmetic-Only Inference. CVPR 2018.",
    "[15] Krishnamoorthi, R. (2018). Quantizing deep convolutional networks for "
    "efficient inference: A whitepaper. arXiv:1806.08342.",
    "[16] PyTorch. FX Graph Mode Quantization / torch.ao.quantization documentation "
    "(pytorch.org). Official framework documentation, accessed 2026.",
    "[17] Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks. "
    "CVPR 2018.",
    "[18] Ramachandran, P., Zoph, B., & Le, Q. V. (2017). Searching for Activation "
    "Functions. arXiv:1710.05941.",
    "[19] PyTorch GitHub Issue #44939: \"Inconsistent result from FBGEMM and "
    "QNNPACK quantization backends.\" github.com/pytorch/pytorch/issues/44939.",
    "[20] PyTorch Forums: \"Understanding differences in the default qconfig for "
    "fbgemm and qnnpack.\" discuss.pytorch.org.",
    "[21] Ive, F. A., & Marks, R. (1968). Tinea Incognito. British Medical Journal, "
    "3(5611), 149-152.",
    "[22] Banbury, C., Zhou, C., Fedorov, I., Matas, R., Thakker, U., Gope, D., "
    "Reddi, V. J., Mattina, M., & Whatmough, P. (2021). MicroNets: Neural Network "
    "Architectures for Deploying TinyML Applications on Commodity Microcontrollers. "
    "MLSys 2021.",
    "[23] Internal project report: pre-compression baseline and original x86/fbgemm "
    "PTQ/pruning simulation results for these 9 architectures on this project's own "
    "curated dataset, reused (not recomputed) as this paper's FP32 and fbgemm "
    "reference points (Section 4.10).",
]
for r in refs:
    para(r, size=9.5, space_after=4)

OUT_PATH = PAPER_DIR / "eczema_compression_deployment_paper_2026-09-20.docx"
doc.save(OUT_PATH)
print(f"Saved: {OUT_PATH}")
print(f"Word count (approx.): {sum(len(p.text.split()) for p in doc.paragraphs)}")
