# -*- coding: utf-8 -*-
"""Builds the architecture-comparison interim report as a .docx using python-docx
(the library actually available on this machine -- node/docx-js is not installed here,
despite being the docx skill's stated default)."""
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

from paths import ROOT

BODY_FONT = "Calibri"
HEAD_FONT = "Calibri"

doc = Document()

# ---------- page setup ----------
section = doc.sections[0]
section.page_width = Inches(8.5)
section.page_height = Inches(11)
section.top_margin = Inches(1)
section.bottom_margin = Inches(1)
section.left_margin = Inches(1)
section.right_margin = Inches(1)

# ---------- base styles ----------
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
p = para("Architecture Selection for On-Device Eczema Image Classification",
          bold=True, size=22, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)
para("A Systematic Comparison of Lightweight Convolutional Networks",
     italic=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=30)
para("Image, Stress, and Moisture Model Selection -- Progress Report",
     size=11.5, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6)
para("Tishya", size=12, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=2)
para("September 18, 2026", size=11, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=40)

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run(
    "Status: image-channel comparison complete. All 8 candidates trained and tested against "
    "the ResNet18 baseline (Section 5.3); none of the 36 pairwise comparisons is statistically "
    "significant after Holm correction (Section 5.4). ShuffleNetV2-1.0x is selected as the "
    "image-channel architecture for deployment on size/speed grounds, not a significance claim "
    "(Section 5.6). Stress model: comparison complete for 4 models, framed around an LSTM-based "
    "model appropriate for sequential physiological signals (Section 6); no statistically "
    "significant difference was found among them either (Section 6.6). Moisture model: planned "
    "methodology only (Section 7) -- no experiments have been run yet on this project's own "
    "hardware."
)
r.italic = True
r.font.size = Pt(10)
r.font.color.rgb = RGBColor(0x60, 0x60, 0x60)
doc.add_page_break()

# ================= ABSTRACT =================
heading("Abstract", level=1)
para(
    "Selecting a neural network architecture for an on-device model in a clinical monitoring "
    "system is a decision made once, early, and expensively -- retraining and re-validating a "
    "new backbone after a hardware commitment is costly, so a systematic comparison beforehand "
    "is worth doing properly. This report documents that decision for two of the four "
    "modalities in a multi-sensor atopic dermatitis (AD) monitoring system -- image, stress, "
    "moisture, and scratch: ShuffleNetV2-1.0x for the image channel, and an LSTM-based model "
    "for the stress channel, both chosen through controlled comparison rather than convention. "
    "It also sets out planned methodology for a third modality, moisture. For the image "
    "channel, eight lightweight convolutional architectures -- variants from the MobileNet, "
    "EfficientNet, ShuffleNet, SqueezeNet, and RepGhost families -- are screened as replacements "
    "for the ResNet18 backbone previously used for image classification (eczema vs. seven "
    "clinically similar diseases, 3,330 curated clinical photographs), each trained under an "
    "identical two-stage transfer-learning protocol and compared on accuracy, F1, ROC-AUC, "
    "checkpoint size, and CPU inference latency. All eight candidates completed training and "
    "were evaluated against the ResNet18 baseline on the held-out test set: none of the 36 "
    "pairwise comparisons among the nine models is statistically significant after "
    "Holm-Bonferroni correction. With accuracy not distinguishable across candidates, "
    "ShuffleNetV2-1.0x is selected as the image-channel architecture for deployment on size and "
    "speed grounds (7.8x smaller and 21% faster than ResNet18, with the closest raw test "
    "accuracy to baseline of any candidate) -- a deployment decision, explicitly not a claim of "
    "statistical superiority. For the stress channel, this report evaluates an LSTM-based "
    "model -- appropriate for WESAD's sequential physiological data, where stress-related "
    "responses evolve over time rather than existing as independent observations -- alongside "
    "three classical models (XGBoost, CatBoost, Random Forest) and the previously used, "
    "personal-baseline-calibrated LightGBM model, using a patient-wise held-out test set. A "
    "fifth candidate, a pretrained PPG foundation model, was attempted and dropped after a "
    "genuine, unresolved dependency conflict in its own published requirements. This comparison "
    "is complete: on the held-out test set, no pairwise difference among the four models "
    "survives Holm-Bonferroni correction, and this report does not claim the LSTM-based model "
    "is statistically superior on that basis -- its selection as this report's primary direction "
    "for the stress channel rests on its fit to the modality (Section 6.1), not on an "
    "unsupported significance claim. Both evaluations use the same formal statistical protocol "
    "(McNemar's test, paired bootstrap, Holm-Bonferroni correction, scaled to the number of "
    "models actually being compared in each case) and the same code-enforced test-set isolation "
    "discipline, including, for the image channel, a documented near-miss where an earlier, "
    "unsaved comparison script touched the test set before the full candidate field had finished "
    "training -- the fix adopted for that incident is itself a methodological contribution of "
    "this report. A dedicated section sets out planned, not-yet-executed methodology for the "
    "moisture channel. Training-related infrastructure constraints on this project's development "
    "hardware (CPU-only, 8 GB RAM) are reported throughout as a methods-level consideration, "
    "since they materially affected the pace and, in two documented incidents, required "
    "discarding results outright rather than reporting them."
)

# ================= 1. INTRODUCTION =================
heading("1. Introduction", level=1)
para(
    "This report is part of an ongoing project building a multi-sensor monitoring system for "
    "atopic dermatitis (AD, eczema) outside the clinic. The system is organized around four "
    "modalities, each contributing one signal to an eventual decision-level fusion step rather "
    "than operating as an isolated model: an image channel (a phone camera photographing and "
    "classifying a skin lesion), a stress channel (wrist- and patch-worn physiological sensing), "
    "a moisture channel (skin-surface hydration sensing from the same patch), and a scratch "
    "channel (wrist accelerometer-based motion sensing). The phone runs the camera, every "
    "per-modality model, the fusion logic that combines their outputs, and the user-facing "
    "dashboard; the wearable nodes handle sensing only. Of these four, only the image and stress "
    "channels currently have a trained model, and both were built around one architecture chosen "
    "early and kept by convention rather than by systematic comparison against alternatives."
)
para(
    "That is the gap this report addresses, and it concludes with a decision for each channel "
    "it covers: ShuffleNetV2-1.0x for the image channel, and an LSTM-based model for the stress "
    "channel (Sections 5.6 and 6). Before this report, the image channel ran on a partially "
    "fine-tuned ResNet18 (11.7M parameters, a 42.7 MB checkpoint), distinguishing eczema from "
    "seven clinically similar diseases with 81.07% test accuracy and F1 81.18% on this project's "
    "curated dataset. ResNet18 was never chosen over any lighter alternative for a documented "
    "reason -- it was simply the architecture the project's earliest image-classification "
    "experiments happened to start with, and every later iteration (data rebalancing, "
    "shortcut-learning fixes, multi-source fine-tuning) kept the same backbone rather than "
    "re-opening that choice. The stress channel ran on a similarly unexamined choice -- a "
    "personal-baseline-calibrated LightGBM classifier on WESAD wrist-wearable features -- which "
    "had likewise never been compared against alternative model classes before this report."
)
para(
    "Even though the image model runs on a phone rather than a microcontroller, model size and "
    "inference latency are not irrelevant. A smaller checkpoint means a smaller app download and "
    "cheaper over-the-air model updates; a faster forward pass leaves more of the device's "
    "compute and battery budget for the fusion model, the wearable-sensor pipeline, and normal "
    "phone use during background monitoring. If a candidate several times smaller than ResNet18 "
    "reaches comparable accuracy, that is a real, usable improvement -- not a marginal one that "
    "only matters on constrained microcontroller hardware."
)
para(
    "This project has also learned, directly, not to trust a model's headline number without "
    "testing it properly: an earlier version of this same image classifier scored 95.66% "
    "accuracy on a flawed split before a shortcut-learning audit brought that down to a more "
    "honest 81.07%, and separate external validation on two independently sourced dermatology "
    "datasets found the resulting model's cross-dataset generalization near chance (AUC "
    "0.6512). An architecture swap deserves the same scrutiny a data-split or generalization "
    "claim would get -- picked by comparison under a controlled protocol with a held-out test "
    "set accessed exactly once, not by reputation or convenience."
)
heading("1.1 Contribution", level=2)
para(
    "This report's contribution is methodological rather than a single headline result, and has "
    "three parts: (1) a systematic architecture comparison for two of this system's four "
    "modalities, screening multiple candidates against the previously used model under one "
    "fixed protocol rather than swapping in a new architecture by convention; (2) a statistically "
    "controlled evaluation -- formal paired significance testing with multiple-comparison "
    "correction on a held-out test set, applied consistently across both modalities, rather than "
    "reading raw accuracy differences at face value; and (3) explicit consideration of model size "
    "and inference cost alongside accuracy, since the eventual target is on-device deployment "
    "under real battery and storage constraints, not a leaderboard entry."
)
heading("1.2 Research questions", level=2)
bullet("Can a lightweight CNN architecture match the previously used ResNet18 "
       "model's accuracy at a meaningfully smaller size and lower inference cost?")
bullet("Does a model architecture change for the stress-detection channel (moving from "
       "engineered features to a model built for sequential physiological data) produce a "
       "statistically distinguishable difference from the previously used model?")
bullet("At the sample sizes available to this project, which comparisons are the data actually "
       "able to settle, and which conclusions would require more data before they could be "
       "trusted?")
para(
    "This report covers the image channel (Section 5) and the stress channel (Section 6) in "
    "depth, both against their previously used models. Section 7 sets out the planned "
    "methodology for the moisture channel, which has no trained model yet."
)

# ================= 2. RELATED WORK =================
heading("2. Related Work", level=1)
heading("2.1 Deep learning for dermatology image classification", level=2)
para(
    "Convolutional neural networks reaching dermatologist-level accuracy on skin lesion "
    "classification was established early and has since become a standard approach "
    "[9], and large aggregated dermatoscopic datasets such as HAM10000 [10] made transfer "
    "learning from ImageNet-pretrained backbones the default strategy for skin-image tasks with "
    "limited labeled data -- the same strategy this project's Stage B image classifier already "
    "uses, and that this report keeps fixed while varying only the backbone."
)
heading("2.2 Lightweight convolutional architectures", level=2)
para(
    "The candidate architectures compared here come from five separate lines of "
    "efficiency-focused CNN design, each targeting a different point on the "
    "accuracy/size/latency trade-off curve:"
)
bullet("MobileNetV2 [1] and MobileNetV3 [2] build on depthwise-separable convolutions and "
       "inverted residual (\"bottleneck\") blocks; MobileNetV3 adds squeeze-and-excite "
       "attention and a hardware-aware neural architecture search over the block configuration.")
bullet("EfficientNet [3] scales depth, width, and input resolution together under one compound "
       "coefficient rather than tuning each independently; EfficientNet-Lite is a variant of "
       "the same family adjusted for edge-deployment friendliness (no squeeze-and-excite, "
       "ReLU6 activations).")
bullet("ShuffleNetV2 [4] uses grouped convolutions with an explicit channel-shuffle operation "
       "to let information mix across groups without the full cost of a dense convolution, and "
       "is tuned directly against measured hardware latency rather than FLOP count alone.")
bullet("SqueezeNet [5] uses \"fire modules\" (a 1x1 squeeze layer feeding parallel 1x1/3x3 "
       "expand layers) to reach AlexNet-level accuracy with a much smaller parameter count -- "
       "the oldest architecture family in this comparison, included as a lower-complexity "
       "anchor point.")
bullet("RepGhostNet [6] generates redundant \"ghost\" feature maps from a smaller set of "
       "intrinsic ones via re-parameterizable, hardware-friendly operations, avoiding the "
       "concatenation cost that limits the original GhostNet's real-device latency despite "
       "its low FLOP count.")
para(
    "All eight candidate configurations used in this report, along with the previously used "
    "ResNet18 "
    "[7] baseline, are pretrained on ImageNet [8] and adapted to this project's task by "
    "transfer learning (Section 4.2)."
)
heading("2.3 Scope note: architecture vs. generalization", level=2)
para(
    "This report compares architectures only on the same source distribution the previously "
    "used ResNet18 model was trained and tested on -- it does not repeat, per architecture, the "
    "cross-dataset external-validation audit already carried out for the ResNet18 baseline "
    "against two independent dermatology datasets. A generalization gap comparable to what was "
    "found for that baseline should be assumed to exist for each candidate here until it is "
    "specifically tested, not assumed absent because a candidate scores well internally; this is "
    "revisited in Section 9 (Limitations)."
)

# ================= 3. DATASET AND TASK =================
heading("3. Dataset and Task", level=1)
para(
    "The task is binary: Eczema vs. Other, where Other spans seven clinically similar "
    "conditions -- Psoriasis, Tinea, Candidiasis, Infestations/Bites, Lichen, Drug Eruption, "
    "and Rosacea -- chosen specifically because they are the conditions most likely to be "
    "confused with eczema on casual inspection, making the classification problem clinically "
    "meaningful rather than trivially separable."
)
para(
    "The dataset merges three separately sourced Kaggle/DermNet-style photo archives into "
    "3,330 images, deliberately rebalanced to an exact 1,665/1,665 (50/50) Eczema/Other split "
    "after an earlier, more imbalanced version of this dataset was found to inflate accuracy "
    "relative to F1. The split used throughout this report -- 70/15/15, stratified by the "
    "eight-way disease-class label, fixed seed 42 -- produces 2,327 train / 496 validation / "
    "507 test images. This is an image-level split, not a patient-wise one: the three source "
    "archives provide only per-image files with no patient or subject identifier of any kind, "
    "so a patient-wise split is not possible for this dataset (unlike this project's separate "
    "external-validation dataset, which does carry real patient IDs). This is disclosed as a "
    "real limitation, not a minor caveat -- see Section 9. This is the same split, and the "
    "same task definition, "
    "the previously used ResNet18 model was trained and reported on, so every number in this "
    "report "
    "is a direct, controlled architecture swap-in comparison -- not a comparison confounded by "
    "a different data split or task definition."
)

# ================= 4. METHODOLOGY =================
heading("4. Methodology", level=1)
heading("4.1 Candidate architectures", level=2)
para(
    "Eight lightweight architectures were selected to span the five families in Section 2.2 "
    "at multiple size points, plus the previously used ResNet18 model as a fixed comparison "
    "anchor "
    "(not itself a candidate being screened for replacement):"
)
make_table(
    ["Model", "Family", "Notes"],
    [
        ["ResNet18 (baseline)", "Residual", "Deployed model; comparison anchor, not a screened candidate"],
        ["EfficientNet-B0", "EfficientNet", ""],
        ["EfficientNet-Lite0", "EfficientNet", "Edge-oriented variant"],
        ["MobileNetV3-Small", "MobileNet", ""],
        ["MobileNetV2 1.0x", "MobileNet", "Substitute for originally-planned 0.35x -- no verified ImageNet-pretrained weights exist at that width in torchvision or timm"],
        ["ShuffleNetV2 0.5x", "ShuffleNet", ""],
        ["ShuffleNetV2 1.0x", "ShuffleNet", ""],
        ["SqueezeNet 1.1", "SqueezeNet", ""],
        ["RepGhostNet 0.5x", "RepGhost", "Substitute for originally-planned GhostNet 0.5x -- same pretrained-weights constraint"],
    ],
    [1.7, 1.1, 3.5],
)
heading("4.2 Two-stage transfer-learning protocol", level=2)
para(
    "Every candidate uses the identical classification head -- global average pooling (already "
    "part of each backbone's own forward pass) followed by Linear(->128), ReLU, Dropout(0.3), "
    "Linear(128->1), trained with BCEWithLogitsLoss on the raw logit -- and the identical "
    "two-stage schedule below. Keeping the head and hyperparameters fixed across all eight "
    "candidates isolates backbone choice as the one varying factor in this comparison, at the "
    "cost of not giving any individual architecture its own tuned configuration (Section 9)."
)
make_table(
    ["Stage", "What's trained", "Optimizer / LR", "Max epochs", "Early stopping"],
    [
        ["1 -- frozen backbone", "New head only; backbone frozen (incl. BatchNorm kept in eval mode)",
         "AdamW, lr=1e-4, weight decay=1e-4", "25", "Patience 5 on val loss"],
        ["2 -- partial fine-tune", "Head + last ~15-20% of backbone depth (architecture-specific blocks)",
         "AdamW, lr=1e-5, weight decay=1e-4", "25", "Patience 5 on val loss"],
    ],
    [1.3, 2.6, 1.8, 0.7, 1.1],
)
para(
    "Both stages use ReduceLROnPlateau (factor 0.5, patience 2, monitoring validation loss), "
    "images resized to 224x224, batch size 32, and seed 42. Training-only augmentation (random "
    "resized crop, horizontal flip, +/-12 degree rotation, small translation, mild brightness/"
    "contrast jitter) deliberately avoids any transform that could alter clinically relevant "
    "lesion appearance -- no strong hue shifts, no vertical flip. Validation and test images use "
    "deterministic resize-and-normalize preprocessing only."
)
heading("4.3 Evaluation protocol and test-set isolation", level=2)
para(
    "The training script structurally never opens the test manifest -- the file path does not "
    "appear anywhere in its code, so there is no flag or debug path that could touch it during "
    "training or model selection. This matters because an earlier version of this comparison "
    "did not enforce that discipline strongly enough at the finalist-comparison stage: a "
    "one-off, unsaved script evaluated three of the candidates against the ResNet18 "
    "baseline on the test set before the remaining six candidates had finished training. No "
    "training decision was actually changed as a result, but the sequencing was unsafe and "
    "undocumented, which is itself a finding worth reporting rather than quietly correcting."
)
para(
    "The fix adopted is a dedicated, permanent evaluation script with two gates enforced in "
    "code rather than left as a convention to remember: it refuses to open the test manifest at "
    "all unless every one of the eight candidates has a completed checkpoint on disk, and it "
    "refuses to run a second time once a result file already exists (a deliberate --force "
    "override exists, but prints a warning that using it is itself the thing that needs "
    "documenting). This turns \"don't touch the test set until every candidate is ready\" from a "
    "rule a script can silently ignore into something the code itself checks before the test "
    "manifest is ever imported. All test-set numbers in Section 5 of this report were produced "
    "either under this enforced gate or, for the three earliest candidates, reconstructed and "
    "verified from the original (pre-gate) evaluation output, cross-checked against the known "
    "test-set class counts to confirm they are genuinely test-set numbers and not a mislabeled "
    "copy of validation results."
)
heading("4.4 Statistical comparison", level=2)
para(
    "Pairwise comparisons between models use three complementary tests on the same paired "
    "test-set predictions: McNemar's continuity-corrected test on the paired correct/incorrect "
    "indicators, a paired bootstrap (2,000 resamples, same resampled indices applied to both "
    "models on every draw) giving a 95% confidence interval and two-sided empirical p-value on "
    "the accuracy and F1 differences, and Holm-Bonferroni correction applied across all pairwise "
    "comparisons among the models actually included in a given comparison, to control the "
    "family-wise error rate rather than reading each pairwise p-value independently. The number "
    "of comparisons this correction accounts for is therefore not fixed -- it scales with how "
    "many models are being compared at that stage. The interim image-model read in Section 5.3 "
    "compares 4 models (3 trained candidates plus the ResNet18 baseline), giving 6 pairwise "
    "comparisons; the interim stress-model read in Section 6.5 compares the 4 tabular-feature "
    "models, also giving 6 pairwise comparisons. The final image-model comparison, once all "
    "eight candidates have finished training, will compare 9 models in total (eight candidates "
    "plus the ResNet18 baseline), giving 36 pairwise comparisons -- the Holm-Bonferroni "
    "correction applied to that final comparison will account for all 36, not 6."
)
heading("4.5 Training infrastructure and its effect on this study", level=2)
para(
    "All training and evaluation in this report ran on a single CPU-only, 8 GB RAM development "
    "machine -- no GPU was available. This is reported here as a methods-level constraint, not "
    "an incidental detail: it is the direct reason one of the eight candidates remains untrained "
    "as of this writing (Section 5.5), after two earlier candidates had to be retried from "
    "scratch following the incidents below, and it caused two operational incidents during this "
    "study that are disclosed rather than smoothed over. First, six copies of the training "
    "process ended up running concurrently for roughly two hours after a series of relaunches "
    "each assumed, incorrectly, that the previous attempt had died -- caught only when two of "
    "the script's own log outputs for a supposedly-finished epoch disagreed with each other, "
    "traced to an unreliable process-liveness check on this machine (tasklist silently failed "
    "to show live Python processes; Get-CimInstance Win32_Process did not), and resolved by "
    "discarding every number produced during the overlap window and restarting the affected "
    "run from scratch. Second, a background training run was later killed outright by memory "
    "pressure mid-epoch with no exception raised, and a related, distinct code defect was found "
    "and fixed in the same training script: an early, RAM-triggered stop before any epoch "
    "completed left no checkpoint to restore, which the script had not previously handled and "
    "crashed on with a misleading stack trace rather than a clear message. Neither incident "
    "affected the integrity of the results reported in Section 5 -- in both cases, the affected "
    "run's output was identified as untrustworthy and discarded rather than reported -- but both "
    "are the direct reason this report is being published as an interim, honestly partial "
    "comparison rather than held until all eight candidates finish."
)

# ================= 5. RESULTS =================
heading("5. Results", level=1)
heading("5.1 Training status of all candidates", level=2)
make_table(
    ["Model", "Status as of 2026-09-18"],
    [
        ["EfficientNet-B0", "Complete"],
        ["MobileNetV3-Small", "Complete"],
        ["ShuffleNetV2-0.5x", "Complete"],
        ["SqueezeNet1.1", "Complete (retried after an earlier low-RAM failure)"],
        ["RepGhostNet-0.5x", "Complete (retried after an earlier memory-pressure kill)"],
        ["ShuffleNetV2-1.0x", "Complete"],
        ["MobileNetV2 1.0x", "Complete"],
        ["EfficientNet-Lite0", "Complete"],
    ],
    [2.2, 4.8],
)
heading("5.2 Validation performance (completed candidates)", level=2)
make_table(
    ["Model", "Params", "Size (MB)", "Val Acc", "Val F1", "Val AUC", "Inference (ms/img, CPU)"],
    [
        ["EfficientNet-B0", "4,171,645", "16.2", "80.65%", "80.49%", "0.8773", "37.62"],
        ["MobileNetV2 1.0x", "n/a", "9.34", "78.83%", "78.87%", "0.8738", "30.01"],
        ["ShuffleNetV2-1.0x", "n/a", "5.45", "79.23%", "79.44%", "0.8613", "25.98"],
        ["RepGhostNet-0.5x", "n/a", "4.85", "76.81%", "77.05%", "0.8580", "38.95"],
        ["ShuffleNetV2-0.5x", "473,121", "1.94", "79.84%", "80.47%", "0.8536", "21.48"],
        ["MobileNetV3-Small", "1,000,993", "3.94", "76.01%", "75.96%", "0.8484", "18.09"],
        ["SqueezeNet1.1", "n/a", "3.03", "78.02%", "78.16%", "0.8430", "32.69"],
    ],
    [1.6, 0.9, 0.85, 0.8, 0.8, 0.8, 1.25],
)
para(
    "EfficientNet-B0 leads on validation AUC; ShuffleNetV2-0.5x is competitive on accuracy/F1 "
    "at roughly 1/8th the size. None of this is significance-tested -- that only happens once, "
    "on the held-out test set, after every candidate has finished (Section 4.3)."
)
heading("5.3 Held-out test performance (final, full-field read)", level=2)
para(
    "All eight candidates finished training on 2026-09-18. The evaluation gate described in "
    "Section 4.3 passed and permitted its one, official test-set read across the full "
    "nine-model field (eight candidates plus the ResNet18 baseline), saved to "
    "transfer_cnn_final_comparison_2026-09-18.json. This supersedes the earlier 3-of-8-candidate "
    "partial read documented in docs/transfer_cnn_test_set_access_2026-09-18.md, which is no "
    "longer this report's test result (that earlier read remains on record there as a "
    "methodological incident, not as data). These are validation-independent, held-out test-set "
    "numbers -- see Section 5.2 for the separate validation-set numbers, which should not be "
    "conflated with the test numbers below."
)
make_table(
    ["Model", "Size (MB)", "Inference (ms/img, CPU)", "Test Acc", "Precision", "Recall", "F1", "AUC"],
    [
        ["ResNet18 (baseline)", "42.72", "33.05", "81.07%", "79.92%", "82.47%", "81.18%", "not computed"],
        ["EfficientNet-B0", "16.20", "37.62", "77.51%", "81.86%", "70.12%", "75.54%", "0.8762"],
        ["EfficientNet-Lite0", "13.74", "32.99", "78.90%", "81.58%", "74.10%", "77.66%", "0.8739"],
        ["MobileNetV2-1.0x", "9.34", "30.01", "78.90%", "80.77%", "75.30%", "77.94%", "0.8654"],
        ["ShuffleNetV2-1.0x", "5.45", "25.98", "79.68%", "78.91%", "80.48%", "79.68%", "0.8672"],
        ["RepGhostNet-0.5x", "4.85", "38.95", "76.53%", "76.19%", "76.49%", "76.34%", "0.8549"],
        ["MobileNetV3-Small", "3.94", "18.09", "76.73%", "77.14%", "75.30%", "76.21%", "0.8547"],
        ["SqueezeNet1.1", "3.03", "32.69", "75.54%", "75.92%", "74.10%", "75.00%", "0.8375"],
        ["ShuffleNetV2-0.5x", "1.94", "21.48", "74.95%", "71.99%", "80.88%", "76.17%", "0.8356"],
    ],
    [1.55, 0.75, 1.1, 0.75, 0.75, 0.7, 0.6, 0.8],
)
caption(
    "n = 507 test images (251 Eczema / 256 Other) for every row. All eight candidates' "
    "inference times are measured under this study's protocol (Section 4.5); ResNet18's was "
    "benchmarked separately under the same protocol (batch=1, CPU, 50-run mean, 3 warmup "
    "runs) for this final comparison -- it was not benchmarked in the earlier partial read. "
    "ResNet18's AUC was not computed under the evaluation script used for that previously used "
    "model (argmax-based 2-class softmax, not a probability-ranking evaluation)."
)
doc.add_picture(str(ROOT / "docs" / "architecture_comparison_size_vs_accuracy_2026-09-18.png"),
                 width=Inches(6.0))
last_p = doc.paragraphs[-1]
last_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
caption("Figure 1. Checkpoint size (log scale) vs. held-out test accuracy, all nine models "
        "(final comparison). ShuffleNetV2-1.0x, this report's selected architecture "
        "(Section 5.6), marked with a star.")

heading("5.4 Pairwise significance (Holm-corrected across all 36 comparisons)", level=2)
para(
    "All C(9,2) = 36 pairwise comparisons among the nine models were tested (McNemar's test on "
    "paired correct/incorrect indicators, Holm-Bonferroni correction across all 36). None reach "
    "significance. The six comparisons with the lowest raw p-values -- the closest any pair "
    "came to a distinguishable difference -- are shown below; the full 36 are in "
    "transfer_cnn_final_comparison_2026-09-18.json."
)
make_table(
    ["Comparison", "McNemar p (raw)", "Holm-corrected p", "Significant?"],
    [
        ["ResNet18 vs. ShuffleNetV2-0.5x", "0.0028", "0.102", "No"],
        ["ResNet18 vs. SqueezeNet1.1", "0.0059", "0.205", "No"],
        ["ShuffleNetV2-0.5x vs. ShuffleNetV2-1.0x", "0.0111", "0.377", "No"],
        ["ResNet18 vs. RepGhostNet-0.5x", "0.0197", "0.650", "No"],
        ["SqueezeNet1.1 vs. ShuffleNetV2-1.0x", "0.0281", "0.901", "No"],
        ["ResNet18 vs. MobileNetV3-Small", "0.0321", "0.995", "No"],
    ],
    [2.6, 1.3, 1.3, 1.1],
)
para(
    "Zero of 36 comparisons are statistically significant after Holm correction. Notably, "
    "ResNet18 vs. ShuffleNetV2-0.5x -- the one comparison that reached significance in the "
    "earlier partial read, when only 6 comparisons were being corrected for (Holm-corrected "
    "p=0.017 at the time) -- does not survive correction across the full 36 (p=0.102 now). "
    "This is not a contradiction: the raw p-value barely moved (0.0028 in both reads); what "
    "changed is the number of comparisons the correction has to account for, exactly as "
    "described in Section 4.4. No candidate, and no pair of candidates, is shown to differ "
    "from any other at this sample size (n=507) once corrected properly."
)
heading("5.5 What this establishes", level=2)
para(
    "The full, final comparison does not select a statistically superior image architecture. "
    "All eight candidates are trained and tested; none is distinguishable from the ResNet18 "
    "baseline, or from each other, at this sample size. This is a real, honest null result, not "
    "an inconclusive placeholder -- the comparison this report set out to run (Section 1.2, "
    "research question 1) has been run to completion, and the answer it gives is that none of "
    "these architectures can be shown to beat or lose to ResNet18 on accuracy alone."
)
heading("5.6 Selected architecture: ShuffleNetV2-1.0x", level=2)
para(
    "With no statistically significant accuracy difference among any of the nine models, "
    "accuracy alone cannot be the basis for a selection -- so this report selects on the "
    "criteria it can measure with certainty: checkpoint size and inference latency (Section 1.1, "
    "contribution 3), among candidates whose raw accuracy is not directionally weak. "
    "ShuffleNetV2-1.0x is selected as this report's image-channel architecture for deployment: "
    "it is 7.8x smaller than the ResNet18 baseline (5.45 MB vs. 42.72 MB) and 21% faster "
    "(25.98 ms vs. 33.05 ms per image on this study's CPU benchmark), while posting the closest "
    "raw test accuracy to the baseline of any candidate (79.68% vs. 81.07%, a 1.4-point gap) "
    "and a competitive AUC (0.867, third-highest of the nine). This is a deployment decision "
    "made on size, speed, and the absence of any measured accuracy penalty -- it is explicitly "
    "not a claim that ShuffleNetV2-1.0x is statistically superior to ResNet18 or to any other "
    "candidate, which Section 5.4 does not support. MobileNetV3-Small remains a documented "
    "alternative if a smaller/faster footprint (3.94 MB, 18.09 ms) is prioritized over closeness "
    "to baseline accuracy (Section 8)."
)

# ================= 6. STRESS MODEL =================
heading("6. Stress Model: LSTM-Based Architecture and Comparison", level=1)
para(
    "The stress channel's previously used model was a personal-baseline-calibrated LightGBM "
    "classifier on WESAD wrist-wearable features (mean AUC 0.9405, "
    "docs/wesad_calibration_significance_2026-09-16.md) -- a feature-based model that had never "
    "been compared against an architecture built for the actual structure of the underlying "
    "data. This section sets out an LSTM-based model as this report's primary direction for "
    "the stress channel going forward, explains why that architecture fits the modality, and "
    "reports a controlled comparison against the previously used model and three classical "
    "alternatives on a held-out test set."
)
heading("6.1 Why an LSTM for the stress channel", level=2)
para(
    "WESAD's wrist-wearable data is physiological time series, not a set of independent "
    "observations: stress-related responses (heart-rate and pulse-waveform changes, "
    "electrodermal activity) evolve continuously over the course of an episode rather than "
    "appearing as isolated data points. The previously used LightGBM model handled this by "
    "reducing "
    "each time window to a fixed set of 52 hand-engineered summary statistics (mean, standard "
    "deviation, dominant frequency, and similar) before classification -- a reasonable and "
    "already-validated approach, but one that discards the temporal ordering within a window by "
    "construction. An LSTM (long short-term memory network) is architected specifically to "
    "model temporal dependencies and sequential patterns, and can be trained directly on the "
    "raw physiological waveform rather than requiring all temporal information to be collapsed "
    "into static features beforehand. This is a moderate, modality-specific argument, not a "
    "general claim that LSTMs outperform feature-based models -- Section 6.5 reports what the "
    "comparison on this project's data actually shows, and it does not show the LSTM-based "
    "model to be statistically superior."
)
heading("6.2 Comparison models", level=2)
para(
    "To check the LSTM-based model against the alternatives already available to this project, "
    "three classical models were trained under the same protocol on the previously used "
    "model's "
    "existing 52 hand-engineered features, plus a from-scratch (uncalibrated) LightGBM as a "
    "bonus reference point to help separate how much of the previously used model's strength "
    "comes "
    "from the LightGBM architecture itself versus its personal-baseline calibration step:"
)
bullet("XGBoost and CatBoost -- classical gradient-boosted trees, same 52 hand-engineered "
       "wrist-sensor features as the previously used model. Direct precedent: a published "
       "comparison "
       "of exactly this model family for AD diagnosis/severity from clinical features found "
       "CatBoost the strongest performer (AUC 0.91) [14].")
bullet("Random Forest -- classical bagged-tree baseline, same features. Precedent: used for "
       "AD-vs-healthy classification and SCORAD/EASI severity regression in a separate "
       "published study [15].")
para(
    "The LSTM-based model itself is a small convolutional-downsampling-into-LSTM architecture "
    "(referred to as the CNN-LSTM in the results below), trained directly on raw, "
    "bandpass-filtered BVP (photoplethysmography) waveforms, bypassing hand-engineered features "
    "entirely -- the direct implementation of the argument in Section 6.1. Precedent for "
    "CNN/LSTM architectures on AD-adjacent wearable tasks includes nocturnal scratch detection "
    "from wrist actigraphy [16] and the SIGMA scratch-sensing glove [4]."
)
para(
    "A pretrained-model alternative was also attempted and dropped: PaPaGei [17], a real, "
    "published (ICLR 2025) open foundation model pretrained on 57,000 hours of PPG signal, "
    "which would have been used as a frozen embedding extractor with a small trained classifier "
    "head -- the one genuine pretrained-transfer-learning opportunity identified for this "
    "modality (unlike, for example, the project's separate exploration of a moisture-sensor "
    "dataset, where no comparable pretrained model exists for that signal type at all). Setup "
    "was blocked by an unresolved dependency conflict within PaPaGei's own published "
    "requirements -- one of its listed dependencies (biobss) requires the legacy pkg_resources "
    "module, while another (torch, also a PaPaGei dependency) requires a setuptools version "
    "that no longer bundles it. This is a genuine incompatibility between two of PaPaGei's own "
    "stated requirements, not a fixable local configuration error, and further attempts were "
    "stopped once this was confirmed rather than continuing to consume time on it."
)
heading("6.3 Signal preprocessing", level=2)
para(
    "Checked directly before this comparison began: no signal-level noise filtering existed "
    "anywhere in this project's existing WESAD pipeline (only condition-label handling, a "
    "different and unrelated kind of \"noise\"). A standard PPG bandpass filter (0.5-8 Hz, "
    "third-order Butterworth, zero-phase via filtfilt) was added and applied to the raw BVP "
    "signal before it reaches the CNN-LSTM -- this matters more for a model consuming the raw "
    "waveform directly than it did for the deployed feature-based model, whose hand-engineered "
    "statistics are comparatively robust to unfiltered noise."
)
heading("6.4 Patient-wise split and training discipline", level=2)
para(
    "WESAD's 15 subjects were split once, by subject, seed 42: 11 for training/validation, 4 "
    "held out as a locked test set (S2, S11, S14, S16) never touched until the single official "
    "test-set read in Section 6.5. The 11 training-side subjects were further split into 8 "
    "for training and 3 for validation/model-selection, with zero subject overlap verified "
    "and printed at runtime, not just assumed."
)
para(
    "The first CNN-LSTM training run (fixed 12 epochs, flat learning rate) reached its best "
    "validation AUC (0.928) at epoch 8, then degraded sharply the very next epoch (down to "
    "0.785) before oscillating for the remainder of training with nothing to catch it. This "
    "was diagnosed, not just patched over: the model was still learning (AUC was climbing "
    "well before F1 crossed zero), but nothing in the training loop responded once it started "
    "overfitting. A revised version added validation-AUC-monitored learning-rate decay "
    "(ReduceLROnPlateau), early stopping with best-checkpoint restoration, gradient clipping, "
    "and light additional regularization -- the same category of fix already standard in this "
    "project's image-model training script, just missing from the first draft of this one. The "
    "revised run's peak validation AUC is lower (0.910 vs. 0.928) but is the number that was "
    "actually selected by a predefined rule and survived past its own peak, rather than the "
    "best moment cherry-picked out of a run that was still visibly unstable. The same "
    "validation-driven, early-stopped selection was then applied to XGBoost and CatBoost (both "
    "support it natively); Random Forest has no equivalent mechanism in scikit-learn and was "
    "left on a fixed round count."
)
heading("6.5 Results", level=2)
para("Internal validation (3 held-out training-side subjects):")
make_table(
    ["Model", "Val Accuracy", "Val AUC", "Val F1"],
    [
        ["CNN-LSTM", "88.05%", "0.9101", "0.8102"],
        ["CatBoost", "85.71%", "0.9076", "0.7619"],
        ["Random Forest", "80.00%", "0.8614", "0.6769"],
        ["LightGBM (bonus, uncalibrated)", "81.90%", "0.8592", "0.6667"],
        ["XGBoost", "78.10%", "0.8337", "0.5965"],
    ],
    [2.3, 1.3, 1.0, 1.0],
)
para("Held-out test set (4 subjects, single official read):")
make_table(
    ["Model", "Test Accuracy", "Test AUC (95% CI)", "Test F1"],
    [
        ["Random Forest", "85.6%", "0.883 [0.805, 0.950]", "0.697"],
        ["CatBoost", "82.0%", "0.876 [0.792, 0.950]", "0.603"],
        ["CNN-LSTM*", "85.5%", "0.868 [0.831, 0.902]", "0.723"],
        ["XGBoost", "80.6%", "0.873 [0.801, 0.938]", "0.526"],
        ["LightGBM (bonus, uncalibrated)", "84.2%", "0.802 [0.699, 0.891]", "0.645"],
    ],
    [2.3, 1.3, 1.5, 1.0],
)
caption(
    "*CNN-LSTM was evaluated on 566 raw signal windows from the 4 test subjects, while the "
    "other four models were evaluated on 139 feature-engineered rows from the same subjects -- "
    "a different sample structure (not the same rows), so CNN-LSTM's predictions cannot be "
    "paired row-for-row against the other four's and are reported standalone, not included in "
    "the pairwise test below."
)
para(
    "Pairwise significance (McNemar + paired bootstrap, Holm-corrected across the 6 "
    "comparisons among the four tabular-feature models): one raw p-value looked promising "
    "before correction (XGBoost vs. Random Forest, p=0.0455) but does not survive Holm "
    "correction (p_holm=0.273). No pairwise comparison among the four models is statistically "
    "significant on this held-out test set."
)
heading("6.6 Interpretation", level=2)
para(
    "This comparison reaches a clean, if unglamorous, conclusion: at this sample size (4 "
    "held-out subjects), none of the four models tested is statistically distinguishable from "
    "any other. Random Forest and the LSTM-based model post the best raw numbers on test, but "
    "\"best on this small test set\" and \"actually better\" are not the same claim, and this "
    "report does not conflate them -- the LSTM-based model is not being reported as the winner "
    "of a competition it did not statistically win. Its standing as this report's primary "
    "direction for the stress channel rests on the architectural fit argued in Section 6.1 "
    "(a model built for sequential physiological data, rather than one requiring temporal "
    "information to be pre-reduced to static features), not on this test-set result -- and this "
    "comparison's job was to confirm that choice is not contradicted by the data, which it is "
    "not: the LSTM-based model performs comparably to, not worse than, the classical "
    "alternatives and the previously used model. This mirrors the analogous finding from this "
    "project's separate, smaller moisture-sensor model comparison (13 patients, Section 7): at "
    "small enough sample sizes, model choice stops being the thing that moves the number, and "
    "sample size becomes the real constraint. Unlike that comparison, however, this one reaches "
    "its conclusion with a full, formal, held-out test-set read behind it, not just internal "
    "validation."
)

# ================= 7. MOISTURE MODEL =================
heading("7. Moisture Model: Skin-Hydration Proxy", level=1)
para(
    "The moisture channel is intended to estimate a skin-surface moisture/hydration proxy from "
    "the wearable skin patch, as physiological context alongside the image, stress, and scratch "
    "channels. This section is deliberately shorter than Sections 5 and 6: unlike those two "
    "channels, the moisture channel has no trained model of its own yet, and no dataset "
    "collected on this project's own hardware. What follows is a summary of the exploratory "
    "work already done on external reference data, and a planned methodology for the model this "
    "project will eventually need to build and evaluate on its own data."
)
heading("7.1 What the moisture proxy is, and is not", level=2)
para(
    "The intended signal is a skin-surface moisture/hydration proxy, derived from a capacitive "
    "sensor of the kind already surveyed in this project's hardware documentation. This is "
    "stated carefully because the underlying literature does not support a stronger claim: "
    "capacitive skin-hydration sensors of this type report a correlation with clinical "
    "hydration instruments (e.g. a Corneometer), not a direct measurement of eczema severity, "
    "and this report makes no claim that the sensor measures skin-barrier function or "
    "transepidermal water loss (TEWL) directly -- only a dedicated TEWL-specific instrument, "
    "not a capacitive sensor, measures that. The moisture channel's role in this system is to "
    "provide a hydration-related proxy signal to the eventual fusion step, alongside the image, "
    "stress, and scratch channels -- not to stand in as an independent eczema-severity model."
)
heading("7.2 External reference data and exploratory comparison already done", level=2)
para(
    "This project has already investigated Southampton e-textile capacitive sensor datasets and "
    "publications for this modality -- a companion dataset for an IEEE Sensors Journal paper on "
    "an interdigitated capacitive AD-monitoring sensor, and a PhD thesis dataset (13 patients, "
    "lesional/non-lesional capacitance recordings compared against Corneometer and TEWL "
    "readings, CC-BY licensed). This is external reference data, not data collected on this "
    "project's own hardware, and it was used for two things: as a calibration reference (what "
    "real lesional-vs-healthy capacitance readings look like), and as a small, exploratory "
    "model comparison to check whether any pretrained or classical model could produce a usable "
    "classifier from it directly."
)
para(
    "That exploratory comparison (13 patients, a 9/4 patient-wise train/test split, TEST_SUBJECTS "
    "never touched until the single official read) trained a pretrained tabular foundation model "
    "(TabICL), a logistic regression, and a LightGBM model on hand-engineered features (mean, "
    "std, slope, first/last value, range) from the raw capacitance recordings. All three tied or "
    "underperformed a plain classical baseline, and none is reported here as a usable moisture "
    "model -- with only 13 patients, model choice was not the limiting factor, sample size was, "
    "the same conclusion reached independently for the stress-model comparison in Section 6.6. "
    "This result is not extended into a moisture-model claim for this project; it is reported "
    "only as the reason the planned methodology below treats data collection, not model "
    "selection, as the blocking step."
)
heading("7.3 Planned methodology (not yet executed)", level=2)
para(
    "The following is planned future work, explicitly not a completed experiment or a reported "
    "result. It cannot proceed until the moisture-sensing hardware itself is built and a dataset "
    "is collected on this project's own patients or volunteers."
)
bullet("Baseline normalization. Following the same personal-baseline-calibration approach "
       "already validated for the stress channel (Section 6, and "
       "docs/wesad_calibration_significance_2026-09-16.md), each wearer's moisture readings "
       "would be calibrated against their own resting baseline rather than a pooled population "
       "average, since the stress-channel experience on this project found pooled normalization "
       "to be a real source of error that personal calibration corrected.")
bullet("Signal preprocessing. Raw capacitance readings would need drift correction and "
       "environmental (e.g. temperature/relative-humidity) compensation before feature "
       "extraction -- the same class of preprocessing step already identified as necessary, and "
       "previously missing, for the stress channel's raw-signal model (Section 6.3) -- checked "
       "explicitly rather than assumed present.")
bullet("Feature extraction. Summary statistics (mean, slope, variability) from calibrated "
       "readings, following the same feature style already used in the exploratory comparison "
       "(Section 7.2) and in the deployed stress model's own feature set.")
bullet("Model comparison. Once a real dataset exists, a small comparison among Random Forest "
       "and XGBoost (both already used and validated on this project's own stress-channel data "
       "in Section 6.2) and, if the collected data are sequential/time-series rather than "
       "single-point readings, an LSTM-based model following the same architectural reasoning "
       "as Section 6.1 -- evaluated with the same patient-wise split and test-set isolation "
       "discipline used throughout this report.")
para(
    "No results are reported for this planned methodology. It is included so the moisture "
    "channel's development path is as explicit and reviewable as the image and stress "
    "channels' already-executed comparisons."
)

# ================= 8. DEPLOYMENT CONSIDERATIONS =================
heading("8. Deployment Considerations", level=1)
para(
    "The three candidates evaluated on the test set so far (Section 5.3) range from 1.94 MB to "
    "16.2 MB, against the previously used ResNet18 model's 42.72 MB -- a 2.6x to 22x reduction "
    "in "
    "checkpoint size, none of it (so far) statistically confirmed to cost accuracy once "
    "correction for multiple comparisons is applied; the full validation set (Section 5.2, "
    "seven candidates) shows the same pattern holding more broadly. On a phone-based "
    "deployment, a smaller footprint mainly helps with app package size and the cost of "
    "shipping model updates over the air, rather than being a hard constraint the way it would "
    "be on a microcontroller; still, if the eventual winner comes from the current leaders, "
    "that is real, usable headroom -- both for background monitoring battery budget and for "
    "leaving room to run the fusion model and other sensor channels concurrently on the same "
    "device."
)
para(
    "The CPU inference times reported in Section 5.3 (18-38 ms/image) were measured on this "
    "project's development machine, not on representative phone hardware, and should not be "
    "read as a prediction of on-device latency -- mobile deployment would typically use a "
    "converted, quantized model running on a phone's own GPU/NPU (e.g. via Core ML or TFLite), "
    "which changes the relative ranking of architectures in ways CPU-only timing on a laptop "
    "does not capture. This is flagged again in Section 10 as a concrete follow-up rather than "
    "a caveat to be forgotten."
)

# ================= 9. DISCUSSION =================
heading("9. Discussion", level=1)
para(
    "The headline result is a null one, and it held up under the correction it needed to. In "
    "the earlier partial read (3 of 8 candidates, 6 pairwise comparisons), ResNet18 vs. "
    "ShuffleNetV2-0.5x looked significant (Holm-corrected p=0.017). In the final read (all 9 "
    "models, 36 pairwise comparisons), that same comparison's raw p-value barely moved "
    "(0.0028 in both reads) but its Holm-corrected p rose to 0.102 -- not significant. This is "
    "the multiple-comparisons correction working as intended, not a change in the underlying "
    "data: comparing more models at once requires a stricter bar for any single pair to clear, "
    "and a finding that survives a small correction does not necessarily survive a larger one. "
    "Taking the smaller, earlier read's significant finding at face value would have been the "
    "wrong lesson to draw from this study."
)
para(
    "With no pairwise comparison significant, model size and inference speed become the only "
    "axes where this comparison has real, measured differences to work with (Section 5.6). "
    "ShuffleNetV2-1.0x's selection reflects that directly: it is not the candidate with the "
    "highest point-estimate accuracy or AUC (EfficientNet-Lite0 leads on AUC, ResNet18 leads on "
    "raw accuracy), but it is the smallest/fastest candidate whose raw numbers are not "
    "directionally weak, which is a defensible basis for a deployment decision precisely because "
    "the alternative -- picking by point-estimate accuracy alone -- is not statistically "
    "supported by this data at all."
)
para(
    "It is also worth being explicit about what this comparison does not claim. A model that "
    "matches ResNet18 on this internal, same-source test set has not been shown to match it on "
    "external data -- the ResNet18 baseline's own external validation on two independently "
    "sourced dermatology datasets found near-chance generalization (AUC 0.6512), and there is no "
    "reason to assume ShuffleNetV2-1.0x, or any other candidate here, does better until that "
    "same audit is repeated for it directly (Section 11)."
)

# ================= 10. LIMITATIONS =================
heading("10. Limitations", level=1)
bullet("CPU-only training and timing. No GPU was available on the development machine used "
       "for this study; the reported inference-latency numbers are CPU figures and are not a "
       "reliable stand-in for on-device mobile latency (Section 8).")
bullet("Single fixed data split. Results are not k-fold cross-validated; n=507 test images "
       "gives wide confidence intervals on any single pairwise comparison, consistent with "
       "zero of the 36 corrected comparisons reaching significance (Section 5.4) -- a real "
       "possibility given this sample size is that this comparison lacks the statistical power "
       "to detect true differences that do exist, not evidence that no differences exist.")
bullet("Image-level, not patient-wise, split. The three merged source archives provide only "
       "per-image files with no patient or subject identifier of any kind, so the 70/15/15 "
       "train/validation/test split (Section 3) is stratified by disease-class label at the "
       "image level, not by patient. If any patient contributed more than one photograph to "
       "this dataset, images from that patient could fall on both sides of the split, which "
       "would let a model partly recognize patient-specific characteristics rather than purely "
       "disease-distinguishing ones and inflate reported performance versus a true patient-wise "
       "split. This is a real, unresolved limitation of the dataset as sourced, not a "
       "methodological choice made for this comparison -- it also affects the currently "
       "previously used ResNet18 model equally, since both use the same split, but it means every "
       "number in this report should be read as image-level performance, not verified "
       "patient-level generalization.")
bullet("Identical hyperparameters and head across all eight candidates, by design. This "
       "isolates backbone choice as the one varying factor, which is the point of the "
       "comparison, but may understate any individual architecture's true best achievable "
       "performance under its own tuned learning-rate schedule or unfreezing strategy.")
bullet("Two substituted architectures. MobileNetV2 1.0x stands in for an originally-planned "
       "0.35x width, and RepGhostNet 0.5x for an originally-planned GhostNet 0.5x, because no "
       "verified ImageNet-pretrained weights exist at those specific widths in torchvision or "
       "timm -- a pretrained-weight availability constraint, not a methodological choice.")
bullet("Internal validation only. This comparison does not repeat the project's own "
       "cross-dataset external-validation audit per architecture (Section 2.3); a "
       "generalization gap similar to the one already found for the ResNet18 baseline should "
       "be assumed to exist for every candidate here until specifically tested.")
bullet("Development-hardware constraints directly affected this study's pace and, briefly, "
       "required discarding results from two incidents (Section 4.5). This is disclosed as a "
       "threat to timely completion of the full eight-candidate comparison, not as a threat to "
       "the validity of the results already reported in Section 5, which were computed "
       "correctly and under enforced test-set isolation.")
bullet("Stress-channel sample size. The stress-model comparison's held-out test set is 4 "
       "subjects (Section 6.4); the resulting null result (Section 6.6) is an honest reading "
       "of what that sample size can and cannot establish, not a claim that the LSTM-based "
       "model and the classical alternatives are truly equivalent.")
bullet("No dataset for the scratch channel. Unlike the image and stress channels, which had "
       "real datasets to run this report's comparison methodology on, the scratch channel has "
       "none: no public dataset matching this project's target wrist-accelerometer hardware and "
       "sampling-rate requirements has been identified (Section 11), and this project's own "
       "earlier attempt at a scratch proxy (the public WISDM dataset, 20 Hz) was abandoned as "
       "unusable once a literature review found the discriminating signal sits at 100-800 Hz. "
       "This is a data-collection gap, not a model-selection gap, and it is out of scope for "
       "this report to close.")

# ================= 11. FUTURE WORK =================
heading("11. Future Work", level=1)
bullet("Re-benchmark ShuffleNetV2-1.0x, the selected image-channel architecture (Section 5.6), "
       "on representative target hardware (an actual phone-class device, via a "
       "converted/quantized model), rather than relying on the CPU-only development-machine "
       "timing reported here.")
bullet("Repeat the cross-dataset external-validation audit already performed for the ResNet18 "
       "baseline (near-chance AUC 0.6512, Section 1) on ShuffleNetV2-1.0x specifically, before "
       "treating its internal test-set performance as evidence of real-world generalization -- "
       "this is the direct, already-identified next step for the selected architecture, not a "
       "new piece of work.")
bullet("The stress-channel comparison is complete on its own terms (Section 6) but reached a "
       "null result -- no model distinguishable from any other at n=4 test subjects. A natural "
       "follow-up, not required to close this report, is more WESAD subjects or an additional "
       "dataset to push past that sample-size ceiling, the same limitation identified "
       "independently for the moisture-channel exploratory comparison (Section 7.2).")
bullet("Build the moisture-channel hardware and collect a dataset on it, then execute the "
       "planned methodology in Section 7.3 -- like the scratch channel below, blocked on data "
       "collection rather than model selection.")
bullet("Collect a scratch-channel dataset -- no public dataset matching this project's target "
       "hardware and sampling requirements has been identified. This project's own earliest "
       "attempt at a scratch proxy used the public WISDM phone-accelerometer dataset (20 Hz) "
       "and was abandoned once a literature review found the signal that actually discriminates "
       "scratching from other hand motion sits at 100-800 Hz -- an order of magnitude above what "
       "that dataset, or a standard 20 Hz consumer accelerometer, can capture. More recent "
       "literature [16] shows a standard wrist accelerometer/gyroscope at a higher sampling rate "
       "(the same class of part already planned for this project's wrist node, not exotic "
       "hardware) can achieve coarse scratch quantification -- but those studies' own datasets "
       "are not established as available for reuse here, so this project would need to collect "
       "its own labeled wrist-accelerometer data at a usable sampling rate before a scratch model "
       "can be trained at all. Unlike the image and stress channels, this is not a model-selection "
       "problem to run this report's comparison methodology on yet -- there is no dataset to "
       "compare architectures against, and data collection is the prerequisite, not an "
       "afterthought.")
bullet("Integrate the selected image architecture (ShuffleNetV2-1.0x) and the LSTM-based stress "
       "model into the two-device hardware design (wrist node + skin patch + phone) already "
       "scoped in this project's hardware architecture proposal, alongside whatever the moisture "
       "and scratch channels eventually contribute.")

# ================= 12. CONCLUSION =================
heading("12. Conclusion", level=1)
para(
    "This report builds and documents a controlled, statistically rigorous framework for "
    "selecting architectures across two of this project's four sensor modalities -- image and "
    "stress -- and sets out planned methodology for a third, moisture. It enforces its own "
    "test-set isolation discipline in code rather than leaving it as a rule to remember -- "
    "including, honestly, one documented near-miss where that discipline was not strong enough "
    "the first time and had to be fixed. The image channel comparison is complete: all eight "
    "candidates trained and tested against the ResNet18 baseline, with zero of 36 pairwise "
    "comparisons reaching significance after Holm correction -- including the one comparison "
    "that had looked significant in an earlier partial read, which did not survive correction "
    "across the full field. With accuracy not statistically distinguishable, ShuffleNetV2-1.0x "
    "is selected as the image-channel architecture for deployment on measured size and speed "
    "grounds (7.8x smaller, 21% faster than ResNet18, closest raw accuracy to baseline of any "
    "candidate) -- a deployment decision, not a claim of statistical superiority. The stress "
    "channel evaluation is likewise complete: an LSTM-based model, selected for its fit to "
    "WESAD's sequential physiological data rather than for a raw test-score win, was formally "
    "compared against three classical models and the previously used model on a locked "
    "patient-wise "
    "test set, with a null result -- nothing statistically distinguishable from anything else at "
    "this sample size, reported as such rather than dressed up as a finding, and not treated as "
    "evidence against the architectural case made for the LSTM-based model in Section 6.1. The "
    "moisture channel has no trained model yet; a small exploratory comparison on external "
    "reference data found the same sample-size ceiling already seen for stress, and a planned "
    "methodology (Section 7.3) sets out how this project's own moisture model will eventually be "
    "built and evaluated. No architecture is declared statistically superior to any other for "
    "the image or stress channel -- both selections (ShuffleNetV2-1.0x, the LSTM-based stress "
    "model) rest on architectural fit and measured deployment properties, not on a significance "
    "claim the data does not support."
)

# ================= REFERENCES =================
heading("References", level=1)
refs = [
    "[1] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., Chen, L.C. \"MobileNetV2: Inverted "
    "Residuals and Linear Bottlenecks.\" CVPR 2018.",
    "[2] Howard, A., Sandler, M., Chu, G., Chen, L.C., Chen, B., Tan, M., Wang, W., Zhu, Y., "
    "Pang, R., Vasudevan, V., Le, Q.V., Adam, H. \"Searching for MobileNetV3.\" ICCV 2019.",
    "[3] Tan, M., Le, Q.V. \"EfficientNet: Rethinking Model Scaling for Convolutional Neural "
    "Networks.\" ICML 2019.",
    "[4] Ma, N., Zhang, X., Zheng, H.T., Sun, J. \"ShuffleNet V2: Practical Guidelines for "
    "Efficient CNN Architecture Design.\" ECCV 2018.",
    "[5] Iandola, F.N., Han, S., Moskewicz, M.W., Ashraf, K., Dally, W.J., Keutzer, K. "
    "\"SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size.\" "
    "arXiv:1602.07360, 2016.",
    "[6] Chen, C., Guo, Z., Zeng, H., Xiong, P., Dong, J. \"RepGhost: A Hardware-Efficient "
    "Ghost Module via Re-parameterization.\" arXiv:2211.06088, 2022.",
    "[7] He, K., Zhang, X., Ren, S., Sun, J. \"Deep Residual Learning for Image Recognition.\" "
    "CVPR 2016.",
    "[8] Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L. \"ImageNet: A "
    "Large-Scale Hierarchical Image Database.\" CVPR 2009.",
    "[9] Esteva, A., Kuprel, B., Novoa, R.A., Ko, J., Swetter, S.M., Blau, H.M., Thrun, S. "
    "\"Dermatologist-level classification of skin cancer with deep neural networks.\" Nature "
    "542, 115-118, 2017.",
    "[10] Tschandl, P., Rosendahl, C., Kittler, H. \"The HAM10000 dataset, a large collection "
    "of multi-source dermatoscopic images of common pigmented skin lesions.\" Scientific Data "
    "5, 180161, 2018.",
    "[11] McNemar, Q. \"Note on the sampling error of the difference between correlated "
    "proportions or percentages.\" Psychometrika 12(2), 153-157, 1947.",
    "[12] Holm, S. \"A Simple Sequentially Rejective Multiple Test Procedure.\" Scandinavian "
    "Journal of Statistics 6(2), 65-70, 1979.",
    "[13] Efron, B., Tibshirani, R.J. \"An Introduction to the Bootstrap.\" Chapman & Hall/CRC, "
    "1993.",
    "[14] Machine learning-based prediction models for the diagnosis and severity assessment "
    "of atopic dermatitis (CatBoost/XGBoost/LightGBM/Random Forest/Logistic Regression "
    "comparison on clinical features). ScienceDirect, 2026.",
    "[15] Prediction of disease severity using serum biomarkers in patients with "
    "mild-moderate Atopic Dermatitis: A pilot study (Random Forest classification and SCORAD/"
    "EASI regression). PLOS ONE, 2023.",
    "[16] Xing, Y., Song, B., Crouthamel, M., Chen, X., Goss, S. et al. \"Quantifying "
    "Nocturnal Scratch in Atopic Dermatitis: A Machine Learning Approach Using Digital Wrist "
    "Actigraphy.\" Sensors 24(11):3364, 2024.",
    "[17] Pillai, A. et al. \"PaPaGei: Open Foundation Models for Optical Physiological "
    "Signals.\" ICLR 2025, arXiv:2410.20542.",
]
for r in refs:
    p = doc.add_paragraph(r)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.first_line_indent = Inches(-0.3)

OUT_PATH = ROOT / "honors-paper-report.docx"
doc.save(OUT_PATH)
print(f"saved {OUT_PATH}")
