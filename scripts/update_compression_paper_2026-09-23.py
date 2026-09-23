# -*- coding: utf-8 -*-
"""Revise eczema_compression_deployment_paper_2026-09-20.docx -> ..._2026-09-23.docx.

Edits the existing .docx in place-by-copy (it contains later hand edits, e.g. Section
5.5.3, that build_eczema_compression_paper_2026-09-20.py does not), so every untouched
paragraph, table and figure is preserved exactly.

Changes (requested 2026-09-23, matching edge_ai_eval_presentation_FINAL_v3.pptx):
  1. New Section 1.3 "Project history" -- everything that was tried, the dead ends, and
     what the project settled on -- plus eczema background in the Introduction.
  2. New Section 3.4 naming and linking every merged source dataset and the 7 comparison
     diseases; provenance paragraph and reference [24] corrected (the SkinDisease folder is
     youssefmohmmed/human-skin-diseases-image, not haroonalam16/20-skin-diseases-dataset).
  3. Operational status thresholds replaced with literature-backed ones:
        collapsed      -- 95% CI of Youden's J includes 0 (chance level)
        degraded       -- above chance, accuracy retention R below the MLPerf target
        backend-stable -- R >= 0.98 (lightweight) / 0.99 (ResNet18) (MLPerf Inference)
     All status columns (Tables 5, 5b, 5c, 9) and every sentence that quotes status counts
     are recomputed from the result JSONs.
  4. Abstract corrected: it claimed all 9 architectures quantize on fbgemm "with only small
     accuracy changes", which contradicts Table 2 (25-28 pt drops for SE/swish models).
  5. Figure 2 regenerated with the new status colours; new references added.
"""
import copy
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

from paths import ROOT

PAPER_DIR = ROOT / "papers" / "edge-ai-lightweight-deployment"
SRC = PAPER_DIR / "eczema_compression_deployment_paper_2026-09-20.docx"
OUT = PAPER_DIR / "eczema_compression_deployment_paper_2026-09-23.docx"
FIG2 = ROOT / "docs" / "candidate_set_status_2026-09-23.png"

# =====================================================================================
# 1. Recompute every status from the raw result files
# =====================================================================================
ARCH = ["resnet18", "efficientnet_b0", "efficientnet_lite0", "mobilenetv3_small", "mobilenetv2_100",
        "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "squeezenet1_1", "repghostnet_050"]
NAME = {"resnet18": "ResNet18", "efficientnet_b0": "EfficientNet-B0", "efficientnet_lite0": "EfficientNet-Lite0",
        "mobilenetv3_small": "MobileNetV3-Small", "mobilenetv2_100": "MobileNetV2-1.0x",
        "shufflenet_v2_x0_5": "ShuffleNetV2-0.5x", "shufflenet_v2_x1_0": "ShuffleNetV2-1.0x",
        "squeezenet1_1": "SqueezeNet1.1", "repghostnet_050": "RepGhostNet-0.5x"}
B_BOOT = 2000


def youden_ci(se, sp, npos, nneg, seed):
    rng = np.random.default_rng(seed)
    j = rng.binomial(npos, se, B_BOOT) / npos + rng.binomial(nneg, sp, B_BOOT) / nneg - 1
    return se + sp - 1, float(np.percentile(j, 2.5)), float(np.percentile(j, 97.5))


def target(a):
    return 0.99 if a == "resnet18" else 0.98


def status(a, lo, retention):
    if lo <= 0:
        return "collapsed"
    if retention is None or retention >= target(a):
        return "backend-stable"
    return "degraded"


bj = json.load(open(PAPER_DIR / "edge_ai_qnnpack_backend_check_2026-09-19.json"))
tj = json.load(open(PAPER_DIR / "final_test_frozen_results_2026-09-20.json"))["per_model"]
pj = json.load(open(PAPER_DIR / "edge_ai_pareto_analysis_2026-09-19.json"))
zj = json.load(open(PAPER_DIR / "skindisnet_all9_external_validation_2026-09-20.json"))["per_model"]
fj = json.load(open(PAPER_DIR / "skindisnet_finetuned_backend_check_2026-09-21.json"))["per_model"]
FP32 = {p["label"].split()[0]: p["accuracy"] for p in pj["accuracy_vs_size_points"] if p["label"].endswith("(FP32)")}
INT8_MB = {p["label"].split()[0]: p["size_mb"] for p in pj["accuracy_vs_size_points"] if p["label"].endswith("(INT8)")}

# ---- 2026-09-23 correction: qnnpack quantized relu6/hardtanh/clamp is wrong on channels_last input
# (FROZEN_TEST_PROTOCOL_DEVIATION_2026-09-23.md). Replace the qnnpack INT8 results of the two affected
# architectures with the fixed-pipeline results; keep the originals for the before/after table.
BUGGY = ["mobilenetv2_100", "efficientnet_lite0"]
fixv = json.load(open(PAPER_DIR / "qnnpack_relu6_layout_fix_2026-09-23.json"))
fixt = json.load(open(PAPER_DIR / "final_test_relu6_fix_2026-09-23.json"))
fixf = json.load(open(PAPER_DIR / "skindisnet_finetuned_relu6_fix_2026-09-23.json"))
ORIG_BUG = {a: {"val": dict(bj[a]["qnnpack_backend_this_mac"]["point"]),
                "test_cm": dict(tj[a]["qnnpack_int8"]["confusion_matrix"]),
                "ft": dict(fj[a]["finetuned_qnnpack_int8"]["point"])} for a in BUGGY}
for a in BUGGY:
    assert fixv[a]["n_clamp_nodes_fixed"] > 0 and fixt[a]["n_clamp_nodes_fixed"] > 0
    bj[a]["qnnpack_backend_this_mac"]["point"] = {k: fixv[a]["fixed"][k] for k in ORIG_BUG[a]["val"]}
    tj[a]["qnnpack_int8"]["confusion_matrix"] = fixt[a]["fixed"]["confusion_matrix"]
    tj[a]["qnnpack_int8"]["point"] = fixt[a]["fixed"]["point"]
    fj[a]["finetuned_qnnpack_int8"]["point"] = {k: fixf[a]["fixed"][k] for k in ORIG_BUG[a]["ft"]}
for a in ARCH:
    if a not in BUGGY:   # the fix is a no-op for every other architecture
        assert fixv[a]["n_clamp_nodes_fixed"] == 0 and fixv[a]["as_is"] == fixv[a]["fixed"], a
SIZE = {p["label"].split()[0]: p["size_mb"] for p in pj["accuracy_vs_size_points"] if p["label"].endswith("(FP32)")}

S = {a: {} for a in ARCH}
for i, a in enumerate(ARCH):
    # validation split (249 eczema / 247 other), same seeds as the v3 deck generator
    for tag, m in (("fbgemm", bj[a]["x86_fbgemm_backend_original"]),
                   ("qnnpack", bj[a]["qnnpack_backend_this_mac"]["point"])):
        j, lo, hi = youden_ci(m["sensitivity_recall"], m["specificity"], 249, 247, 100 + i)
        r = m["accuracy"] / FP32[a]
        S[a][tag] = dict(acc=m["accuracy"], j=j, lo=lo, hi=hi, r=r, st=status(a, lo, r))
    # frozen test split
    f, q = tj[a]["fp32"]["confusion_matrix"], tj[a]["qnnpack_int8"]["confusion_matrix"]
    npos, nneg = f["tp"] + f["fn"], f["tn"] + f["fp"]
    acc_f, acc_q = (f["tp"] + f["tn"]) / (npos + nneg), (q["tp"] + q["tn"]) / (npos + nneg)
    j, lo, hi = youden_ci(q["tp"] / npos, q["tn"] / nneg, npos, nneg, 200 + i)
    S[a]["test"] = dict(j=j, lo=lo, hi=hi, r=acc_q / acc_f, st=status(a, lo, acc_q / acc_f))
    # SkinDisNet zero-shot (imbalanced: retention on balanced accuracy)
    zz = {}
    for tag in ("fp32", "qnnpack_int8"):
        c = zj[a][tag]["confusion_matrix"]
        np_, nn_ = c["tp"] + c["fn"], c["tn"] + c["fp"]
        se, sp = c["tp"] / np_, c["tn"] / nn_
        jj, lo, hi = youden_ci(se, sp, np_, nn_, 300 + i + (50 if tag != "fp32" else 0))
        zz[tag] = dict(j=jj, lo=lo, hi=hi, ba=(se + sp) / 2)
    zz["qnnpack_int8"]["st"] = status(a, zz["qnnpack_int8"]["lo"], zz["qnnpack_int8"]["ba"] / zz["fp32"]["ba"])
    zz["fp32"]["st"] = status(a, zz["fp32"]["lo"], None)
    S[a]["zs"] = zz
    # SkinDisNet fine-tuned (298-image val split: 88 eczema / 210 other)
    ff = {}
    for tag in ("finetuned_fp32", "finetuned_qnnpack_int8"):
        p = fj[a][tag]["point"]
        se, sp = p["sensitivity_recall"], p["specificity"]
        assert abs(round(se * 88) / 88 - se) < 1e-9 and abs(round(sp * 210) / 210 - sp) < 1e-9, (a, tag)
        jj, lo, hi = youden_ci(se, sp, 88, 210, 400 + i + (50 if "int8" in tag else 0))
        ff[tag] = dict(j=jj, lo=lo, hi=hi, ba=(se + sp) / 2)
    ff["finetuned_fp32"]["st"] = status(a, ff["finetuned_fp32"]["lo"], None)
    ff["finetuned_qnnpack_int8"]["st"] = status(
        a, ff["finetuned_qnnpack_int8"]["lo"], ff["finetuned_qnnpack_int8"]["ba"] / ff["finetuned_fp32"]["ba"])
    S[a]["ft"] = ff


def n_st(key, st, sub=None):
    return sum(1 for a in ARCH if (S[a][key][sub]["st"] if sub else S[a][key]["st"]) == st)


def names(pred):
    return ", ".join(NAME[a] for a in ARCH if pred(a))


print("status recompute:")
for a in ARCH:
    print(f"  {NAME[a]:18s} fbgemm={S[a]['fbgemm']['st']:14s} qnnpack={S[a]['qnnpack']['st']:14s} "
          f"test={S[a]['test']['st']:14s} zs_fp32={S[a]['zs']['fp32']['st']:14s} "
          f"ft_fp32={S[a]['ft']['finetuned_fp32']['st']:14s} ft_int8={S[a]['ft']['finetuned_qnnpack_int8']['st']}")

Q_COLL, Q_DEG, Q_STB = n_st("qnnpack", "collapsed"), n_st("qnnpack", "degraded"), n_st("qnnpack", "backend-stable")
F_COLL, F_DEG, F_STB = n_st("fbgemm", "collapsed"), n_st("fbgemm", "degraded"), n_st("fbgemm", "backend-stable")
T_COLL, T_DEG, T_STB = n_st("test", "collapsed"), n_st("test", "degraded"), n_st("test", "backend-stable")
BIN_MATCH = sum(1 for a in ARCH if (S[a]["qnnpack"]["st"] == "collapsed") == (S[a]["test"]["st"] == "collapsed"))
EXACT_MATCH = sum(1 for a in ARCH if S[a]["qnnpack"]["st"] == S[a]["test"]["st"])
ZS_FP32_CHANCE = sum(1 for a in ARCH if S[a]["zs"]["fp32"]["st"] == "collapsed")
assert (Q_COLL, Q_DEG, Q_STB) == (3, 5, 1), (Q_COLL, Q_DEG, Q_STB)   # prose below is written for these

KJ = json.load(open(PAPER_DIR / "pruning_knee_mcnemar_2026-09-23.json"))["per_model"]
KNEE = {a: KJ[a]["knee_mcnemar"] for a in ARCH}
OLD_KNEE = {a: v for a, v in pj["pruning_knees"].items()}
assert all(abs(r["diff_vs_original_pt"]) < 1e-6 for a in ARCH for r in KJ[a]["sweep"]), "rerun != original sweep"
KNEE_CHANGED = [a for a in ARCH if KNEE[a] != OLD_KNEE[a]]
assert KNEE_CHANGED == ["shufflenet_v2_x0_5"], KNEE_CHANGED   # prose below is written for this
P_AT = lambda a, sp_: [r for r in KJ[a]["sweep"] if r["sparsity"] == sp_][0]["p_holm"]
KNEE_DEF = ('Each architecture\'s "pruning knee" (Table 1) is the lowest sparsity whose validation accuracy is '
    'significantly lower than that architecture\'s own unpruned (0%-sparsity) model: for each sparsity, the '
    'exact two-sided McNemar test [39] compares the per-image correctness of the pruned and unpruned models on '
    'the same 496 validation images -- the paired test recommended for comparing two classifiers on one test '
    'set [38] -- and p-values are Holm-corrected across the 9 sparsities of each architecture [40]; the knee is '
    'the first sparsity with adjusted p < 0.05 and lower accuracy (scripts/pruning_knee_mcnemar_2026-09-23.py). '
    'Because the original sweep stored only aggregate metrics, the sweep was re-run with per-image predictions; '
    'the rerun reproduced every original accuracy exactly. An earlier draft used a fixed > 5-pt drop instead; the '
    'significance-based knee agrees with it for 8 of 9 architectures and moves ShuffleNetV2-0.5x from 40% to 30% '
    f'(a 3.0-pt drop at 30%, adjusted p = {P_AT("shufflenet_v2_x0_5", 0.3):.3f}). There is no standard knee '
    'definition in the pruning literature, which recommends reporting full accuracy-sparsity curves [30]; knees '
    'are architecture-relative, not comparable in absolute sparsity across architectures with different '
    'parameter redundancy, and they mark where a statistically detectable loss begins, not an optimal '
    'deployment sparsity.')

# =====================================================================================
# 2. Figure 2, regenerated with the new status colours
# =====================================================================================
COL = {"collapsed": "#A32C2C", "degraded": "#96600E", "backend-stable": "#2F6B4F"}
fig, ax = plt.subplots(figsize=(7.2, 4.3), dpi=200)
ax.axhspan(45, 55, color="#F1EFE6", zorder=0)
ax.text(1.8, 53.3, "chance level (≈50%)", fontsize=8, color="#5B6169")
for a in ARCH:
    st = S[a]["qnnpack"]["st"]
    ax.scatter(SIZE[a], S[a]["qnnpack"]["acc"] * 100, s=70, color=COL[st], zorder=3,
               edgecolor="white", linewidth=0.8)
    off = {"mobilenetv3_small": (-30, -13), "repghostnet_050": (-40, 8), "mobilenetv2_100": (-45, 8),
           "efficientnet_lite0": (-5, 8), "efficientnet_b0": (-10, -14)}.get(a, (6, 4))
    ax.annotate(NAME[a], (SIZE[a], S[a]["qnnpack"]["acc"] * 100), textcoords="offset points",
                xytext=off, fontsize=7.5, color="#3A3F44")
ax.set_xscale("log")
ax.set_xticks([2, 5, 10, 20, 40])
ax.set_xticklabels(["2", "5", "10", "20", "40"])
ax.set_xlabel("FP32 dense checkpoint size (MB, log scale)")
ax.set_ylabel("ARM/qnnpack INT8 accuracy (%)")
ax.set_ylim(44, 84)
for st in ("backend-stable", "degraded", "collapsed"):
    ax.scatter([], [], color=COL[st], label=st)
ax.legend(frameon=False, fontsize=8, loc="center right")
for sp_ in ("top", "right"):
    ax.spines[sp_].set_visible(False)
fig.tight_layout()
fig.savefig(FIG2)
plt.close(fig)

# =====================================================================================
# 3. docx helpers
# =====================================================================================
doc = Document(SRC)
P = list(doc.paragraphs)       # snapshot of ORIGINAL paragraph indices (see structure map)


def find(prefix, start=0):
    hits = [i for i, p in enumerate(P) if i >= start and p.text.startswith(prefix)]
    assert len(hits) >= 1, prefix
    return P[hits[0]]


def set_text(p, text):
    """Replace a paragraph's text, keeping the first run's formatting."""
    runs = p.runs
    if not runs:
        if text:
            p.add_run(text)
        return
    runs[0].text = text
    for r in runs[1:]:
        r._r.getparent().remove(r._r)


def insert_after(anchor, template, text=None):
    """Insert a copy of `template` paragraph after `anchor` element; returns new paragraph."""
    from docx.text.paragraph import Paragraph
    el = copy.deepcopy(template._p)
    anchor_el = anchor._p if hasattr(anchor, "_p") else anchor._tbl if hasattr(anchor, "_tbl") else anchor
    anchor_el.addnext(el)
    np_ = Paragraph(el, template._parent)
    if text is not None:
        set_text(np_, text)
    return np_


def shade(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def new_table(after_para, headers, rows, widths_in):
    t = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    t.style = doc.tables[0].style
    for c, h in enumerate(headers):
        cell = t.cell(0, c)
        cell.text = ""
        r = cell.paragraphs[0].add_run(h)
        r.bold = True
        r.font.size = Pt(9)
        shade(cell, "D9E2F3")
    for ri, row in enumerate(rows):
        for c, v in enumerate(row):
            cell = t.cell(ri + 1, c)
            cell.text = ""
            r = cell.paragraphs[0].add_run(str(v))
            r.font.size = Pt(9)
    for c, w in enumerate(widths_in):
        for rr in t.rows:
            rr.cells[c].width = Inches(w)
    after_para._p.addnext(t._tbl)
    return t


def set_cell(cell, text):
    runs = [r for r in cell.paragraphs[0].runs]
    target = next((r for r in runs if r.text.strip()), runs[-1] if runs else None)
    if target is None:
        cell.paragraphs[0].add_run(text)
        return
    target.text = text
    for r in runs:
        if r is not target:
            r.text = ""


BODY = find("Deploying an image classifier on wearable")     # a Normal body paragraph
BULLET = find("RQ1: For each of 9 architectures")             # a List Bullet paragraph
H2 = find("1.2 Clinical scope and terminology")               # a Heading 2 paragraph
CAPTION = find("Table 1. FP32 baseline")                      # table caption style
SPACER = P[P.index(CAPTION) - 1]                              # the empty paragraph after tables

# =====================================================================================
# 4. Abstract (corrected fbgemm claim + new status counts)
# =====================================================================================
set_text(BODY,
    "Deploying an image classifier on wearable or edge healthcare hardware usually requires compressing "
    "it -- through pruning, INT8 quantization, or both -- to fit memory, latency, and power budgets. This "
    "paper studies what that compression does to an eczema-vs-other-skin-condition image classifier "
    "(eczema vs. seven visually similar diseases) across nine convolutional architectures (eight "
    "lightweight candidates plus a ResNet18 baseline). It is the image component of a planned wearable "
    "eczema-monitoring system; the project first attempted a multimodal (image + wearable-sensor) "
    "design, but no public dataset pairs the two for the same patients, so the image model was built and "
    "stress-tested first (Section 1.3). We measure accuracy, sensitivity, specificity, balanced accuracy, "
    "and F1 under one-shot magnitude pruning, post-training static INT8 quantization (PTQ), "
    "quantization-aware training (QAT), structured (channel) pruning, and pruning-with-fine-tuning. "
    "Quantized models are classified with literature-backed criteria: collapsed if the 95% bootstrap "
    "confidence interval of Youden's J includes zero (no better than chance), backend-stable if they "
    "retain at least the MLPerf Inference quality target of their FP32 accuracy (98% for lightweight "
    "models, 99% for ResNet-class models), and degraded otherwise. The central finding is that static "
    "INT8 behaviour depends jointly on architecture and on the CPU quantized backend. With identical "
    f"trained weights, {Q_COLL} of 9 architectures collapse to chance on the ARM/qnnpack backend versus "
    f"{F_COLL} of 9 on x86/fbgemm; two (EfficientNet-Lite0 and MobileNetV2-1.0x) are clearly above chance "
    "on fbgemm but collapse on qnnpack, and a third (EfficientNet-B0) is borderline on fbgemm. Only "
    "ResNet18 meets its MLPerf retention target on qnnpack. Architectures using squeeze-and-excitation "
    "blocks and swish-family activations show a mean FP32-to-INT8 (fbgemm) accuracy drop of 27.2 pt "
    "versus 3.9 pt for the other six -- an association, not a demonstrated causal mechanism. A forced "
    "per-channel-quantization diagnostic explains most of the backend gap for one architecture, part "
    "of it for a second, and none for three others. Filtering out architectures that collapse on qnnpack "
    "and excluding the non-lightweight baseline leaves a candidate set of three (ShuffleNetV2-0.5x, "
    "ShuffleNetV2-1.0x, SqueezeNet1.1), within which the two ShuffleNetV2 variants are Pareto "
    "non-dominant, so no single winner is declared. On a frozen, once-evaluated held-out test split, "
    f"the collapsed-versus-above-chance status reproduces for {BIN_MATCH} of 9 architectures. A check on "
    "a second dataset (SkinDisNet) is uninformative zero-shot -- every FP32 model is already at chance "
    "there -- but after fine-tuning on SkinDisNet, three architectures with above-chance FP32 baselines "
    "collapse under qnnpack INT8 while all three lightweight candidates remain backend-stable. Every claim "
    "is scoped to these 9 architectures, the default recipes on the two machines used, and this dataset.")

# =====================================================================================
# 5. Introduction: eczema background; 1.2 terminology; new 1.3 project history
# =====================================================================================
intro = find("Wearable and edge devices for chronic-condition monitoring")
set_text(intro,
    "Eczema (atopic dermatitis) is a common, chronic inflammatory skin disease characterised by recurrent "
    "eczematous lesions and intense itch; it affects people of all ages, has no cure, carries a substantial "
    "psychosocial burden, and is associated with food allergy, asthma, allergic rhinitis, and mental-health "
    "disorders [31]. In the Global Burden of Disease study it has the highest disease burden of any skin "
    "disease and ranks 15th among all nonfatal diseases, with its highest prevalence in early childhood "
    "[32]. The itch-scratch cycle disrupts sleep for patients and families [33], and because disease "
    "activity fluctuates between clinic visits, skin-conformable wearable sensors have been developed to "
    "quantify symptoms objectively and continuously [34]. " + intro.text)

set_text(find("This paper makes no diagnostic claim."),
    "This paper makes no diagnostic claim. \"Eczema\" denotes the label in this project's own curated "
    "dataset (Section 3), not a dermatologist-verified diagnosis for every image, and the task is binary "
    "(eczema-labeled vs. seven other skin diseases, Section 3.4), not a clinical differential-diagnosis "
    "task with graded severity. The operational terms \"collapsed,\" \"degraded,\" and \"backend-stable\" "
    "(Section 4.8) describe the behaviour of a quantized classifier using statistical and industry-benchmark "
    "criteria [26, 29]; they are not clinical safety classifications, and \"sensitivity\"/\"specificity\" "
    "describe binary-classifier behaviour on a curated dataset, not diagnostic performance in clinical use.")

h13 = insert_after(find("This paper makes no diagnostic claim."), H2, "1.3 Project history: what was tried and what was settled on")
p = insert_after(h13, BODY,
    "This compression study is the current end point of a longer project whose goal is a flexible-electronics "
    "system that monitors eczema continuously: a phone camera for skin images, plus a wristband (motion, "
    "pulse) and a skin patch (moisture, temperature, electrodermal activity) for signals linked to flares. "
    "Several directions were tried before settling on the scope of this paper. They are summarised in "
    "Table H so that the choices made here can be read in context; the dead ends are kept in the project "
    "repository (dead_ends/) rather than deleted.")

hist_rows = [
    ["v1 image model: eczema vs. normal skin", "~95% accuracy, but the classes came from different photo sources (clinical vs. stock) and the model learned the source. Brightness normalisation (95.18% -> 93.98%) and skin-region cropping (95.66% -> 95.66%) did not remove the shortcut.", "Dead end"],
    ["Scratch detection from a phone/watch accelerometer (WISDM)", "20 Hz sampling cannot capture the 100-800 Hz vibration signal that dedicated scratch sensors rely on [34].", "Dead end"],
    ["Stress detection (WESAD, 15 subjects, wrist sensors)", "LightGBM with per-person baseline calibration: mean AUC 0.94 under leave-one-subject-out validation.", "Kept (supporting signal)"],
    ["Sleep-disruption detection (AAUWSS, 13 subjects)", "Mean AUC 0.46 -- chance level, corroborated by two standard actigraphy scoring formulas.", "Dead end"],
    ["Image model rebuilt: eczema vs. 7 look-alike diseases", "Same-source photos for both classes remove the photo-source shortcut; three sources merged and deduplicated (Section 3.4).", "Kept (this paper's task)"],
    ["Zero-shot external validation (SCIN [36], SkinDisNet [25])", "AUC 0.535 and 0.487: the model does not yet transfer to new photo sources.", "Reported limitation"],
    ["Multimodal fusion (image + wearables) and severity scoring (EASI/SCORAD)", "No public dataset pairs skin images with wearable data from the same patients, and no public dataset has clinician-validated severity labels.", "Dead end (future work)"],
    ["9-architecture architecture comparison and compression study", "Pruning, INT8 PTQ, QAT and two CPU backends, with a frozen held-out test evaluation.", "This paper"],
]
t0 = new_table(p, ["Direction tried", "What happened", "Outcome"], hist_rows, [1.9, 3.6, 1.0])
sp0 = insert_after(t0, SPACER, "")
cap0 = insert_after(sp0, CAPTION, "Table H. Project history: every direction tried, its outcome, and what was kept.")
insert_after(cap0, BODY,
    "What the project settled on. The image model is the core of the system, trained on a same-source, "
    "deduplicated, class-balanced eczema-vs-look-alike dataset; the WESAD stress model is kept as a "
    "supporting, literature-motivated flare-trigger signal, fused at decision level because no paired "
    "data exists; and, since the image model must run on phone-class ARM hardware, this paper asks which "
    "architectures survive compression there. Its answer is two finalists -- ShuffleNetV2-0.5x (smallest) "
    "and SqueezeNet1.1 (most accurate lightweight model) -- that are Pareto non-dominant, rather than a "
    "single winner (Section 7). Along the way the study found that a defect in the ARM quantized backend "
    "silently turned two architectures into one-class predictors (Section 5.5.4), which is itself a "
    "practical lesson for deploying medical image models on phones. On-device testing of the finalists is "
    "the next step.")

# =====================================================================================
# 6. Dataset: provenance fix + new Section 3.4
# =====================================================================================
prov = find("This project's primary training/validation/test images")
set_text(prov,
    "This project's primary training/validation/test images (used throughout Sections 3-9) are a merge "
    "of three public, DermNet-style dermatology image collections described in Section 3.4: the \"Human "
    "Skin Diseases (Image)\" Kaggle dataset [24], the \"Eczema Infected + Normal\" Kaggle dataset [37], and "
    "a 17-subtype eczema image archive whose original download link was not recorded (it contributes only "
    "5 unique images). The external SkinDisNet dataset used in Sections 5.5.2-5.5.3 is 1,710 real "
    "clinical smartphone photographs (416 patients) collected at two hospitals in Bangladesh, released "
    "under a CC BY-NC 4.0 (academic-use-only) license by its original publishers [25]; this project uses "
    "SkinDisNet strictly for evaluation and fine-tuned evaluation, consistent with that license, and does "
    "not redistribute the images. All datasets are used as secondary, already-public data: this project "
    "did not collect any new patient images, and ethical review and consent for the original image "
    "collection is the responsibility of each dataset's original publishers. No image carries a patient "
    "name or other direct identifier in the manifests this project works from (Section 3.1).")
h34 = insert_after(prov, H2, "3.4 Source datasets, merge procedure, and comparison classes")
p = insert_after(h34, BODY,
    "Table D lists every source. The eczema class pools all three sources; the seven comparison "
    "(\"other\") classes -- chosen because they can look similar to eczema -- come only from source 1, so "
    "both classes share a photographic source and the photo-source shortcut of the project's first model "
    "(Section 1.3) cannot recur. Every image was checked to decode, and exact and near duplicates were "
    "removed with MD5 hashing plus perceptual hashing (pHash Hamming distance <= 4), both within and "
    "across sources; 714 of the third source's 719 eczema images turned out to be duplicates of the other "
    "two. Any photo appearing under two different labels was dropped from both classes. Because there is "
    "not enough unique eczema data to match the other classes, the other classes were downsampled "
    "proportionally (random seed 42) to match the 1,665 eczema images, giving 3,330 images split "
    "70/15/15 into 2,327 train, 496 validation, and 507 test images (scripts/merge_all_eczema_sources.py).")
src_rows = [
    ["1. Human Skin Diseases (Image), Y. Mohamed, Kaggle [24] -- kaggle.com/datasets/youssefmohmmed/human-skin-diseases-image", "232 eczema + 1,665 other", "Eczema and all 7 comparison classes"],
    ["2. Eczema Infected + Normal, adityush, Kaggle [37] -- kaggle.com/datasets/adityush/eczema2", "1,428 eczema", "Eczema only; its 'Normal' class was dropped (shortcut learning)"],
    ["3. 17-subtype eczema archive (DermNet-style; download link not recorded)", "5 eczema", "11 eczema subtypes kept, 6 distinct diagnoses excluded"],
    ["SkinDisNet, Sultana et al. [25] -- data.mendeley.com/datasets/yj3md44hxg", "1,558 after deduplication (of 1,710 raw)", "Evaluation only: zero-shot on all 1,558 (467 eczema / 1,091 other); fine-tuned split 765 / 298 / 495 (Sections 5.5.2-5.5.3)"],
]
t = new_table(p, ["Source", "Images used", "Role"], src_rows, [3.6, 1.2, 1.7])
sp = insert_after(t, SPACER, "")
cap = insert_after(sp, CAPTION, "Table D. Source datasets and the images each contributed to the 3,330-image set.")
insert_after(cap, BODY,
    "The seven comparison classes and their final counts are: tinea (364), psoriasis (324), rosacea (248), "
    "lichen (220), infestations/bites (209), drug eruption (201), and candidiasis (99). In the SkinDisNet "
    "checks, Eczema and Atopic Dermatitis form the positive class and Contact Dermatitis, Scabies, "
    "Seborrheic Dermatitis and Tinea Corporis the negative class.")

# =====================================================================================
# 7. Metrics: Youden's J + new status definitions
# =====================================================================================
bal = find("Balanced accuracy = (Sensitivity + Specificity) / 2")
insert_after(bal, BULLET,
    "Youden's J = Sensitivity + Specificity - 1 [26] -- 0 for any classifier that performs at chance "
    "(including a constant classifier that always predicts one class), 1 for a perfect classifier; "
    "equivalently J = 2 x balanced accuracy - 1 [27]")
set_text(find("95% confidence intervals on all point metrics"),
    "95% confidence intervals on point metrics use a nonparametric bootstrap [28] (2000 resamples). "
    "Confidence intervals for Youden's J are class-stratified bootstrap intervals computed from each "
    "condition's confusion-matrix counts (2000 resamples, fixed seeds), because per-image predictions "
    "were not stored for every condition. No parametric significance test (e.g. a two-proportion z-test) "
    "was run anywhere in this paper unless explicitly stated, and none is stated for the SE/swish group "
    "comparison (Sections 2.3, 5.2) or the corruption-ranking comparisons (Section 6.1).")
set_text(find("Operational classification used in Tables 5"),
    "Operational classification used in Tables 5, 5b, 5c, and 9 (this replaces an earlier draft's "
    "F1 < 0.05 / Sensitivity-or-Specificity < 20% thresholds, which had no literature basis): "
    "\"collapsed\" if the 95% CI of Youden's J includes 0, i.e. the model cannot be distinguished from "
    "chance [26-28]; otherwise \"backend-stable\" if its accuracy retention R = Acc(INT8) / Acc(FP32) meets "
    "the MLPerf Inference quality target for its model class -- at least 98% of FP32 accuracy for "
    "lightweight image classifiers (MLPerf's MobileNet target) and at least 99% for ResNet18 (MLPerf's "
    "ResNet-50 target) [29]; otherwise \"degraded.\" On the class-imbalanced SkinDisNet splits, R is "
    "computed on balanced accuracy instead of accuracy [27], because a collapsed model that always predicts "
    "the majority class keeps its accuracy there. FP32 models are labelled only collapsed or not.")

# =====================================================================================
# 8. Table 5 and Section 5.5 prose
# =====================================================================================
T = doc.tables  # original tables unchanged in order; our new tables were appended at the end of doc.tables
t5 = [t for t in T if t.rows[0].cells[0].text == "Architecture" and len(t.columns) == 10][0]
for ri, a in enumerate(ARCH, start=1):
    assert t5.rows[ri].cells[0].text.startswith(NAME[a].split("-")[0]), (t5.rows[ri].cells[0].text, a)
    set_cell(t5.rows[ri].cells[8], S[a]["qnnpack"]["st"])
    q = S[a]["qnnpack"]
    set_cell(t5.rows[ri].cells[9], f"J {q['j']:+.2f} [{q['lo']:+.2f}, {q['hi']:+.2f}]")
set_cell(t5.rows[0].cells[9], "qnnpack Youden's J [95% CI]")

set_text(find("With identical trained weights and the same nominal static-quantization procedure"),
    "With identical trained weights and the same nominal static-quantization procedure, INT8 behaviour "
    "differs between the x86/fbgemm and ARM/qnnpack backends. Table 5 reports, per architecture: FP32 "
    "accuracy; fbgemm and qnnpack INT8 accuracy; the backend delta; qnnpack sensitivity, specificity, "
    "and balanced accuracy; this study's status (Section 4.8); and qnnpack Youden's J with its 95% CI. "
    f"On qnnpack, {Q_COLL} architectures collapse to chance ({names(lambda a: S[a]['qnnpack']['st'] == 'collapsed')}), "
    f"{Q_DEG} are above chance but below their MLPerf retention target "
    f"({names(lambda a: S[a]['qnnpack']['st'] == 'degraded')}), and only ResNet18 is backend-stable. "
    f"Applying the same criteria to the fbgemm results gives {F_COLL} collapsed "
    f"({names(lambda a: S[a]['fbgemm']['st'] == 'collapsed')}), {F_DEG} degraded, and {F_STB} backend-stable "
    f"({names(lambda a: S[a]['fbgemm']['st'] == 'backend-stable')}). The backend effect is therefore "
    "clearest for EfficientNet-Lite0 and MobileNetV2-1.0x, which are well above chance on fbgemm "
    f"(J = {S['efficientnet_lite0']['fbgemm']['j']:.2f} and {S['mobilenetv2_100']['fbgemm']['j']:.2f}) but "
    "predict a single class on qnnpack (sensitivity 0%, specificity 100%); EfficientNet-B0 is borderline "
    f"on fbgemm (J = {S['efficientnet_b0']['fbgemm']['j']:.2f}, CI lower bound "
    f"{S['efficientnet_b0']['fbgemm']['lo']:+.3f}) and collapsed on qnnpack. The other two SE/swish "
    "architectures (MobileNetV3-Small, RepGhostNet-0.5x) are at chance on both backends, so their "
    "failure is a quantization-fragility effect rather than a backend effect (Section 5.2). For ResNet18, "
    "accuracy is identical to four decimal places between backends (79.64%), yet sensitivity and "
    "specificity shift in opposite directions (-1.2 pt / +1.2 pt) -- accuracy alone can hide a backend "
    "effect. The 95% bootstrap accuracy intervals of the collapsed architectures (43.8-54.2%) and the "
    "above-chance architectures (69.6-83.1%) do not overlap.")
set_text(find("Table 5. Backend-dependent INT8 behavior"),
    "Table 5. Backend-dependent INT8 behavior, all 9 architectures (validation split). \"Status\" uses "
    "Section 4.8's criteria (Youden's J CI and MLPerf retention target). The rightmost column is qnnpack "
    "Youden's J with its 95% class-stratified bootstrap CI.")

# ---- 5.5.2 zero-shot SkinDisNet
t5b = [t for t in T if t.rows[0].cells[0].text == "Architecture" and len(t.columns) == 6
       and "SkinDisNet" in t.rows[0].cells[1].text][0]
set_cell(t5b.rows[0].cells[5], "qnnpack status")
for ri, a in enumerate(ARCH, start=1):
    set_cell(t5b.rows[ri].cells[5], S[a]["zs"]["qnnpack_int8"]["st"] + " (FP32 already at chance)"
             if S[a]["zs"]["fp32"]["st"] == "collapsed" else S[a]["zs"]["qnnpack_int8"]["st"])
set_text(find("Table 5b. Zero-shot check on SkinDisNet"),
    "Table 5b. Zero-shot check on SkinDisNet, all 9 architectures. Every FP32 model is already at chance "
    "on SkinDisNet (Youden's J 95% CI includes 0) before any quantization, so this table cannot isolate "
    "a quantization effect.")
set_text(find("The 3 architectures that fully collapse on this project's own validation"),
    f"Under Section 4.8's criteria, all {ZS_FP32_CHANCE} FP32 models are already at chance on SkinDisNet "
    "zero-shot (Youden's J between "
    f"{min(S[a]['zs']['fp32']['j'] for a in ARCH):+.3f} and {max(S[a]['zs']['fp32']['j'] for a in ARCH):+.3f}, "
    "every 95% CI including or lying below 0; FP32 F1 0.013-0.100), because the models do not transfer "
    "to this new photographic source at all. A quantized model cannot be judged against an FP32 baseline "
    "that is itself at chance, so this zero-shot check is uninformative about the backend effect for all "
    "9 architectures -- including the three that collapse to F1 = 0.000 on qnnpack, and ShuffleNetV2-0.5x, "
    "whose F1 is 0.036 (FP32) vs. 0.032 (INT8). An earlier draft read the F1 = 0 cases as reproducing the "
    "collapse signature; that reading is withdrawn. Section 5.5.3 removes the confound by fine-tuning on "
    "SkinDisNet first.")

# ---- 5.5.3 fine-tuned SkinDisNet
t5c = [t for t in T if len(t.columns) == 7 and t.rows[0].cells[3].text == "FP32 Status"][0]
for ri, a in enumerate(ARCH, start=1):
    set_cell(t5c.rows[ri].cells[3], "above chance" if S[a]["ft"]["finetuned_fp32"]["st"] != "collapsed" else "at chance")
    set_cell(t5c.rows[ri].cells[6], S[a]["ft"]["finetuned_qnnpack_int8"]["st"])
set_text(find("Table 5c. Fine-tuned check on SkinDisNet"),
    "Table 5c. Fine-tuned check on SkinDisNet (Section 5.5.3), all 9 architectures, evaluated on the new "
    "SkinDisNet fine-tune validation split (298 images; 88 eczema / 210 other). FP32 status is whether the "
    "fine-tuned FP32 model is above chance; INT8 status uses Section 4.8's criteria with retention on "
    "balanced accuracy.")
ft_ok = [a for a in ARCH if S[a]["ft"]["finetuned_fp32"]["st"] != "collapsed"]
ft_coll = [a for a in ARCH if S[a]["ft"]["finetuned_qnnpack_int8"]["st"] == "collapsed"]
ft_stb = [a for a in ARCH if S[a]["ft"]["finetuned_qnnpack_int8"]["st"] == "backend-stable"]
set_text(find("Once each architecture has a genuine, non-collapsed FP32 baseline"),
    f"After fine-tuning, {len(ft_ok)} of 9 FP32 models are above chance on SkinDisNet "
    f"({', '.join(NAME[a] for a in ft_ok)}); "
    f"{', '.join(NAME[a] for a in ARCH if a not in ft_ok)} "
    f"{'remains' if len(ft_ok) == 8 else 'remain'} at chance even after fine-tuning, and MobileNetV3-Small is "
    f"only marginally above chance (J = {S['mobilenetv3_small']['ft']['finetuned_fp32']['j']:.3f}). "
    "The backend-collapse pattern then reproduces cleanly: EfficientNet-B0, EfficientNet-Lite0, and "
    "MobileNetV2-1.0x go from an above-chance FP32 baseline (F1 = 0.31-0.43) to full collapse under qnnpack "
    "INT8 (F1 = 0.000, J = 0) -- a quantization/backend effect on a second dataset, not a restatement of "
    "the zero-shot distribution-shift gap. MobileNetV3-Small and RepGhostNet-0.5x also collapse under INT8, "
    "but their fine-tuned FP32 baselines are at or barely above chance, so this check cannot attribute "
    f"their INT8 collapse to the backend. {', '.join(NAME[a] for a in ft_stb)} are backend-stable on "
    "SkinDisNet (balanced-accuracy retention >= 98%, >= 99% for ResNet18) -- the same four architectures "
    "that are above chance on qnnpack on this project's own validation split (Table 5).")
set_text(find("This bears directly on the feasible candidate set (Section 7.2)"),
    "This bears directly on the feasible candidate set (Section 7.2): all three of its lightweight members "
    "-- ShuffleNetV2-0.5x, ShuffleNetV2-1.0x, and SqueezeNet1.1 -- are backend-stable on SkinDisNet under "
    "this fine-tuned check; ShuffleNetV2-0.5x's F1 rises slightly, from 0.448 (FP32) to 0.502 (INT8). "
    "This is a genuine but limited validation of the candidate set on a second dataset -- limited because "
    "it is one dataset, one fine-tuning recipe, and a validation-split evaluation.")

# =====================================================================================
# 9. Section 6.2, Discussion 7.1/7.2, Figure 2
# =====================================================================================
set_text(find("For ShuffleNetV2-1.0x (backend-stable on both backends"),
    find("For ShuffleNetV2-1.0x (backend-stable on both backends").text.replace(
        "(backend-stable on both backends -- Table 5)",
        "(backend-stable on fbgemm and above chance but degraded on qnnpack -- Table 5)"))
set_text(find("PRIMARY: backend-dependent INT8 behavior"),
    "PRIMARY: INT8 behaviour depends jointly on architecture and backend (Section 5.5) -- with identical "
    f"trained weights, {Q_COLL} of 9 architectures collapse to chance on ARM/qnnpack versus {F_COLL} of 9 "
    "on x86/fbgemm; EfficientNet-Lite0 and MobileNetV2-1.0x collapse only on qnnpack. Scoped to these 9 "
    "architectures and this machine's default recipes, not a general ARM claim.")
set_text(find("SECONDARY: SE/swish architectures show a larger"),
    "SECONDARY: SE/swish architectures show a much larger FP32-to-INT8 accuracy drop on x86/fbgemm "
    "(Section 5.2) and are at or near chance on both backends -- an architecture-family association, not "
    "an isolated causal mechanism.")
set_text(find("Restricting to architectures that are backend-stable on qnnpack"),
    "Removing architectures that collapse to chance on qnnpack (Table 5) leaves four: ResNet18, "
    "ShuffleNetV2-0.5x, ShuffleNetV2-1.0x, and SqueezeNet1.1. Excluding the ResNet18 baseline (42.7 MB; "
    "neither the smallest nor the most accurate of the four) gives a feasible candidate set (Figure 2) of "
    "three lightweight architectures for a wearable/edge target. None of the three reaches the MLPerf 98% "
    "retention target on the validation split (R = "
    f"{S['shufflenet_v2_x0_5']['qnnpack']['r'] * 100:.1f}%, {S['shufflenet_v2_x1_0']['qnnpack']['r'] * 100:.1f}% "
    f"and {S['squeezenet1_1']['qnnpack']['r'] * 100:.1f}% respectively); on the held-out test split "
    f"ShuffleNetV2-0.5x ({S['shufflenet_v2_x0_5']['test']['r'] * 100:.1f}%) and SqueezeNet1.1 "
    f"({S['squeezenet1_1']['test']['r'] * 100:.1f}%) do, while ShuffleNetV2-1.0x "
    f"({S['shufflenet_v2_x1_0']['test']['r'] * 100:.1f}%) does not. This is a candidate set, not a "
    "single recommendation -- Section 7.3 shows why a single winner is not declared within it. All three "
    "remain backend-stable on SkinDisNet after fine-tuning (Section 5.5.3).")
fig_para = P[P.index(find("Figure 2. FP32 dense checkpoint size")) - 1]
for r in fig_para.runs:
    r._r.getparent().remove(r._r)
fig_para.add_run().add_picture(str(FIG2), width=Inches(6.1))
set_text(find("Figure 2. FP32 dense checkpoint size"),
    "Figure 2. FP32 dense checkpoint size (log scale) vs. ARM/qnnpack INT8 accuracy, all 9 architectures "
    "(validation split), coloured by Section 4.8's status. The shaded band marks chance level. The three "
    "lightweight candidates (Section 7.2) are the above-chance points below 10 MB; ResNet18 is excluded "
    "as the non-lightweight baseline.")

# =====================================================================================
# 10. Limitations, Section 9.3, Conclusion, Claims audit
# =====================================================================================
lim_last = find("All deployment-characteristic numbers (size, parameters, latency)")
x = insert_after(lim_last, BULLET,
    "The status criteria (Section 4.8) are literature-backed, but the retention threshold is applied to "
    "a point estimate: SqueezeNet1.1 misses the 98% target by 0.1 pt on validation, so candidates near the "
    "line can change tier between splits (Section 9.3). Pruning knees (Section 4.4) are significance-tested "
    "on the validation split only; they were not re-tested on the held-out test split.")
x = insert_after(x, BULLET,
    "The training data come from a single DermNet-style archive family with image-level (not "
    "patient-level) splits; skin-tone composition was not audited, a gap common to dermatology AI datasets "
    "[35]. One of the three merged sources has no recorded download link (Section 3.4).")
lim_q = insert_after(x, BULLET,
    "QAT was run for only 3 epochs; longer or per-channel QAT schedules were not tested.")

set_text(find("The collapsed/severely-degraded/backend-stable status (Section 4.8) matches"),
    f"Under Section 4.8's criteria the held-out test split gives {T_COLL} collapsed, {T_DEG} degraded, and "
    f"{T_STB} backend-stable architectures (Table 9). The collapsed-versus-above-chance status matches "
    f"between validation (Table 5) and test for {BIN_MATCH} of 9 architectures; the exact three-way status "
    f"matches for {EXACT_MATCH} of 9, because the above-chance architectures sit close to the MLPerf "
    "retention line: ResNet18 drops from backend-stable to degraded, while ShuffleNetV2-0.5x and "
    "SqueezeNet1.1 rise from degraded to backend-stable. This reproduction is reported as support for the PRIMARY "
    "finding (Section 7.1) -- INT8 collapse on qnnpack -- not as validation of every secondary or "
    "exploratory claim, several of which (SE/swish causation, corruption-compression interaction, "
    "SkinDisNet generalization) were not re-tested on this split.")
t9 = [t for t in T if len(t.columns) == 5 and t.rows[0].cells[4].text == "Status (test)"][0]
for ri, a in enumerate(ARCH, start=1):
    set_cell(t9.rows[ri].cells[4], S[a]["test"]["st"])

set_text(find("Across the 9 architectures tested in this paper, INT8 quantization behavior"),
    "Across the 9 architectures tested in this paper, INT8 quantization behaviour depends jointly on "
    f"architecture and backend: {Q_COLL}/9 architectures collapse to chance on ARM/qnnpack versus {F_COLL}/9 "
    "on x86/fbgemm with the same trained weights, two of them (EfficientNet-Lite0, MobileNetV2-1.0x) only on "
    "qnnpack, and only ResNet18 meets its MLPerf retention target on qnnpack. This study does not isolate "
    "a single causal mechanism: a forced-granularity diagnostic (Section 5.5.1) accounts for most of one "
    "architecture's gap, part of a second's, and none of three others', and the SE/swish PTQ pattern "
    "(Section 5.2) remains a correlational observation rather than a demonstrated causal effect.")
set_text(find("For deployment, this means: a candidate set"),
    "For deployment, this means that a candidate set of above-chance, lightweight architectures can be "
    "identified for this task on this machine's default recipes (ShuffleNetV2-0.5x, ShuffleNetV2-1.0x, "
    "SqueezeNet1.1 -- Section 7.2), but no single winner is declared, because ShuffleNetV2-0.5x and -1.0x "
    "are formally Pareto non-dominant (Section 7.3) and this task's differential-diagnosis shape means a "
    "sensitivity-priority tie-break borrowed from screening literature does not transfer. This study did "
    "NOT determine: whether the backend-collapse finding generalizes beyond this one ARM machine and "
    "PyTorch build; whether SE/swish components are causally responsible for quantization fragility; "
    "whether the models generalize zero-shot to new photo sources (they do not yet -- Section 5.5.2); or "
    "any measure of real-world diagnostic performance, clinical utility, or on-device latency and power "
    "draw. The next step is hardware-in-the-loop testing of the candidate set on a phone-class ARM "
    "device. Every finding is scoped to the 9 architectures, the qnnpack and fbgemm default recipes on the "
    "two machines used, and this project's own curated dataset.")
set_text(find("Backend-dependent INT8 collapse for 5 of 9 architectures on qnnpack"),
    f"INT8 collapse to chance for {Q_COLL} of 9 architectures on qnnpack vs. {F_COLL} of 9 on fbgemm "
    f"(Table 5), with the collapsed-vs-above-chance status reproduced for {BIN_MATCH} of 9 architectures on "
    "a final, previously untouched held-out test split (Table 9, Section 9.3).")
set_text(find("The SkinDisNet check (Sections 5.5.2-5.5.3): the zero-shot version"),
    "The SkinDisNet check (Sections 5.5.2-5.5.3): the zero-shot version is uninformative because every "
    "FP32 model is already at chance there. The fine-tuned follow-up reproduces the backend collapse for "
    "EfficientNet-B0, EfficientNet-Lite0 and MobileNetV2-1.0x and backend-stability for the four "
    "architectures that are above chance on this project's validation split, but it is one external "
    "dataset, one fine-tuning recipe and a validation-split evaluation, and it cannot attribute the INT8 "
    "collapse of MobileNetV3-Small and RepGhostNet-0.5x (FP32 at or near chance after fine-tuning).")

# =====================================================================================
# 11. References: fix [24], append [26]-[38]
# =====================================================================================
set_text(find("[24] "),
    "[24] Mohamed, Y. (2025). Human Skin Diseases (Image) [dataset]. Kaggle. "
    "https://www.kaggle.com/datasets/youssefmohmmed/human-skin-diseases-image (accessed 2026). "
    "(Corrected: an earlier draft cited a different Kaggle dataset, haroonalam16/20-skin-diseases-dataset.)")
new_refs = [
    "[26] Youden, W. J. (1950). Index for rating diagnostic tests. Cancer, 3(1), 32-35.",
    "[27] Brodersen, K. H., Ong, C. S., Stephan, K. E., & Buhmann, J. M. (2010). The balanced accuracy and its posterior distribution. ICPR 2010.",
    "[28] Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall/CRC.",
    "[29] Reddi, V. J., et al. (2020). MLPerf Inference Benchmark. ISCA 2020 (arXiv:1911.02549), Table I model-quality targets.",
    "[30] Blalock, D., Gonzalez Ortiz, J. J., Frankle, J., & Guttag, J. (2020). What is the state of neural network pruning? MLSys 2020.",
    "[31] Langan, S. M., Irvine, A. D., & Weidinger, S. (2020). Atopic dermatitis. The Lancet, 396(10247), 345-360.",
    "[32] Laughter, M. R., et al. (2021). The global burden of atopic dermatitis: lessons from the Global Burden of Disease Study 1990-2017. British Journal of Dermatology, 184(2), 304-309.",
    "[33] Bawany, F., Northcott, C. A., Beck, L. A., & Pigeon, W. R. (2021). Sleep disturbances and atopic dermatitis: relationships, methods for assessment, and therapies. J Allergy Clin Immunol Pract, 9(4), 1488-1500.",
    "[34] Chun, K. S., et al. (2021). A skin-conformable wireless sensor to objectively quantify symptoms of pruritus. Science Advances, 7, eabf9405.",
    "[35] Daneshjou, R., et al. (2021). Lack of transparency and potential bias in artificial intelligence data sets and algorithms: a scoping review. JAMA Dermatology, 157(11), 1362-1369.",
    "[36] Ward, A., et al. (2024). Creating an empirical dermatology dataset through crowdsourcing with web search advertisements (SCIN). JAMA Network Open, 7(11), e2446615.",
    "[37] adityush. Eczema Infected + Normal [dataset]. Kaggle. https://www.kaggle.com/datasets/adityush/eczema2 (accessed 2026).",
    "[38] Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. Neural Computation, 10(7), 1895-1923.",
    "[39] McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika, 12(2), 153-157.",
    "[40] Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2), 65-70.",
]

g = find("Global unstructured L1 magnitude pruning")
set_text(g, "Global unstructured L1 magnitude pruning (torch.nn.utils.prune.global_unstructured) removes the "
    "smallest-magnitude weights across all prunable layers jointly, at sparsity targets from 0% to 90% in "
    "10% steps for every architecture. " + KNEE_DEF)
set_text(find("Table 1. FP32 baseline (validation split)"),
    "Table 1. FP32 baseline (validation split) and each architecture's own pruning knee -- the lowest one-shot "
    "unstructured-pruning sparsity whose validation accuracy is significantly below its own unpruned model "
    "(exact McNemar test, Holm-corrected; Section 4.4). Knees are architecture-relative, not comparable in absolute terms.")

# ---- knees: Table 1, Table 3, Sections 5.1 / 5.3, Figure 1
t1 = [t for t in T if t.rows[0].cells[-1].text == "Pruning knee"][0]
for ri, a in enumerate(ARCH, start=1):
    set_cell(t1.rows[ri].cells[-1], f"{round(KNEE[a] * 100)}%")
t3 = [t for t in T if t.rows[0].cells[-1].text == "30% vs. own knee"][0]
for ri, a in enumerate(ARCH, start=1):
    set_cell(t3.rows[ri].cells[-1], "past own knee" if KNEE[a] < 0.3 else ("at own knee" if KNEE[a] == 0.3 else "below own knee"))
set_text(find("Table 1 reports each architecture's FP32 accuracy"),
    "Table 1 reports each architecture's FP32 accuracy, sensitivity, specificity, and AUROC on the validation "
    "split, plus its own significance-tested pruning knee (Section 4.4, Figure 1). Knees range from 20% "
    f"(MobileNetV2-1.0x, adjusted p = {P_AT('mobilenetv2_100', 0.2):.3f}) to 60% (ResNet18, adjusted p = "
    f"{P_AT('resnet18', 0.6):.3f}), confirming that a single fixed sparsity target cannot be assumed \"safe\" "
    "across architectures -- a point that matters directly for Section 5.3.")
set_text(find("Table 3 evaluates a fixed 30% one-shot unstructured sparsity target"),
    "Table 3 evaluates a fixed 30% one-shot unstructured sparsity target combined with x86/fbgemm INT8 "
    "quantization. A 30% target is below the pruning knee (Table 1) for 6 of 9 architectures, exactly at the knee "
    "of SqueezeNet1.1 and ShuffleNetV2-0.5x (both 30%), and ABOVE MobileNetV2-1.0x's knee (20%) -- so "
    "MobileNetV2-1.0x's row reflects an architecture already past its own pruning-induced degradation point, "
    "not a clean isolation of quantization's marginal contribution. Consistent with this, MobileNetV2-1.0x drops "
    "from 78.8% (FP32) to 72.0% with INT8 alone, but to 55.4% combined -- a substantially larger drop than INT8 "
    "alone accounts for, consistent with (not proof of) an added pruning-specific contribution once past its own knee.")
fig1_para = P[P.index(find("Figure 1. Pruning knee (Table 1)")) - 1]
FIG1 = ROOT / "docs" / "pruning_knee_mcnemar_2026-09-23.png"
order_k = sorted(ARCH, key=lambda a: (KNEE[a], NAME[a]))
fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=200)
ax.bar(range(len(order_k)), [KNEE[a] * 100 for a in order_k],
       color=["#96600E" if a in KNEE_CHANGED else "#14213D" for a in order_k])
for k, a in enumerate(order_k):
    ax.text(k, KNEE[a] * 100 + 1.2, f"{round(KNEE[a] * 100)}%", ha="center", fontsize=8)
ax.set_xticks(range(len(order_k)))
ax.set_xticklabels([NAME[a] for a in order_k], rotation=35, ha="right", fontsize=7.5)
ax.set_ylabel("Pruning knee (sparsity, %)")
ax.set_ylim(0, 70)
for sp_ in ("top", "right"):
    ax.spines[sp_].set_visible(False)
fig.tight_layout()
fig.savefig(FIG1)
plt.close(fig)
for r in fig1_para.runs:
    r._r.getparent().remove(r._r)
fig1_para.add_run().add_picture(str(FIG1), width=Inches(6.1))
set_text(find("Figure 1. Pruning knee (Table 1)"),
    "Figure 1. Significance-tested pruning knee (Table 1) by architecture, sorted lowest to highest: the first "
    "sparsity with Holm-adjusted McNemar p < 0.05 against the unpruned model. ShuffleNetV2-0.5x (amber) is the "
    "one knee that differs from the earlier > 5-pt rule (40% -> 30%). Knees are architecture-relative (Section 4.4).")

last = find("[25] ")
for ref in new_refs:
    last = insert_after(last, last, ref)

set_text(find("Margin M = min(Sensitivity, Specificity)"),
    "Margin M = min(Sensitivity, Specificity) -- the worse of the two class-wise rates; used as a single "
    "worst-case number when comparing deployment candidates (Section 7.3)")
set_text(find("The zero-shot SkinDisNet check (Section 5.5.2) is confounded"),
    "The zero-shot SkinDisNet check (Section 5.5.2) is uninformative: every FP32 model is already at "
    "chance on SkinDisNet before quantization. The fine-tuned follow-up (Section 5.5.3) reproduces the "
    "backend collapse for EfficientNet-B0, EfficientNet-Lite0 and MobileNetV2-1.0x and backend-stability "
    "for the three lightweight candidates and ResNet18, but it is limited to one external dataset, one "
    "fine-tuning recipe, and a validation-split (not held-out test) evaluation, and cannot attribute the "
    "INT8 collapse of MobileNetV3-Small and RepGhostNet-0.5x, whose fine-tuned FP32 baselines are at or "
    "barely above chance.")


# =====================================================================================
# 12. REVISION 2026-09-23 (b): the qnnpack "collapse" of MobileNetV2 / EfficientNet-Lite0 is a
#     library defect (relu6 on channels_last quantized tensors), not a backend-recipe effect.
# =====================================================================================
def pct_(x):
    return f"{x * 100:.1f}%"


N_REL = {a: fixv[a]["n_clamp_nodes_fixed"] for a in BUGGY}
set_text(P[1], "Compressing an Eczema Image Classifier for Phone-Based Monitoring: Architecture-Dependent "
               "INT8 Fragility and a Silent ARM Quantization Defect")
set_text(BODY,
    "A wearable eczema monitor needs an image classifier that can tell eczema from visually similar skin "
    "diseases on a phone, which in turn requires compressing the model -- through pruning, INT8 quantization, "
    "or both -- to fit memory, latency, and power budgets. This paper studies what that compression does to an "
    "eczema-vs-seven-look-alike-diseases classifier across nine convolutional architectures (eight lightweight "
    "candidates plus a ResNet18 baseline). It is the image component of a planned wearable eczema-monitoring "
    "system; a multimodal (image + wearable-sensor) design was attempted first but no public dataset pairs the "
    "two for the same patients (Section 1.3). We measure accuracy, sensitivity, specificity, balanced accuracy "
    "and F1 under one-shot magnitude pruning, post-training static INT8 quantization (PTQ), quantization-aware "
    "training (QAT), structured pruning, and pruning-with-fine-tuning, classifying quantized models as "
    "collapsed (95% CI of Youden's J includes zero), degraded, or backend-stable (retaining the MLPerf "
    "Inference quality target of 98-99% of FP32 accuracy). Three findings stand out. First, pruning tolerance "
    "is architecture-specific: significance-tested (McNemar) pruning knees range from 20% to 60% sparsity. "
    "Second, the three architectures built on squeeze-and-excitation and swish-family blocks (EfficientNet-B0, "
    "MobileNetV3-Small, RepGhostNet-0.5x) lose 27.2 pt on average to INT8 PTQ versus 3.9 pt for the other six, "
    "and are at or near chance on both the x86/fbgemm and ARM/qnnpack backends; 3-epoch QAT does not recover "
    "them. Third, two further architectures (MobileNetV2-1.0x, EfficientNet-Lite0) appeared to collapse only on "
    "ARM: a controlled diagnosis (reference-simulated vs. kernel execution, single-block and single-operator "
    "tests) traced this to a defect in PyTorch 2.8's qnnpack backend, whose quantized ReLU6/hardtanh/clamp "
    "kernels return wrong values for the channels-last tensors its own convolutions produce. The models "
    "silently predicted one class for every image -- visible in sensitivity/specificity, not in an exception. "
    "A one-line workaround restores them to within about 1 pt of their x86 accuracy on validation and on the "
    "held-out test split (re-opened once under a declared protocol deviation). After correction, INT8 behaviour "
    "is similar on the two backends, and removing collapsed and non-lightweight models and applying a "
    "size-accuracy Pareto criterion leaves two finalists, ShuffleNetV2-0.5x and SqueezeNet1.1, which are Pareto "
    "non-dominant on both splits. A second dataset (SkinDisNet) is uninformative zero-shot but, after "
    "fine-tuning, confirms both finalists as backend-stable. Every claim is scoped to these 9 architectures, "
    "PyTorch 2.8 default recipes on the two machines used, and this dataset.")

set_text(find("PyTorch's quantized backends (fbgemm for x86, qnnpack for ARM)"),
    "PyTorch's quantized backends (fbgemm for x86, qnnpack for ARM) use different default quantization recipes "
    "-- notably per-channel vs. per-tensor weight quantization and different histogram-observer settings "
    "(Section 4.2) -- and engineering reports have documented accuracy differences between them for some "
    "architectures [19,20]. It is also well established that per-tensor INT8 quantization can destroy depthwise-"
    "separable networks: Krishnamoorthi reports MobileNetV2 falling from 70.9% to 0.1% ImageNet top-1 with "
    "per-layer weight quantization, restored by per-channel quantization [15], and Nagel et al. trace this to "
    "large differences in weight ranges across channels [41]. This paper therefore does not claim that "
    "backend recipes or per-tensor quantization can collapse a network as a new finding; it reports what "
    "happens on an applied eczema task, and documents a separate kernel-level defect (Section 5.5.4).")

g42 = find("Quantization uses PyTorch's FX graph-mode static quantization API")
set_text(g42, g42.text + " Because kernel-level defects can masquerade as quantization effects, every "
    "qnnpack result was additionally checked against PyTorch's reference-quantized model "
    "(convert_to_reference_fx), which simulates the same quantization in floating point independently of the "
    "backend's integer kernels (Section 5.5.4); where they disagreed, the defect was isolated and corrected by "
    "inserting a memory-layout conversion (.contiguous()) before every relu6/hardtanh/clamp operation "
    "(scripts/qnnpack_relu6_layout_fix_2026_09_23.py). The correction changes no quantization parameter.")

# ---- Table 5: corrected qnnpack columns for the two affected architectures
for ri, a in enumerate(ARCH, start=1):
    if a in BUGGY:
        m = bj[a]["qnnpack_backend_this_mac"]["point"]
        fb = bj[a]["x86_fbgemm_backend_original"]["accuracy"]
        set_cell(t5.rows[ri].cells[3], pct_(m["accuracy"]) + "*")
        set_cell(t5.rows[ri].cells[4], f"{(fb - m['accuracy']) * 100:+.1f} pt")
        set_cell(t5.rows[ri].cells[5], pct_(m["sensitivity_recall"]))
        set_cell(t5.rows[ri].cells[6], pct_(m["specificity"]))
        set_cell(t5.rows[ri].cells[7], pct_((m["sensitivity_recall"] + m["specificity"]) / 2))
set_text(find("Table 5. Backend-dependent INT8 behavior"),
    "Table 5. INT8 behaviour on the two CPU backends, all 9 architectures (validation split). Status uses "
    "Section 4.8's criteria; the rightmost column is qnnpack Youden's J with its 95% class-stratified bootstrap "
    "CI. * qnnpack result after correcting the relu6 channels-last defect (Section 5.5.4); the uncorrected "
    "pipeline gave 49.8% (sensitivity 0%, specificity 100%) for both.")
set_text(find("With identical trained weights and the same nominal static-quantization procedure"),
    "With identical trained weights and the same nominal static-quantization procedure, Table 5 compares INT8 "
    "behaviour on x86/fbgemm and ARM/qnnpack. After correcting a qnnpack kernel defect that affected two "
    f"architectures (Section 5.5.4), {Q_COLL} architectures are at chance on qnnpack "
    f"({names(lambda a: S[a]['qnnpack']['st'] == 'collapsed')}), {Q_DEG} are above chance but below their MLPerf "
    f"retention target ({names(lambda a: S[a]['qnnpack']['st'] == 'degraded')}), and only ResNet18 is "
    f"backend-stable. On fbgemm the same criteria give {F_COLL} collapsed "
    f"({names(lambda a: S[a]['fbgemm']['st'] == 'collapsed')}), {F_DEG} degraded, and {F_STB} backend-stable "
    f"({names(lambda a: S[a]['fbgemm']['st'] == 'backend-stable')}). The two backends therefore behave similarly: "
    "the architectures that fail, fail on both (the SE/swish family, Section 5.2; EfficientNet-B0 is borderline "
    f"on fbgemm, J = {S['efficientnet_b0']['fbgemm']['j']:.2f}, CI lower bound {S['efficientnet_b0']['fbgemm']['lo']:+.3f}), "
    "and the remaining differences are small retention shifts of 1-5 pt that move ShuffleNetV2-1.0x and "
    "SqueezeNet1.1 across the 98% line. For ResNet18, accuracy is identical to four decimal places between "
    "backends (79.64%), yet sensitivity and specificity shift in opposite directions (-1.2 pt / +1.2 pt) -- "
    "accuracy alone can hide a backend effect.")
set_text(find("Table 5a. Diagnostic: re-quantizing on qnnpack"),
    "Table 5a. Diagnostic: re-quantizing on qnnpack with weight quantization forced to per-channel (qnnpack's "
    "own default is per-tensor; Section 4.2), for the 5 architectures that failed on qnnpack before the "
    "correction. EfficientNet-Lite0 and MobileNetV2-1.0x were run without the relu6 correction, so their "
    "per-channel results are affected by the defect of Section 5.5.4.")
set_text(find("As stated in Section 4.2, the fbgemm-vs-qnnpack comparison in Table 5"),
    "The fbgemm-vs-qnnpack comparison is not a controlled ablation of 'the backend' alone: the default recipes "
    "differ in weight granularity (per-channel vs. per-tensor) and activation-observer range. The forced-per-"
    "channel diagnostic (Table 5a) separates granularity: it substantially rescues EfficientNet-B0 on qnnpack "
    "(F1 0.080 to 0.649) and partly rescues MobileNetV3-Small (F1 0.165), consistent with the known sensitivity "
    "of depthwise/SE networks to per-tensor weight scales [15, 41], and does not rescue RepGhostNet-0.5x. Its "
    "apparent failure to rescue EfficientNet-Lite0 and MobileNetV2-1.0x -- which contradicted the literature, "
    "where per-channel quantization restores MobileNetV2 [15, 41] -- is what led to the defect found in "
    "Section 5.5.4: those two models were failing for a reason unrelated to granularity.")

set_text(find("After fine-tuning, "),
    find("After fine-tuning, ").text.split("The backend-collapse pattern")[0] +
    "Under qnnpack INT8 (with the relu6 correction applied), EfficientNet-B0 goes from an above-chance FP32 "
    "baseline (F1 0.39) to a one-class predictor (F1 = 0.000, J = 0) -- a genuine quantization failure on a "
    "second dataset. EfficientNet-Lite0 remains at chance even after the correction (F1 "
    f"{fixf['efficientnet_lite0']['fixed']['f1']:.3f}) and MobileNetV2-1.0x is degraded (F1 "
    f"{fixf['mobilenetv2_100']['fixed']['f1']:.3f}); both were one-class predictors (F1 = 0.000) before the "
    "correction. MobileNetV3-Small and RepGhostNet-0.5x also collapse, but their fine-tuned FP32 baselines are at "
    f"or barely above chance, so this check cannot attribute their INT8 collapse. {', '.join(NAME[a] for a in ft_stb)} "
    "are backend-stable on SkinDisNet (balanced-accuracy retention >= 98%, >= 99% for ResNet18).")
t5c_ = [t for t in T if len(t.columns) == 7 and t.rows[0].cells[3].text == "above chance" or
        (len(t.columns) == 7 and t.rows[0].cells[0].text == "Architecture" and "FP32 F1" in t.rows[0].cells[2].text)][0]
for ri, a in enumerate(ARCH, start=1):
    if a in BUGGY:
        m = fixf[a]["fixed"]
        set_cell(t5c_.rows[ri].cells[4], pct_(m["accuracy"]) + "*")
        set_cell(t5c_.rows[ri].cells[5], f"{m['f1']:.3f}*")
set_text(find("Table 5c. Fine-tuned check on SkinDisNet"),
    find("Table 5c. Fine-tuned check on SkinDisNet").text + " * after the relu6 correction (Section 5.5.4); "
    "uncorrected, both were 70.5% accuracy with F1 = 0.000.")
set_text(find("This bears directly on the feasible candidate set (Section 7.2)"),
    "This bears directly on the deployment finalists (Section 7.2): ShuffleNetV2-0.5x and SqueezeNet1.1 are both "
    "backend-stable on SkinDisNet under this fine-tuned check (ShuffleNetV2-0.5x F1 0.448 to 0.502; SqueezeNet1.1 "
    "0.434 to 0.433). This is a genuine but limited validation on a second dataset -- one dataset, one "
    "fine-tuning recipe, and a validation-split evaluation.")

# ---- new Section 5.5.4: the defect
anchor = find("This bears directly on the deployment finalists")
h554 = insert_after(anchor, H2, "5.5.4 Root cause of the apparent ARM-only collapse: a quantized ReLU6 layout defect")
p1 = insert_after(h554, BODY,
    "Before correction, MobileNetV2-1.0x and EfficientNet-Lite0 were the only architectures that were clearly "
    "above chance on fbgemm but predicted a single class on qnnpack (sensitivity 0%, specificity 100%), which an "
    "earlier draft of this paper reported as its central 'backend-dependent collapse' finding. Four controlled "
    "steps show that this was a software defect, not a property of the models or of quantization. (1) The same "
    "calibrated MobileNetV2 scores 71-76% when converted to PyTorch's reference-quantized model, which simulates "
    "the quantization in floating point, but 49.8% when converted to qnnpack kernels -- under the qnnpack default "
    "qconfig, forced per-channel weights, and the exact fbgemm qconfig alike. Weight-only per-channel fake "
    "quantization keeps 78.0% (FP32 78.8%), matching the literature [15, 41]. (2) Quantizing any single one of "
    "MobileNetV2's 19 feature blocks, with the rest in FP32, already makes qnnpack diverge from the reference "
    "(mean logit error 1.5-4.6 vs. 0.03-0.44). (3) Single-layer tests are exact on qnnpack for convolution, "
    "depthwise convolution, strided convolution, conv+BN+ReLU and linear layers, but conv+BN+ReLU6 has 141% "
    "relative error. (4) The quantized relu6, hardtanh and clamp kernels are exact on NCHW tensors but return "
    "wrong values (maximum error 6.0 on a 0-6 range) on channels-last tensors -- and qnnpack's quantized "
    "convolution outputs channels-last tensors. Plain ReLU is unaffected because it is fused into the "
    "convolution, which is why ResNet18, ShuffleNetV2 and SqueezeNet were never affected. MobileNetV2 contains "
    f"{N_REL['mobilenetv2_100']} such ReLU6 nodes and EfficientNet-Lite0 {N_REL['efficientnet_lite0']}; no other "
    "architecture in this study contains any.")
p2 = insert_after(p1, BODY,
    "Inserting .contiguous() before every relu6/hardtanh/clamp node of the converted model -- a mechanical graph "
    "rewrite that changes no quantization parameter -- restores both models (Table 5d) to within about 1 pt of "
    "their fbgemm accuracy. The held-out test split was re-evaluated once for all 9 architectures under a "
    "protocol deviation written before re-opening it (FROZEN_TEST_PROTOCOL_DEVIATION_2026-09-23.md); the "
    "correction is a no-op for the 7 unaffected architectures. The practical lesson is that a quantized medical "
    "image model can fail silently on a deployment backend -- no exception, plausible-looking accuracy near 50% "
    "on a balanced set -- and that class-wise metrics plus a reference-vs-kernel comparison catch it where "
    "accuracy alone does not. The defect was observed on PyTorch 2.8.0 (macOS, ARM64); whether other builds or "
    "mobile runtimes share it was not tested.")
rows5d = []
for a in BUGGY:
    o, f = ORIG_BUG[a], fixv[a]["fixed"]
    oc, fc = o["test_cm"], fixt[a]["fixed"]["confusion_matrix"]
    acc_t = lambda c: (c["tp"] + c["tn"]) / sum(c.values())
    rows5d.append([NAME[a], pct_(o["val"]["accuracy"]), pct_(f["accuracy"]),
                   pct_(acc_t(oc)), pct_(acc_t(fc)),
                   pct_(bj[a]["x86_fbgemm_backend_original"]["accuracy"]),
                   f"{o['ft']['f1']:.3f} -> {fixf[a]['fixed']['f1']:.3f}"])
t5d = new_table(p2, ["Architecture", "qnnpack val (before)", "qnnpack val (fixed)", "qnnpack test (before)",
                     "qnnpack test (fixed)", "fbgemm val", "SkinDisNet FT F1"], rows5d,
                [1.3, 0.85, 0.85, 0.85, 0.85, 0.75, 1.0])
sp5d = insert_after(t5d, SPACER, "")
insert_after(sp5d, CAPTION, "Table 5d. Effect of the relu6 channels-last correction on the two affected architectures. "
             "'Before' values are sensitivity 0% / specificity 100% one-class predictors.")

set_text(find("For ShuffleNetV2-1.0x (backend-stable on fbgemm"),
    find("For ShuffleNetV2-1.0x (backend-stable on fbgemm").text.replace(
        "but this is an artifact of Table 5's backend collapse",
        "but this is an artifact of those models already being one-class predictors (for MobileNetV2-1.0x, "
        "because of the defect in Section 5.5.4)"))

set_text(find("PRIMARY: INT8 behaviour depends jointly"),
    "PRIMARY: INT8 PTQ fragility is architecture-dependent and largely backend-independent after correction "
    f"(Sections 5.2, 5.5) -- {Q_COLL} of 9 architectures are at chance on ARM/qnnpack and {F_COLL} (plus one "
    "borderline) on x86/fbgemm, all from the SE/swish family.")
set_text(find("SECONDARY: SE/swish architectures show a much larger"),
    "SECONDARY: a silent qnnpack defect (quantized ReLU6 on channels-last tensors) turned MobileNetV2-1.0x and "
    "EfficientNet-Lite0 into one-class predictors; it was found only through class-wise metrics and a "
    "reference-vs-kernel comparison (Section 5.5.4).")
set_text(find("SECONDARY: weight-quantization granularity"),
    "SECONDARY: forcing per-channel weight quantization rescues EfficientNet-B0 on qnnpack and partly rescues "
    "MobileNetV3-Small, consistent with the known per-tensor sensitivity of depthwise/SE networks [15, 41].")
set_text(find("SECONDARY: fine-tuning each architecture on SkinDisNet"),
    "SECONDARY: after fine-tuning on SkinDisNet, EfficientNet-B0 still collapses under INT8 while both "
    "deployment finalists remain backend-stable (Section 5.5.3).")

above = [a for a in ARCH if S[a]["qnnpack"]["st"] != "collapsed"]
light = [a for a in above if a != "resnet18"]
def front(key):
    acc = {a: (S[a][key]["acc"] if "acc" in S[a][key] else None) for a in light}
    return [a for a in light if not any(INT8_MB[b] <= INT8_MB[a] and acc[b] >= acc[a] and
                                        (INT8_MB[b] < INT8_MB[a] or acc[b] > acc[a]) for b in light if b != a)]
S_test_acc = {}
for a in ARCH:
    c = tj[a]["qnnpack_int8"]["confusion_matrix"]
    S[a]["test"]["acc"] = (c["tp"] + c["tn"]) / sum(c.values())
FV, FT = front("qnnpack"), front("test")
assert FV == FT == ["shufflenet_v2_x0_5", "squeezenet1_1"], (FV, FT)
set_text(find("Removing architectures that collapse to chance on qnnpack"),
    f"Removing architectures that are at chance on qnnpack (Table 5) leaves {len(above)}: "
    f"{', '.join(NAME[a] for a in above)}. Excluding the ResNet18 baseline (42.7 MB FP32) leaves {len(light)} "
    "lightweight models. Comparing their INT8 model size against qnnpack INT8 accuracy, only two are Pareto-optimal "
    "-- ShuffleNetV2-0.5x (0.64 MB) and SqueezeNet1.1 (0.86 MB) -- on both the validation and the held-out test "
    "split; ShuffleNetV2-1.0x, MobileNetV2-1.0x and EfficientNet-Lite0 are each dominated by SqueezeNet1.1, which "
    "is smaller and more accurate. These two finalists are compared in Section 7.3. Neither reaches the MLPerf "
    f"98% retention target on validation (R = {S['shufflenet_v2_x0_5']['qnnpack']['r'] * 100:.1f}% and "
    f"{S['squeezenet1_1']['qnnpack']['r'] * 100:.1f}%); both do on the test split "
    f"({S['shufflenet_v2_x0_5']['test']['r'] * 100:.1f}% and {S['squeezenet1_1']['test']['r'] * 100:.1f}%), and "
    "both remain backend-stable on SkinDisNet after fine-tuning (Section 5.5.3). INT8 sizes are serialized INT8 "
    "checkpoints; accuracy gaps of 1-2 pt between these models are within sampling uncertainty.")
set_text(find("Figure 2. FP32 dense checkpoint size (log scale) vs. ARM/qnnpack"),
    "Figure 2. FP32 dense checkpoint size (log scale) vs. ARM/qnnpack INT8 accuracy (after the relu6 correction), "
    "all 9 architectures (validation split), coloured by Section 4.8's status. The shaded band marks chance level.")

h73 = find("7.3 ShuffleNetV2-0.5x vs. 1.0x")
set_text(h73, "7.3 ShuffleNetV2-0.5x vs. SqueezeNet1.1: a Pareto non-dominance")
t8 = [t for t in T if t.rows[0].cells[0].text.startswith("Metric (qnnpack INT8)")][0]
A5, SQ = "shufflenet_v2_x0_5", "squeezenet1_1"
set_cell(t8.rows[0].cells[1], "ShuffleNetV2-0.5x"); set_cell(t8.rows[0].cells[2], "SqueezeNet1.1")
v5, vq = bj[A5]["qnnpack_backend_this_mac"]["point"], bj[SQ]["qnnpack_backend_this_mac"]["point"]
rows8 = [("Accuracy", v5["accuracy"], vq["accuracy"]), ("Sensitivity", v5["sensitivity_recall"], vq["sensitivity_recall"]),
         ("Specificity", v5["specificity"], vq["specificity"]),
         ("Margin M = min(Sens., Spec.)", min(v5["sensitivity_recall"], v5["specificity"]), min(vq["sensitivity_recall"], vq["specificity"]))]
for ri, (lab, x5, xq) in enumerate(rows8, start=1):
    set_cell(t8.rows[ri].cells[0], lab); set_cell(t8.rows[ri].cells[1], pct_(x5)); set_cell(t8.rows[ri].cells[2], pct_(xq))
    set_cell(t8.rows[ri].cells[3], "ShuffleNetV2-0.5x" if x5 > xq else "SqueezeNet1.1")
set_cell(t8.rows[5].cells[0], "INT8 model size"); set_cell(t8.rows[5].cells[1], f"{INT8_MB[A5]:.2f} MB")
set_cell(t8.rows[5].cells[2], f"{INT8_MB[SQ]:.2f} MB"); set_cell(t8.rows[5].cells[3], "ShuffleNetV2-0.5x")
set_text(find("Table 8. ShuffleNetV2-0.5x vs. 1.0x on qnnpack"),
    "Table 8. ShuffleNetV2-0.5x vs. SqueezeNet1.1 on qnnpack INT8 (validation split): neither Pareto-dominates the "
    "other. The same split of wins holds on the held-out test split.")
set_text(find("1.0x wins on accuracy and sensitivity; 0.5x wins"),
    "SqueezeNet1.1 wins on accuracy, sensitivity and worst-case margin; ShuffleNetV2-0.5x wins on specificity and "
    "size (0.64 vs. 0.86 MB INT8). Neither is at least as good on every metric [30], on validation or on test, "
    "so this is a formal Pareto non-dominance. Resolving it requires an explicit weighting of sensitivity vs. "
    "specificity for THIS task, and that weighting does not transfer from unrelated literature. Sensitivity-first "
    "framings common in skin-cancer screening (e.g. [7]) assume a missed positive risks a missed malignancy. This "
    "task is eczema vs. OTHER SKIN CONDITIONS that can look similar, several of which (e.g. tinea corporis) can be "
    "worsened by a treatment appropriate for eczema -- 'tinea incognito', where topical corticosteroids mask or "
    "worsen a fungal infection [21]. A false positive is therefore not obviously cheaper than a false negative, "
    "and without a task-specific cost model the non-dominance is left unresolved rather than forced.")
set_text(find("One narrower argument that does transfer"),
    "One narrower argument that does transfer: if the deployment target is memory- or power-constrained enough "
    "that a 1.3x INT8 size difference (0.64 vs. 0.86 MB) matters operationally -- plausible for a microcontroller-"
    "class wearable, where on-board memory is a documented hard constraint [22] -- ShuffleNetV2-0.5x's size "
    "advantage is a narrowly-scoped reason to prefer it; on a phone, where both fit easily, SqueezeNet1.1's "
    "higher sensitivity and margin may matter more. This is offered as a conditional resolution, not a "
    "general recommendation.")

x2 = insert_after(lim_q, BULLET,
    "Two architectures' qnnpack results required a correction for a PyTorch 2.8.0 kernel defect (Section 5.5.4); "
    "the correction is a workaround, and other operators or builds may have similar layout defects that this "
    "study's reference-vs-kernel check would only catch for the configurations it ran.")
insert_after(x2, BULLET,
    "INT8 calibration draws a random 256-image subset of the training split; re-running calibration with a "
    "different draw moved the unaffected models' qnnpack accuracy by up to about 1.4 pt, enough to move "
    "architectures near the 98% retention line between 'degraded' and 'backend-stable'.")

set_text(find("FROZEN_TEST_PROTOCOL_2026-09-20.md specifies exactly"),
    find("FROZEN_TEST_PROTOCOL_2026-09-20.md specifies exactly").text + " Declared deviation: after the relu6 "
    "defect was found (Section 5.5.4), the test split was re-opened once to re-run the qnnpack INT8 evaluation for "
    "all 9 architectures with and without the correction, under a written protocol committed before re-opening "
    "(FROZEN_TEST_PROTOCOL_DEVIATION_2026-09-23.md). No threshold, architecture or calibration choice changed.")
for ri, a in enumerate(ARCH, start=1):
    if a in BUGGY:
        m = fixt[a]["fixed"]["point"]
        set_cell(t9.rows[ri].cells[2], pct_(m["accuracy"]) + "*")
        set_cell(t9.rows[ri].cells[3], f"{m['f1']:.3f}*")
set_text(find("Table 9. Final held-out test-split evaluation"),
    "Table 9. Final held-out test-split evaluation, all 9 architectures. * qnnpack result after the relu6 "
    "correction, from the declared re-evaluation (Section 9.1); the original run gave 50.5% (F1 = 0.000) for both.")
set_text(find("Under Section 4.8's criteria the held-out test split gives"),
    f"Under Section 4.8's criteria the held-out test split gives {T_COLL} collapsed, {T_DEG} degraded, and "
    f"{T_STB} backend-stable architectures (Table 9, corrected). The collapsed-versus-above-chance status matches "
    f"between validation and test for {BIN_MATCH} of 9 architectures and the exact three-way status for "
    f"{EXACT_MATCH} of 9; the architectures near the MLPerf retention line move between degraded and "
    "backend-stable (ResNet18 down; ShuffleNetV2-0.5x and SqueezeNet1.1 up). The deployment Pareto front "
    "(Section 7.2) is identical on the two splits. Secondary and exploratory claims (SE/swish causation, "
    "corruption-compression interaction, SkinDisNet generalization, pruning knees) were not re-tested on test.")
set_text(find("Across the 9 architectures tested in this paper, INT8 quantization behaviour depends jointly"),
    "For an eczema-vs-look-alike classifier headed for a phone, compression is feasible but not uniformly: "
    "pruning tolerance varies from 20% to 60% sparsity across architectures, and the three squeeze-and-"
    f"excitation/swish architectures do not survive default INT8 PTQ on either backend ({Q_COLL}/9 at chance on "
    "ARM after correction), which 3 epochs of QAT does not fix. The apparent ARM-only collapse of MobileNetV2-1.0x "
    "and EfficientNet-Lite0 was a PyTorch kernel defect, found only because class-wise metrics exposed one-class "
    "behaviour and a reference-vs-kernel comparison localized it; corrected, both behave as on x86.")
set_text(find("For deployment, this means that a candidate set"),
    "For deployment, two finalists emerge on both validation and test -- ShuffleNetV2-0.5x (smallest) and "
    "SqueezeNet1.1 (most accurate lightweight model) -- and they are Pareto non-dominant, so no single winner is "
    "declared without a task-specific cost model (Section 7.3). Beyond the numbers, the study argues that "
    "deploying a medical image model to a phone should include (i) class-wise metrics, not accuracy alone, and "
    "(ii) a check of the deployed kernels against a reference simulation. This study did NOT determine: whether "
    "the defect exists in other PyTorch builds or mobile runtimes; whether SE/swish components are causally "
    "responsible for quantization fragility; whether the models generalize zero-shot to new photo sources (they "
    "do not yet); or real-world diagnostic performance, clinical utility, or on-device latency and power. The next "
    "step is hardware-in-the-loop testing of the finalists on a phone.")
set_text(find("INT8 collapse to chance for"),
    f"INT8 PTQ failure of the SE/swish-family architectures on both backends ({Q_COLL}/9 at chance on qnnpack, "
    f"{F_COLL}/9 plus one borderline on fbgemm; Table 5), reproduced on the held-out test split (Table 9).")
set_text(find("Backend-dependent INT8 collapse reproducing on a second"),
    "The qnnpack relu6 channels-last defect and its effect (Section 5.5.4): isolated by reference-vs-kernel "
    "comparison, single-block and single-operator tests, and corrected with a no-op-for-other-models graph "
    "rewrite, confirmed on validation, test and SkinDisNet.")
set_text(find("Weight-quantization granularity accounting"),
    "Per-channel weight quantization rescuing EfficientNet-B0 on qnnpack (Table 5a), consistent with prior work "
    "[15, 41].")
set_text(find("The SkinDisNet check (Sections 5.5.2-5.5.3): the zero-shot version is uninformative"),
    "The SkinDisNet check (Sections 5.5.2-5.5.3): uninformative zero-shot (every FP32 model at chance); after "
    "fine-tuning it confirms EfficientNet-B0's INT8 collapse and both finalists' backend-stability, but it is one "
    "external dataset, one fine-tuning recipe, and a validation-split evaluation.")
set_text(find("Whether the backend-collapse finding generalizes"),
    "Whether the relu6 layout defect exists in other PyTorch versions, ARM devices, or mobile runtimes was not "
    "tested and is not claimed.")
set_text(find("Superlative novelty language"),
    "Novelty is stated narrowly. That per-tensor INT8 quantization can destroy depthwise-separable networks, and "
    "that per-channel quantization restores them, is established [15, 41]; that fbgemm and qnnpack can disagree "
    "is documented [19, 20]. This paper's contributions are applied: a 9-architecture compression study on an "
    "eczema-vs-look-alike task with class-wise, significance-tested and held-out evaluation; and the diagnosis of "
    "a silent qnnpack kernel defect that masquerades as a backend effect, with a reproduction and workaround.")
set_text(find("The backend-dependent collapse (Section 5.5) is characterized on one ARM machine"),
    "The backend comparison (Section 5.5) and the relu6 defect (Section 5.5.4) are characterized on one ARM "
    "machine's PyTorch 2.8.0 qnnpack build; no second ARM device, PyTorch build or mobile runtime was tested.")
set_text(find("The zero-shot SkinDisNet check (Section 5.5.2) is uninformative"),
    "The zero-shot SkinDisNet check (Section 5.5.2) is uninformative: every FP32 model is already at chance on "
    "SkinDisNet before quantization. The fine-tuned follow-up (Section 5.5.3) shows EfficientNet-B0's INT8 "
    "collapse and backend-stability for both finalists and ResNet18, but it is limited to one external dataset, "
    "one fine-tuning recipe, and a validation-split evaluation, and cannot attribute the INT8 collapse of "
    "MobileNetV3-Small and RepGhostNet-0.5x, whose fine-tuned FP32 baselines are at or barely above chance.")
last41 = find("[25] ")

lastref = [p_ for p_ in doc.paragraphs if p_.text.startswith("[40] ")][0]
insert_after(lastref, lastref, "[41] Nagel, M., van Baalen, M., Blankevoort, T., & Welling, M. (2019). Data-free "
             "quantization through weight equalization and bias correction. ICCV 2019.")
doc.save(OUT)
print("Saved:", OUT)
