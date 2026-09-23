# Presentation Changelog — 2026-09-20

`edge_ai_eval_presentation_FINAL.pptx` (built by `scripts/build_eval_presentation_FINAL.py`)
replaces `edge_ai_eval_presentation_2026-09-19.pptx`, which was built for the superseded
"Compression Robustness Is Architecture-Dependent" draft and predates the current paper's
central finding (backend-dependent INT8 collapse) entirely. This is a full rebuild, not an
edit pass — the old deck's headline finding (SE/swish quantization fragility on one x86
backend) is now a SECONDARY finding; the PRIMARY finding (backend-dependent collapse) did
not exist in the old deck at all. Every number below was re-verified against this
project's own JSON result files or the current paper in this session before being used.

Slide count: 14 → 14 (coincidental — content is almost entirely new).

| # | Old slide | What changed | Why |
|---|---|---|---|
| 1 | Title: "Compression Robustness Is Architecture-Dependent" / "On-Device Skin-Lesion Classification" | Retitled to "Backend-Dependent INT8 Quantization Behavior in an Eczema Image Classifier" / "A 9-Architecture Study for Edge and Wearable Healthcare Deployment" | Old title reflected the superseded generic lightweight-CNN framing and a generic "skin-lesion" label, not this project's eczema-specific, deployment-scoped current paper. |
| 2 | Motivation citing a separate "companion report" architecture selection | Rewritten around the current paper's 5 research questions (RQ1–RQ5) | Old framing depended on a different document's conclusions not carried by the current paper. |
| 3 | Dataset + architecture table (numbers unchanged) | Kept the FP32 accuracy / size / SE-swish table (verified unchanged against `edge_simulation_all_architectures_2026-09-19.json` and `edge_ai_extended_analysis_2026-09-19.json`) but replaced "test set never opened for this exploratory sweep" | The old caption is now false — Section 9 of the current paper opens the test set exactly once. Dataset counts (3,330 / 2,327 train / 496 val / 507 test) verified directly against `SkinDisease/manifest_curated_v3_{train,val,test}.csv` row counts. |
| 4 | Methodology slide listing only unstructured pruning and "one quantization backend (x86/onednn)" | Rebuilt as a pipeline diagram naming unstructured pruning, structured/channel pruning, pruning+fine-tune, PTQ, QAT, and both x86/fbgemm and ARM/qnnpack | The old claims ("only unstructured pruning tested," "only one quantization backend") are both false for the current paper: structured pruning ran for 7/9 architectures, and the qnnpack-vs-fbgemm comparison is this paper's central experiment. |
| 5 | Pruning-knee table using a ">5pt drop" definition, 6 of 9 architectures shown | Rebuilt as a 9-architecture bar chart using the CURRENT >2pt-drop knee definition | The old knee definition and the old per-architecture knee values are both superseded; the current paper defines knee as the first sparsity level with a >2pt drop, and reports it for all 9 architectures, not 6. |
| 6 | "The Headline Finding" — SE/swish quantization fragility, framed as this paper's main result, with **9.3×**, **26.3 pt vs. 2.8 pt** | Reframed as **Finding 2** (secondary), with the current numbers **27.2% vs. 3.9%, a 23.3-point difference, ≈7.0×**, and an explicit "association, not demonstrated causation" statement | The old ratio/mean-drop numbers are stale (recomputed since). More importantly, the old deck stated this as the paper's central result; the current paper's central result is the backend-dependent collapse (new Slide 7), and this finding is explicitly downgraded to an association, not a cause, per the current paper's Sections 2.3 and 5.2. |
| 7 | *(did not exist)* | **New slide** — the primary finding: backend-dependent INT8 collapse, full 9-architecture fbgemm-vs-qnnpack table, 3/9 collapsed · 2/9 severely degraded · 4/9 backend-stable · 5/9 (55.6%) fail criterion | This is the current paper's actual central finding (Section 5.5) and had no equivalent anywhere in the old deck. |
| 8 | *(did not exist)* | **New slide** — forced per-channel quantization diagnostic, explaining part (not all) of the backend gap for 2 of 5 affected architectures | Directly supports the current paper's methodology audit (Section 5.5.1) and prevents the deck from implying backend alone is a complete, controlled explanation. |
| 9 | Old Slide 3 ("Related Work") and part of old Slide 8 ("Interpretation") — framed SE-block presence as "the operative cause" of quantization collapse; no QAT or fine-tuning content existed | Replaced with QAT results (3 architectures, PTQ vs. after 3-epoch QAT — none rescued) and pruning+fine-tune results (3 architectures × 3 sparsities — 2 of 3 recover fully, MobileNetV3-Small does not at 70%) | The old causal SE-block language is removed. QAT and fine-tune-after-pruning are both real experiments in the current paper (Sections 5.4, 5.7) that had no representation in the old deck at all. |
| 10 | Old Slide 10 ("Discussion") — framed ShuffleNetV2-1.0x's earlier selection as reinforced/validated by this study | Replaced with a "candidate set, not a single winner" framing: 4/9 qnnpack-backend-stable architectures, 3 lightweight candidates | The old framing already leaned toward declaring 1.0x the answer; the current paper explicitly declines to declare a winner (Section 7.2–7.3). |
| 11 | *(did not exist)* | **New slide** — ShuffleNetV2-0.5x vs. 1.0x formal Pareto non-dominance table (accuracy, sensitivity, specificity, margin, size) | The current paper's Table 8 / Section 7.3 result — a genuine multi-objective non-dominance — was entirely absent from the old deck, which instead treated 1.0x as already-validated (see row above). |
| 9 (old) | "Result 3 — Honest Limitation": CPU timing not a valid latency proxy | Folded into Slide 13's limitations list ("no real target-device latency or power measurement") rather than given its own slide | Still true and still stated, but condensed to fit the current paper's structure without dedicating a full slide to a point already covered in Limitations. |
| 12 | *(did not exist as one slide)* | **New slide** — robustness checks in three panels: image corruption (exploratory), SkinDisNet zero-shot (not general external validation), and the frozen 507-image held-out test (8/9 match) | The current paper's Sections 6, 5.5.2, and 9 had no presentation coverage in the old deck at all. |
| 13 | Old Slides 11–12 ("Limitations", "Conclusion") | Rewritten limitations list (added: one ARM machine, SE/swish causality unresolved, SkinDisNet distribution shift, structured-pruning tooling exclusion) and a 5-point conclusion matching the current paper's Section 10 | Old limitations list matched the old paper's scope (single backend, no structured pruning) — both now false as limitations, since both were subsequently tested. |
| 14 | Thank you | Unchanged in substance; paper filename and companion-doc references updated | Cosmetic only. |
| 13 (old) | References slide (citations, companion-report link, old paper filename) | Removed as a standalone slide; the current paper's filename is referenced on the Thank You slide instead | The rebuilt deck is shorter and citation-heavy references belong in the paper itself, not a 14-slide evaluation deck; per the brief's suggested structure, references were not requested as a separate slide. |

## Final cross-check against the brief's specific confirmation list

- No `9.3×` anywhere — confirmed by text search. ✅
- No `26.3` / `2.8 pt` anywhere — confirmed. Current numbers (27.2% / 3.9% / 23.3 pt / ≈7.0×) used throughout (Slide 6). ✅
- No `>5pt` pruning-knee definition anywhere — confirmed. Slide 5 states the current `>2pt` definition verbatim. ✅
- No claim that test data was never opened — confirmed. Slide 4 explicitly states the opposite ("not 'never opened'"). ✅
- No claim that only one backend was tested — confirmed. Slide 4 and Slide 7 both name x86/fbgemm and ARM/qnnpack explicitly. ✅
- No claim that only unstructured pruning was tested — confirmed. Slide 4 names structured/channel pruning and pruning+fine-tune; Slide 13 notes the ShuffleNetV2 tooling exclusion honestly rather than omitting structured pruning altogether. ✅
- No causal SE/swish claim — confirmed. Slide 6 states "association, not demonstrated causation" and explains the confounded ablation; Slide 13 restates "SE/swish causality unresolved." ✅
- No "ShuffleNetV2-1.0x is the winner" claim — confirmed. Slide 10 says "candidate set, not a single winner"; Slide 11 states "Pareto non-dominant — no single winner declared." ✅
- No clinical-diagnostic claim — confirmed. Slide 3 states "Eczema" is a dataset label, not a verified diagnosis; Slide 13 states no diagnostic-accuracy claim is made. ✅
- No real-device latency claim — confirmed. Slide 13 states no real target-device latency/power measurement exists; no latency number appears anywhere in the new deck (the old deck's CPU-timing-direction-flip result was a limitation, not a headline claim, and is now summarized as a limitation only). ✅

## What was intentionally NOT invented

The brief's suggested Slide 12 mentioned specific corruption-drop magnitudes and Slide 9's
QAT/fine-tune panel could have shown per-epoch curves; only point values already verified
against `robustness_cross_architecture_summary.json`, `edge_ai_qat_2026-09-19.json`, and
`edge_ai_pruning_finetune_2026-09-19.json` in this session are shown. No latency figure,
no significance-test result, and no clinical-performance number appears anywhere in the
deck, because none exists in the current paper.
