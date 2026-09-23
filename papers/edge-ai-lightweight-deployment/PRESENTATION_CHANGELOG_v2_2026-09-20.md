# Presentation Changelog — v2 clarity pass (2026-09-20)

**Note on slide numbers below:** the table immediately below describes the first v2 pass,
when the deck was still 14 slides (Slides 1–4 unchanged, 5–13 redesigned, 14 unchanged). A
follow-up request then added a new "Key Definitions" slide right after Slide 4, pushing
every slide from the old "5" onward back by one — the deck is now **15 slides**. Read every
"Slide N" reference in the table below as N for a 14-slide deck; in the actual, current
`edge_ai_eval_presentation_FINAL_v2.pptx` file, add 1 to any number 5 or above (old Slide 5
= current Slide 6, old Slide 13 = current Slide 14, etc.). The new Slide 5 itself, and the
one in-deck cross-reference this insertion required updating, are documented in their own
row below.

`edge_ai_eval_presentation_FINAL_v2.pptx` (built by `scripts/build_eval_presentation_FINAL_v2.py`)
is a clarity/visual-hierarchy pass over `edge_ai_eval_presentation_FINAL.pptx`. **No number,
conclusion, or scientific claim changed in this pass.** Slides 1–4, 10, 14 keep the same
science as v1 with only layout/terminology touch-ups; slides 5–13 were restructured around
"what was tested → what happened → what it means," with a large-number takeaway per slide
and full speaker notes added.

## Slide-by-slide

| # | What changed | Why |
|---|---|---|
| 5 (new) | Added a "Reading the results — key definitions" slide, inserted between the old Slide 4 (Experimental Design) and the old Slide 5 (Pruning Knees), pushing every later slide back by one (deck is now 15 slides, not 14). Contains the exact formulas for Sensitivity, Specificity, F1, Balanced accuracy, Margin M, ΔAcc, Accuracy retention R, Relative degradation, and Fraction of drop recovered, plus the collapsed/severely-degraded/backend-stable thresholds and the bootstrap-CI methodology, all copied verbatim from the paper's Section 4.8. Has speaker notes. | You asked to add the formal definitions the defense-readiness note flagged as the one thing an examiner would still need the paper for. The deck is now fully self-contained on this point; the one in-deck cross-reference to the primary-finding slide (in the Slide 11 funnel) was updated from "Slide 7" to "Slide 8" to match the new numbering. |
| 1–4 | Unchanged content. Slide 3's title font size reduced (30→25pt) and its subtitle/table/card shifted down ~0.2–0.25in | The original title (49 characters) was tight enough against its subtitle to risk visual overlap if it wrapped; added vertical clearance defensively. |
| 2 | Title font size reduced (34→27pt) for the 68-character subtitle line; content boxes below shifted from top=1.55in to top=2.0in/2.55in/3.75in | You flagged this specific slide's text-overlap risk. The long subtitle was at real risk of wrapping to two lines and colliding with the research-questions box beneath it; added a full ~0.3in safety margin regardless of exact wrap behavior. |
| 5 | Retitled "FINDING 1 — No universally safe pruning level" with an explicit subtitle line; added a "20% – 60%" large-number callout in the side panel; added the "A knee marks where degradation begins — not an optimal deployment sparsity" note; added a bottom takeaway bar: "Pruning tolerance is architecture-specific." | Matches your requested title/subtitle exactly; makes the headline number (the 20–60% range) readable in the "5–10 second" sense before the audience reads the chart. |
| 6 | Retitled "FINDING 2 — SE / swish architectures are more PTQ-fragile"; replaced the small table + side card with two large side-by-side number cards (27.2 pt vs. 3.9 pt); switched from "%" to "pt" for all point-difference language; corrected the difference from "23.3 pt" to **23.4 pt** (the paper's own precisely-rounded value, not a re-rounding of the two display means); kept the "association, not demonstrated causation" statement as its own line, not buried in a paragraph | You explicitly asked for pt (not %) terminology and for internal consistency with the paper's own number. Verified: the paper's underlying unrounded difference is 23.353 pt, which rounds to 23.4 pt — using that exact value rather than 27.2−3.9=23.3 (which uses already-rounded inputs). |
| 7 | Retitled "PRIMARY FINDING — INT8 behavior depends on the backend" with the "same weights + same task + INT8 → different backend → very different outcomes" framing as a subtitle; shortened table status labels; enlarged the 3/9, 2/9, 4/9 stat row and 5/9 = 55.6% line; added the MobileNetV2-1.0x sensitivity=0%/specificity=100% callout with the "≈ constant-classifier behavior" annotation | This is the slide the brief asked to make the visual centerpiece. The Sensitivity=0%/Specificity=100% example is now shown explicitly, so the audience understands "collapse" as a qualitative behavior change, not just a lower accuracy number. |
| 8 | Retitled "Why does the backend gap occur?"; unchanged table; tightened side-panel wording to the brief's exact bottom-line sentence | Matches the requested title/structure. |
| 9 | Retitled "Can the compression damage be recovered?"; added a red takeaway bar "QAT DID NOT RESTORE USABLE ACCURACY" under the QAT panel and a green takeaway bar "RECOVERED MOST/ALL PRUNING LOSS FOR 2/3 TESTED" under the fine-tuning panel; added a bottom contrast sentence | Makes the two-panel contrast (QAT failed / fine-tuning mostly worked) immediately visible instead of requiring the audience to read both tables and infer the contrast themselves. |
| 10 | Rebuilt entirely as a 5-stage vertical elimination funnel (9 architectures → apply backend-stability criterion → 4 backend-stable → exclude ResNet18 baseline → 3 lightweight candidates), replacing the old two-card layout | You explicitly asked for a funnel/selection diagram instead of a plain list. |
| 11 | Retitled "ShuffleNetV2 — a multi-objective trade-off"; changed table's "Wins" column to "better: 0.5x" / "better: 1.0x"; changed side-panel labels from "0.5x wins:" / "1.0x wins:" to "0.5x is better on:" / "1.0x is better on:" | You explicitly said not to write "wins" — "is better on" is now used everywhere on this slide. |
| 12 | Retitled "Robustness checks — what reproduces, and what doesn't?"; replaced "hospital-sourced dataset" with "external dermatology dataset" (the paper itself never uses "hospital" — that detail lives only in this project's dataset-build scripts, not in the paper, which is the source of truth per your instruction); labeled panel C "STRONGEST"; added the explicit caveat "Not 'all results replicated' — operational-status agreement only" | Matches your instruction not to claim more sourcing detail than the paper itself states, and not to imply full replication when only operational-status agreement (8/9) is supported. |
| 13 | Rebuilt as "What did the experiments show?" with 5 numbered result cards (pruning, backend/architecture dependence, no single mechanism, recovery depends on method, multi-objective deployment) plus the one-line synthesis sentence you specified verbatim, replacing the old two-column Limitations+Conclusion layout | Matches your requested "results synthesis" framing. The detailed limitations list from v1 is preserved in the paper and the v1 changelog; this slide is now a synthesis, not a repeat of Slide 8's limitations. |
| 14 | Unchanged | No requested change. |

## Speaker notes

Added to slides 5–13 (confirmed programmatically: `slide.has_notes_slide` and non-empty text
on exactly 5–13, absent on 1–4 and 14). Each note follows the four-part structure you asked
for — what was tested, what the numbers mean, what the main result is, what it does NOT
prove — written as spoken sentences, not paper prose. Full text is in the `.pptx` file's
notes pane for each slide; not duplicated here to avoid drift between two copies of the same
text.

## Numerical cross-check performed

Every number in the redesigned slides was re-verified against the same JSON result files
used to build the paper and v1 (not re-derived from memory), specifically:
- `edge_ai_extended_analysis_2026-09-19.json` (FP32/INT8 fbgemm accuracies, SE/swish means)
- `edge_ai_qnnpack_backend_check_2026-09-19.json` (fbgemm vs. qnnpack table, sensitivity/specificity)
- `edge_ai_pareto_analysis_2026-09-19.json` (pruning knees)
- `qnnpack_perchannel_diagnostic_2026-09-20.json` (forced per-channel F1 values)
- `edge_ai_qat_2026-09-19.json`, `edge_ai_pruning_finetune_2026-09-19.json` (recovery tables)
- `edge_simulation_all_architectures_2026-09-19.json` (ShuffleNetV2 size figures)
- `final_test_frozen_results_2026-09-20.json` (held-out test, 8/9 match)
- `SkinDisease/manifest_curated_v3_{train,val,test}.csv` row counts (3,330 = 2,327+496+507)

The one correction made during this cross-check: the SE/swish point difference was **23.4 pt**
in the paper's own precisely-computed text, not 23.3 pt as displayed in v1 (which came from
subtracting the two already-rounded display means, 27.2−3.9=23.3, rather than using the
paper's own unrounded value, 23.353, which rounds to 23.4). v2 uses 23.4 pt throughout.

## Automated QA performed on the final file

- Every shape on every slide checked against slide bounds (13.333in × 7.5in) — zero off-slide shapes.
- Bounding-box overlap scan across all text shapes on all 14 slides — the two genuine risks
  found (Slide 2's 68-character subtitle and Slide 3's 49-character title, both close enough
  to their box width to risk wrapping into the element below) were fixed by reducing font
  size and adding vertical clearance. The remaining flagged bounding-box overlaps on Slides
  5, 6, 7, 9, and 13 are confirmed false positives from the header helper's fixed 0.9in title
  box height against short, comfortably one-line titles — not actual rendered text collisions.
- Full banned-phrase sweep (9.3×, 26.3/2.8, ">5pt", causal SE/swish language, "wins,"
  "hospital-sourced," "23.3 pt," clinical/target-device/real-camera-robustness/general-
  external-validation claims) — zero hits.
- Full presence check for every anchor number in your brief (27.2 pt, 3.9 pt, 23.4 pt, ≈7.0×,
  3/9, 2/9, 4/9, 55.6%, 8/9, 507, 3,330, 2,327, 496, 1.95 MB, 5.45 MB) — all present.

## Defense-readiness assessment

An examiner who has NOT read the paper should be able to follow, from the slides alone:
- **What was tested** — stated as the subtitle or opening line of every results slide (5–13).
- **What happened** — one large number or table per slide, with color-coded status.
- **What it means** — a one-line takeaway bar or bold sentence on every results slide.
- **What it does NOT prove** — an explicit qualifier on every results slide (association not
  causation; partial not complete mechanism; candidate set not a winner; Pareto non-dominant;
  exploratory; operational-status agreement, not full replication).

The one place a reader would still need the paper rather than the slides alone: the exact
definitions behind "margin," "balanced accuracy," or the bootstrap-CI methodology are not
re-derived on any slide (by design — the brief asked for results clarity, not a methods
re-exposition). An examiner asking "how exactly is 'severely degraded' defined" would be
pointed to the paper's Section 4.8, which is fine for a defense (that's a legitimate follow-up
question, not a comprehension gap in the deck).
