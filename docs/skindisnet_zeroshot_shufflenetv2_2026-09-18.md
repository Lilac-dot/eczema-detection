# ShuffleNetV2-1.0x Zero-Shot External Validation on SkinDisNet — 2026-09-18

## Why this ran

`honors-paper-report.docx` selected ShuffleNetV2-1.0x as the image-channel architecture
(Section 5.6) but its own Future Work section (Section 11) flagged that the only
cross-dataset external-validation number on record (AUC 0.4865 zero-shot / 0.6512
fine-tuned) was measured for the ResNet18 baseline, never for ShuffleNetV2-1.0x
specifically. This closes that gap.

## What changed vs. the number already on record

Also rebuilt the SkinDisNet manifest with two cleaning steps that the original zero-shot
matrix work (`cross_dataset_matrix_2026-09-15.md`) didn't apply to the *full* manifest:
removed 39 near-duplicate rows (`docs/external_duplicate_screen_2026-09-16.csv`) and 113
low-skin-content rows (`docs/external_skin_content_check_2026-09-16.csv`, skin_fraction <
0.3) — see `scripts/build_skindisnet_clean_manifest.py`. One near-duplicate pair crosses
disease labels (the same photo appears as both "Eczema" and "Scabies" in SkinDisNet's own
source data) — flagged, not arbitrated. Net: 1710 -> 1558 images (467 Eczema+AD / 1091
Other). Both models below were re-run on this cleaned manifest, so the ResNet18 number here
(AUC 0.4745) is a slightly different, slightly cleaner measurement than the 0.4865 already
on record — not a replacement for it, both are reported.

## Result

Zero-shot, neither model fine-tuned on any SkinDisNet data:

| Model | Accuracy | Precision | Recall | F1 | AUC | AUC 95% CI |
|---|---|---|---|---|---|---|
| ResNet18 (deployed baseline) | 67.65% | 30.1% | 6.0% | 0.100 | 0.4745 | [0.445, 0.505] |
| ShuffleNetV2-1.0x (selected) | 69.64% | 25.0% | 0.6% | 0.013 | 0.4806 | [0.446, 0.511] |

Paired bootstrap significance (ShuffleNetV2-1.0x AUC − ResNet18 AUC, same 1558 images):
**observed diff +0.0061, 95% CI [−0.029, 0.038], p = 0.77 — not significant.**

## Interpretation

**The generalization gap is architecture-independent.** Both models sit at chance (AUC
~0.48, CI spanning 0.5) on SkinDisNet, and the tiny difference between them is nowhere near
significant. This confirms, for the actually-selected architecture rather than just the
old baseline, the exact caution the report's own Section 2.9 (Scope Note) already stated in
advance: *"A generalization gap comparable to what was found for that baseline should be
assumed to exist for each candidate here until it is specifically tested, not assumed
absent because a candidate scores well internally."* ShuffleNetV2-1.0x's strong internal
test performance (Section 5.3) tells you nothing about its performance on an independent
clinical photo source — same conclusion as ResNet18, now confirmed rather than assumed.

Recall is the more visible failure mode for both models (6.0% and 0.6%) — both models are
calling almost everything "Other," not confusing classes symmetrically. ShuffleNetV2-1.0x's
recall is worse in absolute terms, but the CI overlap and non-significant paired test mean
this should be read as noise at this sample size, not a real architecture effect.

## What this does and doesn't settle

- Does NOT mean the architecture choice (Section 5) was wrong — that comparison already
  correctly limited its claim to same-source performance (Section 2.9), and this result is
  exactly why that limitation was stated rather than glossed over.
- Does NOT retest the fine-tuning-recovers-generalization question
  (`cross_dataset_generalization_status`, `skindisnet_patient_leakage`) for
  ShuffleNetV2-1.0x — that would be a separate fine-tuning experiment, not done here.
- Files: `scripts/build_skindisnet_clean_manifest.py`,
  `scripts/eval_external_skindisnet_selected_arch.py`,
  `skindisnet_zeroshot_shufflenetv2_vs_resnet18_2026-09-18.json`.
