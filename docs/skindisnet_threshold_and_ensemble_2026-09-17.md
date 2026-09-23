# SkinDisNet F1 Improvement Attempts: Threshold Tuning and Ensembling — 2026-09-17

## Why

`docs/external_generalization_improvement_2026-09-15.md` established
`curated_resnet18_multisource_weighted.pt` (equal-weighted fine-tuning on Internal+SCIN+
SkinDisNet) as the best-performing intervention for SkinDisNet generalization: AUC 0.6515,
F1 0.4218 on SkinDisNet's held-out test set. The user asked directly whether F1 could be
pushed higher. This doc records two attempts, both real, both evaluated without touching
test data for any tuning decision, and a resulting decision about which number this
project should call its "external validation" result going forward.

**Also documented here**: two background training attempts from 2026-09-16 (a from-scratch
LODO run for `internal_skindisnet`, and a from-pretrained fine-tune on Internal+SCIN only
holding SkinDisNet out as a strict external test) were interrupted by a laptop shutdown
before completion (5/15 and 1/10 epochs respectively) and were **deliberately abandoned**,
not resumed, per explicit user instruction once the multisource-weighted result above was
confirmed sufficient for current evaluation purposes. Their partial checkpoints
(`models/lodo_internal_skindisnet_resnet18.pt`, `models/finetune_internal_scin_resnet18.pt`,
and both `.resume.pt` sidecars) were deleted rather than left sitting on disk implying a
finished result. `models/lodo_internal_scin_resnet18.pt` (the one LODO config that DID
finish, from earlier the same day) was kept — see `docs/cross_dataset_matrix_2026-09-15.md`
Part 5 for that result (AUC 0.498 on SkinDisNet).

## Attempt 1: Validation-tuned decision threshold

**Method**: `curated_resnet18_multisource_weighted.pt`'s F1 (0.4218) was originally computed
at a naive argmax / 0.5 probability cutoff, never actually tuned. Extracted raw probabilities
(`eval_external_common.run_eval(..., return_raw=True)`) on SkinDisNet's **validation** split
(`manifest_skindisnet_val_clean.csv`, 340 images) only, swept 197 candidate thresholds
(0.01-0.99) to find the val-F1-maximizing cutoff, then applied that single fixed threshold
to the **test** split (`manifest_skindisnet_test_clean.csv`, 513 images, never used in the
threshold search) for final reporting. No retraining; same model weights throughout.

**Result**:

| | Threshold | Test F1 | Test Precision | Test Recall | Test AUC |
|---|---|---|---|---|---|
| Original (naive) | 0.500 | 0.4218 | 0.4662 | 0.3851 | 0.6512 |
| Validation-tuned | 0.110 | **0.5009** | 0.3460 | 0.9068 | 0.6512 (unchanged) |

**Interpretation**: a real, leakage-free F1 improvement (threshold chosen on validation data
only), but AUC is identical — nothing about the model's ranking ability changed. The tuned
threshold is far more liberal (0.11 vs. 0.5), so the model now flags nearly everything as
"Eczema" (recall 90.7%) at a real precision cost (46.6% -> 34.6%). This is the same
mechanism already identified and correctly flagged in
`external_generalization_improvement_2026-09-15.md` for SCIN's earlier F1 jump (a
threshold/operating-point shift, not a ranking improvement) — reproduced here honestly via
a proper validation-only tuning procedure rather than accidentally via test-set exposure.
**Whether this threshold is worth adopting depends on whether missing a real Eczema case is
considered worse than a false alarm for the intended use case** — it is a legitimate,
reportable choice, not a free win.

## Attempt 2: Ensembling with the SkinDisNet-specific model

**Method**: averaged softmax probabilities of `curated_resnet18_multisource_weighted.pt`
("Model A") and `models/skindisnet_resnet18.pt` ("Model B", from
`docs/cross_dataset_matrix_2026-09-15.md` — trained directly on SkinDisNet's own training
split), motivated by this project's earlier finding that a CNN/LightGBM ensemble beat both
individual models on the original curated Stage B task
(`docs/curated_eczema_vs_disease_2026-08-26.md`). Threshold again tuned on validation only,
applied to test.

**Result**:

| Model | Val AUC | Test AUC | Test F1 (val-tuned threshold) |
|---|---|---|---|
| A alone (multisource_weighted) | 0.6240 | 0.6512 | 0.5009 |
| B alone (skindisnet_resnet18) | 0.8194 | 0.8281 | 0.6687 |
| Ensemble (A+B averaged) | 0.7919 | 0.7953 | 0.6261 |

**Critical finding: the ensemble underperforms Model B alone on every metric.** Averaging
in the weaker model (A) dilutes the stronger one (B) rather than helping — this is the
opposite of the original curated-task ensemble result, and is reported as such rather than
selectively citing only the ensemble-vs-A comparison (which would look like an improvement
in isolation).

**The more important issue this surfaces is not which combination scores highest, but what
each number actually means**: Model B was trained on SkinDisNet's own training data. Testing
it on SkinDisNet's test set is an **in-domain evaluation** — the same kind of test as the
internal model's own 0.86 AUC on its own test set — not evidence that anything generalizes
to unseen data. It is already reported, correctly framed, as the "SkinDisNet -> SkinDisNet"
diagonal cell in `docs/cross_dataset_matrix_2026-09-15.md`'s matrix. Reporting Model B's or
the ensemble's number under an "external validation" label would misrepresent what was
tested, since Model B has already seen this exact distribution during training.

## Decision: which number counts as "external validation"

Two honestly-labeled numbers now exist for SkinDisNet, and they answer different questions:

| Claim being made | Model | AUC | F1 |
|---|---|---|---|
| "Our model generalizes to data it never trained on" (external validation) | `curated_resnet18_multisource_weighted.pt` | **0.6512** | **0.5009** (val-tuned) / 0.4218 (naive threshold) |
| "A model trained on SkinDisNet performs well on SkinDisNet" (in-domain, not generalization evidence) | `skindisnet_resnet18.pt` | 0.8281 | 0.6687 |

**This project's external-validation claims should use the first row.** The second row is
legitimate and already documented elsewhere (the matrix's diagonal cell) but must never be
substituted in as if it were a generalization result — doing so would contradict the entire
premise of this project's cross-dataset generalization investigation
(`docs/external_validation_2026-09-15.md` onward). Per direct user confirmation, AUC
0.6515/F1 0.4218 (Experiment 2b's original numbers, matching the 0.6512/0.4218 reproduced
here at the naive threshold) is the settled external-validation figure for current
reporting purposes; the val-tuned 0.5009 F1 is available as an alternative operating point
if a higher-recall use case is ever specifically wanted, with its precision cost stated
alongside it.

## Files

New: `scripts/` — no new permanent scripts (both analyses were run as one-off inline
scripts, not saved, since they only recombine existing model outputs and don't need to be
rerun as part of any pipeline); this doc. Deleted (abandoned, incomplete):
`models/lodo_internal_skindisnet_resnet18.pt`, `models/lodo_internal_skindisnet_resnet18.pt.resume.pt`,
`models/finetune_internal_scin_resnet18.pt`, `models/finetune_internal_scin_resnet18.pt.resume.pt`,
and their corresponding log files. Not modified: `models/curated_resnet18_multisource_weighted.pt`,
`models/skindisnet_resnet18.pt`, any manifest, the paper, or Stage A/C code.
