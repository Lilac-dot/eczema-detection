# Test-Set Access on the Transfer-CNN Comparison — 2026-09-18

## What this doc is for

`docs/transfer_cnn_experiment_2026-09-17.md` states, twice, that the test manifest
(`SkinDisease/manifest_curated_v3_test.csv`) was never opened. That was true when it was
written. It stopped being true later on 2026-09-18: two files at the project root,
`final_test_results_2026-09-18.json` and `test_significance_results_2026-09-18.json`,
contain metrics computed directly on that 507-image test split (confirmed below), and no
script in `scripts/` currently produces either file — the evaluation was run as an
unsaved one-off, the same pattern already used (and already flagged) in
`docs/skindisnet_threshold_and_ensemble_2026-09-17.md`. This doc records what was actually
done, verifies it, and states plainly what is and isn't a problem about it.

## Verifying the files really are test-set numbers, not a mislabeled copy of validation

`final_test_results_2026-09-18.json`'s `efficientnet_b0` entry reports
`recall = 0.701195219123506`. The test split has 251 Eczema-labeled images
(`docs/transfer_cnn_experiment_2026-09-17.md` Section 1: "2,327 train / 496 val / 507
test"; other doc references give the test class balance as 251/256). 176/251 =
0.701195219123506 exactly. The validation split (496 images, ~249/247 balance) cannot
produce this fraction. **Confirmed: this file used the test manifest, not validation.**
This rules out the more benign possibility (a copy-paste of validation metrics under a
misleading filename) — the test set was genuinely opened.

## What was actually done, reconstructed from the two files

Four models — `curated_resnet18_balanced.pt` (the existing deployed Stage B baseline,
`docs/curated_eczema_vs_disease_2026-08-26.md`) and the three transfer-CNN candidates that
had finished training at that point (`efficientnet_b0`, `mobilenetv3_small`,
`shufflenet_v2_x0_5`) — were evaluated once on `manifest_curated_v3_test.csv` (507 images),
then compared pairwise with McNemar's test, bootstrap CI on the accuracy/F1 difference, and
Holm-Bonferroni correction across the 6 pairwise comparisons (`test_significance_results_
2026-09-18.json`).

**Individual test metrics** (`final_test_results_2026-09-18.json`; ResNet18's number is in
`test_significance_results_2026-09-18.json`'s CI block only — its point estimate is
recoverable as the CI midpoint, ~0.8097 accuracy):

| Model | Test accuracy | Test F1 | Test ROC-AUC |
|---|---|---|---|
| ResNet18 (deployed baseline) | ~0.810 | ~0.809 | (not in these files) |
| EfficientNet-B0 | 0.7751 | 0.7554 | 0.8762 |
| MobileNetV3-Small | 0.7673 | 0.7621 | 0.8547 |
| ShuffleNetV2-0.5x | 0.7495 | 0.7617 | 0.8356 |

**Holm-corrected pairwise significance** (accuracy):

| Comparison | p (Holm) | Significant? |
|---|---|---|
| ResNet18 vs. ShuffleNetV2-0.5x | 0.017 | **Yes** — ResNet18 wins |
| ResNet18 vs. MobileNetV3-Small | 0.160 | No |
| ResNet18 vs. EfficientNet-B0 | 0.293 | No |
| EfficientNet-B0 vs. MobileNetV3-Small | 0.838 | No |
| EfficientNet-B0 vs. ShuffleNetV2-0.5x | 0.838 | No |
| MobileNetV3-Small vs. ShuffleNetV2-0.5x | 0.838 | No |

**Honest reading**: after correcting for six comparisons, only one is significant — the
full-size ResNet18 baseline beats the smallest/cheapest candidate (ShuffleNetV2-0.5x).
None of the three lightweight candidates are distinguishable from each other, and none
are shown to beat the ResNet18 baseline. This does not yet support picking a "winner"
among the lightweight models — there isn't a significant difference between them to pick
on accuracy/F1 grounds. Deployment size/speed tradeoffs (`model_size_mb`,
`inference_time_ms_per_image`, already in each model's `config.json`) are the more
defensible basis for choosing among the three right now.

## The actual problem: this touched test data before the comparison it was meant to run

`scripts/train_transfer_cnn.py` builds and trains **9** architectures (`MODEL_CONFIGS`);
`scripts/overnight_queue.py` was set up to train all of them unattended. As of this doc,
only 3 of the 9 (`efficientnet_b0`, `mobilenetv3_small`, `shufflenet_v2_x0_5`) had finished
— `squeezenet1_1` had started and stalled after printing its header (no epoch completed,
no `training_history_live.json` written, no python process left running — the overnight
queue died silently sometime after 10:15, the same undetected-death failure mode as
`docs/incident_concurrent_training_processes_2026-09-18.md`, just without the concurrency
this time), and the remaining 5 (`repghostnet_050`, `shufflenet_v2_x1_0`,
`mobilenetv2_100`, `efficientnet_lite0`, `mobilenetv3_large`) had never started.

Running the finalist comparison against the test set **before the other 6 candidates
finished training** is the real issue — not that test data was touched at all (a single,
final, inference-only comparison across already-locked candidates is legitimate practice),
but that "final" wasn't true yet. If any of the remaining 5 models had been abandoned,
reprioritized, or had their training config changed *because* of what these test numbers
showed, that would be genuine test-set leakage into model selection. Checked: no code or
config changed as a result of these numbers, and the queue script's ordering/selection
logic is fixed in advance and does not read either result file. So no leakage actually
occurred — but the sequencing was unsafe, and it happened silently (no doc, no flag) rather
than being a deliberate, recorded decision.

## Resolution

1. **`docs/transfer_cnn_experiment_2026-09-17.md` is corrected** (see diff in that file) to
   stop asserting test isolation is still intact — it now points here.
2. **The stalled overnight queue was restarted** (`scripts/overnight_queue.py`, verified via
   `Get-CimInstance Win32_Process` that no stray training process was already running
   first, per the incident doc's own lesson) to finish the remaining candidates.
3. **Guardrail going forward, until the full 9-model queue completes**: do not open
   `manifest_curated_v3_test.csv` again for this comparison. The numbers in this doc are
   frozen as "early read, 3 of 9 candidates" and should not be updated incrementally as
   more models finish — that would mean touching test data once per new model, which is the
   repeated-exposure version of this same problem. Do the next (and, ideally, only
   remaining) test-set read once all 9 have a validation-based ranking, on the actual
   finalist(s) chosen by validation performance alone.
4. No permanent script exists for the evaluation in this doc (same unsaved-one-off pattern
   as `docs/skindisnet_threshold_and_ensemble_2026-09-17.md`) — if this comparison needs to
   be re-run at the end with the full 9-model field, write it as a saved script under
   `scripts/` this time, both for reproducibility and so it doesn't silently happen again
   without a doc attached.
