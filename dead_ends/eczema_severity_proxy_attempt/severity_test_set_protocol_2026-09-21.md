# Severity Model Test-Set Protocol (committed before training) — 2026-09-21

Written before `scripts/train_severity_cnn.py` is run for either architecture, following
this project's standing test-set discipline (see `docs/transfer_cnn_test_set_access_2026-09-18.md`,
`papers/edge-ai-lightweight-deployment/FROZEN_TEST_PROTOCOL_2026-09-20.md` for precedent).

## What's being held out

`dataset/severity_labels/severity_test.csv` (251 images, the same images as
`SkinDisease/manifest_curated_v3_test.csv`'s Eczema class, minus histology exclusions).
Labels exist in this file (labeling ground truth is not "peeking" — it was assigned by
the same blind visual process as train/val, see `docs/severity_scale_selection_2026-09-21.md`),
but the file itself is never opened by the training script.

## What will be trained now (Pareto-survivor architectures only, per prior discussion)

- **ShuffleNetV2-0.5x**
- **EfficientNet-B0**

Both via `scripts/train_severity_cnn.py --model <name>`, which structurally only ever
opens `severity_train.csv` and `severity_val.csv` — there is no code path in that script
that references `severity_test.csv`, mirroring `train_transfer_cnn.py`'s own isolation
approach for the disease classifier.

## Rules committed before touching test

- No test-set evaluation, of any kind, until Tishya explicitly says so.
- No architecture choice, hyperparameter, threshold, or checkpoint-selection decision
  will ever be based on test-set numbers — those decisions are made on validation only.
- When test evaluation does happen, it will be run once, on both architectures, and the
  full result (accuracy, macro-F1, macro-precision/recall, confusion matrix) reported
  as-is — not a subset selected after seeing which numbers look best.
- This is a non-dermatologist visual-proxy severity label (see
  `docs/severity_scale_selection_2026-09-21.md`), not a validated clinical score — test
  accuracy on it measures agreement with this project's own labeling heuristic, and will
  be reported as such.
