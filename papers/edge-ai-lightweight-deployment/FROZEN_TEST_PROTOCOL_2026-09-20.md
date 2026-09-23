# Frozen Test-Set Evaluation Protocol (committed 2026-09-20, before opening test data)

This document is written and committed BEFORE `SkinDisease/manifest_curated_v3_test.csv`
(507 images) is opened for any purpose in this compression/deployment study. It exists so
that no decision below can be, or be suspected of being, influenced by test-set results.

## What will be evaluated (fixed, decided now)

For each of the 9 already-trained architectures (ResNet18, EfficientNet-B0,
EfficientNet-Lite0, MobileNetV2-1.0x, MobileNetV3-Small, ShuffleNetV2-0.5x,
ShuffleNetV2-1.0x, SqueezeNet1.1, RepGhostNet-0.5x), using the existing checkpoints with
**no retraining, no re-selection, no threshold tuning**:

1. FP32 zero-shot evaluation on the test split.
2. ARM/qnnpack static INT8 evaluation on the test split, calibrated ONLY on the training
   split (`SkinDisease/manifest_curated_v3_train.csv`) -- never on validation, test, or
   SkinDisNet data.

## What will NOT be evaluated on test, and why

- **x86/fbgemm INT8 on test**: not attempted. This machine's PyTorch build only supports
  the `qnnpack` quantized engine (`torch.backends.quantized.supported_engines ==
  ['qnnpack', 'none']`) -- there is no way to execute an x86/fbgemm-quantized model on
  this hardware. This is flagged as an infeasible experiment, not silently skipped or
  approximated.
- **QAT, structured pruning, and pruning+fine-tuned variants**: not evaluated on test.
  These were exploratory comparisons across many configurations (3 sparsity levels x 3
  architectures for pruning+fine-tune, 3 ratios x 7 architectures for structured pruning,
  3 architectures for QAT) during the validation stage. Evaluating all of them on the
  now-opened test set would effectively run dozens of comparisons against the held-out
  set in one sitting, which defeats the purpose of holding it out. Only the two
  conditions above (FP32, qnnpack INT8) are evaluated on test, mirroring exactly the
  central comparison this paper's Section 5.5 and Table 5 already make on validation.

## Rules committed before looking

- No threshold, architecture, pruning ratio, QAT setting, or preprocessing choice will be
  changed based on test results.
- The deployment candidate recommendation (ShuffleNetV2-1.0x) is based on
  validation-stage and SkinDisNet evidence already on record before this protocol was
  written -- test results are reported as confirmatory context alongside that
  recommendation, not as its basis.
- All 9 architectures are reported as a fixed comparison. No "winner" will be chosen or
  re-chosen after seeing test results.
- Metrics reported: accuracy, precision, sensitivity, specificity, F1, AUROC, and the
  raw confusion matrix (TP/FP/TN/FN), for both conditions, for all 9 architectures --
  the full result, not a subset selected after seeing which numbers look best.
- If any architecture's test-set evaluation fails to run (e.g. a compatibility or label
  issue), that failure will be reported explicitly, not silently omitted or
  papered over with a fabricated number.

This file is committed to disk before `scripts/eval_final_test_frozen.py` is written or
run.
