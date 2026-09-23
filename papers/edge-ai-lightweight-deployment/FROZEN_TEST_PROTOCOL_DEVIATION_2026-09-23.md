# Declared deviation from the frozen test protocol (written 2026-09-23, before re-opening test data)

## Why

`FROZEN_TEST_PROTOCOL_2026-09-20.md` evaluated each architecture once on the 507-image test
split (FP32 and ARM/qnnpack static INT8). On 2026-09-23 a software defect was found in the
qnnpack INT8 path that invalidates part of that evaluation:

- In this machine's PyTorch 2.8.0 qnnpack build, quantized convolution outputs are
  channels_last (NHWC), and the quantized `relu6` / `hardtanh` / `clamp` kernels return wrong
  values for channels_last quantized input (max error 6.0 on a 0-6 range; exact on NCHW input).
  Minimal reproduction and diagnostics: `scripts/perchannel_anomaly_diagnostic_2026-09-23.py`,
  `scripts/qnnpack_relu6_layout_fix_2026_09_23.py`.
- Architectures that apply ReLU6 after a quantized conv (MobileNetV2-1.0x: 35 nodes;
  EfficientNet-Lite0: 33 nodes) therefore produced scrambled activations. On validation, inserting
  `.contiguous()` before each such node raises them from 49.8% to 73.0% and 67.7% respectively,
  within ~1 pt of their x86/fbgemm results. The other 7 architectures contain no such node and
  are numerically unchanged by the fix.

## What will be done (fixed now, before looking)

1. Re-run exactly the original test-split qnnpack INT8 evaluation for **all 9** architectures
   (same checkpoints, same `quantize_static`, same training-split calibration, same seed), and
   evaluate each calibrated model twice: `as_is` (original pipeline) and `fixed` (identical model
   with `.contiguous()` inserted before every relu6/hardtanh/clamp node). FP32 test results are
   not recomputed (they do not use the defective path).
2. Report all 9 architectures, both variants, full metrics and confusion matrices -- no subset.

## Rules

- The fix is a correctness fix to a library defect, applied uniformly by a mechanical graph
  rule; it is not tuned, and no threshold, architecture, calibration, or preprocessing choice
  changes.
- No architecture is selected, re-ranked, or excluded on the basis of these test results.
- The original 2026-09-20 test results remain on disk unchanged
  (`final_test_frozen_results_2026-09-20.json`) and will be reported alongside the fixed ones.
- The paper will state explicitly that the test split was opened a second time for this reason.
