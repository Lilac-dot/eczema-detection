# Edge-Deployment Simulation: Pruning + Quantization on ShuffleNetV2-1.0x (2026-09-19)

## What this is

A "lightweight/edge AI" framing (user-proposed "Option 7") of the already-completed
image-architecture comparison
(`papers/architecture-selection-report/honors-paper-report.docx`, Section 5), applied to
the already-selected image-channel model, ShuffleNetV2-1.0x
(`experiments/shufflenet_v2_x1_0/checkpoints/best_overall.pt`).

**No Raspberry Pi or Android device is available in this project.** This simulates the
comparison on the CPU-only development machine instead. Every latency number below is a
CPU-only proxy from a shared, noisy development laptop, not a claim about real phone/edge
hardware -- flagged the same way the main report already flags its own CPU timing
(Section 8/10 of the architecture report).

Script: `scripts/edge_simulation_shufflenet.py`. Raw output:
`papers/edge-ai-lightweight-deployment/edge_simulation_shufflenet_2026-09-19.json`.
Evaluated on the **validation** split only (496 images) -- the test set
(`manifest_curated_v3_test.csv`) was never opened, consistent with this project's
test-set-isolation discipline (`docs/transfer_cnn_test_set_access_2026-09-18.md`):
pruning/quantization variants are an exploratory sweep, not a finalized candidate list,
so they belong on validation like every other sweep in this project's history, not on
the one-time-only test set.

Sanity check before either experiment: the unmodified checkpoint, reloaded fresh, scored
0.7923 validation accuracy -- matching `experiments/shufflenet_v2_x1_0/config/config.json`
exactly, confirming nothing about the loading/eval code itself is broken.

## 1. Pruning sweep (unstructured, global, L1 magnitude, one-shot)

All `Conv2d`/`Linear` weight tensors pruned together (global, not per-layer), by
magnitude, in one shot -- **no fine-tuning after pruning**. This is a sensitivity sweep
(how much accuracy does removing weights cost, with no chance to recover it), not a claim
about the best achievable accuracy at a given sparsity -- fine-tuning after pruning
typically recovers some accuracy and was not attempted here.

| Sparsity | Val Acc | Val F1 | Val AUC | Latency (ms) | Dense size (MB) | Sparse-format estimate (MB) |
|---|---|---|---|---|---|---|
| 0% (baseline) | 79.23% | 0.794 | 0.861 | 38.6 | 5.45 | 10.57 |
| 10% | 79.03% | 0.793 | 0.862 | 47.8 | 5.45 | 9.52 |
| 20% | 77.22% | 0.781 | 0.860 | 54.7 | 5.45 | 8.48 |
| 30% | 77.62% | 0.769 | 0.860 | 54.7 | 5.45 | 7.43 |
| 40% | 77.02% | 0.746 | 0.859 | 52.2 | 5.45 | 6.39 |
| 50% | 69.35% | 0.608 | 0.843 | 33.0 | 5.45 | 5.35 |
| 60% | 57.66% | 0.323 | 0.795 | 29.5 | 5.45 | 4.30 |
| 70% | 49.80% | 0.000 | 0.750 | 31.4 | 5.45 | 3.26 |
| 80% | 50.20% | 0.016 | 0.657 | 40.9 | 5.45 | 2.21 |
| 90% | 49.60% | 0.016 | 0.626 | 37.8 | 5.45 | 1.17 |

### Reading this honestly

**There is a real, usable sensitivity finding**: accuracy is essentially flat through 40%
sparsity (79.2% -> 77.0%, a 2.2-point drop for removing 2 out of every 5 weights), then
falls off a cliff between 50% and 70% -- by 70% sparse, F1 collapses to 0.000 (the model
stops predicting the positive class at all; precision/recall both hit 0) and accuracy sits
at chance (49.8%, this dataset's val split is close to 50/50). This is a real, quantifiable
knee in the curve, not a smooth degradation -- useful for anyone deciding how aggressively
to compress this specific model.

**Two things this table does NOT show, stated plainly rather than glossed over:**

1. **Latency does not improve with sparsity, and at points gets worse.** This is expected,
   not a bug: unstructured pruning zeros out individual weight *values* but the tensor
   stays the same dense shape -- PyTorch's CPU convolution still does the full
   multiply-accumulate over every zeroed position, so there is no real speedup without a
   sparse-aware runtime/kernel this project doesn't have. The `sparse_effective_mb_estimate`
   column is a COO-format (value + index per nonzero) size estimate of what a *real* sparse
   runtime could achieve on disk -- not a measured number, and not what `torch.save`
   actually produces (the dense checkpoint stays 5.45 MB at every sparsity level, also
   shown above). Reporting a "smaller model" from pruning alone, without structured pruning
   or a sparse runtime, would overstate what was actually demonstrated.
2. **Latency numbers here are noisy.** Standard deviations on individual latency
   measurements ran as high as 22-38 ms on a mean of 30-55 ms -- close to 100% relative
   noise -- because this ran on a shared laptop with Chrome, VS Code, and other
   background apps competing for the same CPU cores, not a dedicated benchmarking
   machine. Treat every latency figure in this section as an order-of-magnitude proxy,
   not a precise measurement.

## 2. Quantization: FP32 vs. dynamic INT8 vs. static INT8 (FX graph mode)

| Variant | Val Acc | Val F1 | Val AUC | Latency (ms) | Size (MB) |
|---|---|---|---|---|---|
| FP32 (baseline) | 79.23% | 0.794 | 0.861 | 16.5 | 5.45 |
| Dynamic INT8 (Linear layers only) | 78.83% | 0.790 | 0.861 | 17.3 | 5.08 |
| Static INT8 (full network, FX graph mode) | 78.83% | 0.790 | 0.862 | 27.1 | 1.58 |

Static quantization calibrated on 8 batches (256 images) from the **training** split
only -- never validation or test. Backend: `onednn` (the only one
`torch.backends.quantized.supported_engines` reports on this machine; the more commonly
documented `fbgemm`/`qnnpack` backends are not available in this torch build).

### Reading this honestly

**The accuracy story is good news, straightforwardly**: both INT8 variants cost under half
a point of accuracy (79.23% -> 78.83%) and essentially no AUC. Static INT8 shrinks the
checkpoint 3.4x (5.45 MB -> 1.58 MB) for that same near-zero accuracy cost -- a real,
usable result if checkpoint size were the only axis that mattered.

**The latency story is not what quantization is supposed to deliver, and is reported as
such rather than hidden**: static INT8 measured *slower* than FP32 on this machine (27.1 ms
vs. 16.5 ms), and dynamic INT8 was statistically indistinguishable from FP32 (17.3 ms vs.
16.5 ms, well within this machine's noise band from Section 1). This is the opposite of
quantization's usual purpose. The most likely explanation is that this torch build's
`onednn` CPU backend does not have fast fused INT8 kernels for this architecture's op
pattern, so the quantize/dequantize overhead inserted around each op outweighs any INT8
compute saving at batch size 1 on a model this small -- but that is a plausible explanation,
not a verified root cause; it was not root-caused further. **The one thing this result
reliably shows is that CPU timing on this specific dev machine, with this specific torch
build, is not a stand-in for real edge/mobile latency** -- exactly the caveat the main
architecture report already carries (Section 8), now demonstrated rather than just stated.
A real phone deployment would use a converted, mobile-native runtime (Core ML / TFLite /
ONNX Runtime Mobile) with actual INT8 kernels for that hardware, which this simulation
cannot approximate.

## What this does and doesn't establish

**Established**: a real, usable pruning-sensitivity curve for this model (safe to roughly
40% sparse, degrades sharply past 50%) and a real, usable quantization accuracy result
(INT8 costs under half a point of accuracy, static INT8 gets a genuine 3.4x size
reduction).

**Not established**: any real edge/mobile latency number (this machine's CPU timing is not
a valid proxy, demonstrated directly in Section 2, not just assumed), any benefit from
pruning beyond the checkpoint-size *estimate* (no structured pruning or sparse runtime was
used, so no real speedup or real disk-size reduction was measured), and no fine-tuning was
attempted after pruning (the 40-50% knee could likely be pushed further right with
retraining -- not attempted here).

## Possible next steps (not started)

- Fine-tune the model after pruning at the 40-60% range to see how much of the accuracy
  cliff is recoverable -- the natural next question this sweep raises.
- Structured (channel-level) pruning, which would need a library like `torch-pruning`
  (not installed in this project) to physically shrink the model and realize a real
  speedup, not just a size estimate.
- Export the static-INT8 model to ONNX/TFLite and benchmark on an actual phone-class
  device if one becomes available -- the only way to get a real edge latency number
  rather than this CPU-only proxy.
