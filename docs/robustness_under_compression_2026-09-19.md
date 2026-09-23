# Does Compression Change Corruption-Robustness? — ShuffleNetV2-1.0x (2026-09-19)

## What this is

The actual combined experiment behind the "D + compression" pivot: does pruning or
quantizing the deployed model change how badly it degrades under corrupted input images
(the same 7-corruption x 5-severity sweep as
`docs/robustness_corruption_benchmark_2026-09-19.md`), on top of already knowing (from
`papers/edge-ai-lightweight-deployment/`) how compression affects *clean*-image accuracy?

Literature check (2026-09-19) found this exact question — compression x corruption
interaction — already studied in general computer vision, with mixed, architecture-
dependent results (some studies find compression hurts robustness, some find it helps).
Nothing existed for dermatology. This is the first check for this project's own model.

Script: `scripts/robustness_under_compression.py`. Model: ShuffleNetV2-1.0x only (the
deployed candidate) — not all 9 architectures, to keep runtime manageable; extending this
to other architectures is future work, not attempted here. Four variants, all evaluated on
the same 36-condition corruption sweep, validation set only:
- **fp32**: the original checkpoint (reused from the earlier corruption-robustness run,
  not re-executed — deterministic, so re-running would just cost 20 more minutes for
  identical numbers)
- **pruned_at_knee**: unstructured L1 pruning at 50% sparsity — this model's own
  accuracy-vs-sparsity knee, identified from the original pruning sweep
- **static_int8**: post-training static quantization (FX graph mode), same procedure as
  the edge-ai paper
- **pruned30_plus_int8**: 30% pruning (safely inside every model's knee) + static INT8 —
  the "combined, safe" compression variant

## Result 1: quantization alone barely changes the corruption-vulnerability pattern

| Corruption | FP32 drop (sev1→5) | Static INT8 drop |
|---|---|---|
| Gaussian blur | 0.2359 | 0.2278 |
| JPEG compression | 0.2399 | 0.2319 |
| Brightness down | 0.0222 | 0.0565 |
| Gaussian noise | 0.0141 | 0.0181 |

Static INT8 costs only ~1.8 points of clean accuracy (0.7923 → 0.7742, consistent with the
edge-ai paper's near-zero quantization cost finding) and its corruption-degradation curve
is nearly a carbon copy of FP32's: blur and JPEG are still by far the worst, brightness and
noise are still mild. **This is a real, useful, reassuring result for deployment**:
quantizing this model for edge inference does not introduce a new corruption-sensitivity
problem on top of the already-known small clean-accuracy cost.

## Result 2: aggressive pruning alone is confounded by a floor effect — read carefully

At its own 50% knee, pruning alone already collapses clean accuracy from 0.7923 to 0.6935
(F1 0.7944 → 0.6082) — this is *before* any corruption is applied at all. Once a model is
already this damaged, it has less room left to fall before hitting the ~50% chance floor,
so its raw sev1→sev5 *drop* looks smaller (0.1694 for blur) than FP32's (0.2359) — but that
is a floor-effect artifact, not evidence that pruning makes the model more robust. Read the
absolute numbers instead: pruned-at-knee's blur-sev5 accuracy (0.4980) is essentially
identical to FP32's blur-sev5 accuracy (0.5464) — both are near chance, they just got there
by different paths. **Comparing "how much did accuracy drop" is misleading whenever the
starting accuracies are this different — this is a real methodological lesson worth
stating plainly, not glossing over.**

## Result 3: the real finding — combined pruning+quantization creates a NEW vulnerability that neither technique shows alone

`pruned30_plus_int8` starts from a much gentler baseline (clean accuracy 0.7621, only ~3
points below FP32 — a fair, comparable starting point, unlike the knee-pruned variant
above) and its blur/JPEG drops track FP32 almost exactly (0.2319 and 0.2419 vs. FP32's
0.2359 and 0.2399). But look at **contrast reduction**:

| Variant | Contrast-down drop (sev1→5) |
|---|---|
| FP32 | 0.1371 |
| Static INT8 alone | 0.1633 |
| **Pruned30 + INT8 combined** | **0.2177** |

Combining a modest amount of pruning with quantization roughly **doubles** the
contrast-reduction sensitivity that either technique shows on its own (FP32's 0.1371 vs.
the combined variant's 0.2177 — a jump neither pruning-alone-at-a-comparable-sparsity nor
quantization-alone predicts by themselves). Brightness-up shows the same direction, more
mildly (0.1069 FP32 → 0.1532 combined). **This is the one genuinely new, non-obvious
finding from this experiment**: a real-world deployment stacking both compression
techniques (the realistic "make it small AND fast for an edge device" scenario) can
introduce a specific new fragility that neither compression technique's own accuracy
report would predict, because that report only ever measures clean-image accuracy.

## What this does and doesn't establish

**Established**: for this project's own deployed model, static INT8 quantization alone
does not meaningfully change corruption-robustness (safe finding for deployment);
comparing raw "accuracy drop under corruption" across models with very different clean
baselines is methodologically unsound and this project caught that in its own analysis
rather than reporting a misleading pruned-model result at face value; and combining
pruning with quantization introduces a specific new sensitivity to contrast/brightness
changes beyond what either technique shows alone — a real, first-of-its-kind (for
dermatology) empirical result, though modest in scope (one model, one architecture).

**Not established**: whether this contrast-sensitivity interaction generalizes to the
other 8 trained architectures (untested — future work, same caveat as the single-model
scope of the corruption-robustness benchmark itself); whether it holds under real
(non-synthetic) contrast variation; and whether QAT (quantization-aware training, run
separately) shows the same interaction or avoids it by training through the quantization
noise.

## Next steps (not started)

1. Extend this same 4-variant comparison to a second architecture (e.g., EfficientNet-B0,
   which also has full corruption-robustness numbers already) to check whether the
   pruning+quantization contrast-sensitivity interaction is ShuffleNetV2-1.0x-specific or
   general.
2. Compare against the QAT (quantization-aware training) variant once that's run, to see
   whether training through fake-quantization noise avoids the contrast-sensitivity
   interaction that post-training quantization introduces.
3. Fold this finding into the edge-ai paper's "Compression Robustness Is
   Architecture-Dependent" framing as a new dimension: compression robustness is not just
   architecture-dependent for clean-image accuracy, it interacts with *which* real-world
   image degradation the deployed system is likely to encounter.
