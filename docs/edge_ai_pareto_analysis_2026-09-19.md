# Pareto (Accuracy vs. Cost) Analysis — Edge-AI Compression Paper (2026-09-19)

## What this is

Re-plots and re-tabulates data already on disk (`edge_simulation_all_architectures_2026-
09-19.json`, `edge_ai_extended_analysis_2026-09-19.json`) into two Pareto views — no new
model runs, purely analysis of existing numbers. Script:
`scripts/edge_ai_pareto_analysis.py`.

## Finding 1, stated plainly even though it's a little uncomfortable: the deployed model isn't Pareto-optimal

| Model (variant) | Size (MB) | Accuracy |
|---|---|---|
| ShuffleNetV2-0.5x (INT8) | 0.64 | 0.7843 |
| ShuffleNetV2-0.5x (FP32) | 1.95 | **0.7984** |
| ShuffleNetV2-1.0x (FP32) — **the currently deployed model** | 5.45 | 0.7923 |
| EfficientNet-B0 (FP32) | 16.20 | 0.8065 |

ShuffleNetV2-0.5x FP32 has both a smaller checkpoint (1.95MB vs. 5.45MB) AND higher
accuracy (79.84% vs. 79.23%) than the currently-deployed ShuffleNetV2-1.0x — meaning
ShuffleNetV2-1.0x is **strictly dominated** on the accuracy/size tradeoff, not a reasonable
middle-ground choice. Only three points across all 9 architectures × 2 precisions (18
total) are actually Pareto-optimal: ShuffleNetV2-0.5x at both precisions, and
EfficientNet-B0 FP32 at the high-accuracy end.

**Why this happened, and what it means**: the original architecture-selection report
picked ShuffleNetV2-1.0x on accuracy/size/latency grounds *before* this compression
analysis existed, using the original (uncompressed) numbers side-by-side — and at that
time no candidate's accuracy was statistically distinguishable from any other after
correction for multiple comparisons (the original report's own finding), so pure accuracy
couldn't drive the choice anyway. This Pareto analysis shows that **once quantization is
factored in, a smaller, cheaper architecture (ShuffleNetV2-0.5x) would have been the
better pick** — not because ShuffleNetV2-1.0x's accuracy was wrong, but because the
original comparison didn't have compression-aware numbers to select from. This is exactly
the paper's own reframing argument (`scripts/build_edge_ai_paper_2026-09-19.py`'s "this
paper tests that assumption... and finds it does not hold" framing) — now with a concrete,
named example of the assumption actually costing something.

## Finding 2: pruning-sensitivity knee varies 3x across architectures, no simple pattern by size

| Architecture | Sparsity knee |
|---|---|
| MobileNetV2-1.0x | 0.2 (most fragile) |
| SqueezeNet1.1 | 0.3 |
| MobileNetV3-Small | 0.4 |
| ShuffleNetV2-0.5x | 0.4 |
| RepGhostNet-0.5x | 0.4 |
| EfficientNet-B0 | 0.5 |
| EfficientNet-Lite0 | 0.5 |
| ShuffleNetV2-1.0x | 0.5 |
| ResNet18 | 0.6 (most robust to pruning) |

No relationship to baseline model size or family — MobileNetV2-1.0x (a mid-sized model)
is the most pruning-fragile, while ResNet18 (the largest) tolerates the most pruning
before collapsing. This matches the original paper's own headline finding
(pruning-sensitivity is architecture-dependent, not predictable from size alone) and
extends it with an explicit ranking across all 9 candidates rather than a qualitative
statement.

## Plots

- `docs/edge_ai_pareto_accuracy_vs_size_2026-09-19.png` — full scatter, all 18 points
  (9 architectures × FP32/INT8), Pareto front highlighted.
- `docs/edge_ai_pareto_accuracy_vs_sparsity_2026-09-19.png` — all 9 architectures'
  accuracy-vs-sparsity curves on one plot.

## Caveat carried over from the original pruning analysis

Pruning's "size" isn't plotted on the size-Pareto axis at all — unstructured pruning's
dense checkpoint does not actually shrink on disk (already established:
`papers/edge-ai-lightweight-deployment/edge_deployment_simulation_2026-09-19.md`), so
including pruned points on a size axis would overstate what pruning alone actually
achieves without a sparse runtime this project doesn't have. The two Pareto views are kept
deliberately separate for this reason, not combined into one "cost" axis.
