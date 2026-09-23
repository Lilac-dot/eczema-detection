# Corruption-Robustness Benchmark — Stage B Image Model (2026-09-19)

## What this is and why

This is the first concrete result for the "D/E/F" pivot (controlled imaging hardware +
robustness testing + uncertainty-aware deferral + cross-dataset generalization), following
the 2026-09-19 literature review that found: (a) "does image-based AD severity work" is
already answered by multiple groups (Bang et al. 2021, Keio/Atopiyo 2026), so that's not a
usable novelty claim on its own, but (b) nobody has applied a corruption-robustness
benchmark methodology (ISIC-C/Dermnet-C-style) to eczema specifically, and nobody has
tested whether purpose-built low-cost hardware narrows the resulting robustness gap. This
script runs the software-only half of that question — no new hardware or new data download
needed — using the already-trained, already-selected Stage B model.

Script: `scripts/robustness_corruption_benchmark.py`. Model: ShuffleNetV2-1.0x
(`experiments/shufflenet_v2_x1_0/checkpoints/best_overall.pt`), the same architecture
already selected in the architecture-comparison report and used by the edge-deployment
simulation. **Validation set only** (496 images, `manifest_curated_v3_val.csv`) — the
official test-set read for this model already happened
(`results/transfer_cnn_final_comparison_2026-09-18.json`); this is a new exploratory sweep,
so it belongs on validation, the same reasoning the edge-deployment pruning/quantization
sweep already used. Clean-baseline accuracy reproduces exactly (0.7923), confirming nothing
about the loading/eval code is broken.

36 conditions (7 corruption types × 5 severities + clean), full results in
`results/robustness_corruption_results_shufflenet_v2_x1_0.json` (summary) and
`results/robustness_corruption_probs_shufflenet_v2_x1_0.json` (per-image probabilities, for
re-analysis without rerunning inference). Total runtime: 1314s (~22 min) on this CPU-only
machine. The script and its output paths are parameterized by `--model` (see
`docs/robustness_multi_architecture_2026-09-19.md` for the same benchmark run across the
other 7 trained architectures, to check whether this doc's findings are specific to
ShuffleNetV2-1.0x or general).

## Corruption types, and what "severity" means here

Each type has 5 severities (1=barely visible, 5=severely degraded but still a plausible bad
phone photo), applied to the raw image before resize/normalize — see the script for exact
parameters. **These severity levels are this project's own choice, not taken from a
published scale** — no benchmark reviewed (ISIC-C/Dermnet-C) publishes exact per-severity
parameters in a form this project could reuse directly, so the severities are internally
consistent (monotonically worse) but not calibrated against any external reference.

## Result 1: accuracy degradation is highly corruption-specific, not uniform

This is the main finding, and it's a genuine surprise relative to a naive "any acquisition
error hurts the model" hypothesis:

| Corruption | Acc at sev1 | Acc at sev5 | Drop |
|---|---|---|---|
| Gaussian blur | 0.7823 | **0.5464** | **-0.246** |
| JPEG compression | 0.7863 | **0.5464** | **-0.240** |
| Contrast down | 0.7883 | 0.6512 | -0.137 |
| Brightness up (overexposure) | 0.7903 | 0.6835 | -0.107 |
| Gaussian noise | 0.7681 | 0.7540 | -0.014 |
| Color shift (white balance) | 0.7762 | 0.7238 | -0.052 |
| **Brightness down (underexposure)** | 0.7782 | **0.7560** | **-0.022** (non-monotonic — peaks at 0.7964 at sev4) |

**Blur and heavy JPEG compression are catastrophic** (accuracy collapses to near-chance,
~0.55, by severity 5 — F1 drops to 0.18–0.22, meaning the model has essentially stopped
predicting the positive class). **Brightness, contrast, noise, and color shift are much
more forgiving** — even the underexposure condition barely hurts accuracy at all, and
briefly *improves* it (likely coincidental — darker images may suppress some background
texture the model was over-relying on, not a real robustness gain).

**Why this matters for the hardware design**: the original hardware proposal treated
"lighting" and "distance" as roughly co-equal things to control. This result suggests
**focus/blur (which a fixed, correct capture distance directly prevents) matters far more
than illumination control** for this specific model. That reframes the VL53L1X distance
sensor from "nice to have for framing consistency" to "the single highest-leverage thing
the hardware controls" — a testable, falsifiable claim the real hardware build should be
designed to check, not just assumed from this synthetic result.

**Caveat, stated plainly**: this is a *synthetic* corruption benchmark. It shows the model
is sensitive to blur and compression artifacts in principle — it does NOT yet show that a
real physical camera-to-skin distance error produces the same kind of blur, at the same
severity, as this benchmark's Gaussian blur filter. That link (real defocus blur from a
wrong physical distance vs. this synthetic approximation) is untested and is the natural
next experiment once real hardware exists.

## Result 2: calibration (ECE) degrades sharply with blur/JPEG specifically, not generally

Expected Calibration Error tracks the same pattern: clean ECE is 0.0725, and most
corruptions leave it roughly flat or mildly worse (0.04–0.14). Blur and JPEG are the
outliers: **ECE roughly quadruples under severe blur (0.0725 → 0.3282) and triples under
severe JPEG compression (→ 0.2153)**. This means the model isn't just less accurate under
these two corruptions — its confidence becomes actively misleading (it stays confident
while being wrong far more often), which is exactly the failure mode an uncertainty-aware
deferral system needs to catch.

## Result 3: confidence remains a usable trust signal even under severe corruption

Selective-classification (risk-coverage) analysis: sort predictions by the model's own
confidence, keep only the most-confident X%, measure accuracy on the retained subset.

| Coverage retained | Clean acc | Blur sev3 acc | Blur sev5 acc |
|---|---|---|---|
| 100% (no deferral) | 0.7923 | 0.6935 | 0.5464 |
| 50% | 0.9032 | 0.8387 | 0.6815 |
| 20% | 0.9899 | 0.9293 | **0.8182** |

Even at the worst corruption tested (blur severity 5, where blanket accuracy collapses to
near-chance), **deferring the least-confident 80% of predictions and trusting only the most
confident 20% recovers accuracy to 0.8182** — close to the clean baseline. This is a real,
usable result for the "E" (uncertainty-aware decision support) piece: the model's own
softmax confidence, with no extra uncertainty-quantification machinery (no MC-dropout, no
ensembling), already tracks reliability well enough to support a "trust this prediction or
flag for human review" policy, even when the underlying corruption is severe enough to
break raw accuracy.

**Caveat**: this deferral analysis was run per-corruption-type with the corruption type and
severity known in advance (I picked which conditions to test). A deployed system doesn't
know in advance which corruption a given real photo suffers from or how severe it is —
whether confidence-based deferral works this well on a *mixture* of unknown, real-world
degradations (not one known synthetic corruption at a time) is untested and is the natural
next step.

## What this does and doesn't establish

**Established**: a real, quantified, corruption-specific accuracy/calibration degradation
curve for this project's own deployed eczema classifier (not previously measured for AD by
anyone, per the 2026-09-19 literature review) — blur and JPEG compression are far more
damaging than brightness/contrast/noise/color-balance errors — and a real demonstration
that confidence-based selective classification meaningfully recovers reliability under the
worst corruption tested.

**Not established**: whether real acquisition-condition variation (an actual uncontrolled
phone photo, not a synthetic filter) produces the same pattern; whether the planned
hardware (distance + illumination control) actually narrows this gap when built and tested
for real; how confidence-based deferral performs on unknown/mixed real-world corruption
rather than one known synthetic type at a time; and how this compares across the other 7
already-trained architectures (only ShuffleNetV2-1.0x tested so far, since it's the
already-selected deployment candidate) — extending this to the other architectures would
show whether "blur/JPEG matter most, lighting matters little" is a property of this
specific model or a more general finding.

## Next steps (not started)

1. Extend this same benchmark to the other 8 trained architectures
   (`experiments/*/checkpoints/best_overall.pt`) to check whether the blur/JPEG-dominant
   pattern is model-specific or general.
2. Once the ESP32 imaging-gate firmware (`firmware/`, `DEVICE_ROLE_IMAGING_GATE`) has real
   hardware to run on, capture a small real dataset varying distance (in and out of the
   VL53L1X's tolerance band) and compare real-world blur severity against this synthetic
   benchmark's severity scale, to check whether the synthetic proxy is a reasonable stand-in.
3. Re-run the risk-coverage analysis on a *mixture* of corruption types at random severities
   (simulating "an unknown real photo with some unknown problem"), not one type at a time,
   to test whether confidence-based deferral holds up under more realistic uncertainty
   about what's wrong with a given photo.
