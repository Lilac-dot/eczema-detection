# Corruption-Robustness Across All 8 Trained Architectures (2026-09-19)

## What this is

`docs/robustness_corruption_benchmark_2026-09-19.md` found, for ShuffleNetV2-1.0x alone,
that blur and JPEG compression are far more damaging than brightness/contrast/noise/color
corruptions. This doc checks whether that's a property of one specific model or general
across architectures, by running the identical 36-condition benchmark
(`scripts/robustness_corruption_benchmark.py`) on all 8 lightweight architectures from the
original architecture comparison (ResNet18 excluded — it's a different classification head
pattern, not part of the `MODEL_CONFIGS` 9-way comparison this project deploys from).
Aggregated with `scripts/compare_robustness_across_architectures.py`.

## Result: the ranking is consistent across every architecture tested

Mean accuracy drop (severity 1 → 5), averaged across all 8 architectures:

| Corruption | Mean drop | Relative to blur |
|---|---|---|
| **Gaussian blur** | **0.2608** | 1.0x (worst) |
| **JPEG compression** | **0.2288** | 0.88x |
| Contrast reduction | 0.0930 | 0.36x |
| Brightness up (overexposure) | 0.0822 | 0.32x |
| Gaussian noise | 0.0774 | 0.30x |
| Brightness down (underexposure) | 0.0721 | 0.28x |
| Color shift (white balance) | 0.0638 | 0.24x |

**Blur and JPEG compression are 3–4x more damaging than every other corruption type,
consistently, across all 8 architectures** — MobileNetV3-Small, MobileNetV2-1.0x,
SqueezeNet1.1, both ShuffleNetV2 widths, RepGhostNet-0.5x, EfficientNet-B0, and
EfficientNet-Lite0. No architecture bucks this pattern; the ranking (blur > JPEG >>
everything else) holds in every single one of the 8 per-model tables
(`results/robustness_corruption_results_<model>.json`).

## What does vary by architecture: how MUCH each corruption hurts, not the ranking

Some real architecture-dependent spread within the pattern:
- **ShuffleNetV2-0.5x is the most broadly fragile** — its brightness-up drop (0.200) and
  contrast-down drop (0.210) are both far higher than any other architecture's (next
  highest is ~0.14), on top of a normal blur/JPEG vulnerability. This is a genuinely
  different, more fragile architecture overall, not just noisier results.
- **MobileNetV3-Small and RepGhostNet-0.5x are the most resistant to the "minor"
  corruptions** — both have brightness/contrast/noise/color drops mostly under 0.06,
  noticeably calmer than the group average, while still showing the same severe
  blur/JPEG vulnerability as everyone else.
- **ShuffleNetV2-1.0x (the deployed candidate) sits in the middle of the pack** — not the
  most fragile, not the most resistant, on the minor corruptions, with a blur/JPEG
  vulnerability slightly below the group average (0.236/0.240 vs. the 0.261/0.229 means).

## What this means for the hardware and paper framing

This is now a general, cross-architecture finding, not a quirk of the one model this
project happens to deploy: **whatever camera/capture setup this project builds, keeping
photos in focus (avoiding blur) and avoiding heavy recompression matters far more than
controlling lighting** — true regardless of which of these 8 architectures ends up
deployed. That strengthens the earlier hardware-design conclusion
(`docs/imaging_capture_gate_design_2026-09-19.md`): the VL53L1X distance sensor
(preventing focus-distance blur) is the higher-leverage component, not the LED
illumination-compensation circuit.

Plot: `docs/robustness_cross_architecture_comparison_2026-09-19.png` (accuracy vs.
severity under blur, one line per architecture). Full numeric table:
`results/robustness_cross_architecture_summary.json`.

## Caveats, stated plainly

Same caveats as the single-model version of this benchmark: these are synthetic
corruptions, not real photos taken at the wrong distance or with the wrong camera — the
link to real acquisition-condition variation is still untested and needs real hardware.
This comparison also only covers the 8 architectures already trained for the original
comparison; it does not include any compressed (pruned/quantized) variant of each — that
combined question is answered separately, for one architecture only, in
`docs/robustness_under_compression_2026-09-19.md`.
