# Stage A Pivot (Stress) + Stage C Fusion — 2026-08-27

> **Update, 2026-09-12**: the numbers and the CNN-as-deployed-model decision below are
> superseded. Personal-baseline calibration fixed the threshold-transfer instability
> documented in this file's "Threshold-transfer instability" section, and Stage A's
> deployed model changed from the CNN to LightGBM as a result. See
> `docs/wesad_personal_baseline_calibration_2026-09-12.md` for the full before/after
> comparison and reasoning. This file is kept as-is otherwise for the historical record of
> how Stage A was built and how the instability was first found.

## Why this exists

Stage A's original approach (WISDM teeth-brushing proxy for scratch detection) has a
hardware bandwidth ceiling that no amount of modeling can fix (see
`docs/paper_review_adam_sensor_2026-08-26.md`), and building a real scratch-labelled
dataset would need IRB approval that isn't available. Stress is a documented,
literature-backed driver of eczema flares (Khan et al. 2024, `skinhd_4_5_e449.pdf`), and
unlike scratch detection, there's a public, well-established benchmark dataset for
wearable stress detection: **WESAD** (Schmidt et al. 2018). This pivots Stage A to
"wearable stress detection," validated on WESAD, while being explicit that WESAD's
"stress" (a lab social-stress test on healthy adults) is not eczema-patient data — this
model is a literature-motivated building block, not a validated eczema predictor.

## Dataset

WESAD, 15 subjects (S2-S11, S13-S17), wrist-worn Empatica E4 modalities only (EDA 4Hz,
TEMP 4Hz, BVP 64Hz, ACC 32Hz) — chest/RespiBAN data intentionally excluded, since the
planned hardware replica is wrist-worn. Binary target: stress (TSST) vs. not-stress
(baseline/amusement); meditation and transient/undefined segments dropped. All 15
subject `.pkl` files verified loadable with expected structure before use. Redundant raw
exports (`respiban.txt`, `E4_Data` CSVs — same data as the `.pkl` in another format)
deleted to save ~4GB; only `.pkl` + small metadata kept.

## Model 1: LightGBM on hand-crafted features (`train_wesad_stress.py`)

60s non-overlapping windows, ~6 stats per modality (mean/std/min/max/range/slope,
plus a crude dominant-frequency heart-rate proxy for BVP) — 520 windows, 26 features.
Evaluated with **leave-one-subject-out CV** (15 folds), not a single split — 15 subjects
is too few for one held-out split to be reliable, and this matches the protocol Chun et
al. 2021 uses for the closely related scratch-detection task.

**Result: mean AUC 0.871, mean F1 0.623, pooled AUC 0.816, pooled F1 0.646.** Top
features by gain were EDA range, temperature mean/max/slope, motion magnitude, and BVP
dominant frequency — physiologically sensible (sweat response, vasoconstriction, heart
rate, movement), a good sign the model learned real signal.

## Model 2: CNN + attention fusion on raw signal (`train_wesad_cnn_attention.py`)

30s windows, 15s stride (overlapping, only within-subject so no leakage across the
subject-level split) for more training examples — 2,140 windows. Four separate 1D-CNN
branches (one per modality, since sampling rates and signal character differ too much
for one shared network), fused via a learned attention gate over the 4 modality
embeddings rather than plain concatenation, so the model learns which sensor to weight
more per example. A Transformer was deliberately not used — not enough data (15
subjects) for self-attention over long raw sequences to learn anything but noise. Same
LOSO-CV protocol, fold-local z-score normalization (train-fold stats only, no leakage).

**Result: mean AUC 0.899, mean F1 0.628, pooled AUC 0.831, pooled F1 0.667** — a small
edge over LightGBM, but not a meaningful one given both models show huge fold-to-fold
variance (F1 std 0.25-0.31) that dwarfs the gap between them. **Honest finding: added
architectural complexity did not meaningfully outperform hand-crafted features + gradient
boosting at this sample size (n=15).** Worth stating as a real conclusion, not a failed
experiment. Attention weights were fairly balanced across modalities (EDA 27%, TEMP 27%,
BVP 18%, ACC 29%) — no single sensor dominated, which supports the multi-sensor design.

The same handful of subjects (notably S14, S17) scored poorly under *both* models —
suggestive of genuine individual differences in autonomic stress response, not noise in
either model. Worth naming explicitly as a limitation.

### Threshold-transfer instability (added after re-reviewing the per-fold results directly)

The mean AUC on its own overstates how deployable this is, and the saved per-fold CSV
(`dataset/WESAD/wesad_cnn_loso_fold_results.csv`) shows exactly why. Per-subject test
results, worst-F1-first:

| Subject | Test AUC | F1 | Accuracy | Threshold used |
|---|---|---|---|---|
| S2 | 1.000 | 0.000 | 0.711 | 0.840 |
| S14 | 0.541 | 0.175 | 0.674 | 0.050 |
| S3 | 0.540 | 0.321 | 0.449 | 0.310 |
| S7 | 1.000 | 0.450 | 0.291 | 0.010 |
| S17 | 0.566 | 0.480 | 0.315 | 0.300 |
| S11 | 0.918 | 0.494 | 0.375 | 0.450 |
| S13 | 1.000 | 0.571 | 0.559 | 0.035 |
| S8 | 1.000 | 0.589 | 0.577 | 0.345 |
| S10 | 0.999 | 0.752 | 0.790 | 0.020 |
| S15 | 0.934 | 0.846 | 0.916 | 0.480 |
| S5 | 0.998 | 0.857 | 0.903 | 0.605 |
| S6 | 0.990 | 0.880 | 0.937 | 0.575 |
| S9 | 1.000 | 1.000 | 1.000 | 0.365 |
| S4 | 1.000 | 1.000 | 1.000 | 0.585 |
| S16 | 1.000 | 1.000 | 1.000 | 0.720 |

S2 and S7 both have **perfect AUC (1.0) but F1 of 0.00 and 0.45** — the model's raw
probability *ranking* is flawless for these subjects, but the decision threshold (tuned
on that fold's validation subject, applied to the test subject) lands in the wrong place
on their score distribution, so it predicts almost everyone the same class. The threshold
column ranges from 0.01 to 0.84 across the 15 folds — an 84x spread — meaning there is no
single fixed cutoff that would work well for a new, unseen person.

**What this means:** the 0.899 mean AUC is a real, correctly-measured ceiling on ranking
quality, not a promise about deployed accuracy. Any real use of this model needs either
(a) a per-subject calibration step (a short baseline recording to set that person's own
threshold) or (b) reporting AUC alongside per-subject F1/accuracy at a fixed threshold,
never AUC alone. `scripts/loso_report.py` (used by both `train_wesad_stress.py` and
`train_wesad_cnn_attention.py`) now prints this per-subject breakdown automatically at the
end of every training run, instead of only the mean/std.

## Fine-tuning attempt (`train_wesad_cnn_attention_v2.py`) — made things WORSE

Targeted the high fold-to-fold variance directly: added training-time data augmentation
(Gaussian jitter + small random time-shift per window), more dropout (added inside each
modality branch, classifier dropout 0.3->0.5), stronger weight decay (1e-4->5e-4), and a
learning-rate scheduler (ReduceLROnPlateau).

**Result: mean AUC dropped to 0.829, mean F1 dropped to 0.529** (pooled F1 0.633) — worse
than the untouched v1 model on every metric. Likely cause: the augmentation/regularization
was too aggressive for how little data each fold already has (13 subjects), blurring out
real signal rather than just noise. **Conclusion: v1 (`models/wesad_stress_cnn_attention.pt`)
is the model actually used going forward** — v2 is kept only as a documented negative
result (`models/wesad_stress_cnn_attention_v2.pt`,
`dataset/WESAD/wesad_cnn_v2_loso_fold_results.csv`), the same way the cropped/normalized
Stage B experiments were kept as evidence rather than deleted.

## Stage C: fusion pipeline (`scripts/fusion_pipeline.py`)

No dataset anywhere pairs wearable sensor data with skin images from the same
patients/sessions, so there's nothing to train a joint fusion model on — same missing-data
wall as the scratch and lesion-thermography ideas. Fusion is therefore **decision-level**:
each stage runs independently (already validated on its own data) and their outputs are
combined by an explicit rule, the same way real clinical indices like SCORAD/EASI combine
independently-assessed components rather than being fit to one giant dataset.

```
composite = 0.5 * (Stage B image severity, curated CNN, 81.07% acc / F1 81.18%)
          + 0.5 * (Stage A stress score, WESAD LightGBM, mean AUC 0.9405)
```
(Stage A originally used the CNN v1 here, mean AUC 0.899 -- see the 2026-09-12 update note
at the top of this file for why it changed to the recalibrated LightGBM model.)

Weights are equal by default and explicitly stated as *proposed*, not fit to data (there's
no ground truth for the combined task to fit against).

**Demo result**: ran both real models on real data (a real Stage B test image, real
held-out WESAD windows) and fused them successfully — the pipeline runs correctly
end-to-end. The specific example landed on two individually-weak cases (subject S17's
stress windows were nearly indistinguishable to Stage A, consistent with its known weak
LOSO fold for that subject; one "Other" test image was misclassified as Eczema-leaning by
Stage B, consistent with its ~80% precision). Left as-is deliberately rather than
cherry-picked, since it's an honest illustration of where the system's real limitations
show up, not a pipeline bug. **Important caveat repeated from the fusion design
discussion: the image and the wearable window in any demo are never from the same real
person/moment — no dataset provides that pairing — so the composite score is a
demonstration of the architecture, not a validated joint clinical claim.**

## What to cite in the paper

- WESAD (Schmidt et al., ACM ICMI 2018) for the dataset itself.
- Khan et al. 2024 for the eczema-stress clinical rationale.
- SCORAD/EASI as precedent for expert-weighted composite severity scoring without a
  fitted joint model.
- The v1-vs-v2 comparison as evidence against blindly adding model complexity/regularization
  at small sample sizes — a genuine, reportable methodological finding.
- The threshold-transfer instability above (AUC=1.0 folds with F1 as low as 0.00) as a
  limitation to state explicitly, not omit — report per-subject results, not just the mean
  AUC, wherever this model's performance is cited.
