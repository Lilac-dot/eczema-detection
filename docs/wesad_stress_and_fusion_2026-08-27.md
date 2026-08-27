# Stage A Pivot (Stress) + Stage C Fusion — 2026-08-27

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
          + 0.5 * (Stage A stress score, WESAD CNN v1, mean AUC 0.899)
```

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
