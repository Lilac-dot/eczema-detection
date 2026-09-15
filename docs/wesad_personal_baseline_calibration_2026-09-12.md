# Stage A Fix: Personal-Baseline Calibration — 2026-09-12

## Why

A review of this repo's methodology flagged that Stage A's headline "0.899 mean AUC"
(CNN+attention) hid a severe per-subject problem: several LOSO-CV folds had AUC near 1.0
(the model ranks stress above non-stress essentially perfectly) but F1 as low as 0.00,
because the decision threshold tuned on one subject's validation data didn't transfer to
a different test subject. The per-fold threshold ranged from 0.01 to 0.84 across the 15
folds -- an 84x spread, meaning there was no single fixed cutoff that would work for a new
person. See the "Threshold-transfer instability" section added to
`docs/wesad_stress_and_fusion_2026-08-27.md` for the original evidence.

**Hypothesis**: both models normalized every subject against a single pooled mean/std
(computed per LOSO fold, but still shared across every subject in that fold). Physiological
signals like EDA and skin temperature have large, genuine between-person differences in
absolute level, so pooled normalization leaves those differences sitting in the model's
input -- a plausible driver of the threshold-transfer failures. Fix: normalize (or feature
-ize) each subject's signal relative to their OWN baseline (calm, pre-stress-task) period,
not a pooled group statistic. This is standard practice in this literature and mirrors how
a real wearable would have to work anyway: calibrate briefly against the wearer's own calm
state, then monitor deviations from it.

## What changed

- `build_wesad_raw_windows.py` and `build_wesad_features.py` now retain the raw
  1/baseline/2/stress/3/amusement condition label (previously collapsed straight to the
  binary stress/not-stress target), so each subject's own baseline-condition windows can be
  identified.
- `build_wesad_features.py` adds `add_personal_baseline_features()`: for every extracted
  feature (e.g. `EDA_mean`, `TEMP_slope`), a `_rel` column is added, expressing that
  window's value relative to that subject's own baseline-condition average of the same
  feature. Both the absolute and `_rel` columns are kept (52 features total, up from 26).
  `condition` is written to the CSV for transparency but explicitly excluded from
  `feat_cols` in `train_wesad_stress.py` — it directly encodes the label (`condition==2`
  means stress) and must never be a model input.
- `train_wesad_cnn_attention.py`'s normalization was changed from one pooled mean/std per
  modality per fold to `wesad_calibration.compute_subject_baseline_stats()`: every
  subject's own baseline-condition windows set their own per-modality (mean, std), applied
  to ALL of that subject's windows (train, val, test alike -- every subject calibrates
  against only their own data, so this isn't leakage).
- New shared modules: `wesad_features.py` (window feature extraction, now used by both
  `build_wesad_features.py` and `fusion_pipeline.py` so they can't drift apart) and
  `wesad_calibration.py` (the baseline-stats/normalization helpers, used by both the CNN
  training script and the fusion pipeline).
- `scripts/loso_report.py`'s per-subject reliability summary (added in the prior review
  pass) runs automatically after every retrain, so this kind of instability can't go
  unnoticed again.

## Results

### LightGBM (`train_wesad_stress.py`) — clear improvement

| | Before | After |
|---|---|---|
| Mean AUC | 0.871 | **0.9405** |
| Mean F1 | 0.623 | 0.6352 (small change) |
| Pooled AUC | 0.816 | **0.858** |
| Pooled F1 | 0.646 | **0.7602** |
| Per-fold threshold range | 0.01 – 0.875 (88x) | **0.23 – 0.655 (2.8x)** |
| F1 std across folds | ~0.25-0.31 (per original doc) | 0.283 (no clear change) |
| Subjects with F1 < 0.3 | 1/15 (S14, F1=0.00) | 1/15 (S5, F1=0.00 -- different subject, same count) |

Not every summary statistic moved (F1 mean and std are both roughly flat, and exactly one
subject still lands at F1=0.00 either way -- this fix does not eliminate per-subject
failure, it relocates it). The real, unambiguous win is in the three subjects that were
previously catastrophic specifically because of threshold miscalibration, not poor
ranking -- their AUC was fine before, but the threshold sent them to near-total false
positives:

| Subject | Accuracy before | Accuracy after |
|---|---|---|
| S7 | 28.6% | **97.1%** |
| S13 | 28.6% | **57.1%** |
| S17 | 30.6% | **83.3%** |

And the per-fold threshold range collapsing from 88x to 2.8x is the direct, mechanistic
reason why: there is now much less need for the "right" cutoff to vary wildly from person
to person. Top features by gain are now dominated by `_rel` (baseline-relative) columns:
`EDA_range_rel`, `TEMP_min_rel`, `ACC_mag_mean_rel`, `TEMP_slope_rel`, `BVP_dominant_hz_rel`
are all in the top 10 -- direct evidence the personal-baseline signal is real and useful,
not just noise the model happened to fit.

### CNN+attention (`train_wesad_cnn_attention.py`) — mixed, arguably worse

| | Before | After |
|---|---|---|
| Mean AUC | 0.899 | 0.889 |
| Mean F1 | 0.628 | 0.677 |
| F1 std (fold variance) | 0.25–0.31 | 0.349 (higher) |
| Per-fold threshold range | 0.01 – 0.84 (84x) | **0.01 – 0.97 (97x, worse)** |
| Worst fold | AUC 1.0, F1 0.00 (S2) | **AUC 0.00, F1 0.00 (S14) — a new failure mode: the model's ranking is inverted for this subject, not just its threshold** |

Mean AUC stayed roughly flat and F1 improved, but the specific problem this fix targeted
(threshold instability) got worse, not better, and a new and arguably more concerning
failure appeared (a fold with AUC exactly 0.0). A plausible reason: each fold's CNN
mini-batches mix windows from several different subjects, each already re-scaled to their
own baseline (mean 0, std 1 in their own frame) before batching. BatchNorm inside each
modality branch then computes statistics over a mixture of individually-personalized but
differently-shaped "deviation from calm" distributions, which may be harder for a small,
freshly-initialized network (13 training subjects per fold) to fit consistently than the
previous single shared pooled scale was. This is a plausible mechanism, not a proven one --
treat it as a hypothesis, not a fact, if it comes up again.

### Follow-up: seed ensembling (completed, 2026-09-14) — did not fix it

Hypothesis: each fold trains one small network from scratch on only ~13 subjects, so a
single unlucky random initialization could plausibly explain the S2 (AUC=1.0, F1=0.00)
and S14 (AUC=0.00) failures above. The standard fix for "one bad training run" is not more
regularization (already tried in the v1-vs-v2 comparison and it made things worse) but
averaging several independently-initialized copies, so one bad copy gets outvoted. This
was implemented (`train_ensemble()`/`ensemble_predict()` in `train_wesad_cnn_attention.py`,
3 independently-seeded models per fold, averaged sigmoid probability) and run to completion
across all 15 folds plus the final model.

**Result: mixed to negative, not a fix.**

| | Single model (post-calibration) | 3-model ensemble |
|---|---|---|
| Mean AUC | 0.8894 | 0.8920 (~flat) |
| Mean F1 | 0.6772 | 0.6474 (worse) |
| F1 std | 0.3486 | 0.3811 (worse) |
| Threshold range | 0.01 – 0.97 (97x) | 0.01 – 0.82 (82x, still huge) |
| Subjects with F1 < 0.3 | 3 (S3, S14, S15) | 3 (S3, S14, S17) |
| S14 test AUC | 0.0000 (inverted ranking) | 0.0095 (still inverted) |
| Pooled AUC | 0.8478 | 0.8864 (improved) |
| Pooled F1 | 0.7800 | 0.7741 (~flat) |

Ensembling did NOT fix S14's inverted-ranking problem (AUC still ~0 with 3 independent
seeds, so this isn't one unlucky initialization -- something about that subject's signal
is genuinely adversarial to this architecture post-calibration). It also didn't reduce
per-fold variance; if anything F1 std went up. It did fix S15 (F1 0.24 -> 0.46, now above
the 0.3 cutoff) but broke S17 in the process (F1 0.64 -> 0.00, newly below 0.3) -- a
lateral move, not a net improvement. Pooled AUC improved somewhat (0.848 -> 0.886), which
makes sense (averaging reduces noise in aggregate ranking) but this does not translate to
the per-subject reliability this fix was actually trying to achieve.

**Conclusion: the CNN's instability is not primarily an unlucky-initialization problem**,
which rules out the most tractable, non-regularization fix available. Whatever is
happening with subjects like S14 is either a genuine property of their signal under
per-subject calibration, or a deeper architectural mismatch (see the BatchNorm-on-mixed-
personalized-batches hypothesis above) that ensembling initialization alone cannot
average away. This strengthens, rather than weakens, the decision below to deploy
LightGBM instead of the CNN.

## Decision: Stage A's deployed model changes from CNN to LightGBM

Since the fix, LightGBM is both more accurate (mean AUC 0.9405 vs. 0.889) and far more
stable (threshold range 2.8x vs. 97x) than the CNN. `scripts/fusion_pipeline.py` has been
updated to load `models/wesad_stress_lightgbm.txt` instead of
`models/wesad_stress_cnn_attention.pt`, using `wesad_features.extract_window_features()` +
`wesad_calibration.apply_feature_baseline()` in place of the old raw-signal CNN forward
pass. `stage_a_stress_predict()`'s signature changed accordingly: it now takes a
`baseline_profile` (a per-feature personal calibration profile from
`compute_feature_baseline_profile()`), not a `baseline_stats` per-modality (mean, std)
tuple.

The CNN+attention model and its training script are kept (not deleted) as a documented
comparison, the same way this project has treated every other negative/mixed result
(the v1-vs-v2 fine-tuning comparison, the cropped/normalized Stage B experiments).

## Remaining caveats (still true after this fix)

- **This is still stress detection, not eczema-severity detection.** Nothing about this
  fix changes that scope; see the original scope note in `README.md` and
  `docs/wesad_stress_and_fusion_2026-08-27.md`.
- **A real deployment now explicitly requires a calibration step.** Both the LightGBM and
  CNN versions of Stage A need a new user's own short calm-period recording before any
  prediction is meaningful for them -- there is no fixed global normalization anymore. This
  is a more honest design (it matches how a real wearable would have to work) but it is a
  real added requirement, not a free improvement: a system that skips calibration for a
  new person has no basis for trusting its output for them.
- **1 subject (S5) is still poorly served by LightGBM** (F1 0.00 despite AUC 0.90 -- a
  threshold-transfer failure, not a ranking failure). The CNN's S14 failure (AUC ~0.00,
  an inverted ranking) survived a 3-model seed ensemble unchanged, confirming it isn't
  initialization noise. n=15 subjects remains too small to know if these are stable
  individual differences or sampling noise.
- **The fusion demo's specific example pair (subject S17) shows the recalibrated LightGBM
  model scoring S17's known-stress window (P=0.289) LOWER than their known-calm window
  (P=0.416)** -- backwards. This is left as-is rather than swapped for a better-looking
  example, consistent with this project's existing practice of not cherry-picking demo
  output (see the original fusion demo caveat in
  `docs/wesad_stress_and_fusion_2026-08-27.md`). It's a real illustration that per-subject
  performance still varies meaningfully even after this fix.
