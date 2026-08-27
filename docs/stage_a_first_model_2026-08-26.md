# Stage A — First Model (WISDM Phone-Accelerometer, Teeth-Brushing Proxy)

## Goal

Build the first Stage A (motion/scratch) model. WISDM has no scratch label at all, so
per the existing project notes (`Progress_Log_2026-08-26.docx` §4.2), "teeth" (brushing
teeth) is the closest available proxy for a repetitive hand-motion gesture — this model
predicts **teeth-brushing vs. everything else** from a single ~10-second accelerometer
window, as a first prototype toward an eventual scratch-vs-non-scratch classifier.

Scope for this first iteration: phone accelerometer only (the most-used WISDM signal in
the literature). Watch and gyroscope data are natural next steps, not attempted here.

## Method

1. **`scripts/build_stage_a_features.py`** — parses WISDM's own pre-extracted ARFF files
   (`dataset/WISDM/wisdm-dataset/arff_files/phone/accel/*.arff`), one row per ~10-second
   window per subject. These are WISDM's official windowed statistical/spectral features:
   per-axis histogram bins, average/peak/absolute-deviation/std/variance, 13 MFCCs per
   axis, inter-axis cosine similarity and correlation, and the resultant magnitude — 91
   numeric features per window. The `class` ARFF attribute (which is just the subject id
   repeated on every row, a WISDM labeling artifact) is dropped so it can't leak subject
   identity into the model. Output: `dataset/WISDM/stage_a_features_phone_accel.csv`.
   - 50 subjects (one of the 51 has no phone-accel ARFF file upstream), 23,074 windows
     total, 1,282 (5.6%) labelled "teeth".
2. **`scripts/train_stage_a.py`** — trains a LightGBM binary classifier.
   - **Split: subject-level, not row-level.** Windows from the same subject are highly
     correlated, so a random row split would let the model partly memorize per-subject
     accelerometer signatures rather than learn the activity itself. Subjects were
     shuffled (seed=42) and partitioned 70/15/15 whole (35/7/8 subjects), the same
     principle as the file-level split used for the image dataset. Assignments saved to
     `dataset/WISDM/stage_a_subject_splits.csv`.
   - `is_unbalance=True` for the ~5.6% positive rate; **AUC**, not logloss, used for
     early stopping — an early run using logloss early-stopped after 2 rounds and never
     produced usable probabilities (all predictions stayed below 0.5, i.e. never crossed
     the default decision threshold). AUC as the early-stopping metric fixed this.
   - **Decision threshold tuned on the validation set** (grid search maximizing F1),
     then applied fixed to test — with a 5.6% positive rate, 0.5 is not a meaningful
     default threshold.

## Results

| Metric | Validation (7 subjects, n=3,945) | Test (8 subjects, n=3,507) |
|---|---|---|
| AUC | 0.878 | 0.827 |
| PR-AUC | 0.329 | 0.350 |
| Accuracy (tuned threshold=0.510) | 89.96% | 92.33% |
| Precision (teeth) | 28.07% | 25.00% |
| Recall (teeth) | 56.67% | 25.14% |
| F1 (teeth) | 37.54% | 25.07% |

Confusion matrix, test set (rows = true, columns = predicted):

| | Predicted not-teeth | Predicted teeth |
|---|---|---|
| **True not-teeth** | 3,193 (TN) | 135 (FP) |
| **True teeth** | 134 (FN) | 45 (TP) |

![Stage A confusion matrix](confusion_matrix_stage_a.png)

Top features by gain: `XSTANDDEV`, `YPEAK`, `ZPEAK`, `RESULTANT`, `XAVG`, `YMFCC0`,
`YAVG` — dominated by axis standard deviation, peak values, and overall motion
magnitude, which is consistent with brushing teeth being a small, high-frequency,
repetitive wrist motion rather than a large gross-motor movement (contrast with
walking/jogging, which would be expected to dominate on `RESULTANT`/`AVG` at a larger
scale).

## Interpretation

- AUC of 0.83–0.88 means the model carries real, well-above-chance signal for
  distinguishing teeth-brushing from other activities using accelerometer statistics
  alone — the approach is viable.
- Precision/recall/F1 (~25–38%) are modest, and test performance is noticeably weaker
  than validation (F1 0.375 → 0.251). Because the split is subject-level, this gap is a
  genuine finding, not noise: how people brush their teeth (grip, wrist angle, motion
  amplitude) varies enough between individuals that a model trained on 35 subjects
  doesn't fully generalize to 8 unseen ones. This is expected and worth stating plainly
  in the paper — it's a realistic result for a first cross-subject baseline, not a
  polished final number, and this project has already established (Stage B) that
  suspiciously high first-pass accuracy is a red flag worth interrogating, not a result
  to chase without checking why.
- This also isn't the real target task — brushing teeth is a proxy for "repetitive hand
  motion," not scratching. A meaningful accuracy ceiling on this proxy task doesn't
  guarantee equal performance on real scratch data; it's a pipeline validation step.

## Addendum: a sharper explanation for the modest result

After this was written, a paper the user added (Chun et al. 2021, the "ADAM sensor" —
see `docs/paper_review_adam_sensor_2026-08-26.md`) gave a much more specific reason for
the modest precision/recall here than just "teeth-brushing is an imperfect proxy for
scratching." That paper shows the signal that actually distinguishes scratching from
other hand motion is a **100–800 Hz acousto-mechanic vibration**, and that even a
100 Hz-sampling smartwatch can't capture it (their benchmark smartwatch algorithm
confused hand-waving with scratching for exactly this reason). **WISDM samples at 20 Hz**
— well below even that inadequate bar. So this model's ceiling isn't just about which
activity was chosen as the positive class; the underlying sensor data may structurally
lack the frequency content a genuine scratch/non-scratch distinction would need. Worth
stating plainly in the paper as a specific, literature-backed limitation rather than a
vague "needs more data" caveat.

## Next steps for Stage A

- Add watch accelerometer/gyroscope features (wrist-worn is the actual target sensor
  placement for a scratch detector, whereas phone placement is a weaker analogue).
- Try richer proxies or a multi-class "hand-motion-like" grouping (teeth + writing +
  clapping vs. gross-motor activities) rather than a single proxy class.
- Once real scratch-labelled data exists (or a pilot capture is done), re-validate this
  pipeline (feature extraction → subject-level split → threshold tuning) on the actual
  target label.

## Artifacts produced

- `dataset/WISDM/stage_a_features_phone_accel.csv` — feature table (91 features + label)
- `dataset/WISDM/stage_a_subject_splits.csv` — subject → split assignment
- `models/stage_a_lightgbm.txt` — trained model
- `docs/confusion_matrix_stage_a.png`
- Scripts: `scripts/build_stage_a_features.py`, `scripts/train_stage_a.py`,
  `scripts/make_confusion_matrix_stage_a.py`
