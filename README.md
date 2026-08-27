# Multi-Sensor Wearable System for Monitoring Eczema Severity

Honors project: a three-stage system combining wearable sensor data and skin photos to
estimate eczema (atopic dermatitis) status, and a fusion layer that combines them into
one composite score.

## Stages

- **Stage A — wearable stress detection.** Trained on the public WESAD dataset
  (wrist EDA, temperature, BVP, accelerometer). Two architectures compared with
  leave-one-subject-out cross-validation (15 subjects is too few for a single held-out
  split to be reliable): LightGBM on hand-crafted window statistics (mean AUC 0.871), and
  a 4-branch 1D-CNN with attention-gated fusion across modalities (mean AUC 0.899, the
  better of the two). Scripts: `scripts/build_wesad_*.py`, `scripts/train_wesad_*.py`.
  **Scope note**: this detects general physiological stress, which is a documented driver
  of eczema flares — it is not itself validated as an eczema-flare predictor, since no
  dataset pairs stress and eczema severity for the same patients.
- **Stage B — image-based eczema diagnosis.** Eczema vs. 7 visually-similar skin
  conditions (Psoriasis, Tinea, Candidiasis, Infestations/Bites, Lichen, Drug Eruption,
  Rosacea), curated from a same-source 20-class dermatology dataset specifically to avoid
  a shortcut-learning problem found in the original Eczema-vs-Normal approach (see
  `docs/dataset_usability_check_2026-08-26.md`). ResNet18 CNN, 81.07% test accuracy,
  F1 81.18%. Scripts: `scripts/train_curated_cnn_balanced.py`,
  `scripts/build_curated_subset*.py`.
- **Stage C — fusion.** No dataset anywhere pairs wearable sensor data with skin images
  from the same patients, so this is decision-level (not jointly trained) fusion: each
  stage's independently-validated score is combined by an explicit weighted rule, the way
  clinical severity indices like SCORAD/EASI combine independently-assessed components.
  `scripts/fusion_pipeline.py`.

## Repo layout

- `scripts/` — all data cleaning, feature extraction, training, and evaluation code.
- `docs/` — methodology write-ups, results, and honestly-reported limitations for each
  stage (dataset issues found, fixes tried and ruled out, why certain approaches were
  abandoned).
- `DATASETS.md` — where to get the raw data (not tracked in this repo).
- `archive/` — an earlier prototype of this same project (image CNN + two Random Forest
  models), kept for history. Superseded by the current code for methodological reasons
  documented in `docs/` — notably, its stress and scratch models used labels derived
  directly from the same features fed into the model rather than the dataset's real
  condition labels.

## Setup

```
pip install pandas numpy lightgbm scikit-learn torch torchvision matplotlib pillow python-docx
```

Download datasets per `DATASETS.md`, then run the relevant `clean_*.py` / `build_*.py`
scripts before training.
