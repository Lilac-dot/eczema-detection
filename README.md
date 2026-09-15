# Multi-Sensor Wearable System for Monitoring Eczema Severity

Honors project: image-based eczema (atopic dermatitis) detection as the core diagnosis,
with wearable sensor data (stress, sleep quality) as supporting factors — literature-backed
drivers of eczema flares — fused with the image score into one composite. See
`docs/architecture_v2_2026-09-14.md` for the reasoning behind this structure, including
why lesion-vs-surrounding-skin temperature (a candidate factor) is deferred to future work.

## Stages

- **Stage B — image-based eczema diagnosis (the core stage).** Eczema vs. 7 visually-similar
  skin conditions (Psoriasis, Tinea, Candidiasis, Infestations/Bites, Lichen, Drug Eruption,
  Rosacea), curated from a same-source 20-class dermatology dataset specifically to avoid
  a shortcut-learning problem found in the original Eczema-vs-Normal approach (see
  `docs/dataset_usability_check_2026-08-26.md`). ResNet18 CNN, 81.07% test accuracy,
  F1 81.18%. Scripts: `scripts/train_curated_cnn_balanced.py`,
  `scripts/build_curated_subset*.py`.
- **Stage A-stress — wearable stress detection.** Trained on the public WESAD dataset
  (wrist EDA, temperature, BVP, accelerometer). Two architectures compared with
  leave-one-subject-out cross-validation (15 subjects is too few for a single held-out
  split to be reliable): LightGBM on hand-crafted window statistics, personal-baseline
  calibrated (mean AUC 0.9405, the better and by far the more stable of the two — this is
  the model `scripts/fusion_pipeline.py` actually uses), and a 4-branch 1D-CNN with
  attention-gated fusion across modalities (mean AUC 0.889, kept as a documented
  comparison). Scripts: `scripts/build_wesad_*.py`, `scripts/train_wesad_*.py`.
  **Calibration requirement**: both models now normalize each subject against their OWN
  baseline-condition recording rather than a pooled statistic, which fixed a severe
  per-subject decision-threshold instability found in review (see
  `docs/wesad_personal_baseline_calibration_2026-09-12.md` for the full before/after). This
  means a real deployment needs a short calm calibration recording from each new wearer
  before predictions are meaningful for them — there is no fixed global normalization.
  **Reliability caveat**: even after this fix, a mean AUC is not a promise about every
  individual — see `scripts/loso_report.py`'s per-subject breakdown, run automatically
  after every retrain.
  **Scope note**: this detects general physiological stress, which is a documented driver
  of eczema flares — it is not itself validated as an eczema-flare predictor, since no
  dataset pairs stress and eczema severity for the same patients.
- **Stage A-sleep — wearable sleep-quality detection (in progress).** Trained on AAUWSS
  (13 subjects, overnight stays, same Empatica E4 sensor set as WESAD — EDA, temperature,
  BVP, accelerometer — plus AASM-scored sleep stages). Chosen over PhysioNet's DREAMT
  because AAUWSS is open access with no data-use agreement, matching every other dataset
  in this project; see `DATASETS.md`. Poor sleep is a documented eczema-flare driver (the
  itch-scratch cycle disrupts sleep, and disrupted sleep is linked to worse flares), so
  this is a literature-motivated building block, not a validated eczema predictor — the
  same scope caveat as Stage A-stress. Planned to reuse WESAD's personal-baseline
  calibration approach (`wesad_calibration.py`, `wesad_features.py` — same sensor types
  make this directly reusable) and LOSO-CV given the similarly small subject count.
  Scripts: `scripts/build_aauwss_*.py`, `scripts/train_aauwss_sleep.py` (in progress).
- **Future scope — lesion-vs-surrounding-skin temperature.** Considered as a third
  wearable-adjacent factor, but ruled out for now: no public dataset pairs eczema lesions
  with thermal/IR imaging (thermal skin-imaging datasets exist for cancer and pressure
  injuries, not eczema), and there's no way to build one without IR-capable hardware and
  IRB-approved data collection this project doesn't have. Deferred until custom hardware
  makes it possible to collect this data directly, not designed around now.
- **Stage C — fusion.** No dataset anywhere pairs wearable sensor data with skin images
  from the same patients, so this is decision-level (not jointly trained) fusion.
  Primary rule: `flare_risk = image_score × trigger_index(stress, sleep, ...)` — stress and
  sleep are documented flare *triggers*, not independent severity signals, so they gate a
  risk-of-worsening rather than blending into a flat composite (see
  `docs/stress_moderated_fusion_design_2026-09-14.md`). The original flat
  `0.5 × image + 0.5 × stress` average (`fuse()`) is kept in the same script for comparison.
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
  condition labels. Also holds two superseded top-level status reports
  (`AD_Wearable_Project_Report.docx`, `AD_Wearable_Model_and_Project_Report_2026-08-26.docx`)
  that predate the WISDM->WESAD pivot and the fusion pipeline — `docs/` and this README
  are the current source of truth, not those files.

## Setup

```
pip install pandas numpy lightgbm scikit-learn torch torchvision matplotlib pillow python-docx imagehash scipy openpyxl
```

Download datasets per `DATASETS.md`, then run the relevant `clean_*.py` / `build_*.py`
scripts before training.
