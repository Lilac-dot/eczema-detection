# Moisture Sensor: Pretrained-Model Comparison on the Southampton Patient Data — 2026-09-18

## Why this happened

The architecture now being planned is four modalities feeding one phone-based fusion
model: stress, moisture, scratch, and camera (image). Camera has a mature, actively-compared
pipeline (`docs/transfer_cnn_experiment_2026-09-17.md`); stress has a deployed, calibrated
model (`docs/wesad_calibration_significance_2026-09-16.md`). Moisture and scratch both have
zero trained models. This doc covers a same-day exploration of whether the moisture channel
could get a head start from the real patient data found earlier
(`docs/dataset_usability_check_2026-08-26.md`-style investigation, but for the Southampton
e-textile capacitive sensor thesis dataset, eprints.soton.ac.uk/511063) -- specifically,
whether an existing pretrained model could be used on it rather than training something from
scratch.

## What the data actually is

`dataset/southampton_moisture/PatientTest/` (extracted from the thesis zip; see
`SOURCE.md` there for full attribution). **13 patients**, each with a capacitive-sensor
recording on lesional and non-lesional skin (a few seconds to 30 seconds, one reading per
second -- some patients' sessions stopped early, a real property of the clinical data, not a
loading bug: `load_timeseries()` in the scripts below truncates per-patient rather than
assuming a fixed length). Corneometer and TEWL readings exist too, as the clinical ground
truth the sensor is being validated against -- **not used as model input**, since a deployed
device would only ever have its own sensor reading, not a Corneometer's.

## Why "just fine-tune a pretrained model" doesn't apply the way it does for images

Raised and rejected early in this exploration: there is no equivalent of ImageNet for
capacitive skin-hydration sensors. No one has published a pretrained backbone for this
signal type to transfer from. Separately, training anything from scratch directly on 13
patients was also rejected -- with that few subjects, any model's validation number would be
dominated by which specific patients happened to land in the test set, not by anything the
model actually learned.

**What does genuinely apply**: a different category of pretrained model exists --
tabular foundation models (TabICL, TabDPT, etc.) -- pretrained not on this specific
signal, but on millions of synthetic *small tabular classification problems*, specifically so
they can classify a brand-new small dataset via in-context learning (handing it the training
rows directly, no gradient-descent training step of its own). This is the correct category to
try here: matched to the actual constraint (very few rows), not to the signal type (which
turned out too simple -- see Results).

## Method

`scripts/eval_moisture_sensor_models.py`. Each patient's sensor recording is reduced to 6
features (mean, std, linear-fit slope, first value, last value, range) -- the same
featurization used in the earlier (superseded) LOSO attempt in
`scripts/train_moisture_sensor_loso.py`.

**Test-set isolation**: 13 patients split once, by patient (both a patient's lesional and
non-lesional rows always stay together), fixed seed 42, into **9 train / 4 test**. The 4 test
patients (S-02, S-10, S-11, S-18) were used for exactly one thing: the final evaluation
below. No model selection, threshold tuning, or feature choice used them. Same discipline as
`scripts/eval_transfer_cnn_finalists.py` on the image side, scaled down to fit 13 patients
instead of 507 test images.

**Models attempted**:
- **TabICL** -- pretrained tabular foundation model, in-context learning. Ran successfully.
- **Logistic Regression** -- classical baseline, actually trained on the 18 training rows.
- **LightGBM** -- classical baseline, actually trained on the 18 training rows (same model
  family already deployed for the Stage A stress model).
- **TabDPT** -- a third pretrained tabular foundation model, identified as available
  (`pip install tabdpt`) but not run, given the result below already showed pretraining
  wasn't adding anything over the classical baselines.
- **"LimiX"** -- a tabular foundation model that came up in search, but the `limix` package
  on PyPI is an unrelated, much older genomics statistics library (name collision) -- not
  actually available under that name.

## Results

Test set: 4 held-out patients, 8 recordings (4 lesional, 4 non-lesional).

| Model | Accuracy | AUC | F1 |
|---|---|---|---|
| TabICL (pretrained, in-context) | 0.750 (6/8) | 0.688 | 0.800 |
| Logistic Regression (from scratch) | 0.750 (6/8) | 0.688 | 0.800 |
| LightGBM (from scratch) | 0.750 (6/8) | **0.938** | 0.800 |

All three models made the **same two mistakes** -- both were non-lesional recordings
misclassified as lesional (recall on lesional skin was a perfect 4/4 for every model). That
consistency across completely different model families suggests those two specific patients'
non-lesional readings were genuinely borderline, not that any one model handled them worse.

## Interpretation

**The pretrained model did not outperform the classical baselines.** TabICL tied logistic
regression exactly and lost to LightGBM on ranking quality (AUC 0.688 vs. 0.938). This
confirms what was flagged before running anything: the underlying signal (a 20-30 second
capacitance settling curve, reduced to 6 summary features) is simple enough that a model
pretrained on generic small-tabular-classification priors has nothing extra to offer over a
handful of trees or a linear model.

**The real bottleneck is sample size, not model choice.** 8 test predictions from 4
independent patients means a single flipped call moves accuracy by 12.5 points. Nothing in
this comparison -- pretrained or classical -- changes that. This result should be read as a
demonstration that the comparison was done honestly (real patient-level split, real held-out
test, no peeking), not as a validated model for anything.

## What this settles, going forward

- No more time should go into trying additional pretrained tabular models on this dataset
  (TabDPT, or anything else) -- the category has already been tested and shown not to move
  the result here.
- The moisture channel's real next step is the same one already identified before this
  exploration: it needs its own hardware built and its own larger patient dataset collected.
  This 13-patient dataset remains useful only as a calibration reference (expected value
  ranges, lesional-vs-healthy direction of effect), not as a training or transfer-learning
  source.

## Files

- `dataset/southampton_moisture/PatientTest/` -- extracted patient CSVs (13 patients,
  sensor + Corneometer + TEWL), `SOURCE.md` for attribution.
- `scripts/eval_moisture_sensor_models.py` -- the patient-wise, test-isolated comparison run
  above (the one to reuse/extend).
- `scripts/train_moisture_sensor_loso.py` -- an earlier, superseded leave-one-subject-out
  attempt (same features, different validation protocol); kept for reference, not the basis
  for the results in this doc.
