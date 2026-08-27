# Dataset Usability Check — 2026-08-26

## Scope

Two datasets in this project:

- **Eczema/Normal images** (`dataset/Eczema/`, `dataset/Normal/`) — Stage B (image severity model)
- **WISDM** motion data (`dataset/WISDM/`) — Stage A (motion/scratch model)

## Starting point

A full cleaning pass over both datasets had already been run earlier today
(`scripts/clean_dataset.py`, `scripts/clean_wisdm.py`, `scripts/split_dataset.py`,
`scripts/extract_color_features.py`), with results narrated in
`Progress_Log_2026-08-26.docx`. Rather than re-running that (expensive, and would
just reproduce the same numbers), this pass **independently verified that the
cleaned state is still correct and asked whether it's genuinely trustworthy**, not
just present. Nothing was changed, moved, or deleted — this was a read-only audit.

## Checks performed

1. Re-opened every one of the 2,757 images listed in `manifest_clean.csv` with
   Pillow to confirm each still decodes.
2. Diffed `manifest_clean.csv` against what's actually on disk in `dataset/Eczema/`
   and `dataset/Normal/`, both directions (manifest → disk and disk → manifest).
3. Verified `manifest_train.csv` / `manifest_val.csv` / `manifest_test.csv` form a
   true partition of `manifest_clean.csv`: no overlaps between splits, union is
   exact, class balance matches the reported 70/15/15 stratified split.
4. Cross-checked `dataset/_quarantine/` file counts against `clean_report.csv`'s
   reason counts.
5. Recounted lines in all 204 cleaned WISDM files under `dataset/WISDM/raw_clean/`
   and diffed against `dataset/WISDM/clean_report.csv`'s `rows_kept` column;
   spot-checked line format (6 comma-separated fields per row).
6. Loaded both trained model artifacts (`models/stage_b_resnet18.pt`,
   `models/stage_b_lightgbm.txt`) to confirm neither is corrupted.
7. Independently quantified the previously-flagged shortcut-learning risk using
   the existing `dataset/color_features.csv`: computed per-class mean/std of the
   two features LightGBM ranked as most important.
8. Checked `dataset/WISDM/wisdm-dataset/arff_files/` — this turned out to contain
   200 genuine per-subject ARFF feature files, not just filesystem junk as an
   initial directory listing suggested.

## Results

### Eczema/Normal image dataset — usable, with a known caveat

- All 2,757 images in the clean manifest are present and still decode correctly;
  zero drift between the manifest and disk in either direction.
- Quarantine folder matches the report exactly: 104 exact duplicates + 262
  near-duplicates = 366.
- train/val/test split is clean and non-overlapping: 1,929 / 413 / 415 images,
  class-balanced per split (matches the original report).
- Images are a mix of `.jpg` (1,428 Eczema + 1,053 Normal) and `.png` (276
  Normal) — not itself a problem since the pipeline converts everything to RGB,
  but worth knowing if a future step assumes a single format.
- **Known issue, confirmed and quantified here (not new — see
  `Progress_Log_2026-08-26.docx` §3.4 and §8):** Eczema and Normal images come
  from two different photo sources (clinical macro photography vs. stock
  photography), and a classifier can separate the classes using nothing but a
  whole-image brightness statistic. Measured directly from
  `color_features.csv`: mean of `MEAN_XYZ_2` (overall brightness/luminance) =
  **0.15 for Eczema vs. 0.58 for Normal** (class std devs 0.07 and 0.26) — roughly
  a 4x gap between class means relative to within-class spread. This is why a
  CNN (95.66% test accuracy) and a colour-features-only LightGBM model (95.18%)
  land on almost the same number: neither needs to see the lesion to hit ~95%.
- **Verdict:** usable for pipeline/engineering purposes — loading, training,
  and evaluation code all run end-to-end correctly on this data. **Not** usable
  yet for a genuine eczema-vs-healthy-skin accuracy claim; same-source imagery
  for both classes is a prerequisite fix before reporting any number from this
  dataset (already flagged as the top priority in the existing progress log).

### WISDM motion dataset — usable

- All 204 cleaned files present and well-formed; total kept-row count
  (15,364,411) matches `clean_report.csv` exactly, file by file.
- No corruption or malformed lines found on spot-check.
- No model has been trained on this data yet — it's cleaned and ready, but
  Stage A (motion/scratch classifier) itself hasn't been built.
- Correction to the earlier session's assumption: `arff_files/` is not empty —
  it holds 200 real pre-extracted ARFF feature files, which could save the
  effort of hand-rolling feature extraction from the raw sensor data for Stage A.

### Model artifacts

Both `models/stage_b_resnet18.pt` and `models/stage_b_lightgbm.txt` load without
error, so the existing checkpoints are usable as-is if needed for further
evaluation or fine-tuning.

## Bottom line

Both datasets are technically sound right now — no corruption, no missing
files, nothing that would break a training pipeline, and everything cleaned
earlier today is still intact and self-consistent. The one real blocker is the
confirmed cross-class source confound in the Eczema/Normal image set, which is
a **data-collection problem, not a code or corruption problem** — it needs
same-source imagery for both classes before any accuracy number from that
dataset can be trusted or reported.

## What this doesn't cover

This was a verification/audit pass, not new cleaning work. It doesn't repeat
the original cleaning methodology (documented in `Progress_Log_2026-08-26.docx`
§2–§4) and doesn't attempt the dataset fix recommended there (re-sourcing
same-style imagery, removing watermarked images). No files were changed.
