# Datasets

Raw data isn't tracked in this repo (too large for git, and all of it is publicly
downloadable). Scripts expect the layout described in each script's docstring; run the
matching `clean_*.py` / `build_*.py` script after downloading.

## WESAD (Stage A — wearable stress detection)

15 subjects, wrist + chest wearable sensor data (EDA, temperature, BVP, accelerometer),
lab stress-induction protocol. Official source (no login required, ~2.5GB zip):
https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html

Note: that server can be very slow. Kaggle mirrors exist and download much faster —
search "WESAD Wearable Stress and Affect Detection" on Kaggle.

Expected layout after extracting: `dataset/WESAD/S<n>/S<n>.pkl` for each subject
(S2-S11, S13-S17).

## AAUWSS (Stage A — wearable sleep-quality detection)

13 subjects, one overnight stay each, wrist-worn Empatica E4 (same sensors as WESAD: EDA,
temperature, BVP/PPG, accelerometer) plus PSG, sleep-staged by a human rater per AASM
guidelines (Wake/N1/N2/N3/REM). Open access, no login or data-use agreement required
(CC-BY-4.0), ~6.1GB zip:
https://zenodo.org/records/16919071

Chosen over PhysioNet's DREAMT (100 subjects, same E4 signal set) specifically because
DREAMT requires PhysioNet registration + signing a Restricted Health Data use agreement,
while AAUWSS is immediately downloadable like every other dataset in this project.

The zip (`aauwss.zip`, extract to `dataset/AAUWSS_raw/`) also bundles full PSG/EEG (`edfs/`,
~11GB) and aligned ECG/PPG (`aligned_sleep_data_set/`, ~1.4GB) — not needed here, since this
project (like WESAD) only uses the wrist-worn E4 channels. Only extract:
- `empatica/subject_<NN>/{EDA,TEMP,BVP,ACC,HR,IBI,tags}.csv` — standard Empatica E4 export
  (row 1 = session start Unix timestamp, row 2 = sample rate in Hz, then samples). Same
  sampling rates as WESAD: EDA/TEMP 4Hz, BVP 64Hz, ACC 32Hz (ACC unit is 1/64g, convert to g).
- `annotations/subject_<NN>_manual_annotation.xlsx` — 30-second-epoch AASM sleep-stage
  labels (columns: index, Event Start Time, Sleep Stage [Wake/N1/N2/N3/REM], Epoch Number,
  Subject).
- `Sleep_Study_Participant_info.csv`, `edf_channel_types.json` — small metadata, not
  essential but harmless to keep.

This selective extraction is ~326MB total, vs. 14.6GB for the full archive.

**Timestamp alignment**: annotation "Event Start Time" values are UTC-equivalent and align
directly with each E4 file's own Unix start timestamp — verified by checking that computed
epoch times never predate a subject's E4 recording start, and cross-checking one subject
(S12) where annotated epochs ran past the E4 recording's end: all four E4 modalities
(EDA/TEMP/BVP/ACC) stopped within the same second of each other, confirming the wearable
itself stopped recording early (most likely a flat battery — these are unattended overnight
recordings) rather than a timestamp offset. `scripts/build_aauwss_raw_windows.py` handles
this by dropping any epoch not fully covered by every modality's recorded range, rather than
applying any timezone correction.

## SkinDisease (Stage B — curated Eczema vs. similar-disease comparison)

20-class dermatology image dataset (DermNet-style, same-source photos across classes),
used to build the shortcut-free Eczema-vs-other-disease comparison. Likely source (verify
this is the exact one before re-downloading — there are several similarly-named Kaggle
sets):
https://www.kaggle.com/datasets/haroonalam16/20-skin-diseases-dataset

## SCIN (Stage B — external validation)

Google/Stanford's crowd-sourced consumer dermatology dataset (Ward et al., *JAMA Network
Open* 2024;7(11):e2446615). 5,033 cases, 10,000+ images, self-photographed by US Google
Search users — used to test whether the deployed Stage B model generalizes beyond its own
DermNet-style training archive. Public GCS bucket, no login, no data-use agreement:
```
https://storage.googleapis.com/dx-scin-public-data/dataset/scin_cases.csv
https://storage.googleapis.com/dx-scin-public-data/dataset/scin_labels.csv
```
Images referenced by `image_1_path`/`image_2_path`/`image_3_path` columns in
`scin_cases.csv` (e.g. `dataset/images/<id>.png`), fetchable the same way:
`https://storage.googleapis.com/dx-scin-public-data/<image_path>`. Schema documented in full
at github.com/google-research-datasets/scin (`dataset_schema.md`).

`scripts/build_scin_manifest.py` downloads a balanced 488-Eczema / 488-Other subset (one
image per case, `image_1_path` only) to `dataset/SCIN/images/` and writes
`dataset/SCIN/manifest_scin.csv`. See `docs/external_validation_2026-09-15.md` for what this
was used for and the result.

## SkinDisNet (Stage B — external validation)

Clinical smartphone photos from two Bangladesh hospitals (Sultana et al., *Data in Brief*
2025;63:112239, DOI 10.1016/j.dib.2025.112239). 1,710 real images (416 patients) across 6
classes, plus an 11,970-image synthetic-augmentation folder not used here. CC BY-NC 4.0,
academic use only. Mendeley Data, DOI 10.17632/yj3md44hxg:
https://data.mendeley.com/datasets/yj3md44hxg

The page itself is a client-rendered app with no visible download link in the raw HTML; the
actual file list/direct download URLs come from
`https://data.mendeley.com/public-api/datasets/yj3md44hxg` (a plain JSON GET, no auth). The
most recent zip (`SkinDisNet_2.zip`, ~1.38GB) contains `Preprocessed/<class>/` (the real
1,710 images used here), `Augmented/<class>/` (11,970 synthetic augmentations, not used),
and `SkinDisNet_Metadata.csv`.

`scripts/build_skindisnet_manifest.py` merges "Eczema" + "Atopic Dermatitis" into one
positive class (536 images; the split between them is clinically unexplained in the source
paper) against "Contact Dermatitis" + "Scabies" + "Seborrheic Dermatitis" + "Tinea Corporis"
(1,174 images), writing `SkinDisNet/manifest_skindisnet.csv`. See
`docs/external_validation_2026-09-15.md`.

## Eczema Infected + Normal (original Stage B attempt — superseded)

The original Eczema-vs-Normal dataset. Confirmed to have a shortcut-learning problem
(Eczema = clinical photos, Normal = stock photography, so models learned photo source,
not lesion features — see `docs/dataset_usability_check_2026-08-26.md`). Kept only for
historical reference; not used in the current Stage B model. Likely source (verify before
use):
https://www.kaggle.com/datasets/adityush/eczema2

## WISDM (original Stage A attempt — deleted, no longer used)

Smartphone/watch accelerometer activity dataset, used for an early motion-proxy scratch
model. Abandoned: its 20Hz sampling rate can't capture the 100-800Hz signal that actually
distinguishes scratching from other hand motion (see `dead_ends/negative_results/paper_review_adam_sensor_2026-08-26.md`).
Not re-downloaded or referenced by any current script.
