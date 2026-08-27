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

## SkinDisease (Stage B — curated Eczema vs. similar-disease comparison)

20-class dermatology image dataset (DermNet-style, same-source photos across classes),
used to build the shortcut-free Eczema-vs-other-disease comparison. Likely source (verify
this is the exact one before re-downloading — there are several similarly-named Kaggle
sets):
https://www.kaggle.com/datasets/haroonalam16/20-skin-diseases-dataset

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
distinguishes scratching from other hand motion (see `docs/paper_review_adam_sensor_2026-08-26.md`).
Not re-downloaded or referenced by any current script.
