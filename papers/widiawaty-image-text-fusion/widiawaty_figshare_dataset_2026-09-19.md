# Widiawaty et al. Figshare Dataset — Discovery, Download, and Patient-Wise Split (2026-09-19)

## What this is

A user-added `29925533.xml` (Figshare dataset metadata record) pointed to a public dataset
supporting Widiawaty et al. 2025, **"Multimodal Machine Learning Approach for Diagnosing
Atopic Dermatitis"** — a doctoral-program study out of Indonesia combining ResNet50 (image)
and MPNet (clinical text) via late fusion, reporting 98.28% internal validation accuracy.
Dataset DOI: `10.6084/m9.figshare.29925533.v4`.

This is the first dataset encountered in this project with **genuinely paired image + text
data per patient** — every prior fusion discussion in this project (Section 8.2 of
`honors-paper-report.docx`) has been decision-level specifically because no dataset pairs
wearable sensor data with images from the same patient. This one pairs something else
(structured clinical text) with images, from the same patient/visit — a real, buildable
alternative to the wearable-fusion dead end, not a fix for the wearable gap itself.

## What was downloaded

12 of 13 files from the Figshare record, to `dataset/widiawaty_figshare_29925533/`:

- `Clinical_data_AD_nonAD_phases1_2.xlsx` — the main data, 4 sheets (AD/Non-AD x Phase 1/2)
- `Table1_Demographics.xlsx`, `Table2_ResNet50_vs_ViT_metrics.xlsx` — paper's own summary tables
- `GoogleForm_AD_patients.html`, `GoogleForm_NonAD_patients.html` — the actual data-collection
  instruments (useful for seeing exact question wording behind each column)
- 5 paper figures (training curves, F1 score, algorithm flowchart, research-flow diagrams)
- `TranslatedInformedConsent.pdf` — the consent form template
- `STARD2015Checklist.docx` — reporting-standard checklist

**Deliberately NOT downloaded**: `SignedInformedConsent.pdf` (file id 57237212) — a
*signed* consent document could contain a real individual's actual signature, unlike the
blank translated template. Skipped on privacy grounds even though the dataset is publicly
deposited by the researchers themselves.

Lesion photos are **not** part of the Figshare deposit — each of the 2,811 rows in the main
spreadsheet links to an individual photo hosted on Google Drive
(`drive.google.com/file/d/<id>/view`). Verified one downloads successfully as a real
768x1024 JPEG with no login required (`https://drive.google.com/uc?export=download&id=<id>`
redirects to a working direct-download URL). **Images themselves have not been bulk-
downloaded yet** — this doc covers manifest-building only; see Next Steps.

## Dataset structure

4 sheets: AD Phase 1 (926 rows), Non-AD Phase 1 (697), AD Phase 2 (525), Non-AD Phase 2
(663) — 2,811 total. Each row: demographics, chief complaint, allergen/irritant contact
history, self-reported "Current Disease Trigger Factors" (free text), duration of illness,
lesion location, past/family medical history, a Google Drive photo link, and (AD sheets
only) Hanifin-Rajka major/minor diagnostic criteria strings, or (Non-AD sheets only) an
explicit `Diagnosis` column. Checked the Non-AD `Diagnosis` values directly rather than
trusting the sheet name alone: all four values present (Psoriasis vulgaris, Contact
dermatitis, Nummular dermatitis, Lichen simplex chronicus) are genuinely non-AD conditions,
no missing/ambiguous rows — confirms sheet-derived labeling (AD sheets=1, Non-AD sheets=0)
is safe to use directly.

## The patient-ID problem, and a bug caught before it was trusted

The spreadsheet has **no real patient ID column** — `Sample number` is per-photo, not
per-patient. Confirmed directly: hundreds of rows per sheet share an identical full clinical
profile, almost certainly the same patient photographed more than once, not coincidence.

**First attempt at a grouping key was wrong.** Used only 4 fields (Gender, Age, Past
Medical History, Family Medical History) reasoning that a conservative/coarse key is the
"safe" direction for a leakage-preventing split. This was checked before being trusted, and
the check failed it: 2,809 rows collapsed into only **406** groups. Inspecting why: these 4
fields have very low cardinality (2 genders; many patients share exact ages; Past/Family
Medical History are checkbox-style with a handful of common combinations — "NaN" alone
covers 336 of 926 rows in one sheet). The key was mostly merging **unrelated patients who
happen to share demographics**, not detecting real repeat photos — over-merging this
aggressively would have discarded most of the dataset's real diversity for no genuine
leakage protection, the opposite of what a patient-wise split is for.

**Fixed key**: every available clinical free-text field (chief complaint, allergen history,
source of infection, trigger factors, duration, lesion location, Hanifin-Rajka criteria
where present, past/family medical history) plus gender and age. Much higher cardinality —
an exact match across all of these is a specific same-patient signal, not a demographic-
bucket collision. Result: 2,809 rows -> **1,915 groups** (894 rows merged), matching a
direct exploratory check run before writing any code (767/926, 445/697, 442/525, 605/663
unique clinical-field combinations per sheet, about 80% unique overall) — consistent, not
just "a more plausible-looking number."

Still a heuristic, not a verified patient identifier — stated plainly, not treated as ground
truth. It is deliberately biased toward over-merging (a false merge wastes some diversity
but cannot cause leakage) rather than under-merging (which could actually split one real
patient's photos across train and test).

## Build process and final split

`scripts/build_widiawaty_manifest.py`:
1. Load all 4 sheets, tag each row with its sheet-derived label and phase.
2. Extract the Google Drive file ID from each photo link (regex on `/d/<id>/`).
3. Build the clinical-text field: chief complaint, allergen history, source of infection,
   trigger factors, lesion location, major/minor criteria, joined as `"field: value | ..."`.
4. Build the patient-group key described above; assign a `patient_group_id`.
5. Split **by group, not by row** — every row belonging to one patient group goes to the
   same split. Stratified by label so AD/non-AD balance holds across splits despite group
   sizes varying (1 to 6+ rows per group). Target 70/15/15, seed 42.
6. Hard assertion: no `patient_group_id` appears in more than one split. This ran and
   passed — not just intended, verified.

Dropped 2 of 2,811 rows for an unparseable photo link. Final manifest:
`dataset/widiawaty_figshare_29925533/manifest.csv`
(`row_id, phase, label, diagnosis, drive_file_id, clinical_text, patient_group_id, split`).

**Final split (rows)**:

| Split | Non-AD | AD | Total |
|---|---|---|---|
| Train | 951 | 1,017 | 1,968 |
| Val | 204 | 218 | 422 |
| Test | 203 | 216 | 419 |

Label balance holds close to 48/52 in every split. 1,915 patient groups total (713/625
train, 165/130 val, 149/133 test by non-AD/AD).

## Planned model comparison (not yet run)

Five candidates, same statistical discipline as the image-architecture comparison
(Section 5 of `honors-paper-report.docx`) — McNemar + paired bootstrap + Holm correction
across all pairs, one official test-set read once every candidate is trained:

1. Text-only classical baseline — TF-IDF + LightGBM on `clinical_text`.
2. Image-only baseline — ShuffleNetV2-1.0x (this project's already-selected architecture)
   fine-tuned fresh on this dataset's own images.
3. Decision-level late fusion — average/weighted combination of (1) and (2)'s independent
   probabilities, mirroring the existing `fuse()` philosophy. A lower bound: if joint
   training doesn't beat this, it bought nothing.
4. Jointly-trained fusion — ShuffleNetV2-1.0x image embedding + a lightweight sentence
   encoder (`all-MiniLM-L6-v2`, ~80MB, chosen over the original paper's MPNet for this
   CPU-only development machine) for text, concatenated, small MLP head, trained end-to-end.
   The actual point of using this dataset: the first model in this project that can be
   genuinely jointly trained on paired data, rather than decision-level fusion of
   independently-trained scores.
5. Gated fusion (optional/reach) — same two backbones, a small learned gate instead of
   plain concatenation, letting the model weight image vs. text per example. Only worth
   trying if (4) clearly beats (3).

## Next steps

1. Bulk-download all 2,809 manifest images from Google Drive into `train/`, `val/`,
   `test/` subfolders matching the manifest's own split column (not done yet — flagged as a
   real time/scale commitment before starting, not something to launch silently in the
   background).
2. Run model 1 (text-only) first — needs no images at all, fastest possible sanity check
   on whether the text modality carries real signal before committing to the image side.
3. Then models 2-5 in order, with the same test-set-access discipline already established
   elsewhere in this project (train everything before the one official test read, save
   results to a permanent script, not a one-off).
4. Decide, once results exist, whether/how this feeds back into `honors-paper-report.docx`
   — this is new work outside that report's current image/stress/moisture scope, not yet
   folded in.
