# Severity Scale Selection for Image-Based Eczema Severity Labeling — 2026-09-21

## Why this exists

Following the abandonment of the Mendeley "Multi-Class Skin Disease Image Dataset with
Severity Levels" (contaminated with non-eczema diagnoses, broken test split, no
documented labeling methodology — see chat record, not written up separately since the
dataset was deleted and never used), the severity approach pivots to labeling this
project's own already-vetted Stage B eczema images
(`SkinDisease/manifest_curated_v3_{train,val,test}.csv`, `disease_class == Eczema`,
1,665 images total: 1,165 train / 249 val / 251 test) directly, using visual inspection
against a documented clinical rubric, rather than trusting an external dataset's
unverified severity folders.

## The three standard AD severity scales, and which is actually assessable from a single photo

| Scale | What it measures | Can it be scored from one static lesion photo? |
|---|---|---|
| **SCORAD** | Extent (% BSA, rule of nines) + 6 intensity signs (erythema, edema/papulation, oozing/crusting, excoriation, lichenification, dryness) + **subjective** pruritus and sleep-loss (each patient-reported 0-10 VAS over recent days) | **No** — the subjective component (~40% of the score) requires the patient's own recent symptom report, not observable in an image at all. |
| **POEM** | 7 items on symptom *frequency over the past week* (itchy, sleep loss, bleeding, weeping/oozing, cracking, flaking, dry/rough skin), each 0-4 by number of days | **No** — POEM has no visual/objective component whatsoever; it is a patient diary questionnaire, not a clinical exam. Structurally inapplicable to an image. |
| **EASI** | 4 objective clinical signs (erythema, induration/papulation, excoriation, lichenification), each graded 0-3, assessed per body region, times an area-affected score (0-6) per region, times a fixed regional weight | **Yes, for the 4 signs** — these are purely visual/morphological judgments a trained-enough non-dermatologist can learn to apply from a photo. The area/region-weighting half needs a whole-body view, which single lesion-crop photos (this dataset's format) don't provide. |

**EASI is the only one of the three with a purely observational component** — SCORAD and
POEM are ruled out categorically, not just as "harder," because their subjective/
patient-report halves cannot be reconstructed from a photo under any labeling protocol.
This matches what the smartphone-photo AD-severity validation literature actually does in
practice (`PMC9907712`, `PMC12590331` — both validate photo-based EASI/SCORAD scoring by
having dermatologists grade signs from images, never the subjective components).

## Deviation from full EASI, stated explicitly

This project's images are single close-up lesion crops (`dataset/Eczema/*.jpg`), not
whole-body photographs, and are not reliably attributable to one of EASI's 4 canonical
body regions (head/neck, upper limb, trunk, lower limb) from the image alone. Full EASI's
area-score × regional-weight half is therefore **not computed**. What is labeled is the
**4-sign severity score only** — the directly visualizable half of EASI, on its
documented 0-3 per-sign scale, summed to a 0-12 "EASI-signs" score per image. This is a
deliberate, named simplification, not a full EASI score — reported as such in any future
write-up, not conflated with a validated clinical EASI number.

## The rubric used for labeling (sourced from the standard EASI clinical descriptors)

For each image, score these four signs, each 0 (absent) - 3 (severe):

- **Erythema (redness)**: 0 none/residual discoloration only; 1 light pink to light red;
  2 red; 3 deep/dark red or violaceous.
- **Induration / papulation (thickening, raised lesions)**: 0 none; 1 barely
  palpable/slight thickening or papules; 2 easily visible moderate thickening or
  papules; 3 marked/severe thickening or dense papule clusters.
- **Excoriation (scratch damage)**: 0 none; 1 occasional scratch marks or superficial
  breaks; 2 multiple linear excoriations; 3 extensive scratch damage with crusting or
  bleeding.
- **Lichenification (chronic leathery thickening, exaggerated skin lines)**: 0 none;
  1 subtle skin-line prominence; 2 visible thickening with pronounced creases; 3
  leather-like texture with exaggerated surface pattern.

`easi_signs_sum = erythema + induration_papulation + excoriation + lichenification`
(range 0-12).

**Bands revised after labeling, based on the actual observed distribution, not the
theoretical range.** All 1,659 labeled images (after excluding 6 histology micrographs,
see below) were scored by 5 parallel agents against the rubric above
(`scripts/build_severity_labeling_composites.py` tiled images into 3x3 grids for
efficient vision labeling; raw per-image scores in
`dataset/severity_labels/easi_signs_labels_raw.csv`). The observed `easi_signs_sum`
distribution tops out at **9** (nothing reached 10-12) and is concentrated in the
low-to-mid range — a fixed even 4-way split of the theoretical 0-12 range put only 10/1659
images in a "9-12" band, an unusable class size. Bands were therefore set from the
**empirical tertiles** of the actual score distribution instead, giving a 3-class target
that is well-balanced overall and per-split (~30-37% each class in train/val/test):

| Band | easi_signs_sum | Count (of 1,659) |
|---|---|---|
| Mild | 0-2 | 554 |
| Moderate | 3-4 | 607 |
| Severe | 5-9 | 498 |

Final labels: `dataset/severity_labels/easi_severity_labels_final.csv` (columns: `path`,
`split`, the 4 sign scores, `easi_signs_sum`, `severity_band`). `split` reuses the
existing Stage B `SkinDisease/manifest_curated_v3_{train,val,test}` partition — the same
images, same split membership, so no new leakage-checking was needed (that partition was
already built and vetted for the disease-classification task). The `test` split's labels
were assigned by the same blind visual process as train/val (labeling ground truth is not
"peeking" at model performance) but remains untouched for any training/model-selection
decision, per this project's standing test-set discipline.

## Excluded: 6 histology micrographs found mixed into the curated Eczema class

`dataset/Eczema/eczema-histology-{1..6}.jpg` are stained tissue cross-section
micrographs, not clinical photos — confirmed by direct visual inspection. These aren't
gradable against a visual severity rubric (no erythema/lichenification etc. is
observable in a microscopy slide) and were excluded from the severity label set entirely
(1,665 candidate images -> 1,659 labeled). **This is a pre-existing contamination issue
in the Stage B curated_v3 disease-classifier training data**, not something introduced by
this labeling pass — noted here since it wasn't caught by the original
`dataset_usability_check_2026-08-26.md` audit, but out of scope to fix as part of this
severity work.

## Explicit limitation

This is a **non-dermatologist visual proxy**, not a clinically validated severity score
— every image gets a single annotator's judgment (this project's own labeling pass), not
the 3-dermatologist consensus a real EASI study would use. Treat any model trained on
these labels as learning "this project's own visual severity heuristic," and report it
that way — the same honesty standard already applied throughout this project's other
scope caveats (see `README.md`'s Stage A-stress/A-sleep caveats for the precedent).
