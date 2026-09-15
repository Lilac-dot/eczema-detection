# Stage B External Validation — 2026-09-15

## Why

Every image used to train, validate, and test the deployed Stage B model traces back to
the same underlying dermatology-atlas archive (Section 5.5 of the report). The original
Eczema-vs-Normal classifier was already found to be exploiting a photographic-source
shortcut (Section 5.1) rather than genuine lesion features; the rebuilt Eczema-vs-7-diseases
task fixed the *known* confound, but a subtler same-archive artifact — consistent
compression, framing convention, or watermark presence correlating with class — could not
be ruled out without testing on data from a genuinely different source. This is Experiment 2
from the external-dataset research done earlier: take the deployed model exactly as it is,
with no retraining or fine-tuning, and test it on two independently-sourced eczema datasets.

## Datasets

**SCIN** (Ward et al., *JAMA Network Open* 2024;7(11):e2446615) — crowd-sourced consumer
smartphone photos from US Google Search users, publicly downloadable with no login from a
public GCS bucket (`https://storage.googleapis.com/dx-scin-public-data/dataset/`). This is
the strongest independence claim of any candidate found: a completely different collection
mechanism (self-photographed by ordinary people, not a clinical/atlas archive) from a
different country.

`scin_labels.csv`'s `weighted_skin_condition_label` field gives each case a dict of
`{condition_name: weight}`, already mapping clinical synonyms (e.g. "Nummular eczema",
"Eczematous dermatitis") onto one "Eczema" label. Of 5,033 total cases, 3,061 have a
weighted label at all; of those, **488 have "Eczema" as their single highest-weighted
condition**. A balanced external set was built by taking all 488 Eczema cases plus a random
sample of 488 non-Eczema cases from the remaining pool (seed 42) — 976 images total. Only
`image_1_path` was used per case (SCIN allows up to 3 images per case via `case_id`); using
one image per case avoids introducing a same-case leakage risk that the internal dataset's
own lack of patient IDs (Section 5.5) can't rule out for itself.
`scripts/build_scin_manifest.py`.

**SkinDisNet** (Sultana et al., *Data in Brief* 2025;63:112239, DOI
10.1016/j.dib.2025.112239; Mendeley DOI 10.17632/yj3md44hxg) — clinical smartphone photos
from two Bangladesh hospitals, April 2023–June 2024. Genuinely independent hospital-sourced
data, though a weaker independence claim than SCIN since it's still clinic-acquired
photography (same broad genre as the internal archive, just a different institution/country).
1,710 real images across 6 classes (only the "Preprocessed" folder was used — the
"Augmented" folder is synthetic augmentation of the same 1,710 photos and would just add
near-duplicates, not new evidence). The dataset treats "Eczema" (466 images) and "Atopic
Dermatitis" (70 images) as separate classes with no clinical justification given anywhere in
the source paper, so they were merged into one `label=1` positive class (536 total);
"Contact Dermatitis" (477), "Scabies" (343), "Seborrheic Dermatitis" (79), and "Tinea
Corporis" (275) became `label=0` (1,174 total). `scripts/build_skindisnet_manifest.py`.

Both manifests use the same `path,disease_class,label` schema as the internal dataset.

## Method

`scripts/eval_external_common.py` loads `models/curated_resnet18_balanced.pt` unmodified
(no retraining, no fine-tuning) and evaluates it on each external manifest with the same
224x224 / ImageNet-normalization preprocessing used at training time. In addition to the
accuracy/precision/recall/F1 the original `eval_curated_cnn_balanced.py` reports, this adds
AUC (from the softmax probability, not just the argmax) and a bootstrap 95% CI on AUC and F1
(1,000 resamples of the test images, 2.5th/97.5th percentiles) — the same kind of interval
reporting the report already does for Stage A (Section 4.6), needed here because a few
hundred external images is a small enough sample that a bare point estimate would overstate
precision. `scripts/eval_external_internal_baseline.py` reruns the same code path on the
model's own internal test split, so internal and external numbers are computed identically
and are directly comparable (the original eval script never computed AUC or a CI).

`scripts/diagnose_external_shortcut.py` reuses the exact diagnostic that originally caught
the brightness shortcut (Section 5.1): mean whole-image brightness by class, and — since a
suspiciously close CNN/colour-feature-LightGBM agreement was the first red flag that time —
runs the existing `curated_lightgbm_v3` colour-feature model (also trained only on internal
data, also not retrained here) on each external manifest for comparison.

## Results

| | n (Eczema / Other) | Accuracy | AUC (95% CI) | Precision | Recall | F1 (95% CI) |
|---|---|---|---|---|---|---|
| Internal test set (baseline) | 507 (251/256) | 0.8107 | 0.8644 (0.8307–0.8973) | 0.7992 | 0.8247 | 0.8118 (0.7732–0.8493) |
| SCIN (external) | 976 (488/488) | 0.5092 | 0.5347 (0.5003–0.5692) | 0.5600 | 0.0861 | 0.1492 (0.1119–0.1886) |
| SkinDisNet (external) | 1,710 (536/1,174) | 0.6673 | 0.4865 (0.4559–0.5183) | 0.3429 | 0.0672 | 0.1123 (0.0794–0.1461) |

The internal-baseline row reproduces the report's own 81.07%/81.18% figures exactly (Section
5.5), confirming the eval pipeline is behaving as expected before trusting its external
numbers.

**Confusion detail, SCIN**: TN=455, FP=33, FN=446, TP=42. The model predicts "Eczema" for
only 75 of 976 images (7.7%) — it is not recalling most real Eczema cases (recall 0.0861),
and the handful of "Other" cases it does call Eczema aren't concentrated in any one
condition (the script reports them all under a single "Other" bucket rather than SCIN's
finer condition labels, since the manifest only carries the binary label — a limitation of
this specific run, not of the data).

**Confusion detail, SkinDisNet**: TN=1,105, FP=69, FN=500, TP=36. Recall on the merged
Eczema/Atopic-Dermatitis class is 0.0672 (433 Eczema + 67 Atopic Dermatitis images missed).
False positives cluster on Contact Dermatitis (29), Scabies (26), and Tinea Corporis (14) —
Contact Dermatitis and Tinea Corporis are at least the same general "similar-looking
inflammatory/infectious skin condition" category as the internal task's look-alike classes,
so this isn't a random confusion pattern, but the absolute numbers are small next to 500
missed true positives.

## Shortcut/domain-gap diagnostic

| | Mean brightness, Eczema | Mean brightness, Other | Gap | Colour-feature LightGBM accuracy / AUC / F1 |
|---|---|---|---|---|
| Internal test set | 0.3972 | 0.4534 | 0.0562 | 0.7041 / 0.7829 / 0.7036 |
| SCIN | 0.4923 | 0.4908 | 0.0015 | 0.5195 / 0.5185 / 0.2399 |
| SkinDisNet | 0.5314 | 0.5529 | 0.0215 | 0.6193 / 0.5437 / 0.2661 |

(Brightness is computed as grayscale mean on a 0–1 scale, same 128x128 resize
`extract_color_features.py` uses; the original dataset's known shortcut-confound gap, for
comparison, was 0.15 vs 0.58 = a 0.43 gap. The internal test set's own gap, checked here for
the first time on this exact split, is 0.056 — an order of magnitude smaller, consistent
with the report's claim that the rebuild fixed the known confound.)

Neither external dataset has a brightness gap anywhere close to the original confound's
scale — both are close to zero, well inside the range of normal photographic variation. This
rules out the specific failure mode that broke the *original* Eczema-vs-Normal classifier:
the CNN's poor external performance is not explained by a simple brightness-style shortcut
being absent in the new data. The colour-feature LightGBM — the simpler, historically more
shortcut-prone model — degrades to the same near-chance region as the CNN on both external
sets (AUC 0.52 and 0.54, against 0.78 internally), rather than continuing to perform well
while the CNN fails or vice versa. If the CNN's internal accuracy were resting on some
subtler same-archive artifact invisible to a plain brightness check, a plausible
alternative signature would be the two architecturally different models disagreeing sharply
on external data (the same kind of divergence that was actually reassuring in Section 5.4's
frozen-CNN-vs-LightGBM comparison); instead they move together, both toward chance.

## Interpretation

**This is a real, honestly-reported negative result for cross-dataset generalization**, not
an ambiguous one. Both AUCs are at or barely above chance (SCIN 0.535, 95% CI upper bound
0.569; SkinDisNet 0.487, 95% CI spanning 0.5 on both sides) — a large, clearly-outside-CI
drop from the internal 0.864 baseline. Recall on the positive class collapses to 6–9% on
both external sets: the model is not simply miscalibrated (a threshold shift could fix a
model that ranks correctly but decides wrong), it has little to no discriminative ranking
signal left once the photographic source changes.

The diagnostic evidence points to genuine distributional shift rather than a residual,
findable shortcut: no brightness confound exists in either external set to explain the
result, and the simpler colour-feature model fails in the same way as the CNN rather than
in a different way that would suggest a CNN-specific artifact. The most likely explanation,
consistent with both external datasets' own documented acquisition methods, is that the
model has learned features tied to this specific archive's clinical macro-photography
conventions — close-up framing, consistent lighting/focus, and (per Section 5.5) whatever
compression/imaging pipeline is common across DermNet-style sources — that do not transfer
to either consumer smartphone photography (SCIN) or a different clinical archive's own
photographic conventions (SkinDisNet). This is a *narrower, more defensible* conclusion than
"the model learned nothing real": it performs exactly as designed on same-source data
(Section 5.5), and the original, most obviously exploitable confound (Section 5.1) is
confirmed absent here. What's now supported by direct evidence is that the fix substantially
reduced but did not eliminate the model's dependence on source-specific visual conventions —
the remaining gap is a genuine same-archive artifact of the kind Section 5.5 already flagged
as unruled-out, now quantified rather than merely hypothesized.

This changes the evidence status of the claim in the report's Table 6 ("Image model
distinguishes eczema from selected look-alike diseases" — "Partially supported... no
external, independently sourced test set has been checked"): a check has now been run, and
the result is that the model does **not** generalize to either external source tested. The
report's own claim structure already anticipated and correctly scoped for this possibility;
this finding fills in the previously-open question rather than contradicting anything
currently asserted.

## Status / what this doesn't cover

- No retraining was attempted. Whether fine-tuning on a small amount of external data (e.g.
  a held-out slice of SCIN) recovers performance, or whether the internal and external tasks
  are fundamentally too different in composition (SkinDisNet's "Other" classes only
  partially overlap with the internal task's 7 look-alike diseases) to expect transfer at
  all, are both open questions this experiment doesn't answer.
- SCIN's negative-class sample (a random draw from all non-Eczema top-condition cases) is
  not restricted to eczema look-alikes the way the internal task's 7-disease comparison is —
  it includes conditions as visually distinct as Herpes Zoster and Acne. This makes the SCIN
  task easier in principle (less visually confusable), which makes the near-chance result
  more notable, not less.
- Files created: `scripts/build_scin_manifest.py`, `scripts/build_skindisnet_manifest.py`,
  `scripts/eval_external_common.py`, `scripts/eval_external_scin.py`,
  `scripts/eval_external_skindisnet.py`, `scripts/eval_external_internal_baseline.py`,
  `scripts/diagnose_external_shortcut.py`. `scripts/paths.py` gained `SCIN_DIR` and
  `SKINDISNET_DIR` constants. Raw data: `dataset/SCIN/` (976 images + 2 metadata CSVs) and
  `SkinDisNet/Preprocessed/` (1,710 images, extracted from the full Mendeley zip — the
  `Augmented/` folder and the full zip were left in place but are not used by anything).
  Neither `models/curated_resnet18_balanced.pt` nor `models/curated_lightgbm_v3.txt` was
  modified.
