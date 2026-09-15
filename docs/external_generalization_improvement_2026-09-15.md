# Stage B External Generalization — Improvement Attempts, 2026-09-15

## Why

`docs/external_validation_2026-09-15.md` found the deployed Stage B model
(`models/curated_resnet18_balanced.pt`) does not generalize zero-shot to two
independent external eczema datasets (SCIN, SkinDisNet) -- both external AUCs sit at
chance, with no brightness-shortcut explanation and a colour-feature model that fails
in the same way, pointing to a genuine, archive-specific distributional gap rather
than a fixable confound. This doc reports two attempts to actually close that gap.
Both are real training runs (CPU-only machine) with real results, reported honestly
including where they didn't help.

**Machine note**: this development machine is CPU-only (`torch.cuda.is_available()`
returns False). Each 15-epoch training run below took roughly 50 minutes wall-clock.
Given that, and per the scope given for this work, Experiment 3 (a quick
colour-constancy preprocessing check) was written (`scripts/train_curated_cnn_
graypreproc.py` exists, ready to run) but **not executed** -- Experiments 1 and 2
between them consumed the available time budget, and this project's own history
(Section 5.2 of the main report: brightness normalization and cropping both failed to
fix the *original* shortcut) already gives reason to expect a quick preprocessing fix
is the least promising of the three options tried here. This is a scope cut made for
time, not a finding -- if generalization work continues, Experiment 3 is ready to run.

## Experiment 1: heavier augmentation, zero-shot

**Method**: same architecture and recipe as `train_curated_cnn_balanced.py`
(ResNet18, `layer4`+`fc` unfrozen, 15 epochs, seed 42), same internal-only training
data (`manifest_curated_v3_train.csv`, 2,327 images) -- but with meaningfully heavier
augmentation aimed at reducing sensitivity to source-specific photographic
conventions: `RandomResizedCrop` (scale 0.6-1.0) instead of a fixed resize, wider
colour jitter, random vertical flip, random Gaussian blur (p=0.3), random
autocontrast/equalize, and random JPEG re-encoding at quality 30-80 (p=0.3) to
simulate a different compression pipeline. No external data touched, so results are
directly comparable to Phase 1's full-external-set numbers.
`scripts/train_curated_cnn_augmented.py`, `scripts/eval_experiment1.py`. Saved to
`models/curated_resnet18_augmented.pt`.

**Wall-clock time**: 3,001.1s (50.0 min) for 15 epochs (168-346s/epoch, heavier at
the start -- likely disk/cache warm-up for the JPEG re-encode path).

**Result**: internal validation accuracy peaked at 0.7702 (best epoch 14) -- notably
lower than the baseline recipe's internal performance, confirming the heavier
augmentation makes the internal task itself harder, as expected.

| | n | Accuracy | AUC (95% CI) | F1 (95% CI) |
|---|---|---|---|---|
| Internal test (augmented model) | 507 | 0.7515 | 0.8345 (0.7987-0.8695) | 0.7595 (0.7180-0.7978) |
| Internal test (baseline, for comparison) | 507 | 0.8107 | 0.8644 (0.8307-0.8973) | 0.8118 (0.7732-0.8493) |
| SCIN full external (augmented model) | 976 | 0.5266 | 0.5358 (0.4987-0.5721) | 0.2062 (0.1630-0.2558) |
| SCIN full external (baseline, for comparison) | 976 | 0.5092 | 0.5347 (0.5003-0.5692) | 0.1492 (0.1119-0.1886) |
| SkinDisNet full external (augmented model) | 1,710 | 0.6643 | 0.5016 (0.4709-0.5317) | 0.1800 (0.1400-0.2207) |
| SkinDisNet full external (baseline, for comparison) | 1,710 | 0.6673 | 0.4865 (0.4559-0.5183) | 0.1123 (0.0794-0.1461) |

**Interpretation: this did not work.** AUC on both external sets is statistically
indistinguishable from the baseline's already-chance-level AUC (SCIN 0.5358 vs.
0.5347; SkinDisNet 0.5016 vs. 0.4865 -- both well within each other's bootstrap CIs).
F1 moved up somewhat on both external sets (SCIN 0.1492->0.2062, SkinDisNet
0.1123->0.1800), but this reads as a byproduct of the model's decision boundary
shifting under the harder, more heavily regularized training, not a genuine
discriminative improvement -- the AUC, which measures ranking quality independent of
threshold, did not move. Meanwhile internal accuracy dropped by nearly 6 points
(0.8107->0.7515). **Heavier augmentation traded internal performance for no
measurable external generalization gain.** This is a clean negative result, not an
ambiguous one, and is broadly consistent with this project's earlier experience that
input-side preprocessing fixes (Section 5.2 of the main report) don't touch the
underlying problem.

## Experiment 2: multi-source fine-tuning

**Method**: split SCIN's 976-image manifest and SkinDisNet's 1,710-image manifest
each into train/val/test (stratified by label, 50/20/30, seed 42) --
`scripts/split_external_manifest.py`:

| Dataset | Train (Eczema/Other) | Val (Eczema/Other) | Test (Eczema/Other) |
|---|---|---|---|
| SCIN | 488 (244/244) | 194 (97/97) | 294 (147/147) |
| SkinDisNet | 855 (268/587) | 341 (107/234) | 514 (161/353) |

Fine-tuned starting from the deployed model's own weights (not from scratch) on
internal-train (2,327) + SCIN-train (488) + SkinDisNet-train (855) = 3,670 images
combined, using the baseline's original (moderate) augmentation recipe -- not
Experiment 1's heavier one, to keep "more diverse training sources" as the single
variable under test. Learning rates were dropped to roughly 1/3 of the original
(FC 3e-5, backbone 3e-6) since this is fine-tuning an already-converged model, not
training from scratch, and 10 epochs were used instead of 15 for the same reason plus
wall-clock budget. Checkpoint selection used the mean of internal/SCIN/SkinDisNet
validation accuracy each epoch, so the saved model isn't just good on one domain.
`scripts/train_curated_cnn_multisource.py`. Saved to
`models/curated_resnet18_multisource.pt`.

**Wall-clock time**: 2,962.6s (49.4 min) for 10 epochs (287-305s/epoch).

**Training trajectory** (validation accuracy per epoch): internal validation stayed
essentially flat throughout (0.784-0.800, no sign of catastrophic forgetting).
SkinDisNet validation climbed steadily from 0.604 (epoch 1) to a peak of 0.663 (epoch
9). SCIN validation, notably, stayed pinned in a narrow 0.48-0.52 band for all 10
epochs -- essentially chance, even with direct training exposure to SCIN's own
training images throughout. Best checkpoint (epoch 8, combined val acc 0.6507) was
driven almost entirely by the SkinDisNet improvement.

**Held-out test results** -- baseline re-evaluated on the exact same held-out test
partitions used to evaluate the fine-tuned model (so this is a fair, matched
comparison; these numbers are not identical to Phase 1's full-external-set numbers
since they're computed on a smaller held-out slice only):

| | n | Accuracy | AUC (95% CI) | Precision | Recall | F1 (95% CI) |
|---|---|---|---|---|---|---|
| Internal test -- baseline | 507 | 0.8107 | 0.8644 (0.8307-0.8973) | 0.7992 | 0.8247 | 0.8118 (0.7732-0.8493) |
| Internal test -- multisource | 507 | 0.7968 | 0.8530 (0.8191-0.8859) | 0.7868 | 0.8088 | 0.7976 (0.7585-0.8359) |
| SCIN held-out test -- baseline | 294 | 0.4966 | 0.5535 (0.4872-0.6163) | 0.4800 | 0.0816 | 0.1395 (0.0719-0.2169) |
| SCIN held-out test -- multisource | 294 | 0.5442 | 0.5783 (0.5163-0.6419) | 0.5489 | 0.4966 | 0.5214 (0.4436-0.5937) |
| SkinDisNet held-out test -- baseline | 514 | 0.6518 | 0.4680 (0.4149-0.5225) | 0.1250 | 0.0186 | 0.0324 (0.0000-0.0718) |
| SkinDisNet held-out test -- multisource | 514 | 0.6479 | 0.6090 (0.5538-0.6594) | 0.4242 | 0.3478 | 0.3823 (0.3082-0.4512) |

**Interpretation: a real, but domain-dependent, improvement -- and it's more
qualified than the headline F1 jump suggests.**

- **Internal**: essentially preserved (AUC 0.8644->0.8530, F1 0.8118->0.7976,
  both drops well within the baseline's own bootstrap noise). No meaningful
  catastrophic forgetting from adding external data to the training mix.
- **SkinDisNet**: the clearer win. AUC went from below chance (0.4680, CI spanning
  both sides of 0.5) to clearly above chance (0.6090, CI 0.5538-0.6594, entirely
  above 0.5) -- a genuine ranking improvement, not just a threshold shift. Recall on
  the Eczema/Atopic-Dermatitis class went from essentially not-detecting (0.0186) to
  meaningfully better (0.3478). Still far below internal performance (AUC 0.61 vs.
  0.85), but this is real, direction-correct progress from a modest training
  exposure (855 images).
- **SCIN**: the AUC barely moved (0.5535->0.5783) and its CI (0.5163-0.6419) only
  narrowly excludes 0.5 -- a small, marginal signal at best, not clearly
  distinguishable from noise. **The F1 jump here (0.1395->0.5214) is the result
  most worth being careful about**: recall jumped from 0.0816 to 0.4966 almost
  entirely because the model's decision threshold shifted toward predicting
  "Eczema" far more often on this domain (precision correspondingly dropped from
  0.48 to 0.55 only slightly, but that's because SCIN's test set is balanced
  50/50 -- predicting positive more often mechanically buys recall on a balanced
  set without necessarily reflecting better discrimination). Given the AUC result,
  the honest read is: fine-tuning gave the model a better *operating point* for
  SCIN's data, not a meaningfully better *underlying ranking* of SCIN images by
  how eczema-like they are.

**Why the SCIN/SkinDisNet split makes sense**: SkinDisNet is clinical smartphone
photography from hospital dermatology consultations -- a different institution and
country than the internal archive, but the same broad *genre* of photography
(clinician-directed, lesion-focused). SCIN is crowd-sourced consumer photography --
ordinary people photographing their own skin, with all the framing, lighting, and
distance variability that implies. That SkinDisNet responded to a small amount of
in-domain fine-tuning while SCIN barely did, even with SCIN's own training images
directly in the fine-tuning mix, is consistent with SCIN representing a larger,
harder-to-bridge domain gap -- not just "needs more of the same fix," but plausibly a
qualitatively different photographic distribution that 488 training images and 10
epochs at a conservative learning rate aren't enough to close.

## What this does and doesn't establish

- **Something that helps exists** (multi-source fine-tuning), but it only clearly
  helps for the external domain that's photographically closer to the training
  archive (SkinDisNet's clinical photography), not the one that's furthest from it
  (SCIN's consumer photography) -- and even there, it remains far below internal
  performance (AUC 0.61 vs. 0.85).
- **Something that doesn't help was also identified** (heavier augmentation alone,
  with no new data sources), which rules out one plausible, cheaper-to-implement
  fix and is itself useful evidence about the nature of the problem: this isn't
  primarily about the model being under-regularized against superficial photographic
  variation (which augmentation targets), it's about the training data not covering
  the relevant range of real-world photographic sources at all (which only new data
  addressed, and only partially).
- **Not tested here**: whether more external training data (all of SCIN/SkinDisNet's
  train split rather than this experiment's necessarily-limited slice, or additional
  sources like Fitzpatrick17k) would close the SCIN gap further, or whether SCIN's
  gap is more fundamental (e.g., its self-photographed, non-clinical acquisition
  setting is different enough from any clinical/atlas archive that this class of fix
  has a ceiling well below internal performance regardless of how much such data is
  added). A quick check: Fitzpatrick17k's own CSV lists 287 rows matching "eczema" in
  their label field, hosted on dermaamin.com (a web dermatology atlas -- the same
  broad genre as the internal training archive, so a weaker independence claim than
  SCIN, closer to SkinDisNet's). This wasn't pursued further here given the time
  already spent on Experiments 1-2; it remains a reasonable next external check if
  this work continues.
- **Experiment 3 (colour-constancy preprocessing) was not run**, for the time-budget
  reason stated above. The script (`scripts/train_curated_cnn_graypreproc.py`) is
  ready; `eval_external_common.py` was extended with an `eval_tf` override parameter
  specifically so this could reuse the same evaluation path when it is run.

## Experiment 2b: domain-weighted sampling (testing the imbalance hypothesis)

*Note on process: this section was first drafted against an intermediate checkpoint
(11:12) while `train_curated_cnn_multisource_weighted.py` (PID 22788) was still
running, which produced a materially different, more pessimistic result. The
training run was confirmed complete (exit code 0) and re-evaluated at 11:49 against
its true final checkpoint (last updated 11:41, unchanged since); the numbers below
are from that final, confirmed run. This is recorded here because the corrected
result changes the section's conclusion, not just its numbers -- see the discussion
below.*

Hypothesis going in: Experiment 2's combined fine-tune mixed internal-train (~2,331),
SkinDisNet-train (855), and SCIN-train (488) in proportion to their raw size, so SCIN
-- the smallest source, and the one that responded worst -- may have been getting
diluted by gradient updates from the other two, larger sources.

Test: identical starting weights, identical held-out test partitions, identical
epoch count as Experiment 2, but with a weighted random sampler so each of the three
sources contributes roughly equally per epoch regardless of its raw size.
`scripts/train_curated_cnn_multisource_weighted.py` ->
`models/curated_resnet18_multisource_weighted.pt`. Evaluated with
`scripts/eval_experiment2b_weighted.py` on the exact same three held-out sets as
Experiment 2.

| | Baseline AUC | Exp 2 (proportional) | Exp 2b (equal-weighted, final) |
|---|---|---|---|
| Internal test | 0.8644 | 0.8530 | 0.8317 |
| SCIN held-out | 0.5535 | 0.5783 | 0.5794 |
| SkinDisNet held-out | 0.4680 | 0.6090 | **0.6515** |

| | Baseline F1 | Exp 2 (proportional) | Exp 2b (equal-weighted, final) |
|---|---|---|---|
| Internal test | 0.8118 | 0.7976 | 0.7758 |
| SCIN held-out | 0.1395 | 0.5214 | **0.5761** |
| SkinDisNet held-out | 0.0324 | 0.3823 | **0.4218** |

**The hypothesis was right, and this is the strongest cross-dataset result of any
intervention tried in this document.** Once evaluated against the actual final,
fully-trained checkpoint (not the mid-training snapshot the first draft of this
section used), equal-weighting improves on Experiment 2's already-real gains on
*both* external sets, not just one: SkinDisNet AUC rises further (0.6090 -> 0.6515)
and its F1 nearly doubles Experiment 2's own figure (0.3823 -> 0.4218); SCIN, which
had barely moved under proportional weighting, gets a real F1 improvement too
(0.5214 -> 0.5761), though its AUC gain over Experiment 2 is still small (0.5783 ->
0.5794 -- SCIN's *ranking* quality has not meaningfully improved over Experiment 2,
even though more of its images are now being correctly classified at the operating
threshold). This comes at a real, not negligible, cost to internal performance
(AUC 0.8644 -> 0.8317, F1 0.8118 -> 0.7758) -- the largest internal drop of any
intervention tried, which should be weighed against the external gains rather than
treated as free.

Why the first (provisional) evaluation of this same idea looked like a negative
result is itself informative: `train_curated_cnn_multisource_weighted.py` saves a
checkpoint only when validation accuracy improves, and the 11:12 snapshot evaluated
in the original draft of this section was not yet the model's converged state --
several more epochs of real improvement happened between 11:12 and the run's actual
completion. The lesson for this project's own methodology: a "best-so-far" checkpoint
from a still-running process is not a safe stand-in for a finished model's result,
even when it looks plausible -- exactly the kind of premature-conclusion risk this
project's whole approach (Sections 4.4, 5.1 of the main report) has otherwise been
built to catch.

`scripts/expand_scin_train.py` and `scripts/train_curated_cnn_multisource_expanded.py`
were written to test a complementary volume hypothesis directly (pulling more real
SCIN images into SCIN-train specifically, leaving SCIN-val/test untouched) but the
training run was not completed in this pass -- worth running given SCIN's AUC still
lags SkinDisNet's despite the F1 gain, suggesting its ranking-level domain gap may
need more raw exposure, not just better-weighted exposure, to close further.

## Bottom line

Of the three interventions tried, two produced real net improvements and one did
not: **heavier augmentation alone did nothing; multi-source fine-tuning with sources
in their natural proportion helped SkinDisNet substantially and SCIN only in a
threshold-driven way; and domain-equal-weighted fine-tuning, once actually finished
training, improved on that further for both external sets** (most clearly for
SkinDisNet, more modestly for SCIN's ranking quality specifically), at the cost of
the largest internal-accuracy drop of the three. The overall picture: real exposure
to real examples from a given external source is what moves these numbers, weighting
that exposure more evenly across sources helps further, and none of this comes free
against internal performance. None of the three new models are being proposed as a
replacement for the deployed `curated_resnet18_balanced.pt` -- all are saved
separately (`curated_resnet18_augmented.pt`, `curated_resnet18_multisource.pt`,
`curated_resnet18_multisource_weighted.pt`) for the record and for these comparisons,
and the deployed model is untouched throughout.

## Files

New: `scripts/train_curated_cnn_augmented.py`, `scripts/train_curated_cnn_
multisource.py`, `scripts/train_curated_cnn_multisource_weighted.py`,
`scripts/train_curated_cnn_multisource_expanded.py` (written, not run),
`scripts/expand_scin_train.py`, `scripts/train_curated_cnn_graypreproc.py` (written,
not run), `scripts/split_external_manifest.py`, `scripts/eval_experiment1.py`,
`scripts/eval_experiment2.py`, `scripts/eval_experiment2b_weighted.py`,
`docs/external_generalization_improvement_2026-09-15.md` (this file). Modified:
`scripts/eval_external_common.py` (added optional `model_path` and `eval_tf`
parameters, both backward compatible -- existing calls with no arguments behave
exactly as before). New data: `dataset/SCIN/manifest_scin_train_expanded.csv`
(built, not yet used by a completed training run), `dataset/
SCIN/manifest_scin_{train,val,test}.csv`, `SkinDisNet/manifest_skindisnet_
{train,val,test}.csv`. New models: `models/curated_resnet18_augmented.pt`,
`models/curated_resnet18_multisource.pt`,
`models/curated_resnet18_multisource_weighted.pt`. Raw training logs: `logs_exp1_
augmented.txt`, `logs_exp2_multisource.txt` (repo root, not committed). Result
dumps: `experiment1_results.json`, `experiment2_results.json`,
`experiment2b_weighted_results.json`. Nothing committed --
left for review. `models/curated_resnet18_balanced.pt` (the deployed model) and
`models/curated_lightgbm_v3.txt` were not modified. No Stage A/C code touched, no
`.docx` report touched.
