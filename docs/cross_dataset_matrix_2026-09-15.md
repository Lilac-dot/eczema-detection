# Stage B Cross-Dataset Generalization Matrix — 2026-09-15

## Why

`docs/external_validation_2026-09-15.md` established that the deployed Stage B model
(`models/curated_resnet18_balanced.pt`, trained only on the internal DermNet-style
archive) does not generalize zero-shot to SCIN or SkinDisNet. That answers "does the
internal model transfer out," but not the more general question: is this an internal-model
problem specifically, or does *no* pairwise transfer work between any two of these three
sources? This doc fills in the other 6 cells of the 3x3 train-on-X/test-on-Y matrix,
plus two dataset-hygiene checks (duplicates, stock/non-skin contamination) on SCIN and
SkinDisNet that hadn't been run before using them as training sources (only as eval-only
external test sets so far).

## Part 1: Duplicate/overlap screening (`scripts/check_external_duplicates.py`)

Perceptual-hash screen (phash, Hamming distance <= 5), following the same method as
`scripts/check_eczema_overlap.py`, run on: within-SCIN, within-SkinDisNet, SCIN-vs-
SkinDisNet, SCIN-vs-Internal, SkinDisNet-vs-Internal.

**Cross-dataset overlap: none found.** Zero flagged pairs between SCIN and SkinDisNet,
SCIN and Internal, or SkinDisNet and Internal. No exact (md5) duplicates within either
external dataset.

**Within-SkinDisNet: 41 phash-flagged pairs, but most are false positives.** A
pixel-level MSE check (64x64 resize) on all 41 pairs found a bimodal distribution: 4
pairs with MSE < 20 (near-identical images) and the remaining 37 with MSE in the
180-1060 range (visually distinct images, e.g. an Eczema photo flagged against a
Scabies photo at MSE 384-411). The high false-positive rate is consistent with SkinDisNet
being close-up clinical macro-photography of skin texture — a genre with naturally low
phash diversity, unlike general photography. **Within-SCIN: zero pairs flagged at all**
(SCIN's more varied consumer-photography framing apparently doesn't trigger this).

The 4 genuine near-duplicates (confirmed by low pixel MSE, all same-class, likely
burst-mode photos of the same lesion) were checked against split membership:

| Pair | Split A | Split B |
|---|---|---|
| EC (453) / EC (454) | train | val |
| CD (371) / CD (372) | val | test |
| CD (79) / CD (82) | train | train |
| TC (30) / TC (33) | test | test |

Two of the four straddle a train/val or val/test boundary — real leakage risk for
Experiment 2/2b's original held-out test partitions. Fixed by dropping the
lower-priority-split copy of each pair (train dropped in favor of val; val dropped in
favor of test; for same-split pairs, one arbitrary member dropped to avoid redundant
weight on one lesion). Wrote `SkinDisNet/manifest_skindisnet_{train,val,test}_clean.csv`
(853/340/513, down from 855/341/514) — used for this doc's new training/eval only;
`docs/external_generalization_improvement_2026-09-15.md`'s Experiment 2/2b numbers
predate this cleaning and are not recomputed here (the leakage found is small — 2
images — and does not change that doc's conclusions, but it does mean their exact
numbers would shift by a negligible amount if rerun on the cleaned split).

## Part 2: Stock-photo / non-skin contamination screen (`scripts/check_external_skin_content.py`)

Reused the YCbCr skin-color-fraction heuristic that caught SkinDisease's contaminated
Normal class (median skin fraction ~0.29 there, vs. 0.73-0.99 for real disease classes,
with 12-16.6% of images near-zero skin content).

| Dataset | Class | n | Mean skin fraction | Median | % below 0.02 | % below 0.05 |
|---|---|---|---|---|---|---|
| SCIN | Eczema | 488 | 0.685 | 0.728 | 0.6 | 0.6 |
| SCIN | Other | 488 | 0.705 | 0.735 | 0.2 | 0.4 |
| SkinDisNet | Eczema | 536 | 0.721 | 0.781 | 0.2 | 0.2 |
| SkinDisNet | Other | 1,174 | 0.756 | 0.810 | 0.0 | 0.1 |

**No contamination found in either dataset.** Both classes in both datasets have
median skin fraction in the 0.73-0.81 range (comparable to SkinDisease's real disease
classes, not its contaminated Normal class), and under 1% of images in any class are
near-zero skin content. This is a clean result, reported as such rather than searching
harder for a problem that isn't there — consistent with `docs/external_validation_
2026-09-15.md`'s prior description of both as apparently-real clinical/consumer skin
photography.

## Explicitly skipped: brightness/color-statistic matching to the internal dataset

Not performed, by design. Adjusting SCIN's or SkinDisNet's color statistics to match
the internal dataset's would compromise their value as *independent* test sets — the
entire point of testing on them is that they were captured through a different pipeline;
matching that away would launder the exact distributional difference this experiment
exists to measure. This is also a fix type already tested and found ineffective for a
related problem: `docs/brightness_normalization_experiment_2026-08-26.md` found that
normalizing brightness on the *original* Eczema/Normal confound didn't fix it and just
shifted the shortcut to a different image statistic. Experiment 3 (`train_curated_cnn_
graypreproc.py`, gray-world color constancy applied uniformly to all sources at both
train and eval time — not internal-to-external matching) is the appropriate version of
this idea and is covered in Part 4 below.

## Part 3: The cross-dataset matrix

New models trained (`scripts/train_scin_resnet18.py`, `scripts/train_skindisnet_
resnet18.py`): same architecture and recipe family as the deployed model
(ImageNet-pretrained ResNet18, layer4+fc unfrozen, discriminative LRs 1e-4/1e-5, 15
epochs, plain cross-entropy, no class weighting) — kept identical on purpose so matrix
cells differ only in training data, not modeling choices. `models/scin_resnet18.pt`
trained on SCIN-train (488, balanced 244/244); `models/skindisnet_resnet18.pt` trained
on the cleaned SkinDisNet-train (853, imbalanced 268/585 — no weighting applied, a real
caveat for that cell, see below).

**Important sample-size caveat**: SCIN-train (488) and SkinDisNet-train (853) are 4-5x
smaller than internal-train (2,327). The SCIN-trained and SkinDisNet-trained models are
lower-confidence estimates than the internal-trained model's row — read their cells as
noisier, not as equally reliable data points in the matrix.

| Trained on \ Tested on | Internal | SCIN | SkinDisNet |
|---|---|---|---|
| **Internal** | AUC 0.8644 (n=507) [1] | AUC 0.5347 (n=976) [1] | AUC 0.4865 (n=1,710) [1] |
| **SCIN** | AUC 0.5395 (n=507) | AUC 0.6370 (n=294) | AUC 0.5316 (n=513) |
| **SkinDisNet** | AUC 0.4414 (n=507) | AUC 0.4761 (n=294) | AUC 0.8281 (n=513) |

[1] Internal-trained row reuses `docs/external_validation_2026-09-15.md`'s original
full-external-set numbers (SCIN n=976, SkinDisNet n=1,710) rather than the smaller
held-out slices used for the other two rows, since the internal model never trained on
any SCIN/SkinDisNet data and so has no leakage constraint requiring a held-out subset —
using the full external set there is the more statistically stable choice, at the cost
of the row's SCIN/SkinDisNet columns not sharing an identical n with the other two rows.
Full per-cell precision/recall/F1/95% CI in `cross_dataset_matrix_results.json`.

### Reading the matrix

**Every off-diagonal cell is at or below chance (0.44-0.64 range includes SCIN's own
diagonal, but every true off-diagonal cell sits in 0.44-0.54)** — this is not just "the
internal model doesn't generalize," it's "no pairwise combination of these three
sources generalizes to either of the other two, in either direction." That's a stronger
and more general finding than Section 1 alone supported.

**Each dataset is far more learnable in-domain than any cross-domain transfer
achieves**, but the three in-domain (diagonal) numbers are not equal:
- Internal: 0.8644 — highest, also the largest training set (2,327).
- SkinDisNet: 0.8281 — close to Internal's, on barely a third of the training data
  (853 images). This is a notably strong in-domain result for a compact clinical
  dataset, and reinforces (independent of the multisource fine-tuning result in
  `external_generalization_improvement_2026-09-15.md`) that SkinDisNet's Eczema-vs-other
  distinction is a learnable, well-behaved task on its own terms.
- SCIN: 0.6370 — clearly above chance, but far below the other two diagonals despite
  SCIN-train being roughly balanced (244/244) and the same recipe. SCIN's photos are
  self-taken by ordinary people (framing, lighting, distance all vary far more than
  clinical photography), which plausibly makes the in-domain task itself noisier, not
  just the cross-domain transfer. This is consistent with SCIN being the harder,
  higher-variance source in every result this project has produced about it so far
  (its multisource fine-tuning response was also the weakest — see
  `external_generalization_improvement_2026-09-15.md`).

**A directional asymmetry worth flagging**: SkinDisNet-trained -> Internal (0.4414) is
the single worst cell in the matrix, below-chance more than any other off-diagonal cell,
including the reverse direction Internal -> SkinDisNet (0.4865). SkinDisNet's "Other"
classes (Contact Dermatitis, Scabies, Seborrheic Dermatitis, Tinea Corporis) only
partially overlap in kind with Internal's 7 curated look-alike diseases (Psoriasis,
Tinea, Candidiasis, Infestations_Bites, Lichen, DrugEruption, Rosacea) — a
SkinDisNet-trained model's confusion pattern on Internal spreads fairly evenly across
Tinea, DrugEruption, Psoriasis, Infestations_Bites, Lichen, and Rosacea (see the
confusion detail in the raw eval log), suggesting it hasn't learned a look-alike
distinction transferable to Internal's specific comparison set, not that it's failing
in one particular confusable direction.

## Part 4: Additional generalization-improvement attempts (requested follow-up)

Two follow-ups flagged as open in `docs/external_generalization_improvement_2026-09-15.md`
were attempted after the matrix work above: Experiment 2c (more raw SCIN training volume)
and Experiment 3 (gray-world color constancy, applied uniformly to all sources — not
matching external stats to internal stats, which Part 2 above explains was rejected).

**`scripts/expand_scin_train.py` ran successfully**: SCIN's `weighted_skin_condition_label`
field was used to find 47 additional cases where Eczema is present with weight >= 0.40
but isn't the single top-weighted condition (a real secondary diagnosis, not a token
mention), plus 47 matched negatives — expanding SCIN-train from 488 (244/244) to 582
(291/291). `manifest_scin_val.csv`/`manifest_scin_test.csv` were untouched (read-only,
to build the exclusion set), so any result would have been directly comparable to
Experiment 2/2b's held-out numbers.

**Both training runs (Experiment 2c and Experiment 3) could not be completed in this
session.** `train_curated_cnn_multisource_expanded.py` was OOM-killed twice by the
system, and `train_curated_cnn_graypreproc.py` (a smaller, single-dataset job) was also
OOM-killed once immediately after. A memory check between attempts found only
~1.0-1.2GB free out of this machine's 8GB total — a system-wide memory shortage, not
something either training script did differently from the successful runs earlier in
this session (the SCIN-only and SkinDisNet-only ResNet18 trainings, and the entire prior
Experiment 1/2/2b work, completed fine on this same machine). This matches this
project's existing precedent (`docs/aauwss_sleep_model_results_2026-09-14.md`: the
AAUWSS sleep CNN was OOM-killed twice and abandoned rather than forced through a third
time) — following that same convention, these two runs are recorded here as abandoned
attempts, not as a third and fourth negative result. Both scripts are unmodified and
ready to run in a session with more free memory; `manifest_scin_train_expanded.csv` is
already built and waiting.

**Bottom line on Part 4**: no result to report yet either way — this is an
infrastructure limitation of the current session, not evidence about whether more SCIN
volume or color-constancy preprocessing would help or hurt. Re-run when the machine has
more headroom (close other applications, or retry after a restart) rather than treating
the OOM as a finding.

## Interpretation (overall)

Combined with `external_validation_2026-09-15.md` and `external_generalization_
improvement_2026-09-15.md`, the picture is now: (1) no dataset's model transfers
zero-shot to either other dataset, in any direction — a general finding about this
three-source setting, not an artifact of the internal model specifically; (2) the
datasets differ substantially in how learnable they are in-domain (SCIN's in-domain AUC
of 0.637 is itself far below Internal's or SkinDisNet's ~0.83-0.86, suggesting SCIN's
consumer-photography variability is a harder problem even before cross-dataset transfer
enters the picture); (3) both datasets pass duplicate and content-contamination
screening cleanly (aside from a small, now-fixed leakage risk of 2 images in
SkinDisNet), so the matrix results reflect genuine distributional differences between
sources, not a data-quality artifact; (4) the SkinDisNet-favoring, SCIN-resistant
pattern already seen in multisource fine-tuning repeats here in a completely independent
experiment (single-source training instead of fine-tuning), which is corroborating
evidence, not a coincidence of one experimental design.

## Part 5: Leave-one-dataset-out (LODO) — one result, two infrastructure failures

Follow-up requested after the matrix above: train on the union of two sources, test
zero-shot on the third held-out-entirely source, with a `WeightedRandomSampler` giving
the two combined training sources equal weight (same principle as
`train_curated_cnn_multisource_weighted.py`). `scripts/train_lodo_resnet18.py` (one
script, three configs), `scripts/mem_guard.py` (shared free-RAM check, factored out of
`train_curated_cnn_multisource_weighted_v2.py`'s ad hoc version — see
`docs/finetuning_v2_expanded_2026-09-16.md`).

This machine's memory situation was significantly worse during this part of the session
than earlier: free RAM swung between roughly 440MB and 1.6GB throughout, and the
harness's own memory-pressure watchdog killed three separate background processes
outright (not this project's own graceful mem_guard check — an external kill, before
any of this script's own log lines even printed), including, notably, a **plain
evaluation run with no training at all**.

**Config 1 — Internal-train + SCIN-train -> SkinDisNet test: completed successfully.**
15 epochs, no memory abort (`logs_lodo_internal_scin.txt`), best checkpoint at epoch 10
(combined internal+SCIN validation accuracy 0.6445; val_internal=0.7581, val_scin=0.5309
at that checkpoint). Saved to `models/lodo_internal_scin_resnet18.pt`. Final-epoch
train_acc reached 0.9009, well above the validation numbers -- some overfitting by the
end, consistent with fine-tuning any ImageNet-init model on ~2,800 images for 15 epochs,
though the checkpoint used is the epoch-10 one, not the final epoch.

**Config 2 — Internal-train + SkinDisNet-train -> SCIN test: did not complete, after two
attempts.** Both the original run and one retry were killed by the system before
printing anything (empty log both times) -- an external, harness-level kill triggered by
overall system memory pressure, not this script's own 400MB safety floor being crossed
(which would have printed a message first). Following this project's established
convention (the AAUWSS sleep CNN, and Part 4 above), abandoned after the second attempt
rather than forcing a third. No result, positive or negative -- an infrastructure gap,
not a finding about whether Internal+SkinDisNet transfers to SCIN.

**Config 3 — SCIN-train + SkinDisNet-train -> Internal test (the config that never
touches Internal during training at all): also did not complete, after two attempts,**
same failure mode as config 2 (both empty logs, external kill). Also abandoned. This is
the more scientifically interesting of the two failed configs -- whether the eczema task
is learnable purely from external data, with zero internal-archive exposure, remains
completely open.

**Evaluation of the one successful model could not be completed either, after two
attempts.** `scripts/eval_lodo_matrix.py` (which also carries this doc's planned
`paired_bootstrap_test` comparisons -- see below) was killed by the same external
mechanism twice, despite doing only forward-pass inference on a few hundred images per
manifest -- a workload that should need a small fraction of the memory any of the
training runs above did. Both attempts produced an empty log (killed before the first
print). **This is the strongest evidence in this document that the remaining problem is
not really about this project's training scripts being memory-heavy** (already reduced
to batch_size=8, no worker processes, explicit safety gates) **but about something else
on this machine consuming memory unpredictably enough that even a lightweight inference
job can get caught by it.** Reducing batch size, disabling dataloader workers, and
adding this project's own graceful abort logic have all already been tried and do not
address a kill that originates outside the Python process entirely.

**Net result of Part 5**: one real, honest LODO number exists (Internal+SCIN's
validation performance, above) but **no AUC/F1 test-set numbers, no combined matrix
table entry, and no significance tests could be produced this session** -- not because
the method failed, but because the evaluation step itself could not run. The
`paired_bootstrap_test()` function described below is implemented and ready; it has
simply never been executed against real data yet.

## Part 6: Paired significance testing (implemented, not yet run on this session's data)

`scripts/eval_external_common.py` gained two additions, both backward-compatible
(existing callers pass no new arguments and see no behavior change):

1. `run_eval(..., return_raw=True)` -- an opt-in flag that adds `y_true`/`y_prob` (as
   plain lists) to the returned dict, alongside everything already returned. Safe to
   pair across two separate calls on the *same* `manifest_path`, since `run_eval`'s
   `DataLoader` always uses `shuffle=False`, so row order matches the manifest file's own
   row order every time.
2. `paired_bootstrap_test(y_true, y_prob_a, y_prob_b, n=1000, seed=42)` -- resamples the
   *same* image indices for both models on every bootstrap draw (not independent
   resamples per model, which is what makes this the statistically correct comparison
   for two models scored on identical test images -- see e.g. Efron & Tibshirani 1993).
   Returns the observed AUC difference, its 95% CI, and a two-sided bootstrap p-value.

`scripts/eval_lodo_matrix.py` is written to apply this to every scientifically meaningful
same-test-set comparison this project now has models for (LODO vs. the corresponding
single-source baseline, on the same held-out test images) -- but as Part 5 describes,
it has not yet been able to finish running. **No p-values or significance verdicts are
reported anywhere in this document** -- none currently exist. This is flagged explicitly
so this capability isn't mistaken for something already validated against real numbers;
it is code that has been written and is ready, not a result.

## Recommendation

Re-run `scripts/train_lodo_resnet18.py internal_skindisnet`, `scripts/
train_lodo_resnet18.py scin_skindisnet`, and `scripts/eval_lodo_matrix.py` (in that order,
one at a time) once this machine has a period of genuinely stable free memory -- the
pattern observed here (healthy-looking free-RAM readings immediately before a launch,
followed by an external kill moments later) suggests checking memory right before
launching is not sufficient on its own; something on this machine is causing short,
unpredictable memory spikes that a point-in-time reading can miss.

## Files

New: `scripts/check_external_duplicates.py`, `scripts/check_external_skin_content.py`,
`scripts/train_scin_resnet18.py`, `scripts/train_skindisnet_resnet18.py`,
`scripts/eval_cross_dataset_matrix.py`, `scripts/mem_guard.py`,
`scripts/train_lodo_resnet18.py`, `scripts/eval_lodo_matrix.py`,
`docs/cross_dataset_matrix_2026-09-15.md` (this file, extended with Parts 5-6). New data:
`SkinDisNet/manifest_skindisnet_{train,val,test}_clean.csv`,
`dataset/SCIN/manifest_scin_train_expanded.csv` (built, training on it not yet
completed), `docs/external_duplicate_screen_2026-09-16.csv`, `docs/external_skin_content_
check_2026-09-16.csv`. New models: `models/scin_resnet18.pt`,
`models/skindisnet_resnet18.pt`, `models/lodo_internal_scin_resnet18.pt`. NOT created
(training never completed): `models/lodo_internal_skindisnet_resnet18.pt`,
`models/lodo_scin_skindisnet_resnet18.pt`. Modified: `scripts/eval_external_common.py`
(added `return_raw` parameter to `run_eval`, default `False`, and a new
`paired_bootstrap_test` function -- both additive, no existing behavior changed). Raw
logs (repo root, not committed): `logs_scin_resnet18.txt`, `logs_skindisnet_resnet18.txt`,
`logs_cross_dataset_matrix.txt`, `logs_expand_scin.txt`, `logs_exp2c_expanded.txt` (empty
-- both attempts OOM-killed before first epoch printed), `logs_exp3_graypreproc.txt`
(empty -- OOM-killed before first epoch printed), `logs_lodo_internal_scin.txt`
(complete, 15 epochs), `logs_lodo_internal_skindisnet.txt` (empty x2), `logs_lodo_scin_
skindisnet.txt` (empty x2), `logs_eval_lodo_matrix.txt` (empty x2). Result dump:
`cross_dataset_matrix_results.json` (repo root, not committed; `lodo_matrix_results.json`
was never created since the evaluation script never completed). Nothing committed --
left for review, matching this project's existing convention. Not touched: the paper,
Stage A/C code, `models/curated_resnet18_balanced.pt`, `models/curated_lightgbm_v3.txt`,
or any other pre-existing model/manifest file.
