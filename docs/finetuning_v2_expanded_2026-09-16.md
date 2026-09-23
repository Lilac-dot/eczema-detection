# Experiment 2c: Equal-Weighted Fine-Tune with Expanded SCIN + Cleaned SkinDisNet — 2026-09-16

## Why

`docs/external_generalization_improvement_2026-09-15.md` found domain-equal-weighted
multi-source fine-tuning (`train_curated_cnn_multisource_weighted.py`, "Experiment 2b")
is the strongest external-generalization intervention tried so far: starting from
`models/curated_resnet18_balanced.pt`, fine-tuning on internal + SCIN + SkinDisNet with
each source sampled equally regardless of size gave SkinDisNet AUC 0.4680->0.6515 and
SCIN AUC 0.5535->0.5794 (SCIN's *ranking* barely moved despite a real F1 gain -- flagged
honestly in that doc as unresolved). `docs/cross_dataset_matrix_2026-09-15.md` (Part 4)
identified two follow-ups worth trying -- more raw SCIN volume, and gray-world color
constancy -- but both were abandoned that session after repeated OOM kills. This
experiment combines the equal-weighting method that already works with the two new
inputs made available since then (an expanded SCIN training manifest and a
leakage-fixed SkinDisNet training manifest), as a single next attempt rather than
re-running the two OOM'd experiments separately.

## Method

`scripts/train_curated_cnn_multisource_weighted_v2.py` -- identical to
`train_curated_cnn_multisource_weighted.py` (same starting weights, same discriminative
LRs of FC 3e-5 / backbone 3e-6, same 10 epochs, same equal-weighted `WeightedRandomSampler`,
same checkpoint rule: save whenever the mean of internal/SCIN/SkinDisNet validation
accuracy improves) except for three changes:

1. SCIN-train is the **expanded** manifest (`dataset/SCIN/manifest_scin_train_expanded.csv`,
   582 images, up from 488) built by `expand_scin_train.py` in the prior session (47
   additional real secondary-Eczema cases from SCIN's own weighted-label field, plus 47
   matched negatives).
2. SkinDisNet-train is the **cleaned** manifest (`SkinDisNet/manifest_skindisnet_train_clean.csv`,
   853 images, down from 855) with the 2 confirmed near-duplicate images that leaked
   across the original train/val and val/test split removed (`docs/cross_dataset_matrix_
   2026-09-15.md`, Part 1).
3. `batch_size` reduced from 32 to 8, and a dependency-free free-RAM check (`ctypes`
   call to the Windows `GlobalMemoryStatusEx` API, no `psutil` needed) logs free memory
   before training starts and gates execution: if free RAM is below 400MB before
   starting, or drops below 400MB before any subsequent epoch, the run stops on its own
   terms rather than waiting for the OS to kill it uncleanly. This is a real methodological
   difference from Experiment 2b worth flagging on its own: a smaller batch size changes
   the effective gradient noise per step and is not guaranteed to reproduce 2b's numbers
   exactly even if memory had not been a problem.

Evaluation was planned on the same held-out test partitions as 2b for direct
comparability (Internal test 507, SCIN test 294 unchanged, SkinDisNet test using the
cleaned 513-image version), via `eval_external_common.py`'s existing bootstrap-CI
pattern -- but evaluation was never reached; see Result below.

## Result: blocked before training started, not an OOM kill

This machine's free RAM was highly volatile through this session -- 2,982MB free right
after a restart, then 466MB, then 878MB within the space of about a minute of checking,
with no training running yet. By the time the script actually launched and reached its
own startup check, free RAM had fallen to **74MB**, and a recheck moments later (still
with nothing from this experiment running) showed **81MB**. Both are far below the 400MB
safety floor built into the script.

The script's own gate caught this and exited cleanly with a printed explanation, before
touching the model or any dataset -- this is the safety behavior it was designed to have,
and it worked as intended. This is meaningfully different from the prior session's OOM
kills (`docs/cross_dataset_matrix_2026-09-15.md`, Part 4), which were the operating
system forcibly terminating an already-running training process. Here, nothing was ever
allocated, and no partial/corrupt checkpoint exists.

A second attempt was not made. Free RAM was trending flat-to-worse (74MB -> 81MB) across
the two checks, with no indication of recovery, and the earlier volatility (memory
appearing at 878MB-2,982MB minutes before dropping under 100MB) points to something
external to this experiment -- another process on this machine, unrelated to Python or
this project -- consuming memory unpredictably. Retrying immediately under the same
condition would not be a meaningfully different attempt; per this project's existing
precedent for repeated OOM/resource failures (the AAUWSS sleep CNN, and Part 4 of the
cross-dataset matrix doc), this is recorded as an abandoned attempt due to an
infrastructure limitation, not as a negative result about the method.

## What this doesn't establish

**Nothing about whether expanded SCIN volume or SkinDisNet's leakage cleanup would have
changed Experiment 2b's numbers** -- no training happened, so there is no result to
compare, positive or negative. The open question from `external_generalization_
improvement_2026-09-15.md` (does more real SCIN data move SCIN's AUC, or does its domain
gap have a ceiling regardless of volume) remains exactly as open as it was before this
attempt.

## Recommendation

`scripts/train_curated_cnn_multisource_weighted_v2.py` is unmodified and ready to run
as-is. Re-attempt it when this machine has sustained free memory well above 400MB (the
volatility observed here suggests checking free memory right before launching, not
relying on a reading taken minutes earlier). No script or data change is needed first.

## Files

New: `scripts/train_curated_cnn_multisource_weighted_v2.py`, this file. Raw log (repo
root, not committed): `logs_exp2c_v2_finetune.txt` (3 lines -- confirms the immediate,
clean abort). Not created: `models/curated_resnet18_multisource_weighted_v2.pt` (no
checkpoint was ever saved, since no epoch completed). Not touched: the paper, Stage A/C
code, `models/curated_resnet18_balanced.pt`, `models/curated_resnet18_multisource_weighted.pt`,
or any existing manifest.
