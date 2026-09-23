# WESAD Personal-Baseline Calibration: Paired Significance Testing — 2026-09-16

## Why

`docs/wesad_personal_baseline_calibration_2026-09-12.md` reported a real improvement from
personal-baseline calibration (mean AUC 0.871 -> 0.9405) but only as a descriptive
before/after comparison of two separately-run LOSO-CV sweeps — no formal test of whether
the improvement is statistically distinguishable from fold-to-fold noise, and no per-subject
paired data was kept from the original "before" run to make that test possible after the
fact (`wesad_loso_fold_results.csv` gets overwritten by each retrain). Per this project's own
methodological standard (Section 7.10 of the report) and the explicit instruction to use
paired tests when comparing models on the same subjects, this was an open gap.

## Method

**Reconstructing "before" without re-deriving anything new**: `wesad_wrist_features.csv`
already contains both the absolute per-window features (`EDA_mean`, `TEMP_slope`, ...) and
their personal-baseline-relative `_rel` counterparts, computed together in one pass by
`build_wesad_features.py`. "Before calibration" is exactly the LightGBM LOSO-CV pipeline
in `train_wesad_stress.py`, restricted to the absolute columns only (`_rel` columns
excluded from `feat_cols`) — the identical feature set the pre-fix model used. This is not
a new experiment, it reruns the already-documented ablation with results saved for the
first time. `scripts/train_wesad_stress_before_calibration.py` ->
`dataset/WESAD/wesad_loso_fold_results_before_calibration.csv`. The existing "after" file
(`dataset/WESAD/wesad_loso_fold_results.csv`) was not touched or rerun.

**Sanity check**: the reconstruction's mean AUC (0.8712) and per-subject low-F1 count (1/15,
S14 at F1=0.00) match `docs/wesad_personal_baseline_calibration_2026-09-12.md`'s originally
reported before-state numbers (0.871 mean AUC, S14 the sole F1=0.00 subject) exactly —
confirms this is a faithful reconstruction, not a new, differently-configured run.

**Statistical tests**, treating each of the 15 WESAD subjects as one independent paired
observation (before-AUC, after-AUC) per the project's stated unit-of-analysis rule:
- Wilcoxon signed-rank test (non-parametric, robust to the skewed/outlier-heavy
  distribution of per-subject differences seen below) and a paired t-test (parametric,
  reported alongside for comparison) on AUC and on F1 separately.
- Cohen's d_z (paired effect size: mean difference / std of differences).
- A bootstrap 95% CI on the mean after-minus-before difference, resampling the 15
  subjects with replacement (10,000 resamples, seed 42) — matches this project's existing
  bootstrap-CI convention (`eval_external_common.py`) but resamples subjects, not windows,
  per the stated WESAD unit-of-analysis rule.

## Results

### Per-subject AUC change (after − before)

| Subject | Before AUC | After AUC | Change |
|---|---|---|---|
| S2 | 0.9913 | 0.9565 | -0.0348 |
| S3 | 0.8174 | 0.8304 | +0.0130 |
| S4 | 1.0000 | 1.0000 | +0.0000 |
| S5 | 0.6720 | 0.9000 | **+0.2280** |
| S6 | 0.9750 | 0.8250 | -0.1500 |
| S7 | 1.0000 | 1.0000 | +0.0000 |
| S8 | 0.9280 | 0.9905 | +0.0625 |
| S9 | 0.9437 | 0.9896 | +0.0458 |
| S10 | 0.9848 | 1.0000 | +0.0152 |
| S11 | 0.9508 | 1.0000 | +0.0492 |
| S13 | 0.9680 | 0.9760 | +0.0080 |
| S14 | 0.4673 | 0.8509 | **+0.3836** |
| S15 | 0.8145 | 0.9709 | +0.1564 |
| S16 | 0.9880 | 1.0000 | +0.0120 |
| S17 | 0.5673 | 0.8182 | **+0.2509** |

### AUC: significant improvement, driven by a few large per-subject gains

| Test | Statistic | p-value |
|---|---|---|
| Wilcoxon signed-rank | W=14.0 | **0.0277** |
| Paired t-test | t=2.021 | 0.0629 |

Cohen's d_z = 0.522 (medium effect size). Bootstrap 95% CI on the mean AUC improvement:
**[0.0078, 0.1364]** — entirely above zero.

The Wilcoxon test (significant at α=0.05) is the more appropriate of the two here: the
per-subject differences are visibly non-normal — three large positive outliers (S5 +0.228,
S14 +0.384, S17 +0.251) alongside several near-zero or slightly negative changes (S2, S6) —
which inflates the variance the t-test relies on and costs it power. The bootstrap CI
(which makes no normality assumption) agrees with the Wilcoxon result: it excludes zero,
though its lower bound (0.008) is close to it, so this is a real but not overwhelming
effect by CI width, consistent with a fix that transforms a handful of specific subjects
dramatically rather than lifting everyone uniformly (see S14/S17/S5 above vs. S2/S3/S4/S7's
near-zero change).

### F1: no significant change

| Test | Statistic | p-value |
|---|---|---|
| Wilcoxon signed-rank | W=56.0 | 0.8469 |
| Paired t-test | t=0.145 | 0.8867 |

Cohen's d_z = 0.037 (negligible). Bootstrap 95% CI on the mean F1 change: [-0.147, 0.168] —
comfortably spans zero. This formally confirms what `wesad_personal_baseline_calibration_
2026-09-12.md` already described qualitatively ("F1 mean and std are both roughly flat"):
calibration's real, significant effect is on ranking quality (AUC) and on rescuing specific
threshold-transfer failures (S7, S13, S17 in the original doc's accuracy table), not on
mean F1 across all 15 subjects.

## Interpretation

**The calibration fix produces a statistically real improvement in ranking quality (AUC),
not merely a fold-variance artifact** — Wilcoxon p=0.028 and a bootstrap CI excluding zero
both support this, using subjects (not windows) as the unit of analysis as required. The
improvement is concentrated in a minority of subjects (S5, S14, S17) who were previously
served worst, exactly the "catastrophic threshold-transfer failure" pattern the original
fix targeted — this is consistent with, and now statistically substantiates, the original
qualitative claim rather than contradicting it.

**F1 does not show a significant change**, formally confirming that calibration's value is
in ranking/threshold-transfer stability, not raw classification accuracy at a fixed cutoff
— an important distinction for how this result should be described (a ranking/reliability
fix, not an accuracy fix).

**Caveat**: n=15 is a small sample for any significance test; the Wilcoxon/t-test
disagreement (p=0.028 vs. p=0.063) itself illustrates this — a few subjects behaving very
differently can swing significance either way. This doesn't invalidate the finding (the
bootstrap CI, which doesn't depend on a parametric assumption, agrees with the Wilcoxon
result), but any future report of this result should carry both p-values rather than only
the more favorable one.

## Files

New: `scripts/train_wesad_stress_before_calibration.py`,
`dataset/WESAD/wesad_loso_fold_results_before_calibration.csv`, this file. Not modified:
`dataset/WESAD/wesad_loso_fold_results.csv` (the existing "after" file), any deployed
model, `train_wesad_stress.py`, the paper, or Stage B/C code.
