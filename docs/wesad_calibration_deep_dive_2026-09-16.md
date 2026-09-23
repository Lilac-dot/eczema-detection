# WESAD Personal-Baseline Calibration: Publication-Ready Deep Dive — 2026-09-16

This extends `docs/wesad_personal_baseline_calibration_2026-09-12.md` and
`docs/wesad_calibration_significance_2026-09-16.md` with subject-level analysis, a
baseline-mismatch hypothesis test, a data-leakage audit and fix, cross-dataset
feasibility check, and a modality ablation. Organized in the order requested.

## 1. Reproduction

Re-derived both LOSO-CV sweeps independently: population-baseline (absolute features
only, `train_wesad_stress_before_calibration.py`) mean AUC = **0.8712**, mean F1 =
**0.6232**; personalized-baseline (full feature set, `train_wesad_stress.py`) mean AUC =
**0.9405**, mean F1 = **0.6352**. Both match the previously reported values exactly (the
0.871/0.9405 headline figures). Wilcoxon signed-rank (p=0.0277) and paired t-test
(p=0.0629) were re-verified as paired-by-subject (each test consumes 15 (before, after)
pairs indexed by subject, not independent samples) — confirmed correct by inspecting the
test inputs directly, not assumed. **No discrepancy found anywhere.**

## 2. Complete subject-level table

| Subject | Pop. AUC | Pers. AUC | ΔAUC | Pop. F1 | Pers. F1 | ΔF1 | Category |
|---|---|---|---|---|---|---|---|
| S14 | 0.4673 | 0.8509 | **+0.3836** | 0.000 | 0.308 | +0.308 | improved |
| S17 | 0.5673 | 0.8182 | +0.2509 | 0.468 | 0.727 | +0.259 | improved |
| S5 | 0.6720 | 0.9000 | +0.2280 | 0.625 | 0.000 | -0.625 | improved (AUC) |
| S15 | 0.8145 | 0.9709 | +0.1564 | 0.563 | 0.308 | -0.255 | improved (AUC) |
| S8 | 0.9280 | 0.9905 | +0.0625 | 0.606 | 0.917 | +0.311 | improved |
| S11 | 0.9508 | 1.0000 | +0.0492 | 0.815 | 0.429 | -0.386 | improved (AUC) |
| S9 | 0.9437 | 0.9896 | +0.0458 | 0.462 | 0.824 | +0.362 | improved |
| S10 | 0.9848 | 1.0000 | +0.0152 | 0.900 | 0.880 | -0.020 | negligible |
| S3 | 0.8174 | 0.8304 | +0.0130 | 0.556 | 0.667 | +0.111 | negligible |
| S16 | 0.9880 | 1.0000 | +0.0120 | 0.818 | 0.909 | +0.091 | negligible |
| S13 | 0.9680 | 0.9760 | +0.0080 | 0.444 | 0.571 | +0.127 | negligible |
| S4 | 1.0000 | 1.0000 | +0.0000 | 1.000 | 0.900 | -0.100 | negligible (ceiling) |
| S7 | 1.0000 | 1.0000 | +0.0000 | 0.444 | 0.952 | +0.508 | negligible AUC (ceiling) |
| S2 | 0.9913 | 0.9565 | -0.0348 | 0.824 | 0.667 | -0.157 | worsened |
| S6 | 0.9750 | 0.8250 | **-0.1500** | 0.824 | 0.471 | -0.353 | worsened |

Category (|ΔAUC| < 0.02 = negligible, matching the smallest fold-to-fold noise visible
elsewhere in this table): **7 improved, 6 negligible, 2 worsened**. Largest positive
change: S14 (+0.3836). Largest negative change: S6 (-0.1500). Two subjects (S4, S7) were
already at the AUC ceiling (1.0) before calibration — they cannot show an AUC gain
regardless of whether personalization "works" for them, a ceiling effect distinct from
"calibration doesn't help this subject." Full table:
`dataset/WESAD/wesad_subject_level_table.csv`.

Note the AUC and F1 columns tell different stories per subject (e.g. S5's AUC improved
+0.228 while its F1 collapsed to 0.000) — this is explained mechanistically in Section 9.

## 3. Visualizations

- `docs/wesad_calibration_paired_auc_2026-09-16.png` — paired AUC per subject, population
  vs. personalized (connected-dot plot, green=improved, red=worsened, gray=negligible).
- `docs/wesad_calibration_delta_auc_2026-09-16.png` — sorted ΔAUC bar chart, all 15
  subjects, labeled values.
- `docs/wesad_calibration_delta_f1_2026-09-16.png` — same for ΔF1, deliberately included
  despite the null result: the near-random scatter of positive/negative bars around zero
  **is** the F1 finding (no systematic shift), not an absence of one.
- `docs/wesad_baseline_mismatch_scatter_2026-09-16.png` — Section 4's scatter, labeled by
  subject.

All figures show the full per-subject spread, not just the mean — the effect is
concentrated in a minority of subjects (S5, S14, S17 for gains; S6, S2 for losses), which
a bar-chart-of-means would hide entirely.

## 4. Baseline-mismatch hypothesis

**Hypothesis tested**: subjects whose calm-state EDA differs more from the population's
calm EDA benefit more from personalized calibration.

**Measure**: for each subject, `eda_mismatch_z` = (that subject's mean calm-condition
`EDA_mean` − the population's mean calm `EDA_mean` across all 15 subjects) / population
calm-EDA standard deviation.

**Result: the hypothesis is NOT supported, and the (non-significant) trend points the
opposite direction.**

Spearman rho = **-0.375**, p = **0.168** (not significant at α=0.05), bootstrap 95% CI on
rho (10,000 resamples over subjects) = **[-0.766, 0.199]** — spans zero comfortably. If
anything, subjects with a *larger* absolute EDA mismatch show slightly *smaller*
improvement, the reverse of the hypothesis, though the confidence interval is far too wide
to treat this as a real effect either. Concretely: S13 and S7 have the two largest EDA
mismatches in the whole cohort (z=2.16 and 1.94) yet show almost no AUC change (+0.008 and
+0.000), directly contradicting the hypothesis for those two subjects specifically.

**This is correlational only and is reported as a null/non-significant result, not
reframed as a positive finding.** Raw calm-EDA-level mismatch is not what explains which
subjects benefit.

**An exploratory follow-up motivated by Section 5's findings** (not a pre-specified test —
flagged as exploratory per the multiple-comparisons note in Section 8): baseline
*variability* (coefficient of variation of a subject's own calm EDA, std/mean) correlates
more suggestively, though still short of significance: rho = -0.500, p = 0.082 (n=13,
excluding the two ceiling subjects S4/S7 who cannot show a gain), bootstrap 95% CI
[-0.766, 0.164]. This says subjects with a *noisier, less stable* calm baseline tend to
benefit *less* (or even worsen) — worth investigating further with more subjects, but not
claimed as established here.

## 5. Why S5, S14, S17 benefit — and why S2, S6 don't

Compared the three largest gainers against three non-improving subjects (S2, S6 — both
worsened; S4 — ceiling, can't improve). Full detail: `dataset/WESAD/wesad_focus_subject_
detail.csv`.

| Subject | ΔAUC | Calm EDA (mean±std) | Stress EDA mean | Calm EDA CV |
|---|---|---|---|---|
| S14 | +0.384 | 0.285 ± 0.014 | 0.336 (1.2x) | **0.050** |
| S17 | +0.251 | 1.360 ± 0.246 | 1.048 (0.8x, inverted) | 0.181 |
| S5 | +0.228 | 0.817 ± 0.115 | 3.489 (4.3x) | 0.140 |
| S2 (worsened) | -0.035 | 0.249 ± 0.220 | 0.867 (3.5x) | 0.883 |
| S6 (worsened, largest loss) | -0.150 | 0.684 ± 0.601 | 8.791 (12.9x) | **0.879** |
| S4 (ceiling) | 0.000 | 0.133 ± 0.023 | 2.411 (18.2x) | 0.170 |

**Pattern found, stated with appropriate caution**: the two worsened subjects (S2, S6)
both have a *strikingly noisy calm baseline* — coefficient of variation 0.88-0.88, meaning
their "calm" EDA standard deviation is nearly as large as its mean. S14, the single
largest gainer, has the opposite: the most *stable* calm baseline in the entire cohort
(CV=0.050). This is consistent with a plausible mechanism: calibrating against a noisy,
unreliable personal baseline can inject noise into the `_rel` features rather than remove
it, while calibrating against a stable one cleanly isolates the person-specific offset.

**This is not fully explanatory, and that is stated plainly rather than papered over**:
S17 (second-largest gainer) has a mid-range CV (0.181), not a notably stable baseline, and
its stress-condition EDA is actually *lower* than its calm EDA (an inverted response) —
not explained by the CV hypothesis at all. And S4 (no room to improve, ceiling effect) has
a similarly stable baseline (CV=0.170) to S17 without showing any gain, because it had
nowhere to go. **No single measured feature cleanly separates all three focus subjects
from all comparison subjects.** The calm-baseline-stability pattern fits 2 of 3 focus
subjects and both worsened subjects, which is suggestive but not a complete, data-proven
explanation — reported as a plausible partial mechanism, not a settled one.

## 6. Data leakage check — confirmed, fixed, and quantified

**Verified directly from `build_wesad_features.py`'s `add_personal_baseline_features()`**:
for every subject, the baseline profile subtracted from that subject's windows is the mean
of *all* of that subject's condition==1 (baseline) windows. Since `label = (condition==2)`,
condition==1 (baseline) windows are themselves part of the evaluated "not-stress" set for
that subject — **every baseline window's own value contributes to the very profile it is
then compared against.** This is genuine self-referential leakage, present identically for
every subject (train and test alike), diluted by group-averaging over that subject's ~18-19
baseline windows but not zero. **Stress and amusement windows are not self-referentially
leaked** — their `_rel` features come from a profile built entirely from *other* (baseline)
windows, which is the intended calibration design, not leakage.

**A) Original/current evaluation**: as above — each test subject's baseline profile
includes the exact baseline windows later evaluated. This is what produced the reported
0.9405 mean AUC.

**B) Leakage-free evaluation** (`train_wesad_stress_leakage_free.py`, new): for the TEST
subject only, in each LOSO fold, baseline-condition windows are split in half; the first
half is used *only* to build that subject's baseline profile, the second half (plus all
their amusement/stress windows) forms the evaluated set. Training/validation subjects are
unaffected — using a non-evaluated subject's own full baseline for their own features is
not leakage for them.

**Result across 3 independent random calibration/evaluation splits** (seeds 42, 7, 123 —
see Section 7):

| | Mean AUC | Mean F1 |
|---|---|---|
| Population baseline (no personalization) | 0.8712 | 0.6232 |
| **Personalized, original (leakage-present)** | 0.9405 | 0.6352 |
| **Personalized, leakage-free (avg. of 3 splits)** | **0.9247** | **0.6603** |

**The personalized-calibration advantage survives the leakage-free correction — it
shrinks, but does not disappear.** The improvement over the population baseline goes from
+0.0693 AUC (leaky) to +0.0535 AUC (leakage-free), roughly a 23% reduction in the measured
effect size, not a reversal. This is an honest, quantified answer: the leakage is real and
should be corrected in any future reporting of this result, but it was not the primary
driver of the finding. **Both CSVs are kept separately** (`wesad_loso_fold_results.csv` =
A, `wesad_loso_fold_results_leakage_free_split{42,7,123}.csv` = B) and neither was
overwritten.

## 7. Subject-wise validation appropriateness

The existing LOSO-CV protocol (each of 15 subjects held out as test exactly once) is
already the correct choice for n=15 subjects — a single fixed train/test split would let
one unlucky subject assignment dominate the reported number, which is exactly the failure
mode LOSO-CV is designed to avoid, and matches this project's own established convention
(Section 4.2 of the main report) and the literature it cites (Chun et al. 2021). No
different protocol is warranted or was introduced.

The leakage-free redesign in Section 6 introduces a new source of randomness (which half
of a subject's baseline windows become "calibration" vs. "evaluation"), so it was run with
**3 different random split seeds** (42, 7, 123) rather than just one, to check the result
isn't an artifact of one particular split. Spread across the 3 splits: mean AUC 0.9214 to
0.9313 (range 0.0099), mean F1 0.6491 to 0.6701 (range 0.0210) — a small, stable spread,
giving confidence the leakage-free number in Section 6 is not a fluke of one split.

## 8. Statistical analysis

Wilcoxon signed-rank (p=0.0277) and paired t-test (p=0.0629) were kept, both reported
without suppressing the less favorable one (see `wesad_calibration_significance_2026-09-16.md`
for the original derivation). **Why they differ**: the paired ΔAUC distribution is
right-skewed with three large positive outliers (S5, S14, S17) against a cluster of
near-zero and small-negative changes — this violates the paired t-test's normality
assumption and inflates its variance estimate, costing it power; Wilcoxon, which uses only
the ranks of the differences, is not affected by that skew and is the more appropriate test
here.

**Outlier check** (1.5×IQR rule on the ΔAUC distribution, bounds computed as
[Q1-1.5×IQR, Q3+1.5×IQR] = **[-0.154, 0.268]**): **S14 (+0.384) is a formal outlier** by
this rule; no other subject is. This confirms quantitatively what Section 2's table shows
by eye — S14 is genuinely exceptional, not just the top of a smooth distribution, and the
Wilcoxon-vs-t-test divergence in Section 1 is substantially attributable to this one point.

**Effect size**: Cohen's d_z = 0.522 (medium, paired-samples convention) for AUC; 0.037
(negligible) for F1 — both previously reported, confirmed here again alongside the outlier
context above.

**Multiple-comparisons note**: this document runs several statistical tests beyond the two
primary, pre-specified ones (paired AUC and F1 tests). The AUC/F1 paired tests (Section 1)
are **confirmatory** — they test the specific claim the original 2026-09-12 doc made.
Everything else — the baseline-mismatch correlation (Section 4, both the primary EDA-level
version and the CV follow-up), the modality ablation (Section 11), and the oracle-threshold
ceiling check (Section 9) — is **exploratory/hypothesis-generating**, run once each without
a formal multiple-testing correction (e.g. Bonferroni), and should be read as suggestive
leads for future work with more subjects, not as independently-confirmed findings at the
same evidentiary weight as the primary result.

## 9. F1 vs. AUC

Confirmed directly from `train_wesad_stress.py`: the classification threshold
(`best_f1_threshold`) is **tuned per-fold on a held-out validation subject's predictions,
then transferred to the test subject** — not a fixed global cutoff, and not tuned on the
test subject itself. Precision/recall/confusion-matrix values are already in both fold CSVs
(`tp`, `tn`, `fp`, `fn`, `precision`, `recall` columns).

**Oracle-threshold ceiling analysis** (`dataset/WESAD/wesad_threshold_oracle_ceiling.csv`,
new — clearly a non-deployable upper bound, not a real method, since it tunes the threshold
directly on the test subject's own labels): if each test subject could use their own
best-F1 threshold instead of one transferred from a different validation subject, mean F1
would be **0.888** instead of the actual **0.635** — a large gap. This directly answers the
question: **calibration substantially improved ranking quality (AUC) but the
threshold-transfer problem the original 2026-09-12 fix targeted was reduced, not
eliminated** — a large amount of achievable F1 is still being lost specifically to
threshold miscalibration across people, exactly the mechanism the original document
described (e.g. S5: AUC 0.90, actual F1 0.000, oracle F1 0.800 — a pure
threshold-transfer failure, not a ranking failure). This is reported as a ceiling analysis
illustrating the mechanism, separate from and not a substitute for the primary result.

## 10. Cross-dataset generalization — not feasible with this project's current data

Checked directly rather than assumed: this project's only other wearable physiological
dataset is **AAUWSS** (`build_aauwss_features.py`). Its label is `condition==1` = **asleep**
vs. **Wake/disrupted** — a sleep-disruption label, not a stress label. There is no stress
protocol, no TSST-equivalent task, and no "stress" condition anywhere in AAUWSS's label
space. A WESAD-stress-trained model has no valid target to be evaluated against in AAUWSS —
this is a **label-semantics incompatibility**, not merely a distribution-shift question,
and forcing the comparison (e.g. treating AAUWSS's "Wake" as a stand-in for "stress") would
conflate two physiologically and behaviorally distinct states without justification.
SCIN/SkinDisNet are image datasets — a different modality (RGB photographs vs. wearable
sensor time-series) with no meaningful mapping to a stress-classification task at all.
**No valid cross-dataset stress-generalization experiment exists with data currently in
this project.** The closest scientifically valid alternative, not pursued here, would be a
second public EDA/skin-temperature stress-induction dataset with a comparable
lab-stressor protocol (e.g. a TSST or Trier-type paradigm) — none is currently in this
repository.

## 11. Ablation: EDA vs. temperature vs. combined

Reran the full personalized-calibration LOSO pipeline restricted to feature subsets
(`wesad_calibration_modality_ablation.py`, `dataset/WESAD/wesad_ablation_results.csv`):

| Feature subset | n features | Mean AUC | Mean F1 |
|---|---|---|---|
| EDA only (+ EDA_rel) | 12 | **0.9144** | 0.6157 |
| TEMP only (+ TEMP_rel) | 12 | 0.8015 | 0.6029 |
| EDA + TEMP (+ both _rel) | 24 | 0.9002 | 0.6551 |
| **Full feature set (EDA+TEMP+BVP+ACC, reference)** | 52 | **0.9405** | 0.6352 |

**The personalization benefit is substantially EDA-driven**: EDA alone (0.9144) recovers
most of the full-model's AUC (0.9405), while temperature alone is much weaker (0.8015) and
close to the *population*-baseline full-feature-set AUC (0.8712) — temperature's
personalization signal is weak on its own. **A finding reported honestly despite being
mildly counterintuitive**: EDA+TEMP combined (0.9002) is not simply additive and actually
scores *below* EDA alone (0.9144) — plausibly sampling noise from doubling the feature
count on this small dataset rather than a real interaction effect, but not explained away;
BVP and ACC (not tested individually here, out of the requested scope) evidently
contribute the remaining gap to reach the full model's 0.9405.

## 12. Overclaiming check

No causal language used for Sections 4 or 5 (both explicitly correlational, both say so in
their own text, not only here). Every null/negative result is reported at the same level of
detail as positive ones: F1's non-significant change (Sections 1, 3, 9), the
baseline-mismatch hypothesis's non-support (Section 4), the EDA+TEMP non-additive ablation
result (Section 11), and the cross-dataset infeasibility (Section 10). No result was
softened, hidden, or reframed to look more favorable than what was actually measured.

## Files

New: `scripts/wesad_subject_level_analysis.py`, `scripts/train_wesad_stress_leakage_free.py`,
`scripts/wesad_calibration_modality_ablation.py`, `scripts/plot_wesad_calibration_effect.py`.
New data: `dataset/WESAD/wesad_subject_level_table.csv`, `wesad_focus_subject_detail.csv`,
`wesad_loso_fold_results_leakage_free_split{42,7,123}.csv`, `wesad_ablation_results.csv`,
`wesad_threshold_oracle_ceiling.csv`. New figures: `docs/wesad_calibration_paired_auc_
2026-09-16.png`, `wesad_calibration_delta_auc_2026-09-16.png`,
`wesad_calibration_delta_f1_2026-09-16.png`, `wesad_baseline_mismatch_scatter_2026-09-16.png`.
This file. Not modified: `wesad_loso_fold_results.csv`, `wesad_loso_fold_results_before_
calibration.csv`, any deployed model, the paper, or Stage B/C code.
