# Stage A-sleep: Results — Negative Finding — 2026-09-14

## Bottom line

Stage A-sleep (wrist-signal sleep-disruption detection on AAUWSS, same pipeline as
Stage A-stress) was built, trained, and evaluated. It does not work: three independent
methods — personal-baseline-calibrated LightGBM, and the two literature-standard,
decades-validated actigraphy formulas (Cole-Kripke 1992, Sadeh 1994) — all score at
essentially chance under proper subject-level evaluation. A fourth method (a raw-signal
CNN+attention, mirroring Stage A-stress's second architecture) was attempted but never
completed a full run, for hardware reasons documented below, not because it produced a
bad result. **Stage A-sleep is kept in the codebase as a documented negative result and
is NOT wired into the deployed `trigger_index()`/`flare_risk()` computation** — the same
treatment this project has given every other negative result (Stage A-stress's CNN v2
fine-tuning attempt, Stage B's cropped/normalized experiments).

## Method 1: LightGBM (`train_aauwss_sleep.py`), the primary attempt

Same architecture as the working Stage A-stress model: window feature extraction
(`wesad_features.py`, reused unmodified since AAUWSS's Empatica E4 export uses identical
sampling rates to WESAD's), personal-baseline calibration (each subject's own
asleep-state average as the reference, the sleep-domain analogue of WESAD's calm-baseline
reference — see `build_aauwss_raw_windows.py`), and leave-one-subject-out cross-validation
across the 13 subjects.

**Result: mean AUC 0.4642 (std 0.092), mean F1 0.1071, pooled AUC 0.3867 (worse than
chance).** All 13 subjects have F1 < 0.3. This is not a subtle instability like Stage
A-stress's original threshold-transfer problem — it is a model that has not learned a
generalizable signal at all.

## Diagnosing whether this is a bug before accepting it as a finding

A near-chance (and in the pooled case, *below*-chance) result is unusual enough that it
warranted checking for an implementation bug before reporting it as real:

- **Feature-set ablation**: raw (non-baseline-relative) features alone gave LOSO AUC
  0.475; baseline-relative features alone gave 0.531; the combined default gave 0.464.
  All three are near chance, which rules out "the personal-baseline transform specifically
  broke something" as the primary explanation — the signal isn't reliably there in either
  feature family under proper subject-held-out evaluation.
- **Between- vs. within-subject signal**: raw pooled EDA differs substantially between
  Wake (mean 3.71 µS) and Asleep (mean 1.76 µS) windows, which looks like a strong signal
  until per-subject means are examined: individual subjects' average EDA ranges from 0.13
  to 10.5 µS — nearly two orders of magnitude — and the subject-level correlation between
  a person's mean EDA and their Wake fraction is only 0.22. The pooled difference is
  substantially a between-subject artifact (some people just run high-EDA), not a
  within-subject Wake-vs-Asleep signal a LOSO-evaluated model can exploit for a new person
  — exactly the kind of confound personal-baseline calibration is supposed to remove, and
  consistent with why removing it (the "REL-ONLY" ablation above) didn't help either.
- **Manual epoch-level inspection**: for one subject (subject_02), the raw label sequence
  and movement (`ACC_mag_std`) were inspected directly. The label structure itself looks
  legitimate — long, physiologically plausible contiguous blocks (e.g., ~29 minutes of
  Wake at the start of the recording before sleep onset), not scrambled or misaligned —
  so this rules out a gross alignment bug. Movement differences between the states are
  real but small and noisy at the individual-epoch level.
- **Quiet pre-sleep wakefulness**: across all 13 subjects, 32.7% of all Wake epochs
  (range 14.3%–78.9% by subject) fall within each subject's initial contiguous Wake block
  — i.e., lying in bed before sleep onset, not moving much. This looks like "asleep" to a
  movement/EDA-based signal and is a genuine, literature-connectable difficulty (distinct
  from the *after*-sleep-onset arousals that actigraphy is more established at detecting),
  but at ~33% of the positive class it does not fully explain a near-chance result on its
  own.

**Conclusion: no implementation bug found; this is treated as a genuine result.** The
underlying task — Wake-vs-Asleep at 30-second resolution, from wrist EDA/temperature/
BVP/movement alone, evaluated on genuinely unseen subjects, including a substantial
quiet-pre-sleep-wake component — is harder than either this feature set or this sample
size (13 subjects, some with as few as 21 total Wake windows) can reliably solve.

## Method 2 & 3: Cole-Kripke (1992) and Sadeh (1994) — literature-standard, non-ML baselines

Motivated by the LightGBM result, and to check whether the difficulty was specific to
this project's modeling choices rather than the data itself, the two most established
published actigraphy sleep/wake scoring formulas were applied directly
(`aauwss_actigraphy_baseline.py`). Both are fixed, non-trained formulas (weighted sums of
a movement "activity count" over a window of surrounding epochs) — there is nothing to
fit, so results are reported directly per subject and pooled, not via LOSO-CV.

**Adaptation caveats** (both algorithms were designed for 1-minute epochs and a
proprietary, undocumented "activity count" metric specific to 1990s actigraph hardware):
activity was approximated here as summed jerk (sum of \|diff\| of the tri-axial
acceleration-magnitude signal) per epoch, a standard substitute used in open
reimplementations (e.g. GGIR, pyActigraphy) when only raw accelerometry is available; each
algorithm's published window (e.g. Cole-Kripke's -2/+4 *minutes*) was rescaled to this
project's 30-second epochs by doubling epoch counts, preserving real-world window
duration. These are disclosed adaptations, not a claim of exactly reproducing either
paper's original validation.

**Results:**

| Algorithm | Mean per-subject AUC | Pooled AUC | Mean F1 |
|---|---|---|---|
| Cole-Kripke (1992) | 0.4822 | 0.5045 | 0.2382 |
| Sadeh (1994) | 0.4704 | 0.4805 | 0.0081 |

Both land at chance. This is the strongest evidence that the difficulty is not specific to
this project's LightGBM/calibration design: even decades-validated, movement-only formulas
that (per their own published validation studies) typically reach 85%+ accuracy in general
populations do not separate Wake from Asleep on this dataset under this evaluation.
Whether that reflects something about this specific cohort/recording setup, the
30-second/activity-count adaptation, or a genuine property of the AAUWSS protocol was not
resolved further — this is the point at which further debugging stopped being a good use
of time relative to what it would add.

## Method 4: CNN+attention — attempted, not completed (hardware, not a result)

Following Stage A-stress's pattern of comparing LightGBM against a raw-signal 4-branch
CNN with attention-gated fusion (`train_aauwss_sleep_cnn.py`, architecturally identical to
`train_wesad_cnn_attention.py`), this was attempted specifically because the user asked
for the comparison and because a raw-signal model can sometimes find structure
hand-crafted statistics miss. **It never produced a result**, for reasons specific to the
development machine, not the model:

- First two attempts were killed by the OS for running out of memory (8GB total system
  RAM, well under 1GB free even at baseline given other running applications). Root cause:
  the original ensemble design (3 independently-trained models per fold, each rebuilding
  its own full copy of the fold's training tensors) held multiple redundant large-tensor
  copies in memory at once — workable at WESAD's scale (~2,140 windows) but not at
  AAUWSS's (9,700 windows, ~4.5x more). Fixed by building each fold's tensors once and
  reducing the ensemble to a single model, with explicit cleanup between folds.
- The next two attempts (after the memory fix, and after freeing additional system RAM)
  ran without crashing but did not finish training even the first of 13 folds within
  ~21-24 minutes each, an impractical rate (~5+ hours projected for all folds) on this
  CPU-only, memory-constrained machine. Increasing the batch size 8x (32->256) to reduce
  Python-loop overhead did not resolve this.

Given three independent methods already agree on a near-chance result, and continuing to
fight this specific machine's hardware constraints for a fourth data point had reached
diminishing returns, further attempts were stopped by user decision. The script is kept in
the repository, documented as attempted-but-inconclusive (not as a negative modeling
result — it never ran to completion, so it cannot be reported as agreeing or disagreeing
with the other three methods).

## Decision: Stage A-sleep is not used in the deployed fusion pipeline

`fusion_pipeline.py` retains `stage_a_sleep_predict()` (callable, using the trained
LightGBM model) for completeness and future reference, but it is not called by any
default/demo code path, and `trigger_index()` in practice is only ever invoked with a
stress score. Including a near-chance (and pooled below-chance) sleep signal in a
production trigger_index would not be a conservative "extra, low-weight input" — a
below-chance signal is actively worse than no signal, since noisy-OR treats any elevated
reading as concerning regardless of whether that reading means anything. This mirrors how
Stage A-stress's CNN was benched in favor of the better-performing, more stable LightGBM
model, the same "measured, not assumed" standard applied throughout this project.

## What to cite in the paper

- Cole, R. J., Kripke, D. F., Gruen, W., Mullaney, D. J., & Gillin, J. C. (1992).
  Automatic sleep/wake identification from wrist activity. *Sleep*, 15(5), 461-469.
- Sadeh, A., Sharkey, K. M., & Carskadon, M. A. (1994). Activity-based sleep-wake
  identification: An empirical test of methodological issues. *Sleep*, 17(3), 201-207.
- This negative result as further, independent evidence (alongside Stage A-stress's CNN
  v2 and the Stage B cropped/normalized experiments) that this project's practice of
  running the real evaluation before trusting a plausible-sounding design is doing real
  work, not just process for its own sake.
