**Superseded 2026-09-18**: the publication direction changed. Paper 1 is now a systematic
architecture comparison (Stage B image models first, Stage A wearable models next), not the
domain-generalization/calibration split proposed below — see `honors-paper-report.docx` and
`archive/2026-09-17_multimodal_framework_paper/README.md`. Kept below for the research
history (the underlying experiments this doc tracks are all still real and still used), not
as the current plan.

# Research Workplan — 2026-09-16

## 1. What the current paper already establishes

`Multi_Sensor_Wearable_AD_Monitoring_Paper_2026-09-15_revised.docx` (last revised
2026-09-14/15, not modified since — this workplan does not change it) already documents,
with real results:

- **Stage A**: WESAD stress detection, personal-baseline calibration fix (mean AUC
  0.871→0.9405), reported descriptively (Section 4.4-4.7). A negative result for sleep
  (Section 4.8, AAUWSS, chance-level AUC, corroborated by two literature baselines).
- **Stage B**: the shortcut audit and rebuild (Section 5.1-5.4, final internal AUC 0.8644),
  **zero-shot external validation on SCIN and SkinDisNet** (Section 5.5-5.6, both near
  chance), and **three generalization interventions** (Section 5.7: augmentation — failed;
  natural-proportion multisource fine-tuning — helped SkinDisNet, barely helped SCIN;
  equal-weighted multisource — helped further, biggest internal-accuracy cost).
- **Stage C**: decision-level fusion, stress as a gating trigger (Section 6).
- **Future Work** (Section 9) already explicitly names Priority 4: "the untested volume
  hypothesis... is the most direct next lever" for Stage B generalization — the work below
  is that priority, plus adjacent work the paper doesn't yet anticipate.

**This workplan's job is to extend past that Future Work list, not repeat it.** Everything
below is either (a) directly executing an already-named future-work item, or (b) new work
done since the last paper revision that the paper doesn't know about yet.

## 2. What's been done since the last paper revision (not yet in the paper)

| Work | Status | Doc |
|---|---|---|
| Data hygiene check on SCIN/SkinDisNet as training sources (dupes, contamination) | Done — clean, one small leak fixed | `cross_dataset_matrix_2026-09-15.md` |
| Full 3x3 cross-dataset matrix (train-on-X/test-on-Y, all 9 cells) | Done | `cross_dataset_matrix_2026-09-15.md` |
| Domain-shift quantification (proxy A-distance, embeddings, origin classifier) | Done — found A-distance does NOT predict fine-tuning responsiveness (a real, reported inconsistency) | `domain_shift_analysis_2026-09-15.md` |
| WESAD calibration paired significance testing | Done today | `wesad_calibration_significance_2026-09-16.md` |
| Leave-one-dataset-out (LODO) training + formal paired bootstrap significance testing | **In progress** (background) | pending |
| SCIN training-volume expansion (the paper's own named Priority 4) | **Blocked** — machine ran out of RAM twice; genuinely unresolved, not a negative result | `cross_dataset_matrix_2026-09-15.md` Part 4, `finetuning_v2_expanded_2026-09-16.md` |
| Gray-world color-constancy preprocessing (uniform, not internal-external matching) | **Blocked**, same reason | same |

## 3. Immediate next steps, in order

1. **Let the running background LODO + significance-testing job finish**, then QA its
   numbers against the matrix/domain-shift docs before trusting them (same spot-check
   practice used for every other result so far — read the doc, sanity-check a few numbers,
   don't just relay the agent's summary).
2. **Retry the two blocked experiments** (SCIN volume, color constancy) once the machine
   has stable free memory — these remain genuinely open questions, not negative results,
   and directly answer the paper's own Priority 4.
3. **Decide the write-up path** (Section 4 below) once 1-2 land.
4. Only after that: revisit whether the paired-significance infrastructure built for the
   matrix work should also be applied retroactively to the three Section 5.7 interventions
   already in the paper (right now they're reported with independent, not paired, bootstrap
   CIs — a paired re-analysis would strengthen, not contradict, what's already written).

## 4. Where this could be published

Two distinct, non-competing paper directions have emerged from this work — writing them as
two papers rather than one avoids turning either into an unfocused model-zoo document.

### Paper A — Domain generalization in dermatology image classification (the new, deeper contribution)

**Core claim**: no pairwise combination of three real dermatology-image sources (a curated
clinical archive, crowd-sourced consumer photos, a second hospital's clinical photos)
transfers to either other source zero-shot; a formal domain-divergence measure confirms the
gap is real and large but — surprisingly — does not predict which direction is fixable by
fine-tuning; targeted interventions (equal-weighted multi-source exposure) partially close
the gap for one dataset and not the other.

**What it draws from**: Section 5.5-5.8 of the current paper (already written) as the
foundation, extended with the cross-dataset matrix, the domain-shift/A-distance analysis
and its inconsistency finding, and the LODO + paired-significance results once complete.

**Why this is a real contribution, not just "our model didn't generalize"**: it's a
general finding about the three-source setting (Section 2 of this workplan), a formal
divergence measurement that produces a non-obvious, honestly-reported result, and a
controlled comparison of which fixes work.

**Realistic venue**: a workshop track at a dermatology-imaging or medical-imaging venue
(e.g. an ISIC-style skin-imaging workshop at a computer-vision conference), a domain
generalization workshop, or an undergraduate/student research symposium. Not a flagship
clinical venue — there's no IRB or clinical-severity claim here, and shouldn't be.

### Paper B — Personal-baseline calibration for wearable stress detection (smaller, self-contained)

**Core claim**: pooled-normalization wearable stress models can have a high mean AUC while
catastrophically failing specific subjects due to decision-threshold transfer failure;
calibrating each subject against their own baseline recording fixes this — now with a
formal paired significance test (`wesad_calibration_significance_2026-09-16.md`: Wilcoxon
p=0.028 on AUC, bootstrap CI excluding zero) rather than only a descriptive before/after
comparison.

**What it draws from**: Section 4.4-4.7 of the current paper, plus today's new significance
testing.

**Realistic venue**: a wearable-computing or affective-computing workshop track (e.g. a
UbiComp/IMWUT workshop), or an IEEE EMBS student paper track. This is a narrow, clean,
well-quantified engineering result — publishable on its own, independent of the eczema
application.

### The existing paper itself

Once items 1-2 above land, the current paper (updated with the matrix/domain-shift/LODO
material) remains a reasonable **systems/pipeline paper** in its own right — the full
3-stage architecture with every component honestly audited, including two clearly-labeled
negative results (sleep trigger, cross-dataset generalization). This is the paper's own
existing scope; it doesn't need a new angle, just the updates in Section 2 above folded in
once decided.

## 5. What NOT to do (guardrails carried over from this project's established constraints)

- Do not use any test partition for tuning, model selection, or checkpoint selection —
  every experiment so far has respected this; keep doing so.
- Do not fabricate results for the two blocked experiments — report them as unresolved
  until they actually run.
- Do not merge Paper A and Paper B into one submission — they have different audiences,
  different methods sections, and combining them risks reading as an unfocused model-zoo
  study, which this project has explicitly avoided so far.
- Do not make any clinical flare/severity claim in either paper — neither has the paired
  longitudinal patient data that would support one.
- Preserve every negative result (sleep trigger, SCIN's persistent near-chance ranking,
  the two blocked/unresolved experiments) in whichever paper discusses them — do not quietly
  drop them if they remain negative.
