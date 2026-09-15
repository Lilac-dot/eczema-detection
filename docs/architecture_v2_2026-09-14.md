# Architecture v2: Image Diagnosis as the Core, Wearable Triggers as Modifiers — 2026-09-14

## Why this exists

This records a reframing of how the three (now four-ish) stages relate to each other. The
individual stages aren't new — Stage B (image eczema diagnosis) and Stage A-stress (WESAD)
already existed and are unchanged by this doc — but the relationship between them is now
explicit: **Stage B is the core diagnosis**, and the wearable stages are **flare-risk
triggers**, not co-equal inputs averaged into one severity number. This matches the
`trigger_index`/`flare_risk` redesign in
`docs/stress_moderated_fusion_design_2026-09-14.md`, and adds a second trigger (sleep) plus
a explicit decision on a third candidate trigger (lesion temperature) that was considered
and deferred.

## The four pieces

1. **Stage B — image-based eczema diagnosis (core).** Unchanged: curated CNN, 81.07%
   accuracy. This alone answers "is this eczema."
2. **Stage A-stress — wearable stress trigger.** Unchanged: WESAD, LightGBM,
   personal-baseline calibrated, mean AUC 0.9405.
3. **Stage A-sleep — wearable sleep-quality trigger (built, negative result).** AAUWSS
   dataset (13 subjects, overnight, Empatica E4), trained and LOSO-CV evaluated
   2026-09-14: mean AUC 0.46 (chance), corroborated by two literature-standard actigraphy
   formulas (Cole-Kripke, Sadeh) also scoring at chance on the same data. Not used in the
   deployed pipeline — see `docs/aauwss_sleep_model_results_2026-09-14.md` for the full
   result and diagnostics.
4. **Lesion temperature — considered, deferred to future hardware work.** See "Temperature"
   below.

Stage C combines these via `trigger_index(stress=...)` and `flare_risk(image_score,
trigger_idx)` — see `scripts/fusion_pipeline.py`. `trigger_index` accepts a `sleep=`
score too (no interface change needed) but is not called with one in practice, since
Stage A-sleep doesn't work. The old flat `fuse()` average is kept only for comparison.

## Sleep: why AAUWSS, not DREAMT

Sleep was already named as a candidate trigger in the stress-moderator design doc, blocked
there because WESAD's ~1-hour lab session can't supply overnight sleep signal. Two public
wearable-sleep datasets were evaluated:

| | DREAMT (PhysioNet) | AAUWSS (Zenodo) — chosen |
|---|---|---|
| Subjects | 100 | 13 |
| Sensors | Empatica E4 (same as WESAD) | Empatica E4 (same as WESAD) |
| Labels | AASM sleep stages | AASM sleep stages |
| Access | Restricted Health Data License — requires PhysioNet registration + signing a data use agreement | Open access, CC-BY-4.0, no login |

AAUWSS was chosen specifically because it needs no account or agreement, consistent with
every other dataset this project uses (WESAD, SkinDisease). DREAMT's larger n would give a
statistically stronger result, and is worth revisiting if access friction turns out not to
matter for the project timeline — but it isn't a blocker to start with AAUWSS, which is
already the same order of magnitude as WESAD's own 15 subjects.

**Plan (mirrors the WESAD Stage A-stress pipeline exactly, since the sensors are
identical):**
- `wesad_features.extract_window_features()` and `wesad_calibration.py`'s personal-baseline
  approach are directly reusable — same modalities (EDA, TEMP, BVP, ACC), same rationale
  for why personal calibration matters (between-person differences in absolute signal
  level).
- Output: a sleep-disruption score in [0,1] (not the raw 5-class AASM stage), so it plugs
  into `trigger_index()` the same way the stress score does.
- Same LOSO-CV protocol, same per-subject reliability reporting via `loso_report.py`,
  given the similarly small subject count (13 vs. WESAD's 15).
- **Scope caveat, matching Stage A-stress's**: this detects general sleep disruption, which
  is a documented eczema-flare driver (poor sleep is one of SCORAD's own components) — it
  is not itself validated as an eczema-flare predictor, since no dataset pairs sleep and
  eczema severity for the same patients.

## Temperature: two different things, only one of which is in scope

"Temperature" as a trigger candidate splits into two genuinely different signals, worth
naming explicitly so they aren't conflated:

1. **Wrist skin temperature (already available)** — a single-point sensor reading (rising
   or falling over a window), already one of the four WESAD/AAUWSS Empatica E4 channels and
   already used as a Stage A-stress feature (`TEMP_mean`, `TEMP_slope`, etc.). Nothing new
   needed to add this as its own trigger later — it's already collected, just not yet
   split out as an independent `trigger_index` input distinct from the stress score it
   currently feeds into.
2. **Lesion-vs-surrounding-skin thermal imaging (out of scope, deferred)** — the idea
   originally proposed: does an eczema lesion run measurably warmer/cooler than the
   unaffected skin around it, imaged spatially (not a single wrist point). This needs an
   IR/thermal camera and a dataset pairing thermal images with labeled eczema lesions.
   Searched for one and found none — thermal skin-imaging datasets exist publicly for skin
   cancer and pressure injuries, not eczema, and none of them are structured as
   lesion-vs-surrounding-skin comparisons.

   **Deferred, not attempted**: there's no way to build this dataset without IR-capable
   hardware and a real data-collection protocol (informed consent, imaging procedure) this
   project doesn't have. This is explicitly future scope for when custom hardware exists to
   collect it directly — not designed around in the current architecture, and not blocking
   anything above.

## What changes in the report

- Section on system architecture: reframe as image-core + trigger-based modifiers, not
  three co-equal stages averaged together.
- Stage A section: split into Stage A-stress (done, deployed) and Stage A-sleep (done,
  negative result — trained and evaluated, not deployed), each with its own scope caveat.
  All done as of 2026-09-14 — see `docs/aauwss_sleep_model_results_2026-09-14.md`.
- Stage C section: describe `trigger_index`/`flare_risk` as the primary fusion rule, with
  the flat `fuse()` average kept only as a documented alternative/comparison.
- Future work section: lesion-thermal-imaging as a named, deliberately-deferred future
  direction contingent on custom hardware — not a gap in the current design, a scoped-out
  extension.
