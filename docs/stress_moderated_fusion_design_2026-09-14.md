# Stage C Redesign: Stress as a Moderator, Not an Averaged Score — 2026-09-14

## Why

The current Stage C fusion (`composite = 0.5 × image_score + 0.5 × stress_score`) treats the
wearable stress score and the image score as two equally-weighted, independent pieces of
evidence about the same thing. That's not how stress actually relates to eczema: the
literature this report already cites (Khan et al., 2024) frames stress as a **trigger** for
flares, not a direct severity signal. A flat average lets a calm stress reading drag down a
photo that clearly shows eczema, and lets a high stress reading push up the score for a photo
that shows nothing — neither of which makes clinical sense. This doc records the redesign
discussed to fix that, before it's implemented.

## The reframe: stress as a moderator, not a co-equal input

In statistics, a variable that changes *how strongly* one thing affects another (rather than
contributing its own independent share) is called a **moderator** (Baron & Kenny, 1986). That's
the right frame here: stress should only matter once eczema is already present, and it should
scale a *risk of worsening*, not blend into a *current severity* number.

## Literature found to support this

Saved under `papers/fusion_formula_review/` (five downloaded successfully; three blocked or
paywalled, links only):

- **Kittler, Hatef, Duin & Matas (1998), "On Combining Classifiers," IEEE TPAMI** — downloaded.
  General theory of score-combination rules (average, product, max/min); background for why a
  flat average is not automatically the "safe default" once the two scores mean different
  things.
- **Nandakumar et al. (2006), "Quality-based Score Level Fusion in Multibiometric Systems,"
  ICPR** — downloaded. Reference for weighting a score by how reliable/high-quality that
  particular reading is, relevant if reliability-weighting is preferred over the moderator
  approach.
- **Benchmarking Quality-Dependent and Cost-Sensitive Score-Level Multimodal Biometric Fusion
  Algorithms (arXiv 2111.08703)** — downloaded. More recent benchmark of the same family of
  fusion rules.
- **"Personalised prediction of daily eczema severity scores using a mechanistic machine
  learning model" (medRxiv 2020.01.16.20017772)** — downloaded. Closest real precedent found:
  predicts a patient's day-by-day AD severity trajectory as a function of ongoing triggers,
  fit to real longitudinal severity diaries. Not wearable- or photo-based, but the right
  structural analogue for "trigger drives change over time" rather than "trigger contributes to
  a static score."
- **Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer" (arXiv 1709.07871)**
  — downloaded. Reference architecture for a *trained* (not rule-based) version of "one signal
  modulates another" later, if a fusion model is ever trained on real paired data.
- **Pondeljak & Lugović-Mihić, "Psychological Stress and the Cutaneous Immune Response: Roles of
  the HPA Axis and the Sympathetic Nervous System in Atopic Dermatitis and Psoriasis" (PMC3437281)**
  — mechanism review; download blocked by PMC's bot-check.
  <https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3437281/>
- **"Predictors of Flares and Disease Severity in Patients With Atopic Dermatitis Using Machine
  Learning" (2025, PMC12268521)** — shows flare history predicts future severity, supporting
  "trajectory" as the right thing to model; download blocked by PMC's bot-check.
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC12268521/>
- **Baron & Kenny (1986), "The moderator-mediator variable distinction in social psychological
  research," Journal of Personality and Social Psychology** — the source of the moderator/
  mediator framing used above; paywalled (APA journal), not downloaded.
  <https://pubmed.ncbi.nlm.nih.gov/3806354/>

## Proposed architecture (not yet implemented)

Split Stage C into two separate outputs instead of one blended number:

1. **Eczema status** — Stage B's output, unchanged. This is still the primary read on "is this
   eczema, and roughly how bad does it look."
2. **Flare-risk** — a new, second output, gated on Stage B:

   ```
   flare_risk = eczema_score × stress_score
   ```

   If Stage B says "not eczema," `eczema_score` is low, so `flare_risk` stays near zero
   regardless of stress — there's nothing to flare. If Stage B confirms eczema, stress scales
   the risk up or down. This is a rule (like the current 0.5/0.5 formula), not a trained model —
   there is still no paired dataset to fit real weights against.

## Extending beyond stress: a trigger index

The same gated design generalizes past stress alone:

```
flare_risk = eczema_score × trigger_index
```

where `trigger_index` could combine more than one trigger signal. Candidates, sorted by how
reachable they are with current data:

- **Skin temperature** — already a WESAD wrist channel; could be added as a second trigger with
  no new data collection.
- **Sleep** — well-documented eczema trigger (it's one of SCORAD's own components), but WESAD is
  a single ~1-hour lab session, not an overnight recording, so it can't be pulled from WESAD
  itself. Would need a different dataset.
- **Scratching/motion** — a real trigger signal in principle, but blocked by the same hardware
  bandwidth ceiling already documented for Stage A's original WISDM approach (Chun et al., 2021
  found the discriminating signal sits at 100–800 Hz, well above wrist-accelerometer range).
- **Environmental factors** (humidity, allergens, season) — real triggers, but would need an
  external data source this project doesn't currently have.

## What this does and does not solve

This is a more scientifically defensible *mechanism* than a flat average — it matches the
stress-as-trigger literature instead of contradicting it. It does **not** solve the missing-data
problem the report already names in Sections 6.3 and 7.4; if anything it raises the bar, since
validating "does stress actually predict worsening" needs repeated, longitudinal measurements
per patient (multiple photos + stress logs over time), which is a stronger requirement than the
single-timepoint pairing already flagged as missing.

## Status and next steps

**Implemented (2026-09-14)**: `trigger_index()` and `flare_risk()` added to
`fusion_pipeline.py` alongside (not replacing) the existing `fuse()` composite score.
`trigger_index()` takes named trigger scores (e.g. `trigger_index(stress=0.8)`) and
averages them, so it's already written to accept more than one trigger without a redesign.
`demo()` now prints both `fuse()` and `flare_risk()` side by side for comparison.

**Update (2026-09-14, later the same day)**: **Stage A-sleep** was built and trained on
AAUWSS (13 subjects, overnight, same Empatica E4 sensor set as WESAD) — the "Sleep"
trigger candidate named below, previously blocked because WESAD's single ~1-hour lab
session can't supply overnight sleep data. It does not work: LOSO-CV mean AUC 0.46
(chance), corroborated by two literature-standard actigraphy formulas (Cole-Kripke 1992,
Sadeh 1994) also scoring at chance on the same data — see
`docs/aauwss_sleep_model_results_2026-09-14.md` for the full result and the diagnostics
run to rule out an implementation bug before accepting it. `trigger_index(stress=...,
sleep=...)` already supports a second named score with no interface change, but is not
called with a sleep score in practice, since Stage A-sleep is a documented negative
result, not a working second trigger.

Remaining:
- Longer-term: add WESAD's own skin-temperature channel as a further trigger (this is the
  single-point wrist temperature sensor already collected in WESAD/AAUWSS, distinct from
  lesion-vs-surrounding-skin thermal *imaging*, which remains out of scope — see
  `docs/architecture_v2_2026-09-14.md`).
- If ever revisited: Stage A-sleep's negative result used LightGBM on hand-crafted
  features; a raw-signal CNN comparison was attempted but never completed (development
  machine ran out of memory, then was too slow to finish even one LOSO fold) — this
  remains a genuinely open question, not a second negative data point.

## Refinement (2026-09-14): noisy-OR trigger combination, and a softening knob considered and partly rejected

The user asked whether the plain-product gate (`flare_risk = image_score × trigger_index`,
with `trigger_index` a plain mean of the individual trigger scores) is the best available
formula, and to check the literature for something better. Two independent design
questions came out of that, with two different outcomes:

**1. Combining multiple triggers into `trigger_index` — changed.** A plain mean dilutes a
high trigger toward the middle whenever a calmer second trigger is averaged in (e.g. high
stress 0.8 + calm sleep 0.15 averages to 0.475, even though the high stress reading alone
is arguably still the concerning signal). **Noisy-OR** — `trigger_index = 1 - ∏(1 -
score_i)` — is the standard way multiple independent risk factors are combined in Bayesian
medical-diagnosis networks (Pearl, 1988; e.g. the QMR-DT model), and fits this case better:
it rises toward whichever trigger is most elevated rather than averaging them together.
With exactly one trigger score this is numerically identical to a mean of one value (both
just return that value), so nothing changes in practice since Stage A-sleep (built,
trained, but a documented negative result — see the update above) isn't used as a real
second trigger — `fusion_pipeline.py`'s `trigger_index()` now uses this formula, and
`demo()` prints a synthetic worked example showing the mean vs. noisy-OR difference
directly, rather than a real second trigger score.

**2. Softening the outer product against one noisy score — added as an off-by-default
option, not a new default.** Kittler et al. (1998) — already cited above — found that pure
product combination of scores, despite having the cleanest theoretical justification under
independence, is more fragile in practice to one noisy or miscalibrated input than blended
alternatives. The obvious fix borrowed from that literature is a symmetric weighted
geometric mean of `image_score` and `trigger_idx` (e.g. `image_score^0.5 ×
trigger_idx^0.5`). **This was tried and deliberately rejected as the default**: it can push
`flare_risk` ABOVE `image_score` whenever `trigger_idx > image_score` (e.g. image_score=0.3,
trigger_idx=0.9 gives a geometric mean of ≈0.52, not ≤0.3), silently breaking the
"`flare_risk` can never exceed `image_score`" gating property this design was built around
and that the report already states as a deliberate feature (Section 6.2). Instead,
`flare_risk()` gained an optional `gamma` exponent applied only to `trigger_idx`:
`flare_risk = image_score × trigger_idx**gamma`. Since `trigger_idx**gamma ≤ 1` for any
`gamma > 0`, the bound is preserved for every value of `gamma`, while `gamma < 1` still
softens how harshly one low trigger reading can suppress the result (e.g. `0.01**0.5 =
0.1`, not as punishing as the raw `0.01`). Left at `gamma=1.0` (the original, unmodified
behaviour) by default, same reasoning as every other unfitted weight in this pipeline: no
paired dataset exists to tune it against, so a non-default value would be an unjustified
guess dressed up as an improvement.

**What to cite for this addition**: Pearl, J. (1988). *Probabilistic Reasoning in
Intelligent Systems: Networks of Plausible Inference.* Morgan Kaufmann — the original
noisy-OR formulation, applied here in the same spirit as its use in Bayesian
medical-diagnosis networks (e.g. QMR-DT) for combining multiple independent risk/finding
signals.
