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

Not yet implemented in code or in the report. When it is:

- Implement `flare_risk()` in `fusion_pipeline.py` alongside (not replacing) the existing
  `fuse()` composite score.
- Rewrite Section 6 of the report around this design, with the same "not yet validated" framing
  used throughout.
- Longer-term: add skin temperature as a second trigger once the stress-only version is written
  up and working.
