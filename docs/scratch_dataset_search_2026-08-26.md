# Looking for a Better Scratch Dataset — Findings

## Why WISDM isn't a real fix

The current Stage A model (`docs/stage_a_first_model_2026-08-26.md`) uses WISDM's "teeth"
(brushing teeth) activity as a proxy for scratch-like repetitive hand motion, because
WISDM has no scratch label at all. That's a real limitation, not a preference — it's
worth actively looking for something better, which is what this covers.

## What I searched for

Public/downloadable accelerometer or wearable-sensor datasets specifically labelled with
scratch vs. non-scratch (or itch-related) events, searching general web sources, GitHub,
Kaggle, Zenodo, and PhysioNet, plus checking a 2025 systematic review of scratch-detection
methods and two individual scratch-detection papers directly for data-availability
statements.

## Update: a stronger finding than "no public dataset" (from a paper added later)

`abf9405.pdf` (Chun et al. 2021, *Sci. Adv.*, the "ADAM sensor" paper —
full review in `docs/paper_review_adam_sensor_2026-08-26.md`) was added after this search
was first written. It sharpens the conclusion below considerably: scratching's actual
discriminating signal is a **100–800 Hz acousto-mechanic vibration**, and the paper
directly demonstrates that even a 100 Hz-sampling smartwatch (Apple Watch) can't capture
it — the smartwatch algorithm they benchmarked against misclassified hand-waving as
scratching and missed finger-only/head scratching entirely. WISDM samples at **20 Hz**
(confirmed in its own README) — 5x below even that inadequate smartwatch, 80x below the
paper's own sensor (1600 Hz). So this isn't only a missing-label problem: **no amount of
searching will turn up a usable public dataset built on typical consumer-grade
(~20–100 Hz) accelerometers, because the hardware itself can't record the signal that
matters.** A real fix needs either a high-bandwidth sensor (custom hardware, or licensing
something like ADAM) or accepting that a WISDM-style proxy model is a coarse,
motion-only stand-in with a known, literature-backed ceiling — not a placeholder for a
soon-to-be-fixed data problem.

## What exists in the literature (and why none of it is usable off-the-shelf)

Several research groups have built real scratch detectors, but **none currently have a
public dataset**:

- **ADAM sensor — skin-conformable wireless acousto-mechanic sensor** (Chun et al. 2021,
  reviewed in full in `docs/paper_review_adam_sensor_2026-08-26.md`). Custom hardware,
  1600 Hz z-axis sampling, mounted on the dorsum of the hand. 99.0% accuracy / 84.3%
  sensitivity / 99.3% specificity in real nocturnal AD-patient validation (n=11, 46
  nights). Data availability: not public, "may be requested from the authors" — though
  several authors/the underlying patent are tied to a company (Sibel Health), which may
  make licensing harder than a purely academic dataset request.
- **SIGMA — a sensorised glove for scratch detection** (microtubular stretchable sensors
  + IMU, ML classifier). Checked its data-availability statement directly: *"data...are
  available on request from the corresponding author... not publicly available due to
  the video recording that could compromise the privacy of research participants."*
  ([PMC10748247](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10748247/))
- **Wrist actigraphy + machine learning for nocturnal scratch quantification**
  ([PMC11174528](https://pmc.ncbi.nlm.nih.gov/articles/PMC11174528/)) and an **accelerometer
  + recurrent neural network** approach to nocturnal scratch detection
  ([PubMed 28613187](https://pubmed.ncbi.nlm.nih.gov/28613187/)) — both proprietary
  clinical-trial data.
- A **2025 JMIR analytical validation study** compared wearable and touchless (radar/camera)
  home scratch-monitoring technologies
  ([jmir.org/2025/1/e72216](https://www.jmir.org/2025/1/e72216)) — again a clinical
  validation study, not a released dataset. Pfizer is a named partner running related
  pediatric trials
  ([Pfizer overview](https://www.pfizer.com/news/articles/capturing-itch-using-digital-wearable-devices-help-patients-atopic-dermatitis)).
- I checked a dedicated **2025 systematic review of 21 scratch-detection studies /
  14 distinct solutions**
  ([PMC12299226](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12299226/)) specifically for
  any public dataset across all of them. Direct finding: **none of the 14 solutions have a
  public dataset.** Every study did proprietary data collection (1–31 participants each),
  and the review's own conclusion calls for "a modular, preferably open-source scratch
  detection framework" as something that doesn't yet exist — i.e. the field has already
  identified this exact gap.

## General (non-scratch) accelerometer/gesture datasets that do exist

For contrast, general HAR/gesture datasets are genuinely public and downloadable, but none
have a scratch label — the same limitation WISDM already has:

- [MotionSense](https://github.com/mmalekzadeh/motion-sense) — smartphone accel/gyro,
  activity + attribute recognition.
- uWave — accelerometer-based hand-gesture dataset (controlled gestures, not scratch).
- PAMAP2, MMASH — general physical-activity/health monitoring, not hand-motion focused.
- Child Mind Institute "Detect Sleep States" (Kaggle) — wrist accelerometer, sleep-state
  labels, not motion-type labels.
- None of these are a clear upgrade over WISDM for this specific purpose — they'd swap one
  proxy-activity set for another without adding an actual scratch label.

## Bottom line

**There is no public, downloadable dataset with real scratch labels right now** — this
isn't a search-effort problem, it's a genuine, reviewer-acknowledged gap in the field,
now reinforced by a concrete hardware-bandwidth explanation for *why* (see update above).
Realistic options, roughly in order of effort:

1. **Request access directly.** ADAM's and SIGMA's authors both explicitly offer data
   "on request" — worth an email describing this project; academic groups often share for
   non-commercial research use, sometimes under a data-use agreement. The other clinical
   papers could be worth the same approach even without an explicit request clause. Note
   ADAM's commercial ties (Sibel Health, a filed patent) may make that one slower/harder
   than a purely academic ask.
2. **Consider whether the sensor, not just the dataset, needs to change.** If genuine
   scratch detection (not a coarse motion proxy) is a real project goal, a ~20–100 Hz
   consumer accelerometer feed has a hard ceiling per the ADAM paper's own direct test.
   That points toward either a higher-sampling-rate wearable for any self-collected data
   (see point 4), or explicitly scoping Stage A as motion-only/coarse by design.
3. **Keep WISDM as the pipeline validator, not the final answer.** The Stage A pipeline
   (feature extraction → subject-level split → threshold tuning) built today is
   dataset-agnostic; it can be pointed at real scratch data the moment any becomes
   available, without redoing the methodology.
4. **Small self-collected pilot** — ideally with a higher-sampling-rate device than a
   typical smartwatch (see point 2) given what the ADAM paper shows about the 100–800 Hz
   band mattering. A few volunteers performing a short scripted protocol (scratch vs. a
   few control gestures — typing, brushing hair, waving) would give a genuine, if small,
   scratch-labelled dataset. Worth putting on the ethics/IRB radar now given it's already
   the identified bottleneck for real patient data per `Progress_Log_2026-08-26.docx` §5.2.
5. **Combine WISDM activities into a broader "hand-motion-like" positive class** (teeth +
   writing + clapping vs. gross-motor activities like walking/jogging) rather than a single
   proxy — cheap to try, doesn't require new data, but inherits the same 20Hz ceiling, so
   treat it as a methodology exercise rather than a path to a trustworthy scratch detector.

I did not download or integrate any of the above (no public data to fetch) — this is a
survey to inform the decision, not a change to the pipeline.

Sources:
- Chun et al. 2021, "A skin-conformable wireless sensor to objectively quantify symptoms of pruritus," *Sci. Adv.* 7, eabf9405 (`abf9405.pdf`; full review in `docs/paper_review_adam_sensor_2026-08-26.md`)
- [SIGMA sensorised glove paper (data availability checked directly)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10748247/)
- [Systematic review of scratch-detection methods (checked directly for public datasets)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12299226/)
- [Quantifying Nocturnal Scratch via wrist actigraphy + ML](https://pmc.ncbi.nlm.nih.gov/articles/PMC11174528/)
- [Detection of Nocturnal Scratching via accelerometers + RNN](https://pubmed.ncbi.nlm.nih.gov/28613187/)
- [JMIR 2025 wearable/touchless scratch-monitoring validation study](https://www.jmir.org/2025/1/e72216)
- [Pfizer overview of wearable itch-monitoring partnerships](https://www.pfizer.com/news/articles/capturing-itch-using-digital-wearable-devices-help-patients-atopic-dermatitis)
- [MotionSense dataset (general HAR, no scratch label)](https://github.com/mmalekzadeh/motion-sense)
