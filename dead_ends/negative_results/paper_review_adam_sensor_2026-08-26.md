# Paper Review — Chun et al. 2021, "A Skin-Conformable Wireless Sensor to Objectively Quantify Symptoms of Pruritus" (Sci. Adv., the ADAM sensor)

Source file: `abf9405.pdf`. Full citation: K. S. Chun
et al., *Sci. Adv.* 7, eabf9405 (2021).

## What the paper does

Builds a custom hand-worn sensor (ADAM: ADvanced Acousto-Mechanic) specifically to solve
scratch detection, and directly benchmarks it against a wrist-worn smartwatch approach
(Apple Watch + "Itch Tracker" app). Two studies: an algorithm-development study (10
healthy adults, controlled scratching/non-scratching tasks) and a clinical validation
study (11 predominately pediatric AD patients, 46 nights, 378.4 hours, judged against
manually-labelled infrared video as ground truth).

## The single most important finding for this project: sampling rate, not just labels

Scratching produces two distinct signal components:
1. **Gross hand/arm motion** — low frequency (a few Hz or less). This is what any
   standard accelerometer (WISDM, a smartwatch, a phone) can see.
2. **Acousto-mechanic vibration** from fingertip/fingernail friction against skin —
   extends up to **~600–800 Hz**, with the most useful energy up to ~200 Hz. This is the
   component that actually distinguishes scratching from other hand motions.

The paper shows directly (Fig. 3) that a **wrist-mounted, 100 Hz-sampling smartwatch**
(Apple Watch Series 4) **cannot capture this second component at all**, and as a result
its existing algorithm misclassifies hand-waving as scratching and misses finger-only
scratching and head-scratching entirely. Their own device samples the critical axis at
**1600 Hz** specifically to capture this content, and is mounted on the dorsum of the
hand (not the wrist) because the signal is already strongly attenuated by the time it
reaches the wrist.

**This directly explains — and upgrades the diagnosis of — the weak Stage A result
already documented in `docs/stage_a_first_model_2026-08-26.md`.** WISDM's accelerometer
data is sampled at **20 Hz** (confirmed in `dataset/WISDM/wisdm-dataset/README.txt`) —
by Nyquist, it physically cannot represent anything above 10 Hz, let alone the 100–800 Hz
band this paper identifies as the actual discriminating signal. WISDM is 5x below the
smartwatch this paper already showed was inadequate, and 80x below the paper's own
sensor. So the modest AUC/F1 obtained for the teeth-brushing proxy model wasn't only a
"wrong proxy label" problem — **it's a hardware bandwidth ceiling**. No amount of better
labels, better proxies, or better modelling on 20 Hz accelerometer data can recover a
signal that was never sampled. This is a stronger and more specific conclusion than
"WISDM has no scratch label," and changes what "fixing Stage A" would actually require —
it means fixing the *label*, but not fixing the *sensor*, still caps how good any model
built on WISDM (or any similar ~20–100 Hz consumer accelerometer feed) can ever get at
true scratch detection.

## Other useful findings

- **Methodology validation:** they use leave-one-subject-out cross-validation (LOSO-CV)
  specifically because subject-level splits prevent a classifier from learning
  "confounding [within-subject] relationship...with unrealistically high classification
  accuracy," citing Saeb et al. 2017. This is exactly the subject-level split already
  used for the Stage A model here — good independent confirmation that was the right
  call, not overcaution.
- **Feature set** (Fig. 5C, ranked by RF importance) — all derived from a single z-axis
  signal: sum of |z-accel|, several FFT power-sum/peak bands (2–5, 5–20, 20–50, 80–100,
  100–140 Hz), and power above the 50th-percentile amplitude. Useful reference if this
  project ever works with a higher-sample-rate sensor feed.
- **Reported performance:** 89.1% accuracy / 87.8% sensitivity / 88.1% specificity on
  the healthy-subject validation set (LOSO-CV); 99.0% accuracy / 84.3% sensitivity / 99.3%
  specificity on the real clinical (nocturnal, AD patient) dataset. The clinical number is
  higher mainly because nighttime has far fewer confounding activities (no typing,
  texting, etc.) than the daytime validation protocol — worth remembering as a reason not
  to over-index on a single headline accuracy number without checking what activities were
  actually being distinguished from what.
- Even simple everyday actions like typing, texting, and mouse-clicking show *some*
  energy in the 100–200 Hz band (impulse artifacts) — so even a high-bandwidth sensor
  needs more than one frequency feature to separate scratch from those, not a hard
  100 Hz cutoff rule.

## Data availability (consistent with everything else found so far)

Not public. Data statement: *"All data needed to evaluate the conclusions in the paper
are present in the paper and/or the Supplementary Materials. Additional data related to
this paper may be requested from the authors."* Same "ask the authors" pattern as
everything else in `docs/scratch_dataset_search_2026-08-26.md`, now updated to include
this paper. Worth noting: several authors and the underlying patent are tied to a
company (Sibel Health) with a commercial interest in the sensor, which may make a data
license harder to obtain than a purely academic ask.

## What this changes about the project's plan

- Add to the honors paper's limitations/framing: this project's Stage A approach (WISDM,
  ~20 Hz) is explicitly a coarser-grained, motion-only proxy, not a claim to reproduce
  scratch-detection accuracy at the level ADAM/similar high-bandwidth sensors achieve —
  and now there's a specific, citable technical reason why, rather than a vague caveat.
- If genuine scratch detection is a real goal (not just a proxy pipeline demo), the
  practical options aren't just "find better labels" anymore — they now include
  "acquire/borrow a higher-sampling-rate wearable" (a hardware question) as a
  first-class option, on top of the data-sourcing options already listed in
  `docs/scratch_dataset_search_2026-08-26.md`.
- The LOSO-CV validation principle already used for Stage A now has direct literature
  backing from a paper doing the same task — worth citing in the methods section.
