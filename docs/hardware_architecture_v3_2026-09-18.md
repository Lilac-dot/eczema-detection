# Proposed Hardware Architecture v3 — 2026-09-18

Supersedes `docs/hardware_architecture_proposal_2026-09-14.md` (kept, not deleted, for
history). Still a proposal, not something built or tested — flagged the same way as this
project's other "not yet validated" claims.

## What changed from the v1 (2026-09-14) proposal, and why

The v1 proposal was a single wrist wearable carrying every sensor (EDA, PPG, skin
temperature, accelerometer). Revisiting against the wearable-AD literature specifically
(rather than just matching what WESAD/AAUWSS happened to record) suggested a cleaner
guiding principle:

> Only include a sensor if there is a defensible literature connection to an AD-related
> observable, and split sensors across two devices by what they're actually measuring
> (movement vs. skin-surface state) rather than putting everything on one wrist node
> because that's where the training data came from.

That produced three concrete changes, one of which surfaced a real compatibility problem
that had to be resolved before adopting it:

1. **Split into two devices**: a wrist node (motion only) and a separate skin patch
   (skin-surface sensing), instead of one wrist node carrying everything.
2. **Added a skin-hydration/moisture sensor** — not in the v1 design at all. Justified by
   real, recent, verified literature (below) directly on AD skin-barrier monitoring, not a
   speculative addition.
3. **Added a gyroscope** alongside the accelerometer on the wrist node (BMI270 gives both
   in one part) — no currently-trained model uses gyroscope data, so this is a
   forward-looking addition, not something the deployed Stage A model can use today.

**The compatibility problem, and how it was resolved**: an earlier draft of this revision
dropped the PPG/BVP sensor from the wrist node entirely, keeping only the IMU. Checked
directly against the deployed Stage A stress model
(`models/wesad_stress_lightgbm.txt`) rather than assumed: **12 of its 52 features are
BVP-derived** (`BVP_mean`, `BVP_std`, `BVP_min`, `BVP_max`, `BVP_dominant_hz`,
`BVP_band_power`, and their personal-baseline-relative counterparts). Dropping PPG would
have made the already-deployed, already-calibrated, already-leakage-audited Stage A model
unable to compute nearly a quarter of its own inputs — not a minor accuracy cost, a hard
break. **Resolution: keep a PPG sensor on the wrist node.** Retraining the deployed model
without BVP was the alternative and was explicitly rejected — re-validating a model that's
already been through significant calibration and leakage-correction work (Section 4.1.3,
4.1.4 of the main report) is a larger cost than keeping one more small, cheap sensor.

## Architecture

```text
                         PHONE
                ┌─────────────────────┐
                │ Camera -> Image CNN │
                │ Fusion              │
                │ Dashboard           │
                └──────────┬──────────┘
                           │  BLE
              ┌────────────┴────────────┐
              │                         │
       ┌─────────────┐           ┌──────────────┐
       │ WRISTBAND   │           │  SKIN PATCH  │
       │ BMI270      │           │  Moisture    │
       │  (accel+gyro)│          │  Temperature │
       │ PPG (MAX30101│           │  EDA*        │
       │  -class)    │           │  ESP32-S3    │
       │ ESP32-S3    │           └──────────────┘
       └─────────────┘
```
\* EDA is the least-established of these for AD specifically — treated as an
exploratory/contextual stress signal (feeding the existing WESAD-trained model), not an
AD biomarker in its own right. Same framing as the v1 design already used.

## Wristband: BMI270 (accel+gyro) + PPG

**What it measures**: wrist motion (scratching/itch-related movement) and pulse
(feeds the deployed Stage A stress model's BVP-derived features).

**Scratch-detection literature, reframed from v1**: the v1 doc treated scratch detection
as needing fundamentally different hardware (SIGMA glove or the ADAM acoustomechanic
sensor), citing `docs/paper_review_adam_sensor_2026-08-26.md`'s finding that the signal
discriminating scratching from other hand motion sits at 100-800 Hz, above what a standard
wrist IMU captures. That finding still stands for *high-precision* scratch discrimination.
But newly-verified literature (2026-09-18) shows **standard wrist accelerometer + ML does
achieve reasonable scratch quantification**, just not via the same mechanism ADAM uses:

- Mahadevan et al. (2021), *npj Digital Medicine* — wrist accelerometer-only pipeline for
  nighttime scratch + sleep in AD patients.
- Ji et al. (2023), *npj Digital Medicine* — actigraphy + topological features +
  model-ensembling for nocturnal scratch, validated against video ground truth.
- Xing et al. (2024), *Sensors* 24(11):3364 — Apple Watch accelerometer+gyroscope at
  100 Hz, ML-based scratch quantification (already cited in the main report, ref [26]).
- Au et al.'s SIGMA glove (already cited, ref [4]) is a different form factor (finger
  stretch sensors + IMU) reaching 83-99% scratch-prediction accuracy, useful as an upper
  bound on what dedicated hardware can do, not evidence for or against a plain wrist IMU.

**Correction to the v1 framing**: scratch detection is not "blocked on exotic hardware
this project doesn't have" — a standard wrist accelerometer+gyroscope, the same class of
part already on the wrist node for other reasons, has real literature precedent for
*coarse* scratch quantification (not the high-fidelity discrimination ADAM targets). What
this project still lacks is a **trained model** for it — no scratch-labeled dataset
matching this hardware exists yet in this project (the original WISDM-based attempt was
abandoned for the ADAM-identified bandwidth reasons, at 20 Hz; this hardware runs faster).
Building that dataset is new future work, not a hardware blocker.

## Skin patch: moisture + temperature + EDA

**New in v3** — not present in the v1 wrist-only design.

**Moisture/hydration — the strongest justification for adding a new sensor at all**:
- Dai et al. (2025), *npj Flexible Electronics* — a wearable capacitive textile sensor
  purpose-built for continuous AD monitoring, demonstrated distinguishing lesional from
  non-lesional skin in a clinical case study. Shares three authors (Todorov, Torah, Beeby,
  Ardern-Jones) with Khan et al. (already cited, ref [13]) — a real companion paper from
  the same Southampton group, not an isolated find.
- Shin et al. (2022/2023), *Advanced Healthcare Materials* — wireless soft skin-hydration
  sensor, thermal-conductivity-based, validated on 200+ dermatology-clinic patients.
  Senior author John A. Rogers is a well-established name in flexible bioelectronics.
- Sivakumar et al. (2023), *ACS Sensors* — WASP, a wearable closed-chamber hygrometer for
  continuous transepidermal water loss (TEWL), the more clinically standard skin-barrier
  measurement, explicitly framed for AD/psoriasis monitoring.

**Temperature**: kept from v1, same role — supporting/contextual, not a standalone AD
marker. A contact sensor (e.g. TMP117-class) is adequate; an IR/non-contact option matches
the Empatica E4's own measurement principle more closely if budget allows (same tradeoff
already noted in `docs/wrist_node_hardware_design_2026-09-15.md`'s temperature-sensor
choice).

**EDA**: kept, explicitly exploratory. Two independent recent dermatology reviews (Kiani
et al. 2022, *JEADV Clinical Practice*; Henkel et al. 2026, *JDDG*) describe EDA-based
wearable AD concepts as measuring sympathetic tone/stress rather than AD directly, and
note EDA specifically has not been well-studied in atopic eczema the way actigraphy has.
This matches the project's own WESAD-trained stress model's actual scope (general
physiological stress, not an AD-specific signal) — feeding EDA+temperature into the
existing WESAD model gives *context* (physiological arousal alongside image and motion
signals), not a new diagnostic claim.

**Placement caution carried over from `docs/wrist_node_hardware_design_2026-09-15.md`**:
EDA electrodes should not sit directly on active eczema lesions — irritated/broken skin
barrier complicates both electrode contact and patient comfort. The literature itself
(Khan et al., ref [13]) emphasizes wearable materials for AD need to be breathable and
non-irritating specifically because AD skin is unusually sensitive; this applies more to
the skin patch (in prolonged surface contact) than the wristband.

## Phone

Unchanged from v1's already-settled conclusion (`docs/hardware_architecture_proposal_
2026-09-14.md` Section 6.2 of the main report already rejected a Raspberry Pi 5 hub for
power/size reasons): camera, all trained models, fusion logic, dashboard, and BLE
aggregation all run on the phone. No separate compute hub.

## What still needs updating (not done in this revision)

`docs/wrist_node_hardware_design_2026-09-15.md` is a detailed single-board BOM/PCB design
combining EDA+temp+PPG+IMU on one wrist node — that design predates the two-node split and
would need to be split into two BOMs (wristband: BMI270 + PPG + MCU/BLE; patch: moisture
sensor + temperature + EDA front end + MCU/BLE) before it's an accurate build target for
this architecture. Flagged as a follow-up, not done here — the BOM/PCB-level detail is a
substantially larger revision than this architecture-level update.

## Citations for this section

- Dai, H. et al., "High resolution reverse-offset printed wearable laminated textile
  capacitive sensor for continuous monitoring of atopic dermatitis," *npj Flex. Electron.*,
  2025.
- Shin, J. et al., "Wireless, Soft Sensors of Skin Hydration with Designs Optimized for
  Rapid, Accurate Diagnostics of Dermatological Health," *Adv. Healthc. Mater.*, 2022/2023.
- Sivakumar, A. D. et al., "WASP: Wearable Analytical Skin Probe for Dynamic Monitoring of
  Transepidermal Water Loss," *ACS Sens.*, 8(11):4407-4416, 2023.
- Mahadevan, N. et al., "Development of digital measures for nighttime scratch and sleep
  using wrist-worn wearable devices," *npj Digit. Med.*, 2021.
- Ji, J. et al., "Assessing nocturnal scratch with actigraphy in atopic dermatitis
  patients," *npj Digit. Med.*, 2023.
- Kiani, C., Kain, A., Zink, A., "Wearables and smart skin as new tools for clinical
  practice and research in dermatology," *JEADV Clin. Pract.*, 2022.
- Henkel, L. et al., "Digital Dermatology," *JDDG*, 24(7):e934-e952, 2026.
- Already cited in the main report and unchanged: Khan et al. (ref [13]), Au et al. SIGMA
  glove (ref [4]), Au et al. scratch review (ref [3]), Xing et al. (ref [26]).
