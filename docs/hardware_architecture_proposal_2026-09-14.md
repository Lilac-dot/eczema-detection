# Proposed Hardware Architecture — 2026-09-14

## Why this exists

Every model in this project so far is trained on *public* datasets recorded with existing
research/commercial devices (WESAD and AAUWSS both used the Empatica E4). No custom
hardware has been built. This doc proposes what a real device for this system would look
like, split into what's buildable now (mirrors what the models were actually trained on)
and what's future work (needs new hardware AND a new dataset this project doesn't have).
This is a proposal, not something built or tested — flagged the same way as this
project's other "not yet validated" claims.

## Part 1: the wrist wearable (buildable now — matches training data exactly)

Stage A-stress and Stage A-sleep were both trained on the same sensor set (Empatica E4:
EDA, PPG/BVP, skin temperature, 3-axis accelerometer). A device built for real deployment
needs the *same* sensor set, or the trained models don't apply — this is not a free
choice, it's dictated by what the models were fit to.

**Reference architecture**: EmotiBit (Fitch, 2022, open-source hardware) is the closest
existing open design to what's needed: a Feather-compatible mainboard streaming raw
multi-wavelength PPG, 9-axis motion, skin temperature, and EDA over Bluetooth Low Energy,
buffered to onboard microSD. Its "strap-anywhere" design (sensors on a separate pod
connected to the mainboard) is a reasonable template rather than a fixed wrist-only form
factor.

**Proposed component list**, modeled on EmotiBit and the E4's own sensor architecture:

| Component | Function | Notes |
|---|---|---|
| EDA electrodes + analog front-end | Skin conductance (tonic/phasic) | Two-electrode contact, amplifier into separate tonic/phasic filter paths — standard EDA front-end design |
| PPG (IR emitter-receiver pair) | Blood volume pulse -> HR, IBI | Same signal type as E4's BVP channel; this project's `BVP_dominant_hz` feature depends on this |
| Thermistor or IR skin-temperature sensor | Wrist skin temperature | Feeds `TEMP_mean/slope` features already used by Stage A-stress |
| 3-axis accelerometer (or 9-axis IMU) | Motion / actigraphy | Feeds `ACC_*` features; also the basis for AAUWSS's actigraphy-style sleep-stage estimation |
| MCU + BLE radio | Sensor fusion, streaming | Feather-compatible microcontroller per EmotiBit's design |
| microSD buffer | Local logging when disconnected | Matches EmotiBit's design; avoids data loss on BLE dropout |

**Data flow**: wearable -> BLE -> companion phone app -> runs
`wesad_features.extract_window_features()` + personal-baseline calibration ->
Stage A-stress / Stage A-sleep models -> scores feed `fusion_pipeline.trigger_index()`.

**Calibration requirement carries over from the software design**: per
`docs/wesad_personal_baseline_calibration_2026-09-12.md`, any new wearer needs a short
calm/wake baseline recording before predictions mean anything for them — this is a
hardware+software requirement together (the device must support an explicit "calibration
mode" recording, not just continuous monitoring).

## Part 2: scratch detection (future — needs different hardware, not this project's wrist wearable)

Already ruled out at the current sensor spec: `docs/paper_review_adam_sensor_2026-08-26.md`
found that the accelerometer signal distinguishing scratching from other hand motion sits
at 100-800 Hz, well above what a standard low-power wrist IMU (the kind in Part 1, or
WISDM's original 20 Hz) captures. Two concrete hardware paths exist in the literature,
neither of which this project has built or has data for:

- **SIGMA (SensorIsed Glove for Monitoring Atopic dermatitis)** — a low-cost glove using
  silicone microtube stretch sensors filled with eutectic gallium-indium liquid metal,
  mounted over each finger, ~45g total (Sensors 2023, 23(24):9782). A glove is a very
  different form factor from a wrist wearable — worth naming as a real tradeoff (comfort
  and compliance for an all-day/overnight glove vs. a wristband).
- **ADAM (acoustomechanic sensor)** — already reviewed in this project
  (`docs/paper_review_adam_sensor_2026-08-26.md`) — detects the high-frequency
  acoustic/mechanical signal of scratching directly, rather than trying to infer it from a
  low-rate accelerometer.

Not pursued now: this needs new hardware AND a new labeled dataset (no public
scratch-labeled dataset exists at the needed sampling rate/modality), same missing-data
wall as the rest of this project's deferred ideas.

## Part 3: lesion-vs-surrounding-skin thermal imaging (future — needs new hardware AND a new dataset)

Per `docs/architecture_v2_2026-09-14.md`, this is the deferred "lesion temperature" idea —
distinct from the wrist temperature sensor in Part 1. This would NOT be part of the wrist
wearable (it needs to image a skin region, not sit on the wrist) — it would be a separate
handheld or phone-clip-on imaging accessory, used the same way a patient would take a
Stage B diagnostic photo.

**Concrete candidate hardware**: FLIR Lepton series micro thermal camera modules
(80x60 or 160x120 px, radiometric versions give calibrated per-pixel temperature, coin-sized,
low enough power draw for battery-powered/portable use) — an established, off-the-shelf
component for exactly this kind of miniaturized thermal sensing, already used in other
portable thermographic-imaging builds (see "The Development of a Cost-Effective Imaging
Device Based on Thermographic Technology," PMC10220945, for a worked example of building
a low-cost thermographic imaging device from a similar module).

**Proposed pairing**: thermal module + a standard RGB camera (already needed for Stage B),
co-registered so a lesion segmented in the RGB image can be mapped onto the thermal
image to compute a lesion-vs-surrounding-skin temperature delta — the actual measurement
this idea needs, which no existing dataset provides.

**Still blocked on**: an IRB-approved data collection protocol to build the training
dataset this would need — the hardware existing is necessary but not sufficient. This
remains future scope, not a near-term addition.

## What to cite in the paper for this section

- Fitch WT (or the EmotiBit project) — open-source EmotiBit hardware architecture, as the
  reference design for Part 1.
- Empatica E4 technical description — the commercial equivalent this project's models
  were actually trained on.
- SIGMA sensorised glove (Sensors 2023, 23(24):9782) and the ADAM acoustomechanic sensor
  (already cited) — the two concrete scratch-sensing hardware paths for Part 2.
- "Development of Objective Measurements of Scratching as a Proxy of Atopic
  Dermatitis — A Review" (Sensors 2025, 25(14):4316) — recent survey covering the broader
  scratch-sensing hardware landscape, useful as a single citation covering multiple
  approaches.
- "Skin Sensing and Wearable Technology as Tools to Measure Atopic Dermatitis Severity"
  (Skin Health and Disease, skinhd 4(5):e449) — already cited elsewhere in this project for
  the stress-eczema rationale; also relevant here for wearable AD-monitoring hardware
  context generally.
- FLIR Lepton module documentation and the cost-effective thermographic imaging device
  paper (PMC10220945) — for Part 3's thermal camera proposal.
