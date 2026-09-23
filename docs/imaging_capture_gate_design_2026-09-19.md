# Imaging Capture-Gate Firmware — Design Notes (2026-09-19)

## What this is

A third device role added to `firmware/` for the "D" (controlled imaging hardware)
research question, alongside the existing wristband/skin-patch roles from
`docs/hardware_architecture_v3_2026-09-18.md` (that doc is unchanged by this — this is a
separate addition, not a revision to it). New Kconfig option: `idf.py menuconfig` ->
AD Wearable Node Configuration -> Device role -> **Imaging capture gate**.

## Scope decision: this device does not take the photo

The user's available hardware right now is **a plain ESP32** ("kindaa, i have an esp32" —
not a confirmed ESP32-CAM or other camera-carrying variant, and no camera module, distance
sensor, or LED ring confirmed in hand yet). Given that, this firmware is designed as a
**capture-condition enforcer + trigger**, not a self-contained camera:

- It reads a time-of-flight distance sensor (VL53L1X-class, I2C) and an ambient light
  sensor (BH1750-class, I2C) — **neither is wired up or confirmed owned yet**; these are
  the parts this firmware's `sensor_hal_real.c` stub expects once real hardware exists.
- It drives a diffused LED ring via PWM, compensating (open-loop, see caveat below) for
  ambient brightness.
- Once distance is within tolerance AND the LED isn't railed at its min/max duty (meaning
  it can still meaningfully compensate) for a debounce window, it reports **READY** over
  BLE. The phone — which already owns the camera and every trained model in
  `hardware_architecture_v3` — takes the actual photo only once READY is asserted.

**If an ESP32-CAM-class board with an attached camera becomes available later**, this same
readiness signal could gate an on-device capture instead of a phone capture — noted as a
future option in the code, not designed for now, since it wasn't the hardware in hand.

## What was built (software only — no physical parts wired up, nothing flashed to real hardware)

- `sensor_hal.h`/`.c`: added `sensor_hal_read_distance_mm()` and `sensor_hal_read_lux()` to
  the existing sensor abstraction (same simulated/real backend pattern already used for
  EDA/temp/moisture). Simulated backend produces plausible drifting values so the rest of
  the pipeline can be built and tested before real sensors exist — same philosophy already
  documented in `sensor_hal_simulated.c`'s header comment.
- `firmware/components/capture_gate/`: new component. Owns the LED PWM (ESP-IDF `ledc`
  driver), the distance-tolerance check (target 150mm ± 8mm — an arbitrary starting point,
  not derived from any lens/FOV calculation yet), the ambient-compensation heuristic, and
  the debounced READY state machine (5 consecutive good readings at 5Hz, ~1s, before
  asserting ready — avoids flickering as a hand passes through the sensor's view).
- `ble_service.c`: extended to support a role-conditional GATT table. The imaging-gate role
  gets its own service UUID and three notify characteristics (ready, distance, lux)
  instead of the skin patch's (moisture, temp, EDA) — picked via `#if
  CONFIG_DEVICE_ROLE_IMAGING_GATE` at compile time, mirroring how `main.c` already
  branches by role. Device advertises as `AD-ImagingGate` in this role.
- `main.c`: new `start_imaging_gate_sampling()`, gated the same way as the existing skin
  patch startup path.
- Kconfig: new `DEVICE_ROLE_IMAGING_GATE` choice option.

## Verified: both roles build clean

Built with ESP-IDF v5.4.3 (`C:\Espressif\frameworks\esp-idf-v5.4.3`), target `esp32s3`
(the target this `firmware/` project was already configured for before this addition —
see the hardware mismatch note below). Both `DEVICE_ROLE_SKIN_PATCH` (the existing default)
and the new `DEVICE_ROLE_IMAGING_GATE` compile to a complete binary with no errors under
this project's `-Werror` build (one real bug caught and fixed in the process: the new BLE
UUID constants were unconditionally declared, which failed the skin-patch build with
`-Werror=unused-const-variable` since they're only referenced when the imaging-gate role's
GATT table is compiled in — fixed by making the UUID declarations themselves role-
conditional, not just the table that uses them). sdkconfig was left set back to the
project's existing default (`DEVICE_ROLE_SKIN_PATCH`) after verifying the imaging-gate role
also builds — this addition does not change what role gets built by default.

**Untested at runtime** — same caveat as the rest of `firmware/`: no physical ESP32 has
been flashed with this code, no VL53L1X/light sensor/LED ring has been wired up, and the
ambient-compensation heuristic and debounce timing are unvalidated guesses, not tuned
values.

## Hardware mismatch to resolve before flashing

This `firmware/` project is currently configured for **ESP32-S3** (`idf.py set-target
esp32s3` was already run before this session, for the wristband/skin-patch roles' BLE +
peripheral needs). The user's confirmed hardware is **a plain ESP32** — a different chip
(original Xtensa LX6 core, different peripheral set, no built-in USB-JTAG). Before flashing
this to real hardware:

1. Run `idf.py set-target esp32` in `firmware/`, which will need a fresh `idf.py
   menuconfig` pass (target-specific defaults, especially anything S3-specific like PSRAM
   config) — not done in this session since it would invalidate the S3 config without a
   real device to test the ESP32 config against yet.
2. Confirm the exact ESP32 board (WROOM/WROVER/dev-kit variant) and its actual available
   GPIOs — `LED_RING_GPIO` in `capture_gate.c` is currently `GPIO_NUM_18`, a placeholder
   pin from the S3 config, not checked against a real ESP32 board's pinout or against
   whichever GPIOs the VL53L1X/light-sensor I2C bus ends up using.
3. This mismatch does not block the software work (the corruption-robustness benchmark,
   `docs/robustness_corruption_benchmark_2026-09-19.md`) — that ran entirely on the
   existing trained image model and needed no hardware at all.

## What's still a real simplification, stated plainly

The ambient-light compensation in `capture_gate.c` is **open-loop**: the lux sensor reads
ambient light, and LED duty is adjusted inversely, but there's no way yet to distinguish
"light from the LED ring itself" from "ambient room light" at the sensor (that would need
either a light sensor positioned away from the LED's direct path, or a duty-cycled
measurement scheme sampling between LED pulses). This is a real, deliberate simplification
for a first version, not something already solved — revisit once real hardware exists to
characterize how much this matters in practice.

## Next steps (not started)

1. Confirm exact parts (VL53L1X or equivalent ToF sensor, BH1750 or equivalent lux sensor,
   a diffused LED ring + MOSFET/driver circuit) and whether the user will source them.
2. `idf.py set-target esp32` + a fresh menuconfig pass once the target is confirmed.
3. Wire up real I2C addresses/pins in `sensor_hal_real.c` once parts are on hand (currently
   a stub that fails loudly, matching every other unbuilt sensor in this project).
4. Once flashed to real hardware: capture a small set of real photos varying distance
   in/out of tolerance, and compare against the synthetic blur benchmark
   (`docs/robustness_corruption_benchmark_2026-09-19.md`) to check whether real defocus
   blur from a wrong distance actually matches that benchmark's severity scale.
