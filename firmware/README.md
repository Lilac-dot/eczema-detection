# AD Wearable Node Firmware

ESP-IDF (v5.4.3 tested) firmware for the two-device hardware plan in
`docs/hardware_architecture_v3_2026-09-18.md`. One project, two build-time roles
(wristband / skin patch) picked via `idf.py menuconfig` -> "AD Wearable Node Configuration".

**Status (2026-09-18): skin patch role only, simulated sensors, builds clean, NOT yet run
on real hardware or in emulation.** Wristband role is stubbed (logs a warning, does
nothing) -- not built out yet.

## What it does

Skin patch role samples EDA (4Hz), temperature (4Hz), and moisture (4Hz) and streams each
over BLE (NimBLE, one GATT service, one notify characteristic per sensor) to whatever
central connects -- the phone, per the architecture doc, which does all feature extraction
and model inference. This firmware's only job is reliable raw sensor delivery.

No physical sensor exists yet, so `sensor_hal_simulated.c` generates synthetic
plausible-looking waveforms instead of reading real hardware -- this validates the
firmware pipeline (sampling timers, BLE packaging, GATT service structure), NOT the
sensors themselves. See that file's header comment before reading anything into the
specific numbers it produces.

## Build (verified working on this machine, 2026-09-18)

ESP-IDF's `export.sh` refuses to run under Git Bash/MSYS ("MSys/Mingw is not supported").
Use PowerShell:

```powershell
cd firmware
& "C:\Espressif\frameworks\esp-idf-v5.4.3\export.ps1"
idf.py set-target esp32s3   # first time only
idf.py build
```

Builds clean with zero warnings in `main.c`, `sensor_hal_*.c`, or `ble_service.c` (checked
by grepping a full rebuild log, not just eyeballing the tail).

## What's NOT verified yet

- **Never run**, on real hardware or in emulation -- a clean compile+link is real evidence
  the code is syntactically and API-correctly matched to ESP-IDF 5.4's NimBLE headers, but
  it is not evidence the BLE service actually advertises/notifies correctly at runtime, or
  that the timers fire at the intended rates. Don't describe this as "working" until one
  of the options below has actually been tried.
- **No QEMU installed** on this machine for ESP32-S3 -- `idf.py qemu` isn't available here.
  Espressif publishes a QEMU fork with ESP32-S3 support; installing it is a separate step
  not done as part of this session.
- **Wokwi** (wokwi.com) can simulate an ESP32-S3 running this firmware in-browser, including
  BLE, without installing anything locally -- the most realistic path to "see this actually
  run" without buying hardware. Needs manual setup (create a Wokwi project, copy this
  firmware's build output in) not done here.
- **Real hardware**: any ESP32-S3 dev board (e.g. an ESP32-S3-DevKitC, a few dollars) can be
  flashed directly with `idf.py -p <PORT> flash monitor` and will genuinely advertise as
  "AD-SkinPatch" over BLE, visible in any phone BLE scanner app (e.g. nRF Connect) -- the
  most direct way to confirm this for real, at the cost of buying one board.

## Next steps, in likely order

1. Get this actually running somewhere (Wokwi, QEMU, or a real board) before writing more
   firmware -- confirming the skin patch role works end-to-end is worth more right now
   than starting the wristband role on unverified foundations.
2. Wristband role (BMI270 accel+gyro + PPG) -- same pattern, new sensor_hal channels
   already declared in `sensor_hal.h` (read_bvp/read_accel/read_gyro), just needs the
   BLE service's GATT table and main.c's sampling loop extended, mirroring what's here.
3. Real sensor drivers, once hardware is chosen/bought -- fill in `sensor_hal_real.c`
   (currently a stub that fails loudly by design, not a placeholder that looks like it
   works).
