# Wrist Sensor Node — Low-Cost Custom PCB Design (Draft)

Companion to Section 8 of the main report. This is a from-scratch, low-cost wrist node
targeting the same four WESAD wrist modalities the deployed Stage A model was trained on
(EDA, skin temperature, PPG/BVP, 3-axis acceleration), designed for someone comfortable
with embedded firmware but new to PCB layout. Not yet built, not yet validated — a
concrete starting point for KiCad, not a finished design.

Key design choice that shapes everything below: EDA uses AC excitation (matching the
Empatica E4's own approach, which uses 8Hz/100uA AC excitation specifically to avoid
electrode-polarization drift) but keeps the *analog* circuit deliberately simple — a
single instrumentation amp, no dedicated impedance-AFE chip (AD5941 was considered and
rejected for a first PCB; see the design-decision note in the EDA section) — and pushes
the actual synchronous demodulation into firmware, where embedded skill is stronger than
PCB/analog skill. This is the one subsystem worth the most bring-up time.

## Bill of materials (low-cost target)

All prices are approximate small-quantity (qty 5-10) unit costs from LCSC/Mouser/Digikey
as of this writing — verify current pricing before ordering, prices move.

| Ref | Part | Function | Package | Qty | ~Unit | ~Ext | Datasheet / notes |
|---|---|---|---|---|---|---|---|
| U1 | Raytac MDBT50Q-1MV2 | nRF52840 MCU + BLE, pre-certified, chip antenna | SMD module, ~10.5x15.5mm | 1 | $5.50 | $5.50 | raytac.com/product/ins.php?index_id=24 — internal REG0 DC-DC can run directly off the battery rail (1.7-5.5V), so it does not need the regulated 3V3 rail below |
| U2 | MAX30101EFD+ | PPG / heart-rate | OLGA-14 (~5.6x3.3mm) | 1 | $4.00 | $4.00 | analog.com/media/en/technical-documentation/data-sheets/max30101.pdf — needs **two** supplies: VDD 1.7-2.0V (typ. 1.8V) for logic/ADC, VLED+ 3.1-5.0V (typ. 3.3V) for the LED driver. Do not tie these together. |
| U3 | MCP9808T-A0M/MC | Skin temperature (contact) | DFN-8, 2x3mm | 1 | $1.20 | $1.20 | Cheap default. Swap for MLX90632 (~$9, SFN package, non-contact IR — matches the E4's own measurement principle) as a Rev 2 upgrade if budget allows; keeping MCP9808 for Rev 1 keeps the footprint and bring-up simpler. |
| U4 | LSM6DS3TR-C | IMU, accel+gyro | LGA-14, 2.5x3mm | 1 | $1.80 | $1.80 | st.com/resource/en/datasheet/lsm6ds3tr-c.pdf — CS pin tied HIGH selects I2C mode (CS=1 enables I2C); SDO/SA0 tied LOW sets I2C address 0x6A |
| U5 | INA333AIDGKR | Instrumentation amp, EDA front end | VSSOP-10 | 1 | $3.20 | $3.20 | ti.com/lit/ds/symlink/ina333.pdf — gain G = 1 + (100k / R_G) between pins 1/8; REF (pin 5) tied to GND per TI's own default recommendation, output swings 0V..VDD around the AC excitation carrier |
| U6 | MCP73831T-2ACI/OT | LiPo linear charger | SOT-23-5 | 1 | $0.55 | $0.55 | Standard single-cell charger, sets charge current via PROG resistor to GND |
| U7 | MCP1700-3302E/TT | 3.3V LDO (sensor rail) | SOT-23 | 1 | $0.35 | $0.35 | Feeds U2(VLED)/U3/U4/U5 and I2C pull-ups off a clean 3.3V, independent of U1's own internal regulation |
| U8 | MCP1700-1802E/TT | 1.8V LDO (MAX30101 logic rail) | SOT-23 | 1 | $0.35 | $0.35 | MAX30101 VDD only — do not feed this into anything else |
| — | LiPo cell, 150-250mAh, **with built-in protection PCB** | Battery | — | 1 | $4.50 | $4.50 | Buying a pre-protected cell avoids needing a separate battery-protection IC |
| — | Passives: decoupling caps (100nF/1uF/4.7uF per IC per datasheet), I2C pull-ups (4.7k x2), excitation RC filter (R+C), status LED resistor | — | 0402/0603 | ~35 | $0.01-0.05 | ~$1.20 | Stick to JLCPCB's "Basic Parts" library for these to avoid per-part assembly setup fees |
| J1 | JST-PH 2-pin, SMD | Battery connector | SMD | 1 | $0.20 | $0.20 | |
| J2 | 2x5 1.27mm SWD header | Program/debug | THT | 1 | $0.50 | $0.50 | Or substitute 4 pogo-pin test pads (SWDIO/SWCLK/GND/RESET) to save board space/cost if you have a pogo jig |
| J3 | JST-PH 2-pin, SMD | EDA electrode leads (flying leads to snap electrodes off-board) | SMD | 1 | $0.20 | $0.20 | |
| J4 | USB-C receptacle (power/charge only, no data) | Charge input | SMD | 1 | $0.40 | $0.40 | |
| SW1 | Small SMD slide switch | Power on/off | SMD | 1 | $0.25 | $0.25 | |
| — | Ag/AgCl snap electrodes, pack of 10+ | EDA sensing, off-board | — | 2 | ~$0.30 ea | $0.60 | Wired via J3 |
| — | Status LED | Bring-up/debug indicator | 0603 | 1 | $0.05 | $0.05 | |

**Component total: ~$24.75/board.** PCB fab (JLCPCB, 2-layer, 5 boards): **~$2-5**.
SMT assembly (JLCPCB, ~9 unique SMD parts most in their "extended" library, ~$3 one-time
setup fee each the first time + per-placement cost, batch of 5 boards): **~$40-80**.

**Total for a 5-board prototype batch: roughly $150-220 all-in — about $30-45 per
assembled board**, versus $549.99 for the EmotiBit bundle. Hand-soldering instead of using
assembly service is possible but not recommended for a first board given the fine-pitch
packages (OLGA-14, DFN-8, LGA-14 have no accessible leads — they need reflow/hot-air, not
an iron) — a $30-50 hot-air rework station is the DIY alternative to the ~$40-80 assembly
fee if you'd rather learn hand-reflow than pay for assembly.

## Subsystem net list

Organize your KiCad schematic as hierarchical sheets matching this structure — copy
each IC's own datasheet "typical application circuit" for exact decoupling values rather
than inventing them.

### Sheet: Power

- Battery+ -> J1.1 -> SW1 -> `VBAT_SW`
- Battery- -> J1.2 -> `GND`
- `VBAT_SW` -> U6.BAT (charge target); J4 USB VBUS -> U6.VDD (charge source); U6.PROG -> resistor to GND (sets charge current, see datasheet table); U6.STAT -> optional status LED
- `VBAT_SW` -> U1.VDD directly (module's internal REG0 handles 1.7-5.5V -> its own regulated rail; decouple per Raytac's app circuit, typically 100nF + 4.7uF close to the pin)
- `VBAT_SW` -> U7.VIN -> U7.VOUT = `3V3` (decouple per MCP1700 datasheet, typically 1uF in/out)
- `VBAT_SW` -> U8.VIN -> U8.VOUT = `1V8` (same pattern, 1.8V rail, MAX30101 VDD only)
- `3V3` -> U2.VLED+, U3.VDD, U4.VDD/VDDIO, U5.VDD, and the top of both I2C pull-up resistors
- `1V8` -> U2.VDD only
- All GND pins, including module exposed pad, -> common ground pour

### Sheet: MCU / BLE (U1)

- SWDIO, SWCLK, RESET -> J2
- I2C: one GPIO pair -> `I2C_SDA`/`I2C_SCL` (each pulled to `3V3` via 4.7k, per bus, not per device)
- GPIO -> U2.INT (MAX30101 interrupt, open-drain, needs its own pull-up to `1V8` since that's the MAX30101's logic level)
- GPIO -> U4.INT1 (optional, LSM6DS3TR-C interrupt)
- GPIO (PWM-capable) -> EDA excitation RC filter input (see EDA sheet)
- ADC-capable GPIO <- EDA front-end output (U5.OUT)
- Spare GPIO -> status LED (through ~1k resistor)

### Sheet: PPG (U2, MAX30101)

- `I2C_SDA`/`I2C_SCL` -> U2 SDA/SCL
- U2.INT -> MCU GPIO (pull-up to `1V8`)
- VDD decouple: 1uF close to pin; VLED+ decouple: 4.7uF, low-impedance path, close to pin (per datasheet's own warning about LED-pulse ripple)
- Placement: at a board edge/underside so it can face skin through an enclosure cutout

### Sheet: Temperature (U3, MCP9808)

- `I2C_SDA`/`I2C_SCL` -> U3 SDA/SCL
- A0/A1/A2 address pins -> GND (sets default I2C address)
- 100nF decouple
- Placement: skin-contact side, ideally near U2 since both need skin contact

### Sheet: IMU (U4, LSM6DS3TR-C)

- `I2C_SDA`/`I2C_SCL` -> U4 SDA/SCL
- CS -> `3V3` (tied high, selects I2C mode)
- SDO/SA0 -> GND (I2C address 0x6A)
- INT1 -> MCU GPIO (optional)
- 100nF decouple

### Sheet: EDA front end (U5, INA333 + excitation)

This is the subsystem to bring up first on a breadboard before trusting it in a layout.

- MCU PWM GPIO -> R1 (start ~1k) -> node also to C1 (~100nF to GND) forming an RC
  low-pass that smooths the PWM into a quasi-sine excitation tone (pick the PWM carrier
  frequency and RC corner together on the bench — this needs real tuning, don't treat the
  values above as final)
- Filtered excitation -> series resistor (limits current into skin, target the same
  rough order of magnitude as the E4's own ~100uA excitation) -> J3 pin 1 -> Electrode A
- Electrode B -> J3 pin 2 -> U5.IN+ (through an optional small series protection resistor)
- U5.IN- -> tied to a fixed reference (simplest: GND, matching TI's own default
  recommendation for REF/IN- in the common case)
- U5.REF (pin 5) -> GND (per TI's stated "most common application"; keep this connection
  low-impedance — stray resistance here directly degrades CMRR per the datasheet)
- Gain resistor R_G across pins 1/8: G = 1 + (100k / R_G) — choose R_G on the bench once
  you know the real signal amplitude you're seeing; don't hardcode a gain into the layout
  you haven't measured yet, use a footprint you can easily rework
- U5.OUT -> MCU ADC input
- 100nF decouple on U5.VDD

Firmware does synchronous demodulation: sample U5.OUT at a rate well above the excitation
frequency, multiply by a reference sine/cosine at that same frequency, low-pass filter the
product to recover the conductance-proportional magnitude. This is the piece that replaces
what an AD5941 would do in hardware — keeping it in software is the deliberate trade
described in the main design discussion.

### Sheet: misc

- Status LED + resistor
- Test points on `3V3`, `1V8`, `GND`, `I2C_SDA`, `I2C_SCL` — cheap, and very useful for
  bring-up debugging with a multimeter/scope on a first board

## Floorplan guidance

- Keep the EDA front end (U5, excitation RC, J3) physically away from the module's
  antenna area and away from the PWM/digital switching traces where practical — route the
  excitation and sense lines short and direct. Noise coupling into a sub-microsiemens
  signal is the most likely first-board bug.
- Respect U1's antenna keepout exactly as specified in Raytac's own datasheet/footprint
  package (typically a rectangular no-copper, no-ground-plane zone under/around the chip
  antenna) — this is a copy-the-datasheet step, not a design decision.
- U2 (PPG) and U3 (temperature) both need skin contact — place them together at one edge
  or on the underside, with a matching cutout planned in whatever enclosure you build.
- Target board size: aim for something in the 30-40mm x 20-25mm range for a first
  attempt — small enough to be wearable, large enough that a first-time layout isn't
  fighting for space on top of everything else being new. Miniaturize in a Rev 2 once the
  circuit is proven.

## Why this doc doesn't include a KiCad project file

KiCad isn't installed in this environment, so nothing here could be opened, DRC-checked,
or verified before handing it over — for a first PCB, a broken auto-generated file would
cost more time than it saves. Build the schematic directly in KiCad from the net list
above (each subsystem sheet maps directly to a KiCad hierarchical sheet); it's a few
hours of transcription, not a redesign. JLCPCB's free DRC check
(jlcpcb.com/RGE/freekicad) against their standard 2-layer capability set (6/6mil
trace/space is comfortably within their basic process) will catch layout-rule violations
before you pay to fab anything.

## Suggested build order

1. Breadboard/perfboard bring-up with breakout boards for every subsystem — especially
   the EDA front end — before committing anything to a PCB layout. Get the full firmware
   signal chain (sampling, BLE streaming, software lock-in EDA demodulation) working on
   real hardware first.
2. KiCad schematic, sheet by sheet, from the net list above.
3. KiCad PCB layout — start from the floorplan guidance, copy each IC's datasheet
   reference layout for the tricky parts (especially the module's antenna area).
4. JLCPCB DRC check, then order (2-layer, 5 boards, SMT assembly).
5. Bring-up and debug; budget for at least one respin.

Not yet done: any of the above. This document is the starting point, not a validated
design.
