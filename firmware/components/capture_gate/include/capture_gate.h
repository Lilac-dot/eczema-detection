#pragma once
#include <stdbool.h>
#include "esp_err.h"

/*
 * Capture-condition gate for the standardized-acquisition robustness research question
 * (docs/imaging_capture_gate_design_2026-09-19.md). Third device role alongside the
 * wristband and skin patch from hardware_architecture_v3 -- NOT part of that doc, added
 * separately.
 *
 * Scope decision, stated plainly: this device does not take the photo itself. The user's
 * available hardware is a plain ESP32 (no confirmed camera module), so this firmware
 * controls distance (VL53L1X-class time-of-flight sensor) and illumination (a diffused
 * LED ring, ambient-compensated via a lux sensor), and reports a READY/NOT-READY state
 * over BLE -- the phone (which already owns the camera + CNN in hardware_architecture_v3)
 * only takes the photo once this device reports READY. If an ESP32-CAM-class board with
 * an attached camera becomes available later, this same readiness signal could gate an
 * on-device capture instead -- revisit then, not designed for now.
 */

typedef struct {
    float distance_mm;
    float distance_target_mm;
    float distance_tolerance_mm;
    bool distance_ok;
    float lux;
    float lux_target;
    uint8_t led_duty_percent;   /* what the LED ring is currently driven at, 0-100 */
    bool illumination_ok;       /* true once LED duty has been within its stable band
                                  * for the debounce window below */
    bool ready;                 /* distance_ok AND illumination_ok, debounced */
} capture_gate_status_t;

esp_err_t capture_gate_init(void);

/* Call at SENSOR_HAL_DISTANCE_HZ (sensor_hal.h). Reads distance+lux, drives the LED PWM
 * duty cycle, updates the debounced readiness state, and writes the current status into
 * *out. Never blocks. */
esp_err_t capture_gate_update(capture_gate_status_t *out);
