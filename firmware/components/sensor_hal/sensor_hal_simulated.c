/*
 * Simulated backend: synthetic waveforms, no I2C/ADC hardware touched at all. This is the
 * backend selected by default (idf.py menuconfig -> Sensor backend -> Simulated) precisely
 * because no physical sensor exists for this project yet -- it exists so everything above
 * this file (BLE packaging, the phone-side pipeline) can be built and tested honestly in
 * the meantime, not to produce numbers anyone should read as real physiology.
 *
 * Every value below is a plausible-looking waveform, not a calibrated simulation of any
 * real sensor's noise characteristics or of real skin physiology -- do not use this
 * backend's output to draw any conclusion about the moisture/EDA/temp CHANNELS themselves
 * (e.g. "does the moisture sensor concept work"), only about whether the FIRMWARE PIPELINE
 * around them (sampling, BLE notify, phone-side parsing) is wired up correctly.
 */
#include <math.h>
#include "esp_err.h"
#include "esp_timer.h"
#include "esp_random.h"
#include "sensor_hal.h"

static float seconds_now(void) {
    return (float)esp_timer_get_time() / 1e6f;
}

/* esp_random() is a full 32-bit CSPRNG draw -- cheap, but overkill for "add a bit of
 * jitter to a synthetic signal". Fine for a simulation backend; a real backend would
 * never call this. */
static float unit_noise(void) {
    return ((float)(esp_random() % 2000) / 1000.0f) - 1.0f;  /* [-1, 1] */
}

esp_err_t sensor_hal_init(void) {
    /* Nothing to bring up -- no peripheral exists. Real return value kept meaningful
     * (ESP_OK) so callers that check it behave the same regardless of backend. */
    return ESP_OK;
}

esp_err_t sensor_hal_read_bvp(float *out) {
    float t = seconds_now();
    /* ~72 bpm cardiac-ish waveform: fundamental + a second harmonic to look less like a
     * pure sine, plus noise. Units are arbitrary (real PPG output is a raw ADC count, not
     * a physical unit) -- matches wesad_features.py treating BVP as unitless too. */
    float hr_hz = 1.2f;
    *out = sinf(2.0f * (float)M_PI * hr_hz * t)
         + 0.3f * sinf(4.0f * (float)M_PI * hr_hz * t)
         + 0.05f * unit_noise();
    return ESP_OK;
}

esp_err_t sensor_hal_read_accel(sensor_vec3_t *out) {
    /* Resting wrist: ~1g on one axis, small jitter on all three. Real units would be g;
     * this is unscaled. */
    out->x = 0.02f * unit_noise();
    out->y = 0.02f * unit_noise();
    out->z = 1.0f + 0.02f * unit_noise();
    return ESP_OK;
}

esp_err_t sensor_hal_read_gyro(sensor_vec3_t *out) {
    out->x = 0.5f * unit_noise();
    out->y = 0.5f * unit_noise();
    out->z = 0.5f * unit_noise();
    return ESP_OK;
}

esp_err_t sensor_hal_read_eda(float *out) {
    float t = seconds_now();
    /* Slow drift (simulating a slowly-changing arousal level) + a much slower "phasic"
     * ripple + noise. Real skin EDA sits roughly 1-20 microsiemens; this stays in a
     * similar numeric range so downstream code that assumes "a small positive float"
     * doesn't choke, but it is NOT a validated EDA model. */
    float slow_drift = 5.0f + 2.0f * sinf(2.0f * (float)M_PI * 0.01f * t);
    float phasic = 0.3f * sinf(2.0f * (float)M_PI * 0.2f * t);
    *out = slow_drift + phasic + 0.1f * unit_noise();
    return ESP_OK;
}

esp_err_t sensor_hal_read_temp(float *out) {
    float t = seconds_now();
    /* Skin temperature drifts slowly around a plausible wrist/skin-surface value. */
    *out = 32.0f + 0.5f * sinf(2.0f * (float)M_PI * 0.005f * t) + 0.05f * unit_noise();
    return ESP_OK;
}

esp_err_t sensor_hal_read_moisture(float *out) {
    float t = seconds_now();
    /* PLACEHOLDER RANGE -- no moisture sensor has been selected/spec'd yet (report
     * Section 7.1/7.3: hardware not built, no dataset collected), so there is no real
     * unit or expected range to simulate against yet. This 0-100 "hydration index" shape
     * exists only so the BLE characteristic and phone-side parsing have something to
     * carry -- revisit this function's whole shape once real hardware is chosen. */
    *out = 50.0f + 10.0f * sinf(2.0f * (float)M_PI * 0.003f * t) + 1.0f * unit_noise();
    return ESP_OK;
}

esp_err_t sensor_hal_read_distance_mm(float *out) {
    float t = seconds_now();
    /* Simulates a hand/phone rig drifting near a 150mm target distance -- slow drift
     * (someone repositioning) plus fast jitter (hand tremor / sensor noise). Not a
     * calibrated VL53L1X noise model, just plausible enough to exercise the readiness
     * state machine in capture_gate before real hardware exists. */
    *out = 150.0f + 15.0f * sinf(2.0f * (float)M_PI * 0.05f * t) + 3.0f * unit_noise();
    return ESP_OK;
}

esp_err_t sensor_hal_read_lux(float *out) {
    float t = seconds_now();
    /* Simulates indoor ambient light wandering between ~150 and ~450 lux (someone moving
     * relative to a window/lamp), not a real photometric simulation. */
    *out = 300.0f + 150.0f * sinf(2.0f * (float)M_PI * 0.01f * t) + 10.0f * unit_noise();
    if (*out < 0.0f) *out = 0.0f;
    return ESP_OK;
}
