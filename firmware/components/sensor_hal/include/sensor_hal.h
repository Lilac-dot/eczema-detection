#pragma once
#include <stdint.h>
#include "esp_err.h"

/*
 * Every sensor this project's hardware plan calls for (docs/hardware_architecture_v3_
 * 2026-09-18.md), behind one interface with two interchangeable backends selected at
 * build time (idf.py menuconfig -> AD Wearable Node Configuration -> Sensor backend):
 *
 *   - sensor_hal_simulated.c: synthetic waveforms, no hardware needed. This is what lets
 *     the windowing/BLE-packaging/phone-side pipeline be built and tested honestly before
 *     any physical sensor exists.
 *   - sensor_hal_real.c: intended to wrap the actual I2C/ADC drivers (BMI270, a
 *     MAX30101-class PPG part, the moisture sensor, an EDA front end). Currently a stub
 *     that fails loudly at init -- fill in real driver calls once parts are on hand.
 *
 * Callers (main.c's periodic sampling tasks) never know which backend is linked in; only
 * this header's contract matters. Sample rates match wesad_features.py's FS dict exactly
 * for BVP/EDA/TEMP/ACC, since those three (plus accel) are what the deployed WESAD stress
 * model was trained on -- changing them here would silently change what the phone-side
 * model receives.
 */

typedef struct {
    float x, y, z;
} sensor_vec3_t;

#define SENSOR_HAL_BVP_HZ 64
#define SENSOR_HAL_ACCEL_HZ 32
#define SENSOR_HAL_GYRO_HZ 32
#define SENSOR_HAL_EDA_HZ 4
#define SENSOR_HAL_TEMP_HZ 4
#define SENSOR_HAL_MOISTURE_HZ 4
#define SENSOR_HAL_DISTANCE_HZ 5
#define SENSOR_HAL_LUX_HZ 5

/* Brings up whichever sensors the compiled DEVICE_ROLE actually has. Must be called once
 * before any sensor_hal_read_* call. Returns an error (does not abort) if a real-backend
 * sensor fails to respond -- callers should decide whether to keep running degraded. */
esp_err_t sensor_hal_init(void);

/* Each fills *out with the current reading. Called from a periodic esp_timer at the
 * matching SENSOR_HAL_*_HZ rate above -- these are not expected to block. */
esp_err_t sensor_hal_read_bvp(float *out);            /* wristband: PPG blood volume pulse */
esp_err_t sensor_hal_read_accel(sensor_vec3_t *out);   /* wristband: BMI270 accelerometer */
esp_err_t sensor_hal_read_gyro(sensor_vec3_t *out);    /* wristband: BMI270 gyroscope --
                                                         * no trained model uses this yet
                                                         * (hardware_architecture_v3), sampled
                                                         * and streamed anyway so it's on
                                                         * record for future model work. */
esp_err_t sensor_hal_read_eda(float *out);             /* skin patch: electrodermal activity */
esp_err_t sensor_hal_read_temp(float *out);             /* skin patch: skin temperature, deg C */
esp_err_t sensor_hal_read_moisture(float *out);        /* skin patch: hydration proxy --
                                                         * no trained model yet either
                                                         * (Section 7.3 of the report),
                                                         * streamed for future use */
esp_err_t sensor_hal_read_distance_mm(float *out);     /* imaging gate: time-of-flight
                                                         * distance to skin (VL53L1X-class
                                                         * part), millimeters. Not part of
                                                         * hardware_architecture_v3 -- added
                                                         * for the standardized-acquisition
                                                         * research question (docs/
                                                         * imaging_capture_gate_design_
                                                         * 2026-09-19.md), a third node
                                                         * alongside the wristband/skin patch. */
esp_err_t sensor_hal_read_lux(float *out);             /* imaging gate: ambient light sensor
                                                         * (e.g. BH1750-class part), lux --
                                                         * used to compensate the LED ring's
                                                         * output, not to classify anything
                                                         * itself. */
