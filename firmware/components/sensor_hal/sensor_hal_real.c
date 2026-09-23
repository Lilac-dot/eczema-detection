/*
 * Real backend -- NOT IMPLEMENTED. No physical sensor (BMI270, PPG, moisture, EDA front
 * end) has been wired to this project yet, so every function here fails loudly instead of
 * silently returning a fake value that could be mistaken for a real reading. Fill in the
 * actual I2C/ADC driver calls once parts are on hand -- sensor_hal.h's contract (which
 * function reads what, at what rate) does not need to change to do that; only this file
 * does. See sensor_hal_simulated.c for the interface every function here must match.
 */
#include "esp_err.h"
#include "esp_log.h"
#include "sensor_hal.h"

static const char *TAG = "sensor_hal_real";

esp_err_t sensor_hal_init(void) {
    ESP_LOGE(TAG, "sensor_hal_real is a stub -- no driver code exists yet for BMI270/PPG/"
                  "moisture/EDA hardware. Select the Simulated backend (idf.py menuconfig) "
                  "until real driver calls are written here.");
    return ESP_ERR_NOT_SUPPORTED;
}

esp_err_t sensor_hal_read_bvp(float *out) { (void)out; return ESP_ERR_NOT_SUPPORTED; }
esp_err_t sensor_hal_read_accel(sensor_vec3_t *out) { (void)out; return ESP_ERR_NOT_SUPPORTED; }
esp_err_t sensor_hal_read_gyro(sensor_vec3_t *out) { (void)out; return ESP_ERR_NOT_SUPPORTED; }
esp_err_t sensor_hal_read_eda(float *out) { (void)out; return ESP_ERR_NOT_SUPPORTED; }
esp_err_t sensor_hal_read_temp(float *out) { (void)out; return ESP_ERR_NOT_SUPPORTED; }
esp_err_t sensor_hal_read_moisture(float *out) { (void)out; return ESP_ERR_NOT_SUPPORTED; }
esp_err_t sensor_hal_read_distance_mm(float *out) { (void)out; return ESP_ERR_NOT_SUPPORTED; }
esp_err_t sensor_hal_read_lux(float *out) { (void)out; return ESP_ERR_NOT_SUPPORTED; }
