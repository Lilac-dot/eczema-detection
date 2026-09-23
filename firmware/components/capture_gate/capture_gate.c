#include <math.h>
#include "esp_err.h"
#include "esp_log.h"
#include "driver/ledc.h"
#include "sensor_hal.h"
#include "capture_gate.h"

static const char *TAG = "capture_gate";

/* Placeholder GPIO/PWM assignments -- no physical wiring exists yet (the user has a plain
 * ESP32 dev board, parts not yet confirmed). Adjust once the LED ring driver circuit is
 * actually wired; nothing above this file needs to change to do that. */
#define LED_RING_GPIO           GPIO_NUM_18
#define LED_RING_LEDC_TIMER     LEDC_TIMER_0
#define LED_RING_LEDC_CHANNEL   LEDC_CHANNEL_0
#define LED_RING_LEDC_MODE      LEDC_LOW_SPEED_MODE
#define LED_RING_DUTY_RES       LEDC_TIMER_10_BIT   /* 0-1023 */
#define LED_RING_FREQ_HZ        5000

#define DISTANCE_TARGET_MM      150.0f  /* ~15cm working distance, arbitrary starting
                                          * point -- tune once a real lens/FOV is chosen */
#define DISTANCE_TOLERANCE_MM   8.0f

#define LUX_REFERENCE           300.0f  /* ambient level the LED compensation is tuned
                                          * around; not derived from any measurement */
#define LED_BASE_DUTY_PERCENT   50
#define LED_GAIN_PERCENT_PER_LUX 0.15f
#define LED_MIN_DUTY_PERCENT    5
#define LED_MAX_DUTY_PERCENT    95

#define DEBOUNCE_UPDATES        5   /* at SENSOR_HAL_DISTANCE_HZ (5Hz), ~1s of stable
                                      * conditions before reporting READY -- avoids
                                      * flickering ready/not-ready as a hand passes by */

static uint32_t stable_count = 0;

static uint8_t clamp_u8(float v, float lo, float hi) {
    if (v < lo) return (uint8_t)lo;
    if (v > hi) return (uint8_t)hi;
    return (uint8_t)v;
}

esp_err_t capture_gate_init(void) {
    ledc_timer_config_t timer_cfg = {
        .speed_mode = LED_RING_LEDC_MODE,
        .timer_num = LED_RING_LEDC_TIMER,
        .duty_resolution = LED_RING_DUTY_RES,
        .freq_hz = LED_RING_FREQ_HZ,
        .clk_cfg = LEDC_AUTO_CLK,
    };
    esp_err_t err = ledc_timer_config(&timer_cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "ledc_timer_config failed: %d", err);
        return err;
    }

    ledc_channel_config_t channel_cfg = {
        .gpio_num = LED_RING_GPIO,
        .speed_mode = LED_RING_LEDC_MODE,
        .channel = LED_RING_LEDC_CHANNEL,
        .timer_sel = LED_RING_LEDC_TIMER,
        .duty = 0,
        .hpoint = 0,
    };
    err = ledc_channel_config(&channel_cfg);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "ledc_channel_config failed: %d", err);
        return err;
    }

    stable_count = 0;
    ESP_LOGI(TAG, "capture_gate initialized (LED ring on GPIO%d, target distance %.0fmm +/-%.0fmm)",
              LED_RING_GPIO, DISTANCE_TARGET_MM, DISTANCE_TOLERANCE_MM);
    return ESP_OK;
}

esp_err_t capture_gate_update(capture_gate_status_t *out) {
    float distance_mm = 0.0f, lux = 0.0f;
    esp_err_t d_err = sensor_hal_read_distance_mm(&distance_mm);
    esp_err_t l_err = sensor_hal_read_lux(&lux);
    if (d_err != ESP_OK || l_err != ESP_OK) {
        ESP_LOGW(TAG, "sensor read failed (distance=%d, lux=%d)", d_err, l_err);
        return ESP_FAIL;
    }

    bool distance_ok = fabsf(distance_mm - DISTANCE_TARGET_MM) <= DISTANCE_TOLERANCE_MM;

    /* Simple open-loop ambient-compensation heuristic: more LED duty when the room reads
     * darker than LUX_REFERENCE, less when brighter. This is NOT a closed photometric
     * control loop -- it does not distinguish "light from the LED ring itself" from
     * "ambient light", since that would need the lux sensor positioned to read the
     * combined illuminance at the skin with a way to sample ambient-only (e.g. duty-cycle
     * the LED and measure between pulses), not implemented here. Stated as a real,
     * deliberate simplification, not something already solved. */
    float duty_f = LED_BASE_DUTY_PERCENT - LED_GAIN_PERCENT_PER_LUX * (lux - LUX_REFERENCE);
    uint8_t duty_percent = clamp_u8(duty_f, LED_MIN_DUTY_PERCENT, LED_MAX_DUTY_PERCENT);
    bool illumination_ok = (duty_percent > LED_MIN_DUTY_PERCENT) && (duty_percent < LED_MAX_DUTY_PERCENT);

    uint32_t duty_raw = (uint32_t)((duty_percent / 100.0f) * ((1 << LED_RING_DUTY_RES) - 1));
    ledc_set_duty(LED_RING_LEDC_MODE, LED_RING_LEDC_CHANNEL, duty_raw);
    ledc_update_duty(LED_RING_LEDC_MODE, LED_RING_LEDC_CHANNEL);

    if (distance_ok && illumination_ok) {
        if (stable_count < DEBOUNCE_UPDATES) stable_count++;
    } else {
        stable_count = 0;
    }

    out->distance_mm = distance_mm;
    out->distance_target_mm = DISTANCE_TARGET_MM;
    out->distance_tolerance_mm = DISTANCE_TOLERANCE_MM;
    out->distance_ok = distance_ok;
    out->lux = lux;
    out->lux_target = LUX_REFERENCE;
    out->led_duty_percent = duty_percent;
    out->illumination_ok = illumination_ok;
    out->ready = stable_count >= DEBOUNCE_UPDATES;

    return ESP_OK;
}
