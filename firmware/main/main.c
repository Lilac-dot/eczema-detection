#include "esp_err.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "nvs_flash.h"

#include "sensor_hal.h"
#include "ble_service.h"
#include "capture_gate.h"

static const char *TAG = "main";

#if CONFIG_DEVICE_ROLE_SKIN_PATCH

static void eda_timer_cb(void *arg) {
    (void)arg;
    float value;
    if (sensor_hal_read_eda(&value) == ESP_OK) {
        ble_service_notify_eda(value);
    }
}

static void temp_timer_cb(void *arg) {
    (void)arg;
    float value;
    if (sensor_hal_read_temp(&value) == ESP_OK) {
        ble_service_notify_temp(value);
    }
}

static void moisture_timer_cb(void *arg) {
    (void)arg;
    float value;
    if (sensor_hal_read_moisture(&value) == ESP_OK) {
        ble_service_notify_moisture(value);
    }
}

static void start_skin_patch_sampling(void) {
    /* One esp_timer per sensor, each firing at that sensor's own rate
     * (sensor_hal.h's SENSOR_HAL_*_HZ) -- EDA and TEMP happen to share a rate here, but
     * kept as separate timers rather than one combined callback so a future sensor with
     * a different rate (moisture's real rate is still unknown, Section 7.1 of the report)
     * doesn't require restructuring this. */
    const esp_timer_create_args_t eda_args = {.callback = eda_timer_cb, .name = "eda"};
    const esp_timer_create_args_t temp_args = {.callback = temp_timer_cb, .name = "temp"};
    const esp_timer_create_args_t moisture_args = {.callback = moisture_timer_cb, .name = "moisture"};

    esp_timer_handle_t eda_timer, temp_timer, moisture_timer;
    ESP_ERROR_CHECK(esp_timer_create(&eda_args, &eda_timer));
    ESP_ERROR_CHECK(esp_timer_create(&temp_args, &temp_timer));
    ESP_ERROR_CHECK(esp_timer_create(&moisture_args, &moisture_timer));

    ESP_ERROR_CHECK(esp_timer_start_periodic(eda_timer, 1000000 / SENSOR_HAL_EDA_HZ));
    ESP_ERROR_CHECK(esp_timer_start_periodic(temp_timer, 1000000 / SENSOR_HAL_TEMP_HZ));
    ESP_ERROR_CHECK(esp_timer_start_periodic(moisture_timer, 1000000 / SENSOR_HAL_MOISTURE_HZ));

    ESP_LOGI(TAG, "skin patch sampling started: EDA %dHz, TEMP %dHz, MOISTURE %dHz",
             SENSOR_HAL_EDA_HZ, SENSOR_HAL_TEMP_HZ, SENSOR_HAL_MOISTURE_HZ);
}

#endif  /* CONFIG_DEVICE_ROLE_SKIN_PATCH */

#if CONFIG_DEVICE_ROLE_IMAGING_GATE

static void capture_gate_timer_cb(void *arg) {
    (void)arg;
    capture_gate_status_t status;
    if (capture_gate_update(&status) != ESP_OK) {
        return;
    }
    ble_service_notify_capture_distance(status.distance_mm);
    ble_service_notify_capture_lux(status.lux);
    ble_service_notify_capture_ready(status.ready);
    ESP_LOGD(TAG, "distance=%.1fmm (ok=%d) lux=%.0f led=%u%% ready=%d",
              status.distance_mm, status.distance_ok, status.lux,
              status.led_duty_percent, status.ready);
}

static void start_imaging_gate_sampling(void) {
    ESP_ERROR_CHECK(capture_gate_init());
    const esp_timer_create_args_t timer_args = {
        .callback = capture_gate_timer_cb, .name = "capture_gate",
    };
    esp_timer_handle_t timer;
    ESP_ERROR_CHECK(esp_timer_create(&timer_args, &timer));
    ESP_ERROR_CHECK(esp_timer_start_periodic(timer, 1000000 / SENSOR_HAL_DISTANCE_HZ));
    ESP_LOGI(TAG, "imaging capture-gate sampling started at %dHz", SENSOR_HAL_DISTANCE_HZ);
}

#endif  /* CONFIG_DEVICE_ROLE_IMAGING_GATE */

void app_main(void) {
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_ERROR_CHECK(sensor_hal_init());
    ESP_ERROR_CHECK(ble_service_init());

#if CONFIG_DEVICE_ROLE_SKIN_PATCH
    start_skin_patch_sampling();
#elif CONFIG_DEVICE_ROLE_IMAGING_GATE
    start_imaging_gate_sampling();
#else
    ESP_LOGW(TAG, "DEVICE_ROLE_WRISTBAND is selected but not implemented yet -- only the "
                  "skin patch and imaging-gate roles are wired up so far. Sensors and BLE "
                  "are initialized, but nothing samples or notifies. Switch to a wired-up "
                  "role (idf.py menuconfig) or finish wiring up the wristband path in "
                  "main.c.");
#endif
}
