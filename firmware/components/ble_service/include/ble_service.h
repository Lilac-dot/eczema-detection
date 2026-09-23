#pragma once
#include <stdbool.h>
#include "esp_err.h"
#include "sensor_hal.h"

/*
 * BLE peripheral (NimBLE) that streams raw sensor samples to the phone -- no on-device
 * feature extraction or inference. hardware_architecture_v3_2026-09-18.md puts every
 * trained model on the phone; this firmware's only job is getting clean sensor readings
 * off the wearable reliably.
 *
 * One custom GATT service per device role, each with one notify characteristic per
 * sensor. Only the skin patch's service (moisture/temp/EDA) is implemented so far --
 * building the wristband role currently logs a warning and advertises no service.
 */

esp_err_t ble_service_init(void);

/* Pushes a new sample to whichever connected central (the phone) has subscribed to that
 * characteristic. Safe to call even with no active connection -- returns quietly. */
esp_err_t ble_service_notify_moisture(float value);
esp_err_t ble_service_notify_temp(float value);
esp_err_t ble_service_notify_eda(float value);

/* Imaging-gate role only (DEVICE_ROLE_IMAGING_GATE) -- see capture_gate.h. Declared
 * unconditionally like the skin-patch notifiers above so main.c doesn't need role-specific
 * #ifdefs around the call sites; only meaningful when that role's GATT service is compiled
 * in, but safe to call (no-op-ish, returns ESP_OK) otherwise since no central will have
 * subscribed to a characteristic that doesn't exist. */
esp_err_t ble_service_notify_capture_ready(bool ready);
esp_err_t ble_service_notify_capture_distance(float distance_mm);
esp_err_t ble_service_notify_capture_lux(float lux);
