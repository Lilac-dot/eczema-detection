idf_component_register(
    SRCS "main.c"
    INCLUDE_DIRS "."
    REQUIRES sensor_hal ble_service capture_gate nvs_flash esp_timer
)
