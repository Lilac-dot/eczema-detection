if(CONFIG_SENSOR_BACKEND_SIMULATED)
    set(srcs "sensor_hal_simulated.c")
else()
    set(srcs "sensor_hal_real.c")
endif()

idf_component_register(
    SRCS ${srcs}
    INCLUDE_DIRS "include"
    REQUIRES esp_timer
)
