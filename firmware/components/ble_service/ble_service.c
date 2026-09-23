/*
 * Skin-patch BLE peripheral: one custom GATT service, three notify characteristics
 * (moisture, temperature, EDA). Advertises continuously, accepts one central (the phone),
 * pushes a notification every time main.c's periodic sampling calls
 * ble_service_notify_*(). No pairing/bonding -- this is sensor telemetry, not a security-
 * sensitive link, and adding bonding would need NVS-backed store config this project
 * doesn't need yet.
 *
 * Structurally follows ESP-IDF's own NimBLE "bleprph" example (advertise -> on connect,
 * stop advertising -> on disconnect, advertise again), trimmed to what a single-service
 * sensor peripheral actually needs.
 */
#include <string.h>
#include "esp_err.h"
#include "esp_log.h"
#include "nimble/nimble_port.h"
#include "nimble/nimble_port_freertos.h"
#include "host/ble_hs.h"
#include "services/gap/ble_svc_gap.h"
#include "services/gatt/ble_svc_gatt.h"

#include "ble_service.h"

static const char *TAG = "ble_service";
#if CONFIG_DEVICE_ROLE_IMAGING_GATE
static const char *DEVICE_NAME = "AD-ImagingGate";
#else
static const char *DEVICE_NAME = "AD-SkinPatch";
#endif

/* Custom 128-bit UUIDs -- randomly generated for this project, not from any standard BLE
 * SIG service. Each device role gets its own service UUID (different physical device,
 * different GATT service), not more characteristics crammed under one shared service.
 * Only the active role's UUIDs are compiled in -- with -Werror=unused-const-variable
 * enabled for this project, leaving both sets unconditional fails the build for whichever
 * role isn't selected. */
#if CONFIG_DEVICE_ROLE_IMAGING_GATE
static const ble_uuid128_t SVC_IMAGING_GATE_UUID =
    BLE_UUID128_INIT(0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                      0x09, 0x0a, 0x0b, 0x0c, 0xb0, 0x00, 0x00, 0x00);
static const ble_uuid128_t CHR_CAPTURE_READY_UUID =
    BLE_UUID128_INIT(0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                      0x09, 0x0a, 0x0b, 0x0c, 0xb0, 0x01, 0x00, 0x00);
static const ble_uuid128_t CHR_CAPTURE_DISTANCE_UUID =
    BLE_UUID128_INIT(0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                      0x09, 0x0a, 0x0b, 0x0c, 0xb0, 0x02, 0x00, 0x00);
static const ble_uuid128_t CHR_CAPTURE_LUX_UUID =
    BLE_UUID128_INIT(0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                      0x09, 0x0a, 0x0b, 0x0c, 0xb0, 0x03, 0x00, 0x00);
#else
static const ble_uuid128_t SVC_SKIN_PATCH_UUID =
    BLE_UUID128_INIT(0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                      0x09, 0x0a, 0x0b, 0x0c, 0xa0, 0x00, 0x00, 0x00);
static const ble_uuid128_t CHR_MOISTURE_UUID =
    BLE_UUID128_INIT(0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                      0x09, 0x0a, 0x0b, 0x0c, 0xa0, 0x01, 0x00, 0x00);
static const ble_uuid128_t CHR_TEMP_UUID =
    BLE_UUID128_INIT(0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                      0x09, 0x0a, 0x0b, 0x0c, 0xa0, 0x02, 0x00, 0x00);
static const ble_uuid128_t CHR_EDA_UUID =
    BLE_UUID128_INIT(0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                      0x09, 0x0a, 0x0b, 0x0c, 0xa0, 0x03, 0x00, 0x00);
#endif

static uint16_t moisture_val_handle;
static uint16_t temp_val_handle;
static uint16_t eda_val_handle;
static uint16_t capture_ready_val_handle;
static uint16_t capture_distance_val_handle;
static uint16_t capture_lux_val_handle;

/* Last sample of each channel, so a read (as opposed to a notify/subscribe) returns
 * something sensible instead of garbage. */
static float last_moisture = 0.0f;
static float last_temp = 0.0f;
static float last_eda = 0.0f;
static float last_capture_ready = 0.0f;   /* stored as 0.0f/1.0f -- reuses the same
                                            * float-based notify() helper as every other
                                            * channel rather than a separate bool path */
static float last_capture_distance = 0.0f;
static float last_capture_lux = 0.0f;

static uint8_t own_addr_type;
static uint16_t conn_handle = BLE_HS_CONN_HANDLE_NONE;

static int gatt_access_cb(uint16_t conn, uint16_t attr_handle,
                           struct ble_gatt_access_ctxt *ctxt, void *arg) {
    (void)conn;
    (void)attr_handle;
    if (ctxt->op != BLE_GATT_ACCESS_OP_READ_CHR) {
        return BLE_ATT_ERR_UNLIKELY;  /* write-to-us isn't a supported operation here */
    }
    float *value = (float *)arg;
    int rc = os_mbuf_append(ctxt->om, value, sizeof(*value));
    return rc == 0 ? 0 : BLE_ATT_ERR_INSUFFICIENT_RES;
}

#if CONFIG_DEVICE_ROLE_IMAGING_GATE
static const struct ble_gatt_svc_def gatt_svcs[] = {
    {
        .type = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid = &SVC_IMAGING_GATE_UUID.u,
        .characteristics = (struct ble_gatt_chr_def[]){
            {
                .uuid = &CHR_CAPTURE_READY_UUID.u,
                .access_cb = gatt_access_cb,
                .arg = &last_capture_ready,
                .val_handle = &capture_ready_val_handle,
                .flags = BLE_GATT_CHR_F_READ | BLE_GATT_CHR_F_NOTIFY,
            },
            {
                .uuid = &CHR_CAPTURE_DISTANCE_UUID.u,
                .access_cb = gatt_access_cb,
                .arg = &last_capture_distance,
                .val_handle = &capture_distance_val_handle,
                .flags = BLE_GATT_CHR_F_READ | BLE_GATT_CHR_F_NOTIFY,
            },
            {
                .uuid = &CHR_CAPTURE_LUX_UUID.u,
                .access_cb = gatt_access_cb,
                .arg = &last_capture_lux,
                .val_handle = &capture_lux_val_handle,
                .flags = BLE_GATT_CHR_F_READ | BLE_GATT_CHR_F_NOTIFY,
            },
            {0},  /* characteristics terminator */
        },
    },
    {0},  /* services terminator */
};
#else
static const struct ble_gatt_svc_def gatt_svcs[] = {
    {
        .type = BLE_GATT_SVC_TYPE_PRIMARY,
        .uuid = &SVC_SKIN_PATCH_UUID.u,
        .characteristics = (struct ble_gatt_chr_def[]){
            {
                .uuid = &CHR_MOISTURE_UUID.u,
                .access_cb = gatt_access_cb,
                .arg = &last_moisture,
                .val_handle = &moisture_val_handle,
                .flags = BLE_GATT_CHR_F_READ | BLE_GATT_CHR_F_NOTIFY,
            },
            {
                .uuid = &CHR_TEMP_UUID.u,
                .access_cb = gatt_access_cb,
                .arg = &last_temp,
                .val_handle = &temp_val_handle,
                .flags = BLE_GATT_CHR_F_READ | BLE_GATT_CHR_F_NOTIFY,
            },
            {
                .uuid = &CHR_EDA_UUID.u,
                .access_cb = gatt_access_cb,
                .arg = &last_eda,
                .val_handle = &eda_val_handle,
                .flags = BLE_GATT_CHR_F_READ | BLE_GATT_CHR_F_NOTIFY,
            },
            {0},  /* characteristics terminator */
        },
    },
    {0},  /* services terminator */
};
#endif

static void start_advertising(void) {
    struct ble_hs_adv_fields fields;
    struct ble_gap_adv_params adv_params;
    int rc;

    memset(&fields, 0, sizeof(fields));
    fields.flags = BLE_HS_ADV_F_DISC_GEN | BLE_HS_ADV_F_BREDR_UNSUP;
    fields.name = (uint8_t *)DEVICE_NAME;
    fields.name_len = strlen(DEVICE_NAME);
    fields.name_is_complete = 1;
    fields.tx_pwr_lvl_is_present = 1;
    fields.tx_pwr_lvl = BLE_HS_ADV_TX_PWR_LVL_AUTO;

    rc = ble_gap_adv_set_fields(&fields);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_gap_adv_set_fields failed: %d", rc);
        return;
    }

    memset(&adv_params, 0, sizeof(adv_params));
    adv_params.conn_mode = BLE_GAP_CONN_MODE_UND;
    adv_params.disc_mode = BLE_GAP_DISC_MODE_GEN;

    extern int gap_event_cb(struct ble_gap_event *event, void *arg);
    rc = ble_gap_adv_start(own_addr_type, NULL, BLE_HS_FOREVER, &adv_params,
                            gap_event_cb, NULL);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_gap_adv_start failed: %d", rc);
    }
}

int gap_event_cb(struct ble_gap_event *event, void *arg) {
    (void)arg;
    switch (event->type) {
        case BLE_GAP_EVENT_CONNECT:
            if (event->connect.status == 0) {
                ESP_LOGI(TAG, "phone connected, handle=%d", event->connect.conn_handle);
                conn_handle = event->connect.conn_handle;
            } else {
                ESP_LOGW(TAG, "connect attempt failed, status=%d -- re-advertising",
                         event->connect.status);
                start_advertising();
            }
            return 0;
        case BLE_GAP_EVENT_DISCONNECT:
            ESP_LOGI(TAG, "phone disconnected, reason=%d -- re-advertising",
                      event->disconnect.reason);
            conn_handle = BLE_HS_CONN_HANDLE_NONE;
            start_advertising();
            return 0;
        case BLE_GAP_EVENT_SUBSCRIBE:
            ESP_LOGI(TAG, "subscribe event: attr_handle=%d notify=%d",
                      event->subscribe.attr_handle, event->subscribe.cur_notify);
            return 0;
        case BLE_GAP_EVENT_ADV_COMPLETE:
            start_advertising();
            return 0;
        default:
            return 0;
    }
}

static void on_sync(void) {
    int rc = ble_hs_id_infer_auto(0, &own_addr_type);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_hs_id_infer_auto failed: %d", rc);
        return;
    }
    start_advertising();
}

static void on_reset(int reason) {
    ESP_LOGW(TAG, "NimBLE host reset, reason=%d", reason);
}

static void host_task(void *param) {
    (void)param;
    nimble_port_run();
    nimble_port_freertos_deinit();
}

esp_err_t ble_service_init(void) {
    esp_err_t ret = nimble_port_init();
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "nimble_port_init failed: %d", ret);
        return ret;
    }

    ble_hs_cfg.sync_cb = on_sync;
    ble_hs_cfg.reset_cb = on_reset;

    ble_svc_gap_init();
    ble_svc_gatt_init();

    int rc = ble_gatts_count_cfg(gatt_svcs);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_gatts_count_cfg failed: %d", rc);
        return ESP_FAIL;
    }
    rc = ble_gatts_add_svcs(gatt_svcs);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_gatts_add_svcs failed: %d", rc);
        return ESP_FAIL;
    }

    rc = ble_svc_gap_device_name_set(DEVICE_NAME);
    if (rc != 0) {
        ESP_LOGE(TAG, "ble_svc_gap_device_name_set failed: %d", rc);
        return ESP_FAIL;
    }

    nimble_port_freertos_init(host_task);
    ESP_LOGI(TAG, "BLE service up, advertising as \"%s\"", DEVICE_NAME);
    return ESP_OK;
}

static esp_err_t notify(uint16_t val_handle, float *cache, float value) {
    *cache = value;
    if (conn_handle == BLE_HS_CONN_HANDLE_NONE) {
        return ESP_OK;  /* nobody listening -- not an error, just nothing to do */
    }
    struct os_mbuf *om = ble_hs_mbuf_from_flat(&value, sizeof(value));
    if (om == NULL) {
        return ESP_ERR_NO_MEM;
    }
    int rc = ble_gatts_notify_custom(conn_handle, val_handle, om);
    return rc == 0 ? ESP_OK : ESP_FAIL;
}

esp_err_t ble_service_notify_moisture(float value) {
    return notify(moisture_val_handle, &last_moisture, value);
}

esp_err_t ble_service_notify_temp(float value) {
    return notify(temp_val_handle, &last_temp, value);
}

esp_err_t ble_service_notify_eda(float value) {
    return notify(eda_val_handle, &last_eda, value);
}

esp_err_t ble_service_notify_capture_ready(bool ready) {
    return notify(capture_ready_val_handle, &last_capture_ready, ready ? 1.0f : 0.0f);
}

esp_err_t ble_service_notify_capture_distance(float distance_mm) {
    return notify(capture_distance_val_handle, &last_capture_distance, distance_mm);
}

esp_err_t ble_service_notify_capture_lux(float lux) {
    return notify(capture_lux_val_handle, &last_capture_lux, lux);
}
