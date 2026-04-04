from machine import Pin, UART
import time


UART_BAUDRATE = 256000
REPORT_MS = 1000
READ_DELAY_MS = 50
HEX_PREVIEW_BYTES = 32

SENSORS = (
    {
        "name": "LD2410C_1",
        "uart_id": 0,
        "tx_pin": 0,
        "rx_pin": 1,
        "out_pin": 2,
    },
    {
        "name": "LD2410C_2",
        "uart_id": 1,
        "tx_pin": 8,
        "rx_pin": 9,
        "out_pin": 3,
    },
)

# Change this to 0 or 1 depending on which sensor you want to test.
ACTIVE_SENSOR_INDEX = 0


def hex_preview(data):
    preview = data[:HEX_PREVIEW_BYTES]
    return " ".join("{:02X}".format(byte) for byte in preview)


def main():
    sensor = SENSORS[ACTIVE_SENSOR_INDEX]
    uart = UART(
        sensor["uart_id"],
        baudrate=UART_BAUDRATE,
        tx=Pin(sensor["tx_pin"]),
        rx=Pin(sensor["rx_pin"]),
        bits=8,
        parity=None,
        stop=1,
        timeout=20,
    )
    out_pin = Pin(sensor["out_pin"], Pin.IN)

    total_bytes = 0
    last_out = out_pin.value()
    last_report = time.ticks_ms()
    last_data = b""

    print("PICO_READY")
    print(
        "ACTIVE_SENSOR:{}:UART_ID={},BAUD={},PICO_TX=GP{},PICO_RX=GP{},OUT=GP{}".format(
            sensor["name"],
            sensor["uart_id"],
            UART_BAUDRATE,
            sensor["tx_pin"],
            sensor["rx_pin"],
            sensor["out_pin"],
        )
    )
    print(
        "WIRING:VCC=VBUS_OR_3.3V_AS_TESTED,GND=GND,SENSOR_TX->PICO_RX_GP{},SENSOR_RX->PICO_TX_GP{},SENSOR_OUT->GP{}".format(
            sensor["rx_pin"],
            sensor["tx_pin"],
            sensor["out_pin"],
        )
    )
    print("RAW_UART_DEBUG:FRAME_PARSING=OFF")

    while True:
        out_now = out_pin.value()
        if out_now != last_out:
            print("OUT_CHANGE:{}->{}".format(last_out, out_now))
            last_out = out_now

        data = uart.read()
        if data:
            total_bytes += len(data)
            last_data = data
            print(
                "RX_EVENT:BYTES={},TOTAL_BYTES={},HEX={}".format(
                    len(data),
                    total_bytes,
                    hex_preview(data),
                )
            )

        now = time.ticks_ms()
        if time.ticks_diff(now, last_report) >= REPORT_MS:
            if last_data:
                print(
                    "STATUS:OUT={},TOTAL_BYTES={},LAST_CHUNK_BYTES={},LAST_HEX={}".format(
                        out_now,
                        total_bytes,
                        len(last_data),
                        hex_preview(last_data),
                    )
                )
            else:
                print(
                    "STATUS:OUT={},TOTAL_BYTES=0,LAST_CHUNK_BYTES=0,NOTE=NO_UART_DATA".format(
                        out_now
                    )
                )
            last_report = now

        time.sleep_ms(READ_DELAY_MS)


main()
