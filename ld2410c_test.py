from machine import Pin, UART
import time

UART_BAUDRATE = 256000
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

FRAME_HEADER = b"\xF4\xF3\xF2\xF1"
FRAME_END = b"\xF8\xF7\xF6\xF5"
MIN_TARGET_PAYLOAD_LEN = 13
MAX_BUFFER_SIZE = 512
REPORT_MS = 2000

TARGET_STATES = {
    0x00: "NO_TARGET",
    0x01: "MOVING_TARGET",
    0x02: "STATIONARY_TARGET",
    0x03: "MOVING_AND_STATIONARY",
}


class LD2410CReader:
    def __init__(self, uart):
        self.uart = uart
        self.buffer = bytearray()
        self.total_uart_bytes = 0

    def poll_uart(self):
        data = self.uart.read()
        if not data:
            return 0

        self.buffer.extend(data)
        self.total_uart_bytes += len(data)

        if len(self.buffer) > MAX_BUFFER_SIZE:
            self.buffer = self.buffer[-MAX_BUFFER_SIZE:]

        return len(data)

    def _extract_frame(self):
        while True:
            header_index = self.buffer.find(FRAME_HEADER)
            if header_index < 0:
                if len(self.buffer) > 3:
                    self.buffer = self.buffer[-3:]
                return None

            if header_index > 0:
                self.buffer = self.buffer[header_index:]

            if len(self.buffer) < 6:
                return None

            payload_length = self.buffer[4] | (self.buffer[5] << 8)
            frame_length = 4 + 2 + payload_length + 4

            if len(self.buffer) < frame_length:
                return None

            frame = bytes(self.buffer[:frame_length])
            self.buffer = self.buffer[frame_length:]

            if frame[-4:] != FRAME_END:
                continue

            return frame

    def read_frames(self):
        frames = []

        while True:
            frame = self._extract_frame()
            if frame is None:
                return frames
            frames.append(frame)

    def parse_target_frame(self, frame):
        payload_length = frame[4] | (frame[5] << 8)
        payload = frame[6 : 6 + payload_length]

        if payload_length < MIN_TARGET_PAYLOAD_LEN:
            return None

        if payload[1] != 0xAA:
            return None

        if payload[-2:] != b"\x55\x00":
            return None

        parsed = {
            "frame_type": payload[0],
            "payload_length": payload_length,
            "target_state_raw": payload[2],
            "moving_distance_cm": payload[3] | (payload[4] << 8),
            "moving_energy": payload[5],
            "stationary_distance_cm": payload[6] | (payload[7] << 8),
            "stationary_energy": payload[8],
            "detection_distance_cm": payload[9] | (payload[10] << 8),
        }

        if payload_length > MIN_TARGET_PAYLOAD_LEN:
            parsed["extra_data_len"] = payload_length - MIN_TARGET_PAYLOAD_LEN

        return parsed


def main():
    sensors = []

    for config in SENSORS:
        uart = UART(
            config["uart_id"],
            baudrate=UART_BAUDRATE,
            tx=Pin(config["tx_pin"]),
            rx=Pin(config["rx_pin"]),
            bits=8,
            parity=None,
            stop=1,
            timeout=20,
        )
        out_pin = Pin(config["out_pin"], Pin.IN)
        sensors.append(
            {
                "name": config["name"],
                "uart_id": config["uart_id"],
                "tx_pin": config["tx_pin"],
                "rx_pin": config["rx_pin"],
                "out_pin": config["out_pin"],
                "out": out_pin,
                "reader": LD2410CReader(uart),
                "last_out_state": out_pin.value(),
                "latest_reading": None,
            }
        )

    last_report = time.ticks_ms()

    print("PICO_READY")
    for sensor in sensors:
        print(
            "{}:UART_ID={},BAUD={},PICO_TX=GP{},PICO_RX=GP{},OUT=GP{}".format(
                sensor["name"],
                sensor["uart_id"],
                UART_BAUDRATE,
                sensor["tx_pin"],
                sensor["rx_pin"],
                sensor["out_pin"],
            )
        )
        print(
            "{}_WIRING:VCC=3.3V_OUT,GND=GND,SENSOR_TX->PICO_RX_GP{},SENSOR_RX->PICO_TX_GP{},SENSOR_OUT->GP{}".format(
                sensor["name"],
                sensor["rx_pin"],
                sensor["tx_pin"],
                sensor["out_pin"],
            )
        )
    print("CHECKLIST:COMMON_GROUND=YES,POWER=3.3V_OUT,UART_WIRING=CROSSED,SENSOR_TX_TO_PICO_RX=YES,SENSOR_RX_TO_PICO_TX=YES")
    print("WAITING_FOR_LD2410C_FRAMES")

    while True:
        for sensor in sensors:
            out_state = sensor["out"].value()
            if out_state != sensor["last_out_state"]:
                print("{}:OUT={}".format(sensor["name"], out_state))
                sensor["last_out_state"] = out_state

            bytes_read = sensor["reader"].poll_uart()
            if bytes_read:
                frames = sensor["reader"].read_frames()
                for frame in frames:
                    reading = sensor["reader"].parse_target_frame(frame)
                    if reading is not None:
                        sensor["latest_reading"] = reading

        now = time.ticks_ms()
        if time.ticks_diff(now, last_report) >= REPORT_MS:
            for sensor in sensors:
                out_state = sensor["out"].value()
                latest_reading = sensor["latest_reading"]

                if latest_reading is None:
                    print(
                        "{}:OUT={},UART_BYTES_TOTAL={},BUFFER_LEN={},STATUS=WAITING_FOR_VALID_FRAME".format(
                            sensor["name"],
                            out_state,
                            sensor["reader"].total_uart_bytes,
                            len(sensor["reader"].buffer),
                        )
                    )
                else:
                    target_state = TARGET_STATES.get(
                        latest_reading["target_state_raw"],
                        "UNKNOWN({})".format(latest_reading["target_state_raw"]),
                    )
                    print(
                        "{}:OUT={},STATE={},MOVING_DIST_CM={},MOVING_ENERGY={},STATIONARY_DIST_CM={},STATIONARY_ENERGY={},DETECTION_DIST_CM={},FRAME_TYPE=0x{:02X},PAYLOAD_LEN={}".format(
                            sensor["name"],
                            out_state,
                            target_state,
                            latest_reading["moving_distance_cm"],
                            latest_reading["moving_energy"],
                            latest_reading["stationary_distance_cm"],
                            latest_reading["stationary_energy"],
                            latest_reading["detection_distance_cm"],
                            latest_reading["frame_type"],
                            latest_reading["payload_length"],
                        )
                    )
            last_report = now

        time.sleep_ms(50)


main()
