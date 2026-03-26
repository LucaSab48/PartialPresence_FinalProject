from machine import Pin, UART
import time

UART_ID = 0
UART_BAUDRATE = 256000
UART_TX_PIN = 0
UART_RX_PIN = 1
OUT_PIN = 2

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
    uart = UART(
        UART_ID,
        baudrate=UART_BAUDRATE,
        tx=Pin(UART_TX_PIN),
        rx=Pin(UART_RX_PIN),
        bits=8,
        parity=None,
        stop=1,
        timeout=20,
    )
    out_pin = Pin(OUT_PIN, Pin.IN)
    reader = LD2410CReader(uart)

    last_out_state = out_pin.value()
    last_report = time.ticks_ms()
    latest_reading = None

    print("PICO_READY")
    print(
        "LD2410C_TEST:UART_ID={},BAUD={},TX=GP{},RX=GP{},OUT=GP{}".format(
            UART_ID, UART_BAUDRATE, UART_TX_PIN, UART_RX_PIN, OUT_PIN
        )
    )
    print("OTHER_SENSOR_NOTE:LEAVING_GP3_AND_GP4_UNTOUCHED")
    print("WAITING_FOR_LD2410C")

    while True:
        out_state = out_pin.value()
        if out_state != last_out_state:
            print("LD2410C:OUT={}".format(out_state))
            last_out_state = out_state

        bytes_read = reader.poll_uart()
        if bytes_read:
            frames = reader.read_frames()
            for frame in frames:
                reading = reader.parse_target_frame(frame)
                if reading is not None:
                    latest_reading = reading

        now = time.ticks_ms()
        if time.ticks_diff(now, last_report) >= REPORT_MS:
            if latest_reading is None:
                print(
                    "LD2410C:OUT={},UART_BYTES_TOTAL={},BUFFER_LEN={},STATUS=WAITING_FOR_VALID_FRAME".format(
                        out_state,
                        reader.total_uart_bytes,
                        len(reader.buffer),
                    )
                )
            else:
                target_state = TARGET_STATES.get(
                    latest_reading["target_state_raw"],
                    "UNKNOWN({})".format(latest_reading["target_state_raw"]),
                )
                print(
                    "LD2410C:OUT={},STATE={},MOVING_DIST_CM={},MOVING_ENERGY={},STATIONARY_DIST_CM={},STATIONARY_ENERGY={},DETECTION_DIST_CM={},FRAME_TYPE=0x{:02X},PAYLOAD_LEN={}".format(
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
