import json
import sys
import time

import serial


PORT = "COM8"
BAUDRATE = 115200
SERIAL_TIMEOUT_SECONDS = 1
TEST_DURATION_SECONDS = 20
SERIAL_PREFIX = "SENSOR_DATA:"


def parse_sensor_line(raw_line):
    if not raw_line.startswith(SERIAL_PREFIX):
        return None

    payload = raw_line[len(SERIAL_PREFIX) :].strip()
    if not payload:
        return None

    try:
        packet = json.loads(payload)
    except json.JSONDecodeError:
        return None

    if not isinstance(packet, dict):
        return None
    if packet.get("type") != "sensor_frame":
        return None

    return packet


def main():
    print(f"Opening {PORT} at {BAUDRATE} baud for {TEST_DURATION_SECONDS} seconds")
    print(f"Expected output includes PICO_READY and {SERIAL_PREFIX} packets.")

    try:
        with serial.Serial(PORT, BAUDRATE, timeout=SERIAL_TIMEOUT_SECONDS) as ser:
            time.sleep(2)
            deadline = time.time() + TEST_DURATION_SECONDS
            saw_any_output = False
            saw_sensor_packet = False

            while time.time() < deadline:
                raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not raw_line:
                    continue

                saw_any_output = True
                print(raw_line)

                if parse_sensor_line(raw_line) is not None:
                    saw_sensor_packet = True

            if not saw_any_output:
                print("No serial output received from the Pico.", file=sys.stderr)
                print(
                    "Make sure main.py is saved on the Pico itself and not just run from the editor.",
                    file=sys.stderr,
                )
                print(
                    "After closing any serial monitor, run: py deploy_pico.py",
                    file=sys.stderr,
                )
                sys.exit(1)

            if not saw_sensor_packet:
                print(
                    "Serial output was received, but no valid SENSOR_DATA packets were found.",
                    file=sys.stderr,
                )
                sys.exit(1)

            print("Serial test passed.")
    except serial.SerialException as exc:
        print(f"Could not open {PORT}: {exc}", file=sys.stderr)
        print("Close any VS Code, MicroPico, or Thonny serial monitor and try again.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
