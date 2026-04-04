import json
import sys
import time

import serial


PORT = "COM8"
BAUDRATE = 115200
SERIAL_TIMEOUT_SECONDS = 1
TEST_DURATION_SECONDS = 20
SERIAL_PREFIX = "SENSOR_DATA:"
REQUIRED_PACKET_KEYS = (
    "bme688",
    "sen0628",
    "ld2410c_front",
    "ld2410c_back",
)


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


def packet_has_expected_shape(packet):
    return all(key in packet for key in REQUIRED_PACKET_KEYS)


def summarize_ld_sensor(name, ld_data):
    if not isinstance(ld_data, dict):
        return f"{name}=MISSING"

    status = ld_data.get("status")
    if status == "WAITING_FOR_VALID_FRAME":
        return "{}=OUT:{},STATUS:WAITING".format(
            name,
            ld_data.get("out"),
        )

    return "{}=OUT:{},STATE:{},DIST:{}cm".format(
        name,
        ld_data.get("out"),
        ld_data.get("target_state"),
        ld_data.get("detection_distance_cm"),
    )


def summarize_packet(packet):
    bme = packet.get("bme688") or {}
    sen = packet.get("sen0628") or {}
    front_sensor = packet.get("ld2410c_front") or {}
    back_sensor = packet.get("ld2410c_back") or {}

    return "SEQ={seq} | TEMP={temp}C | HUM={humidity}% | SEN_CENTER={center}mm | {front_sensor} | {back_sensor}".format(
        seq=packet.get("seq"),
        temp=bme.get("temperature_c"),
        humidity=bme.get("humidity_pct"),
        center=sen.get("center_mm"),
        front_sensor=summarize_ld_sensor("FRONT", front_sensor),
        back_sensor=summarize_ld_sensor("BACK", back_sensor),
    )


def main():
    print(f"Opening {PORT} at {BAUDRATE} baud for {TEST_DURATION_SECONDS} seconds")
    print(f"Expected output includes PICO_READY and {SERIAL_PREFIX} packets.")
    print(
        "Expected packet keys: "
        + ", ".join(REQUIRED_PACKET_KEYS)
        + " (light sensor removed)"
    )

    try:
        with serial.Serial(PORT, BAUDRATE, timeout=SERIAL_TIMEOUT_SECONDS) as ser:
            time.sleep(2)
            deadline = time.time() + TEST_DURATION_SECONDS
            saw_any_output = False
            saw_sensor_packet = False
            saw_expected_shape = False

            while time.time() < deadline:
                raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not raw_line:
                    continue

                saw_any_output = True
                print(raw_line)

                packet = parse_sensor_line(raw_line)
                if packet is not None:
                    saw_sensor_packet = True
                    if packet_has_expected_shape(packet):
                        saw_expected_shape = True
                        print("PARSED: " + summarize_packet(packet))

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

            if not saw_expected_shape:
                print(
                    "SENSOR_DATA packets were received, but they do not match the new two-LD2410C payload shape.",
                    file=sys.stderr,
                )
                print(
                    "Deploy the updated main.py so packets include ld2410c_front and ld2410c_back.",
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
