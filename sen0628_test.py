from machine import I2C, Pin
import time

I2C_ID = 1
SDA_PIN = 6
SCL_PIN = 7
I2C_FREQ = 100_000

SENSOR_ADDRESS = 0x33

CMD_SETMODE = 0x01
CMD_ALLDATA = 0x02

STATUS_SUCCESS = 0x53
STATUS_FAILED = 0x63

MATRIX_8X8 = 8
REPORT_DELAY_S = 2
SCAN_RETRY_MS = 10000
STARTUP_DELAY_MS = 1500
WARMUP_READS = 2
RESPONSE_TIMEOUT_MS = 8000
MODE_SWITCH_DELAY_MS = 5000
MOUNT_MODE = "ceiling_down"
CEILING_HEIGHT_MM = 2400
FLOOR_OCCUPANCY_DELTA_MM = 180


def safe_i2c_scan(i2c):
    if i2c is None:
        return []
    try:
        return i2c.scan()
    except Exception as exc:
        print("SEN0628:SCAN_ERROR,{}".format(exc))
        return []


def format_row(row_values):
    return " ".join("{:4d}".format(value) for value in row_values)


def valid_readings(values):
    return [value for value in values if 0 < value < 4000]


def average_distance_mm(readings):
    valid = valid_readings(readings)
    if not valid:
        return None
    return sum(valid) / len(valid)


def count_close_points(readings, threshold_mm):
    return len([value for value in readings if 0 < value < threshold_mm])


def describe_distance(distance_mm):
    if distance_mm is None:
        return "unknown distance"
    if distance_mm < 500:
        return "very close"
    if distance_mm < 1200:
        return "close"
    if distance_mm < 2200:
        return "mid-range"
    return "far"


def describe_zone_strength(close_points):
    if close_points >= 8:
        return "strong"
    if close_points >= 4:
        return "moderate"
    if close_points >= 1:
        return "weak"
    return "none"


def build_scene_summary(values):
    rows = [
        values[row_index * MATRIX_8X8 : (row_index + 1) * MATRIX_8X8]
        for row_index in range(MATRIX_8X8)
    ]
    left_values = []
    center_values = []
    right_values = []
    for row in rows:
        left_values.extend(row[0:2])
        center_values.extend(row[2:6])
        right_values.extend(row[6:8])

    zone_info = {
        "left": {
            "distance_mm": average_distance_mm(left_values),
            "close_points": count_close_points(left_values, 1600),
        },
        "center": {
            "distance_mm": average_distance_mm(center_values),
            "close_points": count_close_points(center_values, 1600),
        },
        "right": {
            "distance_mm": average_distance_mm(right_values),
            "close_points": count_close_points(right_values, 1600),
        },
    }

    strongest_zone = max(
        zone_info.items(),
        key=lambda item: (item[1]["close_points"], -(item[1]["distance_mm"] or 99999)),
    )[0]
    strongest_close_points = zone_info[strongest_zone]["close_points"]
    nearest_distance = min(valid_readings(values)) if valid_readings(values) else None

    if strongest_close_points == 0:
        occupancy_text = "mostly empty or no nearby subject"
    else:
        occupancy_text = "{} activity strongest on the {} side".format(
            describe_zone_strength(strongest_close_points),
            strongest_zone,
        )

    nearest_text = describe_distance(nearest_distance)
    return zone_info, occupancy_text, nearest_text, nearest_distance


def build_floor_summary(values):
    if MOUNT_MODE != "ceiling_down":
        return None

    rows = [
        values[row_index * MATRIX_8X8 : (row_index + 1) * MATRIX_8X8]
        for row_index in range(MATRIX_8X8)
    ]

    def occupied_count(readings):
        return len(
            [
                value
                for value in readings
                if 0 < value < 4000 and value <= (CEILING_HEIGHT_MM - FLOOR_OCCUPANCY_DELTA_MM)
            ]
        )

    left_values = []
    center_values = []
    right_values = []
    for row in rows:
        left_values.extend(row[0:2])
        center_values.extend(row[2:6])
        right_values.extend(row[6:8])

    all_heights = [
        CEILING_HEIGHT_MM - value
        for value in values
        if 0 < value < 4000 and value <= CEILING_HEIGHT_MM
    ]
    max_height = max(all_heights) if all_heights else None
    mean_height = (sum(all_heights) / len(all_heights)) if all_heights else None

    return {
        "left": occupied_count(left_values),
        "center": occupied_count(center_values),
        "right": occupied_count(right_values),
        "total": occupied_count(values),
        "mean_height_mm": mean_height,
        "max_height_mm": max_height,
    }


def recover_i2c_bus():
    sda = Pin(SDA_PIN, Pin.IN, Pin.PULL_UP)
    scl = Pin(SCL_PIN, Pin.OPEN_DRAIN, value=1)

    for _ in range(9):
        scl.value(0)
        time.sleep_us(10)
        scl.value(1)
        time.sleep_us(10)

    # Attempt a stop condition with SDA rising while SCL is high.
    sda = Pin(SDA_PIN, Pin.OPEN_DRAIN, value=0)
    time.sleep_us(10)
    scl.value(1)
    time.sleep_us(10)
    sda.value(1)
    time.sleep_us(10)

    Pin(SDA_PIN, Pin.IN, Pin.PULL_UP)
    Pin(SCL_PIN, Pin.IN, Pin.PULL_UP)

class SEN0628:
    def __init__(self, i2c, address):
        self.i2c = i2c
        self.address = address

    def _write_packet(self, cmd, args=b""):
        payload_len = len(args) + 1
        packet = bytes(
            (
                0x55,
                (payload_len >> 8) & 0xFF,
                payload_len & 0xFF,
                cmd,
            )
        ) + args
        self.i2c.writeto(self.address, packet)

    def _read_exact(self, length):
        return self.i2c.readfrom(self.address, length)

    def _read_response(self, expected_cmd, timeout_ms=RESPONSE_TIMEOUT_MS):
        deadline = time.ticks_add(time.ticks_ms(), timeout_ms)

        while time.ticks_diff(deadline, time.ticks_ms()) > 0:
            status = self._read_exact(1)[0]

            if status == 0xFF:
                time.sleep_ms(20)
                continue

            if status not in (STATUS_SUCCESS, STATUS_FAILED):
                time.sleep_ms(20)
                continue

            cmd = self._read_exact(1)[0]
            if cmd != expected_cmd:
                raise OSError("unexpected response cmd 0x{:02X}".format(cmd))

            length_bytes = self._read_exact(2)
            payload_len = length_bytes[0] | (length_bytes[1] << 8)
            payload = self._read_exact(payload_len) if payload_len else b""

            if status == STATUS_FAILED:
                error_code = payload[0] if payload else None
                raise OSError("sensor error {}".format(error_code))

            return payload

        raise OSError("response timeout")

    def begin(self):
        devices = self.i2c.scan()
        if self.address not in devices:
            raise OSError("SEN0628 not found")

    def set_ranging_mode(self, matrix_size):
        self._write_packet(CMD_SETMODE, bytes((0, 0, 0, matrix_size)))
        self._read_response(CMD_SETMODE)
        time.sleep_ms(MODE_SWITCH_DELAY_MS)

    def get_all_data(self):
        self._write_packet(CMD_ALLDATA)
        payload = self._read_response(CMD_ALLDATA)

        if len(payload) < 128:
            raise OSError("payload too short: {}".format(len(payload)))

        values = []
        for index in range(0, 128, 2):
            values.append(payload[index] | (payload[index + 1] << 8))
        return values


def main():
    print("PICO_READY")
    print(
        "SEN0628_TEST:I2C_ID={},SDA=GP{},SCL=GP{},ADDR={}".format(
            I2C_ID, SDA_PIN, SCL_PIN, hex(SENSOR_ADDRESS)
        )
    )
    print("SEN0628:I2C_FREQ={}".format(I2C_FREQ))
    print("SEN0628:STARTUP_DELAY_MS={}".format(STARTUP_DELAY_MS))
    print("SEN0628:MOUNT_MODE={}".format(MOUNT_MODE))
    if MOUNT_MODE == "ceiling_down":
        print("SEN0628:CEILING_HEIGHT_MM={}".format(CEILING_HEIGHT_MM))
        print("SEN0628:FLOOR_OCCUPANCY_DELTA_MM={}".format(FLOOR_OCCUPANCY_DELTA_MM))

    i2c = None
    sensor = None

    while True:
        try:
            if sensor is None:
                print("SEN0628:RECOVERING_I2C_BUS")
                recover_i2c_bus()
                time.sleep_ms(STARTUP_DELAY_MS)
                scan_deadline = time.ticks_add(time.ticks_ms(), SCAN_RETRY_MS)

                while True:
                    i2c = I2C(I2C_ID, sda=Pin(SDA_PIN), scl=Pin(SCL_PIN), freq=I2C_FREQ)
                    devices = safe_i2c_scan(i2c)
                    print("I2C_SCAN:", [hex(device) for device in devices])

                    if SENSOR_ADDRESS in devices:
                        sensor = SEN0628(i2c, SENSOR_ADDRESS)
                        print("SEN0628:FOUND,ADDR={}".format(hex(SENSOR_ADDRESS)))
                        try:
                            sensor.begin()
                            print("SEN0628:SETTING_MODE=8X8")
                            sensor.set_ranging_mode(MATRIX_8X8)
                            print("SEN0628:MODE=8X8_READY")
                        except Exception as exc:
                            print("SEN0628:INIT_FAILURE,{}".format(exc))
                            devices_after_failure = safe_i2c_scan(i2c)
                            print(
                                "SEN0628:POST_INIT_SCAN={}".format(
                                    [hex(device) for device in devices_after_failure]
                                )
                            )
                            sensor = None
                            time.sleep_ms(500)
                            continue
                        for _ in range(WARMUP_READS):
                            try:
                                sensor.get_all_data()
                                time.sleep_ms(100)
                            except Exception:
                                pass
                        break

                    if time.ticks_diff(scan_deadline, time.ticks_ms()) <= 0:
                        print("SEN0628:NOT_FOUND")
                        time.sleep(REPORT_DELAY_S)
                        scan_deadline = time.ticks_add(time.ticks_ms(), SCAN_RETRY_MS)

                    time.sleep_ms(500)

            values = sensor.get_all_data()
            center = values[(3 * MATRIX_8X8) + 3]
            valid_values = valid_readings(values)
            min_value = min(valid_values) if valid_values else None
            max_value = max(valid_values) if valid_values else None
            valid_count = len(valid_values)
            zone_info, occupancy_text, nearest_text, nearest_distance = build_scene_summary(values)
            floor_summary = build_floor_summary(values)

            print(
                "SEN0628:CENTER_MM={},MIN_MM={},MAX_MM={},VALID_POINTS={}/64".format(
                    center,
                    min_value if min_value is not None else "NONE",
                    max_value if max_value is not None else "NONE",
                    valid_count,
                )
            )
            print(
                "SEN0628:INTERPRETATION=Nearest object is {} ({})".format(
                    nearest_text,
                    "{} mm".format(int(nearest_distance)) if nearest_distance is not None else "no valid reading",
                )
            )
            print("SEN0628:SCENE={}".format(occupancy_text))
            if floor_summary is not None:
                print(
                    "SEN0628:FLOOR="
                    "LEFT:{} | CENTER:{} | RIGHT:{} | TOTAL:{} | MEAN_HEIGHT={}mm | MAX_HEIGHT={}mm".format(
                        floor_summary["left"],
                        floor_summary["center"],
                        floor_summary["right"],
                        floor_summary["total"],
                        int(floor_summary["mean_height_mm"]) if floor_summary["mean_height_mm"] is not None else "NONE",
                        int(floor_summary["max_height_mm"]) if floor_summary["max_height_mm"] is not None else "NONE",
                    )
                )
            print(
                "SEN0628:ZONES="
                "LEFT:{}@{}mm | CENTER:{}@{}mm | RIGHT:{}@{}mm".format(
                    describe_zone_strength(zone_info["left"]["close_points"]),
                    int(zone_info["left"]["distance_mm"]) if zone_info["left"]["distance_mm"] is not None else "NONE",
                    describe_zone_strength(zone_info["center"]["close_points"]),
                    int(zone_info["center"]["distance_mm"]) if zone_info["center"]["distance_mm"] is not None else "NONE",
                    describe_zone_strength(zone_info["right"]["close_points"]),
                    int(zone_info["right"]["distance_mm"]) if zone_info["right"]["distance_mm"] is not None else "NONE",
                )
            )
            for row in range(MATRIX_8X8):
                row_values = values[row * MATRIX_8X8 : (row + 1) * MATRIX_8X8]
                print("ROW{}:{}".format(row, format_row(row_values)))
            print("------------------------------")
        except Exception as exc:
            print("SEN0628:ERROR,{}".format(exc))
            devices_after_error = safe_i2c_scan(i2c)
            print(
                "SEN0628:DIAG=SCAN_AFTER_ERROR,DEVICES={}".format(
                    [hex(device) for device in devices_after_error]
                )
            )
            if SENSOR_ADDRESS in devices_after_error:
                print("SEN0628:DIAG=DEVICE_STILL_PRESENT,BUS_OK_SENSOR_NOT_RESPONDING")
            else:
                print("SEN0628:DIAG=DEVICE_MISSING_FROM_I2C_SCAN,CHECK_WIRING_POWER_MODE_SWITCH")
            sensor = None
            i2c = None

        time.sleep(REPORT_DELAY_S)


main()
