from machine import I2C, Pin, UART
import time

try:
    import ujson as json
except ImportError:
    import json


BME_I2C_ID = 0
BME_SDA_PIN = 4
BME_SCL_PIN = 5
BME_I2C_FREQ = 100_000
BME_ADDRS = (0x76, 0x77)
BME_EXPECTED_CHIP_ID = 0x61

LD_UART_BAUDRATE = 256000
LD_MAX_BUFFER_SIZE = 512
LD_SENSOR_CONFIGS = (
    {
        "name": "ld2410c_front",
        "uart_id": 0,
        "tx_pin": 0,
        "rx_pin": 1,
        "out_pin": 2,
    },
    {
        "name": "ld2410c_back",
        "uart_id": 1,
        "tx_pin": 8,
        "rx_pin": 9,
        "out_pin": 3,
    },
)

SEN_I2C_ID = 1
SEN_SDA_PIN = 6
SEN_SCL_PIN = 7
SEN_I2C_FREQ = 100_000
SEN_SENSOR_ADDRESS = 0x33
SEN_MATRIX_8X8 = 8
SEN_STARTUP_DELAY_MS = 1500
SEN_SCAN_RETRY_MS = 10000
SEN_WARMUP_READS = 2
SEN_MOUNT_MODE = "ceiling_down"
SEN_RUNTIME_MODE = "hardware"
SEN_CEILING_HEIGHT_MM = 2400
SEN_FLOOR_OCCUPANCY_DELTA_MM = 180
SEN_FLOOR_NEAR_HEIGHT_MM = 250
SEN_FLOOR_MID_HEIGHT_MM = 900

REPORT_INTERVAL_MS = 2000

REG_CHIP_ID = 0xD0
REG_CTRL_GAS_1 = 0x71
REG_CTRL_HUM = 0x72
REG_CTRL_MEAS = 0x74
REG_CONFIG = 0x75
REG_FIELD0 = 0x1D
REG_RES_HEAT0 = 0x5A
REG_GAS_WAIT0 = 0x64

LOOKUP_TABLE_1 = (
    2147483647,
    2147483647,
    2147483647,
    2147483647,
    2147483647,
    2126008810,
    2147483647,
    2130303777,
    2147483647,
    2147483647,
    2143188679,
    2136746228,
    2147483647,
    2126008810,
    2147483647,
    2147483647,
)

LOOKUP_TABLE_2 = (
    4096000000,
    2048000000,
    1024000000,
    512000000,
    255744255,
    127110228,
    64000000,
    32258064,
    16016016,
    8000000,
    4000000,
    2000000,
    1000000,
    500000,
    250000,
    125000,
)

LD_FRAME_HEADER = b"\xF4\xF3\xF2\xF1"
LD_FRAME_END = b"\xF8\xF7\xF6\xF5"
LD_MIN_TARGET_PAYLOAD_LEN = 13

LD_TARGET_STATES = {
    0x00: "NO_TARGET",
    0x01: "MOVING_TARGET",
    0x02: "STATIONARY_TARGET",
    0x03: "MOVING_AND_STATIONARY",
}

SEN_CMD_SETMODE = 0x01
SEN_CMD_ALLDATA = 0x02
SEN_STATUS_SUCCESS = 0x53
SEN_STATUS_FAILED = 0x63
SEN_RESPONSE_TIMEOUT_MS = 8000
SEN_MODE_SWITCH_DELAY_MS = 5000


def twos_complement(value, bits):
    if value & (1 << (bits - 1)):
        value -= 1 << bits
    return value


def safe_round(value, digits=2):
    if value is None:
        return None
    return round(value, digits)


class BME688:
    def __init__(self, i2c, address):
        self.i2c = i2c
        self.address = address
        self.t_fine = 0
        self._load_calibration()
        self._configure_sensor()

    def _read(self, register, length=1):
        return self.i2c.readfrom_mem(self.address, register, length)

    def _write(self, register, value):
        self.i2c.writeto_mem(self.address, register, bytes((value,)))

    def _load_calibration(self):
        coeff = self._read(0x89, 25) + self._read(0xE1, 16)

        self.par_t1 = (coeff[34] << 8) | coeff[33]
        self.par_t2 = twos_complement((coeff[2] << 8) | coeff[1], 16)
        self.par_t3 = twos_complement(coeff[3], 8)

        self.par_p1 = (coeff[6] << 8) | coeff[5]
        self.par_p2 = twos_complement((coeff[8] << 8) | coeff[7], 16)
        self.par_p3 = twos_complement(coeff[9], 8)
        self.par_p4 = twos_complement((coeff[12] << 8) | coeff[11], 16)
        self.par_p5 = twos_complement((coeff[14] << 8) | coeff[13], 16)
        self.par_p6 = twos_complement(coeff[16], 8)
        self.par_p7 = twos_complement(coeff[15], 8)
        self.par_p8 = twos_complement((coeff[20] << 8) | coeff[19], 16)
        self.par_p9 = twos_complement((coeff[22] << 8) | coeff[21], 16)
        self.par_p10 = coeff[23]

        self.par_h1 = (coeff[27] << 4) | (coeff[26] & 0x0F)
        self.par_h2 = (coeff[25] << 4) | (coeff[26] >> 4)
        self.par_h3 = twos_complement(coeff[28], 8)
        self.par_h4 = twos_complement(coeff[29], 8)
        self.par_h5 = twos_complement(coeff[30], 8)
        self.par_h6 = coeff[31]
        self.par_h7 = twos_complement(coeff[32], 8)

        self.par_gh1 = twos_complement(coeff[37], 8)
        self.par_gh2 = twos_complement((coeff[36] << 8) | coeff[35], 16)
        self.par_gh3 = twos_complement(coeff[38], 8)

        self.res_heat_range = (self._read(0x02)[0] & 0x30) >> 4
        self.res_heat_val = twos_complement(self._read(0x00)[0], 8)
        self.range_sw_err = twos_complement((self._read(0x04)[0] & 0xF0) >> 4, 4)

    def _configure_sensor(self):
        self._write(REG_CONFIG, 0x00)
        self._write(REG_CTRL_HUM, 0x01)
        self._write(REG_CTRL_MEAS, 0x25)
        self._write(REG_CTRL_GAS_1, 0x10)
        self._write(REG_RES_HEAT0, self._calc_res_heat(320))
        self._write(REG_GAS_WAIT0, 50)

    def _calc_res_heat(self, target_temp_c, ambient_temp_c=25):
        var1 = ((ambient_temp_c * self.par_gh3) // 1000) * 256
        var2 = (
            (self.par_gh1 + 784)
            * (((((self.par_gh2 + 154009) * target_temp_c * 5) // 100) + 3276800) // 10)
        )
        var3 = var1 + (var2 // 2)
        var4 = var3 // (self.res_heat_range + 4)
        var5 = (131 * self.res_heat_val) + 65536
        heatr_res_x100 = ((var4 // var5) - 250) * 34
        heatr_res = (heatr_res_x100 + 50) // 100
        return max(0, min(255, heatr_res))

    def _start_measurement(self):
        self._write(REG_CTRL_HUM, 0x01)
        self._write(REG_CTRL_MEAS, 0x25)
        self._write(REG_CTRL_GAS_1, 0x10)
        self._write(REG_CTRL_MEAS, 0x25 | 0x01)

    def _read_raw_data(self):
        self._start_measurement()

        for _ in range(20):
            data = self._read(REG_FIELD0, 15)
            if data[0] & 0x80:
                return data
            time.sleep_ms(20)

        raise OSError("measurement timeout")

    def _compensate_temperature(self, adc_temp):
        var1 = ((adc_temp >> 3) - (self.par_t1 << 1))
        var2 = (var1 * self.par_t2) >> 11
        var3 = (((var1 >> 1) * (var1 >> 1)) >> 12)
        var3 = ((var3 * (self.par_t3 << 4)) >> 14)
        self.t_fine = var2 + var3
        return ((self.t_fine * 5) + 128) >> 8

    def _compensate_pressure(self, adc_pressure):
        var1 = (self.t_fine >> 1) - 64000
        var2 = ((((var1 >> 2) * (var1 >> 2)) >> 11) * self.par_p6) >> 2
        var2 = var2 + ((var1 * self.par_p5) << 1)
        var2 = (var2 >> 2) + (self.par_p4 << 16)
        var1 = (
            (((((var1 >> 2) * (var1 >> 2)) >> 13) * (self.par_p3 << 5)) >> 3)
            + ((self.par_p2 * var1) >> 1)
        ) >> 18
        var1 = ((32768 + var1) * self.par_p1) >> 15

        if var1 == 0:
            return 0

        pressure = 1048576 - adc_pressure
        pressure = ((pressure - (var2 >> 12)) * 3125)

        if pressure >= (1 << 30):
            pressure = (pressure // var1) * 2
        else:
            pressure = (pressure * 2) // var1

        var1 = (self.par_p9 * (((pressure >> 3) * (pressure >> 3)) >> 13)) >> 12
        var2 = ((pressure >> 2) * self.par_p8) >> 13
        var3 = (
            ((pressure >> 8) * (pressure >> 8) * (pressure >> 8) * self.par_p10) >> 17
        )
        pressure = pressure + ((var1 + var2 + var3 + (self.par_p7 << 7)) >> 4)
        return pressure

    def _compensate_humidity(self, adc_humidity):
        temp_scaled = ((self.t_fine * 5) + 128) >> 8
        var1 = adc_humidity - (
            (self.par_h1 * 16) + (((temp_scaled * self.par_h3) // 100) >> 1)
        )
        var2 = (
            self.par_h2
            * (
                (
                    ((temp_scaled * self.par_h4) // 100)
                    + (
                        (
                            (temp_scaled * ((temp_scaled * self.par_h5) // 100))
                            >> 6
                        )
                        // 100
                    )
                    + (1 << 14)
                )
                >> 10
            )
        )
        var3 = var1 * var2
        var4 = (self.par_h6 << 7) + ((temp_scaled * self.par_h7) // 100)
        var4 >>= 4
        var5 = ((var3 >> 14) * (var3 >> 14)) >> 10
        var6 = (var4 * var5) >> 1
        humidity = (((var3 + var6) >> 10) * 1000) >> 12
        humidity = max(0, min(100000, humidity))
        return humidity

    def _compensate_gas(self, adc_gas, gas_range):
        var1 = ((1340 + (5 * self.range_sw_err)) * LOOKUP_TABLE_1[gas_range]) >> 16
        var2 = ((adc_gas << 15) - 16777216) + var1
        var3 = (LOOKUP_TABLE_2[gas_range] * var1) >> 9
        if var2 == 0:
            return 0
        return (var3 + (var2 >> 1)) // var2

    def read(self):
        data = self._read_raw_data()

        adc_pressure = (data[2] << 12) | (data[3] << 4) | (data[4] >> 4)
        adc_temperature = (data[5] << 12) | (data[6] << 4) | (data[7] >> 4)
        adc_humidity = (data[8] << 8) | data[9]
        adc_gas = (data[13] << 2) | (data[14] >> 6)
        gas_range = data[14] & 0x0F

        temperature_centi_c = self._compensate_temperature(adc_temperature)
        pressure_pa = self._compensate_pressure(adc_pressure)
        humidity_milli_pct = self._compensate_humidity(adc_humidity)
        gas_ohms = self._compensate_gas(adc_gas, gas_range)

        raw_temperature_c = temperature_centi_c / 100.0
        return {
            "temperature_c": safe_round(raw_temperature_c),
            "raw_temperature_c": safe_round(raw_temperature_c),
            "humidity_pct": safe_round(humidity_milli_pct / 1000.0),
            "pressure_hpa": safe_round(pressure_pa / 100.0),
            "gas_ohms": int(gas_ohms),
        }


class LD2410CReader:
    def __init__(self, uart, out_pin):
        self.uart = uart
        self.out_pin = out_pin
        self.buffer = bytearray()
        self.total_uart_bytes = 0
        self.latest_reading = None

    def poll(self):
        data = self.uart.read()
        if not data:
            return

        self.buffer.extend(data)
        self.total_uart_bytes += len(data)

        if len(self.buffer) > LD_MAX_BUFFER_SIZE:
            self.buffer = self.buffer[-LD_MAX_BUFFER_SIZE:]

        while True:
            frame = self._extract_frame()
            if frame is None:
                return

            reading = self._parse_target_frame(frame)
            if reading is not None:
                self.latest_reading = reading

    def snapshot(self):
        out_state = self.out_pin.value()
        if self.latest_reading is None:
            return {
                "out": out_state,
                "uart_bytes_total": self.total_uart_bytes,
                "status": "WAITING_FOR_VALID_FRAME",
            }

        data = dict(self.latest_reading)
        data["out"] = out_state
        data["target_state"] = LD_TARGET_STATES.get(
            data["target_state_raw"],
            "UNKNOWN",
        )
        return data

    def _extract_frame(self):
        while True:
            header_index = self.buffer.find(LD_FRAME_HEADER)
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

            if frame[-4:] != LD_FRAME_END:
                continue

            return frame

    def _parse_target_frame(self, frame):
        payload_length = frame[4] | (frame[5] << 8)
        payload = frame[6 : 6 + payload_length]

        if payload_length < LD_MIN_TARGET_PAYLOAD_LEN:
            return None
        if payload[1] != 0xAA:
            return None
        if payload[-2:] != b"\x55\x00":
            return None

        return {
            "frame_type": payload[0],
            "payload_length": payload_length,
            "target_state_raw": payload[2],
            "moving_distance_cm": payload[3] | (payload[4] << 8),
            "moving_energy": payload[5],
            "stationary_distance_cm": payload[6] | (payload[7] << 8),
            "stationary_energy": payload[8],
            "detection_distance_cm": payload[9] | (payload[10] << 8),
        }


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

    def _read_response(self, expected_cmd, timeout_ms=SEN_RESPONSE_TIMEOUT_MS):
        deadline = time.ticks_add(time.ticks_ms(), timeout_ms)

        while time.ticks_diff(deadline, time.ticks_ms()) > 0:
            status = self._read_exact(1)[0]

            if status == 0xFF:
                time.sleep_ms(20)
                continue

            if status not in (SEN_STATUS_SUCCESS, SEN_STATUS_FAILED):
                time.sleep_ms(20)
                continue

            cmd = self._read_exact(1)[0]
            if cmd != expected_cmd:
                raise OSError("unexpected response cmd 0x{:02X}".format(cmd))

            length_bytes = self._read_exact(2)
            payload_len = length_bytes[0] | (length_bytes[1] << 8)
            payload = self._read_exact(payload_len) if payload_len else b""

            if status == SEN_STATUS_FAILED:
                error_code = payload[0] if payload else None
                raise OSError("sensor error {}".format(error_code))

            return payload

        raise OSError("response timeout")

    def begin(self):
        devices = self.i2c.scan()
        if self.address not in devices:
            raise OSError("SEN0628 not found")

    def set_ranging_mode(self, matrix_size):
        self._write_packet(SEN_CMD_SETMODE, bytes((0, 0, 0, matrix_size)))
        self._read_response(SEN_CMD_SETMODE)
        time.sleep_ms(SEN_MODE_SWITCH_DELAY_MS)

    def get_all_data(self):
        self._write_packet(SEN_CMD_ALLDATA)
        payload = self._read_response(SEN_CMD_ALLDATA)

        if len(payload) < 128:
            raise OSError("payload too short: {}".format(len(payload)))

        values = []
        for index in range(0, 128, 2):
            values.append(payload[index] | (payload[index + 1] << 8))
        return values


def recover_i2c_bus(sda_pin, scl_pin):
    sda = Pin(sda_pin, Pin.IN, Pin.PULL_UP)
    scl = Pin(scl_pin, Pin.OPEN_DRAIN, value=1)

    for _ in range(9):
        scl.value(0)
        time.sleep_us(10)
        scl.value(1)
        time.sleep_us(10)

    sda = Pin(sda_pin, Pin.OPEN_DRAIN, value=0)
    time.sleep_us(10)
    scl.value(1)
    time.sleep_us(10)
    sda.value(1)
    time.sleep_us(10)

    Pin(sda_pin, Pin.IN, Pin.PULL_UP)
    Pin(scl_pin, Pin.IN, Pin.PULL_UP)


def safe_i2c_scan(i2c):
    if i2c is None:
        return []
    try:
        return i2c.scan()
    except Exception:
        return []


def scan_for_bme688(i2c):
    devices = i2c.scan()
    for address in BME_ADDRS:
        if address in devices:
            return address
    return None


def init_bme688():
    i2c = I2C(
        BME_I2C_ID,
        sda=Pin(BME_SDA_PIN),
        scl=Pin(BME_SCL_PIN),
        freq=BME_I2C_FREQ,
    )
    address = scan_for_bme688(i2c)
    if address is None:
        raise OSError("BME688 not found")

    chip_id = i2c.readfrom_mem(address, REG_CHIP_ID, 1)[0]
    if chip_id != BME_EXPECTED_CHIP_ID:
        raise OSError("unexpected BME688 chip id 0x{:02X}".format(chip_id))

    return BME688(i2c, address)


def init_sen0628():
    recover_i2c_bus(SEN_SDA_PIN, SEN_SCL_PIN)
    time.sleep_ms(SEN_STARTUP_DELAY_MS)
    scan_deadline = time.ticks_add(time.ticks_ms(), SEN_SCAN_RETRY_MS)
    last_error = None

    while True:
        i2c = I2C(
            SEN_I2C_ID,
            sda=Pin(SEN_SDA_PIN),
            scl=Pin(SEN_SCL_PIN),
            freq=SEN_I2C_FREQ,
        )
        devices = safe_i2c_scan(i2c)
        if SEN_SENSOR_ADDRESS in devices:
            sensor = SEN0628(i2c, SEN_SENSOR_ADDRESS)
            try:
                sensor.begin()
                sensor.set_ranging_mode(SEN_MATRIX_8X8)
                for _ in range(SEN_WARMUP_READS):
                    try:
                        sensor.get_all_data()
                        time.sleep_ms(100)
                    except Exception:
                        pass
                return sensor
            except Exception as exc:
                last_error = exc
                time.sleep_ms(500)
        else:
            last_error = OSError("SEN0628 not found")

        if time.ticks_diff(scan_deadline, time.ticks_ms()) <= 0:
            raise OSError(str(last_error) if last_error is not None else "SEN0628 not found")

        time.sleep_ms(250)


def read_sen0628(sensor):
    values = sensor.get_all_data()
    valid_values = [value for value in values if 0 < value < 4000]
    center_index = (3 * SEN_MATRIX_8X8) + 3
    center_raw_value = values[center_index]
    center_value = center_raw_value if 0 < center_raw_value < 4000 else None
    rows = [
        values[row_index * SEN_MATRIX_8X8 : (row_index + 1) * SEN_MATRIX_8X8]
        for row_index in range(SEN_MATRIX_8X8)
    ]

    if valid_values:
        min_value = min(valid_values)
        max_value = max(valid_values)
        mean_value = sum(valid_values) / len(valid_values)
    else:
        min_value = None
        max_value = None
        mean_value = None

    def flatten_columns(start_col, end_col):
        flattened = []
        for row in rows:
            flattened.extend(row[start_col:end_col])
        return flattened

    def mean_valid(readings):
        valid = [value for value in readings if 0 < value < 4000]
        if not valid:
            return None
        return safe_round(sum(valid) / len(valid))

    def close_count(readings, threshold_mm):
        return len([value for value in readings if 0 < value < threshold_mm])

    def occupied_points(readings):
        if SEN_MOUNT_MODE != "ceiling_down":
            return 0
        return len(
            [
                value
                for value in readings
                if 0 < value < 4000 and value <= (SEN_CEILING_HEIGHT_MM - SEN_FLOOR_OCCUPANCY_DELTA_MM)
            ]
        )

    def height_above_floor_stats(readings):
        if SEN_MOUNT_MODE != "ceiling_down":
            return None, None
        heights = [
            SEN_CEILING_HEIGHT_MM - value
            for value in readings
            if 0 < value < 4000 and value <= SEN_CEILING_HEIGHT_MM
        ]
        if not heights:
            return None, None
        return safe_round(sum(heights) / len(heights)), max(heights)

    left_values = flatten_columns(0, 2)
    center_values = flatten_columns(2, 6)
    right_values = flatten_columns(6, 8)
    front_values = rows[0] + rows[1]
    mid_values = rows[2] + rows[3] + rows[4] + rows[5]
    back_values = rows[6] + rows[7]
    mean_obstruction_height_mm, max_obstruction_height_mm = height_above_floor_stats(values)

    return {
        "mount_mode": SEN_MOUNT_MODE,
        "ceiling_height_mm": SEN_CEILING_HEIGHT_MM if SEN_MOUNT_MODE == "ceiling_down" else None,
        "center_mm": center_value,
        "center_raw_mm": center_raw_value,
        "min_mm": min_value,
        "max_mm": max_value,
        "mean_mm": safe_round(mean_value),
        "valid_points": len(valid_values),
        "left_zone_mm": mean_valid(left_values),
        "center_zone_mm": mean_valid(center_values),
        "right_zone_mm": mean_valid(right_values),
        "left_close_points": close_count(left_values, 1600),
        "center_close_points": close_count(center_values, 1600),
        "right_close_points": close_count(right_values, 1600),
        "left_occupied_points": occupied_points(left_values),
        "center_occupied_points": occupied_points(center_values),
        "right_occupied_points": occupied_points(right_values),
        "front_zone_mm": mean_valid(front_values),
        "mid_zone_mm": mean_valid(mid_values),
        "back_zone_mm": mean_valid(back_values),
        "front_occupied_points": occupied_points(front_values),
        "mid_occupied_points": occupied_points(mid_values),
        "back_occupied_points": occupied_points(back_values),
        "near_points": close_count(values, 1200),
        "mid_points": len([value for value in values if 1200 <= value < 2200]),
        "far_points": len([value for value in values if 2200 <= value < 4000]),
        "floor_occupied_points": occupied_points(values),
        "floor_clear_points": len(valid_values) - occupied_points(values),
        "mean_obstruction_height_mm": mean_obstruction_height_mm,
        "max_obstruction_height_mm": max_obstruction_height_mm,
        "low_obstruction_points": len(
            [
                value
                for value in values
                if 0 < value < 4000
                and value <= SEN_CEILING_HEIGHT_MM
                and (SEN_CEILING_HEIGHT_MM - value) >= 0
                and (SEN_CEILING_HEIGHT_MM - value) < SEN_FLOOR_NEAR_HEIGHT_MM
            ]
        )
        if SEN_MOUNT_MODE == "ceiling_down"
        else 0,
        "mid_obstruction_points": len(
            [
                value
                for value in values
                if 0 < value < 4000
                and value <= SEN_CEILING_HEIGHT_MM
                and (SEN_CEILING_HEIGHT_MM - value) >= SEN_FLOOR_NEAR_HEIGHT_MM
                and (SEN_CEILING_HEIGHT_MM - value) < SEN_FLOOR_MID_HEIGHT_MM
            ]
        )
        if SEN_MOUNT_MODE == "ceiling_down"
        else 0,
        "tall_obstruction_points": len(
            [
                value
                for value in values
                if 0 < value < 4000
                and value <= SEN_CEILING_HEIGHT_MM
                and (SEN_CEILING_HEIGHT_MM - value) >= SEN_FLOOR_MID_HEIGHT_MM
            ]
        )
        if SEN_MOUNT_MODE == "ceiling_down"
        else 0,
    }


def build_fake_sen0628_reading():
    empty_distance_mm = SEN_CEILING_HEIGHT_MM if SEN_MOUNT_MODE == "ceiling_down" else 3200
    total_points = SEN_MATRIX_8X8 * SEN_MATRIX_8X8
    return {
        "mount_mode": SEN_MOUNT_MODE,
        "ceiling_height_mm": SEN_CEILING_HEIGHT_MM if SEN_MOUNT_MODE == "ceiling_down" else None,
        "center_mm": empty_distance_mm,
        "center_raw_mm": empty_distance_mm,
        "min_mm": empty_distance_mm,
        "max_mm": empty_distance_mm,
        "mean_mm": empty_distance_mm,
        "valid_points": total_points,
        "left_zone_mm": empty_distance_mm,
        "center_zone_mm": empty_distance_mm,
        "right_zone_mm": empty_distance_mm,
        "left_close_points": 0,
        "center_close_points": 0,
        "right_close_points": 0,
        "left_occupied_points": 0,
        "center_occupied_points": 0,
        "right_occupied_points": 0,
        "front_zone_mm": empty_distance_mm,
        "mid_zone_mm": empty_distance_mm,
        "back_zone_mm": empty_distance_mm,
        "front_occupied_points": 0,
        "mid_occupied_points": 0,
        "back_occupied_points": 0,
        "near_points": 0,
        "mid_points": 0,
        "far_points": total_points,
        "floor_occupied_points": 0,
        "floor_clear_points": total_points,
        "mean_obstruction_height_mm": 0 if SEN_MOUNT_MODE == "ceiling_down" else None,
        "max_obstruction_height_mm": 0 if SEN_MOUNT_MODE == "ceiling_down" else None,
        "low_obstruction_points": 0,
        "mid_obstruction_points": 0,
        "tall_obstruction_points": 0,
    }


def read_bme688(sensor):
    return sensor.read()


def print_json_packet(packet):
    print("SENSOR_DATA:{}".format(json.dumps(packet)))


def main():
    print("PICO_READY")
    config_message = "CONFIG:BME688=I2C{},GP{}/GP{};SEN0628=I2C{},GP{}/GP{},MODE={}".format(
        BME_I2C_ID,
        BME_SDA_PIN,
        BME_SCL_PIN,
        SEN_I2C_ID,
        SEN_SDA_PIN,
        SEN_SCL_PIN,
        SEN_RUNTIME_MODE,
    )
    for config in LD_SENSOR_CONFIGS:
        config_message += ";{}=UART{},TX=GP{},RX=GP{},OUT=GP{}".format(
            config["name"].upper(),
            config["uart_id"],
            config["tx_pin"],
            config["rx_pin"],
            config["out_pin"],
        )
    print(config_message)

    ld_readers = []
    for config in LD_SENSOR_CONFIGS:
        ld_uart = UART(
            config["uart_id"],
            baudrate=LD_UART_BAUDRATE,
            tx=Pin(config["tx_pin"]),
            rx=Pin(config["rx_pin"]),
            bits=8,
            parity=None,
            stop=1,
            timeout=20,
        )
        ld_readers.append(
            (
                config["name"],
                LD2410CReader(ld_uart, Pin(config["out_pin"], Pin.IN)),
            )
        )

    bme_sensor = None
    sen_sensor = None
    sequence = 0
    last_report = time.ticks_ms()

    while True:
        for _, ld_reader in ld_readers:
            ld_reader.poll()
        now = time.ticks_ms()

        if time.ticks_diff(now, last_report) < REPORT_INTERVAL_MS:
            time.sleep_ms(50)
            continue

        packet = {
            "type": "sensor_frame",
            "seq": sequence,
            "uptime_ms": time.ticks_ms(),
        }

        if bme_sensor is None:
            try:
                bme_sensor = init_bme688()
                print("INFO:BME688_READY")
            except Exception as exc:
                packet["bme688_error"] = str(exc)
        if bme_sensor is not None:
            try:
                packet["bme688"] = read_bme688(bme_sensor)
            except Exception as exc:
                packet["bme688_error"] = str(exc)
                bme_sensor = None

        for sensor_name, ld_reader in ld_readers:
            packet[sensor_name] = ld_reader.snapshot()

        if SEN_RUNTIME_MODE == "fake":
            packet["sen0628"] = build_fake_sen0628_reading()
            packet["sen0628_status"] = "fake"
        elif SEN_RUNTIME_MODE == "disabled":
            packet["sen0628_status"] = "disabled"
        elif sen_sensor is None:
            try:
                sen_sensor = init_sen0628()
                print("INFO:SEN0628_READY")
            except Exception as exc:
                packet["sen0628_error"] = str(exc)
        if SEN_RUNTIME_MODE == "hardware" and sen_sensor is not None:
            try:
                packet["sen0628"] = read_sen0628(sen_sensor)
            except Exception as exc:
                packet["sen0628_error"] = str(exc)
                sen_sensor = None

        print_json_packet(packet)

        sequence += 1
        last_report = now


main()
