from machine import I2C, Pin
import time

I2C_ID = 0
SDA_PIN = 4
SCL_PIN = 5
I2C_FREQUENCY = 100_000
BME68X_ADDRS = (0x76, 0x77)
EXPECTED_CHIP_ID = 0x61
SAMPLE_DELAY_S = 2
TEMPERATURE_OFFSET_C = 2.0

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


def twos_complement(value, bits):
    if value & (1 << (bits - 1)):
        value -= 1 << bits
    return value


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
        adjusted_temperature_c = (temperature_centi_c / 100.0) - TEMPERATURE_OFFSET_C

        return {
            "temperature_c": adjusted_temperature_c,
            "raw_temperature_c": temperature_centi_c / 100.0,
            "pressure_hpa": pressure_pa / 100.0,
            "humidity_pct": humidity_milli_pct / 1000.0,
            "gas_ohms": gas_ohms,
        }


def scan_for_bme688(i2c):
    devices = i2c.scan()
    print("I2C_SCAN:", [hex(device) for device in devices])

    for address in BME68X_ADDRS:
        if address in devices:
            return address

    return None


def main():
    i2c = I2C(
        I2C_ID,
        sda=Pin(SDA_PIN),
        scl=Pin(SCL_PIN),
        freq=I2C_FREQUENCY,
    )

    print("PICO_READY")
    print(
        "BME688_TEST:I2C_ID={},SDA=GP{},SCL=GP{},TEMP_OFFSET_C={:.2f}".format(
            I2C_ID, SDA_PIN, SCL_PIN, TEMPERATURE_OFFSET_C
        )
    )

    address = scan_for_bme688(i2c)
    if address is None:
        print("BME688:NOT_FOUND")
        return

    chip_id = i2c.readfrom_mem(address, REG_CHIP_ID, 1)[0]
    if chip_id != EXPECTED_CHIP_ID:
        print(
            "BME688:FOUND,ADDR={},CHIP_ID=0x{:02X},STATUS=UNEXPECTED_ID".format(
                hex(address), chip_id
            )
        )
        return

    sensor = BME688(i2c, address)
    print(
        "BME688:FOUND,ADDR={},CHIP_ID=0x{:02X},STATUS=PASS".format(
            hex(address), chip_id
        )
    )

    while True:
        try:
            reading = sensor.read()
            print(
                "BME688:TEMP_C={:.2f},RAW_TEMP_C={:.2f},HUMIDITY_PCT={:.2f},PRESSURE_HPA={:.2f},GAS_OHMS={}".format(
                    reading["temperature_c"],
                    reading["raw_temperature_c"],
                    reading["humidity_pct"],
                    reading["pressure_hpa"],
                    reading["gas_ohms"],
                )
            )
        except Exception as exc:
            print("BME688:ERROR,{}".format(exc))

        time.sleep(SAMPLE_DELAY_S)


main()
