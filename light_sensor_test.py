from machine import ADC, Pin
import time

LIGHT_ADC_PIN = 26
REPORT_MS = 2000
ADC_MAX = 65535


def main():
    sensor = ADC(Pin(LIGHT_ADC_PIN))
    print("PICO_READY")
    print("LIGHT_SENSOR_TEST:ADC_PIN=GP{}".format(LIGHT_ADC_PIN))

    while True:
        raw = sensor.read_u16()
        percent = (raw / ADC_MAX) * 100.0

        print(
            "LIGHT_SENSOR:RAW={},PERCENT={:.1f}".format(
                raw,
                percent,
            )
        )

        time.sleep_ms(REPORT_MS)


main()
