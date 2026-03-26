from machine import Pin
import time
import dht

DATA_PIN = 15
SAMPLE_DELAY_S = 2


def main():
    sensor = dht.DHT11(Pin(DATA_PIN, Pin.IN, Pin.PULL_UP))

    print("PICO_READY")
    print("DHT11_TEST:DATA_PIN=GP{}".format(DATA_PIN))

    while True:
        try:
            sensor.measure()
            print(
                "DHT11:TEMP_C={},HUMIDITY_PCT={}".format(
                    sensor.temperature(),
                    sensor.humidity(),
                )
            )
        except Exception as exc:
            print("DHT11:ERROR,{}".format(exc))

        time.sleep(SAMPLE_DELAY_S)


main()
