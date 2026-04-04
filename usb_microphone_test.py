import argparse
import json
import time

from usb_microphone import AudioLevelMonitor, list_input_devices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect USB microphone input devices and live levels.")
    parser.add_argument("--list", action="store_true", help="List input-capable audio devices and exit.")
    parser.add_argument("--device-index", type=int, help="Specific audio input device index to open.")
    parser.add_argument("--device-name", help="Substring match for the preferred input device name.")
    parser.add_argument(
        "--seconds",
        type=float,
        default=15.0,
        help="How long to print live audio snapshots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        devices = list_input_devices()
        print(json.dumps(devices, indent=2))
        return

    monitor = AudioLevelMonitor(
        preferred_device_index=args.device_index,
        preferred_device_name=args.device_name,
    )
    started = monitor.start()
    print(f"USB microphone: {monitor.status_text}")
    if not started:
        return

    deadline = time.time() + max(1.0, args.seconds)
    try:
        while time.time() < deadline:
            print(json.dumps(monitor.snapshot(), indent=None))
            time.sleep(0.5)
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()
