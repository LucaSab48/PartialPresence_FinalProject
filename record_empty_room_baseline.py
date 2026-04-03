import argparse
import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import serial

from generate_from_pico import (
    BAUDRATE,
    EMPTY_ROOM_BASELINE_PATH,
    MIN_FRAMES_FOR_PROCESSING,
    PORT,
    SERIAL_TIMEOUT_SECONDS,
    SERIAL_STARTUP_SECONDS,
    SMOOTHING_WINDOW_SIZE,
    atomic_write_text,
    parse_sensor_line,
    smooth_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a smoothed empty-room baseline from the Pico sensor stream."
    )
    parser.add_argument(
        "--port",
        default=PORT,
        help="Serial port to open.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=12.0,
        help="How long to collect empty-room frames after startup.",
    )
    parser.add_argument(
        "--output",
        default=str(EMPTY_ROOM_BASELINE_PATH),
        help="Where to save the baseline JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    frame_buffer: deque[dict[str, Any]] = deque(maxlen=SMOOTHING_WINDOW_SIZE * 4)

    print(f"Opening {args.port} at {BAUDRATE} baud")
    print("Leave the room empty while recording this baseline.")
    print(f"Waiting up to {SERIAL_STARTUP_SECONDS} seconds for sensor frames...")

    with serial.Serial(args.port, BAUDRATE, timeout=SERIAL_TIMEOUT_SECONDS) as ser:
        startup_deadline = time.time() + SERIAL_STARTUP_SECONDS
        while time.time() < startup_deadline and len(frame_buffer) < MIN_FRAMES_FOR_PROCESSING:
            raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw_line:
                continue
            frame = parse_sensor_line(raw_line)
            if frame is not None:
                frame_buffer.append(frame)

        if len(frame_buffer) < MIN_FRAMES_FOR_PROCESSING:
            raise RuntimeError("No valid sensor frames received from the Pico.")

        print(f"Recording empty-room data for {args.seconds:.1f} seconds...")
        record_deadline = time.time() + max(1.0, args.seconds)
        while time.time() < record_deadline:
            raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not raw_line:
                continue
            frame = parse_sensor_line(raw_line)
            if frame is not None:
                frame_buffer.append(frame)
                print(f"[BASELINE_FRAME] seq={frame.get('seq')} uptime_ms={frame.get('uptime_ms')}")

    raw_frames = list(frame_buffer)
    smoothed = smooth_frames(raw_frames)
    payload = {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "port": args.port,
        "duration_seconds": args.seconds,
        "raw_frame_summary": {
            "frame_count": len(raw_frames),
            "first_seq": raw_frames[0].get("seq") if raw_frames else None,
            "last_seq": raw_frames[-1].get("seq") if raw_frames else None,
        },
        "smoothed_sensor_values": smoothed,
    }
    atomic_write_text(output_path, json.dumps(payload, indent=2))
    print(f"Saved empty-room baseline to {output_path.resolve()}")
    print(json.dumps(smoothed, indent=2))


if __name__ == "__main__":
    main()
