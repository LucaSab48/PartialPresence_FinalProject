import argparse
import json
from copy import deepcopy
from typing import Any

from huggingface_hub import InferenceClient

from generate_from_pico import (
    EMPTY_ROOM_BASELINE_PATH,
    SMOOTHING_WINDOW_SIZE,
    build_scene_plan,
    generate_seed_comparison_set,
    load_empty_room_baseline,
    smooth_frames,
)
from hf_auth import load_hf_token
from openai_auth import load_openai_settings


DEFAULT_SEEDS = [23, 42, 77, 101, 222, 333, 444, 777]

FAKE_SENSOR_PRESETS: dict[str, dict[str, Any]] = {
    "empty-room": {
        "description": "No target detected in a dim room.",
        "frame": {
            "bme688": {
                "temperature_c": 21.4,
                "raw_temperature_c": 22.0,
                "humidity_pct": 46.0,
                "pressure_hpa": 1012.4,
                "gas_ohms": 18500.0,
            },
            "ld2410c": {
                "out": 0,
                "target_state": "NO_TARGET",
                "target_state_raw": 0,
                "moving_distance_cm": None,
                "moving_energy": 0,
                "stationary_distance_cm": None,
                "stationary_energy": 0,
                "detection_distance_cm": None,
                "status": "OK",
            },
            "sen0628": {
                "center_mm": 2350,
                "min_mm": 2100,
                "max_mm": 3100,
                "mean_mm": 2480,
                "valid_points": 18,
                "left_zone_mm": 2550,
                "center_zone_mm": 2380,
                "right_zone_mm": 2510,
                "left_close_points": 0,
                "center_close_points": 1,
                "right_close_points": 0,
                "near_points": 0,
                "mid_points": 3,
                "far_points": 15,
            },
        },
    },
    "one-person-left": {
        "description": "One person standing deeper toward the back of the room.",
        "frame": {
            "bme688": {
                "temperature_c": 22.8,
                "raw_temperature_c": 23.4,
                "humidity_pct": 49.0,
                "pressure_hpa": 1011.7,
                "gas_ohms": 16200.0,
            },
            "ld2410c": {
                "out": 1,
                "target_state": "STATIONARY_TARGET",
                "target_state_raw": 2,
                "moving_distance_cm": None,
                "moving_energy": 8,
                "stationary_distance_cm": 118,
                "stationary_energy": 42,
                "detection_distance_cm": 118,
                "status": "OK",
            },
            "sen0628": {
                "center_mm": 1260,
                "min_mm": 940,
                "max_mm": 2420,
                "mean_mm": 1540,
                "valid_points": 32,
                "left_zone_mm": 1120,
                "center_zone_mm": 1470,
                "right_zone_mm": 2060,
                "left_close_points": 8,
                "center_close_points": 3,
                "right_close_points": 0,
                "near_points": 10,
                "mid_points": 17,
                "far_points": 5,
            },
        },
    },
    "one-person-right-moving": {
        "description": "One active person moving near the front of the room.",
        "frame": {
            "bme688": {
                "temperature_c": 23.7,
                "raw_temperature_c": 24.3,
                "humidity_pct": 53.0,
                "pressure_hpa": 1010.8,
                "gas_ohms": 14800.0,
            },
            "ld2410c": {
                "out": 1,
                "target_state": "MOVING_TARGET",
                "target_state_raw": 1,
                "moving_distance_cm": 138,
                "moving_energy": 68,
                "stationary_distance_cm": None,
                "stationary_energy": 4,
                "detection_distance_cm": 138,
                "status": "OK",
            },
            "sen0628": {
                "center_mm": 1410,
                "min_mm": 960,
                "max_mm": 2230,
                "mean_mm": 1620,
                "valid_points": 28,
                "left_zone_mm": 2050,
                "center_zone_mm": 1580,
                "right_zone_mm": 1190,
                "left_close_points": 0,
                "center_close_points": 2,
                "right_close_points": 7,
                "near_points": 8,
                "mid_points": 16,
                "far_points": 4,
            },
        },
    },
    "two-people-middle": {
        "description": "Two people inferred between the front and back zones.",
        "frame": {
            "bme688": {
                "temperature_c": 24.6,
                "raw_temperature_c": 25.2,
                "humidity_pct": 57.0,
                "pressure_hpa": 1011.2,
                "gas_ohms": 13900.0,
            },
            "ld2410c": {
                "out": 1,
                "target_state": "MOVING_AND_STATIONARY",
                "target_state_raw": 3,
                "moving_distance_cm": 152,
                "moving_energy": 74,
                "stationary_distance_cm": 168,
                "stationary_energy": 39,
                "detection_distance_cm": 160,
                "status": "OK",
            },
            "sen0628": {
                "center_mm": 1560,
                "min_mm": 890,
                "max_mm": 2290,
                "mean_mm": 1670,
                "valid_points": 35,
                "left_zone_mm": 1490,
                "center_zone_mm": 1420,
                "right_zone_mm": 1540,
                "left_close_points": 4,
                "center_close_points": 6,
                "right_close_points": 4,
                "near_points": 12,
                "mid_points": 19,
                "far_points": 4,
            },
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the existing image-generation seed test against fake Pico sensor frames."
    )
    parser.add_argument(
        "--preset",
        default="two-people-middle",
        choices=sorted(FAKE_SENSOR_PRESETS.keys()),
        help="Synthetic sensor scenario to use.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Space-separated list of seeds to compare.",
    )
    parser.add_argument(
        "--frame-count",
        type=int,
        default=SMOOTHING_WINDOW_SIZE,
        help="How many fake frames to build before smoothing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the interpreted state and prompt without calling the HF image API.",
    )
    parser.add_argument(
        "--dump-frames",
        action="store_true",
        help="Print the generated fake raw frames as JSON.",
    )
    parser.add_argument(
        "--no-openai",
        action="store_true",
        help="Force the old rule-based interpreter even if OpenAI is configured.",
    )
    return parser.parse_args()


def build_fake_frame(preset_name: str, seq: int) -> dict[str, Any]:
    template = deepcopy(FAKE_SENSOR_PRESETS[preset_name]["frame"])
    bme = template["bme688"]
    ld = template["ld2410c"]
    sen = template["sen0628"]

    # Add slight deterministic jitter so the smoothing path is exercised realistically.
    bme["temperature_c"] = round(bme["temperature_c"] + (0.1 * ((seq % 2) - 0.5)), 2)
    bme["raw_temperature_c"] = round(bme["raw_temperature_c"] + (0.1 * ((seq % 2) - 0.5)), 2)
    bme["humidity_pct"] = round(bme["humidity_pct"] + (0.5 * ((seq % 3) - 1)), 2)

    if ld.get("moving_distance_cm") is not None:
        ld["moving_distance_cm"] += (seq % 3) - 1
    if ld.get("stationary_distance_cm") is not None:
        ld["stationary_distance_cm"] += (seq % 3) - 1
    if ld.get("detection_distance_cm") is not None:
        ld["detection_distance_cm"] += (seq % 3) - 1

    sen["center_mm"] += (seq % 5) - 2
    sen["mean_mm"] += (seq % 5) - 2
    for zone_key in ("left_zone_mm", "center_zone_mm", "right_zone_mm"):
        if sen.get(zone_key) is not None:
            sen[zone_key] += (seq % 3) - 1

    ld_front = {
        "out": 0,
        "target_state": "NO_TARGET",
        "target_state_raw": 0,
        "moving_distance_cm": None,
        "moving_energy": 0,
        "stationary_distance_cm": None,
        "stationary_energy": 0,
        "detection_distance_cm": None,
        "status": "OK",
    }
    ld_back = deepcopy(ld_front)

    if preset_name == "one-person-right-moving":
        ld_front = deepcopy(ld)
    elif preset_name == "one-person-left":
        ld_back = deepcopy(ld)
    elif preset_name == "two-people-middle":
        ld_front = deepcopy(ld)
        ld_back = deepcopy(ld)
        ld_back["moving_distance_cm"] = 214
        ld_back["stationary_distance_cm"] = 232
        ld_back["detection_distance_cm"] = 224
        ld_back["moving_energy"] = max(40, (ld.get("moving_energy") or 0) - 12)
        ld_back["stationary_energy"] = max(28, (ld.get("stationary_energy") or 0) - 8)

    return {
        "type": "sensor_frame",
        "seq": seq,
        "uptime_ms": seq * 250,
        "bme688": bme,
        "ld2410c_front": ld_front,
        "ld2410c_back": ld_back,
        "sen0628": sen,
    }


def build_fake_frames(preset_name: str, frame_count: int) -> list[dict[str, Any]]:
    return [build_fake_frame(preset_name, seq) for seq in range(frame_count)]


def build_shared_state(
    raw_frames: list[dict[str, Any]],
    *,
    use_openai: bool,
) -> dict[str, Any]:
    smoothed = smooth_frames(raw_frames)
    openai_settings = load_openai_settings(required=False)
    if not use_openai or not openai_settings.get("api_key"):
        openai_settings["enabled"] = False
    empty_room_baseline = load_empty_room_baseline() if EMPTY_ROOM_BASELINE_PATH.exists() else None
    scene_plan = build_scene_plan(smoothed, openai_settings, empty_room_baseline)
    return {
        "smoothed_sensor_values": smoothed,
        "interpreted_state": scene_plan["descriptors"],
        "prompt_sections": scene_plan["prompt_sections"],
        "image_prompt": scene_plan["prompt"],
        "generation_controls": scene_plan.get("generation_controls"),
        "live_inference_lines": scene_plan["live_lines"],
        "interpretation_source": scene_plan.get("interpretation_source"),
        "last_openai_error": scene_plan.get("openai_error"),
    }


def main() -> None:
    args = parse_args()
    raw_frames = build_fake_frames(args.preset, max(1, args.frame_count))
    shared_state = build_shared_state(raw_frames, use_openai=not args.no_openai)

    print(f"Preset: {args.preset}")
    print(f"Description: {FAKE_SENSOR_PRESETS[args.preset]['description']}")
    print(f"Seeds: {args.seeds}")
    print(f"Interpreter: {shared_state.get('interpretation_source', 'rule_based')}")
    print("Live inference:")
    for line in shared_state["live_inference_lines"]:
        print(f"  - {line}")
    print("Final prompt:")
    print(shared_state["image_prompt"])

    if args.dump_frames:
        print("Raw frames:")
        print(json.dumps(raw_frames, indent=2))

    if args.dry_run:
        return

    token = load_hf_token()
    client = InferenceClient(provider="hf-inference", api_key=token)
    output_paths = generate_seed_comparison_set(
        client,
        shared_state=shared_state,
        raw_frames=raw_frames,
        seeds=args.seeds,
    )
    print("Generated files:")
    for path in output_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
