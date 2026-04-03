import json
import os
import re
import sys
import time
import threading
import atexit
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import serial
import requests
from huggingface_hub import InferenceClient
from PIL import Image, ImageDraw, ImageFont, ImageOps

from hf_auth import load_hf_token
from openai_auth import load_openai_settings


PORT = "COM8"
BAUDRATE = 115200
SERIAL_TIMEOUT_SECONDS = 1
SERIAL_STARTUP_SECONDS = 20

# Timing controls for the two output layers.
TEXT_UPDATE_INTERVAL_SECONDS = 0.6
IMAGE_GENERATION_INTERVAL_SECONDS = 8.0
FORCE_IMAGE_REFRESH_SECONDS = 10.0
STATE_STABLE_HOLD_SECONDS = 0.9

# Thresholds for deciding whether the interpreted state changed enough to matter.
LIGHT_CHANGE_THRESHOLD = 6.0
TEMPERATURE_CHANGE_THRESHOLD = 0.8
HUMIDITY_CHANGE_THRESHOLD = 4.0
DISTANCE_CHANGE_THRESHOLD_CM = 30.0

SMOOTHING_WINDOW_SIZE = 6
MIN_FRAMES_FOR_PROCESSING = 3

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
IMAGE_GUIDANCE_SCALE = 9.0
IMAGE_NUM_INFERENCE_STEPS = 24
DEFAULT_EMPTY_IMAGE_SEED = 23
DEFAULT_OCCUPIED_IMAGE_SEED = 23
SEED_TEST_MODE = False
SANITY_SINGLE_IMAGE_MODE = False
SEED_TEST_VALUES = [23, 42, 77, 101, 222, 333, 444, 777]
OPTIONAL_TEXT_UI_EXCLUSION_NEGATIVE_PROMPT = (
    "text, words, letters, typography, signage, caption, watermark, user interface, "
    "collage, diptych, triptych, split panels, storyboard, poster layout, document layout, "
    "bed, bedroom, mattress, pillow, blanket, duvet, headboard, nightstand, dresser, wardrobe, "
    "sofa, couch, television, domestic furniture, home decor, lampshade, kitchen, office desk, "
    "artwork, framed art, black screen, monitor, pedestal, sculpture"
)
BASE_SCENE_PROMPT = (
    "photorealistic installation space, gallery-like interior, white cube exhibition room, "
    "fixed camera angle, same general room across generations, stable walls and floor layout, open floor, bare walls, minimal architecture, "
    "no domestic furniture, no bedroom elements, no home decor, no objects unless required by human presence, "
    "clean uncluttered exhibition environment, restrained background detail, believable materials, "
    "realistic lens perspective, preserve similar architecture and room geometry, "
    "keep the environment broadly consistent while allowing occupancy changes, "
    "avoid painterly, graphic, or overtly surreal aesthetics"
)
COMPOSITION_DIRECTIVE = (
    "single interior scene, cinematic photography, medium-long framing, stable composition, "
    "background remains broadly stable and secondary, hold a similar camera position and lens perspective, "
    "do not redesign the room or introduce new architecture, human placement should feel estimated "
    "rather than overly staged, allow lateral and depth variation in the figures, the room should feel "
    "inferred from sensors instead of confidently mapped"
)
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "generated_images"
LOCK_FILE_PATH = SCRIPT_DIR / ".generate_from_pico.lock"
SEED_TEST_OUTPUT_DIR = OUTPUT_DIR / "seed_tests"
SHARED_STATE_PATH = OUTPUT_DIR / "current_interpretation_state.json"
SERIAL_PREFIX = "SENSOR_DATA:"
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
CONTACT_SHEET_FILENAME = "seed_contact_sheet.png"
SEED_INDEX_FILENAME = "seed_test_index.json"
OPENAI_RESPONSE_TIMEOUT_FALLBACK_SECONDS = 25
MAX_AI_LIVE_LINES = 5
EMPTY_ROOM_BASELINE_PATH = SCRIPT_DIR / "empty_room_baseline.json"
MAX_BASELINE_CHANGED_METRICS = 5
MAX_LANGUAGE_PASS_LIVE_LINES = 4
MIN_LANGUAGE_PASS_LIVE_LINES = 3
MAX_LANGUAGE_PASS_AGENT_NOTES = 1
MIN_LANGUAGE_PASS_AGENT_NOTES = 0
MAX_LANGUAGE_PASS_PROMPT_MODIFIERS = 6
MIN_LANGUAGE_PASS_PROMPT_MODIFIERS = 3
BASELINE_SUBSTANTIAL_DELTA = 3.0
BASELINE_VERY_SUBSTANTIAL_DELTA = 12.0


@dataclass
class InterpretationCoordinator:
    current_signature: str | None = None
    stable_signature: str | None = None
    last_meaningful_change_time: float = 0.0
    last_text_update_time: float = 0.0
    last_image_generation_time: float = 0.0
    last_image_signature: str | None = None
    latest_shared_state: dict[str, Any] | None = None
    image_generation_in_progress: bool = False
    image_generation_started_at: float = 0.0
    pending_image_result: dict[str, Any] | None = None


def json_text(value: object) -> str:
    return json.dumps(cast(Any, value), indent=2)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp_path.write_text(text, encoding="utf-8")

    last_error: PermissionError | None = None
    for attempt in range(10):
        try:
            temp_path.replace(path)
            return
        except PermissionError as error:
            last_error = error
            # Windows can briefly deny replace() while another process is reading the target file.
            time.sleep(0.05 * (attempt + 1))

    try:
        temp_path.unlink(missing_ok=True)
    except OSError:
        pass

    if last_error is not None:
        raise last_error


def numeric_mean(values):
    clean_values = [value for value in values if isinstance(value, (int, float))]
    if not clean_values:
        return None
    return sum(clean_values) / len(clean_values)


def pick_mode(values):
    clean_values = [value for value in values if value not in (None, "")]
    if not clean_values:
        return None
    return Counter(clean_values).most_common(1)[0][0]


def parse_sensor_line(raw_line):
    if not raw_line.startswith(SERIAL_PREFIX):
        return None

    payload = raw_line[len(SERIAL_PREFIX) :].strip()
    if not payload:
        return None

    try:
        frame = json.loads(payload)
    except json.JSONDecodeError:
        return None

    if not isinstance(frame, dict):
        return None

    if frame.get("type") != "sensor_frame":
        return None

    return frame


def smooth_frames(frames):
    bme_frames = [frame.get("bme688", {}) for frame in frames if isinstance(frame.get("bme688"), dict)]
    light_frames = [frame.get("light", {}) for frame in frames if isinstance(frame.get("light"), dict)]
    ld_frames = [frame.get("ld2410c", {}) for frame in frames if isinstance(frame.get("ld2410c"), dict)]
    sen_frames = [frame.get("sen0628", {}) for frame in frames if isinstance(frame.get("sen0628"), dict)]
    ld_out_mean = numeric_mean([item.get("out") for item in ld_frames])

    smoothed = {
        "frame_count": len(frames),
        "seq_range": [frames[0].get("seq"), frames[-1].get("seq")] if frames else [None, None],
        "uptime_ms": frames[-1].get("uptime_ms") if frames else None,
        "bme688": {
            "temperature_c": numeric_mean([item.get("temperature_c") for item in bme_frames]),
            "raw_temperature_c": numeric_mean([item.get("raw_temperature_c") for item in bme_frames]),
            "humidity_pct": numeric_mean([item.get("humidity_pct") for item in bme_frames]),
            "pressure_hpa": numeric_mean([item.get("pressure_hpa") for item in bme_frames]),
            "gas_ohms": numeric_mean([item.get("gas_ohms") for item in bme_frames]),
        }
        if bme_frames
        else None,
        "light": {
            "raw": numeric_mean([item.get("raw") for item in light_frames]),
            "percent": numeric_mean([item.get("percent") for item in light_frames]),
        }
        if light_frames
        else None,
        "ld2410c": {
            "out": round(ld_out_mean) if ld_out_mean is not None else None,
            "target_state": pick_mode([item.get("target_state") for item in ld_frames]),
            "target_state_raw": pick_mode([item.get("target_state_raw") for item in ld_frames]),
            "moving_distance_cm": numeric_mean([item.get("moving_distance_cm") for item in ld_frames]),
            "moving_energy": numeric_mean([item.get("moving_energy") for item in ld_frames]),
            "stationary_distance_cm": numeric_mean([item.get("stationary_distance_cm") for item in ld_frames]),
            "stationary_energy": numeric_mean([item.get("stationary_energy") for item in ld_frames]),
            "detection_distance_cm": numeric_mean([item.get("detection_distance_cm") for item in ld_frames]),
            "status": pick_mode([item.get("status") for item in ld_frames]),
            "state_history": [item.get("target_state") for item in ld_frames if item.get("target_state")],
        }
        if ld_frames
        else None,
        "sen0628": {
            "center_mm": numeric_mean([item.get("center_mm") for item in sen_frames]),
            "min_mm": numeric_mean([item.get("min_mm") for item in sen_frames]),
            "max_mm": numeric_mean([item.get("max_mm") for item in sen_frames]),
            "mean_mm": numeric_mean([item.get("mean_mm") for item in sen_frames]),
            "valid_points": numeric_mean([item.get("valid_points") for item in sen_frames]),
            "left_zone_mm": numeric_mean([item.get("left_zone_mm") for item in sen_frames]),
            "center_zone_mm": numeric_mean([item.get("center_zone_mm") for item in sen_frames]),
            "right_zone_mm": numeric_mean([item.get("right_zone_mm") for item in sen_frames]),
            "left_close_points": numeric_mean([item.get("left_close_points") for item in sen_frames]),
            "center_close_points": numeric_mean([item.get("center_close_points") for item in sen_frames]),
            "right_close_points": numeric_mean([item.get("right_close_points") for item in sen_frames]),
            "near_points": numeric_mean([item.get("near_points") for item in sen_frames]),
            "mid_points": numeric_mean([item.get("mid_points") for item in sen_frames]),
            "far_points": numeric_mean([item.get("far_points") for item in sen_frames]),
        }
        if sen_frames
        else None,
        "errors": {
            "bme688_error": pick_mode([frame.get("bme688_error") for frame in frames]),
            "sen0628_error": pick_mode([frame.get("sen0628_error") for frame in frames]),
        },
    }
    return smoothed


def load_empty_room_baseline() -> dict[str, Any] | None:
    if not EMPTY_ROOM_BASELINE_PATH.exists():
        return None

    try:
        payload = json.loads(EMPTY_ROOM_BASELINE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None

    smoothed = payload.get("smoothed_sensor_values")
    if isinstance(smoothed, dict):
        return smoothed
    return None


def flatten_numeric_values(data: Any, prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    if not isinstance(data, dict):
        return flat

    for key, value in data.items():
        next_prefix = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_numeric_values(value, next_prefix))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[next_prefix] = float(value)
    return flat


def build_empty_room_comparison(
    smoothed: dict[str, Any],
    empty_room_baseline: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if empty_room_baseline is None:
        return None

    live_values = flatten_numeric_values(smoothed)
    baseline_values = flatten_numeric_values(empty_room_baseline)
    changed_metrics: list[dict[str, Any]] = []

    for key in sorted(set(live_values.keys()) & set(baseline_values.keys())):
        baseline_value = baseline_values[key]
        live_value = live_values[key]
        delta = live_value - baseline_value
        magnitude = abs(delta)
        if magnitude < 0.75:
            continue
        changed_metrics.append(
            {
                "metric": key,
                "baseline": round(baseline_value, 3),
                "live": round(live_value, 3),
                "delta": round(delta, 3),
            }
        )

    changed_metrics.sort(key=lambda item: abs(float(item["delta"])), reverse=True)
    return {
        "available": True,
        "top_changed_metrics": changed_metrics[:12],
        "baseline_seq_range": empty_room_baseline.get("seq_range"),
        "baseline_frame_count": empty_room_baseline.get("frame_count"),
    }


def classify_baseline_departure(empty_room_comparison: dict[str, Any] | None) -> str:
    if not isinstance(empty_room_comparison, dict):
        return "unknown"

    changed_metrics = empty_room_comparison.get("top_changed_metrics")
    if not isinstance(changed_metrics, list):
        return "minimal"
    if not changed_metrics:
        return "minimal"

    departure_score = 0
    for metric in changed_metrics:
        if not isinstance(metric, dict):
            continue
        delta = metric.get("delta")
        if not isinstance(delta, (int, float)) or isinstance(delta, bool):
            continue
        magnitude = abs(float(delta))
        if magnitude >= BASELINE_VERY_SUBSTANTIAL_DELTA:
            departure_score += 2
        elif magnitude >= BASELINE_SUBSTANTIAL_DELTA:
            departure_score += 1

    if departure_score <= 1:
        return "weak"
    if departure_score <= 4:
        return "moderate"
    return "strong"


def interpret_presence(ld_data):
    if not ld_data:
        return "presence uncertain"

    states = ld_data.get("state_history", [])
    state_counts = Counter(states)
    unique_states = len(state_counts)
    target_state = ld_data.get("target_state")
    out_state = ld_data.get("out")
    moving_energy = ld_data.get("moving_energy") or 0
    stationary_energy = ld_data.get("stationary_energy") or 0

    if target_state == "NO_TARGET" and out_state == 0:
        return "no presence"
    if unique_states >= 3:
        return "intermittent movement"
    if target_state == "MOVING_TARGET" or moving_energy >= 45:
        return "active presence"
    if target_state == "MOVING_AND_STATIONARY":
        return "intermittent movement"
    if target_state == "STATIONARY_TARGET" or stationary_energy >= 30 or out_state == 1:
        return "still presence"
    if ld_data.get("status") == "WAITING_FOR_VALID_FRAME":
        return "presence uncertain"
    return "presence uncertain"


def estimate_presence_count(ld_data, sen_data):
    if not ld_data:
        return "uncertain number of people"

    target_state = ld_data.get("target_state")
    out_state = ld_data.get("out")
    moving_energy = ld_data.get("moving_energy") or 0
    stationary_energy = ld_data.get("stationary_energy") or 0

    if target_state == "NO_TARGET" and out_state == 0:
        return "no people"

    if target_state == "MOVING_AND_STATIONARY":
        return "two people"
    if moving_energy >= 70 and stationary_energy >= 35:
        return "two people"
    if moving_energy >= 45 or stationary_energy >= 30 or out_state == 1:
        return "one person"
    return "uncertain number of people"


def classify_depth_band(ld_data, sen_data):
    distance_cm = None
    if ld_data:
        distance_cm = ld_data.get("detection_distance_cm")
        if distance_cm is None:
            distance_cm = ld_data.get("moving_distance_cm") or ld_data.get("stationary_distance_cm")

    if distance_cm is None and sen_data:
        center_mm = sen_data.get("center_mm")
        mean_mm = sen_data.get("mean_mm")
        if center_mm is not None:
            distance_cm = center_mm / 10.0
        elif mean_mm is not None:
            distance_cm = mean_mm / 10.0

    if distance_cm is None:
        return "mid-room"
    if distance_cm < 90:
        return "front"
    if distance_cm < 180:
        return "mid-room"
    return "back"


def estimate_people_layout(ld_data, sen_data, presence_activity, presence_count):
    if presence_activity == "no presence" or presence_count == "no people":
        return {
            "figure_count": 0,
            "placement_summary": "empty room",
            "placement_prompt": "show no people in the room",
            "depth_band": "mid-room",
            "primary_zone": "center",
            "secondary_zone": "center",
            "active_zones": [],
            "layout_mode": "empty",
        }

    zone_strengths = {
        "left": (sen_data or {}).get("left_close_points") or 0,
        "center": (sen_data or {}).get("center_close_points") or 0,
        "right": (sen_data or {}).get("right_close_points") or 0,
    }
    sorted_zones = sorted(zone_strengths.items(), key=lambda item: item[1], reverse=True)
    active_zones = [name for name, value in sorted_zones if value >= 3]
    primary_zone = sorted_zones[0][0]
    secondary_zone = sorted_zones[1][0]
    depth_band = classify_depth_band(ld_data, sen_data)

    zone_labels = {
        "left": "left side of the room",
        "center": "center of the room",
        "right": "right side of the room",
    }
    depth_labels = {
        "front": "near the front of the room",
        "mid-room": "around the middle of the room",
        "back": "toward the back of the room",
    }

    if presence_count == "two people":
        if len(active_zones) >= 2:
            first_zone = active_zones[0]
            second_zone = active_zones[1]
            placement_summary = f"two people split between {first_zone} and {second_zone}"
            placement_prompt = (
                f"show exactly two human figures, one on the {zone_labels[first_zone]}, "
                f"the other on the {zone_labels[second_zone]}, both {depth_labels[depth_band]}"
            )
        else:
            placement_summary = f"two people clustered on the {primary_zone}"
            placement_prompt = (
                f"show exactly two human figures clustered on the {zone_labels[primary_zone]}, "
                f"with slight separation between them, both {depth_labels[depth_band]}"
            )
        return {
            "figure_count": 2,
            "placement_summary": placement_summary,
            "placement_prompt": placement_prompt,
            "depth_band": depth_band,
            "primary_zone": primary_zone,
            "secondary_zone": secondary_zone,
            "active_zones": active_zones,
            "layout_mode": "split" if len(active_zones) >= 2 else "clustered",
        }

    if presence_activity == "presence uncertain" and sorted_zones[0][1] < 2:
        return {
            "figure_count": 0,
            "placement_summary": "empty room",
            "placement_prompt": "show no people in the room",
            "depth_band": depth_band,
            "primary_zone": primary_zone,
            "secondary_zone": secondary_zone,
            "active_zones": active_zones,
            "layout_mode": "empty",
        }

    preferred_zone = primary_zone if sorted_zones[0][1] >= 2 else "center"
    return {
        "figure_count": 1,
        "placement_summary": f"one person near the {preferred_zone}",
        "placement_prompt": (
            f"show exactly one human figure on the {zone_labels[preferred_zone]}, "
            f"{depth_labels[depth_band]}"
        ),
        "depth_band": depth_band,
        "primary_zone": preferred_zone,
        "secondary_zone": secondary_zone,
        "active_zones": active_zones,
        "layout_mode": "single",
    }


def describe_spatial_certainty(
    presence_activity: str,
    active_zones: list[str],
    primary_zone: str,
) -> str:
    if presence_activity == "presence uncertain":
        return "spatial certainty partial"
    if not active_zones:
        return "spatial certainty weak"
    if len(active_zones) >= 3:
        return "multi-zone certainty reduced"
    if len(active_zones) == 2:
        return "split-zone layout supported"
    return f"{primary_zone} weighting more stable"


def build_figure_variation_modifiers(descriptors: dict[str, Any]) -> list[str]:
    figure_count = int(descriptors.get("figure_count", 0))
    layout_mode = str(descriptors.get("layout_mode", "empty"))
    primary_zone = str(descriptors.get("primary_zone", "center"))
    depth_band = str(descriptors.get("depth_band", "mid-room"))

    modifiers = ["same room retained", "background structure held", "camera view kept similar"]
    if figure_count <= 0:
        modifiers.extend(["occupancy reduced", "empty floor emphasis"])
        return modifiers[:MAX_LANGUAGE_PASS_PROMPT_MODIFIERS]

    if figure_count == 1:
        if depth_band == "front":
            modifiers.append("single larger foreground figure")
        elif depth_band == "back":
            modifiers.append("smaller distant figure")
        else:
            modifiers.append("single mid-room figure")
    else:
        if layout_mode == "split":
            modifiers.append("two separated figures")
        else:
            modifiers.append("clustered figure grouping")

    if primary_zone == "left":
        modifiers.append("figure grouping shifted toward left side")
    elif primary_zone == "right":
        modifiers.append("figure grouping shifted toward right side")
    else:
        modifiers.append("one centered or slightly off-center figure emphasis")

    if depth_band == "front":
        modifiers.append("foreground weighting increased")
    elif depth_band == "back":
        modifiers.append("figures shifted toward back depth")
    else:
        modifiers.append("figures shifted toward middle depth")

    if layout_mode == "clustered":
        modifiers.append("tighter figure spacing")
    elif layout_mode == "split":
        modifiers.append("wider spacing between figures")
    else:
        modifiers.append("pose and silhouette may stay somewhat ambiguous")

    modifiers.append("room retained, occupancy pattern revised")
    return modifiers[:MAX_LANGUAGE_PASS_PROMPT_MODIFIERS]


def interpret_sen0628_spatial_estimate(sen_data):
    if not sen_data:
        return "SEN0628 spatial estimate unavailable"

    zone_close_counts = {
        "left": sen_data.get("left_close_points") or 0,
        "center": sen_data.get("center_close_points") or 0,
        "right": sen_data.get("right_close_points") or 0,
    }
    strongest_zone = max(zone_close_counts.items(), key=lambda item: item[1])[0]
    strongest_value = zone_close_counts[strongest_zone]
    active_zones = [name for name, count in zone_close_counts.items() if count >= 3]
    zone_distance_map = {
        "left": sen_data.get("left_zone_mm"),
        "center": sen_data.get("center_zone_mm"),
        "right": sen_data.get("right_zone_mm"),
    }
    strongest_distance = zone_distance_map.get(strongest_zone)

    if strongest_distance is None:
        depth_text = "with uncertain depth"
    elif strongest_distance < 900:
        depth_text = "close to the sensor plane"
    elif strongest_distance < 1800:
        depth_text = "in the mid-room"
    else:
        depth_text = "deeper in the room"

    if strongest_value < 2:
        return "SEN0628 suggests only a weak central trace"
    if len(active_zones) >= 3:
        return f"SEN0628 suggests occupancy spread across the room, {depth_text}"
    if len(active_zones) == 2:
        zone_pair = " and ".join(active_zones)
        return f"SEN0628 suggests occupancy shared across the {zone_pair} zones, {depth_text}"
    if strongest_zone == "left":
        return f"SEN0628 suggests occupancy concentrated on the left side, {depth_text}"
    if strongest_zone == "right":
        return f"SEN0628 suggests occupancy concentrated on the right side, {depth_text}"
    return f"SEN0628 suggests occupancy concentrated near the center, {depth_text}"


def sen0628_location_detail(sen_data):
    if not sen_data:
        return None

    zone_strengths = {
        "left": sen_data.get("left_close_points") or 0,
        "center": sen_data.get("center_close_points") or 0,
        "right": sen_data.get("right_close_points") or 0,
    }
    strongest_zone = max(zone_strengths.items(), key=lambda item: item[1])[0]
    strongest_value = zone_strengths[strongest_zone]
    sorted_strengths = sorted(zone_strengths.values(), reverse=True)
    second_value = sorted_strengths[1] if len(sorted_strengths) > 1 else 0
    active_zones = [name for name, value in zone_strengths.items() if value >= 3]

    if strongest_value < 2:
        return "with only a vague central trace"
    if len(active_zones) >= 3:
        return "spread across the full field of view"
    if len(active_zones) == 2:
        return "distributed across adjacent zones"
    if strongest_value - second_value <= 1 and strongest_value >= 3:
        return "hovering between zones"
    if strongest_zone == "left":
        return "biased toward the left side"
    if strongest_zone == "right":
        return "biased toward the right side"
    return "centered in front of the system"


def interpret_sen0628_figure_side(sen_data):
    if not sen_data:
        return "middle"

    zone_strengths = {
        "left": sen_data.get("left_close_points") or 0,
        "middle": sen_data.get("center_close_points") or 0,
        "right": sen_data.get("right_close_points") or 0,
    }
    strongest_zone = max(zone_strengths.items(), key=lambda item: item[1])[0]
    strongest_value = zone_strengths[strongest_zone]
    if strongest_value < 2:
        return "middle"
    return strongest_zone


def interpret_presence_location(ld_data, sen_data):
    if not ld_data and not sen_data:
        return "location uncertain"

    distance_cm = None
    if ld_data:
        distance_cm = ld_data.get("detection_distance_cm")
        if distance_cm is None:
            distance_cm = ld_data.get("moving_distance_cm") or ld_data.get("stationary_distance_cm")

    if distance_cm is None and sen_data:
        center_mm = sen_data.get("center_mm")
        mean_mm = sen_data.get("mean_mm")
        if center_mm is not None:
            distance_cm = center_mm / 10.0
        elif mean_mm is not None:
            distance_cm = mean_mm / 10.0

    if distance_cm is None:
        depth_phrase = "at an uncertain depth"
    elif distance_cm < 90:
        depth_phrase = "very close to the sensing system"
    elif distance_cm < 160:
        depth_phrase = "in the near field"
    elif distance_cm < 260:
        depth_phrase = "in the middle distance"
    else:
        depth_phrase = "farther back in the space"

    horizontal_phrase = sen0628_location_detail(sen_data)
    if horizontal_phrase is None:
        return depth_phrase
    return f"{depth_phrase}, {horizontal_phrase}"


def interpret_lighting(light_data):
    if not light_data:
        return "lighting uncertain"

    percent = light_data.get("percent")
    if percent is None:
        return "lighting uncertain"
    if percent < 18:
        return "dark"
    if percent < 40:
        return "dim"
    if percent < 70:
        return "moderate light"
    return "bright"


def interpret_atmosphere(bme_data):
    if not bme_data:
        return "atmosphere uncertain"

    temperature = bme_data.get("temperature_c")
    humidity = bme_data.get("humidity_pct")
    gas_ohms = bme_data.get("gas_ohms")

    if temperature is None or humidity is None or gas_ohms is None:
        return "atmosphere uncertain"
    if gas_ohms < 8000 or (temperature >= 25 and humidity >= 55):
        return "stale heavy atmosphere"
    if temperature >= 24 or humidity >= 60:
        return "warm dense air"
    if temperature < 21 and humidity < 45:
        return "cool dry air"
    return "neutral indoor air"


def interpret_spatial_impression(ld_data, sen_data):
    if not sen_data and not ld_data:
        return "spatial impression uncertain"

    if ld_data:
        target_state = ld_data.get("target_state")
        out_state = ld_data.get("out")
        if target_state == "NO_TARGET" and out_state == 0:
            return "open, empty, and waiting"

    near_points = (sen_data or {}).get("near_points") or 0
    far_points = (sen_data or {}).get("far_points") or 0
    zone_counts = [
        (sen_data or {}).get("left_close_points") or 0,
        (sen_data or {}).get("center_close_points") or 0,
        (sen_data or {}).get("right_close_points") or 0,
    ]
    occupied_zones = sum(1 for value in zone_counts if value >= 3)
    detection_distance = (ld_data or {}).get("detection_distance_cm")

    if near_points >= 10 and occupied_zones >= 2:
        return "shallow and fragmented"
    if far_points >= 20 and occupied_zones <= 1:
        return "deep and receding"
    if occupied_zones >= 3:
        return "wide and distributed"
    if detection_distance is not None and detection_distance < 120:
        return "compressed and near"
    return "contained and interior"


def interpret_abstract_background(light_data, bme_data):
    lighting = interpret_lighting(light_data)
    atmosphere = interpret_atmosphere(bme_data)

    if lighting == "dark":
        light_phrase = "broad dark backdrop with faint muted tonal shifts"
    elif lighting == "dim":
        light_phrase = "soft low-contrast gradients with large quiet open areas"
    elif lighting == "moderate light":
        light_phrase = "restrained diffused tones with pale breathing room"
    elif lighting == "bright":
        light_phrase = "washed pale planes with generous openness and faint bloom"
    else:
        light_phrase = "minimal ambiguous tonal fields with open depth"

    if atmosphere == "stale heavy atmosphere":
        air_phrase = "a compressed haze reduced to sparse smoky traces"
    elif atmosphere == "warm dense air":
        air_phrase = "warm suspended haze with softened edges and very little detail"
    elif atmosphere == "cool dry air":
        air_phrase = "clear separation with dry pale textures kept light and minimal"
    elif atmosphere == "neutral indoor air":
        air_phrase = "soft neutral texture barely present in the background"
    else:
        air_phrase = "indeterminate atmospheric traces kept faint and sparse"

    return (
        "abstract interior backdrop interpreted from sensor data, "
        f"{light_phrase}, {air_phrase}, desaturated tones, muted color, large open areas"
    )


def build_people_directive(descriptors):
    presence = descriptors["presence_activity"]
    presence_count = descriptors["presence_count"]
    figure_count = descriptors["figure_count"]
    layout_mode = descriptors.get("layout_mode", "empty")
    primary_zone = descriptors.get("primary_zone", "center")
    depth_band = descriptors.get("depth_band", "mid-room")

    if figure_count == 0 or presence == "no presence" or presence_count == "no people":
        return (
            "keep the installation space empty and unoccupied, preserve the same room with no visible person, "
            "no occupant, no silhouette, no crowd"
        )

    zone_text = {
        "left": "toward the left side",
        "center": "near the middle zone",
        "right": "toward the right side",
    }.get(str(primary_zone), "near the middle zone")
    depth_text = {
        "front": "leaning nearer to the foreground",
        "mid-room": "held around middle depth",
        "back": "leaning farther back in the room",
    }.get(str(depth_band), "held around middle depth")
    count_text = (
        "require two visible human figures as the occupancy revision"
        if figure_count == 2
        else "require one visible human figure as the occupancy revision"
    )
    spacing_text = (
        "allow the figures to separate across the room"
        if layout_mode == "split"
        else "allow tighter grouping between the figures"
        if layout_mode == "clustered"
        else "allow some silhouette ambiguity without removing the body presence"
    )
    return (
        f"human presence is being inferred as {presence}, {count_text}, {zone_text}, {depth_text}, "
        "the room must read as visibly occupied, the figures must be noticeable at first glance, "
        "render adult human bodies clearly enough to read immediately, not just traces, shadows, or implied presence, "
        "keep the figures legible in the frame with enough scale to stand out from the background, "
        f"{spacing_text}, vary the figures more than the room itself"
    )


def build_prompt_sections(descriptors):
    people_directive = build_people_directive(descriptors)
    background_continuity_directive = build_background_continuity_directive()
    figure_variation_directive = build_figure_variation_directive(descriptors)

    if descriptors["figure_count"] == 0:
        composition_directive = (
            f"{COMPOSITION_DIRECTIVE}, keep the same room and camera view, allow the room to remain empty in this no-presence case, "
            "preserve the empty installation-space look, discourage background novelty"
        )
        negative_prompt = (
            f"{OPTIONAL_TEXT_UI_EXCLUSION_NEGATIVE_PROMPT}, "
            "person, people, human, figure, body, face, silhouette, crowd, alternate room, different room, "
            "new architecture, changed camera angle, dramatic set dressing"
        )
    else:
        composition_directive = (
            f"{COMPOSITION_DIRECTIVE}, keep the same general room and camera view, the room should visibly contain occupants when presence is inferred, "
            "do not render an empty room when presence is inferred, keep background secondary to visible human presence, "
            "preserve the room while allowing figure arrangement to change, figures should occupy a noticeable amount "
            "of the frame, remain legible at first glance, and serve as the main changing element rather than tiny distant background details"
        )
        negative_prompt = (
            f"{OPTIONAL_TEXT_UI_EXCLUSION_NEGATIVE_PROMPT}, "
            "empty room, empty scene, unoccupied room, no people, no person, absent occupant, vacant interior, "
            "tiny distant people, people barely visible, occupants lost in the background, alternate room, different room, "
            "changed architecture, changed camera angle, new furniture"
        )

    return {
        "base_scene_prompt": BASE_SCENE_PROMPT,
        "background_continuity_directive": background_continuity_directive,
        "people_directive": people_directive,
        "figure_variation_directive": figure_variation_directive,
        "composition_directive": composition_directive,
        "negative_prompt": negative_prompt,
    }


def select_generation_seed(descriptors: dict[str, Any]) -> int:
    if descriptors.get("figure_count", 0) > 0:
        return DEFAULT_OCCUPIED_IMAGE_SEED
    return DEFAULT_EMPTY_IMAGE_SEED


def interpret_sensor_state(smoothed):
    presence_activity = interpret_presence(smoothed.get("ld2410c"))
    presence_count = estimate_presence_count(smoothed.get("ld2410c"), smoothed.get("sen0628"))
    people_layout = estimate_people_layout(
        smoothed.get("ld2410c"),
        smoothed.get("sen0628"),
        presence_activity,
        presence_count,
    )

    descriptors = {
        "presence_activity": presence_activity,
        "presence_count": presence_count,
        "sen0628_spatial_estimate": interpret_sen0628_spatial_estimate(smoothed.get("sen0628")),
        "sen0628_figure_side": interpret_sen0628_figure_side(smoothed.get("sen0628")),
        "presence_location": interpret_presence_location(smoothed.get("ld2410c"), smoothed.get("sen0628")),
        "lighting_condition": interpret_lighting(smoothed.get("light")),
        "atmospheric_condition": interpret_atmosphere(smoothed.get("bme688")),
        "spatial_impression": interpret_spatial_impression(smoothed.get("ld2410c"), smoothed.get("sen0628")),
        "abstract_background": interpret_abstract_background(
            smoothed.get("light"), smoothed.get("bme688")
        ),
        "figure_count": people_layout["figure_count"],
        "placement_summary": people_layout["placement_summary"],
        "placement_prompt": people_layout["placement_prompt"],
        "depth_band": people_layout["depth_band"],
        "primary_zone": people_layout["primary_zone"],
        "secondary_zone": people_layout["secondary_zone"],
        "active_zones": people_layout["active_zones"],
        "layout_mode": people_layout["layout_mode"],
    }
    descriptors["spatial_certainty"] = describe_spatial_certainty(
        presence_activity,
        list(descriptors["active_zones"]),
        str(descriptors["primary_zone"]),
    )
    descriptors["figure_variation_modifiers"] = build_figure_variation_modifiers(descriptors)
    return descriptors


def build_live_inference_lines(descriptors):
    presence_activity = descriptors["presence_activity"]
    lighting = descriptors["lighting_condition"]
    atmosphere = descriptors["atmospheric_condition"]
    depth_band = descriptors["depth_band"]
    primary_zone = descriptors["primary_zone"]
    figure_count = int(descriptors["figure_count"])
    layout_mode = descriptors["layout_mode"]
    spatial_certainty = descriptors["spatial_certainty"]

    movement_line_map = {
        "no presence": "occupancy weighting reduced",
        "still presence": "occupancy held steady",
        "active presence": "occupancy revision active",
        "intermittent movement": "occupancy state oscillating",
        "presence uncertain": "occupancy remains provisional",
    }
    light_line_map = {
        "dark": "low ambient field retained",
        "dim": "dim ambient field held",
        "moderate light": "ambient brightness stable",
        "bright": "brighter field retained",
        "lighting uncertain": "ambient light weakly resolved",
    }
    atmosphere_line_map = {
        "stale heavy atmosphere": "air density slightly elevated",
        "warm dense air": "warmer air bias retained",
        "cool dry air": "cool dry bias retained",
        "neutral indoor air": "atmosphere near neutral",
        "atmosphere uncertain": "atmospheric weighting minor",
    }

    if figure_count <= 0:
        occupancy_line = "room retained; figures reduced"
    elif figure_count == 1:
        occupancy_line = "single figure favored"
    elif layout_mode == "split":
        occupancy_line = "split figure layout favored"
    else:
        occupancy_line = "clustered figure state favored"

    if primary_zone == "left":
        lateral_line = "left trace dominant"
    elif primary_zone == "right":
        lateral_line = "right trace dominant"
    else:
        lateral_line = "central trace dominant"

    if depth_band == "front":
        depth_line = "near-field signal stronger"
    elif depth_band == "back":
        depth_line = "depth weighting shifted inward"
    else:
        depth_line = "mid-depth weighting held"

    lines = [
        "background held stable",
        movement_line_map.get(presence_activity, "signal weighting is being revised"),
        occupancy_line,
        lateral_line,
        depth_line,
        spatial_certainty,
        light_line_map.get(lighting, "ambient light remains under revision"),
        atmosphere_line_map.get(atmosphere, "air reading remains only partially resolved"),
    ]
    return lines


def build_background_continuity_directive() -> str:
    return (
        "preserve the same general installation room across generations, similar camera angle, similar framing, "
        "similar walls, floor, and architectural layout, keep environment structure consistent while allowing occupant revision, "
        "background continuity is a secondary anchor and the figures should remain the main changing element"
    )


def build_figure_variation_directive(descriptors: dict[str, Any]) -> str:
    modifiers = descriptors.get("figure_variation_modifiers") or []
    modifier_text = ", ".join(str(item) for item in modifiers if isinstance(item, str))
    if not modifier_text:
        modifier_text = "room retained, occupancy pattern revised"
    return (
        "treat human figures as the primary changing element, allow variation in lateral placement, depth placement, spacing, "
        f"scale, clustering, and separation while keeping the room broadly stable, {modifier_text}"
    )


def build_image_prompt(descriptors):
    sections = build_prompt_sections(descriptors)
    if int(descriptors.get("figure_count", 0)) > 0:
        ordered_sections = [
            sections["people_directive"],
            sections["figure_variation_directive"],
            sections["base_scene_prompt"],
            sections["background_continuity_directive"],
            sections["composition_directive"],
        ]
    else:
        ordered_sections = [
            sections["base_scene_prompt"],
            sections["background_continuity_directive"],
            sections["composition_directive"],
            sections["people_directive"],
            sections["figure_variation_directive"],
        ]
    return ", ".join(ordered_sections)


def build_rule_based_scene_plan(smoothed: dict[str, Any]) -> dict[str, Any]:
    descriptors = interpret_sensor_state(smoothed)
    prompt_sections = build_prompt_sections(descriptors)
    prompt = build_image_prompt(descriptors)
    live_lines = build_live_inference_lines(descriptors)
    return {
        "descriptors": descriptors,
        "prompt_sections": prompt_sections,
        "prompt": prompt,
        "live_lines": live_lines,
        "interpretation_source": "rule_based",
        "agent_preview": None,
        "agent_notes": [],
        "state_signature_descriptors": descriptors,
    }


def sanitize_text_value(value: Any, fallback: str) -> str:
    if not isinstance(value, str):
        return fallback
    cleaned = " ".join(value.split()).strip()
    return cleaned or fallback


def compact_machine_phrase(value: str, *, max_length: int) -> str:
    cleaned = " ".join(value.split()).strip().lower()
    cleaned = re.sub(r"[.!?]+$", "", cleaned)
    cleaned = re.sub(
        r"\b(the|a|an|currently|now|next image|in the current reading|in the next image)\b",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*;\s*", "; ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,.;:-")
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip(" ,.;:-")
    return cleaned


def sanitize_live_lines(candidate_lines: Any, fallback_lines: list[str]) -> list[str]:
    if not isinstance(candidate_lines, list):
        return fallback_lines

    cleaned_lines: list[str] = []
    for item in candidate_lines:
        if not isinstance(item, str):
            continue
        cleaned = compact_machine_phrase(item, max_length=44)
        if cleaned:
            cleaned_lines.append(cleaned)

    return cleaned_lines[:MAX_AI_LIVE_LINES] or fallback_lines


def build_language_pass_fallback_preview(fallback_plan: dict[str, Any]) -> str:
    fallback_descriptors = fallback_plan["descriptors"]
    if fallback_descriptors.get("figure_count", 0) > 0:
        return "upcoming image preview: " + fallback_descriptors["placement_summary"]
    return "upcoming image preview: empty installation space"


def sanitize_short_string_list(
    value: Any,
    *,
    fallback: list[str],
    min_items: int,
    max_items: int,
    max_length: int,
) -> list[str]:
    if not isinstance(value, list):
        return fallback

    cleaned_items: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        cleaned = compact_machine_phrase(item, max_length=max_length)
        if not cleaned:
            continue
        if cleaned not in cleaned_items:
            cleaned_items.append(cleaned)

    if len(cleaned_items) < min_items:
        return fallback
    return cleaned_items[:max_items]


def sanitize_prompt_modifiers(value: Any, fallback: list[str], *, force_occupied: bool = False) -> list[str]:
    cleaned_items = sanitize_short_string_list(
        value,
        fallback=fallback,
        min_items=0,
        max_items=MAX_LANGUAGE_PASS_PROMPT_MODIFIERS,
        max_length=72,
    )
    blocked_terms = (
        "different room",
        "alternate room",
        "new architecture",
        "camera angle change",
        "bedroom",
        "kitchen",
        "office",
        "surveillance",
        "security camera",
    )
    occupied_blocked_terms = (
        "empty room",
        "empty scene",
        "unoccupied",
        "vacant",
        "no people",
        "no person",
        "absent occupant",
        "tiny distant people",
        "barely visible people",
    )
    sanitized_items: list[str] = []
    for item in cleaned_items:
        compact = re.sub(r"[^a-zA-Z0-9 ,\\-/]", "", item).strip(" ,.;:-")
        if not compact:
            continue
        lowered = compact.lower()
        if any(term in lowered for term in blocked_terms):
            continue
        if force_occupied and any(term in lowered for term in occupied_blocked_terms):
            continue
        if compact not in sanitized_items:
            sanitized_items.append(compact)
    return sanitized_items[:MAX_LANGUAGE_PASS_PROMPT_MODIFIERS]


def sanitize_openai_language_pass(
    language_pass: dict[str, Any],
    fallback_plan: dict[str, Any],
) -> dict[str, Any]:
    fallback_preview = build_language_pass_fallback_preview(fallback_plan)
    live_lines = sanitize_short_string_list(
        language_pass.get("live_inference_lines"),
        fallback=fallback_plan["live_lines"],
        min_items=MIN_LANGUAGE_PASS_LIVE_LINES,
        max_items=MAX_LANGUAGE_PASS_LIVE_LINES,
        max_length=44,
    )
    agent_notes = sanitize_short_string_list(
        language_pass.get("agent_notes"),
        fallback=[],
        min_items=MIN_LANGUAGE_PASS_AGENT_NOTES,
        max_items=MAX_LANGUAGE_PASS_AGENT_NOTES,
        max_length=40,
    )
    prompt_modifiers = sanitize_prompt_modifiers(
        language_pass.get("prompt_modifiers"),
        fallback=fallback_plan["descriptors"].get("figure_variation_modifiers", []),
        force_occupied=int(fallback_plan["descriptors"].get("figure_count", 0)) > 0,
    )
    image_preview = compact_machine_phrase(
        sanitize_text_value(language_pass.get("image_preview"), fallback_preview),
        max_length=52,
    )
    return {
        "live_inference_lines": live_lines,
        "image_preview": image_preview,
        "agent_notes": agent_notes,
        "prompt_modifiers": prompt_modifiers,
    }


def extract_json_object_from_text(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    if isinstance(parsed, dict):
        return parsed
    return None


def extract_responses_api_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]

    output = payload.get("output")
    if not isinstance(output, list):
        return ""

    text_chunks: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for content_item in content:
            if not isinstance(content_item, dict):
                continue
            if content_item.get("type") == "output_text" and isinstance(content_item.get("text"), str):
                text_chunks.append(content_item["text"])
    return "\n".join(text_chunks).strip()


def build_llm_interpretation_payload(
    smoothed: dict[str, Any],
    fallback_plan: dict[str, Any],
    empty_room_baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    descriptors = fallback_plan["descriptors"]
    empty_room_comparison = build_empty_room_comparison(smoothed, empty_room_baseline)
    top_changed_metrics = []
    if isinstance(empty_room_comparison, dict):
        changed_metrics = empty_room_comparison.get("top_changed_metrics")
        if isinstance(changed_metrics, list):
            top_changed_metrics = changed_metrics[:MAX_BASELINE_CHANGED_METRICS]

    return {
        "deterministic_state": {
            "figure_count": descriptors["figure_count"],
            "presence_activity": descriptors["presence_activity"],
            "presence_count": descriptors["presence_count"],
            "placement_summary": descriptors["placement_summary"],
            "placement_prompt": descriptors["placement_prompt"],
            "presence_location": descriptors["presence_location"],
            "lighting_condition": descriptors["lighting_condition"],
            "atmospheric_condition": descriptors["atmospheric_condition"],
            "spatial_impression": descriptors["spatial_impression"],
            "sen0628_spatial_estimate": descriptors["sen0628_spatial_estimate"],
            "sen0628_figure_side": descriptors["sen0628_figure_side"],
            "depth_band": descriptors["depth_band"],
            "primary_zone": descriptors["primary_zone"],
            "layout_mode": descriptors["layout_mode"],
            "spatial_certainty": descriptors["spatial_certainty"],
            "figure_variation_modifiers": descriptors["figure_variation_modifiers"],
        },
        "baseline_comparison": {
            "available": bool(empty_room_comparison),
            "top_changed_metrics": top_changed_metrics,
            "departure_level": classify_baseline_departure(empty_room_comparison),
        },
        "prompt_intent": {
            "background_priority": "keep the room and camera stable across generations",
            "figure_priority": "allow figures to vary more than the room",
            "live_text_style": (
                "emit terse machine-status fragments about weighting, ambiguity, occupancy revision, "
                "figure placement, and room continuity"
            ),
        },
        "raw_sensor_summary": {
            "ld2410c": smoothed.get("ld2410c"),
            "sen0628": smoothed.get("sen0628"),
            "light": smoothed.get("light"),
            "bme688": smoothed.get("bme688"),
        },
    }


def fetch_openai_language_pass(
    openai_settings: dict[str, Any],
    smoothed: dict[str, Any],
    fallback_plan: dict[str, Any],
    empty_room_baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = build_llm_interpretation_payload(smoothed, fallback_plan, empty_room_baseline)
    system_prompt = (
        "You are a language pass for an installation-space sensing system. "
        "The deterministic_state is the source of truth. "
        "Do not increase figure_count. "
        "If deterministic figure_count is greater than 0, do not erase or weaken occupancy. "
        "When deterministic figure_count is greater than 0, keep visible human figures explicit in wording and do not drift toward an empty room. "
        "Do not contradict placement_summary. "
        "Do not override presence_count, presence_activity, or major placement logic. "
        "Do not invent room changes, camera changes, furniture, or new architecture. "
        "If baseline departure is minimal or weak, prefer caution, uncertainty, and partial interpretation. "
        "Keep the tone machine-like, restrained, observational, and concise. "
        "Avoid poetic excess, storytelling, surveillance language, and explanation. "
        "Write as terse status fragments, not sentences or prose. "
        "Favor wording about tendencies, shifts, ambiguity, continuity, occupancy revision, figure placement, figure scale, and room preservation. "
        "Do not state room contents as hard facts unless certainty is extremely strong. "
        "Keep phrasing projector-friendly and glance-readable. "
        "Return JSON only with these keys: live_inference_lines, image_preview, agent_notes, prompt_modifiers. "
        "Rules: live_inference_lines must be 3 to 4 very short lines; "
        "each live line should usually be 2 to 6 words and under 44 characters; "
        "image_preview must be 1 very short fragment; "
        "agent_notes should be 0 to 1 short technical notes; "
        "prompt_modifiers must be 2 to 4 short visual phrases that can be appended to a prompt; "
        "prompt_modifiers should prioritize visible figure variation such as count, spacing, clustering, side bias, depth, and scale while keeping the room broadly consistent; "
        "keep prompt_modifiers concise rather than prose; "
        "prefer phrases like 'left trace dominant', 'background held stable', 'split figure layout favored'; "
        "do not include markdown fences."
    )
    request_payload = {
        "model": str(openai_settings.get("model", "")),
        "max_output_tokens": 180,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(payload, indent=2),
                    }
                ],
            },
        ],
    }

    timeout_seconds = openai_settings.get("timeout_seconds", OPENAI_RESPONSE_TIMEOUT_FALLBACK_SECONDS)
    try:
        timeout_value = float(timeout_seconds)
    except (TypeError, ValueError):
        timeout_value = float(OPENAI_RESPONSE_TIMEOUT_FALLBACK_SECONDS)

    response = requests.post(
        f"{str(openai_settings.get('base_url', 'https://api.openai.com/v1')).rstrip('/')}/responses",
        headers={
            "Authorization": f"Bearer {openai_settings['api_key']}",
            "Content-Type": "application/json",
        },
        json=request_payload,
        timeout=timeout_value,
    )
    response.raise_for_status()
    response_payload = response.json()
    response_text = extract_responses_api_text(response_payload)
    ai_plan = extract_json_object_from_text(response_text)
    if ai_plan is None:
        raise ValueError("OpenAI interpreter did not return valid JSON")
    return ai_plan


def build_final_prompt_from_language_pass(
    fallback_plan: dict[str, Any],
    language_pass: dict[str, Any],
) -> str:
    prompt_sections = fallback_plan["prompt_sections"]
    figure_count = int(fallback_plan["descriptors"].get("figure_count", 0))
    if figure_count > 0:
        ordered_sections = [
            prompt_sections["people_directive"],
            prompt_sections["figure_variation_directive"],
            prompt_sections["base_scene_prompt"],
            prompt_sections["background_continuity_directive"],
            prompt_sections["composition_directive"],
        ]
    else:
        ordered_sections = [
            prompt_sections["base_scene_prompt"],
            prompt_sections["background_continuity_directive"],
            prompt_sections["composition_directive"],
            prompt_sections["people_directive"],
            prompt_sections["figure_variation_directive"],
        ]

    sanitized_modifiers = sanitize_prompt_modifiers(
        language_pass.get("prompt_modifiers"),
        fallback=fallback_plan["descriptors"].get("figure_variation_modifiers", []),
        force_occupied=figure_count > 0,
    )
    return ", ".join(ordered_sections + sanitized_modifiers)


def build_scene_plan(
    smoothed: dict[str, Any],
    openai_settings: dict[str, Any] | None,
    empty_room_baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    fallback_plan = build_rule_based_scene_plan(smoothed)
    if not openai_settings or not openai_settings.get("enabled"):
        return fallback_plan

    try:
        language_pass = fetch_openai_language_pass(
            openai_settings, smoothed, fallback_plan, empty_room_baseline
        )
        sanitized_language_pass = sanitize_openai_language_pass(language_pass, fallback_plan)
        live_lines = sanitized_language_pass["live_inference_lines"]
        agent_preview = sanitized_language_pass["image_preview"]
        if len(live_lines) > MAX_AI_LIVE_LINES:
            live_lines = live_lines[:MAX_AI_LIVE_LINES]

        result = dict(fallback_plan)
        result["live_lines"] = live_lines
        result["agent_preview"] = agent_preview
        result["agent_notes"] = sanitized_language_pass["agent_notes"]
        result["prompt"] = build_final_prompt_from_language_pass(
            fallback_plan, sanitized_language_pass
        )
        result["interpretation_source"] = "openai_language_pass"
        return result
    except Exception as exc:
        fallback_result = dict(fallback_plan)
        fallback_result["openai_error"] = str(exc)
        return fallback_result


def extract_change_metrics(smoothed):
    light_percent = ((smoothed.get("light") or {}).get("percent"))
    bme_data = smoothed.get("bme688") or {}
    ld_data = smoothed.get("ld2410c") or {}
    detection_distance_cm = ld_data.get("detection_distance_cm")
    if detection_distance_cm is None:
        detection_distance_cm = ld_data.get("moving_distance_cm") or ld_data.get("stationary_distance_cm")

    return {
        "light_percent": light_percent,
        "temperature_c": bme_data.get("temperature_c"),
        "humidity_pct": bme_data.get("humidity_pct"),
        "detection_distance_cm": detection_distance_cm,
    }


def build_state_signature(descriptors):
    combined_text = " ".join(
        str(descriptors.get(field_name, ""))
        for field_name in (
            "placement_summary",
            "placement_prompt",
            "sen0628_figure_side",
            "presence_location",
        )
    ).lower()

    lateral_bucket = "middle"
    if "left" in combined_text and "right" in combined_text:
        lateral_bucket = "left-right"
    elif "left" in combined_text and "center" in combined_text:
        lateral_bucket = "left-center"
    elif "right" in combined_text and "center" in combined_text:
        lateral_bucket = "center-right"
    elif "left" in combined_text:
        lateral_bucket = "left"
    elif "right" in combined_text:
        lateral_bucket = "right"
    elif "center" in combined_text or "middle" in combined_text:
        lateral_bucket = "center"

    depth_bucket = "mid-room"
    if "front" in combined_text or "near field" in combined_text or "near the sensor" in combined_text:
        depth_bucket = "front"
    elif "back" in combined_text or "farther back" in combined_text or "back wall" in combined_text:
        depth_bucket = "back"
    elif "middle" in combined_text or "mid-room" in combined_text or "mid-distance" in combined_text:
        depth_bucket = "mid-room"

    signature_fields = {
        "presence_activity": descriptors["presence_activity"],
        "presence_count": descriptors["presence_count"],
        "figure_count": descriptors["figure_count"],
        "lateral_bucket": lateral_bucket,
        "depth_bucket": depth_bucket,
    }
    return json.dumps(signature_fields, sort_keys=True)


def change_exceeds_threshold(previous_metrics, current_metrics):
    if previous_metrics is None:
        return True

    threshold_map = {
        "light_percent": LIGHT_CHANGE_THRESHOLD,
        "temperature_c": TEMPERATURE_CHANGE_THRESHOLD,
        "humidity_pct": HUMIDITY_CHANGE_THRESHOLD,
        "detection_distance_cm": DISTANCE_CHANGE_THRESHOLD_CM,
    }
    for key, threshold in threshold_map.items():
        previous_value = previous_metrics.get(key)
        current_value = current_metrics.get(key)
        if previous_value is None or current_value is None:
            continue
        if abs(current_value - previous_value) >= threshold:
            return True
    return False


def build_shared_state_payload(
    *,
    raw_frames,
    smoothed,
    descriptors,
    live_lines,
    prompt,
    prompt_sections,
    interpretation_source,
    agent_preview,
    agent_notes,
    openai_error,
    coordinator,
    now,
    generated_image_path,
) -> dict[str, Any]:
    previous_shared_state = coordinator.latest_shared_state or {}
    return {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "timing": {
            "text_update_interval_seconds": TEXT_UPDATE_INTERVAL_SECONDS,
            "image_generation_interval_seconds": IMAGE_GENERATION_INTERVAL_SECONDS,
            "force_image_refresh_seconds": FORCE_IMAGE_REFRESH_SECONDS,
            "state_stable_hold_seconds": STATE_STABLE_HOLD_SECONDS,
        },
        "raw_frame_summary": {
            "frame_count": len(raw_frames),
            "latest_seq": raw_frames[-1].get("seq") if raw_frames else None,
        },
        "smoothed_sensor_values": smoothed,
        "interpreted_state": descriptors,
        "live_inference_lines": live_lines,
        "interpretation_source": interpretation_source,
        "agent_preview": agent_preview,
        "agent_notes": agent_notes,
        "image_prompt": prompt,
        "prompt_sections": prompt_sections,
        "state_signature": coordinator.current_signature,
        "stable_signature": coordinator.stable_signature,
        "last_meaningful_change_time": coordinator.last_meaningful_change_time,
        "last_text_update_time": coordinator.last_text_update_time,
        "last_image_generation_time": coordinator.last_image_generation_time,
        "current_image_path": resolve_image_path_value(generated_image_path),
        "current_image_seed": previous_shared_state.get(
            "current_image_seed", select_generation_seed(descriptors)
        ),
        "seed_test_mode": previous_shared_state.get("seed_test_mode", SEED_TEST_MODE),
        "last_image_error": previous_shared_state.get("last_image_error"),
        "last_openai_error": openai_error,
        "seconds_since_meaningful_change": round(max(0.0, now - coordinator.last_meaningful_change_time), 2),
    }

def resolve_image_path_value(image_path):
    if image_path is None:
        return None
    if isinstance(image_path, Path):
        return str(image_path.resolve())
    return str(Path(image_path).resolve())


def list_generated_image_files() -> list[Path]:
    if not OUTPUT_DIR.exists():
        return []

    image_files = [
        path
        for path in OUTPUT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    return sorted(image_files, key=lambda path: path.stat().st_mtime, reverse=True)


def delete_two_generations_ago_image() -> Path | None:
    image_files = list_generated_image_files()
    if len(image_files) < 3:
        return None

    stale_image_path = image_files[2]
    stale_log_path = stale_image_path.with_suffix(".txt")
    stale_metadata_path = stale_image_path.with_suffix(".json")

    try:
        stale_image_path.unlink()
        print(f"[CLEANUP] Deleted stale image: {stale_image_path}")
    except FileNotFoundError:
        return None

    if stale_log_path.exists():
        try:
            stale_log_path.unlink()
            print(f"[CLEANUP] Deleted stale log:   {stale_log_path}")
        except OSError as exc:
            print(f"[WARN] Could not delete stale log {stale_log_path}: {exc}", file=sys.stderr)

    if stale_metadata_path.exists():
        try:
            stale_metadata_path.unlink()
            print(f"[CLEANUP] Deleted stale metadata: {stale_metadata_path}")
        except OSError as exc:
            print(f"[WARN] Could not delete stale metadata {stale_metadata_path}: {exc}", file=sys.stderr)

    return stale_image_path


def slugify_text(value: str) -> str:
    allowed = []
    for character in value.lower():
        if character.isalnum():
            allowed.append(character)
        elif character in {" ", "-", "_"}:
            allowed.append("-")
    slug = "".join(allowed).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "unspecified"


def build_descriptor_slug(descriptors: dict[str, Any]) -> str:
    presence_count = slugify_text(str(descriptors.get("presence_count", "presence-uncertain")))
    figure_side = slugify_text(str(descriptors.get("sen0628_figure_side", "middle")))
    return f"presence-{presence_count}_{figure_side}"


def make_output_stem(
    *,
    output_dir: Path = OUTPUT_DIR,
    seed: int | None = None,
    descriptors: dict[str, Any] | None = None,
    timestamp: datetime | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_value = timestamp or datetime.now()
    timestamp_text = timestamp_value.strftime("%Y-%m-%d_%H-%M-%S")
    stem_parts = [timestamp_text]
    if seed is not None:
        stem_parts.append(f"seed-{seed}")
    if descriptors:
        stem_parts.append(build_descriptor_slug(descriptors))
    return output_dir / "_".join(stem_parts)


def build_generation_metadata(
    *,
    raw_frames,
    smoothed,
    descriptors,
    prompt,
    prompt_sections,
    seed: int,
    seed_test_mode: bool,
    timestamp: datetime,
) -> dict[str, Any]:
    return {
        "timestamp": timestamp.isoformat(timespec="seconds"),
        "seed": seed,
        "seed_test_mode": seed_test_mode,
        "final_prompt": prompt,
        "prompt_sections": prompt_sections,
        "negative_prompt": prompt_sections.get("negative_prompt", OPTIONAL_TEXT_UI_EXCLUSION_NEGATIVE_PROMPT),
        "interpreted_descriptors": descriptors,
        "smoothed_sensor_values": smoothed,
        "raw_sensor_frames": raw_frames,
    }


def add_debug_overlay(
    image: Image.Image,
    *,
    timestamp: datetime,
    seed: int,
    presence_count: str,
    sen0628_figure_side: str,
) -> Image.Image:
    debug_image = image.convert("RGBA")
    draw = ImageDraw.Draw(debug_image, "RGBA")
    font = ImageFont.load_default()
    lines = [
        timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        f"seed: {seed}",
        f"presence_count: {presence_count}",
        f"sen0628_figure_side: {sen0628_figure_side}",
    ]
    padding_x = 8
    padding_y = 6
    line_gap = 3
    margin = 10

    text_boxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    text_width = max(box[2] - box[0] for box in text_boxes)
    text_height = sum((box[3] - box[1]) for box in text_boxes) + line_gap * (len(lines) - 1)

    rect_width = text_width + (padding_x * 2)
    rect_height = text_height + (padding_y * 2)
    rect_left = debug_image.width - rect_width - margin
    rect_top = debug_image.height - rect_height - margin
    rect_right = rect_left + rect_width
    rect_bottom = rect_top + rect_height

    draw.rectangle((rect_left, rect_top, rect_right, rect_bottom), fill=(0, 0, 0, 170))

    text_y = rect_top + padding_y
    for line, box in zip(lines, text_boxes):
        draw.text((rect_left + padding_x, text_y), line, font=font, fill=(255, 255, 255, 255))
        text_y += (box[3] - box[1]) + line_gap

    return debug_image.convert("RGB")


def save_generation_artifacts(
    image,
    *,
    raw_frames,
    smoothed,
    descriptors,
    prompt,
    prompt_sections,
    seed: int,
    seed_test_mode: bool,
    output_dir: Path = OUTPUT_DIR,
    timestamp: datetime | None = None,
):
    generation_time = timestamp or datetime.now()
    stem = make_output_stem(
        output_dir=output_dir,
        seed=seed,
        descriptors=descriptors,
        timestamp=generation_time,
    )
    image_path = stem.with_suffix(".png")
    metadata_path = stem.with_suffix(".json")
    log_path = stem.with_suffix(".txt")

    debug_image = add_debug_overlay(
        image,
        timestamp=generation_time,
        seed=seed,
        presence_count=str(descriptors.get("presence_count", "unknown")),
        sen0628_figure_side=str(descriptors.get("sen0628_figure_side", "unknown")),
    )
    debug_image.save(image_path)

    metadata = build_generation_metadata(
        raw_frames=raw_frames,
        smoothed=smoothed,
        descriptors=descriptors,
        prompt=prompt,
        prompt_sections=prompt_sections,
        seed=seed,
        seed_test_mode=seed_test_mode,
        timestamp=generation_time,
    )
    atomic_write_text(metadata_path, json.dumps(metadata, indent=2))

    log_text = "\n".join(
        [
            "generation_timestamp: {}".format(generation_time.isoformat(timespec="seconds")),
            f"seed: {seed}",
            f"seed_test_mode: {seed_test_mode}",
            "",
            "raw_sensor_frames:",
            json_text(raw_frames),
            "",
            "smoothed_sensor_values:",
            json_text(smoothed),
            "",
            "interpreted_descriptors:",
            json_text(descriptors),
            "",
            "prompt_sections:",
            json_text(prompt_sections),
            "",
            "final_prompt:",
            prompt,
            "",
            "negative_prompt:",
            prompt_sections.get("negative_prompt", OPTIONAL_TEXT_UI_EXCLUSION_NEGATIVE_PROMPT),
        ]
    )
    atomic_write_text(log_path, log_text)
    return image_path, log_path, metadata_path


def generate_image(client, prompt: str, negative_prompt: str, seed: int):
    # The locally installed InferenceClient exposes a `seed` keyword on text_to_image().
    # If the HF API method changes in a different environment, this is the call site to update.
    return client.text_to_image(
        prompt,
        negative_prompt=negative_prompt,
        model=MODEL_ID,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        guidance_scale=IMAGE_GUIDANCE_SCALE,
        num_inference_steps=IMAGE_NUM_INFERENCE_STEPS,
        seed=seed,
    )


def build_seed_test_folder(timestamp: datetime | None = None) -> Path:
    folder_time = timestamp or datetime.now()
    folder_name = folder_time.strftime("%Y-%m-%d_%H-%M-%S")
    folder_path = SEED_TEST_OUTPUT_DIR / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path


def build_seed_test_index(
    *,
    output_dir: Path,
    prompt: str,
    descriptors: dict[str, Any],
    seeds: list[int],
    generated_filenames: list[str],
) -> Path:
    index_payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seeds": seeds,
        "prompt": prompt,
        "descriptor_snapshot": descriptors,
        "generated_filenames": generated_filenames,
    }
    index_path = output_dir / SEED_INDEX_FILENAME
    atomic_write_text(index_path, json.dumps(index_payload, indent=2))
    return index_path


def create_seed_contact_sheet(image_paths: list[Path], output_dir: Path) -> Path | None:
    if not image_paths:
        return None

    try:
        thumbnail_size = (256, 256)
        padding = 24
        columns = min(4, len(image_paths))
        rows = (len(image_paths) + columns - 1) // columns
        caption_height = 28
        sheet = Image.new(
            "RGB",
            (
                columns * thumbnail_size[0] + (columns + 1) * padding,
                rows * (thumbnail_size[1] + caption_height) + (rows + 1) * padding,
            ),
            "#101010",
        )
        draw = ImageDraw.Draw(sheet)

        for index, image_path in enumerate(image_paths):
            column = index % columns
            row = index // columns
            x = padding + column * (thumbnail_size[0] + padding)
            y = padding + row * (thumbnail_size[1] + caption_height + padding)

            with Image.open(image_path) as source_image:
                thumb = ImageOps.fit(source_image.convert("RGB"), thumbnail_size, Image.Resampling.LANCZOS)
            sheet.paste(thumb, (x, y))
            draw.text((x, y + thumbnail_size[1] + 6), image_path.stem, fill="#f0f0f0")

        contact_sheet_path = output_dir / CONTACT_SHEET_FILENAME
        sheet.save(contact_sheet_path)
        return contact_sheet_path
    except Exception as exc:
        print(f"[WARN] Could not build seed contact sheet: {exc}", file=sys.stderr)
        return None


def wait_for_initial_frames(ser, frame_buffer):
    deadline = time.time() + SERIAL_STARTUP_SECONDS

    while time.time() < deadline and len(frame_buffer) < MIN_FRAMES_FOR_PROCESSING:
        raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not raw_line:
            continue

        print(f"[PICO] {raw_line}")
        frame = parse_sensor_line(raw_line)
        if frame is not None:
            frame_buffer.append(frame)

    return len(frame_buffer) >= MIN_FRAMES_FOR_PROCESSING


def should_update_text(coordinator, now, state_changed):
    if coordinator.latest_shared_state is None:
        return True
    if state_changed:
        return True
    return now - coordinator.last_text_update_time >= TEXT_UPDATE_INTERVAL_SECONDS


def format_debug_timestamp(timestamp: float | None) -> str:
    if timestamp is None:
        return "None"
    return datetime.fromtimestamp(timestamp).isoformat(timespec="seconds")


def is_pid_active(pid: int) -> bool:
    if pid <= 0:
        return False

    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True


def write_lock_file() -> None:
    payload = {
        "pid": os.getpid(),
        "script_path": str(Path(__file__).resolve()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    atomic_write_text(LOCK_FILE_PATH, json.dumps(payload, indent=2))


def cleanup_lock_file() -> None:
    try:
        if not LOCK_FILE_PATH.exists():
            return

        payload = json.loads(LOCK_FILE_PATH.read_text(encoding="utf-8"))
        if payload.get("pid") == os.getpid():
            LOCK_FILE_PATH.unlink(missing_ok=True)
    except (OSError, json.JSONDecodeError, AttributeError):
        pass


def acquire_single_instance_lock() -> None:
    if LOCK_FILE_PATH.exists():
        try:
            payload = json.loads(LOCK_FILE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None

        if isinstance(payload, dict):
            existing_pid = payload.get("pid")
            if isinstance(existing_pid, int) and is_pid_active(existing_pid):
                print("Another generator instance is already running.")
                sys.exit(1)

        try:
            LOCK_FILE_PATH.unlink(missing_ok=True)
        except OSError:
            pass

    write_lock_file()
    atexit.register(cleanup_lock_file)


def print_startup_debug() -> None:
    print(f"[STARTUP] PID: {os.getpid()}")
    print(f"[STARTUP] Script: {Path(__file__).resolve()}")
    print(f"[STARTUP] OUTPUT_DIR: {OUTPUT_DIR.resolve()}")
    print(f"[STARTUP] DEFAULT_EMPTY_IMAGE_SEED: {DEFAULT_EMPTY_IMAGE_SEED}")
    print(f"[STARTUP] DEFAULT_OCCUPIED_IMAGE_SEED: {DEFAULT_OCCUPIED_IMAGE_SEED}")
    print(
        "[STARTUP] IMAGE_GENERATION_INTERVAL_SECONDS: "
        f"{IMAGE_GENERATION_INTERVAL_SECONDS}"
    )


def should_generate_new_image(coordinator, now):
    seconds_since_last_image_generation = now - coordinator.last_image_generation_time
    seconds_since_meaningful_change = now - coordinator.last_meaningful_change_time
    decision = False
    reason = "scheduled interval not reached"

    if coordinator.latest_shared_state is None:
        reason = "no shared state yet"
    elif coordinator.image_generation_in_progress:
        reason = "generation already in progress"
    elif seconds_since_last_image_generation < IMAGE_GENERATION_INTERVAL_SECONDS:
        reason = "image interval not reached"
    elif seconds_since_meaningful_change < STATE_STABLE_HOLD_SECONDS:
        reason = "state stability hold not reached"
    elif coordinator.stable_signature != coordinator.last_image_signature:
        decision = True
        reason = "state signature changed"
    elif seconds_since_last_image_generation >= FORCE_IMAGE_REFRESH_SECONDS:
        decision = True
        reason = "scheduled timed refresh"

    print(
        "[IMAGE_CHECK] "
        f"pid={os.getpid()} "
        f"decision={'True' if decision else 'False'} "
        f"reason={reason} "
        f"current_time={format_debug_timestamp(now)} "
        f"last_image_generation_time={format_debug_timestamp(coordinator.last_image_generation_time)} "
        f"seconds_since_last_image_generation={seconds_since_last_image_generation:.3f} "
        f"last_meaningful_change_time={format_debug_timestamp(coordinator.last_meaningful_change_time)} "
        f"seconds_since_meaningful_change={seconds_since_meaningful_change:.3f} "
        f"stable_signature={coordinator.stable_signature!r} "
        f"last_image_signature={coordinator.last_image_signature!r} "
        f"image_generation_in_progress={coordinator.image_generation_in_progress}"
    )
    return decision


def update_shared_state_file(shared_state: dict[str, Any]) -> None:
    atomic_write_text(SHARED_STATE_PATH, json.dumps(shared_state, indent=2))


def load_shared_state_from_disk() -> dict[str, Any] | None:
    if not SHARED_STATE_PATH.exists():
        return None

    try:
        return json.loads(SHARED_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def generate_and_save_image(
    client,
    shared_state: dict[str, Any],
    raw_frames,
    *,
    seed: int,
    seed_test_mode: bool,
    output_dir: Path = OUTPUT_DIR,
):
    prompt = shared_state["image_prompt"]
    prompt_sections = shared_state.get("prompt_sections") or build_prompt_sections(
        shared_state["interpreted_state"]
    )
    negative_prompt = prompt_sections.get("negative_prompt", OPTIONAL_TEXT_UI_EXCLUSION_NEGATIVE_PROMPT)
    smoothed = shared_state["smoothed_sensor_values"]
    descriptors = shared_state["interpreted_state"]

    print(f"[INFER] {json.dumps(descriptors)}")
    print(f"[PROMPT] {prompt}")
    print(f"[NEGATIVE_PROMPT] {negative_prompt}")
    print(f"[SEED] {seed}")

    image = generate_image(client, prompt, negative_prompt, seed)
    image_path, log_path, metadata_path = save_generation_artifacts(
        image=image,
        raw_frames=raw_frames,
        smoothed=smoothed,
        descriptors=descriptors,
        prompt=prompt,
        prompt_sections=prompt_sections,
        seed=seed,
        seed_test_mode=seed_test_mode,
        output_dir=output_dir,
    )
    print(
        "[IMAGE_SAVE] "
        f"pid={os.getpid()} "
        f"timestamp={datetime.now().isoformat(timespec='seconds')} "
        f"image_path={image_path.resolve()}"
    )
    print(f"[SAVED] Image: {image_path}")
    print(f"[SAVED] Log:   {log_path}")
    print(f"[SAVED] Meta:  {metadata_path}")
    if not seed_test_mode:
        delete_two_generations_ago_image()
    return image_path


def generate_seed_comparison_set(
    client,
    *,
    shared_state: dict[str, Any],
    raw_frames,
    seeds: list[int],
) -> list[Path]:
    # Seed test mode freezes one interpreted state and one final prompt, then varies only the seed.
    shared_state_snapshot = json.loads(json.dumps(shared_state))
    raw_frames_snapshot = json.loads(json.dumps(raw_frames))
    seed_output_dir = build_seed_test_folder()
    generated_paths: list[Path] = []

    for seed in seeds:
        image_path = generate_and_save_image(
            client,
            shared_state_snapshot,
            raw_frames_snapshot,
            seed=seed,
            seed_test_mode=True,
            output_dir=seed_output_dir,
        )
        generated_paths.append(image_path)

        latest_shared_state = json.loads(json.dumps(shared_state_snapshot))
        latest_shared_state["current_image_path"] = resolve_image_path_value(image_path)
        latest_shared_state["current_image_seed"] = seed
        latest_shared_state["seed_test_mode"] = True
        update_shared_state_file(latest_shared_state)

    index_path = build_seed_test_index(
        output_dir=seed_output_dir,
        prompt=shared_state_snapshot["image_prompt"],
        descriptors=shared_state_snapshot["interpreted_state"],
        seeds=seeds,
        generated_filenames=[path.name for path in generated_paths],
    )
    print(f"[SEED_TEST] Index: {index_path}")

    contact_sheet_path = create_seed_contact_sheet(generated_paths, seed_output_dir)
    if contact_sheet_path is not None:
        print(f"[SEED_TEST] Contact sheet: {contact_sheet_path}")

    return generated_paths


def start_image_generation(
    client,
    coordinator: InterpretationCoordinator,
    shared_state: dict[str, Any],
    raw_frames: list[dict[str, Any]],
    now: float,
) -> None:
    shared_state_snapshot = json.loads(json.dumps(shared_state))
    raw_frames_snapshot = json.loads(json.dumps(raw_frames))
    signature = coordinator.stable_signature
    selected_seed = select_generation_seed(shared_state_snapshot["interpreted_state"])

    coordinator.image_generation_in_progress = True
    coordinator.image_generation_started_at = now
    coordinator.last_image_generation_time = now
    coordinator.pending_image_result = None
    print(
        "[IMAGE_START] "
        f"pid={os.getpid()} "
        f"timestamp={format_debug_timestamp(now)} "
        f"signature={signature!r} "
        f"seed={selected_seed} "
        "starting background image generation"
    )

    def worker() -> None:
        try:
            image_path = generate_and_save_image(
                client,
                shared_state_snapshot,
                raw_frames_snapshot,
                seed=selected_seed,
                seed_test_mode=False,
            )
            coordinator.pending_image_result = {
                "signature": signature,
                "image_path": str(image_path.resolve()),
                "seed": selected_seed,
                "error": None,
                "completed_at": time.time(),
            }
        except Exception as exc:
            coordinator.pending_image_result = {
                "signature": signature,
                "image_path": None,
                "seed": selected_seed,
                "error": str(exc),
                "completed_at": time.time(),
            }

    threading.Thread(target=worker, daemon=True).start()


def finalize_pending_image_result(coordinator: InterpretationCoordinator) -> None:
    result = coordinator.pending_image_result
    if result is None:
        return

    print(
        "[IMAGE_RESULT] "
        f"pid={os.getpid()} "
        f"timestamp={datetime.now().isoformat(timespec='seconds')} "
        f"result_signature={result.get('signature')!r} "
        f"result_seed={result.get('seed')} "
        f"completed_at={format_debug_timestamp(result.get('completed_at'))}"
    )

    coordinator.pending_image_result = None
    coordinator.image_generation_in_progress = False
    coordinator.last_image_generation_time = result["completed_at"]

    latest_shared_state = coordinator.latest_shared_state
    if latest_shared_state is None:
        return

    latest_shared_state["last_image_generation_time"] = coordinator.last_image_generation_time
    latest_shared_state["image_generation_in_progress"] = False

    if result["error"] is None:
        coordinator.last_image_signature = result["signature"]
        latest_shared_state["current_image_path"] = result["image_path"]
        latest_shared_state["current_image_seed"] = result.get("seed")
        latest_shared_state["last_image_error"] = None
        print(f"[IMAGE] Updated image for signature: {result['signature']}")
    else:
        latest_shared_state["last_image_error"] = result["error"]
        print(f"[ERROR] Image generation failed: {result['error']}", file=sys.stderr)

    update_shared_state_file(latest_shared_state)


def process_interpretation_cycle(
    coordinator: InterpretationCoordinator,
    frame_buffer,
    now: float,
    openai_settings: dict[str, Any] | None,
    empty_room_baseline: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    raw_frames = list(frame_buffer)
    smoothed = smooth_frames(raw_frames)
    scene_plan = build_scene_plan(smoothed, openai_settings, empty_room_baseline)
    descriptors = scene_plan["descriptors"]
    live_lines = scene_plan["live_lines"]
    prompt_sections = scene_plan["prompt_sections"]
    prompt = scene_plan["prompt"]
    signature = build_state_signature(scene_plan.get("state_signature_descriptors", descriptors))
    change_metrics = extract_change_metrics(smoothed)

    previous_shared_state = coordinator.latest_shared_state
    previous_metrics = None if previous_shared_state is None else previous_shared_state.get("change_metrics")
    previous_signature = coordinator.current_signature

    state_changed = signature != previous_signature or change_exceeds_threshold(previous_metrics, change_metrics)
    if state_changed:
        coordinator.current_signature = signature
        coordinator.stable_signature = signature
        coordinator.last_meaningful_change_time = now
    elif coordinator.current_signature is None:
        coordinator.current_signature = signature
        coordinator.stable_signature = signature

    shared_state = build_shared_state_payload(
        raw_frames=raw_frames,
        smoothed=smoothed,
        descriptors=descriptors,
        live_lines=live_lines,
        prompt=prompt,
        prompt_sections=prompt_sections,
        interpretation_source=scene_plan.get("interpretation_source", "rule_based"),
        agent_preview=scene_plan.get("agent_preview"),
        agent_notes=scene_plan.get("agent_notes", []),
        openai_error=scene_plan.get("openai_error"),
        coordinator=coordinator,
        now=now,
        generated_image_path=(
            previous_shared_state.get("current_image_path") if previous_shared_state else None
        ),
    )
    shared_state["change_metrics"] = change_metrics
    shared_state["image_generation_in_progress"] = coordinator.image_generation_in_progress
    coordinator.latest_shared_state = shared_state
    return raw_frames, shared_state, state_changed


def print_runtime_configuration(openai_settings: dict[str, Any] | None):
    print_startup_debug()
    print(f"Opening {PORT} at {BAUDRATE} baud")
    print(f"Expecting serial packets in the format: {SERIAL_PREFIX}{{...json...}}")
    print(f"Text updates every ~{TEXT_UPDATE_INTERVAL_SECONDS} seconds or on meaningful change")
    print(f"Image generation interval: {IMAGE_GENERATION_INTERVAL_SECONDS} seconds")
    print(f"Forced image refresh interval: {FORCE_IMAGE_REFRESH_SECONDS} seconds")
    print(f"Stable-state hold before image generation: {STATE_STABLE_HOLD_SECONDS} seconds")
    print(f"Smoothing over the latest {SMOOTHING_WINDOW_SIZE} frames")
    print(f"Saving outputs to {OUTPUT_DIR.resolve()}")
    print(f"Shared state file: {SHARED_STATE_PATH.resolve()}")
    print(f"Default empty-room seed: {DEFAULT_EMPTY_IMAGE_SEED}")
    print(f"Default occupied-room seed: {DEFAULT_OCCUPIED_IMAGE_SEED}")
    print(f"Seed test mode: {SEED_TEST_MODE}")
    if openai_settings and openai_settings.get("enabled"):
        print(f"OpenAI sensor interpreter: enabled ({openai_settings.get('model')})")
    else:
        print("OpenAI sensor interpreter: disabled or not configured")
    if EMPTY_ROOM_BASELINE_PATH.exists():
        print(f"Empty-room baseline: loaded from {EMPTY_ROOM_BASELINE_PATH.resolve()}")
    else:
        print("Empty-room baseline: not recorded yet")
    if SEED_TEST_MODE:
        print(f"Seed test values: {SEED_TEST_VALUES}")


def main():
    acquire_single_instance_lock()
    openai_settings = load_openai_settings(required=False)
    if not openai_settings.get("api_key"):
        openai_settings["enabled"] = False
    empty_room_baseline = load_empty_room_baseline()
    print_runtime_configuration(openai_settings)

    token = load_hf_token()
    client = InferenceClient(provider="hf-inference", api_key=token)

    if SANITY_SINGLE_IMAGE_MODE:
        sanity_prompt = (
            "photorealistic interior room, two human figures standing in the room, "
            "both figures clearly visible at first glance, full-body human presence visible, "
            "figures ambiguous and partially unresolved, faces indistinct, muted atmosphere, "
            "sparse environment"
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        sanity_image_path = OUTPUT_DIR / "sanity_visible_people_test.png"
        image = generate_image(
            client,
            sanity_prompt,
            "empty room, empty scene, no people, no person, vacant interior",
            DEFAULT_OCCUPIED_IMAGE_SEED,
        )
        image.save(sanity_image_path)
        print(str(sanity_image_path.resolve()))
        return

    frame_buffer = deque(maxlen=SMOOTHING_WINDOW_SIZE)
    coordinator = InterpretationCoordinator()

    try:
        with serial.Serial(PORT, BAUDRATE, timeout=SERIAL_TIMEOUT_SECONDS) as ser:
            time.sleep(2)
            print(f"Waiting up to {SERIAL_STARTUP_SECONDS} seconds for Pico sensor packets...")

            if not wait_for_initial_frames(ser, frame_buffer):
                print(
                    "Connected to the Pico, but no valid SENSOR_DATA packets were received.",
                    file=sys.stderr,
                )
                print(
                    "Deploy the updated main.py to the Pico and confirm it is streaming JSON sensor frames.",
                    file=sys.stderr,
                )
                sys.exit(1)

            if SEED_TEST_MODE:
                now = time.time()
                raw_frames, shared_state, _ = process_interpretation_cycle(
                    coordinator,
                    frame_buffer,
                    now,
                    openai_settings,
                    empty_room_baseline,
                )
                coordinator.last_text_update_time = now
                shared_state["last_text_update_time"] = coordinator.last_text_update_time
                shared_state["seed_test_mode"] = True
                shared_state["image_generation_in_progress"] = True
                update_shared_state_file(shared_state)
                print("[SEED_TEST] Starting frozen prompt and descriptor batch")
                generate_seed_comparison_set(
                    client,
                    shared_state=shared_state,
                    raw_frames=raw_frames,
                    seeds=SEED_TEST_VALUES,
                )
                latest_shared_state = load_shared_state_from_disk() or shared_state
                latest_shared_state["image_generation_in_progress"] = False
                latest_shared_state["seed_test_mode"] = True
                update_shared_state_file(latest_shared_state)
                print("[SEED_TEST] Finished seed comparison batch")
                return

            while True:
                raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
                if raw_line:
                    print(f"[PICO] {raw_line}")
                    frame = parse_sensor_line(raw_line)
                    if frame is not None:
                        frame_buffer.append(frame)

                if len(frame_buffer) < MIN_FRAMES_FOR_PROCESSING:
                    continue

                now = time.time()
                finalize_pending_image_result(coordinator)
                raw_frames, shared_state, state_changed = process_interpretation_cycle(
                    coordinator,
                    frame_buffer,
                    now,
                    openai_settings,
                    empty_room_baseline,
                )

                if should_update_text(coordinator, now, state_changed):
                    coordinator.last_text_update_time = now
                    shared_state["last_text_update_time"] = coordinator.last_text_update_time
                    update_shared_state_file(shared_state)
                    print(f"[TEXT] {' | '.join(shared_state['live_inference_lines'])}")

                if should_generate_new_image(coordinator, now):
                    latest_shared_state = coordinator.latest_shared_state
                    if latest_shared_state is not None:
                        latest_shared_state["image_generation_in_progress"] = True
                        latest_shared_state["seed_test_mode"] = False
                        latest_shared_state["last_image_error"] = None
                        update_shared_state_file(latest_shared_state)
                    print("[IMAGE] Starting background image generation")
                    start_image_generation(client, coordinator, shared_state, raw_frames, now)
    except serial.SerialException as exc:
        print(f"Could not open {PORT}: {exc}", file=sys.stderr)
        print(
            "Close the Pico serial monitor in VS Code or Thonny, then run this script again.",
            file=sys.stderr,
        )
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cleanup_lock_file()


if __name__ == "__main__":
    main()
