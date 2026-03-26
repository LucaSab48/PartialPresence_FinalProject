import json
import os
import sys
import time
import threading
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import serial
from huggingface_hub import InferenceClient

from hf_auth import load_hf_token


PORT = "COM8"
BAUDRATE = 115200
SERIAL_TIMEOUT_SECONDS = 1
SERIAL_STARTUP_SECONDS = 20

# Timing controls for the two output layers.
TEXT_UPDATE_INTERVAL_SECONDS = 1.0
IMAGE_GENERATION_INTERVAL_SECONDS = 8.0
STATE_STABLE_HOLD_SECONDS = 1.5

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
IMAGE_GUIDANCE_SCALE = 5.0
IMAGE_NUM_INFERENCE_STEPS = 20
IMAGE_NEGATIVE_PROMPT = (
    "text, words, letters, typography, signage, caption, watermark, user interface, "
    "collage, diptych, triptych, split panels, storyboard, poster layout, document layout"
)
REFERENCE_BACKGROUND_DIRECTION = (
    "background inspired by liminal void imagery, vast dark backdrop, sparse floor plane, "
    "muted low-saturation gradients, faint atmospheric haze, smooth abstract curved forms, minimal detail, "
    "quiet composition with generous open areas and strong figure readability"
)
OUTPUT_DIR = Path("generated_images")
SHARED_STATE_PATH = OUTPUT_DIR / "current_interpretation_state.json"
SERIAL_PREFIX = "SENSOR_DATA:"
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


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
        return "one distant person"

    if target_state == "MOVING_AND_STATIONARY":
        return "two people"
    if moving_energy >= 70 and stationary_energy >= 35:
        return "two people"
    if moving_energy >= 45 or stationary_energy >= 30 or out_state == 1:
        return "one person"
    return "uncertain number of people"


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


def build_scene_directive(descriptors):
    presence = descriptors["presence_activity"]
    lighting = descriptors["lighting_condition"]
    atmosphere = descriptors["atmospheric_condition"]
    spatial = descriptors["spatial_impression"]
    background = descriptors["abstract_background"]
    occupancy_directive = (
        "the room must visibly contain human occupants"
        if presence != "no presence"
        else "the room should read as empty and unoccupied"
    )

    return (
        "a believable interior room shaped by sensed human activity, "
        f"{descriptors['presence_count']}, "
        f"{descriptors['presence_location']}, "
        f"the room feels {spatial}, "
        f"lighting reads as {lighting}, "
        f"air feels like {atmosphere}, "
        f"architectural backdrop remains minimal and ambiguous, {background}, "
        f"{REFERENCE_BACKGROUND_DIRECTION}, "
        f"{occupancy_directive}, "
        "photorealistic interior surfaces, plausible materials, "
        "abstract but physically believable space"
    )


def interpret_sensor_state(smoothed):
    descriptors = {
        "presence_activity": interpret_presence(smoothed.get("ld2410c")),
        "presence_count": estimate_presence_count(smoothed.get("ld2410c"), smoothed.get("sen0628")),
        "sen0628_spatial_estimate": interpret_sen0628_spatial_estimate(smoothed.get("sen0628")),
        "sen0628_figure_side": interpret_sen0628_figure_side(smoothed.get("sen0628")),
        "presence_location": interpret_presence_location(smoothed.get("ld2410c"), smoothed.get("sen0628")),
        "lighting_condition": interpret_lighting(smoothed.get("light")),
        "atmospheric_condition": interpret_atmosphere(smoothed.get("bme688")),
        "spatial_impression": interpret_spatial_impression(smoothed.get("ld2410c"), smoothed.get("sen0628")),
        "abstract_background": interpret_abstract_background(
            smoothed.get("light"), smoothed.get("bme688")
        ),
    }
    return descriptors


def build_live_inference_lines(descriptors):
    presence_activity = descriptors["presence_activity"]
    presence_location = descriptors["presence_location"]
    lighting = descriptors["lighting_condition"]
    atmosphere = descriptors["atmospheric_condition"]
    spatial = descriptors["spatial_impression"]
    sen0628_spatial_estimate = descriptors["sen0628_spatial_estimate"]
    sen0628_figure_side = descriptors["sen0628_figure_side"]

    movement_line_map = {
        "no presence": "possible absence of human presence",
        "still presence": "steady human presence inferred",
        "active presence": "clustered activity inferred",
        "intermittent movement": "intermittent movement inferred",
        "presence uncertain": "low-confidence presence reading",
    }
    light_line_map = {
        "dark": "ambient light remains low",
        "dim": "dim ambient field detected",
        "moderate light": "moderate ambient light detected",
        "bright": "bright ambient field detected",
        "lighting uncertain": "ambient light remains uncertain",
    }
    atmosphere_line_map = {
        "stale heavy atmosphere": "dense indoor atmosphere inferred",
        "warm dense air": "warm enclosed atmosphere inferred",
        "cool dry air": "cool dry atmosphere inferred",
        "neutral indoor air": "neutral indoor atmosphere inferred",
        "atmosphere uncertain": "atmospheric reading remains uncertain",
    }

    lines = [
        movement_line_map.get(presence_activity, f"{presence_activity} inferred"),
        f"spatial trace: {sen0628_spatial_estimate}",
        f"lateral bias: {sen0628_figure_side}",
        f"spatial boundary: {presence_location}",
        light_line_map.get(lighting, f"lighting appears {lighting}"),
        atmosphere_line_map.get(atmosphere, f"atmosphere appears {atmosphere}"),
        f"room impression: {spatial}",
    ]
    return lines


def build_image_prompt(descriptors):
    presence = descriptors["presence_activity"]
    scene_directive = build_scene_directive(descriptors)
    sen0628_spatial_estimate = descriptors["sen0628_spatial_estimate"]
    sen0628_figure_side = descriptors["sen0628_figure_side"]

    negative_constraints = (
        "no text, no words, no letters, no signage, no typography, "
        "no collage, no diptych, no triptych, no split panels, no storyboard, no poster layout, "
        "no watermark, no caption, no user interface, no document look"
    )

    if presence == "no presence":
        subject_directive = (
            "LD2410C reports no targets, generate an empty room with no people and no human figures"
        )
        composition_directive = (
            "single interior scene, cinematic photography, photorealistic image, sparse abstract environment, "
            "allow empty space only in this no-target case, "
            "background should stay minimal and interpreted rather than literal, "
            "fine surface detail, restrained background detail, "
            "the room should feel unoccupied and quiet"
        )
        branch_negative_constraints = (
            f"{negative_constraints}, no people, no person, no human figure, no silhouette, no occupant"
        )
    else:
        subject_directive = (
            f"{presence}, {descriptors['presence_count']}, {descriptors['presence_location']}, "
            f"{sen0628_spatial_estimate}, "
            f"SEN0628 places the figure on the {sen0628_figure_side} side of the room, "
            "LD2410C detects a target in the room, so there is a human figure in the image and it must not be an empty room, "
            "the figure must be clearly visible and immediately readable as a person, "
            "LD2410C should determine whether one or two people are shown, "
            "SEN0628 should determine where those people stand within the room, "
            "make the people clearly readable as occupants rather than implied traces, "
            "human presence can appear only as the inferred number of coherent scene elements, not repeated beyond that count, "
            "at least one full human figure must be visible in the final image"
        )
        composition_directive = (
            "single interior scene, cinematic photography, photorealistic image, "
            "the room should visibly contain the detected occupants, "
            "keep the background secondary so the people remain legible, "
            "realistic human scale, natural posture, clear separation between each person, "
            "medium-long shot, full or three-quarter bodies visible, subjects not cropped out, "
            "human figures should be immediately noticeable at first glance, "
            "fine surface detail, restrained background detail, "
            "avoid ambiguous forms when people are present"
        )
        branch_negative_constraints = (
            f"{negative_constraints}, no empty room, no empty scene, no furniture-only composition, "
            "no hidden person, no cropped-out person, no tiny distant person, no obscured face, "
            "no silhouette-only figure"
        )

    return (
        f"{scene_directive}, "
        f"{subject_directive}, "
        f"{composition_directive}, "
        "strong figure readability, people placed against large open background areas only when targets are detected, "
        "background palette should stay muted and desaturated, with charcoal, dusty blue, faded teal, ash gray, and faint dusty rose, "
        "real-world lighting, natural shadows, subtle atmospheric depth, realistic lens perspective, "
        "dreamlike only in interpretation, not in rendering style, "
        "avoid painterly, graphic, or overtly surreal aesthetics, "
        f"{branch_negative_constraints}"
    )


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
    signature_fields = {
        "presence_activity": descriptors["presence_activity"],
        "presence_count": descriptors["presence_count"],
        "sen0628_spatial_estimate": descriptors["sen0628_spatial_estimate"],
        "sen0628_figure_side": descriptors["sen0628_figure_side"],
        "presence_location": descriptors["presence_location"],
        "lighting_condition": descriptors["lighting_condition"],
        "atmospheric_condition": descriptors["atmospheric_condition"],
        "spatial_impression": descriptors["spatial_impression"],
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
            "state_stable_hold_seconds": STATE_STABLE_HOLD_SECONDS,
        },
        "raw_frame_summary": {
            "frame_count": len(raw_frames),
            "latest_seq": raw_frames[-1].get("seq") if raw_frames else None,
        },
        "smoothed_sensor_values": smoothed,
        "interpreted_state": descriptors,
        "live_inference_lines": live_lines,
        "image_prompt": prompt,
        "state_signature": coordinator.current_signature,
        "stable_signature": coordinator.stable_signature,
        "last_meaningful_change_time": coordinator.last_meaningful_change_time,
        "last_text_update_time": coordinator.last_text_update_time,
        "last_image_generation_time": coordinator.last_image_generation_time,
        "current_image_path": resolve_image_path_value(generated_image_path),
        "last_image_error": previous_shared_state.get("last_image_error"),
        "seconds_since_meaningful_change": round(max(0.0, now - coordinator.last_meaningful_change_time), 2),
    }


def make_output_stem():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"sensor_inference_{timestamp}"


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

    return stale_image_path


def save_generation_artifacts(image, raw_frames, smoothed, descriptors, prompt):
    stem = make_output_stem()
    image_path = stem.with_suffix(".png")
    log_path = stem.with_suffix(".txt")

    image.save(image_path)

    log_text = "\n".join(
        [
            "generation_timestamp: {}".format(datetime.now().isoformat(timespec="seconds")),
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
            "final_prompt:",
            prompt,
        ]
    )
    log_path.write_text(log_text, encoding="utf-8")
    return image_path, log_path


def generate_image(client, prompt):
    return client.text_to_image(
        prompt,
        negative_prompt=IMAGE_NEGATIVE_PROMPT,
        model=MODEL_ID,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        guidance_scale=IMAGE_GUIDANCE_SCALE,
        num_inference_steps=IMAGE_NUM_INFERENCE_STEPS,
    )


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


def should_generate_new_image(coordinator, now):
    if coordinator.latest_shared_state is None:
        return False
    if coordinator.image_generation_in_progress:
        return False
    if now - coordinator.last_image_generation_time < IMAGE_GENERATION_INTERVAL_SECONDS:
        return False
    if now - coordinator.last_meaningful_change_time < STATE_STABLE_HOLD_SECONDS:
        return False
    if coordinator.stable_signature != coordinator.last_image_signature:
        return True
    return False


def update_shared_state_file(shared_state: dict[str, Any]) -> None:
    atomic_write_text(SHARED_STATE_PATH, json.dumps(shared_state, indent=2))


def generate_and_save_image(client, shared_state: dict[str, Any], raw_frames):
    prompt = shared_state["image_prompt"]
    smoothed = shared_state["smoothed_sensor_values"]
    descriptors = shared_state["interpreted_state"]

    print(f"[INFER] {json.dumps(descriptors)}")
    print(f"[PROMPT] {prompt}")

    image = generate_image(client, prompt)
    image_path, log_path = save_generation_artifacts(
        image=image,
        raw_frames=raw_frames,
        smoothed=smoothed,
        descriptors=descriptors,
        prompt=prompt,
    )
    print(f"[SAVED] Image: {image_path}")
    print(f"[SAVED] Log:   {log_path}")
    delete_two_generations_ago_image()
    return image_path


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

    coordinator.image_generation_in_progress = True
    coordinator.image_generation_started_at = now
    coordinator.last_image_generation_time = now
    coordinator.pending_image_result = None

    def worker() -> None:
        try:
            image_path = generate_and_save_image(client, shared_state_snapshot, raw_frames_snapshot)
            coordinator.pending_image_result = {
                "signature": signature,
                "image_path": str(image_path.resolve()),
                "error": None,
                "completed_at": time.time(),
            }
        except Exception as exc:
            coordinator.pending_image_result = {
                "signature": signature,
                "image_path": None,
                "error": str(exc),
                "completed_at": time.time(),
            }

    threading.Thread(target=worker, daemon=True).start()


def finalize_pending_image_result(coordinator: InterpretationCoordinator) -> None:
    result = coordinator.pending_image_result
    if result is None:
        return

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
        latest_shared_state["last_image_error"] = None
        print(f"[IMAGE] Updated image for signature: {result['signature']}")
    else:
        latest_shared_state["last_image_error"] = result["error"]
        print(f"[ERROR] Image generation failed: {result['error']}", file=sys.stderr)

    update_shared_state_file(latest_shared_state)


def process_interpretation_cycle(
    coordinator: InterpretationCoordinator, frame_buffer, now: float
) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
    raw_frames = list(frame_buffer)
    smoothed = smooth_frames(raw_frames)
    descriptors = interpret_sensor_state(smoothed)
    live_lines = build_live_inference_lines(descriptors)
    prompt = build_image_prompt(descriptors)
    signature = build_state_signature(descriptors)
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


def print_runtime_configuration():
    print(f"Opening {PORT} at {BAUDRATE} baud")
    print(f"Expecting serial packets in the format: {SERIAL_PREFIX}{{...json...}}")
    print(f"Text updates every ~{TEXT_UPDATE_INTERVAL_SECONDS} seconds or on meaningful change")
    print(f"Image generation interval: {IMAGE_GENERATION_INTERVAL_SECONDS} seconds")
    print(f"Stable-state hold before image generation: {STATE_STABLE_HOLD_SECONDS} seconds")
    print(f"Smoothing over the latest {SMOOTHING_WINDOW_SIZE} frames")
    print(f"Saving outputs to {OUTPUT_DIR.resolve()}")
    print(f"Shared state file: {SHARED_STATE_PATH.resolve()}")


def main():
    print_runtime_configuration()

    token = load_hf_token()
    client = InferenceClient(provider="hf-inference", api_key=token)

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
                    coordinator, frame_buffer, now
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


if __name__ == "__main__":
    main()
