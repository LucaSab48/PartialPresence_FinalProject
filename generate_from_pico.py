import json
import os
import re
import sys
import time
import threading
import atexit
import hashlib
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
from usb_microphone import (
    attach_microphone_snapshot,
    build_microphone_monitor_from_env,
)


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
TEMPERATURE_CHANGE_THRESHOLD = 0.8
HUMIDITY_CHANGE_THRESHOLD = 4.0
DISTANCE_CHANGE_THRESHOLD_CM = 30.0
AUDIO_ACTIVITY_CHANGE_THRESHOLD = 1.5

SMOOTHING_WINDOW_SIZE = 6
MIN_FRAMES_FOR_PROCESSING = 3

MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
IMAGE_WIDTH = 768
IMAGE_HEIGHT = 768
IMAGE_NUM_INFERENCE_STEPS = 28
IMAGE_NUM_INFERENCE_STEPS_MIN = 24
IMAGE_NUM_INFERENCE_STEPS_MAX = 32
IMAGE_TEMPERATURE_MIN = 0.2
IMAGE_TEMPERATURE_MAX = 0.8
IMAGE_TOP_P_MIN = 0.7
IMAGE_TOP_P_MAX = 0.95
IMAGE_GUIDANCE_SCALE_MIN = 7.0
IMAGE_GUIDANCE_SCALE_MAX = 10.5
IMAGE_GUIDANCE_VARIATION_OFFSETS = [-1.1, -0.7, -0.35, 0.0, 0.35, 0.7, 1.1]
IMAGE_STEP_VARIATION_OFFSETS = [-4, -2, -1, 0, 1, 2, 4]
IMAGE_UNCERTAINTY_DEFAULT = 0.22
IMAGE_UNCERTAINTY_GUIDANCE_BIAS = 0.25
IMAGE_UNCERTAINTY_TEMPERATURE_CURVE = 1.0
IMAGE_UNCERTAINTY_TOP_P_CURVE = 0.9
IMAGE_UNCERTAINTY_GUIDANCE_CURVE = 1.15
DEFAULT_EMPTY_IMAGE_SEED = 23
DEFAULT_OCCUPIED_IMAGE_SEED = 23
IMAGE_SEED_CYCLE_VALUES = [252, 244, 264, 256]
IMAGE_SEED_CYCLE_INTERVAL_SECONDS = 45.0
SEED_TEST_MODE = False
SANITY_SINGLE_IMAGE_MODE = False
SEED_TEST_VALUES = [23, 42, 77, 101, 248, 333, 444, 777]
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
MAX_AI_LIVE_LINES = 6
EMPTY_ROOM_BASELINE_PATH = SCRIPT_DIR / "empty_room_baseline.json"
MAX_BASELINE_CHANGED_METRICS = 5
MAX_LANGUAGE_PASS_LIVE_LINES = 5
MIN_LANGUAGE_PASS_LIVE_LINES = 3
MAX_LANGUAGE_PASS_AGENT_NOTES = 1
MIN_LANGUAGE_PASS_AGENT_NOTES = 0
MAX_LANGUAGE_PASS_PROMPT_MODIFIERS = 6
MIN_LANGUAGE_PASS_PROMPT_MODIFIERS = 3
EMPTY_ROOM_BASELINE_METRIC_TOLERANCES = {
    "ld2410c.out": 0.5,
    "ld2410c.moving_energy": 18.0,
    "ld2410c.stationary_energy": 28.0,
    "ld2410c.detection_distance_cm": 85.0,
    "ld2410c_front.out": 0.5,
    "ld2410c_front.moving_energy": 18.0,
    "ld2410c_front.stationary_energy": 28.0,
    "ld2410c_front.detection_distance_cm": 85.0,
    "ld2410c_back.out": 0.5,
    "ld2410c_back.moving_energy": 18.0,
    "ld2410c_back.stationary_energy": 28.0,
    "ld2410c_back.detection_distance_cm": 85.0,
    "sen0628.valid_points": 8.0,
    "sen0628.left_zone_mm": 180.0,
    "sen0628.center_zone_mm": 180.0,
    "sen0628.right_zone_mm": 180.0,
    "sen0628.front_zone_mm": 180.0,
    "sen0628.mid_zone_mm": 180.0,
    "sen0628.back_zone_mm": 180.0,
    "sen0628.left_close_points": 1.5,
    "sen0628.center_close_points": 1.5,
    "sen0628.right_close_points": 1.5,
    "sen0628.left_occupied_points": 1.0,
    "sen0628.center_occupied_points": 1.0,
    "sen0628.right_occupied_points": 1.0,
    "sen0628.front_occupied_points": 1.0,
    "sen0628.mid_occupied_points": 1.0,
    "sen0628.back_occupied_points": 1.0,
    "sen0628.near_points": 1.0,
    "sen0628.mid_points": 6.0,
    "sen0628.far_points": 8.0,
    "sen0628.floor_occupied_points": 1.0,
    "sen0628.floor_clear_points": 8.0,
    "sen0628.mean_obstruction_height_mm": 120.0,
    "sen0628.max_obstruction_height_mm": 120.0,
    "sen0628.low_obstruction_points": 1.0,
    "sen0628.mid_obstruction_points": 1.0,
    "sen0628.tall_obstruction_points": 1.0,
    "usb_microphone.activity_score": 1.0,
    "usb_microphone.relative_db": 6.0,
    "usb_microphone.active_fraction": 0.2,
}
BASELINE_SUBSTANTIAL_DELTA = 3.0
BASELINE_VERY_SUBSTANTIAL_DELTA = 12.0
PRESENCE_COUNT_SCORE_THRESHOLDS = {
    1: 0.95,
    2: 2.15,
    3: 4.15,
    4: 4.95,
}


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
    seed_cycle_started_at: float | None = None


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def lerp(start: float, end: float, amount: float) -> float:
    return start + ((end - start) * amount)


def normalize_unit_interval(value: Any, fallback: float) -> float:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, (int, float)):
        return clamp(float(value), 0.0, 1.0)
    return fallback


def round_generation_value(value: float) -> float:
    return round(float(value), 3)


def stable_variation_index(payload: dict[str, Any], modulo: int) -> int:
    if modulo <= 1:
        return 0
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def quantize_numeric_bucket(value: Any, step: float) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if step <= 0:
        return int(round(float(value)))
    return int(round(float(value) / step))


def build_sensor_variation_profile(
    smoothed: dict[str, Any],
    descriptors: dict[str, Any],
) -> dict[str, Any]:
    bme_data = smoothed.get("bme688") or {}
    ld_data = smoothed.get("ld2410c") or {}
    sen_data = smoothed.get("sen0628") or {}
    microphone_data = smoothed.get("usb_microphone") or {}
    fingerprint_payload = {
        "figure_count": int(descriptors.get("figure_count", 0) or 0),
        "layout_mode": str(descriptors.get("layout_mode", "empty")),
        "primary_zone": str(descriptors.get("primary_zone", "center")),
        "depth_band": str(descriptors.get("depth_band", "mid-room")),
        "presence_activity": str(descriptors.get("presence_activity", "")),
        "audio_activity": str(descriptors.get("audio_activity", "")),
        "temperature_bucket": quantize_numeric_bucket(bme_data.get("temperature_c"), 0.6),
        "humidity_bucket": quantize_numeric_bucket(bme_data.get("humidity_pct"), 2.5),
        "distance_bucket": quantize_numeric_bucket(ld_data.get("detection_distance_cm"), 20.0),
        "moving_bucket": quantize_numeric_bucket(ld_data.get("moving_energy"), 10.0),
        "stationary_bucket": quantize_numeric_bucket(ld_data.get("stationary_energy"), 10.0),
        "audio_bucket": quantize_numeric_bucket(microphone_data.get("activity_score"), 0.8),
        "left_zone_bucket": quantize_numeric_bucket(sen_data.get("left_zone_mm"), 140.0),
        "center_zone_bucket": quantize_numeric_bucket(sen_data.get("center_zone_mm"), 140.0),
        "right_zone_bucket": quantize_numeric_bucket(sen_data.get("right_zone_mm"), 140.0),
        "near_points_bucket": quantize_numeric_bucket(sen_data.get("near_points"), 1.0),
        "mid_points_bucket": quantize_numeric_bucket(sen_data.get("mid_points"), 1.0),
        "far_points_bucket": quantize_numeric_bucket(sen_data.get("far_points"), 1.0),
    }
    variation_index = stable_variation_index(fingerprint_payload, len(IMAGE_GUIDANCE_VARIATION_OFFSETS))
    empty_room_modifiers = [
        "subtle polished floor reflections",
        "slightly cooler gallery light",
        "slightly warmer wall light",
        "soft shadow falloff across the floor",
        "clean open floor with faint depth cues",
    ]
    occupied_posture_modifiers = [
        "natural upright posture",
        "slight weight shift through the hips and shoulders",
        "one figure angled partly sideways",
        "subtle mid-step stance",
        "casual standing posture with relaxed arms",
    ]
    occupied_clothing_modifiers = [
        "muted everyday clothing",
        "dark casual clothing with natural folds",
        "light tops with darker trousers",
        "layered neutral clothing",
        "mixed casual outfits with realistic fabric texture",
    ]
    occupied_spacing_modifiers = [
        "clear separation between bodies",
        "mild overlap and natural occlusion",
        "open floor visible between figures",
        "slightly asymmetrical spacing",
        "one figure more dominant in scale",
    ]

    if int(descriptors.get("figure_count", 0) or 0) <= 0:
        prompt_modifiers = [
            empty_room_modifiers[stable_variation_index({"fingerprint": fingerprint_payload, "slot": 0}, len(empty_room_modifiers))],
            empty_room_modifiers[stable_variation_index({"fingerprint": fingerprint_payload, "slot": 1}, len(empty_room_modifiers))],
        ]
    else:
        prompt_modifiers = [
            occupied_posture_modifiers[
                stable_variation_index({"fingerprint": fingerprint_payload, "slot": 0}, len(occupied_posture_modifiers))
            ],
            occupied_clothing_modifiers[
                stable_variation_index({"fingerprint": fingerprint_payload, "slot": 1}, len(occupied_clothing_modifiers))
            ],
            occupied_spacing_modifiers[
                stable_variation_index({"fingerprint": fingerprint_payload, "slot": 2}, len(occupied_spacing_modifiers))
            ],
        ]

    return {
        "fingerprint": json.dumps(fingerprint_payload, sort_keys=True),
        "variation_index": variation_index,
        "prompt_modifiers": prompt_modifiers,
    }


def derive_uncertainty_score_from_descriptors(descriptors: dict[str, Any]) -> float:
    uncertainty = IMAGE_UNCERTAINTY_DEFAULT
    if descriptors.get("presence_activity") == "presence uncertain":
        uncertainty += 0.33
    if descriptors.get("presence_count") == "presence uncertain":
        uncertainty += 0.22
    if descriptors.get("layout_mode") == "ambiguous":
        uncertainty += 0.12
    spatial_certainty = str(descriptors.get("spatial_certainty", "")).lower()
    if "uncertain" in spatial_certainty or "rough" in spatial_certainty:
        uncertainty += 0.12
    active_zone_count = len(descriptors.get("active_zones") or [])
    if active_zone_count >= 3:
        uncertainty += 0.06
    if int(descriptors.get("figure_count", 0)) == 0 and descriptors.get("presence_activity") != "no presence":
        uncertainty += 0.08
    return clamp(uncertainty, 0.0, 1.0)


def build_generation_controls_from_uncertainty(
    uncertainty_score: float,
    variation_fingerprint: str | None = None,
) -> dict[str, float | int]:
    normalized_uncertainty = clamp(float(uncertainty_score), 0.0, 1.0)
    temperature_amount = normalized_uncertainty ** IMAGE_UNCERTAINTY_TEMPERATURE_CURVE
    top_p_amount = normalized_uncertainty ** IMAGE_UNCERTAINTY_TOP_P_CURVE
    guidance_amount = normalized_uncertainty ** IMAGE_UNCERTAINTY_GUIDANCE_CURVE
    guidance_floor_mix = IMAGE_UNCERTAINTY_GUIDANCE_BIAS + (
        (1.0 - IMAGE_UNCERTAINTY_GUIDANCE_BIAS) * (1.0 - guidance_amount)
    )
    guidance_floor_mix = clamp(guidance_floor_mix, 0.0, 1.0)
    guidance_scale = lerp(IMAGE_GUIDANCE_SCALE_MIN, IMAGE_GUIDANCE_SCALE_MAX, guidance_floor_mix)
    num_inference_steps = IMAGE_NUM_INFERENCE_STEPS

    if variation_fingerprint:
        variation_payload = {"variation_fingerprint": variation_fingerprint}
        guidance_scale += IMAGE_GUIDANCE_VARIATION_OFFSETS[
            stable_variation_index(variation_payload, len(IMAGE_GUIDANCE_VARIATION_OFFSETS))
        ]
        num_inference_steps = clamp_int(
            IMAGE_NUM_INFERENCE_STEPS
            + IMAGE_STEP_VARIATION_OFFSETS[
                stable_variation_index(
                    {"variation_fingerprint": variation_fingerprint, "mode": "steps"},
                    len(IMAGE_STEP_VARIATION_OFFSETS),
                )
            ],
            IMAGE_NUM_INFERENCE_STEPS_MIN,
            IMAGE_NUM_INFERENCE_STEPS_MAX,
        )

    return {
        "uncertainty_score": round_generation_value(normalized_uncertainty),
        "temperature": round_generation_value(
            lerp(IMAGE_TEMPERATURE_MIN, IMAGE_TEMPERATURE_MAX, temperature_amount)
        ),
        "top_p": round_generation_value(lerp(IMAGE_TOP_P_MIN, IMAGE_TOP_P_MAX, top_p_amount)),
        "guidance_scale": round_generation_value(
            clamp(guidance_scale, IMAGE_GUIDANCE_SCALE_MIN, IMAGE_GUIDANCE_SCALE_MAX)
        ),
        "num_inference_steps": num_inference_steps,
    }


def sanitize_generation_controls(
    scene_interpretation: dict[str, Any],
    fallback_descriptors: dict[str, Any],
) -> dict[str, float | int]:
    fallback_controls = build_generation_controls_from_uncertainty(
        derive_uncertainty_score_from_descriptors(fallback_descriptors),
        str((fallback_descriptors.get("sensor_variation_profile") or {}).get("fingerprint") or ""),
    )
    uncertainty_score = normalize_unit_interval(
        scene_interpretation.get("uncertainty_score"),
        fallback_controls["uncertainty_score"],
    )
    generation_controls = scene_interpretation.get("generation_controls")
    if not isinstance(generation_controls, dict):
        generation_controls = {}

    default_controls = build_generation_controls_from_uncertainty(uncertainty_score)
    temperature = generation_controls.get("temperature", default_controls["temperature"])
    top_p = generation_controls.get("top_p", default_controls["top_p"])
    guidance_scale = generation_controls.get("guidance_scale", default_controls["guidance_scale"])
    num_inference_steps = generation_controls.get(
        "num_inference_steps",
        default_controls["num_inference_steps"],
    )

    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        temperature = default_controls["temperature"]
    if isinstance(top_p, bool) or not isinstance(top_p, (int, float)):
        top_p = default_controls["top_p"]
    if isinstance(guidance_scale, bool) or not isinstance(guidance_scale, (int, float)):
        guidance_scale = default_controls["guidance_scale"]
    if isinstance(num_inference_steps, bool) or not isinstance(num_inference_steps, (int, float)):
        num_inference_steps = default_controls["num_inference_steps"]

    return {
        "uncertainty_score": round_generation_value(uncertainty_score),
        "temperature": round_generation_value(
            clamp(float(temperature), IMAGE_TEMPERATURE_MIN, IMAGE_TEMPERATURE_MAX)
        ),
        "top_p": round_generation_value(clamp(float(top_p), IMAGE_TOP_P_MIN, IMAGE_TOP_P_MAX)),
        "guidance_scale": round_generation_value(
            clamp(float(guidance_scale), IMAGE_GUIDANCE_SCALE_MIN, IMAGE_GUIDANCE_SCALE_MAX)
        ),
        "num_inference_steps": clamp_int(
            int(round(float(num_inference_steps))),
            IMAGE_NUM_INFERENCE_STEPS_MIN,
            IMAGE_NUM_INFERENCE_STEPS_MAX,
        ),
    }


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


def sen_distance_value(value):
    if isinstance(value, (int, float)) and 0 < value < 4000:
        return float(value)
    return None


def smooth_sen0628_frames(sen_frames):
    if not sen_frames:
        return None

    def distance_mean(key):
        return numeric_mean([sen_distance_value(item.get(key)) for item in sen_frames])

    sen_summary = {
        "mount_mode": pick_mode([item.get("mount_mode") for item in sen_frames]),
        "ceiling_height_mm": numeric_mean([item.get("ceiling_height_mm") for item in sen_frames]),
        "center_mm": distance_mean("center_mm"),
        "center_raw_mm": numeric_mean([item.get("center_raw_mm") for item in sen_frames]),
        "min_mm": distance_mean("min_mm"),
        "max_mm": distance_mean("max_mm"),
        "mean_mm": distance_mean("mean_mm"),
        "valid_points": numeric_mean([item.get("valid_points") for item in sen_frames]),
        "left_zone_mm": distance_mean("left_zone_mm"),
        "center_zone_mm": distance_mean("center_zone_mm"),
        "right_zone_mm": distance_mean("right_zone_mm"),
        "left_close_points": numeric_mean([item.get("left_close_points") for item in sen_frames]),
        "center_close_points": numeric_mean([item.get("center_close_points") for item in sen_frames]),
        "right_close_points": numeric_mean([item.get("right_close_points") for item in sen_frames]),
        "left_occupied_points": numeric_mean([item.get("left_occupied_points") for item in sen_frames]),
        "center_occupied_points": numeric_mean([item.get("center_occupied_points") for item in sen_frames]),
        "right_occupied_points": numeric_mean([item.get("right_occupied_points") for item in sen_frames]),
        "front_zone_mm": distance_mean("front_zone_mm"),
        "mid_zone_mm": distance_mean("mid_zone_mm"),
        "back_zone_mm": distance_mean("back_zone_mm"),
        "front_occupied_points": numeric_mean([item.get("front_occupied_points") for item in sen_frames]),
        "mid_occupied_points": numeric_mean([item.get("mid_occupied_points") for item in sen_frames]),
        "back_occupied_points": numeric_mean([item.get("back_occupied_points") for item in sen_frames]),
        "near_points": numeric_mean([item.get("near_points") for item in sen_frames]),
        "mid_points": numeric_mean([item.get("mid_points") for item in sen_frames]),
        "far_points": numeric_mean([item.get("far_points") for item in sen_frames]),
        "floor_occupied_points": numeric_mean([item.get("floor_occupied_points") for item in sen_frames]),
        "floor_clear_points": numeric_mean([item.get("floor_clear_points") for item in sen_frames]),
        "mean_obstruction_height_mm": numeric_mean([item.get("mean_obstruction_height_mm") for item in sen_frames]),
        "max_obstruction_height_mm": numeric_mean([item.get("max_obstruction_height_mm") for item in sen_frames]),
        "low_obstruction_points": numeric_mean([item.get("low_obstruction_points") for item in sen_frames]),
        "mid_obstruction_points": numeric_mean([item.get("mid_obstruction_points") for item in sen_frames]),
        "tall_obstruction_points": numeric_mean([item.get("tall_obstruction_points") for item in sen_frames]),
    }

    if (
        sen_summary["center_mm"] is not None
        and sen_summary["max_mm"] is not None
        and sen_summary["center_mm"] > sen_summary["max_mm"]
    ):
        sen_summary["center_mm"] = None

    return sen_summary


def pick_mode(values):
    clean_values = [value for value in values if value not in (None, "")]
    if not clean_values:
        return None
    return Counter(clean_values).most_common(1)[0][0]


def summarize_error_values(values):
    clean_values = [str(value) for value in values if value not in (None, "")]
    if not clean_values:
        return {
            "latest": None,
            "count": 0,
            "samples": [],
        }
    unique_samples = list(dict.fromkeys(clean_values))
    return {
        "latest": clean_values[-1],
        "count": len(clean_values),
        "samples": unique_samples,
    }


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


def smooth_ld_sensor_frames(ld_frames):
    if not ld_frames:
        return None

    ld_out_mean = numeric_mean([item.get("out") for item in ld_frames])
    return {
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


def combine_ld_summaries(*sensor_summaries):
    available = [summary for summary in sensor_summaries if isinstance(summary, dict)]
    if not available:
        return None

    states = []
    status_values = []
    target_state_values = []
    target_state_raw_values = []
    out_values = []
    moving_distance_values = []
    moving_energy_values = []
    stationary_distance_values = []
    stationary_energy_values = []
    detection_distance_values = []

    for summary in available:
        states.extend(summary.get("state_history") or [])
        if summary.get("status") is not None:
            status_values.append(summary.get("status"))
        if summary.get("target_state") is not None:
            target_state_values.append(summary.get("target_state"))
        if summary.get("target_state_raw") is not None:
            target_state_raw_values.append(summary.get("target_state_raw"))
        if summary.get("out") is not None:
            out_values.append(summary.get("out"))
        if summary.get("moving_distance_cm") is not None:
            moving_distance_values.append(summary.get("moving_distance_cm"))
        if summary.get("moving_energy") is not None:
            moving_energy_values.append(summary.get("moving_energy"))
        if summary.get("stationary_distance_cm") is not None:
            stationary_distance_values.append(summary.get("stationary_distance_cm"))
        if summary.get("stationary_energy") is not None:
            stationary_energy_values.append(summary.get("stationary_energy"))
        if summary.get("detection_distance_cm") is not None:
            detection_distance_values.append(summary.get("detection_distance_cm"))

    return {
        "out": max(out_values) if out_values else None,
        "target_state": pick_mode(target_state_values),
        "target_state_raw": pick_mode(target_state_raw_values),
        "moving_distance_cm": numeric_mean(moving_distance_values),
        "moving_energy": max(moving_energy_values) if moving_energy_values else None,
        "stationary_distance_cm": numeric_mean(stationary_distance_values),
        "stationary_energy": max(stationary_energy_values) if stationary_energy_values else None,
        "detection_distance_cm": numeric_mean(detection_distance_values),
        "status": pick_mode(status_values),
        "state_history": states,
    }


def smooth_frames(frames):
    bme_frames = [frame.get("bme688", {}) for frame in frames if isinstance(frame.get("bme688"), dict)]
    front_frames = [
        frame.get("ld2410c_front", {})
        for frame in frames
        if isinstance(frame.get("ld2410c_front"), dict)
    ]
    back_frames = [
        frame.get("ld2410c_back", {})
        for frame in frames
        if isinstance(frame.get("ld2410c_back"), dict)
    ]
    sen_frames = [frame.get("sen0628", {}) for frame in frames if isinstance(frame.get("sen0628"), dict)]
    microphone_frames = [
        frame.get("usb_microphone", {})
        for frame in frames
        if isinstance(frame.get("usb_microphone"), dict)
    ]
    front_summary = smooth_ld_sensor_frames(front_frames)
    back_summary = smooth_ld_sensor_frames(back_frames)
    combined_ld_summary = combine_ld_summaries(front_summary, back_summary)
    bme_error_summary = summarize_error_values([frame.get("bme688_error") for frame in frames])
    sen_error_summary = summarize_error_values([frame.get("sen0628_error") for frame in frames])

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
        "ld2410c_front": front_summary,
        "ld2410c_back": back_summary,
        "ld2410c": combined_ld_summary,
        "sen0628": smooth_sen0628_frames(sen_frames),
        "usb_microphone": {
            "available": pick_mode([item.get("available") for item in microphone_frames]),
            "device_name": pick_mode([item.get("device_name") for item in microphone_frames]),
            "device_index": pick_mode([item.get("device_index") for item in microphone_frames]),
            "sample_rate_hz": numeric_mean([item.get("sample_rate_hz") for item in microphone_frames]),
            "rms": numeric_mean([item.get("rms") for item in microphone_frames]),
            "peak": max(
                [item.get("peak") for item in microphone_frames if isinstance(item.get("peak"), (int, float))],
                default=None,
            ),
            "noise_floor_rms": numeric_mean(
                [item.get("noise_floor_rms") for item in microphone_frames]
            ),
            "relative_rms": numeric_mean([item.get("relative_rms") for item in microphone_frames]),
            "relative_db": numeric_mean([item.get("relative_db") for item in microphone_frames]),
            "activity_score": numeric_mean([item.get("activity_score") for item in microphone_frames]),
            "active_fraction": numeric_mean([item.get("active_fraction") for item in microphone_frames]),
        }
        if microphone_frames
        else None,
        "errors": {
            "bme688_error": bme_error_summary["latest"],
            "bme688_error_count": bme_error_summary["count"],
            "bme688_error_samples": bme_error_summary["samples"],
            "sen0628_error": sen_error_summary["latest"],
            "sen0628_error_count": sen_error_summary["count"],
            "sen0628_error_samples": sen_error_summary["samples"],
        },
    }
    return smoothed


def interpret_audio_activity(microphone_data):
    if not isinstance(microphone_data, dict) or not microphone_data.get("available"):
        return "audio unavailable"

    relative_db = microphone_data.get("relative_db")
    activity_score = microphone_data.get("activity_score")
    active_fraction = microphone_data.get("active_fraction")

    if isinstance(relative_db, (int, float)):
        if (
            relative_db >= 10
            and isinstance(active_fraction, (int, float))
            and active_fraction >= 0.65
        ) or (isinstance(activity_score, (int, float)) and activity_score >= 0.75):
            return "strong shared room noise"
        if (
            relative_db >= 6
            and isinstance(active_fraction, (int, float))
            and active_fraction >= 0.4
        ) or (isinstance(activity_score, (int, float)) and activity_score >= 0.35):
            return "moderate human-made room noise"
        if (
            relative_db >= 3
            and isinstance(active_fraction, (int, float))
            and active_fraction >= 0.2
        ) or (isinstance(activity_score, (int, float)) and activity_score >= 0.12):
            return "light room noise"
    return "very quiet room"


def build_audio_live_line(activity):
    line_map = {
        "audio unavailable": "audio feed unavailable",
        "very quiet room": "audio suggests a very quiet room",
        "light room noise": "audio suggests light occupancy noise",
        "moderate human-made room noise": "audio suggests shared room noise",
        "strong shared room noise": "audio strongly suggests shared occupancy",
    }
    return line_map.get(activity, "audio feed partially resolved")


def build_image_foreshadow_line(descriptors):
    figure_count = int(descriptors.get("figure_count", 0) or 0)
    placement_summary = str(descriptors.get("placement_summary") or "occupied room estimate")
    if figure_count <= 0:
        return "next image likely keeps the room nearly empty"
    return f"next image likely leans toward {placement_summary}"


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


def empty_room_baseline_tolerance_for_metric(metric_name: str) -> float | None:
    tolerance = EMPTY_ROOM_BASELINE_METRIC_TOLERANCES.get(metric_name)
    if isinstance(tolerance, (int, float)) and not isinstance(tolerance, bool):
        return float(tolerance)
    return None


def build_empty_room_comparison(
    smoothed: dict[str, Any],
    empty_room_baseline: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if empty_room_baseline is None:
        return None

    live_values = flatten_numeric_values(smoothed)
    baseline_values = flatten_numeric_values(empty_room_baseline)
    changed_metrics: list[dict[str, Any]] = []
    compared_metric_count = 0

    for key in sorted(set(live_values.keys()) & set(baseline_values.keys())):
        tolerance = empty_room_baseline_tolerance_for_metric(key)
        if tolerance is None:
            continue
        compared_metric_count += 1
        baseline_value = baseline_values[key]
        live_value = live_values[key]
        delta = live_value - baseline_value
        magnitude = abs(delta)
        if magnitude < tolerance:
            continue
        changed_metrics.append(
            {
                "metric": key,
                "baseline": round(baseline_value, 3),
                "live": round(live_value, 3),
                "delta": round(delta, 3),
                "tolerance": round(tolerance, 3),
            }
        )

    changed_metrics.sort(key=lambda item: abs(float(item["delta"])), reverse=True)
    return {
        "available": True,
        "top_changed_metrics": changed_metrics[:12],
        "compared_metric_count": compared_metric_count,
        "changed_metric_count": len(changed_metrics),
        "baseline_seq_range": empty_room_baseline.get("seq_range"),
        "baseline_frame_count": empty_room_baseline.get("frame_count"),
    }


def classify_baseline_departure(empty_room_comparison: dict[str, Any] | None) -> str:
    if not isinstance(empty_room_comparison, dict):
        return "unknown"

    changed_metrics = empty_room_comparison.get("top_changed_metrics")
    compared_metric_count = empty_room_comparison.get("compared_metric_count")
    changed_metric_count = empty_room_comparison.get("changed_metric_count")
    if isinstance(compared_metric_count, int) and compared_metric_count > 0:
        if not isinstance(changed_metric_count, int):
            changed_metric_count = len(changed_metrics) if isinstance(changed_metrics, list) else 0
        change_ratio = float(changed_metric_count) / float(compared_metric_count)
        if changed_metric_count == 0 or change_ratio <= 0.08:
            return "minimal"
        if change_ratio <= 0.22:
            return "weak"
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


def baseline_matches_empty_room_recording(
    smoothed: dict[str, Any],
    empty_room_baseline: dict[str, Any] | None,
) -> bool:
    if empty_room_baseline is None:
        return False

    empty_room_comparison = build_empty_room_comparison(smoothed, empty_room_baseline)
    departure_level = classify_baseline_departure(empty_room_comparison)
    sen_data = smoothed.get("sen0628") or {}
    microphone_data = smoothed.get("usb_microphone") or {}
    occupancy_confidence = compute_occupancy_confidence(smoothed, empty_room_baseline)
    compared_metric_count = 0
    changed_metric_count = 0
    if isinstance(empty_room_comparison, dict):
        compared_metric_count = int(empty_room_comparison.get("compared_metric_count") or 0)
        changed_metric_count = int(empty_room_comparison.get("changed_metric_count") or 0)
    change_ratio = (
        float(changed_metric_count) / float(compared_metric_count)
        if compared_metric_count > 0
        else 1.0
    )

    near_points = float(sen_data.get("near_points") or 0.0)
    floor_occupied_points = float(sen_data.get("floor_occupied_points") or 0.0)
    max_obstruction_height_mm = float(sen_data.get("max_obstruction_height_mm") or 0.0)
    audio_activity_score = float(microphone_data.get("activity_score") or 0.0)

    no_visible_sen0628_occupancy = (
        near_points <= 0.5
        and floor_occupied_points <= 0.5
        and max_obstruction_height_mm <= 80.0
    )
    quiet_audio = audio_activity_score <= 1.0
    strong_live_occupancy = float(occupancy_confidence.get("occupancy_confidence") or 0.0) >= 0.58

    if strong_live_occupancy:
        return False
    if departure_level == "minimal":
        return True
    if departure_level == "weak" and change_ratio <= 0.16:
        return True
    if departure_level == "weak" and no_visible_sen0628_occupancy and quiet_audio:
        return True
    return False


def build_empty_room_match_summary(
    smoothed: dict[str, Any],
    empty_room_baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    empty_room_comparison = build_empty_room_comparison(smoothed, empty_room_baseline)
    departure_level = classify_baseline_departure(empty_room_comparison)
    matches_empty_room = baseline_matches_empty_room_recording(smoothed, empty_room_baseline)
    sen_data = smoothed.get("sen0628") or {}
    microphone_data = smoothed.get("usb_microphone") or {}

    near_points = float(sen_data.get("near_points") or 0.0)
    floor_occupied_points = float(sen_data.get("floor_occupied_points") or 0.0)
    max_obstruction_height_mm = float(sen_data.get("max_obstruction_height_mm") or 0.0)
    audio_activity_score = float(microphone_data.get("activity_score") or 0.0)
    compared_metric_count = 0
    changed_metric_count = 0
    if isinstance(empty_room_comparison, dict):
        compared_metric_count = int(empty_room_comparison.get("compared_metric_count") or 0)
        changed_metric_count = int(empty_room_comparison.get("changed_metric_count") or 0)
    change_ratio = (
        round(float(changed_metric_count) / float(compared_metric_count), 3)
        if compared_metric_count > 0
        else None
    )
    occupancy_confidence = compute_occupancy_confidence(smoothed, empty_room_baseline)

    return {
        "baseline_available": empty_room_baseline is not None,
        "matches_empty_room": matches_empty_room,
        "departure_level": departure_level,
        "compared_metric_count": compared_metric_count,
        "changed_metric_count": changed_metric_count,
        "change_ratio": change_ratio,
        "near_points": near_points,
        "floor_occupied_points": floor_occupied_points,
        "max_obstruction_height_mm": max_obstruction_height_mm,
        "audio_activity_score": audio_activity_score,
        "occupancy_confidence": occupancy_confidence,
    }


def build_llm_occupancy_evidence(
    smoothed: dict[str, Any],
    fallback_plan: dict[str, Any],
    empty_room_comparison: dict[str, Any] | None,
    empty_room_baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    descriptors = fallback_plan["descriptors"]
    combined_ld = smoothed.get("ld2410c") or {}
    front_ld = smoothed.get("ld2410c_front") or {}
    back_ld = smoothed.get("ld2410c_back") or {}
    sen_data = smoothed.get("sen0628") or {}
    top_changed_metrics = []
    if isinstance(empty_room_comparison, dict):
        changed_metrics = empty_room_comparison.get("top_changed_metrics")
        if isinstance(changed_metrics, list):
            top_changed_metrics = [
                item.get("metric")
                for item in changed_metrics[:MAX_BASELINE_CHANGED_METRICS]
                if isinstance(item, dict) and isinstance(item.get("metric"), str)
            ]
    occupancy_confidence = compute_occupancy_confidence(smoothed, empty_room_baseline)

    return {
        "instruction": (
            "Estimate human count from live sensor evidence relative to the empty-room baseline. "
            "Use heuristic_interpretation only as a fallback if the direct evidence is ambiguous."
        ),
        "baseline_departure_level": classify_baseline_departure(empty_room_comparison),
        "baseline_changed_metric_names": top_changed_metrics,
        "heuristic_count_guess": {
            "presence_count": descriptors.get("presence_count"),
            "figure_count": descriptors.get("figure_count"),
            "presence_activity": descriptors.get("presence_activity"),
        },
        "ld2410c_evidence": {
            "combined_target_state": combined_ld.get("target_state"),
            "combined_out": combined_ld.get("out"),
            "combined_moving_energy": combined_ld.get("moving_energy"),
            "combined_stationary_energy": combined_ld.get("stationary_energy"),
            "combined_detection_distance_cm": combined_ld.get("detection_distance_cm"),
            "front_target_state": front_ld.get("target_state"),
            "front_out": front_ld.get("out"),
            "front_moving_energy": front_ld.get("moving_energy"),
            "front_stationary_energy": front_ld.get("stationary_energy"),
            "back_target_state": back_ld.get("target_state"),
            "back_out": back_ld.get("out"),
            "back_moving_energy": back_ld.get("moving_energy"),
            "back_stationary_energy": back_ld.get("stationary_energy"),
        },
        "sen0628_evidence": {
            "valid_points": sen_data.get("valid_points"),
            "near_points": sen_data.get("near_points"),
            "mid_points": sen_data.get("mid_points"),
            "far_points": sen_data.get("far_points"),
            "floor_occupied_points": sen_data.get("floor_occupied_points"),
            "left_occupied_points": sen_data.get("left_occupied_points"),
            "center_occupied_points": sen_data.get("center_occupied_points"),
            "right_occupied_points": sen_data.get("right_occupied_points"),
            "left_close_points": sen_data.get("left_close_points"),
            "center_close_points": sen_data.get("center_close_points"),
            "right_close_points": sen_data.get("right_close_points"),
            "left_zone_mm": sen_data.get("left_zone_mm"),
            "center_zone_mm": sen_data.get("center_zone_mm"),
            "right_zone_mm": sen_data.get("right_zone_mm"),
            "mean_mm": sen_data.get("mean_mm"),
            "occupied_zone_count": occupancy_confidence.get("occupied_zone_count"),
            "spread_score": occupancy_confidence.get("sen_spread_score"),
        },
        "microphone_evidence": {
            "activity_score": (smoothed.get("usb_microphone") or {}).get("activity_score"),
            "relative_db": (smoothed.get("usb_microphone") or {}).get("relative_db"),
        },
        "empty_room_match": build_empty_room_match_summary(
            smoothed,
            empty_room_baseline,
        ),
        "occupancy_confidence": occupancy_confidence,
    }


def radar_sensor_strength(ld_data):
    if not ld_data:
        return 0.0

    target_state = ld_data.get("target_state")
    out_state = ld_data.get("out") or 0
    moving_energy = ld_data.get("moving_energy") or 0
    stationary_energy = ld_data.get("stationary_energy") or 0
    strength = max(float(moving_energy), float(stationary_energy))

    if target_state == "MOVING_AND_STATIONARY":
        strength += 25.0
    elif target_state == "MOVING_TARGET":
        strength += 18.0
    elif target_state == "STATIONARY_TARGET":
        strength += 12.0
    elif target_state == "NO_TARGET" and out_state == 0:
        strength -= 10.0

    if out_state == 1:
        strength += 10.0
    if ld_data.get("status") == "WAITING_FOR_VALID_FRAME":
        strength -= 5.0

    return strength


def determine_ld_zone_activity(front_data, back_data):
    front_strength = radar_sensor_strength(front_data)
    back_strength = radar_sensor_strength(back_data)
    active_zones = []

    if front_strength >= 18:
        active_zones.append("front")
    if back_strength >= 18:
        active_zones.append("back")

    dominant_zone = None
    if front_strength >= 18 or back_strength >= 18:
        dominant_zone = "front" if front_strength >= back_strength else "back"

    return {
        "front_strength": front_strength,
        "back_strength": back_strength,
        "active_zones": active_zones,
        "dominant_zone": dominant_zone,
    }


def figure_count_to_presence_label(figure_count: int) -> str:
    if figure_count <= 0:
        return "no people"
    if figure_count == 1:
        return "one person"
    if figure_count == 2:
        return "two people"
    if figure_count == 3:
        return "three people"
    return "four or more people"


def presence_label_to_figure_count(presence_count: str, fallback: int = 0) -> int:
    normalized = str(presence_count or "").strip().lower()
    if normalized == "no people":
        return 0
    if normalized == "one person":
        return 1
    if normalized == "two people":
        return 2
    if normalized == "three people":
        return 3
    if normalized == "four or more people":
        return 4
    return fallback


def compute_occupancy_confidence(
    smoothed: dict[str, Any],
    empty_room_baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    combined_ld = smoothed.get("ld2410c") or {}
    front_ld = smoothed.get("ld2410c_front") or {}
    back_ld = smoothed.get("ld2410c_back") or {}
    sen_data = smoothed.get("sen0628") or {}
    microphone_data = smoothed.get("usb_microphone") or {}
    baseline_comparison = build_empty_room_comparison(smoothed, empty_room_baseline)
    departure_level = classify_baseline_departure(baseline_comparison)
    ld_zone_activity = determine_ld_zone_activity(front_ld, back_ld)

    front_strength = float(ld_zone_activity.get("front_strength") or 0.0)
    back_strength = float(ld_zone_activity.get("back_strength") or 0.0)
    combined_strength = float(radar_sensor_strength(combined_ld))
    active_ld_zone_count = len(ld_zone_activity.get("active_zones") or [])
    radar_score = (
        clamp((front_strength + back_strength + (0.6 * combined_strength)) / 180.0, 0.0, 1.0)
    )
    if active_ld_zone_count >= 2:
        radar_score = clamp(radar_score + 0.15, 0.0, 1.0)

    zone_strengths = sen0628_zone_strengths(sen_data)
    occupied_zone_count = sum(1 for value in zone_strengths.values() if float(value or 0.0) >= 2.0)
    occupied_zone_count_strong = sum(1 for value in zone_strengths.values() if float(value or 0.0) >= 4.0)
    near_points = float(sen_data.get("near_points") or 0.0)
    mid_points = float(sen_data.get("mid_points") or 0.0)
    floor_occupied_points = float(sen_data.get("floor_occupied_points") or 0.0)
    sen_spread_score = clamp(
        (
            min(1.0, occupied_zone_count / 3.0)
            + min(1.0, occupied_zone_count_strong / 2.0)
            + min(1.0, floor_occupied_points / 12.0)
            + min(1.0, near_points / 10.0)
            + min(1.0, mid_points / 18.0)
        )
        / 4.0,
        0.0,
        1.0,
    )

    audio_activity_score = float(microphone_data.get("activity_score") or 0.0)
    audio_support = clamp(audio_activity_score / 4.0, 0.0, 1.0)
    if radar_score < 0.22 and sen_spread_score < 0.2:
        audio_support = min(audio_support, 0.18)

    baseline_score = {
        "unknown": 0.0,
        "minimal": 0.0,
        "weak": 0.2,
        "moderate": 0.55,
        "strong": 0.9,
    }.get(departure_level, 0.0)
    total_confidence = clamp(
        (0.38 * radar_score)
        + (0.38 * sen_spread_score)
        + (0.12 * audio_support)
        + (0.22 * baseline_score),
        0.0,
        1.0,
    )

    return {
        "occupancy_confidence": round(total_confidence, 3),
        "radar_score": round(radar_score, 3),
        "sen_spread_score": round(sen_spread_score, 3),
        "audio_support": round(audio_support, 3),
        "baseline_departure_score": round(baseline_score, 3),
        "active_ld_zone_count": active_ld_zone_count,
        "occupied_zone_count": occupied_zone_count,
        "occupied_zone_count_strong": occupied_zone_count_strong,
        "front_strength": round(front_strength, 1),
        "back_strength": round(back_strength, 1),
        "floor_occupied_points": round(floor_occupied_points, 2),
        "near_points": round(near_points, 2),
        "mid_points": round(mid_points, 2),
        "audio_activity_score": round(audio_activity_score, 3),
        "baseline_departure_level": departure_level,
    }


def build_presence_count_diagnostics(
    ld_data,
    sen_data,
    front_ld_data=None,
    back_ld_data=None,
    microphone_data=None,
    occupancy_confidence_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    front_strength = radar_sensor_strength(front_ld_data)
    back_strength = radar_sensor_strength(back_ld_data)
    combined_strength = radar_sensor_strength(ld_data)
    active_ld_zones = sum(1 for strength in (front_strength, back_strength) if strength >= 18.0)
    combined_out = int((ld_data or {}).get("out") or 0)
    moving_energy = float((ld_data or {}).get("moving_energy") or 0.0)
    stationary_energy = float((ld_data or {}).get("stationary_energy") or 0.0)
    target_state = (ld_data or {}).get("target_state")

    zone_strengths = sen0628_zone_strengths(sen_data)
    occupied_zone_count = sum(1 for value in zone_strengths.values() if float(value or 0.0) >= 2.0)
    strong_zone_count = sum(1 for value in zone_strengths.values() if float(value or 0.0) >= 4.0)
    floor_occupied_points = float((sen_data or {}).get("floor_occupied_points") or 0.0)
    near_points = float((sen_data or {}).get("near_points") or 0.0)
    mid_points = float((sen_data or {}).get("mid_points") or 0.0)
    audio_activity_score = float((microphone_data or {}).get("activity_score") or 0.0)

    no_people_signature = (
        target_state == "NO_TARGET"
        and combined_out == 0
        and active_ld_zones == 0
        and occupied_zone_count == 0
        and floor_occupied_points <= 0.5
        and near_points <= 0.5
    )

    def radar_zone_contribution(zone_strength: float, zone_data) -> float:
        if zone_strength < 18.0:
            return 0.0
        contribution = 0.55
        if zone_strength >= 42.0:
            contribution = 0.8
        if zone_strength >= 72.0:
            contribution = 1.0
        zone_state = (zone_data or {}).get("target_state")
        if zone_state == "MOVING_AND_STATIONARY":
            contribution += 0.08
        return contribution

    front_radar_contribution = radar_zone_contribution(front_strength, front_ld_data)
    back_radar_contribution = radar_zone_contribution(back_strength, back_ld_data)
    radar_bridge_contribution = 0.0
    if active_ld_zones >= 2 and combined_strength >= 48.0:
        radar_bridge_contribution += 0.18
    if active_ld_zones >= 2 and combined_out >= 1:
        radar_bridge_contribution += 0.08
    if combined_strength >= 88.0 and active_ld_zones >= 2:
        radar_bridge_contribution += 0.08

    sen_contribution = 0.0
    if occupied_zone_count >= 1:
        sen_contribution += 0.3
    if occupied_zone_count >= 2:
        sen_contribution += 0.45
    if occupied_zone_count >= 3:
        sen_contribution += 0.4
    if strong_zone_count >= 2:
        sen_contribution += 0.22
    if floor_occupied_points >= 8.0:
        sen_contribution += 0.22
    if floor_occupied_points >= 14.0:
        sen_contribution += 0.18
    if near_points >= 8.0 and mid_points >= 12.0:
        sen_contribution += 0.18

    preliminary_score = (
        front_radar_contribution
        + back_radar_contribution
        + radar_bridge_contribution
        + sen_contribution
    )

    audio_contribution = 0.0
    if preliminary_score >= PRESENCE_COUNT_SCORE_THRESHOLDS[2]:
        audio_contribution = min(0.16, audio_activity_score / 18.0)

    total_count_score = preliminary_score + audio_contribution
    multi_zone_corroboration = 0
    if active_ld_zones >= 2:
        multi_zone_corroboration += 1
    if occupied_zone_count >= 2:
        multi_zone_corroboration += 1
    if floor_occupied_points >= 12.0 or (near_points >= 8.0 and mid_points >= 12.0):
        multi_zone_corroboration += 1

    max_supported_count = 1
    if active_ld_zones >= 1 or occupied_zone_count >= 1:
        max_supported_count = 2
    if multi_zone_corroboration >= 2 and total_count_score >= PRESENCE_COUNT_SCORE_THRESHOLDS[3]:
        max_supported_count = 3
    if (
        multi_zone_corroboration >= 3
        and active_ld_zones >= 2
        and occupied_zone_count >= 2
        and total_count_score >= PRESENCE_COUNT_SCORE_THRESHOLDS[4]
    ):
        max_supported_count = 4

    if no_people_signature:
        final_figure_count = 0
        final_presence_label = "no people"
    elif total_count_score < 0.72:
        if active_ld_zones > 0 or occupied_zone_count > 0 or moving_energy >= 25 or stationary_energy >= 25:
            final_figure_count = 0
            final_presence_label = "uncertain occupancy"
        else:
            final_figure_count = 0
            final_presence_label = "no people"
    elif total_count_score < PRESENCE_COUNT_SCORE_THRESHOLDS[2]:
        final_figure_count = 1
        final_presence_label = "one person"
    elif total_count_score < PRESENCE_COUNT_SCORE_THRESHOLDS[3]:
        final_figure_count = 2
        final_presence_label = "two people"
    elif total_count_score < PRESENCE_COUNT_SCORE_THRESHOLDS[4]:
        final_figure_count = min(3, max_supported_count)
        final_presence_label = (
            "three people" if final_figure_count >= 3 else "two people"
        )
    else:
        if max_supported_count >= 4:
            final_figure_count = 4
            final_presence_label = "four or more people"
        elif max_supported_count >= 3:
            final_figure_count = 3
            final_presence_label = "three people"
        else:
            final_figure_count = 2
            final_presence_label = "two people"

    baseline_contribution = 0.0
    total_occupancy_confidence = 0.0
    if isinstance(occupancy_confidence_details, dict):
        baseline_contribution = float(occupancy_confidence_details.get("baseline_departure_score") or 0.0)
        total_occupancy_confidence = float(occupancy_confidence_details.get("occupancy_confidence") or 0.0)

    return {
        "front_radar_contribution": round(front_radar_contribution, 3),
        "back_radar_contribution": round(back_radar_contribution, 3),
        "radar_bridge_contribution": round(radar_bridge_contribution, 3),
        "sen_contribution": round(sen_contribution, 3),
        "audio_contribution": round(audio_contribution, 3),
        "baseline_contribution": round(baseline_contribution, 3),
        "total_occupancy_confidence": round(total_occupancy_confidence, 3),
        "total_count_score": round(total_count_score, 3),
        "multi_zone_corroboration": multi_zone_corroboration,
        "active_ld_zones": active_ld_zones,
        "occupied_zone_count": occupied_zone_count,
        "floor_occupied_points": round(floor_occupied_points, 2),
        "final_presence_label": final_presence_label,
        "final_figure_count": final_figure_count,
    }


def log_presence_count_diagnostics(diagnostics: dict[str, Any]) -> None:
    print(
        "[COUNT_DIAG] "
        f"front_radar={diagnostics.get('front_radar_contribution')} "
        f"back_radar={diagnostics.get('back_radar_contribution')} "
        f"sen0628={diagnostics.get('sen_contribution')} "
        f"audio={diagnostics.get('audio_contribution')} "
        f"baseline={diagnostics.get('baseline_contribution')} "
        f"occupancy_confidence={diagnostics.get('total_occupancy_confidence')} "
        f"count_score={diagnostics.get('total_count_score')} "
        f"corroboration={diagnostics.get('multi_zone_corroboration')} "
        f"presence_label={diagnostics.get('final_presence_label')} "
        f"figure_count={diagnostics.get('final_figure_count')}"
    )


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


def estimate_presence_count(
    ld_data,
    sen_data,
    front_ld_data=None,
    back_ld_data=None,
    microphone_data=None,
    occupancy_confidence_details: dict[str, Any] | None = None,
):
    if not ld_data and not sen_data:
        return "uncertain occupancy"

    diagnostics = build_presence_count_diagnostics(
        ld_data,
        sen_data,
        front_ld_data,
        back_ld_data,
        microphone_data,
        occupancy_confidence_details,
    )
    log_presence_count_diagnostics(diagnostics)
    return str(diagnostics["final_presence_label"])


def classify_depth_band(ld_data, sen_data):
    distance_cm = None
    if ld_data:
        distance_cm = ld_data.get("detection_distance_cm")
        if distance_cm is None:
            distance_cm = ld_data.get("moving_distance_cm") or ld_data.get("stationary_distance_cm")

    if distance_cm is None and sen_data and sen_data.get("mount_mode") != "ceiling_down":
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


def sen0628_is_ceiling_mounted(sen_data):
    return bool(sen_data) and str(sen_data.get("mount_mode") or "").lower() == "ceiling_down"


def sen0628_zone_strengths(sen_data):
    if sen0628_is_ceiling_mounted(sen_data):
        return {
            "left": (sen_data or {}).get("left_occupied_points") or 0,
            "center": (sen_data or {}).get("center_occupied_points") or 0,
            "right": (sen_data or {}).get("right_occupied_points") or 0,
        }
    return {
        "left": (sen_data or {}).get("left_close_points") or 0,
        "center": (sen_data or {}).get("center_close_points") or 0,
        "right": (sen_data or {}).get("right_close_points") or 0,
    }


def sen0628_zone_distance_map(sen_data):
    return {
        "left": (sen_data or {}).get("left_zone_mm"),
        "center": (sen_data or {}).get("center_zone_mm"),
        "right": (sen_data or {}).get("right_zone_mm"),
    }


def sen0628_depth_phrase_for_ceiling_mount(sen_data):
    if not sen0628_is_ceiling_mounted(sen_data):
        return None
    max_height = sen_data.get("max_obstruction_height_mm")
    mean_height = sen_data.get("mean_obstruction_height_mm")
    height_mm = max_height if isinstance(max_height, (int, float)) else mean_height
    if not isinstance(height_mm, (int, float)):
        return "close to the floor plane"
    if height_mm < 120:
        return "very close to the floor plane"
    if height_mm < 450:
        return "slightly raised from the floor"
    if height_mm < 1100:
        return "rising clearly from the floor"
    return "reaching high into the sensing volume"


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

    figure_count = presence_label_to_figure_count(str(presence_count), 1)
    zone_strengths = sen0628_zone_strengths(sen_data)
    sorted_zones = sorted(zone_strengths.items(), key=lambda item: item[1], reverse=True)
    active_zones = [name for name, value in sorted_zones if value >= 2]
    primary_zone = sorted_zones[0][0]
    secondary_zone = sorted_zones[1][0]
    depth_band = classify_depth_band(ld_data, sen_data)
    floor_occupied_points = float((sen_data or {}).get("floor_occupied_points") or 0.0)
    near_points = float((sen_data or {}).get("near_points") or 0.0)

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

    if presence_activity == "presence uncertain" and sorted_zones[0][1] < 2 and figure_count <= 1:
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

    if figure_count == 1:
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

    if figure_count == 2:
        if len(active_zones) >= 2:
            first_zone = active_zones[0]
            second_zone = active_zones[1]
            placement_summary = f"two people split between {first_zone} and {second_zone}"
            placement_prompt = (
                f"show exactly two human figures, one on the {zone_labels[first_zone]}, "
                f"the other on the {zone_labels[second_zone]}, both {depth_labels[depth_band]}"
            )
            layout_mode = "split"
        else:
            placement_summary = f"two people clustered on the {primary_zone}"
            placement_prompt = (
                f"show exactly two human figures clustered on the {zone_labels[primary_zone]}, "
                f"with slight separation between them, both {depth_labels[depth_band]}"
            )
            layout_mode = "clustered"
        return {
            "figure_count": 2,
            "placement_summary": placement_summary,
            "placement_prompt": placement_prompt,
            "depth_band": depth_band,
            "primary_zone": primary_zone,
            "secondary_zone": secondary_zone,
            "active_zones": active_zones,
            "layout_mode": layout_mode,
        }

    distributed_count_text = "three human figures" if figure_count == 3 else "several human figures"
    if len(active_zones) >= 3 or floor_occupied_points >= 15:
        layout_mode = "room-wide"
        placement_summary = f"{presence_count} distributed across the room"
        placement_prompt = (
            f"show {distributed_count_text} spread across the left, center, and right areas of the room, "
            f"with visible spacing between bodies, {depth_labels[depth_band]}"
        )
    elif len(active_zones) >= 2:
        layout_mode = "dispersed"
        placement_summary = f"{presence_count} dispersed across {active_zones[0]} and {active_zones[1]}"
        placement_prompt = (
            f"show {distributed_count_text} distributed across the {zone_labels[active_zones[0]]} "
            f"and the {zone_labels[active_zones[1]]}, with varied spacing and readable separation, "
            f"{depth_labels[depth_band]}"
        )
    elif depth_band == "front" or near_points >= 8:
        layout_mode = "front-heavy"
        placement_summary = f"{presence_count} weighted toward the front"
        placement_prompt = (
            f"show {distributed_count_text} weighted toward the front half of the room, "
            "with one or two figures larger in scale and others receding behind them"
        )
    elif depth_band == "back":
        layout_mode = "back-heavy"
        placement_summary = f"{presence_count} held deeper in the room"
        placement_prompt = (
            f"show {distributed_count_text} deeper toward the back of the room, "
            "with layered spacing and smaller more distant bodies behind nearer ones"
        )
    else:
        layout_mode = "clustered"
        placement_summary = f"{presence_count} gathered near the {primary_zone}"
        placement_prompt = (
            f"show {distributed_count_text} gathered around the {zone_labels[primary_zone]}, "
            "with partial clustering but readable body separation"
        )

    return {
        "figure_count": figure_count,
        "placement_summary": placement_summary,
        "placement_prompt": placement_prompt,
        "depth_band": depth_band,
        "primary_zone": primary_zone,
        "secondary_zone": secondary_zone,
        "active_zones": active_zones,
        "layout_mode": layout_mode,
    }


def apply_ld_zone_bias(people_layout, ld_zone_activity):
    active_ld_zones = list(ld_zone_activity.get("active_zones") or [])
    dominant_ld_zone = ld_zone_activity.get("dominant_zone")
    if not active_ld_zones and not dominant_ld_zone:
        return people_layout

    updated_layout = dict(people_layout)
    if updated_layout["figure_count"] <= 0:
        updated_layout["active_ld_zones"] = active_ld_zones
        updated_layout["dominant_ld_zone"] = dominant_ld_zone
        return updated_layout

    if updated_layout["figure_count"] >= 3 and len(active_ld_zones) >= 2:
        updated_layout["placement_summary"] = (
            f"{figure_count_to_presence_label(int(updated_layout['figure_count']))} layered from front to back"
        )
        updated_layout["placement_prompt"] = (
            "show several human figures clearly visible from the front zone into the back of the room, "
            "with staggered depth, mixed scale, and readable spacing between bodies"
        )
        updated_layout["layout_mode"] = "dispersed"
        updated_layout["primary_zone"] = "front"
        updated_layout["secondary_zone"] = "back"
    elif updated_layout["figure_count"] >= 2 and len(active_ld_zones) >= 2:
        updated_layout["placement_summary"] = "two people split between front and back"
        updated_layout["placement_prompt"] = (
            "show exactly two human figures, one nearer the front of the room and "
            "the other deeper toward the back of the room"
        )
        updated_layout["layout_mode"] = "depth_split"
        updated_layout["primary_zone"] = "front"
        updated_layout["secondary_zone"] = "back"
    elif dominant_ld_zone == "front" and updated_layout["figure_count"] >= 2:
        updated_layout["placement_summary"] = (
            f"{figure_count_to_presence_label(int(updated_layout['figure_count']))} weighted toward the front"
        )
        updated_layout["placement_prompt"] = (
            "show multiple human figures with front-heavy placement, larger figures nearer the foreground "
            "and smaller figures receding behind them"
        )
        updated_layout["layout_mode"] = "front-heavy"
        updated_layout["depth_band"] = "front"
        updated_layout["primary_zone"] = "front"
        updated_layout["secondary_zone"] = "back"
    elif dominant_ld_zone == "front":
        updated_layout["placement_summary"] = "one person near the front"
        updated_layout["placement_prompt"] = (
            "show exactly one human figure toward the front of the room, leaning nearer to the foreground"
        )
        updated_layout["depth_band"] = "front"
        updated_layout["primary_zone"] = "front"
        updated_layout["secondary_zone"] = "back"
    elif dominant_ld_zone == "back" and updated_layout["figure_count"] >= 2:
        updated_layout["placement_summary"] = (
            f"{figure_count_to_presence_label(int(updated_layout['figure_count']))} held toward the back"
        )
        updated_layout["placement_prompt"] = (
            "show multiple human figures deeper in the room, layered toward the back with visible spacing"
        )
        updated_layout["layout_mode"] = "back-heavy"
        updated_layout["depth_band"] = "back"
        updated_layout["primary_zone"] = "back"
        updated_layout["secondary_zone"] = "front"
    elif dominant_ld_zone == "back":
        updated_layout["placement_summary"] = "one person near the back"
        updated_layout["placement_prompt"] = (
            "show exactly one human figure toward the back of the room, set deeper in the space"
        )
        updated_layout["depth_band"] = "back"
        updated_layout["primary_zone"] = "back"
        updated_layout["secondary_zone"] = "front"

    updated_layout["active_ld_zones"] = active_ld_zones
    updated_layout["dominant_ld_zone"] = dominant_ld_zone
    return updated_layout


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
    elif figure_count == 2:
        if layout_mode in ("split", "depth_split"):
            modifiers.append("two separated figures")
        else:
            modifiers.append("clustered figure grouping")
    else:
        if layout_mode == "room-wide":
            modifiers.append("distributed human presence across the space")
        elif layout_mode == "dispersed":
            modifiers.append("several spaced bodies across the room")
        elif layout_mode == "front-heavy":
            modifiers.append("front-weighted multi-figure grouping")
        elif layout_mode == "back-heavy":
            modifiers.append("back-layered figure grouping")
        else:
            modifiers.append("clustered multi-figure grouping")

    if primary_zone == "left":
        modifiers.append("figure grouping shifted toward left side")
    elif primary_zone == "right":
        modifiers.append("figure grouping shifted toward right side")
    elif primary_zone == "front":
        modifiers.append("figure grouping shifted toward front")
    elif primary_zone == "back":
        modifiers.append("figure grouping shifted toward back")
    else:
        modifiers.append("one centered or slightly off-center figure emphasis")

    if depth_band == "front":
        modifiers.append("foreground weighting increased")
    elif depth_band == "back":
        modifiers.append("figures shifted toward back depth")
    else:
        modifiers.append("figures shifted toward middle depth")

    if figure_count >= 3:
        modifiers.append("multiple bodies visible at first glance")
    if layout_mode == "clustered":
        modifiers.append("tighter figure spacing")
    elif layout_mode in ("split", "depth_split"):
        modifiers.append("wider spacing between figures")
    elif layout_mode == "room-wide":
        modifiers.append("left-center-right spread retained")
    elif layout_mode in ("dispersed", "front-heavy", "back-heavy"):
        modifiers.append("mixed scale and spacing across figures")
    else:
        modifiers.append("pose and silhouette may stay somewhat ambiguous")

    modifiers.append("room retained, occupancy pattern revised")
    return modifiers[:MAX_LANGUAGE_PASS_PROMPT_MODIFIERS]


def interpret_sen0628_spatial_estimate(sen_data):
    if not sen_data:
        return "SEN0628 spatial estimate unavailable"

    zone_close_counts = sen0628_zone_strengths(sen_data)
    strongest_zone = max(zone_close_counts.items(), key=lambda item: item[1])[0]
    strongest_value = zone_close_counts[strongest_zone]
    active_zones = [name for name, count in zone_close_counts.items() if count >= 3]
    zone_distance_map = sen0628_zone_distance_map(sen_data)
    strongest_distance = zone_distance_map.get(strongest_zone)

    if sen0628_is_ceiling_mounted(sen_data):
        depth_text = sen0628_depth_phrase_for_ceiling_mount(sen_data)
    elif strongest_distance is None:
        depth_text = "with uncertain depth"
    elif strongest_distance < 900:
        depth_text = "close to the sensor plane"
    elif strongest_distance < 1800:
        depth_text = "in the mid-room"
    else:
        depth_text = "deeper in the room"

    if strongest_value < 2:
        if sen0628_is_ceiling_mounted(sen_data):
            return "SEN0628 suggests only a weak floor trace"
        return "SEN0628 suggests only a weak central trace"
    if len(active_zones) >= 3:
        if sen0628_is_ceiling_mounted(sen_data):
            return f"SEN0628 suggests floor occupancy spread across the mapped area, {depth_text}"
        return f"SEN0628 suggests occupancy spread across the room, {depth_text}"
    if len(active_zones) == 2:
        zone_pair = " and ".join(active_zones)
        if sen0628_is_ceiling_mounted(sen_data):
            return f"SEN0628 suggests floor occupancy shared across the {zone_pair} floor zones, {depth_text}"
        return f"SEN0628 suggests occupancy shared across the {zone_pair} zones, {depth_text}"
    if strongest_zone == "left":
        if sen0628_is_ceiling_mounted(sen_data):
            return f"SEN0628 suggests a floor footprint concentrated on the left side, {depth_text}"
        return f"SEN0628 suggests occupancy concentrated on the left side, {depth_text}"
    if strongest_zone == "right":
        if sen0628_is_ceiling_mounted(sen_data):
            return f"SEN0628 suggests a floor footprint concentrated on the right side, {depth_text}"
        return f"SEN0628 suggests occupancy concentrated on the right side, {depth_text}"
    if sen0628_is_ceiling_mounted(sen_data):
        return f"SEN0628 suggests a floor footprint concentrated near the center, {depth_text}"
    return f"SEN0628 suggests occupancy concentrated near the center, {depth_text}"


def sen0628_location_detail(sen_data):
    if not sen_data:
        return None

    zone_strengths = sen0628_zone_strengths(sen_data)
    strongest_zone = max(zone_strengths.items(), key=lambda item: item[1])[0]
    strongest_value = zone_strengths[strongest_zone]
    sorted_strengths = sorted(zone_strengths.values(), reverse=True)
    second_value = sorted_strengths[1] if len(sorted_strengths) > 1 else 0
    active_zones = [name for name, value in zone_strengths.items() if value >= 3]

    if strongest_value < 2:
        if sen0628_is_ceiling_mounted(sen_data):
            return "with only a faint floor trace"
        return "with only a vague central trace"
    if len(active_zones) >= 3:
        if sen0628_is_ceiling_mounted(sen_data):
            return "spread across the floor map"
        return "spread across the full field of view"
    if len(active_zones) == 2:
        if sen0628_is_ceiling_mounted(sen_data):
            return "distributed across adjacent floor zones"
        return "distributed across adjacent zones"
    if strongest_value - second_value <= 1 and strongest_value >= 3:
        if sen0628_is_ceiling_mounted(sen_data):
            return "bridging between floor zones"
        return "hovering between zones"
    if strongest_zone == "left":
        return "biased toward the left side"
    if strongest_zone == "right":
        return "biased toward the right side"
    if sen0628_is_ceiling_mounted(sen_data):
        return "centered under the sensor"
    return "centered in front of the system"


def interpret_sen0628_figure_side(sen_data):
    if not sen_data:
        return "middle"

    raw_zone_strengths = sen0628_zone_strengths(sen_data)
    zone_strengths = {
        "left": raw_zone_strengths["left"],
        "middle": raw_zone_strengths["center"],
        "right": raw_zone_strengths["right"],
    }
    strongest_zone = max(zone_strengths.items(), key=lambda item: item[1])[0]
    strongest_value = zone_strengths[strongest_zone]
    if strongest_value < 2:
        return "middle"
    return strongest_zone


def interpret_presence_location(ld_data, sen_data, ld_zone_activity=None):
    if not ld_data and not sen_data and not ld_zone_activity:
        return "location uncertain"

    distance_cm = None
    if ld_data:
        distance_cm = ld_data.get("detection_distance_cm")
        if distance_cm is None:
            distance_cm = ld_data.get("moving_distance_cm") or ld_data.get("stationary_distance_cm")

    if distance_cm is None and sen_data and not sen0628_is_ceiling_mounted(sen_data):
        center_mm = sen_data.get("center_mm")
        mean_mm = sen_data.get("mean_mm")
        if center_mm is not None:
            distance_cm = center_mm / 10.0
        elif mean_mm is not None:
            distance_cm = mean_mm / 10.0

    if distance_cm is None and sen0628_is_ceiling_mounted(sen_data):
        depth_phrase = sen0628_depth_phrase_for_ceiling_mount(sen_data) or "close to the floor plane"
    elif distance_cm is None:
        depth_phrase = "at an uncertain depth"
    elif distance_cm < 90:
        depth_phrase = "very close to the sensing system"
    elif distance_cm < 160:
        depth_phrase = "in the near field"
    elif distance_cm < 260:
        depth_phrase = "in the middle distance"
    else:
        depth_phrase = "farther back in the space"

    active_ld_zones = []
    dominant_ld_zone = None
    if isinstance(ld_zone_activity, dict):
        active_ld_zones = list(ld_zone_activity.get("active_zones") or [])
        dominant_ld_zone = ld_zone_activity.get("dominant_zone")

    if len(active_ld_zones) >= 2:
        return "split between the front and back zones"
    if dominant_ld_zone == "front":
        return "near the front of the room"
    if dominant_ld_zone == "back":
        return "toward the back of the room"

    horizontal_phrase = sen0628_location_detail(sen_data)
    if horizontal_phrase is None:
        return depth_phrase
    return f"{depth_phrase}, {horizontal_phrase}"


def interpret_lighting():
    return "lighting unavailable"


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


def interpret_spatial_impression(ld_data, sen_data, ld_zone_activity=None):
    if not sen_data and not ld_data:
        return "spatial impression uncertain"

    if ld_data:
        target_state = ld_data.get("target_state")
        out_state = ld_data.get("out")
        if target_state == "NO_TARGET" and out_state == 0:
            return "open, empty, and waiting"

    if sen0628_is_ceiling_mounted(sen_data):
        occupied_points = (sen_data or {}).get("floor_occupied_points") or 0
        occupied_zone_counts = [
            (sen_data or {}).get("left_occupied_points") or 0,
            (sen_data or {}).get("center_occupied_points") or 0,
            (sen_data or {}).get("right_occupied_points") or 0,
        ]
        occupied_zones = sum(1 for value in occupied_zone_counts if value >= 3)
        tall_points = (sen_data or {}).get("tall_obstruction_points") or 0
        if len((ld_zone_activity or {}).get("active_zones") or []) >= 2:
            return "layered across the floor plan"
        if occupied_points >= 18 and occupied_zones >= 2:
            return "broad floor coverage"
        if tall_points >= 8:
            return "vertically prominent from overhead"
        if occupied_zones >= 2:
            return "distributed across the floor"
        if occupied_points >= 4:
            return "localized floor activity"
        return "mostly clear floor plane"

    near_points = (sen_data or {}).get("near_points") or 0
    far_points = (sen_data or {}).get("far_points") or 0
    zone_counts = list(sen0628_zone_strengths(sen_data).values())
    occupied_zones = sum(1 for value in zone_counts if value >= 3)
    detection_distance = (ld_data or {}).get("detection_distance_cm")

    active_ld_zones = []
    if isinstance(ld_zone_activity, dict):
        active_ld_zones = list(ld_zone_activity.get("active_zones") or [])

    if len(active_ld_zones) >= 2:
        return "deep and layered"
    if near_points >= 10 and occupied_zones >= 2:
        return "shallow and fragmented"
    if far_points >= 20 and occupied_zones <= 1:
        return "deep and receding"
    if occupied_zones >= 3:
        return "wide and distributed"
    if detection_distance is not None and detection_distance < 120:
        return "compressed and near"
    if active_ld_zones == ["front"]:
        return "weighted toward the front"
    if active_ld_zones == ["back"]:
        return "weighted toward the back"
    return "contained and interior"


def interpret_abstract_background(bme_data):
    lighting = interpret_lighting()
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
    placement_prompt = str(descriptors.get("placement_prompt") or "").strip()

    if figure_count == 0 or presence == "no presence" or presence_count == "no people":
        return (
            "keep the installation space empty and unoccupied, preserve the same room with no visible person, "
            "no occupant, no silhouette, no crowd"
        )

    zone_text = {
        "left": "toward the left side",
        "center": "near the middle zone",
        "right": "toward the right side",
        "front": "toward the front zone",
        "back": "toward the back zone",
    }.get(str(primary_zone), "near the middle zone")
    depth_text = {
        "front": "leaning nearer to the foreground",
        "mid-room": "held around middle depth",
        "back": "leaning farther back in the room",
    }.get(str(depth_band), "held around middle depth")
    if figure_count >= 4:
        count_text = "require several visible human figures as the occupancy revision"
        exact_count_text = (
            "show approximately four or more adult human figures, clearly visible at first glance, "
            "without turning the room into a dense crowd"
        )
    elif figure_count == 3:
        count_text = "require three visible human figures as the occupancy revision"
        exact_count_text = "show approximately three adult human figures, all visibly readable in the room"
    elif figure_count == 2:
        count_text = "require two visible human figures as the occupancy revision"
        exact_count_text = "show exactly two adult human figures and no additional people anywhere in the frame"
    else:
        count_text = "require one visible human figure as the occupancy revision"
        exact_count_text = "show exactly one adult human figure and no additional people anywhere in the frame"

    if figure_count >= 3:
        spacing_text = (
            "show several bodies visible at first glance, keep spacing, clustering, and scale differences readable"
            if layout_mode in ("dispersed", "room-wide", "front-heavy", "back-heavy")
            else "keep the multi-figure grouping legible, with overlapping ambiguity but still readable separation"
        )
    elif figure_count == 2:
        spacing_text = (
            "keep the two bodies clearly separated with readable negative space between them"
            if layout_mode in ("split", "depth_split")
            else "keep the two bodies close but still clearly distinct from each other"
        )
    else:
        spacing_text = "keep the single body stable and clearly readable"
    audio_text = {
        "strong shared room noise": "treat the audio as strong evidence of multiple occupants or sustained shared activity, but do not force extra bodies without support from the other sensors",
        "moderate human-made room noise": "treat the audio as supporting evidence of ongoing occupancy and possible multiple people, while keeping the count conservative",
        "light room noise": "treat the audio as a weak occupancy cue only",
        "very quiet room": "treat the audio as weak evidence for additional occupants",
        "audio unavailable": "do not infer extra occupants from missing audio",
    }.get(str(descriptors.get("audio_activity")), "keep the mood restrained and observational")
    multi_figure_text = (
        "for three or more figures, show distributed human presence across the space and multiple bodies visible at first glance, "
        if figure_count >= 3
        else ""
    )
    return (
        f"human presence is being inferred as {presence}, {count_text}, {exact_count_text}, {zone_text}, {depth_text}, "
        f"{placement_prompt}, "
        "the room must read as visibly occupied, the figures must be noticeable at first glance, "
        "render full adult human bodies with stable anatomy and normal proportions, "
        "show visible human bodies instead of fragments, floating parts, shadows, or implied presence only, "
        "keep the figures legible in the frame with enough scale to stand out from the background, "
        "keep figures inferred and imperfect rather than portrait-clean, with faces and identities unresolved, "
        f"{multi_figure_text}"
        "avoid crowds, duplicated bodies, merged silhouettes, extra limbs, and distorted anatomy, "
        f"{spacing_text}, {audio_text}, keep the room secondary to the clearly readable human figures"
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
        figure_visibility_text = (
            "for three or more inferred figures, show several human figures clearly visible, distributed human presence across the space, "
            "and multiple bodies readable at first glance"
            if int(descriptors.get("figure_count", 0)) >= 3
            else "figures should occupy a noticeable amount of the frame and remain legible at first glance"
        )
        composition_directive = (
            f"{COMPOSITION_DIRECTIVE}, keep the same general room and camera view, the room should visibly contain occupants when presence is inferred, "
            "do not render an empty room when presence is inferred, keep background secondary to visible human presence, "
            "preserve the room while allowing figure arrangement to change, figures should occupy a noticeable amount "
            f"of the frame, remain legible at first glance, and serve as the main changing element rather than tiny distant background details, {figure_visibility_text}"
        )
        if int(descriptors.get("figure_count", 0)) >= 3:
            negative_prompt = (
                f"{OPTIONAL_TEXT_UI_EXCLUSION_NEGATIVE_PROMPT}, "
                "empty room, empty scene, unoccupied room, no people, no person, absent occupant, vacant interior, "
                "tiny distant people, people barely visible, occupants lost in the background, alternate room, different room, "
                "changed architecture, changed camera angle, new furniture, duplicate body, duplicated person, merged bodies, "
                "fused silhouettes, extra limbs, extra arms, extra legs, disconnected limbs, cropped body, cut off body, "
                "floating torso, malformed anatomy"
            )
        else:
            negative_prompt = (
                f"{OPTIONAL_TEXT_UI_EXCLUSION_NEGATIVE_PROMPT}, "
                "empty room, empty scene, unoccupied room, no people, no person, absent occupant, vacant interior, "
                "tiny distant people, people barely visible, occupants lost in the background, alternate room, different room, "
                "changed architecture, changed camera angle, new furniture, crowd, group, extra person, extra people, "
                "duplicate body, duplicated person, merged bodies, fused silhouettes, extra limbs, extra arms, extra legs, "
                "disconnected limbs, cropped body, cut off body, floating torso, malformed anatomy"
            )

    return {
        "base_scene_prompt": BASE_SCENE_PROMPT,
        "background_continuity_directive": background_continuity_directive,
        "people_directive": people_directive,
        "figure_variation_directive": figure_variation_directive,
        "composition_directive": composition_directive,
        "negative_prompt": negative_prompt,
    }


def get_seed_cycle_values() -> list[int]:
    valid_values = [int(value) for value in IMAGE_SEED_CYCLE_VALUES if isinstance(value, int)]
    if valid_values:
        return valid_values
    return [DEFAULT_OCCUPIED_IMAGE_SEED, DEFAULT_EMPTY_IMAGE_SEED]


def resolve_seed_cycle_started_at(
    coordinator: InterpretationCoordinator,
    previous_shared_state: dict[str, Any],
    now: float,
) -> float:
    if coordinator.seed_cycle_started_at is not None:
        return coordinator.seed_cycle_started_at

    candidate = previous_shared_state.get("seed_cycle_started_at")
    if isinstance(candidate, (int, float)) and candidate > 0:
        coordinator.seed_cycle_started_at = float(candidate)
    else:
        coordinator.seed_cycle_started_at = now
    return coordinator.seed_cycle_started_at


def build_seed_cycle_state(
    coordinator: InterpretationCoordinator,
    previous_shared_state: dict[str, Any],
    now: float,
) -> dict[str, Any]:
    seeds = get_seed_cycle_values()
    interval_seconds = max(1.0, float(IMAGE_SEED_CYCLE_INTERVAL_SECONDS))
    started_at = resolve_seed_cycle_started_at(coordinator, previous_shared_state, now)
    elapsed_seconds = max(0.0, now - started_at)
    cycle_slot = int(elapsed_seconds // interval_seconds)
    cycle_index = cycle_slot % len(seeds)
    seconds_into_slot = elapsed_seconds % interval_seconds
    seconds_until_next_seed = max(0.0, interval_seconds - seconds_into_slot)

    return {
        "started_at": round(started_at, 3),
        "interval_seconds": interval_seconds,
        "seeds": seeds,
        "cycle_slot": cycle_slot,
        "cycle_index": cycle_index,
        "current_seed": seeds[cycle_index],
        "seconds_until_next_seed": round(seconds_until_next_seed, 2),
    }


def has_active_seed_cycle(shared_state: dict[str, Any] | None) -> bool:
    if not isinstance(shared_state, dict):
        return False
    seed_cycle = shared_state.get("seed_cycle")
    if not isinstance(seed_cycle, dict):
        return False
    seeds = seed_cycle.get("seeds")
    interval_seconds = seed_cycle.get("interval_seconds")
    return (
        isinstance(seeds, list)
        and len(seeds) > 1
        and isinstance(interval_seconds, (int, float))
        and float(interval_seconds) > 0
    )


def get_seed_cycle_interval_seconds(shared_state: dict[str, Any] | None) -> float | None:
    if not isinstance(shared_state, dict):
        return None
    seed_cycle = shared_state.get("seed_cycle")
    if not isinstance(seed_cycle, dict):
        return None
    interval_seconds = seed_cycle.get("interval_seconds")
    if isinstance(interval_seconds, (int, float)) and float(interval_seconds) > 0:
        return float(interval_seconds)
    return None


def select_generation_seed(shared_state: dict[str, Any]) -> int:
    seed_cycle = shared_state.get("seed_cycle")
    if isinstance(seed_cycle, dict):
        current_seed = seed_cycle.get("current_seed")
        if isinstance(current_seed, int):
            return current_seed
    descriptors = shared_state.get("interpreted_state") or {}
    if isinstance(descriptors, dict) and descriptors.get("figure_count", 0) > 0:
        return DEFAULT_OCCUPIED_IMAGE_SEED
    return DEFAULT_EMPTY_IMAGE_SEED


def interpret_sensor_state(smoothed):
    combined_ld_data = smoothed.get("ld2410c")
    front_ld_data = smoothed.get("ld2410c_front")
    back_ld_data = smoothed.get("ld2410c_back")
    microphone_data = smoothed.get("usb_microphone")
    ld_zone_activity = determine_ld_zone_activity(front_ld_data, back_ld_data)
    occupancy_confidence = compute_occupancy_confidence(smoothed)

    presence_activity = interpret_presence(combined_ld_data)
    presence_count = estimate_presence_count(
        combined_ld_data,
        smoothed.get("sen0628"),
        front_ld_data,
        back_ld_data,
        microphone_data,
        occupancy_confidence,
    )
    people_layout = estimate_people_layout(
        combined_ld_data,
        smoothed.get("sen0628"),
        presence_activity,
        presence_count,
    )
    people_layout = apply_ld_zone_bias(people_layout, ld_zone_activity)

    descriptors = {
        "presence_activity": presence_activity,
        "presence_count": presence_count,
        "sen0628_spatial_estimate": interpret_sen0628_spatial_estimate(smoothed.get("sen0628")),
        "sen0628_figure_side": interpret_sen0628_figure_side(smoothed.get("sen0628")),
        "presence_location": interpret_presence_location(
            combined_ld_data,
            smoothed.get("sen0628"),
            ld_zone_activity,
        ),
        "lighting_condition": interpret_lighting(),
        "atmospheric_condition": interpret_atmosphere(smoothed.get("bme688")),
        "spatial_impression": interpret_spatial_impression(
            combined_ld_data,
            smoothed.get("sen0628"),
            ld_zone_activity,
        ),
        "abstract_background": interpret_abstract_background(smoothed.get("bme688")),
        "audio_activity": interpret_audio_activity(microphone_data),
        "audio_device": None if not isinstance(microphone_data, dict) else microphone_data.get("device_name"),
        "audio_relative_db": None if not isinstance(microphone_data, dict) else microphone_data.get("relative_db"),
        "front_presence": interpret_presence(front_ld_data),
        "back_presence": interpret_presence(back_ld_data),
        "front_strength": round(ld_zone_activity["front_strength"], 1),
        "back_strength": round(ld_zone_activity["back_strength"], 1),
        "active_ld_zones": list(ld_zone_activity["active_zones"]),
        "dominant_ld_zone": ld_zone_activity["dominant_zone"],
        "occupancy_confidence": occupancy_confidence.get("occupancy_confidence"),
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
    descriptors["uncertainty_score"] = round_generation_value(
        derive_uncertainty_score_from_descriptors(descriptors)
    )
    descriptors["sensor_variation_profile"] = build_sensor_variation_profile(smoothed, descriptors)
    descriptors["figure_variation_modifiers"] = build_figure_variation_modifiers(descriptors)
    return descriptors


def build_live_inference_lines(descriptors):
    presence_activity = descriptors["presence_activity"]
    atmosphere = descriptors["atmospheric_condition"]
    depth_band = descriptors["depth_band"]
    primary_zone = descriptors["primary_zone"]
    figure_count = int(descriptors["figure_count"])
    layout_mode = descriptors["layout_mode"]
    spatial_certainty = descriptors["spatial_certainty"]
    audio_activity = descriptors.get("audio_activity")

    movement_line_map = {
        "no presence": "occupancy weighting reduced",
        "still presence": "occupancy held steady",
        "active presence": "occupancy revision active",
        "intermittent movement": "occupancy state oscillating",
        "presence uncertain": "occupancy remains provisional",
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
    elif figure_count >= 4:
        occupancy_line = "room-wide multi-figure state favored"
    elif figure_count == 3:
        occupancy_line = "three-figure occupancy favored"
    elif layout_mode in ("split", "depth_split"):
        occupancy_line = "split figure layout favored"
    else:
        occupancy_line = "clustered figure state favored"

    if primary_zone == "left":
        lateral_line = "left trace dominant"
    elif primary_zone == "right":
        lateral_line = "right trace dominant"
    elif primary_zone == "front":
        lateral_line = "front trace dominant"
    elif primary_zone == "back":
        lateral_line = "back trace dominant"
    else:
        lateral_line = "central trace dominant"

    if depth_band == "front":
        depth_line = "near-field signal stronger"
    elif depth_band == "back":
        depth_line = "depth weighting shifted inward"
    else:
        depth_line = "mid-depth weighting held"

    dominant_ld_zone = descriptors.get("dominant_ld_zone")
    if dominant_ld_zone == "front":
        radar_line = "front radar leads"
    elif dominant_ld_zone == "back":
        radar_line = "back radar leads"
    elif len(descriptors.get("active_ld_zones", [])) >= 2:
        radar_line = "front/back radar split held"
    else:
        radar_line = "radar zone weighting partial"

    lines = [
        "background continuity held across the room",
        movement_line_map.get(presence_activity, "signal weighting is being revised"),
        f"{occupancy_line} under current sensor weighting",
        f"{lateral_line}; {depth_line}",
        radar_line,
        build_audio_live_line(audio_activity),
        spatial_certainty,
        atmosphere_line_map.get(atmosphere, "air reading remains only partially resolved"),
        build_image_foreshadow_line(descriptors),
    ]
    return lines


def apply_count_hysteresis(
    scene_plan: dict[str, Any],
    smoothed: dict[str, Any],
    previous_shared_state: dict[str, Any] | None,
    empty_room_baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(previous_shared_state, dict):
        return scene_plan

    previous_descriptors = previous_shared_state.get("interpreted_state")
    if not isinstance(previous_descriptors, dict):
        return scene_plan

    descriptors = dict(scene_plan.get("descriptors") or {})
    if not descriptors:
        return scene_plan

    previous_count = int(previous_descriptors.get("figure_count", 0) or 0)
    current_count = int(descriptors.get("figure_count", 0) or 0)
    if previous_count == current_count:
        return scene_plan

    occupancy_confidence = compute_occupancy_confidence(smoothed, empty_room_baseline)
    count_diagnostics = build_presence_count_diagnostics(
        smoothed.get("ld2410c"),
        smoothed.get("sen0628"),
        smoothed.get("ld2410c_front"),
        smoothed.get("ld2410c_back"),
        smoothed.get("usb_microphone"),
        occupancy_confidence,
    )
    count_score = float(count_diagnostics.get("total_count_score") or 0.0)
    corroboration = int(count_diagnostics.get("multi_zone_corroboration") or 0)

    adjusted_count = current_count
    if current_count > previous_count:
        required_score = PRESENCE_COUNT_SCORE_THRESHOLDS.get(current_count, PRESENCE_COUNT_SCORE_THRESHOLDS[4])
        upgrade_margin = 0.18 if current_count <= 2 else 0.32
        if current_count >= 3 and corroboration < 2:
            adjusted_count = previous_count
        elif previous_count == 0 and current_count >= 2 and corroboration < 2:
            adjusted_count = 1 if count_score >= PRESENCE_COUNT_SCORE_THRESHOLDS[1] else 0
        elif count_score < (required_score + upgrade_margin):
            adjusted_count = previous_count
    else:
        hold_threshold = {
            1: PRESENCE_COUNT_SCORE_THRESHOLDS[1] - 0.12,
            2: PRESENCE_COUNT_SCORE_THRESHOLDS[2] - 0.2,
            3: PRESENCE_COUNT_SCORE_THRESHOLDS[3] - 0.28,
            4: PRESENCE_COUNT_SCORE_THRESHOLDS[4] - 0.3,
        }.get(previous_count, 0.0)
        if count_score >= hold_threshold:
            adjusted_count = previous_count

    if adjusted_count == current_count:
        return scene_plan

    updated_descriptors = dict(descriptors)
    updated_descriptors["figure_count"] = adjusted_count
    updated_descriptors["presence_count"] = figure_count_to_presence_label(adjusted_count)
    if adjusted_count <= 0 and updated_descriptors.get("presence_activity") != "presence uncertain":
        updated_descriptors["presence_activity"] = "no presence"

    people_layout = estimate_people_layout(
        smoothed.get("ld2410c"),
        smoothed.get("sen0628"),
        str(updated_descriptors.get("presence_activity")),
        str(updated_descriptors.get("presence_count")),
    )
    people_layout = apply_ld_zone_bias(
        people_layout,
        determine_ld_zone_activity(smoothed.get("ld2410c_front"), smoothed.get("ld2410c_back")),
    )
    updated_descriptors["figure_count"] = people_layout["figure_count"]
    updated_descriptors["placement_summary"] = people_layout["placement_summary"]
    updated_descriptors["placement_prompt"] = people_layout["placement_prompt"]
    updated_descriptors["depth_band"] = people_layout["depth_band"]
    updated_descriptors["primary_zone"] = people_layout["primary_zone"]
    updated_descriptors["secondary_zone"] = people_layout["secondary_zone"]
    updated_descriptors["active_zones"] = people_layout["active_zones"]
    updated_descriptors["layout_mode"] = people_layout["layout_mode"]
    updated_descriptors["figure_variation_modifiers"] = build_figure_variation_modifiers(updated_descriptors)

    updated_scene_plan = dict(scene_plan)
    updated_scene_plan["descriptors"] = updated_descriptors
    updated_scene_plan["prompt_sections"] = build_prompt_sections(updated_descriptors)
    updated_scene_plan["prompt"] = build_image_prompt(updated_descriptors)
    updated_scene_plan["live_lines"] = build_live_inference_lines(updated_descriptors)
    updated_scene_plan["state_signature_descriptors"] = updated_descriptors
    return updated_scene_plan


def build_background_continuity_directive() -> str:
    return (
        "preserve the same general installation room across generations, similar camera angle, similar framing, "
        "similar walls, floor, and architectural layout, keep environment structure consistent while allowing occupant revision, "
        "background continuity is a secondary anchor and the figures should remain the main changing element"
    )


def build_figure_variation_directive(descriptors: dict[str, Any]) -> str:
    modifiers = list(descriptors.get("figure_variation_modifiers") or [])
    sensor_variation_profile = descriptors.get("sensor_variation_profile") or {}
    sensor_modifiers = sensor_variation_profile.get("prompt_modifiers") or []
    for modifier in sensor_modifiers:
        if isinstance(modifier, str) and modifier not in modifiers:
            modifiers.append(modifier)
    modifier_text = ", ".join(str(item) for item in modifiers if isinstance(item, str))
    if not modifier_text:
        modifier_text = "room retained, occupancy pattern revised"
    return (
        "treat human figures as the primary changing element, keep the count and placement stable, "
        "allow only subtle variation in pose and clothing, preserve full-body readability and clean silhouettes, "
        f"keep the room broadly stable, {modifier_text}"
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
    generation_controls = build_generation_controls_from_uncertainty(
        derive_uncertainty_score_from_descriptors(descriptors),
        str((descriptors.get("sensor_variation_profile") or {}).get("fingerprint") or ""),
    )
    return {
        "descriptors": descriptors,
        "prompt_sections": prompt_sections,
        "prompt": prompt,
        "live_lines": live_lines,
        "generation_controls": generation_controls,
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
        cleaned = compact_machine_phrase(item, max_length=58)
        if cleaned:
            cleaned_lines.append(cleaned)

    return cleaned_lines[:MAX_AI_LIVE_LINES] or fallback_lines


def build_language_pass_fallback_preview(fallback_plan: dict[str, Any]) -> str:
    fallback_descriptors = fallback_plan["descriptors"]
    if fallback_descriptors.get("figure_count", 0) > 0:
        return "upcoming image preview: " + fallback_descriptors["placement_summary"]
    return "upcoming image preview: likely empty installation space"


def coerce_descriptors_to_empty_room(
    descriptors: dict[str, Any],
    fallback_plan: dict[str, Any],
) -> dict[str, Any]:
    coerced = dict(descriptors)
    fallback_descriptors = fallback_plan["descriptors"]
    coerced["presence_activity"] = "no presence"
    coerced["presence_count"] = "no people"
    coerced["figure_count"] = 0
    coerced["presence_location"] = "baseline-like empty room; no visible occupant"
    coerced["placement_summary"] = "likely empty room"
    coerced["placement_prompt"] = "show no visible people in the room"
    coerced["layout_mode"] = "empty"
    coerced["active_zones"] = []
    coerced["primary_zone"] = str(fallback_descriptors.get("primary_zone", "center"))
    coerced["secondary_zone"] = str(fallback_descriptors.get("secondary_zone", "center"))
    coerced["depth_band"] = str(fallback_descriptors.get("depth_band", "mid-room"))
    coerced["spatial_certainty"] = "baseline supports likely empty room"
    coerced["figure_variation_modifiers"] = build_figure_variation_modifiers(coerced)
    return coerced


def sanitize_choice(value: Any, allowed_values: set[str], fallback: str) -> str:
    if not isinstance(value, str):
        return fallback
    cleaned = " ".join(value.split()).strip().lower()
    if cleaned in allowed_values:
        return cleaned
    return fallback


def sanitize_active_zones(value: Any, fallback: list[str]) -> list[str]:
    allowed_values = {"front", "back", "left", "center", "right"}
    if not isinstance(value, list):
        return fallback

    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        normalized = " ".join(item.split()).strip().lower()
        if normalized not in allowed_values:
            continue
        if normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned or fallback


def sanitize_figure_count(value: Any, fallback: int) -> int:
    if isinstance(value, bool):
        return fallback
    if isinstance(value, (int, float)):
        count = int(round(float(value)))
        if 0 <= count <= 5:
            return count
    return fallback


def sanitize_short_text(value: Any, fallback: str, *, max_length: int) -> str:
    if not isinstance(value, str):
        return fallback
    cleaned = " ".join(value.split()).strip()
    if not cleaned:
        return fallback
    cleaned = cleaned[:max_length].strip(" ,.;:-")
    return cleaned or fallback


def sanitize_scene_interpretation(
    scene_interpretation: dict[str, Any],
    fallback_plan: dict[str, Any],
) -> dict[str, Any]:
    fallback_descriptors = dict(fallback_plan["descriptors"])
    allowed_presence_activity = {
        "no presence",
        "still presence",
        "active presence",
        "intermittent movement",
        "presence uncertain",
    }
    allowed_presence_count = {
        "no people",
        "one person",
        "two people",
        "three people",
        "four or more people",
        "presence uncertain",
        "uncertain occupancy",
    }
    allowed_zone_values = {"left", "center", "right", "front", "back"}
    allowed_depth_band = {"front", "mid-room", "back"}
    allowed_layout_mode = {
        "empty",
        "single",
        "split",
        "depth_split",
        "clustered",
        "ambiguous",
        "dispersed",
        "front-heavy",
        "back-heavy",
        "room-wide",
    }

    descriptors = dict(fallback_descriptors)
    descriptors["presence_activity"] = sanitize_choice(
        scene_interpretation.get("presence_activity"),
        allowed_presence_activity,
        str(fallback_descriptors["presence_activity"]),
    )
    descriptors["presence_count"] = sanitize_choice(
        scene_interpretation.get("presence_count"),
        allowed_presence_count,
        str(fallback_descriptors["presence_count"]),
    )
    descriptors["figure_count"] = sanitize_figure_count(
        scene_interpretation.get("figure_count"),
        int(fallback_descriptors["figure_count"]),
    )
    descriptors["primary_zone"] = sanitize_choice(
        scene_interpretation.get("primary_zone"),
        allowed_zone_values,
        str(fallback_descriptors["primary_zone"]),
    )
    descriptors["secondary_zone"] = sanitize_choice(
        scene_interpretation.get("secondary_zone"),
        allowed_zone_values,
        str(fallback_descriptors["secondary_zone"]),
    )
    descriptors["depth_band"] = sanitize_choice(
        scene_interpretation.get("depth_band"),
        allowed_depth_band,
        str(fallback_descriptors["depth_band"]),
    )
    descriptors["layout_mode"] = sanitize_choice(
        scene_interpretation.get("layout_mode"),
        allowed_layout_mode,
        str(fallback_descriptors["layout_mode"]),
    )
    descriptors["active_zones"] = sanitize_active_zones(
        scene_interpretation.get("active_zones"),
        list(fallback_descriptors.get("active_zones") or []),
    )
    descriptors["presence_location"] = sanitize_short_text(
        scene_interpretation.get("presence_location"),
        str(fallback_descriptors["presence_location"]),
        max_length=72,
    )
    descriptors["placement_summary"] = sanitize_short_text(
        scene_interpretation.get("placement_summary"),
        str(fallback_descriptors["placement_summary"]),
        max_length=72,
    )
    descriptors["placement_prompt"] = sanitize_short_text(
        scene_interpretation.get("placement_prompt"),
        str(fallback_descriptors["placement_prompt"]),
        max_length=160,
    )
    descriptors["spatial_certainty"] = sanitize_short_text(
        scene_interpretation.get("spatial_certainty"),
        str(fallback_descriptors["spatial_certainty"]),
        max_length=48,
    )

    if descriptors["figure_count"] <= 0 or descriptors["presence_count"] == "no people":
        descriptors["figure_count"] = 0
        descriptors["presence_count"] = "no people"
        descriptors["layout_mode"] = "empty"
    elif descriptors["presence_count"] in {"presence uncertain", "uncertain occupancy"}:
        if descriptors["figure_count"] <= 0:
            descriptors["layout_mode"] = "ambiguous"
    else:
        inferred_from_label = presence_label_to_figure_count(
            str(descriptors["presence_count"]),
            int(descriptors["figure_count"]),
        )
        if descriptors["figure_count"] <= 0 and inferred_from_label > 0:
            descriptors["figure_count"] = inferred_from_label
        descriptors["presence_count"] = figure_count_to_presence_label(int(descriptors["figure_count"]))

    if descriptors["figure_count"] == 0 and descriptors["presence_activity"] != "presence uncertain":
        descriptors["presence_activity"] = "no presence"

    descriptors["figure_variation_modifiers"] = build_figure_variation_modifiers(descriptors)

    live_lines = sanitize_short_string_list(
        scene_interpretation.get("live_inference_lines"),
        fallback=fallback_plan["live_lines"],
        min_items=MIN_LANGUAGE_PASS_LIVE_LINES,
        max_items=MAX_AI_LIVE_LINES,
        max_length=58,
    )
    agent_notes = sanitize_short_string_list(
        scene_interpretation.get("agent_notes"),
        fallback=[],
        min_items=MIN_LANGUAGE_PASS_AGENT_NOTES,
        max_items=MAX_LANGUAGE_PASS_AGENT_NOTES,
        max_length=40,
    )
    prompt_modifiers = sanitize_prompt_modifiers(
        scene_interpretation.get("prompt_modifiers"),
        fallback=descriptors.get("figure_variation_modifiers", []),
        force_occupied=int(descriptors.get("figure_count", 0)) > 0,
    )
    image_preview = compact_machine_phrase(
        sanitize_text_value(
            scene_interpretation.get("image_preview"),
            build_language_pass_fallback_preview({"descriptors": descriptors}),
        ),
        max_length=68,
    )
    generation_controls = sanitize_generation_controls(scene_interpretation, descriptors)

    return {
        "descriptors": descriptors,
        "live_inference_lines": live_lines,
        "agent_notes": agent_notes,
        "prompt_modifiers": prompt_modifiers,
        "image_preview": image_preview,
        "generation_controls": generation_controls,
    }


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
        compact = re.sub(r"[^a-zA-Z0-9 ,\\/\\-]", "", item).strip(" ,.;:-")
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
    occupancy_evidence = build_llm_occupancy_evidence(
        smoothed,
        fallback_plan,
        empty_room_comparison,
        empty_room_baseline,
    )
    top_changed_metrics = []
    if isinstance(empty_room_comparison, dict):
        changed_metrics = empty_room_comparison.get("top_changed_metrics")
        if isinstance(changed_metrics, list):
            top_changed_metrics = changed_metrics[:MAX_BASELINE_CHANGED_METRICS]

    return {
        "heuristic_interpretation": {
            "figure_count": descriptors["figure_count"],
            "presence_activity": descriptors["presence_activity"],
            "presence_count": descriptors["presence_count"],
            "placement_summary": descriptors["placement_summary"],
            "placement_prompt": descriptors["placement_prompt"],
            "presence_location": descriptors["presence_location"],
            "lighting_condition": descriptors["lighting_condition"],
            "atmospheric_condition": descriptors["atmospheric_condition"],
            "audio_activity": descriptors.get("audio_activity"),
            "spatial_impression": descriptors["spatial_impression"],
            "sen0628_spatial_estimate": descriptors["sen0628_spatial_estimate"],
            "sen0628_figure_side": descriptors["sen0628_figure_side"],
            "dominant_ld_zone": descriptors.get("dominant_ld_zone"),
            "active_ld_zones": descriptors.get("active_ld_zones"),
            "front_presence": descriptors.get("front_presence"),
            "back_presence": descriptors.get("back_presence"),
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
            "empty_room_match": build_empty_room_match_summary(smoothed, empty_room_baseline),
        },
        "occupancy_evidence": occupancy_evidence,
        "task": {
            "goal": (
                "infer occupancy, approximate human count, approximate location, and image-driving scene interpretation "
                "from the smoothed sensor state"
            ),
            "priority_order": [
                "presence_activity",
                "presence_count",
                "figure_count",
                "placement",
                "depth and zone estimate",
                "stable-room prompt language",
            ],
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
            "ld2410c_front": smoothed.get("ld2410c_front"),
            "ld2410c_back": smoothed.get("ld2410c_back"),
            "sen0628": smoothed.get("sen0628"),
            "bme688": smoothed.get("bme688"),
            "usb_microphone": smoothed.get("usb_microphone"),
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
        "You are the primary scene interpreter for an installation-space sensing system. "
        "Your job is to interpret the smoothed sensor data itself, not just rewrite heuristic text. "
        "Use heuristic_interpretation as a fallback suggestion, not as the source of truth. "
        "Occupancy and approximate placement should come from your reasoning over raw_sensor_summary, occupancy_evidence, and baseline_comparison. "
        "Estimate figure_count and presence_count from the live sensor values relative to the empty-room baseline whenever baseline data is available. "
        "Treat the empty-room baseline as the reference state for deciding whether occupancy is present, but do not let a weak baseline match override strong live occupancy evidence. "
        "Use LD2410C target states and energies, front/back agreement, and SEN0628 point distribution as the main occupancy evidence. "
        "Figure_count should be an approximate visible occupancy estimate, not an exact census. "
        "Strong simultaneous front/back radar activity, high moving and stationary energy, multiple SEN0628 occupied zones, and strong shared audio can justify three or more visible figures. "
        "If occupancy_evidence.empty_room_match.matches_empty_room is true, treat that as strong evidence for zero people unless multiple sensors clearly contradict it. "
        "When occupancy_evidence.empty_room_match.matches_empty_room is true, return figure_count=0 and presence_count=no people. "
        "In that case, placement_summary should explicitly say likely empty room or baseline-like empty room, and placement_prompt should keep visible people out of the image. "
        "If occupancy_evidence.empty_room_match.change_ratio is low and the SEN0628 occupied and near-point signals remain near zero, follow the empty-room recording even if one radar channel is noisy. "
        "If occupancy_confidence is high, prefer occupied interpretations even if the empty-room baseline match is weak. "
        "Stay conservative when the evidence is weak or contradictory. "
        "Do not invent extra certainty. "
        "Do not invent room changes, camera changes, furniture, or new architecture. "
        "If baseline departure is minimal or weak, prefer caution, uncertainty, and partial interpretation. "
        "Keep the tone machine-like, restrained, observational, and concise. "
        "Allow fresh wording and small variations in phrasing rather than repeating a fixed template. "
        "Avoid poetic excess, storytelling, surveillance language, and long explanation. "
        "Write in short projector-friendly fragments or clipped sentences, not full prose paragraphs. "
        "Favor wording about tendencies, shifts, ambiguity, continuity, occupancy revision, figure placement, figure scale, and room preservation. "
        "Do not state room contents as hard facts unless certainty is extremely strong. "
        "Keep phrasing projector-friendly and glance-readable. "
        "Return JSON only with these keys: "
        "presence_activity, presence_count, figure_count, presence_location, primary_zone, secondary_zone, "
        "active_zones, depth_band, layout_mode, placement_summary, placement_prompt, spatial_certainty, "
        "live_inference_lines, image_preview, agent_notes, prompt_modifiers, uncertainty_score, generation_controls. "
        "Allowed values: "
        "presence_activity in [no presence, still presence, active presence, intermittent movement, presence uncertain]; "
        "presence_count in [no people, one person, two people, three people, four or more people, presence uncertain, uncertain occupancy]; "
        "figure_count in [0, 1, 2, 3, 4, 5]; "
        "primary_zone and secondary_zone in [left, center, right, front, back]; "
        "active_zones as a short list using those same values; "
        "depth_band in [front, mid-room, back]; "
        "layout_mode in [empty, single, split, depth_split, clustered, ambiguous, dispersed, front-heavy, back-heavy, room-wide]. "
        "Rules: live_inference_lines must be 4 to 5 short lines; "
        "the set of lines should usually include some audio signal and some foreshadowing of the next image state; "
        "not every line needs a rigid function, and the language can vary naturally as long as it stays concise; "
        "each live line should usually be under 58 characters, with most lines landing around 4 to 12 words; "
        "image_preview must be 1 short forward-looking fragment; "
        "agent_notes should be 0 to 1 short technical notes; "
        "prompt_modifiers must be 2 to 4 short visual phrases that can be appended to a prompt; "
        "prompt_modifiers should prioritize visible figure variation such as count, spacing, clustering, side bias, depth, and scale while keeping the room broadly consistent; "
        "keep prompt_modifiers concise rather than prose; "
        "you may invent new wording instead of copying stock phrases; "
        "uncertainty_score must be a number from 0.0 to 1.0 representing how uncertain your interpretation is; "
        "higher uncertainty should usually increase temperature and top_p modestly while reducing guidance_scale modestly; "
        "generation_controls must be an object with numeric temperature, top_p, guidance_scale values; "
        "keep generation_controls restrained and close to the room-consistency goal rather than pushing into surreal abstraction; "
        "when baseline departure and live occupancy evidence strongly support people, do not leave figure_count as uncertain; choose an approximate visible count that can extend above two when justified; "
        "when baseline departure is minimal and occupancy evidence is weak, prefer no people or presence uncertain instead of overstating occupancy; "
        "when the empty-room baseline match is true, do not keep a person in the scene just because heuristic_interpretation guessed one; "
        "when the installation still resembles the empty-room recording and occupancy_confidence stays low, the generated image should remain an empty room; "
        "If the evidence is ambiguous, use presence uncertain and cautious placement phrasing rather than guessing. "
        "do not include markdown fences."
    )
    request_payload = {
        "model": str(openai_settings.get("model", "")),
        "max_output_tokens": 420,
        "text": {"format": {"type": "json_object"}},
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
    scene_plan: dict[str, Any],
    prompt_modifiers: list[str],
) -> str:
    prompt_sections = scene_plan["prompt_sections"]
    figure_count = int(scene_plan["descriptors"].get("figure_count", 0))
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
    return ", ".join(ordered_sections + prompt_modifiers)


def build_scene_plan(
    smoothed: dict[str, Any],
    openai_settings: dict[str, Any] | None,
    empty_room_baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    fallback_plan = build_rule_based_scene_plan(smoothed)
    if not openai_settings or not openai_settings.get("enabled"):
        return fallback_plan

    try:
        scene_interpretation = fetch_openai_language_pass(
            openai_settings, smoothed, fallback_plan, empty_room_baseline
        )
        sanitized_scene = sanitize_scene_interpretation(scene_interpretation, fallback_plan)
        occupancy_confidence = compute_occupancy_confidence(smoothed, empty_room_baseline)
        if (
            baseline_matches_empty_room_recording(smoothed, empty_room_baseline)
            and float(occupancy_confidence.get("occupancy_confidence") or 0.0) < 0.58
        ):
            sanitized_scene["descriptors"] = coerce_descriptors_to_empty_room(
                sanitized_scene["descriptors"],
                fallback_plan,
            )
            sanitized_scene["live_inference_lines"] = build_live_inference_lines(
                sanitized_scene["descriptors"]
            )
            sanitized_scene["prompt_modifiers"] = build_figure_variation_modifiers(
                sanitized_scene["descriptors"]
            )
            sanitized_scene["image_preview"] = build_language_pass_fallback_preview(
                {"descriptors": sanitized_scene["descriptors"]}
            )
        descriptors = sanitized_scene["descriptors"]
        prompt_sections = build_prompt_sections(descriptors)
        result = {
            "descriptors": descriptors,
            "prompt_sections": prompt_sections,
            "prompt": "",
            "live_lines": sanitized_scene["live_inference_lines"],
            "generation_controls": sanitized_scene["generation_controls"],
            "interpretation_source": "openai_scene_interpreter",
            "agent_preview": sanitized_scene["image_preview"],
            "agent_notes": sanitized_scene["agent_notes"],
            "state_signature_descriptors": descriptors,
        }
        result["prompt"] = build_final_prompt_from_language_pass(
            result,
            sanitized_scene["prompt_modifiers"],
        )
        live_lines = result["live_lines"]
        agent_preview = result["agent_preview"]
        if len(live_lines) > MAX_AI_LIVE_LINES:
            result["live_lines"] = live_lines[:MAX_AI_LIVE_LINES]
        result["agent_preview"] = agent_preview
        return result
    except Exception as exc:
        fallback_result = dict(fallback_plan)
        fallback_result["openai_error"] = str(exc)
        return fallback_result


def extract_change_metrics(smoothed):
    bme_data = smoothed.get("bme688") or {}
    ld_data = smoothed.get("ld2410c") or {}
    microphone_data = smoothed.get("usb_microphone") or {}
    detection_distance_cm = ld_data.get("detection_distance_cm")
    if detection_distance_cm is None:
        detection_distance_cm = ld_data.get("moving_distance_cm") or ld_data.get("stationary_distance_cm")

    return {
        "temperature_c": bme_data.get("temperature_c"),
        "humidity_pct": bme_data.get("humidity_pct"),
        "detection_distance_cm": detection_distance_cm,
        "audio_activity_score": microphone_data.get("activity_score"),
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
        "audio_activity": descriptors.get("audio_activity"),
    }
    return json.dumps(signature_fields, sort_keys=True)


def change_exceeds_threshold(previous_metrics, current_metrics):
    if previous_metrics is None:
        return True

    threshold_map = {
        "temperature_c": TEMPERATURE_CHANGE_THRESHOLD,
        "humidity_pct": HUMIDITY_CHANGE_THRESHOLD,
        "detection_distance_cm": DISTANCE_CHANGE_THRESHOLD_CM,
        "audio_activity_score": AUDIO_ACTIVITY_CHANGE_THRESHOLD,
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
    generation_controls,
    interpretation_source,
    agent_preview,
    agent_notes,
    openai_error,
    coordinator,
    now,
    generated_image_path,
) -> dict[str, Any]:
    previous_shared_state = coordinator.latest_shared_state or {}
    seed_cycle = build_seed_cycle_state(coordinator, previous_shared_state, now)
    return {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "timing": {
            "text_update_interval_seconds": TEXT_UPDATE_INTERVAL_SECONDS,
            "image_generation_interval_seconds": IMAGE_GENERATION_INTERVAL_SECONDS,
            "force_image_refresh_seconds": FORCE_IMAGE_REFRESH_SECONDS,
            "state_stable_hold_seconds": STATE_STABLE_HOLD_SECONDS,
            "seed_cycle_interval_seconds": IMAGE_SEED_CYCLE_INTERVAL_SECONDS,
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
        "generation_controls": generation_controls,
        "seed_cycle": seed_cycle,
        "seed_cycle_started_at": seed_cycle["started_at"],
        "state_signature": coordinator.current_signature,
        "stable_signature": coordinator.stable_signature,
        "last_meaningful_change_time": coordinator.last_meaningful_change_time,
        "last_text_update_time": coordinator.last_text_update_time,
        "last_image_generation_time": coordinator.last_image_generation_time,
        "current_image_path": resolve_image_path_value(generated_image_path),
        "current_image_seed": previous_shared_state.get(
            "current_image_seed", seed_cycle["current_seed"]
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
    generation_controls,
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
        "generation_controls": generation_controls,
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
    uncertainty_score: float | None = None,
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
    if isinstance(uncertainty_score, (int, float)):
        lines.append(f"uncertainty_score: {float(uncertainty_score):.3f}")
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
    generation_controls,
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
        uncertainty_score=(
            float(descriptors["uncertainty_score"])
            if isinstance(descriptors.get("uncertainty_score"), (int, float))
            else None
        ),
    )
    debug_image.save(image_path)

    metadata = build_generation_metadata(
        raw_frames=raw_frames,
        smoothed=smoothed,
        descriptors=descriptors,
        prompt=prompt,
        prompt_sections=prompt_sections,
        generation_controls=generation_controls,
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
            "generation_controls:",
            json_text(generation_controls),
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


def generate_image(
    client,
    prompt: str,
    negative_prompt: str,
    seed: int,
    generation_controls: dict[str, float | int],
):
    # Diffusion image generation is driven primarily by seed, prompt, guidance scale,
    # and denoising steps. Temperature/top_p are not reliable structural controls here.
    request_kwargs = {
        "negative_prompt": negative_prompt,
        "model": MODEL_ID,
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT,
        "guidance_scale": generation_controls["guidance_scale"],
        "num_inference_steps": generation_controls["num_inference_steps"],
        "seed": seed,
    }
    return client.text_to_image(prompt, **request_kwargs)


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


def wait_for_initial_frames(ser, frame_buffer, microphone_monitor=None):
    deadline = time.time() + SERIAL_STARTUP_SECONDS

    while time.time() < deadline and len(frame_buffer) < MIN_FRAMES_FOR_PROCESSING:
        raw_line = ser.readline().decode("utf-8", errors="ignore").strip()
        if not raw_line:
            continue

        print(f"[PICO] {raw_line}")
        frame = parse_sensor_line(raw_line)
        if frame is not None:
            frame_buffer.append(attach_microphone_snapshot(frame, microphone_monitor))

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
    active_seed = None
    prerequisites_met = False
    force_refresh_enabled = not has_active_seed_cycle(coordinator.latest_shared_state)
    seed_cycle_interval_seconds = get_seed_cycle_interval_seconds(coordinator.latest_shared_state)
    seed_cycle_elapsed = (
        seed_cycle_interval_seconds is not None
        and seconds_since_last_image_generation >= seed_cycle_interval_seconds
    )

    if coordinator.latest_shared_state is None:
        reason = "no shared state yet"
    elif coordinator.image_generation_in_progress:
        reason = "generation already in progress"
    elif seconds_since_last_image_generation < IMAGE_GENERATION_INTERVAL_SECONDS:
        reason = "image interval not reached"
    elif seed_cycle_elapsed:
        prerequisites_met = True
        reason = "seed cycle interval elapsed"
    elif seconds_since_meaningful_change < STATE_STABLE_HOLD_SECONDS:
        reason = "state stability hold not reached"
    else:
        prerequisites_met = True
        seed_cycle = coordinator.latest_shared_state.get("seed_cycle")
        if isinstance(seed_cycle, dict) and isinstance(seed_cycle.get("current_seed"), int):
            active_seed = seed_cycle["current_seed"]

    if prerequisites_met and active_seed is None:
        seed_cycle = coordinator.latest_shared_state.get("seed_cycle")
        if isinstance(seed_cycle, dict) and isinstance(seed_cycle.get("current_seed"), int):
            active_seed = seed_cycle["current_seed"]

    if prerequisites_met and seed_cycle_elapsed:
        decision = True
        reason = "seed cycle interval elapsed"

    if prerequisites_met and active_seed is not None:
        current_image_seed = coordinator.latest_shared_state.get("current_image_seed")
        if current_image_seed != active_seed:
            decision = True
            reason = "seed cycle advanced"
    if prerequisites_met and not decision and coordinator.stable_signature != coordinator.last_image_signature:
        decision = True
        reason = "state signature changed"
    elif (
        prerequisites_met
        and force_refresh_enabled
        and not decision
        and seconds_since_last_image_generation >= FORCE_IMAGE_REFRESH_SECONDS
    ):
        decision = True
        reason = "scheduled timed refresh"
    elif prerequisites_met and not decision and not force_refresh_enabled:
        reason = "waiting for state change or next seed in cycle"

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
        f"active_seed={active_seed} "
        f"force_refresh_enabled={force_refresh_enabled} "
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
    generation_controls = shared_state.get("generation_controls") or build_generation_controls_from_uncertainty(
        derive_uncertainty_score_from_descriptors(shared_state["interpreted_state"])
    )
    negative_prompt = prompt_sections.get("negative_prompt", OPTIONAL_TEXT_UI_EXCLUSION_NEGATIVE_PROMPT)
    smoothed = shared_state["smoothed_sensor_values"]
    descriptors = shared_state["interpreted_state"]

    print(f"[INFER] {json.dumps(descriptors)}")
    print(f"[PROMPT] {prompt}")
    print(f"[NEGATIVE_PROMPT] {negative_prompt}")
    print(f"[GENERATION_CONTROLS] {json.dumps(generation_controls)}")
    print(f"[SEED] {seed}")

    image = generate_image(client, prompt, negative_prompt, seed, generation_controls)
    image_path, log_path, metadata_path = save_generation_artifacts(
        image=image,
        raw_frames=raw_frames,
        smoothed=smoothed,
        descriptors=descriptors,
        prompt=prompt,
        prompt_sections=prompt_sections,
        generation_controls=generation_controls,
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
    selected_seed = select_generation_seed(shared_state_snapshot)

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
    previous_shared_state = coordinator.latest_shared_state
    scene_plan = build_scene_plan(smoothed, openai_settings, empty_room_baseline)
    scene_plan = apply_count_hysteresis(
        scene_plan,
        smoothed,
        previous_shared_state,
        empty_room_baseline,
    )
    descriptors = scene_plan["descriptors"]
    live_lines = scene_plan["live_lines"]
    prompt_sections = scene_plan["prompt_sections"]
    prompt = scene_plan["prompt"]
    signature = build_state_signature(scene_plan.get("state_signature_descriptors", descriptors))
    change_metrics = extract_change_metrics(smoothed)

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
        generation_controls=scene_plan.get("generation_controls"),
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
    print(f"Seed cycle values: {get_seed_cycle_values()}")
    print(f"Seed cycle interval: {IMAGE_SEED_CYCLE_INTERVAL_SECONDS} seconds")
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
    microphone_monitor = build_microphone_monitor_from_env()
    microphone_enabled = microphone_monitor.start()
    print_runtime_configuration(openai_settings)
    if microphone_enabled:
        print(f"USB microphone: {microphone_monitor.status_text}")
    else:
        print(f"USB microphone: {microphone_monitor.status_text}")

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

            if not wait_for_initial_frames(ser, frame_buffer, microphone_monitor):
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
                        frame_buffer.append(attach_microphone_snapshot(frame, microphone_monitor))

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
        microphone_monitor.stop()
        cleanup_lock_file()


if __name__ == "__main__":
    main()
