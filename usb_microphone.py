import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any


try:
    import numpy as np
except ImportError:  # pragma: no cover - handled at runtime
    np = None

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - handled at runtime
    sd = None


DEFAULT_DEVICE_NAME_HINTS = (
    "quadcast",
    "hyperx",
    "microphone",
)
DEFAULT_WINDOW_SECONDS = 4.0
DEFAULT_BLOCKSIZE = 2048
DEFAULT_NOISE_FLOOR_WINDOW_SECONDS = 20.0
NOISE_FLOOR_ACTIVITY_MULTIPLIER = 3.0
NOISE_FLOOR_ACTIVITY_OFFSET = 0.002
MIN_NOISE_FLOOR_RMS = 1e-6


@dataclass
class MicrophoneSnapshot:
    available: bool
    device_name: str | None
    device_index: int | None
    sample_rate_hz: float | None
    rms: float | None
    peak: float | None
    noise_floor_rms: float | None
    relative_rms: float | None
    relative_db: float | None
    activity_score: float | None
    active_fraction: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "device_name": self.device_name,
            "device_index": self.device_index,
            "sample_rate_hz": self.sample_rate_hz,
            "rms": self.rms,
            "peak": self.peak,
            "noise_floor_rms": self.noise_floor_rms,
            "relative_rms": self.relative_rms,
            "relative_db": self.relative_db,
            "activity_score": self.activity_score,
            "active_fraction": self.active_fraction,
        }


def _clean_device_name(device: dict[str, Any]) -> str:
    name = device.get("name")
    if not isinstance(name, str):
        return "Unknown input"
    return " ".join(name.split())


def list_input_devices() -> list[dict[str, Any]]:
    if sd is None:
        return []

    devices: list[dict[str, Any]] = []
    for index, raw_device in enumerate(sd.query_devices()):
        if int(raw_device.get("max_input_channels", 0)) <= 0:
            continue
        devices.append(
            {
                "index": index,
                "name": _clean_device_name(raw_device),
                "max_input_channels": int(raw_device.get("max_input_channels", 0)),
                "default_samplerate": float(raw_device.get("default_samplerate", 0.0) or 0.0),
            }
        )
    return devices


def _select_device(
    preferred_index: int | None,
    preferred_name: str | None,
) -> dict[str, Any] | None:
    devices = list_input_devices()
    if not devices:
        return None

    if preferred_index is not None:
        for device in devices:
            if int(device["index"]) == preferred_index:
                return device

    if preferred_name:
        preferred = preferred_name.lower()
        for device in devices:
            if preferred in str(device["name"]).lower():
                return device

    for hint in DEFAULT_DEVICE_NAME_HINTS:
        for device in devices:
            if hint in str(device["name"]).lower():
                return device

    return devices[0]


class AudioLevelMonitor:
    def __init__(
        self,
        *,
        preferred_device_index: int | None = None,
        preferred_device_name: str | None = None,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        blocksize: int = DEFAULT_BLOCKSIZE,
    ) -> None:
        self.window_seconds = max(1.0, float(window_seconds))
        self.blocksize = max(256, int(blocksize))
        self.preferred_device_index = preferred_device_index
        self.preferred_device_name = preferred_device_name

        self._lock = threading.Lock()
        self._history: deque[dict[str, float]] = deque()
        self._quiet_history: deque[dict[str, float]] = deque()
        self._stream = None
        self._device_info: dict[str, Any] | None = None
        self._start_error: str | None = None
        self._noise_floor_rms: float | None = None

    def start(self) -> bool:
        if sd is None or np is None:
            missing = []
            if sd is None:
                missing.append("sounddevice")
            if np is None:
                missing.append("numpy")
            self._start_error = f"missing dependency: {', '.join(missing)}"
            return False

        if self._stream is not None:
            return True

        self._device_info = _select_device(
            self.preferred_device_index,
            self.preferred_device_name,
        )
        if self._device_info is None:
            self._start_error = "no input-capable audio devices found"
            return False

        sample_rate = float(self._device_info.get("default_samplerate") or 48_000.0)
        device_index = int(self._device_info["index"])
        channels = max(1, min(2, int(self._device_info.get("max_input_channels", 1))))

        def callback(indata, frames, callback_time, status) -> None:
            if status:
                self._start_error = str(status)
            if frames <= 0:
                return

            samples = np.asarray(indata, dtype=np.float32)
            if samples.ndim == 2:
                samples = samples.mean(axis=1)
            elif samples.ndim != 1:
                samples = samples.reshape(-1)

            if samples.size == 0:
                return

            rms = float(np.sqrt(np.mean(np.square(samples))))
            peak = float(np.max(np.abs(samples)))
            timestamp = time.time()

            with self._lock:
                self._history.append({"timestamp": timestamp, "rms": rms, "peak": peak})
                self._trim_history_locked(timestamp)
                self._update_noise_floor_locked(timestamp, rms)

        try:
            self._stream = sd.InputStream(
                device=device_index,
                channels=channels,
                samplerate=sample_rate,
                blocksize=self.blocksize,
                dtype="float32",
                callback=callback,
            )
            self._stream.start()
            self._start_error = None
            return True
        except Exception as exc:  # pragma: no cover - hardware specific
            self._stream = None
            self._start_error = str(exc)
            return False

    def stop(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass

    def _trim_history_locked(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._history and self._history[0]["timestamp"] < cutoff:
            self._history.popleft()

        quiet_cutoff = now - max(self.window_seconds, DEFAULT_NOISE_FLOOR_WINDOW_SECONDS)
        while self._quiet_history and self._quiet_history[0]["timestamp"] < quiet_cutoff:
            self._quiet_history.popleft()

    def _update_noise_floor_locked(self, now: float, rms: float) -> None:
        current_floor = self._noise_floor_rms
        qualifies_as_quiet = current_floor is None or rms <= max(
            current_floor * NOISE_FLOOR_ACTIVITY_MULTIPLIER,
            current_floor + NOISE_FLOOR_ACTIVITY_OFFSET,
        )
        if qualifies_as_quiet:
            self._quiet_history.append({"timestamp": now, "rms": rms})

        quiet_rms_values = [item["rms"] for item in self._quiet_history]
        if quiet_rms_values:
            percentile_index = max(0, int(len(quiet_rms_values) * 0.2) - 1)
            candidate_floor = float(sorted(quiet_rms_values)[percentile_index])
        else:
            candidate_floor = rms

        self._noise_floor_rms = max(candidate_floor, MIN_NOISE_FLOOR_RMS)

    def snapshot(self) -> dict[str, Any]:
        device_name = None
        device_index = None
        sample_rate_hz = None
        if self._device_info is not None:
            device_name = str(self._device_info.get("name"))
            device_index = int(self._device_info.get("index"))
            sample_rate_hz = float(self._device_info.get("default_samplerate") or 0.0) or None

        with self._lock:
            now = time.time()
            self._trim_history_locked(now)
            history = list(self._history)

        if not history:
            return MicrophoneSnapshot(
                available=self._stream is not None,
                device_name=device_name,
                device_index=device_index,
                sample_rate_hz=sample_rate_hz,
                rms=None,
                peak=None,
                noise_floor_rms=None,
                relative_rms=None,
                relative_db=None,
                activity_score=None,
                active_fraction=None,
            ).to_dict()

        rms_values = [item["rms"] for item in history]
        peak_values = [item["peak"] for item in history]
        rms = float(sum(rms_values) / len(rms_values))
        peak = float(max(peak_values))
        noise_floor_rms = float(self._noise_floor_rms or min(rms_values))
        relative_rms = max(0.0, rms - noise_floor_rms)
        relative_db = 20.0 * np.log10((rms + 1e-9) / (noise_floor_rms + 1e-9))
        active_threshold = max(
            noise_floor_rms * NOISE_FLOOR_ACTIVITY_MULTIPLIER,
            noise_floor_rms + NOISE_FLOOR_ACTIVITY_OFFSET,
        )
        active_fraction = sum(1 for value in rms_values if value > active_threshold) / len(rms_values)
        activity_score = max(active_fraction * relative_db / 10.0, 0.0)

        return MicrophoneSnapshot(
            available=self._stream is not None,
            device_name=device_name,
            device_index=device_index,
            sample_rate_hz=sample_rate_hz,
            rms=round(rms, 6),
            peak=round(peak, 6),
            noise_floor_rms=round(noise_floor_rms, 6),
            relative_rms=round(relative_rms, 6),
            relative_db=round(float(relative_db), 3),
            activity_score=round(activity_score, 3),
            active_fraction=round(active_fraction, 3),
        ).to_dict()

    @property
    def status_text(self) -> str:
        if self._stream is not None and self._device_info is not None:
            return (
                f"enabled ({self._device_info['name']} "
                f"index={self._device_info['index']})"
            )
        if self._start_error:
            return f"disabled ({self._start_error})"
        return "disabled"


def build_microphone_monitor_from_env() -> AudioLevelMonitor:
    device_index = None
    raw_index = os.getenv("AUDIO_INPUT_DEVICE_INDEX")
    if raw_index:
        try:
            device_index = int(raw_index)
        except ValueError:
            device_index = None

    device_name = os.getenv("AUDIO_INPUT_DEVICE_NAME")
    window_seconds = float(os.getenv("AUDIO_WINDOW_SECONDS", str(DEFAULT_WINDOW_SECONDS)))
    blocksize = int(os.getenv("AUDIO_BLOCKSIZE", str(DEFAULT_BLOCKSIZE)))
    return AudioLevelMonitor(
        preferred_device_index=device_index,
        preferred_device_name=device_name,
        window_seconds=window_seconds,
        blocksize=blocksize,
    )


def attach_microphone_snapshot(
    frame: dict[str, Any],
    microphone_monitor: AudioLevelMonitor | None,
) -> dict[str, Any]:
    augmented = dict(frame)
    if microphone_monitor is None:
        return augmented
    augmented["usb_microphone"] = microphone_monitor.snapshot()
    return augmented
