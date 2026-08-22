"""Fixed-shape agent contract for bridge-delivered auditory events."""

from __future__ import annotations

from numbers import Real
from typing import Any

import numpy as np
from gymnasium import spaces


AUDIO_SCHEMA = "hytalerl_audio_frame_v1"
AUDIO_VERSION = 1
AUDIO_MAX_EVENTS = 16

AUDIO_KIND_2D = 0
AUDIO_KIND_3D = 1
AUDIO_KIND_ENTITY = 2
AUDIO_KIND_COUNT = 3
AUDIO_CATEGORY_NONE = -1
AUDIO_EVENT_WIRE_WIDTH = 10


def audio_space() -> spaces.Dict:
    """Return the bounded policy-facing auditory observation space."""

    return spaces.Dict(
        {
            "available": spaces.Discrete(2),
            "capture_mask": spaces.MultiBinary(AUDIO_KIND_COUNT),
            "valid": spaces.MultiBinary(AUDIO_MAX_EVENTS),
            "kind": spaces.Box(
                AUDIO_KIND_2D,
                AUDIO_KIND_ENTITY,
                shape=(AUDIO_MAX_EVENTS,),
                dtype=np.int8,
            ),
            "sound_event_index": spaces.Box(
                0,
                np.iinfo(np.int32).max,
                shape=(AUDIO_MAX_EVENTS,),
                dtype=np.int32,
            ),
            "category": spaces.Box(
                AUDIO_CATEGORY_NONE,
                3,
                shape=(AUDIO_MAX_EVENTS,),
                dtype=np.int8,
            ),
            "spatial_valid": spaces.MultiBinary(AUDIO_MAX_EVENTS),
            "relative_position": spaces.Box(
                -np.inf,
                np.inf,
                shape=(AUDIO_MAX_EVENTS, 3),
                dtype=np.float32,
            ),
            "volume_modifier": spaces.Box(
                -np.inf,
                np.inf,
                shape=(AUDIO_MAX_EVENTS,),
                dtype=np.float32,
            ),
            "pitch_modifier": spaces.Box(
                -np.inf,
                np.inf,
                shape=(AUDIO_MAX_EVENTS,),
                dtype=np.float32,
            ),
            "age_seconds": spaces.Box(
                0.0,
                np.inf,
                shape=(AUDIO_MAX_EVENTS,),
                dtype=np.float32,
            ),
        }
    )


def empty_audio() -> dict[str, Any]:
    """Return an explicitly unavailable, correctly typed audio observation."""

    return {
        "available": 0,
        "capture_mask": np.zeros(AUDIO_KIND_COUNT, dtype=np.int8),
        "valid": np.zeros(AUDIO_MAX_EVENTS, dtype=np.int8),
        "kind": np.zeros(AUDIO_MAX_EVENTS, dtype=np.int8),
        "sound_event_index": np.zeros(AUDIO_MAX_EVENTS, dtype=np.int32),
        "category": np.full(
            AUDIO_MAX_EVENTS,
            AUDIO_CATEGORY_NONE,
            dtype=np.int8,
        ),
        "spatial_valid": np.zeros(AUDIO_MAX_EVENTS, dtype=np.int8),
        "relative_position": np.zeros(
            (AUDIO_MAX_EVENTS, 3),
            dtype=np.float32,
        ),
        "volume_modifier": np.zeros(AUDIO_MAX_EVENTS, dtype=np.float32),
        "pitch_modifier": np.zeros(AUDIO_MAX_EVENTS, dtype=np.float32),
        "age_seconds": np.zeros(AUDIO_MAX_EVENTS, dtype=np.float32),
    }


def parse_audio(
    raw: Any,
    *,
    step_duration_seconds: Any | None = None,
) -> dict[str, Any]:
    """Parse sparse wire events into a padding-first fixed-capacity ring.

    Missing legacy data is unavailable. A present frame is strict: malformed
    values raise instead of silently turning an unsupported capture path into
    apparent silence.
    """

    result = empty_audio()
    if raw is None:
        return result
    if not isinstance(raw, dict):
        raise ValueError("audio payload must be an object or null")
    if raw.get("schema") != AUDIO_SCHEMA or raw.get("version") != AUDIO_VERSION:
        raise ValueError("unsupported or missing audio schema/version")

    available = _boolean(raw.get("available"), "audio.available")
    capture_raw = raw.get("capture_mask")
    if (
        not isinstance(capture_raw, (list, tuple))
        or len(capture_raw) != AUDIO_KIND_COUNT
    ):
        raise ValueError("audio.capture_mask must contain three booleans")
    capture = np.asarray(
        [
            int(_boolean(value, f"audio.capture_mask[{index}]"))
            for index, value in enumerate(capture_raw)
        ],
        dtype=np.int8,
    )
    events = raw.get("events")
    if not isinstance(events, (list, tuple)):
        raise ValueError("audio.events must be an array")
    if len(events) > AUDIO_MAX_EVENTS:
        raise ValueError(
            f"audio has {len(events)} events; capacity is {AUDIO_MAX_EVENTS}"
        )
    if not available:
        if np.any(capture) or events:
            raise ValueError(
                "unavailable audio cannot advertise capture or contain events"
            )
        return result
    if not np.any(capture):
        raise ValueError("available audio must advertise at least one captured kind")
    duration = _nonnegative_finite(
        step_duration_seconds,
        "audio step duration",
    )

    result["available"] = 1
    result["capture_mask"] = capture
    offset = AUDIO_MAX_EVENTS - len(events)
    previous_age = np.inf
    for event_index, event in enumerate(events):
        if not isinstance(event, (list, tuple)) or len(event) != AUDIO_EVENT_WIRE_WIDTH:
            raise ValueError(
                f"audio event must use the {AUDIO_EVENT_WIRE_WIDTH}-field layout"
            )
        kind = _integer(event[0], "audio event kind")
        sound_index = _integer(event[1], "audio sound-event index")
        category = _integer(event[2], "audio category")
        spatial = _boolean(event[3], "audio spatial flag")
        numeric = np.asarray(
            [
                _finite_number(value, f"audio event field {field_index}")
                for field_index, value in enumerate(event[4:], start=4)
            ],
            dtype=np.float64,
        )
        rel_x, rel_y, rel_z, volume, pitch, age = numeric.tolist()

        if kind < AUDIO_KIND_2D or kind > AUDIO_KIND_ENTITY:
            raise ValueError("unknown audio event kind")
        if not capture[kind]:
            raise ValueError("audio event kind is outside the capture mask")
        if sound_index <= 0 or sound_index > np.iinfo(np.int32).max:
            raise ValueError("audio sound-event index must be positive int32")
        category_valid = (
            category == AUDIO_CATEGORY_NONE
            if kind == AUDIO_KIND_ENTITY
            else 0 <= category <= 3
        )
        if not category_valid:
            raise ValueError("audio category is not canonical for its packet kind")
        if kind == AUDIO_KIND_3D:
            if not spatial:
                raise ValueError("3-D audio event requires a spatial vector")
        elif spatial or any(value != 0.0 for value in (rel_x, rel_y, rel_z)):
            raise ValueError("non-spatial audio event must use a zero relative vector")
        if age < 0.0 or age > duration or age > previous_age:
            raise ValueError(
                "audio event ages must be within the step and oldest-to-newest"
            )
        previous_age = age

        target = offset + event_index
        result["valid"][target] = 1
        result["kind"][target] = kind
        result["sound_event_index"][target] = sound_index
        result["category"][target] = category
        result["spatial_valid"][target] = int(spatial)
        result["relative_position"][target] = [rel_x, rel_y, rel_z]
        result["volume_modifier"][target] = volume
        result["pitch_modifier"][target] = pitch
        result["age_seconds"][target] = age
    return result


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _nonnegative_finite(value: Any, label: str) -> float:
    if value is None:
        raise ValueError(f"{label} is required for available audio")
    result = _finite_number(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


__all__ = [
    "AUDIO_CATEGORY_NONE",
    "AUDIO_EVENT_WIRE_WIDTH",
    "AUDIO_KIND_2D",
    "AUDIO_KIND_3D",
    "AUDIO_KIND_COUNT",
    "AUDIO_KIND_ENTITY",
    "AUDIO_MAX_EVENTS",
    "AUDIO_SCHEMA",
    "AUDIO_VERSION",
    "audio_space",
    "empty_audio",
    "parse_audio",
]
