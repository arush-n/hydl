"""Validated native BlockChunk perception-channel captures."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Protocol

import numpy as np

from hytalegym.worldgen.region import MIN_Y, WORLD_HEIGHT

NATIVE_PERCEPTION_CHANNEL_SCHEMA = "hytalerl_native_perception_channels_v1"
NATIVE_PERCEPTION_CHANNEL_VERSION = 1
NATIVE_PERCEPTION_CHANNEL_MAX_SAMPLES = 4096

CHANNEL_HEIGHTMAP = 0
CHANNEL_SKY_LIGHT = 1
CHANNEL_BLOCK_LIGHT_RGB = 2
CHANNEL_ENVIRONMENT = 3
CHANNEL_TINT_RGB = 4
PERCEPTION_CHANNEL_COUNT = 5

VALID_HEIGHTMAP = 1
VALID_SKY_LIGHT = 1 << 1
VALID_BLOCK_LIGHT_RGB = 1 << 2
VALID_ENVIRONMENT = 1 << 3
VALID_TINT_RGB = 1 << 4
VALID_ALL = (1 << PERCEPTION_CHANNEL_COUNT) - 1
_VALIDITY_BITS = np.asarray(
    [
        VALID_HEIGHTMAP,
        VALID_SKY_LIGHT,
        VALID_BLOCK_LIGHT_RGB,
        VALID_ENVIRONMENT,
        VALID_TINT_RGB,
    ],
    dtype=np.uint8,
)
_NPZ_FIELDS = frozenset(
    {
        "metadata_json_u8",
        "positions",
        "available",
        "channel_valid",
        "heightmap_block_y",
        "sky_light",
        "block_light_rgb",
        "environment_code",
        "tint_argb",
    }
)


class NativePerceptionChannelTransport(Protocol):
    """One bounded bridge verb over an already-reset native environment."""

    def capture_perception_channels(
        self,
        positions: np.ndarray,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class NativePerceptionChannelCapture:
    """Portable fixed samples with native provenance and validity masks."""

    server_version: str
    world_name: str
    worldgen_provider: str
    worldgen_version: str
    seed: int
    positions: np.ndarray
    available: np.ndarray
    channel_valid: np.ndarray
    heightmap_block_y: np.ndarray
    sky_light: np.ndarray
    block_light_rgb: np.ndarray
    environment_code: np.ndarray
    tint_argb: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "server_version",
            "world_name",
            "worldgen_provider",
            "worldgen_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")

        arrays = {
            "positions": _array(self.positions, np.int32),
            "available": _array(self.available, np.bool_),
            "channel_valid": _array(self.channel_valid, np.bool_),
            "heightmap_block_y": _array(self.heightmap_block_y, np.int16),
            "sky_light": _array(self.sky_light, np.uint8),
            "block_light_rgb": _array(self.block_light_rgb, np.uint8),
            "environment_code": _array(self.environment_code, np.int32),
            "tint_argb": _array(self.tint_argb, np.uint32),
        }
        samples = arrays["positions"].shape[0]
        if arrays["positions"].shape != (samples, 3):
            raise ValueError("positions must have shape [N, 3]")
        if samples < 1 or samples > NATIVE_PERCEPTION_CHANNEL_MAX_SAMPLES:
            raise ValueError("native channel sample capacity exceeded")
        expected_shapes = {
            "available": (samples,),
            "channel_valid": (samples, PERCEPTION_CHANNEL_COUNT),
            "heightmap_block_y": (samples,),
            "sky_light": (samples,),
            "block_light_rgb": (samples, 3),
            "environment_code": (samples,),
            "tint_argb": (samples,),
        }
        for name, shape in expected_shapes.items():
            if arrays[name].shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        _validate_capture_arrays(**arrays)
        for name, value in arrays.items():
            value.flags.writeable = False
            object.__setattr__(self, name, value)

    @property
    def sample_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def tint_rgb(self) -> np.ndarray:
        value = self.tint_argb
        result = np.stack(
            (
                (value >> np.uint32(16)) & np.uint32(0xFF),
                (value >> np.uint32(8)) & np.uint32(0xFF),
                value & np.uint32(0xFF),
            ),
            axis=1,
        ).astype(np.uint8)
        result.flags.writeable = False
        return result

    @property
    def surrogate_sky_fallback(self) -> np.ndarray:
        height = self.heightmap_block_y.astype(np.int32)
        inside_y = (self.positions[:, 1] >= MIN_Y) & (
            self.positions[:, 1] < MIN_Y + WORLD_HEIGHT
        )
        value = np.where(
            self.channel_valid[:, CHANNEL_HEIGHTMAP]
            & inside_y
            & (self.positions[:, 1] >= height + 1),
            np.uint8(15),
            np.uint8(0),
        )
        value.flags.writeable = False
        return value

    def semantic_digest(self) -> str:
        digest = hashlib.sha256()
        semantic_metadata = self.metadata()
        semantic_metadata.pop("world")
        metadata = json.dumps(
            semantic_metadata,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(metadata)
        for name in sorted(_NPZ_FIELDS - {"metadata_json_u8"}):
            value = np.ascontiguousarray(getattr(self, name))
            digest.update(name.encode())
            digest.update(value.dtype.str.encode())
            digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
            digest.update(value.tobytes())
        return digest.hexdigest()

    def metadata(self) -> dict[str, object]:
        return {
            "schema": NATIVE_PERCEPTION_CHANNEL_SCHEMA,
            "version": NATIVE_PERCEPTION_CHANNEL_VERSION,
            "server_version": self.server_version,
            "world": self.world_name,
            "worldgen_provider": self.worldgen_provider,
            "worldgen_version": self.worldgen_version,
            "seed": self.seed,
            "sample_count": self.sample_count,
            "position_encoding": "absolute_block_i32_xyz",
            "tint_encoding": "argb8888_u32_rgb_low_24_bits",
            "published_provenance": "native",
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = np.frombuffer(
            json.dumps(
                self.metadata(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
            dtype=np.uint8,
        )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with temporary.open("wb") as output:
                np.savez_compressed(
                    output,
                    metadata_json_u8=metadata,
                    positions=self.positions,
                    available=self.available,
                    channel_valid=self.channel_valid,
                    heightmap_block_y=self.heightmap_block_y,
                    sky_light=self.sky_light,
                    block_light_rgb=self.block_light_rgb,
                    environment_code=self.environment_code,
                    tint_argb=self.tint_argb,
                )
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> NativePerceptionChannelCapture:
        with np.load(Path(path), allow_pickle=False) as archive:
            if set(archive.files) != _NPZ_FIELDS:
                raise ValueError("native perception fixture fields differ")
            metadata = json.loads(
                np.asarray(
                    archive["metadata_json_u8"],
                    dtype=np.uint8,
                ).tobytes()
            )
            _require_metadata(metadata)
            result = cls(
                server_version=metadata["server_version"],
                world_name=metadata["world"],
                worldgen_provider=metadata["worldgen_provider"],
                worldgen_version=metadata["worldgen_version"],
                seed=metadata["seed"],
                positions=archive["positions"],
                available=archive["available"],
                channel_valid=archive["channel_valid"],
                heightmap_block_y=archive["heightmap_block_y"],
                sky_light=archive["sky_light"],
                block_light_rgb=archive["block_light_rgb"],
                environment_code=archive["environment_code"],
                tint_argb=archive["tint_argb"],
            )
        if result.sample_count != metadata["sample_count"]:
            raise ValueError("native perception fixture sample count differs")
        return result


def native_perception_channel_request(
    positions: np.ndarray,
) -> dict[str, object]:
    """Encode one bounded little-endian bridge request."""

    values = np.asarray(positions)
    if values.dtype != np.dtype(np.int32):
        raise TypeError("positions must use int32")
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or values.shape[0] < 1
        or values.shape[0] > NATIVE_PERCEPTION_CHANNEL_MAX_SAMPLES
    ):
        raise ValueError("positions must have shape [1..4096, 3]")
    return {
        "type": "perception_channels",
        "positions_i32_le_xyz": np.asarray(values, dtype="<i4").tobytes(),
    }


def capture_native_perception_channels(
    transport: NativePerceptionChannelTransport,
    positions: np.ndarray,
) -> NativePerceptionChannelCapture:
    """Capture and validate one exact native sample set."""

    requested = np.asarray(positions)
    native_perception_channel_request(requested)
    result = native_perception_channel_capture_from_wire(
        transport.capture_perception_channels(requested)
    )
    if not np.array_equal(result.positions, requested):
        raise ValueError("native bridge returned different sample positions")
    return result


def native_perception_channel_capture_from_wire(
    response: Mapping[str, Any],
) -> NativePerceptionChannelCapture:
    """Decode the fixed binary response and enforce native validity semantics."""

    source = dict(response)
    if source.get("type") != "perception_channels":
        raise ValueError("native response is not perception channels")
    if source.get("schema") != NATIVE_PERCEPTION_CHANNEL_SCHEMA:
        raise ValueError("unsupported native perception channel schema")
    if _wire_int(source.get("version"), "version") != (
        NATIVE_PERCEPTION_CHANNEL_VERSION
    ):
        raise ValueError("unsupported native perception channel version")
    samples = _wire_int(source.get("sample_count"), "sample_count")
    if samples < 1 or samples > NATIVE_PERCEPTION_CHANNEL_MAX_SAMPLES:
        raise ValueError("native channel sample capacity exceeded")

    positions = _binary_array(
        source,
        "positions_i32_le_xyz",
        "<i4",
        samples * 3,
    ).reshape(samples, 3)
    available_u8 = _binary_array(source, "available_u8", "u1", samples)
    if np.any(available_u8 > 1):
        raise ValueError("native availability must be uint8 0 or 1")
    validity = _binary_array(
        source,
        "channel_validity_u8_bits",
        "u1",
        samples,
    )
    if np.any(validity & np.uint8(~VALID_ALL & 0xFF)):
        raise ValueError("native channel validity contains unknown bits")
    channel_valid = (validity[:, None] & _VALIDITY_BITS[None, :]) != 0

    tint_argb = _binary_array(
        source,
        "tint_argb_i32_le",
        "<u4",
        samples,
    )
    return NativePerceptionChannelCapture(
        server_version=_wire_text(source.get("server_version"), "server_version"),
        world_name=_wire_text(source.get("world"), "world"),
        worldgen_provider=_wire_text(
            source.get("worldgen_provider"),
            "worldgen_provider",
        ),
        worldgen_version=_wire_text(
            source.get("worldgen_version"),
            "worldgen_version",
        ),
        seed=_wire_int(source.get("seed"), "seed"),
        positions=positions,
        available=available_u8.astype(np.bool_),
        channel_valid=channel_valid,
        heightmap_block_y=_binary_array(
            source,
            "heightmap_i16_le",
            "<i2",
            samples,
        ),
        sky_light=_binary_array(source, "sky_light_u8", "u1", samples),
        block_light_rgb=_binary_array(
            source,
            "block_light_rgb_u8",
            "u1",
            samples * 3,
        ).reshape(samples, 3),
        environment_code=_binary_array(
            source,
            "environment_i32_le",
            "<i4",
            samples,
        ),
        tint_argb=tint_argb,
    )


def _validate_capture_arrays(
    *,
    positions: np.ndarray,
    available: np.ndarray,
    channel_valid: np.ndarray,
    heightmap_block_y: np.ndarray,
    sky_light: np.ndarray,
    block_light_rgb: np.ndarray,
    environment_code: np.ndarray,
    tint_argb: np.ndarray,
) -> None:
    inside_y = (positions[:, 1] >= MIN_Y) & (
        positions[:, 1] < MIN_Y + WORLD_HEIGHT
    )
    if np.any(channel_valid[~available]) or np.any(heightmap_block_y[~available]):
        raise ValueError("unavailable native samples must be zero-masked")
    if (
        np.any(sky_light[~available])
        or np.any(block_light_rgb[~available])
        or np.any(environment_code[~available])
        or np.any(tint_argb[~available])
    ):
        raise ValueError("unavailable native channel values must be zero")
    if np.any(available & ~channel_valid[:, CHANNEL_TINT_RGB]):
        raise ValueError("covered native columns must publish tint")
    in_domain = available & inside_y
    if not np.array_equal(
        channel_valid[:, CHANNEL_ENVIRONMENT],
        in_domain,
    ):
        raise ValueError("native environment validity differs from its Y domain")
    sky_valid = channel_valid[:, CHANNEL_SKY_LIGHT]
    block_light_valid = channel_valid[:, CHANNEL_BLOCK_LIGHT_RGB]
    if (
        not np.array_equal(sky_valid, block_light_valid)
        or np.any(sky_valid & ~in_domain)
    ):
        raise ValueError(
            "native light validity differs from domain or global-light readiness"
        )
    invalid_height = ~channel_valid[:, CHANNEL_HEIGHTMAP]
    if np.any(heightmap_block_y[invalid_height]):
        raise ValueError("invalid native heightmap values must be zero")
    invalid_sky = ~channel_valid[:, CHANNEL_SKY_LIGHT]
    invalid_block = ~channel_valid[:, CHANNEL_BLOCK_LIGHT_RGB]
    invalid_environment = ~channel_valid[:, CHANNEL_ENVIRONMENT]
    invalid_tint = ~channel_valid[:, CHANNEL_TINT_RGB]
    if (
        np.any(sky_light[invalid_sky])
        or np.any(block_light_rgb[invalid_block])
        or np.any(environment_code[invalid_environment])
        or np.any(tint_argb[invalid_tint])
    ):
        raise ValueError("invalid native channel values must be zero")
    if np.any(sky_light > 15) or np.any(block_light_rgb > 15):
        raise ValueError("native light encoding exceeds the 0.5.7 nibble range")


def _array(value: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    return np.array(value, dtype=dtype, copy=True, order="C")


def _binary_array(
    source: Mapping[str, Any],
    key: str,
    dtype: str,
    count: int,
) -> np.ndarray:
    value = source.get(key)
    if not isinstance(value, bytes):
        raise TypeError(f"{key} must be bytes")
    expected = np.dtype(dtype).itemsize * count
    if len(value) != expected:
        raise ValueError(f"{key} has the wrong byte length")
    return np.frombuffer(value, dtype=dtype, count=count).copy()


def _wire_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _wire_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _require_metadata(metadata: object) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("native perception fixture metadata must be a map")
    if metadata.get("schema") != NATIVE_PERCEPTION_CHANNEL_SCHEMA:
        raise ValueError("native perception fixture schema differs")
    if metadata.get("version") != NATIVE_PERCEPTION_CHANNEL_VERSION:
        raise ValueError("native perception fixture version differs")
    if metadata.get("published_provenance") != "native":
        raise ValueError("native perception fixture provenance differs")


__all__ = [
    "CHANNEL_BLOCK_LIGHT_RGB",
    "CHANNEL_ENVIRONMENT",
    "CHANNEL_HEIGHTMAP",
    "CHANNEL_SKY_LIGHT",
    "CHANNEL_TINT_RGB",
    "NATIVE_PERCEPTION_CHANNEL_MAX_SAMPLES",
    "NATIVE_PERCEPTION_CHANNEL_SCHEMA",
    "NATIVE_PERCEPTION_CHANNEL_VERSION",
    "NativePerceptionChannelCapture",
    "NativePerceptionChannelTransport",
    "PERCEPTION_CHANNEL_COUNT",
    "VALID_ALL",
    "VALID_BLOCK_LIGHT_RGB",
    "VALID_ENVIRONMENT",
    "VALID_HEIGHTMAP",
    "VALID_SKY_LIGHT",
    "VALID_TINT_RGB",
    "capture_native_perception_channels",
    "native_perception_channel_capture_from_wire",
    "native_perception_channel_request",
]
