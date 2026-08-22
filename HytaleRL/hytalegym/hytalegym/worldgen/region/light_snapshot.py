"""Frozen native lighting paired with, but separate from, Region geometry."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from hytalegym.worldgen.region.capture import RegionCapturePlan
from hytalegym.worldgen.region.contract import (
    CAPTURE_CHUNK_COUNT,
    CHUNK_SIZE,
    HEIGHT_SECTIONS,
)
from hytalegym.worldgen.region.native_light import (
    NativeRegionLightSection,
    NativeRegionLightTransport,
    capture_native_region_light_section,
    native_region_light_section_contract_sha256,
)
from hytalegym.worldgen.region.snapshot import NativeRegionSnapshot


REGION_LIGHT_SNAPSHOT_SCHEMA = "hytalerl_native_region_light_snapshot_v1"
REGION_LIGHT_SNAPSHOT_VERSION = 1
_META_KEY = "__metadata_json__"
_CORE_KEY = "__core_min_chunk_xz__"
_AVAILABLE_KEY = "__section_available__"
_REQUIRED_KEY = "__section_required__"
_LIGHT_KEY = "light_raw_yzx"
_SNAPSHOT_KEYS = {
    _META_KEY,
    _CORE_KEY,
    _AVAILABLE_KEY,
    _REQUIRED_KEY,
    _LIGHT_KEY,
}


@dataclass(frozen=True)
class NativeRegionLightSnapshot:
    """Sparse-valid dense native light for one exact Region snapshot."""

    metadata: Mapping[str, Any]
    core_min_chunk_xz: np.ndarray
    section_available: np.ndarray
    section_required: np.ndarray
    light_raw_yzx: np.ndarray

    def __post_init__(self) -> None:
        metadata = dict(self.metadata)
        if metadata.get("schema") != REGION_LIGHT_SNAPSHOT_SCHEMA:
            raise ValueError("unsupported native Region light snapshot schema")
        if metadata.get("version") != REGION_LIGHT_SNAPSHOT_VERSION:
            raise ValueError("unsupported native Region light snapshot version")
        for field in (
            "source_region_semantic_sha256",
            "native_region_light_contract_sha256",
            "evidence_bridge_sha256",
        ):
            metadata[field] = _sha256(metadata.get(field), field)
        if metadata["native_region_light_contract_sha256"] != (
            native_region_light_section_contract_sha256()
        ):
            raise ValueError("native Region light contract changed")
        core = _integer_array(
            self.core_min_chunk_xz,
            np.int32,
            (2,),
            "core_min_chunk_xz",
        )
        section_shape = (CAPTURE_CHUNK_COUNT, HEIGHT_SECTIONS)
        available = _bool_array(
            self.section_available,
            section_shape,
            "section_available",
        )
        required = _bool_array(
            self.section_required,
            section_shape,
            "section_required",
        )
        if np.any(required & ~available):
            raise ValueError("required native Region light section is unavailable")
        light = _integer_array(
            self.light_raw_yzx,
            np.uint16,
            section_shape + (CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE),
            "light_raw_yzx",
        )
        if np.any(light[~available]):
            raise ValueError("unavailable native Region light must be zero")
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "core_min_chunk_xz", core)
        object.__setattr__(self, "section_available", available)
        object.__setattr__(self, "section_required", required)
        object.__setattr__(self, "light_raw_yzx", light)

    @property
    def source_region_semantic_sha256(self) -> str:
        return str(self.metadata["source_region_semantic_sha256"])

    @property
    def evidence_bridge_sha256(self) -> str:
        return str(self.metadata["evidence_bridge_sha256"])

    def semantic_digest(self) -> str:
        """Hash light meaning while excluding path and bridge provenance."""

        digest = hashlib.sha256()
        semantic = {
            "schema": REGION_LIGHT_SNAPSHOT_SCHEMA,
            "version": REGION_LIGHT_SNAPSHOT_VERSION,
            "source_region_semantic_sha256": (
                self.source_region_semantic_sha256
            ),
            "native_region_light_contract_sha256": self.metadata[
                "native_region_light_contract_sha256"
            ],
        }
        digest.update(
            json.dumps(
                semantic,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        for value in (
            self.core_min_chunk_xz,
            self.section_available,
            self.section_required,
            self.light_raw_yzx,
        ):
            digest.update(np.ascontiguousarray(value).tobytes())
        return digest.hexdigest()

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as output:
                temporary = Path(output.name)
                np.savez_compressed(
                    output,
                    **{
                        _META_KEY: np.asarray(
                            json.dumps(
                                dict(self.metadata),
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                        ),
                        _CORE_KEY: self.core_min_chunk_xz,
                        _AVAILABLE_KEY: self.section_available,
                        _REQUIRED_KEY: self.section_required,
                        _LIGHT_KEY: self.light_raw_yzx,
                    },
                )
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> "NativeRegionLightSnapshot":
        with np.load(Path(path), allow_pickle=False) as archive:
            if set(archive.files) != _SNAPSHOT_KEYS:
                raise ValueError("native Region light snapshot keys changed")
            metadata = json.loads(str(archive[_META_KEY].item()))
            if not isinstance(metadata, dict):
                raise ValueError("native Region light metadata must be an object")
            return cls(
                metadata=metadata,
                core_min_chunk_xz=archive[_CORE_KEY].copy(),
                section_available=archive[_AVAILABLE_KEY].copy(),
                section_required=archive[_REQUIRED_KEY].copy(),
                light_raw_yzx=archive[_LIGHT_KEY].copy(),
            )


def capture_native_region_light_snapshot(
    transport: NativeRegionLightTransport,
    source_region: NativeRegionSnapshot,
    *,
    evidence_bridge_sha256: str,
    maximum_passes: int = 8,
    stable_passes: int = 2,
    poll_interval_seconds: float = 0.05,
    advance_readiness: Callable[[], None] | None = None,
) -> NativeRegionLightSnapshot:
    """Queue and capture every block-bearing section, retaining ready air."""

    if not isinstance(source_region, NativeRegionSnapshot):
        raise TypeError("source_region must be a NativeRegionSnapshot")
    passes = _positive_int(maximum_passes, "maximum_passes")
    stable_required = _positive_int(stable_passes, "stable_passes")
    if stable_required > passes:
        raise ValueError("stable_passes cannot exceed maximum_passes")
    delay = float(poll_interval_seconds)
    if not np.isfinite(delay) or delay < 0.0:
        raise ValueError("poll_interval_seconds must be finite and nonnegative")
    bridge = _sha256(evidence_bridge_sha256, "evidence_bridge_sha256")
    plan = RegionCapturePlan.complete(
        int(source_region.core_min_chunk_xz[0]),
        int(source_region.core_min_chunk_xz[1]),
        maximum_source_reach_blocks=0.0,
    )
    required = source_region.section_known & np.any(
        source_region.cell_code != 0,
        axis=2,
    )
    available = np.zeros(required.shape, dtype=np.bool_)
    light = np.zeros(
        required.shape + (CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE),
        dtype=np.uint16,
    )
    stable = 0
    for pass_index in range(passes):
        before = available.copy()
        for task_index, task in enumerate(plan.tasks):
            section = capture_native_region_light_section(
                transport,
                task.chunk_x,
                task.chunk_z,
                task.section_y,
            )
            slot = task_index // HEIGHT_SECTIONS
            if available[slot, task.section_y] and not section.available:
                raise ValueError(
                    "native Region light became unavailable during capture"
                )
            if section.available:
                assert section.light_raw_yzx is not None
                if available[slot, task.section_y] and not np.array_equal(
                    light[slot, task.section_y],
                    section.light_raw_yzx,
                ):
                    raise ValueError(
                        "native Region light changed during bounded capture"
                    )
                available[slot, task.section_y] = True
                light[slot, task.section_y] = section.light_raw_yzx
        stable = stable + 1 if np.array_equal(before, available) else 0
        if np.all(available[required]) and stable >= stable_required:
            return NativeRegionLightSnapshot(
                metadata={
                    "schema": REGION_LIGHT_SNAPSHOT_SCHEMA,
                    "version": REGION_LIGHT_SNAPSHOT_VERSION,
                    "source_region_semantic_sha256": (
                        source_region.semantic_digest()
                    ),
                    "native_region_light_contract_sha256": (
                        native_region_light_section_contract_sha256()
                    ),
                    "evidence_bridge_sha256": bridge,
                    "capture_passes": pass_index + 1,
                    "capture_policy": (
                        "all_block_bearing_sections_required_air_retained_when_ready"
                    ),
                    "readiness_progression": (
                        "caller_callback_between_passes"
                        if advance_readiness is not None
                        else "external_native_world_progress"
                    ),
                },
                core_min_chunk_xz=source_region.core_min_chunk_xz,
                section_available=available,
                section_required=required,
                light_raw_yzx=light,
            )
        if advance_readiness is not None:
            advance_readiness()
        if delay:
            time.sleep(delay)
    missing = np.argwhere(required & ~available).tolist()
    if not missing:
        raise RuntimeError(
            "native Region light became ready but did not remain stable for "
            f"{stable_required} passes within {passes} capture passes"
        )
    raise RuntimeError(
        "native Region light did not settle for required sections: "
        f"{missing}"
    )


def native_region_light_snapshot_from_sections(
    *,
    source_region_semantic_sha256: str,
    evidence_bridge_sha256: str,
    core_min_chunk_xz: Sequence[int],
    section_required: np.ndarray,
    sections: Sequence[NativeRegionLightSection],
) -> NativeRegionLightSnapshot:
    """Assemble a snapshot from already captured, uniquely addressed rows."""

    core = np.asarray(core_min_chunk_xz, dtype=np.int32)
    if core.shape != (2,):
        raise ValueError("core_min_chunk_xz must have shape [2]")
    required = _bool_array(
        section_required,
        (CAPTURE_CHUNK_COUNT, HEIGHT_SECTIONS),
        "section_required",
    )
    available = np.zeros(required.shape, dtype=np.bool_)
    light = np.zeros(
        required.shape + (CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE),
        dtype=np.uint16,
    )
    minimum_x = int(core[0]) - 1
    minimum_z = int(core[1]) - 1
    seen: set[tuple[int, int, int]] = set()
    for section in sections:
        if not isinstance(section, NativeRegionLightSection):
            raise TypeError("sections must contain NativeRegionLightSection")
        key = (section.chunk_x, section.chunk_z, section.section_y)
        if key in seen:
            raise ValueError("native Region light section is duplicated")
        seen.add(key)
        dx = section.chunk_x - minimum_x
        dz = section.chunk_z - minimum_z
        if not 0 <= dx < 5 or not 0 <= dz < 5:
            raise ValueError("native Region light section is outside capture")
        slot = dx * 5 + dz
        if section.available:
            assert section.light_raw_yzx is not None
            available[slot, section.section_y] = True
            light[slot, section.section_y] = section.light_raw_yzx
    return NativeRegionLightSnapshot(
        metadata={
            "schema": REGION_LIGHT_SNAPSHOT_SCHEMA,
            "version": REGION_LIGHT_SNAPSHOT_VERSION,
            "source_region_semantic_sha256": _sha256(
                source_region_semantic_sha256,
                "source_region_semantic_sha256",
            ),
            "native_region_light_contract_sha256": (
                native_region_light_section_contract_sha256()
            ),
            "evidence_bridge_sha256": _sha256(
                evidence_bridge_sha256,
                "evidence_bridge_sha256",
            ),
            "capture_passes": 0,
            "capture_policy": "caller_supplied_complete_section_set",
        },
        core_min_chunk_xz=core,
        section_available=available,
        section_required=required,
        light_raw_yzx=light,
    )


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA-256 string")
    clean = value.lower()
    if len(clean) != 64 or any(c not in "0123456789abcdef" for c in clean):
        raise ValueError(f"{name} must be a SHA-256 string")
    return clean


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _integer_array(
    value: object,
    dtype: np.dtype[Any] | type[np.generic],
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind not in "iu":
        raise TypeError(f"{name} must be an integer array")
    target_dtype = np.dtype(dtype)
    bounds = np.iinfo(target_dtype)
    if np.any(source < bounds.min) or np.any(source > bounds.max):
        raise ValueError(f"{name} exceeds {target_dtype} range")
    result = np.asarray(source, dtype=target_dtype)
    if result.shape != shape:
        raise ValueError(f"{name} has the wrong shape")
    result = np.ascontiguousarray(result)
    result.flags.writeable = False
    return result


def _bool_array(
    value: object,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype != np.bool_:
        raise TypeError(f"{name} must be a bool array")
    if source.shape != shape:
        raise ValueError(f"{name} has the wrong shape")
    result = np.ascontiguousarray(source)
    result.flags.writeable = False
    return result


__all__ = [
    "REGION_LIGHT_SNAPSHOT_SCHEMA",
    "REGION_LIGHT_SNAPSHOT_VERSION",
    "NativeRegionLightSnapshot",
    "capture_native_region_light_snapshot",
    "native_region_light_snapshot_from_sections",
]
