"""Strict host contract for dense native Region light sections."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Protocol

import numpy as np

from hytalegym.worldgen.region.contract import (
    BLOCK_SECTION_CLASS,
    BLOCK_SECTION_CLASS_SHA256,
    CHUNK_SIZE,
    HEIGHT_SECTIONS,
    SECTION_VOLUME,
)

REGION_LIGHT_SECTION_SCHEMA = "hytalerl_native_region_light_section_v1"
REGION_LIGHT_SECTION_VERSION = 1
REGION_LIGHT_ENCODING = "uint16_le_y_z_x_rgbs_nibbles"
REGION_LIGHT_BYTES = SECTION_VOLUME * np.dtype("<u2").itemsize
REGION_LIGHT_STATUS_READY = "ready"
REGION_LIGHT_STATUS_NOT_READY = "global_light_not_ready"
REGION_LIGHT_STATUS_CHANGED = "changed_during_capture"
CHUNK_LIGHT_DATA_CLASS = (
    "com.hypixel.hytale.server.core.universe.world.chunk.section.ChunkLightData"
)
CHUNK_LIGHT_DATA_CLASS_SHA256 = (
    "27553ca269da82be0a8cd72fded26e76fa837f825243a0f5fae914feebfa5649"
)

_STATUSES = frozenset(
    {
        REGION_LIGHT_STATUS_READY,
        REGION_LIGHT_STATUS_NOT_READY,
        REGION_LIGHT_STATUS_CHANGED,
    }
)


class NativeRegionLightTransport(Protocol):
    """The dense light verb exposed by an already-reset native Region."""

    def capture_region_light_section(
        self,
        chunk_x: int,
        chunk_z: int,
        section_y: int,
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class NativeRegionLightSection:
    """One native light section; ``None`` is unavailable, never darkness."""

    chunk_x: int
    chunk_z: int
    section_y: int
    status: str
    global_change_counter: int
    global_light_change_id: int
    light_raw_yzx: np.ndarray | None

    def __post_init__(self) -> None:
        if not 0 <= int(self.section_y) < HEIGHT_SECTIONS:
            raise ValueError("section_y must be inside the native world")
        if self.status not in _STATUSES:
            raise ValueError("unknown native Region light status")
        for name in ("global_change_counter", "global_light_change_id"):
            value = int(getattr(self, name))
            if not 0 <= value <= 0xFFFF:
                raise ValueError(f"{name} must be unsigned 16-bit")
        value = self.light_raw_yzx
        if self.status == REGION_LIGHT_STATUS_READY:
            if self.global_change_counter != self.global_light_change_id:
                raise ValueError("ready native Region light counters differ")
            if value is None:
                raise ValueError("ready native Region light needs a payload")
            array = np.asarray(value, dtype=np.uint16)
            if array.shape != (CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE):
                raise ValueError("native Region light must have shape [Y,Z,X]")
            array = np.ascontiguousarray(array)
            array.flags.writeable = False
            object.__setattr__(self, "light_raw_yzx", array)
        elif value is not None:
            raise ValueError("unavailable native Region light must omit payload")

    @property
    def available(self) -> bool:
        return self.status == REGION_LIGHT_STATUS_READY

    def channel(self, channel: int) -> np.ndarray:
        """Return one native nibble channel: R=0, G=1, B=2, sky=3."""

        if not 0 <= int(channel) < 4:
            raise ValueError("native light channel must be in [0, 4)")
        if self.light_raw_yzx is None:
            raise ValueError("native Region light is unavailable")
        result = (
            self.light_raw_yzx >> np.uint16(4 * int(channel))
        ) & np.uint16(0xF)
        result = result.astype(np.uint8)
        result.flags.writeable = False
        return result

    @property
    def block_light_rgb(self) -> np.ndarray:
        result = np.stack(
            (self.channel(0), self.channel(1), self.channel(2)),
            axis=-1,
        )
        result.flags.writeable = False
        return result

    @property
    def sky_light(self) -> np.ndarray:
        return self.channel(3)


def native_region_light_section_from_wire(
    response: Mapping[str, Any],
) -> NativeRegionLightSection:
    """Validate one bridge response without accepting dark-for-unavailable."""

    source = dict(response)
    if source.get("type") != "region_light_section":
        raise ValueError("native response is not a Region light section")
    if source.get("schema") != REGION_LIGHT_SECTION_SCHEMA:
        raise ValueError("unsupported native Region light schema")
    if _integer(source.get("version"), "version") != REGION_LIGHT_SECTION_VERSION:
        raise ValueError("unsupported native Region light version")
    if source.get("light_encoding") != REGION_LIGHT_ENCODING:
        raise ValueError("unsupported native Region light encoding")
    status = str(source.get("status", ""))
    if status not in _STATUSES:
        raise ValueError("unknown native Region light status")
    available = source.get("available")
    if not isinstance(available, bool):
        raise TypeError("available must be a bool")
    if available != (status == REGION_LIGHT_STATUS_READY):
        raise ValueError("native Region light status and availability differ")
    payload = source.get("light_data")
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("native Region light payload must be MessagePack binary")
    encoded = memoryview(payload)
    if available:
        if encoded.nbytes != REGION_LIGHT_BYTES:
            raise ValueError("native Region light payload has the wrong size")
        light_raw = np.frombuffer(
            encoded,
            dtype="<u2",
            count=SECTION_VOLUME,
        ).reshape((CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE)).copy()
    else:
        if encoded.nbytes:
            raise ValueError("unavailable native Region light published data")
        light_raw = None
    return NativeRegionLightSection(
        chunk_x=_integer(source.get("chunk_x"), "chunk_x"),
        chunk_z=_integer(source.get("chunk_z"), "chunk_z"),
        section_y=_integer(source.get("section_y"), "section_y"),
        status=status,
        global_change_counter=_integer(
            source.get("global_change_counter"),
            "global_change_counter",
        ),
        global_light_change_id=_integer(
            source.get("global_light_change_id"),
            "global_light_change_id",
        ),
        light_raw_yzx=light_raw,
    )


def capture_native_region_light_section(
    transport: NativeRegionLightTransport,
    chunk_x: int,
    chunk_z: int,
    section_y: int,
) -> NativeRegionLightSection:
    """Capture and validate one dense native light section."""

    return native_region_light_section_from_wire(
        transport.capture_region_light_section(chunk_x, chunk_z, section_y)
    )


def native_region_light_section_contract() -> dict[str, object]:
    """Describe the dense sidecar without changing Region geometry identity."""

    return {
        "schema": "hytalerl_native_region_light_section_host_v1",
        "version": 1,
        "wire_schema": REGION_LIGHT_SECTION_SCHEMA,
        "wire_version": REGION_LIGHT_SECTION_VERSION,
        "source_class": BLOCK_SECTION_CLASS,
        "source_class_sha256": BLOCK_SECTION_CLASS_SHA256,
        "raw_light_source": {
            "class": CHUNK_LIGHT_DATA_CLASS,
            "class_sha256": CHUNK_LIGHT_DATA_CLASS_SHA256,
            "method": "getLightRaw",
        },
        "spatial_scope": {
            "shape_yzx": [CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE],
            "encoding": REGION_LIGHT_ENCODING,
        },
        "channels": {
            "red_block_light": "bits_0_to_3_uint4",
            "green_block_light": "bits_4_to_7_uint4",
            "blue_block_light": "bits_8_to_11_uint4",
            "sky_light": "bits_12_to_15_uint4",
        },
        "readiness": (
            "BlockSection.hasGlobalLight_and_globalLight_is_not_ChunkLightData.EMPTY"
        ),
        "fresh_section_trap": (
            "EMPTY.changeId_equals_initial_counter_zero_before_calculation"
        ),
        "calculation_preparation": {
            "trigger": "first_dense_region_light_request_only",
            "geometry_capture_chunks_per_axis": 5,
            "native_light_halo_chunks": 1,
            "pinned_chunks_per_axis": 7,
            "queued_sections": 7 * 7 * HEIGHT_SECTIONS,
            "placeholder_rule": (
                "invalidate_only_ChunkLightData.EMPTY_local_or_global_light"
            ),
            "computed_light_preserved": True,
            "paused_world_progression": (
                "one_noop_authoritative_tick_between_capture_passes"
            ),
        },
        "not_ready_action": "enqueue_ChunkLightingManager_section",
        "counter_semantics": "unsigned_short_equality_only_not_ordering",
        "unavailable_payload": "empty_not_darkness",
        "capture_atomicity": "pre_and_post_counter_and_object_identity_check",
        "artifact_role": "optional_native_sidecar_not_region_geometry_identity",
        "published_provenance": "native",
    }


def native_region_light_section_contract_sha256() -> str:
    payload = json.dumps(
        native_region_light_section_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)


__all__ = [
    "CHUNK_LIGHT_DATA_CLASS",
    "CHUNK_LIGHT_DATA_CLASS_SHA256",
    "REGION_LIGHT_BYTES",
    "REGION_LIGHT_ENCODING",
    "REGION_LIGHT_SECTION_SCHEMA",
    "REGION_LIGHT_SECTION_VERSION",
    "REGION_LIGHT_STATUS_CHANGED",
    "REGION_LIGHT_STATUS_NOT_READY",
    "REGION_LIGHT_STATUS_READY",
    "NativeRegionLightSection",
    "NativeRegionLightTransport",
    "capture_native_region_light_section",
    "native_region_light_section_contract",
    "native_region_light_section_contract_sha256",
    "native_region_light_section_from_wire",
]
