"""Versioned chunk and fixed-region shapes derived from Hytale's local API."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import operator
from typing import Any, Mapping

CHUNK_API_SCHEMA = "hytalerl_chunk_api_v1"
CHUNK_API_VERSION = 1
HYTALE_SERVER_VERSION = "0.5.7"
CHUNK_API_CLASS = "com.hypixel.hytale.math.util.ChunkUtil"

# These are compiled JAX shapes, not unverified world assumptions. Java
# reflects and exhaustively checks the runtime API before emitting this
# manifest; Python accepts only that exact versioned fingerprint.
CHUNK_BITS = 5
CHUNK_SIZE = 32
CHUNK_SIZE_SQUARED = 1024
SECTION_VOLUME = 32768
HEIGHT_SECTIONS = 10
WORLD_HEIGHT = 320
MIN_Y = 0
MIN_CHUNK_COORDINATE = -67_108_864
MAX_CHUNK_COORDINATE = 67_108_863

CORE_CHUNKS_PER_AXIS = 3
CAPTURE_HALO_CHUNKS = 1
CAPTURE_CHUNKS_PER_AXIS = CORE_CHUNKS_PER_AXIS + 2 * CAPTURE_HALO_CHUNKS
CAPTURE_CHUNK_COUNT = CAPTURE_CHUNKS_PER_AXIS**2
CAPTURE_SECTION_COUNT = CAPTURE_CHUNK_COUNT * HEIGHT_SECTIONS
CORE_BLOCKS_PER_AXIS = CORE_CHUNKS_PER_AXIS * CHUNK_SIZE
CAPTURE_BLOCKS_PER_AXIS = CAPTURE_CHUNKS_PER_AXIS * CHUNK_SIZE
MAX_CELL_PALETTE = 1 << 16
MAX_SHAPE_PALETTE = 1 << 14
MAX_SHAPE_BOXES = 9
# The bundled 0.5.7 asset audit observed less than three horizontal blocks of
# outward shape reach. Four blocks is the versioned conservative capture bound.
CERTIFIED_SOURCE_REACH_BOUND_BLOCKS = 4.0

NATIVE_SECTION_AGGREGATE_SCHEMA = "hytalerl_native_section_aggregates_v2"
NATIVE_SECTION_AGGREGATE_VERSION = 2
REGION_DELTA_SCHEMA = "hytalerl_region_delta_v1"
REGION_DELTA_VERSION = 1
BLOCK_SECTION_CLASS = (
    "com.hypixel.hytale.server.core.universe.world.chunk.section.BlockSection"
)
BLOCK_SECTION_CLASS_SHA256 = (
    "059e4595fe336a95983ac035dfd3ecfd4a560a851962bdb350a90136263520d5"
)

_EXPECTED_FIELDS = {
    "schema": CHUNK_API_SCHEMA,
    "version": CHUNK_API_VERSION,
    "server_version": HYTALE_SERVER_VERSION,
    "api_class": CHUNK_API_CLASS,
    "bits": CHUNK_BITS,
    "size": CHUNK_SIZE,
    "size_squared": CHUNK_SIZE_SQUARED,
    "section_volume": SECTION_VOLUME,
    "height_sections": HEIGHT_SECTIONS,
    "height": WORLD_HEIGHT,
    "min_y": MIN_Y,
    "min_chunk_coordinate": MIN_CHUNK_COORDINATE,
    "max_chunk_coordinate": MAX_CHUNK_COORDINATE,
    "local_coordinate": "arithmetic_shift_and_mask",
    "section_index_order": "y_z_x",
}


class UnsupportedChunkApiError(ValueError):
    """The native API cannot be represented by this fixed JAX build."""


@dataclass(frozen=True)
class ChunkApiContract:
    """Validated runtime manifest for the Hytale 0.5.7 ``ChunkUtil`` API."""

    server_version: str
    bits: int
    size: int
    size_squared: int
    section_volume: int
    height_sections: int
    height: int
    min_y: int
    min_chunk_coordinate: int
    max_chunk_coordinate: int

    @classmethod
    def from_manifest(
        cls,
        manifest: Mapping[str, Any],
    ) -> "ChunkApiContract":
        if not isinstance(manifest, Mapping):
            raise UnsupportedChunkApiError(
                "Hytale chunk API manifest must be a mapping"
            )
        mismatches = {
            name: (expected, manifest.get(name))
            for name, expected in _EXPECTED_FIELDS.items()
            if not _same_manifest_value(manifest.get(name), expected)
        }
        if mismatches:
            details = ", ".join(
                f"{name}={actual!r} (expected {expected!r})"
                for name, (expected, actual) in mismatches.items()
            )
            raise UnsupportedChunkApiError(f"unsupported Hytale chunk API: {details}")
        return cls(
            server_version=str(manifest["server_version"]),
            bits=int(manifest["bits"]),
            size=int(manifest["size"]),
            size_squared=int(manifest["size_squared"]),
            section_volume=int(manifest["section_volume"]),
            height_sections=int(manifest["height_sections"]),
            height=int(manifest["height"]),
            min_y=int(manifest["min_y"]),
            min_chunk_coordinate=int(manifest["min_chunk_coordinate"]),
            max_chunk_coordinate=int(manifest["max_chunk_coordinate"]),
        )

    def manifest(self) -> dict[str, Any]:
        """Return the canonical manifest stored in native region artifacts."""

        return dict(_EXPECTED_FIELDS)

    def chunk_coordinate(self, block_coordinate: int) -> int:
        return _exact_int(block_coordinate, "block coordinate") // self.size

    def local_coordinate(self, block_coordinate: int) -> int:
        return _exact_int(block_coordinate, "block coordinate") & (self.size - 1)

    def section_y(self, block_y: int) -> int:
        y = _exact_int(block_y, "block Y")
        if not self.min_y <= y < self.min_y + self.height:
            raise ValueError(f"block Y {y} is outside the native world")
        return (y - self.min_y) >> self.bits

    def section_index(
        self,
        local_x: int,
        local_y: int,
        local_z: int,
    ) -> int:
        coordinates = (
            _exact_int(local_x, "local X"),
            _exact_int(local_y, "local Y"),
            _exact_int(local_z, "local Z"),
        )
        if any(value < 0 or value >= self.size for value in coordinates):
            raise ValueError("section coordinates must be in [0, 32)")
        x, y, z = coordinates
        return y * self.size_squared + z * self.size + x


def certified_chunk_api_manifest() -> dict[str, Any]:
    """Canonical expected output of Java's runtime-reflected API check."""

    return dict(_EXPECTED_FIELDS)


def native_section_aggregate_contract() -> dict[str, Any]:
    """Describe native section aggregates without claiming capture support."""

    return {
        "schema": NATIVE_SECTION_AGGREGATE_SCHEMA,
        "version": NATIVE_SECTION_AGGREGATE_VERSION,
        "server_version": HYTALE_SERVER_VERSION,
        "provenance": "native_api_contract_not_captured",
        "source_class": BLOCK_SECTION_CLASS,
        "source_class_sha256": BLOCK_SECTION_CLASS_SHA256,
        "capture_status": "bridge_endpoint_unimplemented",
        "spatial_scope": {
            "kind": "native_aligned_section",
            "shape_xyz": [CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE],
            "volume": SECTION_VOLUME,
            "identity": ["world_id", "chunk_x", "chunk_z", "section_y"],
            "agent_centered": False,
            "boundary_behavior": (
                "feature_changes_discontinuously_when_the_agent_crosses_a_section"
            ),
        },
        "methods": {
            "value_counts": {
                "native": "valueCounts()",
                "result": "Int2ShortMap",
                "warning": "signed_short_counts_require_native_encoding_fixture",
            },
            "contains_any": {
                "native": "containsAny(IntList)",
                "result": "bool",
            },
            "count": {
                "native": ["count()", "count(int)"],
                "result": "int",
            },
            "find": {
                "native": "find(IntList,...)",
                "result": "section_local_positions",
            },
            "solid_air": {
                "native": "isSolidAir()",
                "result": "bool",
            },
            "filler": {
                "native": ["getFiller(int)", "getFiller(int,int,int)"],
                "result": "packed_signed_5bit_xyz_offset_to_root",
                "zero": "root_or_not_a_filler",
            },
            "ticking_count": {
                "native": "getTickingBlocksCount()",
                "result": "int",
            },
        },
        "cost": "native_palette_or_fast_path_not_host_32_cubed_scan",
        "consumer_semantics": (
            "coarse_section_composition_not_within_radius_of_agent"
        ),
        "runtime_block_ids": "process_local_not_portable",
    }


def native_section_aggregate_contract_sha256() -> str:
    """Return the canonical section-aggregate contract digest."""

    return _canonical_contract_sha256(native_section_aggregate_contract())


def region_delta_contract() -> dict[str, Any]:
    """Describe fail-closed Region deltas and their full-resync boundary."""

    return {
        "schema": REGION_DELTA_SCHEMA,
        "version": REGION_DELTA_VERSION,
        "server_version": HYTALE_SERVER_VERSION,
        "provenance": "native_api_contract_design_only",
        "source_class": BLOCK_SECTION_CLASS,
        "source_class_sha256": BLOCK_SECTION_CLASS_SHA256,
        "implementation_status": "unimplemented",
        "base_identity": [
            "world_id",
            "full_snapshot_semantic_sha256",
            "chunk_x",
            "chunk_z",
            "section_y",
            "resync_epoch",
        ],
        "changed_positions": {
            "native": "getAndClearChangedPositions()",
            "encoding": "section_index_y_z_x",
            "scope": "loaded_block_filler_or_rotation_changes",
            "delivery": "double_buffered_destructive_drain",
        },
        "light_counters": {
            "local": "getLocalChangeCounter()",
            "global": "getGlobalChangeCounter()",
            "wire_type": "signed_short_wrapping",
            "comparison": "inequality_only",
            "oq_05": {
                "status": "resolved_from_0_5_7_bytecode",
                "block_change": (
                    "records_position_then_invalidates_local_and_global_light"
                ),
                "light_invalidation_only": (
                    "can_increment_counters_without_a_changed_position"
                ),
                "set_light_data": (
                    "setLocalLight_and_setGlobalLight_do_not_increment_counters"
                ),
            },
        },
        "resync_required_on": [
            "episode_or_world_reset",
            "initial_chunk_or_section_load",
            "section_unload_or_reload",
            "bridge_reconnect",
            "missing_duplicate_or_out_of_order_delta",
            "delta_capacity_overflow",
            "counter_change_without_matching_position_delta",
            "counter_wrap_or_ambiguous_baseline",
            "semantic_hash_mismatch",
        ],
        "publication": {
            "sequence": [
                "capture_complete_full_snapshot",
                "publish_snapshot_and_resync_epoch_atomically",
                "acknowledge_base_identity_and_counters",
                "drain_and_apply_matching_deltas_atomically",
            ],
            "failure": (
                "leave_active_atlas_unchanged_and_require_full_resync"
            ),
            "partial_sections": "unknown_never_air",
        },
    }


def region_delta_contract_sha256() -> str:
    """Return the canonical fail-closed delta-design digest."""

    return _canonical_contract_sha256(region_delta_contract())


def capture_chunk_slot(
    chunk_x: int,
    chunk_z: int,
    capture_min_chunk_x: int,
    capture_min_chunk_z: int,
) -> int:
    """Return the X-major 5x5 capture slot, rejecting out-of-region chunks."""

    relative_x = _exact_int(chunk_x, "chunk X") - _exact_int(
        capture_min_chunk_x,
        "capture minimum X",
    )
    relative_z = _exact_int(chunk_z, "chunk Z") - _exact_int(
        capture_min_chunk_z,
        "capture minimum Z",
    )
    if (
        relative_x < 0
        or relative_x >= CAPTURE_CHUNKS_PER_AXIS
        or relative_z < 0
        or relative_z >= CAPTURE_CHUNKS_PER_AXIS
    ):
        raise ValueError("chunk lies outside the capture halo")
    return relative_x * CAPTURE_CHUNKS_PER_AXIS + relative_z


def _same_manifest_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, int):
        return (
            isinstance(actual, int)
            and not isinstance(actual, bool)
            and (actual == expected)
        )
    return isinstance(actual, str) and actual == expected


def _canonical_contract_sha256(contract: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exact_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{label} must be an integer") from error
