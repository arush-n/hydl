"""Typed bridge contract for controlled native mutable-block evidence."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from hytalegym.geometry.contract import (
    FLUID_MOVEMENT_FEATURES,
    MAX_DETAIL_BOXES,
    MOVEMENT_FEATURES,
)
from hytalegym.jax.world.mutable_blocks import (
    BLOCK_SEMANTIC_KEY_WORDS,
    mutable_block_contract_sha256,
)
from hytalegym.worldgen.block_affordances import (
    BLOCK_AFFORDANCE_TAGS,
    GATHER_TYPES,
    block_affordance_dictionary_sha256,
)


NATIVE_MUTABLE_BLOCK_SCHEMA = "hytalerl_native_mutable_block_evidence_v1"
NATIVE_MUTABLE_BLOCK_VERSION = 1
NATIVE_MUTABLE_BLOCK_PHASES = (
    "initial",
    "damaged",
    "repaired",
    "removed",
    "placed",
)
NATIVE_MUTABLE_BLOCK_CELLS_SCHEMA = (
    "hytalerl_native_mutable_block_cells_v1"
)
NATIVE_MUTABLE_BLOCK_CELLS_VERSION = 1
NATIVE_MUTABLE_BLOCK_CELL_CAPACITY = 64


@dataclass(frozen=True)
class NativeMutableBlockRow:
    """One exact native value after a controlled transition."""

    phase: str
    block_present: bool
    block_asset_id: str
    runtime_block_id: int
    semantic_key: npt.NDArray[np.uint32]
    semantic_key_valid: bool
    affordance_valid: bool
    affordance_tags: int
    gather_type_index: int
    required_tool_quality: int
    rotation_index: int
    flags: int
    fluid_level: int
    fluid_fill_height: float
    support: int
    block_damage: int
    fluid_damage: int
    movement: npt.NDArray[np.float64]
    fluid_movement: npt.NDArray[np.float64]
    collision_boxes: npt.NDArray[np.float64]
    block_health: float
    block_health_valid: bool
    seconds_since_damage: float
    damage_age_valid: bool
    local_change_counter: int
    global_change_counter: int

    @classmethod
    def from_response(cls, value: Mapping[str, Any]) -> NativeMutableBlockRow:
        """Parse and validate one bridge row."""

        phase = _string(value, "phase")
        present = _boolean(value, "block_present")
        key_valid = _boolean(value, "semantic_key_valid")
        key_bytes = _bytes(value, "semantic_key_sha256")
        if len(key_bytes) != (32 if key_valid else 0):
            raise ValueError("mutable block semantic key has the wrong length")
        semantic_key = (
            np.frombuffer(key_bytes, dtype=">u4").astype(np.uint32)
            if key_valid
            else np.zeros(BLOCK_SEMANTIC_KEY_WORDS, dtype=np.uint32)
        )
        movement = _float64_payload(
            value,
            "movement_f64_le",
            MOVEMENT_FEATURES,
        )
        fluid_movement = _float64_payload(
            value,
            "fluid_movement_f64_le",
            FLUID_MOVEMENT_FEATURES,
        )
        box_count = _integer(value, "collision_box_count")
        if not 0 <= box_count <= MAX_DETAIL_BOXES:
            raise ValueError("mutable block collision capacity is invalid")
        collision_boxes = _float64_payload(
            value,
            "collision_boxes_f64_le",
            box_count * 6,
        ).reshape(box_count, 6)
        health = _number(value, "block_health")
        health_valid = _boolean(value, "block_health_valid")
        age = _number(value, "seconds_since_damage")
        age_valid = _boolean(value, "damage_age_valid")
        affordance_valid = _boolean(value, "affordance_valid")
        affordance_tags = _integer(value, "affordance_tags")
        gather_type = _integer(value, "gather_type_index")
        required_quality = _integer(value, "required_tool_quality")
        if present != key_valid or present != health_valid:
            raise ValueError(
                "mutable block identity and health must match presence"
            )
        if present != affordance_valid:
            raise ValueError(
                "mutable block affordances must match presence"
            )
        known_tag_mask = (1 << len(BLOCK_AFFORDANCE_TAGS)) - 1
        if (
            not 0 <= affordance_tags <= 0xFFFF
            or affordance_tags & ~known_tag_mask
            or not 0 <= gather_type < len(GATHER_TYPES)
            or not 0 <= required_quality <= 32_767
            or (
                not affordance_valid
                and (affordance_tags or gather_type or required_quality)
            )
        ):
            raise ValueError(
                "mutable block affordance value is outside its dictionary"
            )
        if not 0.0 <= health <= 1.0 or (not present and health != 0.0):
            raise ValueError("mutable block health is outside its domain")
        if age < 0.0 or (not age_valid and age != 0.0):
            raise ValueError("mutable block damage age is outside its domain")
        return cls(
            phase=phase,
            block_present=present,
            block_asset_id=_string(value, "block_asset_id", allow_empty=True),
            runtime_block_id=_integer(value, "runtime_block_id"),
            semantic_key=semantic_key,
            semantic_key_valid=key_valid,
            affordance_valid=affordance_valid,
            affordance_tags=affordance_tags,
            gather_type_index=gather_type,
            required_tool_quality=required_quality,
            rotation_index=_integer(value, "rotation_index"),
            flags=_integer(value, "flags"),
            fluid_level=_integer(value, "fluid_level"),
            fluid_fill_height=_number(value, "fluid_fill_height"),
            support=_integer(value, "support"),
            block_damage=_integer(value, "block_damage"),
            fluid_damage=_integer(value, "fluid_damage"),
            movement=movement,
            fluid_movement=fluid_movement,
            collision_boxes=collision_boxes,
            block_health=health,
            block_health_valid=health_valid,
            seconds_since_damage=age,
            damage_age_valid=age_valid,
            local_change_counter=_signed_short(
                value,
                "local_change_counter",
            ),
            global_change_counter=_signed_short(
                value,
                "global_change_counter",
            ),
        )

    def portable_geometry(self) -> dict[str, object]:
        """Return runtime-ID-independent exact geometry for comparison."""

        return {
            "block_present": self.block_present,
            "block_asset_id": self.block_asset_id,
            "semantic_key": self.semantic_key.tolist(),
            "semantic_key_valid": self.semantic_key_valid,
            "affordance_valid": self.affordance_valid,
            "affordance_tags": self.affordance_tags,
            "gather_type_index": self.gather_type_index,
            "required_tool_quality": self.required_tool_quality,
            "rotation_index": self.rotation_index,
            "flags": self.flags,
            "fluid_level": self.fluid_level,
            "fluid_fill_height": self.fluid_fill_height,
            "support": self.support,
            "block_damage": self.block_damage,
            "fluid_damage": self.fluid_damage,
            "movement": self.movement.tolist(),
            "fluid_movement": self.fluid_movement.tolist(),
            "collision_boxes": self.collision_boxes.tolist(),
        }


@dataclass(frozen=True)
class NativeMutableBlockEvidence:
    """One reversible native damage/repair/remove/place sequence."""

    server_version: str
    world: str
    worldgen_provider: str
    worldgen_version: str
    seed: int
    position: npt.NDArray[np.int32]
    rows: tuple[NativeMutableBlockRow, ...]

    @classmethod
    def from_response(
        cls,
        value: Mapping[str, Any],
    ) -> NativeMutableBlockEvidence:
        """Parse and validate a complete bridge response."""

        if value.get("type") != "mutable_block_evidence":
            raise ValueError("bridge returned the wrong mutable block type")
        if value.get("schema") != NATIVE_MUTABLE_BLOCK_SCHEMA:
            raise ValueError("bridge returned the wrong mutable block schema")
        if value.get("version") != NATIVE_MUTABLE_BLOCK_VERSION:
            raise ValueError("bridge returned the wrong mutable block version")
        if value.get("affordance_dictionary_sha256") != (
            block_affordance_dictionary_sha256()
        ):
            raise ValueError(
                "bridge mutable-block affordance dictionary is stale"
            )
        raw_position = value.get("position_i32_xyz")
        if (
            not isinstance(raw_position, (list, tuple))
            or len(raw_position) != 3
            or any(isinstance(item, bool) or not isinstance(item, int)
                    for item in raw_position)
        ):
            raise ValueError("mutable block position must be one XYZ triple")
        raw_rows = value.get("rows")
        if not isinstance(raw_rows, (list, tuple)):
            raise ValueError("mutable block rows must be an array")
        rows = tuple(
            NativeMutableBlockRow.from_response(_mapping(row))
            for row in raw_rows
        )
        if tuple(row.phase for row in rows) != NATIVE_MUTABLE_BLOCK_PHASES:
            raise ValueError("mutable block transition phases are incomplete")
        return cls(
            server_version=_string(value, "server_version"),
            world=_string(value, "world"),
            worldgen_provider=_string(value, "worldgen_provider"),
            worldgen_version=_string(value, "worldgen_version"),
            seed=_integer(value, "seed"),
            position=np.asarray(raw_position, dtype=np.int32),
            rows=rows,
        )

    def transition_sha256(self) -> str:
        """Hash portable values; exclude runtime IDs and wrapping counters."""

        payload = [
            {
                "phase": row.phase,
                "geometry": row.portable_geometry(),
                "block_health": row.block_health,
                "block_health_valid": row.block_health_valid,
                "seconds_since_damage": row.seconds_since_damage,
                "damage_age_valid": row.damage_age_valid,
            }
            for row in self.rows
        ]
        return _sha256(payload)


@dataclass(frozen=True)
class NativeMutableBlockCell:
    """One requested live cell, preserving unavailable versus exact air."""

    position: npt.NDArray[np.int32]
    available: bool
    row: NativeMutableBlockRow | None


@dataclass(frozen=True)
class NativeMutableBlockCells:
    """Bounded exact live-cell response used to refresh mutable JAX state."""

    server_version: str
    world: str
    worldgen_provider: str
    worldgen_version: str
    seed: int
    bridge_sha256: str
    cells: tuple[NativeMutableBlockCell, ...]

    @classmethod
    def from_response(
        cls,
        value: Mapping[str, Any],
    ) -> NativeMutableBlockCells:
        """Parse a complete live mutable-cell capture, failing closed."""

        if value.get("type") != "mutable_block_cells":
            raise ValueError("bridge returned the wrong mutable-cell type")
        if value.get("schema") != NATIVE_MUTABLE_BLOCK_CELLS_SCHEMA:
            raise ValueError("bridge returned the wrong mutable-cell schema")
        if value.get("version") != NATIVE_MUTABLE_BLOCK_CELLS_VERSION:
            raise ValueError("bridge returned the wrong mutable-cell version")
        if value.get("affordance_dictionary_sha256") != (
            block_affordance_dictionary_sha256()
        ):
            raise ValueError(
                "bridge mutable-cell affordance dictionary is stale"
            )
        bridge_sha256 = _string(value, "bridge_sha256")
        if (
            len(bridge_sha256) != 64
            or any(character not in "0123456789abcdefABCDEF"
                   for character in bridge_sha256)
        ):
            raise ValueError("mutable-cell bridge SHA-256 is invalid")
        raw_cells = value.get("cells")
        if (
            not isinstance(raw_cells, (list, tuple))
            or not 1 <= len(raw_cells) <= NATIVE_MUTABLE_BLOCK_CELL_CAPACITY
        ):
            raise ValueError("mutable-cell response exceeds its capacity")
        cells: list[NativeMutableBlockCell] = []
        positions: set[tuple[int, int, int]] = set()
        for raw_cell in raw_cells:
            cell = _mapping(raw_cell)
            position = _position(cell.get("position_i32_xyz"))
            key = tuple(int(item) for item in position)
            if key in positions:
                raise ValueError("mutable-cell response repeats a position")
            positions.add(key)
            available = _boolean(cell, "available")
            raw_row = cell.get("row")
            if available:
                row = NativeMutableBlockRow.from_response(_mapping(raw_row))
                if row.phase != "snapshot":
                    raise ValueError(
                        "mutable-cell rows must use the snapshot phase"
                    )
            else:
                if raw_row is not None:
                    raise ValueError(
                        "unavailable mutable cells must not publish a row"
                    )
                row = None
            cells.append(NativeMutableBlockCell(
                position=position,
                available=available,
                row=row,
            ))
        return cls(
            server_version=_string(value, "server_version"),
            world=_string(value, "world"),
            worldgen_provider=_string(value, "worldgen_provider"),
            worldgen_version=_string(value, "worldgen_version"),
            seed=_integer(value, "seed"),
            bridge_sha256=bridge_sha256.upper(),
            cells=tuple(cells),
        )


def native_mutable_block_evidence_request() -> dict[str, str]:
    """Return the bridge request for controlled mutable-block evidence."""

    return {"type": "mutable_block_evidence"}


def native_mutable_block_cells_request(
    positions: Sequence[Sequence[int]],
    *,
    expected_world_epoch: str | None = None,
) -> dict[str, object]:
    """Build one bounded exact-cell request for the live bridge."""

    if isinstance(positions, (str, bytes, bytearray)):
        raise TypeError("positions must be a sequence of XYZ triples")
    raw = list(positions)
    if not 1 <= len(raw) <= NATIVE_MUTABLE_BLOCK_CELL_CAPACITY:
        raise ValueError(
            "positions must contain 1.."
            f"{NATIVE_MUTABLE_BLOCK_CELL_CAPACITY} XYZ triples"
        )
    parsed = np.empty((len(raw), 3), dtype=np.int32)
    distinct: set[tuple[int, int, int]] = set()
    for index, value in enumerate(raw):
        if (
            isinstance(value, (str, bytes, bytearray))
            or not isinstance(value, Sequence)
            or len(value) != 3
        ):
            raise ValueError("each mutable-cell position must be XYZ")
        triple: list[int] = []
        for coordinate in value:
            if isinstance(coordinate, bool) or not isinstance(
                coordinate,
                (int, np.integer),
            ):
                raise TypeError(
                    "mutable-cell coordinates must be integers"
                )
            integer = int(coordinate)
            if not -(2**31) <= integer < 2**31:
                raise OverflowError(
                    "mutable-cell coordinates must fit signed int32"
                )
            triple.append(integer)
        key = tuple(triple)
        if key in distinct:
            raise ValueError("mutable-cell positions must be distinct")
        distinct.add(key)
        parsed[index] = triple
    request: dict[str, object] = {
        "type": "mutable_block_cells",
        "positions_i32_le_xyz": parsed.astype("<i4", copy=False).tobytes(),
    }
    if expected_world_epoch is not None:
        if not isinstance(expected_world_epoch, str):
            raise TypeError("expected_world_epoch must be a string")
        if (
            not expected_world_epoch.strip()
            or len(expected_world_epoch) > 128
        ):
            raise ValueError(
                "expected_world_epoch must contain 1..128 characters"
            )
        request["expected_world_epoch"] = expected_world_epoch
    return request


def native_mutable_block_evidence_contract() -> dict[str, object]:
    """Return the checkpoint-pinnable native evidence boundary."""

    return {
        "schema": NATIVE_MUTABLE_BLOCK_SCHEMA,
        "version": NATIVE_MUTABLE_BLOCK_VERSION,
        "server_version": "0.5.7",
        "phases": list(NATIVE_MUTABLE_BLOCK_PHASES),
        "mutation": "controlled_flat_fixture_reversible_in_finally",
        "controlled_sequence": {
            "damage": "BlockHealthChunk.damageBlock_1.0_minus_0.4_is_0.6",
            "direct_repair": (
                "BlockHealthChunk.repairBlock_0.6_plus_0.1_removes_"
                "the_sparse_entry_and_reads_as_full_1.0"
            ),
            "timed_repair": (
                "separate_BlockHealthSystem_path_not_exercised_by_"
                "this_endpoint"
            ),
        },
        "values": "exact_geometry_normalized_health_and_signed_counters",
        "semantic_key": (
            "sha256_words_big_endian_of_utf8_asset_id_nul_decimal_rotation"
        ),
        "affordances": {
            "dictionary_sha256": block_affordance_dictionary_sha256(),
            "tags": "stable_uint16_policy_bits",
            "gather_type": "stable_uint8_dictionary",
            "required_tool_quality": "native_int_nonnegative",
            "unknown": "reject_complete_response",
        },
        "counter_semantics": "signed_short_inequality_only",
        "delta_drain": "never_called",
        "jax_contract_sha256": mutable_block_contract_sha256(),
        "certification": (
            "native_runtime_comparison_requires_current_bridge_frozen_fixture"
        ),
    }


def native_mutable_block_evidence_contract_sha256() -> str:
    """Return the canonical native mutable-block contract digest."""

    return _sha256(native_mutable_block_evidence_contract())


def native_mutable_block_cells_contract() -> dict[str, object]:
    """Return the live exact-cell synchronization boundary."""

    return {
        "schema": NATIVE_MUTABLE_BLOCK_CELLS_SCHEMA,
        "version": NATIVE_MUTABLE_BLOCK_CELLS_VERSION,
        "server_version": "0.5.7",
        "request": {
            "positions": "little_endian_signed_int32_absolute_xyz",
            "capacity": NATIVE_MUTABLE_BLOCK_CELL_CAPACITY,
            "duplicates": "rejected",
            "expected_world_epoch": (
                "optional_exact_native_session_reservation_id"
            ),
        },
        "response": {
            "ordering": "request_order",
            "available": (
                "loaded_chunk_and_y_in_0_320;false_never_means_air"
            ),
            "available_row": "exact_NativeMutableBlockRow_snapshot",
            "unavailable_row": None,
            "bridge_identity": "required_sha256",
        },
        "physical_values": (
            "collision_fluid_movement_support_damage_and_wrapping_counters"
        ),
        "semantic_values": (
            "asset_id_rotation_semantic_sha256_and_affordance_dictionary"
        ),
        "counter_semantics": "signed_short_inequality_only",
        "provenance": "native_live_read_only_snapshot",
        "jax_contract_sha256": mutable_block_contract_sha256(),
    }


def native_mutable_block_cells_contract_sha256() -> str:
    """Return the canonical live-cell contract digest."""

    return _sha256(native_mutable_block_cells_contract())


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("mutable block row must be a map")
    return value


def _position(value: object) -> npt.NDArray[np.int32]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 3
        or any(
            isinstance(item, bool)
                or not isinstance(item, (int, np.integer))
            for item in value
        )
        or any(not -(2**31) <= int(item) < 2**31 for item in value)
    ):
        raise ValueError("mutable-cell position must be one int32 XYZ triple")
    return np.asarray(value, dtype=np.int32)


def _string(
    value: Mapping[str, Any],
    key: str,
    *,
    allow_empty: bool = False,
) -> str:
    result = value.get(key)
    if not isinstance(result, str) or (not allow_empty and not result):
        raise ValueError(f"{key} must be a string")
    return result


def _boolean(value: Mapping[str, Any], key: str) -> bool:
    result = value.get(key)
    if not isinstance(result, bool):
        raise ValueError(f"{key} must be boolean")
    return result


def _integer(value: Mapping[str, Any], key: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int):
        raise ValueError(f"{key} must be an integer")
    return result


def _signed_short(value: Mapping[str, Any], key: str) -> int:
    result = _integer(value, key)
    if not -(2**15) <= result < 2**15:
        raise ValueError(f"{key} must fit signed int16")
    return result


def _number(value: Mapping[str, Any], key: str) -> float:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise ValueError(f"{key} must be numeric")
    number = float(result)
    if not np.isfinite(number):
        raise ValueError(f"{key} must be finite")
    return number


def _bytes(value: Mapping[str, Any], key: str) -> bytes:
    result = value.get(key)
    if not isinstance(result, (bytes, bytearray)):
        raise ValueError(f"{key} must be binary")
    return bytes(result)


def _float64_payload(
    value: Mapping[str, Any],
    key: str,
    count: int,
) -> npt.NDArray[np.float64]:
    result = np.frombuffer(_bytes(value, key), dtype="<f8")
    if result.size != count or not np.all(np.isfinite(result)):
        raise ValueError(f"{key} has the wrong shape or non-finite values")
    return result.astype(np.float64, copy=True)


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


__all__ = [
    "NATIVE_MUTABLE_BLOCK_CELL_CAPACITY",
    "NATIVE_MUTABLE_BLOCK_CELLS_SCHEMA",
    "NATIVE_MUTABLE_BLOCK_CELLS_VERSION",
    "NATIVE_MUTABLE_BLOCK_PHASES",
    "NATIVE_MUTABLE_BLOCK_SCHEMA",
    "NATIVE_MUTABLE_BLOCK_VERSION",
    "NativeMutableBlockCell",
    "NativeMutableBlockCells",
    "NativeMutableBlockEvidence",
    "NativeMutableBlockRow",
    "native_mutable_block_cells_contract",
    "native_mutable_block_cells_contract_sha256",
    "native_mutable_block_cells_request",
    "native_mutable_block_evidence_contract",
    "native_mutable_block_evidence_contract_sha256",
    "native_mutable_block_evidence_request",
]
