"""Strict scalar contract for the native explosion-mutation fixture."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Literal, NamedTuple, cast

import numpy as np

from hytalegym.worldgen.block_affordances import (
    BLOCK_AFFORDANCE_TAGS,
    GATHER_TYPES,
    block_affordance_dictionary_sha256,
)
from hytalegym.worldgen.native_mutable_blocks import (
    NATIVE_MUTABLE_BLOCK_CELL_CAPACITY,
)


NATIVE_EXPLOSION_MUTATION_SCHEMA = "hytalerl_native_explosion_mutation_probe_v1"
NATIVE_EXPLOSION_MUTATION_VERSION = 1
NATIVE_EXPLOSION_MUTATION_CELL_CAPACITY = NATIVE_MUTABLE_BLOCK_CELL_CAPACITY
NATIVE_EXPLOSION_MUTATION_DROP_CAPACITY = 4_096

NativeExplosionFixtureKind = Literal["direct_drop"]

RESPONSE_TYPE = "explosion_mutation_probe"
CONTROLLED_SCOPE = "loaded_dry_direct_blocks_no_filler_no_support_cascade"
NATIVE_METHOD = "ExplosionUtils.performExplosion/processTargetBlocks"
CELL_SCALAR_COMPATIBILITY = "NativeMutableBlockEvidence.Row_scalar_fields_v1"
COUNTER_SEMANTICS = "signed_short_inequality_only"
GEOMETRY_REFRESH = "targeted_requery_via_mutable_block_cells"
CHANGED_CELL_ORDERING = "lexicographic_xyz_ascending"
RESOLVED_DROP_ORDERING = (
    "source_xyz_item_quantity_durability_metadata_spawn_xyz_ascending"
)
SYNTHETIC_CONFIG_SOURCE_ID = "bridge_synthetic_dry_block_explosion_v1"
FIXTURE_BLOCK_ASSET_IDS: dict[NativeExplosionFixtureKind, str] = {
    "direct_drop": "Recipe_Book_Magic_Air",
}
BLOCK_DAMAGE_RADIUS = 2
BLOCK_DAMAGE_FALLOFF = 1.0
BLOCK_DROP_CHANCE = 1.0


@dataclass(frozen=True, slots=True)
class NativeExplosionMutationRequest:
    """One narrow server-owned, entity-disabled v1 fixture request."""

    world_epoch: str
    fixture_kind: NativeExplosionFixtureKind
    cell_capacity: int = NATIVE_EXPLOSION_MUTATION_CELL_CAPACITY
    drop_capacity: int = NATIVE_EXPLOSION_MUTATION_DROP_CAPACITY

    def __post_init__(self) -> None:
        epoch = _text(self.world_epoch, "world_epoch", 128)
        if self.fixture_kind not in FIXTURE_BLOCK_ASSET_IDS:
            raise ValueError("fixture_kind must be direct_drop")
        cell_capacity = _bounded(
            self.cell_capacity,
            1,
            NATIVE_EXPLOSION_MUTATION_CELL_CAPACITY,
            "cell_capacity",
        )
        drop_capacity = _bounded(
            self.drop_capacity,
            1,
            NATIVE_EXPLOSION_MUTATION_DROP_CAPACITY,
            "drop_capacity",
        )
        object.__setattr__(self, "world_epoch", epoch)
        object.__setattr__(self, "cell_capacity", cell_capacity)
        object.__setattr__(self, "drop_capacity", drop_capacity)

    def to_message(self) -> dict[str, object]:
        """Build the bounded request; all mutation scalars stay server-owned."""

        return {
            "type": RESPONSE_TYPE,
            "schema": NATIVE_EXPLOSION_MUTATION_SCHEMA,
            "version": NATIVE_EXPLOSION_MUTATION_VERSION,
            "world_epoch": self.world_epoch,
            "fixture_kind": self.fixture_kind,
            "cell_capacity": self.cell_capacity,
            "drop_capacity": self.drop_capacity,
        }


class NativeExplosionMutationCellState(NamedTuple):
    block_present: bool
    block_asset_id: str
    runtime_block_id: int
    semantic_key_sha256: bytes
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
    block_health: float
    block_health_valid: bool
    seconds_since_damage: float
    damage_age_valid: bool
    local_change_counter: int
    global_change_counter: int

    @property
    def dry(self) -> bool:
        return not (self.fluid_level or self.fluid_fill_height or self.fluid_damage)

    def substantive_value(self) -> tuple[object, ...]:
        return self[:18]  # Age and wrapping counters are not block deltas.


class NativeExplosionMutationChangedCell(NamedTuple):
    ordinal: int
    position: tuple[int, int, int]
    before: NativeExplosionMutationCellState
    after: NativeExplosionMutationCellState

    @property
    def drop_eligible(self) -> bool:
        return self.before.block_present and (
            not self.after.block_present
            or self.before.runtime_block_id != self.after.runtime_block_id
            or self.before.block_asset_id != self.after.block_asset_id
            or self.before.semantic_key_sha256 != self.after.semantic_key_sha256
        )


class NativeExplosionMutationDrop(NamedTuple):
    ordinal: int
    source_position: tuple[int, int, int]
    item_asset_id: str
    quantity: int
    durability: float
    max_durability: float
    metadata_json_sha256: str
    spawn_position: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class NativeExplosionMutationCapture:
    """One identity-checked complete or empty native mutation frame."""

    bridge_sha256: str
    server_version: str
    world: str
    worldgen_provider: str
    worldgen_version: str
    seed: int
    request: NativeExplosionMutationRequest
    fixture_block_asset_id: str
    explosion_config_semantic_sha256: str
    origin: tuple[float, float, float]
    execution_started: bool
    complete: bool
    resync_required: bool
    failure_reason: str
    changed_cells: tuple[NativeExplosionMutationChangedCell, ...]
    resolved_drops: tuple[NativeExplosionMutationDrop, ...]

    @classmethod
    def from_response(
        cls,
        response: Mapping[str, Any],
        *,
        expected_request: NativeExplosionMutationRequest,
        expected_bridge_sha256: str,
    ) -> NativeExplosionMutationCapture:
        if not isinstance(expected_request, NativeExplosionMutationRequest):
            raise TypeError("expected_request must be a mutation request")
        value = dict(response)
        fixed = {
            "type": RESPONSE_TYPE,
            "schema": NATIVE_EXPLOSION_MUTATION_SCHEMA,
            "version": NATIVE_EXPLOSION_MUTATION_VERSION,
            "fixture_kind": expected_request.fixture_kind,
            "fixture_block_asset_id": FIXTURE_BLOCK_ASSET_IDS[
                expected_request.fixture_kind
            ],
            "explosion_config_source_id": SYNTHETIC_CONFIG_SOURCE_ID,
            "scope": CONTROLLED_SCOPE,
            "native_method": NATIVE_METHOD,
            "cell_scalar_compatibility": CELL_SCALAR_COMPATIBILITY,
            "counter_semantics": COUNTER_SEMANTICS,
            "geometry_refresh": GEOMETRY_REFRESH,
            "changed_cell_ordering": CHANGED_CELL_ORDERING,
            "resolved_drop_ordering": RESOLVED_DROP_ORDERING,
            "damage_blocks": True,
            "damage_entities": False,
            "block_damage_radius": BLOCK_DAMAGE_RADIUS,
            "block_damage_falloff": BLOCK_DAMAGE_FALLOFF,
            "block_drop_chance": BLOCK_DROP_CHANCE,
            "entity_damage_radius": 0.0,
            "entity_damage": 0.0,
            "entity_damage_falloff": 0.0,
            "ignore_controlled_actor": False,
            "entity_capacity": 1,
            "affordance_dictionary_sha256": (
                block_affordance_dictionary_sha256()
            ),
        }
        for key, expected in fixed.items():
            _equal(value, key, expected)

        bridge = _sha256(value.get("bridge_sha256"), "bridge_sha256").upper()
        if bridge != _sha256(expected_bridge_sha256, "expected bridge").upper():
            raise ValueError("explosion mutation came from another bridge")
        _equal(value, "world_epoch", expected_request.world_epoch)
        _equal(value, "cell_capacity", expected_request.cell_capacity)
        _equal(value, "drop_capacity", expected_request.drop_capacity)
        config_sha = _sha256(
            value.get("explosion_config_semantic_sha256"),
            "explosion_config_semantic_sha256",
        ).lower()
        expected_config_sha = synthetic_config_semantic_sha256(
            expected_request.fixture_kind
        )
        if config_sha != expected_config_sha:
            raise ValueError("server-owned explosion configuration is not canonical")
        origin = (
            _number(value, "origin_x"),
            _number(value, "origin_y"),
            _number(value, "origin_z"),
        )

        totals = tuple(
            _integer(value, key, 0)
            for key in (
                "total_entity_admissions",
                "total_changed_cells",
                "total_resolved_drops",
            )
        )
        overflows = tuple(
            _boolean(value, key)
            for key in ("entity_overflow", "cell_overflow", "drop_overflow")
        )
        capacities = (1, expected_request.cell_capacity, expected_request.drop_capacity)
        if any(
            flag != (total > cap)
            for flag, total, cap in zip(overflows, totals, capacities, strict=True)
        ):
            raise ValueError("overflow flags must exactly reflect totals")
        if totals[0] or overflows[0] or _rows(value, "entity_admissions"):
            raise ValueError("entity-disabled v1 cannot publish entity admissions")

        unsupported = tuple(
            _boolean(value, key)
            for key in (
                "unsupported_fluid_mutation",
                "unsupported_filler_mutation",
                "unsupported_support_cascade",
            )
        )
        if _boolean(value, "dry_controlled_scope") != (not any(unsupported)):
            raise ValueError("dry_controlled_scope contradicts unsupported flags")
        execution_started = _boolean(value, "execution_started")
        complete = _boolean(value, "complete")
        resync = _boolean(value, "resync_required")
        failure = _string(value, "failure_reason", allow_empty=True)
        raw_cells = _rows(value, "changed_cells")
        raw_drops = _rows(value, "resolved_drops")
        if complete:
            if (
                not execution_started
                or any(unsupported)
                or any(overflows)
                or resync
                or failure
            ):
                raise ValueError("complete mutation frame is unsupported or incomplete")
            if (len(raw_cells), len(raw_drops)) != totals[1:]:
                raise ValueError("complete mutation frames must publish every row")
        elif (
            not failure
            or raw_cells
            or raw_drops
            or resync != execution_started
        ):
            raise ValueError(
                "incomplete frames are empty; only executed frames require resync"
            )

        cells = tuple(_changed_cell(row) for row in raw_cells)
        _validate_changed_cell_order(cells)
        drops = tuple(_drop(row) for row in raw_drops)
        _validate_drop_order(drops, cells)
        if complete:
            _validate_fixture_outcome(expected_request.fixture_kind, origin, cells, drops)
        return cls(
            bridge_sha256=bridge,
            server_version=_string(value, "server_version"),
            world=_string(value, "world"),
            worldgen_provider=_string(value, "worldgen_provider"),
            worldgen_version=_string(value, "worldgen_version"),
            seed=_integer(value, "seed", -(2**63), 2**63 - 1),
            request=expected_request,
            fixture_block_asset_id=FIXTURE_BLOCK_ASSET_IDS[
                expected_request.fixture_kind
            ],
            explosion_config_semantic_sha256=config_sha,
            origin=origin,
            execution_started=execution_started,
            complete=complete,
            resync_required=resync,
            failure_reason=failure,
            changed_cells=cells,
            resolved_drops=drops,
        )


def synthetic_config_semantics(fixture_kind: NativeExplosionFixtureKind) -> str:
    """Canonical Java/Python semantic payload for one server fixture."""

    if fixture_kind not in FIXTURE_BLOCK_ASSET_IDS:
        raise ValueError("fixture_kind must be direct_drop")
    return "\n".join(
        (
            f"source_id={SYNTHETIC_CONFIG_SOURCE_ID}",
            f"fixture_kind={fixture_kind}",
            f"fixture_block_asset_id={FIXTURE_BLOCK_ASSET_IDS[fixture_kind]}",
            "damage_blocks=true",
            "damage_entities=false",
            "block_damage_radius=2",
            "block_damage_falloff=0x1.0p0",
            "block_drop_chance=0x1.0p0",
            "item_tool=null",
            "knockback=null",
            "particles=null",
            "sound=null",
        )
    )


def synthetic_config_semantic_sha256(
    fixture_kind: NativeExplosionFixtureKind,
) -> str:
    return hashlib.sha256(synthetic_config_semantics(fixture_kind).encode()).hexdigest()


def _validate_fixture_outcome(
    fixture_kind: NativeExplosionFixtureKind,
    origin: tuple[float, float, float],
    cells: tuple[NativeExplosionMutationChangedCell, ...],
    drops: tuple[NativeExplosionMutationDrop, ...],
) -> None:
    if len(cells) != 1:
        raise ValueError("complete v1 fixture must change exactly one target cell")
    cell = cells[0]
    if cell.position != _target_position(origin):
        raise ValueError("fixture target must be exactly one cell +X from origin")
    expected_asset = FIXTURE_BLOCK_ASSET_IDS[fixture_kind]
    if (
        not cell.before.block_present
        or cell.before.block_asset_id != expected_asset
        or cell.after.block_present
    ):
        raise ValueError("fixture must destroy its declared target block")
    if (
        len(drops) != 1
        or drops[0].source_position != cell.position
        or drops[0].item_asset_id != expected_asset
        or drops[0].quantity != 1
    ):
        raise ValueError("direct_drop must resolve one fixture-item stack")


def _target_position(origin: tuple[float, float, float]) -> tuple[int, int, int]:
    target_center = (origin[0] + 1.0, origin[1], origin[2])
    result: list[int] = []
    for value in target_center:
        coordinate = value - 0.5
        rounded = round(coordinate)
        if not math.isclose(coordinate, rounded, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("server-selected fixture origin must be cell-centred")
        if not -(2**31) <= rounded < 2**31:
            raise OverflowError("fixture target must fit signed int32")
        result.append(rounded)
    return cast(tuple[int, int, int], tuple(result))


def _validate_changed_cell_order(
    cells: tuple[NativeExplosionMutationChangedCell, ...],
) -> None:
    if any(cell.ordinal != index for index, cell in enumerate(cells)):
        raise ValueError("changed cells must preserve native ordinal order")
    positions = tuple(cell.position for cell in cells)
    if positions != tuple(sorted(positions)) or len(set(positions)) != len(positions):
        raise ValueError("changed cells must be unique canonical XYZ rows")


def _validate_drop_order(
    drops: tuple[NativeExplosionMutationDrop, ...],
    cells: tuple[NativeExplosionMutationChangedCell, ...],
) -> None:
    eligible = {cell.position for cell in cells if cell.drop_eligible}
    if any(drop.ordinal != index for index, drop in enumerate(drops)):
        raise ValueError("resolved drops must preserve native ordinal order")
    keys = tuple(
        (
            *drop.source_position,
            drop.item_asset_id,
            drop.quantity,
            drop.durability,
            drop.max_durability,
            drop.metadata_json_sha256,
            *drop.spawn_position,
        )
        for drop in drops
    )
    if keys != tuple(sorted(keys)) or any(
        drop.source_position not in eligible for drop in drops
    ):
        raise ValueError("resolved drops must be canonical and source-valid")


def _cell_state(value: Mapping[str, Any]) -> NativeExplosionMutationCellState:
    present = _boolean(value, "block_present")
    key_valid = _boolean(value, "semantic_key_valid")
    affordance_valid = _boolean(value, "affordance_valid")
    health_valid = _boolean(value, "block_health_valid")
    key = _bytes(value, "semantic_key_sha256")
    asset = _string(value, "block_asset_id", allow_empty=True)
    runtime = _integer(value, "runtime_block_id", 0)
    tags = _integer(value, "affordance_tags", 0)
    gather = _integer(value, "gather_type_index", 0)
    quality = _integer(value, "required_tool_quality", 0)
    health = _number(value, "block_health")
    age = _number(value, "seconds_since_damage")
    age_valid = _boolean(value, "damage_age_valid")
    fill = _number(value, "fluid_fill_height")
    if not (present == key_valid == affordance_valid == health_valid):
        raise ValueError("cell validity must follow presence")
    known_tags = (1 << len(BLOCK_AFFORDANCE_TAGS)) - 1
    if (
        len(key) != (32 if key_valid else 0)
        or tags > 0xFFFF
        or tags & ~known_tags
        or gather >= len(GATHER_TYPES)
        or quality > 32_767
        or (not affordance_valid and (tags or gather or quality))
    ):
        raise ValueError("cell identity or affordance is outside its dictionary")
    if (
        not 0.0 <= health <= 1.0
        or (not present and health != 0.0)
        or age < 0.0
        or (not age_valid and age != 0.0)
        or not 0.0 <= fill <= 1.0
    ):
        raise ValueError("cell health, age, or fluid fill is outside its domain")
    if (present and not asset) or (not present and (asset or runtime or key)):
        raise ValueError("cell block identity is not canonical")
    return NativeExplosionMutationCellState(
        present,
        asset,
        runtime,
        key,
        key_valid,
        affordance_valid,
        tags,
        gather,
        quality,
        _integer(value, "rotation_index"),
        _integer(value, "flags"),
        _integer(value, "fluid_level", 0),
        fill,
        _integer(value, "support"),
        _integer(value, "block_damage"),
        _integer(value, "fluid_damage"),
        health,
        health_valid,
        age,
        age_valid,
        _integer(value, "local_change_counter", -(2**15), 2**15 - 1),
        _integer(value, "global_change_counter", -(2**15), 2**15 - 1),
    )


def _changed_cell(value: Mapping[str, Any]) -> NativeExplosionMutationChangedCell:
    before = _cell_state(_mapping(value, "before"))
    after = _cell_state(_mapping(value, "after"))
    if before.substantive_value() == after.substantive_value():
        raise ValueError("changed cells require a substantive state delta")
    if not before.dry or not after.dry:
        raise ValueError("complete v1 mutation cells must remain dry")
    partial = after.block_health_valid and 0.0 < after.block_health < 1.0
    if _boolean(value, "partial_block_health") != partial:
        raise ValueError("partial_block_health contradicts the after state")
    return NativeExplosionMutationChangedCell(
        _integer(value, "ordinal", 0),
        _position(value, "position_i32_xyz"),
        before,
        after,
    )


def _drop(value: Mapping[str, Any]) -> NativeExplosionMutationDrop:
    durability = _number(value, "durability")
    maximum = _number(value, "max_durability")
    if durability < 0.0 or maximum < 0.0 or durability > maximum:
        raise ValueError("drop durability is outside its domain")
    metadata = _string(value, "metadata_json_sha256", allow_empty=True)
    if metadata:
        metadata = _sha256(metadata, "metadata_json_sha256").lower()
    return NativeExplosionMutationDrop(
        _integer(value, "ordinal", 0),
        _position(value, "source_position_i32_xyz"),
        _string(value, "item_asset_id"),
        _integer(value, "quantity", 1),
        durability,
        maximum,
        metadata,
        _float_tuple(value.get("spawn_position_f64_xyz"), "spawn position"),
    )


def _equal(value: Mapping[str, Any], key: str, expected: object) -> None:
    if isinstance(expected, bool):
        actual = _boolean(value, key)
    elif isinstance(expected, int):
        actual = _integer(value, key)
    elif isinstance(expected, float):
        actual = _number(value, key)
    else:
        actual = _string(value, key)
    if actual != expected:
        raise ValueError(f"{key} must equal {expected!r}")


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise TypeError(f"{key} must be a map")
    return result


def _rows(value: Mapping[str, Any], key: str) -> tuple[Mapping[str, Any], ...]:
    result = value.get(key)
    if not isinstance(result, (list, tuple)) or not all(
        isinstance(row, Mapping) for row in result
    ):
        raise TypeError(f"{key} must be an array of maps")
    return tuple(result)


def _position(value: Mapping[str, Any], key: str) -> tuple[int, int, int]:
    raw = value.get(key)
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise TypeError(f"{key} must be one XYZ row")
    result = tuple(_raw_int(item, key) for item in raw)
    if not all(-(2**31) <= item < 2**31 for item in result):
        raise OverflowError(f"{key} must fit signed int32")
    return cast(tuple[int, int, int], result)


def _float_tuple(value: object, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise TypeError(f"{label} must contain three values")
    result = tuple(_raw_float(item, label) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must be finite")
    return cast(tuple[float, float, float], result)


def _string(value: Mapping[str, Any], key: str, *, allow_empty: bool = False) -> str:
    return _text(value.get(key), key, 512, allow_empty=allow_empty)


def _text(
    value: object,
    label: str,
    maximum: int,
    *,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    result = value.strip()
    if (not allow_empty and not result) or len(result) > maximum:
        raise ValueError(f"{label} is empty or too long")
    return result


def _boolean(value: Mapping[str, Any], key: str) -> bool:
    result = value.get(key)
    if not isinstance(result, bool):
        raise TypeError(f"{key} must be boolean")
    return result


def _integer(
    value: Mapping[str, Any],
    key: str,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    result = _raw_int(value.get(key), key)
    if (minimum is not None and result < minimum) or (
        maximum is not None and result > maximum
    ):
        raise ValueError(f"{key} is outside its domain")
    return result


def _raw_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer")
    return int(value)


def _bounded(value: object, low: int, high: int, label: str) -> int:
    result = _raw_int(value, label)
    if not low <= result <= high:
        raise ValueError(f"{label} must be in [{low},{high}]")
    return result


def _number(value: Mapping[str, Any], key: str) -> float:
    result = _raw_float(value.get(key), key)
    if not math.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def _raw_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{label} must be numeric")
    return float(value)


def _bytes(value: Mapping[str, Any], key: str) -> bytes:
    result = value.get(key)
    if not isinstance(result, (bytes, bytearray, memoryview)):
        raise TypeError(f"{key} must be binary")
    return bytes(result)


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{label} must be a SHA-256")
    return value


__all__ = [
    "NATIVE_EXPLOSION_MUTATION_CELL_CAPACITY",
    "NATIVE_EXPLOSION_MUTATION_DROP_CAPACITY",
    "NATIVE_EXPLOSION_MUTATION_SCHEMA",
    "NATIVE_EXPLOSION_MUTATION_VERSION",
    "NativeExplosionFixtureKind",
    "NativeExplosionMutationCapture",
    "NativeExplosionMutationCellState",
    "NativeExplosionMutationChangedCell",
    "NativeExplosionMutationDrop",
    "NativeExplosionMutationRequest",
    "synthetic_config_semantic_sha256",
    "synthetic_config_semantics",
]
