"""Fixed-capacity prefab entity descriptors with native placement transforms."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math

import numpy as np

from hytalegym.worldgen.region import CAPTURE_BLOCKS_PER_AXIS, MIN_Y, WORLD_HEIGHT
from hytalegym.worldgen.surrogate.assets import HytaleAssetError
from hytalegym.worldgen.surrogate.contract import SurrogateWorldCapacity
from hytalegym.worldgen.surrogate.prefab import place_prefab_entity_transform
from hytalegym.worldgen.surrogate.structures import GeneratedStructurePlan

ENTITY_KIND_SPAWN_MARKER = 1
ENTITY_KIND_NPC = 2
ENTITY_KIND_TRIGGER_VOLUME = 3
ENTITY_KIND_PATROL_MARKER = 4
ENTITY_KIND_SPAWN_SUPPRESSION = 5
ENTITY_KIND_PREFAB = 6

_KIND_CODE = {
    "spawn_marker": ENTITY_KIND_SPAWN_MARKER,
    "npc": ENTITY_KIND_NPC,
    "trigger_volume": ENTITY_KIND_TRIGGER_VOLUME,
    "patrol_marker": ENTITY_KIND_PATROL_MARKER,
    "spawn_suppression": ENTITY_KIND_SPAWN_SUPPRESSION,
    "prefab_entity": ENTITY_KIND_PREFAB,
}


@dataclass(frozen=True)
class EntityGeometryProfile:
    """Explicit local AABB and native target-ray semantics for one entity type."""

    type_id: str
    local_bounds: tuple[float, float, float, float, float, float]
    collidable: bool
    blocks_los: bool
    provenance_sha256: str
    line_of_sight_offset: tuple[float, float, float] | None = None
    native_certified: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.type_id, str) or not self.type_id:
            raise TypeError("entity geometry type_id must be a non-empty string")
        bounds = tuple(float(value) for value in self.local_bounds)
        if len(bounds) != 6 or not all(math.isfinite(value) for value in bounds):
            raise ValueError("entity local_bounds must contain six finite values")
        if any(bounds[axis] >= bounds[axis + 3] for axis in range(3)):
            raise ValueError("entity local_bounds must have positive extent")
        offset = (
            None
            if self.line_of_sight_offset is None
            else tuple(float(value) for value in self.line_of_sight_offset)
        )
        if offset is not None and (
            len(offset) != 3 or not all(math.isfinite(value) for value in offset)
        ):
            raise ValueError("line_of_sight_offset requires three finite values")
        if not isinstance(self.collidable, bool) or not isinstance(
            self.blocks_los,
            bool,
        ):
            raise TypeError("entity geometry flags must be bool")
        if (
            not isinstance(self.provenance_sha256, str)
            or len(self.provenance_sha256) != 64
            or any(value not in "0123456789abcdef" for value in self.provenance_sha256)
        ):
            raise ValueError("entity geometry provenance must be lowercase SHA-256")
        if not isinstance(self.native_certified, bool):
            raise TypeError("native_certified must be bool")
        object.__setattr__(self, "local_bounds", bounds)
        object.__setattr__(self, "line_of_sight_offset", offset)


@dataclass(frozen=True)
class CompiledEntitySpawns:
    """Padded immutable descriptors; behavior is intentionally external."""

    tile_min_xz: tuple[int, int]
    entity_mask: np.ndarray
    identity_words: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    velocity: np.ndarray
    kind: np.ndarray
    type_code: np.ndarray
    type_identity_words: np.ndarray
    geometry_supported: np.ndarray
    collidable: np.ndarray
    blocks_los: np.ndarray
    local_bounds: np.ndarray
    line_of_sight_offset_supported: np.ndarray
    line_of_sight_offset: np.ndarray
    source_code: np.ndarray
    type_ids: tuple[str, ...]
    geometry_profiles: tuple[EntityGeometryProfile, ...]
    source_sha256: tuple[str, ...]
    component_sha256: tuple[str, ...]
    clipped_entity_count: int

    @property
    def entity_count(self) -> int:
        return int(np.count_nonzero(self.entity_mask))

    @property
    def logical_bytes(self) -> int:
        return sum(
            value.nbytes
            for value in (
                self.entity_mask,
                self.identity_words,
                self.position,
                self.rotation,
                self.velocity,
                self.kind,
                self.type_code,
                self.type_identity_words,
                self.geometry_supported,
                self.collidable,
                self.blocks_los,
                self.local_bounds,
                self.line_of_sight_offset_supported,
                self.line_of_sight_offset,
                self.source_code,
            )
        )


def compile_entity_spawns(
    plan: GeneratedStructurePlan,
    *,
    geometry_profiles: Mapping[str, EntityGeometryProfile] | None = None,
    capacity: SurrogateWorldCapacity | None = None,
) -> CompiledEntitySpawns:
    """Compile entities enabled by assignment ``LoadEntities`` exactly once."""

    if not isinstance(plan, GeneratedStructurePlan):
        raise TypeError("plan must be a GeneratedStructurePlan")
    layout = SurrogateWorldCapacity() if capacity is None else capacity
    if not isinstance(layout, SurrogateWorldCapacity):
        raise TypeError("capacity must be a SurrogateWorldCapacity")
    profiles = {} if geometry_profiles is None else dict(geometry_profiles)
    if any(
        not isinstance(key, str)
        or not isinstance(value, EntityGeometryProfile)
        or key != value.type_id
        for key, value in profiles.items()
    ):
        raise TypeError(
            "geometry_profiles must map exact type IDs to EntityGeometryProfile"
        )

    records: list[
        tuple[
            tuple[int, int],
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
            int,
            str,
            str,
            str,
        ]
    ] = []
    clipped = 0
    identities: set[tuple[int, int]] = set()
    for instance in plan.instances:
        if not instance.load_entities:
            continue
        template = instance.placement.template
        for entity in template.entities:
            if entity.unsupported_wrapper_fields:
                raise HytaleAssetError(
                    "prefab entity contains unsupported wrapper fields: "
                    + ", ".join(entity.unsupported_wrapper_fields)
                )
            position, rotation = place_prefab_entity_transform(
                entity,
                template.anchor,
                instance.anchor_world,
                instance.quarter_turns,
            )
            block_x = math.floor(position[0])
            block_y = math.floor(position[1])
            block_z = math.floor(position[2])
            inside = (
                plan.tile_min_xz[0]
                <= block_x
                < plan.tile_min_xz[0] + CAPTURE_BLOCKS_PER_AXIS
                and MIN_Y <= block_y < MIN_Y + WORLD_HEIGHT
                and plan.tile_min_xz[1]
                <= block_z
                < plan.tile_min_xz[1] + CAPTURE_BLOCKS_PER_AXIS
            )
            if not inside:
                clipped += 1
                continue
            identity = _identity_words(
                template.provenance.content_sha256,
                entity.components_sha256,
                entity.index,
                instance.anchor_world,
                instance.quarter_turns,
            )
            if identity in identities:
                raise HytaleAssetError("prefab entity identity collision")
            identities.add(identity)
            records.append(
                (
                    identity,
                    position,
                    rotation,
                    entity.velocity,
                    _KIND_CODE[entity.kind],
                    entity.type_id,
                    template.provenance.content_sha256,
                    entity.components_sha256,
                )
            )
            if len(records) > layout.entity_capacity:
                raise HytaleAssetError(
                    f"prefab entities exceed entity capacity {layout.entity_capacity}"
                )
    records.sort(key=lambda value: value[0])
    type_ids = tuple(sorted({value[5] for value in records}))
    sources = tuple(sorted({value[6] for value in records}))
    type_index = {value: index for index, value in enumerate(type_ids)}
    source_index = {value: index for index, value in enumerate(sources)}

    size = layout.entity_capacity
    mask = np.zeros(size, dtype=np.bool_)
    identity_words = np.zeros((size, 2), dtype=np.uint32)
    position = np.zeros((size, 3), dtype=np.float32)
    rotation = np.zeros((size, 3), dtype=np.float32)
    velocity = np.zeros((size, 3), dtype=np.float32)
    kind = np.zeros(size, dtype=np.uint8)
    type_code = np.zeros(size, dtype=np.uint16)
    type_identity_words = np.zeros((size, 2), dtype=np.uint32)
    geometry_supported = np.zeros(size, dtype=np.bool_)
    collidable = np.zeros(size, dtype=np.bool_)
    blocks_los = np.zeros(size, dtype=np.bool_)
    local_bounds = np.zeros((size, 6), dtype=np.float32)
    line_of_sight_offset_supported = np.zeros(size, dtype=np.bool_)
    line_of_sight_offset = np.zeros((size, 3), dtype=np.float32)
    source_code = np.zeros(size, dtype=np.uint16)
    component_hashes: list[str] = []
    for index, record in enumerate(records):
        (
            identity,
            placed_position,
            placed_rotation,
            placed_velocity,
            kind_code,
            type_id,
            source_hash,
            component_hash,
        ) = record
        mask[index] = True
        identity_words[index] = identity
        position[index] = placed_position
        rotation[index] = placed_rotation
        velocity[index] = placed_velocity
        kind[index] = kind_code
        type_code[index] = type_index[type_id]
        type_identity_words[index] = entity_type_identity_words(type_id)
        profile = profiles.get(type_id)
        if profile is not None:
            geometry_supported[index] = True
            collidable[index] = profile.collidable
            blocks_los[index] = profile.blocks_los
            local_bounds[index] = profile.local_bounds
            if profile.line_of_sight_offset is not None:
                line_of_sight_offset_supported[index] = True
                line_of_sight_offset[index] = profile.line_of_sight_offset
        source_code[index] = source_index[source_hash]
        component_hashes.append(component_hash)
    for value in (
        mask,
        identity_words,
        position,
        rotation,
        velocity,
        kind,
        type_code,
        type_identity_words,
        geometry_supported,
        collidable,
        blocks_los,
        local_bounds,
        line_of_sight_offset_supported,
        line_of_sight_offset,
        source_code,
    ):
        value.flags.writeable = False
    return CompiledEntitySpawns(
        tile_min_xz=plan.tile_min_xz,
        entity_mask=mask,
        identity_words=identity_words,
        position=position,
        rotation=rotation,
        velocity=velocity,
        kind=kind,
        type_code=type_code,
        type_identity_words=type_identity_words,
        geometry_supported=geometry_supported,
        collidable=collidable,
        blocks_los=blocks_los,
        local_bounds=local_bounds,
        line_of_sight_offset_supported=line_of_sight_offset_supported,
        line_of_sight_offset=line_of_sight_offset,
        source_code=source_code,
        type_ids=type_ids,
        geometry_profiles=tuple(
            profiles[type_id] for type_id in sorted(set(type_ids) & profiles.keys())
        ),
        source_sha256=sources,
        component_sha256=tuple(component_hashes),
        clipped_entity_count=clipped,
    )


def entity_type_identity_words(type_id: str) -> tuple[int, int]:
    """Return the stable 64-bit semantic identity used beside atlas-local codes."""

    if not isinstance(type_id, str) or not type_id:
        raise TypeError("entity type_id must be a non-empty string")
    digest = hashlib.sha256(f"hytalerl_entity_type_v1:{type_id}".encode()).digest()
    return (
        int.from_bytes(digest[:4], "little"),
        int.from_bytes(digest[4:8], "little"),
    )


def _identity_words(
    source_sha256: str,
    component_sha256: str,
    entity_index: int,
    anchor: tuple[int, int, int],
    quarter_turns: int,
) -> tuple[int, int]:
    digest = hashlib.sha256(
        (
            f"{source_sha256}:{component_sha256}:{entity_index}:"
            f"{anchor[0]}:{anchor[1]}:{anchor[2]}:{quarter_turns}"
        ).encode()
    ).digest()
    return (
        int.from_bytes(digest[:4], "little"),
        int.from_bytes(digest[4:8], "little"),
    )


__all__ = [
    "ENTITY_KIND_NPC",
    "ENTITY_KIND_PATROL_MARKER",
    "ENTITY_KIND_PREFAB",
    "ENTITY_KIND_SPAWN_MARKER",
    "ENTITY_KIND_SPAWN_SUPPRESSION",
    "ENTITY_KIND_TRIGGER_VOLUME",
    "CompiledEntitySpawns",
    "EntityGeometryProfile",
    "compile_entity_spawns",
    "entity_type_identity_words",
]
