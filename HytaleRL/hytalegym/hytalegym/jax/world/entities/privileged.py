"""Entity-indexed privileged scene over authoritative world-owned state."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.framework import (
    AuthoritativeSnapshot,
    ContentDigest,
    PrivilegedSceneEnvelope,
    ProviderContract,
    SchemaBundle,
    SchemaRef,
)
from hytalegym.jax.world.surrogate.types import SurrogateRuntimeState


Array = jax.Array
PRIVILEGED_ENTITY_SCHEMA = "hytalerl_privileged_entity_scene_v1"
PRIVILEGED_ENTITY_VERSION = 1
PRIVILEGED_ENTITY_CAPACITY = 256

PRIVILEGED_ENTITY_PROVENANCE_NATIVE_RUNTIME = 1
PRIVILEGED_ENTITY_PROVENANCE_SURROGATE_RUNTIME = 2
PRIVILEGED_ENTITY_COMPONENT_PROVENANCE_NONE = 0
PRIVILEGED_ENTITY_COMPONENT_PROVENANCE_NATIVE = 1
PRIVILEGED_ENTITY_COMPONENT_PROVENANCE_SURROGATE = 2

PRIVILEGED_ENTITY_DIAGNOSTIC_SOURCE_UNAVAILABLE = jnp.uint32(1)
PRIVILEGED_ENTITY_DIAGNOSTIC_SOURCE_OVERFLOW = jnp.uint32(1 << 1)
PRIVILEGED_ENTITY_DIAGNOSTIC_OUTPUT_CAPACITY = jnp.uint32(1 << 2)
PRIVILEGED_ENTITY_DIAGNOSTIC_IDENTITY = jnp.uint32(1 << 3)
PRIVILEGED_ENTITY_DIAGNOSTIC_NONFINITE = jnp.uint32(1 << 4)
PRIVILEGED_ENTITY_DIAGNOSTIC_PROFILE_INVALID = jnp.uint32(1 << 5)
PRIVILEGED_ENTITY_DIAGNOSTIC_PROFILE_MISSING = jnp.uint32(1 << 6)
PRIVILEGED_ENTITY_DIAGNOSTIC_DYNAMICS_UNSUPPORTED = jnp.uint32(1 << 7)
PRIVILEGED_ENTITY_DIAGNOSTIC_PROVENANCE = jnp.uint32(1 << 8)
PRIVILEGED_ENTITY_DIAGNOSTIC_LIFECYCLE_INVALID = jnp.uint32(1 << 9)


class PrivilegedEntityScene(NamedTuple):
    """Training-only neutral scene; no entity is assigned a combat role."""

    available: Array
    physical_available: Array
    dynamics_supported: Array
    capacity_exceeded: Array
    diagnostics: Array
    source_provenance: Array
    entity_count: Array
    entity_mask: Array
    physical_mask: Array
    source_slot: Array
    active: Array
    identity_words: Array
    type_identity_words: Array
    kind: Array
    position: Array
    rotation: Array
    velocity: Array
    behavior_supported: Array
    collidable: Array
    blocks_los: Array
    local_bounds: Array
    los_offset_supported: Array
    los_offset: Array
    behavior_provenance: Array
    geometry_provenance: Array


class PrivilegedEntitySnapshot(NamedTuple):
    """Neutral fixed-array entity state copied into one authoritative tick."""

    source_available: Array
    source_overflow: Array
    source_provenance: int | Array
    initialized: Array
    active: Array
    identity_words: Array
    type_identity_words: Array
    kind: Array
    position: Array
    rotation: Array
    velocity: Array
    behavior_supported: Array
    geometry_supported: Array
    collidable: Array
    blocks_los: Array
    local_bounds: Array
    los_offset_supported: Array
    los_offset: Array
    behavior_provenance: Array
    geometry_provenance: Array


def privileged_entity_snapshot_from_surrogate(
    runtime: SurrogateRuntimeState,
) -> PrivilegedEntitySnapshot:
    """Expose existing surrogate state without inventing entity behavior."""

    if not isinstance(runtime, SurrogateRuntimeState):
        raise TypeError("runtime must be a SurrogateRuntimeState")
    batch = runtime.environment_world_id.shape[0]
    return PrivilegedEntitySnapshot(
        source_available=runtime.failure_bits == 0,
        source_overflow=jnp.zeros((batch,), dtype=jnp.bool_),
        source_provenance=PRIVILEGED_ENTITY_PROVENANCE_SURROGATE_RUNTIME,
        initialized=runtime.entity_initialized,
        active=runtime.entity_active,
        identity_words=runtime.entity_identity_words,
        type_identity_words=runtime.entity_type_identity_words,
        kind=runtime.entity_kind,
        position=runtime.entity_position,
        rotation=runtime.entity_rotation,
        velocity=runtime.entity_velocity,
        behavior_supported=runtime.entity_behavior_supported,
        geometry_supported=runtime.entity_geometry_supported,
        collidable=runtime.entity_collidable,
        blocks_los=runtime.entity_blocks_los,
        local_bounds=runtime.entity_local_bounds,
        los_offset_supported=runtime.entity_los_offset_supported,
        los_offset=runtime.entity_los_offset,
        behavior_provenance=jnp.where(
            runtime.entity_behavior_supported,
            jnp.uint8(PRIVILEGED_ENTITY_COMPONENT_PROVENANCE_SURROGATE),
            jnp.uint8(PRIVILEGED_ENTITY_COMPONENT_PROVENANCE_NONE),
        ),
        geometry_provenance=jnp.where(
            runtime.entity_geometry_supported,
            jnp.uint8(PRIVILEGED_ENTITY_COMPONENT_PROVENANCE_SURROGATE),
            jnp.uint8(PRIVILEGED_ENTITY_COMPONENT_PROVENANCE_NONE),
        ),
    )


def produce_privileged_entity_scene(
    snapshot: PrivilegedEntitySnapshot,
    *,
    entity_capacity: int = PRIVILEGED_ENTITY_CAPACITY,
) -> PrivilegedEntityScene:
    """Compact authoritative entity slots and fail closed by environment."""

    if not isinstance(snapshot, PrivilegedEntitySnapshot):
        raise TypeError("snapshot must be a PrivilegedEntitySnapshot")
    if (
        isinstance(entity_capacity, bool)
        or not isinstance(entity_capacity, int)
        or entity_capacity <= 0
    ):
        raise ValueError("entity_capacity must be a positive integer")

    initialized = _field(
        snapshot.initialized,
        np.bool_,
        None,
        "initialized",
    )
    if initialized.ndim != 2 or min(initialized.shape) <= 0:
        raise ValueError("initialized must have non-empty shape [batch, entities]")
    batch, source_capacity = initialized.shape
    shape = (batch, source_capacity)
    active = _field(snapshot.active, np.bool_, shape, "active")
    identity = _field(
        snapshot.identity_words,
        np.uint32,
        shape + (2,),
        "identity_words",
    )
    type_identity = _field(
        snapshot.type_identity_words,
        np.uint32,
        shape + (2,),
        "type_identity_words",
    )
    kind = _field(snapshot.kind, np.uint8, shape, "kind")
    position = _field(
        snapshot.position,
        np.float32,
        shape + (3,),
        "position",
    )
    rotation = _field(
        snapshot.rotation,
        np.float32,
        shape + (3,),
        "rotation",
    )
    velocity = _field(
        snapshot.velocity,
        np.float32,
        shape + (3,),
        "velocity",
    )
    behavior_supported = _field(
        snapshot.behavior_supported,
        np.bool_,
        shape,
        "behavior_supported",
    )
    behavior_provenance = _field(
        snapshot.behavior_provenance,
        np.uint8,
        shape,
        "behavior_provenance",
    )
    geometry_supported = _field(
        snapshot.geometry_supported,
        np.bool_,
        shape,
        "geometry_supported",
    )
    geometry_provenance = _field(
        snapshot.geometry_provenance,
        np.uint8,
        shape,
        "geometry_provenance",
    )
    collidable = _field(snapshot.collidable, np.bool_, shape, "collidable")
    blocks_los = _field(snapshot.blocks_los, np.bool_, shape, "blocks_los")
    local_bounds = _field(
        snapshot.local_bounds,
        np.float32,
        shape + (6,),
        "local_bounds",
    )
    los_offset_supported = _field(
        snapshot.los_offset_supported,
        np.bool_,
        shape,
        "los_offset_supported",
    )
    los_offset = _field(
        snapshot.los_offset,
        np.float32,
        shape + (3,),
        "los_offset",
    )
    source_available = _field(
        snapshot.source_available,
        np.bool_,
        (batch,),
        "source_available",
    )
    source_overflow = _field(
        snapshot.source_overflow,
        np.bool_,
        (batch,),
        "source_overflow",
    )
    provenance = _source_provenance(snapshot.source_provenance, batch)

    identity_present = jnp.any(identity != 0, axis=2)
    type_present = jnp.any(type_identity != 0, axis=2)
    lifecycle_invalid = jnp.any(active & ~initialized, axis=1)
    identity_invalid = jnp.any(
        initialized & (~identity_present | ~type_present),
        axis=1,
    )
    duplicate_identity = _has_duplicate_identity(initialized, identity)
    pose_nonfinite = jnp.any(
        initialized
        & (
            ~jnp.all(jnp.isfinite(position), axis=2)
            | ~jnp.all(jnp.isfinite(rotation), axis=2)
            | ~jnp.all(jnp.isfinite(velocity), axis=2)
        ),
        axis=1,
    )
    bounds_valid = jnp.all(jnp.isfinite(local_bounds), axis=2) & jnp.all(
        local_bounds[:, :, :3] < local_bounds[:, :, 3:], axis=2
    )
    offset_valid = jnp.all(jnp.isfinite(los_offset), axis=2)
    profile_invalid_entity = initialized & (
        (geometry_supported & (~bounds_valid | (los_offset_supported & ~offset_valid)))
        | ((collidable | blocks_los | los_offset_supported) & ~geometry_supported)
    )
    profile_invalid = jnp.any(profile_invalid_entity, axis=1)
    profile_missing = jnp.any(initialized & ~geometry_supported, axis=1)
    dynamics_missing = jnp.any(initialized & ~behavior_supported, axis=1)
    behavior_provenance_valid = (
        behavior_provenance == PRIVILEGED_ENTITY_COMPONENT_PROVENANCE_NATIVE
    ) | (behavior_provenance == PRIVILEGED_ENTITY_COMPONENT_PROVENANCE_SURROGATE)
    geometry_provenance_valid = (
        geometry_provenance == PRIVILEGED_ENTITY_COMPONENT_PROVENANCE_NATIVE
    ) | (geometry_provenance == PRIVILEGED_ENTITY_COMPONENT_PROVENANCE_SURROGATE)
    behavior_provenance_agrees = jnp.where(
        behavior_supported,
        behavior_provenance_valid,
        behavior_provenance == PRIVILEGED_ENTITY_COMPONENT_PROVENANCE_NONE,
    )
    geometry_provenance_agrees = jnp.where(
        geometry_supported,
        geometry_provenance_valid,
        geometry_provenance == PRIVILEGED_ENTITY_COMPONENT_PROVENANCE_NONE,
    )
    component_provenance_invalid = jnp.any(
        initialized & (~behavior_provenance_agrees | ~geometry_provenance_agrees),
        axis=1,
    )
    entity_count = jnp.sum(initialized, axis=1, dtype=jnp.int32)
    capacity_exceeded = entity_count > entity_capacity
    provenance_valid = (provenance == PRIVILEGED_ENTITY_PROVENANCE_NATIVE_RUNTIME) | (
        provenance == PRIVILEGED_ENTITY_PROVENANCE_SURROGATE_RUNTIME
    )
    available = (
        source_available
        & ~source_overflow
        & ~capacity_exceeded
        & ~identity_invalid
        & ~duplicate_identity
        & ~lifecycle_invalid
        & ~pose_nonfinite
        & ~profile_invalid
        & provenance_valid
        & ~component_provenance_invalid
    )
    physical_available = available & ~profile_missing
    dynamics_supported = available & ~dynamics_missing

    diagnostics = (
        jnp.where(
            source_available,
            jnp.uint32(0),
            PRIVILEGED_ENTITY_DIAGNOSTIC_SOURCE_UNAVAILABLE,
        )
        | jnp.where(
            source_overflow,
            PRIVILEGED_ENTITY_DIAGNOSTIC_SOURCE_OVERFLOW,
            jnp.uint32(0),
        )
        | jnp.where(
            capacity_exceeded,
            PRIVILEGED_ENTITY_DIAGNOSTIC_OUTPUT_CAPACITY,
            jnp.uint32(0),
        )
        | jnp.where(
            identity_invalid | duplicate_identity,
            PRIVILEGED_ENTITY_DIAGNOSTIC_IDENTITY,
            jnp.uint32(0),
        )
        | jnp.where(
            pose_nonfinite,
            PRIVILEGED_ENTITY_DIAGNOSTIC_NONFINITE,
            jnp.uint32(0),
        )
        | jnp.where(
            profile_invalid,
            PRIVILEGED_ENTITY_DIAGNOSTIC_PROFILE_INVALID,
            jnp.uint32(0),
        )
        | jnp.where(
            profile_missing,
            PRIVILEGED_ENTITY_DIAGNOSTIC_PROFILE_MISSING,
            jnp.uint32(0),
        )
        | jnp.where(
            dynamics_missing,
            PRIVILEGED_ENTITY_DIAGNOSTIC_DYNAMICS_UNSUPPORTED,
            jnp.uint32(0),
        )
        | jnp.where(
            provenance_valid & ~component_provenance_invalid,
            jnp.uint32(0),
            PRIVILEGED_ENTITY_DIAGNOSTIC_PROVENANCE,
        )
        | jnp.where(
            lifecycle_invalid,
            PRIVILEGED_ENTITY_DIAGNOSTIC_LIFECYCLE_INVALID,
            jnp.uint32(0),
        )
    )

    source_slot = jnp.arange(source_capacity, dtype=jnp.int32)
    order = jnp.argsort(
        jnp.where(initialized, source_slot[None, :], source_capacity),
        axis=1,
        stable=True,
    )

    def selected(values: Array, fill: int | float | bool = 0) -> Array:
        width = min(source_capacity, entity_capacity)
        gathered = values[jnp.arange(batch)[:, None], order[:, :width]]
        if width == entity_capacity:
            return gathered
        padding = [(0, 0), (0, entity_capacity - width)]
        padding.extend([(0, 0)] * (values.ndim - 2))
        return jnp.pad(gathered, padding, constant_values=fill)

    selected_initialized = selected(initialized)
    entity_mask = selected_initialized & available[:, None]
    selected_geometry = selected(geometry_supported)
    physical_mask = entity_mask & selected_geometry & physical_available[:, None]

    def masked(
        values: Array,
        mask: Array,
        fill: int | float | bool = 0,
    ) -> Array:
        gate = mask.reshape(mask.shape + (1,) * (values.ndim - mask.ndim))
        return jnp.where(gate, values, jnp.asarray(fill, dtype=values.dtype))

    slots = selected(
        jnp.broadcast_to(source_slot, shape),
        -1,
    )
    return PrivilegedEntityScene(
        available=available,
        physical_available=physical_available,
        dynamics_supported=dynamics_supported,
        capacity_exceeded=capacity_exceeded,
        diagnostics=diagnostics,
        source_provenance=jnp.where(
            available,
            provenance.astype(jnp.uint8),
            jnp.uint8(0),
        ),
        entity_count=jnp.where(available, entity_count, jnp.int32(0)),
        entity_mask=entity_mask,
        physical_mask=physical_mask,
        source_slot=masked(slots, entity_mask, -1),
        active=masked(selected(active), entity_mask),
        identity_words=masked(selected(identity), entity_mask),
        type_identity_words=masked(selected(type_identity), entity_mask),
        kind=masked(selected(kind), entity_mask),
        position=masked(selected(position), entity_mask),
        rotation=masked(selected(rotation), entity_mask),
        velocity=masked(selected(velocity), entity_mask),
        behavior_supported=masked(
            selected(behavior_supported),
            entity_mask,
        ),
        behavior_provenance=masked(
            selected(behavior_provenance),
            entity_mask,
        ),
        collidable=masked(selected(collidable), physical_mask),
        blocks_los=masked(selected(blocks_los), physical_mask),
        local_bounds=masked(selected(local_bounds), physical_mask),
        los_offset_supported=masked(
            selected(los_offset_supported),
            physical_mask,
        ),
        los_offset=masked(selected(los_offset), physical_mask),
        geometry_provenance=masked(
            selected(geometry_provenance),
            physical_mask,
        ),
    )


def privileged_entity_contract() -> dict[str, object]:
    """Return the framework- and checkpoint-pinnable entity contract."""

    return {
        "schema": PRIVILEGED_ENTITY_SCHEMA,
        "version": PRIVILEGED_ENTITY_VERSION,
        "producer": "produce_privileged_entity_scene",
        "framework_envelope": "PrivilegedSceneEnvelope",
        "entity_axis": "world_owned_instance_not_agent_target_role",
        "default_capacity": PRIVILEGED_ENTITY_CAPACITY,
        "ordering": "initialized_source_slot_ascending",
        "overflow": "clear_complete_environment_row",
        "identity": {
            "instance": "provider_lifetime_stable_uint32_pair",
            "type": "palette_independent_uint32_pair",
            "kind": "provider_input_schema_code_never_instance_or_type_identity",
            "missing_or_duplicate": "clear_complete_environment_row",
            "lifecycle": (
                "not_preserved_across_despawn_replacement_reload_reset_or_regeneration"
            ),
        },
        "availability": {
            "available": "identity_pose_and_source_complete",
            "physical_available": (
                "available_and_every_instance_has_exact_valid_profile"
            ),
            "dynamics_supported": (
                "available_and_every_instance_has_declared_behavior"
            ),
            "active_requires_initialized": True,
            "missing_profile": "clear_all_physical_fields_never_generic_box",
            "unsupported_behavior": "published_but_never_claimed_as_dynamics",
        },
        "provenance": {
            "source": {
                str(PRIVILEGED_ENTITY_PROVENANCE_NATIVE_RUNTIME): "native_runtime",
                str(
                    PRIVILEGED_ENTITY_PROVENANCE_SURROGATE_RUNTIME
                ): "surrogate_runtime",
            },
            "component": {
                str(PRIVILEGED_ENTITY_COMPONENT_PROVENANCE_NONE): "unsupported",
                str(PRIVILEGED_ENTITY_COMPONENT_PROVENANCE_NATIVE): "native",
                str(PRIVILEGED_ENTITY_COMPONENT_PROVENANCE_SURROGATE): "surrogate",
            },
            "support_and_component_provenance_must_agree": True,
        },
        "capacity_evidence": {
            "artifact": ("artifacts/worldgen/privileged-entity-capacity-v1.json"),
            "installed_prefab_records": 4399,
            "installed_maximum_records_per_prefab": 39,
            "managed_native_census_maximum": 2,
            "outside_measured_scope": "checked_fail_closed",
        },
        "logical_bytes_per_environment_at_default_capacity": 26125,
        "actor_conversion": None,
        "behavior_simulation": None,
    }


def privileged_entity_contract_sha256() -> str:
    encoded = json.dumps(
        privileged_entity_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def privileged_entity_schema_ref() -> SchemaRef:
    return SchemaRef(
        name=PRIVILEGED_ENTITY_SCHEMA,
        version=PRIVILEGED_ENTITY_VERSION,
        digest=ContentDigest(
            "sha256",
            privileged_entity_contract_sha256(),
        ),
    )


@dataclass(frozen=True, slots=True)
class PrivilegedEntityCompiler:
    """Concrete neutral entity compiler for the framework protocol."""

    input_schema: SchemaRef
    entity_capacity: int = PRIVILEGED_ENTITY_CAPACITY

    @property
    def contract(self) -> ProviderContract:
        return ProviderContract(
            provider_id="world.privileged-entities-v1",
            consumes=SchemaBundle.from_mapping({"snapshot": self.input_schema}),
            produces=SchemaBundle.from_mapping(
                {"privileged_entities": privileged_entity_schema_ref()}
            ),
            required_capabilities=("entity_spatial_state",),
            provided_capabilities=("privileged_entity_scene",),
        )

    def compile(
        self,
        snapshot: AuthoritativeSnapshot[PrivilegedEntitySnapshot],
    ) -> PrivilegedSceneEnvelope[PrivilegedEntityScene]:
        if not isinstance(snapshot, AuthoritativeSnapshot):
            raise TypeError("snapshot must be an AuthoritativeSnapshot")
        if snapshot.header.schema != self.input_schema:
            raise ValueError("snapshot schema differs from compiler contract")
        if not isinstance(snapshot.payload, PrivilegedEntitySnapshot):
            raise TypeError("snapshot payload must be PrivilegedEntitySnapshot")
        return PrivilegedSceneEnvelope(
            source=snapshot.header,
            schema=privileged_entity_schema_ref(),
            payload=produce_privileged_entity_scene(
                snapshot.payload,
                entity_capacity=self.entity_capacity,
            ),
        )


def _source_provenance(value: int | Array, batch: int) -> Array:
    if isinstance(value, bool):
        raise TypeError("source_provenance must not be bool")
    if isinstance(value, int):
        if not 0 <= value <= np.iinfo(np.uint8).max:
            raise ValueError("source_provenance integer is outside uint8")
        return jnp.full((batch,), value, dtype=jnp.uint8)
    source_dtype = getattr(value, "dtype", None)
    if source_dtype is None or not np.issubdtype(source_dtype, np.integer):
        raise TypeError("source_provenance must use an integer dtype")
    result = jnp.asarray(value)
    if result.shape == ():
        return jnp.broadcast_to(result, (batch,))
    if result.dtype != jnp.uint8:
        raise TypeError("source_provenance vectors must use dtype uint8")
    if result.shape != (batch,):
        raise ValueError(f"source_provenance must have shape {(batch,)}")
    return result


def _has_duplicate_identity(initialized: Array, identity: Array) -> Array:
    order = jnp.lexsort(
        (~initialized, identity[:, :, 1], identity[:, :, 0]),
        axis=1,
    )
    sorted_identity = jnp.take_along_axis(
        identity,
        order[:, :, None],
        axis=1,
    )
    sorted_initialized = jnp.take_along_axis(
        initialized,
        order,
        axis=1,
    )
    adjacent_match = jnp.all(
        sorted_identity[:, 1:] == sorted_identity[:, :-1],
        axis=2,
    )
    return jnp.any(
        adjacent_match & sorted_initialized[:, 1:] & sorted_initialized[:, :-1],
        axis=1,
    )


def _field(
    value: Array,
    dtype: np.dtype,
    shape: tuple[int, ...] | None,
    label: str,
) -> Array:
    source_dtype = getattr(value, "dtype", None)
    if source_dtype is None or np.dtype(source_dtype) != np.dtype(dtype):
        raise TypeError(f"{label} must use dtype {np.dtype(dtype)}")
    result = jnp.asarray(value)
    if shape is not None and result.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    return result


__all__ = [
    "PRIVILEGED_ENTITY_CAPACITY",
    "PRIVILEGED_ENTITY_COMPONENT_PROVENANCE_NATIVE",
    "PRIVILEGED_ENTITY_COMPONENT_PROVENANCE_NONE",
    "PRIVILEGED_ENTITY_COMPONENT_PROVENANCE_SURROGATE",
    "PRIVILEGED_ENTITY_PROVENANCE_NATIVE_RUNTIME",
    "PRIVILEGED_ENTITY_PROVENANCE_SURROGATE_RUNTIME",
    "PRIVILEGED_ENTITY_SCHEMA",
    "PRIVILEGED_ENTITY_VERSION",
    "PrivilegedEntityCompiler",
    "PrivilegedEntityScene",
    "PrivilegedEntitySnapshot",
    "privileged_entity_contract",
    "privileged_entity_contract_sha256",
    "privileged_entity_schema_ref",
    "privileged_entity_snapshot_from_surrogate",
    "produce_privileged_entity_scene",
]
