"""Validation helpers internal to the mutable-blocks package."""

import operator
from typing import NamedTuple
import jax
import jax.numpy as jnp
from hytalegym.geometry.contract import FLUID_MOVEMENT_FEATURES, MAX_DETAIL_BOXES, MOVEMENT_FEATURES
from hytalegym.worldgen.block_actions import BLOCK_SEMANTIC_KEY_WORDS


Array = jax.Array


class MutableBlockGeometry(NamedTuple):
    """Portable block identity plus exact post-mutation cell semantics."""

    exact: Array
    block_present: Array
    runtime_block_id: Array
    runtime_block_id_valid: Array
    semantic_key: Array
    semantic_key_valid: Array
    affordance_valid: Array
    affordance_tags: Array
    gather_type_index: Array
    required_tool_quality: Array
    rotation_index: Array
    flags: Array
    fluid_level: Array
    fluid_fill_height: Array
    support: Array
    block_damage: Array
    fluid_damage: Array
    movement: Array
    fluid_movement: Array
    collision_boxes: Array
    collision_box_mask: Array


class MutableBlockState(NamedTuple):
    """One atomic base identity plus a bounded exact mutation overlay."""

    synchronized: Array
    world_id: Array
    base_semantic_sha256: Array
    resync_epoch: Array
    mutation_revision: Array
    base_provenance: Array
    section_mask: Array
    section_coordinate: Array
    local_change_counter: Array
    global_change_counter: Array
    cell_mask: Array
    cell_position: Array
    geometry_override: Array
    health_override: Array
    block_health: Array
    seconds_since_damage: Array
    cell_provenance: Array
    geometry: MutableBlockGeometry
    failure_bits: Array


class MutableBlockResync(NamedTuple):
    """A complete replacement base; partial publication is never accepted."""

    publish: Array
    snapshot_complete: Array
    world_id: Array
    base_semantic_sha256: Array
    resync_epoch: Array
    base_provenance: Array
    section_mask: Array
    section_coordinate: Array
    local_change_counter: Array
    global_change_counter: Array
    cell_mask: Array
    cell_position: Array
    geometry_override: Array
    health_override: Array
    block_health: Array
    seconds_since_damage: Array
    cell_provenance: Array
    geometry: MutableBlockGeometry


class MutableBlockUpdate(NamedTuple):
    """One atomic fixed-capacity update batch against an expected revision."""

    mask: Array
    world_id: Array
    base_semantic_sha256: Array
    resync_epoch: Array
    expected_revision: Array
    position: Array
    geometry_changed: Array
    health_changed: Array
    damage_applied: Array
    expected_health: Array
    block_health_after: Array
    provenance: Array
    geometry: MutableBlockGeometry


def _validate_state_shapes(state: MutableBlockState) -> None:
    if state.world_id.ndim != 1:
        raise ValueError("mutable state world_id must have shape [batch]")
    batch = state.world_id.shape[0]
    if state.section_mask.ndim != 2 or state.section_mask.shape[0] != batch:
        raise ValueError("section_mask must have shape [batch, section]")
    if state.cell_mask.ndim != 2 or state.cell_mask.shape[0] != batch:
        raise ValueError("cell_mask must have shape [batch, cell]")
    sections = state.section_mask.shape[1]
    cells = state.cell_mask.shape[1]
    expected = {
        "synchronized": (batch,),
        "base_semantic_sha256": (
            batch,
            BLOCK_SEMANTIC_KEY_WORDS,
        ),
        "resync_epoch": (batch,),
        "mutation_revision": (batch,),
        "base_provenance": (batch,),
        "section_coordinate": (batch, sections, 3),
        "local_change_counter": (batch, sections),
        "global_change_counter": (batch, sections),
        "cell_position": (batch, cells, 3),
        "geometry_override": (batch, cells),
        "health_override": (batch, cells),
        "block_health": (batch, cells),
        "seconds_since_damage": (batch, cells),
        "cell_provenance": (batch, cells),
        "failure_bits": (batch,),
    }
    for name, shape in expected.items():
        if getattr(state, name).shape != shape:
            raise ValueError(f"state.{name} must have shape {shape}")
    _validate_geometry_shapes(state.geometry, (batch, cells), "state.geometry")
    if state.local_change_counter.dtype != jnp.int16:
        raise ValueError("local change counters must be signed int16")
    if state.global_change_counter.dtype != jnp.int16:
        raise ValueError("global change counters must be signed int16")


def _validate_resync_shapes(
    state: MutableBlockState,
    frame: MutableBlockResync,
) -> None:
    _validate_state_shapes(state)
    batch, cells = state.cell_mask.shape
    sections = state.section_mask.shape[1]
    expected = {
        "publish": (batch,),
        "snapshot_complete": (batch,),
        "world_id": (batch,),
        "base_semantic_sha256": (
            batch,
            BLOCK_SEMANTIC_KEY_WORDS,
        ),
        "resync_epoch": (batch,),
        "base_provenance": (batch,),
        "section_mask": (batch, sections),
        "section_coordinate": (batch, sections, 3),
        "local_change_counter": (batch, sections),
        "global_change_counter": (batch, sections),
        "cell_mask": (batch, cells),
        "cell_position": (batch, cells, 3),
        "geometry_override": (batch, cells),
        "health_override": (batch, cells),
        "block_health": (batch, cells),
        "seconds_since_damage": (batch, cells),
        "cell_provenance": (batch, cells),
    }
    for name, shape in expected.items():
        if getattr(frame, name).shape != shape:
            raise ValueError(f"resync.{name} must have shape {shape}")
    _validate_geometry_shapes(frame.geometry, (batch, cells), "resync.geometry")
    if frame.local_change_counter.dtype != jnp.int16:
        raise ValueError("resync local counters must be signed int16")
    if frame.global_change_counter.dtype != jnp.int16:
        raise ValueError("resync global counters must be signed int16")


def _validate_update_shapes(
    state: MutableBlockState,
    update: MutableBlockUpdate,
) -> None:
    _validate_state_shapes(state)
    if update.mask.ndim != 2:
        raise ValueError("update.mask must have shape [batch, query]")
    batch, queries = update.mask.shape
    if batch != state.world_id.shape[0]:
        raise ValueError("update batch must match mutable state")
    shape = (batch, queries)
    expected = {
        "world_id": (batch,),
        "base_semantic_sha256": (
            batch,
            BLOCK_SEMANTIC_KEY_WORDS,
        ),
        "resync_epoch": (batch,),
        "expected_revision": (batch,),
        "position": shape + (3,),
        "geometry_changed": shape,
        "health_changed": shape,
        "damage_applied": shape,
        "expected_health": shape,
        "block_health_after": shape,
        "provenance": shape,
    }
    for name, expected_shape in expected.items():
        if getattr(update, name).shape != expected_shape:
            raise ValueError(
                f"update.{name} must have shape {expected_shape}"
            )
    _validate_geometry_shapes(update.geometry, shape, "update.geometry")


def _validate_geometry_shapes(
    geometry: MutableBlockGeometry,
    prefix: tuple[int, ...],
    label: str,
) -> None:
    scalar = (
        "exact",
        "block_present",
        "runtime_block_id",
        "runtime_block_id_valid",
        "semantic_key_valid",
        "affordance_valid",
        "affordance_tags",
        "gather_type_index",
        "required_tool_quality",
        "rotation_index",
        "flags",
        "fluid_level",
        "fluid_fill_height",
        "support",
        "block_damage",
        "fluid_damage",
    )
    for name in scalar:
        if getattr(geometry, name).shape != prefix:
            raise ValueError(f"{label}.{name} must have shape {prefix}")
    expected = {
        "semantic_key": prefix + (BLOCK_SEMANTIC_KEY_WORDS,),
        "movement": prefix + (MOVEMENT_FEATURES,),
        "fluid_movement": prefix + (FLUID_MOVEMENT_FEATURES,),
        "collision_boxes": prefix + (MAX_DETAIL_BOXES, 6),
        "collision_box_mask": prefix + (MAX_DETAIL_BOXES,),
    }
    for name, shape in expected.items():
        if getattr(geometry, name).shape != shape:
            raise ValueError(f"{label}.{name} must have shape {shape}")


def _array_shape(
    value: Array,
    shape: tuple[int, ...],
    dtype,
    label: str,
) -> Array:
    result = jnp.asarray(value, dtype=dtype)
    if result.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    return result


def _shape_tuple(value: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError("prefix_shape must be a tuple")
    return tuple(_nonnegative_int(item, "prefix dimension") for item in value)


def _positive_int(value: int, label: str) -> int:
    result = _exact_int(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative_int(value: int, label: str) -> int:
    result = _exact_int(value, label)
    if result < 0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _exact_int(value: int, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{label} must be an integer") from error
