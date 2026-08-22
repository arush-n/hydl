"""Fixed-shape falling-block impact dispatch from installed asset evidence."""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.worldgen.block_actions import (
    BLOCK_SEMANTIC_KEY_WORDS,
    stable_asset_id,
)
from hytalegym.worldgen.falling_blocks import (
    FALLING_BLOCK_IMPACT_BREAK,
    FALLING_BLOCK_IMPACT_EXPLODE,
    FALLING_BLOCK_IMPACT_NONE,
    FALLING_BLOCK_IMPACT_PLACE,
    LocalFallingBlockImpact,
    falling_block_impact_contract_sha256 as host_contract_sha256,
    native_falling_block_trace_contract_sha256,
)


Array = jax.Array
FALLING_BLOCK_IMPACT_CATALOG_CAPACITY = 2
FALLING_BLOCK_IMPACT_DISPATCH_SCHEMA = (
    "hytalerl_falling_block_impact_dispatch_v1"
)
FALLING_BLOCK_IMPACT_DISPATCH_VERSION = 1

FALLING_BLOCK_DIAGNOSTIC_INACTIVE = 1 << 0
FALLING_BLOCK_DIAGNOSTIC_INVALID_INPUT = 1 << 1
FALLING_BLOCK_DIAGNOSTIC_CATALOG_MISSING = 1 << 2
FALLING_BLOCK_DIAGNOSTIC_CATALOG_DUPLICATE = 1 << 3
FALLING_BLOCK_DIAGNOSTIC_IMPACT_INVALID = 1 << 4
FALLING_BLOCK_DRY_AIR_MOTION_SCHEMA = (
    "hytalerl_falling_block_dry_air_motion_v1"
)
FALLING_BLOCK_DRY_AIR_MOTION_VERSION = 1
FALLING_BLOCK_AIR_RELATIVE_DENSITY = 0.001225
FALLING_BLOCK_GRAVITY_ACCELERATION = 32.0


class FallingBlockImpactCatalog(NamedTuple):
    entry_mask: Array
    semantic_key: Array
    impact_type: Array
    hitbox_collision_config_key: Array
    explosion_configured: Array


class FallingBlockImpactDispatch(NamedTuple):
    """Native ground-impact branch; downstream effects remain separate."""

    available: Array
    impact_type: Array
    impact_cell: Array
    place_requested: Array
    break_requested: Array
    explode_requested: Array
    break_on_place_rejection: Array
    remove_falling_entity: Array
    hitbox_collision_config_key: Array
    explosion_configured: Array
    diagnostics: Array


class FallingBlockMotionState(NamedTuple):
    """Fixed-shape state at the native pre-gravity trace stage."""

    position: Array
    velocity: Array
    active: Array


class FallingBlockDryAirConfig(NamedTuple):
    mass: Array
    drag_coefficient: Array
    inverted_gravity: Array
    box_width: Array
    box_depth: Array


class FallingBlockDryAirStep(NamedTuple):
    state: FallingBlockMotionState
    terminal_velocity: Array
    valid: Array


def falling_block_impact_catalog_from_local(
    entries: Sequence[LocalFallingBlockImpact],
    *,
    capacity: int = FALLING_BLOCK_IMPACT_CATALOG_CAPACITY,
) -> FallingBlockImpactCatalog:
    """Pack installed impact routes; overflow and duplicate keys reject."""

    if isinstance(capacity, bool) or not isinstance(capacity, int):
        raise TypeError("capacity must be an integer")
    if not 1 <= capacity <= FALLING_BLOCK_IMPACT_CATALOG_CAPACITY:
        raise ValueError("falling-block catalog capacity is out of range")
    source = tuple(entries)
    if not source:
        raise ValueError("falling-block catalog cannot be empty")
    if len(source) > capacity:
        raise ValueError("falling-block catalog capacity exceeded")

    entry_mask = np.zeros((capacity,), dtype=np.bool_)
    semantic_key = np.zeros(
        (capacity, BLOCK_SEMANTIC_KEY_WORDS),
        dtype=np.uint32,
    )
    impact_type = np.zeros((capacity,), dtype=np.uint8)
    collision_key = np.zeros((capacity,), dtype=np.int32)
    explosion = np.zeros((capacity,), dtype=np.bool_)
    keys: set[tuple[int, ...]] = set()
    for slot, entry in enumerate(source):
        if not isinstance(entry, LocalFallingBlockImpact):
            raise TypeError("catalog entry has the wrong type")
        key = tuple(entry.semantic_key)
        if (
            len(key) != BLOCK_SEMANTIC_KEY_WORDS
            or any(
                isinstance(word, bool)
                or not isinstance(word, int)
                or not 0 <= word <= np.iinfo(np.uint32).max
                for word in key
            )
        ):
            raise ValueError("falling-block semantic key is invalid")
        if key in keys:
            raise ValueError("falling-block semantic key is duplicated")
        if entry.impact_type not in (
            FALLING_BLOCK_IMPACT_PLACE,
            FALLING_BLOCK_IMPACT_BREAK,
            FALLING_BLOCK_IMPACT_EXPLODE,
        ):
            raise ValueError("falling-block impact type is invalid")
        keys.add(key)
        entry_mask[slot] = True
        semantic_key[slot] = key
        impact_type[slot] = entry.impact_type
        collision_key[slot] = (
            0
            if entry.hitbox_collision_config_id is None
            else stable_asset_id(entry.hitbox_collision_config_id)
        )
        explosion[slot] = entry.explosion_configured
    return FallingBlockImpactCatalog(
        entry_mask=jnp.asarray(entry_mask),
        semantic_key=jnp.asarray(semantic_key),
        impact_type=jnp.asarray(impact_type),
        hitbox_collision_config_key=jnp.asarray(collision_key),
        explosion_configured=jnp.asarray(explosion),
    )


def falling_block_impact_dispatch(
    catalog: FallingBlockImpactCatalog,
    semantic_key: Array,
    position: Array,
    active: Array,
) -> FallingBlockImpactDispatch:
    """Resolve exact native impact branching without inventing effects."""

    _validate_catalog(catalog)
    key = jnp.asarray(semantic_key)
    point = jnp.asarray(position)
    enabled = jnp.asarray(active)
    prefix = key.shape[:-1]
    if key.shape[-1:] != (BLOCK_SEMANTIC_KEY_WORDS,):
        raise ValueError("semantic_key has the wrong trailing dimension")
    if point.shape != prefix + (3,):
        raise ValueError("position must align with semantic_key")
    if enabled.shape != prefix or enabled.dtype != jnp.bool_:
        raise TypeError("active must be a matching boolean array")
    if key.dtype == jnp.bool_ or not jnp.issubdtype(
        key.dtype,
        jnp.integer,
    ):
        raise TypeError("semantic_key must have an integer dtype")
    if not jnp.issubdtype(point.dtype, jnp.floating):
        raise TypeError("position must have a floating dtype")

    finite = jnp.all(jnp.isfinite(point), axis=-1)
    matches = catalog.entry_mask & jnp.all(
        catalog.semantic_key == key.astype(jnp.uint32)[..., None, :],
        axis=-1,
    )
    match_count = jnp.sum(matches, axis=-1, dtype=jnp.int32)
    slot = jnp.argmax(matches, axis=-1)
    impact = catalog.impact_type[slot].astype(jnp.uint8)
    valid_impact = (
        (impact == FALLING_BLOCK_IMPACT_PLACE)
        | (impact == FALLING_BLOCK_IMPACT_BREAK)
        | (impact == FALLING_BLOCK_IMPACT_EXPLODE)
    )
    available = enabled & finite & (match_count == 1) & valid_impact
    place = available & (impact == FALLING_BLOCK_IMPACT_PLACE)
    broken = available & (impact == FALLING_BLOCK_IMPACT_BREAK)
    exploded = available & (impact == FALLING_BLOCK_IMPACT_EXPLODE)
    diagnostics = (
        jnp.where(
            ~enabled,
            jnp.uint32(FALLING_BLOCK_DIAGNOSTIC_INACTIVE),
            jnp.uint32(0),
        )
        | jnp.where(
            enabled & ~finite,
            jnp.uint32(FALLING_BLOCK_DIAGNOSTIC_INVALID_INPUT),
            jnp.uint32(0),
        )
        | jnp.where(
            enabled & finite & (match_count == 0),
            jnp.uint32(FALLING_BLOCK_DIAGNOSTIC_CATALOG_MISSING),
            jnp.uint32(0),
        )
        | jnp.where(
            enabled & finite & (match_count > 1),
            jnp.uint32(FALLING_BLOCK_DIAGNOSTIC_CATALOG_DUPLICATE),
            jnp.uint32(0),
        )
        | jnp.where(
            enabled & finite & (match_count == 1) & ~valid_impact,
            jnp.uint32(FALLING_BLOCK_DIAGNOSTIC_IMPACT_INVALID),
            jnp.uint32(0),
        )
    )
    return FallingBlockImpactDispatch(
        available=available,
        impact_type=jnp.where(
            available,
            impact,
            FALLING_BLOCK_IMPACT_NONE,
        ).astype(jnp.uint8),
        impact_cell=jnp.where(
            available[..., None],
            jnp.floor(point).astype(jnp.int32),
            jnp.int32(0),
        ),
        place_requested=place,
        break_requested=broken,
        explode_requested=exploded,
        break_on_place_rejection=place,
        remove_falling_entity=available,
        hitbox_collision_config_key=jnp.where(
            available,
            catalog.hitbox_collision_config_key[slot],
            jnp.int32(0),
        ),
        explosion_configured=(
            available & catalog.explosion_configured[slot]
        ),
        diagnostics=diagnostics,
    )


def falling_block_dry_air_step(
    state: FallingBlockMotionState,
    config: FallingBlockDryAirConfig,
    gravity_delta_seconds: Array,
    next_motion_delta_seconds: Array,
) -> FallingBlockDryAirStep:
    """Advance one native pre-gravity sample to the next dry-air sample."""

    if not isinstance(state, FallingBlockMotionState):
        raise TypeError("state must be FallingBlockMotionState")
    if not isinstance(config, FallingBlockDryAirConfig):
        raise TypeError("config must be FallingBlockDryAirConfig")
    position = jnp.asarray(state.position)
    velocity = jnp.asarray(state.velocity)
    active = jnp.asarray(state.active)
    prefix = active.shape
    if (
        position.shape != prefix + (3,)
        or velocity.shape != prefix + (3,)
    ):
        raise ValueError("falling-block vectors must align with active")
    if active.dtype != jnp.bool_:
        raise TypeError("active must be boolean")
    if (
        not jnp.issubdtype(position.dtype, jnp.floating)
        or not jnp.issubdtype(velocity.dtype, jnp.floating)
    ):
        raise TypeError("falling-block vectors must be floating point")
    values = {
        "mass": config.mass,
        "drag_coefficient": config.drag_coefficient,
        "inverted_gravity": config.inverted_gravity,
        "box_width": config.box_width,
        "box_depth": config.box_depth,
        "gravity_delta_seconds": gravity_delta_seconds,
        "next_motion_delta_seconds": next_motion_delta_seconds,
    }
    for name, value in values.items():
        if jnp.shape(value) != prefix:
            raise ValueError(f"{name} must align with active")
    if config.inverted_gravity.dtype != jnp.bool_:
        raise TypeError("inverted_gravity must be boolean")
    for name in (
        "mass",
        "drag_coefficient",
        "box_width",
        "box_depth",
        "gravity_delta_seconds",
        "next_motion_delta_seconds",
    ):
        if not jnp.issubdtype(jnp.asarray(values[name]).dtype, jnp.floating):
            raise TypeError(f"{name} must be floating point")

    mass = jnp.asarray(config.mass)
    drag = jnp.asarray(config.drag_coefficient)
    width = jnp.asarray(config.box_width)
    depth = jnp.asarray(config.box_depth)
    gravity_dt = jnp.asarray(gravity_delta_seconds)
    motion_dt = jnp.asarray(next_motion_delta_seconds)
    finite = (
        jnp.all(jnp.isfinite(position), axis=-1)
        & jnp.all(jnp.isfinite(velocity), axis=-1)
        & jnp.isfinite(mass)
        & jnp.isfinite(drag)
        & jnp.isfinite(width)
        & jnp.isfinite(depth)
        & jnp.isfinite(gravity_dt)
        & jnp.isfinite(motion_dt)
    )
    valid = (
        active
        & finite
        & (mass > 0.0)
        & (drag > 0.0)
        & (width > 0.0)
        & (depth > 0.0)
        & (gravity_dt >= 0.0)
        & (motion_dt >= 0.0)
    )
    safe_mass = jnp.where(valid, mass, jnp.ones_like(mass))
    safe_drag = jnp.where(valid, drag, jnp.ones_like(drag))
    safe_width = jnp.where(valid, width, jnp.ones_like(width))
    safe_depth = jnp.where(valid, depth, jnp.ones_like(depth))
    horizontal_area = safe_width * safe_depth
    terminal_magnitude = jnp.sqrt(
        (
            64.0
            * safe_mass
            * 1000.0
        )
        / (
            FALLING_BLOCK_AIR_RELATIVE_DENSITY
            * horizontal_area
            * 1_000_000.0
            * safe_drag
        )
    )
    vertical = velocity[..., 1]
    ratio = jnp.abs(vertical / terminal_magnitude)
    gravity_step = (
        FALLING_BLOCK_GRAVITY_ACCELERATION
        * (1.0 - ratio * ratio * ratio)
        * gravity_dt
    )
    terminal = jnp.where(
        config.inverted_gravity,
        terminal_magnitude,
        -terminal_magnitude,
    )
    gravity_step = jnp.where(
        config.inverted_gravity,
        gravity_step,
        -gravity_step,
    )
    accelerated = jnp.where(
        (vertical < terminal) & (gravity_step > 0.0),
        jnp.minimum(vertical + gravity_step, terminal),
        jnp.where(
            (vertical > terminal) & (gravity_step < 0.0),
            jnp.maximum(vertical + gravity_step, terminal),
            vertical,
        ),
    )
    next_velocity = velocity.at[..., 1].set(accelerated)
    next_position = position + next_velocity * motion_dt[..., None]
    finite_position = jnp.where(
        jnp.isfinite(position),
        position,
        jnp.zeros_like(position),
    )
    finite_velocity = jnp.where(
        jnp.isfinite(velocity),
        velocity,
        jnp.zeros_like(velocity),
    )
    result_state = FallingBlockMotionState(
        position=jnp.where(
            valid[..., None],
            next_position,
            finite_position,
        ),
        velocity=jnp.where(
            valid[..., None],
            next_velocity,
            finite_velocity,
        ),
        active=valid,
    )
    return FallingBlockDryAirStep(
        state=result_state,
        terminal_velocity=jnp.where(
            valid,
            terminal,
            jnp.zeros_like(terminal),
        ),
        valid=valid,
    )


def falling_block_dry_air_motion_contract() -> dict[str, object]:
    return {
        "schema": FALLING_BLOCK_DRY_AIR_MOTION_SCHEMA,
        "version": FALLING_BLOCK_DRY_AIR_MOTION_VERSION,
        "native_trace_contract_sha256": (
            native_falling_block_trace_contract_sha256()
        ),
        "state_stage": (
            "after_BlockEntitySystems_Ticking_before_"
            "FallingBlockTickingSystem_applyGravity"
        ),
        "equations": {
            "terminal_velocity": (
                "PhysicsMath.getTerminalVelocity_mass_density_area_drag"
            ),
            "acceleration": "32*(1-abs(vy/terminal_velocity)^3)",
            "integrator": (
                "gravity_uses_source_tick_delta_then_position_uses_"
                "destination_tick_delta"
            ),
        },
        "fixed_shape": True,
        "jit_compatible": True,
        "fail_closed": [
            "inactive",
            "nonfinite",
            "nonpositive_mass_drag_or_box_extent",
            "negative_gravity_or_motion_delta",
            "nonfinite_state_sanitized_to_zero",
        ],
        "scope": "dry_air_before_first_collision",
        "unsupported": [
            "fluid_entry_or_exit",
            "collision_response",
            "impact_effect_kernel",
        ],
    }


def falling_block_dry_air_motion_contract_sha256() -> str:
    payload = json.dumps(
        falling_block_dry_air_motion_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def falling_block_impact_dispatch_contract() -> dict[str, object]:
    return {
        "schema": FALLING_BLOCK_IMPACT_DISPATCH_SCHEMA,
        "version": FALLING_BLOCK_IMPACT_DISPATCH_VERSION,
        "host_contract_sha256": host_contract_sha256(),
        "catalog_capacity": FALLING_BLOCK_IMPACT_CATALOG_CAPACITY,
        "lookup": "exact_semantic_key_single_match",
        "position": "MathUtil_floor_XYZ",
        "dispatch": {
            "Place": "place_then_Break_on_rejection",
            "Break": "break_effect",
            "Explode": "explosion_only_if_configured",
            "entity": "removed_after_any_resolved_ground_impact",
        },
        "availability": (
            "active_finite_exact_single_catalog_match_and_known_impact"
        ),
        "provenance": "installed_asset_resolved_native_source_dispatch",
        "unsupported": [
            "falling_entity_physics",
            "place_acknowledgement",
            "break_drop_spawning",
            "explosion_effect_kernel",
        ],
    }


def falling_block_impact_dispatch_contract_sha256() -> str:
    payload = json.dumps(
        falling_block_impact_dispatch_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _validate_catalog(catalog: FallingBlockImpactCatalog) -> None:
    if not isinstance(catalog, FallingBlockImpactCatalog):
        raise TypeError("catalog must be FallingBlockImpactCatalog")
    capacity = catalog.entry_mask.shape[0]
    expected = {
        "entry_mask": (capacity,),
        "semantic_key": (capacity, BLOCK_SEMANTIC_KEY_WORDS),
        "impact_type": (capacity,),
        "hitbox_collision_config_key": (capacity,),
        "explosion_configured": (capacity,),
    }
    for field, shape in expected.items():
        if getattr(catalog, field).shape != shape:
            raise ValueError(f"catalog.{field} has the wrong shape")


__all__ = [
    "FALLING_BLOCK_DIAGNOSTIC_CATALOG_DUPLICATE",
    "FALLING_BLOCK_DIAGNOSTIC_CATALOG_MISSING",
    "FALLING_BLOCK_DIAGNOSTIC_IMPACT_INVALID",
    "FALLING_BLOCK_DIAGNOSTIC_INACTIVE",
    "FALLING_BLOCK_DIAGNOSTIC_INVALID_INPUT",
    "FALLING_BLOCK_AIR_RELATIVE_DENSITY",
    "FALLING_BLOCK_DRY_AIR_MOTION_SCHEMA",
    "FALLING_BLOCK_DRY_AIR_MOTION_VERSION",
    "FALLING_BLOCK_GRAVITY_ACCELERATION",
    "FALLING_BLOCK_IMPACT_CATALOG_CAPACITY",
    "FALLING_BLOCK_IMPACT_DISPATCH_SCHEMA",
    "FALLING_BLOCK_IMPACT_DISPATCH_VERSION",
    "FallingBlockImpactCatalog",
    "FallingBlockImpactDispatch",
    "FallingBlockDryAirConfig",
    "FallingBlockDryAirStep",
    "FallingBlockMotionState",
    "falling_block_dry_air_motion_contract",
    "falling_block_dry_air_motion_contract_sha256",
    "falling_block_dry_air_step",
    "falling_block_impact_catalog_from_local",
    "falling_block_impact_dispatch",
    "falling_block_impact_dispatch_contract",
    "falling_block_impact_dispatch_contract_sha256",
]
