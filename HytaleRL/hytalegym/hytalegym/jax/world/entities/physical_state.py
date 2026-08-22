"""Neutral temporal fall and breathing state over exact world geometry."""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.geometry.contract import FLAG_FLUID
from hytalegym.jax.world.geometry import (
    CellEnvironmentResult,
    cell_environment_result,
)
from hytalegym.jax.world.perception.los import (
    GeometryProvider,
    geometry_hitbox_line_of_sight_result,
)
from hytalegym.jax.world.region.geometry import (
    region_cell_environment_result,
)
from hytalegym.jax.world.region.types import RegionGeometryState


Array = jax.Array
ACTOR_PHYSICAL_STATE_SCHEMA = "hytalerl_actor_physical_state_v1"
ACTOR_PHYSICAL_STATE_VERSION = 1


class ActorPhysicalState(NamedTuple):
    """Persistent per-actor condition state."""

    fall_distance: Array
    submersion_duration_seconds: Array
    fall_valid: Array
    submersion_valid: Array


class ActorPhysicalStateResult(NamedTuple):
    """Updated neutral state, one-tick events, and component validity."""

    state: ActorPhysicalState
    available: Array
    fall_available: Array
    submersion_available: Array
    head_occlusion_available: Array
    landing_event: Array
    landing_fall_distance: Array
    landing_speed: Array
    eyes_submerged: Array
    submersion_duration_seconds: Array
    head_occluded: Array
    fluid_identity_available: Array
    fluid_identity: Array
    geometry_exhausted: Array
    invalid: Array


def initial_actor_physical_state(batch_size: int) -> ActorPhysicalState:
    """Return valid zero state for a fixed actor batch."""

    if (
        isinstance(batch_size, (bool, np.bool_))
        or not isinstance(batch_size, (int, np.integer))
        or int(batch_size) <= 0
    ):
        raise ValueError("batch_size must be a positive integer")
    shape = (int(batch_size),)
    return ActorPhysicalState(
        fall_distance=jnp.zeros(shape, dtype=jnp.float32),
        submersion_duration_seconds=jnp.zeros(shape, dtype=jnp.float32),
        fall_valid=jnp.ones(shape, dtype=jnp.bool_),
        submersion_valid=jnp.ones(shape, dtype=jnp.bool_),
    )


def advance_actor_physical_state(
    state: ActorPhysicalState,
    geometry: GeometryProvider,
    previous_position: Array,
    position: Array,
    eye_position: Array,
    previous_grounded: Array,
    grounded: Array,
    pre_move_in_fluid: Array,
    pre_move_climbing: Array,
    pre_move_flying: Array,
    pre_move_gliding: Array,
    landing_velocity_y: Array,
    delta_seconds: Array | float,
) -> ActorPhysicalStateResult:
    """Advance source-pinned fall state and exact breathing conditions.

    Movement flags are sampled before the move, matching
    ``LivingEntity.moveTo``. ``landing_velocity_y`` is the velocity at the
    native damage-gather boundary; this function only publishes its magnitude
    and never applies damage.
    """

    if not isinstance(state, ActorPhysicalState):
        raise TypeError("state must be ActorPhysicalState")
    points = _positions(previous_position, "previous_position")
    current = _positions(position, "position")
    eyes = _positions(eye_position, "eye_position")
    if current.shape != points.shape or eyes.shape != points.shape:
        raise ValueError("position inputs must have the same shape")
    batch = points.shape[0]
    _validate_state(state, batch)
    was_grounded = _booleans(previous_grounded, batch, "previous_grounded")
    now_grounded = _booleans(grounded, batch, "grounded")
    in_fluid = _booleans(pre_move_in_fluid, batch, "pre_move_in_fluid")
    climbing = _booleans(pre_move_climbing, batch, "pre_move_climbing")
    flying = _booleans(pre_move_flying, batch, "pre_move_flying")
    gliding = _booleans(pre_move_gliding, batch, "pre_move_gliding")
    velocity_y = _scalars(landing_velocity_y, batch, "landing_velocity_y")
    dt = _scalars(delta_seconds, batch, "delta_seconds")

    finite_motion = (
        jnp.all(jnp.isfinite(points) & jnp.isfinite(current), axis=1)
        & jnp.isfinite(velocity_y)
        & jnp.isfinite(dt)
        & (dt > 0.0)
        & jnp.isfinite(state.fall_distance)
        & (state.fall_distance >= 0.0)
    )
    excluded = in_fluid | climbing | flying | gliding
    descent = jnp.maximum(points[:, 1] - current[:, 1], 0.0)
    accumulated = jnp.where(
        ~was_grounded & ~excluded,
        state.fall_distance + descent,
        0.0,
    )
    fall_available = state.fall_valid & finite_motion
    landing = fall_available & ~was_grounded & now_grounded & (accumulated > 0.0)
    next_fall_distance = jnp.where(
        fall_available & ~now_grounded & ~excluded,
        accumulated,
        0.0,
    )

    breathing = _breathing_conditions(geometry, eyes)
    finite_submersion = (
        jnp.all(jnp.isfinite(eyes), axis=1)
        & jnp.isfinite(dt)
        & (dt > 0.0)
        & jnp.isfinite(state.submersion_duration_seconds)
        & (state.submersion_duration_seconds >= 0.0)
    )
    submersion_available = (
        state.submersion_valid & finite_submersion & breathing.submersion_available
    )
    duration = jnp.where(
        submersion_available & breathing.eyes_submerged,
        state.submersion_duration_seconds + dt,
        0.0,
    )
    head_available = breathing.head_occlusion_available
    next_state = ActorPhysicalState(
        fall_distance=jnp.where(fall_available, next_fall_distance, 0.0),
        submersion_duration_seconds=jnp.where(
            submersion_available,
            duration,
            0.0,
        ),
        fall_valid=fall_available,
        submersion_valid=submersion_available,
    )
    invalid = ~finite_motion | ~finite_submersion
    return ActorPhysicalStateResult(
        state=next_state,
        available=fall_available & submersion_available & head_available,
        fall_available=fall_available,
        submersion_available=submersion_available,
        head_occlusion_available=head_available,
        landing_event=landing,
        landing_fall_distance=jnp.where(landing, accumulated, 0.0),
        landing_speed=jnp.where(landing, jnp.abs(velocity_y), 0.0),
        eyes_submerged=submersion_available & breathing.eyes_submerged,
        submersion_duration_seconds=jnp.where(
            submersion_available,
            duration,
            0.0,
        ),
        head_occluded=head_available & breathing.head_occluded,
        # The compiled providers intentionally discard process-local fluid
        # IDs. Zero is exact only when the breathing point is not submerged.
        fluid_identity_available=(submersion_available & ~breathing.eyes_submerged),
        fluid_identity=jnp.zeros((batch,), dtype=jnp.int32),
        geometry_exhausted=breathing.geometry_exhausted,
        invalid=invalid,
    )


class _BreathingConditions(NamedTuple):
    submersion_available: Array
    head_occlusion_available: Array
    eyes_submerged: Array
    head_occluded: Array
    geometry_exhausted: Array


def _breathing_conditions(
    geometry: GeometryProvider,
    eyes: Array,
) -> _BreathingConditions:
    cell = _cell_result(geometry, eyes)
    is_fluid = (cell.flags & jnp.int32(FLAG_FLUID)) != 0
    fill_available = (~is_fluid) | (
        (cell.fluid_fill_height >= 0.0) & (cell.fluid_fill_height <= 1.0)
    )
    finite = jnp.all(jnp.isfinite(eyes), axis=1)
    submersion_available = finite & ~cell.geometry_exhausted & fill_available
    relative_y = eyes[:, 1] - jnp.floor(eyes[:, 1])
    eyes_submerged = (
        submersion_available & is_fluid & (relative_y <= cell.fluid_fill_height)
    )
    hitbox = geometry_hitbox_line_of_sight_result(
        geometry,
        eyes,
        eyes,
        max_cells=1,
    )
    head_available = (
        finite
        & ~hitbox.invalid
        & ~hitbox.capacity_exceeded
        & ~hitbox.geometry_exhausted
    )
    return _BreathingConditions(
        submersion_available=submersion_available,
        head_occlusion_available=head_available,
        eyes_submerged=eyes_submerged,
        head_occluded=head_available & ~hitbox.clear,
        geometry_exhausted=(cell.geometry_exhausted | hitbox.geometry_exhausted),
    )


def _cell_result(
    geometry: GeometryProvider,
    point: Array,
) -> CellEnvironmentResult:
    if isinstance(geometry, RegionGeometryState):
        return region_cell_environment_result(geometry, point)
    return cell_environment_result(geometry, point)


def _positions(value: Array, label: str) -> Array:
    result = jnp.asarray(value, dtype=jnp.float32)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"{label} must have shape [batch, 3]")
    return result


def _booleans(value: Array, batch: int, label: str) -> Array:
    result = jnp.asarray(value)
    if result.dtype != jnp.bool_:
        raise TypeError(f"{label} must have boolean dtype")
    if result.shape != (batch,):
        raise ValueError(f"{label} must have shape [batch]")
    return result


def _scalars(value: Array | float, batch: int, label: str) -> Array:
    result = jnp.asarray(value, dtype=jnp.float32)
    if result.ndim == 0:
        result = jnp.broadcast_to(result, (batch,))
    if result.shape != (batch,):
        raise ValueError(f"{label} must be scalar or shape [batch]")
    return result


def _validate_state(state: ActorPhysicalState, batch: int) -> None:
    for label, value in (
        ("fall_distance", state.fall_distance),
        ("submersion_duration_seconds", state.submersion_duration_seconds),
    ):
        if jnp.asarray(value).shape != (batch,):
            raise ValueError(f"state.{label} must have shape [batch]")
    for label, value in (
        ("fall_valid", state.fall_valid),
        ("submersion_valid", state.submersion_valid),
    ):
        result = jnp.asarray(value)
        if result.shape != (batch,) or result.dtype != jnp.bool_:
            raise ValueError(f"state.{label} must be boolean with shape [batch]")


def actor_physical_state_contract() -> dict[str, object]:
    """Return the checkpoint-pinnable neutral physical-state contract."""

    return {
        "schema": ACTOR_PHYSICAL_STATE_SCHEMA,
        "version": ACTOR_PHYSICAL_STATE_VERSION,
        "function": "advance_actor_physical_state",
        "providers": ["GeometryState", "RegionGeometryState"],
        "fall": {
            "accumulator": (
                "positive previous_y-current_y while pre-move state is "
                "airborne, dry, not climbing, not flying, and not gliding"
            ),
            "reset": (
                "pre-move grounded or excluded controller state; first "
                "grounded result publishes then clears the accumulator"
            ),
            "landing_event": ("airborne-to-grounded with positive accumulated descent"),
            "landing_speed": ("absolute vertical velocity at damage-gather boundary"),
            "native_sources": [
                "LivingEntity.moveTo",
                "DamageSystems.FallDamageNPCs",
            ],
            "damage_owned_by": "combat",
        },
        "breathing": {
            "point": "runtime-derived model breathing/eye position",
            "eyes_submerged": ("fluid cell and fractional y <= level/MaxFluidLevel"),
            "head_occluded": ("breathing point inside a solid rotated detail box"),
            "submersion_duration_seconds": {
                "semantics": "continuous time while eyes_submerged",
                "provenance": "learner_derived_not_native_delayed_system_state",
            },
            "fluid_identity": {
                "value": "int32 zero",
                "available": ("only when the breathing point is proven not submerged"),
                "limitation": ("compiled providers discard process-local fluid IDs"),
            },
            "native_sources": [
                "LivingEntity.getPackedMaterialAndFluidAtBreathingHeight",
                "WorldUtil.getPackedMaterialAndFluidAtPosition",
                "EntityUtils.processEntityBreathing",
            ],
        },
        "availability": "independent_fall_submersion_head_occlusion_bits",
        "fail_closed": [
            "non_finite_or_non_positive_time",
            "non_finite_position_or_velocity",
            "invalid_persistent_state",
            "provider_coverage_exhaustion",
            "legacy_fluid_fill_unavailable",
            "nonzero_fluid_identity_unavailable",
        ],
        "unsupported": [
            "fall_or_breathing_damage",
            "oxygen_stat_evolution",
            "BreathingCheckEvent_overrides",
            "native_one_second_delayed_system_phase",
            "player_client_input_queue_ordering",
        ],
        "certification": {
            "instantaneous_geometry": "native_exact_geometry_semantics",
            "temporal_transition": ("source_pinned_not_runtime_differential_certified"),
        },
    }


def actor_physical_state_contract_sha256() -> str:
    payload = json.dumps(
        actor_physical_state_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "ACTOR_PHYSICAL_STATE_SCHEMA",
    "ACTOR_PHYSICAL_STATE_VERSION",
    "ActorPhysicalState",
    "ActorPhysicalStateResult",
    "actor_physical_state_contract",
    "actor_physical_state_contract_sha256",
    "advance_actor_physical_state",
    "initial_actor_physical_state",
]
