"""Ordered, fail-closed entity dynamics for Hytale explosions.

World publishes raw environment effects. Combat remains responsible for
damage mitigation, death, and authored knockback-controller integration.
"""

from __future__ import annotations

import hashlib
import json
from typing import NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.world.explosions.contact import (
    entity_only_explosion_contract_sha256,
)
from hytalegym.worldgen.native_explosion_ordering import (
    native_explosion_ordering_contract_sha256,
)


Array = jax.Array
ORDERED_EXPLOSION_DYNAMICS_SCHEMA = "hytalerl_ordered_explosion_dynamics_v1"
ORDERED_EXPLOSION_DYNAMICS_VERSION = 1


class OrderedExplosionDynamics(NamedTuple):
    """Raw entity effects emitted only after the block/admission phase."""

    available: Array
    entity_effect_mask: Array
    attenuation: Array
    raw_environment_damage: Array
    radial_direction: Array
    radial_direction_valid: Array
    ordering_unsupported: Array
    invalid: Array


def ordered_explosion_entity_dynamics(
    *,
    block_phase_complete: Array,
    entity_source_complete: Array,
    entity_effects_order_independent: Array,
    entity_candidate_mask: Array,
    entity_distance: Array,
    entity_offset_from_origin: Array,
    entity_damage_radius: Array,
    entity_damage: Array,
    entity_damage_falloff: Array,
    radial_direction_required: Array,
) -> OrderedExplosionDynamics:
    """Apply Hytale's exact raw entity falloff after its block phase.

    The last axis of candidate arrays is an identity-indexed entity set, not
    native iteration order. A multi-entity row is rejected unless the caller
    positively certifies that its effects commute.
    """

    candidates = jnp.asarray(entity_candidate_mask, dtype=jnp.bool_)
    if candidates.ndim < 2 or candidates.shape[-1] <= 0:
        raise ValueError("entity_candidate_mask must end in an entity axis")
    prefix = candidates.shape[:-1]
    entities = candidates.shape[-1]
    distance = _array(
        entity_distance, prefix + (entities,), jnp.float32, "entity_distance"
    )
    offset = _array(
        entity_offset_from_origin,
        prefix + (entities, 3),
        jnp.float32,
        "entity_offset_from_origin",
    )
    block_ready = _array(
        block_phase_complete, prefix, jnp.bool_, "block_phase_complete"
    )
    entity_ready = _array(
        entity_source_complete, prefix, jnp.bool_, "entity_source_complete"
    )
    commutative = _array(
        entity_effects_order_independent,
        prefix,
        jnp.bool_,
        "entity_effects_order_independent",
    )
    radius = _array(entity_damage_radius, prefix, jnp.float32, "entity_damage_radius")
    base_damage = _array(entity_damage, prefix, jnp.float32, "entity_damage")
    falloff = _array(
        entity_damage_falloff,
        prefix,
        jnp.float32,
        "entity_damage_falloff",
    )
    direction_required = _array(
        radial_direction_required,
        prefix,
        jnp.bool_,
        "radial_direction_required",
    )

    offset_distance = jnp.linalg.norm(offset, axis=-1)
    direction_valid = jnp.isfinite(offset_distance) & (offset_distance > 0.0)
    candidate_invalid = candidates & (
        ~jnp.isfinite(distance)
        | (distance < 0.0)
        | (distance >= radius[..., None])
        | ~jnp.all(jnp.isfinite(offset), axis=-1)
        | ~jnp.isclose(distance, offset_distance, rtol=2.0e-5, atol=2.0e-5)
        | (direction_required[..., None] & ~direction_valid)
    )
    scalar_invalid = (
        ~jnp.isfinite(radius)
        | (radius <= 0.0)
        | ~jnp.isfinite(base_damage)
        | (base_damage < 0.0)
        | ~jnp.isfinite(falloff)
        | (falloff < 0.0)
    )
    ordering_unsupported = (
        jnp.sum(candidates.astype(jnp.int32), axis=-1) > 1
    ) & ~commutative
    invalid = scalar_invalid | jnp.any(candidate_invalid, axis=-1)
    available = block_ready & entity_ready & ~ordering_unsupported & ~invalid
    mask = candidates & available[..., None]
    attenuation = jnp.power(
        jnp.clip(
            1.0
            - distance / jnp.maximum(radius[..., None], jnp.finfo(jnp.float32).tiny),
            0.0,
            1.0,
        ),
        falloff[..., None],
    )
    direction = offset / jnp.maximum(
        offset_distance[..., None],
        jnp.finfo(jnp.float32).tiny,
    )
    return OrderedExplosionDynamics(
        available=available,
        entity_effect_mask=mask,
        attenuation=jnp.where(mask, attenuation, 0.0),
        raw_environment_damage=jnp.where(
            mask,
            base_damage[..., None] * attenuation,
            0.0,
        ),
        radial_direction=jnp.where(
            (mask & direction_valid)[..., None],
            direction,
            0.0,
        ),
        radial_direction_valid=mask & direction_valid,
        ordering_unsupported=ordering_unsupported,
        invalid=invalid,
    )


def ordered_explosion_dynamics_contract() -> dict[str, object]:
    return {
        "schema": ORDERED_EXPLOSION_DYNAMICS_SCHEMA,
        "version": ORDERED_EXPLOSION_DYNAMICS_VERSION,
        "producer": "ordered_explosion_entity_dynamics",
        "native_formula": (
            "entityDamage*pow(1-distance/entityDamageRadius,entityDamageFalloff)"
        ),
        "phase_gate": "block_or_admission_phase_complete_before_entity_effects",
        "entity_axis": (
            "identity_indexed_set;permutation_equivariant;not_native_iteration_order"
        ),
        "effect_boundary": {
            "world": "raw_environment_damage_and_radial_direction",
            "combat": "mitigation_death_and_knockback_controller",
        },
        "fail_closed": [
            "incomplete_block_or_entity_source",
            "invalid_radius_damage_falloff_distance_or_offset",
            "required_zero_length_radial_direction",
            "multi_entity_order_dependent_effects",
        ],
        "upstream": {
            "entity_only_explosion_contract_sha256": (
                entity_only_explosion_contract_sha256()
            ),
            "native_ordering_contract_sha256": (
                native_explosion_ordering_contract_sha256()
            ),
        },
    }


def ordered_explosion_dynamics_contract_sha256() -> str:
    payload = json.dumps(
        ordered_explosion_dynamics_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _array(value: Array, shape: tuple[int, ...], dtype, label: str) -> Array:
    result = jnp.asarray(value, dtype=dtype)
    if result.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    return result


__all__ = [
    "ORDERED_EXPLOSION_DYNAMICS_SCHEMA",
    "ORDERED_EXPLOSION_DYNAMICS_VERSION",
    "OrderedExplosionDynamics",
    "ordered_explosion_dynamics_contract",
    "ordered_explosion_dynamics_contract_sha256",
    "ordered_explosion_entity_dynamics",
]
