"""Pure JAX learner projection for episode-pinned melee loadouts."""

from __future__ import annotations

import jax.numpy as jnp

from hytalegym.jax.combat.loadout.schema.contract import (
    LOADOUT_FAILURE_ATTACK_OVERFLOW,
    MAX_MELEE_ATTACKS,
    MELEE_ATTACK_FLOAT_SIZE,
    MELEE_ATTACK_INTEGER_SIZE,
    MELEE_COOLDOWN_NORMALIZATION_SECONDS,
    MELEE_LOADOUT_OBSERVATION_VERSION,
    MELEE_RANGE_NORMALIZATION_BLOCKS,
    MELEE_WEAPON_INTEGER_SIZE,
)
from hytalegym.jax.combat.loadout.schema.types import (
    MeleeLoadoutBatch,
    MeleeLoadoutObservation,
)
from hytalegym.jax.combat.types import CombatParams


def encode_melee_loadout(
    loadout: MeleeLoadoutBatch,
    params: CombatParams,
) -> MeleeLoadoutObservation:
    """Normalize a fixed loadout without changing combat or equipment state."""

    batch = loadout.weapon_id.shape[0]
    effective_mask = (
        loadout.attack_mask
        & loadout.equipped_mask[:, None]
        & ~loadout.overflow[:, None]
    )
    attack_count = jnp.sum(
        effective_mask.astype(jnp.int32),
        axis=1,
        dtype=jnp.int32,
    )
    target_health = jnp.maximum(
        jnp.abs(params.target_max_health),
        jnp.float32(1.0e-6),
    )
    attack_f32 = jnp.stack(
        (
            loadout.range_blocks / jnp.float32(MELEE_RANGE_NORMALIZATION_BLOCKS),
            loadout.half_angle_degrees / jnp.float32(180.0),
            loadout.damage / target_health,
            loadout.cooldown_min_seconds
            / jnp.float32(MELEE_COOLDOWN_NORMALIZATION_SECONDS),
            loadout.cooldown_max_seconds
            / jnp.float32(MELEE_COOLDOWN_NORMALIZATION_SECONDS),
            loadout.requires_line_of_sight.astype(jnp.float32),
        ),
        axis=2,
    )
    attack_f32 = jnp.where(
        effective_mask[:, :, None],
        jnp.clip(attack_f32, 0.0, 1.0),
        jnp.float32(0.0),
    )
    attack_slots = jnp.broadcast_to(
        jnp.arange(MAX_MELEE_ATTACKS, dtype=jnp.int32),
        (batch, MAX_MELEE_ATTACKS),
    )
    attack_i32 = jnp.stack(
        (attack_slots, loadout.hit_delay_ticks),
        axis=2,
    )
    attack_i32 = jnp.where(
        effective_mask[:, :, None],
        attack_i32,
        jnp.int32(0),
    )
    weapon_i32 = jnp.stack(
        (
            loadout.weapon_id,
            loadout.weapon_kind,
            attack_count,
        ),
        axis=1,
    ).astype(jnp.int32)
    weapon_i32 = jnp.where(
        (loadout.equipped_mask & ~loadout.overflow)[:, None],
        weapon_i32,
        jnp.int32(0),
    )
    failure_bits = jnp.where(
        loadout.overflow,
        jnp.uint32(LOADOUT_FAILURE_ATTACK_OVERFLOW),
        jnp.uint32(0),
    )
    valid = failure_bits == jnp.uint32(0)

    if attack_f32.shape[-1] != MELEE_ATTACK_FLOAT_SIZE:
        raise AssertionError("melee loadout float feature contract drift")
    if attack_i32.shape[-1] != MELEE_ATTACK_INTEGER_SIZE:
        raise AssertionError("melee loadout integer feature contract drift")
    if weapon_i32.shape[-1] != MELEE_WEAPON_INTEGER_SIZE:
        raise AssertionError("melee weapon feature contract drift")
    return MeleeLoadoutObservation(
        schema_version=jnp.full(
            (batch,),
            MELEE_LOADOUT_OBSERVATION_VERSION,
            dtype=jnp.int32,
        ),
        weapon_i32=weapon_i32,
        equipped_mask=loadout.equipped_mask & ~loadout.overflow,
        attack_f32=attack_f32.astype(jnp.float32),
        attack_i32=attack_i32.astype(jnp.int32),
        attack_mask=effective_mask,
        valid=valid,
        failure_bits=failure_bits,
    )
