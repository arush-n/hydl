"""Host factories for fixed-capacity melee loadouts."""

from __future__ import annotations

import operator

import jax.numpy as jnp

from hytalegym.jax.combat.loadout.schema.contract import (
    MAX_MELEE_ATTACKS,
    WEAPON_KIND_MELEE,
)
from hytalegym.jax.combat.loadout.schema.types import MeleeLoadoutBatch
from hytalegym.jax.combat.types import CombatParams


def empty_melee_loadout(batch_size: int) -> MeleeLoadoutBatch:
    """Return an unequipped, valid loadout for every batch row."""

    batch = _positive_batch_size(batch_size)
    shape = (batch, MAX_MELEE_ATTACKS)
    return MeleeLoadoutBatch(
        weapon_id=jnp.full((batch,), -1, dtype=jnp.int32),
        weapon_kind=jnp.zeros((batch,), dtype=jnp.int32),
        equipped_mask=jnp.zeros((batch,), dtype=jnp.bool_),
        attack_mask=jnp.zeros(shape, dtype=jnp.bool_),
        hit_delay_ticks=jnp.zeros(shape, dtype=jnp.int32),
        range_blocks=jnp.zeros(shape, dtype=jnp.float32),
        half_angle_degrees=jnp.zeros(shape, dtype=jnp.float32),
        damage=jnp.zeros(shape, dtype=jnp.float32),
        cooldown_min_seconds=jnp.zeros(shape, dtype=jnp.float32),
        cooldown_max_seconds=jnp.zeros(shape, dtype=jnp.float32),
        requires_line_of_sight=jnp.zeros(shape, dtype=jnp.bool_),
        overflow=jnp.zeros((batch,), dtype=jnp.bool_),
    )


def default_melee_loadout(
    batch_size: int,
    params: CombatParams,
    *,
    weapon_id: int = 0,
) -> MeleeLoadoutBatch:
    """Project the calibrated v8 agent attacks into the generic contract."""

    batch = _positive_batch_size(batch_size)
    identifier = _int32_value(weapon_id, "weapon_id")
    attack_count = int(params.agent_hit_delays.shape[0])
    if attack_count > MAX_MELEE_ATTACKS:
        raise ValueError(
            f"default agent has {attack_count} attacks, exceeding "
            f"capacity {MAX_MELEE_ATTACKS}"
        )
    loadout = empty_melee_loadout(batch)
    attack_mask = loadout.attack_mask.at[:, :attack_count].set(True)
    return loadout._replace(
        weapon_id=jnp.full((batch,), identifier, dtype=jnp.int32),
        weapon_kind=jnp.full(
            (batch,),
            WEAPON_KIND_MELEE,
            dtype=jnp.int32,
        ),
        equipped_mask=jnp.ones((batch,), dtype=jnp.bool_),
        attack_mask=attack_mask,
        hit_delay_ticks=loadout.hit_delay_ticks.at[:, :attack_count].set(
            params.agent_hit_delays[None, :]
        ),
        range_blocks=loadout.range_blocks.at[:, :attack_count].set(
            params.agent_attack_ranges[None, :]
        ),
        half_angle_degrees=(
            loadout.half_angle_degrees.at[:, :attack_count].set(
                params.agent_half_angles[None, :]
            )
        ),
        damage=loadout.damage.at[:, :attack_count].set(
            jnp.broadcast_to(
                params.agent_damage,
                (batch, attack_count),
            )
        ),
        cooldown_min_seconds=(
            loadout.cooldown_min_seconds.at[:, :attack_count].set(
                jnp.broadcast_to(
                    params.agent_attack_pause_min_seconds,
                    (batch, attack_count),
                )
            )
        ),
        cooldown_max_seconds=(
            loadout.cooldown_max_seconds.at[:, :attack_count].set(
                jnp.broadcast_to(
                    params.agent_attack_pause_max_seconds,
                    (batch, attack_count),
                )
            )
        ),
        requires_line_of_sight=(
            loadout.requires_line_of_sight.at[:, :attack_count].set(True)
        ),
    )


def _positive_batch_size(value: int) -> int:
    if isinstance(value, bool):
        raise TypeError("batch_size must be a positive integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError("batch_size must be a positive integer") from error
    if result <= 0:
        raise ValueError("batch_size must be a positive integer")
    return result


def _int32_value(value: int, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{label} must be an integer") from error
    if not -(2**31) <= result < 2**31:
        raise ValueError(f"{label} exceeds int32")
    return result
