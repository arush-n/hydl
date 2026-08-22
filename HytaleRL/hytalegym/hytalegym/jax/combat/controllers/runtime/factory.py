"""Host factories for the pinned Hytale 0.5.7 ranged controllers."""

from __future__ import annotations

import operator

import jax.numpy as jnp

from hytalegym.jax.combat.arsenal import (
    CROSSBOW_FAMILY_IDS,
    SHORTBOW_FAMILY_IDS,
)
from hytalegym.jax.combat.arsenal.profiles.variants import SCALAR_VARIANTS
from hytalegym.jax.combat.controllers.schema.contract import PHASE_IDLE
from hytalegym.jax.combat.controllers.schema.types import (
    RangedControllerCommands,
    RangedControllerRules,
    RangedControllerState,
)
from hytalegym.jax.combat.entities import ENTITY_CAPACITY


def hytale_0_5_7_ranged_rules(
    batch_size: int,
    weapon_family=None,
) -> RangedControllerRules:
    batch = _positive_size(batch_size)
    shape = (batch, ENTITY_CAPACITY)
    family = (
        jnp.zeros(shape, dtype=jnp.int32)
        if weapon_family is None
        else jnp.asarray(weapon_family, dtype=jnp.int32)
    )
    if family.shape != shape:
        raise ValueError(f"weapon_family must have shape {shape}")
    shortbow = _family_matches(family, SHORTBOW_FAMILY_IDS)
    crossbow = _family_matches(family, CROSSBOW_FAMILY_IDS)
    active = shortbow | crossbow
    max_durability = jnp.where(active, 120.0, 0.0).astype(jnp.float32)
    durability_loss = jnp.where(
        shortbow,
        0.58,
        jnp.where(crossbow, 0.28, 0.0),
    ).astype(jnp.float32)
    for variant in SCALAR_VARIANTS:
        if variant.template not in {"shortbow", "crossbow"}:
            continue
        selected = family == variant.family_id
        max_durability = jnp.where(
            selected,
            jnp.float32(variant.max_durability),
            max_durability,
        )
        durability_loss = jnp.where(
            selected,
            jnp.float32(variant.durability_loss_on_hit),
            durability_loss,
        )
    return RangedControllerRules(
        weapon_family=family,
        active=active,
        max_ammo=jnp.where(shortbow, 1, jnp.where(crossbow, 6, 0)).astype(
            jnp.int32
        ),
        max_durability=max_durability,
        durability_loss_per_projectile=durability_loss,
        signature_energy_cost=jnp.where(
            shortbow,
            6.0,
            jnp.where(crossbow, 5.0, 0.0),
        ).astype(jnp.float32),
    )


def empty_ranged_controller_state(
    batch_size: int,
    rules: RangedControllerRules,
    *,
    arrow_inventory=0,
    arrow_add_capacity=0,
) -> RangedControllerState:
    batch = _positive_size(batch_size)
    shape = (batch, ENTITY_CAPACITY)
    if rules.active.shape != shape:
        raise ValueError(f"rules must have shape {shape}")
    arrows = jnp.broadcast_to(
        jnp.asarray(arrow_inventory, dtype=jnp.int32),
        shape,
    )
    capacity = jnp.broadcast_to(
        jnp.asarray(arrow_add_capacity, dtype=jnp.int32),
        shape,
    )
    if bool(jnp.any(arrows < 0)) or bool(jnp.any(capacity < 0)):
        raise ValueError("arrow counts and add capacity must be nonnegative")
    return RangedControllerState(
        phase=jnp.full(shape, PHASE_IDLE, dtype=jnp.int32),
        phase_elapsed_seconds=jnp.zeros(shape, dtype=jnp.float32),
        primary_was_held=jnp.zeros(shape, dtype=jnp.bool_),
        primary_cooldown_seconds=jnp.zeros(shape, dtype=jnp.float32),
        pending_inventory_arrow=jnp.zeros(shape, dtype=jnp.bool_),
        arrow_inventory=arrows,
        arrow_add_capacity=capacity,
        ammo=rules.max_ammo,
        durability=rules.max_durability,
        signature_energy=jnp.zeros(shape, dtype=jnp.float32),
        signature_charges=jnp.zeros(shape, dtype=jnp.float32),
        failure_bits=jnp.zeros((batch,), dtype=jnp.uint32),
    )


def empty_ranged_controller_commands(
    batch_size: int,
) -> RangedControllerCommands:
    batch = _positive_size(batch_size)
    shape = (batch, ENTITY_CAPACITY)
    empty = jnp.zeros(shape, dtype=jnp.bool_)
    return RangedControllerCommands(
        primary_held=empty,
        signature_pressed=empty,
        reload_pressed=empty,
        secondary_pressed=empty,
        swap_away=empty,
        interrupted=empty,
    )


def _positive_size(value: int) -> int:
    if isinstance(value, bool):
        raise TypeError("batch_size must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError("batch_size must be an integer") from error
    if result <= 0:
        raise ValueError("batch_size must be positive")
    return result


def _family_matches(family, candidates) -> jnp.ndarray:
    result = jnp.zeros_like(family, dtype=jnp.bool_)
    for candidate in candidates:
        result |= family == jnp.int32(candidate)
    return result
