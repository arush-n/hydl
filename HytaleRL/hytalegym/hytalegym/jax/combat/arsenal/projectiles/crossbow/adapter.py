"""Adapt the runtime entity axis to the shared 32-slot Crossbow program."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal.projectiles.crossbow.types import (
    CrossbowProjectileImpactCommands,
)
from hytalegym.jax.combat.entities import (
    ENTITY_CAPACITY,
    empty_entity_roster,
)
from hytalegym.jax.combat.entities.interactions.schema.contract import (
    CROSSBOW_BIG_ARROW_DAMAGE,
    CROSSBOW_COMBO_2_EFFECT_ID,
    CROSSBOW_COMBO_DAMAGE,
    CROSSBOW_IMPACT_BIG_ARROW,
    CROSSBOW_IMPACT_CAPACITY,
    CROSSBOW_IMPACT_STANDARD,
    CROSSBOW_STANDARD_DAMAGE,
    INTERACTION_CAPABILITIES,
)
from hytalegym.jax.combat.entities.interactions.runtime.crossbow import (
    apply_crossbow_impacts,
)
from hytalegym.jax.combat.entities.interactions.schema.types import (
    CrossbowImpactCommands,
    CrossbowImpactInfo,
    EntityInteractionState,
)
from hytalegym.jax.combat.entities.schema.types import EntityCombatState
from hytalegym.jax.combat.mechanics import (
    CombatMechanicsRules,
    CombatMechanicsState,
    empty_mechanics_state,
)
from hytalegym.jax.combat.types import CombatState


def apply_crossbow_projectile_impacts(
    combat: CombatState,
    mechanics: CombatMechanicsState,
    rules: CombatMechanicsRules,
    commands: CrossbowProjectileImpactCommands,
    *,
    team_id: jax.Array,
    maximum_health: jax.Array,
) -> tuple[
    CombatState,
    CombatMechanicsState,
    CrossbowImpactInfo,
    CrossbowImpactCommands,
]:
    """Apply profile-exact hits through the existing target-local program.

    The shared entity interaction runtime has a 32-slot logical roster, while
    the compatibility Arsenal derives its entity count from runtime arrays.
    This adapter pads only inside an impact-taken branch and copies the first
    ``N`` rows back afterwards. It retains one Crossbow implementation rather
    than recreating combo, guard, status-clear, and force behavior here.
    """

    batch, entity_count = combat.health.shape
    if entity_count > ENTITY_CAPACITY:
        raise ValueError(
            "Crossbow impact entity axis exceeds the shared entity capacity"
        )
    if team_id.shape != (batch, entity_count):
        raise ValueError("team_id must match combat.health")
    if maximum_health.shape != (batch, entity_count):
        raise ValueError("maximum_health must match combat.health")
    _validate_command_layout(commands, batch)

    legacy = _legacy_commands(mechanics, commands, entity_count)
    empty_info = _empty_info(batch)

    def apply(_: None):
        padded_rules = _pad_rules(rules, entity_count)
        padded_mechanics = _pad_mechanics(
            mechanics,
            padded_rules,
            entity_count,
        )
        roster = empty_entity_roster(batch)
        active_slice = (slice(None), slice(0, entity_count))
        roster = roster._replace(
            semantic_id=roster.semantic_id.at[active_slice].set(
                jnp.arange(1, entity_count + 1, dtype=jnp.int32)[None, :]
            ),
            generation=roster.generation.at[active_slice].set(jnp.uint32(1)),
            team_id=roster.team_id.at[active_slice].set(team_id),
            position=roster.position.at[active_slice].set(combat.position),
            velocity=roster.velocity.at[active_slice].set(combat.velocity),
            yaw_degrees=roster.yaw_degrees.at[active_slice].set(combat.yaw),
            health=roster.health.at[active_slice].set(combat.health),
            max_health=roster.max_health.at[active_slice].set(maximum_health),
            damageable=roster.damageable.at[active_slice].set(True),
            dead=roster.dead.at[active_slice].set(combat.health <= 0.0),
            active=roster.active.at[active_slice].set(True),
        )
        state = EntityInteractionState(
            combat=EntityCombatState(
                roster=roster,
                mechanics=padded_mechanics,
            ),
            capability_bits=jnp.full(
                (batch,),
                INTERACTION_CAPABILITIES,
                dtype=jnp.uint32,
            ),
            failure_bits=jnp.zeros((batch,), dtype=jnp.uint32),
        )
        result, info = apply_crossbow_impacts(state, legacy, padded_rules)
        next_mechanics = _unpad_mechanics(
            result.combat.mechanics,
            mechanics,
            entity_count,
        )
        next_combat = combat._replace(
            health=result.combat.roster.health[:, :entity_count]
        )
        return next_combat, next_mechanics, info

    next_combat, next_mechanics, info = jax.lax.cond(
        jnp.any(legacy.requested),
        apply,
        lambda _: (combat, mechanics, empty_info),
        operand=None,
    )
    return next_combat, next_mechanics, info, legacy


def _legacy_commands(
    mechanics: CombatMechanicsState,
    commands: CrossbowProjectileImpactCommands,
    entity_count: int,
) -> CrossbowImpactCommands:
    requested = _pad_slots(commands.requested, False)
    kind = _pad_slots(commands.kind, CROSSBOW_IMPACT_STANDARD)
    source = _pad_slots(commands.source_slot, -1)
    target = _pad_slots(commands.target_slot, -1)
    yaw = _pad_slots(commands.knockback_yaw_degrees, 0.0)
    standard_damage = _pad_slots(commands.standard_damage, 0.0)
    combo_damage = _pad_slots(commands.combo_damage, 0.0)
    big_arrow_damage = _pad_slots(commands.big_arrow_damage, 0.0)

    safe_target = jnp.clip(target, 0, entity_count - 1)
    has_combo_2 = _target_has_effect(
        mechanics.statuses,
        safe_target,
        CROSSBOW_COMBO_2_EFFECT_ID,
    )
    standard = requested & (kind == CROSSBOW_IMPACT_STANDARD)
    big_arrow = requested & (kind == CROSSBOW_IMPACT_BIG_ARROW)
    combo_hit = standard & has_combo_2
    desired_damage = jnp.where(
        big_arrow,
        big_arrow_damage,
        jnp.where(combo_hit, combo_damage, standard_damage),
    )
    reference_damage = jnp.where(
        big_arrow,
        jnp.float32(CROSSBOW_BIG_ARROW_DAMAGE),
        jnp.where(
            combo_hit,
            jnp.float32(CROSSBOW_COMBO_DAMAGE),
            jnp.float32(CROSSBOW_STANDARD_DAMAGE),
        ),
    )
    finite = (
        jnp.isfinite(yaw)
        & jnp.isfinite(desired_damage)
        & (desired_damage >= 0.0)
    )
    overflow = jnp.any(
        requested
        & (
            ~finite
            | (source < 0)
            | (source >= entity_count)
            | (target < 0)
            | (target >= entity_count)
            | (~standard & ~big_arrow)
        ),
        axis=1,
    )
    return CrossbowImpactCommands(
        requested=requested,
        kind=kind,
        source_slot=source,
        source_generation=jnp.where(
            requested,
            jnp.uint32(1),
            jnp.uint32(0),
        ),
        target_slot=target,
        target_generation=jnp.where(
            requested,
            jnp.uint32(1),
            jnp.uint32(0),
        ),
        knockback_yaw_degrees=yaw,
        damage_multiplier=jnp.where(
            requested & finite,
            desired_damage / reference_damage,
            jnp.float32(1.0),
        ),
        overflow=overflow,
    )


def _pad_rules(
    rules: CombatMechanicsRules,
    entity_count: int,
) -> CombatMechanicsRules:
    def pad(value):
        if value.ndim < 2 or value.shape[1] != entity_count:
            return value
        if entity_count == ENTITY_CAPACITY:
            return value
        repeated = jnp.repeat(
            value[:, :1],
            ENTITY_CAPACITY - entity_count,
            axis=1,
        )
        return jnp.concatenate((value, repeated), axis=1)

    return jax.tree_util.tree_map(pad, rules)


def _pad_mechanics(
    mechanics: CombatMechanicsState,
    rules: CombatMechanicsRules,
    entity_count: int,
) -> CombatMechanicsState:
    baseline = empty_mechanics_state(mechanics.resources.shape[0], rules)

    def overlay(empty, current):
        if (
            current.ndim >= 2
            and current.shape[1] == entity_count
            and empty.ndim >= 2
            and empty.shape[1] == ENTITY_CAPACITY
        ):
            return empty.at[:, :entity_count].set(current)
        return current

    return jax.tree_util.tree_map(overlay, baseline, mechanics)


def _unpad_mechanics(
    padded: CombatMechanicsState,
    original: CombatMechanicsState,
    entity_count: int,
) -> CombatMechanicsState:
    def unpad(value, reference):
        if (
            reference.ndim >= 2
            and reference.shape[1] == entity_count
            and value.ndim >= 2
            and value.shape[1] == ENTITY_CAPACITY
        ):
            return value[:, :entity_count]
        return value

    return jax.tree_util.tree_map(unpad, padded, original)


def _pad_slots(value: jax.Array, fill: int | float | bool) -> jax.Array:
    if value.shape[1] > CROSSBOW_IMPACT_CAPACITY:
        raise ValueError("projectile axis exceeds Crossbow impact capacity")
    if value.shape[1] == CROSSBOW_IMPACT_CAPACITY:
        return value
    pad = CROSSBOW_IMPACT_CAPACITY - value.shape[1]
    return jnp.pad(
        value,
        ((0, 0), (0, pad)),
        constant_values=fill,
    )


def _target_has_effect(status, target_slot, effect_id: int) -> jax.Array:
    batch = jnp.arange(target_slot.shape[0], dtype=jnp.int32)[:, None]
    active = status.active[batch, target_slot]
    identifiers = status.effect_id[batch, target_slot]
    return jnp.any(
        active & (identifiers == jnp.int32(effect_id)),
        axis=2,
    )


def _empty_info(batch: int) -> CrossbowImpactInfo:
    shape = (batch, CROSSBOW_IMPACT_CAPACITY)
    return CrossbowImpactInfo(
        combo_stage=jnp.full(shape, -1, dtype=jnp.int32),
        damage_applied=jnp.zeros(shape, dtype=jnp.float32),
        blocked=jnp.zeros(shape, dtype=jnp.bool_),
        invulnerable=jnp.zeros(shape, dtype=jnp.bool_),
        combo_1_applied=jnp.zeros(shape, dtype=jnp.bool_),
        combo_2_applied=jnp.zeros(shape, dtype=jnp.bool_),
        combo_cleared=jnp.zeros(shape, dtype=jnp.bool_),
        gameplay_regen_cleared=jnp.zeros(shape, dtype=jnp.bool_),
        signature_energy_gained=jnp.zeros(shape, dtype=jnp.float32),
        knockback_velocity=jnp.zeros(shape + (3,), dtype=jnp.float32),
        failure_bits=jnp.zeros((batch,), dtype=jnp.uint32),
        valid=jnp.ones((batch,), dtype=jnp.bool_),
    )


def _validate_command_layout(
    commands: CrossbowProjectileImpactCommands,
    batch: int,
) -> None:
    width = commands.requested.shape[1]
    if commands.requested.shape != (batch, width):
        raise ValueError("requested must be [B,P]")
    if width <= 0 or width > CROSSBOW_IMPACT_CAPACITY:
        raise ValueError("projectile impact width is outside capacity")
    for name in commands._fields[1:]:
        if getattr(commands, name).shape != (batch, width):
            raise ValueError(f"{name} must match requested")
