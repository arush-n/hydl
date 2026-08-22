"""Ordered Hytale 0.5.7 crossbow impact interaction chains."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal.schema.contract import FORCE_ADD, FORCE_SET
from hytalegym.jax.combat.entities import ENTITY_CAPACITY
from hytalegym.jax.combat.entities.interactions.schema.contract import (
    CROSSBOW_AIR_RESISTANCE,
    CROSSBOW_AIR_RESISTANCE_MAX,
    CROSSBOW_BIG_ARROW_DAMAGE,
    CROSSBOW_BIG_ARROW_FORCE,
    CROSSBOW_COMBO_1_EFFECT_ID,
    CROSSBOW_COMBO_2_EFFECT_ID,
    CROSSBOW_COMBO_DAMAGE,
    CROSSBOW_COMBO_DURATION_SECONDS,
    CROSSBOW_COMBO_FORCE,
    CROSSBOW_COMBO_NONE,
    CROSSBOW_COMBO_ONE,
    CROSSBOW_COMBO_TWO,
    CROSSBOW_FORCE_DIRECTION,
    CROSSBOW_GROUND_RESISTANCE,
    CROSSBOW_GROUND_RESISTANCE_MAX,
    CROSSBOW_IMPACT_BIG_ARROW,
    CROSSBOW_IMPACT_STANDARD,
    CROSSBOW_RESISTANCE_STYLE_LINEAR,
    CROSSBOW_RESISTANCE_THRESHOLD,
    CROSSBOW_STANDARD_DAMAGE,
    CROSSBOW_STANDARD_FORCE,
    INTERACTION_FAILURE_IMPACT_OVERFLOW,
    INTERACTION_FAILURE_INVALID_COMMAND,
    INTERACTION_FAILURE_INVALID_STATE,
    INTERACTION_FAILURE_MECHANICS,
    INTERACTION_FAILURE_STALE_HANDLE,
    INTERACTION_FAILURE_UNSUPPORTED_TARGET,
    INTERACTION_FAILURE_UPSTREAM,
    POTION_REGEN_EFFECT_IDS,
)
from hytalegym.jax.combat.entities.interactions.runtime.status import (
    clear_target_effects,
    target_has_effect,
)
from hytalegym.jax.combat.entities.interactions.schema.types import (
    CrossbowImpactCommands,
    CrossbowImpactEvents,
    CrossbowImpactInfo,
    EntityInteractionState,
)
from hytalegym.jax.combat.entities.interactions.schema.validation import (
    invalid_crossbow_commands,
    invalid_interaction_rows,
    validate_interaction_layout,
)
from hytalegym.jax.combat.mechanics import (
    DAMAGE_CLASS_CHARGED,
    DAMAGE_CLASS_LIGHT,
    DAMAGE_CLASS_SIGNATURE,
    DAMAGE_PROJECTILE,
    RESOURCE_SIGNATURE_ENERGY,
    STATUS_OVERLAP_IGNORE,
    CombatMechanicsRules,
    apply_damage_forces,
    apply_statuses,
    empty_damage_events,
    empty_status_applications,
    resolve_damage_events,
)


def apply_crossbow_impacts(
    state: EntityInteractionState,
    commands: CrossbowImpactCommands,
    rules: CombatMechanicsRules,
) -> tuple[EntityInteractionState, CrossbowImpactInfo]:
    """Apply command-index-ordered standard and Big Arrow impacts.

    Hit production and projectile/world collision are injected. This function
    owns only the target-local interaction chain after a certified hit.
    """

    validate_interaction_layout(state, commands)
    original = state
    roster = state.combat.roster
    requested = commands.requested
    source = jnp.clip(commands.source_slot, 0, ENTITY_CAPACITY - 1)
    target = jnp.clip(commands.target_slot, 0, ENTITY_CAPACITY - 1)
    source_active = _gather(roster.active, source)
    target_active = _gather(roster.active, target)
    source_generation = _gather(roster.generation, source)
    target_generation = _gather(roster.generation, target)
    handles_valid = (
        source_active
        & target_active
        & (source_generation == commands.source_generation)
        & (target_generation == commands.target_generation)
    )
    target_supported = (
        _gather(roster.damageable, target)
        & ~_gather(roster.intangible, target)
        & ~_gather(roster.dead, target)
    )
    bits = state.failure_bits
    bits = _set_failure(
        bits,
        invalid_interaction_rows(state),
        INTERACTION_FAILURE_INVALID_STATE,
    )
    bits = _set_failure(
        bits,
        invalid_crossbow_commands(commands),
        INTERACTION_FAILURE_INVALID_COMMAND,
    )
    bits = _set_failure(
        bits,
        commands.overflow,
        INTERACTION_FAILURE_IMPACT_OVERFLOW,
    )
    bits = _set_failure(
        bits,
        jnp.any(requested & ~handles_valid, axis=1),
        INTERACTION_FAILURE_STALE_HANDLE,
    )
    bits = _set_failure(
        bits,
        jnp.any(requested & handles_valid & ~target_supported, axis=1),
        INTERACTION_FAILURE_UNSUPPORTED_TARGET,
    )
    upstream = (
        state.combat.roster.failure_bits != jnp.uint32(0)
    ) | (state.combat.mechanics.failure_bits != jnp.uint32(0))
    bits = _set_failure(
        bits,
        upstream,
        INTERACTION_FAILURE_UPSTREAM,
    )
    row_enabled = bits == jnp.uint32(0)
    events = CrossbowImpactEvents(
        *(getattr(commands, name) for name in CrossbowImpactEvents._fields)
    )
    inputs = jax.tree_util.tree_map(
        lambda value: jnp.moveaxis(value, 1, 0),
        events,
    )

    def apply_one(combat, command):
        enabled = command.requested & row_enabled
        target_slot = jnp.clip(
            command.target_slot, 0, ENTITY_CAPACITY - 1
        )
        source_slot = jnp.clip(
            command.source_slot, 0, ENTITY_CAPACITY - 1
        )
        standard = enabled & (
            command.kind == CROSSBOW_IMPACT_STANDARD
        )
        big_arrow = enabled & (
            command.kind == CROSSBOW_IMPACT_BIG_ARROW
        )
        has_combo_1 = target_has_effect(
            combat.mechanics.statuses,
            target_slot,
            CROSSBOW_COMBO_1_EFFECT_ID,
        )
        has_combo_2 = target_has_effect(
            combat.mechanics.statuses,
            target_slot,
            CROSSBOW_COMBO_2_EFFECT_ID,
        )
        combo_stage = jnp.where(
            has_combo_2,
            CROSSBOW_COMBO_TWO,
            jnp.where(
                has_combo_1,
                CROSSBOW_COMBO_ONE,
                CROSSBOW_COMBO_NONE,
            ),
        )
        apply_combo_1 = standard & (
            combo_stage == CROSSBOW_COMBO_NONE
        )
        apply_combo_2 = standard & (
            combo_stage == CROSSBOW_COMBO_ONE
        )
        mechanics = _apply_combo_effect(
            combat.mechanics,
            target_slot,
            source_slot,
            apply_combo_1,
            apply_combo_2,
        )

        combo_hit = standard & (
            combo_stage == CROSSBOW_COMBO_TWO
        )
        damage = jnp.where(
            big_arrow,
            CROSSBOW_BIG_ARROW_DAMAGE,
            jnp.where(
                combo_hit,
                CROSSBOW_COMBO_DAMAGE,
                CROSSBOW_STANDARD_DAMAGE,
            ),
        )
        force = jnp.where(
            big_arrow,
            CROSSBOW_BIG_ARROW_FORCE,
            jnp.where(
                combo_hit,
                CROSSBOW_COMBO_FORCE,
                CROSSBOW_STANDARD_FORCE,
            ),
        )
        force_mode = jnp.where(
            standard & ~combo_hit,
            FORCE_ADD,
            FORCE_SET,
        )
        damage_class = jnp.where(
            big_arrow,
            DAMAGE_CLASS_SIGNATURE,
            jnp.where(
                combo_hit,
                DAMAGE_CLASS_CHARGED,
                DAMAGE_CLASS_LIGHT,
            ),
        )
        invulnerable_component = _gather(
            combat.roster.invulnerable,
            target_slot,
        )
        already_dead = _gather(combat.roster.dead, target_slot)
        damage_requested = enabled & ~invulnerable_component & ~already_dead
        velocity = _force_velocity(
            command.knockback_yaw_degrees,
            force,
        )
        events = empty_damage_events(
            combat.roster.active.shape[0],
            1,
        )._replace(
            requested=damage_requested[:, None],
            source_entity_id=source_slot[:, None],
            target_entity_id=target_slot[:, None],
            amount=jnp.where(
                damage_requested,
                damage * command.damage_multiplier,
                0.0,
            )[:, None],
            damage_class=damage_class[:, None],
            cause=jnp.full(
                (damage.shape[0], 1),
                DAMAGE_PROJECTILE,
                dtype=jnp.int32,
            ),
            knockback_velocity=velocity[:, None, :],
            force_mode=force_mode[:, None],
            air_resistance=jnp.full(
                (damage.shape[0], 1),
                CROSSBOW_AIR_RESISTANCE,
                dtype=jnp.float32,
            ),
            air_resistance_max=jnp.full(
                (damage.shape[0], 1),
                CROSSBOW_AIR_RESISTANCE_MAX,
                dtype=jnp.float32,
            ),
            ground_resistance=jnp.full(
                (damage.shape[0], 1),
                CROSSBOW_GROUND_RESISTANCE,
                dtype=jnp.float32,
            ),
            ground_resistance_max=jnp.full(
                (damage.shape[0], 1),
                CROSSBOW_GROUND_RESISTANCE_MAX,
                dtype=jnp.float32,
            ),
            resistance_threshold=jnp.full(
                (damage.shape[0], 1),
                CROSSBOW_RESISTANCE_THRESHOLD,
                dtype=jnp.float32,
            ),
            resistance_style=jnp.full(
                (damage.shape[0], 1),
                CROSSBOW_RESISTANCE_STYLE_LINEAR,
                dtype=jnp.int32,
            ),
            on_hit_resource_id=jnp.where(
                combo_hit,
                RESOURCE_SIGNATURE_ENERGY,
                -1,
            )[:, None],
            on_hit_resource_delta=jnp.where(
                combo_hit,
                1.0,
                0.0,
            )[:, None],
        )
        mechanics, health, resolution = resolve_damage_events(
            mechanics,
            combat.roster.health,
            combat.roster.position,
            combat.roster.yaw_degrees,
            events,
            rules,
        )
        mechanics = apply_damage_forces(mechanics, resolution)
        resolved = resolution.requested[:, 0]
        chain_succeeded = (
            resolved
            & ~resolution.blocked[:, 0]
            & ~resolution.invulnerable[:, 0]
        )
        clear_combo = chain_succeeded & combo_hit
        clear_regen = chain_succeeded & (
            big_arrow | (standard & ~combo_hit)
        )
        statuses, combo_cleared = clear_target_effects(
            mechanics.statuses,
            target_slot,
            (
                CROSSBOW_COMBO_1_EFFECT_ID,
                CROSSBOW_COMBO_2_EFFECT_ID,
            ),
            clear_combo,
        )
        statuses, regen_cleared = clear_target_effects(
            statuses,
            target_slot,
            POTION_REGEN_EFFECT_IDS,
            clear_regen,
        )
        mechanics = mechanics._replace(statuses=statuses)
        newly_dead = (
            combat.roster.active
            & combat.roster.damageable
            & ~combat.roster.dead
            & (health <= 0.0)
        )
        next_combat = combat._replace(
            roster=combat.roster._replace(
                health=health,
                dead=combat.roster.dead | newly_dead,
            ),
            mechanics=mechanics,
        )
        output = (
            jnp.where(standard, combo_stage, -1),
            resolution.applied_damage[:, 0],
            resolution.blocked[:, 0],
            (
                resolution.invulnerable[:, 0]
                | (enabled & invulnerable_component)
            ),
            apply_combo_1,
            apply_combo_2,
            combo_cleared,
            regen_cleared,
            resolution.on_hit_resource_applied[:, 0],
            resolution.knockback_velocity[:, 0],
        )
        return next_combat, output

    candidate, output = jax.lax.scan(
        apply_one,
        state.combat,
        inputs,
    )
    mechanics_failed = (
        candidate.mechanics.failure_bits != jnp.uint32(0)
    )
    bits = _set_failure(
        bits,
        mechanics_failed,
        INTERACTION_FAILURE_MECHANICS,
    )
    valid = bits == jnp.uint32(0)
    combat = _select_tree(valid, candidate, original.combat)
    result = EntityInteractionState(
        combat=combat,
        capability_bits=state.capability_bits,
        failure_bits=bits,
    )
    (
        combo_stage,
        damage_applied,
        blocked,
        invulnerable,
        combo_1_applied,
        combo_2_applied,
        combo_cleared,
        regen_cleared,
        signature_energy,
        knockback,
    ) = jax.tree_util.tree_map(
        lambda value: jnp.moveaxis(value, 0, 1),
        output,
    )
    event_mask = valid[:, None]
    return result, CrossbowImpactInfo(
        combo_stage=jnp.where(event_mask, combo_stage, -1),
        damage_applied=jnp.where(event_mask, damage_applied, 0.0),
        blocked=blocked & event_mask,
        invulnerable=invulnerable & event_mask,
        combo_1_applied=combo_1_applied & event_mask,
        combo_2_applied=combo_2_applied & event_mask,
        combo_cleared=combo_cleared & event_mask,
        gameplay_regen_cleared=regen_cleared & event_mask,
        signature_energy_gained=jnp.where(
            event_mask, signature_energy, 0.0
        ),
        knockback_velocity=jnp.where(
            event_mask[..., None],
            knockback,
            0.0,
        ),
        failure_bits=bits,
        valid=valid,
    )


def _apply_combo_effect(
    mechanics,
    target,
    source,
    apply_combo_1,
    apply_combo_2,
):
    requested = apply_combo_1 | apply_combo_2
    target_mask = (
        jax.nn.one_hot(
            target,
            ENTITY_CAPACITY,
            dtype=jnp.bool_,
        )
        & requested[:, None]
    )
    applications = empty_status_applications(
        target.shape[0],
        entity_count=ENTITY_CAPACITY,
    )
    slot_mask = target_mask[..., None] & (
        jnp.arange(applications.requested.shape[2]) == 0
    )
    effect = jnp.where(
        apply_combo_1,
        CROSSBOW_COMBO_1_EFFECT_ID,
        CROSSBOW_COMBO_2_EFFECT_ID,
    )
    applications = applications._replace(
        requested=slot_mask,
        effect_id=jnp.where(slot_mask, effect[:, None, None], 0),
        source_entity_id=jnp.where(
            slot_mask, source[:, None, None], -1
        ),
        duration_seconds=jnp.where(
            slot_mask,
            CROSSBOW_COMBO_DURATION_SECONDS,
            0.0,
        ),
        damage_cause=jnp.where(
            slot_mask,
            DAMAGE_PROJECTILE,
            0,
        ),
        speed_multiplier=jnp.ones_like(
            applications.speed_multiplier
        ),
        overlap_mode=jnp.where(
            slot_mask,
            STATUS_OVERLAP_IGNORE,
            0,
        ),
    )
    return apply_statuses(mechanics, applications)


def _force_velocity(yaw_degrees, force):
    direction = jnp.asarray(
        CROSSBOW_FORCE_DIRECTION,
        dtype=jnp.float32,
    )
    direction /= jnp.linalg.norm(direction)
    radians = jnp.deg2rad(yaw_degrees)
    world_x = (
        direction[0] * jnp.cos(radians)
        + direction[2] * jnp.sin(radians)
    )
    world_z = (
        -direction[0] * jnp.sin(radians)
        + direction[2] * jnp.cos(radians)
    )
    return (
        jnp.stack(
            (
                world_x,
                jnp.broadcast_to(direction[1], world_x.shape),
                world_z,
            ),
            axis=1,
        )
        * force[:, None]
    )


def _gather(array, entity_id):
    batch = jnp.arange(array.shape[0]).reshape(
        (array.shape[0],) + (1,) * (entity_id.ndim - 1)
    )
    return array[batch, entity_id]


def _set_failure(bits, mask, code):
    return jnp.where(mask, bits | jnp.uint32(code), bits)


def _select_tree(mask, selected, fallback):
    return jax.tree_util.tree_map(
        lambda yes, no: jnp.where(
            mask.reshape((mask.shape[0],) + (1,) * (yes.ndim - 1)),
            yes,
            no,
        ),
        selected,
        fallback,
    )
