"""Pure-JAX Hytale 0.5.7 resource, defense, damage, and status kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.mechanics.schema.contract import (
    DAMAGE_CLASS_COUNT,
    DAMAGE_COUNT,
    MECHANICS_FAILURE_INVALID_COMMAND,
    RESOURCE_COUNT,
    RESOURCE_STAMINA,
    STAMINA_BROKEN_REGEN_DELAY_SECONDS,
    STATUS_FLAG_INVULNERABLE,
    STATUS_FLAG_IGNORE_KNOCKBACK,
)
from hytalegym.jax.combat.mechanics.schema.types import (
    CombatMechanicsRules,
    CombatMechanicsState,
    DamageEvents,
    DamageResolution,
)
from hytalegym.jax.combat.types import CombatParams

# Re-exported to preserve this module's attribute surface.
from hytalegym.jax.combat.mechanics.runtime.access import (  # noqa: F401
    _dodge_local_direction,
    _entity_dt,
    _gather_entity,
    _gather_entity_cause,
    _gather_entity_class,
    _select_tree,
    _set_entity,
    _set_failure,
    _write_new,
    _write_new_cause,
    clear_statuses,
    damage_resistance_modifiers,
    guard_resource_available,
    status_modifiers,
)
from hytalegym.jax.combat.mechanics.runtime.motion import (  # noqa: F401
    _set_entity_where,
    advance_defense_interactions,
    apply_defense_commands,
    damp_applied_velocity,
    damp_applied_motion,
    native_npc_null_config_force_velocity,
    project_applied_motion_velocity,
    resolve_applied_motion_velocity,
    resolve_dodge_launch_velocity,
    translate_applied_motion,
)
from hytalegym.jax.combat.mechanics.runtime.statuses import (  # noqa: F401
    _resolve_damage_resistance_amount,
    apply_resource_delta,
    apply_statuses,
    tick_resources,
    tick_statuses,
)


def advance_applied_motion(
    state: CombatMechanicsState,
    position: jax.Array,
    grounded: jax.Array,
    dt_seconds: jax.Array,
    rules: CombatMechanicsRules,
    params: CombatParams,
) -> tuple[CombatMechanicsState, jax.Array]:
    """Translate then damp when the caller already knows post-move grounding."""

    translated, candidate_position = translate_applied_motion(
        state,
        position,
        grounded,
        dt_seconds,
        params,
    )
    return (
        damp_applied_motion(
            translated,
            grounded,
            dt_seconds,
            rules,
            params,
        ),
        candidate_position,
    )


# Backward-readable name for the exact dodge use of the generic applied force.
advance_dodge_motion = advance_applied_motion


def resolve_damage_events(
    state: CombatMechanicsState,
    health: jax.Array,
    position: jax.Array,
    yaw_degrees: jax.Array,
    events: DamageEvents,
    rules: CombatMechanicsRules,
    *,
    random_keys: jax.Array | None = None,
) -> tuple[
    CombatMechanicsState,
    jax.Array,
    DamageResolution,
]:
    """Resolve ordered damage with native pre-block stamina drain semantics."""

    batch, entity_count = state.resources.shape[:2]
    if random_keys is not None and random_keys.shape[0] != batch:
        raise ValueError("random_keys must match the damage-event batch")
    requested = events.requested
    (
        resistance_present,
        resistance_inherits,
        resistance_flat,
        resistance_multiplier,
    ) = damage_resistance_modifiers(state, rules)
    force_present = jnp.sum(events.knockback_velocity**2, axis=2) > 0.0
    force_valid = (
        ((events.force_mode == 0) | (events.force_mode == 1))
        & jnp.isfinite(events.air_resistance)
        & jnp.isfinite(events.air_resistance_max)
        & jnp.isfinite(events.ground_resistance)
        & jnp.isfinite(events.ground_resistance_max)
        & jnp.isfinite(events.resistance_threshold)
        & (events.air_resistance >= 0.0)
        & (events.air_resistance <= 1.0)
        & (events.air_resistance_max >= 0.0)
        & (events.air_resistance_max <= 1.0)
        & (events.ground_resistance >= 0.0)
        & (events.ground_resistance <= 1.0)
        & (events.ground_resistance_max >= 0.0)
        & (events.ground_resistance_max <= 1.0)
        & (events.resistance_threshold > 0.0)
        & ((events.resistance_style == 0) | (events.resistance_style == 1))
    )
    event_valid = (
        (events.source_entity_id >= -1)
        & (events.source_entity_id < entity_count)
        & (events.target_entity_id >= 0)
        & (events.target_entity_id < entity_count)
        & jnp.isfinite(events.amount)
        & (events.amount >= 0.0)
        & jnp.isfinite(events.random_percentage)
        & (events.random_percentage >= 0.0)
        & (events.damage_class >= 0)
        & (events.damage_class < DAMAGE_CLASS_COUNT)
        & (events.cause >= 0)
        & (events.cause < DAMAGE_COUNT)
        & jnp.isfinite(events.stamina_drain_multiplier)
        & (events.stamina_drain_multiplier >= 0.0)
        & jnp.all(jnp.isfinite(events.knockback_velocity), axis=2)
        & (~force_present | force_valid)
        & (events.on_hit_resource_id >= -1)
        & (events.on_hit_resource_id < RESOURCE_COUNT)
        & jnp.isfinite(events.on_hit_resource_delta)
        & ((events.on_hit_resource_id < 0) | (events.source_entity_id >= 0))
        & jnp.isfinite(events.on_hit_healing)
        & (events.on_hit_healing >= 0.0)
        & ((events.on_hit_healing <= 0.0) | (events.source_entity_id >= 0))
    )
    state_invalid = (
        ~jnp.all(jnp.isfinite(health), axis=1)
        | ~jnp.all(jnp.isfinite(position), axis=(1, 2))
        | ~jnp.all(jnp.isfinite(yaw_degrees), axis=1)
        | ~jnp.all(jnp.isfinite(resistance_flat), axis=(1, 2))
        | ~jnp.all(jnp.isfinite(resistance_multiplier), axis=(1, 2))
        | ~jnp.all(
            jnp.isfinite(rules.equipment_damage_class_flat),
            axis=(1, 2),
        )
        | ~jnp.all(
            jnp.isfinite(rules.equipment_damage_class_multiplier),
            axis=(1, 2),
        )
    )
    invalid = state_invalid | jnp.any(requested & ~event_valid, axis=1)
    if random_keys is None:
        invalid |= jnp.any(
            requested & (events.random_percentage > 0.0),
            axis=1,
        )
    if random_keys is None:
        randomized_amount = events.amount
    else:
        unit_draw = jax.vmap(
            lambda key: jax.random.uniform(
                key,
                (events.amount.shape[1],),
                dtype=jnp.float32,
            )
        )(random_keys)
        symmetric_draw = jnp.float32(2.0) * unit_draw - jnp.float32(1.0)
        randomized_amount = events.amount * (
            jnp.float32(1.0) + events.random_percentage * symmetric_draw
        )
    source = jnp.clip(events.source_entity_id, 0, entity_count - 1)
    damage_class = jnp.clip(events.damage_class, 0, DAMAGE_CLASS_COUNT - 1)
    source_valid = events.source_entity_id >= 0
    class_flat = _gather_entity_class(
        rules.equipment_damage_class_flat,
        source,
        damage_class,
    )
    class_multiplier = _gather_entity_class(
        rules.equipment_damage_class_multiplier,
        source,
        damage_class,
    )
    class_flat = jnp.where(source_valid, class_flat, jnp.float32(0.0))
    class_multiplier = jnp.where(
        source_valid,
        class_multiplier,
        jnp.float32(0.0),
    )
    enhanced_amount = (randomized_amount + class_flat) * jnp.maximum(
        jnp.float32(0.0),
        jnp.float32(1.0) + class_multiplier,
    )
    invalid |= jnp.any(requested & ~jnp.isfinite(enhanced_amount), axis=1)
    bits = _set_failure(
        state.failure_bits,
        invalid,
        MECHANICS_FAILURE_INVALID_COMMAND,
    )
    row_valid = bits == jnp.uint32(0)
    initial_state = state
    initial_health = health
    resolved_events = events._replace(amount=enhanced_amount)
    inputs = jax.tree_util.tree_map(
        lambda value: jnp.moveaxis(value, 1, 0),
        resolved_events,
    )

    def resolve_one(
        carry: tuple[CombatMechanicsState, jax.Array],
        event,
    ):
        current, current_health = carry
        active = event.requested & row_valid
        target = jnp.clip(event.target_entity_id, 0, entity_count - 1)
        source = jnp.clip(event.source_entity_id, 0, entity_count - 1)
        victim_guard = _gather_entity(current.guard_active, target)
        victim_broken = _gather_entity(current.stamina_broken, target)
        cause_mask = _gather_entity_cause(
            rules.guard_cause_mask,
            target,
            event.cause,
        )
        source_valid = event.source_entity_id >= 0
        source_position = _gather_entity(position, source)
        victim_position = _gather_entity(position, target)
        victim_yaw = _gather_entity(yaw_degrees, target)
        offset = source_position - victim_position
        horizontal_length = jnp.hypot(offset[:, 0], offset[:, 2])
        safe_length = jnp.maximum(
            horizontal_length,
            jnp.finfo(jnp.float32).tiny,
        )
        radians = jnp.deg2rad(victim_yaw)
        heading_x = -jnp.sin(radians)
        heading_z = -jnp.cos(radians)
        dot = (
            heading_x * offset[:, 0] / safe_length
            + heading_z * offset[:, 2] / safe_length
        )
        half_angle = _gather_entity(
            rules.guard_half_angle_degrees,
            target,
        )
        frontal = (horizontal_length > 0.0) & (dot > jnp.cos(jnp.deg2rad(half_angle)))
        status_flags, _ = status_modifiers(current.statuses)
        victim_flags = _gather_entity(status_flags, target)
        dodge_invulnerable = (
            _gather_entity(
                current.dodge_invulnerability_remaining_seconds,
                target,
            )
            > 0.0
        )
        invulnerable = active & (
            dodge_invulnerable | ((victim_flags & STATUS_FLAG_INVULNERABLE) != 0)
        )
        blocked = (
            active
            & ~invulnerable
            & victim_guard
            & ~victim_broken
            & source_valid
            & cause_mask
            & frontal
        )
        damage_modifier = _gather_entity_cause(
            rules.guard_damage_modifier,
            target,
            event.cause,
        )
        filtered = jnp.where(
            active & ~invulnerable,
            event.amount * jnp.where(blocked, damage_modifier, 1.0),
            jnp.float32(0.0),
        )
        filtered = _resolve_damage_resistance_amount(
            filtered,
            active & ~invulnerable,
            target,
            event.cause,
            rules,
            resistance_present,
            resistance_inherits,
            resistance_flat,
            resistance_multiplier,
        )
        rounded = jnp.floor(filtered + jnp.float32(0.5))
        before = _gather_entity(current_health, target)
        after = jnp.maximum(0.0, before - rounded)
        actual = before - after
        landed = active & (actual > 0.0)
        current_health = _set_entity(current_health, target, after)

        stamina_value = jnp.maximum(
            _gather_entity(rules.guard_stamina_value, target),
            jnp.finfo(jnp.float32).tiny,
        )
        stamina_spent = jnp.where(
            blocked,
            (event.amount / stamina_value * event.stamina_drain_multiplier),
            jnp.float32(0.0),
        )
        stamina = _gather_entity(
            current.resources[..., RESOURCE_STAMINA],
            target,
        )
        stamina_minimum = _gather_entity(
            rules.resource_minimum[..., RESOURCE_STAMINA],
            target,
        )
        stamina = jnp.maximum(stamina_minimum, stamina - stamina_spent)
        resources = current.resources.at[..., RESOURCE_STAMINA].set(
            _set_entity(
                current.resources[..., RESOURCE_STAMINA],
                target,
                stamina,
            )
        )
        broke = blocked & (stamina <= 0.0)
        broken = _set_entity(
            current.stamina_broken,
            target,
            _gather_entity(current.stamina_broken, target) | broke,
        )
        guard_active = _set_entity(
            current.guard_active,
            target,
            _gather_entity(current.guard_active, target) & ~broke,
        )
        guard_held = _set_entity(
            current.guard_held,
            target,
            _gather_entity(current.guard_held, target) & ~broke,
        )
        guard_windup = _set_entity(
            current.guard_windup_elapsed_seconds,
            target,
            jnp.where(
                broke,
                jnp.float32(0.0),
                _gather_entity(
                    current.guard_windup_elapsed_seconds,
                    target,
                ),
            ),
        )
        old_delay = _gather_entity(
            current.stamina_regen_delay_seconds,
            target,
        )
        delay = _set_entity(
            current.stamina_regen_delay_seconds,
            target,
            jnp.where(
                broke,
                jnp.minimum(
                    old_delay,
                    jnp.float32(STAMINA_BROKEN_REGEN_DELAY_SECONDS),
                ),
                old_delay,
            ),
        )
        current = current._replace(
            resources=resources,
            stamina_broken=broken,
            guard_held=guard_held,
            guard_active=guard_active,
            guard_windup_elapsed_seconds=guard_windup,
            stamina_regen_delay_seconds=delay,
        )
        resource_requested = landed & (event.on_hit_resource_id >= 0)
        reward_resource_id = jnp.clip(
            event.on_hit_resource_id,
            0,
            RESOURCE_COUNT - 1,
        )
        source_resource = current.resources[
            jnp.arange(current.resources.shape[0]),
            source,
            reward_resource_id,
        ]
        source_minimum = rules.resource_minimum[
            jnp.arange(rules.resource_minimum.shape[0]),
            source,
            reward_resource_id,
        ]
        source_maximum = rules.resource_maximum[
            jnp.arange(rules.resource_maximum.shape[0]),
            source,
            reward_resource_id,
        ]
        rewarded = jnp.clip(
            source_resource + event.on_hit_resource_delta,
            source_minimum,
            source_maximum,
        )
        applied_reward = jnp.where(
            resource_requested,
            rewarded - source_resource,
            jnp.float32(0.0),
        )
        on_hit_healing = jnp.where(
            landed,
            event.on_hit_healing,
            jnp.float32(0.0),
        )
        resources = current.resources.at[
            jnp.arange(current.resources.shape[0]),
            source,
            reward_resource_id,
        ].set(jnp.where(resource_requested, rewarded, source_resource))
        current = current._replace(resources=resources)
        knockback_modifier = _gather_entity_cause(
            rules.guard_knockback_modifier,
            target,
            event.cause,
        )
        ignore_knockback = (
            victim_flags & jnp.uint32(STATUS_FLAG_IGNORE_KNOCKBACK)
        ) != 0
        knockback = (
            event.knockback_velocity
            * jnp.where(
                blocked,
                knockback_modifier,
                jnp.float32(1.0),
            )[:, None]
        )
        knockback = jnp.where(
            (active & ~invulnerable & ~ignore_knockback)[:, None],
            knockback,
            jnp.float32(0.0),
        )
        output = (
            active,
            event.source_entity_id,
            actual,
            blocked,
            invulnerable,
            stamina_spent,
            knockback,
            event.target_entity_id,
            event.force_mode,
            event.air_resistance,
            event.air_resistance_max,
            event.ground_resistance,
            event.ground_resistance_max,
            event.resistance_threshold,
            event.resistance_style,
            event.dampen_y,
            event.on_hit_resource_id,
            event.on_hit_resource_delta,
            applied_reward,
            on_hit_healing,
        )
        return (current, current_health), output

    (candidate_state, candidate_health), outputs = jax.lax.scan(
        resolve_one,
        (state._replace(failure_bits=bits), health),
        inputs,
    )
    candidate_state = candidate_state._replace(failure_bits=bits)
    result_state = _select_tree(
        row_valid,
        candidate_state,
        initial_state,
    )._replace(failure_bits=bits)
    result_health = jnp.where(
        row_valid[:, None],
        candidate_health,
        initial_health,
    )
    outputs = jax.tree_util.tree_map(
        lambda value: jnp.moveaxis(value, 0, 1),
        outputs,
    )
    resolution = DamageResolution(
        requested=outputs[0],
        source_entity_id=outputs[1],
        applied_damage=outputs[2],
        blocked=outputs[3],
        invulnerable=outputs[4],
        stamina_spent=outputs[5],
        knockback_velocity=outputs[6],
        target_entity_id=outputs[7],
        force_mode=outputs[8],
        air_resistance=outputs[9],
        air_resistance_max=outputs[10],
        ground_resistance=outputs[11],
        ground_resistance_max=outputs[12],
        resistance_threshold=outputs[13],
        resistance_style=outputs[14],
        dampen_y=outputs[15],
        on_hit_resource_id=outputs[16],
        on_hit_resource_delta=outputs[17],
        on_hit_resource_applied=outputs[18],
        on_hit_healing=outputs[19],
    )
    return result_state, result_health, resolution


def apply_damage_forces(
    state: CombatMechanicsState,
    resolution: DamageResolution,
) -> CombatMechanicsState:
    """Apply ordered surviving knockback events to the configured velocity."""

    entity_count = state.resources.shape[1]
    inputs = jax.tree_util.tree_map(
        lambda value: jnp.moveaxis(value, 1, 0),
        resolution,
    )

    def apply_one(current: CombatMechanicsState, event):
        active = event.requested & (jnp.sum(event.knockback_velocity**2, axis=1) > 0.0)
        target = jnp.clip(event.target_entity_id, 0, entity_count - 1)
        previous = _gather_entity(current.applied_velocity, target)
        velocity = jnp.where(
            event.force_mode[:, None] == 0,
            event.knockback_velocity,
            previous + event.knockback_velocity,
        )
        current = current._replace(
            applied_velocity=_set_entity_where(
                current.applied_velocity,
                target,
                velocity,
                active,
            ),
            applied_velocity_can_clear=_set_entity_where(
                current.applied_velocity_can_clear,
                target,
                jnp.zeros(active.shape, dtype=jnp.bool_),
                active,
            ),
            applied_air_resistance=_set_entity_where(
                current.applied_air_resistance,
                target,
                event.air_resistance,
                active,
            ),
            applied_air_resistance_max=_set_entity_where(
                current.applied_air_resistance_max,
                target,
                event.air_resistance_max,
                active,
            ),
            applied_ground_resistance=_set_entity_where(
                current.applied_ground_resistance,
                target,
                event.ground_resistance,
                active,
            ),
            applied_ground_resistance_max=_set_entity_where(
                current.applied_ground_resistance_max,
                target,
                event.ground_resistance_max,
                active,
            ),
            applied_resistance_threshold=_set_entity_where(
                current.applied_resistance_threshold,
                target,
                event.resistance_threshold,
                active,
            ),
            applied_resistance_style=_set_entity_where(
                current.applied_resistance_style,
                target,
                event.resistance_style,
                active,
            ),
            applied_dampen_y=_set_entity_where(
                current.applied_dampen_y,
                target,
                event.dampen_y,
                active,
            ),
        )
        return current, None

    result, _ = jax.lax.scan(apply_one, state, inputs)
    return result
