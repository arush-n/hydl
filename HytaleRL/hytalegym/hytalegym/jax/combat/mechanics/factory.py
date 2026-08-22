"""Host factories for the fixed Hytale 0.5.7 mechanics trees."""

from __future__ import annotations

import operator
from typing import Sequence

import jax.numpy as jnp

from hytalegym.jax.combat.mechanics.schema.contract import (
    DAMAGE_CAUSE_IDS,
    DAMAGE_CLASS_COUNT,
    DAMAGE_COUNT,
    DAMAGE_PHYSICAL,
    DAMAGE_POISON,
    DAMAGE_PROJECTILE,
    DODGE_AIR_RESISTANCE,
    DODGE_AIR_RESISTANCE_MAX,
    DODGE_COOLDOWN_SECONDS,
    DODGE_EXECUTION_AUTHORED_CLIENT_CONFIGURED,
    DODGE_EXECUTION_NATIVE_NPC_NULL_CONFIG,
    DODGE_FORCE,
    DODGE_GROUND_RESISTANCE,
    DODGE_GROUND_RESISTANCE_MAX,
    DODGE_INVULNERABILITY_SECONDS,
    DODGE_LAUNCH_TICK,
    DODGE_COST_TICK,
    DODGE_REGEN_DELAY_SECONDS,
    DODGE_RESISTANCE_THRESHOLD,
    DODGE_STAMINA_ADMISSION_COST,
    DODGE_STAMINA_SPEND_COST,
    GUARD_ACTIVATION_TICK,
    GUARD_ENTRY_COST_TICK,
    GUARD_ENTRY_STAMINA_COST,
    GUARD_EXIT_REGEN_DELAY_SECONDS,
    GUARD_HALF_ANGLE_DEGREES,
    GUARD_RELEASE_TICK,
    RESOURCE_AMMO,
    RESOURCE_COUNT,
    RESOURCE_MAGIC_CHARGES,
    RESOURCE_MANA,
    RESOURCE_OXYGEN,
    RESOURCE_SIGNATURE_CHARGES,
    RESOURCE_SIGNATURE_ENERGY,
    RESOURCE_STAMINA,
    OXYGEN_MAXIMUM,
    OXYGEN_MINIMUM,
    STAMINA_MAXIMUM,
    STAMINA_MINIMUM,
    STAMINA_REGEN_DELAY_INTERVAL_SECONDS,
    STAMINA_REGEN_AMOUNT,
    STAMINA_REGEN_INTERVAL_SECONDS,
    STATUS_APPLICATION_CAPACITY,
    STATUS_CAPACITY,
)
from hytalegym.jax.combat.mechanics.schema.types import (
    CombatMechanicsRules,
    CombatMechanicsState,
    DamageEvents,
    DefenseCommands,
    StatusApplications,
    StatusState,
)
from hytalegym.jax.combat.types import ENTITY_COUNT
from hytalegym.rulesets import load_combat_ruleset
from hytalegym.rulesets.damage_causes import load_damage_cause_rules
from hytalegym.rulesets.damage_classes import (
    DamageClassEnhancementProfile,
    combine_damage_class_enhancement_profiles,
)
from hytalegym.rulesets.resistances import (
    DamageResistanceProfile,
    combine_damage_resistance_profiles,
)

_DEFAULT_SERVER_TICKS_PER_SECOND = float(
    load_combat_ruleset()["engine"]["ticks_per_second"]
)


def default_mechanics_rules(
    batch_size: int,
    *,
    entity_count: int = ENTITY_COUNT,
    guard_stamina_value: float = 7.0,
    mana_maximum: float = 0.0,
    magic_charges_maximum: float = 0.0,
    signature_energy_maximum: float = 0.0,
    signature_charges_maximum: float = 0.0,
    ammo_maximum: float = 0.0,
    server_ticks_per_second: float = _DEFAULT_SERVER_TICKS_PER_SECOND,
    dodge_execution_profile: int = DODGE_EXECUTION_AUTHORED_CLIENT_CONFIGURED,
) -> CombatMechanicsRules:
    """Create common 0.5.7 rules, with equipment-provided stat maxima.

    The default dodge profile reproduces the bridge-controlled policy actor.
    The bridge reconstructs the client ``VelocityConfig`` before routing the
    authored ``Dodge_Left``/``Dodge_Right`` force into the actor controller.
    Select ``DODGE_EXECUTION_NATIVE_NPC_NULL_CONFIG`` only for a pure
    server-NPC path that executes ``ApplyForceInteraction.simulateTick0``
    without that configuration.
    """

    batch = _batch_size(batch_size)
    entities = _positive_size(entity_count, "entity_count")
    dodge_profile = _index(dodge_execution_profile, "dodge_execution_profile")
    if dodge_profile not in {
        DODGE_EXECUTION_NATIVE_NPC_NULL_CONFIG,
        DODGE_EXECUTION_AUTHORED_CLIENT_CONFIGURED,
    }:
        raise ValueError("dodge_execution_profile is not a supported profile")
    shape = (batch, entities, RESOURCE_COUNT)
    minimum = jnp.zeros(shape, dtype=jnp.float32)
    maximum = jnp.zeros(shape, dtype=jnp.float32)
    regen_amount = jnp.zeros(shape, dtype=jnp.float32)
    regen_interval = jnp.ones(shape, dtype=jnp.float32)
    minimum = minimum.at[..., RESOURCE_STAMINA].set(STAMINA_MINIMUM)
    maximum = maximum.at[..., RESOURCE_STAMINA].set(STAMINA_MAXIMUM)
    minimum = minimum.at[..., RESOURCE_OXYGEN].set(OXYGEN_MINIMUM)
    maximum = maximum.at[..., RESOURCE_OXYGEN].set(OXYGEN_MAXIMUM)
    regen_amount = regen_amount.at[..., RESOURCE_STAMINA].set(STAMINA_REGEN_AMOUNT)
    regen_interval = regen_interval.at[..., RESOURCE_STAMINA].set(
        STAMINA_REGEN_INTERVAL_SECONDS
    )
    for resource_id, value in (
        (RESOURCE_MANA, mana_maximum),
        (RESOURCE_MAGIC_CHARGES, magic_charges_maximum),
        (RESOURCE_SIGNATURE_ENERGY, signature_energy_maximum),
        (RESOURCE_SIGNATURE_CHARGES, signature_charges_maximum),
        (RESOURCE_AMMO, ammo_maximum),
    ):
        maximum = maximum.at[..., resource_id].set(jnp.float32(value))
    regen_amount = regen_amount.at[..., RESOURCE_MANA].set(1.0)
    regen_interval = regen_interval.at[..., RESOURCE_MANA].set(0.2)
    regen_amount = regen_amount.at[..., RESOURCE_MAGIC_CHARGES].set(1.0)
    regen_interval = regen_interval.at[..., RESOURCE_MAGIC_CHARGES].set(2.0)

    cause_shape = (batch, entities, DAMAGE_COUNT)
    cause_rules = load_damage_cause_rules()
    cause_ids = tuple(rule.asset_id for rule in cause_rules)
    if cause_ids != DAMAGE_CAUSE_IDS:
        raise ValueError(
            "damage-cause asset order differs from the compiled mechanics contract"
        )
    cause_index = {asset_id: index for index, asset_id in enumerate(cause_ids)}

    def cause_table(field, *, dtype):
        values = jnp.asarray(
            [field(rule) for rule in cause_rules],
            dtype=dtype,
        )
        return jnp.broadcast_to(values, cause_shape)

    cause_inherits = cause_table(
        lambda rule: (-1 if rule.inherits is None else cause_index[rule.inherits]),
        dtype=jnp.int32,
    )
    cause_durability_loss = cause_table(
        lambda rule: rule.durability_loss,
        dtype=jnp.bool_,
    )
    cause_stamina_loss = cause_table(
        lambda rule: rule.stamina_loss,
        dtype=jnp.bool_,
    )
    cause_bypass_resistances = cause_table(
        lambda rule: rule.bypass_resistances,
        dtype=jnp.bool_,
    )
    guard_mask = jnp.zeros(cause_shape, dtype=jnp.bool_)
    guard_mask = guard_mask.at[..., DAMAGE_PHYSICAL].set(True)
    guard_mask = guard_mask.at[..., DAMAGE_PROJECTILE].set(True)
    guard_mask = guard_mask.at[..., DAMAGE_POISON].set(True)
    damage_modifier = jnp.ones(cause_shape, dtype=jnp.float32)
    damage_modifier = jnp.where(guard_mask, jnp.float32(0.0), damage_modifier)
    knockback_modifier = jnp.ones(cause_shape, dtype=jnp.float32)
    knockback_modifier = knockback_modifier.at[..., DAMAGE_PHYSICAL].set(0.25)
    knockback_modifier = knockback_modifier.at[..., DAMAGE_PROJECTILE].set(0.25)
    entity_shape = (batch, entities)

    def full(value):
        return jnp.full(entity_shape, value, dtype=jnp.float32)

    return CombatMechanicsRules(
        resource_minimum=minimum,
        resource_maximum=maximum,
        resource_regen_amount=regen_amount,
        resource_regen_interval_seconds=regen_interval,
        cause_inherits=cause_inherits,
        cause_durability_loss=cause_durability_loss,
        cause_stamina_loss=cause_stamina_loss,
        cause_bypass_resistances=cause_bypass_resistances,
        equipment_damage_resistance_present=jnp.zeros(
            cause_shape,
            dtype=jnp.bool_,
        ),
        equipment_damage_resistance_inherits=jnp.zeros(
            cause_shape,
            dtype=jnp.bool_,
        ),
        equipment_damage_resistance_flat=jnp.zeros(
            cause_shape,
            dtype=jnp.float32,
        ),
        equipment_damage_resistance_multiplier=jnp.zeros(
            cause_shape,
            dtype=jnp.float32,
        ),
        equipment_damage_class_flat=jnp.zeros(
            (batch, entities, DAMAGE_CLASS_COUNT),
            dtype=jnp.float32,
        ),
        equipment_damage_class_multiplier=jnp.zeros(
            (batch, entities, DAMAGE_CLASS_COUNT),
            dtype=jnp.float32,
        ),
        guard_cause_mask=guard_mask,
        guard_damage_modifier=damage_modifier,
        guard_knockback_modifier=knockback_modifier,
        guard_entry_cost=full(GUARD_ENTRY_STAMINA_COST),
        guard_stamina_value=full(guard_stamina_value),
        guard_half_angle_degrees=full(GUARD_HALF_ANGLE_DEGREES),
        guard_entry_delay_seconds=full(0.0),
        guard_exit_regen_delay_seconds=full(GUARD_EXIT_REGEN_DELAY_SECONDS),
        guard_required_resource_id=jnp.full(
            entity_shape,
            -1,
            dtype=jnp.int32,
        ),
        guard_required_resource_minimum=full(0.0),
        guard_entry_cost_tick=jnp.full(
            entity_shape,
            GUARD_ENTRY_COST_TICK,
            dtype=jnp.int32,
        ),
        guard_activation_tick=jnp.full(
            entity_shape,
            GUARD_ACTIVATION_TICK,
            dtype=jnp.int32,
        ),
        guard_release_tick=jnp.full(
            entity_shape,
            GUARD_RELEASE_TICK,
            dtype=jnp.int32,
        ),
        dodge_execution_profile=jnp.full(
            entity_shape,
            dodge_profile,
            dtype=jnp.int32,
        ),
        dodge_admission_cost=full(DODGE_STAMINA_ADMISSION_COST),
        dodge_spend_cost=full(DODGE_STAMINA_SPEND_COST),
        dodge_launch_tick=jnp.full(
            entity_shape,
            DODGE_LAUNCH_TICK,
            dtype=jnp.int32,
        ),
        dodge_cost_tick=jnp.full(
            entity_shape,
            DODGE_COST_TICK,
            dtype=jnp.int32,
        ),
        dodge_force=full(DODGE_FORCE),
        dodge_invulnerability_seconds=full(DODGE_INVULNERABILITY_SECONDS),
        dodge_cooldown_seconds=full(DODGE_COOLDOWN_SECONDS),
        dodge_regen_delay_seconds=full(DODGE_REGEN_DELAY_SECONDS),
        dodge_air_resistance=full(DODGE_AIR_RESISTANCE),
        dodge_air_resistance_max=full(DODGE_AIR_RESISTANCE_MAX),
        dodge_ground_resistance=full(DODGE_GROUND_RESISTANCE),
        dodge_ground_resistance_max=full(DODGE_GROUND_RESISTANCE_MAX),
        dodge_resistance_threshold=full(DODGE_RESISTANCE_THRESHOLD),
        server_ticks_per_second=jnp.asarray(
            server_ticks_per_second,
            dtype=jnp.float32,
        ),
    )


def with_damage_resistance_profiles(
    rules: CombatMechanicsRules,
    profiles: Sequence[Sequence[DamageResistanceProfile]],
) -> CombatMechanicsRules:
    """Bind one strict, pre-aggregated equipment profile per batch entity."""

    batch, entities, cause_count = rules.cause_inherits.shape
    if len(profiles) != batch:
        raise ValueError(f"resistance profiles must have {batch} batch rows")
    present = []
    inherits = []
    flat = []
    multiplier = []
    for row_index, row in enumerate(profiles):
        if len(row) != entities:
            raise ValueError(
                f"resistance profile row {row_index} must have {entities} entities"
            )
        validated = [combine_damage_resistance_profiles((profile,)) for profile in row]
        if any(len(profile.present) != cause_count for profile in validated):
            raise ValueError(
                f"resistance profiles must have {cause_count} damage causes"
            )
        present.append([profile.present for profile in validated])
        inherits.append([profile.inherits for profile in validated])
        flat.append([profile.flat for profile in validated])
        multiplier.append([profile.multiplier for profile in validated])
    return rules._replace(
        equipment_damage_resistance_present=jnp.asarray(
            present,
            dtype=jnp.bool_,
        ),
        equipment_damage_resistance_inherits=jnp.asarray(
            inherits,
            dtype=jnp.bool_,
        ),
        equipment_damage_resistance_flat=jnp.asarray(
            flat,
            dtype=jnp.float32,
        ),
        equipment_damage_resistance_multiplier=jnp.asarray(
            multiplier,
            dtype=jnp.float32,
        ),
    )


def with_damage_class_enhancement_profiles(
    rules: CombatMechanicsRules,
    profiles: Sequence[Sequence[DamageClassEnhancementProfile]],
) -> CombatMechanicsRules:
    """Bind one strict source-equipment enhancement profile per batch entity."""

    batch, entities, class_count = rules.equipment_damage_class_flat.shape
    if len(profiles) != batch:
        raise ValueError(f"damage-class profiles must have {batch} batch rows")
    flat = []
    multiplier = []
    for row_index, row in enumerate(profiles):
        if len(row) != entities:
            raise ValueError(
                f"damage-class profile row {row_index} must have {entities} entities"
            )
        validated = [
            combine_damage_class_enhancement_profiles((profile,))
            for profile in row
        ]
        if any(len(profile.flat) != class_count for profile in validated):
            raise ValueError(
                f"damage-class profiles must have {class_count} classes"
            )
        flat.append([profile.flat for profile in validated])
        multiplier.append([profile.multiplier for profile in validated])
    return rules._replace(
        equipment_damage_class_flat=jnp.asarray(
            flat,
            dtype=jnp.float32,
        ),
        equipment_damage_class_multiplier=jnp.asarray(
            multiplier,
            dtype=jnp.float32,
        ),
    )


def empty_mechanics_state(
    batch_size: int,
    rules: CombatMechanicsRules,
    *,
    initial_resources: jnp.ndarray | None = None,
) -> CombatMechanicsState:
    """Reset resources to explicit values, or maxima for legacy callers."""

    batch = _batch_size(batch_size)
    entities = rules.resource_maximum.shape[1]
    entity_shape = (batch, entities)
    resources = (
        rules.resource_maximum
        if initial_resources is None
        else jnp.asarray(initial_resources, dtype=jnp.float32)
    )
    if resources.shape != rules.resource_maximum.shape:
        raise ValueError(
            "initial_resources must match resource rule shape "
            f"{rules.resource_maximum.shape}"
        )
    resources = jnp.clip(
        resources,
        rules.resource_minimum,
        rules.resource_maximum,
    )
    return CombatMechanicsState(
        resources=resources,
        # RegeneratingValue.remainingUntilRegen is zero-initialized natively,
        # so the first eligible tick regenerates immediately.  Store elapsed
        # time here; one full interval represents that initially-due state.
        resource_regen_clock=jnp.where(
            rules.resource_regen_amount != jnp.float32(0.0),
            rules.resource_regen_interval_seconds,
            jnp.float32(0.0),
        ),
        # Native RegeneratingValue and DelayedEntitySystem timers begin due:
        # their zero-initialized countdowns are decremented before the strict
        # below-zero check on the first server tick.
        oxygen_regen_remaining_seconds=jnp.zeros(
            entity_shape,
            dtype=jnp.float32,
        ),
        breathing_damage_remaining_seconds=jnp.zeros(
            entity_shape,
            dtype=jnp.float32,
        ),
        stamina_regen_delay_seconds=jnp.zeros(
            entity_shape,
            dtype=jnp.float32,
        ),
        stamina_regen_delay_clock=jnp.full(
            entity_shape,
            jnp.float32(STAMINA_REGEN_DELAY_INTERVAL_SECONDS),
            dtype=jnp.float32,
        ),
        stamina_broken=jnp.zeros(entity_shape, dtype=jnp.bool_),
        guard_held=jnp.zeros(entity_shape, dtype=jnp.bool_),
        guard_active=jnp.zeros(entity_shape, dtype=jnp.bool_),
        guard_windup_elapsed_seconds=jnp.zeros(
            entity_shape,
            dtype=jnp.float32,
        ),
        guard_operation_tick=jnp.full(
            entity_shape,
            -1,
            dtype=jnp.int32,
        ),
        control_immunity=jnp.zeros(entity_shape, dtype=jnp.float32),
        control_immunity_regen_clock=jnp.zeros(
            entity_shape,
            dtype=jnp.float32,
        ),
        applied_velocity=jnp.zeros(
            (batch, entities, 3),
            dtype=jnp.float32,
        ),
        external_velocity_y=jnp.zeros(entity_shape, dtype=jnp.float32),
        applied_velocity_can_clear=jnp.zeros(
            entity_shape,
            dtype=jnp.bool_,
        ),
        applied_air_resistance=rules.dodge_air_resistance,
        applied_air_resistance_max=rules.dodge_air_resistance_max,
        applied_ground_resistance=rules.dodge_ground_resistance,
        applied_ground_resistance_max=rules.dodge_ground_resistance_max,
        applied_resistance_threshold=rules.dodge_resistance_threshold,
        applied_resistance_style=jnp.ones(entity_shape, dtype=jnp.int32),
        applied_dampen_y=jnp.zeros(entity_shape, dtype=jnp.bool_),
        dodge_invulnerability_remaining_seconds=jnp.zeros(
            entity_shape,
            dtype=jnp.float32,
        ),
        dodge_cooldown_remaining_seconds=jnp.zeros(
            entity_shape,
            dtype=jnp.float32,
        ),
        dodge_operation_tick=jnp.full(
            entity_shape,
            -1,
            dtype=jnp.int32,
        ),
        dodge_pending_direction=jnp.zeros(
            entity_shape,
            dtype=jnp.int32,
        ),
        statuses=_empty_statuses(batch, entities),
        failure_bits=jnp.zeros((batch,), dtype=jnp.uint32),
    )


def empty_defense_commands(
    batch_size: int,
    *,
    entity_count: int = ENTITY_COUNT,
) -> DefenseCommands:
    batch = _batch_size(batch_size)
    shape = (batch, _positive_size(entity_count, "entity_count"))
    return DefenseCommands(
        guard_held=jnp.zeros(shape, dtype=jnp.bool_),
        dodge_direction=jnp.zeros(shape, dtype=jnp.int32),
        dodge_corridor_clear=jnp.zeros(shape, dtype=jnp.bool_),
    )


def empty_status_applications(
    batch_size: int,
    *,
    entity_count: int = ENTITY_COUNT,
) -> StatusApplications:
    batch = _batch_size(batch_size)
    shape = (
        batch,
        _positive_size(entity_count, "entity_count"),
        STATUS_APPLICATION_CAPACITY,
    )
    f32 = jnp.zeros(shape, dtype=jnp.float32)
    i32 = jnp.zeros(shape, dtype=jnp.int32)
    return StatusApplications(
        requested=jnp.zeros(shape, dtype=jnp.bool_),
        effect_id=i32,
        source_entity_id=jnp.full(shape, -1, dtype=jnp.int32),
        duration_seconds=f32,
        cycle_cooldown_seconds=f32,
        damage_per_cycle=f32,
        damage_cause=i32,
        healing_per_cycle=f32,
        resource_id=jnp.full(shape, -1, dtype=jnp.int32),
        resource_delta_per_cycle=f32,
        speed_multiplier=jnp.ones(shape, dtype=jnp.float32),
        damage_resistance_present=jnp.zeros(
            shape + (DAMAGE_COUNT,),
            dtype=jnp.bool_,
        ),
        damage_resistance_flat=jnp.zeros(
            shape + (DAMAGE_COUNT,),
            dtype=jnp.float32,
        ),
        damage_resistance_multiplier=jnp.zeros(
            shape + (DAMAGE_COUNT,),
            dtype=jnp.float32,
        ),
        flags=jnp.zeros(shape, dtype=jnp.uint32),
        overlap_mode=i32,
    )


def empty_damage_events(batch_size: int, capacity: int) -> DamageEvents:
    batch = _batch_size(batch_size)
    count = _nonnegative_size(capacity, "capacity")
    shape = (batch, count)
    return DamageEvents(
        requested=jnp.zeros(shape, dtype=jnp.bool_),
        source_entity_id=jnp.full(shape, -1, dtype=jnp.int32),
        target_entity_id=jnp.zeros(shape, dtype=jnp.int32),
        amount=jnp.zeros(shape, dtype=jnp.float32),
        random_percentage=jnp.zeros(shape, dtype=jnp.float32),
        damage_class=jnp.zeros(shape, dtype=jnp.int32),
        cause=jnp.zeros(shape, dtype=jnp.int32),
        stamina_drain_multiplier=jnp.ones(shape, dtype=jnp.float32),
        knockback_velocity=jnp.zeros((batch, count, 3), dtype=jnp.float32),
        force_mode=jnp.zeros(shape, dtype=jnp.int32),
        air_resistance=jnp.ones(shape, dtype=jnp.float32),
        air_resistance_max=jnp.ones(shape, dtype=jnp.float32),
        ground_resistance=jnp.ones(shape, dtype=jnp.float32),
        ground_resistance_max=jnp.ones(shape, dtype=jnp.float32),
        resistance_threshold=jnp.ones(shape, dtype=jnp.float32),
        resistance_style=jnp.zeros(shape, dtype=jnp.int32),
        dampen_y=jnp.zeros(shape, dtype=jnp.bool_),
        on_hit_resource_id=jnp.full(shape, -1, dtype=jnp.int32),
        on_hit_resource_delta=jnp.zeros(shape, dtype=jnp.float32),
        on_hit_healing=jnp.zeros(shape, dtype=jnp.float32),
    )


def _empty_statuses(batch: int, entity_count: int) -> StatusState:
    shape = (batch, entity_count, STATUS_CAPACITY)
    f32 = jnp.zeros(shape, dtype=jnp.float32)
    i32 = jnp.zeros(shape, dtype=jnp.int32)
    return StatusState(
        effect_id=i32,
        source_entity_id=jnp.full(shape, -1, dtype=jnp.int32),
        remaining_seconds=f32,
        cycle_elapsed_seconds=f32,
        cycle_cooldown_seconds=f32,
        damage_per_cycle=f32,
        damage_cause=i32,
        healing_per_cycle=f32,
        resource_id=jnp.full(shape, -1, dtype=jnp.int32),
        resource_delta_per_cycle=f32,
        speed_multiplier=jnp.ones(shape, dtype=jnp.float32),
        damage_resistance_present=jnp.zeros(
            shape + (DAMAGE_COUNT,),
            dtype=jnp.bool_,
        ),
        damage_resistance_flat=jnp.zeros(
            shape + (DAMAGE_COUNT,),
            dtype=jnp.float32,
        ),
        damage_resistance_multiplier=jnp.zeros(
            shape + (DAMAGE_COUNT,),
            dtype=jnp.float32,
        ),
        flags=jnp.zeros(shape, dtype=jnp.uint32),
        overlap_mode=i32,
        active=jnp.zeros(shape, dtype=jnp.bool_),
        has_cycled=jnp.zeros(shape, dtype=jnp.bool_),
    )


def _batch_size(value: int) -> int:
    return _positive_size(value, "batch_size")


def _positive_size(value: int, label: str) -> int:
    result = _index(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative_size(value: int, label: str) -> int:
    result = _index(value, label)
    if result < 0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _index(value: int, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{label} must be an integer") from error
