"""Legacy target-attack columns projected from data-driven ability programs."""

from __future__ import annotations

import jax.numpy as jnp

from hytalegym.jax.combat.opponents.legacy.controller import COOLDOWN, IDLE, RECOVERY, SWEEP, WINDUP
from hytalegym.jax.combat.arsenal.schema.contract import (
    EF_SELECTOR_RUNTIME_SECONDS,
    OBSERVATION_CAPACITY,
)
from hytalegym.jax.combat.arsenal.programs.scheduling import (
    event_uses_progressive_selector,
)
from hytalegym.jax.combat.arsenal.programs.event_storage import selected_ability_events
from hytalegym.jax.combat.observation.v1.schema.contract import (
    COMBAT_FLOAT_FEATURES,
    COMBAT_INTEGER_FEATURES,
    ENTITY_FLOAT_FEATURES,
    ENTITY_INTEGER_FEATURES,
    TARGET_FLOAT_FEATURES,
    TARGET_INTEGER_FEATURES,
)


_COMBAT_TARGET_PHASE_START = COMBAT_FLOAT_FEATURES.index("target_phase_idle")
_COMBAT_TARGET_PHASE_END = COMBAT_FLOAT_FEATURES.index("target_phase_cooldown") + 1
_COMBAT_TARGET_PROGRESS = COMBAT_FLOAT_FEATURES.index("target_attack_progress")
_COMBAT_TARGET_INDEX = COMBAT_FLOAT_FEATURES.index("target_attack_index")
_COMBAT_I32_PHASE = COMBAT_INTEGER_FEATURES.index("target_attack_phase")
_COMBAT_I32_INDEX = COMBAT_INTEGER_FEATURES.index("target_attack_index")
_COMBAT_I32_ELAPSED = COMBAT_INTEGER_FEATURES.index("target_attack_elapsed_ticks")
_TARGET_PROGRESS = TARGET_FLOAT_FEATURES.index("attack_progress")
_TARGET_I32_PHASE = TARGET_INTEGER_FEATURES.index("attack_phase")
_TARGET_I32_INDEX = TARGET_INTEGER_FEATURES.index("attack_index")
_TARGET_I32_ELAPSED = TARGET_INTEGER_FEATURES.index("attack_elapsed_ticks")
_TARGET_I32_QUEUED = TARGET_INTEGER_FEATURES.index("attack_queued")
_ENTITY_PROGRESS = ENTITY_FLOAT_FEATURES.index("attack_progress")
_ENTITY_I32_PHASE = ENTITY_INTEGER_FEATURES.index("attack_phase")
_ENTITY_I32_INDEX = ENTITY_INTEGER_FEATURES.index("attack_index")

def project_arsenal_target_attack_compatibility(
    base,
    state,
    config,
    target_entity_id,
):
    """Keep the frozen target-attack prefix live for Arsenal opponents.

    Hytale publishes elapsed time and operation progress for interaction-chain
    nodes, not a universal five-phase enum.  The frozen Gym phase columns are
    therefore a policy abstraction: authored event times delimit windup,
    selector/event execution, recovery, and cooldown.  No weapon name or
    profile-specific timing appears here.
    """

    batch, entity_count = state.arsenal.active_ability_slot.shape
    target = jnp.asarray(target_entity_id, dtype=jnp.int32)
    if target.shape != (batch,):
        raise ValueError(f"target_entity_id must have shape [{batch}]")
    row = jnp.arange(batch, dtype=jnp.int32)
    safe_target = jnp.clip(target, 0, entity_count - 1)
    target_index_valid = (target >= 0) & (target < entity_count)
    target_present = target_index_valid & jnp.asarray(
        base.target_mask,
        dtype=jnp.bool_,
    )

    loadout = config.loadout
    ability_capacity = loadout.ability_mask.shape[2]
    active_slot = state.arsenal.active_ability_slot[row, safe_target]
    safe_active_slot = jnp.clip(active_slot, 0, ability_capacity - 1)

    target_ability_mask = loadout.ability_mask[row, safe_target]
    arsenal_target = target_index_valid & jnp.any(target_ability_mask, axis=1)
    event_mask, event_time, event_f32, event_flags = selected_ability_events(
        loadout,
        safe_target,
        safe_active_slot,
    )
    ability_present = target_ability_mask[row, safe_active_slot]
    active = (
        target_present
        & (active_slot >= 0)
        & ability_present
    )

    progressive = event_uses_progressive_selector(event_flags)
    selector_runtime = jnp.where(
        progressive,
        event_f32[..., EF_SELECTOR_RUNTIME_SECONDS],
        jnp.float32(0.0),
    )
    duration = loadout.ability_duration_seconds[
        row,
        safe_target,
        safe_active_slot,
    ]
    elapsed = state.arsenal.ability_elapsed_seconds[row, safe_target]
    first_event = jnp.min(
        jnp.where(event_mask, event_time, jnp.float32(jnp.inf)),
        axis=1,
    )
    last_event = jnp.max(
        jnp.where(
            event_mask,
            event_time + selector_runtime,
            jnp.float32(0.0),
        ),
        axis=1,
    )
    has_event = jnp.any(event_mask, axis=1)
    first_event = jnp.where(has_event, first_event, duration)
    last_event = jnp.where(has_event, last_event, duration)

    active_phase = jnp.where(
        elapsed < first_event,
        WINDUP,
        jnp.where(elapsed <= last_event, SWEEP, RECOVERY),
    ).astype(jnp.int32)
    windup_progress = elapsed / jnp.maximum(first_event, jnp.float32(1.0e-6))
    sweep_progress = (elapsed - first_event) / jnp.maximum(
        last_event - first_event,
        jnp.float32(1.0e-6),
    )
    recovery_progress = (elapsed - last_event) / jnp.maximum(
        duration - last_event,
        jnp.float32(1.0e-6),
    )
    active_progress = jnp.where(
        active_phase == WINDUP,
        windup_progress,
        jnp.where(
            active_phase == SWEEP,
            sweep_progress,
            recovery_progress,
        ),
    )

    cooldown_remaining = state.arsenal.ability_cooldown_seconds[row, safe_target]
    cooldown_duration = loadout.ability_cooldown_seconds[row, safe_target]
    cooldown_fraction = jnp.where(
        target_ability_mask,
        cooldown_remaining / jnp.maximum(cooldown_duration, jnp.float32(1.0e-6)),
        jnp.float32(0.0),
    )
    cooling_slot = jnp.argmax(cooldown_fraction, axis=1).astype(jnp.int32)
    cooling_fraction = jnp.max(cooldown_fraction, axis=1)
    cooling = target_present & ~active & (cooling_fraction > 0.0)

    phase = jnp.where(active, active_phase, jnp.where(cooling, COOLDOWN, IDLE))
    progress = jnp.where(
        active,
        jnp.clip(active_progress, 0.0, 1.0),
        jnp.where(cooling, jnp.float32(1.0) - cooling_fraction, 0.0),
    ).astype(jnp.float32)
    reported_slot = jnp.where(
        active,
        active_slot,
        jnp.where(cooling, cooling_slot, jnp.int32(-1)),
    )
    elapsed_ticks = jnp.where(
        active,
        jnp.maximum(
            state.arsenal.ability_scheduler_tick[row, safe_target],
            jnp.int32(0),
        ),
        jnp.int32(0),
    )
    queued = active & (
        state.arsenal.ability_scheduler_tick[row, safe_target] < jnp.int32(0)
    )
    visible_phase = jax_one_hot_phase(phase) * target_present[:, None]
    normalized_slot = jnp.where(
        reported_slot >= 0,
        reported_slot.astype(jnp.float32)
        / jnp.float32(max(OBSERVATION_CAPACITY - 1, 1)),
        jnp.float32(-1.0),
    )

    projected_combat_f32 = (
        base.combat_f32.at[
            :, _COMBAT_TARGET_PHASE_START:_COMBAT_TARGET_PHASE_END
        ]
        .set(visible_phase)
        .at[:, _COMBAT_TARGET_PROGRESS]
        .set(jnp.where(target_present, progress, 0.0))
        .at[:, _COMBAT_TARGET_INDEX]
        .set(jnp.where(target_present, normalized_slot, 0.0))
    )
    projected_combat_i32 = (
        base.combat_i32.at[:, _COMBAT_I32_PHASE]
        .set(jnp.where(target_present, phase, jnp.int32(IDLE)))
        .at[:, _COMBAT_I32_INDEX]
        .set(jnp.where(target_present, reported_slot, jnp.int32(-1)))
        .at[:, _COMBAT_I32_ELAPSED]
        .set(jnp.where(target_present, elapsed_ticks, jnp.int32(0)))
    )
    projected_target_f32 = base.target_f32.at[:, _TARGET_PROGRESS].set(
        jnp.where(target_present, progress, 0.0)
    )
    projected_target_i32 = (
        base.target_i32.at[:, _TARGET_I32_PHASE]
        .set(jnp.where(target_present, phase, jnp.int32(IDLE)))
        .at[:, _TARGET_I32_INDEX]
        .set(jnp.where(target_present, reported_slot, jnp.int32(-1)))
        .at[:, _TARGET_I32_ELAPSED]
        .set(jnp.where(target_present, elapsed_ticks, jnp.int32(0)))
        .at[:, _TARGET_I32_QUEUED]
        .set(jnp.where(target_present, queued, False).astype(jnp.int32))
    )
    projected_entity_f32 = base.entity_f32.at[:, 0, _ENTITY_PROGRESS].set(
        jnp.where(base.entity_mask[:, 0], progress, 0.0)
    )
    projected_entity_i32 = (
        base.entity_i32.at[:, 0, _ENTITY_I32_PHASE]
        .set(jnp.where(base.entity_mask[:, 0], phase, jnp.int32(IDLE)))
        .at[:, 0, _ENTITY_I32_INDEX]
        .set(jnp.where(base.entity_mask[:, 0], reported_slot, jnp.int32(-1)))
    )
    return base._replace(
        combat_f32=jnp.where(
            arsenal_target[:, None],
            projected_combat_f32,
            base.combat_f32,
        ),
        combat_i32=jnp.where(
            arsenal_target[:, None],
            projected_combat_i32,
            base.combat_i32,
        ),
        target_f32=jnp.where(
            arsenal_target[:, None],
            projected_target_f32,
            base.target_f32,
        ),
        target_i32=jnp.where(
            arsenal_target[:, None],
            projected_target_i32,
            base.target_i32,
        ),
        entity_f32=jnp.where(
            arsenal_target[:, None, None],
            projected_entity_f32,
            base.entity_f32,
        ),
        entity_i32=jnp.where(
            arsenal_target[:, None, None],
            projected_entity_i32,
            base.entity_i32,
        ),
    )


def jax_one_hot_phase(phase):
    """Return the frozen five-phase one-hot without importing policy code."""

    return jnp.arange(5, dtype=jnp.int32)[None, :] == phase[:, None]


__all__ = ["project_arsenal_target_attack_compatibility"]
