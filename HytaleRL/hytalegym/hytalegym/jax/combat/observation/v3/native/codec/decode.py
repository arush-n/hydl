"""Leaf helpers extracted verbatim from native_evidence.py."""

from dataclasses import dataclass
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np

from hytalegym.jax.combat.arsenal.schema.contract import (
    ABILITY_CAPACITY,
    EF_STATUS_COOLDOWN_SECONDS,
    EF_STATUS_DAMAGE,
    EF_STATUS_HEALING,
    EF_STATUS_RESOURCE_DELTA,
    EF_STATUS_SPEED_MULTIPLIER,
    EI_STATUS_DAMAGE_CAUSE,
    EI_STATUS_ID,
    EI_STATUS_OVERLAP_MODE,
    EI_STATUS_RESOURCE_ID,
    EI_TARGET_MODE,
    EVENT_FLAG_STATUS_VALUE_PERCENT,
    EVENT_STATUS_FLAG_MASK,
    TARGET_OTHER,
    TARGET_SELF,
)
from hytalegym.jax.combat.contracts.semantic import semantic_id
from hytalegym.jax.combat.entities.interactions.schema.status_programs import (
    entity_interaction_status_programs_by_semantic_id,
)
from hytalegym.jax.combat.mechanics import (
    DAMAGE_COUNT,
    RESOURCE_COUNT,
    STATUS_CAPACITY,
    STATUS_FLAG_DEBUFF,
    STATUS_FLAG_INVULNERABLE,
    role_status_programs_by_semantic_id,
)
from hytalegym.jax.combat.observation.v3.schema.contract import (
    ACTOR_WORLD_FLOAT_SIZE,
    ACTOR_WORLD_MASK_SIZE,
    DEFENSE_FLOAT_SIZE,
    DODGE_ACTION_COUNT,
    MOVEMENT_STATE_SIZE,
)
from hytalegym.jax.combat.observation.v3.schema.spec import (
    learner_observation_v3_actor_evidence_contract_sha256,
)
from hytalegym.jax.combat.types import AGENT_ENTITY, ENTITY_COUNT, TARGET_ENTITY
from hytalegym.jax.world import geometry_state_from_numpy


NATIVE_ACTOR_EVIDENCE_SCHEMA = (
    "hytalerl_native_learner_v3_actor_evidence_inputs_v4"
)


NATIVE_ACTOR_EVIDENCE_VERSION = 4


NATIVE_ACTOR_EVIDENCE_GEOMETRY_CELL_COUNT = 729

MOTION_FORCE_SOURCE_NONE = 0
MOTION_FORCE_SOURCE_LEGACY_EXTERNAL = 1
MOTION_FORCE_SOURCE_CONFIGURED_APPLIED = 2
MOTION_FORCE_SOURCE_COMBINED_CURRENT = 3
MOTION_FORCE_SOURCE_PENDING_KNOCKBACK = 4
MOTION_FORCE_SOURCE_AMBIGUOUS_PENDING = 5
MOTION_FORCE_SOURCE_UNAVAILABLE = 6


@dataclass(frozen=True)
class _StatusProgram:
    damage: float
    healing: float
    cooldown: float
    resource_delta: float
    speed_multiplier: float
    damage_cause: int
    resource_id: int
    overlap_mode: int
    target_mode: int
    flags: int
    value_percent: bool
    infinite: bool
    damage_resistance_present: tuple[bool, ...]
    damage_resistance_flat: tuple[int, ...]
    damage_resistance_multiplier: tuple[float, ...]


def _wire_frame(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("native actor evidence payload must be an object")
    expected_contract = learner_observation_v3_actor_evidence_contract_sha256()
    identities = {
        "schema": NATIVE_ACTOR_EVIDENCE_SCHEMA,
        "version": NATIVE_ACTOR_EVIDENCE_VERSION,
        "evidence_contract_sha256": expected_contract,
    }
    for name, expected in identities.items():
        if value.get(name) != expected:
            raise ValueError(
                f"native actor evidence {name} mismatch: "
                f"{value.get(name)!r} != {expected!r}"
            )
    capacities = _mapping(value.get("capacities"), "capacities")
    expected_capacities = {
        "entities": ENTITY_COUNT,
        "resources": RESOURCE_COUNT,
        "statuses_per_entity": STATUS_CAPACITY,
        "abilities_per_entity": ABILITY_CAPACITY,
        "skills": 9,
        "dodge_directions": DODGE_ACTION_COUNT,
        "door_candidates": 8,
        "door_intents": 3,
    }
    for name, expected in expected_capacities.items():
        if _integer(capacities.get(name), name) != expected:
            raise ValueError(f"native actor evidence {name} capacity mismatch")
    available = _boolean(value.get("available"), "available")
    reason = value.get("unavailable_reason")
    if not isinstance(reason, str):
        raise ValueError("native actor evidence reason must be a string")
    raw_actors = _sequence(value.get("actors"), "actors")
    raw_abilities = _sequence(value.get("abilities"), "abilities")
    actors = [_actor(row, index) for index, row in enumerate(raw_actors)]
    abilities = [
        _ability(row, index) for index, row in enumerate(raw_abilities)
    ]
    if available:
        if reason or len(actors) != ENTITY_COUNT or len(abilities) != ABILITY_CAPACITY:
            raise ValueError("available native actor evidence is incomplete")
    elif actors or abilities:
        raise ValueError("unavailable native actor evidence carries actor rows")
    role_available = _boolean(
        value.get("role_opaque_cell_mask_available"),
        "role_opaque_cell_mask_available",
    )
    role_mask = _array(
        value.get("role_opaque_cell_mask"),
        (NATIVE_ACTOR_EVIDENCE_GEOMETRY_CELL_COUNT,),
        np.bool_,
        "role_opaque_cell_mask",
    )
    if not role_available and np.any(role_mask):
        raise ValueError("unavailable role opacity carries true cells")
    result = {
        "available": available,
        "unavailable_reason": reason,
        "world_tick": _integer(value.get("world_tick"), "world_tick"),
        "actors": actors,
        "abilities": abilities,
        "world_geometry_available": _boolean(
            value.get("world_geometry_available"),
            "world_geometry_available",
        ),
        "role_opaque_cell_mask_available": role_available,
        "role_opaque_cell_mask": role_mask,
        "skill_action_mask": _array(
            value.get("skill_action_mask"), (9,), np.bool_, "skill_action_mask"
        ),
        "jump_action_available": _boolean(
            value.get("jump_action_available"), "jump_action_available"
        ),
        "guard_action_available": _boolean(
            value.get("guard_action_available"), "guard_action_available"
        ),
        "dodge_action_mask": _array(
            value.get("dodge_action_mask"),
            (DODGE_ACTION_COUNT,),
            np.bool_,
            "dodge_action_mask",
        ),
        "door_action_mask": _array(
            value.get("door_action_mask"),
            (8, 3),
            np.bool_,
            "door_action_mask",
        ),
        "loadout_failure": _integer(
            value.get("loadout_failure"), "loadout_failure"
        ),
        "mechanics_failure_bits": _integer(
            value.get("mechanics_failure_bits"), "mechanics_failure_bits"
        ),
        "arsenal_failure_bits": _integer(
            value.get("arsenal_failure_bits"), "arsenal_failure_bits"
        ),
    }
    usable = (
        result["world_geometry_available"]
        or role_available
        or np.any(result["skill_action_mask"])
        or result["jump_action_available"]
        or result["guard_action_available"]
        or np.any(result["dodge_action_mask"])
        or np.any(result["door_action_mask"])
    )
    if not available and usable:
        raise ValueError("unavailable native actor evidence carries usable rows")
    return result


def _actor(value: Any, entity_id: int) -> dict[str, Any]:
    row = _mapping(value, "actor")
    if _integer(row.get("entity_id"), "entity_id") != entity_id:
        raise ValueError("native actors are not in canonical entity order")
    present = _boolean(row.get("present"), "present")
    perceptible = _boolean(row.get("perceptible"), "perceptible")
    role_id = _string(row.get("role_id"), "role_id")
    item_id = _string(row.get("item_id"), "item_id")
    movement = _mapping(row.get("movement_states"), "movement_states")
    movement_available = _boolean(
        movement.get("available"), "movement_states.available"
    )
    movement_bits = _integer(movement.get("bits"), "movement_states.bits")
    if not 0 <= movement_bits < (1 << MOVEMENT_STATE_SIZE):
        raise ValueError("native movement-state bits are out of range")
    if not movement_available and movement_bits:
        raise ValueError("unavailable movement state carries true bits")
    statuses = [_status(item) for item in _sequence(row.get("statuses"), "statuses")]
    if len(statuses) > STATUS_CAPACITY:
        raise ValueError("native status capacity exceeded")
    motion_force = _motion_force(row.get("motion_force"))
    result = {
        "entity_id": entity_id,
        "present": present,
        "perceptible": perceptible,
        "role_id": role_id,
        "item_id": item_id,
        "item_runtime_index": _integer(
            row.get("item_runtime_index"), "item_runtime_index"
        ),
        "active_ability_slot": _integer(
            row.get("active_ability_slot"), "active_ability_slot"
        ),
        "position": _array(row.get("position"), (3,), np.float64, "position"),
        "velocity": _array(row.get("velocity"), (3,), np.float64, "velocity"),
        "motion_force": motion_force,
        "yaw_degrees": _number(row.get("yaw_degrees"), "yaw_degrees"),
        "pitch_degrees": _number(row.get("pitch_degrees"), "pitch_degrees"),
        "health": _number(row.get("health"), "health"),
        "max_health": _number(row.get("max_health"), "max_health"),
        "resource_values": _array(
            row.get("resource_values"),
            (RESOURCE_COUNT,),
            np.float64,
            "resource_values",
        ),
        "resource_maximums": _array(
            row.get("resource_maximums"),
            (RESOURCE_COUNT,),
            np.float64,
            "resource_maximums",
        ),
        "resource_available": _array(
            row.get("resource_available"),
            (RESOURCE_COUNT,),
            np.bool_,
            "resource_available",
        ),
        "defense_values": _array(
            row.get("defense_values"),
            (DEFENSE_FLOAT_SIZE,),
            np.float64,
            "defense_values",
        ),
        "defense_available": _array(
            row.get("defense_available"),
            (DEFENSE_FLOAT_SIZE,),
            np.bool_,
            "defense_available",
        ),
        "statuses": statuses,
        "actor_world_values": _array(
            row.get("actor_world_values"),
            (ACTOR_WORLD_FLOAT_SIZE,),
            np.float64,
            "actor_world_values",
        ),
        "actor_world_available": _array(
            row.get("actor_world_available"),
            (ACTOR_WORLD_MASK_SIZE,),
            np.bool_,
            "actor_world_available",
        ),
        "movement_states": {
            "available": movement_available,
            "bits": movement_bits,
        },
    }
    if not present and _absent_actor_has_evidence(result):
        raise ValueError("absent native actor is not structurally empty")
    if present and not perceptible:
        raise ValueError("present native target evidence is not perceptible")
    if present:
        if result["max_health"] <= 0.0:
            raise ValueError("present native actor has no positive max health")
        if not 0.0 <= result["health"] <= result["max_health"] + 1.0e-6:
            raise ValueError("native actor health is outside its declared range")
        if not np.all(result["defense_available"]):
            raise ValueError("native defense evidence is incomplete")
        if bool(result["defense_available"][3]) != bool(
            motion_force["projected_available"]
        ):
            raise ValueError(
                "native force projection availability disagrees with defense"
            )
        projected_speed = float(
            np.linalg.norm(motion_force["projected_velocity"])
        )
        if not np.isclose(
            result["defense_values"][3],
            projected_speed,
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise ValueError(
                "native force projection magnitude disagrees with defense"
            )
        if bool(result["defense_values"][6] > 0.5) != (
            result["health"] > 0.0
        ):
            raise ValueError("native alive evidence disagrees with health")
        unavailable_resources = ~result["resource_available"]
        if np.any(result["resource_values"][unavailable_resources]) or np.any(
            result["resource_maximums"][unavailable_resources]
        ):
            raise ValueError("unavailable native resource carries a value")
        actor_world_groups = (
            (0, (0,)),
            (1, (1, 2)),
            (2, (3, 4)),
        )
        for availability_index, value_indices in actor_world_groups:
            if (
                not result["actor_world_available"][availability_index]
                and np.any(result["actor_world_values"][list(value_indices)])
            ):
                raise ValueError(
                    "unavailable native actor-world evidence carries a value"
                )
    return result


def parse_native_actor_evidence_frame(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and decode one versioned native actor-evidence frame."""

    return _wire_frame(value)


def _motion_force(value: Any) -> dict[str, Any]:
    row = _mapping(value, "motion_force")
    legacy_available = _boolean(
        row.get("legacy_external_available"),
        "motion_force.legacy_external_available",
    )
    legacy = _array(
        row.get("legacy_external_velocity"),
        (3,),
        np.float64,
        "motion_force.legacy_external_velocity",
    )
    configured_available = _boolean(
        row.get("configured_applied_available"),
        "motion_force.configured_applied_available",
    )
    configured = _array(
        row.get("configured_applied_velocity"),
        (3,),
        np.float64,
        "motion_force.configured_applied_velocity",
    )
    configured_count = _integer(
        row.get("configured_applied_count"),
        "motion_force.configured_applied_count",
    )
    pending_available = _boolean(
        row.get("pending_knockback_available"),
        "motion_force.pending_knockback_available",
    )
    pending = _array(
        row.get("pending_knockback_velocity"),
        (3,),
        np.float64,
        "motion_force.pending_knockback_velocity",
    )
    projected_available = _boolean(
        row.get("projected_available"),
        "motion_force.projected_available",
    )
    projected = _array(
        row.get("projected_velocity"),
        (3,),
        np.float64,
        "motion_force.projected_velocity",
    )
    source = _integer(
        row.get("projection_source"),
        "motion_force.projection_source",
    )
    if configured_count < 0:
        raise ValueError("configured applied force count must be nonnegative")
    if (
        (not legacy_available and np.any(legacy))
        or (
            not configured_available
            and (configured_count != 0 or np.any(configured))
        )
        or (not pending_available and np.any(pending))
        or (not projected_available and np.any(projected))
    ):
        raise ValueError("unavailable native force channel carries a value")

    channels_available = (
        legacy_available and configured_available and pending_available
    )
    current = legacy + configured
    legacy_active = bool(np.any(legacy))
    configured_active = bool(configured_count > 0 or np.any(configured))
    current_active = bool(np.any(current))
    pending_active = bool(np.any(pending))
    if not channels_available:
        expected_available = False
        expected = np.zeros((3,), dtype=np.float64)
        expected_source = MOTION_FORCE_SOURCE_UNAVAILABLE
    elif current_active and pending_active:
        expected_available = False
        expected = np.zeros((3,), dtype=np.float64)
        expected_source = MOTION_FORCE_SOURCE_AMBIGUOUS_PENDING
    elif current_active:
        expected_available = True
        expected = current
        expected_source = (
            MOTION_FORCE_SOURCE_COMBINED_CURRENT
            if legacy_active and configured_active
            else (
                MOTION_FORCE_SOURCE_CONFIGURED_APPLIED
                if configured_active
                else MOTION_FORCE_SOURCE_LEGACY_EXTERNAL
            )
        )
    elif pending_active:
        expected_available = True
        expected = pending
        expected_source = MOTION_FORCE_SOURCE_PENDING_KNOCKBACK
    else:
        expected_available = True
        expected = np.zeros((3,), dtype=np.float64)
        expected_source = MOTION_FORCE_SOURCE_NONE
    if (
        projected_available != expected_available
        or source != expected_source
        or not np.allclose(projected, expected, rtol=0.0, atol=1.0e-12)
    ):
        raise ValueError("native force projection is inconsistent with raw channels")
    return {
        "legacy_external_available": legacy_available,
        "legacy_external_velocity": legacy,
        "configured_applied_available": configured_available,
        "configured_applied_velocity": configured,
        "configured_applied_count": configured_count,
        "pending_knockback_available": pending_available,
        "pending_knockback_velocity": pending,
        "projected_available": projected_available,
        "projected_velocity": projected,
        "projection_source": source,
    }


def _absent_actor_has_evidence(actor) -> bool:
    scalar = (
        actor["perceptible"]
        or actor["role_id"]
        or actor["item_id"]
        or actor["active_ability_slot"] != -1
        or actor["yaw_degrees"] != 0.0
        or actor["pitch_degrees"] != 0.0
        or actor["health"] != 0.0
        or actor["max_health"] != 0.0
        or actor["statuses"]
        or actor["movement_states"]["available"]
        or actor["movement_states"]["bits"]
    )
    arrays = (
        actor["position"],
        actor["velocity"],
        actor["motion_force"]["legacy_external_velocity"],
        actor["motion_force"]["configured_applied_velocity"],
        actor["motion_force"]["pending_knockback_velocity"],
        actor["motion_force"]["projected_velocity"],
        actor["resource_values"],
        actor["resource_maximums"],
        actor["resource_available"],
        actor["defense_values"],
        actor["defense_available"],
        actor["actor_world_values"],
        actor["actor_world_available"],
    )
    motion_force = actor["motion_force"]
    force_scalar = (
        motion_force["legacy_external_available"]
        or motion_force["configured_applied_available"]
        or motion_force["configured_applied_count"] != 0
        or motion_force["pending_knockback_available"]
        or motion_force["projected_available"]
    )
    return bool(scalar or force_scalar or any(np.any(value) for value in arrays))


def _status(value: Any) -> dict[str, Any]:
    row = _mapping(value, "status")
    effect_id = _string(row.get("effect_id"), "effect_id")
    if not effect_id:
        raise ValueError("native status effect_id is empty")
    initial = _number(
        row.get("initial_duration_seconds"), "initial_duration_seconds"
    )
    remaining = _number(
        row.get("remaining_duration_seconds"), "remaining_duration_seconds"
    )
    if initial < 0.0 or remaining < 0.0:
        raise ValueError("native status durations must be nonnegative")
    return {
        "runtime_effect_index": _integer(
            row.get("runtime_effect_index"), "runtime_effect_index"
        ),
        "effect_id": effect_id,
        "initial_duration_seconds": initial,
        "remaining_duration_seconds": remaining,
        "infinite": _boolean(row.get("infinite"), "infinite"),
        "debuff": _boolean(row.get("debuff"), "debuff"),
        "invulnerable": _boolean(row.get("invulnerable"), "invulnerable"),
    }


def _ability(value: Any, slot: int) -> dict[str, Any]:
    row = _mapping(value, "ability")
    if _integer(row.get("slot"), "slot") != slot:
        raise ValueError("native abilities are not in canonical slot order")
    authored = _boolean(row.get("authored"), "authored")
    result = {
        "slot": slot,
        "interaction_id": _string(
            row.get("interaction_id"), "interaction_id"
        ),
        "interaction_type": _string(
            row.get("interaction_type"), "interaction_type"
        ),
        "authored": authored,
        "host_legal": _boolean(row.get("host_legal"), "host_legal"),
        "active": _boolean(row.get("active"), "active"),
        "start_world_tick": _integer(
            row.get("start_world_tick"), "start_world_tick"
        ),
        "finish_world_tick": _integer(
            row.get("finish_world_tick"), "finish_world_tick"
        ),
    }
    if not authored and (
        result["interaction_id"]
        or result["interaction_type"]
        or result["host_legal"]
        or result["active"]
    ):
        raise ValueError("unauthored ability carries usable evidence")
    return result


def _resources(actors, rules):
    minimum = np.asarray(rules.resource_minimum, dtype=np.float32)
    maximum = np.asarray(rules.resource_maximum, dtype=np.float32)
    values = minimum.copy()
    available = np.zeros_like(values, dtype=np.bool_)
    for entity, actor in enumerate(actors):
        if not actor["present"]:
            continue
        raw_available = actor["resource_available"]
        static_available = maximum[0, entity] > minimum[0, entity]
        maximum_matches = np.isclose(
            actor["resource_maximums"],
            maximum[0, entity],
            rtol=0.0,
            atol=1.0e-5,
        )
        usable = raw_available & static_available & maximum_matches
        values[0, entity] = np.where(
            usable,
            np.clip(
                actor["resource_values"],
                minimum[0, entity],
                maximum[0, entity],
            ),
            minimum[0, entity],
        )
        available[0, entity] = usable
    return values, available


def _required_resources_available(loadout, available):
    minimum = np.asarray(loadout.ability_resource_minimum)
    return np.all((minimum <= 0.0) | available[:, :, None, :], axis=3)


def _status_programs(loadout) -> dict[int, _StatusProgram]:
    result: dict[int, _StatusProgram] = {}
    ambiguous: set[int] = set()
    mask = np.asarray(loadout.event_mask, dtype=np.bool_)
    f32 = np.asarray(loadout.event_f32)
    i32 = np.asarray(loadout.event_i32)
    flags = np.asarray(loadout.event_flags, dtype=np.uint32)
    for raw_index in np.argwhere(mask):
        index = tuple(int(value) for value in raw_index)
        status_id = int(i32[index + (EI_STATUS_ID,)])
        if status_id <= 0 or status_id in ambiguous:
            continue
        program = _StatusProgram(
            damage=float(f32[index + (EF_STATUS_DAMAGE,)]),
            healing=float(f32[index + (EF_STATUS_HEALING,)]),
            cooldown=float(f32[index + (EF_STATUS_COOLDOWN_SECONDS,)]),
            resource_delta=float(f32[index + (EF_STATUS_RESOURCE_DELTA,)]),
            speed_multiplier=float(f32[index + (EF_STATUS_SPEED_MULTIPLIER,)]),
            damage_cause=int(i32[index + (EI_STATUS_DAMAGE_CAUSE,)]),
            resource_id=int(i32[index + (EI_STATUS_RESOURCE_ID,)]),
            overlap_mode=int(i32[index + (EI_STATUS_OVERLAP_MODE,)]),
            target_mode=int(i32[index + (EI_TARGET_MODE,)]),
            flags=int(flags[index] & np.uint32(EVENT_STATUS_FLAG_MASK)),
            value_percent=bool(
                flags[index] & np.uint32(EVENT_FLAG_STATUS_VALUE_PERCENT)
            ),
            infinite=False,
            damage_resistance_present=(False,) * DAMAGE_COUNT,
            damage_resistance_flat=(0,) * DAMAGE_COUNT,
            damage_resistance_multiplier=(0.0,) * DAMAGE_COUNT,
        )
        if status_id in result and result[status_id] != program:
            result.pop(status_id)
            ambiguous.add(status_id)
        else:
            result[status_id] = program
    for status_id, status in role_status_programs_by_semantic_id().items():
        profile = status.damage_resistance
        program = _StatusProgram(
            damage=status.damage_per_cycle,
            healing=status.healing_per_cycle,
            cooldown=status.cycle_cooldown_seconds,
            resource_delta=status.resource_delta_per_cycle,
            speed_multiplier=status.speed_multiplier,
            damage_cause=status.damage_cause,
            resource_id=status.resource_id,
            overlap_mode=status.overlap_mode,
            target_mode=TARGET_SELF,
            flags=status.flags,
            value_percent=False,
            infinite=status.infinite,
            damage_resistance_present=profile.present,
            damage_resistance_flat=profile.flat,
            damage_resistance_multiplier=profile.multiplier,
        )
        if status_id in result and result[status_id] != program:
            result.pop(status_id)
            ambiguous.add(status_id)
        elif status_id not in ambiguous:
            result[status_id] = program
    for status_id, status in (
        entity_interaction_status_programs_by_semantic_id().items()
    ):
        program = _StatusProgram(
            damage=status.damage_per_cycle,
            healing=status.healing_per_cycle,
            cooldown=status.cycle_cooldown_seconds,
            resource_delta=status.resource_delta_per_cycle,
            speed_multiplier=status.speed_multiplier,
            damage_cause=status.damage_cause,
            resource_id=status.resource_id,
            overlap_mode=status.overlap_mode,
            target_mode=(
                TARGET_OTHER if status.source_entity_is_other else TARGET_SELF
            ),
            flags=status.flags,
            value_percent=status.value_percent,
            infinite=status.infinite,
            damage_resistance_present=status.damage_resistance_present,
            damage_resistance_flat=status.damage_resistance_flat,
            damage_resistance_multiplier=(
                status.damage_resistance_multiplier
            ),
        )
        if status_id in result and result[status_id] != program:
            result.pop(status_id)
            ambiguous.add(status_id)
        elif status_id not in ambiguous:
            result[status_id] = program
    return result


def _statuses(actors, programs, omitted_statuses, empty):
    values = {
        name: np.asarray(getattr(empty, name)).copy()
        for name in empty._fields
    }
    unknown = False
    for entity, actor in enumerate(actors):
        if not actor["present"]:
            continue
        slot = 0
        for raw in actor["statuses"]:
            if raw["effect_id"] in omitted_statuses:
                continue
            effect_id = semantic_id(raw["effect_id"])
            program = programs.get(effect_id)
            observed_flags = (
                (STATUS_FLAG_DEBUFF if raw["debuff"] else 0)
                | (STATUS_FLAG_INVULNERABLE if raw["invulnerable"] else 0)
            )
            if program is None or program.infinite != raw["infinite"]:
                unknown = True
                program = _StatusProgram(
                    damage=0.0,
                    healing=0.0,
                    cooldown=0.0,
                    resource_delta=0.0,
                    speed_multiplier=1.0,
                    damage_cause=0,
                    resource_id=-1,
                    overlap_mode=0,
                    target_mode=(
                        TARGET_OTHER if raw["debuff"] else entity
                    ),
                    flags=observed_flags,
                    value_percent=False,
                    infinite=raw["infinite"],
                    damage_resistance_present=(False,) * DAMAGE_COUNT,
                    damage_resistance_flat=(0,) * DAMAGE_COUNT,
                    damage_resistance_multiplier=(0.0,) * DAMAGE_COUNT,
                )
            health = max(float(actor["max_health"]), 1.0e-6)
            damage = program.damage * health if program.value_percent else program.damage
            healing = (
                program.healing * health
                if program.value_percent
                else program.healing
            )
            source = (
                TARGET_ENTITY if entity == AGENT_ENTITY else AGENT_ENTITY
            ) if program.target_mode == TARGET_OTHER else entity
            updates = {
                "effect_id": effect_id,
                "source_entity_id": source,
                "remaining_seconds": (
                    np.finfo(np.float32).max
                    if program.infinite
                    else raw["remaining_duration_seconds"]
                ),
                "cycle_cooldown_seconds": program.cooldown,
                "damage_per_cycle": damage,
                "damage_cause": program.damage_cause,
                "healing_per_cycle": healing,
                "resource_id": program.resource_id,
                "resource_delta_per_cycle": program.resource_delta,
                "speed_multiplier": program.speed_multiplier,
                "damage_resistance_present": (
                    program.damage_resistance_present
                ),
                "damage_resistance_flat": program.damage_resistance_flat,
                "damage_resistance_multiplier": (
                    program.damage_resistance_multiplier
                ),
                "flags": np.uint32(program.flags | observed_flags),
                "overlap_mode": program.overlap_mode,
                "active": True,
            }
            for name, value in updates.items():
                values[name][0, entity, slot] = value
            slot += 1
    return type(empty)(
        *(jnp.asarray(values[name]) for name in empty._fields)
    ), unknown


def _combat_info(info: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "agent_attack_executing": ("combat_agent_attack_executing", 0.0),
        "target_attack_phase": ("combat_target_attack_phase", 0.0),
        "target_attack_index": ("combat_target_attack_index", -1.0),
        "target_attack_elapsed_ticks": (
            "combat_target_attack_elapsed_ticks",
            0.0,
        ),
        "target_head_yaw_degrees": (
            "combat_target_head_yaw_degrees",
            0.0,
        ),
        "target_head_pitch_degrees": (
            "combat_target_head_pitch_degrees",
            0.0,
        ),
    }
    result = {
        name: _number(info.get(wire_name, default), wire_name)
        for name, (wire_name, default) in fields.items()
    }
    result["target_attack_phase"] = int(
        np.clip(np.rint(result["target_attack_phase"]), 0, 4)
    )
    result["target_attack_index"] = int(
        np.rint(result["target_attack_index"])
    )
    result["target_attack_elapsed_ticks"] = max(
        0,
        int(np.rint(result["target_attack_elapsed_ticks"])),
    )
    return result


def _exact_geometry(observation, wire):
    frame = observation.get("geometry")
    if (
        not wire["world_geometry_available"]
        or not isinstance(frame, Mapping)
        or not bool(int(frame.get("available", 0)))
    ):
        return None
    return geometry_state_from_numpy(frame)


def _movement_row(actor):
    movement = actor["movement_states"]
    available = movement["available"]
    bits = movement["bits"]
    values = np.asarray(
        [[(bits >> index) & 1 for index in range(MOVEMENT_STATE_SIZE)]],
        dtype=np.float32,
    )
    mask = np.full(values.shape, available, dtype=np.bool_)
    return values, mask


def _movement_state(actor, index):
    movement = actor["movement_states"]
    return bool(movement["available"] and (movement["bits"] & (1 << index)))


def _active_ability_slot(abilities):
    active = [row["slot"] for row in abilities if row["active"]]
    if len(active) > 1:
        raise ValueError("multiple native abilities are active")
    return active[0] if active else -1


def _loadout_mismatch(wire, loadout):
    item_id = wire["actors"][AGENT_ENTITY]["item_id"]
    if not item_id:
        return False
    return semantic_id(item_id) != int(loadout.weapon_id[0, AGENT_ENTITY])


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"native actor evidence {name} must be an object")
    return value


def _sequence(value, name):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"native actor evidence {name} must be an array")
    return value


def _string(value, name):
    if not isinstance(value, str):
        raise ValueError(f"native actor evidence {name} must be a string")
    return value


def _boolean(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"native actor evidence {name} must be boolean")
    return bool(value)


def _native_use_action_available(info: Mapping[str, Any]) -> bool:
    """Bind per-tick authored Use evidence, never bridge support alone."""

    if "native_use_available" not in info:
        return False
    available = _boolean(
        info.get("native_use_available"),
        "native_use_available",
    )
    supported = info.get("supported_actions")
    if not isinstance(supported, str):
        raise ValueError(
            "native supported-actions evidence is unavailable"
        )
    use_supported = "use" in {
        value.strip()
        for value in supported.split(",")
        if value.strip()
    }
    if available and not use_supported:
        raise ValueError(
            "native Use is actor-available but unsupported by the bridge"
        )
    return available and use_supported


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"native actor evidence {name} must be an integer")
    return int(value)


def _number(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"native actor evidence {name} must be numeric"
        ) from error
    if not np.isfinite(result):
        raise ValueError(f"native actor evidence {name} must be finite")
    return result


def _array(value, shape, dtype, name):
    raw = np.asarray(value)
    if dtype == np.bool_ and raw.dtype != np.bool_:
        raise ValueError(f"native actor evidence {name} must be boolean")
    result = np.asarray(value, dtype=dtype)
    if result.shape != shape:
        raise ValueError(
            f"native actor evidence {name} must have shape {shape}"
        )
    if np.issubdtype(result.dtype, np.floating) and not np.all(
        np.isfinite(result)
    ):
        raise ValueError(f"native actor evidence {name} must be finite")
    return result
