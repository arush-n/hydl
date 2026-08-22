"""Backend-neutral decision inputs and host-side stale-result guards.

The normalized tensors are additive views over upstream diagnostics.  Their
``known`` masks distinguish a real false value from a field that a backend
does not publish.  :class:`DecisionContextFrame` retains the exact original
``info`` object beside that actor-safe view, so normalization never becomes a
lossy replacement for Gym or bridge evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.combat.telemetry import CombatPhase
from hytalegym.jax.combat.types import AGENT_ENTITY
from hytalegym.jax.combat.observation import (
    SELF_FLOAT_FEATURES,
    TARGET_INTEGER_FEATURES,
)
from hytalegym.jax.combat.observation.v3.types import LearnerCombatObservationV3
from hytalegym.jax.combat.arsenal.environment import (
    ACTION_SURFACE_VERB_BLOCK_INTERACTION,
    ACTION_SURFACE_VERB_CRAFT,
    ACTION_SURFACE_VERB_USE,
    ArsenalEnvironmentInfo,
)

from adk.architecture.inputs import ActorPolicyInput
from adk.development import EnvironmentDiagnostics


_TARGET_ATTACK_PHASE = TARGET_INTEGER_FEATURES.index("attack_phase")
_SELF_HEALTH_FRACTION = SELF_FLOAT_FEATURES.index("health_fraction")


class ActionVerb(IntEnum):
    """Stable superset; unsupported backend rows remain explicitly unknown."""

    GUARD = 0
    ABILITY = 1
    ATTACK = 2
    DODGE = 3
    USE = 4
    BLOCK_INTERACTION = 5
    PLACE_BLOCK = 6
    BREAK_BLOCK = 7
    CRAFT_RECIPE = 8


class ExecutionState(IntEnum):
    """PLAN execution states plus an explicit pre-execution rejection."""

    REQUESTED = 0
    VALIDATED = 1
    STARTED = 2
    IN_PROGRESS = 3
    COMPLETED = 4
    CANCELLED = 5
    INTERRUPTED = 6
    FAILED = 7
    REJECTED = 8


class DecisionEvent(IntEnum):
    """Stable early-decision/manager-interrupt event order from PLAN section 7.3."""

    ATTACK_TELEGRAPH_BEGAN = 0
    PROJECTILE_THREAT_ENTERED = 1
    DAMAGE_RECEIVED = 2
    TARGET_DIED = 3
    LINE_OF_SIGHT_CHANGED = 4
    PATH_INVALID = 5
    ABILITY_COMPLETED = 6
    ABILITY_INTERRUPTED = 7
    NPC_STUCK = 8
    GOAL_IMPOSSIBLE = 9
    ACTION_COMPLETED = 10
    ACTION_CANCELLED = 11
    ACTION_FAILED = 12
    ACTION_REJECTED = 13


class DecisionTiming(NamedTuple):
    """Per-lane decision time with availability for each normalized term."""

    tick: Any
    tick_known: Any
    delta_seconds: Any
    delta_seconds_known: Any


class ActionLifecycle(NamedTuple):
    """Boolean ``[batch, verb, state]`` values and matching availability."""

    values: Any
    known: Any

    def state(self, verb: ActionVerb, state: ExecutionState) -> Any:
        return self.values[..., int(verb), int(state)]

    def state_known(self, verb: ActionVerb, state: ExecutionState) -> Any:
        return self.known[..., int(verb), int(state)]


class DecisionEvents(NamedTuple):
    """Boolean ``[batch, event]`` values and matching availability."""

    values: Any
    known: Any

    def event(self, event: DecisionEvent) -> Any:
        return self.values[..., int(event)]

    def event_known(self, event: DecisionEvent) -> Any:
        return self.known[..., int(event)]


class DecisionContext(NamedTuple):
    """JAX-friendly normalized timing, lifecycle, and interrupt inputs."""

    timing: DecisionTiming
    lifecycle: ActionLifecycle
    events: DecisionEvents


@dataclass(frozen=True, slots=True)
class DecisionContextFrame:
    """Normalized input plus the normal lossless diagnostic wrapper.

    Pass only ``context`` to deployed policy/manager code. ``diagnostics`` can
    contain backend-specific or privileged fields and is deliberately a
    different type; its ``raw`` field retains the exact producer object.
    """

    context: DecisionContext
    diagnostics: EnvironmentDiagnostics

    @property
    def raw_info(self) -> Any:
        """Compatibility alias for the exact backend diagnostic object."""

        return self.diagnostics.raw


def jax_decision_context(
    info: ArsenalEnvironmentInfo,
    *,
    previous_info: ArsenalEnvironmentInfo | None = None,
    actor_input: ActorPolicyInput | None = None,
    previous_actor_input: ActorPolicyInput | None = None,
) -> DecisionContextFrame:
    """Map self/action diagnostics and legal structured perception.

    ``previous_info`` remains accepted for source compatibility, but raw
    diagnostics never produce observation-derived events. Those require
    current and previous ``ActorPolicyInput`` values or remain explicitly
    unknown. The current legal observation cannot establish target death or
    navigation-path invalidity, so those events stay unknown.
    """

    if not isinstance(info, ArsenalEnvironmentInfo):
        raise TypeError("info must be an ArsenalEnvironmentInfo")
    if previous_info is not None and not isinstance(
        previous_info,
        ArsenalEnvironmentInfo,
    ):
        raise TypeError("previous_info must be an ArsenalEnvironmentInfo")

    action_valid = jnp.asarray(info.action_valid)
    if action_valid.ndim != 1:
        raise ValueError("ArsenalEnvironmentInfo action_valid must have shape (B,)")
    batch = action_valid.shape[0]
    lifecycle_values = jnp.zeros(
        (batch, len(ActionVerb), len(ExecutionState)),
        dtype=jnp.bool_,
    )
    lifecycle_known = jnp.zeros_like(lifecycle_values)

    def publish(verb: ActionVerb, state: ExecutionState, value: Any):
        nonlocal lifecycle_values, lifecycle_known
        row = _jax_vector(value, batch, f"{verb.name}.{state.name}", jnp.bool_)
        lifecycle_values = lifecycle_values.at[:, int(verb), int(state)].set(row)
        lifecycle_known = lifecycle_known.at[:, int(verb), int(state)].set(True)

    combat = info.combat_info
    arsenal = info.arsenal_info
    attack_requested = _jax_vector(
        combat.attack_requested,
        batch,
        "combat_info.attack_requested",
        jnp.bool_,
    )
    attack_accepted = _jax_vector(
        combat.attack_accepted,
        batch,
        "combat_info.attack_accepted",
        jnp.bool_,
    )
    publish(ActionVerb.ATTACK, ExecutionState.REQUESTED, attack_requested)
    publish(ActionVerb.ATTACK, ExecutionState.VALIDATED, attack_accepted)
    publish(
        ActionVerb.ATTACK,
        ExecutionState.IN_PROGRESS,
        combat.agent_attack_executing,
    )
    publish(
        ActionVerb.ATTACK,
        ExecutionState.REJECTED,
        attack_requested & ~attack_accepted,
    )

    ability_requested = _jax_agent_vector(
        arsenal.ability_requested,
        batch,
        "arsenal_info.ability_requested",
    ).astype(jnp.bool_)
    ability_accepted = _jax_agent_vector(
        arsenal.ability_accepted,
        batch,
        "arsenal_info.ability_accepted",
    ).astype(jnp.bool_)
    active_ability = _jax_agent_vector(
        arsenal.active_ability_slot,
        batch,
        "arsenal_info.active_ability_slot",
    ) >= 0
    publish(ActionVerb.ABILITY, ExecutionState.REQUESTED, ability_requested)
    publish(ActionVerb.ABILITY, ExecutionState.VALIDATED, ability_accepted)
    publish(ActionVerb.ABILITY, ExecutionState.IN_PROGRESS, active_ability)
    publish(
        ActionVerb.ABILITY,
        ExecutionState.REJECTED,
        ability_requested & ~ability_accepted,
    )

    surface = info.action_surface_lifecycle
    surface_verbs = (
        (ActionVerb.USE, ACTION_SURFACE_VERB_USE),
        (ActionVerb.BLOCK_INTERACTION, ACTION_SURFACE_VERB_BLOCK_INTERACTION),
        (ActionVerb.CRAFT_RECIPE, ACTION_SURFACE_VERB_CRAFT),
    )
    surface_fields = (
        (ExecutionState.REQUESTED, "requested"),
        (ExecutionState.VALIDATED, "accepted"),
        (ExecutionState.STARTED, "started"),
        (ExecutionState.COMPLETED, "finished"),
        (ExecutionState.CANCELLED, "cancelled"),
        (ExecutionState.REJECTED, "rejected"),
    )
    for verb, column in surface_verbs:
        for state, name in surface_fields:
            matrix = jnp.asarray(getattr(surface, name))
            if matrix.shape != (batch, len(surface_verbs)):
                raise ValueError(
                    f"action_surface_lifecycle.{name} must have shape "
                    f"({batch}, {len(surface_verbs)})"
                )
            publish(verb, state, matrix[:, int(column)])

    lifecycle = ActionLifecycle(lifecycle_values, lifecycle_known)
    events = _jax_events(
        lifecycle,
        batch,
        actor_input=actor_input,
        previous_actor_input=previous_actor_input,
    )
    tick = _jax_vector(combat.tick_count, batch, "combat_info.tick_count")
    delta = _jax_vector(
        combat.motion_delta_seconds,
        batch,
        "combat_info.motion_delta_seconds",
    )
    return DecisionContextFrame(
        context=DecisionContext(
            timing=DecisionTiming(
                tick=tick,
                tick_known=jnp.ones((batch,), dtype=jnp.bool_),
                delta_seconds=delta,
                delta_seconds_known=jnp.ones((batch,), dtype=jnp.bool_),
            ),
            lifecycle=lifecycle,
            events=events,
        ),
        diagnostics=EnvironmentDiagnostics.from_info(info),
    )


def native_decision_context(
    info: Mapping[str, Any],
    *,
    previous_info: Mapping[str, Any] | None = None,
    actor_input: ActorPolicyInput | None = None,
    previous_actor_input: ActorPolicyInput | None = None,
) -> DecisionContextFrame:
    """Map native self/action evidence and legal structured actor events."""

    if not isinstance(info, Mapping):
        raise TypeError("info must be a native bridge mapping")
    if previous_info is not None and not isinstance(previous_info, Mapping):
        raise TypeError("previous_info must be a native bridge mapping")

    values = np.zeros((1, len(ActionVerb), len(ExecutionState)), dtype=np.bool_)
    known = np.zeros_like(values)

    native_fields = {
        ActionVerb.GUARD: {
            ExecutionState.REQUESTED: "native_guard_requested",
            ExecutionState.VALIDATED: "native_guard_accepted",
            ExecutionState.STARTED: "native_guard_started",
            ExecutionState.IN_PROGRESS: "native_guard_active",
            ExecutionState.COMPLETED: "native_guard_finished",
        },
        ActionVerb.ABILITY: {
            ExecutionState.REQUESTED: "native_ability_requested",
            ExecutionState.VALIDATED: "native_ability_accepted",
            ExecutionState.STARTED: "native_ability_started",
        },
        ActionVerb.ATTACK: {
            ExecutionState.REQUESTED: "native_attack_requested",
            ExecutionState.VALIDATED: "native_attack_accepted",
            ExecutionState.IN_PROGRESS: "native_attack_executing",
        },
        ActionVerb.DODGE: {
            ExecutionState.REQUESTED: "native_dodge_requested",
            ExecutionState.VALIDATED: "native_dodge_accepted",
            ExecutionState.STARTED: "native_dodge_started",
        },
        ActionVerb.USE: {
            ExecutionState.REQUESTED: "native_use_requested",
            ExecutionState.VALIDATED: "native_use_accepted",
            ExecutionState.STARTED: "native_use_started",
            ExecutionState.IN_PROGRESS: "native_use_active",
        },
        ActionVerb.PLACE_BLOCK: {
            ExecutionState.REQUESTED: "native_place_block_requested",
            ExecutionState.VALIDATED: "native_place_block_accepted",
        },
        ActionVerb.BREAK_BLOCK: {
            ExecutionState.REQUESTED: "native_break_block_requested",
            ExecutionState.VALIDATED: "native_break_block_accepted",
        },
        ActionVerb.CRAFT_RECIPE: {
            ExecutionState.REQUESTED: "native_craft_recipe_requested",
            ExecutionState.VALIDATED: "native_craft_recipe_accepted",
        },
    }
    for verb, fields in native_fields.items():
        for state, key in fields.items():
            value, available = _native_bool(info, key)
            if available:
                values[0, int(verb), int(state)] = value
                known[0, int(verb), int(state)] = True

        _publish_native_rejection(info, values, known, verb)

    for verb, prefix in (
        (ActionVerb.ABILITY, "ability"),
        (ActionVerb.DODGE, "dodge"),
        (ActionVerb.USE, "use"),
    ):
        _publish_native_terminal_outcome(info, values, known, verb, prefix)

    active_slot, active_slot_known = _native_integer(
        info,
        "native_ability_active_slot",
    )
    if active_slot_known:
        values[0, int(ActionVerb.ABILITY), int(ExecutionState.IN_PROGRESS)] = (
            active_slot >= 0
        )
        known[0, int(ActionVerb.ABILITY), int(ExecutionState.IN_PROGRESS)] = True

    dodge_start, dodge_start_known = _native_integer(
        info,
        "native_dodge_chain_start_world_tick",
    )
    dodge_finish, dodge_finish_known = _native_integer(
        info,
        "native_dodge_chain_finish_world_tick",
    )
    if dodge_start_known and dodge_finish_known:
        values[0, int(ActionVerb.DODGE), int(ExecutionState.IN_PROGRESS)] = (
            dodge_start >= 0 and dodge_finish < 0
        )
        known[0, int(ActionVerb.DODGE), int(ExecutionState.IN_PROGRESS)] = True

    lifecycle = ActionLifecycle(values, known)
    tick, tick_known = _native_integer(info, "native_world_tick")
    delta, delta_known = _native_number(info, "engine_simulated_seconds")
    return DecisionContextFrame(
        context=DecisionContext(
            timing=DecisionTiming(
                tick=np.asarray([tick], dtype=np.int64),
                tick_known=np.asarray([tick_known], dtype=np.bool_),
                delta_seconds=np.asarray([delta], dtype=np.float64),
                delta_seconds_known=np.asarray([delta_known], dtype=np.bool_),
            ),
            lifecycle=lifecycle,
            events=_native_events(
                lifecycle,
                actor_input=actor_input,
                previous_actor_input=previous_actor_input,
            ),
        ),
        diagnostics=EnvironmentDiagnostics.from_info(info),
    )


def manager_decision_due(
    context: DecisionContext,
    seconds_since_decision: Any,
    cadence_seconds: Any,
    goal_terminated: Any,
) -> jax.Array:
    """Return cadence-or-known-event-or-goal-termination manager decisions."""

    critical = jnp.any(context.events.values & context.events.known, axis=-1)
    return (
        jnp.asarray(seconds_since_decision) >= jnp.asarray(cadence_seconds)
    ) | jnp.asarray(goal_terminated, dtype=jnp.bool_) | critical


def _jax_events(
    lifecycle: ActionLifecycle,
    batch: int,
    *,
    actor_input: ActorPolicyInput | None,
    previous_actor_input: ActorPolicyInput | None,
) -> DecisionEvents:
    values = jnp.zeros((batch, len(DecisionEvent)), dtype=jnp.bool_)
    known = jnp.zeros_like(values)

    values, known = _publish_lifecycle_events_jax(values, known, lifecycle)
    perceptual_values, perceptual_known = _legal_perceptual_events(
        actor_input,
        previous_actor_input,
        batch,
    )
    values = values | perceptual_values
    known = known | perceptual_known
    return DecisionEvents(values, known)


def _publish_lifecycle_events_jax(
    values: jax.Array,
    known: jax.Array,
    lifecycle: ActionLifecycle,
) -> tuple[jax.Array, jax.Array]:
    event_states = (
        (DecisionEvent.ACTION_COMPLETED, ExecutionState.COMPLETED),
        (DecisionEvent.ACTION_CANCELLED, ExecutionState.CANCELLED),
        (DecisionEvent.ACTION_FAILED, ExecutionState.FAILED),
        (DecisionEvent.ACTION_REJECTED, ExecutionState.REJECTED),
    )
    for event, state in event_states:
        result, available = _known_any(
            lifecycle.values[:, :, int(state)],
            lifecycle.known[:, :, int(state)],
            axis=1,
        )
        values = values.at[:, int(event)].set(result)
        known = known.at[:, int(event)].set(available)
    ability_completed = lifecycle.state(ActionVerb.ABILITY, ExecutionState.COMPLETED)
    ability_completed_known = lifecycle.state_known(
        ActionVerb.ABILITY,
        ExecutionState.COMPLETED,
    )
    values = values.at[:, int(DecisionEvent.ABILITY_COMPLETED)].set(
        ability_completed & ability_completed_known
    )
    known = known.at[:, int(DecisionEvent.ABILITY_COMPLETED)].set(
        ability_completed_known
    )
    ability_interrupted = lifecycle.state(
        ActionVerb.ABILITY,
        ExecutionState.INTERRUPTED,
    )
    ability_interrupted_known = lifecycle.state_known(
        ActionVerb.ABILITY,
        ExecutionState.INTERRUPTED,
    )
    values = values.at[:, int(DecisionEvent.ABILITY_INTERRUPTED)].set(
        ability_interrupted & ability_interrupted_known
    )
    known = known.at[:, int(DecisionEvent.ABILITY_INTERRUPTED)].set(
        ability_interrupted_known
    )
    return values, known


def _native_events(
    lifecycle: ActionLifecycle,
    *,
    actor_input: ActorPolicyInput | None,
    previous_actor_input: ActorPolicyInput | None,
) -> DecisionEvents:
    values = np.zeros((1, len(DecisionEvent)), dtype=np.bool_)
    known = np.zeros_like(values)

    for event, state in (
        (DecisionEvent.ACTION_COMPLETED, ExecutionState.COMPLETED),
        (DecisionEvent.ACTION_CANCELLED, ExecutionState.CANCELLED),
        (DecisionEvent.ACTION_FAILED, ExecutionState.FAILED),
        (DecisionEvent.ACTION_REJECTED, ExecutionState.REJECTED),
    ):
        result, available = _known_any_numpy(
            lifecycle.values[:, :, int(state)],
            lifecycle.known[:, :, int(state)],
            axis=1,
        )
        values[:, int(event)] = result
        known[:, int(event)] = available
    for event, state in (
        (DecisionEvent.ABILITY_COMPLETED, ExecutionState.COMPLETED),
        (DecisionEvent.ABILITY_INTERRUPTED, ExecutionState.INTERRUPTED),
    ):
        available = lifecycle.state_known(ActionVerb.ABILITY, state)
        values[:, int(event)] = lifecycle.state(ActionVerb.ABILITY, state) & available
        known[:, int(event)] = available
    perceptual_values, perceptual_known = _legal_perceptual_events(
        actor_input,
        previous_actor_input,
        1,
    )
    values |= np.asarray(jax.device_get(perceptual_values), dtype=np.bool_)
    known |= np.asarray(jax.device_get(perceptual_known), dtype=np.bool_)
    return DecisionEvents(values, known)


def _legal_perceptual_events(
    actor_input: ActorPolicyInput | None,
    previous_actor_input: ActorPolicyInput | None,
    batch: int,
) -> tuple[jax.Array, jax.Array]:
    """Derive actor-observable edges from the leakage-filtered actor product.

    In particular, ``ArsenalInfo.damage_received`` is step diagnostics outside
    the actor observation.  Damage is therefore the observable decrease in
    self health between two valid legal observations, not that privileged
    diagnostic magnitude.
    """

    values = jnp.zeros((batch, len(DecisionEvent)), dtype=jnp.bool_)
    known = jnp.zeros_like(values)
    if actor_input is None and previous_actor_input is None:
        return values, known
    if actor_input is None or previous_actor_input is None:
        raise ValueError(
            "actor-observable decision events require current and previous "
            "ActorPolicyInput values"
        )
    visible, phase, health, available = _legal_actor_evidence(
        actor_input,
        batch,
        "actor_input",
    )
    old_visible, old_phase, old_health, old_available = _legal_actor_evidence(
        previous_actor_input,
        batch,
        "previous_actor_input",
    )
    edge_known = available & old_available
    values = values.at[:, int(DecisionEvent.LINE_OF_SIGHT_CHANGED)].set(
        (visible != old_visible) & edge_known
    )
    known = known.at[:, int(DecisionEvent.LINE_OF_SIGHT_CHANGED)].set(edge_known)
    telegraph = (
        visible
        & (phase == int(CombatPhase.WINDUP))
        & (~old_visible | (old_phase != int(CombatPhase.WINDUP)))
    )
    values = values.at[:, int(DecisionEvent.ATTACK_TELEGRAPH_BEGAN)].set(
        telegraph & edge_known
    )
    known = known.at[:, int(DecisionEvent.ATTACK_TELEGRAPH_BEGAN)].set(edge_known)
    values = values.at[:, int(DecisionEvent.DAMAGE_RECEIVED)].set(
        (health < old_health) & edge_known
    )
    known = known.at[:, int(DecisionEvent.DAMAGE_RECEIVED)].set(edge_known)
    return values, known


def _legal_actor_evidence(
    actor_input: ActorPolicyInput,
    batch: int,
    name: str,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    if not isinstance(actor_input, ActorPolicyInput):
        raise TypeError(f"{name} must be ActorPolicyInput")
    observation = actor_input.legal_observation
    if not isinstance(observation, LearnerCombatObservationV3):
        raise TypeError(
            f"{name}.legal_observation must be LearnerCombatObservationV3"
        )
    base = observation.base
    visible = _jax_vector(base.target_mask, batch, f"{name}.base.target_mask", jnp.bool_)
    valid = _jax_vector(observation.valid, batch, f"{name}.valid", jnp.bool_)
    base_valid = _jax_vector(base.valid, batch, f"{name}.base.valid", jnp.bool_)
    target_i32 = jnp.asarray(base.target_i32)
    if target_i32.ndim != 2 or target_i32.shape[0] != batch:
        raise ValueError(f"{name}.base.target_i32 must have shape (B, features)")
    if target_i32.shape[1] <= _TARGET_ATTACK_PHASE:
        raise ValueError(f"{name}.base.target_i32 has no attack-phase field")
    self_f32 = jnp.asarray(base.self_f32)
    if self_f32.ndim != 2 or self_f32.shape[0] != batch:
        raise ValueError(f"{name}.base.self_f32 must have shape (B, features)")
    if self_f32.shape[1] <= _SELF_HEALTH_FRACTION:
        raise ValueError(f"{name}.base.self_f32 has no health-fraction field")
    return (
        visible,
        target_i32[:, _TARGET_ATTACK_PHASE],
        self_f32[:, _SELF_HEALTH_FRACTION],
        valid & base_valid,
    )


def _jax_vector(
    value: Any,
    batch: int,
    name: str,
    dtype: Any | None = None,
) -> jax.Array:
    result = jnp.asarray(value, dtype=dtype)
    if result.shape != (batch,):
        raise ValueError(f"{name} must have shape ({batch},)")
    return result


def _jax_agent_vector(value: Any, batch: int, name: str) -> jax.Array:
    result = jnp.asarray(value)
    if result.ndim != 2 or result.shape[0] != batch or result.shape[1] <= AGENT_ENTITY:
        raise ValueError(f"{name} must have shape (B, entities)")
    return result[:, AGENT_ENTITY]


def _known_any(values: Any, known: Any, *, axis: int) -> tuple[Any, Any]:
    any_true = jnp.any(values & known, axis=axis)
    return any_true, any_true | jnp.all(known, axis=axis)


def _known_any_numpy(values: Any, known: Any, *, axis: int) -> tuple[Any, Any]:
    any_true = np.any(values & known, axis=axis)
    return any_true, any_true | np.all(known, axis=axis)


def _publish_native_rejection(
    info: Mapping[str, Any],
    values: np.ndarray,
    known: np.ndarray,
    verb: ActionVerb,
) -> None:
    requested_index = int(ExecutionState.REQUESTED)
    validated_index = int(ExecutionState.VALIDATED)
    rejected_index = int(ExecutionState.REJECTED)
    row = int(verb)
    if not known[0, row, requested_index]:
        return
    requested = bool(values[0, row, requested_index])
    if not requested:
        values[0, row, rejected_index] = False
        known[0, row, rejected_index] = True
        return
    if not known[0, row, validated_index]:
        return
    accepted = bool(values[0, row, validated_index])
    if accepted:
        values[0, row, rejected_index] = False
        known[0, row, rejected_index] = True
        return

    reason_key = {
        ActionVerb.GUARD: "native_guard_reject_reason",
        ActionVerb.ABILITY: "native_ability_reject_reason",
        ActionVerb.DODGE: "native_dodge_reject_reason",
        ActionVerb.USE: "native_use_reject_reason",
        ActionVerb.PLACE_BLOCK: "native_place_block_reject_reason",
        ActionVerb.BREAK_BLOCK: "native_break_block_reject_reason",
        ActionVerb.CRAFT_RECIPE: "native_craft_recipe_reject_reason",
    }.get(verb)
    if reason_key is None:
        # Native attack admission is synchronous and publishes no separate
        # rejection-reason field. Its accepted bit is therefore definitive.
        values[0, row, rejected_index] = True
        known[0, row, rejected_index] = True
        return

    reason, reason_known = _native_string(info, reason_key)
    if reason_known and reason:
        values[0, row, rejected_index] = True
        known[0, row, rejected_index] = True
    # Chain-backed verbs can remain unaccepted with no rejection reason until
    # InteractionManager applies its authoritative start gate. Incomplete
    # evidence therefore stays unknown instead of becoming a false rejection.


def _publish_native_terminal_outcome(
    info: Mapping[str, Any],
    values: np.ndarray,
    known: np.ndarray,
    verb: ActionVerb,
    prefix: str,
) -> None:
    """Split native ``finished`` (terminal) into success and failure states."""

    terminal, terminal_known = _native_bool(info, f"native_{prefix}_finished")
    failed, failed_known = _native_bool(info, f"native_{prefix}_failed")
    row = int(verb)
    completed_index = int(ExecutionState.COMPLETED)
    failed_index = int(ExecutionState.FAILED)
    if failed_known:
        values[0, row, failed_index] = failed
        known[0, row, failed_index] = True
    if not terminal_known:
        return
    if not terminal:
        values[0, row, completed_index] = False
        known[0, row, completed_index] = True
        return
    if failed_known:
        values[0, row, completed_index] = not failed
        known[0, row, completed_index] = True


def _native_bool(info: Mapping[str, Any], key: str) -> tuple[bool, bool]:
    if key not in info:
        return False, False
    value = info[key]
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"native field {key!r} must be bool")
    return bool(value), True


def _native_string(info: Mapping[str, Any], key: str) -> tuple[str, bool]:
    if key not in info:
        return "", False
    value = info[key]
    if not isinstance(value, str):
        raise TypeError(f"native field {key!r} must be a string")
    return value, True


def _native_integer(info: Mapping[str, Any], key: str) -> tuple[int, bool]:
    if key not in info:
        return 0, False
    value = info[key]
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"native field {key!r} must be an integer")
    return int(value), True


def _native_number(info: Mapping[str, Any], key: str) -> tuple[float, bool]:
    if key not in info:
        return 0.0, False
    value = info[key]
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"native field {key!r} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"native field {key!r} must be finite")
    return result, True


__all__ = [
    "ActionLifecycle",
    "ActionVerb",
    "DecisionContext",
    "DecisionContextFrame",
    "DecisionEvent",
    "DecisionEvents",
    "DecisionTiming",
    "ExecutionState",
    "jax_decision_context",
    "manager_decision_due",
    "native_decision_context",
]
