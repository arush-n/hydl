"""Recurrent actor-major rollout collection for shared combat arenas."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from hytalegym.jax.combat.arsenal.factory import empty_arsenal_commands
from hytalegym.jax.combat.arsenal.environment import (
    ArsenalOpponentAbilityProvider,
)
from hytalegym.jax.combat.arsenal.schema.types import (
    ArsenalEnvironmentState,
    ArsenalRuntimeConfig,
    ArsenalWorldCapabilities,
)
from hytalegym.jax.combat.observation.v3.policy import (
    sample_arsenal_action_factors,
)
from hytalegym.jax.combat.mechanics import RESOURCE_STAMINA
from hytalegym.jax.combat.types import CombatParams
from hytalegym.jax.training.types import RecurrentPolicyParams

from .assignment import PolicyActorAssignment, policy_controlled_entity_mask
from .decoding import decode_policy_actor_factors_flat
from .inference import apply_assigned_actor_controllers
from .observations import PolicyActorObservation, observe_policy_actors_flat
from .runtime import (
    MultiActorArenaState,
    initialize_multi_actor_arena_state,
    step_multi_actor_arena,
)


ArenaResetProvider = Callable[[jax.Array], ArsenalEnvironmentState]
ArenaCapabilityProvider = Callable[[ArsenalEnvironmentState], ArsenalWorldCapabilities]
ActionMaskTransform = Callable[
    [MultiActorArenaState, PolicyActorObservation, jax.Array], jax.Array
]
ActorTransitionTransform = Callable[
    [MultiActorArenaState, Any],
    Any,
]
ObservationTelemetryTransform = Callable[[PolicyActorObservation], jax.Array]
ActorObservationTransform = Callable[[MultiActorArenaState, Any], Any]


def _apply_transition_transform(transform, arena, transition):
    """Apply an adapter without collapsing flat, Region, or future evidence."""

    transformed = transform(arena, transition)
    if not isinstance(transformed, type(transition)):
        raise TypeError("transition_transform must preserve the transition type")
    return transformed


def _apply_observation_transform(transform, arena, observation):
    """Apply value-only policy context without changing the public tree ABI."""

    transformed = transform(arena, observation)
    if not isinstance(transformed, type(observation)):
        raise TypeError("observation_transform must preserve the observation type")
    before = jax.tree_util.tree_leaves(observation)
    after = jax.tree_util.tree_leaves(transformed)
    if len(before) != len(after) or any(
        left.shape != right.shape or left.dtype != right.dtype
        for left, right in zip(before, after, strict=True)
    ):
        raise ValueError("observation_transform must preserve every leaf ABI")
    return transformed


class MultiActorRollout(NamedTuple):
    """Time-major behavior facts with separate arena and actor axes."""

    observation: jax.Array
    observation_telemetry: jax.Array
    action_mask: jax.Array
    action: jax.Array
    log_probability: jax.Array
    value: jax.Array
    reward: jax.Array
    done: jax.Array
    episode_start: jax.Array
    valid: jax.Array
    policy_id: jax.Array
    trainable: jax.Array
    completed_episode_return: jax.Array
    completed_episode_length: jax.Array
    completed_episode_success: jax.Array
    completed_episode_death: jax.Array
    completed_episode_simultaneous: jax.Array
    completed_episode_other: jax.Array
    entity_position: jax.Array
    entity_yaw: jax.Array
    entity_health: jax.Array
    entity_grounded: jax.Array
    #: The **guard/ability** stamina resource, `mechanics.resources[RESOURCE_STAMINA]`.
    entity_stamina: jax.Array
    entity_stamina_broken: jax.Array
    #: The **locomotion** stamina -- the sprint budget -- which is a different
    #: quantity carried on `EntityLocomotionState`. Two bars share the word
    #: "stamina" and nothing but the field name distinguished them, so a metric
    #: reading `entity_stamina` to measure sprinting silently measured guard
    #: spend instead: sprinting never touches that resource.
    entity_locomotion_stamina: jax.Array


class MultiActorRolloutResult(NamedTuple):
    """Final recurrent/runtime state plus the collected actor-major rollout."""

    state: MultiActorArenaState
    observation: PolicyActorObservation
    recurrent_state: jax.Array
    episode_start: jax.Array
    running_episode_return: jax.Array
    running_episode_length: jax.Array
    rollout: MultiActorRollout


def make_multi_actor_rollout_collector(
    params: CombatParams,
    config: ArsenalRuntimeConfig,
    assignment: PolicyActorAssignment,
    *,
    reset_provider: ArenaResetProvider,
    capability_provider: ArenaCapabilityProvider,
    rollout_steps: int,
    opponent_ability_provider: ArsenalOpponentAbilityProvider | None = None,
    action_mask_transform: ActionMaskTransform | None = None,
    observation_transform: ActorObservationTransform | None = None,
    transition_transform: ActorTransitionTransform | None = None,
    carry_recurrent_state: bool = True,
    record_policy_inputs: bool = True,
    observation_telemetry_transform: ObservationTelemetryTransform | None = None,
    record_actions: bool = True,
    maximum_turn_degrees: float = 45.0,
    scripted_actor_slots: tuple[int, ...] = (),
    record_policy_slots: tuple[int, ...] | None = None,
):
    """Build one JIT-safe shared-arena collector with atomic arena resets.

    A terminal event from any active actor ends the shared match for every
    active actor in that arena. Reset and recurrent-state clearing therefore
    happen on the arena axis ``B`` rather than independently on ``B*P``.
    """

    if opponent_ability_provider is None:
        from hytalegym.jax.combat.opponents.runtime.policy import (
            first_legal_opponent_ability_slots,
        )

        opponent_ability_provider = first_legal_opponent_ability_slots
    batch, policy_slots = assignment.actor_index.shape
    entity_count = config.loadout.weapon_id.shape[1]

    def project(arena):
        capabilities = capability_provider(arena.arsenal)
        return (
            observe_policy_actors_flat(
                arena.arsenal,
                params,
                capabilities,
                config,
                arena.locomotion,
                assignment,
            ),
            capabilities,
        )

    def decode(arena, observation, capabilities, factors):
        return decode_policy_actor_factors_flat(
            arena.arsenal,
            observation.structured,
            factors,
            capabilities,
            config,
            arena.locomotion,
            assignment,
            maximum_turn_degrees=maximum_turn_degrees,
        )

    def advance(arena, controls, _observation, capabilities, environment_keys):
        commands = empty_arsenal_commands(
            batch,
            entity_count=entity_count,
        )._replace(
            ability_slot=scripted_opponent_ability_slots(
                arena.arsenal,
                capabilities,
                config,
                assignment,
                opponent_ability_provider,
            ),
            world=capabilities,
        )
        return step_multi_actor_arena(
            arena,
            controls,
            assignment,
            environment_keys,
            params,
            commands,
            config,
        )

    def reset(reset_keys):
        return initialize_multi_actor_arena_state(
            reset_provider(reset_keys),
            params,
        )

    return _make_actor_major_rollout_collector(
        assignment,
        rollout_steps=rollout_steps,
        project=project,
        decode=decode,
        advance=advance,
        reset=reset,
        action_mask_transform=action_mask_transform,
        observation_transform=observation_transform,
        transition_transform=transition_transform,
        carry_recurrent_state=carry_recurrent_state,
        record_policy_inputs=record_policy_inputs,
        observation_telemetry_transform=observation_telemetry_transform,
        record_actions=record_actions,
        scripted_actor_slots=scripted_actor_slots,
        record_policy_slots=record_policy_slots,
    )


def scripted_opponent_ability_slots(
    state: ArsenalEnvironmentState,
    capabilities: ArsenalWorldCapabilities,
    config: ArsenalRuntimeConfig,
    assignment: PolicyActorAssignment,
    provider: ArsenalOpponentAbilityProvider,
) -> jax.Array:
    """Select only unowned scripted rows through one validated provider."""

    if provider is None or not callable(provider):
        raise TypeError("opponent ability provider must be callable")
    selected = jnp.asarray(provider(state, capabilities, config))
    expected_shape = state.combat.health.shape
    if selected.shape != expected_shape or selected.dtype != jnp.dtype(jnp.int32):
        raise ValueError(
            f"opponent ability provider must return int32{list(expected_shape)}"
        )
    owned = policy_controlled_entity_mask(
        assignment,
        entity_count=expected_shape[1],
    )
    scripted = config.opponent_controller_mask & ~owned
    return jnp.where(scripted, selected, jnp.int32(-1))


def _make_actor_major_rollout_collector(
    assignment: PolicyActorAssignment,
    *,
    rollout_steps: int,
    project: Callable,
    decode: Callable,
    advance: Callable,
    reset: Callable,
    action_mask_transform: ActionMaskTransform | None = None,
    observation_transform: ActorObservationTransform | None = None,
    transition_transform: ActorTransitionTransform | None = None,
    carry_recurrent_state: bool = True,
    record_policy_inputs: bool = True,
    observation_telemetry_transform: ObservationTelemetryTransform | None = None,
    record_actions: bool = True,
    scripted_actor_slots: tuple[int, ...] = (),
    record_policy_slots: tuple[int, ...] | None = None,
):
    """Collect actor-major behavior through caller-owned arena adapters."""

    if isinstance(rollout_steps, bool) or not isinstance(rollout_steps, int):
        raise TypeError("rollout_steps must be an integer")
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be positive")
    if action_mask_transform is not None and not callable(action_mask_transform):
        raise TypeError("action_mask_transform must be callable or None")
    if observation_transform is not None and not callable(observation_transform):
        raise TypeError("observation_transform must be callable or None")
    if transition_transform is not None and not callable(transition_transform):
        raise TypeError("transition_transform must be callable or None")
    if not isinstance(carry_recurrent_state, bool):
        raise TypeError("carry_recurrent_state must be boolean")
    if not isinstance(record_policy_inputs, bool):
        raise TypeError("record_policy_inputs must be boolean")
    if observation_telemetry_transform is not None and not callable(
        observation_telemetry_transform
    ):
        raise TypeError("observation_telemetry_transform must be callable or None")
    if not isinstance(record_actions, bool):
        raise TypeError("record_actions must be boolean")
    batch, policy_slots = assignment.actor_index.shape
    scripted_slots = _policy_slot_tuple(
        scripted_actor_slots,
        policy_slots,
        name="scripted_actor_slots",
        allow_empty=True,
    )
    recorded_slots = (
        tuple(range(policy_slots))
        if record_policy_slots is None
        else _policy_slot_tuple(
            record_policy_slots,
            policy_slots,
            name="record_policy_slots",
            allow_empty=False,
        )
    )
    recorded_slot_index = jnp.asarray(recorded_slots, dtype=jnp.int32)

    def collect(
        state: MultiActorArenaState,
        policy_bank: RecurrentPolicyParams,
        recurrent_state: jax.Array,
        episode_start: jax.Array,
        running_episode_return: jax.Array,
        running_episode_length: jax.Array,
        key: jax.Array,
    ) -> MultiActorRolloutResult:
        if recurrent_state.ndim != 3 or recurrent_state.shape[:2] != (
            batch,
            policy_slots,
        ):
            raise ValueError("recurrent_state must have shape [B,P,R]")
        starts = jnp.asarray(episode_start, dtype=jnp.bool_)
        if starts.shape != (batch, policy_slots):
            raise ValueError("episode_start must have shape [B,P]")
        running_return = jnp.asarray(running_episode_return, dtype=jnp.float32)
        running_length = jnp.asarray(running_episode_length, dtype=jnp.int32)
        if running_return.shape != (batch, policy_slots):
            raise ValueError("running_episode_return must have shape [B,P]")
        if running_length.shape != (batch, policy_slots):
            raise ValueError("running_episode_length must have shape [B,P]")
        scan_keys = jax.random.split(key, rollout_steps)

        def rollout_step(carry, step_key):
            (
                arena,
                actor_carry,
                actor_starts,
                actor_return,
                actor_length,
            ) = carry
            action_key, environment_key, reset_key = jax.random.split(step_key, 3)
            observation, context = project(arena)
            if observation_transform is not None:
                observation = _apply_observation_transform(
                    observation_transform, arena, observation
                )
            action_mask = observation.action_mask
            if action_mask_transform is not None:
                action_mask = jnp.asarray(
                    action_mask_transform(arena, observation, action_mask)
                )
                if action_mask.dtype != jnp.dtype(jnp.bool_):
                    raise TypeError(
                        "action_mask_transform must return a boolean action mask"
                    )
                if action_mask.shape != observation.action_mask.shape:
                    raise ValueError(
                        "action_mask_transform must preserve the action-mask shape"
                    )
            recurrent_input = jnp.where(
                actor_starts[..., None],
                jnp.float32(0.0),
                actor_carry,
            )
            inference = apply_assigned_actor_controllers(
                policy_bank,
                observation.dense,
                recurrent_input,
                action_mask,
                assignment,
                scripted_actor_slots=scripted_slots,
            )
            factors, log_probability = sample_arsenal_action_factors(
                action_key,
                inference.logits,
            )
            controls = decode(arena, observation, context, factors)
            environment_keys = jax.random.split(environment_key, batch)
            transition = advance(
                arena,
                controls,
                observation,
                context,
                environment_keys,
            )
            if transition_transform is not None:
                transition = _apply_transition_transform(
                    transition_transform, arena, transition
                )
            actor_done = transition.terminated | transition.truncated
            arena_done = jnp.any(actor_done & assignment.active, axis=1)

            reset_keys = jax.random.split(reset_key, batch)
            reset_state = reset(reset_keys)
            next_arena = _select_arena_rows(
                arena_done,
                reset_state,
                transition.state,
            )
            policy_carry = (
                inference.recurrent_state
                if carry_recurrent_state
                else jnp.zeros_like(inference.recurrent_state)
            )
            next_carry = jnp.where(
                arena_done[:, None, None],
                jnp.float32(0.0),
                policy_carry,
            )
            next_starts = arena_done[:, None] & assignment.active
            transition_valid = getattr(
                transition,
                "action_valid",
                controls.valid,
            )
            valid = inference.valid & controls.valid & transition_valid
            reward = jnp.where(valid, transition.reward, jnp.float32(0.0))
            accumulated_return = actor_return + reward
            accumulated_length = actor_length + valid.astype(jnp.int32)
            done = arena_done[:, None] & assignment.active
            completed_return = jnp.where(done, accumulated_return, jnp.float32(0.0))
            completed_length = jnp.where(done, accumulated_length, jnp.int32(0))
            completion = transition.reward_components.completion
            death = transition.reward_components.death
            success = done & completion & ~death
            died = done & death & ~completion
            simultaneous = done & completion & death
            other = done & ~completion & ~death
            next_return = jnp.where(done, jnp.float32(0.0), accumulated_return)
            next_length = jnp.where(done, jnp.int32(0), accumulated_length)
            telemetry = (
                observation.dense[..., :0]
                if observation_telemetry_transform is None
                else jnp.asarray(observation_telemetry_transform(observation))
            )
            if telemetry.ndim < 2 or telemetry.shape[:2] != (
                batch,
                policy_slots,
            ):
                raise ValueError(
                    "observation telemetry must begin with actor shape [B,P]"
                )

            def recorded(value):
                return value[:, recorded_slot_index]

            row = MultiActorRollout(
                observation=(
                    recorded(observation.dense)
                    if record_policy_inputs
                    else recorded(observation.dense)[..., :0]
                ),
                observation_telemetry=recorded(telemetry),
                action_mask=(
                    recorded(action_mask)
                    if record_policy_inputs
                    else recorded(action_mask)[..., :0]
                ),
                action=(
                    recorded(factors) if record_actions else recorded(factors)[..., :0]
                ),
                log_probability=recorded(
                    jnp.where(valid, log_probability, jnp.float32(0.0))
                ),
                value=recorded(jnp.where(valid, inference.value, jnp.float32(0.0))),
                reward=recorded(reward),
                done=recorded(done),
                episode_start=recorded(actor_starts),
                valid=recorded(valid),
                policy_id=recorded(assignment.policy_id),
                trainable=recorded(assignment.trainable & valid),
                completed_episode_return=recorded(completed_return),
                completed_episode_length=recorded(completed_length),
                completed_episode_success=recorded(success),
                completed_episode_death=recorded(died),
                completed_episode_simultaneous=recorded(simultaneous),
                completed_episode_other=recorded(other),
                entity_position=transition.state.arsenal.combat.position,
                entity_yaw=transition.state.arsenal.combat.yaw,
                entity_health=transition.state.arsenal.combat.health,
                entity_grounded=transition.state.locomotion.grounded,
                entity_stamina=transition.state.arsenal.mechanics.resources[
                    ..., RESOURCE_STAMINA
                ],
                entity_stamina_broken=(
                    transition.state.arsenal.mechanics.stamina_broken
                ),
                entity_locomotion_stamina=transition.state.locomotion.stamina,
            )
            return (
                next_arena,
                next_carry,
                next_starts,
                next_return,
                next_length,
            ), row

        (
            (
                final_state,
                final_carry,
                final_starts,
                final_running_return,
                final_running_length,
            ),
            rollout,
        ) = jax.lax.scan(
            rollout_step,
            (
                state,
                recurrent_state,
                starts,
                running_return,
                running_length,
            ),
            scan_keys,
        )
        final_observation, _ = project(final_state)
        if observation_transform is not None:
            final_observation = _apply_observation_transform(
                observation_transform, final_state, final_observation
            )
        return MultiActorRolloutResult(
            state=final_state,
            observation=final_observation,
            recurrent_state=final_carry,
            episode_start=final_starts,
            running_episode_return=final_running_return,
            running_episode_length=final_running_length,
            rollout=rollout,
        )

    collect.record_policy_slots = recorded_slots
    collect.scripted_actor_slots = scripted_slots
    return collect


def _policy_slot_tuple(
    value: tuple[int, ...],
    policy_slots: int,
    *,
    name: str,
    allow_empty: bool,
) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple of policy-slot indexes")
    if (not allow_empty and not value) or value != tuple(sorted(set(value))):
        raise ValueError(f"{name} must be sorted and unique")
    if any(
        isinstance(slot, bool)
        or not isinstance(slot, int)
        or slot < 0
        or slot >= policy_slots
        for slot in value
    ):
        raise ValueError(f"{name} contains an invalid policy-slot index")
    return value


def _select_arena_rows(mask, when_true, when_false):
    selector = jnp.asarray(mask, dtype=jnp.bool_)

    def select(true_value, false_value):
        if true_value.shape != false_value.shape:
            raise ValueError("arena reset changed a state-leaf shape")
        if true_value.dtype != false_value.dtype:
            raise ValueError("arena reset changed a state-leaf dtype")
        if true_value.ndim == 0 or true_value.shape[0] != selector.shape[0]:
            return false_value
        expanded = selector.reshape(selector.shape + (1,) * (true_value.ndim - 1))
        return jnp.where(expanded, true_value, false_value)

    return jax.tree_util.tree_map(select, when_true, when_false)


__all__ = [
    "ActorObservationTransform",
    "ActorTransitionTransform",
    "ActionMaskTransform",
    "ArenaCapabilityProvider",
    "ArenaResetProvider",
    "MultiActorRollout",
    "MultiActorRolloutResult",
    "ObservationTelemetryTransform",
    "make_multi_actor_rollout_collector",
    "scripted_opponent_ability_slots",
]
