"""Compiled held-out evaluation for recurrent Arsenal policies and controls."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.combat import (
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
    AGENT_ENTITY,
    TARGET_ENTITY,
    ArsenalRuntimeConfig,
    CombatParams,
    RewardComponents,
    arsenal_policy_action_mask,
    arsenal_policy_observation,
    decode_arsenal_policy_actions,
    decode_learner_arsenal_action,
    empty_injected_world_features,
    empty_world_geometry_policy_tokens,
    encode_learner_observation_v3,
    step_arsenal_batch,
)
from hytalegym.jax.combat.observation.v3.policy.distribution import (
    greedy_arsenal_action_factors,
)
from hytalegym.jax.combat.types import sum_reward_components
from hytalegym.jax.combat.env import TargetNavigationProvider
from hytalegym.jax.training.arsenal import (
    ArsenalActionSurfaceExecutor,
    ArsenalActionSurfaceProvider,
    ArsenalActionSurfaceRuntimeInitializer,
    ArsenalPPOEnvironmentState,
    ArsenalResetProvider,
    ArsenalRuntimeExplosionCandidateProvider,
    ArsenalWorldCapabilityProvider,
    ArsenalWorldFeatureProvider,
    ArsenalWorldRuntimeProvider,
    ArsenalWorldTokenProvider,
    fail_closed_arsenal_world_capabilities,
    make_arsenal_ppo_environment,
)
from hytalegym.jax.training.evaluation import (
    ActionCarryInitializer,
    ActionSource,
)
from hytalegym.jax.training.policy import apply_policy
from hytalegym.jax.training.types import RecurrentPolicyParams
from hytalegym.jax.world import GeometryProvider


class ArsenalEvaluationResult(NamedTuple):
    """Shared evaluation fields plus the immutable semantic weapon identity."""

    success: jax.Array
    terminated: jax.Array
    episode_return: jax.Array
    episode_length: jax.Array
    reward_components: RewardComponents
    profile_weapon_id: jax.Array
    world_capabilities_ready: jax.Array
    world_geometry_ready: jax.Array


ArsenalEvaluator = Callable[
    [RecurrentPolicyParams | None, jax.Array, jax.Array],
    ArsenalEvaluationResult,
]


class ArsenalPolicyActionInput(NamedTuple):
    """Actor-legal policy inputs produced by one Arsenal environment state."""

    learner_observation: Any
    policy_observation: jax.Array
    action_mask: jax.Array


ArsenalPolicyActionSource = Callable[
    [ArsenalPolicyActionInput, Any, jax.Array],
    tuple[jax.Array, Any],
]


class ArsenalPolicyAwareActionSource:
    """Preserve the legacy callable while accepting exact policy inputs."""

    __slots__ = ("_legacy_source", "_policy_source")

    def __init__(
        self,
        legacy_source: ActionSource,
        policy_source: ArsenalPolicyActionSource,
    ) -> None:
        self._legacy_source = legacy_source
        self._policy_source = policy_source

    def __call__(self, observation, carry, key):
        """Retain compatibility with callers that expose no action surface."""

        return self._legacy_source(observation, carry, key)

    def from_policy_input(
        self,
        policy_input: ArsenalPolicyActionInput,
        carry: Any,
        key: jax.Array,
    ):
        """Choose from the exact observation and mask used by the policy."""

        return self._policy_source(policy_input, carry, key)


def policy_aware_arsenal_action_source(
    legacy_source: ActionSource,
    policy_source: ArsenalPolicyActionSource,
) -> ArsenalPolicyAwareActionSource:
    """Bind a rich Arsenal source without breaking the public legacy call."""

    return ArsenalPolicyAwareActionSource(legacy_source, policy_source)


def recurrent_arsenal_policy_action_source(
    policy_params: RecurrentPolicyParams,
    *,
    compile: bool = True,
) -> tuple[ArsenalPolicyAwareActionSource, ActionCarryInitializer]:
    """Expose one recurrent Arsenal policy through the public ActionSource."""

    recurrent_size = int(policy_params.gru.recurrent_kernel.shape[0])

    def initialize_carry(batch_size: int) -> jax.Array:
        return jnp.zeros(
            (batch_size, recurrent_size),
            dtype=jnp.float32,
        )

    def policy_source(policy_input, carry, key):
        del key
        next_carry, logits, _ = apply_policy(
            policy_params,
            policy_input.policy_observation,
            carry,
            policy_input.action_mask,
        )
        return (
            _greedy_factored_actions(
                logits,
                ARSENAL_POLICY_ACTION_HEAD_SIZES,
            ),
            next_carry,
        )

    def legacy_source(observation, carry, key):
        return policy_source(
            ArsenalPolicyActionInput(
                learner_observation=observation,
                policy_observation=arsenal_policy_observation(observation),
                action_mask=arsenal_policy_action_mask(observation),
            ),
            carry,
            key,
        )

    if compile:
        legacy_source = jax.jit(legacy_source)
        policy_source = jax.jit(policy_source)
    return (
        policy_aware_arsenal_action_source(
            legacy_source,
            policy_source,
        ),
        initialize_carry,
    )


def make_arsenal_evaluator(
    environment_params: CombatParams,
    runtime_config: ArsenalRuntimeConfig,
    *,
    geometry_provider: GeometryProvider | None = None,
    target_navigation_provider: TargetNavigationProvider | None = None,
    world_capability_provider: ArsenalWorldCapabilityProvider | None = None,
    world_feature_provider: ArsenalWorldFeatureProvider | None = None,
    world_token_provider: ArsenalWorldTokenProvider | None = None,
    world_runtime_provider: ArsenalWorldRuntimeProvider | None = None,
    explosion_candidate_provider: (
        ArsenalRuntimeExplosionCandidateProvider | None
    ) = None,
    reset_provider: ArsenalResetProvider | None = None,
    action_surface_provider: ArsenalActionSurfaceProvider | None = None,
    action_surface_executor: ArsenalActionSurfaceExecutor | None = None,
    action_surface_runtime_initializer: (
        ArsenalActionSurfaceRuntimeInitializer | None
    ) = None,
    max_policy_steps: int = 256,
    compile: bool = True,
    action_source: ActionSource | None = None,
    action_carry_initializer: ActionCarryInitializer | None = None,
) -> ArsenalEvaluator:
    """Build a deterministic full-batch Arsenal-v3 evaluation scan.

    ``world_capability_provider=None`` deliberately selects the training
    adapter's fail-closed capability producer. Physical motion remains flat
    unless callers separately supply a fixed ``geometry_provider``; this
    evaluator never substitutes the permissive open-flat control. World tokens
    are independently unavailable unless ``world_token_provider`` or the
    unified ``world_runtime_provider`` is supplied.
    """

    if max_policy_steps < 1:
        raise ValueError("max_policy_steps must be positive")
    if action_source is None and action_carry_initializer is not None:
        raise ValueError(
            "action_carry_initializer requires an explicit action_source"
        )
    capability_provider = (
        fail_closed_arsenal_world_capabilities
        if world_capability_provider is None
        else world_capability_provider
    )
    environment = make_arsenal_ppo_environment(
        environment_params,
        runtime_config,
        geometry_provider=geometry_provider,
        target_navigation_provider=target_navigation_provider,
        world_capability_provider=(
            None if world_runtime_provider is not None else capability_provider
        ),
        world_feature_provider=world_feature_provider,
        world_token_provider=world_token_provider,
        world_runtime_provider=world_runtime_provider,
        explosion_candidate_provider=explosion_candidate_provider,
        reset_provider=reset_provider,
        action_surface_provider=action_surface_provider,
        action_surface_executor=action_surface_executor,
        action_surface_runtime_initializer=(
            action_surface_runtime_initializer
        ),
    )

    def projected_capabilities(state: ArsenalPPOEnvironmentState):
        if state.world_capabilities is not None:
            return state.world_capabilities
        if world_runtime_provider is None:
            return capability_provider(state.runtime, environment_params)
        return world_runtime_provider(
            state.runtime,
            state.action_surface_runtime.world,
            environment_params,
        ).capabilities

    def evaluate(
        policy_params: RecurrentPolicyParams | None,
        reset_keys: jax.Array,
        rollout_key: jax.Array,
    ) -> ArsenalEvaluationResult:
        batch_size = reset_keys.shape[0]
        state, policy_observation, action_mask = environment.reset(reset_keys)
        initial_world = projected_capabilities(state)
        world_capabilities_ready = _world_capabilities_ready(initial_world)
        world_geometry_ready = _world_geometry_ready(
            state.learner_observation.world_geometry
        )
        if action_source is None:
            if policy_params is None:
                raise ValueError(
                    "policy_params are required when action_source is omitted"
                )
            recurrent_size = policy_params.gru.recurrent_kernel.shape[0]
            initial_action_carry = jnp.zeros(
                (batch_size, recurrent_size),
                dtype=jnp.float32,
            )
        else:
            initializer = (
                _empty_action_carry
                if action_carry_initializer is None
                else action_carry_initializer
            )
            initial_action_carry = initializer(batch_size)
        initial_carry = (
            state,
            policy_observation,
            action_mask,
            initial_action_carry,
            jnp.zeros((batch_size,), dtype=jnp.bool_),
            jnp.zeros((batch_size,), dtype=jnp.bool_),
            jnp.zeros((batch_size,), dtype=jnp.float32),
            jnp.zeros((batch_size,), dtype=jnp.int32),
            world_capabilities_ready,
            world_geometry_ready,
        )
        step_keys = jax.random.split(rollout_key, max_policy_steps)

        def evaluate_step(carry, step_key):
            (
                current_state,
                current_policy_observation,
                current_action_mask,
                current_action_carry,
                finished,
                success,
                episode_return,
                episode_length,
                world_capabilities_ready,
                world_geometry_ready,
            ) = carry
            if action_source is None:
                action_carry_candidate, logits, _ = apply_policy(
                    policy_params,
                    current_policy_observation,
                    current_action_carry,
                    current_action_mask,
                )
                action_ids = _greedy_factored_actions(
                    logits,
                    ARSENAL_POLICY_ACTION_HEAD_SIZES,
                )
            else:
                action_ids, action_carry_candidate = _invoke_action_source(
                    action_source,
                    ArsenalPolicyActionInput(
                        learner_observation=(
                            current_state.learner_observation
                        ),
                        policy_observation=current_policy_observation,
                        action_mask=current_action_mask,
                    ),
                    current_action_carry,
                    jax.random.fold_in(step_key, 1),
                )
            environment_keys = jax.random.split(step_key, batch_size)
            (
                candidate_state,
                candidate_policy_observation,
                reward,
                done,
                candidate_action_mask,
                environment_info,
            ) = environment.step_detailed(
                current_state,
                current_policy_observation,
                action_ids,
                environment_keys,
            )
            reward_components = environment_info.combat_info.reward_components
            candidate_world = projected_capabilities(candidate_state)
            candidate_world_geometry = (
                candidate_state.learner_observation.world_geometry
            )
            active = ~finished
            newly_done = active & done
            current_state = _batch_select(
                active,
                candidate_state,
                current_state,
            )
            current_policy_observation = jnp.where(
                active[:, None],
                candidate_policy_observation,
                current_policy_observation,
            )
            current_action_mask = jnp.where(
                active[:, None],
                candidate_action_mask,
                current_action_mask,
            )
            current_action_carry = _batch_select(
                active,
                action_carry_candidate,
                current_action_carry,
            )
            episode_return = episode_return + jnp.where(
                active,
                reward,
                jnp.float32(0.0),
            )
            episode_length = episode_length + active.astype(jnp.int32)
            world_capabilities_ready &= ~active | _world_capabilities_ready(
                candidate_world
            )
            world_geometry_ready &= ~active | _world_geometry_ready(
                candidate_world_geometry
            )
            success = success | (
                newly_done
                & (
                    candidate_state.runtime.combat.health[:, TARGET_ENTITY]
                    <= 0.0
                )
            )
            finished = finished | newly_done
            active_components = jax.tree.map(
                lambda value: jnp.where(
                    active,
                    value,
                    jnp.zeros_like(value),
                ),
                reward_components,
            )
            return (
                current_state,
                current_policy_observation,
                current_action_mask,
                current_action_carry,
                finished,
                success,
                episode_return,
                episode_length,
                world_capabilities_ready,
                world_geometry_ready,
            ), active_components

        final_carry, decision_components = jax.lax.scan(
            evaluate_step,
            initial_carry,
            step_keys,
        )
        episode_components = sum_reward_components(decision_components)
        (
            _,
            _,
            _,
            _,
            finished,
            success,
            episode_return,
            episode_length,
            world_capabilities_ready,
            world_geometry_ready,
        ) = final_carry
        return ArsenalEvaluationResult(
            success=success,
            terminated=finished,
            episode_return=episode_return,
            episode_length=episode_length,
            reward_components=episode_components,
            profile_weapon_id=runtime_config.loadout.weapon_id[
                :, AGENT_ENTITY
            ],
            world_capabilities_ready=world_capabilities_ready,
            world_geometry_ready=world_geometry_ready,
        )

    if compile:
        return jax.jit(evaluate)
    return evaluate


def step_arsenal_evaluation(
    state: ArsenalPPOEnvironmentState,
    action_ids: jax.Array,
    keys: jax.Array,
    params: CombatParams,
    config: ArsenalRuntimeConfig,
    capability_provider: ArsenalWorldCapabilityProvider,
    token_provider: ArsenalWorldTokenProvider | None = None,
    geometry_provider: GeometryProvider | None = None,
    target_navigation_provider: TargetNavigationProvider | None = None,
):
    """Mirror the legacy stateless PPO step and retain reward facts.

    This compatibility helper has no action-surface runtime or executor.
    Production evaluation uses ``PPOEnvironment.step_detailed`` through
    :func:`make_arsenal_evaluator` or
    :func:`evaluate_arsenal_action_source`.
    """

    return _step_arsenal_evaluation_with_world(
        state,
        action_ids,
        keys,
        params,
        config,
        capability_provider,
        token_provider,
        geometry_provider,
        target_navigation_provider,
    )[:6]


def _step_arsenal_evaluation_with_world(
    state: ArsenalPPOEnvironmentState,
    action_ids: jax.Array,
    keys: jax.Array,
    params: CombatParams,
    config: ArsenalRuntimeConfig,
    capability_provider: ArsenalWorldCapabilityProvider,
    token_provider: ArsenalWorldTokenProvider | None = None,
    geometry_provider: GeometryProvider | None = None,
    target_navigation_provider: TargetNavigationProvider | None = None,
):
    current_world = capability_provider(state.runtime, params)
    decoded = decode_learner_arsenal_action(
        state.learner_observation,
        decode_arsenal_policy_actions(action_ids),
        current_world,
        desired_body_yaw_degrees=state.runtime.combat.desired_body_yaw,
    )
    transition = step_arsenal_batch(
        state.runtime,
        decoded.low_level_action,
        keys,
        params,
        decoded.commands,
        config,
        geometry_provider,
        target_navigation_provider,
    )
    next_world = capability_provider(transition.state, params)
    next_world_geometry = (
        empty_world_geometry_policy_tokens(action_ids.shape[0])
        if token_provider is None
        else token_provider(transition.state, params)
    )
    next_observation = encode_learner_observation_v3(
        transition.state,
        params,
        empty_injected_world_features(action_ids.shape[0]),
        next_world,
        config,
        world_geometry=next_world_geometry,
    )
    return (
        ArsenalPPOEnvironmentState(
            transition.state,
            next_observation,
            state.action_surface,
            state.action_surface_runtime,
        ),
        arsenal_policy_observation(
            next_observation,
            state.action_surface.block_candidates,
            state.action_surface.recipe_encoding,
        ),
        transition.reward,
        transition.terminated | transition.truncated,
        arsenal_policy_action_mask(
            next_observation,
            state.action_surface.block_candidates,
            state.action_surface.recipe_candidates,
            use_available=state.action_surface.use_available,
        ),
        transition.combat_info.reward_components,
        next_world,
        next_world_geometry,
    )


def evaluate_arsenal_action_source(
    environment_params: CombatParams,
    runtime_config: ArsenalRuntimeConfig,
    action_source: ActionSource,
    reset_keys: jax.Array,
    rollout_key: jax.Array,
    *,
    geometry_provider: GeometryProvider | None = None,
    target_navigation_provider: TargetNavigationProvider | None = None,
    world_capability_provider: ArsenalWorldCapabilityProvider | None = None,
    world_feature_provider: ArsenalWorldFeatureProvider | None = None,
    world_token_provider: ArsenalWorldTokenProvider | None = None,
    world_runtime_provider: ArsenalWorldRuntimeProvider | None = None,
    explosion_candidate_provider: (
        ArsenalRuntimeExplosionCandidateProvider | None
    ) = None,
    action_surface_provider: ArsenalActionSurfaceProvider | None = None,
    action_surface_executor: ArsenalActionSurfaceExecutor | None = None,
    action_surface_runtime_initializer: (
        ArsenalActionSurfaceRuntimeInitializer | None
    ) = None,
    max_policy_steps: int = 256,
    action_carry_initializer: ActionCarryInitializer | None = None,
    compile_step: bool = True,
) -> ArsenalEvaluationResult:
    """Evaluate a host-driven action source against the structured bundle.

    Unlike :func:`make_arsenal_evaluator`, the action source runs at the Python
    boundary once per decision. It may therefore wrap PyTorch or another
    framework, while the expensive environment step remains JIT compiled.
    Returned packed actions are validated before entering JAX. Physical
    geometry, capabilities, and World tokens remain explicit; a bound
    ``world_runtime_provider`` projects all three from one persistent PyTree.
    """

    if max_policy_steps < 1:
        raise ValueError("max_policy_steps must be positive")
    batch_size = int(reset_keys.shape[0])
    if batch_size != runtime_config.loadout.weapon_id.shape[0]:
        raise ValueError("reset key batch must match the Arsenal loadout")
    provider = (
        fail_closed_arsenal_world_capabilities
        if world_capability_provider is None
        else world_capability_provider
    )
    environment = make_arsenal_ppo_environment(
        environment_params,
        runtime_config,
        geometry_provider=geometry_provider,
        target_navigation_provider=target_navigation_provider,
        world_capability_provider=(
            None if world_runtime_provider is not None else provider
        ),
        world_feature_provider=world_feature_provider,
        world_token_provider=world_token_provider,
        world_runtime_provider=world_runtime_provider,
        explosion_candidate_provider=explosion_candidate_provider,
        action_surface_provider=action_surface_provider,
        action_surface_executor=action_surface_executor,
        action_surface_runtime_initializer=(
            action_surface_runtime_initializer
        ),
    )
    reset = jax.jit(environment.reset) if compile_step else environment.reset
    state, policy_observation, action_mask = reset(reset_keys)

    def projected_capabilities(current_state: ArsenalPPOEnvironmentState):
        if current_state.world_capabilities is not None:
            return current_state.world_capabilities
        if world_runtime_provider is None:
            return provider(current_state.runtime, environment_params)
        return world_runtime_provider(
            current_state.runtime,
            current_state.action_surface_runtime.world,
            environment_params,
        ).capabilities

    initial_world = projected_capabilities(state)
    world_capabilities_ready = _world_capabilities_ready(initial_world)
    world_geometry_ready = _world_geometry_ready(
        state.learner_observation.world_geometry
    )
    initializer = (
        _empty_action_carry
        if action_carry_initializer is None
        else action_carry_initializer
    )
    action_carry = initializer(batch_size)
    finished = jnp.zeros((batch_size,), dtype=jnp.bool_)
    success = jnp.zeros((batch_size,), dtype=jnp.bool_)
    episode_return = jnp.zeros((batch_size,), dtype=jnp.float32)
    episode_length = jnp.zeros((batch_size,), dtype=jnp.int32)
    episode_components = RewardComponents(
        target_damage=jnp.zeros((batch_size,), dtype=jnp.float32),
        agent_damage=jnp.zeros((batch_size,), dtype=jnp.float32),
        completion=jnp.zeros((batch_size,), dtype=jnp.bool_),
        death=jnp.zeros((batch_size,), dtype=jnp.bool_),
    )

    def environment_step(
        current_state,
        current_policy_observation,
        action_ids,
        keys,
    ):
        return environment.step_detailed(
            current_state,
            current_policy_observation,
            action_ids,
            keys,
        )

    step = jax.jit(environment_step) if compile_step else environment_step
    step_keys = jax.random.split(rollout_key, max_policy_steps)
    for step_index in range(max_policy_steps):
        step_key = step_keys[step_index]
        action_ids, action_carry_candidate = _invoke_action_source(
            action_source,
            ArsenalPolicyActionInput(
                learner_observation=state.learner_observation,
                policy_observation=policy_observation,
                action_mask=action_mask,
            ),
            action_carry,
            jax.random.fold_in(step_key, 1),
        )
        action_ids = _validated_host_actions(
            action_ids,
            batch_size=batch_size,
            head_sizes=ARSENAL_POLICY_ACTION_HEAD_SIZES,
            step_index=step_index,
        )
        environment_keys = jax.random.split(step_key, batch_size)
        (
            candidate_state,
            candidate_policy_observation,
            reward,
            done,
            candidate_action_mask,
            environment_info,
        ) = step(
            state,
            policy_observation,
            action_ids,
            environment_keys,
        )
        components = environment_info.combat_info.reward_components
        candidate_world = projected_capabilities(candidate_state)
        candidate_world_geometry = (
            candidate_state.learner_observation.world_geometry
        )
        active = ~finished
        newly_done = active & done
        state = _batch_select(active, candidate_state, state)
        policy_observation = jnp.where(
            active[:, None],
            candidate_policy_observation,
            policy_observation,
        )
        action_mask = jnp.where(
            active[:, None],
            candidate_action_mask,
            action_mask,
        )
        action_carry = _batch_select(
            active,
            action_carry_candidate,
            action_carry,
        )
        episode_return = episode_return + jnp.where(
            active,
            reward,
            jnp.float32(0.0),
        )
        episode_length = episode_length + active.astype(jnp.int32)
        world_capabilities_ready &= ~active | _world_capabilities_ready(
            candidate_world
        )
        world_geometry_ready &= ~active | _world_geometry_ready(
            candidate_world_geometry
        )
        success = success | (
            newly_done
            & (candidate_state.runtime.combat.health[:, TARGET_ENTITY] <= 0.0)
        )
        active_components = jax.tree.map(
            lambda value: jnp.where(active, value, jnp.zeros_like(value)),
            components,
        )
        episode_components = RewardComponents(
            target_damage=(
                episode_components.target_damage
                + active_components.target_damage
            ),
            agent_damage=(
                episode_components.agent_damage
                + active_components.agent_damage
            ),
            completion=(
                episode_components.completion
                | active_components.completion
            ),
            death=episode_components.death | active_components.death,
        )
        finished = finished | newly_done
        if bool(np.all(np.asarray(jax.device_get(finished)))):
            break

    return ArsenalEvaluationResult(
        success=success,
        terminated=finished,
        episode_return=episode_return,
        episode_length=episode_length,
        reward_components=episode_components,
        profile_weapon_id=runtime_config.loadout.weapon_id[:, AGENT_ENTITY],
        world_capabilities_ready=world_capabilities_ready,
        world_geometry_ready=world_geometry_ready,
    )


def _world_capabilities_ready(world) -> jax.Array:
    """Return rows whose geometry-backed availability channels stayed valid."""

    return (
        world.actor_world_state_available
        & world.actor_controller_medium_available
        & world.actor_submersion_available
        & world.actor_drop_available
        & jnp.all(world.line_of_sight_valid, axis=1)
        & jnp.all(world.selector_line_of_sight_valid, axis=1)
        & jnp.all(world.muzzle_valid, axis=1)
    )


def _world_geometry_ready(tokens) -> jax.Array:
    """Require an available, non-empty policy token row."""

    return tokens.available & jnp.any(tokens.token_mask, axis=1)


def _validated_host_actions(
    action_ids: Any,
    *,
    batch_size: int,
    head_sizes: tuple[int, ...],
    step_index: int,
) -> jax.Array:
    values = np.asarray(jax.device_get(action_ids))
    expected_shape = (batch_size, len(head_sizes))
    if values.shape != expected_shape:
        raise ValueError(
            f"action source step {step_index} returned shape {values.shape}; "
            f"expected {expected_shape}"
        )
    if not np.issubdtype(values.dtype, np.integer):
        raise TypeError(
            f"action source step {step_index} must return integer factors"
        )
    lower_valid = values >= 0
    upper_valid = values < np.asarray(head_sizes, dtype=np.int64)[None, :]
    if np.any(~(lower_valid & upper_valid)):
        raise ValueError(
            f"action source step {step_index} returned a factor outside "
            "its declared head"
        )
    return jnp.asarray(values, dtype=jnp.int32)


def _invoke_action_source(
    action_source: ActionSource,
    policy_input: ArsenalPolicyActionInput,
    carry: Any,
    key: jax.Array,
):
    """Use exact policy inputs when the source opts into the rich protocol."""

    if isinstance(action_source, ArsenalPolicyAwareActionSource):
        return action_source.from_policy_input(policy_input, carry, key)
    return action_source(policy_input.learner_observation, carry, key)


def _greedy_factored_actions(
    logits: jax.Array,
    head_sizes: tuple[int, ...],
) -> jax.Array:
    if tuple(head_sizes) != tuple(ARSENAL_POLICY_ACTION_HEAD_SIZES):
        raise ValueError("Arsenal greedy decoding requires the public head layout")
    return greedy_arsenal_action_factors(logits)


def _empty_action_carry(batch_size: int) -> jax.Array:
    return jnp.zeros((batch_size, 0), dtype=jnp.float32)


def _batch_select(mask, when_true: Any, when_false: Any):
    def select_leaf(true_leaf, false_leaf):
        expanded_mask = mask.reshape(
            (mask.shape[0],)
            + (1,) * (true_leaf.ndim - 1)
        )
        return jnp.where(expanded_mask, true_leaf, false_leaf)

    return jax.tree.map(select_leaf, when_true, when_false)


__all__ = [
    "ArsenalEvaluationResult",
    "ArsenalEvaluator",
    "ArsenalPolicyActionInput",
    "ArsenalPolicyActionSource",
    "ArsenalPolicyAwareActionSource",
    "evaluate_arsenal_action_source",
    "make_arsenal_evaluator",
    "policy_aware_arsenal_action_source",
    "step_arsenal_evaluation",
]
