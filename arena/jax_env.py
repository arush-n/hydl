"""Direct composition and rollout of the public JAX Arsenal environment."""

from __future__ import annotations

from dataclasses import dataclass
import dataclasses
import hashlib
import inspect
import json
from functools import partial
from types import MappingProxyType
from typing import Any, Callable, Mapping, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.combat.arsenal.factory import arsenal_runtime_capacity
from hytalegym.jax.combat.arsenal.profiles.hytale_0_5_7 import (
    hytale_0_5_7_loadouts,
)
from hytalegym.jax.combat.arsenal.runtime import arsenal_runtime_config
from hytalegym.jax.combat.types import default_combat_params
from hytalegym.jax.training.arsenal import make_arsenal_ppo_environment
from hytalegym.jax.training.checkpoint import (
    ARSENAL_POLICY_SURFACE,
    current_combat_checkpoint_contract,
)
from hytalegym.jax.combat.observation.v3.tokens.world import (
    DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG,
)

from arena.worlds import ArenaWorldSpec, bind_world
from arena.jax_contract import observation_groups


COMBAT_PARAMETER_NAMES = tuple(default_combat_params(microticks=1)._fields)
RUNTIME_OPTIONS = frozenset(
    {
        "specialize",
        "entity_team_id",
        "distance_component_selector",
        "sensor_range",
        "backpack_capacity",
        "entity_role_ids",
        "opponent_controller_mask",
    }
)


class RewardContract(Protocol):
    @property
    def required_groups(self) -> frozenset[str]: ...

    @property
    def required_completion_evidence(self) -> frozenset[str]: ...

    def apply(
        self,
        native_reward: Any,
        previous: Mapping,
        current: Mapping,
        *,
        terminal: Any | None = None,
        completion_evidence: Mapping[str, Any] | None = None,
    ) -> Any: ...

    def describe(self) -> Mapping[str, Any]: ...


class ActorInput(NamedTuple):
    observation: jax.Array
    action_mask: jax.Array
    legal_observation: Any
    action_surface: Any


class Transition(NamedTuple):
    state: Any
    actor_input: ActorInput
    action_factors: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    done: jax.Array
    info: Any
    next_state: Any
    next_actor_input: ActorInput


class RolloutState(NamedTuple):
    environment: Any
    actor_input: ActorInput
    policy_carry: Any
    finished: jax.Array


@dataclass(frozen=True, slots=True)
class ArenaScene:
    name: str
    contract_sha256: str
    environment: Any
    combat_params: Any
    runtime_config: Any
    runtime_capacity: Any
    environment_kwargs: Mapping[str, Any]
    expected_batch: int
    expected_loadout: str
    world: ArenaWorldSpec | None = None
    world_identity: Mapping[str, Any] | None = None
    reward: RewardContract | None = None
    policy_contract: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class ParameterizedPolicy:
    """A stable policy program whose JAX parameters may change per rollout."""

    apply: Callable
    parameters: Any
    initial_carry: Any = None
    compilation_key: Any = None

    def __post_init__(self) -> None:
        if not callable(self.apply):
            raise TypeError("apply must be callable")
        try:
            hash(self.program_identity)
        except TypeError as error:
            raise TypeError("compilation_key must be hashable") from error

    @property
    def program_identity(self) -> Any:
        """Stable cache identity; custom adapters may provide an explicit key."""

        return id(self.apply) if self.compilation_key is None else self.compilation_key

    def __call__(self, carry: Any, actor_input: Any, key: Any):
        return self.apply(self.parameters, carry, actor_input, key)

    def with_parameters(self, parameters: Any) -> "ParameterizedPolicy":
        """Swap weights while retaining the compiled policy program."""

        return dataclasses.replace(self, parameters=parameters)


def _identity(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _identity(item) for key, item in value.items()}
    if isinstance(value, partial):
        return {
            "callable": _identity(value.func),
            "keywords": _identity(value.keywords),
        }
    if callable(value):
        return f"{getattr(value, '__module__', '?')}.{getattr(value, '__qualname__', type(value).__name__)}"
    fields = getattr(value, "_fields", None)
    if fields is not None:
        return {
            "type": type(value).__name__,
            "fields": {name: _identity(getattr(value, name)) for name in fields},
        }
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _identity(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    array = np.asarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype)}


def _digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(_identity(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest().upper()


def make_scene(
    name: str,
    *,
    loadout: str,
    opponent: str | None,
    opponent_ability_provider: Any,
    batch: int,
    parameters: Mapping[str, Any] | None = None,
    providers: Mapping[str, Any] | None = None,
    reward: RewardContract | None = None,
    world: ArenaWorldSpec | None = None,
    microticks: int = 1,
    target_active: bool = True,
    **runtime_options: Any,
) -> ArenaScene:
    """Build one task directly on the JAX PPO/Arsenal environment."""

    unknown = sorted(set(runtime_options) - RUNTIME_OPTIONS)
    if unknown:
        raise ValueError(f"unknown JAX runtime option(s): {', '.join(unknown)}")
    params = default_combat_params(microticks=microticks, target_active=target_active)
    overrides = dict(parameters or {})
    invalid = sorted(set(overrides) - set(COMBAT_PARAMETER_NAMES))
    if invalid:
        raise ValueError(f"unknown CombatParams override(s): {', '.join(invalid)}")
    params = params._replace(**overrides)

    profiles = [loadout] * batch
    targets = None if opponent is None else [opponent] * batch
    loadouts = hytale_0_5_7_loadouts(profiles, target_profiles=targets)
    config = arsenal_runtime_config(loadouts, **runtime_options)
    capacity = arsenal_runtime_capacity(loadouts)
    policy_contract = current_combat_checkpoint_contract(
        arsenal_runtime_capacity=capacity,
        policy_surface=ARSENAL_POLICY_SURFACE,
        world_geometry_config=DEFAULT_WORLD_GEOMETRY_POLICY_CONFIG,
    )

    environment_kwargs = dict(providers or {})
    world_identity = None
    if world is not None:
        binding = bind_world(world, batch, params, config)
        overlap = set(environment_kwargs) & set(binding.providers)
        if overlap:
            raise ValueError(f"world provider(s) supplied twice: {sorted(overlap)}")
        environment_kwargs.update(binding.providers)
        world_identity = binding.identity

    environment = make_arsenal_ppo_environment(
        params,
        config,
        opponent_ability_provider=opponent_ability_provider,
        **environment_kwargs,
    )
    if reward is not None:
        if environment.step_detailed is None:
            raise ValueError("task rewards require a detailed JAX environment step")
        native_step = environment.step_detailed
        episode_outcome = environment.episode_outcome
        groups = tuple(sorted(reward.required_groups))
        reward_parameters = inspect.signature(reward.apply).parameters
        accepts_terminal = "terminal" in reward_parameters
        accepts_completion_evidence = "completion_evidence" in reward_parameters
        required_completion_evidence = frozenset(
            getattr(reward, "required_completion_evidence", ())
        )
        supported_completion_evidence = (
            frozenset({"episode_success"})
            if episode_outcome is not None
            else frozenset()
        )
        missing_completion_evidence = (
            required_completion_evidence - supported_completion_evidence
        )
        if missing_completion_evidence:
            raise ValueError(
                "task reward needs unavailable completion evidence: "
                + ", ".join(sorted(missing_completion_evidence))
            )

        def step_detailed(state, observation, action_ids, keys):
            result = native_step(state, observation, action_ids, keys)
            next_state, next_observation, native_reward, done, mask, info = result
            previous_groups = observation_groups(state.learner_observation, groups)
            current_groups = observation_groups(next_state.learner_observation, groups)
            reward_kwargs = {}
            if accepts_terminal:
                reward_kwargs["terminal"] = done
            if (
                accepts_completion_evidence
                and episode_outcome is not None
                and "episode_success" in required_completion_evidence
            ):
                reward_kwargs["completion_evidence"] = {
                    "episode_success": episode_outcome(next_state, done).success,
                }
            shaped_reward = reward.apply(
                native_reward,
                previous_groups,
                current_groups,
                **reward_kwargs,
            )
            return next_state, next_observation, shaped_reward, done, mask, info

        def step(state, observation, action_ids, keys):
            return step_detailed(state, observation, action_ids, keys)[:5]

        environment = dataclasses.replace(
            environment, step=step, step_detailed=step_detailed
        )
    manifest = {
        "schema": "arena_jax_scene_v2",
        "name": name,
        "batch": batch,
        "loadout": loadout,
        "opponent": opponent,
        "parameters": overrides,
        "runtime_options": runtime_options,
        "providers": environment_kwargs,
        "world": world_identity,
        "reward": None if reward is None else reward.describe(),
        "policy_contract": policy_contract,
    }
    return ArenaScene(
        name=name,
        contract_sha256=_digest(manifest),
        environment=environment,
        combat_params=params,
        runtime_config=config,
        runtime_capacity=capacity,
        environment_kwargs=MappingProxyType(environment_kwargs),
        expected_batch=batch,
        expected_loadout=loadout,
        world=world,
        world_identity=world_identity,
        reward=reward,
        policy_contract=MappingProxyType(policy_contract),
    )


def _actor_input(state: Any, observation: jax.Array, mask: jax.Array) -> ActorInput:
    for _ in range(8):
        if hasattr(state, "learner_observation") and hasattr(state, "action_surface"):
            break
        if not hasattr(state, "environment"):
            raise TypeError(
                "Arena environment state does not expose policy evidence through "
                "learner_observation/action_surface or an environment wrapper"
            )
        state = state.environment
    else:
        raise TypeError("Arena environment wrapper nesting exceeds 8 levels")
    return ActorInput(
        observation, mask, state.learner_observation, state.action_surface
    )


def _select_rows(select: jax.Array, new: Any, old: Any) -> Any:
    def choose(candidate, current):
        if not hasattr(candidate, "shape") or candidate.ndim == 0:
            return candidate
        if candidate.shape[0] != select.shape[0]:
            return candidate
        shape = (select.shape[0],) + (1,) * (candidate.ndim - 1)
        return jnp.where(select.reshape(shape), candidate, current)

    return jax.tree.map(choose, new, old)


def collect(
    scene: ArenaScene,
    policy: Callable,
    record: Callable,
    key: jax.Array,
    ticks: int,
    *,
    initial_carry: Any = None,
) -> tuple[RolloutState, Any]:
    """Run a fixed JAX scan; terminal lanes remain frozen."""

    reset_key, scan_key = jax.random.split(key)
    keys = jax.random.split(reset_key, scene.expected_batch)
    state, observation, mask = scene.environment.reset(keys)
    actor_input = _actor_input(state, observation, mask)
    carry_structure = jax.tree.structure(initial_carry)

    def step(loop: RolloutState, step_key: jax.Array):
        policy_key, environment_key = jax.random.split(step_key)
        next_carry, factors = policy(loop.policy_carry, loop.actor_input, policy_key)
        if jax.tree.structure(next_carry) != carry_structure:
            raise TypeError("policy carry must match initial_carry")
        candidate = scene.environment.step_detailed(
            loop.environment,
            loop.actor_input.observation,
            factors,
            jax.random.split(environment_key, scene.expected_batch),
        )
        candidate_state, candidate_observation, reward, done, candidate_mask, info = (
            candidate
        )
        active = ~loop.finished
        next_state = _select_rows(active, candidate_state, loop.environment)
        next_observation = _select_rows(
            active, candidate_observation, loop.actor_input.observation
        )
        next_mask = _select_rows(active, candidate_mask, loop.actor_input.action_mask)
        next_actor_input = _actor_input(next_state, next_observation, next_mask)
        if next_carry is not None:
            next_carry = _select_rows(active, next_carry, loop.policy_carry)
        transition = Transition(
            loop.environment,
            loop.actor_input,
            factors,
            jnp.where(active, reward, 0.0),
            active & info.terminated,
            active & info.truncated,
            active & done,
            info,
            next_state,
            next_actor_input,
        )
        return (
            RolloutState(
                next_state, next_actor_input, next_carry, loop.finished | done
            ),
            record(transition),
        )

    return jax.lax.scan(
        step,
        RolloutState(
            state,
            actor_input,
            initial_carry,
            jnp.zeros((scene.expected_batch,), dtype=jnp.bool_),
        ),
        jax.random.split(scan_key, ticks),
    )


def _bind_parameters(apply: Callable, parameters: Any) -> Callable:
    def policy(state, actor_input, key):
        return apply(parameters, state, actor_input, key)

    return policy


@dataclass(frozen=True, slots=True)
class ArenaHandle:
    scene: ArenaScene

    def compile_collector(self, policy, record, ticks, *, initial_carry=None):
        return jax.jit(
            lambda key, carry=initial_carry: collect(
                self.scene,
                policy,
                record,
                key,
                ticks,
                initial_carry=carry,
            )
        )

    def compile_parameterized_collector(
        self, apply, record, ticks, *, initial_carry=None
    ):
        """Compile once while keeping policy parameters as dynamic inputs."""

        def run(parameters, key, carry=initial_carry):
            return collect(
                self.scene,
                _bind_parameters(apply, parameters),
                record,
                key,
                ticks,
                initial_carry=carry,
            )

        return jax.jit(run)

    def compile_evaluator(
        self,
        policy,
        record,
        score,
        ticks,
        *,
        initial_carry=None,
    ):
        """Compile rollout and scoring together, returning only success."""

        def evaluate(key, carry=initial_carry):
            _, trajectory = collect(
                self.scene,
                policy,
                record,
                key,
                ticks,
                initial_carry=carry,
            )
            return score(trajectory)

        return jax.jit(evaluate)

    def compile_parameterized_evaluator(
        self,
        apply,
        record,
        score,
        ticks,
        *,
        initial_carry=None,
    ):
        """Fuse rollout and scoring while accepting updated policy weights."""

        def evaluate(parameters, key, carry=initial_carry):
            _, trajectory = collect(
                self.scene,
                _bind_parameters(apply, parameters),
                record,
                key,
                ticks,
                initial_carry=carry,
            )
            return score(trajectory)

        return jax.jit(evaluate)

    def compile_batched_parameterized_evaluator(
        self,
        apply,
        record,
        score,
        ticks,
        *,
        initial_carry=None,
    ):
        """Evaluate a leading batch of policies with common environment keys."""

        def one(parameters, key):
            _, trajectory = collect(
                self.scene,
                _bind_parameters(apply, parameters),
                record,
                key,
                ticks,
                initial_carry=initial_carry,
            )
            return score(trajectory)

        return jax.jit(
            lambda parameters, key: jax.vmap(one, in_axes=(0, None))(parameters, key)
        )


__all__ = [
    "ActorInput",
    "ArenaHandle",
    "ArenaScene",
    "COMBAT_PARAMETER_NAMES",
    "ParameterizedPolicy",
    "RewardContract",
    "RolloutState",
    "Transition",
    "collect",
    "make_scene",
]
