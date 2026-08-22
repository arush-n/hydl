"""Bind an :class:`AgentSpec` to the framework-neutral JAX environment.

The upstream structured environment is canonical.  The ADK additionally
publishes its exact dense bridge-compatible projection, but policy tensors do
not replace or hide the structured learner observation, action surface, raw
state, or diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.combat.arsenal.factory import arsenal_runtime_capacity
from hytalegym.jax.combat.arsenal.environment import (
    ArsenalEnvironment,
    make_arsenal_environment,
)
from hytalegym.jax.combat.arsenal.profiles.hytale_0_5_7 import (
    hytale_0_5_7_loadouts,
)
from hytalegym.jax.combat.arsenal.runtime import arsenal_runtime_config
from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
    decode_arsenal_policy_actions,
    encode_arsenal_policy_actions,
    neutral_arsenal_policy_action_factors,
)
from hytalegym.jax.combat.types import default_combat_params
from hytalegym.jax.policy import sample_actions as gym_sample_actions

from adk.contracts.stamp import ContractStamp, current_stamp
from adk.core.spec import AgentSpec
from adk.runtime.backends import BackendSelection, resolve_backend
from adk.runtime.episode import EpisodeBoundary
from adk.runtime.policy_surface import DensePolicySurface, PolicyInput
from adk.runtime.scenes import JaxScene, resolve_scene
# The one shaping dispatcher, imported rather than reimplemented. It lives in
# `adk.contracts` because importing `adk.scenarios` from here is a cycle, and
# because two copies of "which contract does this term declare" is exactly how
# the console's carry-arity predicate drifted from its sibling wrapper.
from adk.contracts.shaping import call as _shaping_call


_ENVIRONMENT_HEAD_NAME_ALIASES = {
    "guard_held": "guard_off_on",
    "jump_held": "jump_off_on",
}


class JaxEnvironmentState(NamedTuple):
    """Raw Gym state plus its explicit actor-safe structured observation."""

    environment: Any
    learner_observation: Any


class EnvironmentStep(NamedTuple):
    """Backend-neutral structured step with explicit boundary availability."""

    state: JaxEnvironmentState
    policy_input: PolicyInput
    reward: jax.Array
    boundary: EpisodeBoundary
    info: Any

    @classmethod
    def from_done(
        cls,
        state: JaxEnvironmentState,
        policy_input: PolicyInput,
        reward: Any,
        done: Any,
        info: Any,
    ) -> "EnvironmentStep":
        """Adapt a legacy collapsed step without guessing its cause."""

        return cls(
            state=state,
            policy_input=policy_input,
            reward=reward,
            boundary=EpisodeBoundary.from_done(done),
            info=info,
        )

    @property
    def done(self):
        return self.boundary.done

    @property
    def terminated(self):
        return self.boundary.terminated

    @property
    def truncated(self):
        return self.boundary.truncated


@dataclass(frozen=True, slots=True)
class JaxBuildInputs:
    """Upstream objects used to load providers and build one JAX surface."""

    combat_params: Any
    runtime_config: Any
    runtime_capacity: Any
    batch: int


@dataclass(frozen=True, slots=True)
class BuiltAgent:
    """An agent spec resolved against one structured JAX runtime and scene."""

    spec: AgentSpec
    stamp: ContractStamp
    environment: ArsenalEnvironment
    policy_surface: DensePolicySurface
    combat_params: Any
    runtime_config: Any
    runtime_capacity: Any
    backend: BackendSelection
    scene: JaxScene
    world_geometry_config: Any
    batch: int
    #: Optional additive reward term, ``(next_state, info) -> (batch,)`` or
    #: ``(previous_state, next_state, info)`` when marked ``wants_previous``.
    #:
    #: Applied here rather than by each caller wrapping its own step, because
    #: everything downstream -- the collector, PPO's train step, a bare rollout
    #: -- goes through :meth:`step_factors`. Wrapping at one of those call sites
    #: would shape that one and silently leave the others on the native reward,
    #: so a policy could train against a shaped objective and be evaluated
    #: against a different one without anything reporting a difference.
    #:
    #: See :mod:`adk.scenarios.shaping` for the contract. A minigame is one
    #: implementation of it; nothing here knows what a minigame is.
    shaping: Any = None
    #: Optional episode length cap, in engine ticks (30 ticks = 1 second).
    #:
    #: The environment's own rule ends an episode only on death or on a world
    #: error -- there is no time limit anywhere below this. Measured
    #: consequence: episodes ran past 512 ticks, a 256-step update completed
    #: 0-3 of them, and `mean_episode_return` reported 0.000 on most updates
    #: because it had nothing to average. A learner cannot fit a value function
    #: against a boundary it almost never sees.
    #:
    #: Applied here for the same reason as `shaping`: the collector, PPO's
    #: train step and a bare rollout all come through :meth:`step_factors`, so
    #: capping at any one call site would cap that one alone.
    episode_tick_limit: int | None = None

    def with_shaping(self, shaping: Any) -> "BuiltAgent":
        """Return a copy carrying `shaping`. Frozen, so this is a new object."""

        import dataclasses

        return dataclasses.replace(self, shaping=shaping)

    def with_episode_limit(self, ticks: int | None) -> "BuiltAgent":
        """Return a copy that truncates an episode after `ticks` engine ticks."""

        import dataclasses

        if ticks is not None and int(ticks) < 1:
            raise ValueError(f"episode tick limit must be positive, got {ticks}")
        return dataclasses.replace(
            self, episode_tick_limit=None if ticks is None else int(ticks))

    def _truncated(self, next_state) -> Any:
        """Per-row time-limit flag, or None when no limit is set.

        Read off `tick_count`, which the environment already keeps per row --
        `runtime/reset.py:164` zeroes it and `runtime/step.py:298` increments
        it. So this needs no counter of its own, which matters: a counter held
        in a Python attribute would be incremented once at trace time and read
        the same dead value on every iteration of a compiled `lax.scan`. That
        exact bug has already been found and fixed once in the shaping seam.
        """

        if self.episode_tick_limit is None:
            return None
        ticks = next_state.runtime.combat.tick_count
        return jnp.asarray(ticks >= jnp.int32(self.episode_tick_limit), dtype=bool)

    @property
    def observation_size(self) -> int:
        return self.stamp.observation_size

    @property
    def logit_size(self) -> int:
        return sum(self.stamp.action_head_sizes)

    def reset(self, keys) -> tuple[JaxEnvironmentState, PolicyInput]:
        state, structured_observation = self.environment.reset(keys)
        policy_input = self.policy_surface.project(state, structured_observation)
        return (
            JaxEnvironmentState(state, structured_observation),
            policy_input,
        )

    def _step_parts(
        self,
        state: JaxEnvironmentState,
        factors,
        keys,
    ):
        """Everything both public step methods need, boundary causes apart.

        :meth:`step_factors` has to collapse the causes into one ``done``;
        :meth:`step_detailed` must not. Deriving both from one body is what
        stops the two from ever disagreeing about when an episode ended.
        """

        if not isinstance(state, JaxEnvironmentState):
            raise TypeError("state must be the JaxEnvironmentState returned by reset")
        values = jnp.asarray(factors)
        expected_shape = (self.batch, len(self.stamp.action_head_sizes))
        if values.dtype != jnp.dtype(jnp.int32):
            raise TypeError("action factors must have dtype int32")
        if values.shape != expected_shape:
            raise ValueError(
                f"action factors must have shape {expected_shape}, got {values.shape}"
            )
        (
            next_state,
            next_structured_observation,
            reward,
            done,
            info,
        ) = self.environment.step_factors(
            state.environment,
            values,
            keys,
        )
        next_input = self.policy_surface.project(
            next_state,
            next_structured_observation,
        )
        if self.shaping is not None:
            # Additive: the four native terms stay intact underneath, so a
            # shaped run is still comparable to the unshaped baseline on those
            # alone. That comparability is the only reason a shaped number can
            # be read at all.
            #
            # Both raw states go in. A term paying for a CHANGE reads the
            # difference between them instead of remembering the previous tick
            # in a Python dict -- which works eagerly and is silently zero
            # under `jax.jit`, because a traced-once `lax.scan` body re-reads
            # the same dead value every iteration. Training compiles this step.
            #
            # `observe_done` survives for terms that carry their own state, but
            # nothing in `minigames/` needs it now: a boundary is readable from
            # the state itself, purely.
            hook = getattr(self.shaping, "observe_done", None)
            if callable(hook):
                hook(done)
            reward = reward + jnp.asarray(
                _shaping_call(self.shaping, state.environment, next_state, info),
                dtype=jnp.float32)
        env_done = jnp.asarray(done, dtype=bool)
        capped = self._truncated(next_state)
        # Death wins a tie. If the agent died on the very tick the cap landed,
        # that is a real terminal and a learner must NOT bootstrap a future
        # value from it -- there is no future. Labelling the overlap
        # `truncated` would teach the critic that dying is worth whatever
        # V(s') happens to predict for a corpse.
        truncated = None if capped is None else (capped & ~env_done)
        return (
            JaxEnvironmentState(next_state, next_structured_observation),
            next_input,
            reward,
            env_done,
            truncated,
            info,
        )

    def step_factors(
        self,
        state: JaxEnvironmentState,
        factors,
        keys,
    ):
        """Step with explicit ``int32[B, H]`` action factors.

        The episode cap arrives collapsed into ``done``, because this signature
        has nowhere else to put it. The environment does not reset on ``done``
        -- the collector does -- so a capped row behaves exactly like a dead
        one, which is the existing contract rather than a new one.
        """

        wrapped, next_input, reward, env_done, truncated, info = self._step_parts(
            state, factors, keys)
        done = env_done if truncated is None else (env_done | truncated)
        return (wrapped, next_input, reward, done, info)

    def step(self, state: JaxEnvironmentState, factors, keys):
        """Alias for :meth:`step_factors`; no scalar action transport exists."""

        return self.step_factors(state, factors, keys)

    def step_detailed(
        self,
        state: JaxEnvironmentState,
        factors,
        keys,
    ) -> EnvironmentStep:
        """Step while preserving whether the episode-cause split is known.

        Gym's current structured Arsenal environment returns only ``done``, so
        with no episode limit set both causes stay unavailable rather than
        labelling every boundary a termination.

        **With a limit set, the split becomes known and is published.** The
        environment's own rule is death-or-world-error, and the cap is this
        adapter's, so the two are separable here and nowhere below. That
        distinction is worth publishing rather than collapsing: a learner must
        bootstrap from the value function when an episode was cut off by a
        clock and must not when the agent actually died. Collapsing a time
        limit into ``terminated`` teaches the critic that running out of time
        is worth zero future reward, which is the bias
        :meth:`EpisodeBoundary.require_split` exists to let a learner refuse.
        """

        wrapped, next_input, reward, env_done, truncated, info = self._step_parts(
            state, factors, keys)
        if truncated is None:
            return EnvironmentStep.from_done(
                wrapped,
                next_input,
                reward,
                env_done,
                info,
            )
        # Exhaustive and disjoint by construction: `_step_parts` already
        # removed the overlap, so `terminated | truncated` reproduces the
        # `done` that `step_factors` publishes, exactly.
        return EnvironmentStep(
            state=wrapped,
            policy_input=next_input,
            reward=reward,
            boundary=EpisodeBoundary.from_split(
                terminated=env_done,
                truncated=truncated,
            ),
            info=info,
        )


def _resolve_jax_backend(
    backend: str | Mapping[str, Any] | BackendSelection,
) -> BackendSelection:
    if isinstance(backend, BackendSelection):
        selection = backend
    elif isinstance(backend, str):
        selection = resolve_backend(
            {
                "backend": backend,
                "entry_point": "build",
            }
        )
    elif isinstance(backend, Mapping):
        options = dict(backend)
        options.setdefault("entry_point", "build")
        selection = resolve_backend(options)
    else:
        raise TypeError("backend must be a name, mapping, or BackendSelection")
    if selection.backend != "jax":
        raise RuntimeError(
            "BuiltAgent is the batched JAX lane; use the bridge evidence "
            "session for native transfer"
        )
    return selection


def prepare_jax_build_inputs(spec: AgentSpec, batch: int) -> JaxBuildInputs:
    """Create the exact host-side inputs expected by Gym provider loaders."""

    if not isinstance(spec, AgentSpec):
        raise TypeError("spec must be an AgentSpec")
    if isinstance(batch, bool) or not isinstance(batch, int):
        raise TypeError("batch must be an integer")
    if batch < 1:
        raise ValueError(f"batch must be >= 1: {batch}")
    combat_params = default_combat_params(microticks=spec.microticks)
    loadouts = hytale_0_5_7_loadouts([spec.loadout] * batch)
    return JaxBuildInputs(
        combat_params=combat_params,
        runtime_config=arsenal_runtime_config(loadouts),
        runtime_capacity=arsenal_runtime_capacity(loadouts),
        batch=batch,
    )


def build(
    spec: AgentSpec,
    *,
    batch: int = 1,
    backend: str | Mapping[str, Any] | BackendSelection = "jax",
    scene: JaxScene | None = None,
) -> BuiltAgent:
    """Resolve a spec against the current structured JAX environment.

    Custom provider bundles must arrive as a named, contract-stamped
    :class:`JaxScene`; an unknown scene never falls back to empty providers.
    """

    prepared = prepare_jax_build_inputs(spec, batch)
    selection = _resolve_jax_backend(backend)
    resolved_scene = resolve_scene(spec.scene, scene)
    if (
        resolved_scene.expected_batch is not None
        and resolved_scene.expected_batch != batch
    ):
        raise ValueError(
            "JAX scene was prepared for batch "
            f"{resolved_scene.expected_batch}, not {batch}"
        )
    if (
        resolved_scene.expected_loadout is not None
        and resolved_scene.expected_loadout != spec.loadout
    ):
        raise ValueError(
            "JAX scene was prepared for loadout "
            f"{resolved_scene.expected_loadout!r}, not {spec.loadout!r}"
        )
    world_geometry_config = resolved_scene.environment_kwargs.get(
        "world_geometry_config"
    )
    params = resolved_scene.combat_params
    if params is None:
        params = prepared.combat_params
    elif not hasattr(params, "microticks"):
        raise TypeError("scene combat_params must expose microticks")
    else:
        parameter_microticks = np.asarray(
            jax.device_get(params.microticks)
        )
        if parameter_microticks.shape != ():
            raise ValueError("scene combat_params.microticks must be scalar")
        if int(parameter_microticks) != spec.microticks:
            raise ValueError(
                "scene combat params were prepared for different microticks"
            )
    runtime = (
        prepared.runtime_config
        if resolved_scene.runtime_config is None
        else resolved_scene.runtime_config
    )
    runtime_capacity = (
        prepared.runtime_capacity
        if resolved_scene.runtime_capacity is None
        else resolved_scene.runtime_capacity
    )
    environment = make_arsenal_environment(
        params,
        runtime,
        maximum_turn_degrees=spec.maximum_turn_degrees,
        **resolved_scene.environment_kwargs,
    )
    stamp = current_stamp(world_geometry_config)
    if environment.spec.is_dense:
        raise RuntimeError("canonical Arsenal environment unexpectedly became dense")
    environment_head_sizes = tuple(
        int(component.size)
        for component in environment.spec.action_components
    )
    if environment_head_sizes != stamp.action_head_sizes:
        raise RuntimeError(
            "structured environment action components disagree with current stamp"
        )
    environment_head_names = tuple(
        _ENVIRONMENT_HEAD_NAME_ALIASES.get(component.name, component.name)
        for component in environment.spec.action_components
    )
    if environment_head_names != stamp.action_head_names:
        raise RuntimeError(
            "structured environment action-component order disagrees with "
            "current policy head names"
        )
    policy_surface = DensePolicySurface.from_contract(
        environment.spec,
        runtime,
        observation_size=stamp.observation_size,
        action_size=sum(stamp.action_head_sizes),
    )
    return BuiltAgent(
        spec=spec,
        stamp=stamp,
        environment=environment,
        policy_surface=policy_surface,
        combat_params=params,
        runtime_config=runtime,
        runtime_capacity=runtime_capacity,
        backend=selection,
        scene=resolved_scene,
        world_geometry_config=world_geometry_config,
        batch=batch,
    )


def observe(policy_input: Any) -> jax.Array:
    """Return the already-flattened dense policy observation."""

    observation = getattr(policy_input, "observation", None)
    if observation is None:
        raise TypeError(
            "observe expects an ADK input with a dense observation"
        )
    return observation


def legal_mask(policy_input: Any) -> jax.Array:
    """Return the mask composed by the shared dense policy projection."""

    mask = getattr(policy_input, "action_mask", None)
    if mask is None:
        raise TypeError("legal_mask expects an ADK input with an action mask")
    return mask


def encode(action) -> jax.Array:
    """Structured learner action -> explicit ``int32[B, 12]`` factors."""

    return encode_arsenal_policy_actions(action)


def decode(factors) -> Any:
    """Explicit ``int32[B, 12]`` factors -> structured learner action."""

    return decode_arsenal_policy_actions(factors)


def sample_actions(logits, mask, key) -> jax.Array:
    """Sample the twelve independent masked categorical heads."""

    values = jnp.asarray(logits)
    legal = jnp.asarray(mask)
    if values.shape != legal.shape:
        raise ValueError("logits and action mask must have identical shapes")
    if legal.dtype != jnp.dtype(jnp.bool_):
        raise TypeError("action mask must have dtype bool")
    if values.shape[-1] != sum(ARSENAL_POLICY_ACTION_HEAD_SIZES):
        raise ValueError("logit width does not match the live action heads")
    masked = jnp.where(legal, values, -jnp.inf)
    factors, _ = gym_sample_actions(
        key,
        masked,
        tuple(ARSENAL_POLICY_ACTION_HEAD_SIZES),
        transport="factors",
    )
    return factors


def neutral_actions(batch: int) -> jax.Array:
    """Return the current all-neutral ``int32[B, H]`` factor action."""

    if isinstance(batch, bool) or not isinstance(batch, int):
        raise TypeError("batch must be an integer")
    if batch < 1:
        raise ValueError("batch must be positive")
    return neutral_arsenal_policy_action_factors(batch)


def step_with_boundary(
    built: Any,
    state: JaxEnvironmentState,
    factors: Any,
    keys: Any,
) -> EnvironmentStep:
    """Use a detailed backend step, or losslessly adapt the legacy 5-tuple.

    Lightweight wrappers used by algorithms may still implement only
    ``step_factors``.  Their collapsed ``done`` remains supported, but its
    terminated/truncated cause is explicitly unavailable.
    """

    detailed = getattr(built, "step_detailed", None)
    if detailed is not None:
        result = detailed(state, factors, keys)
        if not isinstance(result, EnvironmentStep):
            raise TypeError("step_detailed must return EnvironmentStep")
        # Validate and canonicalize without synchronizing traced values. A
        # known split owns ``done``; a collapsed producer owns only ``done``.
        return result._replace(boundary=result.boundary.canonical())
    next_state, next_input, reward, done, info = built.step_factors(
        state,
        factors,
        keys,
    )
    return EnvironmentStep.from_done(
        next_state,
        next_input,
        reward,
        done,
        info,
    )


__all__ = [
    "BuiltAgent",
    "EnvironmentStep",
    "JaxBuildInputs",
    "JaxEnvironmentState",
    "PolicyInput",
    "build",
    "decode",
    "encode",
    "legal_mask",
    "neutral_actions",
    "observe",
    "prepare_jax_build_inputs",
    "sample_actions",
    "step_with_boundary",
]
