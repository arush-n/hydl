"""Flat two-actor pursuit arena for the composition task, one decision at a time.

``build_flat_arena`` reconstructs the environment the pursuit candidate was
trained on (``agent_skills/training_runs/pursuit_contact/task.json``, run by
``agent_skills/training_runs/tools/train_pursuit.py``). The Arena pursuit runner
is the only place where that construction is written down, and it is one large
training function. The seams below are the same functions that runner calls:

* ``_materialize_design`` (``arena/training/runs/pursuit_run.py``) builds the
  ``completely_flat`` design through ``worlds.flat.flat_arena``. It is private to
  the runner and is reused as it is, so the design cache and reset law match.
* The sensor-range override is the block at ``pursuit_run.py:2409-2426``. That
  block is not a function, so its statements are repeated in
  :func:`build_flat_arena`.
* ``_facing_reset`` (``pursuit_run.py``) spawns the learner facing its target,
  because the candidate sets ``spawn_facing_target``.
* ``bind_pursuit_collector`` (``arena/training/contracts/pursuit.py``) binds the
  candidate's action-mask, observation and transition transforms. It receives
  :func:`_flat_environment` in place of ``make_multi_actor_rollout_collector``.

The one-step body in :meth:`FlatPursuitEnvironment.step_detailed` follows
``rollout_step`` in ``hytalegym/jax/training/multi_actor/rollout.py`` (lines
346-511) and keeps its order: project, observation transform, action-mask
transform, decode, advance, transition transform, done. It drops sampling,
recording and auto-reset. The learner's factors come from a Skill, and
``adk.runtime.hierarchical`` resets lanes itself. The scripted opponent's factors
are the single legal choice of its forced mask, which is what
``apply_assigned_actor_controllers`` (``inference.py``) computes for a scripted slot.

Name clash: :class:`FlatArena` here is not ``worlds.flat.FlatArena``. That class is
the design's reset and capability seam, and this module wraps it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from adk.runtime.env_adapter import EnvironmentStep, JaxEnvironmentState
from adk.runtime.policy_surface import PolicyInput
from arena.training.contracts.pursuit import (
    PERCEPTION_MODES,
    PursuitStageConfig,
    bind_pursuit_collector,
    pursuit_learner_action_scope,
)
from arena.training.runs.pursuit_run import _facing_reset, _materialize_design
from arena.training.skills.program import TargetMode, TargetMotionProgram
from arena.training.skills.stage import skill_duel_roles
from experimental.neural_agent.contracts import ACTION_HEADS
from hytalegym.jax.combat.arsenal.environment import (
    ArsenalActionSurfaceLifecycle,
    ArsenalEnvironmentInfo,
)
from hytalegym.jax.combat.arsenal.factory import empty_arsenal_commands
from hytalegym.jax.combat.observation.v1.schema.types import DoorIntentRequest
from hytalegym.jax.combat.observation.v3.policy import ARSENAL_POLICY_ACTION_HEAD_SIZES
from hytalegym.jax.combat.opponents.runtime.policy import (
    first_legal_opponent_ability_slots,
)
from hytalegym.jax.training.multi_actor import (
    MultiActorArenaState,
    PolicyActorAssignment,
    decode_policy_actor_factors_flat,
    initialize_multi_actor_arena_state,
    observe_policy_actors_flat,
    policy_actor_assignment,
    scripted_opponent_ability_slots,
    step_multi_actor_arena,
)

FLAT_ARENA_SCHEMA = "experimental-neural-agent-flat-pursuit-v1"
FLAT_DESIGN_ID = "completely_flat"
#: Physical actor row and policy slot of the learner. ``skill_duel_roles`` derives
#: these from the assignment, and :func:`build_flat_arena` checks that they match.
LEARNER_ROW = 0
#: Physical actor row and policy slot of the scripted opponent.
OPPONENT_ROW = 1

# Candidate settings, each one traced to its source in this module's docstring.
_LOADOUT = "iron_sword"  # PursuitRunConfig.loadout default; task.json does not set it.
_OPPONENT_LOADOUT = "iron_sword"  # PursuitRunConfig.opponent default (a loadout id).
_TARGET_SEPARATION_RANGE = (4.0, 8.0)  # task.json settings.target_separation_range.
_ARENA_RADIUS = 48.0  # PursuitRunConfig.arena_radius default.
_SENSOR_RANGE = 16.0  # task.json settings.sensor_range.
_GAIT_MIX = (1.0, 0.0, 0.0)  # task.json settings.evader_gait_mix.
_GAIT_PERIOD_TICKS = 20  # PursuitRunConfig.evader_gait_period_ticks default.
_TARGET_CONTROLLER = "scripted_flee_adaptive"  # task.json settings.target_controller.
#: The brief's ``opponent`` names a scripted controller, not a loadout. This maps
#: each accepted name to the candidate's Arena controller.
_OPPONENTS: Mapping[str, str] = {"scripted_flee": _TARGET_CONTROLLER}

_HEAD_SIZES: tuple[int, ...] = tuple(int(size) for size in ARSENAL_POLICY_ACTION_HEAD_SIZES)
_HEAD_STARTS: tuple[int, ...] = tuple(
    int(start) for start in np.cumsum((0,) + _HEAD_SIZES[:-1])
)
if len(_HEAD_SIZES) != ACTION_HEADS:
    raise RuntimeError(
        f"Arsenal policy has {len(_HEAD_SIZES)} action heads; contracts expect {ACTION_HEADS}"
    )


def forced_factors(mask: jax.Array) -> jax.Array:
    """Return the first legal choice of each head for an actor whose mask is forced.

    Args:
        mask: bool[B, 124], the action mask of one actor. Its 12 head segments
            follow ``ARSENAL_POLICY_ACTION_HEAD_SIZES``.

    Returns:
        int32[B, 12]. Entry ``[b, h]`` is the position of the first legal choice
        in head ``h`` of lane ``b``. A scripted mask has exactly one legal choice
        per head, so this is that choice. A head with no legal choice gives 0, so
        pair this with :func:`target_choice_is_forced`.
    """

    columns = [
        jnp.argmax(mask[:, start : start + size], axis=-1).astype(jnp.int32)
        for start, size in zip(_HEAD_STARTS, _HEAD_SIZES, strict=True)
    ]
    return jnp.stack(columns, axis=-1)


def factors_are_legal(mask: jax.Array, factors: jax.Array) -> jax.Array:
    """Return True for each lane whose factors are in range and allowed by the mask.

    Args:
        mask: bool[B, 124], the action mask of one actor.
        factors: int32[B, 12], one factor per head.

    Returns:
        bool[B].
    """

    legal = jnp.ones(factors.shape[:1], dtype=jnp.bool_)
    for index, (start, size) in enumerate(zip(_HEAD_STARTS, _HEAD_SIZES, strict=True)):
        choice = factors[:, index]
        in_range = (choice >= 0) & (choice < size)
        safe = jnp.clip(choice, 0, size - 1)
        allowed = jnp.take_along_axis(
            mask[:, start : start + size], safe[:, None], axis=1
        )[:, 0]
        legal = legal & in_range & allowed
    return legal


def target_choice_is_forced(mask: jax.Array) -> jax.Array:
    """Return True for each lane where every head has exactly one legal choice.

    Args:
        mask: bool[B, 124], the action mask of the scripted opponent.

    Returns:
        bool[B]. This is the invariant that makes the opponent's factors
        deterministic. :meth:`FlatPursuitEnvironment.step_detailed` does not check
        it under ``jit``, because a traced value cannot raise. The tests check it.
    """

    forced = jnp.ones(mask.shape[:1], dtype=jnp.bool_)
    for start, size in zip(_HEAD_STARTS, _HEAD_SIZES, strict=True):
        forced = forced & (jnp.sum(mask[:, start : start + size], axis=-1) == 1)
    return forced


def _learner_factors(factors: Any, batch: int) -> jax.Array:
    """Return ``factors`` as int32[batch, 12], or raise on any other type or shape."""

    values = jnp.asarray(factors)
    if values.dtype != jnp.int32:
        raise TypeError("action factors must have dtype int32")
    if values.shape != (batch, ACTION_HEADS):
        raise ValueError(
            f"action factors must have shape {(batch, ACTION_HEADS)}, got {values.shape}"
        )
    return values


def _no_surface_lifecycle(batch: int) -> ArsenalActionSurfaceLifecycle:
    """Return an all-zero surface lifecycle for three surface verbs.

    This arena publishes no action surface. ``observe_policy_actors_flat`` keeps
    the world-action candidate surfaces closed, so no surface verb can be
    requested and every lifecycle flag is False.
    """

    flags = jnp.zeros((batch, 3), dtype=jnp.bool_)
    return ArsenalActionSurfaceLifecycle(
        requested=flags,
        accepted=flags,
        started=flags,
        finished=flags,
        rejected=flags,
        cancelled=flags,
        reject_reason=jnp.zeros((batch, 3), dtype=jnp.uint32),
    )


def _no_door_intent(batch: int) -> DoorIntentRequest:
    """Return the empty door request. The closed world-action surface cannot request a door."""

    flags = jnp.zeros((batch,), dtype=jnp.bool_)
    return DoorIntentRequest(
        requested=flags,
        accepted=flags,
        intent=jnp.zeros((batch,), dtype=jnp.int32),
        interaction_slot=jnp.full((batch,), -1, dtype=jnp.int32),
        object_id=jnp.full((batch,), -1, dtype=jnp.int32),
    )


@dataclass(frozen=True, slots=True)
class FlatPursuitEnvironment:
    """A batched two-actor pursuit arena with the ``built`` interface of ``adk.runtime.env_adapter``.

    ``adk.runtime.hierarchical.compile_hierarchical_collector`` reads ``batch``,
    ``reset(keys)`` and ``step_detailed`` (or ``step_factors``). The learner is
    physical row 0 and the scripted opponent is row 1 in every lane. Its
    ``step_detailed`` returns ``info`` as an ``ArsenalEnvironmentInfo``, so the
    default ``jax_hierarchical_context`` accepts it.

    Attributes:
        batch: Number of lanes B, a positive int.
        params: The CombatParams for the arena, with the sensor range overridden.
        config: The ArsenalRuntimeConfig for the arena, with the sensor range overridden.
        assignment: The PolicyActorAssignment, learner at slot 0 and opponent at slot 1.
        entity_count: Entity rows per arena, taken from ``config`` as the collector does.
        reset_provider: Keys [B] to ArsenalEnvironmentState, with the facing reset applied.
        capability_provider: ArsenalEnvironmentState to ArsenalWorldCapabilities.
        action_mask_transform: The candidate's mask transform, (arena, observation, mask) to mask.
        observation_transform: The candidate's observation transform, (arena, observation) to observation.
        transition_transform: The candidate's transition transform, (arena, transition) to transition.
        maximum_turn_degrees: Decode turn cap in degrees, 24.3 for the candidate.
    """

    batch: int
    params: Any
    config: Any
    assignment: PolicyActorAssignment
    entity_count: int
    reset_provider: Callable[[jax.Array], Any]
    capability_provider: Callable[[Any], Any]
    action_mask_transform: Callable[[Any, Any, jax.Array], jax.Array]
    observation_transform: Callable[[Any, Any], Any]
    transition_transform: Callable[[Any, Any], Any]
    maximum_turn_degrees: float

    def reset(self, keys: jax.Array) -> tuple[JaxEnvironmentState, PolicyInput]:
        """Reset every lane and return the learner's legal policy input.

        Args:
            keys: PRNG keys with leading axis B = ``batch``.

        Returns:
            ``(state, policy_input)``. ``state.environment`` is a
            MultiActorArenaState. ``state.learner_observation`` is the learner's
            LearnerCombatObservationV3 with leading axis B. ``policy_input`` is a
            PolicyInput whose ``observation`` is float32[B, ARSENAL_POLICY_OBSERVATION_SIZE] (15656 on 0.7.2, 8272 on 0.5.7) and whose
            ``action_mask`` is bool[B, 124].
        """

        arena = initialize_multi_actor_arena_state(
            self.reset_provider(keys), self.params
        )
        observation, _, mask = self._policy_view(arena)
        return self._learner_state(arena, observation), self._learner_input(
            observation, mask
        )

    def step_detailed(
        self,
        state: JaxEnvironmentState,
        factors: Any,
        keys: jax.Array,
    ) -> EnvironmentStep:
        """Advance every lane by one decision, using the learner's factors.

        Args:
            state: The JaxEnvironmentState from :meth:`reset` or the previous step.
            factors: int32[B, 12], the learner's factors. They must be legal under
                the learner's policy input, which the Skill contract guarantees.
            keys: PRNG keys with leading axis B, one per lane.

        Returns:
            An EnvironmentStep. Its ``state`` and ``policy_input`` are the next
            decision's inputs. ``reward`` is float32[B] for the learner. ``done``
            is bool[B], a collapsed boundary that includes the tick-budget
            truncation, which the pursuit transition sets at ``maximum_ticks``.
            ``info`` is an ArsenalEnvironmentInfo whose ``terminated | truncated``
            equals ``done``. Its ``truncated`` field is the non-terminal part of
            ``done``: the tick horizon, plus any other truncation the transition
            reports, such as arena-geometry exhaustion (``arsenal/runtime.py``).

        Raises:
            TypeError: If ``state`` is not from this environment, or ``factors`` is
                not int32.
            ValueError: If ``factors`` does not have shape [B, 12].
        """

        if not isinstance(state, JaxEnvironmentState) or not isinstance(
            state.environment, MultiActorArenaState
        ):
            raise TypeError("state must be the JaxEnvironmentState from this environment's reset")
        values = _learner_factors(factors, self.batch)
        arena = state.environment
        observation, capabilities, mask = self._policy_view(arena)
        opponent = forced_factors(mask[:, OPPONENT_ROW])
        both = jnp.stack((values, opponent), axis=1)
        controls = decode_policy_actor_factors_flat(
            arena.arsenal,
            observation.structured,
            both,
            capabilities,
            self.config,
            arena.locomotion,
            self.assignment,
            maximum_turn_degrees=self.maximum_turn_degrees,
        )
        commands = empty_arsenal_commands(
            self.batch, entity_count=self.entity_count
        )._replace(
            ability_slot=scripted_opponent_ability_slots(
                arena.arsenal,
                capabilities,
                self.config,
                self.assignment,
                first_legal_opponent_ability_slots,
            ),
            world=capabilities,
        )
        transition = step_multi_actor_arena(
            arena,
            controls,
            self.assignment,
            keys,
            self.params,
            commands,
            self.config,
        )
        transition = self.transition_transform(arena, transition)
        actor_done = transition.terminated | transition.truncated
        done = jnp.any(actor_done & self.assignment.active, axis=1)
        reward = jnp.where(
            controls.valid[:, LEARNER_ROW],
            transition.reward[:, LEARNER_ROW],
            jnp.float32(0.0),
        )
        next_observation, _, next_mask = self._policy_view(transition.state)
        # A diagnostic split of ``done``: terminal for any actor (catch or death),
        # and truncated for every other boundary the transition reports, the horizon
        # included. The boundary stays collapsed, so the ADK still treats the cause
        # as unknown.
        terminated_lanes = jnp.any(transition.terminated & self.assignment.active, axis=1)
        info = ArsenalEnvironmentInfo(
            action_valid=controls.valid[:, LEARNER_ROW]
            & factors_are_legal(mask[:, LEARNER_ROW], values),
            # No action surface is published, so nothing is requested through
            # one. Both legality flags are therefore vacuously True.
            action_surface_legal=jnp.ones((self.batch,), dtype=jnp.bool_),
            world_verb_legal=jnp.ones((self.batch,), dtype=jnp.bool_),
            action_surface_lifecycle=_no_surface_lifecycle(self.batch),
            door=_no_door_intent(self.batch),
            combat_info=transition.combat_info,
            arsenal_info=transition.arsenal_info,
            terminated=terminated_lanes,
            truncated=done & ~terminated_lanes,
        )
        return EnvironmentStep.from_done(
            self._learner_state(transition.state, next_observation),
            self._learner_input(next_observation, next_mask),
            reward,
            done,
            info,
        )

    def step_factors(
        self,
        state: JaxEnvironmentState,
        factors: Any,
        keys: jax.Array,
    ) -> tuple[JaxEnvironmentState, PolicyInput, jax.Array, jax.Array, Any]:
        """Return the tuple form of :meth:`step_detailed`, for legacy collectors.

        Args:
            state: As for :meth:`step_detailed`.
            factors: As for :meth:`step_detailed`.
            keys: As for :meth:`step_detailed`.

        Returns:
            ``(state, policy_input, reward, done, info)``, the same values as
            :meth:`step_detailed`.
        """

        step = self.step_detailed(state, factors, keys)
        return step.state, step.policy_input, step.reward, step.done, step.info

    def _policy_view(self, arena: MultiActorArenaState):
        """Return the observation, the world capabilities and the learner-and-opponent mask.

        This follows ``project``, then the observation transform, then the
        action-mask transform, in the same order as ``rollout_step``.
        """

        capabilities = self.capability_provider(arena.arsenal)
        observation = observe_policy_actors_flat(
            arena.arsenal,
            self.params,
            capabilities,
            self.config,
            arena.locomotion,
            self.assignment,
        )
        observation = self.observation_transform(arena, observation)
        mask = jnp.asarray(
            self.action_mask_transform(arena, observation, observation.action_mask)
        )
        if mask.dtype != jnp.bool_:
            raise TypeError("action_mask_transform must return a boolean action mask")
        if mask.shape != observation.action_mask.shape:
            raise ValueError("action_mask_transform must preserve the action-mask shape")
        return observation, capabilities, mask

    def _learner_state(self, arena: MultiActorArenaState, observation: Any) -> JaxEnvironmentState:
        """Pair the arena with the learner's structured observation, slot 0 of every leaf."""

        legal = jax.tree_util.tree_map(
            lambda leaf: leaf[:, LEARNER_ROW], observation.structured
        )
        return JaxEnvironmentState(environment=arena, learner_observation=legal)

    def _learner_input(self, observation: Any, mask: jax.Array) -> PolicyInput:
        """Return the learner's dense observation and action mask as a PolicyInput."""

        return PolicyInput(
            observation=observation.dense[:, LEARNER_ROW],
            action_mask=mask[:, LEARNER_ROW],
        )


def _flat_environment(
    params: Any,
    config: Any,
    assignment: PolicyActorAssignment,
    *,
    reset_provider: Callable[[jax.Array], Any],
    capability_provider: Callable[[Any], Any],
    action_mask_transform: Callable[[Any, Any, jax.Array], jax.Array],
    observation_transform: Callable[[Any, Any], Any],
    transition_transform: Callable[[Any, Any], Any],
    maximum_turn_degrees: float,
) -> FlatPursuitEnvironment:
    """Return the environment for the transforms that ``bind_pursuit_collector`` binds.

    This factory takes the place of ``make_multi_actor_rollout_collector``.
    ``bind_pursuit_collector`` calls it with the candidate's transforms, so the
    transforms are not re-derived here.
    """

    return FlatPursuitEnvironment(
        batch=int(assignment.actor_index.shape[0]),
        params=params,
        config=config,
        assignment=assignment,
        entity_count=int(config.loadout.weapon_id.shape[1]),
        reset_provider=reset_provider,
        capability_provider=capability_provider,
        action_mask_transform=action_mask_transform,
        observation_transform=observation_transform,
        transition_transform=transition_transform,
        maximum_turn_degrees=float(maximum_turn_degrees),
    )


class FlatArena(NamedTuple):
    """One built flat pursuit arena, ready for ``compile_hierarchical_collector``.

    Attributes:
        built: The :class:`FlatPursuitEnvironment`, with ``batch``, ``reset`` and
            ``step_detailed``.
        reset_keys: PRNG keys with leading axis ``batch``, for the first reset.
        learner_role_index: Physical actor row of the learner, 0.
        opponent_role_index: Physical actor row of the scripted opponent, 1.
        description: The construction record. :meth:`describe` returns a copy.
    """

    built: FlatPursuitEnvironment
    reset_keys: jax.Array
    learner_role_index: int
    opponent_role_index: int
    description: Mapping[str, Any]

    def describe(self) -> dict[str, Any]:
        """Return a shallow copy of the construction record. Nested values stay shared with it.

        The record names the functions that produced the environment and the
        settings they were given, so a result can be traced to the candidate.
        """

        return dict(self.description)


def build_flat_arena(
    batch: int,
    seed: int,
    *,
    opponent: str = "scripted_flee",
    perception: str = "field_of_view",
    ticks: int = 256,
) -> FlatArena:
    """Build the candidate's flat two-actor pursuit arena with ``batch`` lanes.

    The candidate settings are fixed to ``task.json``, apart from the arguments
    listed here: design ``completely_flat``, separation 4 to 8 blocks, sensor range
    16 blocks, the ``scripted_flee_adaptive`` controller with gait mix (1, 0, 0),
    the learner spawned facing its target, and jump and combat closed.

    Args:
        batch: Number of lanes B, a positive int.
        seed: Seed for the first reset keys, an int. ``jax.random.key(seed)`` is
            split into ``batch`` keys.
        opponent: Scripted opponent controller. Only ``"scripted_flee"`` is
            accepted, and it maps to ``scripted_flee_adaptive``.
        perception: ``"field_of_view"``, the candidate's setting and the default. Under
            it the privileged target channel is withheld, so the learner's observation
            is legal. ``"omniscient"`` writes that channel. An omniscient arena is an
            oracle ablation under ADR-001 D5: label it, and never report it as agent
            performance.
        ticks: Episode horizon in decision ticks, an int in ``[9, 1000]``. The
            pursuit transition truncates each lane at this count. The stage
            config refuses 8 or fewer, because its stuck patience is 8 ticks.

    Returns:
        A :class:`FlatArena`. ``built`` has ``batch`` lanes, its ``reset`` takes
        keys of shape [batch], and its observations are float32[batch, ARSENAL_POLICY_OBSERVATION_SIZE] (15656 on 0.7.2) with
        an action mask of bool[batch, 124].

    Raises:
        TypeError: If ``batch``, ``seed`` or ``ticks`` has the wrong type. The
            pursuit stage config raises it for ``ticks``.
        ValueError: If ``batch`` is not positive, ``opponent`` or ``perception``
            is not accepted, or the target can start outside the sensor range.
    """

    if isinstance(batch, bool) or not isinstance(batch, (int, np.integer)):
        raise TypeError(f"batch must be an integer, got {type(batch).__name__}")
    if batch < 1:
        raise ValueError(f"batch must be positive, got {batch}")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer, got {type(seed).__name__}")
    lanes = int(batch)
    if opponent not in _OPPONENTS:
        raise ValueError(f"opponent must be one of {sorted(_OPPONENTS)}, got {opponent!r}")
    if perception not in PERCEPTION_MODES:
        raise ValueError(
            f"perception must be one of {sorted(PERCEPTION_MODES)}, got {perception!r}"
        )
    if _TARGET_SEPARATION_RANGE[1] > _SENSOR_RANGE:
        raise ValueError(
            "target separation begins outside the sensor range: "
            f"{_TARGET_SEPARATION_RANGE[1]} > {_SENSOR_RANGE} blocks"
        )

    # pursuit_run.py: PursuitStageConfig(maximum_ticks, allow_jump, **stage_options).
    # Built before the design, so an invalid ``ticks`` fails before any arena is built.
    stage = PursuitStageConfig(
        maximum_ticks=ticks,
        allow_jump=False,
        perception=perception,
        spawn_facing_target=True,
    )
    # pursuit_run.py: run_pursuit_training builds the design arena with this call.
    design = _materialize_design(
        FLAT_DESIGN_ID,
        _LOADOUT,
        _OPPONENT_LOADOUT,
        lanes,
        _TARGET_SEPARATION_RANGE,
        _ARENA_RADIUS,
        1.0,
        1.0,
    )
    # pursuit_run.py:2409-2426. Both the params and the targeting rules move, or
    # the smaller of the two would silently stay in force.
    params = design.params._replace(sensor_range=jnp.float32(_SENSOR_RANGE))
    rules = design.runtime.targeting_rules
    runtime = design.runtime._replace(
        targeting_rules=rules._replace(
            sensor_range=jnp.full_like(rules.sensor_range, jnp.float32(_SENSOR_RANGE))
        )
    )
    # pursuit_run.py: _assignment(batch), the learner trainable at slot 0.
    assignment = policy_actor_assignment(
        jnp.broadcast_to(jnp.asarray((0, 1), dtype=jnp.int32), (lanes, 2)),
        jnp.broadcast_to(jnp.asarray((0, 1), dtype=jnp.int32), (lanes, 2)),
        trainable=jnp.broadcast_to(jnp.asarray((True, False), dtype=jnp.bool_), (lanes, 2)),
        entity_count=2,
    )
    roles = skill_duel_roles(assignment)
    if int(np.asarray(roles.learner_policy_slot)[0]) != LEARNER_ROW or int(
        np.asarray(roles.target_policy_slot)[0]
    ) != OPPONENT_ROW:
        raise RuntimeError("skill_duel_roles disagrees with the flat arena's learner row")
    # pursuit_run.py: the scripted_flee_adaptive branch of run_pursuit_training.
    target = TargetMotionProgram(
        lane_modes=(TargetMode.FLEE_ADAPTIVE,),
        adaptive_gait_mix=_GAIT_MIX,
        adaptive_gait_period_ticks=_GAIT_PERIOD_TICKS,
        adaptive_arena_radius=_ARENA_RADIUS,
    )

    def reset_provider(keys: jax.Array):
        """Design reset, then the learner faces its target, as ``run_pursuit_training`` does."""

        state = design.reset_provider(keys)
        return _facing_reset(state, params) if stage.spawn_facing_target else state

    # pursuit_run.py: bind_pursuit_collector(...) for a design world. The line of
    # sight is None because catch_geometry is "horizontal", and path distance is None.
    built = bind_pursuit_collector(
        _flat_environment,
        params,
        runtime,
        assignment,
        target=target,
        stage=stage,
        learner_scope=pursuit_learner_action_scope(allow_jump=stage.allow_jump),
        line_of_sight_provider=None,
        path_distance=None,
        reset_provider=reset_provider,
        capability_provider=design.capability_provider,
    )
    description = {
        "schema": FLAT_ARENA_SCHEMA,
        "built_by": {
            "design": "arena.training.runs.pursuit_run._materialize_design",
            "reset": "arena.training.runs.pursuit_run._facing_reset",
            "sensor_override": "arena/training/runs/pursuit_run.py:2409-2426 (repeated)",
            "bind": "arena.training.contracts.pursuit.bind_pursuit_collector",
            "step": "experimental.neural_agent.tasks.flat_arena.FlatPursuitEnvironment.step_detailed (mirrors hytalegym rollout.py rollout_step)",
        },
        "batch": lanes,
        "seed": int(seed),
        "ticks": int(ticks),
        "design_id": FLAT_DESIGN_ID,
        "design": dict(design.metadata),
        "opponent": opponent,
        "target_controller": _OPPONENTS[opponent],
        "target_lane_mode": TargetMode.FLEE_ADAPTIVE.name,
        "adaptive_gait_mix": list(_GAIT_MIX),
        "adaptive_gait_period_ticks": _GAIT_PERIOD_TICKS,
        "arena_radius": _ARENA_RADIUS,
        "perception": perception,
        "view_sector_degrees": float(stage.view_sector_degrees),
        "spawn_facing_target": bool(stage.spawn_facing_target),
        "allow_jump": bool(stage.allow_jump),
        "catch_distance": float(stage.catch_distance),
        "maximum_turn_degrees": float(stage.maximum_turn_degrees),
        "sensor_range": _SENSOR_RANGE,
        "target_separation_range": list(_TARGET_SEPARATION_RANGE),
        "loadout": _LOADOUT,
        "opponent_loadout": _OPPONENT_LOADOUT,
        "learner_role_index": LEARNER_ROW,
        "opponent_role_index": OPPONENT_ROW,
    }
    return FlatArena(
        built=built,
        reset_keys=jax.random.split(jax.random.key(int(seed)), lanes),
        learner_role_index=LEARNER_ROW,
        opponent_role_index=OPPONENT_ROW,
        description=description,
    )


__all__ = [
    "FLAT_ARENA_SCHEMA",
    "FLAT_DESIGN_ID",
    "FlatArena",
    "FlatPursuitEnvironment",
    "LEARNER_ROW",
    "OPPONENT_ROW",
    "build_flat_arena",
    "factors_are_legal",
    "forced_factors",
    "target_choice_is_forced",
]
