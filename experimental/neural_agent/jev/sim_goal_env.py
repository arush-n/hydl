"""SimGoalDecisionEnv: J1 goal decisions over a real arena, for semi-MDP PPO.

``train_j1_ppo.GoalDecisionEnv`` asks for one environment step per goal
decision. This module implements that protocol over the composition stack
of ``eval/run_composition.py``: the arena of ``tasks/arenas.py`` (default
``combat_open_flat``), the shared belief, the skill dispatcher, and
``eval/arms.py::HoldGoalManager`` in the core's manager slot, which is the
``j_lm`` arm's composition. The PPO policy is the host decider. ADK's core
calls its manager inside ``jax.lax.cond`` (``adk/architecture/hierarchy.py``),
so the policy never runs there. ``step`` writes the chosen goal into the
carried hierarchy state between core calls, as ``session.run_session`` writes a
host goal: ``carry.goal`` is the goal, ``carry.goal_age`` is 0 and
``goal_valid`` is True. HoldGoalManager never requests a switch and the
dispatcher never requests termination, so the goal stays in force for the
whole decision. Its 10 s maximum duration is far longer than a decision.

Unlike ``session.run_session``, this is not a Python loop. A decision is a
``jax.lax.while_loop`` over ticks, and each tick is one body of the compiled
collector (``adk/runtime/hierarchical.py`` ``collect_hierarchical``): actor
input, core, ``step_with_boundary``, then the decision context. ``reset`` and
``step`` are therefore pure JAX and trace under ``jax.jit``, as the protocol
requires, and ``train_j1_ppo`` compiles the whole update with the arena inside.

Decision interval. A decision taken before tick t ends after its k-th tick,
at the first k >= 1 where one of these holds for the lane:

* k = ``cadence_ticks``, the host cadence (default 11 ticks, the
  ``SessionConfig.host_cadence_ticks`` default, about 0.5 s at 0.045 s per tick);
* the arena's ``EpisodeBoundary.done`` after that tick (a death, a world
  error, or the episode limit);
* an emergency event at tick t + k: ``valid & any(active_events &
  emergency_interrupt_mask)``, the core's ``emergency_interrupt`` test, read
  from the decision context that the core will read at tick t + k and from the
  goal in force. J0's mask is used, so only DAMAGE_RECEIVED fires on the JAX
  scenes, and an idle goal has no emergency events.

A lane whose decision has ended waits while the other lanes finish theirs.
Its environment, context and hierarchy rows are not advanced
(``select_environment_rows``). The arena still steps the whole batch, and
those rows are discarded.

Reward. With r_i the arena's ``transition.reward`` on tick i of the decision::

    R = sum_{i < k} gamma ** i * r_i,      gamma = train_j1_ppo.GAMMA = 0.99

This is the in-interval return in the ``train_j1_ppo`` module docstring. Source
of r_i: the native combat reward of the hytalegym Arsenal environment. Read on
2026-10-17 and checked against the built arena's ``combat_params``::

    r_i = 2.0 * target_damage - 1.0 * agent_damage
          + 50.0 * [completion] - 20.0 * [death],     and 0 on an invalid tick

``target_damage`` is the damage the agent dealt to opponents on that tick.
``agent_damage`` is the damage it received. ``completion`` is awarded once,
when every opponent's health is at or below 0. ``death`` is awarded once, when
the agent's health is at or below 0. Sources: ``compose_reward`` in
``HytaleRL/hytalegym/hytalegym/jax/combat/types.py``; ``completion`` and
``death`` in ``jax/combat/arsenal/runtime.py`` (``_resolve_damage``); the
scales are the ``reward`` block of
``rulesets/hytale_0_5_7/kweebec_razorleaf_vs_trork_brawler_v9.json``.
``tasks/arenas.py`` sets no shaping, so ``BuiltAgent`` adds nothing.

Episode boundary. ``done`` is ``EpisodeBoundary.done``, the signal that
``collect_hierarchical`` and ``eval/eager_loop.py`` autoreset on. ``step`` does
not reset a finished lane; ``train_j1_ppo.collect_decisions`` replaces its
rows with ``reset`` rows. Limitation: ``done`` includes the episode limit
(``truncated``), and ``decision_advantages`` treats every ``done`` as
terminal, so a time-limit end is not bootstrapped. The protocol has no
truncation channel.

Observation. The observation for a decision is the input that the manager
slot receives at that tick:

* ``belief`` comes from SharedBelief, applied to that tick's actor input with
  the carried belief state, the previous action and the core's clock rule.
  SharedBelief reads neither its goal nor its key (``belief/core.py``), so this
  is the belief that the core computes at that tick;
* ``executive`` is ``carry.manager_state``, the lowered E_t
  (``contracts.executive_arrays``) that HoldGoalManager carries;
* ``current_goal`` is the payload in force: the previous decision's goal, or
  idle after a reset;
* ``mask`` is ``network.j1_allowed_skills``.

These are the records that ``NetworkJev`` reads in the manager slot, and the
trained J1Params load through ``eval/arms.py::load_j1_params``. No test checks
that the learned choice carries over to ``NetworkJev``. That instance takes the
argmax, decides on the core's 0.5 s cadence, and switches only when its replace
head fires. PPO here does not train the replace head.

Goal. ``scripted.py`` (J0) builds the goal with its constants: a skill other
than idle gets a minimum commitment of ``GOAL_MINIMUM_COMMITMENT_SECONDS``
(0.5 s), a maximum duration of ``MAXIMUM_DURATION_SECONDS`` (10 s) and the
mask ``EMERGENCY_EVENTS``. Idle gets 0 s, no limit and an empty mask. The
payload takes the matching E_t slot (``network._matching_slot``), that slot's
parameters (zeros when no slot matches) and ``E_t.version``. J0's 2 s
evade commitment is part of J0's damage rule and is not used here. Here no
commitment can change a run: the host replaces the goal at every decision, and
the core never switches a goal within one.

PRNG. The core and the arena are batched programs that take one key per tick,
as in ``collect_hierarchical``. The ticks of a decision draw their keys from
``keys[0]``, and the other lane keys are not used.
"""

from __future__ import annotations

import numbers
from typing import Any

import jax
import jax.numpy as jnp

from adk.architecture.hierarchy import HierarchicalAgentCore, HierarchicalCoreState
from adk.architecture.inputs import actor_policy_input
from adk.runtime.env_adapter import step_with_boundary
from adk.runtime.hierarchical import (
    HierarchicalActorInputFn,
    HierarchicalContextFn,
    HierarchicalLoopState,
    jax_hierarchical_context,
)
from adk.runtime.loop import unknown_decision_context
from adk.runtime.tree import select_environment_rows
from experimental.neural_agent.contracts import ExecutiveArrays, event_mask, make_goal
from experimental.neural_agent.jev.network import (
    IDLE_SKILL_INDEX,
    _matching_slot,
    j1_allowed_skills,
)
from experimental.neural_agent.jev.scripted import (
    EMERGENCY_EVENTS,
    GOAL_MINIMUM_COMMITMENT_SECONDS,
    MAXIMUM_DURATION_SECONDS,
)
from experimental.neural_agent.jev.train_j1_ppo import GAMMA
from experimental.neural_agent.session import SessionConfig

#: The arena that :func:`build_sim_goal_env` builds unless told otherwise. On
#: this arena the target is visible and DAMAGE_RECEIVED fires (``tasks/arenas.py``).
DEFAULT_ARENA: str = "combat_open_flat"
#: Ticks per decision when no event ends it first: the session's host cadence.
DEFAULT_CADENCE_TICKS: int = SessionConfig(steps=1).host_cadence_ticks
# SharedBelief ignores its key (belief/core.py), so the observation's belief
# readout passes a fixed one.
_BELIEF_KEY = jax.random.key(0)


def _whole(value: Any, name: str) -> int:
    """Return ``value`` as an int of at least 1. Bools raise TypeError."""

    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}")
    return int(value)


class SimGoalDecisionEnv:
    """A ``train_j1_ppo.GoalDecisionEnv`` over an arena and a HoldGoalManager core.

    See the module docstring for the decision interval, the reward and the
    observation. The state is a ``HierarchicalLoopState`` whose leaves all
    have the leading lane axis B.

    Args:
        built: The arena environment, with ``batch``, ``reset(keys)`` and
            ``step_detailed`` (``tasks/arenas.py`` ``BuiltArena.built``).
        core: A ``HierarchicalAgentCore`` whose manager holds the goal in
            force (``HoldGoalManager``).
        initial_state: The core's reset state for ``built.batch`` lanes. Its
            ``carry.manager_state`` must be the lowered E_t, ``ExecutiveArrays``.
        num_skills: Catalog size K.
        cadence_ticks: Longest decision, in ticks.
        context_fn: The collector's map from step info to ``DecisionContext``.
        actor_input_fn: The collector's map from environment state to
            ``ActorPolicyInput``.

    Attributes:
        num_skills: K.
        gamma: Per-tick discount, ``train_j1_ppo.GAMMA``.
        batch: Lanes B, ``built.batch``.
        cadence_ticks: Longest decision, in ticks.
        built: The arena environment.
        core: The hierarchy core.

    Raises:
        TypeError: If ``core`` or ``initial_state`` has the wrong type, or the
            manager state is not ``ExecutiveArrays``.
        ValueError: If a count is not positive, or the state's lane count is
            not ``built.batch``.
    """

    gamma: float = GAMMA

    def __init__(
        self,
        built: Any,
        core: HierarchicalAgentCore,
        initial_state: HierarchicalCoreState,
        *,
        num_skills: int,
        cadence_ticks: int = DEFAULT_CADENCE_TICKS,
        context_fn: HierarchicalContextFn = jax_hierarchical_context,
        actor_input_fn: HierarchicalActorInputFn = actor_policy_input,
    ) -> None:
        if not isinstance(core, HierarchicalAgentCore):
            raise TypeError("core must be a HierarchicalAgentCore")
        if not isinstance(initial_state, HierarchicalCoreState):
            raise TypeError("initial_state must be a HierarchicalCoreState")
        if not isinstance(initial_state.carry.manager_state, ExecutiveArrays):
            raise TypeError(
                "initial_state.carry.manager_state must be ExecutiveArrays "
                "(the HoldGoalManager state)"
            )
        self.batch = _whole(getattr(built, "batch", None), "built.batch")
        lanes = jnp.shape(initial_state.carry.previous_action)[0]
        if lanes != self.batch:
            raise ValueError(
                f"initial_state has {lanes} lanes; built.batch is {self.batch}"
            )
        self.num_skills = _whole(num_skills, "num_skills")
        self.cadence_ticks = _whole(cadence_ticks, "cadence_ticks")
        self.built = built
        self.core = core
        self._initial = initial_state
        self._context_fn = context_fn
        self._actor_input_fn = actor_input_fn

    def reset(self, keys: jax.Array) -> tuple[HierarchicalLoopState, dict[str, Any]]:
        """Start one episode per lane (``GoalDecisionEnv.reset``).

        Args:
            keys: B PRNG keys, passed to ``built.reset``.

        Returns:
            ``(state, obs)``: the arena's reset rows, an unknown decision
            context and the core's reset state, and their observation.
        """

        if jnp.shape(keys)[0] != self.batch:
            raise ValueError(f"reset needs {self.batch} keys, got {jnp.shape(keys)[0]}")
        environment_state, policy_input = self.built.reset(keys)
        state = HierarchicalLoopState(
            environment_state=environment_state,
            policy_input=policy_input,
            decision_context=unknown_decision_context(self.batch),
            hierarchy_state=self._initial,
        )
        return state, self._observe(state)

    def step(
        self, state: HierarchicalLoopState, skill_index: jax.Array, keys: jax.Array
    ) -> tuple[HierarchicalLoopState, dict[str, Any], jax.Array, jax.Array, jax.Array]:
        """Write the chosen goal and run it to the end of the decision (``GoalDecisionEnv.step``).

        Args:
            state: The state from ``reset`` or the previous ``step``.
            skill_index: int32[B], the chosen skill.
            keys: B PRNG keys. The ticks draw from ``keys[0]``.

        Returns:
            ``(state, obs, reward float32[B], elapsed_ticks int32[B], done
            bool[B])`` with the reward and interval of the module docstring.
            ``elapsed_ticks`` is in [1, ``cadence_ticks``].
        """

        lanes = self.batch
        skill = jnp.asarray(skill_index, dtype=jnp.int32)
        if skill.shape != (lanes,):
            raise ValueError(f"skill_index must have shape ({lanes},), got {skill.shape}")
        hierarchy = state.hierarchy_state
        goal = self._goal(skill, hierarchy.carry.manager_state)
        hierarchy = hierarchy._replace(
            carry=hierarchy.carry._replace(
                goal=goal, goal_age=jnp.zeros((lanes,), dtype=jnp.float32)
            ),
            goal_valid=jnp.ones((lanes,), dtype=jnp.bool_),
        )
        step_keys = jax.random.split(keys[0], self.cadence_ticks)
        gamma = jnp.float32(self.gamma)

        def tick(carry):
            loop, active, elapsed, reward, discount, done, index = carry
            hierarchy_key, environment_key = jax.random.split(step_keys[index])
            actor_input = self._actor_input_fn(loop.environment_state, loop.policy_input)
            hierarchy_step = self.core(
                loop.hierarchy_state, actor_input, loop.decision_context, hierarchy_key
            )
            environment_step = step_with_boundary(
                self.built,
                loop.environment_state,
                hierarchy_step.actor_decision.action_factors,
                jax.random.split(environment_key, lanes),
            )
            next_actor_input = self._actor_input_fn(
                environment_step.state, environment_step.policy_input
            )
            next_context = self._context_fn(
                environment_step.info, next_actor_input, actor_input
            )
            advanced = HierarchicalLoopState(
                environment_state=environment_step.state,
                policy_input=environment_step.policy_input,
                decision_context=next_context,
                hierarchy_state=hierarchy_step.state,
            )
            loop = select_environment_rows(active, advanced, loop)
            tick_reward = jnp.asarray(environment_step.reward, dtype=jnp.float32)
            reward = reward + jnp.where(active, discount * tick_reward, 0.0)
            discount = jnp.where(active, discount * gamma, discount)
            elapsed = elapsed + active.astype(jnp.int32)
            ended = active & jnp.asarray(environment_step.boundary.done, dtype=jnp.bool_)
            in_force = hierarchy_step.state
            events = next_context.events.values & next_context.events.known
            emergency = in_force.goal_valid & jnp.any(
                events & in_force.carry.goal.control.emergency_interrupt_mask, axis=-1
            )
            active = active & ~ended & ~emergency & (elapsed < self.cadence_ticks)
            return loop, active, elapsed, reward, discount, done | ended, index + 1

        start = (
            state._replace(hierarchy_state=hierarchy),
            jnp.ones((lanes,), dtype=jnp.bool_),
            jnp.zeros((lanes,), dtype=jnp.int32),
            jnp.zeros((lanes,), dtype=jnp.float32),
            jnp.ones((lanes,), dtype=jnp.float32),
            jnp.zeros((lanes,), dtype=jnp.bool_),
            jnp.int32(0),
        )
        loop, _, elapsed, reward, _, done, _ = jax.lax.while_loop(
            lambda carry: jnp.any(carry[1]), tick, start
        )
        return loop, self._observe(loop), reward, elapsed, done

    def _goal(self, skill: jax.Array, executive: ExecutiveArrays) -> Any:
        """Return the ManagedGoal for the chosen skills, controlled as J0 controls a goal."""

        lanes = skill.shape[0]
        has_goal = skill != IDLE_SKILL_INDEX
        slot = _matching_slot(executive, skill)
        slot_params = executive.params[jnp.arange(lanes), jnp.maximum(slot, 0)]
        return make_goal(
            skill,
            jnp.where((slot >= 0)[:, None], slot_params, 0.0),
            executive.version,
            slot,
            minimum_commitment_seconds=jnp.where(
                has_goal, GOAL_MINIMUM_COMMITMENT_SECONDS, 0.0
            ),
            maximum_duration_seconds=jnp.where(has_goal, MAXIMUM_DURATION_SECONDS, jnp.inf),
            emergency_interrupt_mask=jnp.where(
                has_goal[:, None],
                event_mask(*EMERGENCY_EVENTS, batch=lanes),
                event_mask(batch=lanes),
            ),
        )

    def _observe(self, state: HierarchicalLoopState) -> dict[str, Any]:
        """Return the observation dict that the manager slot would read at the next tick."""

        hierarchy = state.hierarchy_state
        timing = state.decision_context.timing
        # The core's clock rule: unknown time is zero, negative time is clamped.
        delta = jnp.where(
            timing.delta_seconds_known, jnp.maximum(timing.delta_seconds, 0.0), 0.0
        )
        belief = self.core.belief(
            hierarchy.carry.belief_state,
            self._actor_input_fn(state.environment_state, state.policy_input),
            hierarchy.carry.previous_action,
            hierarchy.carry.goal.payload,
            delta,
            _BELIEF_KEY,
        ).belief
        executive = hierarchy.carry.manager_state
        return {
            "belief": belief,
            "executive": executive,
            "current_goal": hierarchy.carry.goal.payload,
            "mask": j1_allowed_skills(executive, num_skills=self.num_skills),
        }


def build_sim_goal_env(
    batch: int,
    ticks: int,
    *,
    arena: str = DEFAULT_ARENA,
    seed: int = 0,
    cadence_ticks: int = DEFAULT_CADENCE_TICKS,
) -> SimGoalDecisionEnv:
    """Build :class:`SimGoalDecisionEnv` on a named arena with the composition stack.

    ``eval/run_composition.py`` supplies the pieces: ``load_context`` (the
    skill registry and E_t of the composition task), ``load_arena`` and
    ``_compose``. The core is composed as the ``j_lm`` arm:
    HoldGoalManager, the skill dispatcher and the shared belief. The arm's
    language-model host decider is not used, because the PPO policy decides.

    Args:
        batch: Lanes B, a positive int.
        ticks: Episode limit of the arena, in ticks.
        arena: A name from ``tasks/arenas.py`` ``ARENA_NAMES``.
        seed: Seed of the arm context and of the arena's construction.
        cadence_ticks: Longest decision, in ticks.

    Returns:
        The environment, with ``num_skills`` equal to the catalog size.
    """

    from experimental.neural_agent.eval.arms import get_arm
    from experimental.neural_agent.eval.run_composition import (
        _compose,
        load_arena,
        load_context,
    )

    ctx, _ = load_context(batch, ticks, seed=seed)
    built = load_arena(arena, batch, seed, ticks).built
    composed = _compose(get_arm("j_lm"), ctx, built=built, ticks=ticks, episodes=1, seed=seed)
    return SimGoalDecisionEnv(
        built,
        composed.core,
        composed.initial,
        num_skills=len(ctx.skill_ids),
        cadence_ticks=cadence_ticks,
    )


__all__ = [
    "DEFAULT_ARENA",
    "DEFAULT_CADENCE_TICKS",
    "SimGoalDecisionEnv",
    "build_sim_goal_env",
]
