"""SkillDispatcher: the ADK ``LowLevelActor`` that runs the active skill.

The core calls ``dispatcher(state, belief, goal_payload, actor_input, key)`` once per
decision step and reads an ``ActorDecision`` (``hierarchy.py``, actor call site).
The dispatcher maps the goal's ``skill_index`` to one skill of the registry.

Reset rule (ADR-001 D6, skill-local state resets on entry):

    switched_b = (skill_index_b != active_skill_b)
    skill_state_b <- initial_state(skill_index_b)   if switched_b
    ticks_in_skill_b <- 0                           if switched_b

The shared belief is never reset here. Only the episode boundary resets it,
and that happens in the hierarchy core, not in the dispatcher.

Selection. Every skill runs on the whole batch. ``jnp.where`` then keeps, for
each lane, the factors and the new state of its active skill. The output is
therefore a static computation that ``jax.jit`` accepts with a static registry.
A skill's concrete checks would otherwise refuse lanes that it does not serve,
so each skill sees the real mask on its own lanes and an all-True mask elsewhere.

Fail closed. A concrete call with ``skill_index`` outside the catalog raises
``ValueError``. Under ``jax.jit`` such a lane runs the idle skill, and
``diagnostics["in_catalog"]`` is False. A concrete call that emits a masked
factor raises ``ValueError``. The traced case is reported in
``diagnostics["legal"]``.

Not implemented here: the outcome report on exit (ADR-001 section 5, seam 1).
``contracts.GoalReport`` exists, but the dispatcher returns only an ``ActorDecision``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from adk.architecture.components import ActorDecision
from adk.architecture.inputs import ActorPolicyInput
from experimental.neural_agent.contracts import (
    ACTION_HEADS,
    SKILL_STATE_WIDTH,
    GoalPayload,
    SharedBeliefState,
)
from experimental.neural_agent.skill_library._common import (
    factors_legal,
    require_concrete_true,
    require_dtype,
    require_shape,
    validate_batch,
    validate_skill_inputs,
)
from experimental.neural_agent.skill_library.library import SkillRegistry


class DispatcherState(NamedTuple):
    """Carried dispatcher state, one row per lane.

    Shapes, with B lanes:
        skill_state: float32[B, SKILL_STATE_WIDTH], state of the active skill.
        active_skill: int32[B], catalog index of the skill that ran last.
        ticks_in_skill: int32[B], decisions completed in the current activation.
    """

    skill_state: jax.Array
    active_skill: jax.Array
    ticks_in_skill: jax.Array


class SkillDispatcher:
    """Map a ``GoalPayload`` to one skill of ``registry`` per lane.

    Implements the ADK ``LowLevelActor`` call. Pass the instance to
    ``HierarchicalAgentCore(actor=...)``.
    """

    def __init__(self, registry: SkillRegistry) -> None:
        """Bind the catalog.

        Raises:
            TypeError: If ``registry`` is not a SkillRegistry.
        """

        if not isinstance(registry, SkillRegistry):
            raise TypeError("registry must be a SkillRegistry")
        self.registry = registry

    def initial_state(self, batch: int) -> DispatcherState:
        """Return the reset dispatcher state for ``batch`` lanes.

        The state starts on the idle skill (index 0) with zero ticks. Use it as
        ``HierarchicalCarry.actor_state``.

        Args:
            batch: Number of lanes B, a positive int.
        """

        lanes = validate_batch(batch)
        return DispatcherState(
            skill_state=self.registry.skills[0].initial_state(lanes),
            active_skill=jnp.zeros((lanes,), dtype=jnp.int32),
            ticks_in_skill=jnp.zeros((lanes,), dtype=jnp.int32),
        )

    def __call__(
        self,
        state: DispatcherState,
        belief: SharedBeliefState,
        goal_payload: GoalPayload,
        actor_input: ActorPolicyInput,
        key: jax.Array,
        /,
    ) -> ActorDecision:
        """Run the goal's skill for one decision step.

        Args:
            state: DispatcherState from the previous step, or ``initial_state``.
            belief: The shared belief B_t.
            goal_payload: GoalPayload. ``skill_index`` is int32[B] and ``params``
                is float32[B, 4].
            actor_input: The legal ActorPolicyInput. ``JaxTrainingInput`` is refused.
            key: A JAX PRNG key, split across the skills.

        Returns:
            ActorDecision. ``state`` is the new DispatcherState,
            ``action_factors`` is int32[B, 12], ``logits`` is None,
            ``behavior_log_probability`` and ``value`` are zeros float32[B],
            ``terminate_goal`` is zeros bool[B]. ``diagnostics`` holds
            ``active_skill`` int32[B], ``switched`` bool[B],
            ``ticks_in_skill`` int32[B] (the count before this decision), and
            ``in_catalog`` and ``legal`` bool[B].

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If the shapes are wrong, a concrete ``skill_index`` is
                outside the catalog, or a concrete factor is masked.
        """

        if not isinstance(state, DispatcherState):
            raise TypeError("state must be a DispatcherState")
        if not isinstance(goal_payload, GoalPayload):
            raise TypeError("goal_payload must be a GoalPayload")
        lanes = validate_skill_inputs(
            state.skill_state,
            belief,
            actor_input,
            goal_payload.params,
            skill_id="dispatcher",
        )
        skills = self.registry.skills
        count = len(skills)
        index = jnp.asarray(goal_payload.skill_index)
        require_dtype(index, jnp.int32, "dispatcher: goal_payload.skill_index")
        require_shape(index, (lanes,), "dispatcher: goal_payload.skill_index")
        in_catalog = (index >= 0) & (index < count)
        require_concrete_true(
            in_catalog,
            f"dispatcher: skill_index must lie in [0, {count}) for the catalog",
        )
        active = jnp.where(in_catalog, index, 0).astype(jnp.int32)
        switched = active != state.active_skill
        lane_ids = jnp.arange(lanes, dtype=jnp.int32)
        entry_states = jnp.stack([skill.initial_state(lanes) for skill in skills])
        entered = entry_states[active, lane_ids]
        skill_in = jnp.where(switched[:, None], entered, state.skill_state)
        ticks_before = jnp.where(switched, 0, state.ticks_in_skill).astype(jnp.int32)

        keys = jax.random.split(key, count)
        factors = jnp.zeros((lanes, ACTION_HEADS), dtype=jnp.int32)
        next_skill_state = jnp.zeros((lanes, SKILL_STATE_WIDTH), dtype=jnp.float32)
        for position, skill in enumerate(skills):
            chosen = active == position
            lane_input = actor_input._replace(
                action_mask=jnp.where(chosen[:, None], actor_input.action_mask, True)
            )
            skill_state, skill_factors = skill(
                skill_in, belief, lane_input, goal_payload.params, keys[position]
            )
            next_skill_state = jnp.where(chosen[:, None], skill_state, next_skill_state)
            factors = jnp.where(chosen[:, None], skill_factors, factors)

        legal = factors_legal(factors, actor_input.action_mask)
        require_concrete_true(
            legal,
            "dispatcher: the active skill emitted a factor the runtime mask forbids",
        )
        zeros = jnp.zeros((lanes,), dtype=jnp.float32)
        return ActorDecision(
            state=DispatcherState(
                skill_state=next_skill_state,
                active_skill=active,
                ticks_in_skill=ticks_before + 1,
            ),
            action_factors=factors,
            logits=None,
            behavior_log_probability=zeros,
            value=zeros,
            terminate_goal=jnp.zeros((lanes,), dtype=jnp.bool_),
            diagnostics={
                "active_skill": active,
                "switched": switched,
                "ticks_in_skill": ticks_before,
                "in_catalog": in_catalog,
                "legal": legal,
            },
        )


__all__ = ["DispatcherState", "SkillDispatcher"]
