"""J0: the scripted Level 2 instance, the baseline of ADR-001 D2 and phase P2.

J0 applies one fixed rule to the shared belief B_t and the executive state E_t.
It keeps no state, reads no privileged input and runs IN_GRAPH in pure JAX, so
it can sit inside ``HierarchicalAgentCore``. ADR-001 section 8 (P2) asks every
learned instance to beat this rule under the same interface.

Rule for lane b, with ticks_b = features[b, ticks_since_damage] * TICKS_SATURATION
and known_b = feature_known[b, ticks_since_damage]::

    evading_b   = known_b and ticks_b < 40 and E_t holds a valid evader goal
    chosen_b    = highest-priority valid evader goal   if evading_b
                = highest-priority valid goal          elif E_t holds a valid goal
                = idle: skill 0, goal_slot -1          otherwise
    terminate_b = chosen_skill_b != current_skill_b    (False means continue)

Control of the chosen goal. The skill owns it (SYSTEM_ONE.md section 8.12,
item 1), so J0 gathers the chosen skill's row of its ``SkillControlTable``::

    (minimum_commitment_seconds, maximum_duration_seconds,
     emergency_interrupt_mask) = control_for(control, chosen_skill_b)

The default table, ``default_control_table(skill_ids)``, holds the contracts
that ``skill_library`` declares: evader 2.0 s / 10 s / {DAMAGE_RECEIVED,
TARGET_DIED, NPC_STUCK}; pursuit and any unknown id 0.5 s / 10 s / the same
events; idle 0 s / +inf / no events. Until 2026-10-17 J0 applied the constants
below itself (2.0 s if evading_b, 0.5 s for another goal, idle's control when
no goal is valid). The two rules give different control in three cases only:
an evader goal chosen as the highest-priority goal outside the damage window
(0.5 s before, 2.0 s now), an executive goal that names idle (0.5 s / 10 s /
the three events before, idle's control now), and an evader whose id is not
``"evader"`` under the default table (2.0 s before when evading, the contract
default 0.5 s now). The composition executive reaches none of them: its
pursuit goal has priority 1.0 and its evader goal 0.5.

``ExecutiveArrays`` carries no planning horizon, so the
``GoalSpec.horizon_seconds`` value is not used here.

Units. ``belief/core.py`` publishes the damage counter as
``counter / TICKS_SATURATION``. The counter counts ticks and saturates at
``TICKS_SATURATION``, so the published value lies in [0, 1]. J0 multiplies the
value by ``TICKS_SATURATION`` before the 40-tick comparison. Without that step every
known timer would read below 40, and every lane with an evader goal would evade.
Scaling by a power of two is exact in float32, so the counter is recovered exactly.
"""

from __future__ import annotations

from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from adk.architecture.components import GoalDecision
from adk.architecture.decision_context import DecisionContext, DecisionEvent
from adk.architecture.hierarchy import GoalRuntimeContext
from adk.architecture.inputs import ActorPolicyInput
from experimental.neural_agent.contracts import (
    BELIEF_FEATURE_COUNT,
    MAX_GOALS,
    PARAM_WIDTH,
    SKILL_IDLE,
    TICKS_SATURATION,
    ComputeBudget,
    ExecutiveArrays,
    ExecutiveState,
    GoalPayload,
    JevMode,
    SharedBeliefState,
    SkillControlTable,
    _require_control_table,
    control_for,
    default_control_table,
    executive_arrays,
    feature_index,
    make_goal,
)

# Id of the evader skill. When the catalog has no such id, one id containing "evad" is used.
EVADER_SKILL_ID: str = "evader"
# Evasion applies while ticks_since_damage, in ticks, is strictly below this value.
EVADE_DAMAGE_WINDOW_TICKS: float = 40.0
# The goal control below belongs to the skill contracts since 2026-10-17
# (SYSTEM_ONE.md section 8.12, item 1). J0 no longer applies these constants: it
# reads the chosen skill's row of its SkillControlTable. They keep the values that
# the contracts declare and that FORMULAS (D9) cites, and test_na_jev_core.py pins
# default_control_table(("idle", "pursuit", "evader")) to them row by row.
# Minimum commitment that the evader's contract declares, in seconds.
EVADE_MINIMUM_COMMITMENT_SECONDS: float = 2.0
# Minimum commitment that the pursuit contract (and the contract default) declares, in seconds.
GOAL_MINIMUM_COMMITMENT_SECONDS: float = 0.5
# Planning horizon used for every goal, in seconds. E_t carries no horizon.
GOAL_HORIZON_SECONDS: float = 10.0
# Upper cap on the maximum duration of a goal, in seconds.
MAXIMUM_DURATION_CAP_SECONDS: float = 30.0
# Maximum duration of a chosen goal: min(horizon, cap) = 10.0 s.
MAXIMUM_DURATION_SECONDS: float = min(GOAL_HORIZON_SECONDS, MAXIMUM_DURATION_CAP_SECONDS)
# Events that bypass the minimum commitment of a pursuit or evader goal.
EMERGENCY_EVENTS: tuple[DecisionEvent, ...] = (
    DecisionEvent.DAMAGE_RECEIVED,
    DecisionEvent.TARGET_DIED,
    DecisionEvent.NPC_STUCK,
)

_IDLE_INDEX: int = 0
_EMPTY_CATALOG_STATE = ExecutiveState(
    version=0,
    issued_tick=0,
    goals=(),
    budget=ComputeBudget(),
    facts=(),
)


class ScriptedJev:
    """J0: one fixed rule over B_t and E_t. IN_GRAPH, stateless, deterministic.

    Args:
        skill_ids: The skill catalog in dispatcher order. Index 0 must be
            ``SKILL_IDLE``. The catalog must name one evader: the id
            ``"evader"``, or else exactly one id that contains ``"evad"``.
        control: Keyword only. The goal control of each catalog skill, one row
            per id. Default ``default_control_table(skill_ids)``; pass
            ``SkillRegistry.control_table()`` to use the registry's contracts.

    Raises:
        TypeError: If ``skill_ids`` is one string, or holds a non-string id, or
            ``control`` is not a SkillControlTable.
        ValueError: If the catalog is empty, repeats an id, does not start with
            ``"idle"``, or names no evader or more than one, or ``control`` does
            not have one row per catalog id.
    """

    name: str = "J0"
    mode: JevMode = JevMode.IN_GRAPH

    def __init__(
        self, skill_ids: Sequence[str], *, control: SkillControlTable | None = None
    ) -> None:
        if isinstance(skill_ids, (str, bytes)):
            raise TypeError("skill_ids must be a sequence of skill ids, not one string")
        catalog = tuple(skill_ids)
        # The contract's own check of the catalog: types, non-empty ids, no repeats.
        executive_arrays(_EMPTY_CATALOG_STATE, catalog, 1)
        if not catalog or catalog[0] != SKILL_IDLE:
            raise ValueError(
                f"ScriptedJev needs {SKILL_IDLE!r} at catalog index 0 (contracts.SKILL_IDLE)"
            )
        self._evader_index = _evader_index(catalog)
        self._ticks_column = feature_index("ticks_since_damage")
        self._control = _require_control_table(
            default_control_table(catalog) if control is None else control, len(catalog)
        )

    def initial_state(self, batch: int) -> tuple[()]:
        """Return the empty state. J0 keeps none, so the batch has no lane leaves.

        Args:
            batch: Number of lanes B, a positive int.

        Returns:
            The empty tuple ``()``.

        Raises:
            TypeError: If ``batch`` is not an integer.
            ValueError: If ``batch`` is not positive.
        """

        if isinstance(batch, bool) or not isinstance(batch, (int, np.integer)):
            raise TypeError(f"batch must be an integer, got {type(batch).__name__}")
        if batch < 1:
            raise ValueError(f"batch must be positive, got {batch}")
        return ()

    def __call__(
        self,
        state: Any,
        belief: SharedBeliefState,
        current_goal: GoalPayload,
        actor_input: ActorPolicyInput,
        key: jax.Array,
        *,
        decision_context: DecisionContext | None,
        goal_runtime: GoalRuntimeContext | None,
        executive: ExecutiveArrays,
    ) -> GoalDecision:
        """Choose the goal for every lane by the J0 rule (see the module docstring).

        Args:
            state: The empty instance state. It is returned unchanged.
            belief: The shared belief B_t. Only ``features`` float32[B, 10] and
                ``feature_known`` bool[B, 10] are read. Column ``ticks_since_damage``
                holds ``counter / TICKS_SATURATION``, as the belief publishes it.
            current_goal: The goal in force, a GoalPayload batched [B].
            actor_input: Not read. J0 uses no observation.
            key: Not read. J0 is deterministic.
            decision_context: Not read.
            goal_runtime: Not read.
            executive: The lowered executive state E_t.

        Returns:
            A GoalDecision. ``goal`` is a ManagedGoal whose GoalPayload has
            ``skill_index`` int32[B], ``params`` float32[B, 4],
            ``executive_version`` int32[B] and ``goal_slot`` int32[B], which is
            -1 for idle. Its control is the chosen skill's row of the control
            table. ``terminate`` is bool[B]. ``diagnostics`` is
            ``{"chosen_slot": int32[B], "evading": bool[B]}``.

        Raises:
            TypeError: If ``belief``, ``executive`` or ``current_goal`` has the wrong type.
            ValueError: If an array has a shape that J0 cannot index.
        """

        del actor_input, key, decision_context, goal_runtime
        lanes = _check_inputs(belief, current_goal, executive)
        rows = jnp.arange(lanes)
        # The belief publishes counter / TICKS_SATURATION. Multiply back to ticks.
        ticks = belief.features[:, self._ticks_column] * TICKS_SATURATION
        ticks_known = belief.feature_known[:, self._ticks_column]
        valid = executive.goal_valid
        priority = executive.priority
        is_evader = valid & (executive.skill_index == self._evader_index)
        evading = ticks_known & (ticks < EVADE_DAMAGE_WINDOW_TICKS) & jnp.any(is_evader, axis=1)

        evader_slot = jnp.argmax(jnp.where(is_evader, priority, -jnp.inf), axis=1)
        best_slot = jnp.argmax(jnp.where(valid, priority, -jnp.inf), axis=1)
        slot = jnp.where(evading, evader_slot, best_slot).astype(jnp.int32)
        has_goal = evading | jnp.any(valid, axis=1)

        chosen_skill = jnp.where(
            has_goal, executive.skill_index[rows, slot], _IDLE_INDEX
        ).astype(jnp.int32)
        chosen_params = jnp.where(has_goal[:, None], executive.params[rows, slot], 0.0)
        chosen_slot = jnp.where(has_goal, slot, -1).astype(jnp.int32)
        minimum, maximum, mask = control_for(self._control, chosen_skill)

        goal = make_goal(
            chosen_skill,
            chosen_params,
            executive.version,
            chosen_slot,
            minimum_commitment_seconds=minimum,
            maximum_duration_seconds=maximum,
            emergency_interrupt_mask=mask,
        )
        return GoalDecision(
            state=state,
            goal=goal,
            terminate=chosen_skill != current_goal.skill_index,
            diagnostics={"chosen_slot": chosen_slot, "evading": evading},
        )


def _evader_index(catalog: tuple[str, ...]) -> int:
    """Return the catalog index of the evader: the id ``"evader"``, else the single id containing ``"evad"``."""

    if EVADER_SKILL_ID in catalog:
        return catalog.index(EVADER_SKILL_ID)
    candidates = [index for index, skill_id in enumerate(catalog) if "evad" in skill_id]
    if len(candidates) != 1:
        raise ValueError(
            f"ScriptedJev needs one evader skill: name it {EVADER_SKILL_ID!r}, or give "
            f"exactly one id that contains 'evad'; found {len(candidates)} candidates"
        )
    return candidates[0]


def _check_inputs(belief: Any, current_goal: Any, executive: Any) -> int:
    """Check the types and shapes that J0 indexes, and return the lane count B."""

    if not isinstance(belief, SharedBeliefState):
        raise TypeError("belief must be a SharedBeliefState")
    if not isinstance(executive, ExecutiveArrays):
        raise TypeError("executive must be an ExecutiveArrays")
    if not isinstance(current_goal, GoalPayload):
        raise TypeError("current_goal must be a GoalPayload")
    features = tuple(jnp.shape(belief.features))
    if len(features) != 2 or features[1] != BELIEF_FEATURE_COUNT:
        raise ValueError(
            f"belief.features must have shape (B, {BELIEF_FEATURE_COUNT}), got {features}"
        )
    lanes = features[0]
    _expect_shape(belief.feature_known, features, "belief.feature_known")
    _expect_shape(executive.version, (lanes,), "executive.version")
    _expect_shape(executive.skill_index, (lanes, MAX_GOALS), "executive.skill_index")
    _expect_shape(executive.priority, (lanes, MAX_GOALS), "executive.priority")
    _expect_shape(executive.goal_valid, (lanes, MAX_GOALS), "executive.goal_valid")
    _expect_shape(executive.params, (lanes, MAX_GOALS, PARAM_WIDTH), "executive.params")
    _expect_shape(current_goal.skill_index, (lanes,), "current_goal.skill_index")
    return lanes


def _expect_shape(value: Any, shape: tuple[int, ...], name: str) -> None:
    """Raise ValueError unless ``value`` has exactly ``shape``."""

    actual = tuple(jnp.shape(value))
    if actual != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {actual}")


__all__ = [
    "EMERGENCY_EVENTS",
    "EVADE_DAMAGE_WINDOW_TICKS",
    "EVADE_MINIMUM_COMMITMENT_SECONDS",
    "EVADER_SKILL_ID",
    "GOAL_HORIZON_SECONDS",
    "GOAL_MINIMUM_COMMITMENT_SECONDS",
    "MAXIMUM_DURATION_CAP_SECONDS",
    "MAXIMUM_DURATION_SECONDS",
    "ScriptedJev",
]
