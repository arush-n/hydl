"""Comparison arms for the composition run.

An :class:`Arm` is a name and three factories. Each factory takes an
:class:`ArmContext` and returns the object that ``HierarchicalAgentCore`` takes
in its slot: the belief, the manager, or the actor. The sibling modules
(``belief``, ``skills``, ``jev``) are imported inside the factories, so this
module imports without them.

The arms follow the composition brief and
``experimental/hierarchical_planner/DESIGN.md`` section 6:

* ``flat``: the pursuit skill alone, with no hierarchy. Its manager always
  proposes the pursuit goal with terminate False.
* ``j0``: scripted rules (``ScriptedJev``) inside ``JevRunner``, with the
  executive bound.
* ``shuffled``: ``j0`` whose proposed skill index is passed through a random
  permutation of the catalogue at every decision. This is the honesty control.
* ``j1``: ``NetworkJev`` (``jev/network.py``) inside ``JevRunner``, wired like
  ``j0``. Without ``--j1-params`` its parameters are ``init_j1`` with the run
  seed, a random initialization, and the report labels the arm ``j1-untrained``
  (:func:`arm_label`). ``--j1-params PATH`` loads them from an ``.npz`` whose
  keys are the ``J1Params`` field names (:func:`load_j1_params`).
* ``j_lm``: a host-mode arm. ``LanguageModelJev`` (``jev/lm/instance.py``)
  decides between steps through :class:`LanguageModelHostDecider`, a
  ``session.HostDecider``, while :class:`HoldGoalManager` holds the in-graph
  manager slot. The ADK core calls its manager inside ``jax.lax.cond``, so this
  arm runs only through ``session.run_session`` (the eager runner). Its backend
  is ``FakeDecisionLM`` scripted to answer pursuit (``fake``, the default, no
  network) or ``OllamaQwenBackend`` with ``qwen3:4b`` (``ollama``).
* ``js1``: a host-mode arm wired like ``j_lm``. ``SystemOneJev``
  (``jev/system_one/instance.py``) over ``HashedBagTrunk`` decides through
  :class:`LanguageModelHostDecider`. Without ``--js1-heads`` its heads are
  ``init_heads`` with the run seed and the report labels the arm ``js1-untrained``;
  ``--js1-heads PATH`` loads them (``jev/system_one/train_supervised.load_heads``).
* ``jprog``: a host-mode arm wired like ``j_lm``. ``ProgramExecutor``
  (``jev/programs.py``, SYSTEM_ONE.md section 9.9) runs one hand-written behaviour
  program on every lane: ``NA_PROGRAM=J0`` (the default) or any other key of
  ``jev/programs.py`` ``PROGRAMS`` (SEQ, COMBAT, the combat styles COUNTER,
  PRESSURE and HIT_AND_RUN, the guerrilla program GUERRILLA and its candidates
  GUERRILLA_SHOT_FIRST and GUERRILLA_FACING), with the feedback rule's window ``NA_NO_REISSUE``
  seconds (default 0, off). It uses the
  registry's control table. The report labels it ``jprog-<program>``, plus
  ``-noreissue<w>s`` when the rule is on. In the eager runner's timed session,
  ``session.run_session`` binds its scratchpad to the executor, which then also
  writes one ``ActionRequest`` per lane per decision into the ``requests``
  section, checked against the registry's contracts, and reads the statuses
  back (SYSTEM_ONE.md section 9.10). The goal path still decides.

The JEV arms (``j0``, ``shuffled``, ``j1``, ``j_lm``) build their instance and
runner with the registry's control table (:func:`_control_table`), so each goal
carries the chosen skill's declared control (SYSTEM_ONE.md section 8.12, item 1).
``js1`` is not wired to it yet.

Every manager in this module returns ``diagnostics=()``. The core runs the
manager inside ``jax.lax.cond``, and both branches must have the same PyTree
structure, so an empty diagnostic structure is the only one that every arm can
share without knowing a sibling's diagnostics.
"""

from __future__ import annotations

import atexit
import json
import math
import numbers
import os
import sys
from pathlib import Path
from typing import Any, Callable, NamedTuple, Protocol, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from adk.architecture.components import GoalDecision
from adk.architecture.decision_context import DecisionEvent
from adk.architecture.hierarchy import ManagedGoal
from experimental.neural_agent.contracts import (
    PARAM_WIDTH,
    SKILL_IDLE,
    ExecutiveArrays,
    ExecutiveState,
    JevBudget,
    JevMode,
    JevStats,
    default_control_table,
    executive_arrays,
    idle_goal,
    make_goal,
)

#: Declared JEV latency budget of the scripted arms, in milliseconds. The
#: budget is set per deployment target (``JevBudget``). This value matches the
#: ``ComputeBudget`` default.
J0_LATENCY_BUDGET_MS: float = 20.0

#: Seed of the belief parameters. They are a random initialization, because this
#: run pins no trained belief checkpoint, and every arm shares them so that the
#: arms differ only in their managers.
BELIEF_PARAMS_SEED: int = 0

#: Skill that the flat arm runs on every tick.
PURSUIT_SKILL_ID: str = "pursuit"

#: Backends of the ``j_lm`` arm. ``fake`` is the default and needs no network.
LM_BACKENDS: tuple[str, ...] = ("fake", "ollama")

#: Ollama model tag of the ``ollama`` backend.
OLLAMA_MODEL: str = "qwen3:4b"

#: Environment variables of the ``jprog`` arm: the program name (``jev/programs.py`` PROGRAMS, default
#: ``J0``) and the feedback rule's window in seconds (default 0, off).
PROGRAM_ENV: str = "NA_PROGRAM"
NO_REISSUE_ENV: str = "NA_NO_REISSUE"

#: The fake backend's reply on every call: switch to pursuit. With
#: ``continue_current`` False, ``LanguageModelJev`` sets terminate only on a lane
#: whose skill is not pursuit yet, so a lane already pursuing keeps its goal.
_FAKE_PURSUIT_REPLY: dict[str, Any] = {
    "skill_id": PURSUIT_SKILL_ID,
    "params": [],
    "continue_current": False,
    "reason": "fake backend: always pursuit",
}


class ArmContext(NamedTuple):
    """Inputs that every arm factory may read. Build it with :func:`build_arm_context`.

    Attributes:
        skill_ids: Catalogue ids in dispatcher order. ``skill_ids[0]`` is ``"idle"``.
        registry: The sibling skill registry with ``.ids`` and ``.skills``. It is
            passed through to the actor factory unchanged, and the JEV arms take
            its ``control_table()`` (:func:`_control_table`).
        batch: Number of lanes B, a positive int.
        executive: The executive state E_t that the scripted arms run under.
        seed: The run seed. The ``j1`` arm initializes its parameters from it, and
            the ``j_lm`` decider keys its calls with it.
        lm_backend: The ``j_lm`` backend, one of :data:`LM_BACKENDS`.
        j1_params: Path of trained J1 parameters (``.npz``), or None for ``init_j1``.
        js1_heads: Path of trained J-S1 heads (``.npz``), or None for ``init_heads``.
    """

    skill_ids: tuple[str, ...]
    registry: Any
    batch: int
    executive: ExecutiveState
    seed: int = 0
    lm_backend: str = "fake"
    j1_params: Path | None = None
    js1_heads: Path | None = None


def build_arm_context(
    *,
    skill_ids: Any,
    registry: Any,
    batch: int,
    executive: ExecutiveState,
    seed: int = 0,
    lm_backend: str = "fake",
    j1_params: str | Path | None = None,
    js1_heads: str | Path | None = None,
) -> ArmContext:
    """Validate the arm inputs and return an :class:`ArmContext`.

    Args:
        skill_ids: Catalogue ids in dispatcher order. The first id must be ``"idle"``,
            because ``idle_goal`` selects index 0.
        registry: The skill registry, passed through unchanged.
        batch: Number of lanes B, a positive int.
        executive: The executive state E_t.
        seed: The run seed, a non-negative int.
        lm_backend: The ``j_lm`` backend, one of :data:`LM_BACKENDS`.
        j1_params: Path of trained J1 parameters, or None.
        js1_heads: Path of trained J-S1 heads, or None.

    Returns:
        The validated ArmContext.

    Raises:
        TypeError: If an id is not a string, or ``batch`` is not an integer, or
            ``executive`` is not an ExecutiveState.
        ValueError: If the catalogue is empty, repeats an id, or does not start
            with ``"idle"``, ``batch`` is not positive, ``seed`` is negative or
            ``lm_backend`` is unknown.
    """

    ids = tuple(skill_ids)
    if not ids:
        raise ValueError("skill_ids must not be empty")
    for position, skill_id in enumerate(ids):
        if not isinstance(skill_id, str):
            raise TypeError(f"skill_ids[{position}] must be a string")
        if not skill_id:
            raise ValueError(f"skill_ids[{position}] must be a non-empty string")
    if len(set(ids)) != len(ids):
        raise ValueError(f"skill_ids repeats an id: {ids}")
    if ids[0] != SKILL_IDLE:
        raise ValueError(
            f"skill_ids[0] must be {SKILL_IDLE!r} because idle_goal selects index 0; "
            f"got {ids[0]!r}"
        )
    if isinstance(batch, bool) or not isinstance(batch, numbers.Integral):
        raise TypeError(f"batch must be an integer, got {type(batch).__name__}")
    if batch < 1:
        raise ValueError(f"batch must be positive, got {batch}")
    if not isinstance(executive, ExecutiveState):
        raise TypeError("executive must be an ExecutiveState")
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed!r}")
    if lm_backend not in LM_BACKENDS:
        raise ValueError(f"lm_backend must be one of {LM_BACKENDS}, got {lm_backend!r}")
    return ArmContext(
        skill_ids=ids,
        registry=registry,
        batch=int(batch),
        executive=executive,
        seed=int(seed),
        lm_backend=lm_backend,
        j1_params=None if j1_params is None else Path(j1_params),
        js1_heads=None if js1_heads is None else Path(js1_heads),
    )


class ArmManager(Protocol):
    """The manager surface that the composition runner uses.

    The core calls ``__call__`` as an ADK ``TacticalManager``. The runner calls
    ``initial_state`` once, and ``jev_stats`` after the run.
    """

    def initial_state(self, batch: int) -> Any:
        """Return the reset manager state, with a leading lane axis B."""

    def __call__(
        self,
        state: Any,
        belief: Any,
        current_goal: Any,
        actor_input: Any,
        key: jax.Array,
        /,
        *,
        decision_context: Any = None,
        goal_runtime: Any = None,
    ) -> GoalDecision:
        """Return the GoalDecision for this decision, with ``diagnostics=()``."""

    def jev_stats(self) -> JevStats | None:
        """Return the JEV statistics of the run, or None when the arm has no JEV."""


class ConstantGoalManager:
    """Manager that always proposes one skill with terminate False. It is the flat arm's manager.

    Shapes, with B lanes: the manager state is int32[B] and is returned
    unchanged. The goal comes from ``make_goal`` with zero parameters, zero
    minimum commitment, unlimited duration and an empty interrupt mask.
    ``terminate`` is bool[B], all False. ``diagnostics`` is ``()``.

    Args:
        skill_index: Index of the skill to propose, a non-negative int.

    Raises:
        ValueError: If ``skill_index`` is negative or not an integer.
    """

    def __init__(self, skill_index: int) -> None:
        if (
            isinstance(skill_index, bool)
            or not isinstance(skill_index, numbers.Integral)
            or skill_index < 0
        ):
            raise ValueError(f"skill_index must be a non-negative int, got {skill_index!r}")
        self._skill_index = int(skill_index)

    def initial_state(self, batch: int) -> jax.Array:
        """Return the reset manager state, zeros of shape int32[batch]."""

        return jnp.zeros((batch,), dtype=jnp.int32)

    def __call__(
        self,
        state: jax.Array,
        belief: Any,
        current_goal: Any,
        actor_input: Any,
        key: jax.Array,
        /,
        *,
        decision_context: Any = None,
        goal_runtime: Any = None,
    ) -> GoalDecision:
        """Propose the constant skill for every lane. Inputs are ignored."""

        del belief, current_goal, actor_input, key, decision_context, goal_runtime
        lanes = state.shape[0]
        goal = make_goal(
            jnp.full((lanes,), self._skill_index, dtype=jnp.int32),
            jnp.zeros((lanes, PARAM_WIDTH), dtype=jnp.float32),
            jnp.zeros((lanes,), dtype=jnp.int32),
            jnp.full((lanes,), -1, dtype=jnp.int32),
            minimum_commitment_seconds=jnp.zeros((lanes,), dtype=jnp.float32),
            maximum_duration_seconds=jnp.full((lanes,), jnp.inf, dtype=jnp.float32),
            emergency_interrupt_mask=jnp.zeros(
                (lanes, len(DecisionEvent)), dtype=jnp.bool_
            ),
        )
        return GoalDecision(
            state=state,
            goal=goal,
            terminate=jnp.zeros((lanes,), dtype=jnp.bool_),
            diagnostics=(),
        )

    def jev_stats(self) -> None:
        """Return None. The flat arm has no JEV, so there are no JEV statistics."""

        return None


class ShuffledGoalManager:
    """Wrap a manager and pass each proposed skill index through a random permutation.

    On every call, a permutation ``perm`` of ``range(n_skills)`` is drawn from
    the second half of ``jax.random.split(key)``. A proposed index ``s >= 0``
    becomes ``perm[s]``, and -1 stays -1. Parameters, priority and commitment
    are kept, and ``diagnostics`` is ``()``. The core adopts a proposal only
    when it switches, so each switch carries a random catalogue skill. This is
    the control described in ``experimental/hierarchical_planner/DESIGN.md``
    section 6: if it matches the unshuffled arm, the goal content adds nothing.

    Args:
        inner: The manager to wrap. It must provide ``initial_state`` and ``jev_stats``.
        n_skills: Size of the catalogue, a positive int.

    Raises:
        ValueError: If ``n_skills`` is not a positive integer.
    """

    def __init__(self, inner: ArmManager, n_skills: int) -> None:
        if isinstance(n_skills, bool) or not isinstance(n_skills, numbers.Integral) or n_skills < 1:
            raise ValueError(f"n_skills must be a positive integer, got {n_skills!r}")
        self._inner = inner
        self._n_skills = int(n_skills)

    def initial_state(self, batch: int) -> Any:
        """Return the inner manager's reset state."""

        return self._inner.initial_state(batch)

    def __call__(
        self,
        state: Any,
        belief: Any,
        current_goal: Any,
        actor_input: Any,
        key: jax.Array,
        /,
        *,
        decision_context: Any = None,
        goal_runtime: Any = None,
    ) -> GoalDecision:
        """Run the inner manager, then permute its skill index for every lane."""

        inner_key, permutation_key = jax.random.split(key)
        decision = self._inner(
            state,
            belief,
            current_goal,
            actor_input,
            inner_key,
            decision_context=decision_context,
            goal_runtime=goal_runtime,
        )
        permutation = jax.random.permutation(permutation_key, self._n_skills)
        payload = decision.goal.payload
        proposed = payload.skill_index
        index = jnp.clip(proposed, 0, self._n_skills - 1)
        shuffled = jnp.where(proposed >= 0, permutation[index], -1).astype(jnp.int32)
        goal = ManagedGoal(
            payload=payload._replace(skill_index=shuffled),
            control=decision.goal.control,
        )
        return GoalDecision(
            state=decision.state,
            goal=goal,
            terminate=decision.terminate,
            diagnostics=(),
        )

    def jev_stats(self) -> JevStats | None:
        """Return the inner manager's JEV statistics."""

        return self._inner.jev_stats()


class JevArmManager:
    """Adapter from a ``JevRunner`` to the arm manager surface, with the executive bound.

    ``initial_state`` writes the executive into the runner's state with
    ``set_executive``. ``__call__`` runs the runner and drops its diagnostics,
    so the core's ``lax.cond`` branches keep the empty diagnostic structure.
    ``jev_stats`` returns the runner's :class:`JevStats`.

    Args:
        runner: A ``JevRunner`` from ``experimental.neural_agent.jev.runner``.
        executive: The ExecutiveState to bind at the start of each run.
    """

    def __init__(
        self,
        runner: Any,
        executive: ExecutiveState,
        *,
        measured: bool,
    ) -> None:
        self._runner = runner
        self._executive = executive
        self._measured = measured

    def initial_state(self, batch: int) -> Any:
        """Return the runner's reset state with the executive written into it."""

        return self._runner.set_executive(
            self._runner.initial_state(batch),
            self._executive,
        )

    def __call__(
        self,
        state: Any,
        belief: Any,
        current_goal: Any,
        actor_input: Any,
        key: jax.Array,
        /,
        *,
        decision_context: Any = None,
        goal_runtime: Any = None,
    ) -> GoalDecision:
        """Run the JEV for one decision and return its goal with ``diagnostics=()``."""

        decision = self._runner(
            state,
            belief,
            current_goal,
            actor_input,
            key,
            decision_context=decision_context,
            goal_runtime=goal_runtime,
        )
        return GoalDecision(
            state=decision.state,
            goal=decision.goal,
            terminate=decision.terminate,
            diagnostics=(),
        )

    def jev_stats(self) -> JevStats | None:
        """Return the runner's :class:`JevStats`, or None for an instance that is not measured.

        ``JevRunner`` counts eager calls only, and its latency is NaN for an
        IN_GRAPH instance. An IN_GRAPH instance inside a compiled collector is
        therefore not measured, and the arm reports None.
        """

        if not self._measured:
            return None
        return self._runner.stats()


class HoldGoalManager:
    """In-graph manager of a host-mode arm: it never changes the goal.

    The ADK core calls its manager inside ``jax.lax.cond``
    (``adk/architecture/hierarchy.py``), so a host-mode decision model cannot sit
    in that slot. This manager holds it. It re-wraps the payload in force as a
    ``ManagedGoal`` with the idle control (no commitment, unlimited duration,
    empty interrupt mask) and returns terminate False, so the core replaces a
    goal only when it must (invalid, timed out, or ended by the actor), and then
    with the same payload. ``diagnostics`` is ``()``.

    Its state is the lowered executive, ``ExecutiveArrays`` with leading axis B,
    returned unchanged. :class:`LanguageModelHostDecider` reads it from
    ``carry.manager_state``.

    Args:
        executive: The executive state E_t.
        skill_ids: The catalogue in dispatcher order.

    Raises:
        KeyError: If a goal of E_t names a skill outside the catalogue.
    """

    def __init__(self, executive: ExecutiveState, skill_ids: Sequence[str]) -> None:
        executive_arrays(executive, skill_ids, 1)
        self._executive = executive
        self._skill_ids = tuple(skill_ids)

    def initial_state(self, batch: int) -> ExecutiveArrays:
        """Return E_t lowered for ``batch`` lanes."""

        return executive_arrays(self._executive, self._skill_ids, batch)

    def __call__(
        self,
        state: ExecutiveArrays,
        belief: Any,
        current_goal: Any,
        actor_input: Any,
        key: jax.Array,
        /,
        *,
        decision_context: Any = None,
        goal_runtime: Any = None,
    ) -> GoalDecision:
        """Hold the payload in force for every lane."""

        del belief, actor_input, key, decision_context, goal_runtime
        lanes = jnp.shape(current_goal.skill_index)[0]
        return GoalDecision(
            state=state,
            goal=ManagedGoal(payload=current_goal, control=idle_goal(lanes).control),
            terminate=jnp.zeros((lanes,), dtype=jnp.bool_),
            diagnostics=(),
        )

    def jev_stats(self) -> None:
        """Return None. The host decider's statistics come from the session metrics."""

        return None


class LanguageModelHostDecider:
    """A ``session.HostDecider`` over a host-mode instance such as ``LanguageModelJev``.

    ``decide`` calls the instance once for the batch with B_t from the step just
    recorded (``transition.hierarchy_step.belief_update.belief``), and with the
    goal in force and the lowered E_t from the carried state (``carry.goal`` and
    ``carry.manager_state``). ``actor_input``, ``decision_context`` and
    ``goal_runtime`` are passed through; ``LanguageModelJev`` reads none of them.

    Merge rule, per lane b: the instance's goal where ``terminate[b]``, else the
    goal in force, payload and control unchanged. ``terminate`` is the switch
    request of the ADK switch rule, so a lane that continues keeps its goal
    exactly. When no lane switches, ``decide`` returns None, which
    ``session.run_session`` reads as keeping the goal (not a miss). Otherwise the
    session writes the merged goal with goal age 0 on every lane.

    Lane failures (``diagnostics["lane_failed"]``, where the instance kept the
    goal after a backend error or timeout) are counted here. The session cannot
    see them, so without this count a dead backend would read as a model that
    always continues.

    Args:
        instance: The host-mode instance. It must provide ``set_executive_state``.
        executive: E_t, bound before the first call.
        batch: Number of lanes B.
        seed: Seed of the PRNG key passed to the instance, split per decision.

    Attributes:
        lane_calls: Lane decisions requested, B per call.
        lane_failures: Lane decisions that kept the goal after a failure.
        first_failure: The first failure message, or None.
    """

    def __init__(self, instance: Any, executive: ExecutiveState, batch: int, *, seed: int = 0) -> None:
        if getattr(instance, "mode", None) is not JevMode.HOST:
            raise TypeError("instance must be a host-mode JEV instance (JevMode.HOST)")
        instance.set_executive_state(executive)
        self._instance = instance
        self._state = instance.initial_state(batch)
        self._key = jax.random.key(seed)
        self.lane_calls = 0
        self.lane_failures = 0
        self.first_failure: str | None = None

    def decide(
        self, transition: Any, hierarchy_state: Any, executive: ExecutiveState | None, /
    ) -> ManagedGoal | None:
        """Return the merged goal, or None when no lane switches. ``executive``, when given, replaces E_t first."""

        if executive is not None:
            self._instance.set_executive_state(executive)
        self._key, key = jax.random.split(self._key)
        current = hierarchy_state.carry.goal
        decision = self._instance(
            self._state,
            transition.hierarchy_step.belief_update.belief,
            current.payload,
            transition.next_policy_input.actor_input,
            key,
            decision_context=transition.next_policy_input.decision_context,
            goal_runtime=transition.hierarchy_step.goal_runtime,
            executive=hierarchy_state.carry.manager_state,
        )
        self._state = decision.state
        switch = np.asarray(decision.terminate, dtype=np.bool_)
        self.lane_calls += int(switch.shape[0])
        failed = decision.diagnostics.get("lane_failed")
        if failed is not None:
            self.lane_failures += int(np.count_nonzero(np.asarray(failed)))
        errors = getattr(self._instance, "last_errors", ())
        if errors and self.first_failure is None:
            self.first_failure = str(errors[0])
        if not switch.any():
            return None

        def pick(proposed: Any, held: Any) -> jax.Array:
            held_array = jnp.asarray(held)
            mask = switch.reshape(switch.shape + (1,) * (held_array.ndim - 1))
            return jnp.where(mask, jnp.asarray(proposed, dtype=held_array.dtype), held_array)

        return jax.tree.map(pick, decision.goal, current)


def _shared_belief(ctx: ArmContext) -> Any:
    """Return the shared belief B_t core from the sibling ``belief`` package."""

    from experimental.neural_agent.belief.core import SharedBelief, init_belief_params

    del ctx
    params = init_belief_params(jax.random.key(BELIEF_PARAMS_SEED))
    return SharedBelief(params=params)


def _skill_dispatcher(ctx: ArmContext) -> Any:
    """Return the Level 1 actor, the dispatcher over the sibling skill registry."""

    from experimental.neural_agent.skill_library.dispatcher import SkillDispatcher

    return SkillDispatcher(ctx.registry)


def _control_table(ctx: ArmContext) -> Any:
    """Return the goal control of the catalogue, one row per ``ctx.skill_ids`` entry.

    That is the registry's ``control_table()`` when the registry has one, and
    otherwise ``default_control_table(ctx.skill_ids)`` (a test context passes no
    registry).

    Raises:
        ValueError: If the registry's ids differ from ``ctx.skill_ids``.
    """

    table_of = getattr(ctx.registry, "control_table", None)
    if table_of is None:
        return default_control_table(ctx.skill_ids)
    if tuple(ctx.registry.ids) != ctx.skill_ids:
        raise ValueError(
            f"the registry's ids {tuple(ctx.registry.ids)} differ from the catalogue {ctx.skill_ids}"
        )
    return table_of()


def _flat_manager(ctx: ArmContext) -> ConstantGoalManager:
    """Return the flat arm's manager, which proposes the pursuit skill every tick."""

    if PURSUIT_SKILL_ID not in ctx.skill_ids:
        raise ValueError(
            f"the flat arm needs the {PURSUIT_SKILL_ID!r} skill; the catalogue has {ctx.skill_ids}"
        )
    return ConstantGoalManager(skill_index=ctx.skill_ids.index(PURSUIT_SKILL_ID))


def _scripted_manager(ctx: ArmContext) -> JevArmManager:
    """Return the J0 arm: ``ScriptedJev`` inside ``JevRunner`` with the executive bound."""

    from experimental.neural_agent.jev.runner import JevRunner
    from experimental.neural_agent.jev.scripted import ScriptedJev

    control = _control_table(ctx)
    instance = ScriptedJev(ctx.skill_ids, control=control)
    runner = JevRunner(
        instance,
        JevBudget(latency_ms=J0_LATENCY_BUDGET_MS),
        ctx.skill_ids,
        control=control,
    )
    return JevArmManager(
        runner,
        ctx.executive,
        measured=instance.mode is JevMode.HOST,
    )


def _shuffled_manager(ctx: ArmContext) -> ShuffledGoalManager:
    """Return the shuffled-goal control: the J0 arm with each skill index permuted."""

    return ShuffledGoalManager(_scripted_manager(ctx), n_skills=len(ctx.skill_ids))


def load_j1_params(path: str | Path) -> Any:
    """Load J1 parameters from an ``.npz`` whose keys are the ``J1Params`` field names.

    The file holds exactly the ten arrays ``w1, b1, w2, b2, w_skill, b_skill,
    w_replace, b_replace, w_params, b_params``, as ``np.savez(path,
    **params._asdict())`` writes them. ``NetworkJev`` checks shapes and dtypes.

    Raises:
        ValueError: If the path is not an ``.npz``, or its keys differ from the fields.
    """

    from experimental.neural_agent.jev.network import J1Params

    target = Path(path)
    if target.suffix != ".npz":
        raise ValueError(f"J1 parameters must be an .npz file, got {target}")
    expected = set(J1Params._fields)
    with np.load(target) as data:
        found = set(data.files)
        if found != expected:
            raise ValueError(
                f"{target} must hold exactly the J1Params fields {sorted(expected)}, got {sorted(found)}"
            )
        return J1Params(**{name: jnp.asarray(data[name]) for name in J1Params._fields})


def _network_manager(ctx: ArmContext) -> JevArmManager:
    """Return the J1 arm: ``NetworkJev`` inside ``JevRunner`` with the executive bound.

    The parameters are ``init_j1(key(ctx.seed))``, a random initialization, unless
    ``ctx.j1_params`` names a file.
    """

    from experimental.neural_agent.jev.network import NetworkJev, init_j1
    from experimental.neural_agent.jev.runner import JevRunner

    if ctx.j1_params is None:
        params = init_j1(jax.random.key(ctx.seed), num_skills=len(ctx.skill_ids))
    else:
        params = load_j1_params(ctx.j1_params)
    scored = NetworkJev(params).num_skills
    if scored != len(ctx.skill_ids):
        raise ValueError(
            f"the J1 parameters score {scored} skills; "
            f"the catalogue has {len(ctx.skill_ids)}"
        )
    control = _control_table(ctx)
    instance = NetworkJev(params, control=control)
    runner = JevRunner(
        instance,
        JevBudget(latency_ms=J0_LATENCY_BUDGET_MS),
        ctx.skill_ids,
        control=control,
    )
    return JevArmManager(
        runner,
        ctx.executive,
        measured=instance.mode is JevMode.HOST,
    )


def _hold_manager(ctx: ArmContext) -> HoldGoalManager:
    """Return the in-graph manager of the ``j_lm`` arm."""

    return HoldGoalManager(ctx.executive, ctx.skill_ids)


def _lm_host_decider(ctx: ArmContext, steps: int) -> LanguageModelHostDecider:
    """Return the ``j_lm`` host decider for one session of ``steps`` steps.

    The session decides at most once per step, and each decision asks the
    backend once per lane, so the fake backend is scripted with ``steps * B``
    pursuit replies. The latency budget is :data:`J0_LATENCY_BUDGET_MS`; it is
    also each backend call's timeout (``LanguageModelJev``).

    Raises:
        ValueError: If ``steps`` is not a positive integer, or the fake backend's
            catalogue has no pursuit skill.
    """

    from experimental.neural_agent.jev.lm.instance import LanguageModelJev

    if isinstance(steps, bool) or not isinstance(steps, numbers.Integral) or steps < 1:
        raise ValueError(f"steps must be a positive integer, got {steps!r}")
    if ctx.lm_backend == "fake":
        from experimental.neural_agent.jev.lm.protocol import FakeDecisionLM

        if PURSUIT_SKILL_ID not in ctx.skill_ids:
            raise ValueError(
                f"the fake backend answers {PURSUIT_SKILL_ID!r}; the catalogue has {ctx.skill_ids}"
            )
        backend: Any = FakeDecisionLM([_FAKE_PURSUIT_REPLY] * (int(steps) * ctx.batch))
    else:
        from experimental.neural_agent.jev.lm.ollama_qwen import OllamaQwenBackend

        backend = OllamaQwenBackend(model=OLLAMA_MODEL)
    instance = LanguageModelJev(
        f"j-lm-{ctx.lm_backend}",
        backend,
        ctx.skill_ids,
        JevBudget(latency_ms=J0_LATENCY_BUDGET_MS),
        control=_control_table(ctx),
    )
    return LanguageModelHostDecider(instance, ctx.executive, ctx.batch, seed=ctx.seed)


def _js1_host_decider(ctx: ArmContext, steps: int) -> LanguageModelHostDecider:
    """Return the ``js1`` host decider: ``SystemOneJev`` over ``HashedBagTrunk``.

    The heads are ``init_heads(key(ctx.seed))``, a random initialization of the
    catalogue-agnostic matching head, unless ``ctx.js1_heads`` names a file. The skills
    are described by their ids. ``steps`` is unused: the instance needs no script.
    """

    from experimental.neural_agent.jev.system_one.heads import init_heads
    from experimental.neural_agent.jev.system_one.instance import SystemOneJev
    from experimental.neural_agent.jev.system_one.train_supervised import load_heads
    from experimental.neural_agent.jev.system_one.trunk import HashedBagTrunk

    del steps
    trunk = HashedBagTrunk()
    if ctx.js1_heads is None:
        heads = init_heads(jax.random.key(ctx.seed), trunk.dim)
    else:
        heads = load_heads(ctx.js1_heads)
    instance = SystemOneJev(trunk, heads, ctx.skill_ids, control=_control_table(ctx))
    return LanguageModelHostDecider(instance, ctx.executive, ctx.batch, seed=ctx.seed)


def program_settings() -> tuple[str, float]:
    """Return the ``jprog`` arm's (program name, no-reissue seconds) from ``NA_PROGRAM`` and ``NA_NO_REISSUE``.

    Raises:
        ValueError: If the program is not in ``jev/programs.py`` PROGRAMS, or the window is not a
            finite number >= 0.
    """

    from experimental.neural_agent.jev.programs import PROGRAMS

    name = os.environ.get(PROGRAM_ENV, "J0")
    if name not in PROGRAMS:
        raise ValueError(f"{PROGRAM_ENV} must be one of {tuple(PROGRAMS)}, got {name!r}")
    text = os.environ.get(NO_REISSUE_ENV, "0")
    try:
        window = float(text)
    except ValueError:
        raise ValueError(f"{NO_REISSUE_ENV} must be a number of seconds, got {text!r}") from None
    if not math.isfinite(window) or window < 0.0:
        raise ValueError(f"{NO_REISSUE_ENV} must be finite and >= 0, got {text!r}")
    return name, window


def _program_host_decider(ctx: ArmContext, steps: int) -> Any:
    """Return the ``jprog`` host decider: ``ProgramExecutor`` with one program on every lane.

    The program and window come from :func:`program_settings`; the control table and the contracts are the
    registry's (:func:`_control_table`; ``registry.contracts``, None without a registry). The session binds
    its scratchpad to the executor, which then writes its requests there (``jev/programs.py``, items 8 and
    9). For a session longer than one step (the timed session, not the one-step
    warm-up), the executor's per-lane counters are written to stderr as one JSON line when the process
    exits, because the composition report has no field for them.
    """

    from experimental.neural_agent.jev.programs import PROGRAMS, ProgramExecutor

    name, window = program_settings()
    executor = ProgramExecutor(
        ctx.skill_ids,
        _control_table(ctx),
        (PROGRAMS[name],) * ctx.batch,
        no_reissue_seconds=window,
        contracts=getattr(ctx.registry, "contracts", None),
    )
    if steps > 1:
        atexit.register(_write_program_counters, arm_label("jprog", ctx), steps, executor)
    return executor


def _typed_host_decider(ctx: ArmContext, steps: int) -> Any:
    """Return the ``jtyped`` host decider: the exact typed path (``jev/typed_rule.py``) over the context's E_t.

    Typed conditions (or the legacy constraint labels) are evaluated exactly on the belief and the priority
    rule chooses the goal; nothing is learned. The control table is the registry's (:func:`_control_table`).
    For a session longer than one step the counters go to stderr as one JSON line at exit, like ``jprog``.
    """

    from experimental.neural_agent.jev.typed_rule import TypedRuleDecider

    decider = TypedRuleDecider(ctx.skill_ids, _control_table(ctx), ctx.executive, ctx.batch)
    if steps > 1:
        atexit.register(_write_program_counters, arm_label("jtyped", ctx), steps, decider)
    return decider


def _write_program_counters(label: str, steps: int, executor: Any) -> None:
    """Write the executor's counters to stderr as one JSON line (registered with ``atexit``)."""

    sys.stderr.write(f"{label} program counters ({steps}-step session): {json.dumps(executor.counters())}\n")


def arm_label(name: str, ctx: ArmContext) -> str:
    """Return the report label of an arm.

    ``j1`` without trained parameters is ``j1-untrained``. ``j_lm`` names its
    backend (``j_lm-fake``, ``j_lm-ollama``). Every other arm keeps its name.
    """

    if name == "j1" and ctx.j1_params is None:
        return "j1-untrained"
    if name == "js1" and ctx.js1_heads is None:
        return "js1-untrained"
    if name == "j_lm":
        return f"j_lm-{ctx.lm_backend}"
    if name == "jprog":
        program, window = program_settings()
        return f"jprog-{program}" + (f"-noreissue{window:g}s" if window > 0.0 else "")
    return name


class Arm(NamedTuple):
    """A named comparison arm: three factories, each taking an :class:`ArmContext`.

    Attributes:
        name: The key used on the command line, for example ``"j0"``.
        make_manager: Returns the manager (an :class:`ArmManager`).
        make_actor: Returns the actor (an ADK ``LowLevelActor``).
        make_belief: Returns the belief core (an ADK ``BeliefCore``).
        make_host_decider: For a host-mode arm, ``(ctx, steps)`` returns the
            ``session.HostDecider`` for one session of ``steps`` steps. None for
            an arm whose decision model runs in the manager slot.
    """

    name: str
    make_manager: Callable[[ArmContext], Any]
    make_actor: Callable[[ArmContext], Any]
    make_belief: Callable[[ArmContext], Any]
    make_host_decider: Callable[[ArmContext, int], Any] | None = None


#: The arms by command-line name. ``j_lm``, ``js1``, ``jprog`` and ``jtyped`` are the host-mode arms.
ARMS: dict[str, Arm] = {
    "flat": Arm("flat", _flat_manager, _skill_dispatcher, _shared_belief),
    "j0": Arm("j0", _scripted_manager, _skill_dispatcher, _shared_belief),
    "shuffled": Arm("shuffled", _shuffled_manager, _skill_dispatcher, _shared_belief),
    "j1": Arm("j1", _network_manager, _skill_dispatcher, _shared_belief),
    "j_lm": Arm("j_lm", _hold_manager, _skill_dispatcher, _shared_belief, _lm_host_decider),
    "js1": Arm("js1", _hold_manager, _skill_dispatcher, _shared_belief, _js1_host_decider),
    "jprog": Arm("jprog", _hold_manager, _skill_dispatcher, _shared_belief, _program_host_decider),
    "jtyped": Arm("jtyped", _hold_manager, _skill_dispatcher, _shared_belief, _typed_host_decider),
}


def get_arm(name: str) -> Arm:
    """Return the arm registered under ``name``.

    Raises:
        ValueError: If ``name`` is not a registered arm.
    """

    if name not in ARMS:
        raise ValueError(f"unknown arm {name!r}; expected one of {tuple(ARMS)}")
    return ARMS[name]


__all__ = [
    "ARMS",
    "Arm",
    "ArmContext",
    "ArmManager",
    "ConstantGoalManager",
    "HoldGoalManager",
    "J0_LATENCY_BUDGET_MS",
    "JevArmManager",
    "LM_BACKENDS",
    "LanguageModelHostDecider",
    "NO_REISSUE_ENV",
    "OLLAMA_MODEL",
    "PROGRAM_ENV",
    "PURSUIT_SKILL_ID",
    "ShuffledGoalManager",
    "arm_label",
    "build_arm_context",
    "get_arm",
    "load_j1_params",
    "program_settings",
]
