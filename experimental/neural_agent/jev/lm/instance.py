"""LanguageModelJev: the quick language-model JEV instance (ADR-001 D2, J-LM).

Host-mode instance. Each lane b in [0, B) gets one call to a DecisionLM. The
reply is parsed by ``prompting.parse_decision`` into a skill index k, four
parameters p and a continue flag c, and the lane's goal is then formed as below.
Each formula line names its source in brackets. SKILL.md carries the same table.

Formulas (lane b; k_cur,b, p_cur,b, v_cur,b, s_cur,b are the current goal):

    reply:        (k_b, p_b, c_b) = parse_decision(LM(prompt_b)),  c_b in {0, 1}
                  [ADR-001 D2 J-LM row; prompting.parse_decision]
    continue      c_b = 1:  goal'_b = goal_b, and k_b must equal k_cur,b
                  [contracts.JevInstance: terminate false means CONTINUE]
    switch        c_b = 0:  goal'_b = (k_b, p_b, v_b, s_b)
                  [ADR-001 D2 J-LM row: "emitting the same ManagedGoal"]
    version       v_b = E_t.version, which must equal executive.version[b]
                  [contracts.GoalPayload.executive_version]
    slot          s_b = min{ g : goal_valid[b,g] and skill_index[b,g] = k_b }, or -1
                  [contracts.GoalPayload.goal_slot; executive_arrays lowers goal g to slot g]
    alignment     for every goal g of E_t: goal_valid[b,g] and skill_index[b,g] equal the
                  index that this catalogue gives goals[g].skill_id, else a lane error
                  [implementation; parent SKILL.md rule 5, fail closed]
    terminate     term_b = (1 - c_b) and [k_b != k_cur,b]
                  [implementation; hierarchy.py:347-356 applies a manager switch only when
                  terminate is true, or the goal is invalid, timed out or has a pending
                  actor termination]
    fallback      on any exception in lane b: goal'_b = goal_b, term_b = 0, lane_failed[b]
                  = true, and "lane b: <ErrorType>: <message>" is written to last_errors
                  [implementation; contracts.JevInstance: a missed budget keeps the goal]
    timeout       timeout_s = latency_ms / 1000, from the JevBudget. A TimeoutError on
                  lane b keeps lane b's goal. The runner counts a miss when the whole call
                  exceeds the budget, then re-emits the current goal for every lane
                  [ADR-001 D2 timeout semantics; jev/runner.py:280-282 and 334-340]
    controls      minimum commitment 0.5 s, maximum duration 10 s,
                  emergency mask {DAMAGE_RECEIVED, TARGET_DIED, NPC_STUCK}
                  [values chosen in this file; ADR-001 D2 retains PLAN section 19.4, which
                  names the commit rule but not these values]
    state         state' = state + 1 per lane, errors included
                  [implementation; initial_state: decisions made per lane]

Reads only B_t (features, feature_known) and E_t (through ExecutiveArrays and the
ExecutiveState it was lowered from). actor_input, key, decision_context and
goal_runtime are accepted for the JevInstance contract and not read (ADR-001 D2).

``terminate`` follows the ADK core. Under hierarchy.py:347-356 a manager switch needs
``terminate`` true, so a same-skill params change with c = 0 gets terminate false and
is not applied while the goal stays valid. SKILL.md lists this as open gap G1.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NamedTuple

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
    ExecutiveArrays,
    ExecutiveState,
    GoalPayload,
    JevBudget,
    JevMode,
    SharedBeliefState,
    event_mask,
    make_goal,
)
from experimental.neural_agent.jev.lm.prompting import (
    DECISION_JSON_SCHEMA,
    build_decision_prompt,
    parse_decision,
    validate_skill_ids,
)
from experimental.neural_agent.jev.lm.protocol import DecisionLM

MINIMUM_COMMITMENT_SECONDS: float = 0.5
MAXIMUM_DURATION_SECONDS: float = 10.0
EMERGENCY_EVENTS: tuple[DecisionEvent, ...] = (
    DecisionEvent.DAMAGE_RECEIVED,
    DecisionEvent.TARGET_DIED,
    DecisionEvent.NPC_STUCK,
)


class _HostArrays(NamedTuple):
    """One call's batched inputs as NumPy arrays. Shapes, with B lanes:

    features float64[B, 10]; known bool[B, 10]; current_skill int32[B];
    current_params float32[B, 4]; current_version int32[B]; current_slot int32[B];
    exec_skill int32[B, 4]; exec_valid bool[B, 4]; exec_version int32[B].
    """

    features: np.ndarray
    known: np.ndarray
    current_skill: np.ndarray
    current_params: np.ndarray
    current_version: np.ndarray
    current_slot: np.ndarray
    exec_skill: np.ndarray
    exec_valid: np.ndarray
    exec_version: np.ndarray


class _LaneChoice(NamedTuple):
    """The goal chosen for one lane, before batching."""

    skill: int
    params: np.ndarray
    version: int
    slot: int
    terminate: bool


class LanguageModelJev:
    """Host-mode JEV instance that asks a quick language model for each goal.

    Host mode (``JevMode.HOST``): :meth:`__call__` converts the batched JAX
    inputs to NumPy and runs Python. It cannot run under ``jax.jit`` or inside
    ``jax.lax.cond``, so it cannot be a branch of the jitted hierarchy core either.
    Its failure messages stay host-side in ``last_errors``. Use it for small batches.
    The model is called once per lane, in order, so one call can take up to
    B times the per-decision budget. Each lane's own call is bounded by the budget.

    Attributes:
        name: Instance id used in reports, for example ``'j-lm-qwen'``.
        mode: ``JevMode.HOST``.
        last_errors: Lane failure messages of the most recent call, each
            ``"lane b: <ErrorType>: <message>"``. Host-side only: the batched output
            carries ``lane_failed`` instead. Empty before the first call.
    """

    def __init__(
        self,
        name: str,
        backend: DecisionLM,
        skill_ids: Sequence[str],
        budget: JevBudget,
    ) -> None:
        """Bind the instance to a language model, a skill catalogue and a budget.

        Args:
            name: Non-empty instance id.
            backend: A DecisionLM. Called once per lane per decision.
            skill_ids: The skill catalogue, in dispatcher order. Checked by
                ``validate_skill_ids``.
            budget: The per-decision latency budget. Its ``latency_ms`` is the
                timeout of each model call, as ``latency_ms / 1000`` seconds.

        Raises:
            TypeError: If ``backend`` does not implement DecisionLM, or ``budget``
                is not a JevBudget.
            ValueError: If ``name`` is empty or the catalogue is invalid.
        """

        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        if not isinstance(backend, DecisionLM):
            raise TypeError("backend must implement DecisionLM (a decide method)")
        if not isinstance(budget, JevBudget):
            raise TypeError("budget must be a JevBudget")
        self.name = name
        self.mode = JevMode.HOST
        self._backend = backend
        self._skill_ids = validate_skill_ids(skill_ids)
        self._timeout_s = budget.latency_ms / 1000.0
        self._executive: ExecutiveState | None = None
        self.last_errors: tuple[str, ...] = ()

    def set_executive_state(self, state: ExecutiveState) -> None:
        """Set the E_t that the prompts are built from.

        The executive arrays passed to each call must carry the same version
        (checked per lane at decision time).

        Args:
            state: The ExecutiveState. Every goal's skill id must be in the catalogue.

        Raises:
            TypeError: If ``state`` is not an ExecutiveState.
            ValueError: If a goal names a skill outside the catalogue.
        """

        if not isinstance(state, ExecutiveState):
            raise TypeError("state must be an ExecutiveState")
        for slot, goal in enumerate(state.goals):
            if goal.skill_id not in self._skill_ids:
                raise ValueError(f"executive goal {slot} names unknown skill {goal.skill_id!r}")
        self._executive = state

    def initial_state(self, batch: int) -> jax.Array:
        """Return the reset instance state: decisions made per lane, all zero.

        Args:
            batch: Number of lanes B, a positive int.

        Returns:
            int32[B], all zero.

        Raises:
            TypeError: If ``batch`` is not an integer.
            ValueError: If ``batch`` is not positive.
        """

        if isinstance(batch, bool) or not isinstance(batch, (int, np.integer)):
            raise TypeError(f"batch must be an integer, got {type(batch).__name__}")
        if batch < 1:
            raise ValueError(f"batch must be positive, got {batch}")
        return jnp.zeros((int(batch),), dtype=jnp.int32)

    def __call__(
        self,
        state: Any,
        belief: SharedBeliefState,
        current_goal: GoalPayload,
        actor_input: ActorPolicyInput,
        key: jax.Array,
        *,
        decision_context: DecisionContext,
        goal_runtime: GoalRuntimeContext,
        executive: ExecutiveArrays,
    ) -> GoalDecision:
        """Ask the language model for one goal per lane.

        Args:
            state: int32[B], the decisions made per lane so far, from
                :meth:`initial_state`. The returned state adds one to every lane,
                errors included.
            belief: SharedBeliefState. ``features`` float[B, 10] and
                ``feature_known`` bool[B, 10] are read. ``latent`` is not read.
            current_goal: GoalPayload. ``skill_index`` int32[B], ``params``
                float32[B, 4], ``executive_version`` int32[B], ``goal_slot`` int32[B].
            actor_input: Not read.
            key: Not read. The Ollama backend decodes greedily (temperature 0). The
                Anthropic backend sends no temperature, so it samples at the API default.
            decision_context: Not read.
            goal_runtime: Not read.
            executive: ExecutiveArrays, lowered from the executive state. Its
                ``skill_index`` int32[B, 4], ``goal_valid`` bool[B, 4] and
                ``version`` int32[B] are read.

        Returns:
            A GoalDecision. ``state`` is int32[B]. ``goal`` is a ManagedGoal with a
            GoalPayload (``skill_index`` int32[B], ``params`` float32[B, 4],
            ``executive_version`` int32[B], ``goal_slot`` int32[B]), commitment
            0.5 s, maximum duration 10 s and the emergency mask. ``terminate`` is
            bool[B]. ``diagnostics`` is ``{"lane_failed": bool[B]}``, True where the
            lane kept its goal after a failure. Only batched arrays cross the ADK core
            (parent SKILL.md rule 4). The failure messages are in ``last_errors``.

        Raises:
            TypeError: If an argument has the wrong record type.
            ValueError: If a batched array has the wrong shape or dtype. A lane
                failure does not raise. It keeps the lane's goal, sets lane_failed,
                and writes the message to last_errors.
        """

        counts = _lane_counts(state)
        lanes = counts.shape[0]
        arrays = _host_arrays(belief, current_goal, executive, lanes)

        choices: list[_LaneChoice] = []
        errors: list[str] = []
        failed: list[bool] = []
        for lane in range(lanes):
            try:
                choices.append(self._decide_lane(arrays, lane))
                failed.append(False)
            except Exception as error:  # Lane boundary: keep this goal, record the error.
                errors.append(f"lane {lane}: {type(error).__name__}: {error}")
                choices.append(_keep_current(arrays, lane))
                failed.append(True)
        self.last_errors = tuple(errors)

        goal = make_goal(
            jnp.asarray(np.array([c.skill for c in choices], dtype=np.int32)),
            jnp.asarray(np.stack([c.params for c in choices]).astype(np.float32)),
            jnp.asarray(np.array([c.version for c in choices], dtype=np.int32)),
            jnp.asarray(np.array([c.slot for c in choices], dtype=np.int32)),
            minimum_commitment_seconds=jnp.full(
                (lanes,), MINIMUM_COMMITMENT_SECONDS, dtype=jnp.float32
            ),
            maximum_duration_seconds=jnp.full(
                (lanes,), MAXIMUM_DURATION_SECONDS, dtype=jnp.float32
            ),
            emergency_interrupt_mask=event_mask(*EMERGENCY_EVENTS, batch=lanes),
        )
        return GoalDecision(
            state=jnp.asarray(counts + 1, dtype=jnp.int32),
            goal=goal,
            terminate=jnp.asarray(
                np.array([c.terminate for c in choices], dtype=np.bool_)
            ),
            diagnostics={"lane_failed": jnp.asarray(np.array(failed, dtype=np.bool_))},
        )

    def _decide_lane(self, arrays: _HostArrays, lane: int) -> _LaneChoice:
        """Ask the model about one lane and map its reply to a choice.

        Raises on any failure. :meth:`__call__` turns the failure into a kept goal.
        """

        executive = self._require_executive()
        served = int(arrays.exec_version[lane])
        if served != executive.version:
            raise ValueError(
                f"executive arrays are version {served} but the executive state is "
                f"version {executive.version}"
            )
        _check_catalogue_alignment(arrays, lane, executive, self._skill_ids)
        current = int(arrays.current_skill[lane])
        if not 0 <= current < len(self._skill_ids):
            raise ValueError(
                f"current goal skill index {current} is outside the catalogue of "
                f"{len(self._skill_ids)} skills"
            )
        prompt = build_decision_prompt(
            arrays.features[lane],
            arrays.known[lane],
            executive,
            self._skill_ids,
            self._skill_ids[current],
        )
        reply = self._backend.decide(prompt, DECISION_JSON_SCHEMA, self._timeout_s)
        skill, params, continue_current = parse_decision(reply, self._skill_ids)
        if continue_current:
            if skill != current:
                raise ValueError(
                    f"continue_current is true but skill_id {self._skill_ids[skill]!r} "
                    f"differs from the current skill {self._skill_ids[current]!r}"
                )
            return _keep_current(arrays, lane)
        return _LaneChoice(
            skill=skill,
            params=np.asarray(params, dtype=np.float32),
            version=served,
            slot=_matching_slot(arrays, lane, skill),
            terminate=skill != current,
        )

    def _require_executive(self) -> ExecutiveState:
        """Return the E_t, or raise if none has been set."""

        if self._executive is None:
            raise ValueError(
                "no executive state is set; call set_executive_state before deciding"
            )
        return self._executive


def _keep_current(arrays: _HostArrays, lane: int) -> _LaneChoice:
    """Return the goal already in force for ``lane``, with terminate false."""

    return _LaneChoice(
        skill=int(arrays.current_skill[lane]),
        params=arrays.current_params[lane],
        version=int(arrays.current_version[lane]),
        slot=int(arrays.current_slot[lane]),
        terminate=False,
    )


def _matching_slot(arrays: _HostArrays, lane: int, skill: int) -> int:
    """Return the first valid executive slot of ``lane`` that names ``skill``, or -1."""

    matches = np.flatnonzero(arrays.exec_valid[lane] & (arrays.exec_skill[lane] == skill))
    return int(matches[0]) if matches.size else -1


def _check_catalogue_alignment(
    arrays: _HostArrays, lane: int, executive: ExecutiveState, skill_ids: tuple[str, ...]
) -> None:
    """Raise ValueError unless the executive arrays index the skills of this catalogue.

    ``executive_arrays`` lowers goal ``g`` to slot ``g``, so each goal must sit in its
    slot with the index that ``skill_ids`` gives its skill. A catalogue in another order
    would otherwise choose by name while the runner reads the index differently.
    """

    for slot, goal in enumerate(executive.goals):
        expected = skill_ids.index(goal.skill_id)
        served = int(arrays.exec_skill[lane, slot])
        valid = bool(arrays.exec_valid[lane, slot])
        if not valid or served != expected:
            raise ValueError(
                f"executive slot {slot} is valid={valid} with skill index {served}, but "
                f"this catalogue has {goal.skill_id!r} at index {expected}"
            )


def _lane_counts(state: Any) -> np.ndarray:
    """Return the per-lane decision counts, int[B] with B >= 1."""

    counts = np.asarray(state)
    if counts.ndim != 1 or counts.shape[0] < 1 or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError(
            "state must be an integer array of shape (B,) with B >= 1; "
            f"got shape {counts.shape}, dtype {counts.dtype}"
        )
    return counts


def _host_arrays(
    belief: SharedBeliefState,
    current_goal: GoalPayload,
    executive: ExecutiveArrays,
    lanes: int,
) -> _HostArrays:
    """Check the batched inputs and convert them to NumPy with fixed dtypes."""

    if not isinstance(belief, SharedBeliefState):
        raise TypeError("belief must be a SharedBeliefState")
    if not isinstance(current_goal, GoalPayload):
        raise TypeError("current_goal must be a GoalPayload")
    if not isinstance(executive, ExecutiveArrays):
        raise TypeError("executive must be an ExecutiveArrays")
    features = np.asarray(belief.features)
    _expect_shape(features, (lanes, BELIEF_FEATURE_COUNT), "belief.features")
    if not np.issubdtype(features.dtype, np.number):
        raise ValueError(f"belief.features must be numeric, got dtype {features.dtype}")
    known = np.asarray(belief.feature_known)
    _expect_shape(known, (lanes, BELIEF_FEATURE_COUNT), "belief.feature_known")
    if known.dtype != np.bool_:
        raise ValueError(f"belief.feature_known must be bool, got dtype {known.dtype}")
    return _HostArrays(
        features=features.astype(np.float64),
        known=known,
        current_skill=_int_array(current_goal.skill_index, (lanes,), "current_goal.skill_index"),
        current_params=_float_array(
            current_goal.params, (lanes, PARAM_WIDTH), "current_goal.params"
        ),
        current_version=_int_array(
            current_goal.executive_version, (lanes,), "current_goal.executive_version"
        ),
        current_slot=_int_array(current_goal.goal_slot, (lanes,), "current_goal.goal_slot"),
        exec_skill=_int_array(executive.skill_index, (lanes, MAX_GOALS), "executive.skill_index"),
        exec_valid=_bool_array(executive.goal_valid, (lanes, MAX_GOALS), "executive.goal_valid"),
        exec_version=_int_array(executive.version, (lanes,), "executive.version"),
    )


def _expect_shape(array: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    """Raise ValueError unless ``array`` has exactly ``shape``."""

    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")


def _int_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Return an integer array of ``shape`` as int32. Floats are rejected, not truncated."""

    array = np.asarray(value)
    _expect_shape(array, shape, name)
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be an integer array, got dtype {array.dtype}")
    return array.astype(np.int32)


def _float_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Return a floating array of ``shape`` as float32."""

    array = np.asarray(value)
    _expect_shape(array, shape, name)
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{name} must be a floating array, got dtype {array.dtype}")
    return array.astype(np.float32)


def _bool_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    """Return a bool array of ``shape``."""

    array = np.asarray(value)
    _expect_shape(array, shape, name)
    if array.dtype != np.bool_:
        raise ValueError(f"{name} must be bool, got dtype {array.dtype}")
    return array
