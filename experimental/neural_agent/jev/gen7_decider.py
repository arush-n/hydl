"""The decision layer's real programs on the gen-7 host session: a :class:`Gen7ProgramDecider` around ProgramExecutor.

``session_gen7.run_gen7_session`` calls a decider on every tick with ``(tick, document, facts, executive)``. The
decision layer's executor (:class:`~experimental.neural_agent.jev.programs.ProgramExecutor`, unchanged) is a
``session.HostDecider``: it reads ADK hierarchy state. This adapter keeps the goal state the ADK core carried, builds
the belief from the legal gen-7 observation, and hands the executor stand-ins with exactly the fields it reads.

Per tick ``t`` (``decide``):

1. Belief. Every call advances the shared belief (``belief.belief.SharedBelief`` or any object with its
   ``initial_state``/``__call__`` interface) on the tick's legal learner observation, the previous action factors and
   the seconds since the previous call. A lane whose episode ended (``Gen7DecisionFacts.done``) or the first tick
   restarts its belief carry. The belief never reads privileged state: the only observation input is
   ``Gen7DecisionFacts.legal_observation``.
2. Goal clock. ``goal_age`` (per lane) grows by ``(t - t_prev) * tick_seconds`` and is zeroed where a goal is applied,
   as the ADK session zeroes ``carry.goal_age`` on a write (``session.py`` step 4, ``host_respects_commitment``).
3. Termination. A lane's goal is terminated when one of its live requests reports ``done`` on a status of this
   decision's window (``skills.statuses``, the verb of the request from the executor's own requests).
4. Cadence. The executor is called on ticks where ``t % cadence == 0`` (cadence = round(0.5 s / tick_seconds), the
   core's manager period) and, in the ADK seam, on emergencies. On other ticks the goal in force is kept.
5. Executor. Stand-ins (below) go to ``ProgramExecutor.decide``; its goal is applied per lane where the ADK seam
   would (``session.py`` step 4: the goal differs and (not valid | commitment elapsed | timed out | emergency)).
6. Positions. The executor sets an entity or a position only for verbs whose ``ARGUMENT_SCHEMAS`` entry takes one
   (``requests.py``, ``belief/target_memory.py`` ``target_arguments``); ``walk_to`` and ``look_at`` have none, so their
   requests leave the executor with no position. The adapter fills the position from the skill layer's known-entity
   record of the lane (the entity ``choose_target`` picks; the same record the executor names its target by), else
   from the belief's remembered target position. A request with no point is withheld from S_t, not sent: the runner
   batches the bodies on every live request before the combiner checks grants (``session_gen7.py`` step 4, the
   ``live`` dict built from ``requests.for_agent``), and a walk body's lowering reads ``args.position`` with no check
   (``skill_agent/neural_skills/walk_to/skill.py`` line 102; the first decision of a run has no known-entity record,
   since the state binding's first tick is -1). The executor keeps the withheld request for its correspondence and
   re-issues at its next cadence. The same goes for a finished program's idle goal: the executor names the verb
   ``idle`` (``skill_ids[0]``), the gen-7 library has no such verb, and the combiner would reject it as verb_unknown,
   which the executor's correspondence reads as a rejected step (16 information requests in the first 300-tick run).
   Idle needs no request, so the adapter withholds it. Nothing else is rewritten.

Events. Each tick the adapter keeps the legal observation of the previous tick and calls the ADK's
``_legal_perceptual_events`` on (previous, current), which publishes LINE_OF_SIGHT_CHANGED, ATTACK_TELEGRAPH_BEGAN and
DAMAGE_RECEIVED. The ADK function checks ``isinstance(..., LearnerCombatObservationV3)``; the gen-7 observation is the
same fields plus ``perception_v2`` and is not a subclass, so the adapter re-wraps the v3 fields (``_make``) and drops
the channel. The lifecycle events (ACTION_*, ABILITY_*) need the arsenal's action lifecycle, which the gen-7 runner does
not publish, and the target and navigation events (TARGET_DIED, PATH_INVALID, NPC_STUCK, GOAL_IMPOSSIBLE,
PROJECTILE_THREAT_ENTERED) are not computed by the ADK from legal observations either; all of these stay unknown. A
lane is unknown on the first tick and on the tick its belief restarts (``Gen7DecisionFacts.done``). When any lane has an
active event the adapter calls the executor on that tick too (the ADK's ``manager_decision_due`` treats a known event
as critical).

Stand-ins, and the executor's reads (``jev/programs.py`` ``ProgramExecutor.decide``, line numbers as written):

- ``transition.hierarchy_step.belief_update.belief.features`` and ``.feature_known`` (lines 899 and 903).
- ``transition.hierarchy_step.belief_update.state.target_memory`` through ``host_target_memory`` (line 988;
  ``belief/target_memory.py`` line 105).
- ``transition.hierarchy_step.goal_runtime.emergency_interrupt`` (line 905), bool[B]. Derived per lane as the ADK core
  does (``adk/architecture/hierarchy.py`` lines 280-282): the lane's goal is valid and an event in the goal's emergency
  mask is active (value and known). The events are the ADK's own ``_legal_perceptual_events`` on consecutive legal
  observations (see "Events" below). Program interrupts do not read events: their predicates read the belief and the
  legal target phase (``jev/programs.py`` ``_desired``), so the events reach the executor only through this flag.
- ``transition.done`` (line 906).
- ``transition.actor_input.legal_observation.base.target_i32`` (``_target_codes``, line 1111): read only when a program
  has a ``TargetPhase`` predicate; the adapter sets the legal observation there anyway.
- ``hierarchy_state.goal_valid`` (907), ``goal_termination_requested`` (908), ``manager_age_seconds`` (909),
  ``carry.goal_age`` (914), ``carry.goal.payload`` (915-918: skill_index, params, executive_version, goal_slot),
  ``carry.manager_state`` (``ExecutiveArrays``, 920-922).

The manager age is 0 at every executor call: the executor's due check is ``manager_age < dt`` (line 942, ``ticked``), and at a cadence call ``dt`` is the cadence interval, so a manager age of that interval would never be
due and no step would advance. The adapter's manager is due at each of its calls by construction.

Executor requests issued with the executor's own check (``requests.validate`` with ``ARGUMENT_SCHEMAS``) report
``verb_unknown`` for ``walk_to`` and ``look_at`` (they have no schema there). The executor counts that and still
sends the request; the gen-7 combiner checks each request with its own skill specs (``skill_agent/runtime/library.py``
``check``), which is the check that decides grants.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from adk.architecture.decision_context import DecisionEvent, _legal_perceptual_events
from adk.architecture.inputs import ActorPolicyInput
from experimental.neural_agent.belief.target_memory import host_target_memory
from experimental.neural_agent.contracts import (
    ExecutiveState,
    SkillContract,
    control_table,
    empty_executive_arrays,
    executive_arrays,
    idle_goal,
)
from experimental.neural_agent.jev.programs import (
    Program,
    ProgramExecutor,
    choose_target,
)
from experimental.neural_agent.shared_state import RequestsSection, SharedState
from hytalegym.jax.combat.observation.v3.types import LearnerCombatObservationV3

POSITION_VERBS: tuple[str, ...] = ("walk_to", "look_at")
#: Blocks above an entity's position (its feet) that look_at aims at: the eye. The 0.5.7 NPC ruleset's
#: ``agent.effective_eye_height`` is 1.0 for the agent and the NPCs alike (``skill_agent/tasks/simple_tasks.py``
#: ``AGENT_EYE_HEIGHT``; ``look_at/teacher.py``:25). Aiming at the feet from 0.7 blocks away is a 55 degree pitch, outside
#: look_at's trained range of +-30 degrees (the first end-to-end run: 0 of 16 lanes facing).
LOOK_EYE_HEIGHT: float = 1.0
"""The verbs whose requests the adapter gives a point (module docstring, item 6)."""
CADENCE_SECONDS: float = 0.5
"""The core's manager period (``decision_context.manager_decision_due``, 0.5 s), which is the executor's cadence."""


class Gen7DecisionFacts(NamedTuple):
    """What the adapter reads from a tick's facts (the world gives them through ``facts_for_decider``).

    Attributes:
        legal_observation: The tick's legal learner observation (``LearnerCombatObservationV3``, with or without the
            gen-7 perception channel), one row per lane. The only observation the belief reads.
        previous_action: int32[B, ACTION_HEADS], the factors executed on the previous tick.
        done: bool[B], lanes whose episode ended on the previous tick (their belief and programs restart).
    """

    legal_observation: Any
    previous_action: Any
    done: Any


class _GoalRuntime(NamedTuple):
    emergency_interrupt: Any


class _HierarchyStep(NamedTuple):
    belief_update: Any
    goal_runtime: _GoalRuntime


class _Transition(NamedTuple):
    hierarchy_step: _HierarchyStep
    actor_input: Any
    done: Any


class _Carry(NamedTuple):
    goal_age: Any
    goal: Any
    manager_state: Any


class _HierarchyState(NamedTuple):
    goal_valid: Any
    goal_termination_requested: Any
    manager_age_seconds: Any
    carry: _Carry


def _as_learner_observation(observation: Any) -> LearnerCombatObservationV3 | None:
    """The observation as the ADK's LearnerCombatObservationV3: the v3 fields, re-wrapped; None when there is none.

    The gen-7 observation has the v3 fields and ``perception_v2`` and is not a subclass of the v3 type. Its v3 fields are
    the legal learner observation, so re-wrapping them keeps the channel out of the events and changes no value.
    """

    if observation is None or isinstance(observation, LearnerCombatObservationV3):
        return observation
    fields = LearnerCombatObservationV3._fields
    missing = [
        name for name in fields if name not in getattr(observation, "_fields", ())
    ]
    if missing:
        raise TypeError(f"legal_observation lacks the v3 fields {missing}")
    return LearnerCombatObservationV3._make(
        getattr(observation, name) for name in fields
    )


def _lane_mask(mask: Any, leaf: Any) -> jax.Array:
    """Reshape a bool[B] lane mask so it broadcasts over the leaf's trailing axes."""

    return jnp.reshape(
        jnp.asarray(mask), (mask.shape[0],) + (1,) * (jnp.ndim(leaf) - 1)
    )


class Gen7ProgramDecider:
    """The gen-7 session's decider: ProgramExecutor, driven through stand-ins (module docstring).

    Args:
        programs_per_lane: One :class:`Program` per lane.
        contracts: The catalogue's SkillContracts in catalogue order, ``idle`` first.
        belief: The shared belief: ``initial_state(lanes)`` and ``__call__(state, actor_input, previous_action,
            previous_goal, delta_time, key)`` (``belief.core.SharedBelief``).
        tick_seconds: The gen-7 tick length in seconds.
        facts_for_decider: Maps the world's facts to :class:`Gen7DecisionFacts`.
        cadence_ticks: The executor's cadence in ticks; default ``round(0.5 s / tick_seconds)``.
        no_reissue_seconds: The executor's feedback window (``ProgramExecutor``), 0 by default.
        jit_belief: Compile the belief's call once (the default for the real belief; off for stubs).
        belief_key: The key passed to the belief (it is deterministic and ignores it).

    Attributes:
        executor: The :class:`ProgramExecutor`; its counters are the decision layer's counters.
        goal: The ManagedGoal in force, batched over lanes. ``goal_valid`` and ``goal_age`` per lane.
        executor_calls: Executor calls (cadence ticks), ``goal_writes``: lane goal changes applied,
            ``goal_held``: lane goal changes held by commitment, ``terminations``: lanes terminated by a done status,
            ``positions_filled`` and ``positions_withheld``: walk_to/look_at requests given a point, or withheld for want of one;
            ``idle_withheld``: the idle requests of a finished or idle lane, which are not sent.
    """

    def __init__(
        self,
        programs_per_lane: Sequence[Program],
        *,
        contracts: Sequence[SkillContract],
        belief: Any,
        tick_seconds: float,
        facts_for_decider: Callable[[Any], Gen7DecisionFacts],
        cadence_ticks: int | None = None,
        no_reissue_seconds: float = 0.0,
        jit_belief: bool = False,
        belief_key: jax.Array | None = None,
    ) -> None:
        contracts = tuple(contracts)
        if not contracts or contracts[0].skill_id != "idle":
            raise ValueError(
                "contracts must be the catalogue in order, starting with the idle skill"
            )
        self.skill_ids = tuple(contract.skill_id for contract in contracts)
        self.lanes = len(tuple(programs_per_lane))
        self.tick_seconds = float(tick_seconds)
        self.cadence_ticks = (
            max(1, round(CADENCE_SECONDS / self.tick_seconds))
            if cadence_ticks is None
            else int(cadence_ticks)
        )
        self.facts_for_decider = facts_for_decider
        self.belief = belief
        self._belief_call = (
            jax.jit(self._belief_step) if jit_belief else self._belief_step
        )
        self._belief_key = jax.random.key(0) if belief_key is None else belief_key
        table = control_table(contracts)
        self._minimum = np.asarray(table.minimum_commitment_seconds, dtype=np.float32)
        self._maximum = np.asarray(table.maximum_duration_seconds, dtype=np.float32)
        self.executor = ProgramExecutor(
            self.skill_ids,
            table,
            tuple(programs_per_lane),
            no_reissue_seconds=no_reissue_seconds,
            contracts=contracts,
        )
        self.goal = idle_goal(self.lanes)
        self.goal_valid = np.zeros(self.lanes, dtype=np.bool_)
        self.goal_age = np.zeros(self.lanes, dtype=np.float32)
        self._belief_state: Any = None
        self._update: Any = None
        self._last_tick: int | None = None
        self._arrays = empty_executive_arrays(self.lanes)
        self._arrays_version: int | None = None
        self._status_tick = 0
        self._verb_of: dict[tuple[int, int], str] = {}
        self._previous_legal: LearnerCombatObservationV3 | None = None
        self.event_values = np.zeros((self.lanes, len(DecisionEvent)), np.bool_)
        self.event_known = np.zeros((self.lanes, len(DecisionEvent)), np.bool_)
        self.emergency_interrupt = np.zeros(self.lanes, np.bool_)
        self._event_now = False
        self.executor_calls = 0
        self.goal_writes = 0
        self.goal_held = 0
        self.emergency_writes = 0
        self.event_lane_ticks = np.zeros(len(DecisionEvent), np.int64)
        self.terminations = 0
        self.positions_filled = 0
        self.positions_withheld = 0
        self.idle_withheld = 0

    # ---------------------------------------------------------------- the belief

    def _belief_step(self, state: Any, legal: Any, previous: Any, delta: Any) -> Any:
        return self.belief(
            state,
            ActorPolicyInput(None, None, legal, None),
            previous,
            None,
            delta,
            self._belief_key,
        )

    def _advance_belief(self, tick: int, facts: Gen7DecisionFacts) -> None:
        reset = np.asarray(facts.done, dtype=np.bool_).reshape(self.lanes)
        first = self._last_tick is None
        if first:
            self._belief_state = self.belief.initial_state(self.lanes)
            seconds = np.zeros(self.lanes, dtype=np.float32)
        else:
            seconds = np.where(
                reset, 0.0, (tick - self._last_tick) * self.tick_seconds
            ).astype(np.float32)
            if reset.any():
                fresh = self.belief.initial_state(self.lanes)
                self._belief_state = jax.tree.map(
                    lambda old, new: jnp.where(_lane_mask(reset, old), new, old),
                    self._belief_state,
                    fresh,
                )
        self._update = self._belief_call(
            self._belief_state,
            facts.legal_observation,
            jnp.asarray(facts.previous_action, jnp.int32),
            jnp.asarray(seconds),
        )
        self._belief_state = self._update.state

    # ---------------------------------------------------------------- the events and the emergency flag

    def _advance_events(self, facts: Gen7DecisionFacts) -> None:
        """Publish the tick's events and the emergency flag, as the ADK core derives them (module docstring, Events)."""

        current = _as_learner_observation(facts.legal_observation)
        previous, self._previous_legal = self._previous_legal, current
        reset = np.asarray(facts.done, dtype=np.bool_).reshape(self.lanes)
        if current is None or previous is None:
            values = np.zeros((self.lanes, len(DecisionEvent)), np.bool_)
            known = np.zeros_like(values)
        else:
            perceptual_values, perceptual_known = _legal_perceptual_events(
                ActorPolicyInput(None, None, current, None),
                ActorPolicyInput(None, None, previous, None),
                self.lanes,
            )
            values = np.asarray(jax.device_get(perceptual_values), dtype=np.bool_)
            known = np.asarray(jax.device_get(perceptual_known), dtype=np.bool_)
            known = known & ~reset[:, None]
        self.event_values, self.event_known = values, known
        active = values & known
        self.event_lane_ticks += active.sum(axis=0)
        mask = np.asarray(self.goal.control.emergency_interrupt_mask, dtype=np.bool_)
        self.emergency_interrupt = self.goal_valid & np.any(active & mask, axis=-1)
        self._event_now = bool(active.any())

    # ---------------------------------------------------------------- the goal clock and termination

    def _terminations(self, tick: int, document: SharedState) -> np.ndarray:
        """bool[B]: lanes whose goal in force reported ``done`` on a status of this decision's window."""

        for request in document.requests.live:
            self._verb_of[(int(request.agent), int(request.request_id))] = request.verb
        done = np.zeros(self.lanes, dtype=np.bool_)
        current = np.asarray(self.goal.payload.skill_index)
        for status in document.skills.statuses:
            if not (self._status_tick <= status.tick < tick) or status.kind != "done":
                continue
            verb = self._verb_of.get((int(status.agent), int(status.request_id)))
            lane = int(status.agent)
            if (
                verb is not None
                and 0 <= lane < self.lanes
                and verb == self.skill_ids[int(current[lane])]
            ):
                done[lane] = True
        self._status_tick = tick
        return done & self.goal_valid

    # ---------------------------------------------------------------- the decision

    def decide(
        self,
        tick: int,
        document: SharedState,
        facts: Any,
        executive: ExecutiveState | None,
        /,
    ) -> None:
        gen7_facts = self.facts_for_decider(facts)
        self._advance_belief(tick, gen7_facts)
        self._advance_events(gen7_facts)
        completed = self._terminations(tick, document)
        self.terminations += int(completed.sum())
        if self._last_tick is None:
            self.goal_age = np.zeros(self.lanes, dtype=np.float32)
        else:
            self.goal_age = self.goal_age + np.float32(
                (tick - self._last_tick) * self.tick_seconds
            )
        previous_tick, self._last_tick = self._last_tick, tick
        if previous_tick is None or tick % self.cadence_ticks == 0 or self._event_now:
            self._executor_call(tick, document, gen7_facts, executive, completed)

    def _lowered(self, executive: ExecutiveState | None) -> Any:
        if executive is None:
            return self._arrays
        if self._arrays_version != executive.version:
            self._arrays = executive_arrays(executive, self.skill_ids, self.lanes)
            self._arrays_version = executive.version
        return self._arrays

    def _executor_call(
        self,
        tick: int,
        document: SharedState,
        facts: Gen7DecisionFacts,
        executive: ExecutiveState | None,
        completed: np.ndarray,
    ) -> None:
        self.executor_calls += 1
        update = self._update
        goal = self.goal
        legal = facts.legal_observation
        standins = _HierarchyState(
            goal_valid=jnp.asarray(self.goal_valid),
            goal_termination_requested=jnp.asarray(completed),
            manager_age_seconds=jnp.zeros((self.lanes,), dtype=jnp.float32),
            carry=_Carry(
                goal_age=jnp.asarray(self.goal_age),
                goal=goal,
                manager_state=self._lowered(executive),
            ),
        )
        transition = _Transition(
            hierarchy_step=_HierarchyStep(
                belief_update=update,
                goal_runtime=_GoalRuntime(
                    emergency_interrupt=jnp.asarray(self.emergency_interrupt)
                ),
            ),
            actor_input=ActorPolicyInput(None, None, legal, None),
            done=jnp.asarray(facts.done, dtype=jnp.bool_).reshape(self.lanes),
        )
        executor = self.executor
        executor.scratchpad = document
        executor.current_tick = tick
        executor.tick_seconds = self.tick_seconds
        executor.host_cadence_ticks = self.cadence_ticks
        executor.host_respects_commitment = True
        proposed = executor.decide(transition, standins, executive)
        self._apply(proposed)
        self._fill_positions(tick, document, update)

    def _apply(self, proposed: Any) -> None:
        """The ADK seam's rule (``session.py`` step 4): a changed goal is written where the lane may switch."""

        current = self.goal.payload
        new = proposed.payload
        skill = np.asarray(current.skill_index)
        differs = (
            (np.asarray(new.skill_index) != skill)
            | (np.asarray(new.goal_slot) != np.asarray(current.goal_slot))
            | np.any(np.asarray(new.params) != np.asarray(current.params), axis=-1)
        )
        age = self.goal_age
        emergency = self.emergency_interrupt
        by_time = (
            ~self.goal_valid
            | (age >= self._minimum[skill])
            | (age >= self._maximum[skill])
        )
        allowed = by_time | emergency
        apply = differs & allowed
        self.goal_held += int((differs & ~allowed).sum())
        self.emergency_writes += int((apply & ~by_time).sum())
        if apply.any():
            mask = jnp.asarray(apply)
            self.goal = jax.tree.map(
                lambda n, o: jnp.where(_lane_mask(mask, o), n, o), proposed, self.goal
            )
            self.goal_age = np.where(apply, np.float32(0.0), self.goal_age).astype(
                np.float32
            )
            self.goal_valid = self.goal_valid | apply
            self.goal_writes += int(apply.sum())

    def _fill_positions(self, tick: int, document: SharedState, update: Any) -> None:
        """Withhold idle's requests and walk/look requests with no point, and give the rest their point (item 6)."""

        entities = {record.agent: record for record in document.skills.known_entities}
        memory = host_target_memory(update)
        changed = False
        live = []
        for request in document.requests.live:
            if (
                request.verb == self.skill_ids[0]
            ):  # idle: the library has no idle verb, so no request is sent
                self.idle_withheld += 1
                changed = True
                continue
            if request.verb in POSITION_VERBS and request.args.position is None:
                record = entities.get(request.agent)
                chosen = None if record is None else choose_target(record.entities)
                fill = None
                if chosen is not None:
                    fill = chosen.position
                elif memory is not None and bool(memory.known[request.agent]):
                    fill = tuple(float(v) for v in memory.position[request.agent])
                changed = True
                if fill is None:
                    self.positions_withheld += 1
                    continue
                self.positions_filled += 1
                if request.verb == "look_at":
                    fill = (float(fill[0]), float(fill[1]) + LOOK_EYE_HEIGHT, float(fill[2]))
                request = dataclasses.replace(
                    request,
                    args=dataclasses.replace(request.args, position=tuple(fill)),
                )
            live.append(request)
        if changed:
            document.write(
                "requests",
                RequestsSection(live=tuple(live), limit=document.requests.limit),
                writer="jev",
                tick=tick,
            )

    def counters(self) -> dict[str, Any]:
        """The executor's counters plus the adapter's."""

        return {
            **self.executor.counters(),
            "executor_calls": self.executor_calls,
            "goal_writes": self.goal_writes,
            "goal_held": self.goal_held,
            "emergency_writes": self.emergency_writes,
            "event_lane_ticks": {
                event.name: int(count)
                for event, count in zip(DecisionEvent, self.event_lane_ticks)
            },
            "terminations": self.terminations,
            "positions_filled": self.positions_filled,
            "positions_withheld": self.positions_withheld,
            "idle_withheld": self.idle_withheld,
        }


__all__ = [
    "CADENCE_SECONDS",
    "POSITION_VERBS",
    "Gen7DecisionFacts",
    "Gen7ProgramDecider",
]
