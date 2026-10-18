"""The exact typed path of the middle layer: typed conditions evaluated on the belief, the priority rule.

docs/SYSTEM_ONE.md section 9.12 decided that the layer acts on typed fields: the skill id selects, the
typed condition says when. This module is that path with nothing learned in it. ``TypedRuleDecider`` is a
``session.HostDecider`` (the shape ``jev/programs.py`` ProgramExecutor has): the session calls ``decide`` at
its host cadence and applies the returned goal per lane where the skill's commitment allows a switch
(``SessionConfig.host_respects_commitment``). The ``jtyped`` arm of ``eval/arms.py`` runs it through the
composition driver.

What is read, once per executive version (``read``), for goal slot k:
    typed    goal.condition is set              -> Predicate(kind, argument)                        confidence 1
    label    a ``constraints`` label that is a non-numeric condition kind (the legacy hint, for example
             "on_damage" in tasks/composition.py)  -> Predicate(label)                            confidence 1
    words    a scratchpad is bound and its executive section holds this version's ``when`` text
             -> condition_rules.read_condition(when).predicate(); below CLEAR it is also an information
             request (``information_requests``); INVALID (a numeric kind without a number) makes the slot
             ineligible until the LLM writes the typed field
    always   otherwise                          -> ALWAYS
The scratchpad's words are consulted only when neither the typed field nor a label is present: the typed
fields are the contract, the words the fallback.

What is decided per host decision (``decide``), per lane b, exactly ``executive/goal_language.teacher``
(the rule M13 of section 8.11):
    conditional_b = { k : valid_bk, predicate_k != always, predicate_k(B_bt) }
    pool_b        = conditional_b if not empty else { k : valid_bk, predicate_k == always }
    chosen_b      = argmax_{k in pool_b} (priority_bk, -k);  idle when pool_b is empty
    the lane's goal is rewritten only when the chosen skill differs from the one in force: skill, the goal's
    params, the lowered E_t's version, slot k. Belief tick columns are published as counter / TICKS_SATURATION
    and multiplied back before a predicate reads them (as the executor does).
A slot whose predicate is None (ineligible) is in neither set. Counters: ``counters()``.

On tasks/composition.py's E_t (pursuit always, evader with the "on_damage" label) this is J0's selection rule;
the commitment (evader 2 s) is the skill contract's, applied by the session's seam.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from experimental.neural_agent.contracts import (
    BELIEF_FEATURE_COUNT,
    TICKS_SATURATION,
    ExecutiveArrays,
    ExecutiveState,
    GoalSpec,
    ManagedGoal,
    SkillControlTable,
    control_for,
    make_goal,
)
from experimental.neural_agent.executive.goal_language import ALWAYS, Predicate
from experimental.neural_agent.jev.programs import TICK_COLUMNS
from experimental.neural_agent.jev.system_one.condition_rules import CLEAR, read_condition
from experimental.neural_agent.jev.system_one.interpret import KINDS, NUMERIC_UNITS

JEV_NAME: str = "jtyped"

SOURCE_TYPED, SOURCE_LABEL, SOURCE_WORDS, SOURCE_ALWAYS = "typed", "label", "words", "always"


@dataclass(frozen=True)
class SlotReading:
    """What one goal slot's condition was read as.

    Attributes:
        slot: The goal's index in E_t.
        predicate: The exact predicate, or None when the slot is ineligible (an INVALID words reading).
        source: SOURCE_TYPED, SOURCE_LABEL, SOURCE_WORDS or SOURCE_ALWAYS.
        confidence: 1.0 unless the reading came from words (then ``condition_rules`` confidence).
        when: The words that were read, or "".
    """

    slot: int
    predicate: Predicate | None
    source: str
    confidence: float
    when: str = ""
    executive_version: int = 0


def read_goal(slot: int, goal: GoalSpec, when: str = "") -> SlotReading:
    """Read one goal's condition by the order of the module docstring: typed, label, words, always."""

    if goal.condition is not None:
        return SlotReading(slot, Predicate(goal.condition.kind, goal.condition.argument), SOURCE_TYPED, CLEAR, when)
    for label in goal.constraints:
        if label in KINDS and label not in NUMERIC_UNITS and label != ALWAYS.kind:
            return SlotReading(slot, Predicate(label), SOURCE_LABEL, CLEAR, when)
    if when.strip():
        reading = read_condition(when)
        return SlotReading(slot, reading.predicate(), SOURCE_WORDS, reading.confidence, when)
    return SlotReading(slot, ALWAYS, SOURCE_ALWAYS, CLEAR, when)


def typed_choice(
    predicates: Sequence[Predicate | None],
    priorities: Sequence[float],
    valid: Sequence[bool],
    features: Sequence[float],
    known: Sequence[bool],
) -> int:
    """Return the chosen slot by the rule of the module docstring, or -1 for idle.

    ``features`` is one belief row in physical units (ticks in ticks), ``known`` its flags; a slot whose
    predicate is None is ineligible.
    """

    conditional = [
        k for k, predicate in enumerate(predicates)
        if valid[k] and predicate is not None and predicate.kind != ALWAYS.kind and predicate.evaluate(features, known)
    ]
    pool = conditional or [
        k for k, predicate in enumerate(predicates) if valid[k] and predicate is not None and predicate.kind == ALWAYS.kind
    ]
    if not pool:
        return -1
    return max(pool, key=lambda k: (float(priorities[k]), -k))


class TypedRuleDecider:
    """The exact typed path as a host decider (module docstring).

    Args:
        skill_ids: The catalogue in dispatcher order, idle first.
        control: The skill contracts' control table (``contracts.control_table`` / ``default_control_table``).
        executive: E_t at the start; later versions arrive through ``decide``'s ``executive`` argument.
        batch: Number of lanes B.

    Attributes:
        scratchpad: Bound by the session when it runs with a shared state; read for the goals' words.
        current_tick: Bound by the session before each decision.
        information_requests: One dict per words reading below CLEAR: version, slot, when, kind, argument,
            confidence. Forwarding them to the LLM is the executive side's (section 9.12).
    """

    name: str = JEV_NAME

    def __init__(self, skill_ids: Sequence[str], control: SkillControlTable, executive: ExecutiveState, batch: int) -> None:
        self._skill_ids = tuple(str(s) for s in skill_ids)
        if not self._skill_ids:
            raise ValueError("skill_ids must not be empty")
        if not isinstance(control, SkillControlTable):
            raise TypeError("control must be a SkillControlTable")
        if not isinstance(executive, ExecutiveState):
            raise TypeError("executive must be an ExecutiveState")
        if not isinstance(batch, int) or isinstance(batch, bool) or batch < 1:
            raise ValueError(f"batch must be a positive int, got {batch!r}")
        self._control = control
        self._lanes = batch
        self.scratchpad: Any = None
        self.current_tick: int | None = None
        self.decisions = 0
        self.lane_calls = 0
        self.switches = 0
        self.versions_read = 0
        self.unsure = 0
        self.ineligible = 0
        self.version_mismatches = 0
        self.information_requests: list[dict[str, Any]] = []
        self._executive: ExecutiveState | None = None
        self._readings: tuple[SlotReading, ...] = ()
        self.read(executive)

    @property
    def readings(self) -> tuple[SlotReading, ...]:
        """The slot readings of the executive version in force."""

        return self._readings

    def read(self, executive: ExecutiveState) -> None:
        """Read E_t's goals (module docstring); the scratchpad's words are used when they are this version's."""

        if not isinstance(executive, ExecutiveState):
            raise TypeError("executive must be an ExecutiveState")
        words: tuple[str, ...] = ("",) * len(executive.goals)
        section = None if self.scratchpad is None else getattr(self.scratchpad, "executive", None)
        if section is not None and section.version == executive.version and len(section.goals) == len(executive.goals):
            words = tuple(str(getattr(goal, "when", "")) for goal in section.goals)
        readings = tuple(
            dataclasses.replace(read_goal(k, goal, when), executive_version=executive.version)
            for k, (goal, when) in enumerate(zip(executive.goals, words))
        )
        for reading in readings:
            if reading.source == SOURCE_WORDS and reading.confidence < CLEAR:
                self.unsure += 1
                predicate = reading.predicate
                self.information_requests.append({
                    "version": executive.version, "slot": reading.slot, "when": reading.when,
                    "kind": None if predicate is None else predicate.kind,
                    "argument": None if predicate is None else predicate.argument,
                    "confidence": reading.confidence,
                })
            if reading.predicate is None:
                self.ineligible += 1
        self._readings = readings
        self._executive = executive
        self.versions_read += 1

    def decide(self, transition: Any, hierarchy_state: Any, executive: Any, /) -> ManagedGoal:
        """Return the batched goal for all B lanes (module docstring). ``executive``, when new, is read first.

        Raises:
            TypeError: If ``carry.manager_state`` is not an ExecutiveArrays.
            ValueError: If the belief does not have B rows of BELIEF_FEATURE_COUNT features.
        """

        if isinstance(executive, ExecutiveState) and executive is not self._executive:
            self.read(executive)
        lanes = self._lanes
        belief = transition.hierarchy_step.belief_update.belief
        features = np.array(belief.features, dtype=np.float32)
        if features.shape != (lanes, BELIEF_FEATURE_COUNT):
            raise ValueError(f"belief.features must have shape ({lanes}, {BELIEF_FEATURE_COUNT}), got {features.shape}")
        features[:, list(TICK_COLUMNS)] *= np.float32(TICKS_SATURATION)
        known = np.asarray(belief.feature_known, dtype=np.bool_)
        carry = hierarchy_state.carry
        arrays = carry.manager_state
        if not isinstance(arrays, ExecutiveArrays):
            raise TypeError("carry.manager_state must be the lowered executive (ExecutiveArrays)")
        skill_index = np.asarray(arrays.skill_index, dtype=np.int32)
        priority = np.asarray(arrays.priority, dtype=np.float32)
        goal_params = np.asarray(arrays.params, dtype=np.float32)
        goal_valid = np.asarray(arrays.goal_valid, dtype=np.bool_)
        executive_version = np.asarray(arrays.version, dtype=np.int32)
        payload = carry.goal.payload
        skill = np.array(payload.skill_index, dtype=np.int32)
        params = np.array(payload.params, dtype=np.float32)
        version = np.array(payload.executive_version, dtype=np.int32)
        slot = np.array(payload.goal_slot, dtype=np.int32)
        slots = skill_index.shape[1]
        predicates = [self._readings[k].predicate if k < len(self._readings) else None for k in range(slots)]
        read_version = None if self._executive is None else self._executive.version
        for lane in range(lanes):
            if read_version is not None and int(executive_version[lane]) != read_version:
                self.version_mismatches += 1
            valid = [bool(goal_valid[lane, k]) and k < len(self._readings) for k in range(slots)]
            k = typed_choice(predicates, priority[lane], valid, features[lane], known[lane])
            desired = 0 if k < 0 else int(skill_index[lane, k])
            if desired != int(skill[lane]):
                self.switches += 1
                skill[lane] = desired
                params[lane] = 0.0 if k < 0 else goal_params[lane, k]
                version[lane] = executive_version[lane]
                slot[lane] = k
        self.decisions += 1
        self.lane_calls += lanes
        minimum, maximum, mask = control_for(self._control, jnp.asarray(skill))
        return make_goal(
            jnp.asarray(skill),
            jnp.asarray(params),
            jnp.asarray(version),
            jnp.asarray(slot),
            minimum_commitment_seconds=minimum,
            maximum_duration_seconds=maximum,
            emergency_interrupt_mask=mask,
        )

    def counters(self) -> dict[str, Any]:
        """The decider's counters as plain ints, for the composition driver's stderr line."""

        return {
            "decisions": self.decisions,
            "lane_calls": self.lane_calls,
            "switches": self.switches,
            "versions_read": self.versions_read,
            "unsure": self.unsure,
            "ineligible": self.ineligible,
            "version_mismatches": self.version_mismatches,
            "information_requests": len(self.information_requests),
            "sources": [reading.source for reading in self._readings],
        }


__all__ = [
    "JEV_NAME",
    "SOURCE_ALWAYS",
    "SOURCE_LABEL",
    "SOURCE_TYPED",
    "SOURCE_WORDS",
    "SlotReading",
    "TypedRuleDecider",
    "read_goal",
    "typed_choice",
]
