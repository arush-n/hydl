"""Behaviour programs: a short series of skill steps, executed exactly on the host fast path.

SYSTEM_ONE.md section 9.9 (owner principle, 2026-10-17 night): the middle layer "gives a series of neural
actions". A program is up to L = 4 steps, each a skill, an exit condition and a hold time, plus the LLM's
conditional goals as interrupts. Before any learned program head exists, :class:`ProgramExecutor` runs
hand-written programs exactly, one per lane, as a ``session.HostDecider``. No learning: numpy on the host,
then ``make_goal`` and ``control_for`` for the emitted goal.

Execution, for lane b at every host decision (the session decides every ``host_cadence_ticks`` steps and on
every step with an active event, ``session.run_session`` step 4):

1. Clock. ``dt_b = carry.goal_age[b]``. The executor writes a goal on every lane at every decision, and the
   session sets every lane's goal age to 0 when it accepts a write, so ``dt_b`` is the virtual time since the
   previous decision, summed from the per-step ``delta_seconds`` exactly as the core's goal age and the
   belief's tick counters are. (Measured 2026-10-17 on ``combat_open_flat``: the per-step delta is 0.045 s
   or 0.0333 s and differs by lane, so ``ticks x 0.045`` would be wrong.) The lane's goal age ``age_b``, its
   time in the current step ``elapsed_b`` (frozen while an interrupt holds) and its no-reissue windows
   advance by ``dt_b``. The clocks are float32, the dtype of the core's goal age: with a host decision on
   every step, ``fl32(age_b + dt_b)`` is the core's own accumulation bit for bit. With fewer decisions the
   sum of interval sums rounds differently, which matters at an exact boundary: 60 steps of fl32(1/30) s
   accumulate to 1.9999994 in the core but to 2.0 here (measured 2026-10-17), so on such a lane J0
   refuses the evader's release at its 2.0 s cadence tick and this executor allows it. A write the session
   rejects as a miss is not seen here: the next ``dt_b`` then also holds the missed interval.
2. Reset. A lane is fresh, and restarts its program at step 0 with every clock at 0, before its first
   decision, when its episode ended (``transition.done``), when its goal is not valid, or when the goal in
   force carries another executive version than the executor wrote (the autoreset's idle goal, version 0).
   Cadence. A lane decides (items 4 to 6) only at a cadence tick of the decision layer: when it is fresh,
   when its goal ended (item 3), or when the ADK core's manager was due on it since the previous host
   decision, ``m_b < dt_b`` with ``m_b = hierarchy_state.manager_age_seconds[b]``. The core keeps that
   schedule for the hold manager by the rule it applies to J0 (``adk/architecture/decision_context.py``
   ``manager_decision_due``: 0.5 s since the lane's last manager decision, or a known event on the lane).
   The session decides more often (every 11 steps and on any lane's event), so without this gate the
   executor acted between the manager's ticks: measured 2026-10-17 (seed 0, batch 8, 512 ticks), J0's
   evasions lasted 52 or 75 steps, because J0 is only asked at 0.5 s ticks and a tick at age 1.99999 s
   refuses the switch, while the ungated executor's lasted 52 to 64 steps. At a cadence tick the goal's age
   is read at the due step, ``age_b - m_b``, for the commitment and the maximum duration of item 6. A lane
   that does not decide re-emits its goal; its clocks still advance.
3. End of the goal in force. COMPLETED when the skill asked to end it (``goal_termination_requested``;
   the current dispatcher never sets it, ``skill_library/dispatcher.py``). TIMEOUT when ``age_b`` reaches
   the skill contract's maximum duration: on this path the core never times a goal out itself, because the
   session's writes reset its goal age.
4. Exit of the current step k, while the program is not suspended by an interrupt::

       fire_b = elapsed_b >= hold_seconds_k  and  ( exit_k = "timer"
                                                  or exit_k = "done" and the skill reported COMPLETED
                                                  or exit_k is a Predicate P and P(B_t^phys)_b )

   On fire, k <- k + 1 and elapsed_b <- 0, at most one step per decision; k = len(steps) is finished.
   ``B_t^phys`` is the belief row with the two tick columns (``ticks_since_damage``,
   ``ticks_since_target_seen``) multiplied by ``TICKS_SATURATION``, because the belief publishes
   ``counter / TICKS_SATURATION`` (``executive/goal_language.py`` units note; J0 does the same,
   ``jev/scripted.py``). ``Predicate.evaluate`` gives the predicate's value: False when its feature is unknown.
5. Interrupts, by (M13): ``S_b = {i : P_i(B_t^phys)_b and skill_i not blocked}``. If ``S_b`` is non-empty
   the desired goal is ``argmax_{i in S_b} priority_i`` (ties to the lower index) and the program is
   suspended; otherwise it is step k's skill, idle when that skill is blocked or the program is finished.
6. Switch rule. The session writes host goals past the ADK core's own rule, so the executor applies that
   rule here (``adk/architecture/hierarchy.py``, ``switch_allowed``; FORMULAS (D9) gives J0's goals the same
   control). A lane whose desired skill differs from its current skill changes only if::

       fresh_b or age_b >= min_commitment(current) or emergency_b or age_b >= max_duration(current)

   ``emergency_b`` is the core's flag for the step just taken: an active event in the current goal's
   emergency mask. As in J0 (terminate = chosen skill != current skill), a lane whose desired skill equals
   its current skill keeps its goal, and an ended goal (step 3) is renewed.
7. Emit, on every lane: ``make_goal(skill, params, version, slot, *control_for(table, skill))``. A lane that
   changes takes E_t's version (``ExecutiveArrays`` in ``carry.manager_state``, which ``HoldGoalManager``
   keeps), the program's params zero-padded to PARAM_WIDTH, and the highest-priority valid executive slot
   that names the skill (-1 if none, as for idle). A lane that keeps its goal re-emits the payload in force.

Feedback rule (owner: "the neural actions feed back into the decision state"), ``no_reissue_seconds`` w:
when the goal in force ends with TIMEOUT or FAILED and w > 0, its skill is blocked on that lane for w
seconds: no step and no interrupt issues it. If the goal came from step k and a step k + 1 exists, the
program takes step k + 1; otherwise it holds step k and idles until the block expires. With w = 0 the ended
goal is renewed, which is what J0 gets from the core at a forced switch. FAILED has no producer on this
path today (``session._outcome_for_switch`` never reports it).

P_J0 is one step, pursuit until "done", with the interrupt evader while ``on_damage`` at the evader's
priority. On the composition executive it chooses what J0 chooses (tests/.../test_na_programs.py, 200 rows).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np

from adk.architecture.hierarchy import ManagedGoal
from experimental.neural_agent.contracts import (
    BELIEF_FEATURE_COUNT,
    PARAM_WIDTH,
    SKILL_IDLE,
    TICKS_SATURATION,
    ExecutiveArrays,
    SkillControlTable,
    SkillOutcome,
    _require_control_table,
    control_for,
    feature_index,
    make_goal,
)
from experimental.neural_agent.executive.goal_language import ALWAYS, Predicate
from experimental.neural_agent.tasks.composition import (
    EVADER_CONSTRAINT,
    EVADER_PRIORITY,
    EVADER_SKILL,
    PURSUIT_SKILL,
)

#: Steps in one program at most (SYSTEM_ONE.md section 9.9, L = 4).
MAX_PROGRAM_STEPS: int = 4
#: Exit kinds besides a Predicate: the skill reported completion, or the step's hold time elapsed.
EXIT_DONE: str = "done"
EXIT_TIMER: str = "timer"
#: Belief columns published as counter / TICKS_SATURATION; multiplied back before a predicate reads them.
TICK_COLUMNS: tuple[int, ...] = (
    feature_index("ticks_since_damage"),
    feature_index("ticks_since_target_seen"),
)

_PROGRAM, _INTERRUPT = 0, 1


def _params(values: Any, owner: str) -> tuple[float, ...]:
    """Return ``values`` as at most PARAM_WIDTH finite floats."""

    items = tuple(float(value) for value in values)
    if len(items) > PARAM_WIDTH or not all(math.isfinite(value) for value in items):
        raise ValueError(f"{owner}: params must be at most {PARAM_WIDTH} finite numbers, got {items}")
    return items


def _skill_id(value: Any, owner: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner}: skill_id must be a non-empty string, got {value!r}")
    return value


@dataclass(frozen=True)
class ProgramStep:
    """One step: run ``skill_id`` with ``params`` until ``exit`` fires, after at least ``hold_seconds``.

    ``exit`` is a ``Predicate`` over the belief, ``"done"`` (the skill reported completion) or ``"timer"``
    (``hold_seconds`` elapsed). See the module docstring, item 4.
    """

    skill_id: str
    params: tuple[float, ...] = ()
    exit: Predicate | str = EXIT_DONE
    hold_seconds: float = 0.0

    def __post_init__(self) -> None:
        _skill_id(self.skill_id, "ProgramStep")
        object.__setattr__(self, "params", _params(self.params, "ProgramStep"))
        if not isinstance(self.exit, Predicate) and self.exit not in (EXIT_DONE, EXIT_TIMER):
            raise ValueError(f"ProgramStep.exit must be a Predicate, {EXIT_DONE!r} or {EXIT_TIMER!r}")
        hold = float(self.hold_seconds)
        if not math.isfinite(hold) or hold < 0.0:
            raise ValueError(f"ProgramStep.hold_seconds must be finite and >= 0, got {self.hold_seconds!r}")
        object.__setattr__(self, "hold_seconds", hold)


@dataclass(frozen=True)
class Interrupt:
    """A conditional goal of the executive: ``skill_id`` pre-empts the program while ``predicate`` holds."""

    skill_id: str
    priority: float
    predicate: Predicate
    params: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        _skill_id(self.skill_id, "Interrupt")
        if not isinstance(self.predicate, Predicate) or self.predicate == ALWAYS:
            raise ValueError("Interrupt.predicate must be a Predicate other than 'always'")
        priority = float(self.priority)
        if not math.isfinite(priority):
            raise ValueError(f"Interrupt.priority must be finite, got {self.priority!r}")
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "params", _params(self.params, "Interrupt"))


@dataclass(frozen=True)
class Program:
    """1 to MAX_PROGRAM_STEPS steps run in order, and the interrupts that pre-empt them."""

    steps: tuple[ProgramStep, ...]
    interrupts: tuple[Interrupt, ...] = ()

    def __post_init__(self) -> None:
        steps, interrupts = tuple(self.steps), tuple(self.interrupts)
        if not 1 <= len(steps) <= MAX_PROGRAM_STEPS:
            raise ValueError(f"a program has 1 to {MAX_PROGRAM_STEPS} steps, got {len(steps)}")
        if not all(isinstance(step, ProgramStep) for step in steps):
            raise TypeError("Program.steps must hold ProgramStep values")
        if not all(isinstance(interrupt, Interrupt) for interrupt in interrupts):
            raise TypeError("Program.interrupts must hold Interrupt values")
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "interrupts", interrupts)


#: The composition executive's conditional goal: evade while damaged, at the evader's priority.
EVADE_ON_DAMAGE = Interrupt(EVADER_SKILL, EVADER_PRIORITY, Predicate(EVADER_CONSTRAINT))
#: J0 as a program: pursue (no exit fires within a horizon), evade on damage.
P_J0 = Program(steps=(ProgramStep(PURSUIT_SKILL, exit=EXIT_DONE),), interrupts=(EVADE_ON_DAMAGE,))
#: Sequencing and resumption: pursue until within 3 blocks, idle 1 s, pursue again; evade on damage.
P_SEQ = Program(
    steps=(
        ProgramStep(PURSUIT_SKILL, exit=Predicate("distance_below", 3.0)),
        ProgramStep(SKILL_IDLE, exit=EXIT_TIMER, hold_seconds=1.0),
        ProgramStep(PURSUIT_SKILL, exit=EXIT_DONE),
    ),
    interrupts=(EVADE_ON_DAMAGE,),
)
PROGRAMS: dict[str, Program] = {"J0": P_J0, "SEQ": P_SEQ}


class ProgramExecutor:
    """A ``session.HostDecider`` that runs one program per lane (module docstring, items 1 to 7).

    Args:
        skill_ids: The catalogue in dispatcher order, ``"idle"`` first.
        control_table: The catalogue's ``SkillControlTable`` (the skills' contracts).
        programs_per_lane: One :class:`Program` per lane; B is its length.
        no_reissue_seconds: Keyword only. The feedback rule's window w in seconds; 0 turns it off.

    Attributes:
        lane_calls: Lane decisions made, B per call. ``lane_failures`` is always 0 and ``first_failure``
            always None: there is no backend to fail. ``eval/run_composition.py`` reads all three.

    Raises:
        ValueError: If the catalogue is empty, repeats an id or does not start with idle, a program is
            missing, or ``no_reissue_seconds`` is negative or not finite.
        KeyError: If a step or interrupt names a skill outside the catalogue.
        TypeError: If a program is not a Program, or the table is not a SkillControlTable.
    """

    name: str = "J-PROG"

    def __init__(
        self,
        skill_ids: Sequence[str],
        control_table: SkillControlTable,
        programs_per_lane: Sequence[Program],
        *,
        no_reissue_seconds: float = 0.0,
    ) -> None:
        ids = tuple(skill_ids)
        if not ids or ids[0] != SKILL_IDLE or len(set(ids)) != len(ids):
            raise ValueError(f"skill_ids must be distinct ids starting with {SKILL_IDLE!r}, got {ids}")
        self._control = _require_control_table(control_table, len(ids))
        programs = tuple(programs_per_lane)
        if not programs:
            raise ValueError("programs_per_lane must hold one Program per lane")
        if not all(isinstance(program, Program) for program in programs):
            raise TypeError("programs_per_lane must hold Program values")
        window = float(no_reissue_seconds)
        if not math.isfinite(window) or window < 0.0:
            raise ValueError(f"no_reissue_seconds must be finite and >= 0, got {no_reissue_seconds!r}")
        index = {skill_id: position for position, skill_id in enumerate(ids)}

        def lookup(skill_id: str) -> int:
            if skill_id not in index:
                raise KeyError(f"skill {skill_id!r} is not in the catalogue {ids}")
            return index[skill_id]

        self._programs = programs
        self._step_skill = [tuple(lookup(step.skill_id) for step in p.steps) for p in programs]
        self._interrupt_skill = [tuple(lookup(i.skill_id) for i in p.interrupts) for p in programs]
        self._used_skills = sorted(
            {0} | {k for row in self._step_skill for k in row} | {k for row in self._interrupt_skill for k in row}
        )
        self._minimum = np.asarray(self._control.minimum_commitment_seconds, dtype=np.float32)
        self._maximum = np.asarray(self._control.maximum_duration_seconds, dtype=np.float32)
        self._no_reissue = window
        lanes = len(programs)
        self._lanes = lanes
        self._started = np.zeros(lanes, dtype=np.bool_)
        self._step = np.zeros(lanes, dtype=np.int64)
        self._elapsed = np.zeros(lanes, dtype=np.float32)
        self._age = np.zeros(lanes, dtype=np.float32)
        self._source = np.full(lanes, _PROGRAM, dtype=np.int64)
        self._blocked = np.zeros((lanes, len(ids)), dtype=np.float32)
        self._issued_version = np.full(lanes, -1, dtype=np.int64)
        longest = max(len(p.steps) for p in programs)
        self.step_completions = np.zeros((lanes, longest), dtype=np.int64)
        self.steps_skipped = np.zeros(lanes, dtype=np.int64)
        self.programs_finished = np.zeros(lanes, dtype=np.int64)
        self.interrupts = np.zeros(lanes, dtype=np.int64)
        self.resumes = np.zeros(lanes, dtype=np.int64)
        self.timeouts = np.zeros(lanes, dtype=np.int64)
        self.blocked_decisions = np.zeros(lanes, dtype=np.int64)
        self.restarts = np.zeros(lanes, dtype=np.int64)
        self.decisions = 0
        self.lane_calls = 0
        self.lane_failures = 0
        self.first_failure: str | None = None

    # ------------------------------------------------------------------ lane state, for tests and reports

    @property
    def step_index(self) -> np.ndarray:
        """int[B]: each lane's current step; ``len(steps)`` once its program is finished."""

        return self._step.copy()

    @property
    def interrupted(self) -> np.ndarray:
        """bool[B]: True where an interrupt's goal is in force and the program is suspended."""

        return self._source == _INTERRUPT

    @property
    def step_elapsed_seconds(self) -> np.ndarray:
        """float[B]: seconds spent in the current step, interrupts excluded."""

        return self._elapsed.copy()

    def counters(self) -> dict[str, Any]:
        """Return the per-lane counters as JSON-ready lists."""

        return {
            "decisions": self.decisions,
            "step_completions": self.step_completions.tolist(),
            "steps_skipped": self.steps_skipped.tolist(),
            "programs_finished": self.programs_finished.tolist(),
            "interrupts": self.interrupts.tolist(),
            "resumes": self.resumes.tolist(),
            "timeouts": self.timeouts.tolist(),
            "blocked_decisions": self.blocked_decisions.tolist(),
            "restarts": self.restarts.tolist(),
        }

    # ------------------------------------------------------------------ the host decision

    def decide(self, transition: Any, hierarchy_state: Any, executive: Any, /) -> ManagedGoal:
        """Advance every lane's program and return the batched goal for all B lanes.

        ``executive`` is not read: E_t is read from ``hierarchy_state.carry.manager_state``, where the
        session's executive writer puts every new version.

        Raises:
            TypeError: If ``carry.manager_state`` is not an ExecutiveArrays.
            ValueError: If the belief does not have B rows of BELIEF_FEATURE_COUNT features.
        """

        del executive
        lanes = self._lanes
        step = transition.hierarchy_step
        belief = step.belief_update.belief
        features = np.array(belief.features, dtype=np.float32)
        if features.shape != (lanes, BELIEF_FEATURE_COUNT):
            raise ValueError(f"belief.features must have shape ({lanes}, {BELIEF_FEATURE_COUNT}), got {features.shape}")
        features[:, list(TICK_COLUMNS)] *= np.float32(TICKS_SATURATION)
        known = np.asarray(belief.feature_known, dtype=np.bool_)
        emergency = np.asarray(step.goal_runtime.emergency_interrupt, dtype=np.bool_)
        done = np.asarray(transition.done, dtype=np.bool_)
        valid = np.asarray(hierarchy_state.goal_valid, dtype=np.bool_)
        completed = np.asarray(hierarchy_state.goal_termination_requested, dtype=np.bool_)
        manager_age = np.asarray(hierarchy_state.manager_age_seconds, dtype=np.float32)
        carry = hierarchy_state.carry
        dt = np.asarray(carry.goal_age, dtype=np.float32)
        payload = carry.goal.payload
        skill = np.array(payload.skill_index, dtype=np.int32)
        params = np.array(payload.params, dtype=np.float32)
        version = np.array(payload.executive_version, dtype=np.int32)
        slot = np.array(payload.goal_slot, dtype=np.int32)
        arrays = carry.manager_state
        if not isinstance(arrays, ExecutiveArrays):
            raise TypeError("carry.manager_state must be the lowered executive (ExecutiveArrays)")
        executive_version = np.asarray(arrays.version, dtype=np.int32)
        slots = self._executive_slots(arrays)
        fresh = ~self._started | done | ~valid | (version != self._issued_version)

        for lane in range(lanes):
            program = self._programs[lane]
            steps = program.steps
            ended = self._advance_clocks(lane, bool(fresh[lane]), dt[lane], int(skill[lane]), bool(completed[lane]))
            ticked = manager_age[lane] < dt[lane]  # the core's manager was due on this lane since the last decision
            if not (fresh[lane] or ended is not None or ticked):
                continue
            # The goal's age at the due step (module docstring, item 2).
            age = self._age[lane] - (manager_age[lane] if ticked and ended is None and not fresh[lane] else 0.0)
            advanced = ended is not None and self._on_goal_end(lane, ended, int(skill[lane]))
            k = int(self._step[lane])
            if not advanced and self._source[lane] == _PROGRAM and k < len(steps):
                current = steps[k]
                if self._elapsed[lane] >= current.hold_seconds and (
                    current.exit == EXIT_TIMER
                    or (isinstance(current.exit, Predicate) and current.exit.evaluate(features[lane], known[lane]))
                ):
                    self._advance(lane, len(steps))

            desired, desired_params, source = self._desired(lane, features[lane], known[lane])
            current_skill = int(skill[lane])
            if fresh[lane] or ended is not None or desired != current_skill:
                allowed = (
                    fresh[lane]
                    or age >= self._minimum[current_skill]
                    or emergency[lane]
                    or age >= self._maximum[current_skill]
                )
                if not allowed:
                    continue
                skill[lane] = desired
                params[lane] = 0.0
                params[lane, : len(desired_params)] = desired_params
                version[lane] = executive_version[lane]
                slot[lane] = slots[desired][lane]
                self._age[lane] = 0.0
            self._set_source(lane, source)

        self._started[:] = True
        self._issued_version[:] = version
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

    # ------------------------------------------------------------------ helpers, one lane at a time

    def _advance_clocks(
        self, lane: int, fresh: bool, dt: float, current_skill: int, completed: bool
    ) -> SkillOutcome | None:
        """Restart a fresh lane, else advance its clocks by ``dt`` and return how its goal ended, if it did."""

        if fresh:
            if self._started[lane]:
                self.restarts[lane] += 1
            self._step[lane] = 0
            self._elapsed[lane] = 0.0
            self._age[lane] = 0.0
            self._source[lane] = _PROGRAM
            self._blocked[lane] = 0.0
            return None
        self._age[lane] += dt
        np.maximum(self._blocked[lane] - dt, 0.0, out=self._blocked[lane])
        if self._source[lane] == _PROGRAM:
            self._elapsed[lane] += dt
        if completed and current_skill != 0:
            return SkillOutcome.COMPLETED
        if self._age[lane] >= self._maximum[current_skill]:
            self.timeouts[lane] += 1
            return SkillOutcome.TIMEOUT
        return None

    def _on_goal_end(self, lane: int, ended: SkillOutcome, current_skill: int) -> bool:
        """Apply the end of the goal in force (module docstring, item 3). Return True when the program advanced."""

        steps = self._programs[lane].steps
        k = int(self._step[lane])
        # The goal in force came from step k: the program is not suspended and it runs step k's skill.
        on_step = self._source[lane] == _PROGRAM and k < len(steps) and self._step_skill[lane][k] == current_skill
        if ended is SkillOutcome.COMPLETED:
            if on_step and steps[k].exit == EXIT_DONE and self._elapsed[lane] >= steps[k].hold_seconds:
                self._advance(lane, len(steps))
                return True
            return False
        if self._no_reissue > 0.0:  # TIMEOUT or FAILED: the feedback rule
            self._blocked[lane, current_skill] = self._no_reissue
            if on_step and k + 1 < len(steps):
                self._step[lane] = k + 1
                self._elapsed[lane] = 0.0
                self.steps_skipped[lane] += 1
                return True
        return False

    def _advance(self, lane: int, step_count: int) -> None:
        k = int(self._step[lane])
        self.step_completions[lane, k] += 1
        self._step[lane] = k + 1
        self._elapsed[lane] = 0.0
        if k + 1 == step_count:
            self.programs_finished[lane] += 1

    def _desired(self, lane: int, features: np.ndarray, known: np.ndarray) -> tuple[int, tuple[float, ...], int]:
        """Return (skill index, params, source) of the goal the lane wants now (module docstring, item 5)."""

        program = self._programs[lane]
        chosen, best = -1, -math.inf
        for position, interrupt in enumerate(program.interrupts):
            skill = self._interrupt_skill[lane][position]
            if self._blocked[lane, skill] > 0.0 or interrupt.priority <= best:
                continue
            if interrupt.predicate.evaluate(features, known):
                chosen, best = position, interrupt.priority
        if chosen >= 0:
            return self._interrupt_skill[lane][chosen], program.interrupts[chosen].params, _INTERRUPT
        k = int(self._step[lane])
        if k >= len(program.steps):
            return 0, (), _PROGRAM
        skill = self._step_skill[lane][k]
        if skill != 0 and self._blocked[lane, skill] > 0.0:
            self.blocked_decisions[lane] += 1
            return 0, (), _PROGRAM
        return skill, program.steps[k].params, _PROGRAM

    def _set_source(self, lane: int, source: int) -> None:
        """Record where the lane's goal now comes from, counting pre-emptions and resumptions."""

        if source == _INTERRUPT and self._source[lane] == _PROGRAM:
            self.interrupts[lane] += 1
        elif source == _PROGRAM and self._source[lane] == _INTERRUPT:
            self.resumes[lane] += 1
        self._source[lane] = source

    def _executive_slots(self, arrays: ExecutiveArrays) -> dict[int, np.ndarray]:
        """Return, per used skill index, int32[B]: the highest-priority valid slot naming it, or -1."""

        skill = np.asarray(arrays.skill_index)
        priority = np.asarray(arrays.priority, dtype=np.float64)
        valid = np.asarray(arrays.goal_valid, dtype=np.bool_)
        slots = {}
        for index in self._used_skills:
            match = valid & (skill == index)
            best = np.argmax(np.where(match, priority, -np.inf), axis=1)
            slots[index] = np.where(match.any(axis=1), best, -1).astype(np.int32)
        return slots


__all__ = [
    "EVADE_ON_DAMAGE",
    "EXIT_DONE",
    "EXIT_TIMER",
    "Interrupt",
    "MAX_PROGRAM_STEPS",
    "PROGRAMS",
    "P_J0",
    "P_SEQ",
    "Program",
    "ProgramExecutor",
    "ProgramStep",
    "TICK_COLUMNS",
]
