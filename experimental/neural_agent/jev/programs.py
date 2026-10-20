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

Requests (SYSTEM_ONE.md section 9.10). With a scratchpad S_t bound (``shared_state=``, or the one that
``session.run_session`` binds as ``scratchpad``), every decision also speaks the request stream to the lower
layer and reads its statuses back. The goal returned by items 1 to 7 stays the source of truth: until a
combiner writes statuses, items 8 and 9 change nothing in items 1 to 7. Ticks are the session's clock ticks
(``current_tick``, which ``run_session`` sets before each decide; the 1-based call count when it is unset);
``host_cadence_ticks`` and ``tick_seconds`` are the session config's (``run_session`` binds them too).

8. Correspondence, per lane, right after its clocks advance (item 1). The lane's live requests (the ones the
   previous decision issued) and their statuses in ``S_t.skills.statuses`` go to
   ``jev/correspondence.correspond``, with ``hold_ticks = max(1, ceil(hold_seconds_k / tick_seconds))`` when
   the request served step k (an interrupt's request holds 0 s) and the step's rejections acted on so far.
   An abort with the code ``expired`` is left out: the request lapsed at the end of its one-cadence life, it
   did not fail. Each action changes the lane's state::

       Advance             the request's verb is the goal in force: that goal ends COMPLETED (item 3), so a
                           "done" exit fires exactly as for a completion that the core reports
       ResampleStep        a rejection adds to the step's rejections; when the request served step k, the
                           step's clock restarts (elapsed_b <- 0). Item 9 re-issues the step's verb: a
                           hand-written step names one verb, so keep_verb False cannot change it
       RaiseUrgency        the lane's urgency boost u_b grows by ``by``
       Drop                the request is withdrawn
       InformationRequest  ``information_requests`` gains (tick, lane, the request's E_t version, step,
                           verb, reason), and the request is withdrawn

   A lane whose request was withdrawn issues none until its request key (item 9) changes, so one step asks
   the LLM once. A fresh lane (item 2) drops its live requests unread.
9. Requests, after every lane decided. Each lane's live request is replaced by one new request for the goal
   in force; a request lives one host cadence, so every decision re-issues it::

       request_id    n_b, then n_b <- n_b + 1 (monotonic per agent, never reset)
       agent         b
       issued_tick   t;   expires_tick = t + host_cadence_ticks
       verb          the goal's skill id
       args.entity   TARGET_ENTITY (0) when ``requests.ARGUMENT_SCHEMAS[verb]`` needs an entity, else absent
       args.scalars  the goal's params named by the verb's contract ``param_names``
       mode          "interrupt" while an interrupt's goal is in force, else "parallel"
       priority      the interrupt's priority; else E_t's priority of the goal's slot; else 0 (no slot)
       urgency       min(1, clip(priority, 0, 1) + u_b);   confidence 1
       goal_slot, executive_version   the goal's payload;   program_step   k_b (len(steps) once finished)

   The request key (verb, interrupt or not, k_b, the payload's version, E_t's version) scopes a step: when it
   changes, the step's rejections, u_b and the withdrawal reset. Each request is checked with
   ``requests.validate`` against its verb's contract (``contracts=``; without them the check reports
   verb_unknown) and the entities the belief knows: {TARGET_ENTITY} when ``ticks_since_target_seen`` is
   known and below TICKS_SATURATION (the belief starts it saturated), else none. The world box and the
   action mask are the combiner's to check, so the executor passes the widest finite box and every scope.
   A request that fails is still sent, so the combiner rejects it and item 8 reacts; ``invalid_requests``
   counts it. The lanes' requests and every other agent's live requests are written as one ``requests``
   section by its writer "jev": one document version per decision.

Combat program (2026-10-19, ``P_COMBAT``). One pursuit step (exit "done") with four rules and the evader as
interrupts. The evader on damage (``EVADE_ON_DAMAGE``: the belief's ``ticks_since_damage``, as above) has the top
priority. Then: ``guard`` while the target winds up; ``punish`` in its recovery and ``sword`` within
STRIKE_REACH_BLOCKS (the reach of a strike through an ability slot); ``keep_range`` in its sweep within
MELEE_REACH_BLOCKS (the target's sweep reach, a proxy). Pursuit runs when no rule holds,
which is the target out of reach, or idle or cooling down. The reads, with the belief's field names: the distance is
``target_distance_blocks`` (known only while ``target_visible`` is known and above 0.5). The attack phase is the
legal observation's ``target_i32`` column ``attack_phase`` (``ActorPolicyInput.legal_observation.base``), read as the
code of ``ATTACK_PHASE_CODES`` and only while the target is visible. The belief itself carries no attack field: its
ten features are ``contracts.BELIEF_FEATURE_NAMES``. The attack index (``target_i32`` ``attack_index``) is not read:
the phase already names each window, and a rule that fired once per attack would need per-lane memory. Priorities:
the evader (``EVADER_PRIORITY``, 0.5) above the phase rules (``COMBAT_PHASE_PRIORITY``, 0.4) above ``sword``
(``SWORD_PRIORITY``, 0.3). ``P_COMBAT``'s evader is gated (``EVADE_WHEN_HURT``, 2026-10-20): it fires while damaged
only when the agent's own health is below ``EVADE_HEALTH_FRACTION`` (0.5), through ``AllOf((Predicate("on_damage"),
Predicate("self_health_below", 0.5)))``; ``Predicate("self_health_below", t)`` reads the belief's ``self_health_fraction``,
``Not`` and ``AllOf`` combine conditions, and an interrupt takes any of them. Under this package's idle verb bodies
(``DeclaredVerbSkill``, ``skill_library/library.py``; gen 5, seeds 0 to 2, lane #94) the gate looked worse, since no goal
dealt damage and evading cost the least HP per lane-tick (14.8 against 68 per 100 lane-ticks). Under the skill layer's
bodies (the referee run, gen 7, ``combat_open_flat``, 8 lanes x 1023 ticks, seed 0) it won by a wide margin: evader share
56% to 3.5%, HP dealt 957 to 3665, HP taken 1687 to 1544, kills 7 to 55, deaths 13 to 2; the 0.7 threshold gave 21 kills 9 deaths, and a gate on the target's recovery and reach moved
nothing. ``P_J0`` and ``P_SEQ`` keep the ungated ``EVADE_ON_DAMAGE``. ``P_COMBAT`` is ``PROGRAMS['COMBAT']``
(``NA_PROGRAM=COMBAT``, 2026-10-19). It runs on the combat catalogue
(``skill_library.library.build_combat_registry``) with ``--arms jprog`` alone, which the composition CLI enforces.
Its scalars: sword's ``hold_seconds`` is 0.0 (a tap), keep_range's band is KEEP_RANGE_BAND, (3.1, 3.6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, NamedTuple, Sequence

import jax.numpy as jnp
import numpy as np

from adk.architecture.hierarchy import ManagedGoal
from adk.policy import fields
from experimental.neural_agent.contracts import (
    BELIEF_FEATURE_COUNT,
    INT32_MAX,
    PARAM_WIDTH,
    SKILL_IDLE,
    TICKS_SATURATION,
    ExecutiveArrays,
    SkillContract,
    SkillControlTable,
    SkillOutcome,
    _require_control_table,
    control_for,
    feature_index,
    make_goal,
)
from experimental.neural_agent.belief.target_memory import host_target_memory, target_arguments
from experimental.neural_agent.executive.goal_language import ALWAYS, Predicate
from experimental.neural_agent.jev.correspondence import (
    Advance,
    Drop,
    InformationRequest,
    RaiseUrgency,
    ResampleStep,
    correspond,
)
from experimental.neural_agent.requests import (
    ARGUMENT_SCHEMAS,
    COMBAT_VERBS,
    MODE_INTERRUPT,
    MODE_PARALLEL,
    SCOPES,
    ActionRequest,
    RequestArgs,
    RequestStatus,
    rejection_code,
    validate,
)
from experimental.neural_agent.shared_state import SECTION_WRITERS, RequestsSection, SharedState
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
#: The entity handle of the belief's single target, written as ``args.entity`` (module docstring, item 9).
#: The belief (``belief/core.py``) follows one target per lane and publishes no handle, so a request names
#: the target by its index among the belief's targets: 0. The lower layer registers the same convention
#: (agreed between the two sessions, 2026-10-17).
TARGET_ENTITY: int = 0
#: Reach of the target's sweep, in blocks: a proxy, the base agent attack range of the 0.5.7 ruleset
#: (``hytalegym/rulesets/hytale_0_5_7/kweebec_razorleaf_vs_trork_brawler_v9.json``, ``agent.attacks[0].range``). The
#: target's own reach is not measured. The learner's base attack is switched off (``observation/v3/encoding/encoder.py``
#: 470-476), so this is not the strike reach.
MELEE_REACH_BLOCKS: float = 3.0
#: Reach of a strike through an ability slot, in blocks. Measured by the skill layer on 2026-10-19 on both engines
#: (iron sword, first ability pressed every tick at an inert target straight ahead, 8 lanes at stepped distances):
#: hits from 3.4 to 3.7 blocks centre to centre, none from 3.8; 3.5 keeps 0.2 under the measured maximum. The
#: opponent's own hits landed at 2.54 to 3.00 blocks (MELEE_REACH_BLOCKS 3.0 confirmed as the sweep reach).
STRIKE_REACH_BLOCKS: float = 3.5
#: keep_range's band (band_min, band_max): outside the opponent's 3.0 sweep and inside the 3.7 strike reach, so the
#: agent can hit without being hit (the skill layer's measurement above).
KEEP_RANGE_BAND: tuple[float, float] = (3.1, 3.6)
#: The target's attack phases as the legal observation publishes them (``target_i32`` ``attack_phase``): the legacy
#: controller's codes (``hytalegym/jax/combat/opponents/legacy/controller.py``).
ATTACK_PHASE_CODES: dict[str, int] = {"idle": 0, "windup": 1, "sweep": 2, "recovery": 3, "cooldown": 4}
#: Priorities of the combat rules. The evader's (EVADER_PRIORITY) is above them. The phase rules name one phase each,
#: so they never compete; sword holds in reach whatever the phase, so it is below them.
COMBAT_PHASE_PRIORITY: float = 0.4
SWORD_PRIORITY: float = 0.3

_PROGRAM, _INTERRUPT = 0, 1
_SEEN: int = feature_index("ticks_since_target_seen")
_DISTANCE_COLUMN: int = feature_index("target_distance_blocks")
_VISIBLE_PREDICATE: Predicate = Predicate("target_visible")
_ATTACK_PHASE_COLUMN: int = fields.field_index("target_i32", "attack_phase")
#: The widest finite world box: the executor's own check of a request cannot know the world (item 9).
_ANY_WORLD: tuple[tuple[float, ...], tuple[float, ...]] = ((-float(INT32_MAX),) * 3, (float(INT32_MAX),) * 3)


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
class TargetPhase:
    """The target's attack is in ``phase`` and, when ``within_blocks`` is given, the target is that near.

    An interrupt's predicate beside :class:`Predicate` (the combat program, module docstring). It holds only while the
    belief says the target is visible, because the encoder publishes the target's fields only while it is visible. The
    phase comes from the legal observation's target row, not from the belief.

    Raises:
        ValueError: If ``phase`` is not a key of ``ATTACK_PHASE_CODES``, or ``within_blocks`` is not finite and positive.
    """

    phase: str
    within_blocks: float | None = None

    def __post_init__(self) -> None:
        if self.phase not in ATTACK_PHASE_CODES:
            raise ValueError(f"TargetPhase.phase must be one of {tuple(ATTACK_PHASE_CODES)}, got {self.phase!r}")
        if self.within_blocks is not None:
            within = float(self.within_blocks)
            if not math.isfinite(within) or within <= 0.0:
                raise ValueError(f"TargetPhase.within_blocks must be finite and positive, got {self.within_blocks!r}")
            object.__setattr__(self, "within_blocks", within)

    def holds(self, features: Sequence[float], known: Sequence[bool], code: int) -> bool:
        """Return whether, on one lane, the target is visible, in ``phase`` by its legal ``code``, and within reach."""

        if not _VISIBLE_PREDICATE.evaluate(features, known) or int(code) != ATTACK_PHASE_CODES[self.phase]:
            return False
        if self.within_blocks is None:
            return True
        return bool(known[_DISTANCE_COLUMN]) and float(features[_DISTANCE_COLUMN]) < self.within_blocks


def _check_condition(condition: Any, owner: str) -> None:
    """Raise TypeError unless ``condition`` is a Predicate, a TargetPhase, a Not or an AllOf."""

    if not isinstance(condition, (Predicate, TargetPhase, Not, AllOf)):
        raise TypeError(f"{owner} takes a Predicate, a TargetPhase, a Not or an AllOf, got {condition!r}")


def _holds(condition: Any, features: Sequence[float], known: Sequence[bool], code: int) -> bool:
    """Return whether a condition holds on one lane. A Predicate reads the belief row; the others also read ``code``."""

    if isinstance(condition, Predicate):
        return condition.evaluate(features, known)
    return condition.holds(features, known, code)


def _reads_target_phase(condition: Any) -> bool:
    """True when the condition reads the target's attack phase: a TargetPhase, also inside a Not or an AllOf."""

    if isinstance(condition, TargetPhase):
        return True
    if isinstance(condition, Not):
        return _reads_target_phase(condition.part)
    if isinstance(condition, AllOf):
        return any(_reads_target_phase(part) for part in condition.parts)
    return False


@dataclass(frozen=True)
class Not:
    """The condition ``part`` does not hold. A feature that is unknown makes ``part`` False, so its negation holds.

    Raises:
        TypeError: If ``part`` is not a Predicate, a TargetPhase, a Not or an AllOf.
    """

    part: Any

    def __post_init__(self) -> None:
        _check_condition(self.part, "Not")

    def holds(self, features: Sequence[float], known: Sequence[bool], code: int) -> bool:
        """Return whether ``part`` does not hold on one lane (``code`` is the target's phase code, or -1)."""

        return not _holds(self.part, features, known, code)


@dataclass(frozen=True)
class AllOf:
    """The conjunction of conditions: it holds where every part holds.

    Raises:
        ValueError: If there are no parts.
        TypeError: If a part is not a Predicate, a TargetPhase, a Not or an AllOf.
    """

    parts: tuple[Any, ...]

    def __post_init__(self) -> None:
        parts = tuple(self.parts)
        if not parts:
            raise ValueError("AllOf needs at least one part")
        for part in parts:
            _check_condition(part, "AllOf")
        object.__setattr__(self, "parts", parts)

    def holds(self, features: Sequence[float], known: Sequence[bool], code: int) -> bool:
        """Return whether every part holds on one lane (``code`` is the target's phase code, or -1)."""

        return all(_holds(part, features, known, code) for part in self.parts)


@dataclass(frozen=True)
class Interrupt:
    """A conditional goal of the executive: ``skill_id`` pre-empts the program while ``predicate`` holds."""

    skill_id: str
    priority: float
    predicate: Predicate | TargetPhase | Not | AllOf
    params: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        _skill_id(self.skill_id, "Interrupt")
        if not isinstance(self.predicate, (Predicate, TargetPhase, Not, AllOf)) or self.predicate == ALWAYS:
            raise ValueError("Interrupt.predicate must be a Predicate other than 'always', a TargetPhase, a Not or an AllOf")
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
#: The combat program's evader (2026-10-20): evade while damaged only when the agent's own health is below half; the
#: skill layer's referee judged it under its bodies (gen 7, combat_open_flat, 8 lanes x 1023 ticks): evader share 56%
#: to 3.5%, HP dealt 957 to 3665, HP taken 1687 to 1544, kills 7 to 55, deaths 13 to 2; the 0.7 threshold gave 21/9.
EVADE_HEALTH_FRACTION = 0.5
EVADE_WHEN_HURT = Interrupt(
    EVADER_SKILL,
    EVADER_PRIORITY,
    AllOf((Predicate(EVADER_CONSTRAINT), Predicate("self_health_below", EVADE_HEALTH_FRACTION))),
)
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
#: The combat program (module docstring): evade on damage below half health; guard, punish and keep range by the
#: target's phase; sword in reach.
P_COMBAT = Program(
    steps=(ProgramStep(PURSUIT_SKILL, exit=EXIT_DONE),),
    interrupts=(
        EVADE_WHEN_HURT,
        Interrupt("guard", COMBAT_PHASE_PRIORITY, TargetPhase("windup")),
        Interrupt("punish", COMBAT_PHASE_PRIORITY, TargetPhase("recovery", STRIKE_REACH_BLOCKS)),
        Interrupt("keep_range", COMBAT_PHASE_PRIORITY, TargetPhase("sweep", MELEE_REACH_BLOCKS), params=KEEP_RANGE_BAND),
        Interrupt("sword", SWORD_PRIORITY, Predicate("distance_below", STRIKE_REACH_BLOCKS), params=(0.0,)),
    ),
)
PROGRAMS: dict[str, Program] = {"J0": P_J0, "SEQ": P_SEQ, "COMBAT": P_COMBAT}


def program_uses_combat(program: Program) -> bool:
    """True when the program names a combat verb (``COMBAT_VERBS``) as a step or an interrupt."""

    verbs = {step.skill_id for step in program.steps} | {interrupt.skill_id for interrupt in program.interrupts}
    return bool(verbs & set(COMBAT_VERBS))


class LaneInformationRequest(NamedTuple):
    """One information request to the LLM: a lane's step was rejected REPEATED_REJECTIONS times (item 8).

    ``reason`` is the correspondence rule's; the version, step and verb are those of the rejected request.
    """

    tick: int
    lane: int
    executive_version: int
    program_step: int
    verb: str
    reason: str


class ProgramExecutor:
    """A ``session.HostDecider`` that runs one program per lane (module docstring, items 1 to 7).

    Args:
        skill_ids: The catalogue in dispatcher order, ``"idle"`` first.
        control_table: The catalogue's ``SkillControlTable`` (the skills' contracts).
        programs_per_lane: One :class:`Program` per lane; B is its length.
        no_reissue_seconds: Keyword only. The feedback rule's window w in seconds; 0 turns it off.
        shared_state: Keyword only. The scratchpad S_t that items 8 and 9 read and write, or None (the
            default) until ``session.run_session`` binds its own as ``scratchpad``.
        contracts: Keyword only. The catalogue's ``SkillContract`` values in catalogue order
            (``SkillRegistry.contracts``), for the check of item 9, or None.

    Attributes:
        lane_calls: Lane decisions made, B per call. ``lane_failures`` is always 0 and ``first_failure``
            always None: there is no backend to fail. ``eval/run_composition.py`` reads all three.
        scratchpad, current_tick, host_cadence_ticks, tick_seconds: What ``session.run_session`` binds:
            its scratchpad, its clock tick before each decide, and its config's cadence and tick length. The
            defaults 11 ticks and 0.045 s are SessionConfig's.
        information_requests: The :class:`LaneInformationRequest` records of item 8, oldest first.
        requests_issued: int[B], the requests written per lane. ``invalid_requests`` counts the ones that
            failed the check of item 9 and ``first_invalid_request`` keeps the first reason.
            ``correspondence`` counts the actions of item 8 by kind.

    Raises:
        ValueError: If the catalogue is empty, repeats an id or does not start with idle, a program is
            missing, ``no_reissue_seconds`` is negative or not finite, or ``contracts`` are not the
            catalogue's in its order.
        KeyError: If a step or interrupt names a skill outside the catalogue.
        TypeError: If a program is not a Program, the table is not a SkillControlTable, ``shared_state``
            is not a SharedState, or a contract is not a SkillContract.
    """

    name: str = "J-PROG"

    def __init__(
        self,
        skill_ids: Sequence[str],
        control_table: SkillControlTable,
        programs_per_lane: Sequence[Program],
        *,
        no_reissue_seconds: float = 0.0,
        shared_state: SharedState | None = None,
        contracts: Sequence[SkillContract] | None = None,
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
        if shared_state is not None and not isinstance(shared_state, SharedState):
            raise TypeError(f"shared_state must be a SharedState, got {type(shared_state).__name__}")
        if contracts is not None:
            contracts = tuple(contracts)
            if not all(isinstance(contract, SkillContract) for contract in contracts):
                raise TypeError("contracts must hold SkillContract values")
            if tuple(contract.skill_id for contract in contracts) != ids:
                raise ValueError(f"contracts must be the catalogue's, in its order {ids}")
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
        self._reads_phase = any(_reads_target_phase(i.predicate) for p in programs for i in p.interrupts)
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
        self._interrupt_priority = np.zeros(lanes, dtype=np.float64)
        # The request stream (module docstring, items 8 and 9); session.run_session binds the first four.
        self.scratchpad: SharedState | None = shared_state
        self.current_tick: int | None = None
        self.host_cadence_ticks: int = 11
        self.tick_seconds: float = 0.045
        # Bound by run_session: whether the host seam keeps unchanged lanes' goal ages (the clock regime).
        self.host_respects_commitment: bool = False
        self._skill_ids = ids
        self._contracts: tuple[SkillContract, ...] | None = contracts
        self._next_request = np.zeros(lanes, dtype=np.int64)
        # Per lane: request id -> the live request; its program_step maps a status back to its step.
        self._live: list[dict[int, ActionRequest]] = [{} for _ in range(lanes)]
        self._request_key: list[tuple[Any, ...] | None] = [None] * lanes
        self._step_rejections = np.zeros(lanes, dtype=np.int64)
        self._urgency_boost = np.zeros(lanes, dtype=np.float64)
        self._withdrawn = np.zeros(lanes, dtype=np.bool_)
        self.requests_issued = np.zeros(lanes, dtype=np.int64)
        self.invalid_requests = 0
        self.first_invalid_request: str | None = None
        self.correspondence = dict.fromkeys(
            ("advance", "resample_step", "raise_urgency", "drop", "information_request"), 0
        )
        self.information_requests: list[LaneInformationRequest] = []

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
        """Return the per-lane counters as JSON-ready lists, and the request stream's when a scratchpad is bound."""

        counters: dict[str, Any] = {
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
        if self.scratchpad is not None:
            counters.update(
                requests_issued=self.requests_issued.tolist(),
                requests_invalid=self.invalid_requests,
                correspondence=dict(self.correspondence),
                information_requests=len(self.information_requests),
            )
        return counters

    # ------------------------------------------------------------------ the host decision

    def decide(self, transition: Any, hierarchy_state: Any, executive: Any, /) -> ManagedGoal:
        """Advance every lane's program and return the batched goal for all B lanes.

        ``executive`` is not read: E_t is read from ``hierarchy_state.carry.manager_state``, where the
        session's executive writer puts every new version. With a scratchpad bound, the lanes also
        correspond with the request stream (module docstring, items 8 and 9).

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
        codes = self._target_codes(transition) if self._reads_phase else None
        emergency = np.asarray(step.goal_runtime.emergency_interrupt, dtype=np.bool_)
        done = np.asarray(transition.done, dtype=np.bool_)
        valid = np.asarray(hierarchy_state.goal_valid, dtype=np.bool_)
        completed = np.asarray(hierarchy_state.goal_termination_requested, dtype=np.bool_)
        manager_age = np.asarray(hierarchy_state.manager_age_seconds, dtype=np.float32)
        carry = hierarchy_state.carry
        # Seconds since this executor's previous decision, per lane. With the old host seam the session
        # zeroes goal_age on every write, so goal_age is that interval; with host_respects_commitment an
        # unchanged lane keeps its age, so the interval is the age's increase since the last decision.
        dt = self._elapsed_since_last(np.asarray(carry.goal_age, dtype=np.float32))
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
        bound = self.scratchpad is not None
        tick = self.decisions + 1 if self.current_tick is None else int(self.current_tick)
        history: dict[int, list[RequestStatus]] = {}  # each agent's statuses, newest last
        if bound:
            for status in self.scratchpad.skills.statuses:
                history.setdefault(status.agent, []).append(status)

        for lane in range(lanes):
            program = self._programs[lane]
            steps = program.steps
            ended = self._advance_clocks(lane, bool(fresh[lane]), dt[lane], int(skill[lane]), bool(completed[lane]))
            if bound:
                ended = self._correspond(lane, bool(fresh[lane]), int(skill[lane]), ended, tick, history.get(lane, ()))
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

            code = -1 if codes is None else int(codes[lane])
            desired, desired_params, source, priority = self._desired(lane, features[lane], known[lane], code)
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
            self._set_source(lane, source, priority)

        self._started[:] = True
        self._issued_version[:] = version
        self.decisions += 1
        self.lane_calls += lanes
        if bound:
            self._issue_requests(
                tick, skill, params, version, slot, executive_version, arrays, features, known,
                host_target_memory(step.belief_update),
            )
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

    def _elapsed_since_last(self, goal_age: np.ndarray) -> np.ndarray:
        """Return per lane the seconds elapsed since the previous decision, from the carry's goal age.

        Without ``host_respects_commitment`` (the session binds the flag) every host write zeroes the age, so
        the age is the interval. With it, an unchanged lane keeps its age: a lane whose age is below the age
        seen last time was reset by a goal write and its age is the interval, otherwise the interval is the
        increase. The first call (or a lane-count change) starts from zero.
        """

        last = getattr(self, "_last_goal_age", None)
        if last is None or last.shape != goal_age.shape:
            last = np.zeros_like(goal_age)
        self._last_goal_age = goal_age.copy()
        if not getattr(self, "host_respects_commitment", False):
            # The old seam zeroes every lane's goal age on each host write, so the age IS the interval
            # (and two equal consecutive ages mean two equal intervals, not zero time).
            return goal_age.astype(np.float32)
        reset = goal_age < last
        return np.where(reset, goal_age, goal_age - last).astype(np.float32)

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

    def _desired(
        self, lane: int, features: np.ndarray, known: np.ndarray, code: int = -1
    ) -> tuple[int, tuple[float, ...], int, float]:
        """Return (skill index, params, source, priority) of the goal the lane wants now (module docstring, item 5).

        ``priority`` is the chosen interrupt's, or 0.0 for the program (its requests take E_t's, item 9).
        """

        program = self._programs[lane]
        chosen, best = -1, -math.inf
        for position, interrupt in enumerate(program.interrupts):
            skill = self._interrupt_skill[lane][position]
            if self._blocked[lane, skill] > 0.0 or interrupt.priority <= best:
                continue
            if _holds(interrupt.predicate, features, known, code):
                chosen, best = position, interrupt.priority
        if chosen >= 0:
            return self._interrupt_skill[lane][chosen], program.interrupts[chosen].params, _INTERRUPT, best
        k = int(self._step[lane])
        if k >= len(program.steps):
            return 0, (), _PROGRAM, 0.0
        skill = self._step_skill[lane][k]
        if skill != 0 and self._blocked[lane, skill] > 0.0:
            self.blocked_decisions[lane] += 1
            return 0, (), _PROGRAM, 0.0
        return skill, program.steps[k].params, _PROGRAM, 0.0

    def _target_codes(self, transition: Any) -> np.ndarray:
        """int[B]: the target's attack phase code per lane, from the legal observation's target row (module docstring)."""

        target = np.asarray(transition.actor_input.legal_observation.base.target_i32, dtype=np.int64)
        if target.ndim != 2 or target.shape[0] != self._lanes:
            raise ValueError(f"legal_observation.base.target_i32 must have one row per lane ({self._lanes}), got {target.shape}")
        return target[:, _ATTACK_PHASE_COLUMN]

    def _set_source(self, lane: int, source: int, priority: float) -> None:
        """Record where the lane's goal now comes from and the interrupt's priority, counting pre-emptions
        and resumptions."""

        if source == _INTERRUPT and self._source[lane] == _PROGRAM:
            self.interrupts[lane] += 1
        elif source == _PROGRAM and self._source[lane] == _INTERRUPT:
            self.resumes[lane] += 1
        self._source[lane] = source
        self._interrupt_priority[lane] = priority

    # ------------------------------------------------------------------ the request stream (items 8 and 9)

    def _correspond(
        self,
        lane: int,
        fresh: bool,
        current_skill: int,
        ended: SkillOutcome | None,
        tick: int,
        statuses: Sequence[RequestStatus],
    ) -> SkillOutcome | None:
        """Apply the correspondence rules to the lane's live requests (module docstring, item 8).

        Return how the goal in force ended: ``ended`` as item 3 found it, or COMPLETED on an Advance.
        """

        live = self._live[lane]
        if fresh:
            live.clear()
            self._request_key[lane] = None
            return ended
        if not live:
            return ended
        requests = tuple(live.values())
        steps = self._programs[lane].steps
        k = int(self._step[lane])
        on_step = (
            self._source[lane] == _PROGRAM
            and k < len(steps)
            and all(request.mode == MODE_PARALLEL and request.program_step == k for request in requests)
        )
        statuses = [
            status
            for status in statuses
            if not (status.kind == "aborted" and rejection_code(status.reason) == "expired")
        ]
        actions = correspond(
            requests,
            statuses,
            tick=tick,
            hold_ticks=max(1, math.ceil((steps[k].hold_seconds if on_step else 0.0) / self.tick_seconds)),
            step_rejections=int(self._step_rejections[lane]),
        )
        for action in actions:
            self.correspondence[action.kind] += 1
            if isinstance(action, Advance):
                if ended is None and all(request.verb == self._skill_ids[current_skill] for request in requests):
                    ended = SkillOutcome.COMPLETED
            elif isinstance(action, ResampleStep):
                latest = {status.request_id: status.kind for status in statuses if status.tick <= tick}
                self._step_rejections[lane] += sum(latest.get(request_id) == "rejected" for request_id in live)
                if on_step:
                    self._elapsed[lane] = 0.0
            elif isinstance(action, RaiseUrgency):
                self._urgency_boost[lane] += action.by
            elif isinstance(action, Drop):
                live.pop(action.request_id, None)
                self._withdrawn[lane] = True
            elif isinstance(action, InformationRequest):
                request = requests[0]
                self.information_requests.append(
                    LaneInformationRequest(
                        tick, lane, request.executive_version, request.program_step, request.verb, action.reason
                    )
                )
                live.clear()
                self._withdrawn[lane] = True
        return ended

    def _issue_requests(
        self,
        tick: int,
        skill: np.ndarray,
        params: np.ndarray,
        version: np.ndarray,
        slot: np.ndarray,
        executive_version: np.ndarray,
        arrays: ExecutiveArrays,
        features: np.ndarray,
        known: np.ndarray,
        target_memory: Any,
    ) -> None:
        """Replace each lane's live request with one for its goal in force, and write S_t's requests (item 9).

        ``target_memory`` is the belief's remembered target position as numpy arrays (``host_target_memory``), or None.
        """

        priorities = np.asarray(arrays.priority, dtype=np.float64)
        for lane in range(self._lanes):
            index, goal_slot, k = int(skill[lane]), int(slot[lane]), int(self._step[lane])
            verb = self._skill_ids[index]
            interrupt = bool(self._source[lane] == _INTERRUPT)
            key = (verb, interrupt, k, int(version[lane]), int(executive_version[lane]))
            if key != self._request_key[lane]:
                self._request_key[lane] = key
                self._step_rejections[lane] = 0
                self._urgency_boost[lane] = 0.0
                self._withdrawn[lane] = False
            live = self._live[lane]
            live.clear()
            if self._withdrawn[lane]:
                continue
            if interrupt:
                priority = float(self._interrupt_priority[lane])
            else:
                priority = float(priorities[lane, goal_slot]) if goal_slot >= 0 else 0.0
            contract = None if self._contracts is None else self._contracts[index]
            schema = ARGUMENT_SCHEMAS.get(verb)
            named = 0 if contract is None else len(contract.param_names)
            target_entity, target_position = target_arguments(schema, features[lane], target_memory, lane, TARGET_ENTITY)
            request = ActionRequest(
                request_id=int(self._next_request[lane]),
                agent=lane,
                issued_tick=tick,
                expires_tick=tick + int(self.host_cadence_ticks),
                verb=verb,
                args=RequestArgs(
                    entity=target_entity,
                    position=target_position,
                    scalars=tuple(float(value) for value in params[lane, :named]),
                ),
                mode=MODE_INTERRUPT if interrupt else MODE_PARALLEL,
                urgency=min(1.0, min(max(priority, 0.0), 1.0) + float(self._urgency_boost[lane])),
                confidence=1.0,
                priority=priority,
                goal_slot=goal_slot,
                executive_version=int(version[lane]),
                program_step=k,
            )
            self._next_request[lane] += 1
            self.requests_issued[lane] += 1
            seen = bool(known[lane, _SEEN]) and float(features[lane, _SEEN]) < TICKS_SATURATION
            reason = validate(
                request,
                contract,
                tick=tick,
                world_bounds=_ANY_WORLD,
                known_entities=(TARGET_ENTITY,) if seen else (),
                legal_scopes=SCOPES,
                done_ids=(),
            )
            if reason is not None:
                self.invalid_requests += 1
                if self.first_invalid_request is None:
                    self.first_invalid_request = reason
            live[request.request_id] = request
        state = self.scratchpad
        others = tuple(request for request in state.requests.live if not 0 <= request.agent < self._lanes)
        mine = tuple(request for live in self._live for request in live.values())
        section = RequestsSection(live=others + mine, limit=state.requests.limit)
        state.write("requests", section, writer=SECTION_WRITERS["requests"], tick=tick)

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
    "ATTACK_PHASE_CODES",
    "COMBAT_PHASE_PRIORITY",
    "EVADE_ON_DAMAGE",
    "EVADE_HEALTH_FRACTION",
    "EVADE_WHEN_HURT",
    "EXIT_DONE",
    "EXIT_TIMER",
    "AllOf",
    "Interrupt",
    "LaneInformationRequest",
    "MAX_PROGRAM_STEPS",
    "KEEP_RANGE_BAND",
    "MELEE_REACH_BLOCKS",
    "PROGRAMS",
    "P_COMBAT",
    "P_J0",
    "P_SEQ",
    "Not",
    "Program",
    "ProgramExecutor",
    "ProgramStep",
    "STRIKE_REACH_BLOCKS",
    "SWORD_PRIORITY",
    "TARGET_ENTITY",
    "TICK_COLUMNS",
    "TargetPhase",
    "program_uses_combat",
]
