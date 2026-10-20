"""Combat styles beside P_COMBAT: five programs from the same verbs and predicates (task #100, 2026-10-20).

The owner's direction (2026-10-19 night): "more variable combat with the middle and neural layer, from swinging styles to
combat styles, so there is real skill instead of brute force". P_COMBAT (``programs.py``) is one style. The styles here are
:class:`~experimental.neural_agent.jev.programs.Program` values built from the verbs and predicates that exist, so the
executive can name one per agent and moment through ``NA_PROGRAM`` (``eval/arms.py``, ``program_settings``).

Built and registered in ``PROGRAMS``: COUNTER, PRESSURE, HIT_AND_RUN. Designs, not registered: AMBUSH (needs the ``hide``
verb) and ZONING (needs the ``shortbow`` verb). Their builders raise ``ValueError`` until ``requests.COMBAT_CONTRACTS``
declares those verbs.

Rule order. A style's interrupts are its priority list: the highest priority whose predicate holds pre-empts the step
(``programs.py`` item 5); on a tie the interrupt listed first wins. Where none holds, the step's skill runs. A program runs
its steps once and then idles, so a cycle that repeats is written as interrupts, which the executor re-decides at every
cadence tick (HIT_AND_RUN).

Import order. ``programs.py`` calls :func:`built_styles` at its end to register the built styles, so this module imports
the program language inside :func:`_language`, not at the top: a top-level import would be circular. Either module may be
imported first; ``test_na_combat_styles.py`` checks both orders.

Sword charges. The sword's ``hold_seconds`` scalar is the charge of its move, and each style has its own value (the
constants below). The skill layer's sword accepts the scalar and builds no charged holds yet
(``skill_agent/runtime/scripted_bodies.py``, the table at line 22), so until it does the built styles strike alike and the
values record intent only.

Target timing. The windup and recovery lengths behind the charges come from the 0.5.7 ruleset
``kweebec_razorleaf_vs_trork_brawler_v9.json`` (``target.attacks[0]``: windup 12 ticks, sweep 5, recovery 13) at
0.045 s a tick. Whether the composition scene (``combat_open_flat``) loads that ruleset is not confirmed (INFERRED).

Step exits. A step with exit done ends when the lower layer reports done for its request (``jev/correspondence.py``,
Advance). The steps that never end here (pursuit, keep_range) follow P_COMBAT and P_J0; a done report from the skill
layer would finish the program, and an idle program then holds nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from experimental.neural_agent.executive.goal_language import Predicate
from experimental.neural_agent.requests import COMBAT_CONTRACTS

if TYPE_CHECKING:
    from experimental.neural_agent.jev.programs import Program

#: A tap: PRESSURE strikes on every tick the target is in reach, so it does not wait to charge.
PRESSURE_HOLD_SECONDS: float = 0.0
#: A short charge: COUNTER's fallback strike lands inside the target's 13-tick recovery (0.585 s at 0.045 s a tick).
COUNTER_HOLD_SECONDS: float = 0.3
#: A charge about the length of the target's 12-tick windup (0.54 s): the swing is spent before its sweep would start.
HIT_AND_RUN_HOLD_SECONDS: float = 0.6
#: AMBUSH's opening strike from cover (design only), charged for the same windup reason as HIT_AND_RUN.
AMBUSH_HOLD_SECONDS: float = 0.6
#: PRESSURE guards a wind-up only below this self health (low health).
PRESSURE_GUARD_FRACTION: float = 0.5
#: PRESSURE evades only when damaged and below this self health.
PRESSURE_EVADE_FRACTION: float = 0.3
#: ZONING's band in blocks (design only): the skill layer's AVOID_BAND_BLOCKS default, not measured for a bow.
ZONING_BAND: tuple[float, float] = (6.0, 10.0)


def _language() -> Any:
    """Return ``jev.programs``, imported late (module docstring, import order)."""

    from experimental.neural_agent.jev import programs

    return programs


def _require_declared(owner: str, verbs: Sequence[str]) -> None:
    """Raise ValueError naming the verbs that ``requests.COMBAT_CONTRACTS`` does not declare."""

    missing = [verb for verb in verbs if verb not in COMBAT_CONTRACTS]
    if missing:
        raise ValueError(f"{owner} needs the verbs {missing}, which requests.COMBAT_CONTRACTS does not declare yet")


def counter() -> Program:
    """COUNTER: a counter-puncher. It waits at the band, guards the wind-up and strikes in the recovery.

    Rule order (the priority list):
      1. guard, COMBAT_PHASE_PRIORITY (0.4): the target winds up, at any distance. The skill layer's guard body acts
         only within GUARD_RANGE_BLOCKS (5.0, ``scripted_bodies.py``), so beyond 5 blocks the verb idles.
      2. punish, COMBAT_PHASE_PRIORITY (0.4): the target is in its recovery and within STRIKE_REACH_BLOCKS.
      3. sword, SWORD_PRIORITY (0.3), charge COUNTER_HOLD_SECONDS: the same recovery window in reach. The skill layer's
         ``punish`` is this strike gated to the recovery (``scripted_bodies.py``, line 24), so rule 3 acts only where rule 2
         cannot: when punish is blocked by the feedback rule (``programs.py`` item 3, a no_reissue window above 0 s). With
         the default window of 0 s it never acts. It stays because the brief lists it.
      step. keep_range on KEEP_RANGE_BAND, exit done: whenever no rule holds it walks in when the target is beyond 3.6
         blocks and out when nearer than 3.1, so the counter-puncher waits at the band. No evader: the rule list has none.
    """

    lang = _language()
    return lang.Program(
        steps=(lang.ProgramStep("keep_range", params=lang.KEEP_RANGE_BAND, exit=lang.EXIT_DONE),),
        interrupts=(
            lang.Interrupt("guard", lang.COMBAT_PHASE_PRIORITY, lang.TargetPhase("windup")),
            lang.Interrupt("punish", lang.COMBAT_PHASE_PRIORITY, lang.TargetPhase("recovery", lang.STRIKE_REACH_BLOCKS)),
            lang.Interrupt(
                "sword",
                lang.SWORD_PRIORITY,
                lang.TargetPhase("recovery", lang.STRIKE_REACH_BLOCKS),
                params=(COUNTER_HOLD_SECONDS,),
            ),
        ),
    )


def pressure() -> Program:
    """PRESSURE: close and strike continuously. It guards and evades only when it is low.

    Rule order (the priority list):
      1. evader, EVADER_PRIORITY (0.5): the agent is damaged and its self health is below PRESSURE_EVADE_FRACTION (0.3).
      2. guard, COMBAT_PHASE_PRIORITY (0.4): the target winds up and self health is below PRESSURE_GUARD_FRACTION (0.5).
      3. sword, SWORD_PRIORITY (0.3), a tap (PRESSURE_HOLD_SECONDS): the target is within STRIKE_REACH_BLOCKS, in any phase.
      step. pursuit, exit done: close on the target until a rule holds.
    """

    lang = _language()
    return lang.Program(
        steps=(lang.ProgramStep(lang.PURSUIT_SKILL, exit=lang.EXIT_DONE),),
        interrupts=(
            lang.Interrupt(
                lang.EVADER_SKILL,
                lang.EVADER_PRIORITY,
                lang.AllOf((Predicate("on_damage"), Predicate("self_health_below", PRESSURE_EVADE_FRACTION))),
            ),
            lang.Interrupt(
                "guard",
                lang.COMBAT_PHASE_PRIORITY,
                lang.AllOf((lang.TargetPhase("windup"), Predicate("self_health_below", PRESSURE_GUARD_FRACTION))),
            ),
            lang.Interrupt(
                "sword",
                lang.SWORD_PRIORITY,
                Predicate("distance_below", lang.STRIKE_REACH_BLOCKS),
                params=(PRESSURE_HOLD_SECONDS,),
            ),
        ),
    )


def hit_and_run() -> Program:
    """HIT_AND_RUN: strike from the band where the sword outranges the target's sweep, then back out.

    The strike band is 3.1 to 3.5 blocks: from KEEP_RANGE_BAND's near edge, 0.1 outside the target's sweep reach
    (MELEE_REACH_BLOCKS 3.0), to the sword's press reach (STRIKE_REACH_BLOCKS 3.5). The brief's 3.6 is KEEP_RANGE_BAND's
    far edge; the skill layer's sword presses only within 3.5 (``scripted_bodies.py``, REACH_BLOCKS), so a strike band to
    3.6 would hold a sword that does not press.

    Rule order (the priority list):
      1. evader, EVADER_PRIORITY (0.5): EVADE_WHEN_HURT (on damage and self health below EVADE_HEALTH_FRACTION).
      2. keep_range, COMBAT_PHASE_PRIORITY (0.4), band KEEP_RANGE_BAND: the target is inside 3.1 blocks. The back-off
         ends at 3.1, where keep_range's body holds (it moves only while the target is outside its band), and rule 3
         takes over.
      3. sword, SWORD_PRIORITY (0.3), charge HIT_AND_RUN_HOLD_SECONDS: the target is in the strike band, in any phase.
      step. pursuit, exit done: close on the target until a rule holds.

    The cycle repeats by construction: pursuit closes into the band, sword strikes there, keep_range backs out when the
    target closes inside 3.1, and sword resumes once the target is back in the band. The brief's timed keep_range hold is
    not in the language: an interrupt has no timer, so the back-off lasts only while the target is inside 3.1.
    """

    lang = _language()
    strike_band = lang.AllOf(
        (
            Predicate("distance_below", lang.STRIKE_REACH_BLOCKS),
            lang.Not(Predicate("distance_below", lang.KEEP_RANGE_BAND[0])),
        )
    )
    return lang.Program(
        steps=(lang.ProgramStep(lang.PURSUIT_SKILL, exit=lang.EXIT_DONE),),
        interrupts=(
            lang.EVADE_WHEN_HURT,
            lang.Interrupt(
                "keep_range",
                lang.COMBAT_PHASE_PRIORITY,
                Predicate("distance_below", lang.KEEP_RANGE_BAND[0]),
                params=lang.KEEP_RANGE_BAND,
            ),
            lang.Interrupt("sword", lang.SWORD_PRIORITY, strike_band, params=(HIT_AND_RUN_HOLD_SECONDS,)),
        ),
    )


def ambush(hide_verb: str = "hide") -> Program:
    """AMBUSH (design, not registered): hold cover and strike when the target comes into reach.

    Needs ``hide_verb`` (the locomotion verb that runs to a cover point; the skill layer's catalogue names it ``hide``) and
    ``sword``. Raises ValueError until the request layer declares the hide verb.

    Rule order (the priority list):
      1. evader, EVADER_PRIORITY (0.5): EVADE_WHEN_HURT.
      2. sword, SWORD_PRIORITY (0.3), charge AMBUSH_HOLD_SECONDS: the target is within STRIKE_REACH_BLOCKS.
      step. ``hide_verb``, exit done: stay in cover. Risk for its designers: if the verb reports completion on arrival,
         the step ends and the program idles at cover, so the bodies must hold the cover point without reporting it.
    """

    _require_declared("AMBUSH", (hide_verb, "sword"))
    lang = _language()
    return lang.Program(
        steps=(lang.ProgramStep(hide_verb, exit=lang.EXIT_DONE),),
        interrupts=(
            lang.EVADE_WHEN_HURT,
            lang.Interrupt(
                "sword",
                lang.SWORD_PRIORITY,
                Predicate("distance_below", lang.STRIKE_REACH_BLOCKS),
                params=(AMBUSH_HOLD_SECONDS,),
            ),
        ),
    )


def zoning(shoot_verb: str = "shortbow") -> Program:
    """ZONING (design, not registered): hold a band at range and shoot into it; back off when the target closes.

    Needs ``shoot_verb`` (the shortbow verb, not yet declared) and ``keep_range``. Raises ValueError until the request layer
    declares the shot verb. The shot takes no scalars, since the verb's contract is not known yet.

    Rule order (the priority list):
      1. evader, EVADER_PRIORITY (0.5): EVADE_WHEN_HURT.
      2. ``shoot_verb``, COMBAT_PHASE_PRIORITY (0.4): the target is inside ZONING_BAND, beyond its near edge and nearer
         than its far edge.
      step. keep_range on ZONING_BAND, exit done: it walks out when the target is nearer than the near edge and in when it
         is beyond the far edge, so the shot runs while the band holds.
    """

    _require_declared("ZONING", (shoot_verb, "keep_range"))
    lang = _language()
    near, far = ZONING_BAND
    return lang.Program(
        steps=(lang.ProgramStep("keep_range", params=ZONING_BAND, exit=lang.EXIT_DONE),),
        interrupts=(
            lang.EVADE_WHEN_HURT,
            lang.Interrupt(
                shoot_verb,
                lang.COMBAT_PHASE_PRIORITY,
                lang.AllOf((Predicate("distance_above", near), Predicate("distance_below", far))),
            ),
        ),
    )


def built_styles() -> dict[str, Program]:
    """Return the three styles ``programs.py`` registers in ``PROGRAMS``, by their ``NA_PROGRAM`` names."""

    return {"COUNTER": counter(), "PRESSURE": pressure(), "HIT_AND_RUN": hit_and_run()}


__all__ = [
    "AMBUSH_HOLD_SECONDS",
    "COUNTER_HOLD_SECONDS",
    "HIT_AND_RUN_HOLD_SECONDS",
    "PRESSURE_EVADE_FRACTION",
    "PRESSURE_GUARD_FRACTION",
    "PRESSURE_HOLD_SECONDS",
    "ZONING_BAND",
    "ambush",
    "built_styles",
    "counter",
    "hit_and_run",
    "pressure",
    "zoning",
]
