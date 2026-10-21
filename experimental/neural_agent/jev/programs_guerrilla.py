"""The guerrilla program (2026-10-20, task #99): avoid when threatened, shoot from range, hide when seen.

An example, not a deliverable (owner, 2026-10-20): a workflow like this is what the LLM teaches the decision model, and
nobody needs to hand-code it. What must hold is that the features it uses work on their own and compose: the verbs
(``shortbow``, ``avoid``, ``hide``, ``keep_range``), companion requests (``Interrupt.beside``), the bodies' reports and the
remembered target position. The program stays as a worked example and a test fixture; it is not tuned further.

One step, pursuit until "done", and three interrupts. The priorities decide (``programs.py`` item 5): the highest priority
whose condition holds on the lane wins, and ties go to the lower index.

1. ``shortbow`` (SHOT_PRIORITY, 0.6) when the shot is clean: the target is seen, its distance is inside BOW_WINDOW_BLOCKS
   (the band ``avoid`` keeps, 6 to 10 blocks), and it is not in melee reach (report ``in_reach`` is off). Scalar:
   ``hold_seconds`` = SHOT_HOLD_SECONDS = 1.2 s, the full draw (ability slot 4) under the contract's mapping; the
   measurement behind both values is in the constants' docstrings (#105, 2026-10-20).
   Beside it, while the shot is in force, a ``keep_range`` companion holds AVOID_BAND_BLOCKS facing the target
   (``programs.py``, item 9, "Companions"). The body faces the target while it shoots, which the skill layer
   measured to matter (commit 1112c77).
2. ``avoid`` (AVOID_PRIORITY, the evader's 0.5) while the target faces the agent (report ``facing_me``) or the agent's
   health is below HURT_HEALTH_FRACTION (``Predicate("self_health_below", 0.5)``, strict). The OR is built from ``Not`` and
   ``AllOf``: ``Not(AllOf((Not(facing_me), Not(self_health_below))))``. Scalars: AVOID_BAND_BLOCKS, the skill layer's default
   band of (6, 10) blocks.
3. ``hide`` (HIDE_PRIORITY, 0.3) while the target is seen. It names no cover point, so the hide body retreats as ``avoid``
   does (``skill_agent/runtime/scripted_bodies.py``, ``hide``) until a cover position is supplied.
4. Search is the step: while the target is unseen no interrupt holds, and the step's pursuit runs. The executor's request
   for pursuit carries the remembered position while the target is unseen (``belief/target_memory.py``,
   ``target_arguments``, task #96), so the body heads for it. The brief's "turn" has no verb of its own, so this rule
   does not turn the agent before it moves. Open.

The reports and their belief fallback are the ``Report`` class in ``programs.py`` (its docstring has the mapping).

Judged (2026-10-20, task #111). The referee run (the skill layer's harness, gen 5, combat_cover, 8 lanes x 1023
ticks, seed 0; candidates in 5bd3fc1, referee in 52632ec) judged three programs: the shot below ``avoid`` (the old
GUERRILLA), shot first (the shot at 0.6), and facing (shot first plus the ``keep_range`` companion). Two gaps
showed. (1) GUERRILLA made 0 shot requests: ``avoid`` (0.5) holds while the target faces the agent, and a chaser
always does. (2) The shot ran alone, and the bow aims with the head only, within 45 degrees. Shot first and facing
each loosed 4 arrows and landed 4, with 1 kill, from one lane (HP taken 529 and 644, inside the per-lane spread).
Under facing, keep_range held locomotion on all 204 lane-ticks the shortbow held the hands. The tie goes to the
archery range (commit 1112c77): a body that faces the target lets the bow hit a strafer on 85% (gen 5) and 79%
(gen 7) of its arrows, and the bow alone loses a strafer past the head's 45 degrees. So facing is
``P_GUERRILLA``. Shot first and the old GUERRILLA structure are removed.
Known limit of companions (task #114): a goal with a companion ends by its maximum duration, not by COMPLETED.
Open (task #116): on combat_cover the agent cannot back off past about 4 blocks without leaving the arena's 9x9
geometry window. On open ground a same-speed melee chaser follows a retreating agent at 3.4 to 3.9 blocks, so the
6 to 10 window rarely opens. Kiting at range needs cover at range. The window stays 6 to 10 for now.

Not built: "not yet in cover". No report or belief feature measures cover: ``REPORTS`` has none, and the cover asks of
``skill_agent/docs/PERCEPTION_ASKS.md`` are not in the belief. Rule 3 therefore tests "seen" alone. That reads as "not yet
in cover" only if line of sight is symmetric, which is inferred and not measured.

Import this module through ``experimental.neural_agent.jev.programs``: that module registers ``P_GUERRILLA`` as
``PROGRAMS["GUERRILLA"]`` right after ``PROGRAMS``, and this module imports the program DSL from it. Importing this module
first fails on that cycle.
"""

from __future__ import annotations

from experimental.neural_agent.executive.goal_language import Predicate
from experimental.neural_agent.jev.programs import (
    EXIT_DONE,
    AllOf,
    Interrupt,
    Not,
    Program,
    ProgramStep,
    Report,
)
from experimental.neural_agent.tasks.composition import EVADER_PRIORITY, PURSUIT_SKILL

HURT_HEALTH_FRACTION: float = 0.5
"""``avoid`` also fires below this health fraction (``Predicate("self_health_below", ...)`` is strict)."""
AVOID_BAND_BLOCKS: tuple[float, float] = (6.0, 10.0)
"""``avoid``'s band (near, far) in blocks: the skill layer's default (``scripted_bodies.AVOID_BAND_BLOCKS``)."""
BOW_WINDOW_BLOCKS: tuple[float, float] = AVOID_BAND_BLOCKS
"""The distances a shortbow shot is taken at: the band ``avoid`` keeps, 6 to 10 blocks. Measured by the skill layer
(2026-10-20, its shortbow body against a still target, 300 ticks, both lines): every loosed arrow hits at 4.5 and at 8
blocks for the tiers that reach, and the full draw gives 11.4 damage a second at 8 blocks on both lines (9.5 at 4.5 on
gen 7); the first arrow lands about 1.6 s after a facing start, so the earlier window of 3.5 to 4.5 blocks (the local
window's freeze distance) is crossed by a chasing body before the shot. Reach on level ground, head at most 45 degrees
up: gen 7 tier 0 3.25 blocks, tier 1 10.5, tier 2 26.5, tiers 3 and 4 beyond 40 (gen 5: 4.75, 17, then beyond 40).
Not measured: moving targets (the body aims at the current position with no lead)."""
SHOT_HOLD_SECONDS: float = 1.2
"""The shortbow's ``hold_seconds``, the draw time (``requests.py``, the guerrilla verbs): the body takes the largest tier
whose authored charge threshold (0.1, 0.3, 0.6, 0.9, 1.2 s for the iron shortbow on both lines, confirmed by the sim's
adapter) is at most the hold, so 1.2 s is tier 4, the full draw. Chosen on the skill layer's per-tier measurement (#105,
2026-10-20): gen 7 authors 1, 2, 4, 7, 19 damage and launch speeds 12, 22, 35, 50, 85 for the tiers, so the full draw is
the only strong tier there (damage a second 2.4, 3.6, 4.9, 9.5 for tiers 1 to 4 at 4.5 blocks; 11.4 for tier 4 at 8) and
tier 0 cannot reach the window at all; on gen 5 every tier deals about 11 a second (6, 9, 12, 16, 19 damage), so the
hold costs nothing there. The earlier 0.2 s tap (tier 0) was a 1-damage arrow on gen 7."""
AVOID_PRIORITY: float = EVADER_PRIORITY
SHOT_PRIORITY: float = 0.6
"""The shortbow's priority: above ``avoid`` (AVOID_PRIORITY, 0.5), so a clean shot wins over ``avoid``'s facing rule,
which holds on every chase. At 0.4 (the old combat phase priority) the shot made no request in the referee run
(task #111)."""
HIDE_PRIORITY: float = 0.3

_AVOID = Not(AllOf((Not(Report("facing_me")), Not(Predicate("self_health_below", HURT_HEALTH_FRACTION)))))
_SHOT = AllOf(
    (
        Report("seen"),
        Report("distance_above", BOW_WINDOW_BLOCKS[0]),
        Report("distance_below", BOW_WINDOW_BLOCKS[1]),
        Not(Report("in_reach")),
    )
)

P_GUERRILLA = Program(
    steps=(ProgramStep(PURSUIT_SKILL, exit=EXIT_DONE),),
    interrupts=(
        Interrupt(
            "shortbow",
            SHOT_PRIORITY,
            _SHOT,
            params=(SHOT_HOLD_SECONDS,),
            beside=(("keep_range", AVOID_BAND_BLOCKS),),
        ),
        Interrupt("avoid", AVOID_PRIORITY, _AVOID, params=AVOID_BAND_BLOCKS),
        Interrupt("hide", HIDE_PRIORITY, Report("seen")),
    ),
)
"""The guerrilla program (module docstring): ``PROGRAMS["GUERRILLA"]`` in ``programs.py``."""
