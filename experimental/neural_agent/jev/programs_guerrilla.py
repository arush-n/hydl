"""The guerrilla program (2026-10-20, task #99): avoid when threatened, shoot from range, hide when seen.

One step, pursuit until "done", and three interrupts. The priorities decide (``programs.py`` item 5): the highest priority
whose condition holds on the lane wins, and ties go to the lower index.

1. ``avoid`` (AVOID_PRIORITY, the evader's 0.5) while the target faces the agent (report ``facing_me``) or the agent's
   health is below HURT_HEALTH_FRACTION (``Predicate("self_health_below", 0.5)``, strict). The OR is built from ``Not`` and
   ``AllOf``: ``Not(AllOf((Not(facing_me), Not(self_health_below))))``. Scalars: AVOID_BAND_BLOCKS, the skill layer's default
   band of (6, 10) blocks.
2. ``shortbow`` (SHOT_PRIORITY, 0.4) when the shot is clean: the target is seen, its distance is inside BOW_WINDOW_BLOCKS,
   and it is not in melee reach (report ``in_reach`` is off). Scalar: ``hold_seconds`` = SHOT_HOLD_SECONDS = 0.2 s, which
   selects the smallest draw tier (ability slot 0, the quick shot) under the contract's mapping (the constant's docstring).
3. ``hide`` (HIDE_PRIORITY, 0.3) while the target is seen. It names no cover point, so the hide body retreats as ``avoid``
   does (``skill_agent/runtime/scripted_bodies.py``, ``hide``) until a cover position is supplied.
4. Search is the step: while the target is unseen no interrupt holds, and the step's pursuit runs. The executor's request
   for pursuit carries the remembered position while the target is unseen (``belief/target_memory.py``,
   ``target_arguments``, task #96), so the body heads for it. The brief's "turn" has no verb of its own, so this rule
   does not turn the agent before it moves. Open.

The reports and their belief fallback are the ``Report`` class in ``programs.py`` (its docstring has the mapping).

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
    COMBAT_PHASE_PRIORITY,
    EXIT_DONE,
    STRIKE_REACH_BLOCKS,
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
BOW_WINDOW_BLOCKS: tuple[float, float] = (STRIKE_REACH_BLOCKS, 4.5)
"""The distances a shortbow shot is taken at: beyond the strike reach, and inside 4.5 blocks. The 4.5 is the freeze
distance of the 9-block local window (``skill_agent/docs/PERCEPTION_ASKS.md``, the coordinator's bow recipe of 2026-10-20:
a shot fires at 3.5 blocks, and the arsenal freezes at 4.5). The 0.5.7 shortbow (``hytalegym/jax/combat/arsenal/profiles/
hytale_0_5_7.py``, ``_iron_shortbow``) gives each tier's projectile speed and gravity and no range, so no bow range is
measured for the Region route."""
SHOT_HOLD_SECONDS: float = 0.2
"""The shortbow's ``hold_seconds``, the draw time (``requests.py``, the guerrilla verbs): the body takes the largest tier
whose authored charge threshold (0.1, 0.3, 0.6, 0.9, 1.2 s for the 0.5.7 iron shortbow) is at most the hold, and a hold
below the first threshold is tier 0. 0.2 s selects tier 0, the quick shot (the arrow spawns at tick 9), and would also
under the two spawn-tick readings that were in play before the contract pinned the mapping (tiers from 0.41 s at 0.045 s
a tick, or from 0.3 s at 30 ticks a second), since it sits below their first tier. The body is not built yet."""
AVOID_PRIORITY: float = EVADER_PRIORITY
SHOT_PRIORITY: float = COMBAT_PHASE_PRIORITY
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
        Interrupt("avoid", AVOID_PRIORITY, _AVOID, params=AVOID_BAND_BLOCKS),
        Interrupt("shortbow", SHOT_PRIORITY, _SHOT, params=(SHOT_HOLD_SECONDS,)),
        Interrupt("hide", HIDE_PRIORITY, Report("seen")),
    ),
)
"""The guerrilla program (module docstring): ``PROGRAMS["GUERRILLA"]`` in ``programs.py``."""
