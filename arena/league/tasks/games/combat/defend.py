"""Take a beating and survive it -- the only game that exercises the guard head.

**Read GAP-15 before trusting a score here.** Guard was measured exactly inert
on the scripted-target scene: 414.0 damage taken with guard forced on and 414.0
with it forced off, because the scripted target's damage path
(`_step_core.py:419-431`) is a flat subtraction with no mitigation term at all.
Against an armed Arsenal opponent it does work, and costs the attacker about
14% (60.0 against 70.0).

That creates a real tension with the framework's ladder rule, which requires an
unarmed opening rung because against a lethal opponent avoidance is a cheap
local optimum a policy will not leave. The gate rejected an all-armed version of
this game, and the rule is right -- so the opening rung stays, with the caveat
stated instead of hidden: **`scripted_inert` teaches the motion and measures no
mitigation whatsoever.** Only the armed rungs score guard. Read a promotion off
the first rung as "the policy learned to press the button", nothing more.
"""

from __future__ import annotations

from arena.tasks.framework.base import Difficulty, Task
from arena.tasks.framework.goals import keep_health_above

DEFEND = Task(
    name="defend",
    description="Hold most of your health while an armed opponent presses you.",
    heads_exercised=("guard_off_on", "locomotion_gait_compass"),
    loadout="iron_sword",
    difficulties=(
        Difficulty("scripted_inert", opponent_armed=False, opponent_profile="iron_sword"),
        Difficulty("armed_dagger", opponent_armed=True, opponent_profile="iron_daggers"),
        Difficulty("armed_battleaxe", opponent_armed=True, opponent_profile="iron_battleaxe"),
    ),
    success=keep_health_above(0.5),
)

__all__ = ["DEFEND"]
