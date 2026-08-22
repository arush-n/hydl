"""Stay airborne on purpose -- the only game that exercises the jump head.

`hold_movement_state(state="jumping", ...)` reads `movement_state_f32.jumping`,
which the full-observation audit found **flat at 0.0** on
`combat/open_flat_control` under a uniform-legal agent across 384 ticks. That
is a candidate dead column, not a proven one: a random agent rarely holds a jump,
and the control scene is a bare plane. This game is the deliberate forcing
function that tells those two apart -- if `jumping` stays flat with a policy
*trying* to jump, the column is dead and that is a finding.

The fraction is deliberately low. A jump is a short arc, so demanding a large
fraction of ticks in the air would make the goal unreachable rather than hard.
"""

from __future__ import annotations

from arena.tasks.framework.base import Difficulty, Task
from arena.tasks.framework.goals import hold_movement_state

TRAVERSE = Task(
    name="traverse",
    description="Spend part of the episode airborne rather than grounded.",
    heads_exercised=("jump_off_on", "locomotion_gait_compass"),
    loadout="iron_sword",
    difficulties=(
        Difficulty("unpressed", opponent_armed=False, opponent_profile="iron_sword"),
        Difficulty("pressed", opponent_armed=True, opponent_profile="iron_daggers"),
    ),
    success=hold_movement_state(state="jumping", fraction=0.05),
)

__all__ = ["TRAVERSE"]
