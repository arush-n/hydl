"""Bring an interaction target inside reach.

Exercises the world-interaction heads. On a bare combat scene the block,
recipe and use heads are never legal -- 48 of the 54 permanently-illegal
logits are explained by scene content, not broken mechanics -- so a game like
this is the only way they get selected at all.
"""

from __future__ import annotations

from arena.tasks.framework.base import Difficulty, Task
from arena.tasks.framework.goals import reach_interaction

REACH = Task(
    name="reach",
    description="Bring an interaction target within reach.",
    heads_exercised=(),
    loadout="iron_sword",
    difficulties=(
        Difficulty("adjacent", opponent_armed=False),
        Difficulty("across_room", opponent_armed=False, sensor_range=48.0),
    ),
    success=reach_interaction(),
)

__all__ = ["REACH"]
