"""Reduce the opponent to zero health."""

from __future__ import annotations

from arena.tasks.framework.base import Difficulty, Task
from arena.tasks.framework.goals import defeat_target

DUEL = Task(
    name="duel",
    description="Reduce the opponent to zero health.",
    heads_exercised=(),
    loadout="iron_sword",
    difficulties=(
        # The inert rungs remove survival pressure so offence is the only axis;
        # against a lethal opponent avoidance is a cheap local optimum.
        Difficulty("target_inert", opponent_armed=False, opponent_profile="iron_sword"),
        Difficulty(
            "target_inert_tanky",
            opponent_armed=False,
            opponent_profile="iron_sword",
            target_max_health=180.0,
        ),
        Difficulty("target_armed", opponent_armed=True, opponent_profile="iron_sword"),
    ),
    success=defeat_target(),
)

__all__ = ["DUEL"]
