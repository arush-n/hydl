"""Keep the opponent visible and centred -- an aim skill no fight rewards."""

from __future__ import annotations

from arena.tasks.framework.base import Difficulty, Task
from arena.tasks.framework.goals import keep_target_visible

TRACK = Task(
    name="track",
    description="Keep the opponent visible for most of the episode.",
    heads_exercised=(),
    loadout="iron_sword",
    difficulties=(
        Difficulty("static_inert", opponent_armed=False, opponent_profile="iron_sword"),
        Difficulty("mobile_armed", opponent_armed=True, opponent_profile="iron_daggers"),
    ),
    success=keep_target_visible(fraction=0.9),
)

__all__ = ["TRACK"]
