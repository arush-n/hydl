"""Outlast an armed opponent."""

from __future__ import annotations

from arena.tasks.framework.base import Difficulty, Task
from arena.tasks.framework.goals import stay_alive

SURVIVE = Task(
    name="survive",
    description="Outlast an armed opponent for the whole episode.",
    heads_exercised=(),
    loadout="iron_sword",
    difficulties=(
        # An unarmed first rung is not a formality: against an inert target
        # every policy survives, so it establishes the measurement floor.
        Difficulty("floor_unarmed", opponent_armed=False, opponent_profile="iron_sword"),
        Difficulty(
            "armed_fragile",
            opponent_armed=True,
            opponent_profile="iron_daggers",
            agent_max_health=160.0,
        ),
        Difficulty("armed", opponent_armed=True, opponent_profile="iron_sword"),
        Difficulty(
            "armed_heavy",
            opponent_armed=True,
            opponent_profile="iron_mace",
            agent_max_health=80.0,
        ),
    ),
    success=stay_alive(),
)

__all__ = ["SURVIVE"]
