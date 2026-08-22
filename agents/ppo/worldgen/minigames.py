"""Small custom WorldGen V2 games used to verify generic IL composition."""

from __future__ import annotations

from pathlib import Path

from arena.publication_worlds import PublishedWorldSpec
from arena.tasks.framework.base import Difficulty, Task
from arena.tasks.framework.goals import (
    Goal,
    all_goals,
    keep_health_above,
    keep_target_visible,
    reduce_target_health_below,
)
from arena.tasks.framework.rewards import (
    RewardConfig,
    compose_rewards,
    reward_for_goal,
)
from arena.worlds import ArenaWorldSpec, WorldSpec


_DENSE = RewardConfig(native_scale=0.0, state_scale=0.01, step_penalty=0.001)


def combat_minigame(
    name: str,
    description: str,
    *,
    goals: tuple[Goal, ...],
    loadout: str,
    opponent: str,
    world: ArenaWorldSpec,
) -> Task:
    """Compose a custom task without coupling its rules to the trainer."""

    success = all_goals(*goals)
    reward = compose_rewards(
        *(reward_for_goal(goal, _DENSE) for goal in goals),
        label=f"{name}_reward",
        config=_DENSE,
    )
    return Task(
        name=name,
        description=description,
        heads_exercised=(),
        loadout=loadout,
        difficulties=(
            Difficulty("inert", opponent_armed=False, opponent_profile=opponent),
            Difficulty("armed", opponent_armed=True, opponent_profile=opponent),
        ),
        success=success,
        reward=reward,
        world=world,
    )


def publication_minigame(
    publication: str | Path,
    *,
    name: str = "mountain_ruin_pressure",
) -> Task:
    """Bind custom rules to one verified, immutable WorldGen publication."""

    return combat_minigame(
        name,
        "Track and pressure an inert target in the published mountain ruins.",
        goals=(
            keep_target_visible(fraction=0.75),
            reduce_target_health_below(0.9),
        ),
        loadout="iron_sword",
        opponent="iron_daggers",
        world=PublishedWorldSpec(name, publication, resident_capacity=2),
    )


TAIGA_PRESSURE = combat_minigame(
    "taiga_pressure",
    "Track and wound a dagger opponent in a two-seed Taiga world.",
    goals=(keep_target_visible(fraction=0.75), reduce_target_health_below(0.85)),
    loadout="iron_sword",
    opponent="iron_daggers",
    world=WorldSpec(
        "taiga_pressure_pool",
        "Zone3_Taiga1",
        570069,
        pool_seeds=(570073,),
    ),
)

STONE_CONTROL = combat_minigame(
    "stone_control",
    "Trade safely while pressuring a mace opponent on a distinct stone world.",
    goals=(keep_health_above(0.5), reduce_target_health_below(0.9)),
    loadout="iron_daggers",
    opponent="iron_mace",
    world=WorldSpec(
        "stone_control_pool",
        "Default",
        570070,
        pool_seeds=(570075,),
    ),
)

CUSTOM_MINIGAMES = (TAIGA_PRESSURE, STONE_CONTROL)

__all__ = [
    "CUSTOM_MINIGAMES",
    "STONE_CONTROL",
    "TAIGA_PRESSURE",
    "combat_minigame",
    "publication_minigame",
]
