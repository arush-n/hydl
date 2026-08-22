"""Small task ports on three distinct exact WorldGen V2 worlds."""

from __future__ import annotations

from arena.tasks.framework.base import Difficulty, Task
from arena.tasks.framework.goals import (
    all_goals,
    defeat_target,
    keep_target_visible,
    reduce_target_health_below,
    stay_alive,
)
from arena.tasks.framework.rewards import (
    RewardConfig,
    compose_rewards,
    reward_for_goal,
)
from arena.worlds import WorldSpec

_DENSE = RewardConfig(native_scale=0.0, state_scale=0.01, step_penalty=0.001)
_PLAINS_GOAL = defeat_target()
_DESERT_TRACK = keep_target_visible(fraction=0.9)
_DESERT_PRESSURE = reduce_target_health_below(0.75)
_DESERT_GOAL = all_goals(_DESERT_TRACK, _DESERT_PRESSURE)
_DESERT_REWARD = compose_rewards(
    reward_for_goal(_DESERT_TRACK),
    reward_for_goal(_DESERT_PRESSURE),
    label="track_and_pressure",
    config=_DENSE,
)
_VOLCANIC_GOAL = stay_alive()

PLAINS_DUEL = Task(
    name="plains_duel",
    description="Duel across Plains seeds 570057 and 570061.",
    heads_exercised=(),
    loadout="iron_sword",
    difficulties=(
        Difficulty("inert", opponent_armed=False, opponent_profile="iron_sword"),
        Difficulty("armed", opponent_armed=True, opponent_profile="iron_sword"),
    ),
    success=_PLAINS_GOAL,
    reward=reward_for_goal(_PLAINS_GOAL, _DENSE),
    world=WorldSpec(
        "plains_pool", "Zone1_Plains1", 570057, pool_seeds=(570061,)
    ),
)

DESERT_TRACK = Task(
    name="desert_track",
    description="Track and pressure an opponent across Desert seeds 570058 and 570060.",
    heads_exercised=(),
    loadout="iron_daggers",
    difficulties=(
        Difficulty("inert", opponent_armed=False, opponent_profile="iron_sword"),
        Difficulty("armed", opponent_armed=True, opponent_profile="iron_daggers"),
    ),
    success=_DESERT_GOAL,
    reward=_DESERT_REWARD,
    world=WorldSpec(
        "desert_pool", "Zone2_Desert1", 570058, pool_seeds=(570060,)
    ),
)

VOLCANIC_SURVIVE = Task(
    name="volcanic_survive",
    description="Stay alive across Volcanic seeds 570060 and 570064.",
    heads_exercised=(),
    loadout="iron_mace",
    difficulties=(
        Difficulty("unarmed", opponent_armed=False, opponent_profile="iron_sword"),
        Difficulty("armed", opponent_armed=True, opponent_profile="iron_mace"),
    ),
    success=_VOLCANIC_GOAL,
    reward=reward_for_goal(_VOLCANIC_GOAL, _DENSE),
    world=WorldSpec(
        "volcanic_pool",
        "Zone4_Volcanic1",
        570060,
        pool_seeds=(570064,),
    ),
)

GENERATED = (PLAINS_DUEL, DESERT_TRACK, VOLCANIC_SURVIVE)

__all__ = ["DESERT_TRACK", "GENERATED", "PLAINS_DUEL", "VOLCANIC_SURVIVE"]
