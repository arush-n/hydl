"""The machinery: how a game is defined, checked and registered.

Nothing here knows about any particular game. Content lives in
``arena.tasks.games``.
"""

from arena.tasks.framework.base import (
    PERCEPTION_PROVIDERS,
    Difficulty,
    Task,
    head_coverage,
)
from arena.tasks.framework.goals import Goal, all_goals, any_goals
from arena.tasks.framework.rewards import (
    RewardConfig,
    TaskReward,
    compose_rewards,
    reward_for_goal,
)
from arena.tasks.framework.registry import (
    all_tasks,
    clear,
    get,
    names,
    register,
    unregister,
)
from arena.tasks.framework.validate import TaskConformanceError, report, validate
from arena.worlds import PublishedWorldSpec, WorldSpec

__all__ = [
    "PERCEPTION_PROVIDERS",
    "Difficulty",
    "Goal",
    "RewardConfig",
    "Task",
    "TaskReward",
    "TaskConformanceError",
    "WorldSpec",
    "PublishedWorldSpec",
    "all_goals",
    "all_tasks",
    "any_goals",
    "clear",
    "compose_rewards",
    "get",
    "head_coverage",
    "names",
    "register",
    "report",
    "reward_for_goal",
    "unregister",
    "validate",
]
