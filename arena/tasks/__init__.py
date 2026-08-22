"""Compatibility namespace for tasks now physically owned by ``arena.league``.

The task tree moved to ``arena/league/tasks`` without changing its published
``arena.tasks`` import contract.  Point this package's search path at the new
owner so existing task, curriculum, PPO, and plugin imports resolve to one
implementation rather than requiring a second copied task tree.
"""

from pathlib import Path


__path__ = [str(Path(__file__).resolve().parent.parent / "league" / "tasks")]

from arena.tasks.framework import (  # noqa: E402
    PERCEPTION_PROVIDERS,
    Difficulty,
    Goal,
    PublishedWorldSpec,
    RewardConfig,
    Task,
    TaskConformanceError,
    TaskReward,
    WorldSpec,
    all_goals,
    all_tasks,
    any_goals,
    clear,
    compose_rewards,
    get,
    head_coverage,
    names,
    register,
    report,
    reward_for_goal,
    unregister,
    validate,
)


__all__ = [
    "PERCEPTION_PROVIDERS",
    "Difficulty",
    "Goal",
    "PublishedWorldSpec",
    "RewardConfig",
    "Task",
    "TaskConformanceError",
    "TaskReward",
    "WorldSpec",
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
