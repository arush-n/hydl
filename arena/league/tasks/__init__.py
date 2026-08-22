"""Games and the framework that defines them.

* ``arena.tasks.framework`` -- the machinery (Task, Difficulty, goals,
  criteria, the conformance gate, the registry).
* ``arena.tasks.games`` -- the content: authored games, grouped by domain.

Framework names are re-exported here so ``from arena.tasks import Task`` keeps
working.
"""

from arena.tasks.framework import (
    PERCEPTION_PROVIDERS,
    Difficulty,
    Goal,
    RewardConfig,
    Task,
    TaskReward,
    TaskConformanceError,
    PublishedWorldSpec,
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
    "RewardConfig",
    "Task",
    "TaskReward",
    "TaskConformanceError",
    "PublishedWorldSpec",
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
