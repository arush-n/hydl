"""Task metadata and direct JAX scene construction.

Every scene binds perception explicitly. ``Difficulty.opponent_armed`` also
selects the JAX opponent controller explicitly, so an inert teaching rung can
never be confused with an armed one. ``Task.world`` optionally binds one exact
WorldGen V2 capture through the JAX Region runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from hytalegym.jax.combat.arsenal.environment import (
    open_flat_arsenal_world_capabilities,
)
from hytalegym.jax.combat.opponents.runtime.policy import (
    first_legal_opponent_ability_slots,
    inert_opponent_ability_slots,
)

from arena.jax_env import COMBAT_PARAMETER_NAMES, ArenaScene, make_scene
from arena.tasks.framework.rewards import TaskReward
from arena.worlds import ArenaWorldSpec, PublishedWorldSpec, WorldSpec


#: Perception providers, bound on every task scene.  Without one of these the
#: agent cannot see the opponent at all.
PERCEPTION_PROVIDERS: Mapping[str, Any] = {
    "world_capability_provider": open_flat_arsenal_world_capabilities
}


@dataclass(frozen=True, slots=True)
class Difficulty:
    """One rung of a task's ladder.

    ``opponent_armed`` has no default on purpose.  An unarmed opponent is a
    legitimate teaching tool -- it is how you isolate offence from defence --
    but it must be *chosen*, because an unarmed opponent silently removes two
    of the four reward components.
    """

    name: str
    opponent_armed: bool
    opponent_profile: str | None = None
    agent_max_health: float | None = None
    target_max_health: float | None = None
    sensor_range: float | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise ValueError("difficulty name must be a non-empty label")
        unknown = sorted(set(self.parameters) - set(COMBAT_PARAMETER_NAMES))
        if unknown:
            raise ValueError(
                "unknown CombatParams override(s): "
                + ", ".join(unknown)
                + f"; CombatParams declares {len(COMBAT_PARAMETER_NAMES)} fields"
            )
        if self.opponent_armed and self.opponent_profile is None:
            raise ValueError(
                f"difficulty {self.name!r} arms the opponent but names no "
                "opponent_profile; an armed opponent needs a loadout to swing"
            )

    @property
    def ability_provider(self) -> Any:
        """The provider that decides whether the target ever attacks."""

        return (
            first_legal_opponent_ability_slots
            if self.opponent_armed
            else inert_opponent_ability_slots
        )

    def combat_parameters(self) -> dict[str, Any]:
        overrides = dict(self.parameters)
        if self.agent_max_health is not None:
            overrides["agent_max_health"] = self.agent_max_health
        if self.target_max_health is not None:
            overrides["target_max_health"] = self.target_max_health
        if self.sensor_range is not None:
            overrides["sensor_range"] = self.sensor_range
        return overrides


@dataclass(frozen=True, slots=True)
class Task:
    """One unique game, with a ladder of difficulties.

    ``success`` is handed the recorded rollout (a mapping of observation group
    name -> array, plus whatever the task's ``record`` added) and returns a
    per-environment boolean array.  It must read columns **by name**; the
    observation width has moved twice in one day and any width-indexed reader
    is a latent silent failure.
    """

    name: str
    description: str
    #: Which action heads this game is built to exercise, for coverage checks.
    heads_exercised: tuple[str, ...]
    loadout: str
    difficulties: tuple[Difficulty, ...]
    success: Callable[[Mapping[str, Any]], Any]
    #: Optional exact WorldGen V2 terrain for this task.
    world: ArenaWorldSpec | None = None
    #: Small escape hatch for JAX providers, ``microticks`` and ``target_active``.
    scene_options: Mapping[str, Any] = field(default_factory=dict)
    #: Optional explicit per-step JAX reward; success remains an evaluator.
    reward: TaskReward | None = None

    def __post_init__(self) -> None:
        if "/" in self.name:
            raise ValueError("task name must not contain '/'; it is a segment")
        if not self.difficulties:
            raise ValueError(f"task {self.name!r} declares no difficulties")
        if self.world is not None and not isinstance(
            self.world, (WorldSpec, PublishedWorldSpec)
        ):
            raise TypeError("world must be a WorldSpec or PublishedWorldSpec")
        if self.reward is not None and not isinstance(self.reward, TaskReward):
            raise TypeError("reward must be a TaskReward")
        seen = [d.name for d in self.difficulties]
        if len(set(seen)) != len(seen):
            raise ValueError(f"task {self.name!r} has duplicate difficulty names")
        if not self.heads_exercised and self.goal is not None:
            # A Goal already knows which heads it implies, so a task built from
            # one need not restate them.
            object.__setattr__(self, "heads_exercised", self.goal.heads_implied)
        if not self.heads_exercised:
            raise ValueError(
                f"task {self.name!r} names no action heads; a game that "
                "exercises nothing cannot be justified in a curriculum"
            )

    @property
    def goal(self) -> Any:
        """The :class:`arena.tasks.goals.Goal` behind this task, if any.

        ``success`` may be a bare callable, so this is ``None`` for tasks that
        hand-write their criterion.
        """

        from arena.tasks.framework.goals import Goal  # local: goals imports criteria

        return self.success if isinstance(self.success, Goal) else None

    def difficulty(self, name: str) -> Difficulty:
        for rung in self.difficulties:
            if rung.name == name:
                return rung
        available = ", ".join(d.name for d in self.difficulties)
        raise KeyError(
            f"task {self.name!r} has no difficulty {name!r}; have: {available}"
        )

    def build_scene(self, difficulty: str, *, batch: int) -> ArenaScene:
        """Build this game at one difficulty, with both traps closed."""

        if batch < 1:
            raise ValueError("batch must be positive")
        rung = self.difficulty(difficulty)

        providers = {} if self.world is not None else dict(PERCEPTION_PROVIDERS)
        options = dict(self.scene_options)
        supplied = dict(options.pop("providers", {}) or {})
        providers.update(supplied)

        microticks = options.pop("microticks", 1)
        target_active = options.pop("target_active", True)
        if options:
            raise ValueError("unknown scene option(s): " + ", ".join(sorted(options)))

        runtime_options: dict[str, Any] = {}
        if rung.sensor_range is not None:
            runtime_options["sensor_range"] = rung.sensor_range

        opponents = rung.opponent_profile

        goal = self.goal
        if (
            getattr(goal, "requires_world", False)
            and self.world is None
            and not (
                providers.get("geometry_provider")
                or providers.get("world_feature_provider")
            )
        ):
            import warnings

            warnings.warn(
                f"game {self.name!r} scores {goal.label} against terrain, "
                "traversal, interaction or hazard columns, but this scene binds "
                "neither a WorldGen V2 world nor a geometry/feature provider, so "
                "those columns are structurally zero. Set Task.world or bind a "
                "JAX Region provider through scene_options. See GAP-19.",
                RuntimeWarning,
                stacklevel=2,
            )

        return make_scene(
            f"arena/{self.name}.{rung.name}",
            loadout=self.loadout,
            opponent=opponents,
            opponent_ability_provider=rung.ability_provider,
            batch=batch,
            providers=providers,
            parameters=rung.combat_parameters(),
            reward=self.reward,
            world=self.world,
            microticks=microticks,
            target_active=target_active,
            **runtime_options,
        )


def head_coverage(tasks: Sequence[Task]) -> dict[str, tuple[str, ...]]:
    """Which tasks exercise each action head.

    A head with no task is a capability nothing in the curriculum ever
    teaches -- which is exactly how an action surface grows dead logits.
    """

    coverage: dict[str, list[str]] = {}
    for task in tasks:
        for head in task.heads_exercised:
            coverage.setdefault(head, []).append(task.name)
    return {head: tuple(names) for head, names in sorted(coverage.items())}


__all__ = [
    "PERCEPTION_PROVIDERS",
    "Difficulty",
    "Task",
    "head_coverage",
]
