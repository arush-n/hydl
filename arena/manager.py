"""The game manager: hold games, build their scenes, run and score them.

One object that a training loop can talk to. It accepts a :class:`Task`, or a
:class:`~arena.component.Component` that builds Tasks -- registering a
component expands it into variants, so one template contributes tens of games
without any of them being written out.

    manager = GameManager()
    manager.register(DUEL)                      # one game
    manager.register(endure_template, steps=4)  # 64 variants of one template

    manager.build("duel", "target_inert", batch=8)
    result = manager.evaluate("duel", "target_inert", policy, ticks=512)
    result.success_rate            # -> feeds a curriculum ladder
    result.scene_contract_sha256   # -> what it was actually measured against

Scoring is the task's own success criterion, so a game means the same thing
here as it does in a ladder or a league match.

Every game is validated on the way in, including a scene build, because a game
that cannot construct its own scene is worth discovering at registration rather
than mid-run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import numpy as np

from hytalegym.jax.compilation_cache import configure_persistent_compilation_cache

from arena.component import Component
from arena.jax_contract import all_observation_groups, observation_groups
from arena.jax_env import ArenaHandle, ParameterizedPolicy
from arena.tasks.framework.base import Task
from arena.tasks.framework.criteria import criterion_groups
from arena.tasks.framework.validate import validate


def default_record(transition: Any) -> dict[str, Any]:
    """Retain every named group that Arena criteria can read."""

    return all_observation_groups(transition.next_actor_input.legal_observation)


def record_groups(groups: tuple[str, ...]):
    """Build a traced recorder for one static observation subset."""

    def record(transition: Any) -> dict[str, Any]:
        return observation_groups(
            transition.next_actor_input.legal_observation,
            groups,
        )

    return record


@dataclass(frozen=True, slots=True)
class GameResult:
    """One evaluation of one game at one difficulty."""

    task: str
    difficulty: str
    goal: str | None
    success: np.ndarray
    ticks: int
    batch: int
    scene_name: str
    scene_contract_sha256: str

    @property
    def success_rate(self) -> float:
        return float(np.asarray(self.success).mean())

    def describe(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "difficulty": self.difficulty,
            "goal": self.goal,
            "success_rate": self.success_rate,
            "successes": int(np.asarray(self.success).sum()),
            "batch": self.batch,
            "ticks": self.ticks,
            "scene": self.scene_name,
            "scene_contract_sha256": self.scene_contract_sha256,
        }


class GameManager:
    """Games in, built scenes and scored rollouts out."""

    def __init__(
        self,
        *,
        record: Callable[[Any], Mapping[str, Any]] | None = None,
        cache_limit: int | None = None,
    ):
        """``record`` overrides what every rollout retains.

        Built-in criteria automatically retain only their required groups.
        Opaque custom criteria fall back to every readable group unless an
        explicit recorder is supplied. ``cache_limit`` defaults to four on a
        GPU and 32 on CPU so compiled specializations cannot casually fill
        accelerator memory.
        """

        import jax

        if cache_limit is not None and (
            isinstance(cache_limit, bool)
            or not isinstance(cache_limit, int)
            or not 1 <= cache_limit <= 64
        ):
            raise ValueError("cache_limit must be an integer from 1 to 64")
        self.backend = jax.default_backend()
        self.cache_limit = cache_limit or (4 if self.backend == "gpu" else 32)
        self._games: dict[str, Task] = {}
        self._record_override = record
        self._build_cache: dict[tuple[str, str, int], tuple[Any, Any]] = {}
        self._record_cache: dict[tuple[str, ...], Callable] = {}
        self._collector_cache: dict[tuple[Any, ...], Callable] = {}
        self._evaluator_cache: dict[tuple[Any, ...], Callable] = {}
        self.compilation_cache = configure_persistent_compilation_cache(
            Path(__file__).resolve().parents[1] / ".jax_cache"
        )

    # -- registration ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._games)

    def __contains__(self, name: object) -> bool:
        return name in self._games

    def __iter__(self) -> Iterator[Task]:
        return iter(self._games[name] for name in sorted(self._games))

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._games))

    def get(self, name: str) -> Task:
        try:
            return self._games[name]
        except KeyError:
            known = ", ".join(sorted(self._games)) or "<no games registered>"
            raise KeyError(f"no game {name!r}; registered: {known}") from None

    def register(
        self,
        game: Task | Component,
        *,
        replace: bool = False,
        build_scenes: bool = True,
        steps: int = 3,
    ) -> tuple[Task, ...]:
        """Add a game, or expand a template into its grid of variants.

        ``steps`` applies only to a Component: it is the grid resolution per
        parameter axis. ``build_scenes`` validates that every difficulty can
        actually construct, which is the check worth paying for at registration.
        """

        if isinstance(game, Component):
            produced = [variant.value for variant in game.grid(steps=steps)]
            if not produced:
                raise ValueError(f"component {game.name!r} produced no variants")
            for task in produced:
                if not isinstance(task, Task):
                    raise TypeError(
                        f"component {game.name!r} builds {type(task).__name__}, "
                        "not a Task; a game manager can only hold Tasks"
                    )
        elif isinstance(game, Task):
            produced = [game]
        else:
            raise TypeError(
                f"expected a Task or a Component of Tasks, got {type(game).__name__}"
            )

        added: list[Task] = []
        for task in produced:
            validate(task, build_scenes=build_scenes)
            existing = self._games.get(task.name)
            if existing is not None and existing is not task and not replace:
                raise ValueError(
                    f"game {task.name!r} is already registered; pass replace=True "
                    "to override it deliberately. A silently swapped game makes "
                    "a training run impossible to reproduce."
                )
            self._games[task.name] = task
            if existing is not task:
                self._invalidate(task.name)
            added.append(task)
        return tuple(added)

    # -- building and running --------------------------------------------------

    def build(self, name: str, difficulty: str, *, batch: int):
        """Build one game at one difficulty and return a runnable handle."""

        task = self.get(name)
        key = (name, difficulty, batch)
        cached = self._build_cache.get(key)
        if cached is not None:
            return cached
        scene = task.build_scene(difficulty, batch=batch)
        built = (ArenaHandle(scene), scene)
        if len(self._build_cache) >= self.cache_limit:
            oldest = next(iter(self._build_cache))
            self._build_cache.pop(oldest)
            self._collector_cache = {
                collector_key: value
                for collector_key, value in self._collector_cache.items()
                if collector_key[:3] != oldest
            }
            self._evaluator_cache = {
                evaluator_key: value
                for evaluator_key, value in self._evaluator_cache.items()
                if evaluator_key[:3] != oldest
            }
        self._build_cache[key] = built
        return built

    def _invalidate(self, name: str) -> None:
        self._build_cache = {
            key: value for key, value in self._build_cache.items() if key[0] != name
        }
        self._collector_cache = {
            key: value for key, value in self._collector_cache.items() if key[0] != name
        }
        self._evaluator_cache = {
            key: value for key, value in self._evaluator_cache.items() if key[0] != name
        }

    def _record_for(self, task: Task):
        if self._record_override is not None:
            return self._record_override
        groups = criterion_groups(task.success)
        if groups is None:
            return default_record
        key = tuple(sorted(groups))
        if key not in self._record_cache:
            if len(self._record_cache) >= self.cache_limit:
                self._record_cache.pop(next(iter(self._record_cache)))
            self._record_cache[key] = record_groups(key)
        return self._record_cache[key]

    def evaluate(
        self,
        name: str,
        difficulty: str,
        policy: Any,
        *,
        ticks: int,
        batch: int = 8,
        key: Any = None,
    ) -> GameResult:
        """Run ``policy`` on one game and score the task's own criterion."""

        import jax

        task = self.get(name)
        goal = task.goal
        required = getattr(goal, "minimum_ticks", 1) if goal else 1
        if ticks < required:
            raise ValueError(
                f"game {name!r} needs at least {required} ticks (its goal "
                f"declares a {required}-tick window); got {ticks}"
            )

        handle, scene = self.build(name, difficulty, batch=batch)
        initial_carry = getattr(policy, "initial_carry", None)
        parameterized = isinstance(policy, ParameterizedPolicy)
        compiled_policy = policy.apply if parameterized else policy
        policy_identity = (
            ("parameterized", policy.program_identity)
            if parameterized
            else ("bound", id(policy))
        )
        recorder = self._record_for(task)
        collector_key = (
            name,
            difficulty,
            batch,
            ticks,
            policy_identity,
            id(recorder),
            str(jax.tree.structure(initial_carry)),
        )
        root_key = jax.random.key(0) if key is None else key
        runner_args = (
            (policy.parameters, root_key, initial_carry)
            if parameterized
            else (root_key, initial_carry)
        )
        if criterion_groups(task.success) is not None:
            evaluator = self._evaluator_cache.get(collector_key)
            if evaluator is None:
                compile_evaluator = (
                    handle.compile_parameterized_evaluator
                    if parameterized
                    else handle.compile_evaluator
                )
                evaluator = compile_evaluator(
                    compiled_policy,
                    recorder,
                    task.success,
                    ticks,
                    initial_carry=initial_carry,
                )
                if len(self._evaluator_cache) >= self.cache_limit:
                    self._evaluator_cache.pop(next(iter(self._evaluator_cache)))
                self._evaluator_cache[collector_key] = evaluator
            success = np.asarray(evaluator(*runner_args))
        else:
            collector = self._collector_cache.get(collector_key)
            if collector is None:
                compile_collector = (
                    handle.compile_parameterized_collector
                    if parameterized
                    else handle.compile_collector
                )
                collector = compile_collector(
                    compiled_policy,
                    recorder,
                    ticks,
                    initial_carry=initial_carry,
                )
                if len(self._collector_cache) >= self.cache_limit:
                    self._collector_cache.pop(next(iter(self._collector_cache)))
                self._collector_cache[collector_key] = collector
            _final, record = collector(*runner_args)
            success = np.asarray(task.success(record))
        if success.shape != (batch,):
            raise ValueError(
                f"game {name!r} scored shape {success.shape}, expected ({batch},); "
                "its success criterion collapsed the batch"
            )
        return GameResult(
            task=name,
            difficulty=difficulty,
            goal=goal.label if goal else None,
            success=success,
            ticks=ticks,
            batch=batch,
            scene_name=scene.name,
            scene_contract_sha256=scene.contract_sha256,
        )

    def evaluate_all(
        self, policy: Any, *, ticks: int, batch: int = 8, key: Any = None
    ) -> list[GameResult]:
        """Every registered game at every difficulty -- a coverage sweep."""

        results: list[GameResult] = []
        for task in self:
            for rung in task.difficulties:
                results.append(
                    self.evaluate(
                        task.name, rung.name, policy, ticks=ticks, batch=batch, key=key
                    )
                )
        return results

    # -- reporting -------------------------------------------------------------

    def head_coverage(self) -> dict[str, tuple[str, ...]]:
        """Which games exercise each action head.

        A head no game touches is a capability nothing in training ever teaches,
        which is how an action surface accumulates dead logits.
        """

        from arena.tasks.framework.base import head_coverage as coverage

        return coverage(tuple(self))

    def describe(self) -> dict[str, Any]:
        return {
            "games": [
                {
                    "name": task.name,
                    "loadout": task.loadout,
                    "difficulties": [rung.name for rung in task.difficulties],
                    "goal": task.goal.label if task.goal else None,
                    "reward": None if task.reward is None else task.reward.describe(),
                    "heads_exercised": list(task.heads_exercised),
                    "world": None if task.world is None else task.world.describe(),
                }
                for task in self
            ],
            "head_coverage": {
                head: list(names) for head, names in self.head_coverage().items()
            },
        }


__all__ = ["GameManager", "GameResult", "default_record", "record_groups"]
