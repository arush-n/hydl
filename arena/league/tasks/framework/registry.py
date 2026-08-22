"""Where custom games live once you have written one.

The arena ships no canonical task list. A game is whatever you register, and
nothing in the curriculum or league machinery knows the difference between a
game that shipped here and one you wrote this afternoon.

    from arena.tasks import Difficulty, Task, register
    from arena.tasks.framework.criteria import above, at_end, col

    register(Task(
        name="ledge",
        description="Cross a gap without falling.",
        heads_exercised=("jump_off_on", "locomotion_gait_compass"),
        loadout="iron_sword",
        difficulties=(Difficulty("narrow", opponent_armed=False),),
        success=above(at_end(col("self_f32", "alive")), 0.5),
    ))

Registration is by name and refuses silent replacement: re-registering the
same name with a different task raises rather than shadowing, because a
curriculum that quietly swapped a rung underneath a run is unreproducible.
"""

from __future__ import annotations

from typing import Iterator

from arena.tasks.framework.base import Task
from arena.tasks.framework.validate import validate

_REGISTRY: dict[str, Task] = {}


def register(
    task: Task,
    *,
    replace: bool = False,
    build_scenes: bool = False,
    check: bool = True,
) -> Task:
    """Add a game, after checking it conforms. Returns it, so it can be used
    as a declaration expression.

    ``check`` runs :func:`arena.tasks.validate.validate`, which executes the
    success criterion against a synthetic rollout. It is on by default: a game
    that scores the wrong axis or names a moved column is otherwise only
    discovered hours into a run. ``build_scenes`` additionally constructs every
    difficulty, which is slower but catches bad providers and parameters.
    """

    if not isinstance(task, Task):
        raise TypeError("register expects a Task")
    if check:
        validate(task, build_scenes=build_scenes)
    existing = _REGISTRY.get(task.name)
    if existing is not None and not replace:
        if existing is task:
            return task
        raise ValueError(
            f"task {task.name!r} is already registered. Pass replace=True if "
            "you mean to override it; a silently swapped rung makes a "
            "curriculum run impossible to reproduce."
        )
    _REGISTRY[task.name] = task
    return task


def unregister(name: str) -> None:
    _REGISTRY.pop(name, None)


def get(name: str) -> Task:
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "<none registered>"
        raise KeyError(f"no arena task {name!r}; registered: {known}") from None


def names() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def all_tasks() -> tuple[Task, ...]:
    return tuple(_REGISTRY[name] for name in sorted(_REGISTRY))


def __iter__() -> Iterator[Task]:  # pragma: no cover - module-level convenience
    return iter(all_tasks())


def clear() -> None:
    """Drop every registration. Intended for tests."""

    _REGISTRY.clear()


__all__ = ["all_tasks", "clear", "get", "names", "register", "unregister"]
