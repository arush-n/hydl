"""Conformance checking: a game is only accepted if it actually works.

Registration runs this. The point is that every failure below is one that
would otherwise surface *hours into a training run* as a task that scores
nothing, scores everything, or scores one number for a whole batch.

The load-bearing check is :func:`check_success`, which **executes** the success
criterion against a synthetic record shaped exactly like a real rollout. That
catches, at authoring time:

* a column name that no longer exists (the schema moved twice on 2026-08-03);
* a reduction over the wrong axis -- the single most common bug, because
  ``values.min()`` looks right and silently collapses the batch to a scalar;
* a criterion returning floats instead of booleans, which makes every episode
  "succeed" at rate 0.37;
* a group the task never records.

Synthetic values are deliberately varied, not zeros: a criterion that only
works on an all-zero record passes a zero probe and fails on real data.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from arena.jax_contract import (
    GROUP_FEATURES,
    HEAD_SPANS,
    READABLE_GROUPS,
    list_loadouts,
)

from arena.tasks.framework.base import Difficulty, Task

#: Shape of the synthetic record used to execute a criterion.
PROBE_TICKS = 7
PROBE_BATCH = 3


class TaskConformanceError(ValueError):
    """A game does not satisfy the framework's contract."""


def synthetic_record(
    *, ticks: int = PROBE_TICKS, batch: int = PROBE_BATCH, seed: int = 0
) -> dict[str, np.ndarray]:
    """A record shaped like a real rollout, with varied values.

    Covers ``READABLE_GROUPS``, not all of ``GROUP_FEATURES``. Two named groups
    -- ``inventory_token_f32`` and ``light_token_f32`` -- are absent from the
    observation tree on scenes that did not enable their providers, and several
    more had no reader at all before 2026-08-03. Handing the gate a group the
    recorder cannot produce lets a criterion pass validation and then fail at
    scoring, which is the exact failure the gate exists to prevent.
    """

    rng = np.random.default_rng(seed)
    return {
        group: rng.uniform(0.0, 1.0, size=(ticks, batch, len(GROUP_FEATURES[group])))
        for group in sorted(READABLE_GROUPS)
    }


def check_identity(task: Task, problems: list[str]) -> None:
    if not isinstance(task, Task):
        problems.append(f"not a Task instance: {type(task).__name__}")
        return
    if not task.name or task.name != task.name.strip():
        problems.append("name must be a non-empty, unpadded label")
    if "/" in task.name or "." in task.name:
        problems.append(
            f"name {task.name!r} must be a bare segment: scene names are built "
            "as 'arena/<task>.<difficulty>', so '/' and '.' would corrupt them"
        )
    if not task.description.strip():
        problems.append(
            "description is empty; a curriculum rung must say what it teaches"
        )


def check_heads(task: Task, problems: list[str]) -> None:
    unknown = sorted(set(task.heads_exercised) - set(HEAD_SPANS))
    if unknown:
        problems.append(
            f"heads_exercised names non-existent head(s): {unknown}; "
            f"the action surface has: {', '.join(sorted(HEAD_SPANS))}"
        )


def check_loadout(task: Task, problems: list[str]) -> None:
    profiles = set(list_loadouts())
    if task.loadout not in profiles:
        problems.append(
            f"loadout {task.loadout!r} is not a combat profile; "
            f"choose from {len(profiles)} names e.g. " + ", ".join(sorted(profiles)[:5])
        )
    for rung in task.difficulties:
        if rung.opponent_profile is not None and rung.opponent_profile not in profiles:
            problems.append(
                f"difficulty {rung.name!r} names opponent_profile "
                f"{rung.opponent_profile!r}, which is not a combat profile"
            )


def check_difficulties(task: Task, problems: list[str]) -> None:
    if not task.difficulties:
        problems.append("declares no difficulties")
        return
    for rung in task.difficulties:
        if not isinstance(rung, Difficulty):
            problems.append(f"difficulty {rung!r} is not a Difficulty")
    armed = [r.opponent_armed for r in task.difficulties]
    if len(task.difficulties) > 1 and all(armed):
        problems.append(
            "every difficulty arms the opponent; ladders should open with an "
            "unarmed rung, because against a lethal opponent avoidance is a "
            "cheap local optimum that a policy will not leave"
        )


def check_success(task: Task, problems: list[str]) -> None:
    """Execute the criterion. This is the check that earns its keep."""

    if not callable(task.success):
        problems.append("success is not callable")
        return

    # Goals with a time window refuse a rollout shorter than the window rather
    # than passing vacuously, so the probe has to be at least that long. A goal
    # that wants 500 ticks gets a 500-tick probe.
    goal = task.goal
    ticks = max(PROBE_TICKS, getattr(goal, "minimum_ticks", 1) if goal else 1)
    record = synthetic_record(ticks=ticks)
    try:
        result = task.success(record)
    except KeyError as error:
        problems.append(
            f"success criterion referenced something absent: {error}. "
            "Read observation columns by name via arena.tasks.criteria.col."
        )
        return
    except Exception as error:  # noqa: BLE001 - report any authoring error
        problems.append(f"success criterion raised {type(error).__name__}: {error}")
        return

    array = np.asarray(result)
    if array.dtype != np.bool_:
        problems.append(
            f"success must return booleans, got dtype {array.dtype}. Wrap the "
            "value in a comparison, e.g. above(at_end(col(...)), 0.5)."
        )
    if array.shape != (PROBE_BATCH,):
        problems.append(
            f"success must return one value per environment -- expected shape "
            f"({PROBE_BATCH},), got {array.shape}. This almost always means a "
            "reduction collapsed the batch: use the reducers in "
            "arena.tasks.criteria, which collapse ticks only."
        )


def check_reward(task: Task, problems: list[str]) -> None:
    """Execute the optional transition reward against named one-step groups."""

    if task.reward is None:
        return
    record = synthetic_record(ticks=2)
    groups = task.reward.required_groups
    previous = {group: record[group][0] for group in groups}
    current = {group: record[group][1] for group in groups}
    try:
        result = task.reward.apply(
            np.zeros((PROBE_BATCH,), dtype=np.float32), previous, current
        )
    except Exception as error:  # noqa: BLE001
        problems.append(f"reward raised {type(error).__name__}: {error}")
        return
    array = np.asarray(result)
    if array.shape != (PROBE_BATCH,) or array.dtype != np.float32:
        problems.append(
            "reward must return float32 per environment -- expected shape "
            f"({PROBE_BATCH},), got {array.dtype}{array.shape}"
        )
    elif not np.isfinite(array).all():
        problems.append("reward returned a non-finite value")


def check_scene(task: Task, problems: list[str], *, batch: int = 2) -> None:
    """Build every difficulty. Catches bad providers and parameter names."""

    for rung in task.difficulties:
        try:
            task.build_scene(rung.name, batch=batch)
        except Exception as error:  # noqa: BLE001
            problems.append(
                f"difficulty {rung.name!r} failed to build a scene: "
                f"{type(error).__name__}: {error}"
            )


def validate(task: Task, *, build_scenes: bool = False) -> Task:
    """Raise :class:`TaskConformanceError` unless the game conforms.

    ``build_scenes`` additionally constructs every difficulty. It is off by
    default because it imports and runs Gym factories, which is slow enough to
    matter when registering many games; turn it on in a test.
    """

    problems: list[str] = []
    check_identity(task, problems)
    if problems:
        raise TaskConformanceError(_format(getattr(task, "name", "<?>"), problems))

    check_heads(task, problems)
    check_loadout(task, problems)
    check_difficulties(task, problems)
    check_success(task, problems)
    check_reward(task, problems)
    if build_scenes:
        check_scene(task, problems)

    if problems:
        raise TaskConformanceError(_format(task.name, problems))
    return task


def _format(name: str, problems: Sequence[str]) -> str:
    lines = "\n".join(f"  - {problem}" for problem in problems)
    return f"task {name!r} does not conform to the arena framework:\n{lines}"


def report(task: Task, *, build_scenes: bool = False) -> dict[str, Any]:
    """Non-raising form, for tooling that wants to list every problem."""

    try:
        validate(task, build_scenes=build_scenes)
    except TaskConformanceError as error:
        return {"name": getattr(task, "name", "<?>"), "ok": False, "detail": str(error)}
    return {"name": task.name, "ok": True, "detail": None}


__all__ = [
    "PROBE_BATCH",
    "PROBE_TICKS",
    "TaskConformanceError",
    "report",
    "synthetic_record",
    "validate",
]
