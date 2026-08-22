"""Staged progression: teach one axis at a time, and prove it before moving on.

A :class:`Stage` is one task at one difficulty plus the bar for leaving it.
A :class:`Ladder` is an ordered list of stages and the promotion rule.

Why a ladder rather than "train on the hard thing":
a 30-update PPO run against a lethal opponent scored **0 successes on every
policy** while driving deaths to 0/16. Survival is a cheap local optimum and
offence is rare under exploration, so the gradient never leaves avoidance.
Ladders exist to make the cheap optimum unavailable until the expensive skill
is already in the policy.

Promotion is deliberately conservative:

* **Promote** on a sustained success rate, measured over a *window* of recent
  evaluations, never a single lucky batch.
* **Demote** when performance collapses, because a policy that regresses on an
  earlier rung has usually overfit the current one -- and silently continuing
  produces a policy that is good at nothing.
* **Never skip.** A skipped rung is an untested assumption about transfer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Protocol, Sequence, runtime_checkable

from arena.tasks.framework.base import Task


@runtime_checkable
class Rung(Protocol):
    """What :class:`Ladder` actually needs from a rung.

    The promotion machinery below never touches ``stage.task`` -- it reads a
    name and four thresholds and nothing else. Stating that as a protocol is
    what lets a curriculum whose rungs are NOT arena tasks (world selections,
    episode horizons, opponent difficulty) reuse the same promote/demote logic
    instead of hand-rolling a second, subtly different copy of it.
    """

    @property
    def name(self) -> str: ...

    promote_at: float
    demote_below: float
    window: int
    minimum_evaluations: int


def _validate_thresholds(rung: Rung) -> None:
    """Shared by every rung type, so one cannot drift from the other."""

    if not 0.0 < rung.promote_at <= 1.0:
        raise ValueError("promote_at must be in (0, 1]")
    if not 0.0 <= rung.demote_below < rung.promote_at:
        raise ValueError("demote_below must be in [0, promote_at)")
    if rung.window < 1:
        raise ValueError("window must be positive")
    if rung.minimum_evaluations < 1:
        raise ValueError("minimum_evaluations must be positive")


@dataclass(frozen=True, slots=True)
class SettingsStage:
    """One rung of a curriculum whose difficulty is a set of run settings.

    The pursuit curriculum's rungs are world selections and episode horizons,
    not `Difficulty` names on a `Task`, so :class:`Stage` cannot express them.
    ``settings`` is opaque here on purpose -- this module owns WHEN to advance,
    and the caller owns what advancing means.
    """

    stage_name: str
    settings: Mapping[str, Any] = field(default_factory=dict)
    promote_at: float = 0.7
    demote_below: float = 0.2
    window: int = 5
    minimum_evaluations: int = 3

    def __post_init__(self) -> None:
        if not self.stage_name:
            raise ValueError("a settings stage needs a name")
        _validate_thresholds(self)

    @property
    def name(self) -> str:
        return self.stage_name


@dataclass(frozen=True, slots=True)
class Stage:
    """One rung: a task at a difficulty, with the bar for leaving it."""

    task: Task
    difficulty: str
    #: Fraction of episodes that must succeed to promote.
    promote_at: float = 0.7
    #: Fraction below which we fall back a rung. Must be < promote_at.
    demote_below: float = 0.2
    #: How many recent evaluations the rates are computed over.
    window: int = 5
    #: Refuse to promote before this many evaluations, however good they look.
    minimum_evaluations: int = 3

    def __post_init__(self) -> None:
        # Validates the difficulty exists, and fails at construction rather
        # than mid-training.
        self.task.difficulty(self.difficulty)
        _validate_thresholds(self)

    @property
    def name(self) -> str:
        return f"{self.task.name}.{self.difficulty}"


@dataclass(slots=True)
class LadderState:
    """Where a run currently is, and what it has seen at this rung."""

    index: int = 0
    history: list[float] = field(default_factory=list)

    def clear(self) -> None:
        self.history.clear()


class Ladder:
    """An ordered set of stages plus promotion/demotion bookkeeping."""

    def __init__(self, stages: Sequence[Rung]) -> None:
        if not stages:
            raise ValueError("a ladder needs at least one stage")
        names = [stage.name for stage in stages]
        if len(set(names)) != len(names):
            raise ValueError("ladder has duplicate stages")
        self._stages = tuple(stages)
        self.state = LadderState()

    def __len__(self) -> int:
        return len(self._stages)

    def __iter__(self) -> Iterator[Rung]:
        return iter(self._stages)

    @property
    def stages(self) -> tuple[Rung, ...]:
        return self._stages

    @property
    def current(self) -> Rung:
        return self._stages[self.state.index]

    @property
    def finished(self) -> bool:
        """True once the top rung has been passed."""

        return self.state.index >= len(self._stages)

    def record(self, success_rate: float) -> str:
        """Log one evaluation and return the transition taken.

        Returns ``"promote"``, ``"demote"`` or ``"hold"``. The caller rebuilds
        its scene when this is not ``"hold"``.
        """

        if not 0.0 <= success_rate <= 1.0:
            raise ValueError("success_rate must be a fraction in [0, 1]")
        if self.finished:
            return "hold"

        stage = self.current
        self.state.history.append(float(success_rate))
        window = self.state.history[-stage.window :]
        rate = sum(window) / len(window)

        if len(self.state.history) >= stage.minimum_evaluations and rate >= stage.promote_at:
            self.state.index += 1
            self.state.clear()
            return "promote"

        if (
            self.state.index > 0
            and len(self.state.history) >= stage.minimum_evaluations
            and rate < stage.demote_below
        ):
            self.state.index -= 1
            self.state.clear()
            return "demote"

        return "hold"

    def describe(self) -> dict[str, object]:
        window = self.state.history[-self.current.window :] if not self.finished else []
        return {
            "finished": self.finished,
            "index": self.state.index,
            "total_stages": len(self._stages),
            "current": None if self.finished else self.current.name,
            "evaluations_at_stage": len(self.state.history),
            "recent_success_rate": (
                sum(window) / len(window) if window else None
            ),
            "stages": [stage.name for stage in self._stages],
        }


def ladder_for(task: Task, **stage_options: object) -> Ladder:
    """Every difficulty of one task, in declaration order.

    Declaration order is the authored progression -- ``Difficulty`` tuples are
    written easiest-first -- so this is the default ladder for a single game.
    """

    return Ladder(
        [
            Stage(task=task, difficulty=rung.name, **stage_options)  # type: ignore[arg-type]
            for rung in task.difficulties
        ]
    )


__all__ = ["Ladder", "LadderState", "Rung", "SettingsStage", "Stage", "ladder_for"]
