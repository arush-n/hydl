"""A versioned opponent population, and the discipline that makes it mean
something.

Self-play fails in a specific, well-known way: the population converges on a
shared convention, every member beats every other member, and the whole
population is weak against anything outside it.  **You cannot detect that after
the fact from the population's own numbers** -- internal win rates look healthy
precisely because everyone learned the same blind spot.

So three things are structural here rather than optional:

* every opponent is **immutably versioned** -- an opponent that silently
  changes makes every historical comparison meaningless;
* evaluation runs against a **frozen** snapshot -- a moving opponent during a
  measured run produces an uninterpretable number;
* a **fixed baseline** is retained permanently, because it is the only thing
  that can reveal drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import jax
import jax.numpy as jnp


class Selection(str, Enum):
    """How an opponent is drawn from the population.

    This changes the learning dynamics as much as any hyperparameter, so it is
    recorded rather than implied.
    """

    #: Always the newest member.  Fastest, and the most prone to cycling --
    #: A beats B beats C beats A, forever, with no progress.
    LATEST = "latest"
    #: Uniform over the whole history.  Slower, and resists cycling because
    #: old strategies stay in the pool.
    UNIFORM = "uniform"
    #: Uniform over the newest ``window`` members.  A compromise.
    RECENT = "recent"


@dataclass(frozen=True, slots=True)
class OpponentSpec:
    """One immutable, versioned opponent."""

    name: str
    generation: int
    parameters: Any
    #: True for opponents that must never be removed -- the fixed anchors that
    #: make a population comparable across months.
    baseline: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("an opponent must be named; unnamed history is not history")
        if self.generation < 0:
            raise ValueError(f"{self.name}: generation must be >= 0")


@dataclass(frozen=True, slots=True)
class Population:
    """An append-only set of versioned opponents.

    Immutable by construction: :meth:`add` returns a new population rather than
    mutating this one, so a snapshot handed to an evaluation cannot change
    underneath it.
    """

    members: tuple[OpponentSpec, ...] = field(default_factory=tuple)

    def add(self, spec: OpponentSpec) -> "Population":
        """Append an opponent, rejecting a name already present.

        Reusing a name would silently redefine a historical result.
        """

        if any(existing.name == spec.name for existing in self.members):
            raise ValueError(
                f"opponent {spec.name!r} already exists; versions are immutable, "
                "so publish a new name rather than replacing one"
            )
        return Population(self.members + (spec,))

    @property
    def baselines(self) -> tuple[OpponentSpec, ...]:
        """The permanently retained non-self-play anchors."""

        return tuple(member for member in self.members if member.baseline)

    @property
    def learned(self) -> tuple[OpponentSpec, ...]:
        """Members produced by training, excluding fixed baselines."""

        return tuple(member for member in self.members if not member.baseline)

    def select(self, rule: Selection, key: jax.Array, *, window: int = 5) -> OpponentSpec:
        """Draw an opponent under ``rule``.

        Baselines are excluded: they are the yardstick, not the curriculum.
        Training against your own anchor makes the anchor meaningless.
        """

        pool = self.learned
        if not pool:
            raise ValueError(
                "the population has no learned members to select from; seed it "
                "with at least one before self-play"
            )
        if rule is Selection.LATEST:
            return max(pool, key=lambda member: member.generation)
        if rule is Selection.RECENT:
            if window < 1:
                raise ValueError("window must be >= 1")
            ordered = sorted(pool, key=lambda member: member.generation)[-window:]
            return ordered[int(jax.random.randint(key, (), 0, len(ordered)))]
        return pool[int(jax.random.randint(key, (), 0, len(pool)))]


def drift(internal_win_rate: jax.Array, baseline_win_rate: jax.Array) -> jax.Array:
    """How much stronger the population looks against itself than against the
    fixed baseline.

    **The number self-play exists to watch.** A population that has converged
    on a shared convention beats itself at roughly even odds while losing to
    anything outside it, so a large positive drift means the internal ladder
    has stopped measuring skill.

    Near zero is healthy.  Large and rising means the population is training
    against its own blind spot.
    """

    return jnp.asarray(internal_win_rate) - jnp.asarray(baseline_win_rate)


def is_drifting(
    internal_win_rate: jax.Array,
    baseline_win_rate: jax.Array,
    *,
    tolerance: float = 0.2,
) -> jax.Array:
    """True when the population's internal ladder has stopped tracking skill."""

    return drift(internal_win_rate, baseline_win_rate) > tolerance
