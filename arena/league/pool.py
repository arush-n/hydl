"""Legacy single-actor opponent pools, ratings, and matchmaking.

This pool configures :class:`arena.manager.GameManager`, whose compatibility
scene owns only entity zero's learner row. It therefore keeps learned entries
fail-closed rather than silently substituting a scripted controller.

Actual learned-policy self-play now uses :mod:`arena.training.selfplay` and
HytaleGym's actor-major multi-actor rollout/trainers. This module remains the
small host-side pool for scripted and parameterised legacy scenes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

import math
import random

from arena.tasks.framework.base import Difficulty


#: Driven by a fixed rule over privileged state. Expressible today.
SCRIPTED = "scripted"
#: A scripted opponent with altered parameters (health, weapon, damage scale).
PARAMETERISED = "parameterised"
#: Frozen learned policy; unsupported only by this legacy single-actor pool.
LEARNED = "learned"

KINDS = (SCRIPTED, PARAMETERISED, LEARNED)

DEFAULT_RATING = 1200.0


@dataclass(frozen=True, slots=True)
class Opponent:
    """One entry in a pool: a name, how it is driven, and its configuration."""

    name: str
    kind: str
    difficulty: Difficulty
    #: For LEARNED entries: where the frozen parameters live. The multi-actor
    #: training path uses policy-bank rows instead of this legacy field.
    checkpoint: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise ValueError("opponent name must be a non-empty label")
        if self.kind not in KINDS:
            raise ValueError(
                f"unknown opponent kind {self.kind!r}; have: {', '.join(KINDS)}"
            )
        if self.kind == LEARNED and self.checkpoint is None:
            raise ValueError(
                f"opponent {self.name!r} is LEARNED but names no checkpoint"
            )

    @property
    def playable(self) -> bool:
        """Whether the legacy single-actor GameManager can drive this entry."""

        return self.kind != LEARNED

    def require_playable(self) -> None:
        if self.playable:
            return
        raise NotImplementedError(
            f"opponent {self.name!r} is LEARNED, but the legacy GameManager "
            "scene owns one learner row. Use arena.training.selfplay with the "
            "HytaleGym multi-actor collector for frozen learned opponents."
        )


@dataclass(slots=True)
class Record:
    """Elo plus the raw tallies, because Elo alone hides sample size."""

    rating: float = DEFAULT_RATING
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def played(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float | None:
        return None if self.played == 0 else self.wins / self.played


def expected_score(rating: float, opponent_rating: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((opponent_rating - rating) / 400.0))


class Pool:
    """A set of opponents with ratings, plus rating-aware sampling."""

    def __init__(
        self, opponents: Sequence[Opponent] = (), *, k_factor: float = 24.0
    ) -> None:
        if k_factor <= 0.0:
            raise ValueError("k_factor must be positive")
        self._opponents: dict[str, Opponent] = {}
        self._records: dict[str, Record] = {}
        self.k_factor = float(k_factor)
        for opponent in opponents:
            self.add(opponent)

    def __len__(self) -> int:
        return len(self._opponents)

    def __iter__(self) -> Iterator[Opponent]:
        return iter(self._opponents[name] for name in sorted(self._opponents))

    def __contains__(self, name: object) -> bool:
        return name in self._opponents

    def add(self, opponent: Opponent, *, rating: float = DEFAULT_RATING) -> Opponent:
        if opponent.name in self._opponents:
            raise ValueError(f"opponent {opponent.name!r} is already in the pool")
        self._opponents[opponent.name] = opponent
        self._records[opponent.name] = Record(rating=float(rating))
        return opponent

    def get(self, name: str) -> Opponent:
        try:
            return self._opponents[name]
        except KeyError:
            known = ", ".join(sorted(self._opponents)) or "<empty pool>"
            raise KeyError(f"no opponent {name!r}; pool: {known}") from None

    def record(self, name: str) -> Record:
        if name not in self._records:
            raise KeyError(f"no opponent {name!r} in this pool")
        return self._records[name]

    def playable(self) -> tuple[Opponent, ...]:
        return tuple(o for o in self if o.playable)

    def report_result(self, name: str, *, score: float, learner_rating: float) -> float:
        """Apply one match outcome from the LEARNER's perspective.

        ``score`` is 1.0 learner win, 0.0 loss, 0.5 draw. Returns the learner's
        updated rating; the opponent's is updated in place.
        """

        if score not in (0.0, 0.5, 1.0):
            raise ValueError("score must be 1.0 (win), 0.5 (draw) or 0.0 (loss)")
        entry = self.record(name)
        expected_learner = expected_score(learner_rating, entry.rating)

        if score == 1.0:
            entry.losses += 1
        elif score == 0.0:
            entry.wins += 1
        else:
            entry.draws += 1

        entry.rating += self.k_factor * ((1.0 - score) - (1.0 - expected_learner))
        return learner_rating + self.k_factor * (score - expected_learner)

    def sample(
        self,
        *,
        learner_rating: float,
        rng: random.Random | None = None,
        temperature: float = 200.0,
    ) -> Opponent:
        """Pick an opponent, favouring those near the learner's rating.

        Even matches carry the most information: a pool sampled uniformly
        spends most of its budget on opponents that are already trivially beaten
        or hopeless. ``temperature`` is in Elo points -- larger is flatter.
        """

        candidates = self.playable()
        if not candidates:
            raise ValueError(
                "pool has no playable opponents in legacy GameManager; use "
                "arena.training.selfplay for LEARNED policy-bank rows"
            )
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        chooser = rng or random
        weights = [
            math.exp(-abs(self.record(o.name).rating - learner_rating) / temperature)
            for o in candidates
        ]
        return chooser.choices(candidates, weights=weights, k=1)[0]

    def standings(self) -> list[dict[str, Any]]:
        rows = [
            {
                "name": name,
                "kind": self._opponents[name].kind,
                "rating": round(entry.rating, 1),
                "played": entry.played,
                "win_rate": entry.win_rate,
                "playable": self._opponents[name].playable,
            }
            for name, entry in self._records.items()
        ]
        rows.sort(key=lambda row: row["rating"], reverse=True)
        return rows


__all__ = [
    "DEFAULT_RATING",
    "KINDS",
    "LEARNED",
    "PARAMETERISED",
    "SCRIPTED",
    "Opponent",
    "Pool",
    "Record",
    "expected_score",
]
