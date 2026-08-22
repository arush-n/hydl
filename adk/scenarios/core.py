"""A scenario is a scene plus what it teaches and how you know it worked.

The environment must be able to support a **perfect** agent even when no agent
can yet reach it.  So every scenario declares a ``ceiling``: the score an
optimal policy would achieve.  Without it a scenario cannot distinguish "the
agent is bad" from "the scenario is unwinnable", and this project has spent
real time on exactly that ambiguity.

``score`` and ``ceiling`` are deliberately separate from the environment's own
reward.  A scenario is an *evaluation*, and mixing the two lets a shaped reward
quietly redefine success.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import jax.numpy as jnp

#: A recorded rollout: named arrays, time-first.
Trajectory = Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class Scenario:
    """One skill, one scene, one number that says whether it was learned."""

    name: str
    #: One line: the single skill this scenario isolates.
    teaches: str
    #: Builds the compiled handle.  Takes ``batch``.
    build: Callable[[int], Any]
    #: Trajectory -> score, higher is better.  Batched: returns ``(B,)``.
    score: Callable[[Trajectory], jnp.ndarray]
    #: What an optimal policy scores.  The scenario is meaningless without it.
    ceiling: float
    #: What an idle or random policy scores.  Anything at or below this has
    #: learned nothing, however good the raw number looks.
    floor: float
    #: Why the ceiling is what it is -- the arithmetic, not a guess.
    ceiling_rationale: str = ""

    def __post_init__(self) -> None:
        if not self.ceiling > self.floor:
            raise ValueError(
                f"scenario {self.name!r} has ceiling {self.ceiling} <= floor "
                f"{self.floor}; there is no room to demonstrate skill"
            )

    def normalised(self, trajectory: Trajectory) -> jnp.ndarray:
        """Score mapped so 0.0 is the floor and 1.0 is a perfect policy.

        Comparable across scenarios, which raw scores are not.  Values above
        1.0 mean the ceiling was set too low and should be re-derived, not
        clipped away -- so this does not clip.
        """

        raw = jnp.asarray(self.score(trajectory), dtype=jnp.float32)
        return (raw - self.floor) / (self.ceiling - self.floor)


def registry(*scenarios: Scenario) -> dict[str, Scenario]:
    """Index scenarios by name, rejecting duplicates.

    A silently shadowed scenario would report another one's numbers under the
    wrong heading.
    """

    index: dict[str, Scenario] = {}
    for scenario in scenarios:
        if scenario.name in index:
            raise ValueError(f"duplicate scenario name {scenario.name!r}")
        index[scenario.name] = scenario
    return index
