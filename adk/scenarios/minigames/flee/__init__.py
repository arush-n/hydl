"""Escape: increase separation from the opponent.

Complements the `survivor` task. That one only says "do not get hit", which a
corner-camping policy satisfies without learning to disengage; this pays the
disengagement itself.

**Reads as a success against an inert opponent.** A target that never chases
lets separation grow for free. Confirmed on flat scenes: the shipped
`first_legal_opponent_ability_slots` produced 0.0 agent damage taken, identical
to the explicitly inert provider. Gate any flee result on the opponent having
actually moved or attacked.
"""

from __future__ import annotations

from ..framework import Minigame, progress, separation


def build(*, horizontal: bool = True, safe_distance: float | None = None,
          safe_bonus: float = 0.0):
    return progress(
        separation(horizontal=horizontal),
        sign=+1.0,
        bonus=safe_bonus,
        # `progress` pays the bonus BELOW `within`; fleeing wants it above, so
        # a safe-distance bonus is expressed as a separate band if needed
        # rather than by inverting the comparison here.
        within=None if safe_distance is None else safe_distance)


MINIGAME = Minigame(
    name="flee",
    teaches="break contact and open distance",
    tell="mean separation over the episode, vs an idle control",
    build=build,
    requires=("armed_opponent",),
)
