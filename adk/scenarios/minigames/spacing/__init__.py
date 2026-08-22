"""Spacing: hold the effective damage range.

Melee damage is a band, not a threshold -- it peaks near 0.83 units and is
**zero** at sustained point-blank. So "get close" is actively wrong, and a
policy trained only on damage can sit inside its own dead zone.

The default band brackets the measured sword peak. Reach differs per weapon --
daggers are shorter than the sword -- so a weapon-specific band is the honest
way to use this. Passing the sword band while holding daggers trains the agent
to stand where daggers do nothing.
"""

from __future__ import annotations

from ..framework import Minigame, band, separation

#: Measured sword peak ~0.83; the band brackets it rather than targeting a
#: point, because a point target is unreachable under discrete movement.
SWORD_BAND = (0.6, 1.1)


def build(*, low: float = SWORD_BAND[0], high: float = SWORD_BAND[1],
          reward: float = 1.0, horizontal: bool = True):
    return band(separation(horizontal=horizontal), low, high, reward=reward)


MINIGAME = Minigame(
    name="spacing",
    teaches="hold the range where the equipped weapon actually deals damage",
    tell="damage per accepted ability, vs the same policy without shaping",
    build=build,
    requires=("armed_opponent",),
)
