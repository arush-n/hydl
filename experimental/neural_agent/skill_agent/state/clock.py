"""The world's tick length in the state layer (``docs/VIRTUAL_HUMAN.md`` section 27).

Every state-layer rule is stated in seconds, and turned into ticks or per-tick factors here, once per world: gen 7 and
the duel step 1/30 s, gen 5 steps 0.045 s, and live passes its measured mean. The conversions run in Python at trace
time, so a world's tick length is a static float, never a traced argument.
"""

from __future__ import annotations

import math

from experimental.neural_agent.skill_agent.contract import surface

DEFAULT_TICK_SECONDS: float = surface.tick_seconds("gen7")
"""Gen 7's tick, 1/30 s: the default line (``contract/surface.py``, the owner's 30 Hz ruling of 2026-10-17)."""
GEN5_TICK_SECONDS: float = surface.tick_seconds("gen5")
"""The gen-5 simulator's tick, 0.045 s (``contract/surface.py``, measured)."""


def at_least(seconds: float, tick_seconds: float) -> int:
    """A lower bound in ticks: ``seconds`` rounded up to whole ticks, and at least one."""

    return max(1, math.ceil(seconds / tick_seconds - 1e-9))


def at_most(seconds: float, tick_seconds: float) -> int:
    """An upper bound in ticks: ``seconds`` rounded down to whole ticks."""

    return math.floor(seconds / tick_seconds + 1e-9)


def about(seconds: float, tick_seconds: float) -> int:
    """A duration or horizon in ticks: ``seconds`` rounded to the nearest whole tick."""

    return round(seconds / tick_seconds)


def fade(tau_seconds: float, tick_seconds: float) -> float:
    """The per-tick factor of an exponential fade with time constant ``tau_seconds``: ``exp(-tick_seconds / tau)``."""

    return math.exp(-tick_seconds / tau_seconds)


__all__ = ["DEFAULT_TICK_SECONDS", "GEN5_TICK_SECONDS", "about", "at_least", "at_most", "fade"]
