"""Canonical 30 Hz time conversion for Hytale combat fidelity.

Hytale assets author durations in seconds while the combat environment advances
them on server ticks.  Integer tick budgets always round upward: an effect must
remain closed for every tick whose start time is still below its authored
duration. Continuous rates do not round: the per-tick budget is the authored
per-second rate divided by 30.
"""

from __future__ import annotations

import math
import operator

from hytalegym.rulesets import load_combat_ruleset


HYTALE_TICKS_PER_SECOND = operator.index(
    load_combat_ruleset()["engine"]["ticks_per_second"]
)
if HYTALE_TICKS_PER_SECOND != 30:
    raise RuntimeError(
        "the pinned Hytale combat ruleset must run at exactly 30 ticks per second"
    )


def seconds_to_ticks_ceil(seconds: float) -> int:
    """Convert a finite nonnegative duration to its first eligible tick.

    Values which are mathematically integral but land a few ulps above an
    integer are snapped before applying ``ceil``.  This prevents ``0.1 * 30``
    from ever becoming four ticks on a different Python/libm build.
    """

    value = float(seconds)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("seconds must be finite and nonnegative")
    scaled = value * HYTALE_TICKS_PER_SECOND
    nearest = round(scaled)
    if math.isclose(scaled, nearest, rel_tol=0.0, abs_tol=1.0e-9):
        return int(nearest)
    return math.ceil(scaled)


def ticks_to_seconds(ticks: int) -> float:
    """Return the exact authored-time budget represented by whole ticks."""

    if isinstance(ticks, bool):
        raise TypeError("ticks must be an integer")
    try:
        count = operator.index(ticks)
    except TypeError as exc:
        raise TypeError("ticks must be an integer") from exc
    if count < 0:
        raise ValueError("ticks must be nonnegative")
    return count / HYTALE_TICKS_PER_SECOND


def rate_per_second_to_per_tick(rate: float) -> float:
    """Convert a finite continuous rate without integer-boundary rounding.

    Rounding upward is reserved for discrete duration boundaries. For example,
    729 degrees/second is exactly 24.3 degrees/tick and 3 stamina/second is
    exactly 0.1 stamina/tick at the pinned 30 Hz clock.
    """

    value = float(rate)
    if not math.isfinite(value):
        raise ValueError("rate must be finite")
    return value / HYTALE_TICKS_PER_SECOND


__all__ = [
    "HYTALE_TICKS_PER_SECOND",
    "rate_per_second_to_per_tick",
    "seconds_to_ticks_ceil",
    "ticks_to_seconds",
]
