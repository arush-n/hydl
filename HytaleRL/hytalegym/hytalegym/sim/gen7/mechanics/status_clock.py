"""Status clock (gen 7, task #106): a timed status lasts round(D / tick) ticks, as the live 0.7.2 game counts them.

Switch ``Constructs.status_clock``: False (gen 5, and the committed render: the stock clock) or True (gen 7). Hook:
``tick_statuses`` in ``combat/mechanics/runtime/statuses.py`` (sim/gen7/overlay.py row ``status_clock``).

Stock clock (``statuses.py``, ``tick_statuses``): each step subtracts the step dt from ``remaining_seconds`` in float32
(``remaining = max(0, remaining - dt)``) and the status is active while ``remaining > 0``. The subtraction is sequential, so the
rounding accumulates: at 1/30 s, ``0.5 - 15 * float32(1/30)`` in one product is exactly 0.0, but fifteen sequential float32
subtractions leave 2.2e-8 > 0, and a 0.5 s status is removed on the 16th tick (receipt: ``tests/sim/test_status_clock.py``).

This construct keeps a status active after a tick only while more than half a tick of it remains: the status is removed on the
first tick k with ``D - k * dt <= dt / 2``, so it lasts ``round(D / dt)`` ticks (a remainder within 0.001 tick below one half
counts as one half). The stored remaining seconds are the stock decrement; only ``active`` changes, and everything that reads
statuses reads ``active`` (``status_modifiers``, the overlap rules, the cycles of periodic statuses fire from the pre-tick state).

Measured (whole ticks only): Flinch 0.5 s removed after 15 frames and Stunned 0.7 s after 21 frames on the live 0.7.2 server
(``sim/gen7/mechanics_072.md``, hit interrupt). Assumed: the rounding of durations that are not whole ticks (21 of the 88
gen-7 effect durations, e.g. Dodge 0.25 s = 7.5 ticks, Potion regen 5.05 s = 151.5 ticks); the native rule for those is not
measured, and the half-up tie is a choice. Durations below half a tick last one tick, as the stock clock gives.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import jax.numpy as jnp

SWITCH = "status_clock"
IMPLEMENTED = True
#: A remainder this many ticks below half a tick still counts as half a tick: ties round up, and the float32 residue of an
#: authored duration (5.05 s = 151.5 ticks) does not decide them.
TIE_SNAP_TICKS = 1e-3


def _enabled(profile: Any) -> bool:
    value = profile.constructs.status_clock
    if value is False:
        return False
    if value is not True:
        raise ValueError(f"status_clock must be a bool, got {value!r}")
    return True


def construct(original: Any, profile: Any) -> Any:
    """Hook for ``tick_statuses``: after the stock decrement, a status stays active while more than half a tick remains."""

    if not _enabled(profile):
        return original

    @wraps(original)
    def tick_statuses(state: Any, dt_seconds: Any, rules: Any) -> Any:
        next_state, tick = original(state, dt_seconds, rules)
        statuses = next_state.statuses
        batch = statuses.active.shape[0]
        dt = jnp.broadcast_to(jnp.asarray(dt_seconds, jnp.float32), (batch,))[:, None, None]
        keep = statuses.remaining_seconds > (0.5 - TIE_SNAP_TICKS) * dt
        return next_state._replace(statuses=statuses._replace(active=statuses.active & keep)), tick

    return tick_statuses
