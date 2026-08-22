"""Controlled measurement of what an agent can actually do.

Every capability claim in this project that turned out wrong was wrong for the
same reason: **the action heads are not independent**, so a probe that samples
freely measures conflict resolution rather than the head it meant to test.

`adk/docs/CAPABILITY-AUDIT.md` records the three mechanisms:

1. ``base_action`` APPROACH / RETREAT / STRAFE_* move the body themselves;
2. ``dodge`` displaces;
3. an **active interaction** forces ``world_move`` neutral, masks each
   approach/retreat/strafe skill against its native direction bit, and disables
   jump (``observation/v3/policy/surface.py:202-212``).

Plus tick cooldowns, which make a per-tick mean silently average in no-ops.

This package exists so a correct measurement is the easy one to write:

    from adk.capability import isolate, NEUTRAL_OPTION

    factors = isolate("locomotion_gait_compass", 3, batch=2)
    # -> every other head pinned neutral, so nothing suppresses the one
    #    under test

Pair it with :func:`adk.probes.repeat`, which emits an explicit factor vector
every step. **Do not use ``prefer()`` for this** — it is a bias, not a force,
and produced a misleading number three separate times before this package
existed.
"""

from adk.capability.forced import (
    NEUTRAL_OPTION,
    isolate,
    neutral_factors,
    override,
)

__all__ = [
    "NEUTRAL_OPTION",
    "isolate",
    "neutral_factors",
    "override",
]
