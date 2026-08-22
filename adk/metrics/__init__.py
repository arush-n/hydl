"""Derived statistics — more signal than moves and hits.

Damage says what landed.  It does not say whether the agent played well, and a
policy optimised on damage alone learns to trade hits rather than win
exchanges.  These are the quantities competitive games actually score on:
combos, punishes, whiffs, spacing, resource efficiency, tempo.

Everything is a pure function over a recorded trajectory and is jittable, so
the same code serves as a reward term and as an evaluation statistic — there is
no second implementation to drift.

    from adk.metrics import combo_count, whiff_rate, rising_edges, ticks

    starts = rising_edges(active_slot >= 0)
    combos = combo_count(landed_hit, gap=ticks(0.4))
    waste  = whiff_rate(starts, landed_hit, window=ticks(0.5))

See [`DEV.md`](DEV.md) for what each number is for and how to read pairs
of them.
"""

from adk.metrics.combat import (
    TICKS_PER_SECOND,
    aggression,
    combo_count,
    combo_lengths,
    damage_per_stamina,
    engagement_fraction,
    longest_combo,
    punish_rate,
    rising_edges,
    ticks,
    time_to_first_hit,
    whiff_rate,
)

__all__ = [
    "TICKS_PER_SECOND",
    "aggression",
    "combo_count",
    "combo_lengths",
    "damage_per_stamina",
    "engagement_fraction",
    "longest_combo",
    "punish_rate",
    "rising_edges",
    "ticks",
    "time_to_first_hit",
    "whiff_rate",
]
