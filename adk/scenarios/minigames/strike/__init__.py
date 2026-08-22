"""Successful attacks: pay per hit that actually landed.

The distinction this exists to draw: **acceptance is not damage.** Swept over
300 ticks, two opponent policies got abilities accepted 36 and 16 times and
dealt **0.0**. Anything paying `accepted` counts intent; this counts effect.

Against `accuracy`, which pays damage-per-accepted as a *ratio*: a ratio is the
right objective once a policy already lands hits, and the wrong one before
that, because a policy that never attacks has an undefined-to-flattering ratio
and no gradient toward attacking at all. This pays the event, so the first hit
a random policy stumbles into is already worth something. Run this first and
`accuracy` after, in that order.

Read straight off `info.arsenal_info.damage_dealt` rather than inferred from a
health drop. The health-drop heuristic cannot separate a hit that was blocked
from one that missed, and cannot attribute damage when more than one thing can
deal it on the same tick.

Melee damage is a **band** -- it peaks near 0.83 units and is zero at sustained
point-blank -- so a policy can attack constantly at the wrong range and score
nothing here. That is the intended reading, not a defect: pair with `spacing`
if the agent is attacking and this stays flat.
"""

from __future__ import annotations

from ..framework import Minigame, damage, event, landed


def build(*, per_damage: float = 1.0, per_hit: float = 0.0,
          hit_threshold: float = 0.0):
    """Pay `per_damage` per point of damage, and `per_hit` per landing tick.

    Two terms because they answer different questions. `per_damage` scales with
    how good the hit was and is the default. `per_hit` is flat per landing tick
    and is what you want when the objective is "connect at all" -- an early
    policy that lands one weak hit gets the same credit as one that lands a
    strong one, which is the right signal before the agent can aim.

    Leaving both at their defaults pays damage only.
    """

    dealt = damage(dealt=True)
    hits = landed(threshold_value=hit_threshold)

    if not per_hit:
        return event(dealt, reward=per_damage)

    pay_damage = event(dealt, reward=per_damage)
    pay_hits = event(hits, reward=per_hit)

    def shape(state, info):
        return pay_damage(state, info) + pay_hits(state, info)

    return shape


MINIGAME = Minigame(
    name="strike",
    teaches="land attacks, as opposed to throwing them",
    tell=("damage per episode and landing ticks per episode; both must move, "
          "and accepted-per-episode must NOT be what moved"),
    build=build,
    requires=("armed_opponent",),
)
