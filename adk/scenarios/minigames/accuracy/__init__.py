"""Ability accuracy: pay for damage per accepted ability, not damage total.

A policy optimising raw damage learns to throw more abilities. This pays the
*ratio*, so landing one good hit beats three that whiff -- which is the
difference between a weapon being used correctly and being used constantly.

Weapon-specific by nature. Reach differs per weapon and melee damage is a band
that is **zero at sustained point-blank**, so the same policy is accurate with
one weapon and useless with another at the same range. Read a result here only
against the weapon it was measured on; pair with `spacing` for the band.

Implementation note: damage-dealt is read as the drop in target health rather
than from a reward component, so it stays correct when the reward weights are
retuned by a `Task`.
"""

from __future__ import annotations

import jax.numpy as jnp

from ..framework import AGENT_ENTITY, Minigame, TARGET_ENTITY, terminated
from adk.contracts.shaping import wants_previous


def build(*, entity: int = AGENT_ENTITY, target: int = TARGET_ENTITY,
          miss_cost: float = 0.25):
    """Pay damage landed per accepted ability; charge `miss_cost` for a whiff.

    An accepted ability that produced no health drop this tick is the whiff.
    That is approximate -- multi-tick abilities land later than they are
    accepted -- so treat the absolute number as a comparison between policies
    on the same weapon, not as a true hit rate.
    """

    @wants_previous
    def shape(previous_state, state, info):
        health = state.runtime.combat.health[:, target]
        last = previous_state.runtime.combat.health[:, target]
        acted = info.arsenal_info.ability_accepted[:, entity].astype(jnp.float32)
        landed = jnp.maximum(last - health, 0.0)
        whiffed = (acted > 0) & (landed <= 0.0)
        paid = landed - jnp.where(whiffed, jnp.float32(miss_cost),
                                  jnp.float32(0.0))
        return jnp.where(terminated(previous_state), jnp.float32(0.0), paid)

    return shape


MINIGAME = Minigame(
    name="accuracy",
    teaches="land the ability rather than throw it; weapon-conditioned",
    tell="damage per accepted ability, reported per weapon",
    build=build,
    requires=("armed_opponent",),
)
