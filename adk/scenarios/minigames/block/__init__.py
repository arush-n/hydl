"""Blocking: pay for guard being up when it actually absorbs something.

Guard uptime alone is the wrong target -- a policy that holds guard forever
scores perfectly and never fights. This pays guard only on ticks where the
agent would otherwise have taken damage, so it trains *timing* rather than
turtling, and it charges a small idle cost so permanent guard is not free.

Two facts that decide whether a result here means anything:

* Guard is gated on stamina. `guard_stamina_value > 0` AND `~stamina_broken`
  (`observation/v3/encoding/encoder.py:416-421`), and entry costs stamina --
  measured 10.00 -> 9.50 on the first guard tick with iron_sword. 114 of 223
  profiles guard at all; the other 109 cannot, and this minigame is
  meaningless on them.
* Guard and ability are a confirmed joint exclusion the per-head mask calls
  legal. PPO samples heads independently, so some guard choices are silently
  arbitrated away -- a low guard rate is not automatically the policy's fault.
"""

from __future__ import annotations

import jax.numpy as jnp

from ..framework import AGENT_ENTITY, Minigame, terminated
from adk.contracts.shaping import wants_previous


def build(*, entity: int = AGENT_ENTITY, absorb_reward: float = 2.0,
          idle_cost: float = 0.05):
    """Reward guard on damage ticks; charge a small cost for guard otherwise."""

    @wants_previous
    def shape(previous_state, state, info):
        del info
        mechanics = state.runtime.mechanics
        health = state.runtime.combat.health[:, entity]
        guarding = mechanics.guard_active[:, entity]
        last = previous_state.runtime.combat.health[:, entity]
        under_fire = (last - health) > 0.0
        paid = (
            jnp.where(guarding & under_fire, jnp.float32(absorb_reward), 0.0)
            - jnp.where(guarding & ~under_fire, jnp.float32(idle_cost), 0.0)
        )
        return jnp.where(terminated(previous_state), jnp.float32(0.0), paid)

    return shape


MINIGAME = Minigame(
    name="block",
    teaches="raise guard when a hit is coming, not permanently",
    tell="damage taken per opponent attack, and guard uptime (should NOT be ~1.0)",
    build=build,
    requires=("armed_opponent", "guarding_weapon"),
)
