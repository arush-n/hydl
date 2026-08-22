"""Dodging: pay for spending invulnerability frames on an actual incoming hit.

Dodge grants i-frames -- `dodge_invulnerability_remaining_seconds` in mechanics
state, sized by `dodge_invulnerability_seconds` in the rules. A "last-second
dodge" is one whose window is open on the tick a hit would have landed, which is
what `late_reward` pays. Dodging into empty air pays `any_reward` (small, or
zero) so the policy is not merely trained to dodge constantly.

**Check this before trusting any dodge number:** the default execution profile
nulls authored dodge displacement (`arsenal/factory.py:84`). Under it a dodge
grants i-frames but does not move the agent, so a positional read of dodging
will show nothing while the mechanic is working fine. Verify which profile the
scene built with before concluding dodge is broken.

Also: dodge no longer has its own head. It is the four-option tail of
`locomotion_gait_compass`, starting at `ARSENAL_POLICY_LOCOMOTION_DODGE_START`.
Measured against the retired five-wide dodge head, no dodge at all was legal
under `fail_closed`, and only two of the four directions were ever legal on
`open_flat` and on Region+geometry. Only that count carried over to the merged
head -- re-measure before quoting option indices.
"""

from __future__ import annotations

import jax.numpy as jnp

from ..framework import AGENT_ENTITY, Minigame, terminated
from adk.contracts.shaping import wants_previous


def build(*, entity: int = AGENT_ENTITY, late_reward: float = 5.0,
          any_reward: float = 0.0):
    """Reward i-frames active while damage would land; optionally any dodge."""

    @wants_previous
    def shape(previous_state, state, info):
        del info
        field = "dodge_invulnerability_remaining_seconds"
        health = state.runtime.combat.health[:, entity]
        iframes = getattr(state.runtime.mechanics, field)[:, entity] > 0.0
        last = previous_state.runtime.combat.health[:, entity]
        last_iframes = getattr(
            previous_state.runtime.mechanics, field)[:, entity] > 0.0
        under_fire = (last - health) > 0.0
        started = iframes & ~last_iframes
        paid = (
            jnp.where(iframes & under_fire, jnp.float32(late_reward), 0.0)
            + jnp.where(started, jnp.float32(any_reward), 0.0)
        )
        return jnp.where(terminated(previous_state), jnp.float32(0.0), paid)

    return shape


MINIGAME = Minigame(
    name="dodge",
    teaches="time the i-frame window onto an incoming hit",
    tell="damage taken per opponent attack; dodges started vs dodges that absorbed",
    build=build,
    requires=("armed_opponent", "dodge_displacement_profile"),
)
