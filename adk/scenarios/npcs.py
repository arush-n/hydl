"""Opponent ability policies -- NPC variety at the seam the Gym already exposes.

The movement half of NPC behaviour is already modelled: `OpponentMemoryState`
carries mode (inactive/chase/search/return-home), home position, last-seen
position, pursuit and search timers. That is aggro-and-leash, and it is not
mine to reimplement.

What is thin is ability *choice*. The Gym ships exactly two providers:
`first_legal_opponent_ability_slots` (always the lowest legal slot) and
`inert_opponent_ability_slots`. Every opponent in every run so far has been one
of those, so "the agent beat the opponent" has only ever meant "beat a
deterministic lowest-slot bot".

Each policy here has the Gym's signature -- `(state, world, config) -> (batch,
entity) int32`, -1 meaning no ability -- and reuses `ability_legal_mask`, so
cooldown, resources, statuses and world evidence still gate every choice. None
of these bypass legality; they only choose differently among legal slots.

Pass one as `opponent_ability_provider=` to `make_arsenal_ppo_environment`.
"""

from __future__ import annotations

import jax.numpy as jnp

from hytalegym.jax.combat.arsenal.programs.ability import ability_legal_mask
from hytalegym.jax.combat.opponents.schema.contract import OPPONENT_MODE_CHASE


def _legal(state, world, config):
    """(legal (B,E,S), controlled-and-chasing (B,E)) -- the shared preamble."""

    legal = ability_legal_mask(
        state.arsenal, state.mechanics, config.loadout, world,
        state.combat.health > jnp.float32(0.0))
    active = config.opponent_controller_mask & (
        state.opponent_memory.mode == jnp.int32(OPPONENT_MODE_CHASE))
    return legal, active & jnp.any(legal, axis=2)


def highest_slot(state, world, config):
    """Mirror of the shipped baseline, from the other end of the bank.

    Authored banks are ordered, so the lowest slot is usually the cheap basic
    attack. Taking the highest instead is the cheapest possible test of whether
    a result depends on WHICH ability the opponent picks -- if a win rate is
    unchanged between this and the shipped provider, ability choice is not what
    decided the fight.
    """

    legal, ok = _legal(state, world, config)
    slots = jnp.arange(legal.shape[2], dtype=jnp.int32)
    highest = jnp.max(jnp.where(legal, slots[None, None, :], -1), axis=2)
    return jnp.where(ok, highest, jnp.int32(-1))


def random_slot(state, world, config):
    """Uniform over legal slots, keyed off the pursuit clock.

    A deterministic opponent is exploitable by memorisation: the policy can
    learn the fixed reply rather than the mechanic. This is the control that
    separates those.

    Uses `pursuit_elapsed_ticks` rather than a PRNG key because the provider
    signature takes no key -- it is a fixed function of state, so the scene
    stays reproducible.
    """

    legal, ok = _legal(state, world, config)
    tick = state.opponent_memory.pursuit_elapsed_ticks.astype(jnp.int32)
    # Hash the tick per entity so two opponents do not act in lockstep.
    # Multiplier must fit int32 -- Knuth's 2654435761 overflows and raises
    # OverflowError at trace time, not at import, so it survives a smoke test.
    entity = jnp.arange(legal.shape[1], dtype=jnp.int32)[None, :]
    draw = (tick * jnp.int32(1103515245) + entity * jnp.int32(40503))
    count = jnp.sum(legal, axis=2)
    target = jnp.where(count > 0, jnp.abs(draw) % jnp.maximum(count, 1), 0)
    # index of the target-th legal slot
    rank = jnp.cumsum(legal.astype(jnp.int32), axis=2) - 1
    hit = legal & (rank == target[..., None])
    chosen = jnp.argmax(hit, axis=2).astype(jnp.int32)
    return jnp.where(ok & jnp.any(hit, axis=2), chosen, jnp.int32(-1))


def patient(state, world, config):
    """Hold fire until the pursuit has been going a while, then commit.

    Models an NPC that closes before swinging instead of flailing at max range.
    Directly relevant because melee damage is a band, not a threshold -- an
    opponent that attacks immediately spends most of its swings outside its own
    damage window and reads as harmless for the wrong reason.
    """

    legal, ok = _legal(state, world, config)
    slots = jnp.arange(legal.shape[2], dtype=jnp.int32)
    lowest = jnp.argmax(legal, axis=2).astype(jnp.int32)
    ready = state.opponent_memory.pursuit_elapsed_ticks >= jnp.int32(12)
    del slots
    return jnp.where(ok & ready, lowest, jnp.int32(-1))


#: Name -> provider. `first_legal` and `inert` stay the Gym's own so a sweep can
#: include the shipped baseline without importing from two places.
POLICIES = {
    "highest_slot": highest_slot,
    "random_slot": random_slot,
    "patient": patient,
}
