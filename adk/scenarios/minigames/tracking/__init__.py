"""Hold aim on a target that is moving.

Distinct from `footing`'s aim term, which pays the *level* of alignment and (by
its own docstring) is maximised by standing still and staring. This weights
alignment by **how far the target's bearing swung since the last tick**, so a
stationary target pays ~0 however well aimed, and the term is driven by the
target's motion rather than the agent's.

Pays nothing when the opponent does not move -- correct, but it makes the
`tell` unreadable on an inert or parked opponent.
"""

from __future__ import annotations

import jax.numpy as jnp

from adk.contracts.shaping import wants_previous

from ..framework import AGENT_ENTITY, Minigame, TARGET_ENTITY, terminated

#: Bearing swing (radians/tick) that counts as a full unit of "trying to leave".
#: Set so a target crossing at the ruleset's 8.0 max speed near 2 units of
#: separation saturates around 1.0. Not an authored constant -- retune it with
#: `opponent_distance`.
REFERENCE_SWING = 0.12


def _bearing(state, *, entity: int, target: int):
    position = state.runtime.combat.position
    delta = position[:, target] - position[:, entity]
    return jnp.arctan2(delta[..., 0], delta[..., 2])


def _wrap(angle):
    """Fold an angle difference into [-pi, pi].

    `arctan2` readings either side of the branch cut differ by ~2*pi while the
    real motion was ~0. Unwrapped, every pass behind the agent would score as a
    full-circle swing -- the largest weight exactly where tracking is hardest.
    """

    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


def build(*, reward: float = 1.0, hold: float = 0.25, floor: float = 0.0,
          reference_swing: float = REFERENCE_SWING,
          entity: int = AGENT_ENTITY, target: int = TARGET_ENTITY):
    """Pay alignment scaled by the target's bearing swing.

    `hold` charges per radian of the agent's own yaw change, so sweeping the
    view back and forth across the target cannot farm the crossings; 0.0
    measures alignment alone. `floor` clips alignment from below -- 0.0 means
    facing away pays nothing rather than punishing the rotation that fixes it.
    """

    @wants_previous
    def shape(previous_state, state, info):
        del info
        bearing = _bearing(state, entity=entity, target=target)
        swing = jnp.abs(_wrap(
            bearing - _bearing(previous_state, entity=entity, target=target)))

        yaw = state.runtime.combat.yaw[:, entity]
        aimed = jnp.maximum(jnp.cos(bearing - yaw), jnp.float32(floor))
        weight = jnp.minimum(
            swing / jnp.float32(reference_swing), jnp.float32(1.0))
        total = jnp.float32(reward) * aimed * weight

        if hold:
            turned = jnp.abs(_wrap(
                yaw - previous_state.runtime.combat.yaw[:, entity]))
            total = total - jnp.float32(hold) * turned

        # Across a reset the previous bearing belongs to the last episode.
        return jnp.where(terminated(previous_state), jnp.float32(0.0), total)

    return shape


MINIGAME = Minigame(
    name="tracking",
    teaches="hold aim on a moving target, not on any target",
    tell=("mean alignment weighted by bearing swing. Compare with `footing`'s "
          "raw alignment on the same run -- a policy that only learned to "
          "stare scores well there and near zero here"),
    build=build,
    requires=("armed_opponent",),
)
