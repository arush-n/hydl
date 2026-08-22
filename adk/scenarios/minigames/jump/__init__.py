"""Gain height on purpose.

Paying for airtime is the wrong term: airborne entities get **no horizontal
translation**, so an agent that holds the jump head stops moving entirely --
the failure `footing` exists to correct. A `grounded(want=False)` reward would
pay for causing it.

So this pays height actually gained, `max(0, dy)` while airborne. Rising pays;
apex and descent pay nothing. `drag` charges per airborne tick, which is what
makes jump-spam a net loss instead of break-even.

Vertical by construction, unlike every other positional game here. It reads the
delta rather than the level because spawn height varies hugely across node
seeds (y=126 / y=26 / y=46) -- an absolute height reward would rank seeds, not
agents.
"""

from __future__ import annotations

import jax.numpy as jnp

from adk.contracts.shaping import wants_previous

from ..framework import AGENT_ENTITY, Minigame, terminated


def build(*, reward: float = 1.0, drag: float = 0.1, landing: float = 0.0,
          entity: int = AGENT_ENTITY):
    """Pay for height gained while airborne.

    `drag` is charged per airborne tick, so airtime that gains nothing costs;
    0.0 measures raw ascent. `landing` pays once on the airborne->grounded edge
    and defaults off -- it only means something next to a goal that makes
    *where* you land matter, so pair it with `reach` or `checkpoint`.
    """

    @wants_previous
    def shape(previous_state, state, info):
        del info
        position = state.runtime.combat.position
        rise = jnp.maximum(
            position[:, entity, 1]
            - previous_state.runtime.combat.position[:, entity, 1],
            jnp.float32(0.0),
        )
        on_ground = jnp.asarray(state.runtime.combat.agent_grounded).astype(bool)
        airborne = ~on_ground

        total = jnp.where(airborne, jnp.float32(reward) * rise, jnp.float32(0.0))
        if drag:
            total = total - jnp.where(
                airborne, jnp.float32(drag), jnp.float32(0.0))
        if landing:
            was_airborne = ~jnp.asarray(
                previous_state.runtime.combat.agent_grounded).astype(bool)
            total = total + jnp.where(
                on_ground & was_airborne, jnp.float32(landing), jnp.float32(0.0))

        # A reset teleports the agent; the spawn delta would read as a huge rise.
        return jnp.where(terminated(previous_state), jnp.float32(0.0), total)

    return shape


MINIGAME = Minigame(
    name="jump",
    teaches="jump to gain height, and not otherwise",
    tell=("height gained per airborne interval, and airborne fraction. If both "
          "rise together that is jump-spam, not learning -- they must move in "
          "opposite directions. Runs against `footing`, which pays to stay "
          "grounded; do not weight both heavily in one run"),
    build=build,
)
