"""Stay on the ground and face the fight.

Two failure modes that look like broken steering and are not:

**Jumping every tick blocks walking.** Airborne entities get no horizontal
translation, so a policy that discovers the jump head and holds it stops moving
altogether. In a positional objective that reads as "locomotion does not
train", and the fault is the action, not the reward.

**Facing is not free.** Attacks resolve toward where the agent is looking, and
`visible_target_planar_distance` is masked when the target is outside the view
sector -- so a policy that walks correctly while looking away perceives
nothing, and every target column it needs reads as absent rather than as zero.

Both are cheap correctives rather than objectives in their own right. Give this
a small weight alongside a real game; on its own it trains an agent to stand
still and stare, which scores perfectly and does nothing.
"""

from __future__ import annotations

import jax.numpy as jnp

from ..framework import AGENT_ENTITY, Minigame, TARGET_ENTITY, facing, grounded


def build(*, ground_reward: float = 1.0, aim_reward: float = 1.0,
          aim_floor: float = 0.0, entity: int = AGENT_ENTITY,
          target: int = TARGET_ENTITY):
    """Pay for being grounded, and for facing the target.

    `aim_floor` clips the alignment term from below, so looking away pays zero
    rather than a penalty. A negative payment for facing away is a penalty for
    turning *through* the far side, which punishes the rotation that fixes it.
    """

    on_ground = grounded(want=True)
    aim = facing(entity=entity, target=target)

    def shape(state, info):
        del info
        total = jnp.zeros(state.runtime.combat.position.shape[0],
                          dtype=jnp.float32)
        if ground_reward:
            total = total + jnp.float32(ground_reward) * on_ground(state)
        if aim_reward:
            total = total + jnp.float32(aim_reward) * jnp.maximum(
                aim(state), jnp.float32(aim_floor))
        return total

    return shape


MINIGAME = Minigame(
    name="footing",
    teaches="stay grounded and keep the target in view; a corrective, not a goal",
    tell=("fraction of ticks grounded, and mean facing alignment. Pair with a "
          "positional game -- alone this is maximised by standing still"),
    build=build,
)
