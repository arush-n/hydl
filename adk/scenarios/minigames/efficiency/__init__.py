"""Kill in the fewest moves. Charge per accepted action; the kill still pays.

"Kill it" and "kill it efficiently" are the same objective until wasted actions
cost something -- the native reward pays damage and completion with no action
cost, so a flailing policy scores identically to a precise one.

Charged per **accepted** ability, not per request. A request refused on cooldown
or resources costs the agent nothing in the world, and charging it would train
the policy to stop asking rather than to ask better. Cooldowns already make
per-tick request rates include silent no-ops.

Tuning: `cost` must stay well under the damage a landed ability is worth, or the
optimum is to do nothing. With `target_damage_reward_scale = 2.0` and a sword
ability landing meaningful damage, a cost near 0.5 is a starting point -- and
"the agent stopped attacking" is the signal that it is set too high.
"""

from __future__ import annotations

from ..framework import AGENT_ENTITY, Minigame, accepted, penalise


def build(*, cost: float = 0.5, entity: int = AGENT_ENTITY):
    return penalise(accepted(entity=entity), cost=cost)


MINIGAME = Minigame(
    name="efficiency",
    teaches="win with fewer actions; stop spamming abilities that miss",
    tell="accepted abilities per kill, vs the unshaped baseline at equal win rate",
    build=build,
    requires=("armed_opponent",),
)
