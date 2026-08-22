"""Defence that fired: pay per hit actually absorbed or evaded.

`block` and `dodge` both infer success from a drop in health that did not
happen. That inference cannot separate **"the guard absorbed it"** from **"the
attack missed anyway"**, and on a scene where the opponent whiffs often those
look identical while meaning opposite things -- one is a defensive skill, the
other is the opponent being bad. A policy can score well on the inferred
version by standing out of range.

`info.arsenal_info` reports both directly: `blocked_hits` and
`invulnerable_hits`, `(batch,)` int32. Nothing was reading them. This does.

So this is the honest counterpart to `block`/`dodge`, not a replacement:
those pay guard *timing* against incoming damage, which is the behaviour; this
pays the mechanic having demonstrably fired, which is the evidence. If they
disagree, the inferred one is the one to distrust.
"""

from __future__ import annotations

from ..framework import Minigame, defended, event


def build(*, block_reward: float = 2.0, evade_reward: float = 3.0):
    """Pay per hit stopped by guard and per hit evaded by i-frames.

    Evasion pays more by default because it costs a resource and a commitment
    window, while guard can be held. Equal weights would train the agent to
    hold guard permanently, which is exactly the degenerate policy `block`'s
    idle cost exists to prevent.
    """

    blocks = defended(blocked=True)
    evades = defended(blocked=False)

    if not evade_reward:
        return event(blocks, reward=block_reward)
    if not block_reward:
        return event(evades, reward=evade_reward)

    pay_blocks = event(blocks, reward=block_reward)
    pay_evades = event(evades, reward=evade_reward)

    def shape(state, info):
        return pay_blocks(state, info) + pay_evades(state, info)

    return shape


MINIGAME = Minigame(
    name="bulwark",
    teaches="actually absorb or evade incoming hits, not merely avoid range",
    tell=("blocked_hits + invulnerable_hits per episode. Damage taken should "
          "fall WITHOUT mean separation rising -- if separation rose, the "
          "agent solved it by leaving, which is `flee`, not defence"),
    build=build,
    requires=("armed_opponent",),
)
