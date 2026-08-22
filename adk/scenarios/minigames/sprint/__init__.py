"""Sprint: keep moving. Pays per-tick horizontal displacement.

The cheapest check that locomotion is learnable at all. If this does not train,
no positional objective will, and the problem is the movement heads or the
action arbitration rather than the reward -- so run this before `reach` or
`flee` and treat it as their precondition.

Known interaction worth expecting in the numbers: **jumping every tick blocks
walking.** Airborne entities get no horizontal translation, so a policy that
learns to hold jump scores near zero here and looks like broken steering. An
active interaction also suppresses movement outright, so displacement and
attack rate are not independent.
"""

from __future__ import annotations

from ..framework import AGENT_ENTITY, Minigame, displacement


def build(*, entity: int = AGENT_ENTITY, horizontal: bool = True):
    """Pay per-tick horizontal displacement.

    Displacement is already a per-tick delta, so it is paid directly rather
    than through `progress` -- wrapping it would pay the *change in speed*.
    """

    return displacement(entity=entity, horizontal=horizontal)


#: Kept as a name because callers import it. `displacement` no longer holds any
#: state -- it reads the two states it is handed -- so there is nothing for a
#: "stateful" variant to do differently, and the reason the old `build` paid a
#: constant zero (it rebuilt the reader every call, discarding the remembered
#: position) no longer exists either.
build_stateful = build


MINIGAME = Minigame(
    name="sprint",
    teaches="sustained locomotion; precondition for every positional task",
    tell="mean per-tick displacement vs an idle control (which must be ~0)",
    build=build,
)
