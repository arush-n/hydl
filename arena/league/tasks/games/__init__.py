"""Authored games, grouped by domain.

Nothing here is canonical. These are the games that happen to exist; add your
own module beside them (start from ``TEMPLATE.py``) and register it. The
manager, curriculum and league cannot tell the difference.

Importing this package registers nothing -- call :func:`register_all`, or
register individual games yourself.
"""

from __future__ import annotations

from arena.tasks.framework.base import Task
from arena.tasks.framework.registry import register

from arena.tasks.games.combat.defend import DEFEND
from arena.tasks.games.combat.duel import DUEL
from arena.tasks.games.combat.survive import SURVIVE
from arena.tasks.games.movement.track import TRACK
from arena.tasks.games.movement.traverse import TRAVERSE
from arena.tasks.games.world.build import BUILD
from arena.tasks.games.world.generated import (
    DESERT_TRACK,
    PLAINS_DUEL,
    VOLCANIC_SURVIVE,
)
from arena.tasks.games.world.reach import REACH

#: Every game shipped in this package, by domain.
BY_DOMAIN: dict[str, tuple[Task, ...]] = {
    "combat": (SURVIVE, DUEL, DEFEND),
    "movement": (TRACK, TRAVERSE),
    "world": (
        REACH,
        BUILD,
        PLAINS_DUEL,
        DESERT_TRACK,
        VOLCANIC_SURVIVE,
    ),
}

#: Heads no shipped game exercises, with the reason, so coverage gaps are
#: declared rather than discovered. Pinned by a test.
UNCOVERED_HEADS: dict[str, str] = {
    "hotbar_none_plus_slots": (
        "weapon switching is plumbed end to end -- policy head, ArsenalCommands, "
        "set_active_hotbar_slot, and the native bridge -- but no shipped game can "
        "exercise it yet, because `inventory_from_loadout` seeds exactly ONE item "
        "into hotbar slot zero and leaves slots 1-8 empty. Switching today can "
        "only unequip. Give a game a multi-weapon loadout and this declaration "
        "goes away."
    ),
}

SHIPPED: tuple[Task, ...] = tuple(
    task for domain in sorted(BY_DOMAIN) for task in BY_DOMAIN[domain]
)


def register_all(*, replace: bool = False, domains: tuple[str, ...] | None = None):
    """Opt in to the shipped games, optionally only some domains."""

    selected = sorted(BY_DOMAIN) if domains is None else list(domains)
    unknown = [name for name in selected if name not in BY_DOMAIN]
    if unknown:
        raise KeyError(
            f"unknown domain(s): {unknown}; have: {', '.join(sorted(BY_DOMAIN))}"
        )
    registered = []
    for domain in selected:
        for task in BY_DOMAIN[domain]:
            registered.append(register(task, replace=replace))
    return tuple(registered)


__all__ = [
    "BUILD",
    "BY_DOMAIN",
    "DEFEND",
    "DESERT_TRACK",
    "DUEL",
    "PLAINS_DUEL",
    "REACH",
    "SHIPPED",
    "SURVIVE",
    "TRACK",
    "TRAVERSE",
    "UNCOVERED_HEADS",
    "VOLCANIC_SURVIVE",
    "register_all",
]
