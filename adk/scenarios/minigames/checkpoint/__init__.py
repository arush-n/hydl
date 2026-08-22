"""Checkpoint route: reach a sequence of points in order.

`reach` pays for closing on **one** goal and stops meaning anything once the
agent arrives. A route keeps paying: arrive, the target advances, close again.
That is the difference between "can it walk to a spot" and "can it navigate",
and it is the objective that actually exercises a traversal graph rather than a
straight line.

The objective is *remaining route distance*, evaluated per row from position
alone -- so rows progressing at different rates from the same spawn never drag
each other, and nothing has to be remembered between ticks. See
:func:`~adk.scenarios.minigames.framework.waypoints` for what that costs: a
potential admits a shortcut to the last checkpoint, where the per-row index it
replaced enforced order but did not survive `jax.jit`.

**Checkpoints must be real traversal nodes.** The region graph has 21,174
candidates, 4,477 of them navigation-legal; an invented coordinate can sit
inside terrain and the agent is then paid for walking into a wall. Build a
route from the graph, not from arithmetic on the spawn:

    from worlds.zones import zones
    zone = zones(region_seed)[0]
    route = zone.nodes[::len(zone.nodes) // 6][:6]      # spread along the zone

**Strong connectivity matters here specifically.** Drops are one-way doors, so
a route through a weakly-connected component can be unfinishable from the far
side. `worlds.zones` returns strongly-connected components, which is
the property that makes a route guaranteed-completable in both directions.
"""

from __future__ import annotations

from ..framework import AGENT_ENTITY, Minigame, progress, waypoints


def build(route, *, tolerance: float = 1.5, arrive_bonus: float = 10.0,
          entity: int = AGENT_ENTITY, horizontal: bool = True):
    """Pay for closing on the current checkpoint, bonus on each arrival.

    No time penalty, for the same reason `reach` has none: with a per-tick
    progress payment, arriving sooner collects the same total sooner and the
    PPO discount does the rest. An explicit time term would double-count.

    `arrive_bonus` is paid while the route's remaining distance is under
    `tolerance` -- that is, on the final approach -- rather than once at the
    very end. A sparse end-of-route bonus on an objective this long is a reward
    almost no early policy ever sees.
    """

    read = waypoints(route, entity=entity, horizontal=horizontal)
    shape = progress(read, sign=-1.0, bonus=arrive_bonus, within=tolerance)
    shape.route_length = read.route_length          # type: ignore[attr-defined]
    shape.total_distance = read.total_distance      # type: ignore[attr-defined]
    return shape


MINIGAME = Minigame(
    name="checkpoint",
    teaches="navigate a route in order, not just close one distance",
    tell=("checkpoints cleared per episode, against a random-walk control; "
          "the control clears ~0 beyond the one it spawns next to"),
    build=build,
    requires=("route",),
)
