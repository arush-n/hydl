"""Race: get to a point as fast as possible.

Pays for closing distance and adds a bonus on arrival. There is no time term in
the native reward and none is added here -- with a per-tick progress payment,
arriving sooner collects the same total sooner, and the PPO discount does the
rest. An explicit time penalty on top would double-count.

**The goal must be a real traversal node.** The region's graph has 21,174
candidate nodes (4,477 navigation-legal); an invented coordinate can sit inside
terrain, and the agent would then be paid for approaching a wall.
`scene.fixture.metadata` reports `source_node` / `destination_node` for the
spawn actually used.
"""

from __future__ import annotations

from ..framework import Minigame, goal_distance, progress


def build(goal, *, tolerance: float = 1.5, arrive_bonus: float = 10.0,
          horizontal: bool = True):
    return progress(
        goal_distance(goal, horizontal=horizontal),
        sign=-1.0, bonus=arrive_bonus, within=tolerance)


MINIGAME = Minigame(
    name="reach",
    teaches="navigate to a specific location under time pressure",
    tell="ticks-to-arrival, and arrival rate vs a random-walk control",
    build=build,
    requires=("goal",),
)
