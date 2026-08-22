"""Change the world at a chosen spot -- the block-placement heads.

Two heads only this game reaches: `block_none_plus_candidates` picks *which*
block, `block_primary_secondary_trigger` picks place against break. That pairing
is the whole point -- a curriculum that never varies the block ID trains a
policy that can only ever place one thing.

**The success criterion is weaker than the name suggests, and deliberately so.**
No observation column counts placed blocks. `changed_terrain_at_interaction`
detects that traversability *moved* at the interaction target, which a placement
and a break both produce. It proves the world changed there, not what is now
there. Do not read a pass here as "the agent placed block N".

Not yet verified against the bridge. The world verbs fail closed without a
Region action provider (GAP-5), so on a scene without one this measures the
fail-closed path rather than building.
"""

from __future__ import annotations

from arena.tasks.framework.base import Difficulty, Task
from arena.tasks.framework.goals import changed_terrain_at_interaction

BUILD = Task(
    name="build",
    description="Change the terrain at the interaction target.",
    heads_exercised=(
        "block_none_plus_candidates",
        "block_primary_secondary_trigger",
        "use_off_on",
        "locomotion_gait_compass",
    ),
    loadout="iron_sword",
    difficulties=(
        Difficulty("undisturbed", opponent_armed=False, opponent_profile="iron_sword"),
        Difficulty("contested", opponent_armed=True, opponent_profile="iron_daggers"),
    ),
    success=changed_terrain_at_interaction(),
)

__all__ = ["BUILD"]
