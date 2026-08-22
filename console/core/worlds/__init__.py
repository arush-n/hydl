"""The world a run happens in.

``worldgen``      deterministic custom-world design previews
``custom_games``  validated Arena minigame drafts bound to WorldGen designs
``custom_goals``  finite goal options for the Custom minigame editor
``terrain``       the patch of ground under a run, for the map to draw
``replay_terrain`` seed + semantic-hash references to immutable terrain
"""

from __future__ import annotations

from console.core.worlds import custom_games as custom_games
from console.core.worlds import custom_goals as custom_goals
from console.core.worlds import terrain as terrain
from console.core.worlds import worldgen as worldgen
from console.core.worlds import replay_terrain as replay_terrain


__all__ = [
    "custom_games",
    "custom_goals",
    "replay_terrain",
    "terrain",
    "worldgen",
]
