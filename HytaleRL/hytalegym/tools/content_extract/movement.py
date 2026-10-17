"""Movement configs: player locomotion constants (speeds, jump, air, slopes).

Sources: every ``Server/Entity/MovementConfig/*.json`` (flat scalar documents)
and the player binding ``Server/GameplayConfigs/Default.json`` ->
``Player.MovementConfig``. Fields are copied verbatim (minus comments), so a key
the next version adds (0.7.2: ``MaxSlopeAngleDegrees``) appears as a new field.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from .archive import GameArchive
from .common import gameplay

FAMILY = "movement"
VERSION = 1
CONFIG_PREFIX = "Server/Entity/MovementConfig/"
GAMEPLAY_CONFIG = "Server/GameplayConfigs/Default.json"


def extract(archive: GameArchive) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for path in archive.list(CONFIG_PREFIX):
        document = archive.read_json(path)
        if not isinstance(document, dict):
            raise ValueError(f"{path} is not an object")
        rows[f"config:{PurePosixPath(path).stem}"] = {
            "kind": "movement_config",
            "source": path,
            "fields": gameplay(document),
        }
    gameplay_config = archive.read_json(GAMEPLAY_CONFIG)
    player = gameplay_config.get("Player", {})
    rows["binding:player"] = {
        "kind": "player_binding",
        "source": GAMEPLAY_CONFIG,
        "movement_config": player.get("MovementConfig"),
        "hitbox_collision_config": player.get("HitboxCollisionConfig"),
    }
    return rows
