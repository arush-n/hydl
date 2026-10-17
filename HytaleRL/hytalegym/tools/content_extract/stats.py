"""Entity stats: stamina, mana, oxygen, health, signature energy, charges.

Sources: every ``Server/Entity/Stats/*.json`` and the stamina plugin block of
``Server/GameplayConfigs/Default.json`` (``Plugin.Stamina``: the sprint regen
delay the simulator applies). Regeneration rules keep their authored order;
the simulator's per-second constants are derived from them in
``hytalegym.sim.common.content.adapters``.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

from .archive import GameArchive
from .common import gameplay

FAMILY = "stats"
VERSION = 1
STATS_PREFIX = "Server/Entity/Stats/"
GAMEPLAY_CONFIG = "Server/GameplayConfigs/Default.json"


def _rule(rule: dict[str, Any]) -> dict[str, Any]:
    return {
        "interval": rule.get("Interval"),
        "amount": rule.get("Amount"),
        "regen_type": rule.get("RegenType"),
        "clamp_at_zero": bool(rule.get("ClampAtZero", False)),
        "conditions": gameplay(rule.get("Conditions", [])),
    }


def extract(archive: GameArchive) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for path in archive.list(STATS_PREFIX):
        document = archive.read_json(path)
        stem = PurePosixPath(path).stem
        rows[f"stat:{stem}"] = {
            "kind": "stat",
            "source": path,
            "initial_value": document.get("InitialValue"),
            "min": document.get("Min"),
            "max": document.get("Max"),
            "shared": document.get("Shared"),
            "reset_type": document.get("ResetType"),
            "regenerating": [_rule(rule) for rule in document.get("Regenerating", [])],
            "min_value_effects": gameplay(document.get("MinValueEffects")),
            "max_value_effects": gameplay(document.get("MaxValueEffects")),
            "extra": gameplay(
                {
                    key: value
                    for key, value in document.items()
                    if key
                    not in {
                        "InitialValue",
                        "Min",
                        "Max",
                        "Shared",
                        "ResetType",
                        "Regenerating",
                        "MinValueEffects",
                        "MaxValueEffects",
                    }
                }
            ),
        }
    plugin = archive.read_json(GAMEPLAY_CONFIG).get("Plugin", {})
    stamina = plugin.get("Stamina", {})
    sprint = stamina.get("SprintRegenDelay", {})
    rows["plugin:stamina"] = {
        "kind": "stamina_plugin",
        "source": GAMEPLAY_CONFIG,
        "sprint_regen_delay_stat": sprint.get("EntityStatId"),
        "sprint_regen_delay_value": sprint.get("Value"),
        "raw": gameplay(stamina),
    }
    return rows
