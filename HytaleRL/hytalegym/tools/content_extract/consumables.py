"""Consumables: potions, food and every item with ``Consumable: true``.

For each consumable the item's ``InteractionVars`` and its reachable
interaction graph (the weapons walker) are scanned for ``ApplyEffect``,
``ClearEntityEffect`` and ``ChangeStat`` nodes. Every effect id they name gets
an ``effect:<id>`` row from ``Server/Entity/Effects/**`` (Parent-resolved):
duration, stat modifiers, value type, overlap, cooldown.
"""

from __future__ import annotations

from typing import Any

from .archive import GameArchive
from .common import gameplay, resolve_parented
from .weapons import _Graph, _Visitor

FAMILY = "consumables"
VERSION = 1
EFFECT_PREFIX = "Server/Entity/Effects/"


def _scan(value: Any, applied: set[str], cleared: set[str], stats: list[dict[str, Any]]) -> None:
    if isinstance(value, list):
        for child in value:
            _scan(child, applied, cleared, stats)
        return
    if not isinstance(value, dict):
        return
    kind = value.get("Type")
    if kind == "ApplyEffect" and isinstance(value.get("EffectId"), str):
        applied.add(value["EffectId"])
    if kind == "ClearEntityEffect" and isinstance(value.get("EntityEffectId"), str):
        cleared.add(value["EntityEffectId"])
    if kind == "ChangeStat" and isinstance(value.get("StatModifiers"), dict):
        stats.append(
            {
                "stat_modifiers": value["StatModifiers"],
                "behaviour": value.get("Behaviour"),
                "value_type": value.get("ValueType"),
            }
        )
    for child in value.values():
        _scan(child, applied, cleared, stats)


def extract(archive: GameArchive) -> dict[str, Any]:
    graph = _Graph(archive)
    effects = archive.index(EFFECT_PREFIX)
    rows: dict[str, Any] = {}
    referenced: set[str] = set()
    for asset_id in sorted(graph.items):
        item, chain = graph.resolve("item", asset_id)
        if item.get("Consumable") is not True:
            continue
        variables = item.get("InteractionVars", {})
        variables = variables if isinstance(variables, dict) else {}
        visitor = _Visitor(graph, variables)
        equipped = item.get("Interactions", {})
        visitor.visit(equipped, "Interactions", prefer_root=True, allow_graph=True)
        applied: set[str] = set()
        cleared: set[str] = set()
        stats: list[dict[str, Any]] = []
        _scan(variables, applied, cleared, stats)
        for namespace, stem in visitor.named.values():
            document = graph.resolve(namespace, stem)[0]
            _scan(document, applied, cleared, stats)
        referenced |= applied | cleared
        rows[f"item:{asset_id}"] = {
            "kind": "consumable",
            "source": graph.items[asset_id],
            "parent_chain": list(chain),
            "max_stack": item.get("MaxStack"),
            "interactions": {
                key: value for key, value in sorted(equipped.items()) if isinstance(value, str)
            }
            if isinstance(equipped, dict)
            else {},
            "effects_applied": sorted(applied),
            "effects_cleared": sorted(cleared),
            "stat_changes": stats,
            "graph": sorted(visitor.named),
        }
    for effect_id in sorted(referenced):
        if effect_id not in effects:
            rows[f"effect:{effect_id}"] = {"kind": "effect", "source": None, "missing": True}
            continue
        document, chain = resolve_parented(effect_id, effects, archive.read_json)
        rows[f"effect:{effect_id}"] = {
            "kind": "effect",
            "source": effects[effect_id],
            "parent_chain": list(chain),
            "duration": document.get("Duration"),
            "stat_modifiers": document.get("StatModifiers"),
            "value_type": document.get("ValueType"),
            "overlap_behavior": document.get("OverlapBehavior"),
            "damage_calculator_cooldown": document.get("DamageCalculatorCooldown"),
            "debuff": document.get("Debuff"),
            "document": gameplay(
                {key: value for key, value in document.items() if key != "StatusEffectIcon"}
            ),
        }
    return rows
