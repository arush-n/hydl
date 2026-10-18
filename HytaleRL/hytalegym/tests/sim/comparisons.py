"""Adapter output vs today's engine value, flattened to ``adapter:key`` pairs."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from hytalegym.sim.common.content import adapters as A

from . import gen5_live as live


def _flat(prefix: str, value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in value.items():
            out.update(_flat(f"{prefix}.{key}" if prefix else str(key), child))
        return out
    return {prefix: value}


def _pairs(tables) -> list[tuple[str, Callable[[], Any], Callable[[], Any]]]:
    variants = live.scalar_variants()
    spears = live.spear_rows()
    return [
        ("player_locomotion", lambda: A.player_locomotion(tables), live.player_locomotion),
        ("player_stamina", lambda: A.player_stamina(tables), live.player_stamina),
        ("mechanics_resource_constants", lambda: A.mechanics_resource_constants(tables), live.mechanics_resource_constants),
        ("control_immunity", lambda: A.control_immunity(tables), live.control_immunity),
        ("dodge_constants", lambda: A.dodge_constants(tables), live.dodge_constants),
        ("mechanics_factory_resources", lambda: A.mechanics_factory_resources(tables), live.mechanics_factory_resources),
        ("walk_steering_sprint_multiplier", lambda: {"value": A.walk_steering_sprint_multiplier(tables)}, lambda: {"value": live.walk_steering_sprint_multiplier()}),
        ("health_regeneration", lambda: A.health_regeneration(tables), live.health_regeneration),
        ("npc_matchup", lambda: A.npc_matchup(tables), live.npc_matchup),
        (
            "compiled_weapon_durability",
            lambda: {row["item_asset_id"]: row for row in A.compiled_weapon_durability(tables, live.compiled_source_ids())},
            lambda: {row["item_asset_id"]: row for row in live.compiled_weapon_durability()},
        ),
        (
            "scalar_variant_values",
            lambda: {row.asset_id: A.scalar_variant_values(tables, row.asset_id, row.template) for row in variants},
            lambda: {
                row.asset_id: {
                    "guard_stamina_value": row.guard_stamina_value,
                    "max_durability": row.max_durability,
                    "durability_loss_on_hit": row.durability_loss_on_hit,
                    "damage": tuple(row.damage),
                    "angled_damage": tuple(row.angled_damage),
                }
                for row in variants
            },
        ),
        ("iron_template_damage", lambda: A.iron_template_damage(tables), live.iron_template_damage),
        ("sword_knockback", lambda: A.sword_knockback(tables), live.sword_knockback),
        ("sword_selectors", lambda: A.sword_selectors(tables), live.sword_selectors),
        ("melee_hold_constants", lambda: A.melee_hold_constants(tables), live.melee_hold_constants),
        ("shortbow_tiers", lambda: A.shortbow_tiers(tables), live.shortbow_tiers),
        ("spear_rows", lambda: A.spear_rows(tables, spears), lambda: spears),
        ("potions", lambda: A.potions(tables), live.potions),
        ("outlander_arrow", lambda: A.outlander_arrow(tables), live.outlander_arrow),
        ("crafting_capacities", lambda: A.crafting_capacities(tables), live.crafting_capacities),
    ]


def same(adapted: Any, engine: Any) -> bool:
    """Bit-for-bit, except that float32 engine arrays compare at float32."""

    if isinstance(engine, np.floating) or isinstance(adapted, np.floating):
        return np.float32(adapted).tobytes() == np.float32(engine).tobytes()
    if isinstance(adapted, (list, tuple)) and isinstance(engine, (list, tuple)):
        return len(adapted) == len(engine) and all(same(a, e) for a, e in zip(adapted, engine))
    if isinstance(adapted, float) and isinstance(engine, float):
        return np.float64(adapted).tobytes() == np.float64(engine).tobytes()
    return type(adapted) in (int, float, bool) and type(engine) in (int, float, bool) and adapted == engine and (
        isinstance(adapted, bool) == isinstance(engine, bool)
    ) or adapted == engine


def compare(tables) -> tuple[int, dict[str, tuple[Any, Any]]]:
    """Number of values compared and every ``adapter:key`` that differs."""

    compared = 0
    differences: dict[str, tuple[Any, Any]] = {}
    for name, adapted_fn, engine_fn in _pairs(tables):
        adapted, engine = _flat("", adapted_fn()), _flat("", engine_fn())
        for key in sorted(set(adapted) | set(engine)):
            compared += 1
            a, e = adapted.get(key, "<absent>"), engine.get(key, "<absent>")
            if not same(a, e):
                differences[f"{name}:{key}"] = (a, e)
    return compared, differences
