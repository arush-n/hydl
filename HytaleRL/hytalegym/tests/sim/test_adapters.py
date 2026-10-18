"""Adapters on the gen-5 tables (overlay applied) equal today's engine, bit for bit."""

from __future__ import annotations

import json

from hytalegym.sim.common.content import adapters as A
from hytalegym.sim.common.content import load

from . import support
from .comparisons import _pairs, compare


def test_adapters_reproduce_every_engine_value() -> None:
    compared, differences = compare(load("0.5.7"))
    assert compared > 700
    assert differences == {}, differences


def test_swap_plan_entries_resolve_to_adapters_sources_and_lines() -> None:
    package = support.HYTALEGYM / "hytalegym"
    exercised = {name for name, _, _ in _pairs(load("0.5.7"))} | {"native_crafting_recipes"}
    for entry in A.SWAP_PLAN:
        assert callable(getattr(A, entry.adapter)), entry
        assert entry.adapter in exercised, entry  # compared with the engine (A1/adapters)
        assert entry.adapter in A.ADAPTER_SOURCES, entry
        assert entry.kind in ("module_constant", "inline_literal", "function_default", "json_table")
        path = (package / entry.module_path).resolve()
        assert path.is_file(), entry
        assert entry.symbol in path.read_text(encoding="utf-8"), entry


def test_redirect_list_covers_the_swap_plan() -> None:
    document = json.loads((support.SIM / "common" / "content" / "redirects.json").read_text(encoding="utf-8"))
    assert document["schema"] == "hytalerl_render_redirects_v1"
    records = {(r["target"], r["module"], r["symbol"]) for r in document["redirects"]}
    plan = {
        (e.target, ("HytaleRL/hytalegym/hytalegym/" + e.module_path).replace("hytalegym/hytalegym/../tests/", "hytalegym/tests/"), e.symbol)
        for e in A.SWAP_PLAN
    }
    assert records == plan
    for record in document["redirects"]:
        assert record["status"] in ("clean", "uncommitted", "absent_in_head")
        assert record["head_line"] is not None or record["status"] == "absent_in_head", record
        assert record["content"], record


def test_gen7_tables_feed_the_same_adapters() -> None:
    gen7 = load("0.7.2")
    assert A.player_locomotion(gen7) == A.player_locomotion(load("0.5.7"))  # MovementConfig values unchanged
    assert A.mechanics_factory_resources(gen7)["mana_maximum"] == 100.0
    assert A.crafting_capacities(gen7)["RECIPE_TABLE_CAPACITY"] == 2047
    assert A.shortbow_tiers(gen7)["speeds"] == (12.0, 22.0, 35.0, 50.0, 85.0)
    assert A.sword_knockback(gen7)["Weapon_Sword_Primary_Swing_Left"]["force"] == (0.0, 0.2, -1.0, 4.3)


def test_redirect_list_carries_every_step_duration_site() -> None:
    document = json.loads((support.SIM / "common" / "content" / "redirects.json").read_text(encoding="utf-8"))
    step = document["step_duration"]
    assert step["site_count"] == len(step["sites"]) > 300
    for site in step["sites"]:
        assert site["module"] and site["symbol"] and site["head_line"] >= 1 and site["scope"] in ("engine", "consumer")
    keys = {(s["kind"], s["symbol"]) for s in step["sites"]}
    assert ("ruleset_json", "engine.loaded_delta_seconds") in keys
    assert ("ruleset_json", "engine.nominal_delta_seconds") in keys
    assert any(s["kind"] == "delta_field" and s["symbol"] == "_motion_delta" for s in step["sites"])


def test_sprint_swap_record_names_the_value_not_the_key():
    """2026-10-18: the record for locomotion.py:177 names the literal 1.273, not the dict key."""
    document = json.loads((support.SIM / "common" / "content" / "redirects.json").read_text(encoding="utf-8"))
    (record,) = [r for r in document["redirects"] if r["adapter"] == "walk_steering_sprint_multiplier"]
    assert (record["symbol"], record["symbol_kind"], record["head_line"]) == ("1.273", "numeric_literal", 177)
