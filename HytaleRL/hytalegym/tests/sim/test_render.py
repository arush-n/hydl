"""Overlay spec, renderer and bench helpers (static: no JAX rollouts).

The rollout-level identity gate is ``test_render_identity.py``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from hytalegym.sim.common import bench
from hytalegym.sim.common import render as R
from hytalegym.sim.common.overlay import (
    SOURCE_PREFIX,
    HeadSource,
    Hook,
    OverlayError,
    Patch,
    Redirect,
    imported_modules,
    module_name,
    resolve_modules,
    source_path,
    validate,
)
from hytalegym.sim.gen5.overlay import OVERLAY as GEN5
from hytalegym.sim.gen7.overlay import OVERLAY as GEN7

TOKENS = "world/tokens/__init__.py"


@pytest.fixture(scope="module")
def source() -> HeadSource:
    return HeadSource()


@pytest.fixture(scope="module")
def gen5(source, tmp_path_factory):
    out = tmp_path_factory.mktemp("render") / "gen5"
    return out, R.render(GEN5, out, source=source)


@pytest.fixture(scope="module")
def gen7(source, tmp_path_factory):
    out = tmp_path_factory.mktemp("render") / "gen7"
    return out, R.render(GEN7, out, source=source)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_overlays_validate_against_head(source):
    assert validate(GEN5, source) == []
    assert validate(GEN7, source) == []
    assert not GEN5.redirects and not GEN5.hooks and not GEN5.patches


def test_module_closure_is_jax_only_and_closed(source):
    modules = set(resolve_modules(GEN5, source))
    assert all(path.startswith(SOURCE_PREFIX + "/") for path in modules)
    for rel in ("__init__.py", "combat/__init__.py", "world/__init__.py", "crafting/contract.py"):
        assert source_path(rel) in modules
    by_name = {module_name(p): p for p in modules}
    for path in modules:
        for target in imported_modules(source, path):
            if target.startswith("hytalegym.jax"):
                assert target in by_name, f"{path} imports {target}, missing from the closure"


def test_gen5_render_is_relocation_only(gen5, source):
    out, manifest = gen5
    package = manifest["package"]
    assert manifest["source"]["commit"] == source.commit
    assert manifest["totals"]["applied"] == 0 and manifest["totals"]["pending"] == 0
    assert manifest["totals"]["files_changed_beyond_relocation"] == 0
    for entry in manifest["files"]:
        written = (out / entry["path"]).read_bytes()
        head = source.read_bytes(entry["source"])
        assert entry["sha256_before"] == _sha(head)
        assert entry["sha256_after"] == _sha(written)
        if entry["changes"] == ["data"]:
            assert written == head
            continue
        restored = written.decode().replace(package, "hytalegym.jax").splitlines()
        original = head.decode().splitlines()
        differing = [i for i, (a, b) in enumerate(zip(restored, original)) if a != b]
        assert len(restored) == len(original)
        if entry["path"] == TOKENS:
            assert len(differing) == 1 and "parents[5]" in original[differing[0]]
            assert 'Path(__import__("hytalegym.envs", fromlist=["envs"]).__file__).resolve().parents[3]' in restored[differing[0]]
        else:
            assert differing == [], entry["path"]


def test_render_is_deterministic(source, tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    R.render(GEN5, first, source=source)
    R.render(GEN5, second, source=source)
    files = sorted(p.relative_to(first) for p in first.rglob("*") if p.is_file())
    assert files == sorted(p.relative_to(second) for p in second.rglob("*") if p.is_file())
    assert all((first / f).read_bytes() == (second / f).read_bytes() for f in files)


def test_relocation_rules():
    package = "hytalegym.sim.genX.engine"
    text, changes = R.relocate(
        "import hytalegym.jax_cache_bootstrap\n"
        "from hytalegym.jax.combat import types\n"
        "NAME = 'hytalegym.jax.combat.observation.v3'\n"
        "ROOT = Path(__file__).resolve().parents[5]\n"
        "LOCAL = Path(__file__).resolve().parents[1]\n",
        TOKENS,
        package,
    )
    assert "import hytalegym.jax_cache_bootstrap" in text
    assert f"from {package}.combat import types" in text
    assert f"'{package}.combat.observation.v3'" in text
    assert 'Path(__import__("hytalegym.envs", fromlist=["envs"]).__file__).resolve().parents[3]' in text
    assert "LOCAL = Path(__file__).resolve().parents[1]" in text
    assert changes[0] == "relocate:refs 2"
    with pytest.raises(R.RenderError, match="not relocatable"):
        R.relocate("from hytalegym import jax\n", "combat/x.py", package)
    with pytest.raises(R.RenderError, match="unhandled __file__"):
        R.relocate("import os\nHERE = os.path.dirname(__file__)\n", "combat/x.py", package)


def test_a_patch_that_does_not_match_once_refuses_and_writes_nothing(source, tmp_path):
    out = tmp_path / "refused"
    missing = replace(GEN5, patches=(Patch("absent", "combat/types.py", "NOT IN HEAD\n", "x\n", "0", "none"),))
    with pytest.raises(OverlayError, match="occurs 0 times"):
        R.render(missing, out, source=source)
    twice = replace(GEN5, patches=(Patch(
        "ambiguous", "combat/arsenal/profiles/hytale_0_5_7.py",
        "                force=(0, 1, -1, 7),\n", "                force=(0, 1, -1, 8),\n", "0", "none"),))
    with pytest.raises(OverlayError, match="occurs 2 times"):
        R.render(twice, out, source=source)
    outside = replace(GEN5, patches=(Patch("outside", "training/ppo.py", "import", "import", "0", "none"),))
    assert any("not a rendered module" in p for p in validate(outside, source))
    assert not out.exists()


def test_refuses_to_replace_a_directory_that_is_not_a_render(source, tmp_path):
    out = tmp_path / "occupied"
    out.mkdir()
    (out / "notes.txt").write_text("keep me")
    with pytest.raises(R.RenderError, match="not a previous render"):
        R.render(GEN5, out, source=source)
    assert (out / "notes.txt").read_text() == "keep me"


class _StubTables:
    content_sha256 = "stub"

    def effective_row(self, family, key):
        assert (family, key) == ("stats", "stat:Stamina")
        return {"max": 12.5}


def test_redirects_and_hooks_render_as_declared(source, tmp_path):
    overlay = replace(
        GEN5,
        redirects=(
            Redirect("stamina", "combat/mechanics/schema/contract.py", "STAMINA_MAXIMUM",
                     "stats", "stat:Stamina", "max", cast="float"),
            Redirect("loaded_dt", "combat/types.py", "default_combat_params", "profile", "",
                     "step_duration.loaded_seconds", cast="float",
                     expression='engine["loaded_delta_seconds"]'),
            Redirect("ticks", "combat/arsenal/programs/timing.py", "NATIVE_TICKS_PER_SECOND", "profile", "",
                     "step_duration.nominal_seconds", cast="reciprocal_int"),
            Redirect("later_redirect", "combat/types.py", "PLAYER_BASE_SPEED", "movement", "config:Default",
                     "fields.BaseSpeed", status="todo"),
        ),
        hooks=(
            Hook("los", "world/geometry/__init__.py", "line_of_sight_result",
                 "tests_fake.constructs:construct", "line_of_sight", status="active"),
            Hook("later_hook", "combat/mechanics/factory.py", "default_mechanics_rules",
                 "tests_fake.constructs:construct", "mana_pool"),
        ),
    )
    out = tmp_path / "steps"
    manifest = R.render(overlay, out, source=source, content_loader=lambda profile: _StubTables())
    contract = (out / "combat/mechanics/schema/contract.py").read_text()
    assert "STAMINA_MAXIMUM = 12.5\n" in contract
    assert "# render redirect stamina: stats/stat:Stamina#max" in contract
    types_text = (out / "combat/types.py").read_text()
    assert "loaded_dt=jnp.asarray(0.045, dtype=f32)" in types_text  # the gen-5 profile's value
    assert "NATIVE_TICKS_PER_SECOND = 30\n" in (out / "combat/arsenal/programs/timing.py").read_text()
    geometry = (out / "world/geometry/__init__.py").read_text()
    hook_line = "line_of_sight_result = _render_hook_los(line_of_sight_result, _render_profile_los)"
    assert hook_line in geometry
    assert geometry.index("def line_of_sight_result(") < geometry.index(hook_line) < geometry.index("def line_of_sight(")
    assert [(a["kind"], a["id"]) for a in manifest["applied"]] == [
        ("redirect", "ticks"), ("redirect", "stamina"), ("redirect", "loaded_dt"), ("hook", "los")]
    bad = replace(overlay, redirects=(Redirect("bad", "combat/types.py", "default_combat_params", "profile", "",
                                               "step_duration.no_such_field", expression="engine"),))
    assert any("profile has no" in problem for problem in validate(bad, source))
    assert {(p["kind"], p["id"]) for p in manifest["pending"]} == {("redirect", "later_redirect"), ("hook", "later_hook")}
    assert manifest["content_sha256"] == "stub"


def test_gen7_overlay_applies_its_patches_and_step_clock(gen7):
    out, manifest = gen7
    applied = {item["id"]: item for item in manifest["applied"]}
    assert {i for i, a in applied.items() if a["kind"] == "patch"} == {p.id for p in GEN7.patches}
    assert all(applied[p.id]["fidelity_row"] and applied[p.id]["differential"] for p in GEN7.patches)
    from hytalegym.sim.gen7.overlay import STEP_CLOCK

    clock = [r.id for r in STEP_CLOCK]
    assert len(clock) == 13 and all(applied[i]["kind"] == "redirect" for i in clock)
    assert {(p["kind"], p["status"]) for p in manifest["pending"]} == {("redirect", "todo"), ("hook", "stub")}
    profile = (out / "combat/arsenal/profiles/hytale_0_5_7.py").read_text()
    assert "    push = (0.0, 0.2, -1.0, 4.3)\n" in profile
    assert "force=(0, 0.2, -1, 5.0)," in profile and "force=(0, 0.2, -2, 10.8)," in profile
    assert profile.count("_guard(guard_stamina_value, interrupting_type_mask=0),") == 4
    assert "0.05 + 0.03333333333333333," in profile and "1.0 / 30.0" not in profile
    types_text = (out / "combat/types.py").read_text()
    assert "loaded_dt=jnp.asarray(0.03333333333333333, dtype=f32)" in types_text
    assert "PLAYER_MAX_SLOPE_ANGLE_DEGREES = 30.0" in types_text
    clock = (out / "combat/clock.py").read_text()
    assert "HYTALE_TICKS_PER_SECOND = operator.index(\n    # render redirect step_clock:combat/clock.py:19" in clock
    assert '["ticks_per_second"]' not in clock
    assert "NATIVE_TICKS_PER_SECOND = 30\n" in (out / "combat/arsenal/programs/timing.py").read_text()
    assert "RECIPE_TABLE_CAPACITY = 2047\n" in (out / "crafting/contract.py").read_text()
    assert manifest["profile"]["step_duration"] == {
        "nominal_seconds": 1.0 / 30.0, "loaded_seconds": 1.0 / 30.0, "timing_profile_count": 1}


def test_gen7_content_values_render_the_072_table_values(gen7):
    from hytalegym.sim.common.content import adapters, load
    from hytalegym.sim.gen7.overlay import CONTENT_VALUES
    from hytalegym.sim.gen7.profile import PROFILE as GEN7_PROFILE

    out, manifest = gen7
    applied = {item["id"]: item for item in manifest["applied"]}
    pending = {item["id"] for item in manifest["pending"] if item["kind"] == "redirect"}
    assert len(CONTENT_VALUES) == 13 and not {r.id for r in CONTENT_VALUES} & pending
    assert all(applied[r.id]["kind"] == "redirect" and applied[r.id]["fidelity_row"] == r.fidelity_row
               for r in CONTENT_VALUES)
    assert {r.fidelity_row for r in CONTENT_VALUES} == {"5f", "6a", "6c"}
    assert all("differential" in r.source for r in CONTENT_VALUES)
    values = {r.id: applied[r.id]["value"] for r in CONTENT_VALUES}
    assert values["sword_selectors:_SWORD_SWING_LEFT_SELECTOR"] == 0.048
    assert values["sword_selectors:_SWORD_SWING_RIGHT_SELECTOR"] == 0.048
    assert values["sword_selectors:_SWORD_SWING_DOWN_SELECTOR"] == 0.056
    assert values["sword_selectors:_SWORD_VORTEX_SPIN_SELECTOR"] == 0.13
    assert values["sword_selectors:_SWORD_THRUST_SELECTOR"] == 0.267
    assert values["sword_selectors:_SWORD_VORTEX_STAB_SELECTOR"] == 0.267
    assert values["melee_hold_constants:_NESTED_OUTER_GATE_SECONDS"] == 0.174
    assert values["melee_hold_constants:_SWORD_THRUST_HOLD_SECONDS"] == 0.739
    assert values["melee_hold_constants:_DOWNSTRIKE_HOLD_SECONDS"] == 0.746
    assert values["melee_hold_constants:_MACE_CHARGED_HOLD_SECONDS"] == 1.841
    assert values["iron_template_damage:_iron_shortbow defaults"] == [1.0, 2.0, 4.0, 7.0, 19.0, 14.0]
    assert values["shortbow_tiers:shortbow draw ladder launch speeds"] == [12.0, 22.0, 35.0, 50.0, 85.0]
    assert values["iron_template_damage:_iron_crossbow defaults"] == 13.0
    # The rendered values are the 0.7.2 table values (the redirects read the table, not these numbers).
    tables = load(GEN7_PROFILE)
    assert adapters.melee_hold_constants(tables)["_DOWNSTRIKE_HOLD_SECONDS"] == 0.746
    assert adapters.shortbow_tiers(tables)["damage"][:4] == (1.0, 2.0, 4.0, 7.0)
    assert adapters.shortbow_tiers(tables)["speeds"] == (12.0, 22.0, 35.0, 50.0, 85.0)
    assert tables.effective_row("weapons", "item:Weapon_Crossbow_Iron")["damage"]["Combo_Projectile_Damage"]["base_first"] == 13.0
    profile = (out / "combat/arsenal/profiles/hytale_0_5_7.py").read_text()
    for literal in (
        "    0.048, 0.1, 2.75, 0.5, 0.5, 90.0, 1.0, -60.0, 0.0, 30.0\n",
        "    0.048, 0.1, 2.75, 0.5, 0.5, 90.0, -1.0, -60.0, 0.0, 30.0\n",
        "    0.056, 0.1, 2.75, 0.4, 0.4, 90.0, 1.0, -60.0, 0.0, 90.0\n",
        "    0.13, 0.1, 3.0, 1.0, 0.5, 360.0, -1.0, -90.0, 0.0, 0.0\n",
        "_NESTED_OUTER_GATE_SECONDS = 0.174\n",
        "_SWORD_THRUST_HOLD_SECONDS = 0.739\n",
        "_DOWNSTRIKE_HOLD_SECONDS = 0.746\n",
        "_MACE_CHARGED_HOLD_SECONDS = 1.841\n",
        "    damage: tuple[float, ...] = (1.0, 2.0, 4.0, 7.0, 19.0, 14.0),\n",
        "            (12.0, 22.0, 35.0, 50.0, 85.0),\n",
        "    damage: tuple[float, ...] = (10, 13, 78),\n",
    ):
        assert profile.count(literal) == 1, literal
    assert profile.count("    runtime_seconds=0.267,\n") == 2
    for gen5_literal in ("0.05, 0.1, 2.75, 0.5, 0.5, 90.0, 1.0", "0.067, 0.1", "    0.15, 0.1, 3.0, 1.0, 0.5, 360.0, -1.0, -90.0, 0.0, 0.0\n", "runtime_seconds=0.3,",
                         "_SWORD_THRUST_HOLD_SECONDS = 0.85\n", "_MACE_CHARGED_HOLD_SECONDS = 2.2\n",
                         "(6, 9, 12, 16, 19, 14)", "(15, 30, 50, 70, 85),", "(10, 27, 78)"):
        assert gen5_literal not in profile, gen5_literal
    # Not applied (see the overlay's comment): the sword thrust hold speed stays 0.8 and the potion row stays pending.
    assert "hold_speed_multiplier=0.8," in profile
    assert "potions:_potions values" in pending


def test_step_clock_rows_are_the_swap_plan_source_sites():
    from hytalegym.sim.gen7.overlay import STEP_CLOCK, STEP_CLOCK_SITES

    assert STEP_CLOCK_SITES == {"sites": 369, "in_source": 321, "redirected": 13, "consumers": 308,
                                "outside_source": 48}
    assert len(STEP_CLOCK) == 13 and all(r.line and r.family == "profile" for r in STEP_CLOCK)
    assert {r.field for r in STEP_CLOCK} == {"step_duration.nominal_seconds", "step_duration.loaded_seconds",
                                             "step_duration.timing_profile_count"}


def test_swap_plan_rows_render_the_gen5_values_through_the_adapters(source, tmp_path):
    import ast

    from hytalegym.sim.gen7.overlay import CONTENT_LOCOMOTION, CONTENT_TODO

    wanted = ("player_locomotion:PLAYER_JUMP_FORCE", "mechanics_resource_constants:STAMINA_MAXIMUM",
              "dodge_constants:DODGE_FORCE", "crafting_capacities:RECIPE_TABLE_CAPACITY")
    rows = tuple(replace(r, status="active") for r in CONTENT_LOCOMOTION + CONTENT_TODO if r.id in wanted)
    assert len(rows) == len(wanted)
    out = tmp_path / "adapters"
    manifest = R.render(replace(GEN5, redirects=rows), out, source=source)
    assert len(manifest["applied"]) == len(wanted) and manifest["content_sha256"]

    def value(text, symbol):
        for node in ast.parse(text).body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == symbol for t in node.targets):
                return ast.literal_eval(node.value)
        raise AssertionError(symbol)

    for row in rows:
        head = source.read(source_path(row.path))
        rendered = (out / row.path).read_text()
        assert value(rendered, row.symbol) == value(head, row.symbol), row.id


def test_redirect_imports_must_run_before_hytalegym(gen5):
    out, _ = gen5
    with pytest.raises(R.RenderError, match="before hytalegym"):
        R.redirect_imports(out)


def test_bench_compare_outputs_is_bit_exact():
    left = {"rollout": {"x": np.arange(6, dtype=np.float32), "n": np.int32(3)}}
    same = {"rollout": {"x": np.arange(6, dtype=np.float32), "n": np.int32(3)}}
    summary = bench.compare_outputs(left, same)
    assert summary["arrays"] == 2 and summary["mismatched_arrays"] == 0
    assert summary["left_sha256"] == summary["right_sha256"]
    nudged = np.arange(6, dtype=np.float32)
    nudged[4] = np.nextafter(nudged[4], np.float32(10))
    summary = bench.compare_outputs(left, {"rollout": {"x": nudged, "n": np.int32(3)}})
    assert summary["mismatched_arrays"] == 1 and 0 < summary["max_abs_difference"] < 1e-5
    assert "['x']" in summary["first_mismatch"]


def test_bench_attributes_hlo_instructions_to_source_lines():
    text = (
        "HloModule jit_tick\n\nFileNames\n1 \"/r/hytalegym/sim/gen5/engine/combat/arsenal/runtime.py\"\n\n"
        "FunctionNames\n1 \"_resolve_damage\"\n\nFileLocations\n"
        "1 {file_name_id=1 function_name_id=1 line=2501 end_line=2501 column=1 end_column=9}\n\n"
        "StackFrames\n1 {file_location_id=1 parent_frame_id=1}\n\n"
        "ENTRY %main {\n  %sort.0 = f32[4]{0} sort(%a), metadata={op_name=\"jit(tick)/sort\" stack_frame_id=1}\n}\n"
    )
    sources = bench._hlo_sources(text)
    assert sources["sort.0"][1:3] == ("_resolve_damage", 2501)
    assert bench._engine_relative(sources["sort.0"][0]) == "combat/arsenal/runtime.py"


def test_bench_table_rows_cover_the_brief():
    rows = bench._rows({"original": {}, "rendered": {}})
    labels = " | ".join(label for _, label, _, _ in rows)
    for needle in ("batch 8", "batch 64", "compile", "p95, 1 NPC", "p95, 8 NPCs", "real-time factor",
                   "Region library load", "path table"):
        assert needle in labels
    assert json.dumps(rows)
    assert bench.TICK_BUDGET_MS == pytest.approx(33.333, abs=1e-3) and bench.P95_TARGET_MS == 15.0


LOCOMOTION_SYMBOLS = (
    "PLAYER_JUMP_FORCE", "PLAYER_VARIABLE_JUMP_FALL_FORCE", "PLAYER_BASE_SPEED",
    "PLAYER_GAIT_SPEED_MULTIPLIERS", "PLAYER_GAIT_FORWARD_SPEED_MULTIPLIERS",
    "PLAYER_GAIT_BACKWARD_SPEED_MULTIPLIERS", "PLAYER_GAIT_STRAFE_SPEED_MULTIPLIERS",
    "PLAYER_AIR_DRAG_MIN", "PLAYER_AIR_DRAG_MAX", "PLAYER_AIR_DRAG_MIN_SPEED", "PLAYER_AIR_DRAG_MAX_SPEED",
    "PLAYER_AIR_CONTROL_MIN_SPEED", "PLAYER_AIR_CONTROL_MAX_SPEED",
    "PLAYER_AIR_CONTROL_MIN_MULTIPLIER", "PLAYER_AIR_CONTROL_MAX_MULTIPLIER",
)


def _module_value(text, symbol):
    import ast

    nodes = [n for n in ast.parse(text).body if isinstance(n, ast.Assign)
             and [getattr(t, "id", None) for t in n.targets] == [symbol]]
    assert len(nodes) == 1, (symbol, len(nodes))
    return ast.literal_eval(nodes[0].value)


def test_locomotion_rows_are_active_and_the_two_tables_agree_on_them():
    from hytalegym.sim.common.content import adapters, load
    from hytalegym.sim.gen5.profile import PROFILE as GEN5_PROFILE
    from hytalegym.sim.gen7.overlay import CONTENT_LOCOMOTION, CONTENT_TODO
    from hytalegym.sim.gen7.profile import PROFILE as GEN7_PROFILE

    names = {"player_locomotion", "walk_steering_sprint_multiplier"}
    assert len(CONTENT_LOCOMOTION) == 16
    assert all(r.status == "active" and r.row in names for r in CONTENT_LOCOMOTION)
    assert not any(r.row in names for r in CONTENT_TODO)
    sprint = next(r for r in CONTENT_LOCOMOTION if r.row == "walk_steering_sprint_multiplier")
    assert (sprint.line, sprint.expression) == (177, "1.273")
    # 0.7.2 and the gen-5 table (a 0.5.10 extraction) agree on every value these rows read.
    gen5, gen7 = load(GEN5_PROFILE), load(GEN7_PROFILE)
    assert adapters.player_locomotion(gen5) == adapters.player_locomotion(gen7)
    assert adapters.walk_steering_sprint_multiplier(gen5) == adapters.walk_steering_sprint_multiplier(gen7) == 1.273


def test_gen7_render_keeps_the_committed_locomotion_literals(gen7, source):
    from hytalegym.sim.common.overlay import SOURCE_PREFIX

    out, _ = gen7
    head = source.read(SOURCE_PREFIX + "/combat/types.py")
    rendered = (out / "combat/types.py").read_text()
    for symbol in LOCOMOTION_SYMBOLS:
        expected, actual = _module_value(head, symbol), _module_value(rendered, symbol)
        assert type(actual) is type(expected) and actual == expected, symbol
    loco = (out / "world/entities/locomotion.py").read_text()
    assert '"installed_default_forward_multiplier": 1.273,' in loco
    assert '"1.273":' not in loco and "render redirect walk_steering_sprint_multiplier" in loco


def test_gen7_stamina_drink_has_no_status_and_the_072_cooldown(gen7):
    out, manifest = gen7
    applied = {item["id"]: item for item in manifest["applied"]}
    assert applied["potion_stamina_072_drink"]["kind"] == "patch"
    assert applied["potion_stamina_072_drink"]["fidelity_row"] == "8e"
    profile = (out / "combat/arsenal/profiles/hytale_0_5_7.py").read_text()
    start = profile.index('        _ability(\n            "Potion_Stamina",\n')
    block = profile[start:profile.index("\n        ),\n", start)]
    assert '"Potion_Stamina_Cooldown"' not in block and "cooldown=15" not in block  # the comment may name it
    assert '"Potion_Stamina_Cooldown"' not in profile
    assert "cooldown=30," in block
    assert "_resource(0.5, RESOURCE_STAMINA, 45, percent=True)," in block


def test_gen7_durability_hook_wraps_the_catalog_loader(gen5, gen7):
    gen5_out, _ = gen5
    out, manifest = gen7
    applied = {item["id"]: item for item in manifest["applied"]}
    assert applied["weapon_durability_catalog"]["kind"] == "hook"
    assert applied["weapon_durability_catalog"]["fidelity_row"] == "8d"
    text = (out / "combat/arsenal/item_programs.py").read_text()
    assert "from hytalegym.sim.gen7.mechanics.durability import construct as _render_hook_weapon_durability_catalog" in text
    assert ("hytale_0_5_7_item_program_catalog = _render_hook_weapon_durability_catalog("
            "hytale_0_5_7_item_program_catalog, _render_profile_weapon_durability_catalog)") in text
    assert "_render_hook_weapon_durability_catalog" not in (gen5_out / "combat/arsenal/item_programs.py").read_text()
