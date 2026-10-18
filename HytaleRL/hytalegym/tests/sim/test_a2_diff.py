"""A2: the gen-5 -> gen-7 table diff reproduces the measured 0.7.2 changes.

Sources of the expected values: TRAINING-0.7.2.md (gameplay changes 1-10),
FIDELITY-0.7.2.md (rows 1d, 3b, 5a-5d, 6a-6c, 7a, 8a, 8d, 9, 11) and
skill_agent/docs/BUILD.md section 4. The diff tool is exercised on the real data.
"""

from __future__ import annotations

import pytest

from hytalegym.sim.common.content import adapters as A
from hytalegym.sim.common.content import load

from . import support
from content_extract.diff import diff_directories


@pytest.fixture(scope="module")
def diff():
    return diff_directories(support.data_dir("gen5"), support.data_dir("gen7"))["families"]


@pytest.fixture(scope="module")
def gen5():
    return load("0.5.7", apply_overlay=False)


@pytest.fixture(scope="module")
def gen7():
    return load("0.7.2")


def _knockback(tables, attack):
    node = tables.interaction(f"Weapon_Sword_Primary_{attack}_Damage")
    return node.knockback.direction, node.knockback.force


def test_sword_knockback_and_interrupt(diff, gen5, gen7) -> None:
    assert _knockback(gen5, "Swing_Left") == ((0.0, 1.0, -1.0), 6.0)
    assert _knockback(gen7, "Swing_Left") == ((0.0, 0.2, -1.0), 4.3)
    assert _knockback(gen5, "Swing_Right") == ((0.0, 1.0, -1.0), 7.0)
    assert _knockback(gen7, "Swing_Right") == ((0.0, 0.2, -1.0), 5.0)
    assert _knockback(gen5, "Swing_Down") == ((0.0, 1.0, -2.0), 12.0)
    assert _knockback(gen7, "Swing_Down") == ((0.0, 0.2, -2.0), 10.8)
    changed = diff["weapons"]["changed"]["interaction:Weapon_Sword_Primary_Swing_Left_Damage"]
    assert changed["knockback.force"] == [6.0, 4.3]
    assert changed["hit_interrupt"] == [False, True]


def test_hit_interrupt_flags_on_31_attack_files(gen5, gen7) -> None:
    def flagged(tables):
        return [
            row["source"]
            for key, row in tables.table("weapons").rows.items()
            if key.startswith("interaction:") and row.get("hit_interrupt")
        ]

    assert flagged(gen5) == []
    files = flagged(gen7)
    folders: dict[str, int] = {}
    for path in files:
        folder = "/".join(path.split("/")[3:5])
        folders[folder] = folders.get(folder, 0) + 1
    assert len(files) == 31
    assert folders == {
        "Weapons/Daggers": 9,
        "Weapons/Mace": 7,
        "Weapons/Sword": 6,
        "Weapons/Battleaxe": 5,
        "Abilities/Fireball": 1,
        "Abilities/GroundSlam": 2,
        "Abilities/WindStrike": 1,
    }
    assert gen7.effect("Hit_Interrupt_Flinch").duration == 0.5
    assert gen7.effect("Hit_Interrupt_Flinch").document["ApplicationEffects"]["HorizontalSpeedMultiplier"] == 0
    assert gen7.effect("Hit_Interrupt_Stunned").duration == 0.7
    assert gen7.effect("Hit_Interrupt_Enraged").duration == 8


def _damage(tables, item, var):
    return tables.weapon_item(item).damage[var].base_first


def test_daggers_plus_50_percent_sword_unchanged(gen5, gen7) -> None:
    for item, old, new in (("Weapon_Daggers_Iron", 4.0, 6.0), ("Weapon_Daggers_Mithril", 8.0, 12.0)):
        for var in ("Swing_Left_Damage", "Swing_Right_Damage"):
            assert (_damage(gen5, item, var), _damage(gen7, item, var)) == (old, new)
    assert _damage(gen7, "Weapon_Sword_Iron", "Swing_Left_Damage") == _damage(gen5, "Weapon_Sword_Iron", "Swing_Left_Damage") == 10.0
    assert _damage(gen7, "Weapon_Spear_Leaf", "Spear_Stab_Damage") == _damage(gen5, "Weapon_Spear_Leaf", "Spear_Stab_Damage")


def test_ranged_damage_cut_and_launch(gen5, gen7) -> None:
    tiers = [f"Primary_Shoot_Damage_Strength_{n}" for n in range(4)]
    assert [_damage(gen5, "Weapon_Shortbow_Iron", v) for v in tiers] == [6, 9, 12, 16]
    assert [_damage(gen7, "Weapon_Shortbow_Iron", v) for v in tiers] == [1, 2, 4, 7]
    assert [_damage(gen5, "Weapon_Shortbow_Mithril", v) for v in tiers] == [11, 17, 23, 31]
    assert [_damage(gen7, "Weapon_Shortbow_Mithril", v) for v in tiers] == [2, 4, 7, 13]
    steel = "Weapon_Crossbow_Ancient_Steel"
    assert (_damage(gen5, steel, "Combo_Projectile_Damage"), _damage(gen7, steel, "Combo_Projectile_Damage")) == (42, 14)
    assert (_damage(gen5, steel, "Signature_BigArrow_Damage"), _damage(gen7, steel, "Signature_BigArrow_Damage")) == (123, 82)
    assert _damage(gen7, "Weapon_Crossbow_Iron", "Combo_Projectile_Damage") == 13
    assert A.shortbow_tiers(gen5)["speeds"] == (15, 30, 50, 70, 85)
    assert A.shortbow_tiers(gen7)["speeds"] == (12, 22, 35, 50, 85)
    assert A.shortbow_tiers(gen7)["gravities"] == A.shortbow_tiers(gen5)["gravities"] == (45, 40, 32, 28, 25)
    for config, launch, gravity in (("Projectile_Config_Arrow_Crossbow", 40, 10), ("Projectile_Config_Arrow_Crossbow_Signature", 100, 40)):
        for tables in (gen5, gen7):
            row = tables.projectile_config(config)
            assert (row.launch_force, row.gravity) == (launch, gravity)


def test_mana_exists(gen5, gen7) -> None:
    old, new = gen5.stat("Mana"), gen7.stat("Mana")
    assert (old.initial_value, old.maximum, new.initial_value, new.maximum) == (0, 0, 100, 100)
    assert (old.regenerating[0].interval, new.regenerating[0].interval) == (0.2, 0.3)
    assert new.regenerating[0].amount == 1 and new.regenerating[0].condition_ids() == (("Alive", False),)
    ability = gen7.row("weapons", "item:Rune_Fireball")["ability"]
    assert (ability["Cost"], ability["CostType"], ability["Cooldown"]) == (30, "Mana", 2)


def test_durability_changes(diff, gen5, gen7) -> None:
    sword = (gen5.durability("Weapon_Sword_Iron"), gen7.durability("Weapon_Sword_Iron"))
    assert (sword[0].max_durability, sword[0].durability_loss_on_hit) == (120.0, 0.21)
    assert (sword[1].max_durability, sword[1].durability_loss_on_hit) == (250.0, 0.13)
    from . import gen5_live as live

    ids = [row["item_asset_id"] for row in live.compiled_weapon_durability()]
    pairs = [(gen5.durability(i), gen7.durability(i)) for i in ids]
    assert len(pairs) == 143 and all(new is not None for _, new in pairs)
    assert sum(old.max_durability != new.max_durability for old, new in pairs) == 57
    assert sum(old.durability_loss_on_hit != new.durability_loss_on_hit for old, new in pairs) == 82


def test_world_content_changes(gen5, gen7) -> None:
    old, new = gen5.block("Deco_SpiderWeb").movement_settings, gen7.block("Deco_SpiderWeb").movement_settings
    assert (old["HorizontalSpeedMultiplier"], new["HorizontalSpeedMultiplier"]) == (0.65, 0.45)
    assert (old["JumpForceMultiplier"], new["JumpForceMultiplier"]) == (0.5, 0.45)
    assert (old["Drag"], new["Drag"]) == (0.45, 0.82)
    climbable = [sum(t.block(k.split(":", 1)[1]).climbable for k in t.keys("blocks", "block")) for t in (gen5, gen7)]
    assert climbable == [52, 57]  # resolved block types (files authoring IsClimbable: 41 -> 44)
    m5, m7 = gen5.movement_config(), gen7.movement_config()
    assert "MaxSlopeAngleDegrees" not in m5.fields
    assert (m7.number("MaxSlopeAngleDegrees"), m7.number("MaxWallAngleDegrees")) == (30.0, 15.0)
    for key in ("JumpForce", "BaseSpeed", "ForwardSprintSpeedMultiplier"):
        assert m5.number(key) == m7.number(key)


def test_crafting_catalogue(gen5, gen7) -> None:
    assert (len(gen5.recipes()), len(gen7.recipes())) == (1947, 2047)
    benches = {b.bench_id for r in gen7.recipes() for b in r.bench_requirements}
    assert len(benches) == 22
    assert sum(not r.bench_requirements for r in gen7.recipes()) == 19


def test_npc_melee_and_roles_unchanged_paths_moved(diff, gen5, gen7) -> None:
    for role in ("Trork_Brawler", "Kweebec_Razorleaf"):
        a, b = gen5.role(role), gen7.role(role)
        assert (a.max_health, a.knockback_scale, a.walk_controller()) == (b.max_health, b.knockback_scale, b.walk_controller())
        assert "/Faction/" in b.source and "/Faction/" not in a.source
    assert A.npc_matchup(gen5) == A.npc_matchup(gen7)
    assert "role:Trork_Brawler" in diff["npc_roles"]["moved"] or "role:Trork_Brawler" in diff["npc_roles"]["changed"]


def test_guard_roots_stop_interrupting_primary(gen5, gen7) -> None:
    for weapon in ("Sword", "Battleaxe", "Daggers", "Mace"):
        key = f"root:Root_Weapon_{weapon}_Secondary_Guard"
        assert gen5.row("weapons", key)["document"]["Rules"] == {"Interrupting": ["Primary"]}
        assert "Rules" not in gen7.row("weapons", key)["document"]


def test_dodge_unchanged(gen5, gen7) -> None:
    assert A.dodge_constants(gen5) == A.dodge_constants(gen7)
    for key in ("interaction:Dodge", "interaction:Dodge_Left", "interaction:Dodge_Right", "effect:Dodge_Invulnerability"):
        assert gen5.row("weapons", key)["document"] == gen7.row("weapons", key)["document"]


def test_stamina_gains_use_ignore_flag(diff) -> None:
    changed = diff["stats"]["changed"]["stat:Stamina"]
    assert any(path.endswith("UseIgnoreFlag") and values == [None, True] for path, values in changed.items())
