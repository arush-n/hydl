"""Gen-7 overlay (Hytale 0.7.2): a skeleton. Rendering it does NOT make the
simulator faithful to 0.7.2; every change below is gated by a differential that
has not run yet (``HytaleRL/hytale-plugin-0.7.2/FIDELITY-0.7.2.md`` is the
ledger, table (a) rows cited per change).

* the active redirects: the step clock. Owner ruling (2026-10-17): the agent
  operates at 30 Hz, the game's tick. The value is the profile field
  ``SimProfile.step_duration`` (gen 7: nominal = loaded = 1/30 s, one timing
  profile; gen 5 keeps 1/30 s, 0.045 s and three per-lane profiles). The rows
  are generated from the swap plan's ``step_duration`` sites
  (``overlay.step_clock_redirects``): the 13 sites in the rendered source that
  take the clock from a constant (reads of the ruleset ``engine`` block, a
  literal tick rate, one-tick literals); the other 301 sites read one of them
  and follow. Only ``loaded_delta_seconds`` (0.045 -> 1/30) and the
  timing-profile count (3 -> 1) change value; the rest are value-identical.
* ``status="todo"`` redirects: the swap plan's 90 records
  (``sim/common/content/redirects.json``, owned by the content agent), loaded
  as rows valued by their adapter; the records a render cannot reach (JSON
  tables, shared modules) are counted in ``audit``. Listed, not applied.
* ``patches``: the items of FIDELITY section (e) that are pure constant or rule
  changes with an unambiguous ``old`` text at HEAD.
* ``hooks``: named construct insertion points, stubs until the constructs in
  ``sim/gen7/mechanics/`` exist and their differentials are named.

Not patched, and why: the dodge head size (a remap across the action surface,
not a constant: hook ``dodge_action_head``); shield bash ``StaminaRegenDelay``
Set -1.5 -> Min -1 (the engine models only Set; needs a construct); Mithril and
other dagger materials (rows of ``hytalegym/combat/assets/weapon_variants``, a
shared module outside the rendered package); the sword thrust's HorizontalSpeedMultiplier
0.8 -> 0.75 and its chained RunTime 0.043 (``Next/0.565/Interactions/2``: no swap-plan site;
leads under row 5f, not among the 14 values); the potion drink's status and cooldown (patch
``potion_stamina_072_drink``, row 8e); durability (a JSON table: hook ``weapon_durability_catalog`` on the catalog loader,
row 8d). The sword, shortbow and crossbow values
are the ``CONTENT_VALUES`` rows below.
"""

from __future__ import annotations

from dataclasses import replace

from hytalegym.sim.common.overlay import (
    Redirect,
    Hook,
    Overlay,
    Patch,
    load_swap_plan,
    step_clock_redirects,
    swap_plan_redirects,
)

PROFILE_JAX = "combat/arsenal/profiles/hytale_0_5_7.py"

_SWAP_PLAN = load_swap_plan()
STEP_CLOCK, STEP_CLOCK_SITES = step_clock_redirects(_SWAP_PLAN)
CONTENT_ROWS, UNREACHABLE_RECORDS = swap_plan_redirects(_SWAP_PLAN, status="todo")
#: Locomotion adapters, active from phase 3 (2026-10-18). Their 0.7.2 values equal the
#: 0.5.10 values on every MovementConfig key the two tables share (only MaxSlopeAngleDegrees
#: and MaxWallAngleDegrees are new; A2 and tests/sim/test_locomotion_rows.py), so these rows
#: render the same numbers today and make the table the source for later versions.
LOCOMOTION_ADAPTERS = ("player_locomotion", "walk_steering_sprint_multiplier")

CONTENT_LOCOMOTION = tuple(replace(r, status="active") for r in CONTENT_ROWS if r.row in LOCOMOTION_ADAPTERS)
#: 0.7.2 content values that the gen-5 engine literals still carry (handoff 2026-10-18). Twelve rows
#: replace the swap-plan todo rows of the same id (the same engine site, now valued by the 0.7.2 table),
#: so the value follows the table. The shortbow launch speeds have no swap-plan record (the shortbow
#: record is the charge ladder), so their row is an adapter redirect on the same literal. The renderer
#: checks the HEAD expression once at its site. Checked on both tables by every content adapter
#: (receipts: ~/.claude/jobs/31947bd7/tmp/sim_values/). Not applied here: the potion drink's status and
#: cooldown (patch potion_stamina_072_drink, row 8e; the swap-plan row stays todo), the weapon durability
#: (a JSON table: hook weapon_durability_catalog, row 8d), and the sword thrust chained
#: RunTime 0.043 and HorizontalSpeedMultiplier 0.8 -> 0.75 (leads under row 5f, not among the 14).
_SWORD_TIMING = (
    "0.7.2 selector RunTime of the six sword attack rows (weapons/interaction:<attack>_Selector#document.RunTime; "
    "file-sourced). differential: row 5f, the iron sword left swing hit tick (native 9 in FIDELITY-0.7.2.md "
    "section (l) and in the live run of 2026-10-18; JAX 7 before and after this value on the gen-7 render, "
    "seeds 570057 and 1215520385): the value reaches the engine and does not move the tick, so row 5f stays open"
)
_HOLD_TIMING = (
    "0.7.2 charge_thresholds of the sword, thrust, battleaxe downstrike and mace rows (file-sourced; the adapter's "
    "sum rule, which reproduces the gen-5 literals). differential: row 5f (the sword thrust and mace hold gates); "
    "not measured on this render"
)
_SHORTBOW = (
    "0.7.2 Iron shortbow damage (weapons/item:Weapon_Shortbow_Iron, Primary_Shoot_Damage_Strength_0..3) and launch "
    "force of Projectile_Config_Arrow_Shortbow_Strength_0..4 (file-sourced). differential: row 6a, tier-0 open "
    "contact, seeds 20260729 and 7: native damage 1.0 at hit tick 7 on the live bridge, jar 4A5E5D6D (2026-10-18, "
    "sim_values run2 bridge_after4.json); JAX on this render 6.0 at tick 6 before and 1.0 at tick 7 after; the "
    "other tiers' range error (2 % bar) not measured"
)
_CROSSBOW = (
    "0.7.2 Iron crossbow combo damage (weapons/item:Weapon_Crossbow_Iron damage.Combo_Projectile_Damage; "
    "file-sourced). differential: row 6c, the native combo's applied damage 10, 10, 13 (health 61, 51, 41, 28 at "
    "ticks 100, 111, 122; FIDELITY-0.7.2.md section (u), jar CAFAD9B7); the JAX combo on this render not run"
)


def _sword_runtime(constant: str, interaction: str, expression: str, template: str = "{value}") -> Redirect:
    return Redirect(
        f"sword_selectors:{constant}", PROFILE_JAX, constant, "weapons", f"interaction:{interaction}",
        "document.RunTime", cast="float", status="active", fidelity_row="5f", source=_SWORD_TIMING,
        expression=expression, template=template,
    )


def _hold(constant: str) -> Redirect:
    return Redirect(
        f"melee_hold_constants:{constant}", PROFILE_JAX, constant, "adapter", "melee_hold_constants", constant,
        cast="float", status="active", fidelity_row="5f", source=_HOLD_TIMING,
    )


CONTENT_VALUES = (
    _sword_runtime("_SWORD_SWING_LEFT_SELECTOR", "Weapon_Sword_Primary_Swing_Left_Selector", "0.05"),
    _sword_runtime("_SWORD_SWING_RIGHT_SELECTOR", "Weapon_Sword_Primary_Swing_Right_Selector", "0.05"),
    _sword_runtime("_SWORD_SWING_DOWN_SELECTOR", "Weapon_Sword_Primary_Swing_Down_Selector", "0.067"),
    _sword_runtime("_SWORD_VORTEX_SPIN_SELECTOR", "Weapon_Sword_Signature_Vortexstrike_Spin_Selector", "0.15"),
    _sword_runtime("_SWORD_THRUST_SELECTOR", "Weapon_Sword_Primary_Thrust_Selector", "runtime_seconds=0.3",
                   "runtime_seconds={value}"),
    _sword_runtime("_SWORD_VORTEX_STAB_SELECTOR", "Weapon_Sword_Signature_Vortexstrike_Stab_Selector",
                   "runtime_seconds=0.3", "runtime_seconds={value}"),
    _hold("_NESTED_OUTER_GATE_SECONDS"),
    _hold("_SWORD_THRUST_HOLD_SECONDS"),
    _hold("_DOWNSTRIKE_HOLD_SECONDS"),
    _hold("_MACE_CHARGED_HOLD_SECONDS"),
    Redirect(
        "iron_template_damage:_iron_shortbow defaults", PROFILE_JAX, "_iron_shortbow", "adapter", "shortbow_tiers",
        "damage", cast="tuple_float", status="active", fidelity_row="6a", source=_SHORTBOW,
        expression="(6, 9, 12, 16, 19, 14)",
    ),
    Redirect(
        "shortbow_tiers:shortbow draw ladder launch speeds", PROFILE_JAX, "_iron_shortbow", "adapter",
        "shortbow_tiers", "speeds", cast="tuple_float", status="active", fidelity_row="6a", source=_SHORTBOW,
        expression="(15, 30, 50, 70, 85)",
    ),
    Redirect(
        "iron_template_damage:_iron_crossbow defaults", PROFILE_JAX, "_iron_crossbow", "weapons",
        "item:Weapon_Crossbow_Iron", "damage.Combo_Projectile_Damage.base_first", cast="int", status="active",
        fidelity_row="6c", source=_CROSSBOW, expression="(10, 27, 78)", template="(10, {value}, 78)",
    ),
)
_VALUE_IDS = frozenset(row.id for row in CONTENT_VALUES)
CONTENT_TODO = tuple(r for r in CONTENT_ROWS if r.row not in LOCOMOTION_ADAPTERS and r.id not in _VALUE_IDS)

_SWORD_KNOCKBACK = (
    "knockback displacement differential on 0.7.2 (TRAINING-0.7.2.md: Trork 0.19 m vs 0.44 m on "
    "0.5.10) and the test_jax_native_combat_compare.py sword tests recalibrated to 0.7.2"
)
_A1 = (
    "VelocityType unchanged: the asset authors Add, the engine applies _melee's default FORCE_SET "
    "(A1; gen 5 keeps Set through calibration_057.json); the knockback differential decides the "
    "gen-7 model before this row changes"
)
_GUARD = (
    "tests/fidelity/combat/defense/test_native_guard_attack_cancel.py on the live 0.7.2 bridge: Guard raised while "
    "the attack root runs cancels it (sword left onsets 2-5, sword down 2-12, mace 2-18, battleaxe down-left "
    "2-12), later it does not; plus test_native_shield_bash_resource_cost_matches_and_regen_phase_is_explicit"
)
_JUMP_ARC = (
    "native_motion_differential --jump-steps 1 (ledge fixture, Kweebec_Razorleaf) on the gen-7 render against the "
    "live 0.7.2 bridge, <= 5e-4 m and 1e-3 m/s per step; tests/sim/test_npc_jump_arc.py replays a live jump in "
    "place"
)
_BENCH_CRAFTING = (
    "tests/sim/test_bench_crafting.py (a bench-only recipe commits, the Fieldcraft commit is unchanged, the first "
    "16 of 20 legal recipes are published); the Region crafting count on the 0.7.2 catalog (FIDELITY-0.7.2.md "
    "(l): 2 of 197 offered recipes commit before the construct); a native bench-craft differential is open "
    "(no live bench-craft test on 0.7.2 yet)"
)
_STRUCTURAL = (
    "the 0.7.2 catalog count through a render (903 of the 910 structural recipes admitted, all committing); the live "
    "crafting oracle's structural scenarios (FIDELITY-0.7.2.md (s): Cloth_Block_Wool_Black_Half and _Stairs finish, "
    "a missing input is refused)"
)
_CRAFTING_MATERIALS = (
    "tests/sim/test_crafting_materials.py (admission, resolution, legality from a resource type, the output "
    "exclusion, the Region commit); on the live 0.7.2 catalog 263 recipes are offered and all 263 commit, the 66 "
    "with resource inputs stocked with a non-output item that carries the type (FIDELITY-0.7.2.md (o) follow-up)"
)
_GUARD_NOTE = (
    "0.7.2 removes Rules.Interrupting [Primary] from this family's guard root "
    "(ASSET-DIFF-0.7.2.md:237-243, 266-322); the cancel now comes from the attack roots' InterruptedBy "
    "[Secondary] during their RunTime (hook guard_cancel_window)"
)


def _guard_root(family: str, energy: int) -> Patch:
    old = f"        guard_stamina_value,\n        {{RESOURCE_SIGNATURE_ENERGY: {energy}}},\n    )\n"
    new = (
        f"        _guard(guard_stamina_value, interrupting_type_mask=0),\n"
        f"        {{RESOURCE_SIGNATURE_ENERGY: {energy}}},\n    )\n"
    )
    return Patch(f"guard_{family}_no_primary_interrupt", PROFILE_JAX, old, new, "5d", _GUARD, _GUARD_NOTE)


_POTION = (
    "file-sourced, read from the 0.7.2 Assets.zip on 2026-10-18: Potion_Stamina.json applies only "
    "Potion_Stamina_Instant_Small (Stamina 45 percent, equal to 0.5.10) and its ConsumeCooldown var triggers "
    "Consume_Potion_Stamina_Cooldown.json (TriggerCooldown, Id Potion_Consume_Stamina, Cooldown 30); "
    "Condition_Consume_Potion_Stamina.json refuses a drink on that CooldownCondition. No live drink differential: "
    "no bridge loadout holds a potion and the bridge has no item route into the agent's hotbar"
)


_LANDING_WALK = (
    "native_motion_differential on the gen-7 render against the live 0.7.2 bridge, npc_walk, ledge fixture: "
    "--steps 30 --jump-steps 1 (single jump then forward), --steps 12 --jump-steps 1 --forward-steps 0 (jump in "
    "place) and --steps 30 --jump-steps 0 (walk off the ledge, no jump: must not be held), seeds 570057 and 913771; "
    "bar 5e-4 m and 1e-3 m/s per step; after an accepted jump the walk starts on the third tick after touchdown "
    "(native z first moves on L+3)"
)

_BENCH_TIERS = (
    "tests/sim/test_crafting_oracle_replay.py: the gen-7 render replayed on the live 0.7.2 oracle receipts (FIDELITY-0.7.2.md "
    "section (s); Farmingbench Lettuce refused at tier 1 and accepted at tier 2, Cauliflower accepted at tier 4, Onion refused at "
    "tier 7 and accepted at tier 8, the consumed and output counts equal); tests/sim/test_bench_tiers.py for the row tiers"
)
_MEMORIES = (
    "tests/sim/test_crafting_oracle_replay.py: Furniture_Ancient_Chest_Small refused at memories level 1 and accepted at level 2 "
    "with the receipt's consumed and output counts (FIDELITY-0.7.2.md section (s), receipts repo_mem_*.json); "
    "tests/sim/test_crafting_memories.py for the thresholds 10/25/50/100/200"
)

PATCHES = (
    Patch("sword_swing_left_knockback", PROFILE_JAX,
          "    push = (0.0, 1.0, -1.0, 6.0)\n",
          "    push = (0.0, 0.2, -1.0, 4.3)\n",
          "5b", _SWORD_KNOCKBACK, "Weapon_Sword_Primary_Swing_Left_Damage: Force 6.0 -> 4.3, Dir Y 1 -> 0.2. " + _A1),
    Patch("sword_swing_right_knockback", PROFILE_JAX,
          "                force=(0, 1, -1, 7),\n                on_hit=(RESOURCE_SIGNATURE_ENERGY, 1),\n"
          "                horizontal_selector=_SWORD_SWING_RIGHT_SELECTOR,\n",
          "                force=(0, 0.2, -1, 5.0),\n                on_hit=(RESOURCE_SIGNATURE_ENERGY, 1),\n"
          "                horizontal_selector=_SWORD_SWING_RIGHT_SELECTOR,\n",
          "5b", _SWORD_KNOCKBACK, "Weapon_Sword_Primary_Swing_Right_Damage: Force 7 -> 5.0, Dir Y 1 -> 0.2. " + _A1),
    Patch("sword_swing_down_knockback", PROFILE_JAX,
          "                force=(0, 1, -2, 12),\n",
          "                force=(0, 0.2, -2, 10.8),\n",
          "5b", _SWORD_KNOCKBACK, "Weapon_Sword_Primary_Swing_Down_Damage: Force 12 -> 10.8, Dir Y 1 -> 0.2"),
    Patch("iron_daggers_swing_damage", PROFILE_JAX,
          "    damage: tuple[float, ...] = (4, 4, 8, 12, 25, 31, 37),\n",
          "    damage: tuple[float, ...] = (6, 6, 8, 12, 25, 31, 37),\n",
          "5c",
          "seeded dagger melee contact on 0.7.2 with damage exact (tests/fidelity/combat; native 5.0 vs "
          "JAX 3.0 on bronze daggers in the fidelity run)",
          "Iron template only (Weapon_Daggers_Iron Swing_Left/Right 4 -> 6). Mithril (8 -> 12) and the "
          "other materials are rows of hytalegym/combat/assets/weapon_variants/hytale_0_5_7.py (a shared "
          "module outside the rendered package): their swings come from the hook dagger_swing_damage (5c, phase 4)"),
    _guard_root("sword", 20),
    _guard_root("mace", 8),
    _guard_root("battleaxe", 9),
    _guard_root("daggers", 27),
    Patch("slope_wall_limits_parameters", "combat/types.py",
          "PLAYER_SPRINT_REQUIRES_FORWARD_DOMINANT_TRAVEL = True\n",
          "PLAYER_SPRINT_REQUIRES_FORWARD_DOMINANT_TRAVEL = True\n\n"
          "#: 0.7.2 ``MovementConfig/Default.json`` adds ``MaxSlopeAngleDegrees`` and\n"
          "#: ``MaxWallAngleDegrees`` (absent in 0.5.x; sim/gen7/data/movement.json\n"
          "#: config:Default). No 0.7.2 server class reads them (jar scan, 2026-10-18; the\n"
          "#: keys sit in MovementConfig and the client protocol struct); sim/gen7/SKILL.md.\n"
          "PLAYER_MAX_SLOPE_ANGLE_DEGREES = 30.0\n"
          "PLAYER_MAX_WALL_ANGLE_DEGREES = 15.0\n",
          "1d", "a new stairs/slope bridge fixture and the motion differential (<= 5e-4 m per step)",
          "new parameters only: the 0.7.2 server reads neither key (2026-10-18 scan of 38,436 classes: "
          "MovementConfig and MovementSettings only); decided 2026-10-18: slope_limits False in the gen-7 profile, "
          "matching gen 5 on this field (phase 3b)"),
    Patch("perception_detail_hitbox_los", "world/perception/los.py",
          "            opaque = (\n"
          "                (selected.flags & jnp.int32(FLAG_OPAQUE)) != 0\n"
          "            ) | selected.role_opaque\n"
          "            terminal_here = _reached(cell, terminal)\n",
          "            opaque = (\n"
          "                (selected.flags & jnp.int32(FLAG_OPAQUE)) != 0\n"
          "            ) | selected.role_opaque\n"
          "            # 0.7.2 detail hitboxes (FIDELITY 2b): an opaque cell that carries detail boxes occludes\n"
          "            # only where the ray meets one of them; a cell without boxes keeps the gen-5 rule.\n"
          "            relative_cell = (cell - start_block).astype(jnp.float32)\n"
          "            detail_boxes = selected.collision_boxes + jnp.concatenate(\n"
          "                (relative_cell, relative_cell),\n"
          "                axis=1,\n"
          "            )[:, None, :]\n"
          "            detail_hit = _segment_intersects_closed_boxes(\n"
          "                start_fraction[:, None, :],\n"
          "                relative_end[:, None, :],\n"
          "                detail_boxes,\n"
          "            )\n"
          "            opaque = opaque & (\n"
          "                ~jnp.any(selected.collision_box_mask, axis=1)\n"
          "                | jnp.any(detail_hit & selected.collision_box_mask, axis=1)\n"
          "            )\n"
          "            terminal_here = _reached(cell, terminal)\n",
          "2b",
          "perception LOS probe on the live 0.7.2 bridge (full-cube fixtures: los_wall, diagonal graze, diagonal block) "
          "against native target_los; partial-box rays unmeasured (no fixture)",
          "the production perception opacity walk (jax/world/perception/los.py, visit) under the detail rule; the "
          "gen-5 walk is unchanged; cells without detail boxes keep the gen-5 opacity rule"),
    Patch("npc_rejump_import", "combat/runtime/step.py",
          "AGENT_ENTITY, GAIT_COUNT,", "AGENT_ENTITY, AGENT_MOTION_NPC_WALK, GAIT_COUNT,",
          "1c", "with npc_rejump_after_landing", "the motion-model constant the re-jump patch reads"),
    Patch("npc_rejump_after_landing", "combat/runtime/step.py",
          "    grounded_jump = jump & state.agent_grounded & ~landing_recovery\n",
          "    # 0.7.2 NPC walk controller (live jump-held captures): landing on tick L, the next jump on L + 1, with no\n"
          "    # landing-recovery deferral; the player model keeps the deferral.\n"
          "    grounded_jump = jump & state.agent_grounded & (\n"
          "        ~landing_recovery | (params.agent_motion_model == jnp.int32(AGENT_MOTION_NPC_WALK))\n"
          "    )\n",
          "1c",
          "native_motion_differential replays of the two live jump-held captures (jump-forward-30, "
          "jump_in_place12): FAIL 0.30 / 0.18 m before, PASS (<= 1.1e-5 m) after, with the npc_jump_arc construct",
          "the planar suppression on that tick (jump_recovery) stays; not modelled: native walks again only on "
          "the third tick after landing (the engine on the first), which needs engine state"),
    Patch("guard_release_closes_guard_choice", "combat/observation/v3/encoding/encoder.py",
          "        & (\n            state.mechanics.guard_active[:, AGENT_ENTITY]\n            | (\n                movement_enabled\n",
          "        & (\n"
          "            # Gen 7 (row 5d): a guard being released (active, no longer held) closes the \"on\" choice, as on\n"
          "            # the live 0.7.2 bridge for the tick after the release request; the engine refuses that re-raise too.\n"
          "            (state.mechanics.guard_active[:, AGENT_ENTITY] & state.mechanics.guard_held[:, AGENT_ENTITY])\n"
          "            | (\n"
          "                ~state.mechanics.guard_active[:, AGENT_ENTITY]\n"
          "                & movement_enabled\n",
          "5d",
          "the shield mask probe (shield_mask_probe.py, jar A4487CF3, both seeds: guard held 12 ticks, released on 13; "
          "native closes guard_off_on \"on\" on tick 13 only) and the learner-row guard_held_release cases",
          "mask only: the engine already refuses a re-raise while the guard is still active (can_prepare_guard). Not "
          "modelled: natively a new guard also waits about 0.35 s from the last guard's start, the default "
          "interaction cooldown (hold-duration probe: after holds of 2, 4 and 6 ticks the choice reopens on tick 11), "
          "which needs a mechanics state field; and the shield bash, a Primary fork of Guard_Wield in the 0.7.2 asset, "
          "is never advertised natively during a held guard (a bridge legality item, #61)"),
    Patch("structural_bench_import", "combat/arsenal/world_actions/crafting.py",
          "    BENCH_TYPE_CRAFTING,\n    INPUT_MODE_NORMAL,\n",
          "    BENCH_TYPE_CRAFTING,\n    BENCH_TYPE_STRUCTURAL_CRAFTING,\n    INPUT_MODE_NORMAL,\n",
          "11", "with structural_bench_types", "the structural bench type constant the two patches below read"),
    Patch("structural_bench_types", "combat/arsenal/world_actions/crafting.py",
          "_BENCH_TYPE_NAMES = {\n    \"Crafting\": BENCH_TYPE_CRAFTING,\n}\n",
          "_BENCH_TYPE_NAMES = {\n    \"Crafting\": BENCH_TYPE_CRAFTING,\n"
          "    # Gen 7 (row 11): the Builders bench (Bench.Type StructuralCrafting) is a Region bench too; the live 0.7.2\n"
          "    # crafting oracle (FIDELITY-0.7.2.md (s)) crafts its recipes from the player's inventory.\n"
          "    \"StructuralCrafting\": BENCH_TYPE_STRUCTURAL_CRAFTING,\n}\n",
          "11", _STRUCTURAL,
          "Builders benches become Region bench rows (bench contexts of type 3, tier 0)"),
    Patch("structural_bench_admission", "combat/arsenal/world_actions/crafting.py",
          "            requirement.bench_type == BENCH_TYPE_CRAFTING\n",
          "            requirement.bench_type in (BENCH_TYPE_CRAFTING, BENCH_TYPE_STRUCTURAL_CRAFTING)\n",
          "11", _STRUCTURAL,
          "a recipe whose bench requirement is the structural bench is admitted like an ordinary bench's; the commit "
          "runs under its type-3 requirement (bench_crafting) and consumes from the player's inventory in normal "
          "mode, because the bridge moves the input into the window's slot for the server's ordered queue. Not "
          "modelled: the output lands one tick after the request natively (the bridge's queued path, the oracle's "
          "observation; a native player's window timing not yet measured)"),
    Patch("crafting_table_capacity", "crafting/contract.py",
          "# Installed 0.5.7 has 403 explicit plus 1,544 item-generated recipes.\n"
          "RECIPE_TABLE_CAPACITY = 1947\n",
          "# Installed 0.5.7 has 403 explicit plus 1,544 item-generated recipes; the live\n"
          "# 0.7.2 catalog has 2,047 (FIDELITY-0.7.2.md row 11).\n"
          "RECIPE_TABLE_CAPACITY = 2047\n",
          "11",
          "surrogate_crafting_native_compare.py once its 0.5.7 gate is profile-keyed, and the crafting "
          "source audit at a 0.7.2 baseline",
          "capacity only: the packer still rejects the 0.7.2 catalog (988 excluded-material inputs, 19 "
          "recipes without a bench, 22 bench ids); hytalegym.worldgen.native_action_resolution (shared, "
          "not rendered) still imports the gen-5 contract"),
    Patch("dodge_starts_one_drain_later", "combat/mechanics/schema/contract.py",
          "DODGE_LAUNCH_TICK = 2\nDODGE_COST_TICK = 3\n",
          "# Live 0.7.2 bridge (2026-10-18, jar 32EF5EDD, two seeds): a dodge requested on tick 1 is accepted on\n"
          "# tick 2 (the next command drain), launches on tick 3 and spends its stamina on tick 4; its force\n"
          "# sequence is the engine's one tick later (FIDELITY-0.7.2.md row 5f).\n"
          "DODGE_LAUNCH_TICK = 3\nDODGE_COST_TICK = 4\n",
          "5f",
          "the dodge timing probe on the render (first force on tick 3, stamina spent on tick 4, the force sequence "
          "equal to native) and the learner-row dodge_left cases (test_dynamic_learner_row_world_inputs.py)",
          "the force (applied velocity) moves to tick 3 and the stamina cost and regeneration delay to tick 4; the "
          "invulnerability and cooldown stay on the acceptance tick (dodge_effects_on_acceptance)"),
    Patch("dodge_effects_on_acceptance", "combat/mechanics/runtime/motion.py",
          "        dodge_invulnerability_remaining_seconds=jnp.where(\n            dodge_launch,\n            rules.dodge_invulnerability_seconds + dt,\n            state.dodge_invulnerability_remaining_seconds,\n        ),\n        dodge_cooldown_remaining_seconds=jnp.where(\n            dodge_launch,\n            rules.dodge_cooldown_seconds + dt,\n            state.dodge_cooldown_remaining_seconds,\n        ),\n",
          "        # Gen 7 (row 5f): on the live 0.7.2 bridge the invulnerability and the cooldown start on the tick the\n        # dodge is accepted, one tick before its force (DODGE_LAUNCH_TICK).\n        dodge_invulnerability_remaining_seconds=jnp.where(\n            dodge_pending & alive & (next_dodge_tick == rules.dodge_launch_tick - 1),\n            rules.dodge_invulnerability_seconds + dt,\n            state.dodge_invulnerability_remaining_seconds,\n        ),\n        dodge_cooldown_remaining_seconds=jnp.where(\n            dodge_pending & alive & (next_dodge_tick == rules.dodge_launch_tick - 1),\n            rules.dodge_cooldown_seconds + dt,\n            state.dodge_cooldown_remaining_seconds,\n        ),\n",
          "5f",
          "the learner-row dodge_left cases on the live bridge: after dodge_starts_one_drain_later alone the invulnerability fraction and the dodge action mask ran one tick late (native: invulnerability 1.0 and the dodge closed on tick 2, the acceptance tick)",
          "the stock engine started both on its launch tick 2, which was right; the timing patch had moved them with the force"),
    Patch("potion_stamina_072_drink", PROFILE_JAX,
          "        _ability(\n"
          "            \"Potion_Stamina\",\n"
          "            0.7,\n"
          "            _resource(0.5, RESOURCE_STAMINA, 45, percent=True),\n"
          "            _status(\n"
          "                0.5,\n"
          "                \"Potion_Stamina_Cooldown\",\n"
          "                15,\n"
          "                status_flags=STATUS_FLAG_DEBUFF,\n"
          "            ),\n"
          "            cooldown=15,\n"
          "        ),\n",
          "        _ability(\n"
          "            \"Potion_Stamina\",\n"
          "            0.7,\n"
          "            _resource(0.5, RESOURCE_STAMINA, 45, percent=True),\n"
          "            # Gen 7 (row 8e): 0.7.2 applies no Potion_Stamina_Cooldown status; the drink triggers the\n"
          "            # Potion_Consume_Stamina cooldown of 30 s instead (Consume_Potion_Stamina_Cooldown.json).\n"
          "            cooldown=30,\n"
          "        ),\n",
          "8e", _POTION,
          "the restore (45 percent) and the 0.7 s ability are unchanged; the status goes and the cooldown is 30 s. "
          "Residual (inferred from the asset, not measured): the engine starts the cooldown at admission, the server "
          "when the 0.5 s charge completes (Consume_Charge_Potion_Fastest, Next 0.5)"),
    # Row 1c, landing walk (construct landing_walk_resume): the countdown the construct reads and writes.
    Patch("landing_walk_state_field", "combat/types.py",
          "    agent_locomotion_stamina_regen_delay: Array\n",
          "    agent_locomotion_stamina_regen_delay: Array\n"
          "    # Gen 7 (row 1c, landing walk): ticks the npc_walk agent still holds its planar walk after touchdown.\n"
          "    agent_landing_hold_ticks: Array\n",
          "1c", _LANDING_WALK,
          "the CombatState field the landing_walk construct reads and writes (int32 countdown; gen 5 does not apply this patch, so it has no field)"),
    Patch("landing_walk_state_init", "combat/runtime/reset.py",
          "        agent_locomotion_stamina_regen_delay=zeros_f,\n",
          "        agent_locomotion_stamina_regen_delay=zeros_f,\n"
          "        agent_landing_hold_ticks=zeros_i,\n",
          "1c", _LANDING_WALK,
          "the countdown starts at zero at reset (no hold before the first landing)"),
    # Row 11b, bench tiers (construct crafting_bench_tier): a tier requirement up to 8 is admitted, not only tier 0.
    Patch("bench_tier_admission", "combat/arsenal/world_actions/crafting.py",
          "            and requirement.required_tier_level == 0\n",
          "            # Gen 7 (row 11b): a tier requirement up to the highest native bench tier (8); the tier gate is the\n"
          "            # provider's, under each bench context's own tier (crafting/runtime._recipe_precondition).\n"
          "            and requirement.required_tier_level <= 8\n",
          "11b", _BENCH_TIERS,
          "the native tier gate (requiredTierLevel > tier refuses, section (s)) is the engine's existing precondition; "
          "only the admission filter changes. Not modelled: one tier per bench row, not per placed bench instance"),
    # Row 11m, memories level (construct crafting_memories): the level is the scenario's, and recipes up to level 6 are admitted.
    Patch("memories_state_field", "combat/arsenal/world_actions/crafting.py",
          "    bench_valid: jax.Array\n",
          "    bench_valid: jax.Array\n"
          "    # Gen 7 (row 11m): the scenario's universe-wide memories level, 1 to 6 (crafting_memories.construct sets it).\n"
          "    memories_level: int = 1\n",
          "11m", _MEMORIES,
          "the level the fieldcraft and bench contexts carry; gen 5 has no field (the RegionFieldcraftAssets default is level 1)"),
    Patch("memories_admission", "combat/arsenal/world_actions/crafting.py",
          "        and recipe.required_memories_level == 1\n",
          "        # Gen 7 (row 11m): levels up to 6 (crafting_memories.MAX_LEVEL).\n"
          "        and recipe.required_memories_level <= 6\n",
          "11m", _MEMORIES,
          "the lossless filter admits the recipe's own level; the gate is _recipe_precondition's memories term"),
    Patch("memories_resolution", "combat/arsenal/world_actions/crafting.py",
          "        required_memories_level=1,\n",
          "        required_memories_level=recipe.required_memories_level,\n",
          "11m", _MEMORIES,
          "the packed table carries each recipe's level (gen 5 resolves every recipe to level 1)"),
    Patch("memories_fieldcraft_signature", "combat/arsenal/world_actions/crafting.py",
          "def _fieldcraft_context(batch: int, entities: int) -> CraftingContext:\n",
          "def _fieldcraft_context(batch: int, entities: int, memories_level: int = 1) -> CraftingContext:\n",
          "11m", _MEMORIES, "the fieldcraft context takes the scenario level (default 1, as gen 5)"),
    Patch("memories_fieldcraft_body", "combat/arsenal/world_actions/crafting.py",
          "        memories_level=jnp.ones(prefix, dtype=jnp.int32),\n"
          "        known_recipe=jnp.ones(prefix + (RECIPE_TABLE_CAPACITY,), dtype=jnp.bool_),\n",
          "        memories_level=jnp.full(prefix, memories_level, dtype=jnp.int32),\n"
          "        known_recipe=jnp.ones(prefix + (RECIPE_TABLE_CAPACITY,), dtype=jnp.bool_),\n",
          "11m", _MEMORIES, "the fieldcraft context's level"),
    Patch("memories_bench_context", "combat/arsenal/world_actions/crafting.py",
          "        memories_level=jnp.ones(prefix, dtype=jnp.int32),\n"
          "        known_recipe=jnp.ones(\n",
          "        memories_level=jnp.full(prefix, assets.memories_level, dtype=jnp.int32),\n"
          "        known_recipe=jnp.ones(\n",
          "11m", _MEMORIES, "each in-reach bench context's level"),
    Patch("memories_provider_context", "combat/arsenal/world_actions/crafting.py",
          "        fieldcraft = _fieldcraft_context(batch, entities)\n",
          "        fieldcraft = _fieldcraft_context(batch, entities, assets.memories_level)\n",
          "11m", _MEMORIES, "the provider's fieldcraft legality uses the scenario level"),
    Patch("memories_commit_context", "combat/arsenal/world_actions/crafting.py",
          "        _fieldcraft_context(batch, 1),\n",
          "        _fieldcraft_context(batch, 1, assets.memories_level),\n",
          "11m", _MEMORIES, "the commit re-checks the same level as the provider (a commit at another level would not match its candidate)"),
)

_MECHANICS = "hytalegym.sim.gen7.mechanics"
HOOKS = (
    Hook("hit_interrupt_flinch", "combat/mechanics/runtime/kernel.py", "resolve_damage_events",
         f"{_MECHANICS}.hit_interrupt:construct", "hit_interrupt", "stub", "5a",
         "interrupt/flinch differential against an eligible NPC (Skeleton_Fighter or Goblin_Scrapper): "
         "flinch 0.5 s, stun 0.7 s, attack cancelled",
         "31 attacks parent DamageEntityParent_Hit_Interrupt in 0.7.2; Trork_Brawler, Trork_Unarmed and "
         "Kweebec_Razorleaf carry no eligibility (static reading)"),
    Hook("mana_pool", "combat/mechanics/factory.py", "default_mechanics_rules",
         f"{_MECHANICS}.mana_pool:construct", "mana_pool", "active", "7a",
         "native_duel refill from 0 (live 0.7.2 bridge, native_agent_mana): +1 per 8.95 world ticks = 0.2982 s, "
         "against the 0.3 s bar within 5 %; JAX: tick_resources from 0 at 1/30 s (rendered engine)",
         "maximum 100 and initial 100 from stats/stat:Mana; the regeneration interval 0.3 s replaces the gen-5 "
         "0.2 s literal (factory.py line 134); rune casts spend this pool only once rune_casting is built"),
    Hook("rune_casting", PROFILE_JAX, "hytale_0_5_7_loadouts",
         f"{_MECHANICS}.mana:construct", "rune_casting", "stub", "7b",
         "one differential per rune: cost, cooldown, cast time, flight, effect",
         "7 ability runes plus Rune_Empty and 11 modifier runes under Server/Item/Items/Rune/"),
    Hook("detail_hitbox_line_of_sight", "world/geometry/__init__.py", "line_of_sight_result",
         f"{_MECHANICS}.line_of_sight:construct", "line_of_sight", "active", "2b",
         "full-cube rays (native_geometry_compare probes: los_wall, diagonal graze and block) must still "
         "agree; a partial-box ray that the bridge can place is NOT available (no slab fixture aims through "
         "the empty half of Rock_Stone_Half; status open)"),
    Hook("npc_walk_jump_velocity", "combat/types.py", "default_combat_params",
         f"{_MECHANICS}.npc_jump:construct", "npc_walk_jump_velocity", "active", "1c",
         "native_motion_differential --jump-steps 1 on the gen-7 render against the live 0.7.2 bridge "
         "(max position error <= 5e-4 m per step): first step exact; the rest of the arc is the npc_jump_arc "
         "construct (F1c)",
         "npc_walk only: the player model keeps PLAYER_JUMP_FORCE 11.8; gen 5 keeps player_force"),
    Hook("npc_jump_arc_previous_delta", "combat/runtime/step.py", "_microtick",
         f"{_MECHANICS}.npc_jump_arc:capture_previous_delta", "npc_jump_arc", "active", "1c", _JUMP_ARC,
         "holds the previous microtick's duration (state.last_motion_delta_seconds, which _microtick overwrites "
         "before the agent moves) for the agent's motion; changes no state"),
    Hook("npc_jump_arc_fall_speed", "combat/runtime/agent.py", "_tick_agent_geometry_motion",
         f"{_MECHANICS}.npc_jump_arc:fall_speed_integration", "npc_jump_arc", "active", "1c", _JUMP_ARC,
         "npc_walk jump rows of the legacy branch after the impulse: v_k = v_(k-1) - 32 (1 - |(v_(k-1) - v_jump) "
         "/ T|^3) dt_(k-1), with T = 140.488 m/s from the agent bounds and the ModelAsset default PhysicsValues "
         "(mass 68, drag 0.5); the walk controller's fall, knockback arcs, the no-geometry integrator and the "
         "player model are unchanged"),
    Hook("dagger_swing_damage", PROFILE_JAX, "_scalar_variant_profile",
         f"{_MECHANICS}.dagger_damage:construct", "dagger_swing_damage", "active", "5c",
         "scalar dagger variants (15 materials) against the 0.7.2 item rows: JAX swing tuple equals the table, "
         "and a seeded bronze-dagger contact on the live bridge (native 5.0; tests/fidelity)",
         "Swing_Left and Swing_Right per material; the other direct values and the angled values are unchanged; "
         "Void_Crystal is in the 0.7.2 table but has no simulator variant row (not modelled)"),
    Hook("guard_cancel_window_roots", PROFILE_JAX, "_ability",
         f"{_MECHANICS}.guard_window:ability_root_window", "guard_cancel_window", "active", "5d", _GUARD,
         "stamps each Sword, Mace, Battleaxe and Daggers Primary attack root that 0.7.2 declares InterruptedBy "
         "[Secondary] with its RunTime (weapons rows interaction:<id>); replaces the inert thrust patch (the engine "
         "resolves a one-tick thrust request through the primary hold table to the left swing)"),
    Hook("guard_cancel_window_operations", "combat/arsenal/programs/interaction_rules.py",
         "active_operation_interrupted_by",
         f"{_MECHANICS}.guard_window:operation_window", "guard_cancel_window", "active", "5d", _GUARD,
         "for those attacks the boundary ends the Secondary window instead of opening it; not modelled: a Guard "
         "held through the end of an uncancelled attack never rises natively, and native admits an attack one "
         "tick before the engine's legality view (the first window tick)"),
    Hook("native_binding_types", PROFILE_JAX, "hytale_0_5_7_native_profile_bindings",
         f"{_MECHANICS}.native_bindings:construct", "native_binding_types", "active", "6c",
         "the crossbow tests on the live 0.7.2 bridge (jar 32EF5EDD): the reload slot is advertised and its "
         "Ability4 chain runs (tests/fidelity/combat/ability_lifecycle/test_native_crossbow_reload.py, "
         "projectiles/crossbow)",
         "names only: a binding takes its type from the 0.7.2 item's Interactions map where that map binds the "
         "root under one other type (the two crossbow reloads, compact slot 2, and no other binding); the engine "
         "keeps the reload's ordinal INTERACTION_TYPE_ABILITY3, which no ability mask references; Primary is "
         "still not advertised during the native reload (bridge legality, FIDELITY-0.7.2.md (m), open)"),
    Hook("bench_crafting_commit", "crafting/inventory_adapter.py", "craft_into_combat_inventory",
         f"{_MECHANICS}.bench_crafting:bench_commit", "bench_crafting", "active", "11", _BENCH_CRAFTING,
         "a lane whose context satisfies none of its recipe's bench requirements commits under the first one; a "
         "recipe the Fieldcraft context satisfies keeps it (every gen-5 commit unchanged); the executor's only "
         "caller is world_actions/crafting.execute_region_fieldcraft"),
    Hook("bench_crafting_first_candidates", "world/interactions/crafting.py",
         "produce_actor_recipe_candidates_from_legal_mask",
         f"{_MECHANICS}.bench_crafting:first_candidates", "bench_crafting", "active", "11", _BENCH_CRAFTING,
         "more than 16 legal recipes: the first 16 in table order (canonical recipe-id order), row available, "
         "source_count the full count, diagnostics ACTOR_RECIPE_DIAGNOSTIC_CAPACITY (truncated), "
         "capacity_exceeded False; shapes, radix and contract stamps unchanged"),
    Hook("crafting_materials_item_table", "combat/arsenal/world_actions/crafting.py", "compile_region_fieldcraft_assets",
         f"{_MECHANICS}.crafting_materials:item_resource_types", "crafting_materials", "active", "11",
         _CRAFTING_MATERIALS, "the archive's item ResourceTypes as a table, item semantic id -> resource semantic ids"),
    Hook("crafting_materials_admission", "combat/arsenal/world_actions/crafting.py", "_lossless_region_crafting",
         f"{_MECHANICS}.crafting_materials:admission", "crafting_materials", "active", "11", _CRAFTING_MATERIALS,
         "a resource-type input that excludes exactly its recipe's outputs (applyOutputExclusions); outputs stay "
         "concrete items; the four raw-fish recipes, whose input excludes more, stay out"),
    Hook("crafting_materials_resolution", "combat/arsenal/world_actions/crafting.py", "_resolved_recipe",
         f"{_MECHANICS}.crafting_materials:resolution", "crafting_materials", "active", "11", _CRAFTING_MATERIALS,
         "such inputs keep their resource type in the packed table"),
    Hook("crafting_materials_inventory", "combat/arsenal/world_actions/crafting.py", "_crafting_inventory",
         f"{_MECHANICS}.crafting_materials:inventory_resource_types", "crafting_materials", "active", "11",
         _CRAFTING_MATERIALS, "each input slot's resource types, for the candidate provider's legality"),
    Hook("crafting_materials_commit", "crafting/inventory_adapter.py", "craft_into_combat_inventory",
         f"{_MECHANICS}.crafting_materials:commit_resource_types", "crafting_materials", "active", "11",
         _CRAFTING_MATERIALS, "the commit catalog's slot resource types (the executor builds them absent)"),
    Hook("crafting_materials_exclusions", "crafting/runtime.py", "_slot_matches",
         f"{_MECHANICS}.crafting_materials:output_exclusions", "crafting_materials", "active", "11",
         _CRAFTING_MATERIALS, "a resource-type input never consumes a slot holding one of its recipe's outputs"),
    Hook("dodge_action_head", "combat/observation/v3/policy/layout.py", "split_locomotion_choice",
         f"{_MECHANICS}.dodge:construct", "dodge_action_head", "stub", "9",
         "test_native_policy_dodge_boundary_matches_advertised_capability and "
         "test_native_lateral_dodge_projects_mechanics_when_supported (both pass on 0.7.2)",
         "not a constant: DODGE_ACTION_COUNT (observation/v3/schema/contract.py:123) indexes "
         "[forward, back, left, right] in this decode (layout.py:664-673), the encode "
         "(layout.py:805-813), the 4-wide mask concat (policy/surface.py:441-443), "
         "encoding/encoder.py:507 and native/codec/decode.py:112,169; setting it to 2 keeps the masked "
         "forward/back slots and drops left/right"),
    Hook("weapon_durability_catalog", "combat/arsenal/item_programs.py", "hytale_0_5_7_item_program_catalog",
         f"{_MECHANICS}.durability:construct", "weapon_durability", "active", "8d",
         "file-sourced (sim/gen7/data/durability.json, the 0.7.2 item rows); loss per hit not measured: the bridge "
         "exposes durability on its inventory frame (NativeInventoryCapture.java:72), not in info, and a read needs a "
         "policy attack",
         "the catalog's weapon_items: max_durability and durability_loss_on_hit of all 143 rows from the 0.7.2 table "
         "(57 maxima and 82 losses change); the rest of the catalog, semantic_sha256 included, unchanged"),
    Hook("landing_walk_resume", "combat/runtime/step.py", "_apply_action",
         f"{_MECHANICS}.landing_walk:construct", "landing_walk_resume", "active", "1c", _LANDING_WALK,
         "after an accepted jump, holds forward, back, left and right for the two grounded ticks after touchdown "
         "(npc_walk only); a landing without a jump is not held; the jump request is not held; the native pause before "
         "a re-jump (F1c) is the existing patch npc_rejump_after_landing"),
    Hook("bench_tier_assets", "combat/arsenal/world_actions/crafting.py", "compile_region_fieldcraft_assets",
         f"{_MECHANICS}.bench_tiers:construct", "crafting_bench_tier", "active", "11b", _BENCH_TIERS,
         "the bench rows' tier from the scenario (bench_tiers=: an int for every bench, or a {bench id: tier} mapping; a fresh "
         "bench, 1, by default); gen 5 keeps the zero fill"),
    Hook("crafting_memories_assets", "combat/arsenal/world_actions/crafting.py", "compile_region_fieldcraft_assets",
         f"{_MECHANICS}.crafting_memories:construct", "crafting_memories", "active", "11m", _MEMORIES,
         "the scenario's memories level (memories_level=, 1 to 6; default 1); a caller passes the bridge's "
         "native_memories_level when the bridge reports one (none does yet)"),
)

OVERLAY = Overlay(
    generation="gen7",
    profile="hytalegym.sim.gen7.profile:PROFILE",
    modules=("combat/",),
    redirects=STEP_CLOCK + CONTENT_LOCOMOTION + CONTENT_VALUES + CONTENT_TODO,
    hooks=HOOKS,
    patches=PATCHES,
    audit=(
        ("step_duration_sites", STEP_CLOCK_SITES),
        ("swap_plan_rows", len(CONTENT_TODO)),
        ("swap_plan_unreachable", [f"{r['kind']} {r['module'].split('hytalegym/', 1)[-1]}:{r['head_line']} "
                                   f"[{r['status']}] {r['target']}" for r in UNREACHABLE_RECORDS]),
    ),
    description="skeleton: 30 Hz step clock, FIDELITY (e) constant/rule patches, construct hook stubs",
)
