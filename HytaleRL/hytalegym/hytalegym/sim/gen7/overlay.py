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
shared module outside the rendered package); the potion drink's status and cooldown (patch
``potion_stamina_072_drink``, row 8e); durability (a JSON table: hook ``weapon_durability_catalog`` on the catalog loader,
row 8d). The sword (the thrust's event time and hold
speed included; its total stays 0.601, see mechanics_072.md row 5f), shortbow and crossbow values are the
``CONTENT_VALUES`` rows below.
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
#: (a JSON table: hook weapon_durability_catalog, row 8d). The sword thrust's event time and hold speed are
#: the three rows at the end of CONTENT_VALUES (row 5f, 2026-10-22); its 0.533 total is not applied (live ends r+44).
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
_THRUST_TIMING = (
    "0.7.2 iron sword thrust event time and hold speed (file-sourced from weapons/interaction:"
    "Weapon_Sword_Primary_Thrust: the charged branch's Simple RunTime; HorizontalSpeedMultiplier). differential: the "
    "live forced-charge thrust (fidelity_evidence/gen7_probes/thrust_lunge, jar 1F874DFC, seeds 570057 and 913771) "
    "first moves at r+27, the render at r+26 with 0.05 or 0.043; the hold speed is not measurable live until the "
    "hold wire (#138)"
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
    Redirect(
        "sword_thrust_timing:dash", PROFILE_JAX, "_iron_sword", "adapter", "sword_thrust_timing", "event_seconds",
        cast="float", status="active", fidelity_row="5f", source=_THRUST_TIMING,
        expression="_force(0.05, (0, 0, -10), 20)", template="_force({value}, (0, 0, -10), 20)",
    ),
    Redirect(
        "sword_thrust_timing:stab", PROFILE_JAX, "_iron_sword", "adapter", "sword_thrust_timing", "event_seconds",
        cast="float", status="active", fidelity_row="5f", source=_THRUST_TIMING,
        expression="_melee(\n                0.05,\n                thrust,",
        template="_melee(\n                {value},\n                thrust,",
    ),
    Redirect(
        "melee_hold_constants:sword_thrust_hold_speed_multiplier", PROFILE_JAX, "_iron_sword", "adapter",
        "melee_hold_constants", "sword_thrust_hold_speed_multiplier", cast="float", status="active",
        fidelity_row="5f", source=_THRUST_TIMING, expression="hold_speed_multiplier=0.8,",
        template="hold_speed_multiplier={value},",
    ),
)
_VALUE_IDS = frozenset(row.id for row in CONTENT_VALUES)
CONTENT_TODO = tuple(r for r in CONTENT_ROWS if r.row not in LOCOMOTION_ADAPTERS and r.id not in _VALUE_IDS)

_SWORD_KNOCKBACK = (
    "knockback displacement differential on 0.7.2 (TRAINING-0.7.2.md: Trork 0.19 m vs 0.44 m on "
    "0.5.10) and the test_jax_native_combat_compare.py sword tests recalibrated to 0.7.2"
)
_A1 = (
    "force values from the 0.7.2 table; VelocityType Add on Left (sword_swing_left_velocity_add) and Right (inside "
    "sword_swing_right_knockback), Set on Down and Thrust; gen 5 keeps Set through calibration_057.json"
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


_JUMP_CARRY = (
    "tests/sim/test_jump_carry.py: the running jump's horizontal speed per tick on the gen-7 render against the live 0.7.2 "
    "bridge DBB3736B (receipt knockback_landing_jump_Kweebec_Razorleaf_carry_DBB3736B.json, two seeds, five jumps each, all "
    "identical): within 0.02 blocks/s per tick, touchdown on L = 50, walk at full speed on L+5; the switch off restores the "
    "HEAD arc"
)

_LANDING_WALK = (
    "native_motion_differential on the gen-7 render against the live 0.7.2 bridge, npc_walk, ledge fixture: "
    "--steps 30 --jump-steps 1 (single jump then forward), --steps 12 --jump-steps 1 --forward-steps 0 (jump in "
    "place) and --steps 30 --jump-steps 0 (walk off the ledge, no jump: must not be held), seeds 570057 and 913771; "
    "bar 5e-4 m and 1e-3 m/s per step; after an accepted standing jump the walk starts on the third tick after touchdown "
    "(native z first moves on L+3); a running jump is carried by npc_jump_carry when that switch is on"
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

_TIMED_CRAFTING = (
    "tests/sim/test_crafting_oracle_replay.py (timed section): the stage 3 receipts replayed on the gen-7 render (FIDELITY-0.7.2.md "
    "section (t)): the plain craft (inputs out on request+1, output on request+90: 122 for the request at 32, native 120 in that run "
    "and 122 in the second); the reach-out cancel on request+11 with the refund to the original slots; the second request queued and "
    "consumed on the first job's completion, with and without stock; damage on request+11 without a cancel; the no-window control; "
    "the fieldcraft refusal; the queue depth of one; the gen-5 original"
)

_PROCESSING = (
    "tests/sim/test_processing_benches.py: the Campfire and Furnace receipts (FIDELITY-0.7.2.md section (v), seeds 570057 and 31337) "
    "replayed on the gen-7 render: the request takes the inputs and the fuel stack from the inventory; the first progress tick is "
    "R+2; the bench's output and the inputs leave the bench on the completion tick (the simulator's 1/30 s tick: 2.0 s on R+61, "
    "10 s on R+301, against native R+58/R+59 and R+289/R+286, the native world step being about 5 % longer); the second fuel "
    "consumption yields the charcoal; the fuel is burned at the progress rate; gen 5 keeps every original"
)
_DIAGRAM = (
    "tests/sim/test_diagram_crafting.py: the Armory receipts (FIDELITY-0.7.2.md section (v), seeds 570057 and 31337) replayed on the "
    "gen-7 render: the Weapon_Axe_Stone_Trork request is queued on its tick, its inputs leave and the axe enters the inventory on "
    "R+2; the queue is one deep; a timed diagram recipe stays with timed_crafting; gen 5 keeps every original"
)


_AUGMENT = (
    "tests/sim/test_crafting_augment_tags.py: a tagged recipe is refused at a bare GoblinLab_Workbench and accepted with the "
    "grants; the 50 tagged recipes are enumerated from the 0.7.2 archive; the defaults leave every other recipe bit-identical",
)

PATCHES = (
    # Row 11x, augment tags (construct crafting_augment_tags): the grant bits ride the context and the assets; the gate is a hook.
    Patch("augment_context_field", "crafting/types.py",
          "    input_mode: Array\n",
          "    input_mode: Array\n"
          "    # Gen 7 (row 11x): the bench's granted augment-tag bits (crafting_augment_tags.gate_hook); None without a bench.\n"
          "    bench_augment_mask: Array = None\n",
          "11x", _AUGMENT, "the bench context carries its grant bits; the fieldcraft context keeps None (no bench, no grants)"),
    Patch("augment_assets_field", "combat/arsenal/world_actions/crafting.py",
          "    bench_valid: jax.Array\n",
          "    bench_valid: jax.Array\n"
          "    # Gen 7 (row 11x): the scenario's granted augment-tag bits per bench row (crafting_augment_tags.compile_hook).\n"
          "    bench_augment_mask: jax.Array = None\n",
          "11x", _AUGMENT, "the per-row grant bits; gen 5 has no field"),
    Patch("augment_bench_context", "combat/arsenal/world_actions/crafting.py",
          "        bench_tier_level=jnp.where(\n",
          "        bench_augment_mask=None if assets.bench_augment_mask is None else jnp.where(\n"
          "            available,\n"
          "            assets.bench_augment_mask[row],\n"
          "            jnp.uint32(0),\n"
          "        ),\n"
          "        bench_tier_level=jnp.where(\n",
          "11x", _AUGMENT, "each in-reach bench context carries its row's grant bits"),
    Patch("sword_swing_left_knockback", PROFILE_JAX,
          "    push = (0.0, 1.0, -1.0, 6.0)\n",
          "    push = (0.0, 0.2, -1.0, 4.3)\n",
          "5b", _SWORD_KNOCKBACK, "Weapon_Sword_Primary_Swing_Left_Damage: Force 6.0 -> 4.3, Dir Y 1 -> 0.2. " + _A1),
    Patch("sword_swing_right_knockback", PROFILE_JAX,
          "                force=(0, 1, -1, 7),\n                on_hit=(RESOURCE_SIGNATURE_ENERGY, 1),\n"
          "                horizontal_selector=_SWORD_SWING_RIGHT_SELECTOR,\n",
          "                force=(0, 0.2, -1, 5.0),\n                force_mode=FORCE_ADD,\n"
          "                on_hit=(RESOURCE_SIGNATURE_ENERGY, 1),\n"
          "                horizontal_selector=_SWORD_SWING_RIGHT_SELECTOR,\n",
          "5b", _SWORD_KNOCKBACK, "Weapon_Sword_Primary_Swing_Right_Damage: Force 7 -> 5.0, Dir Y 1 -> 0.2, VelocityType Set -> Add. " + _A1),
    Patch("sword_swing_down_knockback", PROFILE_JAX,
          "                force=(0, 1, -2, 12),\n",
          "                force=(0, 0.2, -2, 10.8),\n",
          "5b", _SWORD_KNOCKBACK, "Weapon_Sword_Primary_Swing_Down_Damage: Force 12 -> 10.8, Dir Y 1 -> 0.2"),
    Patch("sword_swing_left_velocity_add", PROFILE_JAX,
          "                force=push,\n                on_hit=(RESOURCE_SIGNATURE_ENERGY, 1),\n",
          "                force=push,\n                force_mode=FORCE_ADD,\n                on_hit=(RESOURCE_SIGNATURE_ENERGY, 1),\n",
          "5b", _SWORD_KNOCKBACK, "Weapon_Sword_Primary_Swing_Left_Damage: VelocityType Set -> Add (the 0.7.2 table authors Add; "
          "the engine's FORCE_ADD adds the force to the target's velocity). Down and Thrust stay Set"),
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
    # Row 1c, jump carry (construct npc_jump_carry): the carried jump's counter (1 on take-off, +1 per grounded tick).
    Patch("jump_carry_state_field", "combat/types.py",
          "    agent_locomotion_stamina_regen_delay: Array\n",
          "    agent_locomotion_stamina_regen_delay: Array\n"
          "    # Gen 7 (row 1c, jump carry): 1 on the take-off tick of a carried jump, +1 per grounded tick, 0 idle.\n"
          "    agent_jump_carry_ticks: Array\n",
          "1c", _JUMP_CARRY,
          "the CombatState field the npc_jump_carry construct reads and writes (int32; gen 5 does not apply this patch)"),
    Patch("jump_carry_state_init", "combat/runtime/reset.py",
          "        agent_locomotion_stamina_regen_delay=zeros_f,\n",
          "        agent_locomotion_stamina_regen_delay=zeros_f,\n"
          "        agent_jump_carry_ticks=zeros_i,\n",
          "1c", _JUMP_CARRY, "the counter starts at zero at reset (no carry before the first jump)"),
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
    # Row 5d, guard re-entry cooldown (construct guard_reentry_cooldown): the countdown the construct reads and writes.
    Patch("guard_reentry_seconds_field", "combat/mechanics/schema/types.py",
          "    guard_operation_tick: Array\n",
          "    guard_operation_tick: Array\n"
          "    # Gen 7 (row 5d, guard re-entry cooldown): seconds before a new guard claim is admitted (float32 countdown).\n"
          "    guard_reentry_seconds: Array\n",
          "5d", "section (af) guard reopen receipts (24 runs) and tests/sim/test_timing_constructs.py",
          "the mechanics state field that guard_cooldown reads and writes (gen 5 does not apply this patch, so it has no field)"),
    Patch("guard_reentry_seconds_init", "combat/mechanics/factory.py",
          "        guard_active=jnp.zeros(entity_shape, dtype=jnp.bool_),\n",
          "        guard_active=jnp.zeros(entity_shape, dtype=jnp.bool_),\n"
          "        guard_reentry_seconds=jnp.zeros(entity_shape, dtype=jnp.float32),\n",
          "5d", "section (af) guard reopen receipts (24 runs) and tests/sim/test_timing_constructs.py",
          "the countdown starts at zero at reset, so the first guard claim is never held"),
    Patch("guard_reentry_closes_new_claim", "combat/observation/v3/encoding/encoder.py",
          "                & guard_rules.can_start[:, AGENT_ENTITY]\n"
          "            )\n",
          "                & guard_rules.can_start[:, AGENT_ENTITY]\n"
          "                # Gen 7 (row 5d): a new guard claim is off the policy mask while the re-entry countdown runs; a\n"
          "                # guard already held (active or not yet active) stays on it.\n"
          "                & (\n"
          "                    state.mechanics.guard_held[:, AGENT_ENTITY]\n"
          "                    | (state.mechanics.guard_reentry_seconds[:, AGENT_ENTITY] <= 0)\n"
          "                )\n"
          "            )\n",
          "5d", "section (af) guard reopen receipts (24 runs) and tests/sim/test_timing_constructs.py",
          "the guard-on choice reopens on the tick the countdown reaches zero (gen 5 keeps the countdown at zero, so the term is always true)"),
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
    # Row 11j, timed crafting (construct timed_crafting): the controlled actor's job leaves and their zero state.
    Patch("timed_job_fields", "combat/types.py",
          "    agent_locomotion_stamina_regen_delay: Array\n",
          "    agent_locomotion_stamina_regen_delay: Array\n"
          "    # Gen 7 (row 11j, timed crafting): the controlled actor's bench job (timed_crafting.py), per lane.\n"
          "    craft_job_recipe: Array\n"
          "    craft_job_elapsed_seconds: Array\n"
          "    craft_job_inputs_pending: Array\n"
          "    craft_queued_recipe: Array\n"
          "    craft_bench_position: Array\n"
          "    craft_refund_quantity: Array\n"
          "    craft_refund_item_id: Array\n",
          "11j", _TIMED_CRAFTING,
          "the job leaves the timed_crafting construct reads and writes (gen 5 applies no patch, so it has none)"),
    Patch("timed_job_init", "combat/runtime/reset.py",
          "        agent_locomotion_stamina_regen_delay=zeros_f,\n",
          "        agent_locomotion_stamina_regen_delay=zeros_f,\n"
          "        craft_job_recipe=jnp.full((batch_size,), -1, dtype=i32),\n"
          "        craft_job_elapsed_seconds=zeros_f,\n"
          "        craft_job_inputs_pending=zeros_b,\n"
          "        craft_queued_recipe=jnp.full((batch_size,), -1, dtype=i32),\n"
          "        craft_bench_position=jnp.zeros((batch_size, 3), dtype=i32),\n"
          "        # 76 slots: the default inventory layout (combat/inventory/schema/contract.py DEFAULT_CONTAINER_CAPACITIES)\n"
          "        craft_refund_quantity=jnp.zeros((batch_size, 76), dtype=i32),\n"
          "        craft_refund_item_id=jnp.full((batch_size, 76), -1, dtype=i32),\n",
          "11j", _TIMED_CRAFTING,
          "every job leaf starts idle (recipe -1, no queue, no refund, no progress) at reset"),
    Patch("timed_resolved_time", "combat/arsenal/world_actions/crafting.py",
          "        time_seconds=0.0,\n    )\n",
          "        time_seconds=recipe.time_seconds,\n    )\n",
          "11j", _TIMED_CRAFTING,
          "the resolved recipe keeps the recipe's time (the packed table's time column); gen 5 writes 0.0 for every recipe"),
    Patch("timed_executor_import", "combat/arsenal/world_actions/executor.py",
          "import numpy as np\n",
          "import numpy as np\n\nfrom hytalegym.sim.gen7.mechanics.timed_crafting import start as timed_crafting_start\n",
          "11j", _TIMED_CRAFTING, "the start function the craft-verb patch calls"),
    Patch("timed_executor_start", "combat/arsenal/world_actions/executor.py",
          "            craft_rechecked = craft.target_rechecked\n"
          "            craft_accepted = craft.accepted\n"
          "            next_state = next_state._replace(inventory=craft.state)\n",
          "            craft_rechecked = craft.target_rechecked\n"
          "            # Gen 7 (row 11j): an accepted timed craft starts or queues a job in the CombatState leaves; its inputs\n"
          "            # leave on the next tick and its output at completion (timed_crafting.start).\n"
          "            craft_accepted, next_state = timed_crafting_start(\n"
          "                execute_region_fieldcraft,\n"
          "                next_state,\n"
          "                craft,\n"
          "                candidates,\n"
          "                next_world,\n"
          "                crafting_assets,\n"
          "                inventory_layout,\n"
          "            )\n"
          "            next_state = next_state._replace(inventory=craft.state)\n",
          "11j", _TIMED_CRAFTING,
          "a timed craft is accepted when it starts or queues (refused with a running job and a queued one, and when no bench "
          "candidate makes the recipe legal); the zero-time acceptance is unchanged"),
    # Rows 11p and 11q (stage 5): the leaves, the bench types and the executor request path. Each patch's anchor is a HEAD text
    # that the earlier patches keep (the validator checks every anchor against HEAD); admission is a hook, not a patch.
    Patch("craft_pre_inventory", "combat/arsenal/world_actions/executor.py",
          "            craft = execute_region_fieldcraft(\n",
          "            # Gen 7 (rows 11p, 11q): the inventory before the craft, which a processing or diagram request works from.\n"
          "            craft_inventory_before = next_state.inventory\n"
          "            craft = execute_region_fieldcraft(\n",
          "11p", _PROCESSING, "the inventory a processing or diagram request starts from (the zero-time commit is undone for those lanes)"),
    Patch("processing_fields", "combat/types.py",
          "    agent_locomotion_stamina_regen_delay: Array\n",
          "    agent_locomotion_stamina_regen_delay: Array\n"
          "    # Gen 7 (row 11p, processing benches): the controlled actor's bench, per lane (processing_benches.py).\n"
          "    proc_bench_cell: Array\n"
          "    proc_recipe: Array\n"
          "    proc_active: Array\n"
          "    proc_has_fuel: Array\n"
          "    proc_wait_ticks: Array\n"
          "    proc_progress_seconds: Array\n"
          "    proc_fuel_seconds: Array\n"
          "    proc_fuel_item: Array\n"
          "    proc_fuel_quantity: Array\n"
          "    proc_fuel_quality: Array\n"
          "    proc_extra_countdown: Array\n"
          "    proc_input_item: Array\n"
          "    proc_input_quantity: Array\n"
          "    proc_output_item: Array\n"
          "    proc_output_quantity: Array\n",
          "11p", _PROCESSING, "the bench leaves the processing_benches construct reads and writes (gen 5 applies no patch)"),
    Patch("processing_init", "combat/runtime/reset.py",
          "        agent_locomotion_stamina_regen_delay=zeros_f,\n",
          "        agent_locomotion_stamina_regen_delay=zeros_f,\n"
          "        proc_bench_cell=jnp.zeros((batch_size, 3), dtype=i32),\n"
          "        proc_recipe=jnp.full((batch_size,), -1, dtype=i32),\n"
          "        proc_active=zeros_b,\n"
          "        proc_has_fuel=zeros_b,\n"
          "        proc_wait_ticks=jnp.zeros((batch_size,), dtype=i32),\n"
          "        proc_progress_seconds=zeros_f,\n"
          "        proc_fuel_seconds=zeros_f,\n"
          "        proc_fuel_item=jnp.full((batch_size,), -1, dtype=i32),\n"
          "        proc_fuel_quantity=jnp.zeros((batch_size,), dtype=i32),\n"
          "        proc_fuel_quality=zeros_f,\n"
          "        # the bench's by-product countdown starts at PER_FUEL_ITEMS (ExtraOutput.PerFuelItemsConsumed 2)\n"
          "        proc_extra_countdown=jnp.full((batch_size,), 2, dtype=i32),\n"
          "        proc_input_item=jnp.full((batch_size, 28), -1, dtype=i32),\n"
          "        proc_input_quantity=jnp.zeros((batch_size, 28), dtype=i32),\n"
          "        proc_output_item=jnp.full((batch_size, 4), -1, dtype=i32),\n"
          "        proc_output_quantity=jnp.zeros((batch_size, 4), dtype=i32),\n",
          "11p", _PROCESSING, "every bench leaf starts idle (no recipe, no fuel, no stacks) at reset"),
    Patch("diagram_fields", "combat/types.py",
          "    agent_locomotion_stamina_regen_delay: Array\n",
          "    agent_locomotion_stamina_regen_delay: Array\n"
          "    # Gen 7 (row 11q, diagram crafting): the queued Armory craft, per lane (diagram_crafting.py).\n"
          "    diagram_recipe: Array\n"
          "    diagram_ticks: Array\n",
          "11q", _DIAGRAM, "the queued diagram craft the diagram_crafting construct reads and writes"),
    Patch("diagram_init", "combat/runtime/reset.py",
          "        agent_locomotion_stamina_regen_delay=zeros_f,\n",
          "        agent_locomotion_stamina_regen_delay=zeros_f,\n"
          "        diagram_recipe=jnp.full((batch_size,), -1, dtype=i32),\n"
          "        diagram_ticks=jnp.zeros((batch_size,), dtype=i32),\n",
          "11q", _DIAGRAM, "no diagram craft is queued at reset"),
    Patch("processing_bench_import", "combat/arsenal/world_actions/crafting.py",
          "\n    INPUT_MODE_NORMAL,\n",
          "\n    BENCH_TYPE_PROCESSING,\n    BENCH_TYPE_DIAGRAM_CRAFTING,\n    INPUT_MODE_NORMAL,\n",
          "11p", _PROCESSING, "the bench type constants the two patches below read (crafting/contract.py)"),
    Patch("processing_bench_types", "combat/arsenal/world_actions/crafting.py",
          "_BENCH_TYPE_NAMES = {\n    \"Crafting\": BENCH_TYPE_CRAFTING,\n",
          "_BENCH_TYPE_NAMES = {\n    \"Crafting\": BENCH_TYPE_CRAFTING,\n"
          "    # Gen 7 (row 11p): the Processing benches (Campfire, Furnace, Salvagebench, Tannery) are Region bench rows (type 1).\n"
          "    \"Processing\": BENCH_TYPE_PROCESSING,\n",
          "11p", _PROCESSING, "Processing benches become Region bench rows (bench contexts of type 1, tier 0)"),
    Patch("diagram_bench_types", "combat/arsenal/world_actions/crafting.py",
          "_BENCH_TYPE_NAMES = {\n    \"Crafting\": BENCH_TYPE_CRAFTING,\n",
          "_BENCH_TYPE_NAMES = {\n    \"Crafting\": BENCH_TYPE_CRAFTING,\n"
          "    # Gen 7 (row 11q): the Armory (Bench.Type DiagramCrafting) is a Region bench row (type 2).\n"
          "    \"DiagramCrafting\": BENCH_TYPE_DIAGRAM_CRAFTING,\n",
          "11q", _DIAGRAM, "the Armory becomes a Region bench row (bench contexts of type 2, tier 0)"),
    Patch("processing_executor_import", "combat/arsenal/world_actions/executor.py",
          "import numpy as np\n",
          "import numpy as np\n\nfrom hytalegym.sim.gen7.mechanics.processing_benches import start as processing_benches_start\n",
          "11p", _PROCESSING, "the processing request function the craft verb calls"),
    Patch("diagram_executor_import", "combat/arsenal/world_actions/executor.py",
          "import numpy as np\n",
          "import numpy as np\n\nfrom hytalegym.sim.gen7.mechanics.diagram_crafting import start as diagram_crafting_start\n",
          "11q", _DIAGRAM, "the diagram request function the craft verb calls"),
    Patch("processing_executor_start", "combat/arsenal/world_actions/executor.py",
          "            accepted = accepted.at[:, ACTION_SURFACE_VERB_CRAFT].set(\n"
          "                craft_accepted\n"
          "            )\n",
          "            # Gen 7 (rows 11p, 11q): a processing request starts its bench, and a diagram craft is queued; both work from\n"
          "            # craft_inventory_before and replace the zero-time commit on their lanes (processing_benches.start). Both scan the\n"
          "            # block candidates (the executor's `candidates`, not the recipe rows): bench_of_type reads visible_position.\n"
          "            craft_accepted, next_state = processing_benches_start(\n"
          "                execute_region_fieldcraft,\n"
          "                next_state,\n"
          "                craft,\n"
          "                craft_inventory_before,\n"
          "                candidates,\n"
          "                next_world,\n"
          "                crafting_assets,\n"
          "                inventory_layout,\n"
          "            )\n"
          "            craft_accepted, next_state = diagram_crafting_start(\n"
          "                execute_region_fieldcraft,\n"
          "                next_state,\n"
          "                craft._replace(accepted=craft_accepted),\n"
          "                craft_inventory_before,\n"
          "                candidates,\n"
          "                next_world,\n"
          "                crafting_assets,\n"
          "                inventory_layout,\n"
          "            )\n"
          "            accepted = accepted.at[:, ACTION_SURFACE_VERB_CRAFT].set(\n"
          "                craft_accepted\n"
          "            )\n",
          "11p", _PROCESSING, "a processing request is accepted when its bench starts (in reach, idle, fuel loaded); a refused request keeps its inventory"),
    # Row 5a, hit interrupt (construct hit_interrupt, part B): the eligible-role flags ride the arsenal config (built from
    # entity_role_ids) and the mechanics state (copied at reset). The config field defaults to None, so the config pytree is
    # unchanged until the builder hook sets it; the state field is all False until a reset with eligible roles.
    Patch("hit_interrupt_config_field", "combat/arsenal/schema/types.py",
          "    legal_observation: bool = False\n",
          "    legal_observation: bool = False\n"
          "    # Gen 7 (row 5a, hit interrupt): per-entity eligible-role flags from entity_role_ids; None without the builder hook.\n"
          "    hit_interrupt_eligible: object = None\n",
          "5a", "tests/sim/test_hit_interrupt.py",
          "the config field the reset hook reads; None keeps the config pytree (gen 5, and gen 7 with the switch off)"),
    Patch("hit_interrupt_state_field", "combat/mechanics/schema/types.py",
          "    failure_bits: Array\n",
          "    failure_bits: Array\n"
          "    # Gen 7 (row 5a, hit interrupt): per-entity eligible-role flags, copied from the config at reset (bool [batch, entity]).\n"
          "    hit_interrupt_eligible: Array\n",
          "5a", "tests/sim/test_hit_interrupt.py",
          "the state field the clock hook reads; gen 5 does not apply this patch; gen 7 with the switch off holds all False"),
    Patch("hit_interrupt_state_init", "combat/mechanics/factory.py",
          "        guard_active=jnp.zeros(entity_shape, dtype=jnp.bool_),\n",
          "        guard_active=jnp.zeros(entity_shape, dtype=jnp.bool_),\n"
          "        hit_interrupt_eligible=jnp.zeros(entity_shape, dtype=jnp.bool_),\n",
          "5a", "tests/sim/test_hit_interrupt.py",
          "the eligible flags start False at construction; the reset hook sets them from the config"),
    Patch("hit_interrupt_loadout_argument", "combat/arsenal/runtime.py",
          "            engagement_target_id=decision_targeting.engagement_target_id,\n        )\n",
          "            engagement_target_id=decision_targeting.engagement_target_id,\n"
          "            # Gen 7 (row 5a, hit interrupt): the agent's loadout maps each fired event to its attack row.\n"
          "            loadout=config.loadout,\n        )\n",
          "5a", "tests/sim/test_hit_interrupt.py",
          "the loadout keyword of process_direct_events; the hit wrapper strips it when the switch is off"),
    Patch("perception_role_config_field", "combat/arsenal/schema/types.py",
          "    legal_observation: bool = False\n",
          "    legal_observation: bool = False\n"
          "    # Gen 7 (task #130, perception_v2): per-entity semantic role ids [batch, entity] from the role names; None without\n"
          "    # the perception role hook (perception_v2 off).\n"
          "    entity_role_semantic_ids: object = None\n",
          "PV2", "tests/sim/test_perception_v2.py",
          "the config field the perception channel reads (PERCEPTION_V2.md section 6 seam, not a FIDELITY-0.7.2.md row); None keeps the config pytree (gen 5, and gen 7 with the switch off)"),

)

_MECHANICS = "hytalegym.sim.gen7.mechanics"
HOOKS = (
    Hook("crafting_augment_assets", "combat/arsenal/world_actions/crafting.py", "compile_region_fieldcraft_assets",
         f"{_MECHANICS}.crafting_augment_tags:compile_hook", "crafting_augment_tags", "active", "11x", _AUGMENT,
         "the scenario's grants (bench_augment_tags=) as one bit mask per bench row, from the 0.7.2 tag table; default none"),
    Hook("crafting_augment_gate", "crafting/runtime.py", "_recipe_precondition",
         f"{_MECHANICS}.crafting_augment_tags:gate_hook", "crafting_augment_tags", "active", "11x", _AUGMENT,
         "a requirement slot whose tags the context's bench does not grant is cleared before the engine's precondition"),
    Hook("hit_interrupt_roles", "combat/arsenal/runtime.py", "arsenal_runtime_config",
         f"{_MECHANICS}.hit_interrupt:construct_roles", "hit_interrupt", "active", "5a",
         differential="tests/sim/test_hit_interrupt.py: the eligible flags follow entity_role_ids (the measured role on the target: [False, True]; another role: all False)",
         note="sets ArsenalRuntimeConfig.hit_interrupt_eligible; the measured roles are the default, the 22-role static set is a comment"),
    Hook("hit_interrupt_reset", "combat/arsenal/runtime.py", "reset_arsenal_batch",
         f"{_MECHANICS}.hit_interrupt:construct_reset", "hit_interrupt", "active", "5a",
         differential="tests/sim/test_hit_interrupt.py: the state flags equal the config flags after reset",
         note="copies the config flags into CombatMechanicsState.hit_interrupt_eligible"),
    Hook("hit_interrupt_clock", "combat/arsenal/runtime.py", "step_arsenal_batch",
         f"{_MECHANICS}.hit_interrupt:construct_clock", "hit_interrupt", "active", "5a",
         differential="tests/sim/test_hit_interrupt.py: Eligible from tick 42 and refreshed on the 0.25 s grid (45, 53, 60) on the render",
         note="before each step: Eligible for the eligible rows from tick 42, refreshed every 0.25 s; Enraged when a flinch ends with Recovery_Pending"),
    Hook("hit_interrupt_flinch", "combat/arsenal/effects/events.py", "process_direct_events",
         f"{_MECHANICS}.hit_interrupt:construct_hits", "hit_interrupt", "active", "5a",
         differential="tests/sim/test_hit_interrupt.py: a hit before tick 42 adds nothing; the first hit at or after 42 adds Stunned, Flinch and Reaction A in one tick; removal after 15, 21 and 120 ticks; a non-eligible role is never affected",
         note="a requested direct hit on the target from a flagged attack of the agent's loadout applies the gate chain and effects (landed approximated)"),
    Hook("mana_pool", "combat/mechanics/factory.py", "default_mechanics_rules",
         f"{_MECHANICS}.mana_pool:construct", "mana_pool", "active", "7a",
         "native_duel refill from 0 (live 0.7.2 bridge, native_agent_mana): +1 per 8.95 world ticks = 0.2982 s, "
         "against the 0.3 s bar within 5 %; JAX: tick_resources from 0 at 1/30 s (rendered engine)",
         "maximum 100 from stats/stat:Mana (the factory default); the regeneration interval 0.3 s replaces the gen-5 "
         "0.2 s literal (factory.py line 134); the loadout's maximum and initial come from mana_pool_loadout; rune casts "
         "spend this pool only once rune_casting is built"),
    Hook("mana_pool_loadout", "combat/arsenal/runtime.py", "arsenal_runtime_config",
         f"{_MECHANICS}.mana_pool:construct_loadout", "mana_pool", "active", "7a",
         "tests/sim/test_mana_pool.py: the agent reads Mana 100 / 100 at reset on the single-agent env and the duel's reset "
         "path (both were 0 / 0 before); gen 5 reads 0 / 0; a loadout's own mana maximum is kept",
         "fills the input loadout's Mana maximum and initial (stats/stat:Mana) where it has none; the rules follow the "
         "loadout (arsenal/factory.py mechanics_rules_for_loadout), so the reset and the observation read 100 / 100"),
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
    # Row 1f, automatic step-up (lane_traversal, 2026-10-23). Listed before npc_jump_arc_fall_speed so it is outermost on
    # _tick_agent_geometry_motion (hooks on one function: the first listed is outermost); the construct unwraps to read globals.
    # Not differential-gated against the server: no 0.7.2 up-step fixture exists.
    Hook("auto_step_up", "combat/runtime/agent.py", "_tick_agent_geometry_motion",
         f"{_MECHANICS}.auto_step_up:construct", "auto_step_up", "active", "1f",
         "tests/sim/test_auto_step_up.py: a grounded npc_walk move blocked by a one-block ledge lifts and settles on the ledge top "
         "(the raised move, the settle and the accept mask); a two-block wall, a blocked headroom and a free move are unchanged; "
         "the switch-off render equals the committed render leaf by leaf",
         note="max_climb_height from the agent walk (1.3, 0.5.7 ruleset; the 0.7.2 learner role carries none, INFERRED); the "
         "construct returns the original function when the switch is off"),
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
    Hook("crossbow_scalar_variant", PROFILE_JAX, "_scalar_variant_profile",
         f"{_MECHANICS}.crossbow_variant:construct", "crossbow_scalar_variant", "active", "6c",
         "Ancient Steel scalar contacts (FIDELITY-0.7.2.md section (ag)): the native 0.7.2 combo and signature rows on two "
         "seeds apply 11, 11, 14 for the table's 10.5, 10.5, 14 and 82 for Big Arrow (tests/fidelity/combat/projectiles/"
         "crossbow); offline tests/sim/test_crossbow_variant.py (gen-7 triple, gen-5 original, iron triple reads back "
         "unchanged at 10, 13, 78)",
         "The table triple (10.5, 14, 82) replaces the 0.5.7 triple (16, 42, 123) on the gen-7 render only; the applied "
         "integers follow from the kernel's half-up rule (mechanics/runtime/kernel.py:320-322), which this hook does not "
         "touch; durability stays with weapon_durability (row 8d); non-crossbow rows pass through"),
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
    # Row 1a, NPC sprint is run (construct npc_walk_sprint_as_run): a sprint request under npc_walk is a run request for the
    # learner. Listed before the landing hook: hooks on one function apply in tuple order and the first applied is the
    # outermost, so landing_walk (which reads original.__globals__) stays innermost (sim/PORTING.md, hook-chaining note).
    Hook("npc_walk_sprint_learner", "combat/runtime/step.py", "_apply_action",
         f"{_MECHANICS}.npc_sprint:construct", "npc_walk_sprint_as_run", "active", "1a",
         "live receipt pending (bridge 6DFD8682, 2026-10-21: NPC sprint measures 5.0 against 6.37); sim side: "
         "tests/sim/test_npc_sprint.py (sprint and run trajectories bit-identical on npc_walk)",
         "a requested sprint under npc_walk is a run for actor zero: same multipliers on every axis, no stamina drain and "
         "no regen block (stamina regenerates as for run); the player model keeps its sprint (1.273)"),
    Hook("npc_jump_carry", "combat/runtime/step.py", "_apply_action",
         f"{_MECHANICS}.npc_jump_carry:construct", "npc_jump_carry", "active", "1c", _JUMP_CARRY,
         "a running npc_walk jump (take-off speed > 0) keeps its take-off horizontal velocity: the engine's force path "
         "(air factor 0.9216, ground factor 0.6724 per tick, no steering, move speed held) and a 0.45 factor on L+4 and a "
         "release on L+5 (the live list: tests/sim/test_jump_carry.py); a standing jump is not carried; a knockback "
         "during the carry ends it; the L+4 factor 0.45 is fitted to one live value (0.203), its mechanism is not "
         "identified; the carry is inferred for a walk-ramp take-off and for re-jumps during a carry"),
    Hook("landing_walk_resume", "combat/runtime/step.py", "_apply_action",
         f"{_MECHANICS}.landing_walk:construct", "landing_walk_resume", "active", "1c", _LANDING_WALK,
         "after an accepted jump, holds forward, back, left and right for the two grounded ticks after touchdown "
         "(npc_walk only; with npc_jump_carry on, only a standing jump, since a running jump is carried); a landing "
         "without a jump is not held; the jump request is not held; the native pause before "
         "a re-jump (F1c) is the existing patch npc_rejump_after_landing; a knockback landing is not held here (the "
         "engine's knockback control lock holds one tick natively: the render walks on L+2, as live; "
         "tests/sim/test_landing_walk_knockback.py)"),
    Hook("bench_tier_assets", "combat/arsenal/world_actions/crafting.py", "compile_region_fieldcraft_assets",
         f"{_MECHANICS}.bench_tiers:construct", "crafting_bench_tier", "active", "11b", _BENCH_TIERS,
         "the bench rows' tier from the scenario (bench_tiers=: an int for every bench, or a {bench id: tier} mapping; a fresh "
         "bench, 1, by default); gen 5 keeps the zero fill"),
    Hook("crafting_memories_assets", "combat/arsenal/world_actions/crafting.py", "compile_region_fieldcraft_assets",
         f"{_MECHANICS}.crafting_memories:construct", "crafting_memories", "active", "11m", _MEMORIES,
         "the scenario's memories level (memories_level=, 1 to 6; default 1); a caller passes the bridge's "
         "native_memories_level when the bridge reports one (none does yet)"),
    Hook("timed_crafting_admission", "combat/arsenal/world_actions/crafting.py", "_lossless_region_crafting",
         f"{_MECHANICS}.timed_crafting:admission", "crafting_time", "active", "11j", _TIMED_CRAFTING,
         "a timed recipe (time_seconds > 0) passes the zero-time gate with its time removed; its other gates are the original's"),
    Hook("timed_crafting_execution", "combat/arsenal/world_actions/crafting.py", "execute_region_fieldcraft",
         f"{_MECHANICS}.timed_crafting:execution", "crafting_time", "active", "11j", _TIMED_CRAFTING,
         "an accepted timed craft is accepted but not committed and not finished; the job starts in the executor patch"),
    Hook("timed_crafting_executor", "combat/arsenal/world_actions/executor.py",
         "make_region_block_action_surface_executor",
         f"{_MECHANICS}.timed_crafting:executor", "crafting_time", "active", "11j", _TIMED_CRAFTING,
         "each executor step first advances the running jobs (first progress tick: inputs out; completion: output; queued start; "
         "reach cancel with refund), then runs the verbs as before"),
    Hook("processing_admission", "combat/arsenal/world_actions/crafting.py", "_lossless_region_crafting",
         f"{_MECHANICS}.processing_benches:admission", "processing_benches", "active", "11p", _PROCESSING,
         "a processing requirement (bench type 1) passes the zero-time gate as an ordinary bench requirement; the timed hook keeps the time"),
    Hook("diagram_admission", "combat/arsenal/world_actions/crafting.py", "_lossless_region_crafting",
         f"{_MECHANICS}.diagram_crafting:admission", "diagram_crafting", "active", "11q", _DIAGRAM,
         "a diagram requirement (bench type 2) passes the zero-time gate as an ordinary bench requirement (the 19 Armory rows also name a type-0 bench)"),
    Hook("processing_executor", "combat/arsenal/world_actions/executor.py", "make_region_block_action_surface_executor",
         f"{_MECHANICS}.processing_benches:executor", "processing_benches", "active", "11p", _PROCESSING,
         "each executor step first advances the processing benches (the start wait, fuel burn, progress, by-products, completion)"),
    Hook("diagram_executor", "combat/arsenal/world_actions/executor.py", "make_region_block_action_surface_executor",
         f"{_MECHANICS}.diagram_crafting:executor", "diagram_crafting", "active", "11q", _DIAGRAM,
         "each executor step first runs the queued diagram commit (two ticks after the request)"),
    Hook("guard_reentry_cooldown", "combat/mechanics/runtime/motion.py", "apply_defense_commands",
         f"{_MECHANICS}.guard_cooldown:construct", "guard_reentry_cooldown", "active", "5d",
         "section (af) guard reopen receipts: 24 runs, reopen on claim + 10 (17) or + 11 (7) on the loaded server; the "
         "gen-7 clock counts 0.35 s from the claim at 1/30 s, reopening on claim + 11",
         "a new guard claim is refused while the countdown runs (0.35 s from the claim, one nominal tick per step); an "
         "active or held guard is unaffected; gen 5 returns the original"),
    Hook("ability_start_split", "combat/arsenal/programs/ability.py", "start_abilities",
         f"{_MECHANICS}.ability_start_split:construct", "ability_start_split", "active", "5f",
         "section (af) kunai, iron sword and shortbow receipts: claim on tick 1, chain start on tick 2; kunai finish 11, iron "
         "sword finish 14; the recharge is unchanged at the nominal 1.0 s from the claim",
         "each accepted ability chain starts one drain after its claim (scheduler offset -1 on the admitted rows); charges "
         "are spent and recharged on the claim; gen 5 returns the original"),
    Hook("replace_default_hop", PROFILE_JAX, "_write_profile",
         f"{_MECHANICS}.replace_default_hop:construct", "replace_default_hop", "active", "5f",
         differential="tests/sim/test_replace_default_hop.py: with the switch on the iron sword Swing_Left, Swing_Right and "
         "Swing_Down, the dagger swings and stabs, the mace swings and the battleaxe swings hit one step later than the switch-off "
         "render (sword Left 9 against 8; the table in the test), their chain finishes are unchanged, and the other attacks are "
         "identical; the switch-off render equals the committed render leaf by leaf",
         note="wraps _write_profile: the server-selector event time of the 13 Simple -> Replace -> Selector attacks moves one tick "
         "(1/30 s) later, so the selector window and sweep move with it; the ability duration is unchanged; _ability is left to "
         "the guard hook; gen 5 returns the original"),
    Hook("shortbow_reset_ammo", "combat/arsenal/profiles/hytale_0_5_7.py", "hytale_0_5_7_loadouts",
         f"{_MECHANICS}.shortbow_reset_ammo:construct", "shortbow_reset_ammo", "active", "5f",
         "section (af) shortbow receipts (seeds 570057, 913771): native ammo 0 on ticks 0 and 1, 1 from tick 2; differential "
         "mismatch on the reset row only",
         "the shortbow's initial RESOURCE_AMMO is 0 (the gen-5 phase already gives 0 on tick 1 and 1 from tick 2); gen 5 "
         "returns the original"),
    Hook("shield_bash_root", "combat/arsenal/programs/ability.py", "_ability_root_admission_mask",
         f"{_MECHANICS}.shield_bash_fork:root_admission", "shield_bash_root", "active", "5d",
         "section (x) and its addendum: the bash advertised on ticks 3 to 12 of a 12-tick hold and on no other tick "
         "(live 0.7.2, seeds 570057 and 1553616197). Render replay tests/sim/test_shield_bash_fork.py (ticks 1 to 16): "
         "legal on ticks 3 to 12 only; a request refused on tick 2 (held, not active) and on ticks 14 and 15 (after "
         "release); accepted on tick 4 while held and active. A request on the release tick 13 is admitted by stock and "
         "by this construct and is not asserted. No live learner-row run on the render (test_dynamic_learner_row_world_"
         "inputs.py compares HEAD JAX only)",
         "a guard fork root (the Iron Shield bash) is legal and admissible only while the guard is held and active "
         "(guard_active); gen 5 returns the original mask, which offers it from the held level (ticks 1 and 2 included)"),
    Hook("reset_head_heading", "combat/arsenal/runtime.py", "reset_arsenal_batch",
         f"{_MECHANICS}.reset_head_heading:construct", "reset_head_heading", "active", "5g",
         "section (ak) on the live 0.7.2 bridge (seeds 570057 and 913771): a target spawned facing the agent (180 degrees) "
         "has HeadRotation equal to its body yaw at tick 0 and keeps it with no input; the reset zeroed both heads. Render "
         "test tests/sim/test_reset_head_heading.py: entity_yaw [0, 180] gives agent head 0 and target head 180 at tick 0, "
         "and the returned observation equals observe_batch of the state; entity_yaw None and gen 5 are unchanged",
         "an Arsenal reset given entity_yaw (entity-position path) seeds agent_head_yaw and target_head_yaw from the stored "
         "body yaw (pitch unchanged) and rebuilds the returned observation from that state; gen 5 returns the original"),
    Hook("ability_ground_continuation", "combat/arsenal/programs/ability.py", "tick_ability_programs",
         f"{_MECHANICS}.ability_ground_continuation:construct", "ability_ground_continuation", "active", "5h",
         differential="tests/sim/test_ability_ground_continuation.py: the charged battleaxe Landing (slot 5) starts on the tick the "
         "Downstrike finishes (cont 5 with active -1) and lands 29 at a measured tick; a premature resume starts nothing; the "
         "sword thrust and mace charged hit are unchanged; the switch-off render equals the committed render",
         note="wraps tick_ability_programs: a child starts when the parent finished this tick and the continuation resolved to a "
         "child; the child start resets the scheduler fields start_abilities resets and keeps the requesting root; charges, "
         "cooldowns and the regen delay are untouched; gen 5 returns the original"),
    Hook("status_clock", "combat/mechanics/runtime/statuses.py", "tick_statuses",
         f"{_MECHANICS}.status_clock:construct", "status_clock", "active",
         differential="tests/sim/test_status_clock.py: a 0.5 s status is active for 15 ticks on the gen-7 render and 16 with the switch "
         "off; the rendered clock matches the construct on every row; the switch-off render equals the committed render leaf by leaf; "
         "tests/sim/test_hit_interrupt.py keeps 15, 21 and 120 steps",
         note="wraps tick_statuses: after the stock decrement a status stays active while more than half a tick remains (round(D / "
         "tick)); only the active flags change; gen 5 returns the original"),
    Hook("perception_v2", "combat/observation/v3/encoding/encoder.py", "encode_learner_observation_v3",
         f"{_MECHANICS}.perception_v2:construct", "perception_v2", "active",
         differential="tests/sim/test_perception_v2.py: channel layout, row gate (view sector, occluder), ids, inventory "
         "floats, body yaw; the switch-off identity gate and the rollout fingerprints are unchanged",
         note="sim/gen7/PERCEPTION_V2.md: opt-in beside the v3 fields; the role id is the target's NPC role (perception_roles)"),
    Hook("perception_roles", "combat/arsenal/runtime.py", "arsenal_runtime_config",
         f"{_MECHANICS}.perception_v2:construct_roles", "perception_v2", "active",
         differential="tests/sim/test_perception_v2.py: the config's role ids for the matchup default, the duel's roles= (swapped), "
         "an empty role, and a player-backed target (the channel's role id reads 0)",
         note="sets ArsenalRuntimeConfig.entity_role_semantic_ids (semantic_id of each role name) from entity_role_ids or the "
         "matchup default; applied after hit_interrupt_roles, which replaces its measured roles with '' before the original"),
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
