"""Gen-7 profile: Hytale 0.7.2, bound to ``server_versions.v0_7``.

Data: ``data/`` extracted from the 0.7.2 ``Assets.zip`` (``data/MANIFEST.json``).
Constructs: set to what 0.7.2 does. A switch whose construct is still a stub in
``mechanics/`` raises ``NotImplementedError`` when the engine asks for it, so a
gen-7 run cannot silently fall back to gen-5 behaviour (see
``mechanics_072.md``: construct, inputs, differential, phase).
"""

from __future__ import annotations

from pathlib import Path

from hytalegym.sim.common.profile import Constructs, SimProfile, StepDuration

_HERE = Path(__file__).resolve().parent

#: Content identity of ``data/`` (no overlay) for the contract stamp;
#: ``tests/sim/test_registry.py`` fails if the data drifts.
CONTENT_SHA256 = "FE91CAEFCDF22E9C3AEEB396A8010934FA31E5DEC32217DC4FA394B08FC6BA53"

PROFILE = SimProfile(
    generation="gen7",
    certified_version="0.7.2",
    release_line=(0, 7),
    data_dir=_HERE / "data",
    constructs=Constructs(
        line_of_sight="detail_hitbox",
        dodge_action_head=("left", "right"),
        hit_interrupt=True,
        mana_pool=True,
        rune_casting=True,
        hookshot=True,
        climbing=True,
        doors="world_v0_5",
        swimming="fluidfx_v0_5",
        gliding=True,
        riding=True,
        npc_velocity_model="role_set_velocity",
        npc_walk_jump_velocity="native_npc",
        npc_jump_arc="physics_math",
        dagger_swing_damage="gen7_table",
        crossbow_scalar_variant="gen7_table",
        guard_cancel_window="attack_root",
        native_binding_types="item_interactions",
        bench_crafting="recipe_bench",
        crafting_materials="resource_types",
        walk_descent="minimum_descent_rate",
        # 0.7.2 MaxSlopeAngleDegrees / MaxWallAngleDegrees have no server reader (jar scan, 2026-10-18):
        # gen 7 matches gen 5 on this field; the two parameters stay as rendered data (decision, phase 3b).
        slope_limits=False,
        block_health_reset_on_write=True,
        weapon_durability="gen7_table",
        landing_walk_resume="third_tick",
        crafting_bench_tier="placed_bench",
        crafting_memories="recorded_count",
        crafting_time="timed_job",
        processing_benches="ticking_bench",
        diagram_crafting="click_craft",
        # Timing (section (ad), live 0.7.2 receipts): the guard's 0.35 s claim cooldown, the one-drain chain start, and
        # the shortbow's empty reset.
        guard_reentry_cooldown="claim_0_35s",
        ability_start_split="next_drain",
        shortbow_reset_ammo="native_zero",
        # Shield bash offer (section (x), row 5d): the 0.7.2 fork of the held guard is legal only while the guard is active.
        shield_bash_root="held_guard_fork",
        # Head yaw at reset (section (ak), row 5g): both heads start on their body headings when the reset gets entity_yaw.
        reset_head_heading="body_heading",
    ),
    # Owner ruling 2026-10-17: the agent operates at 30 Hz; one step duration.
    step_duration=StepDuration(nominal_seconds=1.0 / 30.0, loaded_seconds=1.0 / 30.0, timing_profile_count=1),
)
