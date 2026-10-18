"""The generation profile: version, data directory, construct switches.

A generation is a profile, its extracted content tables, and the constructs
for mechanics that exist only in some versions. Every behaviour that differs
between generations is one field of :class:`Constructs`, implemented once in
the engine behind that field; the engine never branches on a version string.
Defaults are gen-5 behaviour (what ``hytalegym.jax.combat`` does today), so a
field a profile does not set keeps today's rollouts. Construct designs, inputs
and differentials: ``sim/gen7/mechanics_072.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Constructs:
    """One switch per version-dependent behaviour (gen-5 values by default)."""

    # Line of sight through blocks. "non_transparent_blocks": any block whose
    # Opacity is not Transparent stops sight (0.5.x PositionCache).
    # "detail_hitbox": 0.7.2 passes non-transparent blocks unless the ray hits
    # one of their detail hitboxes. Owner: sim/gen7/mechanics/line_of_sight.py (P2).
    line_of_sight: str = "non_transparent_blocks"
    # Dodge choices the locomotion action head exposes. Not a new mechanic: the
    # dodge assets are identical in 0.5.10 and 0.7.2 (Left/Right chains only,
    # impulse 13, 2 stamina, 0.25 s invulnerable) and the engine already
    # executes left/right only; the gen-5 head also carries forward/back, which
    # are authored no-ops. Gen 7 shrinks the head. Owner: dodge.py (P3).
    dodge_action_head: tuple[str, ...] = ("left", "right", "forward", "back")
    # Melee hits that inherit DamageEntityParent_Hit_Interrupt cancel the
    # target's attack, stun it and apply a flinch. Owner: hit_interrupt.py (P3).
    hit_interrupt: bool = False
    # A regenerating mana pool spent by casts (Stats/Mana.json Max > 0).
    # Owner: sim/gen7/mechanics/mana.py (P3).
    mana_pool: bool = False
    # Rune items cast abilities (fireball, wind strike, chain hook, ground slam,
    # charged shot, enrage, poison imbue...). Owner: mana.py (P3).
    rune_casting: bool = False
    # Tool_Hookshot: fire, attach, pull. Owner: hookshot.py (P3, on demand).
    hookshot: bool = False
    # Climbing on blocks with MovementSettings.IsClimbable. Owner: climbing.py.
    climbing: bool = False
    # Door model. "world_v0_5": today's World door semantics (doors source
    # audit); a 0.7.2 model is added only if its differential differs.
    doors: str = "world_v0_5"
    # Swimming model. "fluidfx_v0_5": today's FluidFX movement medium.
    swimming: str = "fluidfx_v0_5"
    # Glider items (gliding movement state). Owner: gliding.py (on demand).
    gliding: bool = False
    # Mounts (MovementConfig/Mount.json, new in 0.7.2). Owner: riding.py.
    riding: bool = False
    # NPC set-velocity knockback: "zero_then_add" (0.5.10 zeroes then adds) or
    # "role_set_velocity" (0.7.2 Role.processSetVelocityInstruction, ~23x for
    # set-type NPC knockback; never reaches the bridge-driven agent).
    npc_velocity_model: str = "zero_then_add"
    # NPC learner (npc_walk motion model) first-step jump velocity: "player_force" (HEAD: the player's
    # PLAYER_JUMP_FORCE 11.8) or "native_npc" (sqrt(2 * gravity floor * jump height parameter) = 5.099,
    # the 0.7.2 NPC first-step vertical speed). Owner: sim/gen7/mechanics/npc_jump.py (phase 3b, row 1c).
    npc_walk_jump_velocity: str = "player_force"
    # NPC learner (npc_walk) jump arc after the impulse: "constant_gravity" (HEAD: g over the current tick) or
    # "physics_math" (0.7.2 MotionControllerWalk.computeNewFallSpeed: the previous tick's duration and
    # PhysicsMath's cubic drag on the speed gained since takeoff). Owner: sim/gen7/mechanics/npc_jump_arc.py
    # (row 1c, F1c).
    npc_jump_arc: str = "constant_gravity"
    # Swing_Left and Swing_Right damage of every scalar dagger variant: "shared_rows" (the 0.5.7 tuples of the
    # shared weapon_variants module, HEAD) or "gen7_table" (the 0.7.2 item rows). Owner: sim/gen7/mechanics/
    # dagger_damage.py (phase 4, item 5c).
    dagger_swing_damage: str = "shared_rows"
    # Guard raised during a melee attack: "guard_root" (the guard root's Interrupting [Primary] cancels the attack at
    # any time, 0.5.x) or "attack_root" (0.7.2: the attack root's InterruptedBy [Secondary] cancels it only while
    # the root's RunTime runs). Owner: sim/gen7/mechanics/guard_window.py (row 5d).
    guard_cancel_window: str = "guard_root"
    # Interaction type names the bridge gets for a profile's ability slots: "profile_sets" (the 0.5.7 membership
    # sets of the profile module) or "item_interactions" (a bound root takes its type from the 0.7.2 item's
    # Interactions map, which moves the crossbow reload from Ability3 to Ability4). Owner: sim/gen7/mechanics/
    # native_bindings.py (row 6c).
    native_binding_types: str = "profile_sets"
    # Region crafting at an ordinary bench: "fieldcraft_context" (HEAD: every candidate commits under the Fieldcraft
    # context, so a bench-only recipe is offered and then refused, and more than 16 legal recipes fail the actor
    # row closed) or "recipe_bench" (commit under a bench the recipe names; keep the first 16 legal recipes in
    # table order). Owner: sim/gen7/mechanics/bench_crafting.py (row 11, stage 0).
    bench_crafting: str = "fieldcraft_context"
    # Region crafting inputs: "concrete_only" (HEAD: a recipe enters only when every input names an item) or
    # "resource_types" (a resource-type input that excludes exactly its recipe's outputs, the server's
    # applyOutputExclusions, with slot resource types from the archive's items). Owner: sim/gen7/mechanics/
    # crafting_materials.py (row 11, stage 1).
    crafting_materials: str = "concrete_only"
    # Walk controller while descending: "free_fall" (0.5.x) or
    # "minimum_descent_rate" (0.7.2 code change, unmeasured). P2.
    walk_descent: str = "free_fall"
    # Player slope and wall limits (MovementConfig MaxSlopeAngleDegrees /
    # MaxWallAngleDegrees, keys new in 0.7.2: 30 and 15). No 0.7.2 class reads them (jar scan, 2026-10-18),
    # so gen 7 sets False (decision, phase 3b); slope_limits.py stays a stub.
    slope_limits: bool = False
    # 0.7.2 drops a block's health entry on any block write. Owner: block_health.py.
    block_health_reset_on_write: bool = False
    # Weapon durability maximum and loss per hit: "shipped_json" (HEAD: hytale_0_5_7_item_programs_v1.json) or
    # "gen7_table" (the 0.7.2 durability table). Owner: sim/gen7/mechanics/durability.py (row 8d).
    weapon_durability: str = "shipped_json"
    # Walk after a landing: "at_once" (HEAD: the walk resumes on the first tick after touchdown) or "third_tick" (0.7.2
    # NPC walk: after an accepted jump the planar walk is held for the two ticks after touchdown). Owner:
    # sim/gen7/mechanics/landing_walk.py (row 1c, F1c).
    landing_walk_resume: str = "at_once"
    # Region bench tier: "zero_fill" (HEAD: every Region bench row is tier 0 and only tier-0 requirements are admitted) or
    # "placed_bench" (a fresh bench is tier 1, tiers 1 to 8 settable; requirements up to tier 8 admitted). Owner:
    # sim/gen7/mechanics/bench_tiers.py (row 11b, stage 2).
    crafting_bench_tier: str = "zero_fill"
    # Region memories level: "level_one" (HEAD: every recipe needs level 1 or less) or "recorded_count" (the universe-wide
    # recorded count sets the level, 1 to 6; recipes up to level 6 admitted). Owner: sim/gen7/mechanics/crafting_memories.py
    # (row 11m, stage 4).
    crafting_memories: str = "level_one"
    # Region timed bench crafts: "instant" (HEAD: a craft commits on its request tick) or "timed_job" (a recipe with
    # time_seconds > 0 starts a job: inputs leave on the next tick, the output at completion, cancelled out of reach).
    # Owner: sim/gen7/mechanics/timed_crafting.py (row 11j, stage 3).
    crafting_time: str = "instant"


@dataclass(frozen=True)
class StepDuration:
    """The engine step duration, as content (owner ruling 2026-10-17: 30 Hz).

    Gen 5 draws one of ``timing_profile_count`` per-lane profiles at reset:
    profile 0 always steps ``nominal_seconds``; profiles 1 and 2 step
    ``loaded_seconds`` on alternating ticks (``jax/combat/runtime/math.py``
    ``_motion_delta``; ruleset ``engine`` block). Gen 7 sets one value, 1/30 s.
    Every site that reads these is listed in ``redirects.json`` (``step_duration``).
    """

    nominal_seconds: float = 0.03333333333333333
    loaded_seconds: float = 0.045
    timing_profile_count: int = 3


@dataclass(frozen=True)
class SimProfile:
    """Everything the engine needs to know about one game generation."""

    generation: str
    certified_version: str
    release_line: tuple[int, int]
    data_dir: Path
    constructs: Constructs = field(default_factory=Constructs)
    step_duration: StepDuration = field(default_factory=StepDuration)
    # Hand-maintained values that differ from the extracted tables but must be
    # kept (gen 5: reproduce today's rollouts byte for byte); None for none.
    calibration_overlay: Path | None = None
    deprecated: bool = False

    def server_profile(self):
        """The bridge-side ``ServerProfile`` of the same release line."""

        from hytalegym.server_versions import server_profile

        return server_profile(self.certified_version)
