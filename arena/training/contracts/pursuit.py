"""JIT-safe pursuit lesson over the shared two-actor JAX arena."""

from __future__ import annotations

from dataclasses import dataclass
import functools
import hashlib
import json
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from arena.training.skills.program import (
    FactoredActionScope,
    TargetMode,
    TargetMotionProgram,
)
from arena.training.skills.signals import (
    aim_tracking_geometry_signals,
    duel_head_yaw,
    pursuit_transition_signals,
    spatial_tracking_signal_contract_sha256,
    spatial_tracking_signals,
)
from arena.training.contracts.evader_duel import (
    EvaderDuelConfig,
    evader_duel_signals,
)
from arena.training.skills.stage import (
    SkillDuelRoles,
    scripted_target_action_mask,
)
from hytalegym.jax.combat import default_combat_params
from hytalegym.jax.combat.types import (
    GAIT_SPRINT,
    PLAYER_GAIT_FORWARD_SPEED_MULTIPLIERS,
)
from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_ACTION_HEAD_NAMES,
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
)
from hytalegym.jax.combat.observation.v1.schema.contract import (
    NEARBY_ENTITY_RADIUS_BLOCKS,
    SELF_FLOAT_FEATURES,
    TARGET_FLOAT_FEATURES,
)
from hytalegym.jax.training.multi_actor import (
    MultiActorArenaState,
    PolicyActorAssignment,
)


PURSUIT_STAGE_SCHEMA = "arena-recurrent-pursuit-stage-v15"
#: What the learner is told about the opponent. See `PursuitStageConfig`.
PERCEPTION_MODES = frozenset({"omniscient", "field_of_view"})
PURSUIT_DEFAULT_TICKS = 512
PURSUIT_MAXIMUM_TICKS = 1_000
PURSUIT_CONTEXT_SCHEMA = "arena-pursuit-privileged-context-v3"
_TARGET_CONTEXT_FEATURES = (
    "relative_forward",
    "relative_right",
    "relative_up",
    "relative_velocity_forward",
    "relative_velocity_right",
    "relative_velocity_up",
    "planar_distance",
    "distance",
    "bearing_sin",
    "bearing_cos",
)
_TARGET_CONTEXT_INDICES = tuple(
    TARGET_FLOAT_FEATURES.index(name) for name in _TARGET_CONTEXT_FEATURES
)
#: Looked up rather than assumed: the `visible` column moves whenever the
#: target feature block changes, and writing a stale index would silently mark
#: some other feature as the visibility flag.
_VISIBLE_TARGET_INDEX = TARGET_FLOAT_FEATURES.index("visible")


@functools.lru_cache(maxsize=1)
def _sprint_blocks_per_tick() -> float:
    """The fastest closure one environment step can physically produce.

    Derived rather than written down, so it tracks the gait table and the tick
    rate instead of going stale beside them. One environment step is one loaded
    motion tick, which the archived trajectories confirm: maximum per-step
    displacement in lane u0200-l000 was 0.2475 blocks, exactly run speed
    (5.5 b/s) times ``loaded_dt`` (0.045 s).

    Sprint is the top of `PLAYER_GAIT_FORWARD_SPEED_MULTIPLIERS` and is
    forward-only, so this is also the closure rate that is unreachable without
    facing the target.
    """

    params = default_combat_params(microticks=1, target_active=True)
    sprint = PLAYER_GAIT_FORWARD_SPEED_MULTIPLIERS[GAIT_SPRINT]
    return float(params.agent_max_speed) * float(sprint) * float(params.loaded_dt)


_SPRINT_BLOCKS_PER_TICK = _sprint_blocks_per_tick()


def _hash(value: object) -> str:
    return (
        hashlib.sha256(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        )
        .hexdigest()
        .upper()
    )


@dataclass(frozen=True, slots=True)
class PursuitStageConfig:
    """Dense learner-owned reward with a bounded, bootstrapped horizon."""

    maximum_ticks: int = PURSUIT_DEFAULT_TICKS
    minimum_chase_ticks: int = 1
    # One block, so a catch means the learner actually closed to contact rather
    # than merely loitering inside melee-ish range. The earlier 6.0 credited a
    # catch from six blocks out, which a policy can satisfy without ever
    # arriving.
    catch_distance: float = 1.0
    proximity_radius: float = 20.0
    #: Distance at which `proximity` stops paying more for closing further.
    #:
    #: The ramp used to run all the way to `catch_distance`, so the last two
    #: blocks were the most valuable ground on the field and a policy could keep
    #: earning by pressing into contact. Below this floor the term is flat: the
    #: reward for arriving is the same at 3 blocks as at 1, and what happens
    #: inside 3 is decided by `success_reward` on the catch, not by shaping.
    proximity_floor_distance: float = 3.0
    evader_safety_radius: float = 14.0
    progress_scale: float = 4.0
    # The physical ceiling on one tick of closure, in blocks, and the reason it
    # is no longer the hardcoded 0.25 it was through run 29d65816.
    #
    # One environment step is one loaded motion tick (`loaded_dt` = 0.045 s), so
    # a gait's per-tick closure is its forward speed times 0.045: run reaches
    # 5.5 * 0.045 = 0.2475 and sprint reaches 7.0015 * 0.045 = 0.3151. The old
    # clip sat *between* them. Sprinting is 27% faster and costs stamina, and it
    # bought 4.0 * (0.25 - 0.2475) = +1% reward. The policy correctly declined:
    # measured across all 211 steps of archived lane u0200-l000, maximum
    # displacement was 0.2475 and sprint was never selected once, so the clip
    # never actually bound -- it just removed the reason to ever try.
    #
    # 0.32 clears sprint's 0.3151 with a small margin for the diagonal case, so
    # the clip now guards against teleports and physics glitches rather than
    # against the fastest legal gait.
    progress_clip_blocks: float = 0.32
    # Pays for *rate* of closure, and is deliberately convex.
    #
    # A term linear in per-tick radial progress telescopes: summed over an
    # episode it equals scale * (first_distance - last_distance), so closing a
    # gap in ten ticks and in a hundred pay exactly the same. That is the defect
    # already documented for `tracking_gain_scale` below, and it is why raising
    # `progress_scale` alone cannot make the learner hurry. Squaring the
    # normalized rate breaks the telescope: covering ground quickly strictly
    # beats covering the same ground slowly.
    #
    # Because the rate is normalized by `progress_clip_blocks` -- sprint speed --
    # the top of this term is reachable only while sprinting, and sprint is
    # forward-only (`PLAYER_SPRINT_REQUIRES_FORWARD_DOMINANT_TRAVEL`). Facing the
    # target therefore stops being a cosmetic preference and becomes the
    # precondition for the fastest closure, which is the whole point: aim is
    # taught by making it instrumental rather than by paying more for it.
    closing_speed_reward: float = 0.30
    evader_progress_scale: float = 4.0
    evader_close_cost: float = 0.05
    # Hytale's own per-tick rotation ceiling, not an arbitrary throttle. The
    # engine slews yaw toward the desired heading at `agent_turn_speed_degrees`
    # (540 deg/s), and a loaded motion tick is `loaded_dt` = 0.045 s, so the
    # most an actor can physically rotate in one tick is 540 * 0.045 = 24.3
    # degrees (18.0 on a nominal 30 Hz tick). Capping the policy's *requested*
    # delta below that made `maximum_yaw_delta_degrees` -- which measures
    # *physical* yaw change -- report violations the policy never committed.
    maximum_turn_degrees: float = 24.3
    # Off by default: the learner spawns with whatever heading the Region reset
    # gave it and must turn to find its target, which is the harder lesson and
    # the historical behaviour. Enabling it spawns the learner already looking
    # at the target, so aim error starts at zero instead of ~140 degrees that
    # `maximum_turn_degrees` needs many ticks to work off.
    #: Opens the jump head for BOTH actors. Off on a plane, where a jump buys
    #: nothing and an airborne actor cannot accelerate; a curriculum rung with
    #: terrain turns it on. Same value both sides, or the matchup measures the
    #: handicap rather than the policies.
    allow_jump: bool = False
    spawn_facing_target: bool = False
    # Aiming has to be worth roughly what walking is. At 1/45 a full 90 -> 0
    # degree convergence paid 2.0 total, against ~4.0 for closing a single
    # block, and run 077b2caa spent 192 updates with its aim error pinned near
    # 90 degrees while it happily closed distance sideways. At 4/45 the same
    # convergence pays ~8.0, comparable to the distance term it competes with.
    tracking_gain_scale: float = 4.0 / 45.0
    # Line of sight and facing were 0.01 against a 4.0-per-block progress term,
    # so both rounded to nothing in the gradient: over a 75-tick episode at the
    # measured 0.45 visible fraction, sight paid 0.34 in total. These are still
    # deliberately below the progress scale -- they shape *how* the learner
    # closes, they should not pay it to stand still and stare.
    line_of_sight_reward: float = 0.05
    facing_reward: float = 0.05
    facing_cone_degrees: float = 60.0
    # A *level* term, and the reason one is needed: `tracking_gain_scale` below
    # pays the per-tick *change* in aim error, which telescopes over an episode
    # to scale * (aim_first - aim_last). Its +/-45 clip never binds, because the
    # engine caps physical yaw at `maximum_turn_degrees` = 24.3 per tick, so the
    # identity is exact -- verified against all eight archived replays of run
    # 20260817T191558Z-4d7c079e, where reconstructed and telescoped totals agree
    # to three decimals on every lane. A difference term therefore pays nothing
    # for *holding* a good heading, only for ending better than it started, and
    # raising its scale from 1/45 to 4/45 earlier the same day multiplied a
    # telescoping sum by four and changed nothing: measured aim error sat at
    # 77-82 degrees across all 192 updates while the term contributed -0.9% of
    # total shaping.
    #
    # This pays every tick the learner is pointed at the target, falling
    # linearly to zero at `aim_level_degrees`. Graded rather than a second cone
    # so there is a gradient to climb instead of a cliff to sit behind.
    # Raised 0.12 -> 0.18 alongside `closing_speed_reward`. Kept a modest step on
    # purpose: the measured failure is not that aiming pays too little in the
    # abstract but that nothing made it *necessary*, and the closing-speed term
    # above is what supplies the necessity. Paying much more per tick for a held
    # heading without that term is how "stand and stare" becomes attractive --
    # see `stationary_cost`, which had to rise with it.
    aim_level_reward: float = 0.18
    aim_level_degrees: float = 90.0
    #: Same graded law as `aim_level_reward`, but scored on the HEAD instead of
    #: the body. Everything else in this contract measures `combat.yaw`, which
    #: `combat/types.py` 412 documents as the BODY, so until this existed the
    #: head was never rewarded or even reported -- an agent could hold perfect
    #: body facing while looking somewhere else entirely and score full marks.
    #:
    #: Zero by default, which keeps every existing pursuit run bit-identical:
    #: pursuit is a CHASING lesson that happens to reward facing, and adding a
    #: second aim term to it silently would change what those runs optimise. A
    #: TRACKING stage -- no chasing, hold a moving target at range with body and
    #: head -- sets this alongside `aim_level_reward` and zeroes `progress_scale`.
    head_aim_level_reward: float = 0.0
    #: Charged on `stationary`, which is keyed to actual planar
    #: DISPLACEMENT, unlike `stationary_cost` which is keyed to closing.
    #: That distinction is the whole point: a mobile-tracking lesson wants
    #: the learner moving in some direction of its own choosing while its
    #: head stays on the target, and charging it for not CLOSING would
    #: just recreate pursuit. Zero by default so pursuit is unchanged.
    idle_motion_cost: float = 0.0
    facing_away_cost: float = 0.04
    facing_away_degrees: float = 120.0
    approach_reward_margin: float = 0.005
    stationary_deadband_blocks: float = 0.01
    # Must strictly exceed what a motionless learner can still collect, or
    # raising the sight and facing rewards turns "stand and stare at the
    # target" into a positive-reward policy: at 0.05 each that is +0.10 a tick
    # against the old 0.03 cost. Keep this above
    # `line_of_sight_reward + facing_reward` whenever either of those moves.
    # Raised 0.15 -> 0.25 with `aim_level_reward`: a parked learner that happens
    # to be aimed now collects sight (0.05) + facing (0.05) + a full aim level
    # (0.12) = 0.22 a tick, so the old 0.15 would have made standing still and
    # staring positive-reward -- exactly the exploit this cost exists to close.
    # Raised again 0.25 -> 0.32 when `aim_level_reward` went to 0.18, which puts
    # the parked total at 0.28. `closing_speed_reward` needs no allowance here:
    # it is zero for a stationary learner by construction.
    stationary_cost: float = 0.32
    stuck_patience_ticks: int = 8
    # Paid on the closing half of `proximity_radius`, so it rises as the gap
    # shrinks and gives the last few blocks -- the ones that actually produce a
    # catch -- more pull than the open-field approach.
    proximity_reward: float = 0.10
    harmful_turn_cost: float = 0.05
    harmful_turn_margin_degrees: float = 5.0
    # The terminal has to dominate the shaping it cuts short. At 2.0 a catch was
    # worth half a block of progress (4.0/block) while ending the episode and
    # forfeiting every future shaping tick: measured mean episode return was
    # 8.41, so continuing to wander paid about four times what catching did, and
    # the catch rate duly fell after update 96. 50.0 makes the catch worth more
    # than a whole episode of perfect shaping.
    success_reward: float = 50.0
    tick_cost: float = 0.002
    #: What the learner is told about the opponent's position.
    #:
    #: ``"omniscient"`` writes exact relative geometry and marks the target
    #: visible EVERY tick, whatever the actor is looking at. That is the right
    #: default for a lesson that trains combat rather than search -- a policy
    #: that loses its opponent and spends the episode re-acquiring is learning a
    #: different problem, and its reward becomes dominated by visibility luck.
    #:
    #: ``"field_of_view"`` gates those writes on the target actually being
    #: inside the learner's horizontal view sector and within `sensor_range`.
    #: Outside it the target slot is masked and `visible` reads 0, which by the
    #: observation convention means NO READING rather than "at the origin", so
    #: the recurrent state is the only thing carrying where the opponent went.
    #: The learner must either infer it or turn and look.
    #:
    #: This matters most for evasion. Being hard to locate IS the evader's task,
    #: and an omniscient chaser cannot be evaded on open ground -- it tracks
    #: through its own back. Round 19 measured exactly that: a 1.000 catch rate,
    #: which left the evader with zero positive examples and therefore no
    #: gradient.
    perception: str = "omniscient"
    #: Full horizontal view sector in degrees, used only by `field_of_view`.
    #: 120 is a conventional human-ish forward arc; the engine authors a real
    #: per-actor `view_sector_full_angle_radians`, and this is the training
    #: contract's own knob rather than a claim about that value.
    view_sector_degrees: float = 120.0
    schema: str = PURSUIT_STAGE_SCHEMA

    def __post_init__(self) -> None:
        if isinstance(self.maximum_ticks, bool) or not isinstance(
            self.maximum_ticks, int
        ):
            raise TypeError("maximum_ticks must be an integer")
        if not 2 <= self.maximum_ticks <= PURSUIT_MAXIMUM_TICKS:
            raise ValueError(f"pursuit stages require 2..{PURSUIT_MAXIMUM_TICKS} ticks")
        if isinstance(self.minimum_chase_ticks, bool) or not isinstance(
            self.minimum_chase_ticks, int
        ):
            raise TypeError("minimum_chase_ticks must be an integer")
        if not 1 <= self.minimum_chase_ticks < self.maximum_ticks:
            raise ValueError(
                "minimum_chase_ticks must be positive and below maximum_ticks"
            )
        if isinstance(self.stuck_patience_ticks, bool) or not isinstance(
            self.stuck_patience_ticks, int
        ):
            raise TypeError("stuck_patience_ticks must be an integer")
        if not 1 <= self.stuck_patience_ticks < self.maximum_ticks:
            raise ValueError(
                "stuck_patience_ticks must be positive and below maximum_ticks"
            )
        values = (
            self.catch_distance,
            self.proximity_radius,
            self.evader_safety_radius,
            self.progress_scale,
            self.progress_clip_blocks,
            self.closing_speed_reward,
            self.evader_progress_scale,
            self.evader_close_cost,
            self.maximum_turn_degrees,
            self.tracking_gain_scale,
            self.line_of_sight_reward,
            self.facing_reward,
            self.facing_cone_degrees,
            self.aim_level_reward,
            self.head_aim_level_reward,
            self.idle_motion_cost,
            self.aim_level_degrees,
            self.facing_away_cost,
            self.facing_away_degrees,
            self.approach_reward_margin,
            self.stationary_deadband_blocks,
            self.stationary_cost,
            self.proximity_reward,
            self.harmful_turn_cost,
            self.harmful_turn_margin_degrees,
            self.success_reward,
            self.tick_cost,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("pursuit reward values must be finite and nonnegative")
        if self.stationary_deadband_blocks <= 0.0:
            raise ValueError("stationary_deadband_blocks must be positive")
        # The clip must clear the fastest legal gait or it prices sprinting out
        # again, which is the exact defect this field was introduced to fix.
        # 0.3151 blocks is sprint's forward speed on one loaded motion tick.
        if self.progress_clip_blocks < _SPRINT_BLOCKS_PER_TICK:
            raise ValueError(
                "progress_clip_blocks must not clip below sprint's per-tick "
                f"closure ({_SPRINT_BLOCKS_PER_TICK:.4f} blocks), or the fastest "
                "gait earns no more than running"
            )
        if not 0.0 < self.maximum_turn_degrees <= 180.0:
            raise ValueError("maximum_turn_degrees must be in (0, 180]")
        if not 0.0 < self.facing_cone_degrees < self.facing_away_degrees <= 180.0:
            raise ValueError(
                "facing cones must satisfy 0 < facing < away <= 180 degrees"
            )
        if not 0.0 < self.aim_level_degrees <= 180.0:
            raise ValueError("aim_level_degrees must be in (0, 180]")
        # The invariant that keeps standing still unprofitable. A motionless
        # learner can still collect sight, facing and a full aim level, so the
        # cost has to strictly exceed all three together.
        #
        # It applies ONLY when the stage actually pays for closing. With
        # `progress_scale == 0` the stage is a pure TRACKING lesson: the learner
        # is not asked to move at all, only to find the target and hold it with
        # body and head while the target moves. "Standing still and staring" is
        # then the whole task, not an exploit, and forcing `stationary_cost`
        # above the aim rewards makes the lesson unlearnable -- it turns the
        # per-tick reward negative everywhere and couples return to episode
        # LENGTH rather than to aim quality, which is exactly what run
        # 20260820T024323Z-8db1062f measured: episode length ran 52 -> 400+ ticks
        # while entropy sat flat at 7.70, i.e. the policy never moved at all.
        # `head_aim_level_reward` belongs in this sum and was missing from it.
        # It was added alongside the head/body split and is earned on exactly
        # the same terms as `aim_level_reward` -- a parked actor holding the
        # target with its head collects it every tick -- so leaving it out let
        # the guard pass a configuration it exists to reject. Caught when
        # aim 0.18 -> 0.60 tripped the check on the body term while the head
        # term, equally large, was not being counted at all.
        stationary_earnings = (
            self.line_of_sight_reward
            + self.facing_reward
            + self.aim_level_reward
            + self.head_aim_level_reward
        )
        if self.progress_scale > 0.0 and self.stationary_cost <= stationary_earnings:
            raise ValueError(
                "stationary_cost must exceed line_of_sight_reward + "
                "facing_reward + aim_level_reward + head_aim_level_reward "
                f"({stationary_earnings:g}), or standing still and "
                "staring at the target is positive-reward. This check is "
                "skipped when progress_scale is 0, which marks a tracking-only "
                "stage where holding station is the lesson."
            )
        if not self.catch_distance < self.proximity_radius:
            raise ValueError("catch_distance must be below proximity_radius")
        if not (
            self.catch_distance
            <= self.proximity_floor_distance
            < self.proximity_radius
        ):
            raise ValueError(
                "proximity_floor_distance must sit between catch_distance and "
                "proximity_radius; at or below catch_distance the floor is "
                "inert, and at the radius the proximity term is constant zero"
            )
        if not self.catch_distance < self.evader_safety_radius <= self.proximity_radius:
            raise ValueError(
                "evader safety radius must be above catch distance and no larger "
                "than proximity radius"
            )
        if not isinstance(self.spawn_facing_target, bool):
            raise TypeError("spawn_facing_target must be a bool")
        if self.perception not in PERCEPTION_MODES:
            raise ValueError(
                f"perception must be one of {sorted(PERCEPTION_MODES)}, "
                f"got {self.perception!r}"
            )
        if not 0.0 < self.view_sector_degrees <= 360.0:
            raise ValueError("view_sector_degrees must be in (0, 360]")
        if self.schema != PURSUIT_STAGE_SCHEMA:
            raise ValueError("pursuit stage schema is not current")

    def manifest(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "maximum_ticks": self.maximum_ticks,
            "minimum_chase_ticks": self.minimum_chase_ticks,
            "terminal": "distance_catch_or_natural_combat_outcome",
            "horizon": "time_limit_truncation_with_bootstrap",
            "catch_distance": self.catch_distance,
            "proximity_radius": self.proximity_radius,
            # On the receipt because it changes what the shaping pays for:
            # below this the approach term is flat, so two runs at different
            # floors are not comparable.
            "proximity_floor_distance": self.proximity_floor_distance,
            "maximum_turn_degrees": self.maximum_turn_degrees,
            # Recorded in the manifest so a run's report states which reset
            # orientation produced its aim-error numbers.
            "spawn_facing_target": self.spawn_facing_target,
            # What the learner could see. Two runs at different perception
            # modes are not comparable -- an omniscient chaser cannot be evaded
            # on open ground -- so the mode belongs on the receipt, not only in
            # the launch spec.
            "perception": self.perception,
            "view_sector_degrees": (
                self.view_sector_degrees
                if self.perception == "field_of_view" else None
            ),
            "reward": {
                name: getattr(self, name)
                for name in (
                    "progress_scale",
                    "progress_clip_blocks",
                    "closing_speed_reward",
                    "evader_progress_scale",
                    "evader_close_cost",
                    "tracking_gain_scale",
                    "line_of_sight_reward",
                    "facing_reward",
                    "facing_cone_degrees",
                    # Previously absent from the manifest, so no run report
                    # stated the aim terms it was trained under.
                    "aim_level_reward",
                    "head_aim_level_reward",
                    "idle_motion_cost",
                    "aim_level_degrees",
                    "facing_away_cost",
                    "facing_away_degrees",
                    "approach_reward_margin",
                    "stationary_deadband_blocks",
                    "stationary_cost",
                    "proximity_reward",
                    "harmful_turn_cost",
                    "harmful_turn_margin_degrees",
                    "success_reward",
                    "tick_cost",
                )
            },
            "credit": ("learner_motion_yaw_and_pitch_with_target_motion_removed"),
            "steady_state_reward": (
                "physical_line_of_sight_and_broad_facing_pay_each_tick;"
                "opposite_facing_and_non_closing_behavior_are_negative;"
                "proximity_pays_only_during_meaningful_learner_approach;"
                "closing_rate_pays_quadratically_so_faster_closure_beats_equal_"
                "distance_closed_slowly;"
                "the_rate_is_normalized_by_sprint_speed_and_sprint_is_forward_"
                "only_so_the_top_of_that_term_requires_facing_the_target"
            ),
            "anti_stall": {
                "patience_ticks": self.stuck_patience_ticks,
                "law": (
                    "charge_absent_radial_progress_while_outside_catch;"
                    "jump_and_dodge_are_unrewarded_means_and_pay_only_through_"
                    "restored_goal_progress"
                ),
            },
            "catch_credit": ("every_valid_in_range catch_ends_and_rewards_pursuer"),
            "evader_reward": {
                "safety_radius": self.evader_safety_radius,
                "law": (
                    "reward_separation_gain_and_penalize_proximity_inside_safety_"
                    "radius_and_catch"
                ),
            },
            "opponent_context": {
                "schema": PURSUIT_CONTEXT_SCHEMA,
                "shared": "actor_target_features_and_recurrent_history",
                "supplement": (
                    "jax_privileged_current_relative_pose_velocity_every_tick"
                ),
                "supplemented_features": _TARGET_CONTEXT_FEATURES,
                "visibility": "physical_line_of_sight_is_preserved",
                "availability": (
                    "target_mask_keeps_privileged_relative_context_available_"
                    "through_occlusion"
                ),
                "java_transfer": "supplement_is_not_required_by_the_live_policy_contract",
            },
            "spatial_tracking_signal_contract_sha256": (
                spatial_tracking_signal_contract_sha256()
            ),
        }

    @property
    def contract_sha256(self) -> str:
        return _hash(self.manifest())


class PursuitLessonSignals(NamedTuple):
    """Dense learner credit plus an unambiguous anti-stall episode boundary."""

    reward: jax.Array
    target_reward: jax.Array
    catch: jax.Array
    pursuer_success: jax.Array
    meaningful_approach: jax.Array
    stationary: jax.Array
    target_separation_gain: jax.Array
    target_proximity_pressure: jax.Array
    line_of_sight: jax.Array
    facing_target: jax.Array
    facing_away: jax.Array


def pursuit_lesson_signals(
    config: PursuitStageConfig,
    *,
    current_tick: jax.Array,
    distance_after: jax.Array,
    learner_radial_progress: jax.Array,
    learner_planar_displacement: jax.Array,
    opponent_radial_contribution: jax.Array,
    learner_alignment_gain: jax.Array,
    aim_error_degrees: jax.Array,
    unnecessary_spin: jax.Array,
    valid: jax.Array,
    line_of_sight: jax.Array | None = None,
    # HEAD error, measured the same way as `aim_error_degrees` but on the head
    # rather than the body. ``None`` means "not supplied", which is not the same
    # as zero: zero would read as perfect head aim and pay full reward for a
    # signal nobody measured, so an absent head is scored as no reward at all.
    head_aim_error_degrees: jax.Array | None = None,
) -> PursuitLessonSignals:
    """Score pursuit without allowing a close stationary policy to farm reward."""

    # Floored, so closing inside `proximity_floor_distance` earns nothing extra.
    # Without it the steepest income sits in the last blocks before contact,
    # which pays a policy to crowd the target rather than to catch it.
    proximity = jnp.clip(
        (
            jnp.float32(config.proximity_radius)
            - jnp.maximum(
                distance_after,
                jnp.float32(config.proximity_floor_distance),
            )
        )
        / jnp.float32(config.proximity_radius - config.catch_distance),
        0.0,
        1.0,
    )
    meaningful = learner_radial_progress > jnp.float32(config.approach_reward_margin)
    if learner_planar_displacement.shape != valid.shape:
        raise ValueError("learner displacement and validity must share one lane shape")
    stationary = (
        learner_planar_displacement <= jnp.float32(config.stationary_deadband_blocks)
    ) & (distance_after > jnp.float32(config.catch_distance))
    # What `stationary_cost` is actually charged for, and it is deliberately not
    # `stationary`. That flag asks whether the learner MOVED; this asks whether
    # it got anywhere. The cost exists to enforce one invariant, stated in its
    # own comment above: a learner that is not closing must not be able to farm
    # the sight/facing/aim terms. Keyed to displacement it only ever bound
    # against a literally parked actor -- which is the one option a policy never
    # has to choose. Anything that merely drifts, circles at constant radius, or
    # (before the airborne-physics fix) glided ballistically cleared the 0.01
    # deadband and collected 0.28 a tick against a 0.002 tick cost.
    #
    # Measured in run 20260818T191204Z-59bbc599 at update 154: +0.187 reward per
    # tick while mean radial progress was -0.0107 blocks per tick. The policy
    # was being paid to retreat, and had learned to, for 154 updates.
    #
    # `stationary` itself is left keyed to displacement on purpose -- it is
    # returned as the stuck signal, and the selection gate's coordinate-stall
    # flag genuinely means "wedged against geometry", not "not closing".
    #
    # No magnitude changed. The constructor already requires `stationary_cost`
    # (0.32) to exceed `line_of_sight + facing + aim_level` (0.28), so charging
    # it on the right condition makes a non-closing tick net -0.042 -- negative,
    # as designed, but not so negative that it swamps the aim signal. A learner
    # turning to face its target still earns `tracking_gain_scale` on every
    # degree it recovers, up to 2.16 a tick against this 0.32.
    unproductive = ~meaningful & (distance_after > jnp.float32(config.catch_distance))
    elapsed = jnp.asarray(current_tick, dtype=jnp.int32)
    if elapsed.shape != valid.shape:
        raise ValueError("pursuit current_tick and validity must share one lane shape")
    catch = (
        valid
        & (elapsed >= jnp.int32(config.minimum_chase_ticks))
        & (distance_after <= jnp.float32(config.catch_distance))
    )
    visible = (
        jnp.zeros_like(valid)
        if line_of_sight is None
        else jnp.asarray(line_of_sight, dtype=jnp.bool_)
    )
    if visible.shape != valid.shape:
        raise ValueError("line of sight and validity must share one lane shape")
    aim_error = jnp.asarray(aim_error_degrees, dtype=jnp.float32)
    if aim_error.shape != valid.shape:
        raise ValueError("aim error and validity must share one lane shape")
    facing = valid & (aim_error <= jnp.float32(config.facing_cone_degrees))
    facing_away = valid & (aim_error >= jnp.float32(config.facing_away_degrees))
    # Graded, per tick, zero beyond `aim_level_degrees`. This is the term that
    # actually pays for *holding* a heading; `tracking_gain_scale` above pays
    # the change, which telescopes to the episode endpoints and so cannot.
    aim_level = jnp.where(
        valid,
        jnp.clip(
            1.0 - aim_error / jnp.float32(config.aim_level_degrees), 0.0, 1.0
        ),
        0.0,
    )
    if head_aim_error_degrees is None:
        head_aim_level = jnp.zeros_like(aim_level)
    else:
        head_error = jnp.asarray(head_aim_error_degrees, dtype=jnp.float32)
        if head_error.shape != valid.shape:
            raise ValueError("head aim error and validity must share one lane shape")
        head_aim_level = jnp.where(
            valid,
            jnp.clip(
                1.0 - head_error / jnp.float32(config.aim_level_degrees), 0.0, 1.0
            ),
            0.0,
        )
    # Rate of closure as a fraction of the fastest legal gait, squared. Only the
    # closing half counts: retreating is already charged through the (negative)
    # progress term, and paying a squared penalty for it as well would double it.
    closing_rate = jnp.clip(
        learner_radial_progress / jnp.float32(config.progress_clip_blocks),
        0.0,
        1.0,
    )
    closing_speed = jnp.where(valid, closing_rate * closing_rate, jnp.float32(0.0))
    reward = (
        jnp.float32(config.progress_scale)
        * jnp.clip(
            learner_radial_progress,
            -jnp.float32(config.progress_clip_blocks),
            jnp.float32(config.progress_clip_blocks),
        )
        + jnp.float32(config.closing_speed_reward) * closing_speed
        + jnp.float32(config.tracking_gain_scale)
        * jnp.clip(learner_alignment_gain, -45.0, 45.0)
        + jnp.float32(config.line_of_sight_reward) * visible
        + jnp.float32(config.facing_reward) * facing
        + jnp.float32(config.aim_level_reward) * aim_level
        + jnp.float32(config.head_aim_level_reward) * head_aim_level
        - jnp.float32(config.idle_motion_cost) * stationary
        + jnp.float32(config.proximity_reward) * proximity * meaningful
        - jnp.float32(config.facing_away_cost) * facing_away
        - jnp.float32(config.harmful_turn_cost) * unnecessary_spin
        - jnp.float32(config.stationary_cost) * unproductive
        + jnp.float32(config.success_reward) * catch
        - jnp.float32(config.tick_cost)
    )
    target_separation_gain = -opponent_radial_contribution
    target_pressure = jnp.clip(
        (jnp.float32(config.evader_safety_radius) - distance_after)
        / jnp.float32(config.evader_safety_radius - config.catch_distance),
        0.0,
        1.0,
    )
    target_reward = (
        jnp.float32(config.evader_progress_scale)
        # Same physical bound as the learner's: the evader runs on the same gait
        # table, so a tighter clip here would quietly price its sprint out too.
        * jnp.clip(
            target_separation_gain,
            -jnp.float32(config.progress_clip_blocks),
            jnp.float32(config.progress_clip_blocks),
        )
        - jnp.float32(config.evader_close_cost) * target_pressure
        - jnp.float32(config.success_reward) * catch
    )
    return PursuitLessonSignals(
        jnp.where(valid, reward, jnp.float32(0.0)),
        jnp.where(valid, target_reward, jnp.float32(0.0)),
        catch,
        catch,
        meaningful,
        stationary & valid,
        jnp.where(valid, target_separation_gain, jnp.float32(0.0)),
        jnp.where(valid, target_pressure, jnp.float32(0.0)),
        visible & valid,
        facing,
        facing_away,
    )


def pursuit_learner_action_scope(*, allow_jump: bool = False) -> FactoredActionScope:
    """Allow run/dash tracking while attack heads stay neutral.

    JUMP IS CLOSED BY DEFAULT, matching the evader. On a plane a jump buys
    nothing and costs mobility -- an airborne actor cannot accelerate, and an
    earlier lineage stayed airborne on 92% of ticks while covering 2.9 blocks in
    a whole episode. `allow_jump` reopens it for a curriculum rung that has
    something to jump over.

    Set it the SAME on both sides. The head is masked, not absent, so flipping
    it changes the legal action set and not the network shape -- but opening it
    for one actor and not the other hands that actor a mechanic its opponent
    cannot answer, which makes the matchup measure the handicap instead of the
    policies. Restore `airborne_idle_cost` alongside it: with the head open, a
    hop that covers no ground becomes payable again.
    """

    choices = []
    for name, size in zip(
        ARSENAL_POLICY_ACTION_HEAD_NAMES,
        ARSENAL_POLICY_ACTION_HEAD_SIZES,
        strict=True,
    ):
        # The body steer centres like the camera steer does: bin zero is a hard
        # turn, not a no-op, so omitting it here would make every "pinned" head
        # command a maximum body rotation on every tick.
        neutral = (
            size // 2
            if name
            in {"yaw_delta_bins", "body_yaw_delta_bins", "pitch_delta_bins"}
            else 0
        )
        choices.append(
            tuple(range(size))
            if name in (
                {
                    "locomotion_gait_compass",
                    "yaw_delta_bins",
                    "body_yaw_delta_bins",
                    "pitch_delta_bins",
                }
                | ({"jump_off_on"} if allow_jump else set())
            )
            else (neutral,)
        )
    return FactoredActionScope(tuple(choices))


def make_pursuit_action_mask_transform(
    target: TargetMotionProgram,
    roles: SkillDuelRoles,
    *,
    learner_scope: FactoredActionScope | None = None,
):
    """Force a physical target controller and lock learner combat actions."""

    scope = learner_scope or pursuit_learner_action_scope()
    frozen_target = all(mode is TargetMode.FROZEN_POLICY for mode in target.lane_modes)
    if TargetMode.FROZEN_POLICY in target.lane_modes and not frozen_target:
        raise ValueError(
            "pursuit target modes cannot mix sampled and scripted controllers"
        )

    def transform(arena, observation, base_mask):
        lanes = jnp.arange(base_mask.shape[0], dtype=jnp.int32)
        learner_mask = scope.restrict(base_mask[lanes, roles.learner_policy_slot])
        if frozen_target:
            target_mask = target.frozen_factor_scope.restrict(
                base_mask[lanes, roles.target_policy_slot]
            )
            resolved = base_mask.at[lanes, roles.target_policy_slot].set(target_mask)
        else:
            resolved = scripted_target_action_mask(
                target,
                roles,
                arena,
                observation,
                base_mask,
            ).action_mask
        return resolved.at[lanes, roles.learner_policy_slot].set(learner_mask)

    return transform


def make_pursuit_observation_transform(
    roles: SkillDuelRoles,
    params,
    *,
    perception: str = "omniscient",
    view_sector_degrees: float = 120.0,
):
    """Expose JAX target geometry, either always or only when it is in view.

    ``perception="omniscient"`` is the historical behaviour and the default:
    exact relative geometry every tick regardless of where the learner is
    looking. ``"field_of_view"`` writes it only when the target is inside the
    learner's horizontal view sector AND within `sensor_range`.
    """

    if perception not in PERCEPTION_MODES:
        raise ValueError(f"unknown perception mode {perception!r}")
    lanes = jnp.arange(roles.learner_policy_slot.shape[0], dtype=jnp.int32)
    # Half-angle, because the test is |bearing| <= half on either side of
    # forward. Precomputed as a cosine so the per-tick check is a dot product
    # against the forward vector rather than an arctangent.
    half_sector_cosine = jnp.cos(
        jnp.deg2rad(jnp.float32(view_sector_degrees) * 0.5)
    )
    gated = perception == "field_of_view"
    target_start = len(SELF_FLOAT_FEATURES)
    target_end = target_start + len(TARGET_FLOAT_FEATURES)
    context_indices = jnp.asarray(_TARGET_CONTEXT_INDICES, dtype=jnp.int32)

    def transform(arena, observation):
        combat = arena.arsenal.combat
        learner = roles.learner_actor_index
        target = roles.target_actor_index
        position = combat.position[lanes, learner]
        target_position = combat.position[lanes, target]
        velocity = combat.velocity[lanes, learner]
        target_velocity = combat.velocity[lanes, target]
        yaw = jnp.deg2rad(combat.yaw[lanes, learner])
        offset = target_position - position
        relative_velocity = target_velocity - velocity
        forward_x, forward_z = -jnp.sin(yaw), -jnp.cos(yaw)
        right_x, right_z = jnp.cos(yaw), -jnp.sin(yaw)
        forward = offset[:, 0] * forward_x + offset[:, 2] * forward_z
        right = offset[:, 0] * right_x + offset[:, 2] * right_z
        velocity_forward = (
            relative_velocity[:, 0] * forward_x + relative_velocity[:, 2] * forward_z
        )
        velocity_right = (
            relative_velocity[:, 0] * right_x + relative_velocity[:, 2] * right_z
        )
        planar = jnp.hypot(forward, right)
        safe_planar = jnp.maximum(planar, jnp.float32(1.0e-6))
        radius = jnp.float32(NEARBY_ENTITY_RADIUS_BLOCKS)
        chase_speed = jnp.maximum(
            jnp.asarray(params.target_chase_speed, dtype=jnp.float32),
            jnp.float32(1.0e-6),
        )
        vertical_speed = jnp.maximum(
            jnp.asarray(params.vertical_speed_scale, dtype=jnp.float32),
            jnp.float32(1.0e-6),
        )
        context = jnp.stack(
            (
                forward / radius,
                right / radius,
                offset[:, 1] / radius,
                velocity_forward / chase_speed,
                velocity_right / chase_speed,
                relative_velocity[:, 1] / vertical_speed,
                planar / radius,
                jnp.linalg.norm(offset, axis=1) / radius,
                right / safe_planar,
                forward / safe_planar,
            ),
            axis=1,
        )
        context = jnp.clip(context, -1.0, 1.0).astype(jnp.float32)
        slot = roles.learner_policy_slot
        target_f32 = observation.structured.base.target_f32
        learner_target = target_f32[lanes, slot]

        if gated:
            # In view when the bearing to the target is inside the half sector
            # AND it is within sensor range. `forward / planar` is the cosine
            # of that bearing; `safe_planar` keeps an exactly-coincident pair
            # from dividing by zero, and such a pair is inside any sector
            # anyway.
            in_sector = (forward / safe_planar) >= half_sector_cosine
            in_range = jnp.linalg.norm(offset, axis=1) <= jnp.asarray(
                params.sensor_range, dtype=jnp.float32
            )
            seen = (in_sector & in_range)[:, None]
            # Masked means NO READING, not "at the origin": leave whatever the
            # engine already reported rather than writing zeros, because a zero
            # relative position reads as "the opponent is on top of me".
            patched = learner_target.at[:, context_indices].set(
                jnp.where(seen, context, learner_target[:, context_indices])
            )
            patched = patched.at[:, _VISIBLE_TARGET_INDEX].set(
                seen[:, 0].astype(jnp.float32)
            )
        else:
            patched = learner_target.at[:, context_indices].set(context)
            seen = None

        target_f32 = target_f32.at[lanes, slot].set(patched)
        available = True if seen is None else seen[:, 0]
        target_mask = observation.structured.base.target_mask.at[
            lanes, slot
        ].set(available)
        structured = observation.structured._replace(
            base=observation.structured.base._replace(
                target_f32=target_f32,
                target_mask=target_mask,
            )
        )
        dense = observation.dense.at[lanes, slot, target_start:target_end].set(patched)
        # The column past the target block is the availability flag the encoder
        # reads. Under `field_of_view` it has to agree with `target_mask`, or
        # the policy would be told the slot is unavailable while a 1.0 here
        # says otherwise.
        dense = dense.at[lanes, slot, target_end].set(
            jnp.float32(1.0) if seen is None else seen[:, 0].astype(jnp.float32)
        )
        return observation._replace(structured=structured, dense=dense)

    return transform


def make_pursuit_transition_transform(
    config: PursuitStageConfig,
    roles: SkillDuelRoles,
    params,
    line_of_sight_provider=None,
    evader: "EvaderDuelConfig | None" = None,
):
    """Relabel combat transitions with causal pursuit and tracking reward.

    With ``evader``, the learner is paid the EVASION objective instead of the
    pursuit one -- staying near the opponent, surviving, and manoeuvring -- while
    everything else about the lesson is unchanged. The geometry is symmetric, so
    the same catch signal serves both: it is the learner's success when it is
    pursuing and its failure when it is evading.
    """

    if not isinstance(config, PursuitStageConfig):
        raise TypeError("pursuit transition transform needs PursuitStageConfig")
    if evader is not None and not isinstance(evader, EvaderDuelConfig):
        raise TypeError("pursuit transition transform needs EvaderDuelConfig")
    if evader is not None and evader.catch_distance != config.catch_distance:
        # Both halves must agree on what a catch is, or the evader is dodging a
        # threshold the simulator does not enforce.
        raise ValueError(
            f"evader catch_distance {evader.catch_distance:g} does not match "
            f"the stage's {config.catch_distance:g}"
        )
    lanes = jnp.arange(roles.learner_policy_slot.shape[0], dtype=jnp.int32)

    def transform(
        previous: MultiActorArenaState,
        transition,
    ):
        valid = jnp.asarray(transition.arsenal.arsenal_info.valid, dtype=jnp.bool_)
        pursuit = pursuit_transition_signals(
            previous,
            transition.state,
            roles.learner_actor_index,
            valid,
        )
        before_position = previous.arsenal.combat.position[
            lanes, roles.learner_actor_index
        ]
        after_position = transition.state.arsenal.combat.position[
            lanes, roles.learner_actor_index
        ]
        planar_displacement = jnp.linalg.norm(
            (after_position - before_position)[:, (0, 2)], axis=-1
        )
        tracking = spatial_tracking_signals(
            previous,
            transition.state,
            roles.learner_actor_index,
            valid,
            params,
            harmful_turn_margin_degrees=config.harmful_turn_margin_degrees,
        )
        visible = jnp.zeros_like(valid)
        if line_of_sight_provider is not None:
            visible = jnp.asarray(
                line_of_sight_provider(transition.state.arsenal.combat),
                dtype=jnp.bool_,
            )
        # The head is scored with the SAME kernel as the body, just fed the
        # head's yaw series instead of `combat.yaw`. Reusing the kernel keeps
        # the two errors defined identically, so `head_aim_level_reward` and
        # `aim_level_reward` are directly comparable numbers.
        head_yaw_before, head_bearing_before = duel_head_yaw(
            previous, roles.learner_actor_index
        )
        head_yaw_after, head_bearing_after = duel_head_yaw(
            transition.state, roles.learner_actor_index
        )
        head_tracking = aim_tracking_geometry_signals(
            head_yaw_before,
            head_yaw_after,
            head_bearing_before,
            head_bearing_after,
            valid,
        )
        lesson = pursuit_lesson_signals(
            config,
            current_tick=transition.state.arsenal.combat.tick_count,
            distance_after=pursuit.distance_after,
            learner_radial_progress=pursuit.learner_radial_progress,
            learner_planar_displacement=planar_displacement,
            opponent_radial_contribution=pursuit.opponent_radial_contribution,
            learner_alignment_gain=tracking.learner_alignment_gain,
            aim_error_degrees=tracking.error_after,
            head_aim_error_degrees=head_tracking.error_after,
            unnecessary_spin=tracking.unnecessary_adjustment,
            valid=valid,
            line_of_sight=visible,
        )
        natural_terminated = jnp.any(transition.terminated, axis=1)
        natural_truncated = jnp.any(transition.truncated, axis=1)
        horizon = (
            transition.state.arsenal.combat.tick_count
            >= jnp.int32(config.maximum_ticks)
        ) & ~natural_terminated
        if evader is None:
            learner_reward = lesson.reward
        else:
            # Body yaw, not head yaw: the manoeuvre being paid for is a change of
            # travel direction, and travel rides the body. Wrapped to the shorter
            # arc so a turn through 350 degrees counts as 10.
            body_before = previous.arsenal.combat.yaw[lanes, roles.learner_actor_index]
            body_after = transition.state.arsenal.combat.yaw[
                lanes, roles.learner_actor_index
            ]
            heading_change = jnp.abs(
                (body_after - body_before + 180.0) % 360.0 - 180.0
            )
            # Ground covered this tick, XZ only. Vertical motion is deliberately
            # excluded: a hopping policy racks up 3D path length while going
            # nowhere, which is exactly the behaviour this term must not pay.
            before_xz = previous.arsenal.combat.position[
                lanes, roles.learner_actor_index
            ][:, (0, 2)]
            after_xz = transition.state.arsenal.combat.position[
                lanes, roles.learner_actor_index
            ][:, (0, 2)]
            step_xz = after_xz - before_xz
            ground_travelled = jnp.linalg.norm(step_xz, axis=-1)
            # How much of that step ran ACROSS the bearing to the pursuer
            # rather than along it. Straight-line flight is nearly all radial
            # and earns nothing here; circling the standoff earns the lot.
            # Taken at the tick's START so the evader is scored on the bearing
            # it decided against, not the one its own move produced.
            bearing_xz = (
                previous.arsenal.combat.position[lanes, roles.target_actor_index]
                - previous.arsenal.combat.position[lanes, roles.learner_actor_index]
            )[:, (0, 2)]
            span = jnp.maximum(
                jnp.linalg.norm(bearing_xz, axis=-1), jnp.float32(1.0e-6)
            )
            unit = bearing_xz / span[:, None]
            # 2D cross product: the component perpendicular to the bearing.
            # Overlapping actors leave `unit` arbitrary, but the step is then
            # scored against a bearing that carries no information either way.
            lateral_travelled = jnp.abs(
                step_xz[:, 0] * unit[:, 1] - step_xz[:, 1] * unit[:, 0]
            )
            # Where the BODY ended up pointing, against the bearing it started
            # the tick with. `body_after` because the posture being scored is
            # the one this tick's action chose; the start bearing for the same
            # reason the lateral split uses it -- the evader is judged on the
            # geometry it decided against, not the one its own move produced.
            #
            # Same forward convention as the view sector above
            # (`forward_x, forward_z = -sin(yaw), -cos(yaw)`), so `alignment` is
            # +1 staring straight at the pursuer and -1 pointing directly away.
            after_yaw = jnp.deg2rad(body_after)
            alignment = (
                -jnp.sin(after_yaw) * unit[:, 0] - jnp.cos(after_yaw) * unit[:, 1]
            )
            facing_away = (1.0 - alignment) * 0.5
            # Airborne is the complement of grounded. Paired with
            # `ground_travelled` this distinguishes a jump that carries the
            # evader somewhere from one that does not.
            aloft = ~transition.state.arsenal.combat.agent_grounded
            if aloft.ndim > 1:
                aloft = aloft[lanes, roles.learner_actor_index]
            # How much of the step ran ALONG the body's facing. Same forward
            # convention as the view sector and `facing_away`, and taken against
            # the body yaw the tick's action chose, so it scores the alignment
            # the policy actually commanded.
            #
            # This is what separates a sprint from a strafe. Movement is
            # compass-driven, so an actor can travel in any direction without
            # turning -- but the authored speeds only reward pointing where you
            # go: sprint-forward 7.00 against run-strafe 4.40 blocks/s, and
            # `sprinting` requires `forward_dominant`. Nothing else in this
            # contract can tell the two apart.
            forward_travelled = (
                step_xz[:, 0] * -jnp.sin(after_yaw)
                + step_xz[:, 1] * -jnp.cos(after_yaw)
            )
            learner_reward = evader_duel_signals(
                evader,
                distance=pursuit.distance_after,
                heading_change_degrees=heading_change,
                ground_travelled=ground_travelled,
                lateral_travelled=lateral_travelled,
                forward_travelled=forward_travelled,
                # Negated because `learner_radial_progress` is positive when the
                # learner CLOSES, and on this branch the learner is the evader.
                # Its own contribution rather than the raw gap change, so a
                # pursuer that wanders off does not pay the evader for standing
                # still -- the same attribution
                # `target_separation_gain = -opponent_radial_contribution`
                # already makes for the evader on the other side of the swap.
                separation_gain=-pursuit.learner_radial_progress,
                facing_away=facing_away,
                airborne=aloft,
                caught=lesson.catch,
                truncated=horizon | natural_truncated,
                valid=valid,
            ).reward
        reward = (
            jnp.zeros_like(transition.reward)
            .at[lanes, roles.learner_policy_slot]
            .set(learner_reward)
            .at[lanes, roles.target_policy_slot]
            .set(lesson.target_reward)
        )
        skill_terminated = lesson.catch & ~natural_truncated
        active = transition.controlled | transition.terminated | transition.truncated
        terminated = transition.terminated | (skill_terminated[:, None] & active)
        # `transition.truncated` MUST be propagated. Suppressing it to make the
        # episode end only on contact was tried on 2026-08-17 and silently
        # destroyed training: run 20260817T210627Z-1fc0ab55 reported
        # `total_trainable_actor_steps` of 0.0 for 96 consecutive updates, with
        # `invalid_trainable_steps` at 3968, `approximate_kl` exactly 0.0 and
        # entropy exactly 0.0, against 6.29M steps and a healthy 0.006 KL on the
        # run before it. Its evaluations were byte-identical at u0/u24/u48
        # because the policy never moved at all.
        #
        # The reason is that the underlying truncation is what *resets* the
        # arena. The arsenal raises it from the sticky `combat.geometry_exhausted`
        # bit; drop the signal and the lane stays parked in an exhausted-geometry
        # state forever, so every subsequent step is counted invalid and nothing
        # is trainable. A terminal that the environment needs in order to reset
        # cannot be filtered out downstream in the reward contract.
        #
        # The underlying problem is real and stays open: by ablation, forcing
        # `_arsenal_geometry_truncated` to False took episodes reaching the
        # 256-tick horizon from 27.7% to 100.0% and removed a 13-tick mode worth
        # 16.8% of episodes. `region_resolve_aabb_motion` reads collision
        # geometry from the 160-block capture but certifies against the 96-block
        # core, and a learner at ~3 blocks/s over 256 ticks covers ~50 against a
        # core half-extent of 48. The fix belongs at placement -- a core-edge
        # inset on the traversal-graph spawn pool -- not here.
        truncated = transition.truncated | (horizon[:, None] & active)

        completion = transition.reward_components.completion.at[
            lanes, roles.learner_policy_slot
        ].set(
            transition.reward_components.completion[lanes, roles.learner_policy_slot]
            | lesson.catch
        )
        death = transition.reward_components.death.at[
            lanes, roles.target_policy_slot
        ].set(
            transition.reward_components.death[lanes, roles.target_policy_slot]
            | lesson.catch
        )
        return transition._replace(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            reward_components=transition.reward_components._replace(
                completion=completion,
                death=death,
            ),
        )

    return transform


def bind_pursuit_collector(
    collector_factory,
    params,
    runtime_config,
    assignment: PolicyActorAssignment,
    *,
    target: TargetMotionProgram,
    stage: PursuitStageConfig,
    line_of_sight_provider=None,
    evader: EvaderDuelConfig | None = None,
    **collector_options,
):
    """Bind the same pursuit lesson to flat, Region, or future collectors.

    ``evader`` flips which side of the duel the learner is paid for without
    changing anything else, so a pursuer run and an evader run share one scene,
    one target program and one set of transforms.
    """

    owned = {
        "action_mask_transform",
        "observation_transform",
        "transition_transform",
        "maximum_turn_degrees",
    }
    supplied = sorted(owned & collector_options.keys())
    if supplied:
        raise ValueError(f"pursuit owns collector transform seams: {supplied}")
    from arena.training.skills.stage import skill_duel_roles

    roles = skill_duel_roles(assignment)
    return collector_factory(
        params,
        runtime_config,
        assignment,
        action_mask_transform=make_pursuit_action_mask_transform(
            target,
            roles,
            learner_scope=pursuit_learner_action_scope(allow_jump=stage.allow_jump),
        ),
        observation_transform=make_pursuit_observation_transform(
            roles,
            params,
            perception=stage.perception,
            view_sector_degrees=stage.view_sector_degrees,
        ),
        transition_transform=make_pursuit_transition_transform(
            stage,
            roles,
            params,
            line_of_sight_provider,
            evader,
        ),
        maximum_turn_degrees=stage.maximum_turn_degrees,
        **collector_options,
    )


__all__ = [
    "PURSUIT_CONTEXT_SCHEMA",
    "PURSUIT_DEFAULT_TICKS",
    "PURSUIT_MAXIMUM_TICKS",
    "PURSUIT_STAGE_SCHEMA",
    "PursuitLessonSignals",
    "PursuitStageConfig",
    "bind_pursuit_collector",
    "make_pursuit_action_mask_transform",
    "make_pursuit_observation_transform",
    "make_pursuit_transition_transform",
    "pursuit_lesson_signals",
    "pursuit_learner_action_scope",
]
