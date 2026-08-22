"""Train and evaluate recurrent pursuit against a sampled actor in JAX Region."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from types import SimpleNamespace
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter, time
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from arena.training.contracts.common import HorizonConfig
from arena.training.contracts.evader_duel import EvaderDuelConfig
from arena.training.contracts.pursuit import (
    PERCEPTION_MODES,
    PURSUIT_DEFAULT_TICKS,
    PURSUIT_MAXIMUM_TICKS,
    PursuitStageConfig,
    bind_pursuit_collector,
    pursuit_learner_action_scope,
)
from arena.training.skills.program import TargetMode, TargetMotionProgram
from arena.training.skills.stage import skill_duel_roles
from arena.training.runtime.monitor import TrainingMonitor
from arena.training.runtime.evaluation import AsyncEvaluationQueue
from hytalegym.jax.compilation_cache import configure_persistent_compilation_cache
from hytalegym.jax.combat import AGENT_ENTITY, TARGET_ENTITY
from hytalegym.jax.combat.arsenal.environment import ArsenalActionSurfaceRuntime
from hytalegym.jax.combat.arsenal.runtime import reset_arsenal_batch
from hytalegym.jax.combat.block_interactions import empty_block_interaction_state
from hytalegym.jax.combat.observation.v1.schema.contract import (
    SELF_FLOAT_FEATURES,
    TARGET_FLOAT_FEATURES,
)
from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_ACTION_HEAD_NAMES,
    ARSENAL_POLICY_COMPASS_DIRECTIONS,
    ARSENAL_POLICY_LOCOMOTION_DODGE_START,
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
)
from hytalegym.jax.combat.types import GAIT_SPRINT
from hytalegym.jax.training.checkpoint import (
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from hytalegym.jax.training.multi_actor import (
    initialize_multi_actor_arena_state,
    make_multi_actor_region_rollout_collector,
    make_multi_actor_rollout_collector,
    make_multi_actor_shared_policy_trainer,
    policy_actor_assignment,
)
from hytalegym.jax.world import geometry_perception_line_of_sight_result


PURSUIT_RUN_SCHEMA = "arena-recurrent-pursuit-run-v16"
PURSUIT_SELECTION_SCHEMA = "arena-recurrent-pursuit-selection-v7"
PURSUIT_COMPUTE_COMPOSITION_SCHEMA = "arena-pursuit-compute-composition-v1"
_MIN_LEARNER_MOTION_FRACTION = 0.50
_MIN_TARGET_MOTION_FRACTION = 0.20
_MIN_PATH_LENGTH = 10.0
# A coordinate stall is a policy wedged against geometry, not a policy that
# brushed a wall once. This was 0.0, which no Region evaluation has ever met:
# every milestone of every run so far carried `learner_coordinate_stall`, so
# `eligible` was never True and every promotion in the project's history has
# been a flagged one. Measured stuck fractions on the two comparable runs were
# 0.0068 (20260817T191558Z-4d7c079e) and 0.0142 (20260817T220845Z-c1265fc5);
# 0.05 is an order of magnitude above incidental terrain contact and still an
# order of magnitude below a policy that is genuinely stuck.
_MAX_STUCK_FRACTION = 0.05
#: Weight on the aim/line-of-sight term once catch rate leads the objective.
#: Bounded to a tiebreaker on purpose. The aim term spans [-1, 1] in principle
#: but moved only 0.049 across all twelve archived milestones of the two runs
#: above, which at this weight shifts the score by 0.0005 -- two orders of
#: magnitude below the 0.046 catch-rate difference it previously overturned when
#: it *was* the whole objective. See `pursuit_selection_assessment`.
_AIM_TIEBREAK_WEIGHT = 0.01
PURSUIT_SELECTION_LAW = (
    "fixed_common_random_number_evaluation;maximize_catch_rate_then_physical_"
    "line_of_sight_minus_normalized_omniscient_3d_aim_error_as_tiebreak;"
    "candidate_success_rate_at_least_source_minus_0.01;"
    "privileged_target_context_fraction_at_least_0.99;"
    "physical_line_of_sight_fraction_at_least_declared_floor;"
    "physical_yaw_delta_never_exceeds_stage_turn_cap;"
    "learner_physical_motion_fraction_at_least_0.50;"
    "no_stationary_run_reaches_declared_stuck_patience;"
    "authored_target_motion_cadence_at_least_0.20;"
    "learner_and_target_reset_safe_path_at_least_10_blocks;"
    "vertical_aim_exercised_at_least_0.10;pitch_action_fraction_at_least_0.01;"
    "all_metrics_finite_and_all_evaluation_rows_valid;"
    "retain_host_policy_at_best_milestone;replay_uses_longest_both_moving_"
    "archived_lane;"
    "publish_selected_not_latest_policy"
)
PERSISTENT_COMPILATION_CACHE = configure_persistent_compilation_cache()
_STAGE_OPTION_FIELDS = frozenset(PursuitStageConfig.__dataclass_fields__) - {
    "maximum_ticks",
    "schema",
}
_TARGET_OPTION_FIELDS = frozenset(TargetMotionProgram.__dataclass_fields__) - {
    "lane_modes",
    "frozen_policy_sha256",
    "frozen_factor_scope",
    "approach_hold_distance",
    "schema",
}


@dataclass(frozen=True, slots=True)
class PursuitSelectionAssessment:
    """Host summary used only at explicit fixed-seed evaluation milestones."""

    objective_score: float
    success_rate: float
    valid_fraction: float
    eligible: bool
    flags: tuple[str, ...]


def pursuit_selection_assessment(
    metrics: Mapping[str, float | int],
    *,
    source_success_rate: float | None = None,
    minimum_line_of_sight_fraction: float = 0.50,
    maximum_turn_degrees: float = 24.3,
    objective: str = "pursue",
    perception: str = "omniscient",
) -> PursuitSelectionAssessment:
    """Validate one evaluation and apply the source retention floor.

    ``objective="evade"`` inverts what success means: the learner is the
    evader, so an episode it SURVIVED is the win and a catch is the loss.
    Without this the gate reads the raw catch rate and promotes whichever
    update is caught most -- run 20260821T030402Z-7e7a0372 duly promoted a
    policy caught in 2461 of 2461 episodes and flagged it eligible.
    """

    if objective not in {"pursue", "evade"}:
        raise ValueError("objective must be 'pursue' or 'evade'")
    if perception not in PERCEPTION_MODES:
        raise ValueError(
            f"perception must be one of {sorted(PERCEPTION_MODES)}, got {perception!r}"
        )

    required = (
        "mean_return",
        "mean_visible_aim_error_degrees",
        "mean_omniscient_aim_error_degrees",
        "mean_distance",
        "visible_fraction",
        "target_context_available_fraction",
        "maximum_yaw_delta_degrees",
        "turn_cap_violation_fraction",
        "vertical_aim_exercised_fraction",
        "pitch_turning_fraction",
        "moving_fraction",
        "learner_motion_fraction",
        "stationary_fraction",
        "stuck_fraction",
        "maximum_stationary_ticks",
        "jump_request_fraction",
        "jump_takeoff_fraction",
        "dodge_request_fraction",
        "airborne_fraction",
        "target_motion_fraction",
        "learner_path_length",
        "target_path_length",
        "valid_fraction",
        "episodes",
        "catches",
    )
    missing = tuple(name for name in required if name not in metrics)
    if missing:
        raise ValueError(f"pursuit selection metrics are missing {missing}")
    mean_return = float(metrics["mean_return"])
    aim_error = float(metrics["mean_omniscient_aim_error_degrees"])
    distance = float(metrics["mean_distance"])
    visible = float(metrics["visible_fraction"])
    context_available = float(metrics["target_context_available_fraction"])
    maximum_yaw_delta = float(metrics["maximum_yaw_delta_degrees"])
    turn_cap_violations = float(metrics["turn_cap_violation_fraction"])
    vertical = float(metrics["vertical_aim_exercised_fraction"])
    pitch_turning = float(metrics["pitch_turning_fraction"])
    moving = float(metrics["moving_fraction"])
    learner_motion = float(metrics["learner_motion_fraction"])
    stationary = float(metrics["stationary_fraction"])
    stuck = float(metrics["stuck_fraction"])
    maximum_stationary = int(metrics["maximum_stationary_ticks"])
    jump_request = float(metrics["jump_request_fraction"])
    jump_takeoff = float(metrics["jump_takeoff_fraction"])
    dodge_request = float(metrics["dodge_request_fraction"])
    airborne = float(metrics["airborne_fraction"])
    target_motion = float(metrics["target_motion_fraction"])
    learner_path = float(metrics["learner_path_length"])
    target_path = float(metrics["target_path_length"])
    valid = float(metrics["valid_fraction"])
    episodes = int(metrics["episodes"])
    catches = int(metrics["catches"])
    finite = all(
        math.isfinite(value)
        for value in (
            mean_return,
            aim_error,
            distance,
            visible,
            context_available,
            maximum_yaw_delta,
            turn_cap_violations,
            vertical,
            pitch_turning,
            moving,
            learner_motion,
            stationary,
            stuck,
            jump_request,
            jump_takeoff,
            dodge_request,
            airborne,
            target_motion,
            learner_path,
            target_path,
            valid,
        )
    )
    counts_valid = (
        episodes >= 0 and 0 <= catches <= episodes and maximum_stationary >= 0
    )
    ranges_valid = all(
        0.0 <= value <= 1.0
        for value in (
            visible,
            context_available,
            turn_cap_violations,
            vertical,
            pitch_turning,
            moving,
            learner_motion,
            stationary,
            stuck,
            jump_request,
            jump_takeoff,
            dodge_request,
            airborne,
            target_motion,
            valid,
        )
    )
    caught_rate = catches / max(episodes, 1)
    # The learner's own success, which is the complement when it is evading.
    success_rate = caught_rate if objective == "pursue" else 1.0 - caught_rate
    if (
        not math.isfinite(minimum_line_of_sight_fraction)
        or not 0.0 <= minimum_line_of_sight_fraction <= 1.0
    ):
        raise ValueError("minimum line-of-sight fraction must be finite in [0, 1]")
    if not math.isfinite(maximum_turn_degrees) or maximum_turn_degrees <= 0.0:
        raise ValueError("maximum turn degrees must be finite and positive")
    if source_success_rate is None:
        success_floor = 0.0
    else:
        source = float(source_success_rate)
        if not math.isfinite(source) or not 0.0 <= source <= 1.0:
            raise ValueError("source pursuit success rate must be finite in [0, 1]")
        success_floor = max(0.0, source - 0.01)
    flags = []
    if not finite or not counts_valid or not ranges_valid:
        flags.append("invalid_evaluation_metrics")
    if valid != 1.0:
        flags.append("invalid_transition_evidence")
    if success_rate < success_floor:
        flags.append("catch_rate_regression")
    # Under omniscient perception the privileged channel is always written, so
    # anything below 1.0 means the evidence itself is broken. Under
    # `field_of_view` it is withheld on purpose, and the same threshold would
    # reject every milestone of every gated run by construction -- which it did,
    # for 27 self-play rounds, reading as "nothing promoted" rather than as a
    # guardrail. What still has to hold there is that the privileged channel
    # agrees with what was physically visible: context is granted when, and only
    # when, the learner could see. A divergence is the real broken-evidence
    # signal in that mode.
    if perception == "field_of_view":
        if abs(context_available - visible) > 0.01:
            flags.append("privileged_target_context_disagrees_with_line_of_sight")
    elif context_available < 0.99:
        flags.append("privileged_target_context_below_guardrail")
    if visible < minimum_line_of_sight_fraction:
        flags.append("physical_line_of_sight_below_guardrail")
    if maximum_yaw_delta > maximum_turn_degrees + 1.0e-3 or turn_cap_violations:
        flags.append("physical_turn_cap_violated")
    if vertical < 0.10:
        flags.append("vertical_aim_not_exercised")
    if vertical >= 0.10 and pitch_turning < 0.01:
        flags.append("pitch_control_inert")
    if (
        moving < _MIN_LEARNER_MOTION_FRACTION
        or learner_motion < _MIN_LEARNER_MOTION_FRACTION
        or learner_path < _MIN_PATH_LENGTH
    ):
        flags.append("learner_motion_collapse")
    if stuck > _MAX_STUCK_FRACTION:
        flags.append("learner_coordinate_stall")
    if target_motion < _MIN_TARGET_MOTION_FRACTION or target_path < _MIN_PATH_LENGTH:
        flags.append("sampled_target_motion_collapse")
    # Catch rate leads, aim quality only settles ties. The objective used to be
    # the aim term alone, which ranked on noise: at update 192 run
    # 20260817T220845Z-c1265fc5 caught 42.1% of its targets and lost promotion
    # to its own update 48 at 37.5%, on an aim-score difference of 0.001. Aim was
    # random in both runs -- 75.3 to 82.6 degrees with no trend across twelve
    # milestones -- so the term being maximised carried no signal, while the one
    # the stage exists to teach was not in the score at all.
    #
    # Aim is deliberately kept rather than dropped: a policy that reaches the
    # target *and* looks at it is the better checkpoint to carry into combat,
    # where an authored melee selector fires from the actor's facing.
    aim_quality = visible - aim_error / 180.0
    return PursuitSelectionAssessment(
        success_rate + _AIM_TIEBREAK_WEIGHT * aim_quality,
        success_rate,
        valid,
        not flags,
        tuple(flags),
    )


def pursuit_candidate_is_better(
    candidate: PursuitSelectionAssessment,
    incumbent: PursuitSelectionAssessment,
) -> bool:
    """Return whether a fixed-seed candidate may replace the incumbent."""

    if not isinstance(candidate, PursuitSelectionAssessment) or not isinstance(
        incumbent, PursuitSelectionAssessment
    ):
        raise TypeError("pursuit selection assessments are invalid")
    # Eligibility still dominates: a clean candidate always displaces a flagged
    # incumbent, and a flagged candidate never displaces a clean one.
    if candidate.eligible != incumbent.eligible:
        return bool(candidate.eligible)
    # Same eligibility on both sides -- rank on the objective. Requiring
    # `candidate.eligible` here instead made the gate unable to compare two
    # flagged assessments, and the source at update 0 is admitted
    # unconditionally, so a single always-on flag pinned the promotion to the
    # untrained policy no matter how good training got. Run
    # 20260817T062230Z-d6552974 promoted update 0 (score -0.172) over update
    # 256 (score +0.678, catch rate 0.994) for exactly this reason. A flagged
    # promotion is still recorded as flagged in `selection.flags`.
    return bool(candidate.objective_score > incumbent.objective_score)


def _runtime_environment() -> dict[str, str | None]:
    return {
        name: os.environ.get(name)
        for name in (
            "HYTALERL_JAX_RUNTIME_PROFILE",
            "JAX_PLATFORMS",
            "XLA_PYTHON_CLIENT_PREALLOCATE",
            "XLA_PYTHON_CLIENT_MEM_FRACTION",
        )
    }


@dataclass(frozen=True, slots=True)
class PursuitRunConfig:
    source_policy: Path
    output: Path
    loadout: str = "iron_sword"
    opponent: str = "iron_sword"
    batch: int = 128
    updates: int = 64
    rollout_steps: int = 128
    evaluation_steps: int = PURSUIT_DEFAULT_TICKS
    evaluation_updates: tuple[int, ...] = (0, 8, 16, 32, 64)
    memory_reset_updates: tuple[int, ...] | None = ()
    archive_lanes: int = 8
    update_epochs: int = 4
    num_minibatches: int = 8
    learning_rate: float = 3.0e-4
    entropy_coefficient: float = 0.003
    #: Weight on the critic's half of the shared loss. Worth lowering whenever a
    #: run starts from a checkpoint trained against a DIFFERENT reward, as the
    #: recursive evader/pursuer pair does on every role swap: the inherited
    #: critic's predictions are then maximally wrong, its loss runs three or
    #: four orders of magnitude above the policy loss, and because both halves
    #: share a trunk and one `max_gradient_norm` clip, the value error takes
    #: essentially the whole gradient budget while the policy rides along and
    #: degrades.
    value_coefficient: float = 0.5
    seed: int = 570_701
    #: Region selection rank, or ``None`` for "no explicit choice".
    #: The console omits this field entirely unless the caller set it, so
    #: ``None`` is the only way to tell an unset key from one that happens to
    #: equal the old default. Before this existed a one-world run discarded
    #: the key outright -- see `_resolved_region_selection`.
    selection_key: int | None = None
    node_seed: int = 570_071
    world_artifact_seed: int | None = None
    #: How many library worlds are resident, with the batch spread across them.
    #: 1 reproduces the historical single-world run. Above 1, ``selection_key``
    #: chooses *which* worlds, so varying it rotates the subset between runs.
    world_count: int = 1
    #: Draw a separate spawn/target pair per environment inside its own world.
    #: Without this every lane shares one pair and a policy can memorise a
    #: single opening instead of learning to pursue.
    environment_diversity: bool = False
    #: Compose the resident set as half flattest-first, half digest-ranked,
    #: instead of taking the whole set from ``selection_key``. The key is a rank
    #: over the split rather than a filter, so a deliberate mix is not reachable
    #: through it at any value -- see ``_mixed_terrain_world_seeds``. Needs
    #: ``world_count`` above 1 to mean anything.
    mixed_terrain_worlds: bool = False
    #: Take the resident set flattest-first instead of by digest rank. The
    #: digest rank is indifferent to terrain, so it can hand a learner worlds
    #: that are mostly cliff. Wins over ``mixed_terrain_worlds`` when both are
    #: set. Needs ``world_count`` above 1 to mean anything.
    flattest_worlds: bool = False
    #: Actor perception range in blocks. Defaults to **64.0**, two 32-block
    #: chunks, so the agent can see a target two chunks ahead and pursuit is not
    #: capped at the opening distance. Pass ``None`` for the authored 16.0 from
    #: ``rulesets/hytale_0_5_7/kweebec_razorleaf_vs_trork_brawler_v9.json``,
    #: which remains the fidelity value and is what combat runs should use.
    #:
    #: This is a declared deviation, not a free knob: ``target_forward``,
    #: ``target_right`` and ``target.planar_distance`` in the learner
    #: observation are all divided by it (``runtime/reset.py``), so at 64 a
    #: target 16 blocks away reads 0.25 where it read 1.0 at 16. A checkpoint
    #: trained at one range reads distance differently at another.
    sensor_range: float | None = 64.0
    target_controller: str = "scripted_flee_weave"
    #: Train on an authored design instead of a captured Region. The value is a
    #: `worlds.designs` id, and the design's own content digest -- not its seed
    #: -- becomes the run's world identity. Skips the Region artifact hash
    #: entirely, so there is no multi-minute scene build.
    world_design: str | None = None
    #: Which side of the duel the learner is paid for. "evade" swaps the
    #: learner's reward to the evasion objective and leaves the scene, the
    #: target program and every transform identical, so a recursive pair is two
    #: runs of the same lesson rather than two lessons.
    objective: str = "pursue"
    #: Share of the full stamina bar the learner may hold. The bar belongs to
    #: the agent actor, which is the learner, so this handicaps exactly the
    #: side being trained -- below 1.0 it cannot sprint indefinitely.
    learner_stamina_fraction: float = 1.0
    #: Multiplier on the authored stamina regeneration rate for the learner.
    #: 1.0 is the shipped Hytale rate; below it a spent bar refills slower, so a
    #: handicap lasts the whole episode instead of only its first sprint.
    learner_stamina_regen_scale: float = 1.0
    #: Magnitude of the evader's two terminals (survive / caught). The
    #: default 1000 against a 0.20/tick contact reward is roughly 50:1 over a
    #: 100-tick episode, which makes every advantage terminal-dominated.
    evader_terminal_reward: float = 1000.0
    #: Per-tick reward at the peak of the evader's standoff band.
    evader_contact_reward: float = 0.10
    #: Paid per block of HORIZONTAL ground the evader covers. Vertical motion
    #: earns nothing, so hopping on the spot is worthless.
    evader_travel_reward: float = 0.10
    #: Paid for pointing the BODY away from the pursuer, scaled by how directly
    #: away it points. Zero disables the term. Whether it is ALSO scaled by the
    #: standoff band depends on `evader_band_leads`.
    #: Pays the evader per block its OWN motion adds to the gap. Zero by
    #: default; the objective ladder turns it on for `long_range`, where
    #: increasing distance IS the objective, and back off wherever the band
    #: leads. Without it nothing in the evader's per-tick reward distinguishes
    #: fleeing from circling.
    #: The three motion terms that were contract defaults with no launch-spec
    #: field, so the ladder could not reach them. Exposed 2026-08-21 because two
    #: of them pay for exactly the behaviour a learn-to-move rung must not buy:
    #: `strafe` pays the component of a step that runs ACROSS the bearing, i.e.
    #: circling, and `turn` pays for swinging the body at all. The evader turned
    #: on 98% of ticks and covered half its pursuer's ground.
    evader_strafe_reward: float = 0.12
    evader_sprint_reward: float = 0.005
    evader_turn_reward: float = 0.02
    #: Pays only the part of a step that runs along the body's facing, which is
    #: the difference between a sprint and a strafe. `evader_travel_reward` pays
    #: both identically per block.
    #: Paid every tick the evader is alive, so a longer episode scores higher
    #: even when it never reaches the horizon.
    #: Opens the jump head for BOTH actors on this run. The rung decides.
    #: Charged per tick the evader is airborne having covered no ground. Zero
    #: while the jump head is closed -- a cost that can never be charged is dead
    #: weight -- and restored on the rungs that open it.
    evader_airborne_idle_cost: float = 0.0
    evader_allow_jump: bool = False
    evader_alive_reward: float = 0.0
    evader_forward_travel_reward: float = 0.0
    evader_separation_reward: float = 0.0
    evader_body_away_reward: float = 0.04
    #: Whether the standoff band is the evader's objective or merely a nudge.
    #: False keeps a small band pulling it back toward the fight while letting
    #: the per-tick motion terms out-earn it, which is the locomotion-first
    #: phase; it also ungates `evader_body_away_reward`.
    evader_band_leads: bool = True
    #: Subtracted when the evader is caught. Unset it tracks
    #: `evader_terminal_reward`, which is the historical behaviour and what a
    #: single-shape run wants.
    #:
    #: They have to be separable for a curriculum that RETIRES the survival
    #: bonus. The two were one number, so lowering the terminal to hand the
    #: objective over to proximity also lowered the catch penalty -- at the
    #: limit leaving an evader that pays nothing for being caught, which is not
    #: a subtler objective but the absence of one. Being caught is bad at every
    #: rung; only the survival BONUS is on a schedule.
    evader_caught_penalty: float | None = None
    #: A separate frozen checkpoint for the opponent slot. Unset means the
    #: opponent is a copy of `source_policy`, which is what a single-policy run
    #: wants; set, it is the previous generation, which is what a recursive
    #: pursuer/evader pair wants. Only meaningful with a policy-driven target.
    opponent_checkpoint: Path | None = None
    #: Half-width of the design arena. Bounds an otherwise unbounded plane and
    #: is what the adaptive evader steers away from.
    arena_radius: float = 48.0
    #: (walk, run, sprint) shares of each adaptive-evader duty cycle. Must sum
    #: to one. The default averages 0.194 blocks/tick, the same pace
    #: `scripted_flee_weave` travels at. Only read by the adaptive controllers.
    evader_gait_mix: tuple[float, float, float] = (0.1, 0.6, 0.3)
    #: 20 divides the default mix exactly into 2 walk / 12 run / 6 sprint.
    evader_gait_period_ticks: int = 20
    #: (inner, outer) blocks of the standoff band the `scripted_flee_standoff`
    #: evader holds. It circles the pursuer inside the band, so the bearing
    #: keeps moving while the range does not. Only read by that controller.
    evader_standoff_band: tuple[float, float] = (6.0, 14.0)
    #: Ticks the standoff evader orbits one way before reversing.
    evader_standoff_strafe_ticks: int = 40
    target_separation_range: tuple[float, float] = (10.0, 14.0)
    minimum_baseline_visible_fraction: float = 0.50
    #: JOINT KL ceiling across every action head. ``None`` derives it from the
    #: live head count, which is the only stable way to set it: `ppo.py` 857
    #: means over the BATCH only, and the log-ratio it consumes is a sum over
    #: all heads (`distribution.py` 104-112 accumulates one logit per head), so
    #: this number's natural scale grows every time the action surface gains a
    #: head. A fixed 0.03 was calibrated at ten heads; at twelve it sat about
    #: 2.2 sd above the measured mean and stopped healthy runs on noise --
    #: 6 of 36 updates crossed it in 20260819T222021Z-3da75418 while
    #: clip_fraction held at 0.22, which is a conservative step size.
    maximum_approximate_kl: float | None = None
    stage_options: Mapping[str, Any] = field(default_factory=dict)
    target_options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        integers = (
            self.batch,
            self.updates,
            self.rollout_steps,
            self.evaluation_steps,
            self.update_epochs,
            self.num_minibatches,
            self.archive_lanes,
        )
        if any(isinstance(value, bool) or value < 1 for value in integers):
            raise ValueError("pursuit run counts must be positive integers")
        if self.world_artifact_seed is not None and (
            isinstance(self.world_artifact_seed, bool)
            or not isinstance(self.world_artifact_seed, int)
            or self.world_artifact_seed < 0
        ):
            raise ValueError("world artifact seed must be a nonnegative integer")
        if self.sensor_range is not None and (
            isinstance(self.sensor_range, bool)
            or not math.isfinite(float(self.sensor_range))
            or float(self.sensor_range) <= 0.0
        ):
            raise ValueError("sensor range override must be finite and positive")
        if self.evaluation_steps > PURSUIT_MAXIMUM_TICKS:
            raise ValueError(
                f"pursuit evaluation cannot exceed {PURSUIT_MAXIMUM_TICKS} ticks"
            )
        milestones = tuple(self.evaluation_updates)
        if (
            milestones != tuple(sorted(set(milestones)))
            or not milestones
            or milestones[0] != 0
            or milestones[-1] != self.updates
        ):
            raise ValueError("evaluation updates must be sorted unique 0..updates")
        memory_updates = (
            ()
            if self.memory_reset_updates is None
            else tuple(self.memory_reset_updates)
        )
        if memory_updates != tuple(sorted(set(memory_updates))) or not set(
            memory_updates
        ).issubset(milestones):
            raise ValueError("memory reset updates must be sorted evaluation updates")
        if (self.batch * self.rollout_steps) % self.num_minibatches:
            raise ValueError("batch*rollout_steps must divide into minibatches")
        if self.archive_lanes > self.batch:
            raise ValueError("archive_lanes must not exceed batch")
        if self.target_controller not in {
            "scripted_flee_weave",
            "scripted_flee_adaptive",
            "scripted_flee_mixed",
            "scripted_flee_standoff",
            "scripted_flee_standoff_mixed",
            "frozen_source_policy",
        }:
            raise ValueError("unsupported pursuit target controller")
        if self.target_controller != "scripted_flee_weave" and (
            self.target_controller != "frozen_source_policy"
            and self.world_design is None
        ):
            # The adaptive evader steers away from the arena bound, and only a
            # design world declares one. Refusing here beats running it on a
            # Region world where `arena_radius` describes nothing.
            raise ValueError(
                "adaptive target controllers require world_design"
            )
        if not math.isfinite(self.arena_radius) or self.arena_radius <= 0.0:
            raise ValueError("arena radius must be finite and positive")
        if len(tuple(self.evader_gait_mix)) != 3 or abs(
            sum(self.evader_gait_mix) - 1.0
        ) > 1.0e-6:
            raise ValueError("evader gait mix must be (walk, run, sprint) summing to 1")
        if self.evader_gait_period_ticks < 1:
            raise ValueError("evader gait period must be positive")
        band = tuple(self.evader_standoff_band)
        if len(band) != 2 or not all(math.isfinite(edge) for edge in band):
            raise ValueError("evader standoff band must be two finite distances")
        if not 0.0 < band[0] < band[1]:
            raise ValueError(
                f"evader standoff band must satisfy 0 < inner < outer, got {band}"
            )
        if band[1] >= self.arena_radius:
            raise ValueError(
                f"standoff outer {band[1]:g} does not fit an arena of radius "
                f"{self.arena_radius:g}"
            )
        if self.evader_standoff_strafe_ticks < 1:
            raise ValueError("evader standoff strafe ticks must be positive")
        if self.objective not in {"pursue", "evade"}:
            raise ValueError("objective must be 'pursue' or 'evade'")
        if (
            not math.isfinite(self.learner_stamina_fraction)
            or not 0.0 < self.learner_stamina_fraction <= 1.0
        ):
            raise ValueError("learner_stamina_fraction must be in (0, 1]")
        if (
            not math.isfinite(self.learner_stamina_regen_scale)
            or not 0.0 <= self.learner_stamina_regen_scale <= 1.0
        ):
            raise ValueError("learner_stamina_regen_scale must be in [0, 1]")
        if self.objective == "evade" and self.target_controller != (
            "frozen_source_policy"
        ):
            # A scripted flee target would run away from an evader, leaving
            # nothing to evade and no contact reward to earn.
            raise ValueError(
                "objective 'evade' needs target_controller "
                "'frozen_source_policy', so the opponent actually pursues"
            )
        if (
            not math.isfinite(self.minimum_baseline_visible_fraction)
            or not 0.0 <= self.minimum_baseline_visible_fraction <= 1.0
        ):
            raise ValueError(
                "minimum physical line-of-sight fraction must be in [0, 1]"
            )
        if self.maximum_approximate_kl is not None and (
            not math.isfinite(self.maximum_approximate_kl)
            or self.maximum_approximate_kl <= 0.0
        ):
            raise ValueError("maximum approximate KL must be finite and positive")
        separation = tuple(float(value) for value in self.target_separation_range)
        if (
            len(separation) != 2
            or any(not math.isfinite(value) for value in separation)
            or separation[0] <= 0.0
            # Equal bounds mean a fixed spawn distance, which is a real
            # setting: holding the gap constant isolates closing skill from the
            # spread. Only an inverted range is wrong.
            or separation[1] < separation[0]
        ):
            raise ValueError("target separation range must be finite and ordered")
        unknown_stage = sorted(set(self.stage_options) - _STAGE_OPTION_FIELDS)
        unknown_target = sorted(set(self.target_options) - _TARGET_OPTION_FIELDS)
        if unknown_stage:
            raise ValueError(f"unsupported pursuit stage options: {unknown_stage}")
        if unknown_target:
            raise ValueError(f"unsupported pursuit target options: {unknown_target}")
        object.__setattr__(self, "memory_reset_updates", memory_updates)
        object.__setattr__(self, "target_separation_range", separation)
        object.__setattr__(self, "stage_options", dict(self.stage_options))
        object.__setattr__(self, "target_options", dict(self.target_options))


def pursuit_launch_options() -> dict[str, Any]:
    """Preset workflow plus open stage extension fields for console/ADK callers."""

    return {
        "schema": "arena-pursuit-launch-options-v16",
        "stage": "pursuit_tracking",
        "strategy": "recurrent_ppo",
        "default_preset": "large",
        "presets": {
            "standard": {
                "batch": 64,
                "updates": 64,
                "rollout_steps": 128,
                "evaluation_steps": PURSUIT_DEFAULT_TICKS,
                "evaluation_updates": [0, 8, 16, 32, 64],
                "memory_reset_updates": [],
                "archive_lanes": 8,
            },
            "large": {
                "batch": 256,
                "updates": 64,
                "rollout_steps": 128,
                "evaluation_steps": PURSUIT_DEFAULT_TICKS,
                "evaluation_updates": [0, 8, 16, 32, 64],
                "memory_reset_updates": [],
                "archive_lanes": 8,
            },
            "throughput": {
                "batch": 256,
                "updates": 64,
                "rollout_steps": 256,
                "evaluation_steps": PURSUIT_DEFAULT_TICKS,
                "evaluation_updates": [0, 8, 16, 32, 64],
                "memory_reset_updates": [],
                "archive_lanes": 8,
            },
            "deep": {
                "batch": 256,
                "updates": 128,
                "rollout_steps": 128,
                "evaluation_steps": PURSUIT_DEFAULT_TICKS,
                "evaluation_updates": [0, 8, 16, 32, 64, 96, 128],
                "memory_reset_updates": [],
                "archive_lanes": 8,
            },
        },
        "custom": {
            "stage_options": sorted(_STAGE_OPTION_FIELDS),
            "target_options": sorted(_TARGET_OPTION_FIELDS),
            "fixed": {
                "target_mode": "scripted_flee_weave",
                "target_separation_range": [10.0, 14.0],
                "evaluation_horizon": f"1..{PURSUIT_MAXIMUM_TICKS}",
                "maximum_episode_ticks": PURSUIT_DEFAULT_TICKS,
                "catch_distance": PursuitStageConfig().catch_distance,
                "minimum_chase_ticks": 1,
                "minimum_baseline_visible_fraction": 0.50,
                "maximum_approximate_kl": (
                    f"{_KL_CEILING_PER_HEAD} per head x "
                    f"{len(ARSENAL_POLICY_ACTION_HEAD_SIZES)} heads, "
                    f"{_KL_VIOLATION_PATIENCE} consecutive to stop"
                ),
            },
            "rule": "preset values are defaults; explicit contract fields win",
        },
        "world_backend": {
            "current_training_source": "saved_region_artifact",
            "artifacts": [dict(item) for item in pursuit_region_artifacts()],
            "default_artifact_seed": "deterministic_rotation_from_training_seed",
            "generation": "console_worldgen_v2_procedural_or_authored",
            "persistence": "save_design_then_publish_hash_stamped_jax_artifact",
            "runtime_rule": "materialize_saved_areas_before_jit_never_generate_in_scan",
        },
        "compute": {
            "schema": PURSUIT_COMPUTE_COMPOSITION_SCHEMA,
            "composition": "resolve_static_features_before_one_jax_trace",
            "controls": [
                {
                    "key": "num_envs",
                    "preset_key": "batch",
                    "label": "parallel environments",
                    "kind": "integer",
                    "minimum": 1,
                    "maximum": 512,
                },
                {
                    "key": "rollout_steps",
                    "preset_key": "rollout_steps",
                    "label": "ticks per update",
                    "kind": "integer",
                    "minimum": 1,
                    "maximum": 2048,
                },
                {
                    "key": "world_artifact_seed",
                    "label": "saved world seed",
                    "kind": "enum",
                    "choices": [item["seed"] for item in pursuit_region_artifacts()],
                    "optional": True,
                },
                {
                    "key": "target_controller",
                    "label": "opponent controller",
                    "kind": "enum",
                    "choices": ["scripted_flee_weave", "frozen_source_policy"],
                },
            ],
            "features": [
                {
                    "key": "actor_inference.scripted_slots",
                    "scope": "collector",
                    "mode": "conditional",
                    "condition": "target_controller=scripted_flee_weave",
                    "value": [1],
                    "summary": "skip the target neural forward when its JAX controller is exact",
                },
                {
                    "key": "collector.policy_storage",
                    "scope": "collector",
                    "mode": "enforced",
                    "value": "learner_slot_0_only",
                    "summary": "retain policy tensors only for the trainable actor",
                },
                {
                    "key": "optimizer.state_buffer_donation",
                    "scope": "optimizer",
                    "mode": "enforced",
                    "value": True,
                    "summary": "reuse prior device buffers across compiled updates",
                },
                {
                    "key": "launcher.resident_worker",
                    "scope": "launcher",
                    "mode": "enforced",
                    "value": True,
                    "summary": "reuse imports, CUDA context, and executable cache between jobs",
                },
                {
                    "key": "launcher.adaptive_vram",
                    "scope": "launcher",
                    "mode": "enforced",
                    "value": "auto",
                    "summary": "reserve an idle GPU or allocate on demand when shared",
                },
            ],
            "extension_rule": (
                "controls and features are resolved before JIT; changing a static "
                "shape produces a different executable cache identity"
            ),
        },
    }


@lru_cache(maxsize=1)
def pursuit_region_artifacts() -> tuple[dict[str, Any], ...]:
    """Saved train-split worlds addressable by seed without regeneration."""

    from worlds.terrain import LIBRARY
    from worlds.zones import selection_for

    manifest = json.loads((LIBRARY / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for entry in manifest["entries"]:
        if entry["split"] != "train":
            continue
        seed = int(entry["seed"])
        selection_key = selection_for(seed)
        if selection_key is None:
            continue
        rows.append(
            {
                "seed": seed,
                "semantic_sha256": str(entry["semantic_sha256"]),
                "selection_key": selection_key,
                "source": "saved_native_region",
            }
        )
    return tuple(sorted(rows, key=lambda item: item["seed"]))


#: The `selection_key` a multi-world run used before the field could be left
#: unset. Preserved exactly so no existing multi-world launch changes worlds.
_DEFAULT_SELECTION_KEY = 570_057

#: Per-head share of the joint KL budget. The stop-loss reads a JOINT quantity
#: (see `maximum_approximate_kl`), so the only head-count-stable way to express
#: the ceiling is per head.
#:
#: Raised 0.005 -> 0.010 on 2026-08-20. The 0.005 figure came from
#: 20260819T222021Z-3da75418, whose joint mean was 0.0203 over 12 heads with
#: clip_fraction 0.22 -- a quiet trajectory. It is too tight for a policy that
#: is actually moving. Run 20260820T043730Z-44101aef stopped at update 55 of 256
#: on `ppo_approximate_kl_exceeded` while it was learning WELL:
#:
#:     eval update     0      8     16     32     55
#:     aim error   60.5   59.1   56.5   45.8   30.1   degrees
#:     return     129.4  131.3  135.8  154.5  186.6
#:
#: Return rose monotonically, aim error fell 50%, and entropy was falling
#: (7.62 -> 7.56), so there was no divergence to catch -- the guardrail cut
#: productive learning. Joint KL during that stretch ran 0.018-0.081, peaking at
#: 0.0807. A 0.010/head ceiling is 0.12 joint: ~50% headroom over the observed
#: productive maximum, and still far below a real divergence, which shows as KL
#: climbing WITH return collapsing rather than climbing beside it.
#:
#: If this proves too loose, prefer adding a "return is not improving" term to
#: the stop condition over lowering the number again: KL alone cannot tell a
#: policy that is moving fast from one that is falling apart.
_KL_CEILING_PER_HEAD = 0.010

#: Consecutive over-ceiling updates required before stopping. Learning is slow
#: and noisy, so this has to be long enough to sit through a stretch of large
#: updates that the policy recovers from. At 3 it ended a run after roughly
#: forty seconds of training whose evaluation showed the policy essentially
#: unchanged, which is a false alarm rather than a saved run.
#:
#: A non-finite KL is exempt and still stops immediately: that is an arithmetic
#: failure the policy cannot come back from, not a policy moving quickly.
_KL_VIOLATION_PATIENCE = 32

#: Largest typical ``|log ratio|`` accepted alongside an over-ceiling KL.
#:
#: ``approximate_kl`` averages ``exp(log_ratio)``, so a handful of extreme
#: samples set it: two adjacent updates measured 689 and 6.86e4 from the same
#: median and the same 99.9th percentile. On a twelve-head joint distribution it
#: never falls near the derived ceiling, so on its own it stops every run.
#:
#: The median is what tracks the policy, and it is read as a second opinion
#: rather than a replacement: a run stops when the KL is over its ceiling *and*
#: the median agrees the distribution really moved. At 1.0 a typical sample's
#: probability has changed by e**1.0, nearly threefold, which is far outside
#: what a 0.2 clip band intends and is real divergence rather than tail noise.
_MEDIAN_LOG_RATIO_CEILING = 1.0


def _evaluation_device():
    """Where evaluation runs: off the trainer's device when there is one.

    Requires a CPU device to exist. Under ``JAX_PLATFORMS=cuda`` there is none
    and `jax.devices("cpu")` raises "Unknown backend cpu", so the launcher asks
    for ``cuda,cpu`` -- CUDA still first, and therefore still the default that
    training uses. Falls back to the default device rather than failing a run:
    a shared allocator is a risk, an exception here is a certainty.
    """

    try:
        return jax.devices("cpu")[0]
    except RuntimeError:
        return jax.devices()[0]


def pursuit_update_diverged(
    approximate_kl: float,
    median_log_ratio: float,
    maximum_approximate_kl: float,
    median_ceiling: float = _MEDIAN_LOG_RATIO_CEILING,
) -> bool:
    """Whether one update looks like divergence rather than tail noise.

    Non-finite is divergence outright. Otherwise both statistics have to agree:
    the exponential mean says the ratio is large somewhere, and the median says
    it is large typically.
    """

    if not math.isfinite(approximate_kl) or not math.isfinite(median_log_ratio):
        return True
    return (
        approximate_kl > maximum_approximate_kl and median_log_ratio > median_ceiling
    )


def _resolved_maximum_approximate_kl(run: PursuitRunConfig) -> float:
    """Return the joint KL ceiling, derived from head count when unset."""

    if run.maximum_approximate_kl is not None:
        return float(run.maximum_approximate_kl)
    return _KL_CEILING_PER_HEAD * len(ARSENAL_POLICY_ACTION_HEAD_SIZES)


def _resolved_region_selection(run: PursuitRunConfig) -> int:
    artifacts = pursuit_region_artifacts()
    if run.world_artifact_seed is None:
        # An EXPLICIT key wins at any `world_count`. This used to be reached
        # only when `world_count > 1`, so a one-world run silently discarded the
        # key and took the curated `seed % len(artifacts)` pick instead. Two
        # runs differing only in `selection_key` therefore loaded the SAME world
        # and reported `visible_fraction` identical to 17 significant figures --
        # which is how the bug was found, since two distinct worlds cannot do
        # that. A crafted key pins which world sorts first, which is exactly
        # what a one-world run wants; with several worlds resident it is a
        # subset choice. Both are the caller's declared intent.
        if run.selection_key is not None:
            return int(run.selection_key)
        if run.world_count > 1:
            # Several worlds resident and no declared key: keep the historical
            # default. The curated pick below names a single world, so it is the
            # wrong fallback once the set is bigger than one.
            return _DEFAULT_SELECTION_KEY
        return int(artifacts[run.seed % len(artifacts)]["selection_key"])
    match = next(
        (item for item in artifacts if item["seed"] == run.world_artifact_seed),
        None,
    )
    if match is None:
        raise ValueError(
            f"saved training world seed {run.world_artifact_seed} is unavailable"
        )
    return int(match["selection_key"])


@lru_cache(maxsize=2)
def _flattest_world_seeds(
    world_count: int,
    selection_key: int,
    split: str = "train",
) -> tuple[int, ...]:
    """The flattest `world_count` worlds in the split, flattest first.

    Distinct from `_mixed_terrain_world_seeds`, which is deliberately half flat
    and half ranked so a learner meets terrain it has to generalise to. This is
    the curriculum floor instead: ground a starting agent can actually walk on.

    `selection_key` still scopes the candidates, for the same reason the mixed
    variant does: `zones.library()` spans held-out seeds too, and naming one
    makes the library refuse the selection at scene build.
    """

    from worlds import zones

    largest: dict[int, object] = {}
    for zone in zones.library():
        if zone.seed not in largest or zone.size > largest[zone.seed].size:
            largest[zone.seed] = zone
    allowed = set(zones.selected_seeds(selection_key, count=len(largest), split=split))
    flatness = {
        seed: zone.terrain.get("open_flat", 0) / max(zone.size, 1)
        for seed, zone in largest.items()
        if seed in allowed
    }
    if len(flatness) < world_count:
        raise ValueError(
            f"terrain census has {len(flatness)} seeds, fewer than the "
            f"{world_count} worlds requested"
        )
    # Ties broken by seed so the set is reproducible, not census-order dependent.
    ranked = sorted(flatness, key=lambda seed: (-flatness[seed], seed))
    return tuple(ranked[:world_count])


def _mixed_terrain_world_seeds(
    world_count: int,
    selection_key: int,
    split: str = "train",
) -> tuple[int, ...]:
    """Half the working set flattest-first, half by the ordinary digest rank.

    `selection_key` is a rank over the whole split, not a filter, so it can only
    ever say "the N that hash first" -- `worlds.zones.selection_key_for_terrain`
    documents that no key composes a deliberate mix. This builds one explicitly:
    the flat half gives the learner ground where the task is legible, the ranked
    half keeps the varied terrain it has to generalise to.

    The ranked half is drawn with the run's own `selection_key`, so a key that
    was *searched* for flatness biases it too. Measured 2026-08-19: key 326's 16
    worlds span 0.342-0.651 `open_flat` share against a library median of 0.377,
    i.e. every one of them is above median. Pass a key that was not terrain
    searched if the second half is meant to be representative.

    Flatness is the `open_flat` share of each seed's largest zone, which is the
    arena an episode is almost always placed in. `zones.library()` costs ~11.6s
    once (measured, 288 train seeds) against a ~412s scene build, and is cached.
    """

    from worlds import zones

    largest: dict[int, object] = {}
    for zone in zones.library():
        if zone.seed not in largest or zone.size > largest[zone.seed].size:
            largest[zone.seed] = zone
    # `zones.library()` spans the WHOLE library, train and held-out alike, so the
    # flattest worlds in it are not all selectable. Restricting to the split is
    # not a tidiness fix: without it the flat half names held-out seeds and the
    # library refuses the selection at scene build, which is exactly how run
    # 20260819T082417Z-81cbdfa8 died on seed 1269345179. `selected_seeds` slices
    # the manifest, so an oversized count safely returns the whole split.
    allowed = set(zones.selected_seeds(selection_key, count=len(largest), split=split))
    flatness = {
        seed: zone.terrain.get("open_flat", 0) / max(zone.size, 1)
        for seed, zone in largest.items()
        if seed in allowed
    }
    if len(flatness) < world_count:
        raise ValueError(
            f"terrain census has {len(flatness)} seeds, fewer than the "
            f"{world_count} worlds requested"
        )
    # Ties broken by seed so the set is reproducible, not census-order dependent.
    flat_half = sorted(flatness, key=lambda seed: (-flatness[seed], seed))
    chosen = list(flat_half[: world_count // 2])
    seen = set(chosen)
    for seed in zones.selected_seeds(
        selection_key, count=len(flatness), split=split
    ):
        if len(chosen) >= world_count:
            break
        if seed in seen or seed not in flatness:
            continue
        chosen.append(seed)
        seen.add(seed)
    if len(chosen) != world_count:
        raise ValueError(
            f"mixed terrain selection resolved {len(chosen)} worlds, "
            f"expected {world_count}"
        )
    return tuple(chosen)


#: Which lane modes each scripted controller runs. `scripted_flee_mixed`
#: alternates the two evaders across lanes, so one run measures the learner
#: against both under an identical policy, world and spawn distribution --
#: comparing them across runs would confound the evader with everything else.
_SCRIPTED_TARGET_LANE_MODES = {
    "scripted_flee_weave": (TargetMode.FLEE_WEAVE,),
    "scripted_flee_adaptive": (TargetMode.FLEE_ADAPTIVE,),
    "scripted_flee_mixed": (TargetMode.FLEE_WEAVE, TargetMode.FLEE_ADAPTIVE),
    "scripted_flee_standoff": (TargetMode.FLEE_STANDOFF,),
    "scripted_flee_standoff_mixed": (
        TargetMode.FLEE_ADAPTIVE,
        TargetMode.FLEE_STANDOFF,
    ),
}


@dataclass
class _DesignFixture:
    """The two fixture seams a design world still has to answer.

    A plane has no captured geometry, so the Region seams -- navigation,
    explosions, the block-action surface -- are not stubbed here. They are not
    bound at all; the flat collector never asks for them.
    """

    metadata: dict[str, Any]
    reset_provider: Any


@lru_cache(maxsize=4)
def _materialize_design(
    design_id: str,
    loadout: str,
    opponent: str,
    batch: int,
    target_separation_range: tuple[float, float],
    arena_radius: float,
    agent_stamina_fraction: float = 1.0,
    stamina_regen_scale: float = 1.0,
):
    """Build a pursuit arena from an authored design, with no Region artifact."""

    from hytalegym.jax.combat import default_combat_params
    from hytalegym.jax.combat.arsenal.profiles import hytale_0_5_7_entity_loadouts
    from hytalegym.jax.combat.arsenal.runtime import arsenal_runtime_config
    from worlds import designs
    from worlds.flat import flat_arena

    design = designs.by_id(design_id)
    params = default_combat_params(microticks=1, target_active=False)
    runtime = arsenal_runtime_config(
        hytale_0_5_7_entity_loadouts(((loadout, opponent),) * batch),
        opponent_controller_mask=jnp.asarray((False, False)),
    )
    arena = flat_arena(
        design,
        params=params,
        config=runtime,
        target_separation_range=target_separation_range,
        arena_radius=arena_radius,
        agent_stamina_fraction=agent_stamina_fraction,
        stamina_regen_scale=stamina_regen_scale,
    )
    # `runtime` rides along so the caller binds the same config the reset was
    # built against; a second `arsenal_runtime_config` would be a different
    # object and the loadout batch would not match the reset keys.
    return SimpleNamespace(
        design=design,
        params=arena.params,
        runtime=runtime,
        reset_provider=arena.reset_provider,
        capability_provider=arena.capability_provider,
        metadata=arena.metadata,
    )


def _design_world_metadata(arena, run: "PursuitRunConfig") -> dict[str, Any]:
    """Replay identity for a design world, in the keys the replay block reads.

    The Region keys are answered with design facts rather than left missing, so
    every consumer of `region_reset` keeps working and the recorded identity
    says what actually ran. `artifact_semantic_sha256` carries the design's
    seed-independent recipe digest, which is the honest world identity here.
    """

    design = arena.design
    low, high = run.target_separation_range
    return {
        **arena.metadata,
        "world_source": "authored_design",
        "artifact_seed": int(design.seed),
        "artifact_semantic_sha256": design.recipe_digest.upper(),
        "region_library_semantic_sha256": design.recipe_digest.upper(),
        # A plane has no traversal graph, so there is no node to name. -1 says
        # that rather than pointing at a node that does not exist.
        "source_node": -1,
        "destination_node": -1,
        "target_initial_horizontal_separation": (low + high) / 2.0,
        "working_set_capacity": 1,
        "distinct_environment_worlds": 1,
        "environment_diversity": True,
        "distinct_environment_spawns": int(run.batch),
        "distinct_environment_targets": int(run.batch),
    }


@lru_cache(maxsize=4)
def _materialize_region(
    loadout: str,
    opponent: str,
    batch: int,
    selection_key: int,
    node_seed: int,
    target_separation_range: tuple[float, float],
    world_count: int = 1,
    environment_diversity: bool = False,
    selection_seeds: tuple[int, ...] | None = None,
):
    """Keep recent immutable saved-world fixtures resident between GPU jobs."""

    from worlds.region import load_region

    return load_region(
        selection_seeds=selection_seeds,
        weapons=(loadout,) * batch,
        target_weapons=(opponent,) * batch,
        policy_controlled_targets=True,
        target_active=False,
        native_evidence=False,
        selection_key=selection_key,
        node_seed=node_seed,
        target_separation_range=target_separation_range,
        require_initial_line_of_sight=True,
        working_set_capacity=world_count,
        environment_diversity=environment_diversity,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _host(value):
    if hasattr(value, "_asdict"):
        return {name: _host(item) for name, item in value._asdict().items()}
    array = np.asarray(jax.device_get(value))
    if array.ndim == 0:
        scalar = array.item()
        return bool(scalar) if isinstance(scalar, np.bool_) else scalar
    return array.tolist()


def _numeric_scalars(value: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten scalar telemetry for live charts without copying rollout tensors."""

    out: dict[str, float] = {}
    for name, item in value.items():
        key = f"{prefix}.{name}" if prefix else name
        if isinstance(item, Mapping):
            out.update(_numeric_scalars(item, key))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            out[key] = float(item)
    return out


def _clock(seconds: float) -> str:
    minutes, remainder = divmod(int(seconds), 60)
    return f"{minutes:3d}:{remainder:02d}"


def _number(value: Any) -> str:
    """Compact enough that a column of these stays scannable."""

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return str(value)
    number = float(value)
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "+Inf" if number > 0 else "-Inf"
    if number and (abs(number) >= 10000.0 or abs(number) < 0.001):
        return f"{number:.2e}"
    return f"{number:.3g}"


def _percent(value: Any) -> str:
    return "-" if not isinstance(value, (int, float)) else f"{float(value) * 100:.2g}%"


def _wrap(parts: list[str], width: int = 72) -> list[str]:
    """Pack ``parts`` into ` | `-joined lines that fit inside ``width``.

    Detail belongs under an event, not off the right-hand side of it, so a long
    run of annotations becomes more lines rather than a wider one.
    """

    lines: list[str] = []
    for part in parts:
        if lines and len(lines[-1]) + len(part) + 3 <= width:
            lines[-1] += f" | {part}"
        else:
            lines.append(part)
    return lines


def _training_update_line(payload: Mapping[str, Any], ceiling: float | None) -> str:
    """The values that decide whether a run survives the next update.

    Two lines at most: the per-update headline always, and a second one only
    when something needs explaining. Neither is allowed to run off the side of
    a terminal, so detail goes downward rather than rightward.
    """

    metrics = payload.get("metrics") or {}
    kl = metrics.get("loss.approximate_kl")
    over = isinstance(kl, float) and ceiling is not None and kl > ceiling
    steps = payload.get("steps_per_second")
    headline = [
        f"kl {_number(kl)}" + (" OVER" if over else ""),
        f"clip {_percent(metrics.get('loss.clip_fraction'))}",
        f"ent {_number(metrics.get('loss.entropy'))}",
        f"loss {_number(metrics.get('loss.total_loss'))}",
        f"{steps:.0f}/s" if isinstance(steps, (int, float)) else "-/s",
    ]
    update = payload.get("update")
    lines = [f"u{update}/{payload.get('updates')}".ljust(13) + " | ".join(headline)]

    detail: list[str] = []
    ratio_shape = [
        metrics.get(f"loss.{name}")
        for name in ("log_ratio_median", "log_ratio_p999", "maximum_log_ratio")
    ]
    if over and all(value is not None for value in ratio_shape):
        # An over-ceiling KL is a mean of exponentials, so it cannot say whether
        # the whole distribution moved or one thin tail carried it. These three
        # answer that, and are only worth a line when the KL is the problem.
        median, p999, maximum = (_number(value) for value in ratio_shape)
        detail.append(f"|ratio| med {median} p99.9 {p999} max {maximum}")
    if metrics.get("loss.incomparable_fraction"):
        detail.append(f"incomparable {_percent(metrics['loss.incomparable_fraction'])}")
    if metrics.get("invalid_trainable_steps"):
        detail.append(f"invalid steps {_number(metrics['invalid_trainable_steps'])}")
    if not metrics.get("episodes_completed"):
        detail.append("NO EPISODES COMPLETED")
    if update == 1:
        detail.append("includes program compile")
    lines.extend(" " * 13 + line for line in _wrap(detail))
    return "\n".join(lines)


def _evaluation_line(payload: Mapping[str, Any]) -> str:
    metrics = payload.get("metrics") or {}
    selection = payload.get("selection") or {}
    aim = metrics.get("learner.mean_visible_aim_error_degrees")
    verdict = (
        "incumbent"
        if selection.get("became_incumbent")
        else "rejected: " + (", ".join(selection.get("flags") or []) or "not eligible")
    )
    return (
        f"eval u{payload.get('update')}".ljust(13)
        + f"aim {_number(aim)} deg | "
        f"success {_number(selection.get('success_rate'))} | "
        f"score {_number(selection.get('objective_score'))} | {verdict}"
    )


def _progress_line(
    elapsed: float, kind: str, payload: Mapping[str, Any], context: Mapping[str, Any]
) -> str:
    """One scannable line per event; the whole payload goes to the event log."""

    get = payload.get
    if kind == "preparing":
        body = (
            "start".ljust(13) + f"{get('backend')} {get('device')} | "
            f"{get('batch')}x{get('rollout_steps')} | {get('updates')} updates"
        )
    elif kind == "scene_build":
        body = "scene".ljust(13) + f"building {get('world_count')} worlds"
    elif kind == "scene_ready":
        cached = "cached" if get("materialization_cache_hit") else "cache miss"
        body = "scene".ljust(13) + (
            f"ready in {_number(get('scene_build_seconds'))}s ({cached}) | "
            f"{get('distinct_environment_worlds')} worlds, "
            f"{get('distinct_environment_spawns')} spawns"
        )
    elif kind == "program_ready":
        body = "compile".ljust(13) + f"{_number(get('program_build_seconds'))}s"
    elif kind == "training_start":
        body = "training".ljust(13) + (
            f"lr {_number(get('learning_rate'))} | "
            f"ent {_number(get('entropy_coefficient'))} | "
            f"kl ceiling {_number(get('maximum_approximate_kl'))} | "
            f"{get('action_head_count')} heads"
        )
    elif kind == "training_update":
        body = _training_update_line(payload, context.get("maximum_approximate_kl"))
    elif kind == "evaluation":
        body = _evaluation_line(payload)
    elif kind == "stop_loss":
        body = "STOPPED".ljust(13) + (
            f"{get('reason')} at u{get('update')} | "
            f"kl {_number(get('approximate_kl'))} > "
            f"{_number(get('maximum_approximate_kl'))} | "
            f"med {_number(get('log_ratio_median'))} > "
            f"{_number(get('maximum_log_ratio_median'))}"
        )
    elif kind == "failed":
        body = "FAILED".ljust(13) + f"{get('error_type')}: {get('message')}"
    elif kind == "completed":
        body = "done".ljust(13) + f"{_number(get('wall_seconds'))}s"
    else:
        return ""
    clock = _clock(elapsed)
    # Continuation lines carry no clock, so they indent to the same column and
    # read as detail belonging to the event above them.
    first, *rest = body.split("\n")
    return "\n".join(
        [f"{clock}  {first}", *(" " * (len(clock) + 2) + line for line in rest)]
    )


def _assignment(batch: int):
    return policy_actor_assignment(
        jnp.broadcast_to(jnp.asarray((0, 1), dtype=jnp.int32), (batch, 2)),
        jnp.broadcast_to(jnp.asarray((0, 1), dtype=jnp.int32), (batch, 2)),
        trainable=jnp.broadcast_to(
            jnp.asarray((True, False), dtype=jnp.bool_), (batch, 2)
        ),
        entity_count=2,
    )


_PURSUIT_TELEMETRY_NAMES = (
    "visible",
    "bearing_sin",
    "bearing_cos",
    "planar_distance",
    "relative_up",
    "target_context_available",
    "learner_pitch",
)
_PURSUIT_TELEMETRY_INDEX = {
    name: index for index, name in enumerate(_PURSUIT_TELEMETRY_NAMES)
}


def _pursuit_observation_telemetry(observation):
    target_names = _PURSUIT_TELEMETRY_NAMES[:5]
    target_indices = jnp.asarray(
        tuple(TARGET_FLOAT_FEATURES.index(name) for name in target_names),
        dtype=jnp.int32,
    )
    pitch = SELF_FLOAT_FEATURES.index("pitch")
    return jnp.concatenate(
        (
            jnp.take(observation.structured.base.target_f32, target_indices, axis=-1),
            observation.structured.base.target_mask[..., None].astype(jnp.float32),
            observation.structured.base.self_f32[..., pitch : pitch + 1],
        ),
        axis=-1,
    )


def _vertical_aim_offsets(params):
    eye = jnp.stack((params.agent_eye_offset[1], params.target_eye_offset[1]))
    center = jnp.stack(
        (
            (params.agent_bounds[1] + params.agent_bounds[4]) * 0.5,
            (params.target_bounds[1] + params.target_bounds[4]) * 0.5,
        )
    )
    return eye, center


def _facing_reset(state, params):
    """Spawn the learner already looking at its target.

    Hytale bounds how far an actor rotates in one tick and the stage keeps that
    bound (`maximum_turn_degrees`), so a random spawn heading is aim error the
    learner physically cannot undo for many ticks -- at 12 degrees a tick, a
    137 degree spawn error costs about eleven before pursuit can even begin.
    The reset already certifies the target is *visible*; this makes it *faced*
    too, so the lesson scores pursuit rather than spawn orientation.

    Yaw and pitch use the same convention `_metrics` scores against, including
    the target-centre and learner-eye offsets, so a fresh spawn reads as zero
    aim error rather than a residual the learner is charged for.
    """

    combat = state.combat
    delta = combat.position[:, TARGET_ENTITY] - combat.position[:, AGENT_ENTITY]
    planar_distance = jnp.linalg.norm(delta[..., (0, 2)], axis=-1)
    eye, center = _vertical_aim_offsets(params)
    vertical = delta[..., 1] + center[TARGET_ENTITY] - eye[AGENT_ENTITY]
    facing_yaw = jnp.degrees(jnp.arctan2(-delta[..., 0], -delta[..., 2]))
    facing_pitch = jnp.degrees(
        jnp.arctan2(vertical, jnp.maximum(planar_distance, jnp.float32(1.0e-6)))
    )
    return state._replace(
        combat=combat._replace(
            # `yaw` is per entity; the target keeps whatever the reset gave it.
            # `pitch`/`desired_*` are the agent's own scalars.
            yaw=combat.yaw.at[:, AGENT_ENTITY].set(facing_yaw),
            pitch=facing_pitch,
            desired_yaw=facing_yaw,
            desired_pitch=facing_pitch,
            # The head is steered separately and is bounded to a window around
            # the body, so orienting the body alone would spawn the learner
            # with its head pinned at the edge of that window, looking 45
            # degrees off the target this reset exists to face.
            agent_head_yaw=facing_yaw,
            agent_head_pitch=facing_pitch,
        )
    )


def _angular_separation(yaw_error, pitch, desired_pitch):
    yaw = jnp.deg2rad(yaw_error)
    actual = jnp.deg2rad(pitch)
    desired = jnp.deg2rad(desired_pitch)
    cosine = jnp.sin(actual) * jnp.sin(desired) + jnp.cos(actual) * jnp.cos(
        desired
    ) * jnp.cos(yaw)
    return jnp.degrees(jnp.arccos(jnp.clip(cosine, -1.0, 1.0)))


def _metrics(result, roles, params, stage) -> dict[str, float | int]:
    rollout = result.rollout
    batch = roles.learner_policy_slot.shape[0]
    lanes = jnp.arange(batch, dtype=jnp.int32)
    observation = rollout.observation_telemetry[:, lanes, roles.learner_policy_slot]
    action = rollout.action[:, lanes, roles.learner_policy_slot]
    valid = rollout.valid[:, lanes, roles.learner_policy_slot]

    def target(name):
        return observation[..., _PURSUIT_TELEMETRY_INDEX[name]]

    visible = valid & (target("visible") > 0.5)
    context_available = valid & (target("target_context_available") > 0.5)
    observed_distance = target("planar_distance") * jnp.float32(24.0)
    eye, center = _vertical_aim_offsets(params)
    learner_pitch = target("learner_pitch") * jnp.float32(90.0)
    # Movement and dodge share one published head now, so both are read off
    # the same factor: any non-idle choice is motion, and the dodge branch sits
    # above the gait ring.
    locomotion = ARSENAL_POLICY_ACTION_HEAD_NAMES.index("locomotion_gait_compass")
    jump = ARSENAL_POLICY_ACTION_HEAD_NAMES.index("jump_off_on")
    yaw = ARSENAL_POLICY_ACTION_HEAD_NAMES.index("yaw_delta_bins")
    pitch = ARSENAL_POLICY_ACTION_HEAD_NAMES.index("pitch_delta_bins")
    yaw_turn = action[..., yaw] != ARSENAL_POLICY_ACTION_HEAD_SIZES[yaw] // 2
    pitch_turn = action[..., pitch] != ARSENAL_POLICY_ACTION_HEAD_SIZES[pitch] // 2
    positions = rollout.entity_position
    learner_position = positions[:, lanes, roles.learner_actor_index]
    target_position = positions[:, lanes, roles.target_actor_index]
    delta = target_position - learner_position
    distance = jnp.linalg.norm(delta[..., (0, 2)], axis=-1)
    vertical = (
        delta[..., 1]
        + center[roles.target_actor_index]
        - eye[roles.learner_actor_index]
    )
    desired_pitch = jnp.degrees(
        jnp.arctan2(vertical, jnp.maximum(distance, jnp.float32(1.0e-6)))
    )
    desired_yaw = jnp.degrees(jnp.arctan2(-delta[..., 0], -delta[..., 2]))
    learner_yaw = rollout.entity_yaw[:, lanes, roles.learner_actor_index]
    yaw_delta = jnp.abs((learner_yaw[1:] - learner_yaw[:-1] + 180.0) % 360.0 - 180.0)
    yaw_error = jnp.abs((desired_yaw - learner_yaw + 180.0) % 360.0 - 180.0)
    pitch_error = jnp.abs(desired_pitch - learner_pitch)
    aim_error = _angular_separation(yaw_error, learner_pitch, desired_pitch)
    denominator = jnp.maximum(jnp.sum(visible), 1)
    valid_denominator = jnp.maximum(jnp.sum(valid), 1)
    target_path = jnp.linalg.norm(
        jnp.diff(target_position[..., (0, 2)], axis=0), axis=-1
    )
    learner_path = jnp.linalg.norm(
        jnp.diff(learner_position[..., (0, 2)], axis=0), axis=-1
    )
    learner_vertical = jnp.abs(jnp.diff(learner_position[..., 1], axis=0))
    learner_done = rollout.done[:, lanes, roles.learner_policy_slot]
    learner_start = rollout.episode_start[:, lanes, roles.learner_policy_slot]
    same_episode = valid[1:] & valid[:-1] & ~learner_start[1:] & ~learner_done[:-1]
    learner_motion = same_episode & (learner_path > jnp.float32(1.0e-3))
    target_motion = same_episode & (target_path > jnp.float32(1.0e-3))
    motion_denominator = jnp.maximum(jnp.sum(same_episode), 1)
    deadband = jnp.float32(stage.stationary_deadband_blocks)
    stationary = same_episode & (learner_path <= deadband)
    _, stationary_runs = jax.lax.scan(
        lambda count, row: (
            jnp.where(row, count + jnp.int32(1), jnp.int32(0)),
            jnp.where(row, count + jnp.int32(1), jnp.int32(0)),
        ),
        jnp.zeros((batch,), dtype=jnp.int32),
        stationary,
    )
    stuck = stationary_runs >= jnp.int32(stage.stuck_patience_ticks)
    jump_requested = valid & (action[..., jump] != 0)
    dodge_requested = valid & (
        action[..., locomotion] >= jnp.int32(ARSENAL_POLICY_LOCOMOTION_DODGE_START)
    )
    # The locomotion head is idle, then a ring of `gait x compass`, then dodge:
    # `split_locomotion_choice` reads gait as `(choice - 1) // 8 + 1`. So the
    # sprint band is the 8 consecutive choices whose gait resolves to
    # GAIT_SPRINT. Derived rather than written as 17..24 so it follows the head
    # if the gait count or compass width ever moves.
    _sprint_first = 1 + (GAIT_SPRINT - 1) * ARSENAL_POLICY_COMPASS_DIRECTIONS
    sprint_requested = (
        valid
        & (action[..., locomotion] >= jnp.int32(_sprint_first))
        & (
            action[..., locomotion]
            < jnp.int32(_sprint_first + ARSENAL_POLICY_COMPASS_DIRECTIONS)
        )
    )
    # Requesting a sprint is not getting one: it is refused off-axis and when
    # the bar is empty, so the request rate alone cannot tell you whether the
    # learner is actually moving at 7.0 b/s. Stamina spent is the evidence that
    # it was granted -- a sprint is the only thing that drains this bar.
    # `entity_locomotion_stamina`, NOT `entity_stamina`: the latter is the
    # guard/ability resource, which a sprint never touches. Reading it here
    # would report a flat bar and look exactly like "the agent never sprinted".
    learner_stamina = rollout.entity_locomotion_stamina[
        :, lanes, roles.learner_actor_index
    ]
    learner_grounded = rollout.entity_grounded[:, lanes, roles.learner_actor_index]
    takeoff = (
        same_episode
        & learner_grounded[:-1]
        & ~learner_grounded[1:]
        & jump_requested[1:]
    )
    jump_denominator = jnp.maximum(jnp.sum(jump_requested[1:] & same_episode), 1)
    learner_path = jnp.where(same_episode, learner_path, 0.0)
    target_path = jnp.where(same_episode, target_path, 0.0)
    learner_success = rollout.completed_episode_success[
        :, lanes, roles.learner_policy_slot
    ]
    completed_length = rollout.completed_episode_length[
        :, lanes, roles.learner_policy_slot
    ]
    episode_count = jnp.sum(learner_done)
    return {
        "valid_fraction": float(jnp.mean(valid)),
        "mean_visible_aim_error_degrees": float(
            jnp.sum(jnp.where(visible, aim_error, 0.0)) / denominator
        ),
        "mean_visible_yaw_error_degrees": float(
            jnp.sum(jnp.where(visible, yaw_error, 0.0)) / denominator
        ),
        "mean_visible_pitch_error_degrees": float(
            jnp.sum(jnp.where(visible, pitch_error, 0.0)) / denominator
        ),
        "mean_visible_target_elevation_degrees": float(
            jnp.sum(jnp.where(visible, jnp.abs(desired_pitch), 0.0)) / denominator
        ),
        "vertical_aim_exercised_fraction": float(
            jnp.sum(visible & (jnp.abs(desired_pitch) >= 1.0)) / denominator
        ),
        "aligned_fraction": float(jnp.sum(visible & (aim_error <= 10.0)) / denominator),
        "visible_fraction": float(jnp.mean(visible)),
        "target_context_available_fraction": float(jnp.mean(context_available)),
        "mean_omniscient_aim_error_degrees": float(
            jnp.sum(jnp.where(valid, aim_error, 0.0)) / valid_denominator
        ),
        "mean_observed_distance": float(
            jnp.sum(jnp.where(visible, observed_distance, 0.0)) / denominator
        ),
        "mean_distance": float(
            jnp.sum(jnp.where(valid, distance, 0.0)) / valid_denominator
        ),
        "initial_distance": float(jnp.mean(distance[0])),
        "final_distance": float(jnp.mean(distance[-1])),
        "minimum_distance": float(jnp.min(jnp.where(valid, distance, jnp.inf))),
        "moving_fraction": float(jnp.mean((action[..., locomotion] != 0) & valid)),
        "learner_motion_fraction": float(jnp.sum(learner_motion) / motion_denominator),
        "stationary_fraction": float(jnp.sum(stationary) / motion_denominator),
        "stuck_fraction": float(jnp.sum(stuck) / motion_denominator),
        "maximum_stationary_ticks": int(jnp.max(stationary_runs, initial=0)),
        "jump_request_fraction": float(jnp.sum(jump_requested) / valid_denominator),
        "jump_takeoff_fraction": float(jnp.sum(takeoff) / jump_denominator),
        "dodge_request_fraction": float(jnp.sum(dodge_requested) / valid_denominator),
        "airborne_fraction": float(
            jnp.sum(valid & ~learner_grounded) / valid_denominator
        ),
        "vertical_motion_fraction": float(
            jnp.sum(same_episode & (learner_vertical > deadband)) / motion_denominator
        ),
        "target_motion_fraction": float(jnp.sum(target_motion) / motion_denominator),
        # `valid_denominator`, not `denominator`: `sprint_requested` is masked by
        # `valid`, while `denominator` counts *visible* ticks, which are a subset.
        # Dividing one by the other reported ~51% when the true rate was ~36%.
        "sprint_request_fraction": float(
            jnp.sum(sprint_requested) / valid_denominator
        ),
        # Falls to zero if sprint is requested but never granted, which is the
        # state every run through 29d65816 was in: 21 requests, 0 grants, the
        # bar pinned at full.
        "sprint_stamina_spent": float(
            jnp.sum(jnp.maximum(learner_stamina[:-1] - learner_stamina[1:], 0.0))
            / jnp.maximum(episode_count, 1)
        ),
        # Same mismatch as above, and this one was self-evidently wrong: the bar
        # is clipped to [PLAYER_STAMINA_MINIMUM, PLAYER_STAMINA_MAXIMUM] = [-4, 10],
        # yet a valid-masked sum over a visible-count denominator reported a mean
        # of 14.22. A mean cannot exceed the maximum of its inputs.
        "mean_stamina": float(
            jnp.sum(jnp.where(valid, learner_stamina, 0.0)) / valid_denominator
        ),
        "turning_fraction": float(jnp.mean((yaw_turn | pitch_turn) & valid)),
        "yaw_turning_fraction": float(jnp.mean(yaw_turn & valid)),
        "pitch_turning_fraction": float(jnp.mean(pitch_turn & valid)),
        "maximum_yaw_delta_degrees": float(
            jnp.max(jnp.where(same_episode, yaw_delta, 0.0), initial=0.0)
        ),
        "turn_cap_violation_fraction": float(
            jnp.sum(
                same_episode
                & (yaw_delta > jnp.float32(stage.maximum_turn_degrees + 1.0e-3))
            )
            / motion_denominator
        ),
        "aligned_turn_fraction": float(
            jnp.sum(visible & (aim_error <= 10.0) & (yaw_turn | pitch_turn))
            / denominator
        ),
        "learner_path_length": float(jnp.mean(jnp.sum(learner_path, axis=0))),
        "target_path_length": float(jnp.mean(jnp.sum(target_path, axis=0))),
        "episodes": int(episode_count),
        "catches": int(jnp.sum(learner_success)),
        "mean_episode_length": float(
            jnp.sum(jnp.where(learner_done, completed_length, 0.0))
            / jnp.maximum(episode_count, 1)
        ),
        "maximum_episode_length": int(jnp.max(completed_length)),
        "mean_return": float(
            jnp.mean(
                jnp.sum(rollout.reward[:, lanes, roles.learner_policy_slot], axis=0)
            )
        ),
        "mean_target_return": float(
            jnp.mean(
                jnp.sum(rollout.reward[:, lanes, roles.target_policy_slot], axis=0)
            )
        ),
        "target_speed_scale": float(params.target_chase_speed),
    }


def _lane_world_identity(
    world_identity: Mapping[str, object],
    lane: int,
) -> dict[str, object]:
    """Name the world THIS lane actually ran in, not the working set's slot 0.

    The run-level identity carries ``artifact_seed``/``artifact_semantic_sha256``
    for the first resident world only. Stamping that onto a per-lane replay is
    not a rounding error -- it is a different world, with different terrain at
    different coordinates, and a consumer resolving terrain from it draws
    geometry the episode never touched.

    When the working set cannot be resolved the run-level ``artifact_seed`` is
    left in place, because the terrain reference is built from it unconditionally
    and removing it would trade a mislabelled replay for a crashed one. That case
    is instead marked ``lane_world_resolved: False``, which is the signal a
    consumer must check before trusting the seed on a multi-world run.
    """

    resolved = dict(world_identity)
    world_ids = world_identity.get("environment_world_id") or ()
    seeds = world_identity.get("resident_artifact_seed") or ()
    shas = world_identity.get("resident_artifact_semantic_sha256") or ()
    if lane >= len(world_ids):
        resolved["lane_world_resolved"] = False
        return resolved
    slot = int(world_ids[lane])
    if slot >= len(seeds):
        resolved["lane_world_resolved"] = False
        return resolved
    resolved["artifact_seed"] = int(seeds[slot])
    if slot < len(shas):
        resolved["artifact_semantic_sha256"] = str(shas[slot])
    resolved["environment_world_slot"] = slot
    resolved["lane_world_resolved"] = True
    # The lane occupies one world, so the batch-wide list is noise on a
    # per-lane record and invites exactly the confusion this function exists
    # to remove.
    resolved.pop("environment_world_id", None)
    return resolved


def _replay(
    result,
    roles,
    params,
    stage,
    *,
    lane: int = 0,
    world_identity: Mapping[str, object],
) -> dict[str, object]:
    rollout = result.rollout
    learner_slot = int(np.asarray(roles.learner_policy_slot)[lane])
    target_slot = int(np.asarray(roles.target_policy_slot)[lane])
    learner_actor = int(np.asarray(roles.learner_actor_index)[lane])
    target_actor = int(np.asarray(roles.target_actor_index)[lane])
    observation = np.asarray(rollout.observation_telemetry[:, lane, learner_slot])
    position = np.asarray(rollout.entity_position[:, lane])
    yaw = np.asarray(rollout.entity_yaw[:, lane])
    health = np.asarray(rollout.entity_health[:, lane])
    action = np.asarray(rollout.action[:, lane, learner_slot])
    target_action = np.asarray(rollout.action[:, lane, target_slot])
    reward = np.asarray(rollout.reward[:, lane, learner_slot])
    target_reward = np.asarray(rollout.reward[:, lane, target_slot])
    done = np.asarray(rollout.done[:, lane, learner_slot])
    grounded = np.asarray(
        rollout.entity_grounded[:, lane, learner_actor], dtype=np.bool_
    )
    stamina = np.asarray(rollout.entity_stamina[:, lane, learner_actor])
    # The sprint budget, published beside the guard bar rather than instead of
    # it: a replay that shows only `stamina` cannot distinguish "spent its guard"
    # from "sprinted", and sprinting is the behaviour these replays exist to show.
    locomotion_stamina = np.asarray(
        rollout.entity_locomotion_stamina[:, lane, learner_actor]
    )
    stamina_broken = np.asarray(
        rollout.entity_stamina_broken[:, lane, learner_actor], dtype=np.bool_
    )
    visible = observation[:, _PURSUIT_TELEMETRY_INDEX["visible"]] > 0.5
    context_available = (
        observation[:, _PURSUIT_TELEMETRY_INDEX["target_context_available"]] > 0.5
    )
    observed_distance = (
        24.0 * observation[:, _PURSUIT_TELEMETRY_INDEX["planar_distance"]]
    )
    delta = position[:, target_actor] - position[:, learner_actor]
    true_distance = np.linalg.norm(delta[:, (0, 2)], axis=-1)
    eye, center = (np.asarray(value) for value in _vertical_aim_offsets(params))
    relative_height = delta[:, 1] + center[target_actor] - eye[learner_actor]
    desired_pitch = np.degrees(
        np.arctan2(relative_height, np.maximum(true_distance, 1.0e-6))
    )
    agent_pitch = 90.0 * observation[:, _PURSUIT_TELEMETRY_INDEX["learner_pitch"]]
    desired_yaw = np.degrees(np.arctan2(-delta[:, 0], -delta[:, 2]))
    yaw_error = np.abs(
        np.mod(desired_yaw - yaw[:, learner_actor] + 180.0, 360.0) - 180.0
    )
    pitch_error = np.abs(desired_pitch - agent_pitch)
    aim_error = np.asarray(_angular_separation(yaw_error, agent_pitch, desired_pitch))
    end = int(np.argmax(done)) + 1 if np.any(done) else len(done)
    learner_path = float(
        np.sum(
            np.linalg.norm(
                np.diff(position[:end, learner_actor, (0, 2)], axis=0), axis=-1
            )
        )
    )
    target_path = float(
        np.sum(
            np.linalg.norm(
                np.diff(position[:end, target_actor, (0, 2)], axis=0), axis=-1
            )
        )
    )
    planar_steps = np.linalg.norm(
        np.diff(position[:end, learner_actor, (0, 2)], axis=0), axis=-1
    )
    stationary = planar_steps <= 0.01
    yaw_delta = np.zeros((end,), dtype=np.float32)
    if end > 1:
        yaw_delta[1:] = np.abs(
            np.mod(np.diff(yaw[:end, learner_actor]) + 180.0, 360.0) - 180.0
        )
    stationary_runs = np.zeros_like(stationary, dtype=np.int32)
    for index, value in enumerate(stationary):
        stationary_runs[index] = (
            (stationary_runs[index - 1] if index else 0) + 1 if value else 0
        )
    jump = ARSENAL_POLICY_ACTION_HEAD_NAMES.index("jump_off_on")
    locomotion = ARSENAL_POLICY_ACTION_HEAD_NAMES.index("locomotion_gait_compass")
    jump_requested = action[:end, jump] != 0
    dodge_requested = action[:end, locomotion] >= ARSENAL_POLICY_LOCOMOTION_DODGE_START
    jump_takeoff = grounded[: end - 1] & ~grounded[1:end] & jump_requested[1:end]
    values = {
        "agent_x": position[:end, learner_actor, 0],
        "agent_y": position[:end, learner_actor, 1],
        "agent_z": position[:end, learner_actor, 2],
        "target_x": position[:end, target_actor, 0],
        "target_y": position[:end, target_actor, 1],
        "target_z": position[:end, target_actor, 2],
        "agent_yaw": yaw[:end, learner_actor],
        "agent_pitch": agent_pitch[:end],
        "desired_pitch": desired_pitch[:end],
        "yaw_error": yaw_error[:end],
        "pitch_error": pitch_error[:end],
        "aim_error": aim_error[:end],
        "yaw_delta_degrees": yaw_delta,
        "target_yaw": yaw[:end, target_actor],
        "agent_health": health[:end, learner_actor],
        "target_health": health[:end, target_actor],
        "true_distance": true_distance[:end],
        "observed_distance": observed_distance[:end],
        "reward": reward[:end],
        "target_reward": target_reward[:end],
    }
    trajectory = {
        name: np.asarray(series).round(5).tolist() for name, series in values.items()
    }
    trajectory.update(
        {
            "visible": visible[:end].astype(int).tolist(),
            "target_context_available": context_available[:end].astype(int).tolist(),
            "done": done[:end].astype(int).tolist(),
            "action": action[:end].astype(int).tolist(),
            "target_action": target_action[:end].astype(int).tolist(),
            "grounded": grounded[:end].astype(int).tolist(),
            "requested": [0] * end,
            "accepted": [0] * end,
            "attack_executing": [0] * end,
            "damage_dealt": [0.0] * end,
            "target_phase": target_action[
                :end,
                ARSENAL_POLICY_ACTION_HEAD_NAMES.index("base_action"),
            ]
            .astype(int)
            .tolist(),
            "active_slot": [-1] * end,
            "ability_start": [0] * end,
            "projectiles": [0] * end,
            "stamina": stamina[:end].round(5).tolist(),
            "locomotion_stamina": locomotion_stamina[:end].round(5).tolist(),
            "stamina_available": [1] * end,
            "stamina_exhausted": stamina_broken[:end].astype(int).tolist(),
            "guard_active": [0] * end,
        }
    )
    return {
        "steps": end,
        "trajectory": trajectory,
        "world_identity": dict(world_identity),
        "region": dict(world_identity),
        "terrain": {
            "schema": "console-region-terrain-reference-v1",
            "seed": int(world_identity["artifact_seed"]),
            "artifact_semantic_sha256": world_identity["artifact_semantic_sha256"],
            "reference_only": True,
            "nodes": 0,
            "x": [],
            "y": [],
            "z": [],
            "material": [],
            "colors": [],
        },
        "motion": {
            "learner_path_length": learner_path,
            "target_path_length": target_path,
            "both_agents_moved": learner_path >= 10.0 and target_path >= 10.0,
            "stationary_ticks": int(np.sum(stationary)),
            "maximum_stationary_ticks": int(np.max(stationary_runs, initial=0)),
            "jump_requests": int(np.sum(jump_requested)),
            "jump_takeoffs": int(np.sum(jump_takeoff)),
            "dodge_requests": int(np.sum(dodge_requested)),
            "maximum_yaw_delta_degrees": float(np.max(yaw_delta, initial=0.0)),
            "turn_cap_degrees": stage.maximum_turn_degrees,
            "turn_cap_violations": int(
                np.sum(yaw_delta > stage.maximum_turn_degrees + 1.0e-3)
            ),
        },
    }


def run_pursuit_training(
    run: PursuitRunConfig,
    *,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Run one long CUDA study; evaluation uses fixed common random numbers."""

    run.output.mkdir(parents=True, exist_ok=True)
    monitor = TrainingMonitor(run.output)
    started = perf_counter()
    launched_at = time()

    # Values the printed log needs but the individual payloads do not carry,
    # filled in as the run resolves them.
    progress_context: dict[str, Any] = {}

    def emit(kind: str, payload: Mapping[str, Any], *, phase: str) -> None:
        monitor.emit(kind, payload, phase=phase)
        # The event log keeps the whole payload for tooling; stdout gets one
        # short line, and nothing at all for events a reader cannot act on.
        line = _progress_line(perf_counter() - started, kind, payload, progress_context)
        if line:
            print(line, flush=True)
        if progress is not None:
            progress({"kind": kind, **payload})

    emit(
        "preparing",
        {
            "batch": run.batch,
            "updates": run.updates,
            "rollout_steps": run.rollout_steps,
            "evaluation_steps": run.evaluation_steps,
            "backend": jax.default_backend(),
            "device": jax.devices()[0].device_kind,
            "persistent_compilation_cache": (
                None
                if PERSISTENT_COMPILATION_CACHE is None
                else str(PERSISTENT_COMPILATION_CACHE)
            ),
        },
        phase="prepare",
    )
    build_started = perf_counter()
    source_params, source_config, source_metadata = load_policy_checkpoint(
        run.source_policy
    )
    # The frozen opponent, when it is not simply a copy of the learner's own
    # starting weights. This is what makes a recursive pair possible: generation
    # N trains against generation N-1's promoted checkpoint.
    opponent_params = None
    opponent_metadata = None
    if run.opponent_checkpoint is not None:
        opponent_params, opponent_config, opponent_metadata = load_policy_checkpoint(
            run.opponent_checkpoint
        )
        # Refused rather than broadcast: two policies of different widths cannot
        # share one bank, and silently padding would train against a network
        # nobody built.
        for axis in ("observation_size", "action_size"):
            expected = getattr(source_config, axis, None)
            actual = getattr(opponent_config, axis, None)
            if expected != actual:
                raise ValueError(
                    f"opponent checkpoint {axis} {actual} does not match the "
                    f"learner's {expected}"
                )
    selection_key = _resolved_region_selection(run)
    emit(
        "scene_build",
        {
            "world_count": run.world_count,
            "selection_key": selection_key,
            "node_seed": run.node_seed,
            "environment_diversity": run.environment_diversity,
            "loadout": run.loadout,
            "opponent": run.opponent,
        },
        phase="prepare",
    )
    design_arena = None
    if run.world_design is not None:
        design_arena = _materialize_design(
            run.world_design,
            run.loadout,
            run.opponent,
            run.batch,
            run.target_separation_range,
            run.arena_radius,
            run.learner_stamina_fraction,
            run.learner_stamina_regen_scale,
        )
        fixture = _DesignFixture(
            metadata=_design_world_metadata(design_arena, run),
            reset_provider=design_arena.reset_provider,
        )
        params, runtime = design_arena.params, design_arena.runtime
        region_fixture_cache_hit = False
    cache_hits = _materialize_region.cache_info().hits
    loaded = None if design_arena is not None else _materialize_region(
        run.loadout,
        run.opponent,
        run.batch,
        selection_key,
        run.node_seed,
        run.target_separation_range,
        run.world_count,
        run.environment_diversity,
        (
            _flattest_world_seeds(run.world_count, selection_key)
            if run.flattest_worlds
            else _mixed_terrain_world_seeds(run.world_count, selection_key)
            if run.mixed_terrain_worlds
            else None
        ),
    )
    if loaded is not None:
        region_fixture_cache_hit = _materialize_region.cache_info().hits > cache_hits
        fixture, params, runtime = loaded.fixture, loaded.params, loaded.runtime
    if run.sensor_range is not None:
        # `sensor_range` is authored at 16.0 in
        # `rulesets/hytale_0_5_7/kweebec_razorleaf_vs_trork_brawler_v9.json`,
        # so an override is a declared deviation and is stamped into the replay
        # identity below rather than edited into the shipped ruleset.
        #
        # Both sides must move: `step_arsenal_batch` takes
        # `jnp.minimum(config.targeting_rules.sensor_range,
        # params.sensor_range)`, so raising only one leaves the smaller in
        # force and the change silently does nothing.
        override = jnp.float32(run.sensor_range)
        params = params._replace(sensor_range=override)
        rules = runtime.targeting_rules
        runtime = runtime._replace(
            targeting_rules=rules._replace(
                sensor_range=jnp.full_like(rules.sensor_range, override)
            )
        )
    fixture.metadata["privileged_target_context"] = True
    if (
        run.world_artifact_seed is not None
        and int(fixture.metadata["artifact_seed"]) != run.world_artifact_seed
    ):
        raise RuntimeError("saved world seed resolved to a different Region artifact")
    replay_world_identity = {
        "schema": "arena-pursuit-replay-world-v1",
        "backend": "saved_region_artifact",
        "training_seed": run.seed,
        "artifact_seed": int(fixture.metadata["artifact_seed"]),
        "artifact_semantic_sha256": fixture.metadata["artifact_semantic_sha256"],
        "region_library_semantic_sha256": fixture.metadata[
            "region_library_semantic_sha256"
        ],
        "selection_key": selection_key,
        # With a composed set the key no longer decides membership, so a reader
        # comparing two runs by selection_key alone would call different world
        # sets the same. `resident_artifact_seed` above is the actual answer.
        "mixed_terrain_worlds": bool(run.mixed_terrain_worlds),
        "flattest_worlds": bool(run.flattest_worlds),
        "requested_artifact_seed": run.world_artifact_seed,
        "materialization_cache_hit": region_fixture_cache_hit,
        "node_seed": run.node_seed,
        "source_node": int(fixture.metadata["source_node"]),
        "destination_node": int(fixture.metadata["destination_node"]),
        # ``artifact_seed`` above names only the first resident world, so a
        # multi-world run needs these to be readable at all.
        #
        # These three are what make a *per-lane* world resolvable:
        # ``environment_world_id[lane]`` is that lane's slot in the working set,
        # and the resident tuples turn the slot back into a seed and a hash.
        # Without them every lane inherits slot 0 and any consumer that resolves
        # terrain from the identity draws the wrong world -- measured on run
        # 409da2cb, where 43 of 48 archived lanes rendered world 0's geometry
        # under a trajectory from somewhere else, and the one lane that rendered
        # nothing was the only honest one.
        "environment_world_id": [
            int(value) for value in fixture.metadata.get("environment_world_id", ())
        ],
        "resident_artifact_seed": [
            int(value) for value in fixture.metadata.get("resident_artifact_seed", ())
        ],
        "resident_artifact_semantic_sha256": [
            str(value)
            for value in fixture.metadata.get("resident_artifact_semantic_sha256", ())
        ],
        "world_count": int(fixture.metadata.get("working_set_capacity", 1)),
        "distinct_environment_worlds": int(
            fixture.metadata.get("distinct_environment_worlds", 1)
        ),
        "environment_diversity": bool(
            fixture.metadata.get("environment_diversity", False)
        ),
        "distinct_environment_spawns": int(
            fixture.metadata.get("distinct_environment_spawns", 1)
        ),
        "distinct_environment_targets": int(
            fixture.metadata.get("distinct_environment_targets", 1)
        ),
        "sensor_range": float(np.asarray(jax.device_get(params.sensor_range))),
        "sensor_range_overridden": run.sensor_range is not None,
    }
    assignment = _assignment(run.batch)
    roles = skill_duel_roles(assignment)
    # `allow_jump` first so an explicit `stage_options` entry still wins, rather
    # than colliding as a duplicate keyword and failing the run at construction.
    stage = PursuitStageConfig(
        maximum_ticks=run.evaluation_steps,
        **{"allow_jump": run.evader_allow_jump, **run.stage_options},
    )
    # Built once here rather than inside the traced transform, so the two halves
    # are checked to agree on the catch distance at setup.
    evader_stage = (
        EvaderDuelConfig(
            horizon=HorizonConfig(maximum_ticks=run.evaluation_steps),
            catch_distance=stage.catch_distance,
            stamina_fraction=run.learner_stamina_fraction,
            contact_reward=run.evader_contact_reward,
            travel_reward=run.evader_travel_reward,
            alive_reward=run.evader_alive_reward,
            airborne_idle_cost=run.evader_airborne_idle_cost,
            forward_travel_reward=run.evader_forward_travel_reward,
            separation_reward=run.evader_separation_reward,
            strafe_reward=run.evader_strafe_reward,
            sprint_reward=run.evader_sprint_reward,
            turn_reward=run.evader_turn_reward,
            body_away_reward=run.evader_body_away_reward,
            band_leads=run.evader_band_leads,
            survival_reward=run.evader_terminal_reward,
            caught_penalty=(
                run.evader_terminal_reward
                if run.evader_caught_penalty is None
                else run.evader_caught_penalty
            ),
        )
        if run.objective == "evade"
        else None
    )
    if run.target_separation_range[0] <= stage.catch_distance:
        raise ValueError("target reset minimum must exceed pursuit catch distance")
    initial_separation = float(fixture.metadata["target_initial_horizontal_separation"])
    sensor_range = float(np.asarray(jax.device_get(params.sensor_range)))
    if initial_separation > sensor_range:
        raise ValueError(
            "target reset begins outside the actor sensor range: "
            f"{initial_separation:.3f} > {sensor_range:.3f} blocks"
        )
    # A requested turn larger than the engine can physically deliver in one
    # tick is not a stricter lesson, it is an unmeasurable one: the selection
    # gate scores *physical* yaw change, so the two must be stated in the same
    # units. Derived rather than hardcoded so a change to the authored turn
    # speed or loaded tick length fails here instead of silently reappearing as
    # phantom turn-cap violations.
    engine_turn_ceiling = float(
        np.asarray(jax.device_get(params.agent_turn_speed_degrees * params.loaded_dt))
    )
    if stage.maximum_turn_degrees > engine_turn_ceiling + 1.0e-3:
        raise ValueError(
            "stage turn cap exceeds the engine per-tick rotation ceiling: "
            f"{stage.maximum_turn_degrees:.3f} > {engine_turn_ceiling:.3f} degrees"
        )
    # A cap *below* the ceiling is the phantom-violation case the comment above
    # warns about, and it is silent: nothing clamps the look action to the
    # declared cap, so the gate measures the engine's physical yaw, flags
    # `physical_turn_cap_violated` on every milestone including update 0, and
    # promotes the untrained source. Run 20260817T062230Z-d6552974 reached a
    # 0.994 catch rate and shipped update 0 this way.
    if stage.maximum_turn_degrees < engine_turn_ceiling - 1.0e-3:
        raise ValueError(
            "stage turn cap is below the engine per-tick rotation ceiling, so "
            "the selection gate can never pass: "
            f"{stage.maximum_turn_degrees:.3f} < {engine_turn_ceiling:.3f} "
            "degrees. Nothing clamps the look action to the declared cap; "
            "state the cap in the same units the gate measures."
        )
    source_policy_sha256 = _sha256(run.source_policy)
    if run.target_controller in _SCRIPTED_TARGET_LANE_MODES:
        lane_modes = _SCRIPTED_TARGET_LANE_MODES[run.target_controller]
        # Both gait-driven modes read the same cadence and arena fields.
        adaptive_options = (
            {
                "adaptive_gait_mix": tuple(run.evader_gait_mix),
                "adaptive_gait_period_ticks": run.evader_gait_period_ticks,
                "adaptive_arena_radius": run.arena_radius,
            }
            if {TargetMode.FLEE_ADAPTIVE, TargetMode.FLEE_STANDOFF} & set(lane_modes)
            else {}
        )
        if TargetMode.FLEE_STANDOFF in lane_modes:
            adaptive_options |= {
                "standoff_inner_distance": run.evader_standoff_band[0],
                "standoff_outer_distance": run.evader_standoff_band[1],
                "standoff_strafe_period_ticks": run.evader_standoff_strafe_ticks,
            }
        target = TargetMotionProgram(
            lane_modes=lane_modes,
            **adaptive_options,
            **run.target_options,
        )
    else:
        target = TargetMotionProgram(
            lane_modes=(TargetMode.FROZEN_POLICY,),
            frozen_policy_sha256=source_policy_sha256,
            frozen_factor_scope=pursuit_learner_action_scope(
                allow_jump=run.evader_allow_jump
            ),
            **run.target_options,
        )
    ppo = replace(
        source_config,
        num_envs=run.batch,
        rollout_steps=run.rollout_steps,
        update_epochs=run.update_epochs,
        num_minibatches=run.num_minibatches,
        learning_rate=run.learning_rate,
        entropy_coefficient=run.entropy_coefficient,
        value_coefficient=run.value_coefficient,
    )

    def reset_provider(keys):
        if design_arena is not None:
            # The design arena places both actors itself, on a plane, so there
            # is no traversal graph to consult and no geometry to reset into.
            state = design_arena.reset_provider(keys)
            return state if not stage.spawn_facing_target else _facing_reset(
                state, params
            )
        reset = fixture.reset_provider(keys)
        inventory = (
            None
            if fixture.inventory_reset_provider is None
            else fixture.inventory_reset_provider(keys, runtime)
        )
        state, _ = reset_arsenal_batch(
            keys,
            params,
            runtime,
            fixture.geometry_provider,
            agent_position=reset.agent_position,
            target_position=reset.target_position,
            initial_inventory=inventory,
        )
        if not stage.spawn_facing_target:
            return state
        # `reset_arsenal_batch` only honours `entity_yaw` on its explicit
        # entity_position path; the Region path used here routes through
        # `reset_batch_region_at`, which takes no orientation and would drop it
        # silently. Orient after the reset instead.
        return _facing_reset(state, params)

    if design_arena is not None:
        # A design arena is a plane: no captured geometry, so no world runtime,
        # no block-action surface and nothing to occlude a sightline. It binds
        # the flat collector, which `bind_pursuit_collector` supports by
        # construction -- "the same pursuit lesson to flat, Region, or future
        # collectors".
        collector_factory = make_multi_actor_rollout_collector
        common = dict(
            reset_provider=reset_provider,
            capability_provider=design_arena.capability_provider,
        )
        line_of_sight_provider = None
    else:
        collector_factory = make_multi_actor_region_rollout_collector
        required = {
            name: getattr(fixture, name)
            for name in (
                "world_runtime_provider",
                "actor_world_runtime_provider",
                "action_surface_provider",
                "action_surface_runtime_initializer",
            )
        }
        common = dict(
            reset_provider=reset_provider,
            target_navigation_provider=fixture.target_navigation_provider,
            explosion_candidate_provider=fixture.explosion_candidate_provider,
            action_surface_executor=fixture.action_surface_executor,
            **required,
        )

        def line_of_sight_provider(combat):
            result = geometry_perception_line_of_sight_result(
                fixture.geometry_provider,
                combat.position[:, AGENT_ENTITY] + params.agent_eye_offset,
                combat.position[:, TARGET_ENTITY] + params.target_eye_offset,
                role_opaque_mask=None,
            )
            return (
                result.visible
                & ~result.geometry_exhausted
                & ~result.capacity_exceeded
                & ~result.invalid
            )

    def collector(
        steps: int,
        *,
        carry: bool = True,
        record_policy_inputs: bool = True,
        record_policy_slots: tuple[int, ...] | None = None,
        device=None,
    ):
        # `device` binds this collector's world to somewhere other than the
        # default. Evaluation uses it to run off the training accelerator: the
        # two otherwise share one allocator, and an evaluation overlapping a
        # training step is what exhausted it at update 33 of run 172042Z while
        # `jit_train_step` asked for another 2.03 GiB.
        bound_params = params if device is None else jax.device_put(params, device)
        bound_runtime = runtime if device is None else jax.device_put(runtime, device)
        return bind_pursuit_collector(
            collector_factory,
            bound_params,
            bound_runtime,
            assignment,
            target=target,
            stage=stage,
            evader=evader_stage,
            line_of_sight_provider=line_of_sight_provider,
            rollout_steps=steps,
            carry_recurrent_state=carry,
            record_policy_inputs=record_policy_inputs,
            scripted_actor_slots=(
                (1,) if run.target_controller in _SCRIPTED_TARGET_LANE_MODES else ()
            ),
            record_policy_slots=record_policy_slots,
            observation_telemetry_transform=_pursuit_observation_telemetry,
            **common,
        )

    keys = jax.random.split(jax.random.key(run.seed), 4 + run.updates)
    reset_keys = jax.random.split(keys[0], run.batch)
    arsenal = reset_provider(reset_keys)
    if design_arena is None:
        initial_arena = initialize_multi_actor_arena_state(
            arsenal,
            params,
            ArsenalActionSurfaceRuntime(
                block_interactions=empty_block_interaction_state(
                    run.batch, entity_count=2
                ),
                world=fixture.action_surface_runtime_initializer(
                    reset_keys, arsenal, params, runtime
                ),
            ),
        )
    else:
        # A design world has no block-action surface to initialise: there are
        # no voxels to break or place. The arena state is built without one,
        # the same shape a flat scene uses.
        initial_arena = initialize_multi_actor_arena_state(arsenal, params)
    # Slot order follows the assignment's own `policy_id`, which is (0, 1) with
    # slot 0 trainable -- so slot 1 is whatever the opponent should be, and no
    # actor index is named here. Without `opponent_checkpoint` both slots hold
    # the source, which is the behaviour every run before this one had.
    if opponent_params is None:
        policy_bank = jax.tree_util.tree_map(
            lambda value: jnp.stack((value, value)), source_params
        )
    else:
        policy_bank = jax.tree_util.tree_map(
            lambda learner, opponent: jnp.stack((learner, opponent)),
            source_params,
            opponent_params,
        )
    scene_build_seconds = perf_counter() - build_started
    emit(
        "scene_ready",
        {
            "scene_build_seconds": scene_build_seconds,
            "materialization_cache_hit": region_fixture_cache_hit,
            "distinct_environment_worlds": replay_world_identity[
                "distinct_environment_worlds"
            ],
            "distinct_environment_spawns": replay_world_identity[
                "distinct_environment_spawns"
            ],
        },
        phase="compile",
    )
    compile_started = perf_counter()
    trainer = make_multi_actor_shared_policy_trainer(
        collector(run.rollout_steps, record_policy_slots=(0,)),
        assignment,
        ppo,
        compile=True,
        donate_state=True,
    )
    training_execution = {
        "schema": "arena-multi-actor-training-execution-v2",
        "scripted_actor_slots": (
            [1] if run.target_controller in _SCRIPTED_TARGET_LANE_MODES else []
        ),
        "record_policy_slots": [0],
        "full_actor_slots": 2,
        "neural_policy_forwards_per_environment": (
            1 if run.target_controller in _SCRIPTED_TARGET_LANE_MODES else 2
        ),
        "state_buffer_donation": trainer.donates_state,
        "donation_initial_alias_policy": "copy_each_state_leaf_once_before_first_update",
        "saved_world": {
            "artifact_seed": int(fixture.metadata["artifact_seed"]),
            "semantic_sha256": fixture.metadata["artifact_semantic_sha256"],
            "materialized_before_jit": True,
            "resident_fixture_cache_hit": region_fixture_cache_hit,
            "generation_inside_scan": False,
        },
    }
    state = trainer.initialize(policy_bank, initial_arena)
    # Evaluation runs off the training accelerator. It is submitted to a
    # background thread and so overlaps a training step; on one device the two
    # share an allocator, and run 172042Z died at update 33 when `jit_train_step`
    # could not get 2.03 GiB immediately after an evaluation. A separate device
    # makes the overlap free rather than contended.
    evaluation_device = _evaluation_device()
    evaluate_memory = jax.jit(
        collector(
            run.evaluation_steps,
            carry=True,
            record_policy_inputs=False,
            device=evaluation_device,
        )
    )
    evaluate_reset = jax.jit(
        collector(
            run.evaluation_steps,
            carry=False,
            record_policy_inputs=False,
            device=evaluation_device,
        )
    )
    program_build_seconds = perf_counter() - compile_started
    emit(
        "program_ready",
        {
            "scene_build_seconds": scene_build_seconds,
            "saved_world_materialization_cache_hit": region_fixture_cache_hit,
            "program_build_seconds": program_build_seconds,
            "stage_contract_sha256": stage.contract_sha256,
            "target_motion_sha256": target.contract_sha256,
        },
        phase="compile",
    )
    evaluation_key = keys[1]
    curve = []
    updates = []
    replay_archive = []
    milestone_params = {}
    last_evaluation = None
    selected_params = None
    selected_replay = None
    selected_update = None
    source_selection = None
    incumbent_selection = None
    cancelled = False
    stop_reason = None
    maximum_approximate_kl = _resolved_maximum_approximate_kl(run)
    progress_context["maximum_approximate_kl"] = maximum_approximate_kl
    consecutive_kl_violations = 0
    evaluation_device_observed = None

    def evaluate(update: int, policy_bank):
        nonlocal selected_params
        nonlocal selected_replay
        nonlocal selected_update
        nonlocal source_selection
        nonlocal incumbent_selection
        evaluation_started = perf_counter()
        # Everything evaluation touches is built and run under the evaluation
        # device as the DEFAULT, not merely copied to it.
        #
        # Committing the arguments with `device_put` was not enough: the three
        # `jnp.zeros` below were still constructed on the trainer's device
        # before being copied, and `jax.jit` carries no `device=`, so placement
        # was inferred rather than stated. An evaluation rollout is
        # `batch * evaluation_steps` deep against the training rollout's
        # `batch * rollout_steps` -- four times the training buffer here -- so
        # anything of it that lands on the accelerator dominates the allocator.
        # That is what kept exhausting it at both 512 and 256 environments,
        # which is also why halving the batch did not help.
        with jax.default_device(evaluation_device):
            zero_carry = jnp.zeros(
                (run.batch, 2, ppo.recurrent_size), dtype=jnp.float32
            )
            args = jax.device_put(
                (
                    initial_arena,
                    policy_bank,
                    zero_carry,
                    assignment.active,
                    jnp.zeros((run.batch, 2), dtype=jnp.float32),
                    jnp.zeros((run.batch, 2), dtype=jnp.int32),
                    evaluation_key,
                ),
                evaluation_device,
            )
            memory = evaluate_memory(*args)
            reset = (
                evaluate_reset(*args)
                if update in run.memory_reset_updates
                else None
            )
        # Evaluation must finish before training resumes.
        #
        # JAX dispatches asynchronously, and the only `block_until_ready` in this
        # loop is on the *training* loss, so an evaluation used to stay in flight
        # while training kept issuing work -- its device buffers alive the whole
        # time. Measured on 20260818T175923Z-76031c53: 200 `training_update`
        # events against 5 `evaluation_queued` but only 3 `evaluation`, i.e. two
        # evaluations still resident when the process died.
        #
        # That is what made the OOM land at update 200 regardless of `num_envs`:
        # an evaluation rollout is `evaluation_steps` (512) deep against the
        # training rollout's 128, so two outstanding evaluations dwarf the
        # training buffer, and halving `num_envs` halved both without changing
        # the ratio. Blocking here bounds live evaluation memory to exactly one.
        jax.block_until_ready(memory if reset is None else (memory, reset))
        # Where evaluation ACTUALLY ran, read off a result buffer rather than
        # from the device we asked for. `jax.jit` here carries no `device=`; the
        # placement comes from the arguments being committed above, so the
        # request and the outcome are separate facts and only the outcome tells
        # us whether evaluation is competing with training for the allocator.
        nonlocal evaluation_device_observed
        if evaluation_device_observed is None:
            leaves = [
                leaf for leaf in jax.tree_util.tree_leaves(memory)
                if hasattr(leaf, "device")
            ]
            if leaves:
                evaluation_device_observed = str(leaves[0].device)
        row = {
            "update": update,
            "environment_steps": update * run.batch * run.rollout_steps,
            "recurrent": _metrics(memory, roles, params, stage),
            "memory_reset_each_tick": (
                None if reset is None else _metrics(reset, roles, params, stage)
            ),
            "wall_seconds": perf_counter() - evaluation_started,
        }
        snapshots = []
        replay_dir = run.output / "replays"
        replay_dir.mkdir(exist_ok=True)
        for lane in range(run.archive_lanes):
            replay = _replay(
                memory,
                roles,
                params,
                stage,
                lane=lane,
                world_identity=_lane_world_identity(replay_world_identity, lane),
            )
            relative = Path("replays") / f"update-{update:04d}-lane-{lane:03d}.json"
            (run.output / relative).write_text(
                json.dumps(replay, separators=(",", ":")) + "\n", encoding="utf-8"
            )
            entry = {
                "update": update,
                "lane": lane,
                "environment_steps": row["environment_steps"],
                "steps": replay["steps"],
                "path": relative.as_posix(),
            }
            replay_archive.append(entry)
            snapshots.append(replay)
        current_params = jax.tree_util.tree_map(
            lambda value: np.asarray(jax.device_get(value[0])).copy(),
            policy_bank,
        )
        if source_selection is None:
            raw_source = pursuit_selection_assessment(
                row["recurrent"],
                objective=run.objective,
                minimum_line_of_sight_fraction=run.minimum_baseline_visible_fraction,
                maximum_turn_degrees=stage.maximum_turn_degrees,
                perception=stage.perception,
            )
            source_selection = pursuit_selection_assessment(
                row["recurrent"],
                source_success_rate=raw_source.success_rate,
                objective=run.objective,
                minimum_line_of_sight_fraction=run.minimum_baseline_visible_fraction,
                maximum_turn_degrees=stage.maximum_turn_degrees,
                perception=stage.perception,
            )
            assessment = source_selection
            accepted = True
        else:
            assessment = pursuit_selection_assessment(
                row["recurrent"],
                source_success_rate=source_selection.success_rate,
                objective=run.objective,
                minimum_line_of_sight_fraction=run.minimum_baseline_visible_fraction,
                maximum_turn_degrees=stage.maximum_turn_degrees,
                perception=stage.perception,
            )
            assert incumbent_selection is not None
            accepted = pursuit_candidate_is_better(assessment, incumbent_selection)
        if accepted:
            selected_params = current_params
            selected_replay = max(
                snapshots,
                key=lambda replay: (
                    bool(replay["motion"]["both_agents_moved"]),
                    min(
                        float(replay["motion"]["learner_path_length"]),
                        float(replay["motion"]["target_path_length"]),
                    ),
                    int(replay["steps"]),
                ),
            )
            selected_update = update
            incumbent_selection = assessment
        row["selection"] = {
            "schema": PURSUIT_SELECTION_SCHEMA,
            "objective_score": assessment.objective_score,
            "success_rate": assessment.success_rate,
            "valid_fraction": assessment.valid_fraction,
            "eligible": assessment.eligible,
            "flags": list(assessment.flags),
            "became_incumbent": accepted,
        }
        curve.append(row)
        emit(
            "evaluation",
            {
                "update": update,
                "environment_steps": row["environment_steps"],
                "wall_seconds": row["wall_seconds"],
                # The replay is already on disk before this event is emitted.
                # Publish its bounded relative path so observers can render the
                # milestone immediately instead of waiting for training to end.
                "replay": replay_archive[-run.archive_lanes],
                "selection": row["selection"],
                "metrics": _numeric_scalars(
                    {
                        "learner": row["recurrent"],
                        **(
                            {}
                            if row["memory_reset_each_tick"] is None
                            else {"memory_reset": row["memory_reset_each_tick"]}
                        ),
                    }
                ),
            },
            phase="evaluate",
        )
        return memory

    evaluation_queue = AsyncEvaluationQueue(evaluate, thread_name="pursuit-evaluation")

    def submit_evaluation(update: int) -> None:
        ticket = evaluation_queue.submit(update, state.policy_bank)
        milestone_params[update] = jax.tree_util.tree_map(
            lambda value: value[0].copy(), ticket.snapshot
        )
        emit(
            "evaluation_queued",
            {
                "update": update,
                "environment_steps": update * run.batch * run.rollout_steps,
            },
            phase="train",
        )

    emit(
        "training_start",
        {
            "updates": run.updates,
            "batch": run.batch,
            "rollout_steps": run.rollout_steps,
            "learning_rate": run.learning_rate,
            "entropy_coefficient": run.entropy_coefficient,
            "maximum_approximate_kl": maximum_approximate_kl,
            "action_head_count": len(ARSENAL_POLICY_ACTION_HEAD_SIZES),
        },
        phase="train",
    )
    try:
        if 0 in run.evaluation_updates:
            submit_evaluation(0)
        for update in range(1, run.updates + 1):
            if stop_reason is not None:
                break
            if (run.output / "CANCEL").exists():
                cancelled = True
                break
            step_started = perf_counter()
            state, metrics = trainer.train_step(state, keys[3 + update])
            jax.block_until_ready(metrics.loss.total_loss)
            row = {
                "update": update,
                "wall_seconds": perf_counter() - step_started,
                **_host(metrics),
            }
            row["environment_steps"] = update * run.batch * run.rollout_steps
            row["steps_per_second"] = (
                run.batch * run.rollout_steps / max(row["wall_seconds"], 1e-9)
            )
            updates.append(row)
            emit(
                "training_update",
                {
                    "update": update,
                    "updates": run.updates,
                    "environment_steps": row["environment_steps"],
                    "wall_seconds": row["wall_seconds"],
                    "steps_per_second": row["steps_per_second"],
                    "metrics": _numeric_scalars(row),
                },
                phase="train",
            )
            approximate_kl = float(row["loss"]["approximate_kl"])
            median_log_ratio = float(row["loss"]["log_ratio_median"])
            # A non-finite KL is unrecoverable -- the policy has already been
            # poisoned -- so it stops immediately. Merely exceeding the ceiling
            # is not: it has to persist, and the median has to agree that the
            # distribution moved rather than one tail sample.
            if not math.isfinite(approximate_kl) or not math.isfinite(
                median_log_ratio
            ):
                consecutive_kl_violations = _KL_VIOLATION_PATIENCE
            elif pursuit_update_diverged(
                approximate_kl, median_log_ratio, maximum_approximate_kl
            ):
                consecutive_kl_violations += 1
            else:
                consecutive_kl_violations = 0
            if consecutive_kl_violations >= _KL_VIOLATION_PATIENCE:
                stop_reason = "ppo_approximate_kl_exceeded"
                emit(
                    "stop_loss",
                    {
                        "reason": stop_reason,
                        "update": update,
                        "approximate_kl": approximate_kl,
                        "maximum_approximate_kl": maximum_approximate_kl,
                        "log_ratio_median": median_log_ratio,
                        "maximum_log_ratio_median": _MEDIAN_LOG_RATIO_CEILING,
                        "consecutive_violations": consecutive_kl_violations,
                        "action_head_count": len(ARSENAL_POLICY_ACTION_HEAD_SIZES),
                    },
                    phase="stopped",
                )
                break
            if update in run.evaluation_updates:
                submit_evaluation(update)

        completed_updates = len(updates)
        submitted_updates = set(milestone_params)
        if completed_updates and completed_updates not in submitted_updates:
            submit_evaluation(completed_updates)
        evaluation_results = evaluation_queue.results()
        last_evaluation = evaluation_results[-1][1]
    finally:
        evaluation_queue.close()

    assert last_evaluation is not None
    assert selected_params is not None
    assert selected_replay is not None
    assert selected_update is not None
    assert source_selection is not None
    assert incumbent_selection is not None
    base_metadata = {
        "schema": PURSUIT_RUN_SCHEMA,
        "live_policy_compatible": False,
        "promotion_status": "not_assessed",
        "transfer_contract": source_metadata.get("transfer_contract"),
        "source_policy_sha256": source_policy_sha256,
        "pursuit_stage": stage.manifest(),
        "pursuit_stage_sha256": stage.contract_sha256,
        "target_motion": target.manifest(),
        "region_reset": fixture.metadata,
        "selection_schema": PURSUIT_SELECTION_SCHEMA,
        "selection_law": PURSUIT_SELECTION_LAW,
        "evaluation_execution": evaluation_queue.describe(),
        "evaluation_device_requested": str(evaluation_device),
        "evaluation_device_observed": evaluation_device_observed,
        "training_execution": training_execution,
        "completed_updates": completed_updates,
        "completed_environment_steps": (
            completed_updates * run.batch * run.rollout_steps
        ),
        "stop_reason": stop_reason,
    }
    milestone_checkpoints = []
    for update, params_at_update in sorted(milestone_params.items()):
        milestone_checkpoint = run.output / "policies" / f"update-{update:04d}.npz"
        save_policy_checkpoint(
            milestone_checkpoint,
            params_at_update,
            ppo,
            metadata={
                **base_metadata,
                "checkpoint_role": "pursuit_evaluation_milestone",
                "updates": update,
                "environment_steps": update * run.batch * run.rollout_steps,
                "resumable": False,
                "omitted_training_state": [
                    "optimizer",
                    "environment",
                    "recurrent_carry",
                    "rng",
                ],
            },
        )
        milestone_checkpoints.append(
            {
                "update": update,
                "environment_steps": update * run.batch * run.rollout_steps,
                "path": milestone_checkpoint.relative_to(run.output).as_posix(),
                "sha256": _sha256(milestone_checkpoint),
            }
        )
    latest_checkpoint = run.output / "latest_policy.npz"
    save_policy_checkpoint(
        latest_checkpoint,
        jax.tree_util.tree_map(lambda value: value[0], state.policy_bank),
        ppo,
        metadata={
            **base_metadata,
            "checkpoint_role": "latest_recurrent_pursuit_diagnostic",
            "updates": completed_updates,
            "environment_steps": completed_updates * run.batch * run.rollout_steps,
        },
    )
    checkpoint = run.output / "pursuit_policy.npz"
    save_policy_checkpoint(
        checkpoint,
        selected_params,
        ppo,
        metadata={
            **base_metadata,
            "checkpoint_role": "selected_recurrent_pursuit_curriculum_output",
            "selected_update": selected_update,
            "updates": selected_update,
            "selected_environment_steps": (
                selected_update * run.batch * run.rollout_steps
            ),
            "environment_steps": selected_update * run.batch * run.rollout_steps,
        },
    )
    report = {
        "schema": PURSUIT_RUN_SCHEMA,
        "status": (
            "cancelled_selected_best_not_promotion"
            if cancelled
            else (
                "stopped_by_guardrail_selected_best_not_promotion"
                if stop_reason is not None
                else "training_complete_selected_best_not_promotion"
            )
        ),
        "runtime": {
            "backend": jax.default_backend(),
            "device": jax.devices()[0].device_kind,
            "wall_seconds": perf_counter() - started,
            "launched_unix_seconds": launched_at,
            "scene_build_seconds": scene_build_seconds,
            "saved_world_materialization_cache_hit": region_fixture_cache_hit,
            "program_build_seconds": program_build_seconds,
            "persistent_compilation_cache": (
                None
                if PERSISTENT_COMPILATION_CACHE is None
                else str(PERSISTENT_COMPILATION_CACHE)
            ),
            "runtime_environment": _runtime_environment(),
        },
        "settings": {
            **{
                name: str(value) if isinstance(value, Path) else value
                for name, value in asdict(run).items()
            },
            "evaluation_updates": list(run.evaluation_updates),
            "memory_reset_updates": list(run.memory_reset_updates),
        },
        "contracts": {
            "stage": stage.manifest(),
            "stage_sha256": stage.contract_sha256,
            "target_motion": target.manifest(),
            "target_motion_sha256": target.contract_sha256,
            "region_reset": fixture.metadata,
            "source_transfer": source_metadata.get("transfer_contract"),
            "selection_schema": PURSUIT_SELECTION_SCHEMA,
            "selection_law": PURSUIT_SELECTION_LAW,
            "evaluation_execution": base_metadata["evaluation_execution"],
            # Copied across explicitly, like every other key here: `report` is
            # assembled independently of `base_metadata`, so a field added only
            # there reaches the checkpoint and never the run report.
            "evaluation_device_requested": base_metadata[
                "evaluation_device_requested"
            ],
            "evaluation_device_observed": base_metadata[
                "evaluation_device_observed"
            ],
            "opponent_checkpoint": (
                str(run.opponent_checkpoint)
                if run.opponent_checkpoint is not None
                else None
            ),
            "training_execution": training_execution,
        },
        "selection": {
            "selected_update": selected_update,
            "completed_updates": completed_updates,
            "source": asdict(source_selection),
            "selected": asdict(incumbent_selection),
            "metrics": next(
                row["recurrent"] for row in curve if row["update"] == selected_update
            ),
            "latest": curve[-1]["selection"],
            "latest_was_rejected": selected_update != curve[-1]["update"],
        },
        "stop_loss": {
            "reason": stop_reason,
            "minimum_baseline_visible_fraction": (
                run.minimum_baseline_visible_fraction
            ),
            # The RESOLVED ceiling, not the raw field: an unset field derives
            # from head count, and a report that published `null` would not say
            # what was actually enforced.
            "maximum_approximate_kl": _resolved_maximum_approximate_kl(run),
            "maximum_approximate_kl_declared": run.maximum_approximate_kl,
            "kl_violation_patience": _KL_VIOLATION_PATIENCE,
            "action_head_count": len(ARSENAL_POLICY_ACTION_HEAD_SIZES),
        },
        "checkpoint": {"path": str(checkpoint), "sha256": _sha256(checkpoint)},
        "latest_checkpoint": {
            "path": str(latest_checkpoint),
            "sha256": _sha256(latest_checkpoint),
        },
        "milestone_checkpoints": milestone_checkpoints,
        "head_names": list(ARSENAL_POLICY_ACTION_HEAD_NAMES),
        "target_phases": (
            ["flee_retreat", "flee_strafe_left", "flee_strafe_right"]
            if run.target_controller in _SCRIPTED_TARGET_LANE_MODES
            else ["frozen_sampled_policy"]
        ),
        "curve": curve,
        "updates": updates,
        "replay": selected_replay,
        "replay_archive": replay_archive,
    }
    (run.output / "report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    emit(
        "cancelled" if cancelled else "completed",
        {
            "report": str(run.output / "report.json"),
            "checkpoint": str(checkpoint),
            "wall_seconds": report["runtime"]["wall_seconds"],
        },
        phase="cancelled" if cancelled else "completed",
    )
    return report


def main(*, default_source_policy: Path | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        help="JSON PursuitRunConfig; explicit CLI fields are used when omitted",
    )
    parser.add_argument("--source-policy", type=Path, default=default_source_policy)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--evaluation-steps", type=int, default=PURSUIT_DEFAULT_TICKS)
    parser.add_argument(
        "--evaluation-updates", type=int, nargs="+", default=(0, 8, 16, 32, 64)
    )
    parser.add_argument("--seed", type=int, default=570_701)
    parser.add_argument("--archive-lanes", type=int, default=8)
    parser.add_argument("--selection-key", type=int, default=570_057)
    parser.add_argument("--world-artifact-seed", type=int)
    parser.add_argument("--node-seed", type=int, default=570_071)
    args = parser.parse_args()
    if args.spec is not None:
        values = json.loads(args.spec.read_text(encoding="utf-8-sig"))
        values["source_policy"] = Path(values["source_policy"])
        values["output"] = Path(values["output"])
        if "evaluation_updates" in values:
            values["evaluation_updates"] = tuple(values["evaluation_updates"])
        if "memory_reset_updates" in values:
            values["memory_reset_updates"] = tuple(values["memory_reset_updates"])
    else:
        values = vars(args)
        values.pop("spec")
        if values["source_policy"] is None or values["output"] is None:
            parser.error("--source-policy and --output are required without --spec")
    config = PursuitRunConfig(**values)
    try:
        report = run_pursuit_training(config)
    except Exception as error:
        TrainingMonitor(config.output).emit(
            "failed",
            {
                "error_type": type(error).__name__,
                "message": str(error),
            },
            phase="failed",
        )
        raise
    print(json.dumps({"status": report["status"], "curve": report["curve"]}, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "PURSUIT_RUN_SCHEMA",
    "PURSUIT_SELECTION_SCHEMA",
    "PURSUIT_COMPUTE_COMPOSITION_SCHEMA",
    "PursuitSelectionAssessment",
    "PursuitRunConfig",
    "pursuit_candidate_is_better",
    "pursuit_launch_options",
    "pursuit_region_artifacts",
    "pursuit_selection_assessment",
    "run_pursuit_training",
]
