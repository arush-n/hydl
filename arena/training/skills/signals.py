"""Pure-JAX causal signals and episode endings for reusable skill stages."""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral
from typing import NamedTuple

import jax
import jax.numpy as jnp

from arena.curriculum.strategy import EpisodeContract
from hytalegym.jax.training.multi_actor import MultiActorArenaState


PURSUIT_SUCCESS_DISTANCE = 3.0
PURSUIT_RADIAL_REWARD_SCALE = 4.0
PURSUIT_ALIGNMENT_REWARD_SCALE = 0.5
PURSUIT_SUCCESS_REWARD = 1.0
PURSUIT_TICK_COST = 0.002
PURSUIT_REWARD_LAW = (
    "4*clip(learner_displacement_toward_previous_target,-0.25,0.25)"
    "+0.5*learner_owned_alignment_gain"
    "+1[distance<=3 and learner_displacement_toward_previous_target>0]-0.002"
)

AIM_SUCCESS_DEGREES = 10.0
AIM_ALIGNMENT_REWARD_SCALE = 1.0 / 45.0
AIM_ALIGNED_REWARD = 0.02
AIM_UNNECESSARY_SPIN_COST = 0.01
AIM_REWARD_LAW = (
    "clip(counterfactual_current_bearing_error-before_to_after_yaw_error,-45,45)/45"
    "+0.02[absolute_current_bearing_error<=10]-0.01[turn_without_alignment_gain]"
)
AIM_TRACKING_SIGNAL_SCHEMA = "arena-aim-tracking-signal-v3"
_AIM_TRACKING_SIGNAL_CONTRACT = {
    "schema": AIM_TRACKING_SIGNAL_SCHEMA,
    "bearing_reference": "current_target_bearing_after_target_motion",
    "learner_credit": (
        "counterfactual_error_with_current_bearing_and_previous_learner_yaw"
        "-actual_error_after_learner_yaw"
    ),
    "target_motion_credit": (
        "previous_error-counterfactual_error_with_previous_learner_yaw"
    ),
    "causal_alignment": (
        "aligned_and(learner_alignment_gain>0_or(previously_aligned_and_"
        "target_bearing_stationary))"
    ),
    "success_threshold_degrees": AIM_SUCCESS_DEGREES,
    "reward_law": AIM_REWARD_LAW,
    "target_only_motion_can_certify_success": False,
    "small_error_policy": "ignore_sub_degree_corrections_and_only_penalize_material_wrong_turns",
}
PITCH_TRACKING_SIGNAL_SCHEMA = "arena-pitch-tracking-signal-v1"
_PITCH_TRACKING_SIGNAL_CONTRACT = {
    "schema": PITCH_TRACKING_SIGNAL_SCHEMA,
    "target": "source_eye_to_target_bounds_center_elevation",
    "learner_credit": (
        "counterfactual_error_with_current_elevation_and_previous_learner_pitch"
        "-actual_error_after_learner_pitch"
    ),
    "target_motion_credit": (
        "previous_error-counterfactual_error_with_previous_learner_pitch"
    ),
    "success_threshold_degrees": AIM_SUCCESS_DEGREES,
    "small_error_policy": (
        "ignore_sub_degree_corrections_and_only_penalize_material_wrong_turns"
    ),
}
SPATIAL_TRACKING_SIGNAL_SCHEMA = "arena-spatial-tracking-signal-v1"
SPATIAL_TRACKING_REWARD_LAW = (
    "clip(learner_owned_great_circle_error_reduction,-45,45)/45"
    "+0.02[causal_alignment]-0.01[materially_harmful_view_adjustment]"
)
_SPATIAL_TRACKING_SIGNAL_CONTRACT = {
    "schema": SPATIAL_TRACKING_SIGNAL_SCHEMA,
    "error": "great_circle_angle_between_view_and_target_direction_degrees",
    "target": "source_eye_to_target_bounds_center",
    "learner_credit": (
        "current_target_direction_with_previous_view_error-minus-current_"
        "target_direction_with_current_view_error"
    ),
    "axes": "yaw_and_pitch_are_one_spherical_objective_not_additive_rewards",
    "success_threshold_degrees": AIM_SUCCESS_DEGREES,
    "small_error_policy": (
        "ignore_sub_degree_corrections_and_only_penalize_material_wrong_turns"
    ),
    "reward_law": SPATIAL_TRACKING_REWARD_LAW,
    "reward_weights": {
        "alignment_gain_scale": AIM_ALIGNMENT_REWARD_SCALE,
        "causal_alignment": AIM_ALIGNED_REWARD,
        "harmful_adjustment": AIM_UNNECESSARY_SPIN_COST,
    },
}
AIM_SUSTAINED_ALIGNMENT_SCHEMA = "arena-aim-sustained-alignment-v1"
_AIM_SUSTAINED_ALIGNMENT_CONTRACT = {
    "schema": AIM_SUSTAINED_ALIGNMENT_SCHEMA,
    "acquisition": "first_aligned_tick_requires_learner_causal_alignment",
    "continuation": (
        "after_acquisition_each_valid_active_absolute_aligned_tick_advances_streak"
    ),
    "reset": "invalid_inactive_or_absolute_alignment_loss_resets_streak",
    "success": "streak_reaches_configured_required_ticks",
    "progress": "learner_alignment_gain_at_least_configured_epsilon",
    "target_only_entry_can_arm_streak": False,
}

ATTACK_TIMING_DAMAGE_SCALE = 0.05
ATTACK_TIMING_START_REWARD = 0.05
ATTACK_TIMING_FALSE_ACTIVATION_COST = 0.05
ATTACK_TIMING_TICK_COST = 0.001
ATTACK_TIMING_REWARD_LAW = (
    "0.05*engine_attributed_damage+0.05*causal_accepted_start"
    "-0.05*false_activation-0.001"
)
ATTACK_INTERVAL_OUTCOME_SCHEMA = "arena-attack-interval-outcome-v2"
_ATTACK_INTERVAL_OUTCOME_CONTRACT = {
    "schema": ATTACK_INTERVAL_OUTCOME_SCHEMA,
    "opens_on": "engine_ability_accepted",
    "authoritative_interval_end_sources": (
        "engine_lifecycle_completion",
        "authoritative_task_terminal",
    ),
    "positive_on": "authoritative_interval_end_after_engine_attributed_damage",
    "delayed_false_activation_on": (
        "authoritative_interval_end_without_engine_attributed_damage"
    ),
    "task_terminal_rule": (
        "close_pending_interval_before_episode_exit; classify from accumulated_"
        "engine_attributed_damage"
    ),
    "teacher_prediction_is_truth": False,
    "geometric_window_is_causal_label": False,
    "geometry_use": "diagnostic_only_and_bound_by_runtime_config_identity",
    "invalid_transition": "preserve_pending_interval_and_emit_no_outcome",
}


def _contract_sha256(manifest: dict[str, object]) -> str:
    return (
        hashlib.sha256(
            json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        )
        .hexdigest()
        .upper()
    )


def attack_interval_outcome_contract_manifest() -> dict[str, object]:
    """Return the policy-independent accepted-attack interval contract."""

    return dict(_ATTACK_INTERVAL_OUTCOME_CONTRACT)


def attack_interval_outcome_contract_sha256() -> str:
    """Hash the exact engine-attributed interval outcome semantics."""

    return (
        hashlib.sha256(
            json.dumps(
                _ATTACK_INTERVAL_OUTCOME_CONTRACT,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        )
        .hexdigest()
        .upper()
    )


def aim_tracking_signal_contract_manifest() -> dict[str, object]:
    """Return the exact target-motion-counterfactual aim semantics."""

    return dict(_AIM_TRACKING_SIGNAL_CONTRACT)


def aim_tracking_signal_contract_sha256() -> str:
    """Hash the learner-owned aim reward and causal-alignment semantics."""

    return (
        hashlib.sha256(
            json.dumps(
                _AIM_TRACKING_SIGNAL_CONTRACT,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        )
        .hexdigest()
        .upper()
    )


def pitch_tracking_signal_contract_manifest() -> dict[str, object]:
    """Return the exact vertical target-motion-counterfactual semantics."""

    return dict(_PITCH_TRACKING_SIGNAL_CONTRACT)


def pitch_tracking_signal_contract_sha256() -> str:
    """Hash learner-owned vertical tracking semantics."""

    return _contract_sha256(_PITCH_TRACKING_SIGNAL_CONTRACT)


def spatial_tracking_signal_contract_manifest() -> dict[str, object]:
    """Return the exact joint yaw/pitch tracking semantics."""

    return dict(_SPATIAL_TRACKING_SIGNAL_CONTRACT)


def spatial_tracking_signal_contract_sha256() -> str:
    """Hash the joint three-dimensional tracking semantics."""

    return _contract_sha256(_SPATIAL_TRACKING_SIGNAL_CONTRACT)


class PursuitTransitionSignals(NamedTuple):
    """Causal pursuit evidence for one lane transition."""

    distance_before: jax.Array
    distance_after: jax.Array
    learner_radial_progress: jax.Array
    opponent_radial_contribution: jax.Array
    learner_alignment_gain: jax.Array
    target_alignment_contribution: jax.Array
    geometric_success: jax.Array
    success: jax.Array
    reward: jax.Array

    @property
    def alignment_delta(self) -> jax.Array:
        """Compatibility name for the learner-owned alignment potential."""

        return self.learner_alignment_gain


class AimTrackingSignals(NamedTuple):
    """Yaw tracking with target motion removed from learner credit."""

    error_before: jax.Array
    counterfactual_error: jax.Array
    error_after: jax.Array
    learner_yaw_delta: jax.Array
    target_bearing_delta: jax.Array
    learner_alignment_gain: jax.Array
    target_alignment_contribution: jax.Array
    aligned: jax.Array
    causal_alignment: jax.Array
    unnecessary_spin: jax.Array
    reward: jax.Array


class PitchTrackingSignals(NamedTuple):
    """Pitch tracking with target elevation motion removed from learner credit."""

    error_before: jax.Array
    counterfactual_error: jax.Array
    error_after: jax.Array
    learner_pitch_delta: jax.Array
    target_elevation_delta: jax.Array
    learner_alignment_gain: jax.Array
    target_alignment_contribution: jax.Array
    aligned: jax.Array
    causal_alignment: jax.Array
    unnecessary_pitch: jax.Array
    reward: jax.Array


class SpatialTrackingSignals(NamedTuple):
    """One spherical tracking objective spanning yaw and pitch."""

    error_before: jax.Array
    counterfactual_error: jax.Array
    error_after: jax.Array
    learner_alignment_gain: jax.Array
    target_alignment_contribution: jax.Array
    aligned: jax.Array
    causal_alignment: jax.Array
    unnecessary_adjustment: jax.Array
    reward: jax.Array


class AimAlignmentProgressSignals(NamedTuple):
    """Stateful learner-owned acquisition followed by sustained alignment."""

    streak: jax.Array
    maximum_streak: jax.Array
    learner_owned_alignment: jax.Array
    success: jax.Array
    meaningful_progress: jax.Array


class AttackTimingSignals(NamedTuple):
    """Causal evidence for one authored basic-attack decision."""

    causal_start: jax.Array
    false_activation: jax.Array
    attributed_damage_event: jax.Array
    attributed_damage: jax.Array
    next_pending_start: jax.Array
    reward: jax.Array


class AttackIntervalCarry(NamedTuple):
    """Unresolved accepted interval and whether it has caused damage."""

    pending: jax.Array
    attributed_damage_seen: jax.Array


class AttackIntervalOutcomeSignals(NamedTuple):
    """Authoritative result emitted only when an accepted interval ends."""

    accepted_start: jax.Array
    attributed_damage_event: jax.Array
    authoritative_interval_end: jax.Array
    positive_interval_end: jax.Array
    delayed_false_activation: jax.Array
    overlapping_start: jax.Array
    next_carry: AttackIntervalCarry


class EpisodeProgressSignals(NamedTuple):
    """One lane-wise progress-clock update inside a JAX scan."""

    last_progress_tick: jax.Array
    terminated: jax.Array
    no_progress: jax.Array
    safety_horizon: jax.Array
    support_failure: jax.Array
    truncated: jax.Array
    done: jax.Array


def _angle_delta(target: jax.Array, current: jax.Array) -> jax.Array:
    return jnp.mod(target - current + 180.0, 360.0) - 180.0


def _duel_geometry(
    state: MultiActorArenaState,
    learner_actor_index: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    learner = jnp.asarray(learner_actor_index, dtype=jnp.int32)
    if learner.ndim != 1:
        raise ValueError("learner actor identity must have shape [B]")
    opponent = jnp.int32(1) - learner
    lanes = jnp.arange(learner.shape[0], dtype=jnp.int32)
    combat = state.arsenal.combat
    if combat.position.ndim != 3 or combat.position.shape[1] != 2:
        raise ValueError("skill signals currently require an exact two-entity duel")
    delta = combat.position[lanes, opponent] - combat.position[lanes, learner]
    delta_xz = delta[:, (0, 2)]
    distance = jnp.linalg.norm(delta_xz, axis=-1)
    bearing = jnp.degrees(jnp.arctan2(-delta_xz[:, 0], -delta_xz[:, 1]))
    return (
        combat.position[lanes, learner],
        combat.position[lanes, opponent],
        combat.yaw[lanes, learner],
        distance,
        bearing,
    )


def duel_head_yaw(state, learner_actor_index: jax.Array) -> jax.Array:
    """Return the learner's HEAD yaw and the bearing to its opponent.

    `_duel_geometry` above reports ``combat.yaw``, which `combat/types.py` 412
    documents as the BODY. Since the head became an independent lever the two
    are different numbers, and every tracking measurement taken so far has been
    the body's -- the head has never been scored at all.

    The head is stored per role rather than per lane: ``agent_head_yaw``
    describes actor 0 and ``target_head_yaw`` actor 1, so the learner's head has
    to be selected by identity instead of indexed like ``yaw``.
    """

    learner = jnp.asarray(learner_actor_index, dtype=jnp.int32)
    if learner.ndim != 1:
        raise ValueError("learner actor identity must have shape [B]")
    combat = state.arsenal.combat
    head = jnp.where(
        learner == jnp.int32(0),
        combat.agent_head_yaw,
        combat.target_head_yaw,
    )
    _, _, _, _, bearing = _duel_geometry(state, learner)
    return head, bearing


def _duel_pitch_geometry(state, learner: jax.Array, params):
    opponent = jnp.int32(1) - learner
    lanes = jnp.arange(learner.shape[0], dtype=jnp.int32)
    eye_y = jnp.stack((params.agent_eye_offset[1], params.target_eye_offset[1]))
    center_y = jnp.stack(
        (
            (params.agent_bounds[1] + params.agent_bounds[4]) * 0.5,
            (params.target_bounds[1] + params.target_bounds[4]) * 0.5,
        )
    )
    position = state.arsenal.combat.position
    delta = position[lanes, opponent] - position[lanes, learner]
    elevation = jnp.degrees(
        jnp.arctan2(
            delta[:, 1] + center_y[opponent] - eye_y[learner],
            jnp.maximum(jnp.linalg.norm(delta[:, (0, 2)], axis=-1), 1.0e-6),
        )
    )
    return state.locomotion.pitch[lanes, learner], elevation


def pursuit_transition_signals(
    previous: MultiActorArenaState,
    next_state: MultiActorArenaState,
    learner_actor_index: jax.Array,
    valid: jax.Array,
) -> PursuitTransitionSignals:
    """Reward only learner-owned closing and learner-owned facing change.

    A target walking toward an idle learner is diagnostic opposition motion;
    it cannot earn reward or certify the distance terminal.  Alignment uses a
    counterfactual that holds the *current* target bearing fixed while applying
    only the learner's yaw change, so lateral target motion also earns nothing.
    """

    learner = jnp.asarray(learner_actor_index, dtype=jnp.int32)
    evidence = jnp.asarray(valid, dtype=jnp.bool_)
    if learner.ndim != 1 or evidence.shape != learner.shape:
        raise ValueError("pursuit learner identity and validity must have shape [B]")
    before_learner, before_opponent, yaw_before, distance_before, bearing_before = (
        _duel_geometry(previous, learner)
    )
    after_learner, after_opponent, yaw_after, distance_after, bearing_after = (
        _duel_geometry(next_state, learner)
    )
    before_delta = before_opponent[:, (0, 2)] - before_learner[:, (0, 2)]
    before_direction = before_delta / jnp.maximum(distance_before[:, None], 1.0e-6)
    learner_motion = (after_learner - before_learner)[:, (0, 2)]
    opponent_motion = (after_opponent - before_opponent)[:, (0, 2)]
    learner_progress = jnp.sum(learner_motion * before_direction, axis=-1)
    opponent_contribution = -jnp.sum(opponent_motion * before_direction, axis=-1)

    error_before = jnp.abs(_angle_delta(bearing_before, yaw_before))
    counterfactual = jnp.abs(_angle_delta(bearing_after, yaw_before))
    error_after = jnp.abs(_angle_delta(bearing_after, yaw_after))
    alignment_before = jnp.cos(jnp.deg2rad(error_before))
    alignment_counterfactual = jnp.cos(jnp.deg2rad(counterfactual))
    alignment_after = jnp.cos(jnp.deg2rad(error_after))
    learner_alignment_gain = alignment_after - alignment_counterfactual
    target_alignment_contribution = alignment_counterfactual - alignment_before
    geometric_success = evidence & (
        distance_after <= jnp.float32(PURSUIT_SUCCESS_DISTANCE)
    )
    success = geometric_success & (learner_progress > 0.0)
    reward = (
        jnp.float32(PURSUIT_RADIAL_REWARD_SCALE)
        * jnp.clip(learner_progress, -0.25, 0.25)
        + jnp.float32(PURSUIT_ALIGNMENT_REWARD_SCALE) * learner_alignment_gain
        + jnp.float32(PURSUIT_SUCCESS_REWARD) * success
        - jnp.float32(PURSUIT_TICK_COST)
    )
    return PursuitTransitionSignals(
        distance_before,
        distance_after,
        jnp.where(evidence, learner_progress, 0.0),
        jnp.where(evidence, opponent_contribution, 0.0),
        jnp.where(evidence, learner_alignment_gain, 0.0),
        jnp.where(evidence, target_alignment_contribution, 0.0),
        geometric_success,
        success,
        jnp.where(evidence, reward, 0.0),
    )


def aim_tracking_signals(
    previous: MultiActorArenaState,
    next_state: MultiActorArenaState,
    learner_actor_index: jax.Array,
    valid: jax.Array,
    *,
    success_degrees: float = AIM_SUCCESS_DEGREES,
    gain_deadband_degrees: float = 0.5,
    harmful_turn_margin_degrees: float = 2.0,
) -> AimTrackingSignals:
    """Measure tracking against the current bearing, not target self-motion."""

    learner = jnp.asarray(learner_actor_index, dtype=jnp.int32)
    evidence = jnp.asarray(valid, dtype=jnp.bool_)
    if learner.ndim != 1 or evidence.shape != learner.shape:
        raise ValueError("aim learner identity and validity must have shape [B]")
    _, _, yaw_before, _, bearing_before = _duel_geometry(previous, learner)
    _, _, yaw_after, _, bearing_after = _duel_geometry(next_state, learner)
    return aim_tracking_geometry_signals(
        yaw_before,
        yaw_after,
        bearing_before,
        bearing_after,
        evidence,
        success_degrees=success_degrees,
        gain_deadband_degrees=gain_deadband_degrees,
        harmful_turn_margin_degrees=harmful_turn_margin_degrees,
    )


def pitch_tracking_signals(
    previous: MultiActorArenaState,
    next_state: MultiActorArenaState,
    learner_actor_index: jax.Array,
    valid: jax.Array,
    params,
    *,
    success_degrees: float = AIM_SUCCESS_DEGREES,
    gain_deadband_degrees: float = 0.5,
    harmful_turn_margin_degrees: float = 2.0,
) -> PitchTrackingSignals:
    """Measure eye-to-body-center pitch tracking for either duel actor."""

    learner = jnp.asarray(learner_actor_index, dtype=jnp.int32)
    evidence = jnp.asarray(valid, dtype=jnp.bool_)
    if learner.ndim != 1 or evidence.shape != learner.shape:
        raise ValueError("pitch learner identity and validity must have shape [B]")
    pitch_before, elevation_before = _duel_pitch_geometry(previous, learner, params)
    pitch_after, elevation_after = _duel_pitch_geometry(next_state, learner, params)
    return pitch_tracking_geometry_signals(
        pitch_before,
        pitch_after,
        elevation_before,
        elevation_after,
        evidence,
        success_degrees=success_degrees,
        gain_deadband_degrees=gain_deadband_degrees,
        harmful_turn_margin_degrees=harmful_turn_margin_degrees,
    )


def _spatial_error(
    yaw: jax.Array,
    pitch: jax.Array,
    bearing: jax.Array,
    elevation: jax.Array,
) -> jax.Array:
    """Great-circle angle between a view vector and target direction."""

    yaw_error = jnp.deg2rad(_angle_delta(bearing, yaw))
    pitch_rad = jnp.deg2rad(pitch)
    elevation_rad = jnp.deg2rad(elevation)
    cosine = jnp.sin(pitch_rad) * jnp.sin(elevation_rad) + jnp.cos(pitch_rad) * jnp.cos(
        elevation_rad
    ) * jnp.cos(yaw_error)
    return jnp.degrees(jnp.arccos(jnp.clip(cosine, -1.0, 1.0)))


def spatial_tracking_geometry_signals(
    yaw_before: jax.Array,
    yaw_after: jax.Array,
    pitch_before: jax.Array,
    pitch_after: jax.Array,
    bearing_before: jax.Array,
    bearing_after: jax.Array,
    elevation_before: jax.Array,
    elevation_after: jax.Array,
    valid: jax.Array,
    *,
    success_degrees: float = AIM_SUCCESS_DEGREES,
    gain_deadband_degrees: float = 0.5,
    harmful_turn_margin_degrees: float = 2.0,
) -> SpatialTrackingSignals:
    """Measure causal view correction as one three-dimensional angle."""

    values = tuple(
        jnp.asarray(value, dtype=jnp.float32)
        for value in (
            yaw_before,
            yaw_after,
            pitch_before,
            pitch_after,
            bearing_before,
            bearing_after,
            elevation_before,
            elevation_after,
        )
    )
    evidence = jnp.asarray(valid, dtype=jnp.bool_)
    if any(value.shape != evidence.shape for value in values):
        raise ValueError("spatial tracking values must share one lane shape")
    for name, value in (
        ("success_degrees", success_degrees),
        ("gain_deadband_degrees", gain_deadband_degrees),
        ("harmful_turn_margin_degrees", harmful_turn_margin_degrees),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not 0.0 < success_degrees <= 180.0:
        raise ValueError("spatial success threshold must be in (0, 180]")
    if harmful_turn_margin_degrees > 180.0:
        raise ValueError("harmful turn margin must not exceed 180 degrees")

    (
        yaw_before,
        yaw_after,
        pitch_before,
        pitch_after,
        bearing_before,
        bearing_after,
        elevation_before,
        elevation_after,
    ) = values
    error_before = _spatial_error(
        yaw_before, pitch_before, bearing_before, elevation_before
    )
    counterfactual = _spatial_error(
        yaw_before, pitch_before, bearing_after, elevation_after
    )
    error_after = _spatial_error(yaw_after, pitch_after, bearing_after, elevation_after)
    raw_gain = counterfactual - error_after
    learner_gain = jnp.where(
        jnp.abs(raw_gain) >= jnp.float32(gain_deadband_degrees), raw_gain, 0.0
    )
    target_contribution = error_before - counterfactual
    target_changed = (jnp.abs(_angle_delta(bearing_after, bearing_before)) > 1.0e-4) | (
        jnp.abs(elevation_after - elevation_before) > 1.0e-4
    )
    aligned = evidence & (error_after <= jnp.float32(success_degrees))
    causal_alignment = aligned & (
        (learner_gain > 0.0)
        | ((error_before <= jnp.float32(success_degrees)) & ~target_changed)
    )
    unnecessary = evidence & (learner_gain < -jnp.float32(harmful_turn_margin_degrees))
    reward = (
        jnp.clip(
            learner_gain * jnp.float32(AIM_ALIGNMENT_REWARD_SCALE),
            -1.0,
            1.0,
        )
        + jnp.float32(AIM_ALIGNED_REWARD) * causal_alignment
        - jnp.float32(AIM_UNNECESSARY_SPIN_COST) * unnecessary
    )
    return SpatialTrackingSignals(
        jnp.where(evidence, error_before, 0.0),
        jnp.where(evidence, counterfactual, 0.0),
        jnp.where(evidence, error_after, 0.0),
        jnp.where(evidence, learner_gain, 0.0),
        jnp.where(evidence, target_contribution, 0.0),
        aligned,
        causal_alignment,
        unnecessary,
        jnp.where(evidence, reward, 0.0),
    )


def spatial_tracking_signals(
    previous: MultiActorArenaState,
    next_state: MultiActorArenaState,
    learner_actor_index: jax.Array,
    valid: jax.Array,
    params,
    *,
    success_degrees: float = AIM_SUCCESS_DEGREES,
    gain_deadband_degrees: float = 0.5,
    harmful_turn_margin_degrees: float = 2.0,
) -> SpatialTrackingSignals:
    """Measure joint yaw/pitch tracking against the moving duel target."""

    learner = jnp.asarray(learner_actor_index, dtype=jnp.int32)
    evidence = jnp.asarray(valid, dtype=jnp.bool_)
    if learner.ndim != 1 or evidence.shape != learner.shape:
        raise ValueError("spatial learner identity and validity must have shape [B]")
    _, _, yaw_before, _, bearing_before = _duel_geometry(previous, learner)
    _, _, yaw_after, _, bearing_after = _duel_geometry(next_state, learner)
    pitch_before, elevation_before = _duel_pitch_geometry(previous, learner, params)
    pitch_after, elevation_after = _duel_pitch_geometry(next_state, learner, params)
    return spatial_tracking_geometry_signals(
        yaw_before,
        yaw_after,
        pitch_before,
        pitch_after,
        bearing_before,
        bearing_after,
        elevation_before,
        elevation_after,
        evidence,
        success_degrees=success_degrees,
        gain_deadband_degrees=gain_deadband_degrees,
        harmful_turn_margin_degrees=harmful_turn_margin_degrees,
    )


def aim_tracking_geometry_signals(
    yaw_before: jax.Array,
    yaw_after: jax.Array,
    bearing_before: jax.Array,
    bearing_after: jax.Array,
    valid: jax.Array,
    *,
    success_degrees: float = AIM_SUCCESS_DEGREES,
    gain_deadband_degrees: float = 0.5,
    harmful_turn_margin_degrees: float = 2.0,
) -> AimTrackingSignals:
    """Score learner-owned tracking from already-projected yaw and bearing.

    This is the shared kernel for actor-major and ordinary Arsenal rollouts.
    Target motion is removed with a counterfactual that holds the learner's
    previous yaw fixed. Sub-degree noise is ignored, and a turn is negative
    only when it materially increases the current target error.
    """

    values = tuple(
        jnp.asarray(value, dtype=jnp.float32)
        for value in (yaw_before, yaw_after, bearing_before, bearing_after)
    )
    evidence = jnp.asarray(valid, dtype=jnp.bool_)
    if any(value.shape != evidence.shape for value in values):
        raise ValueError("aim geometry and validity must share one lane shape")
    for name, value in (
        ("success_degrees", success_degrees),
        ("gain_deadband_degrees", gain_deadband_degrees),
        ("harmful_turn_margin_degrees", harmful_turn_margin_degrees),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not 0.0 < success_degrees <= 180.0:
        raise ValueError("aim success threshold must be in (0, 180]")
    if harmful_turn_margin_degrees > 180.0:
        raise ValueError("harmful turn margin must not exceed 180 degrees")

    yaw_before, yaw_after, bearing_before, bearing_after = values
    error_before = jnp.abs(_angle_delta(bearing_before, yaw_before))
    counterfactual = jnp.abs(_angle_delta(bearing_after, yaw_before))
    error_after = jnp.abs(_angle_delta(bearing_after, yaw_after))
    learner_yaw_delta = _angle_delta(yaw_after, yaw_before)
    target_bearing_delta = _angle_delta(bearing_after, bearing_before)
    raw_learner_gain = counterfactual - error_after
    learner_gain = jnp.where(
        jnp.abs(raw_learner_gain) >= jnp.float32(gain_deadband_degrees),
        raw_learner_gain,
        jnp.float32(0.0),
    )
    target_contribution = error_before - counterfactual
    aligned = evidence & (error_after <= jnp.float32(success_degrees))
    causal_alignment = aligned & (
        (learner_gain > 0.0)
        | (
            (error_before <= jnp.float32(success_degrees))
            & (jnp.abs(target_bearing_delta) <= 1.0e-4)
        )
    )
    unnecessary_spin = evidence & (
        learner_gain < -jnp.float32(harmful_turn_margin_degrees)
    )
    reward = (
        jnp.clip(
            learner_gain * jnp.float32(AIM_ALIGNMENT_REWARD_SCALE),
            -1.0,
            1.0,
        )
        + jnp.float32(AIM_ALIGNED_REWARD) * causal_alignment
        - jnp.float32(AIM_UNNECESSARY_SPIN_COST) * unnecessary_spin
    )
    return AimTrackingSignals(
        jnp.where(evidence, error_before, 0.0),
        jnp.where(evidence, counterfactual, 0.0),
        jnp.where(evidence, error_after, 0.0),
        jnp.where(evidence, learner_yaw_delta, 0.0),
        jnp.where(evidence, target_bearing_delta, 0.0),
        jnp.where(evidence, learner_gain, 0.0),
        jnp.where(evidence, target_contribution, 0.0),
        aligned,
        causal_alignment,
        unnecessary_spin,
        jnp.where(evidence, reward, 0.0),
    )


def pitch_tracking_geometry_signals(
    pitch_before: jax.Array,
    pitch_after: jax.Array,
    elevation_before: jax.Array,
    elevation_after: jax.Array,
    valid: jax.Array,
    *,
    success_degrees: float = AIM_SUCCESS_DEGREES,
    gain_deadband_degrees: float = 0.5,
    harmful_turn_margin_degrees: float = 2.0,
) -> PitchTrackingSignals:
    """Score learner-owned pitch change against current target elevation."""

    values = tuple(
        jnp.asarray(value, dtype=jnp.float32)
        for value in (pitch_before, pitch_after, elevation_before, elevation_after)
    )
    evidence = jnp.asarray(valid, dtype=jnp.bool_)
    if any(value.shape != evidence.shape for value in values):
        raise ValueError("pitch geometry and validity must share one lane shape")
    for name, value in (
        ("success_degrees", success_degrees),
        ("gain_deadband_degrees", gain_deadband_degrees),
        ("harmful_turn_margin_degrees", harmful_turn_margin_degrees),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not 0.0 < success_degrees <= 180.0:
        raise ValueError("pitch success threshold must be in (0, 180]")

    pitch_before, pitch_after, elevation_before, elevation_after = values
    error_before = jnp.abs(elevation_before - pitch_before)
    counterfactual = jnp.abs(elevation_after - pitch_before)
    error_after = jnp.abs(elevation_after - pitch_after)
    raw_gain = counterfactual - error_after
    learner_gain = jnp.where(
        jnp.abs(raw_gain) >= jnp.float32(gain_deadband_degrees), raw_gain, 0.0
    )
    target_delta = elevation_after - elevation_before
    aligned = evidence & (error_after <= jnp.float32(success_degrees))
    causal_alignment = aligned & (
        (learner_gain > 0.0)
        | (
            (error_before <= jnp.float32(success_degrees))
            & (jnp.abs(target_delta) <= 1.0e-4)
        )
    )
    unnecessary = evidence & (learner_gain < -jnp.float32(harmful_turn_margin_degrees))
    reward = (
        jnp.clip(learner_gain * jnp.float32(AIM_ALIGNMENT_REWARD_SCALE), -1.0, 1.0)
        + jnp.float32(AIM_ALIGNED_REWARD) * causal_alignment
        - jnp.float32(AIM_UNNECESSARY_SPIN_COST) * unnecessary
    )
    return PitchTrackingSignals(
        jnp.where(evidence, error_before, 0.0),
        jnp.where(evidence, counterfactual, 0.0),
        jnp.where(evidence, error_after, 0.0),
        jnp.where(evidence, pitch_after - pitch_before, 0.0),
        jnp.where(evidence, target_delta, 0.0),
        jnp.where(evidence, learner_gain, 0.0),
        jnp.where(evidence, error_before - counterfactual, 0.0),
        aligned,
        causal_alignment,
        unnecessary,
        jnp.where(evidence, reward, 0.0),
    )


def aim_sustained_alignment_contract_manifest() -> dict[str, object]:
    """Return the reusable stateful aim-success contract."""

    return dict(_AIM_SUSTAINED_ALIGNMENT_CONTRACT)


def aim_sustained_alignment_contract_sha256() -> str:
    """Hash stateful aim-success semantics independently of Dawn."""

    payload = json.dumps(
        aim_sustained_alignment_contract_manifest(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest().upper()


def advance_aim_alignment_progress(
    streak: jax.Array,
    maximum_streak: jax.Array,
    aligned: jax.Array,
    causal_acquisition: jax.Array,
    learner_alignment_gain: jax.Array,
    active: jax.Array,
    valid: jax.Array,
    *,
    required_ticks: int,
    progress_epsilon_degrees: float,
) -> AimAlignmentProgressSignals:
    """Require learner acquisition once, then measure sustained absolute aim.

    A target cannot move itself into the learner's crosshair and start the
    streak.  Once a learner-owned correction acquires the target, however,
    holding the target within the absolute alignment threshold is sufficient;
    demanding another positive angular gain every tick would make a stable aim
    state impossible to sustain.
    """

    current = jnp.asarray(streak, dtype=jnp.int32)
    maximum = jnp.asarray(maximum_streak, dtype=jnp.int32)
    aligned = jnp.asarray(aligned, dtype=jnp.bool_)
    causal = jnp.asarray(causal_acquisition, dtype=jnp.bool_)
    gain = jnp.asarray(learner_alignment_gain, dtype=jnp.float32)
    active = jnp.asarray(active, dtype=jnp.bool_)
    evidence = jnp.asarray(valid, dtype=jnp.bool_)
    if current.ndim != 1 or any(
        value.shape != current.shape
        for value in (maximum, aligned, causal, gain, active, evidence)
    ):
        raise ValueError("aim progress evidence must share one lane shape [B]")
    if isinstance(required_ticks, bool) or not isinstance(required_ticks, Integral):
        raise TypeError("aim required ticks must be an integer")
    if int(required_ticks) < 1:
        raise ValueError("aim required ticks must be positive")
    epsilon = float(progress_epsilon_degrees)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("aim progress epsilon must be positive and finite")

    acquired = current > 0
    learner_owned_alignment = active & evidence & aligned & (causal | acquired)
    next_streak = jnp.where(learner_owned_alignment, current + 1, 0)
    next_maximum = jnp.maximum(maximum, next_streak)
    success = active & evidence & (next_streak >= jnp.int32(required_ticks))
    meaningful_progress = active & evidence & (gain >= jnp.float32(epsilon))
    return AimAlignmentProgressSignals(
        next_streak,
        next_maximum,
        learner_owned_alignment,
        success,
        meaningful_progress,
    )


def attack_timing_signals(
    requested: jax.Array,
    accepted: jax.Array,
    damage: jax.Array,
    start_eligible: jax.Array,
    pending_start: jax.Array,
    valid: jax.Array,
    *,
    pending_expired: jax.Array | None = None,
) -> AttackTimingSignals:
    """Attribute delayed damage to one eligible, accepted attack start.

    ``start_eligible`` is environment/task evidence.  A policy-specific teacher
    may predict that evidence, but its label must not replace engine truth in a
    reusable objective. ``pending_expired`` is positive lifecycle evidence that
    an admitted attempt ended without damage; mere readiness cannot erase a
    pending causal attribution.
    """

    requested = jnp.asarray(requested, dtype=jnp.bool_)
    accepted = jnp.asarray(accepted, dtype=jnp.bool_)
    damage = jnp.asarray(damage, dtype=jnp.float32)
    start_eligible = jnp.asarray(start_eligible, dtype=jnp.bool_)
    pending_start = jnp.asarray(pending_start, dtype=jnp.bool_)
    evidence = jnp.asarray(valid, dtype=jnp.bool_)
    expired = (
        jnp.zeros_like(pending_start)
        if pending_expired is None
        else jnp.asarray(pending_expired, dtype=jnp.bool_)
    )
    shape = requested.shape
    if any(
        value.shape != shape
        for value in (
            accepted,
            damage,
            start_eligible,
            pending_start,
            evidence,
            expired,
        )
    ):
        raise ValueError("attack timing evidence must share one lane shape")

    attributed_event = evidence & pending_start & (damage > 0.0)
    pending_ended = evidence & pending_start & expired & ~attributed_event
    start_slot_available = ~pending_start | attributed_event | pending_ended
    causal_start = (
        evidence & requested & accepted & start_eligible & start_slot_available
    )
    false_activation = evidence & requested & ~causal_start
    attributed_damage = jnp.where(attributed_event, damage, 0.0)
    retained_pending = pending_start & ~attributed_event & ~pending_ended
    next_pending = evidence & (retained_pending | causal_start)
    reward = (
        jnp.float32(ATTACK_TIMING_DAMAGE_SCALE) * attributed_damage
        + jnp.float32(ATTACK_TIMING_START_REWARD) * causal_start
        - jnp.float32(ATTACK_TIMING_FALSE_ACTIVATION_COST) * false_activation
        - jnp.float32(ATTACK_TIMING_TICK_COST)
    )
    return AttackTimingSignals(
        causal_start,
        false_activation,
        attributed_event,
        attributed_damage,
        next_pending,
        jnp.where(evidence, reward, 0.0),
    )


def attack_interval_outcome_signals(
    accepted: jax.Array,
    attributed_damage: jax.Array,
    engine_lifecycle_completion: jax.Array,
    authoritative_task_terminal: jax.Array,
    carry: AttackIntervalCarry,
    valid: jax.Array,
) -> AttackIntervalOutcomeSignals:
    """Resolve accepted attacks from engine outcomes, never teacher predictions.

    An accepted start remains pending across ticks.  It becomes a positive
    example only when an authoritative interval end follows at least one
    engine-attributed damage event.  An interval ends either when the engine
    lifecycle completes or when an authoritative task terminal closes the
    episode (for example, the sixth attributed hit).  An interval end with no
    such event is the delayed hard negative.  Geometry may explain either
    outcome but is not a causal label here.
    """

    if not isinstance(carry, AttackIntervalCarry):
        raise TypeError("attack interval outcome needs an AttackIntervalCarry")
    accepted = jnp.asarray(accepted, dtype=jnp.bool_)
    damage = jnp.asarray(attributed_damage, dtype=jnp.float32)
    lifecycle_completed = jnp.asarray(engine_lifecycle_completion, dtype=jnp.bool_)
    task_terminal = jnp.asarray(authoritative_task_terminal, dtype=jnp.bool_)
    evidence = jnp.asarray(valid, dtype=jnp.bool_)
    pending = jnp.asarray(carry.pending, dtype=jnp.bool_)
    damage_seen = jnp.asarray(carry.attributed_damage_seen, dtype=jnp.bool_)
    shape = accepted.shape
    if any(
        value.shape != shape
        for value in (
            damage,
            lifecycle_completed,
            task_terminal,
            evidence,
            pending,
            damage_seen,
        )
    ):
        raise ValueError("attack interval evidence must share one lane shape")

    accepted_start = evidence & accepted & ~pending
    overlapping_start = evidence & accepted & pending
    interval_open = pending | accepted_start
    attributed_event = evidence & interval_open & (damage > 0.0)
    interval_damage_seen = (pending & damage_seen) | attributed_event
    authoritative_interval_end = lifecycle_completed | task_terminal
    resolved = evidence & interval_open & authoritative_interval_end
    positive = resolved & interval_damage_seen
    delayed_false = resolved & ~interval_damage_seen
    next_pending = jnp.where(evidence, interval_open & ~resolved, pending)
    next_damage_seen = jnp.where(
        evidence,
        interval_damage_seen & ~resolved,
        damage_seen,
    )
    return AttackIntervalOutcomeSignals(
        accepted_start,
        attributed_event,
        resolved,
        positive,
        delayed_false,
        overlapping_start,
        AttackIntervalCarry(next_pending, next_damage_seen),
    )


def advance_episode_progress(
    contract: EpisodeContract,
    current_tick: jax.Array,
    last_progress_tick: jax.Array,
    meaningful_progress: jax.Array,
    success: jax.Array,
    natural_terminal: jax.Array,
    valid: jax.Array,
    active: jax.Array,
) -> EpisodeProgressSignals:
    """Apply one common success/no-progress/horizon law inside ``lax.scan``."""

    if not isinstance(contract, EpisodeContract):
        raise TypeError("episode progress requires an EpisodeContract")
    active = jnp.asarray(active, dtype=jnp.bool_)
    shape = active.shape
    values = tuple(
        jnp.asarray(value, dtype=jnp.bool_)
        for value in (meaningful_progress, success, natural_terminal, valid)
    )
    if active.ndim != 1 or any(value.shape != shape for value in values):
        raise ValueError("episode evidence must share one lane shape [B]")
    meaningful_progress, success, natural_terminal, valid = values
    tick = jnp.asarray(current_tick, dtype=jnp.int32)
    if tick.ndim == 0:
        tick = jnp.broadcast_to(tick, shape)
    last = jnp.asarray(last_progress_tick, dtype=jnp.int32)
    if tick.shape != shape or last.shape != shape:
        raise ValueError("episode ticks must be scalar or int32[B]")
    next_last = jnp.where(active & meaningful_progress, tick, last)
    terminated = active & (success | natural_terminal)
    support_failure = active & ~terminated & ~valid
    no_progress = (
        active
        & ~terminated
        & valid
        & (tick - next_last >= jnp.int32(contract.no_progress_patience_ticks))
    )
    safety_horizon = (
        active
        & ~terminated
        & valid
        & ~no_progress
        & (tick >= jnp.int32(contract.maximum_ticks))
    )
    truncated = ~terminated & (no_progress | safety_horizon | support_failure)
    return EpisodeProgressSignals(
        next_last,
        terminated,
        no_progress,
        safety_horizon,
        support_failure,
        truncated,
        terminated | truncated,
    )


__all__ = [
    "AIM_REWARD_LAW",
    "AIM_SUSTAINED_ALIGNMENT_SCHEMA",
    "AIM_SUCCESS_DEGREES",
    "AIM_TRACKING_SIGNAL_SCHEMA",
    "ATTACK_TIMING_REWARD_LAW",
    "ATTACK_INTERVAL_OUTCOME_SCHEMA",
    "AimAlignmentProgressSignals",
    "AimTrackingSignals",
    "AttackTimingSignals",
    "AttackIntervalCarry",
    "AttackIntervalOutcomeSignals",
    "EpisodeProgressSignals",
    "PURSUIT_REWARD_LAW",
    "PURSUIT_SUCCESS_DISTANCE",
    "PITCH_TRACKING_SIGNAL_SCHEMA",
    "PitchTrackingSignals",
    "PursuitTransitionSignals",
    "SPATIAL_TRACKING_SIGNAL_SCHEMA",
    "SpatialTrackingSignals",
    "advance_aim_alignment_progress",
    "advance_episode_progress",
    "aim_sustained_alignment_contract_manifest",
    "aim_sustained_alignment_contract_sha256",
    "aim_tracking_signal_contract_manifest",
    "aim_tracking_signal_contract_sha256",
    "aim_tracking_geometry_signals",
    "aim_tracking_signals",
    "attack_timing_signals",
    "attack_interval_outcome_contract_manifest",
    "attack_interval_outcome_contract_sha256",
    "attack_interval_outcome_signals",
    "pursuit_transition_signals",
    "pitch_tracking_geometry_signals",
    "pitch_tracking_signal_contract_manifest",
    "pitch_tracking_signal_contract_sha256",
    "pitch_tracking_signals",
    "spatial_tracking_geometry_signals",
    "spatial_tracking_signal_contract_manifest",
    "spatial_tracking_signal_contract_sha256",
    "spatial_tracking_signals",
]
