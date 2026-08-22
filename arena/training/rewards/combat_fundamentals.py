"""Dense, engine-grounded combat rewards for JAX policy training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real as RealNumber
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from arena.jax_contract import HEAD_SPANS
from arena.params import Real
from arena.training.runtime.contracts import (
    EnvironmentTrainingProfile,
    MetricSpec,
    StopLossSpec,
    TrainingFeature,
)
from arena.training.skills.signals import aim_tracking_geometry_signals
from hytalegym.jax.combat import AGENT_ENTITY, TARGET_ENTITY
from hytalegym.jax.combat.arsenal.schema.contract import (
    EF_HALF_ANGLE_DEGREES,
    EF_RANGE,
    EVENT_FLAG_SELECTOR_IGNORES_LINE_OF_SIGHT,
    EVENT_MELEE_CONE,
)
from hytalegym.jax.combat.arsenal.programs.event_storage import ability_event_bank
from hytalegym.jax.combat.observation.v3.policy.look_deltas import decode_look_delta
from hytalegym.jax.training.ppo import PPOEnvironment


COMBAT_FUNDAMENTALS_REWARD_SCHEMA = "arena_combat_fundamentals_reward_v4"
COMBAT_ATTACK_ENVELOPE_SCHEMA = "arena_combat_attack_envelope_v1"
_ABILITY_HEAD = tuple(HEAD_SPANS).index("ability_none_plus_slots")
_ABILITY_CHOICES = HEAD_SPANS["ability_none_plus_slots"][1]
_YAW_HEAD = tuple(HEAD_SPANS).index("yaw_delta_bins")
_YAW_CHOICES = HEAD_SPANS["yaw_delta_bins"][1]


class CombatAttackEnvelope(NamedTuple):
    """Per-environment basic-attack geometry projected from authored events."""

    range_blocks: jax.Array
    half_angle_degrees: jax.Array
    requires_line_of_sight: jax.Array
    valid: jax.Array

    def describe(self) -> dict[str, Any]:
        return {
            "schema": COMBAT_ATTACK_ENVELOPE_SCHEMA,
            "range_blocks": np.asarray(jax.device_get(self.range_blocks)).tolist(),
            "half_angle_degrees": np.asarray(
                jax.device_get(self.half_angle_degrees)
            ).tolist(),
            "requires_line_of_sight": np.asarray(
                jax.device_get(self.requires_line_of_sight)
            ).tolist(),
            "valid": np.asarray(jax.device_get(self.valid)).tolist(),
            "source": "arsenal.authored_basic_ability.melee_cone_events",
        }


def arsenal_basic_attack_envelope(
    runtime_config: Any, ability_slot: int = 0
) -> CombatAttackEnvelope:
    """Project reach and cone from the selected loadout without weapon branches."""

    loadout = runtime_config.loadout
    if ability_slot < 0 or ability_slot >= loadout.ability_mask.shape[2]:
        raise ValueError("basic attack ability slot is outside the loadout")
    event_mask, _, event_kind, event_f32, _, event_flags = ability_event_bank(loadout)
    mask = event_mask[:, AGENT_ENTITY, ability_slot] & (
        event_kind[:, AGENT_ENTITY, ability_slot] == EVENT_MELEE_CONE
    )
    values = event_f32[:, AGENT_ENTITY, ability_slot]
    flags = event_flags[:, AGENT_ENTITY, ability_slot]
    reach = jnp.max(jnp.where(mask, values[..., EF_RANGE], jnp.float32(0.0)), axis=1)
    half_angle = jnp.max(
        jnp.where(
            mask,
            values[..., EF_HALF_ANGLE_DEGREES],
            jnp.float32(0.0),
        ),
        axis=1,
    )
    ignores_line_of_sight = (
        flags & jnp.asarray(EVENT_FLAG_SELECTOR_IGNORES_LINE_OF_SIGHT, flags.dtype)
    ) != 0
    valid = jnp.any(mask, axis=1) & jnp.isfinite(reach) & (reach > 0.0)
    return CombatAttackEnvelope(
        reach.astype(jnp.float32),
        half_angle.astype(jnp.float32),
        jnp.any(mask & ~ignores_line_of_sight, axis=1),
        valid,
    )


@dataclass(frozen=True, slots=True)
class CombatFundamentalsConfig:
    """Weights for dense combat skills; every positive term has a counter-cost."""

    base_reward_scale: float = 1.0
    rollout_horizon_ticks: int = 512
    discount: float = 0.99
    engagement_min_reach_fraction: float = 0.35
    engagement_max_reach_fraction: float = 0.95
    proximity_reach_multiplier: float = 4.0
    attack_range_margin_fraction: float = 0.15
    attack_range_margin_blocks: float = 0.25
    aim_margin_degrees: float = 10.0
    spin_tolerance_degrees: float = 10.0
    approach_progress_scale: float = 0.10
    alignment_progress_scale: float = 0.04
    tracking_state_scale: float = 0.25
    engagement_state_scale: float = 0.25
    well_timed_attack_scale: float = 0.005
    damage_dealt_scale: float = 1.0
    damage_received_scale: float = 0.50
    blocked_hit_scale: float = 0.0
    accepted_attack_cost: float = 0.0
    missed_attack_penalty: float = 0.03
    cooldown_request_penalty: float = 0.04
    busy_request_penalty: float = 0.04
    rejected_attack_penalty: float = 0.02
    illegal_joint_action_penalty: float = 0.04
    excess_spin_penalty: float = 0.01
    victory_bonus: float = 0.50
    death_penalty: float = 0.50
    simultaneous_death_penalty: float = 0.25
    clip: float | None = 3.0
    basic_ability_slot: int = 0
    basic_attack_enabled: bool = True
    restrict_to_fundamentals: bool = True
    tracking_success_degrees: float = 15.0
    tracking_gain_deadband_degrees: float = 0.5
    tracking_harmful_turn_margin_degrees: float = 2.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name in {
                "basic_ability_slot",
                "rollout_horizon_ticks",
                "restrict_to_fundamentals",
                "basic_attack_enabled",
                "clip",
            }:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, RealNumber)
                or not math.isfinite(float(value))
                or value < 0.0
            ):
                raise ValueError(f"combat reward {name} must be finite and nonnegative")
        if not 0.0 <= self.discount <= 1.0:
            raise ValueError("combat reward discount must be in [0, 1]")
        if not (
            0.0
            < self.engagement_min_reach_fraction
            < self.engagement_max_reach_fraction
            <= 1.0
            < self.proximity_reach_multiplier
        ):
            raise ValueError("combat reward reach fractions are not ordered")
        if self.spin_tolerance_degrees > 45.0:
            raise ValueError("combat reward spin tolerance must not exceed 45 degrees")
        if not 0.0 < self.tracking_success_degrees <= 180.0:
            raise ValueError("tracking_success_degrees must be in (0, 180]")
        if self.tracking_harmful_turn_margin_degrees > 180.0:
            raise ValueError("tracking harmful-turn margin must not exceed 180 degrees")
        if self.clip is not None and (
            isinstance(self.clip, bool)
            or not isinstance(self.clip, RealNumber)
            or not math.isfinite(float(self.clip))
            or self.clip <= 0.0
        ):
            raise ValueError("combat reward clip must be positive or None")
        if (
            isinstance(self.basic_ability_slot, bool)
            or not isinstance(self.basic_ability_slot, int)
            or self.basic_ability_slot < 0
        ):
            raise ValueError("basic_ability_slot must be a nonnegative integer")
        if self.basic_ability_slot + 1 >= _ABILITY_CHOICES:
            raise ValueError("basic_ability_slot is outside the ability action head")
        if not isinstance(self.restrict_to_fundamentals, bool) or not isinstance(
            self.basic_attack_enabled, bool
        ):
            raise TypeError("combat action-scope flags must be bools")
        if (
            isinstance(self.rollout_horizon_ticks, bool)
            or not isinstance(self.rollout_horizon_ticks, int)
            or self.rollout_horizon_ticks != 512
        ):
            raise ValueError("combat fundamentals require a 512-tick PPO horizon")

    def describe(self) -> dict[str, Any]:
        values = asdict(self)
        return {
            "schema": COMBAT_FUNDAMENTALS_REWARD_SCHEMA,
            "semantics": {
                "episode_boundary": (
                    "natural combat outcome or exact 512-tick time limit; "
                    "time limits bootstrap the final pre-reset value"
                ),
                "proximity": "discounted potential into a bounded engagement band",
                "alignment": (
                    "learner-owned reduction in current target-bearing error; "
                    "target motion is removed counterfactually"
                ),
                "tracking_memory": (
                    "actor receives current relative pose/velocity; recurrent carry "
                    "persists until the exact episode boundary"
                ),
                "tracking_state": (
                    "per-horizon occupancy inside the configured centered cone"
                ),
                "proper_attack": "accepted basic onset inside authored range/angle margins",
                "hit": "engine-attributed damage by the learner actor",
                "miss": "penalty only beyond authored range/angle plus tolerance",
                "cooldown": "one penalty per held request streak during cooldown",
                "busy": "one penalty for a new request during another lifecycle",
                "normalization": (
                    "distance by authored reach; damage by health; persistent state "
                    "by episode horizon; penalties by decision/lifecycle event"
                ),
                "unknown": "no inferred hit, miss, cooldown, or terminal labels",
                "action_scope": (
                    (
                        "movement, yaw, pitch, none/basic-attack only"
                        if self.basic_attack_enabled
                        else "movement, yaw, pitch only; every attack choice is masked"
                    )
                    if self.restrict_to_fundamentals
                    else "full authored action surface"
                ),
            },
            "weights": values,
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.describe(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest().upper()


def combat_fundamentals_training_feature(
    config: CombatFundamentalsConfig,
    attack_envelope: CombatAttackEnvelope,
) -> TrainingFeature:
    """Environment-owned tuning, evaluation, and stop-loss declaration."""

    return TrainingFeature(
        key="combat.fundamentals",
        version=2,
        parameter_space={
            "combat.attack_range_margin_fraction": Real(0.05, 0.35),
            "combat.aim_margin_degrees": Real(2.0, 20.0),
            "combat.tracking_success_degrees": Real(5.0, 30.0),
            "combat.missed_attack_penalty": Real(0.005, 0.10),
            "combat.cooldown_request_penalty": Real(0.005, 0.10),
            "combat.excess_spin_penalty": Real(0.001, 0.05),
        },
        metrics=(
            MetricSpec("combat.return", "reward", weight=0.05),
            MetricSpec("combat.target_damage", "progress", weight=1.0),
            MetricSpec("combat.aim_error_degrees", "objective", "minimize", weight=1.0),
            MetricSpec("combat.aligned_fraction", "progress", weight=0.5),
            MetricSpec(
                "combat.harmful_turn_fraction", "progress", "minimize", weight=0.5
            ),
            MetricSpec("combat.strict_success", "outcome", weight=0.5),
            MetricSpec("combat.death_rate", "outcome", "minimize", weight=0.25),
            MetricSpec("combat.censoring_rate", "censoring", "minimize", weight=0.1),
            MetricSpec("combat.invalid_rate", "invalid", "minimize"),
            MetricSpec("combat.geometry_exhausted_rate", "support", "minimize"),
        ),
        stop_losses=(
            StopLossSpec(
                "combat.invalid_transition",
                "combat.invalid_rate",
                "above",
                0.0,
                "stop_training",
            ),
            StopLossSpec(
                "combat.geometry_exhausted",
                "combat.geometry_exhausted_rate",
                "above",
                0.0,
            ),
        ),
        capabilities=frozenset(
            {
                "jax.jit",
                "jax.vmap",
                "combat.authored_attack_envelope",
                "combat.engine_attributed_damage",
                "combat.exact_episode_outcome",
                "combat.causal_target_tracking",
                "policy.recurrent_memory",
            }
        ),
        manifest={
            "reward": {**config.describe(), "sha256": config.sha256},
            "attack_envelope": attack_envelope.describe(),
        },
    )


def combat_fundamentals_training_profile(
    environment_contract_sha256: str,
    config: CombatFundamentalsConfig,
    attack_envelope: CombatAttackEnvelope,
) -> EnvironmentTrainingProfile:
    """Return an immutable base profile that another agent may replace/extend."""

    return EnvironmentTrainingProfile(environment_contract_sha256).add(
        combat_fundamentals_training_feature(config, attack_envelope)
    )


class CombatFundamentalsTerms(NamedTuple):
    approach_progress: jax.Array
    alignment_progress: jax.Array
    tracking_state: jax.Array
    engagement_state: jax.Array
    well_timed_attack: jax.Array
    damage_dealt: jax.Array
    damage_received: jax.Array
    blocked_hit: jax.Array
    accepted_attack_cost: jax.Array
    missed_attack: jax.Array
    cooldown_request: jax.Array
    busy_request: jax.Array
    rejected_attack: jax.Array
    illegal_joint_action: jax.Array
    excess_spin: jax.Array
    victory: jax.Array
    death: jax.Array
    simultaneous_death: jax.Array


class CombatFundamentalsState(NamedTuple):
    environment: Any
    target_distance: jax.Array
    facing_error_degrees: jax.Array
    attack_open: jax.Array
    attack_hit: jax.Array
    attack_start_distance: jax.Array
    attack_start_error_degrees: jax.Array
    rejected_request_latched: jax.Array
    last_terms: CombatFundamentalsTerms


def _runtime(state: Any) -> Any:
    runtime = getattr(state, "runtime", None)
    if (
        runtime is None
        or not hasattr(runtime, "combat")
        or not hasattr(runtime, "arsenal")
    ):
        raise TypeError("combat fundamentals require an Arsenal PPO environment state")
    return runtime


def _target_geometry(
    state: Any,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    combat = _runtime(state).combat
    actor = combat.position[:, AGENT_ENTITY]
    target = combat.position[:, TARGET_ENTITY]
    offset = target - actor
    distance = jnp.linalg.norm(offset[:, (0, 2)], axis=1)
    bearing = jnp.rad2deg(jnp.arctan2(-offset[:, 0], -offset[:, 2]))
    error = (bearing - combat.yaw[:, AGENT_ENTITY] + 180.0) % 360.0 - 180.0
    return (
        distance.astype(jnp.float32),
        bearing.astype(jnp.float32),
        combat.yaw[:, AGENT_ENTITY].astype(jnp.float32),
        jnp.abs(error).astype(jnp.float32),
    )


def _engagement_score(
    distance: jax.Array,
    reach: jax.Array,
    config: CombatFundamentalsConfig,
) -> jax.Array:
    minimum = reach * config.engagement_min_reach_fraction
    maximum = reach * config.engagement_max_reach_fraction
    proximity = reach * config.proximity_reach_multiplier
    below = jnp.clip(distance / jnp.maximum(minimum, 1.0e-6), 0.0, 1.0)
    above = jnp.clip(
        (proximity - distance) / jnp.maximum(proximity - maximum, 1.0e-6),
        0.0,
        1.0,
    )
    return jnp.minimum(below, above)


def _zero_terms(shape: tuple[int, ...]) -> CombatFundamentalsTerms:
    zero = jnp.zeros(shape, dtype=jnp.float32)
    return CombatFundamentalsTerms(*(zero for _ in CombatFundamentalsTerms._fields))


def _weighted_reward(terms: CombatFundamentalsTerms, config):
    reward = (
        config.approach_progress_scale * terms.approach_progress
        + config.alignment_progress_scale * terms.alignment_progress
        + (config.tracking_state_scale / config.rollout_horizon_ticks)
        * terms.tracking_state
        + (config.engagement_state_scale / config.rollout_horizon_ticks)
        * terms.engagement_state
        + config.well_timed_attack_scale * terms.well_timed_attack
        + config.damage_dealt_scale * terms.damage_dealt
        - config.damage_received_scale * terms.damage_received
        + config.blocked_hit_scale * terms.blocked_hit
        - config.accepted_attack_cost * terms.accepted_attack_cost
        - config.missed_attack_penalty * terms.missed_attack
        - config.cooldown_request_penalty * terms.cooldown_request
        - config.busy_request_penalty * terms.busy_request
        - config.rejected_attack_penalty * terms.rejected_attack
        - config.illegal_joint_action_penalty * terms.illegal_joint_action
        - config.excess_spin_penalty * terms.excess_spin
        + config.victory_bonus * terms.victory
        - config.death_penalty * terms.death
        - config.simultaneous_death_penalty * terms.simultaneous_death
    )
    return (
        reward if config.clip is None else jnp.clip(reward, -config.clip, config.clip)
    )


def make_combat_fundamentals_environment(
    environment: PPOEnvironment,
    config: CombatFundamentalsConfig,
    *,
    attack_envelope: CombatAttackEnvelope,
    agent_max_health: Any,
    target_max_health: Any,
) -> PPOEnvironment:
    """Add dense combat feedback without exposing privileged facts to the actor."""

    if not isinstance(environment, PPOEnvironment):
        raise TypeError("environment must be a PPOEnvironment")
    if not isinstance(config, CombatFundamentalsConfig):
        raise TypeError("config must be CombatFundamentalsConfig")
    if not isinstance(attack_envelope, CombatAttackEnvelope):
        raise TypeError("attack_envelope must be CombatAttackEnvelope")
    if environment.step_detailed is None or environment.episode_outcome is None:
        raise ValueError("combat fundamentals require detailed steps and outcomes")
    envelope_leaves = tuple(jnp.asarray(value) for value in attack_envelope)
    if any(value.ndim != 1 for value in envelope_leaves):
        raise ValueError("attack envelope leaves must have shape [batch]")
    if len({value.shape for value in envelope_leaves}) != 1:
        raise ValueError("attack envelope leaves must have equal shapes")
    if not bool(np.all(np.asarray(jax.device_get(attack_envelope.valid)))):
        raise ValueError("basic attack has no authored melee-cone envelope")
    reach = jnp.asarray(attack_envelope.range_blocks, dtype=jnp.float32)
    half_angle = jnp.asarray(attack_envelope.half_angle_degrees, dtype=jnp.float32)
    agent_health = jnp.maximum(jnp.asarray(agent_max_health, jnp.float32), 1.0e-6)
    target_health = jnp.maximum(jnp.asarray(target_max_health, jnp.float32), 1.0e-6)
    action_scope = jnp.concatenate(
        tuple(
            (
                (jnp.arange(size) == 0)
                | (
                    config.basic_attack_enabled
                    & (name == "ability_none_plus_slots")
                    & (jnp.arange(size) == config.basic_ability_slot + 1)
                )
                if name
                not in {
                    "locomotion_gait_compass",
                    "yaw_delta_bins",
                    "pitch_delta_bins",
                }
                else jnp.ones((size,), dtype=jnp.bool_)
            )
            for name, (_, size) in HEAD_SPANS.items()
        )
    )

    def scoped(mask):
        return mask & action_scope if config.restrict_to_fundamentals else mask

    def reset(keys):
        state, observation, mask = environment.reset(keys)
        distance, _, _, error = _target_geometry(state)
        zeros = jnp.zeros_like(distance, dtype=jnp.bool_)
        zero_f32 = jnp.zeros_like(distance, dtype=jnp.float32)
        return (
            CombatFundamentalsState(
                state,
                distance,
                error,
                zeros,
                zeros,
                zero_f32,
                zero_f32,
                zeros,
                _zero_terms(distance.shape),
            ),
            observation,
            scoped(mask),
        )

    def advance(state, observation, action, keys, *, detailed):
        result = environment.step_detailed(state.environment, observation, action, keys)
        next_environment, next_observation, base_reward, done, mask, info = result
        next_distance, bearing_after, yaw_after, next_error = _target_geometry(
            next_environment
        )
        _, bearing_before, yaw_before, _ = _target_geometry(state.environment)
        tracking = aim_tracking_geometry_signals(
            yaw_before,
            yaw_after,
            bearing_before,
            bearing_after,
            jnp.ones_like(next_error, dtype=jnp.bool_),
            success_degrees=config.tracking_success_degrees,
            gain_deadband_degrees=config.tracking_gain_deadband_degrees,
            harmful_turn_margin_degrees=(config.tracking_harmful_turn_margin_degrees),
        )
        runtime = _runtime(state.environment)
        next_runtime = _runtime(next_environment)

        selected_basic = action[:, _ABILITY_HEAD] == config.basic_ability_slot + 1
        accepted = info.arsenal_info.ability_accepted[:, AGENT_ENTITY]
        selected_accepted = selected_basic & accepted
        active_before = runtime.arsenal.active_ability_root_slot[:, AGENT_ENTITY] >= 0
        active_after = (
            next_runtime.arsenal.active_ability_root_slot[:, AGENT_ENTITY]
            == config.basic_ability_slot
        )
        cooldown = (
            runtime.arsenal.ability_cooldown_seconds[
                :, AGENT_ENTITY, config.basic_ability_slot
            ]
            > 1.0e-6
        )

        dealt = info.arsenal_info.entity_damage_dealt[:, AGENT_ENTITY] / target_health
        received = (
            info.arsenal_info.entity_damage_received[:, AGENT_ENTITY] / agent_health
        )
        hit_now = dealt > 1.0e-6
        open_attack = state.attack_open | selected_accepted
        attack_hit = state.attack_hit | (open_attack & hit_now)
        attack_start_distance = jnp.where(
            selected_accepted,
            state.target_distance,
            state.attack_start_distance,
        )
        attack_start_error = jnp.where(
            selected_accepted,
            state.facing_error_degrees,
            state.attack_start_error_degrees,
        )
        resolved = open_attack & ~active_after & ~done
        significantly_out_of_range = attack_start_distance > (
            reach * (1.0 + config.attack_range_margin_fraction)
            + config.attack_range_margin_blocks
        )
        significantly_misaligned = attack_start_error > (
            half_angle + config.aim_margin_degrees
        )
        missed = (
            resolved
            & ~attack_hit
            & (significantly_out_of_range | significantly_misaligned)
        )

        minimum_distance = reach * config.engagement_min_reach_fraction
        maximum_distance = reach * config.engagement_max_reach_fraction
        in_band = (next_distance >= minimum_distance) & (
            next_distance <= maximum_distance
        )
        aligned = next_error <= half_angle
        proper_start = (
            selected_accepted
            & (state.target_distance <= reach)
            & (state.facing_error_degrees <= half_angle)
        )
        yaw_delta = decode_look_delta(action[:, _YAW_HEAD], _YAW_CHOICES)
        excess_spin = jnp.where(
            tracking.unnecessary_spin
            & (jnp.abs(yaw_delta) > config.spin_tolerance_degrees),
            jnp.clip(
                (-tracking.learner_alignment_gain)
                / jnp.float32(max(config.tracking_success_degrees, 1.0)),
                0.0,
                1.0,
            ),
            jnp.float32(0.0),
        )
        outcome = environment.episode_outcome(next_environment, done)
        own_lifecycle = state.attack_open
        illegal_joint = selected_basic & ~info.action_surface_legal
        cooldown_request = selected_basic & ~illegal_joint & cooldown & ~own_lifecycle
        busy_request = (
            selected_basic
            & ~illegal_joint
            & ~cooldown_request
            & active_before
            & ~own_lifecycle
        )
        other_rejected = (
            selected_basic
            & ~selected_accepted
            & ~own_lifecycle
            & ~cooldown_request
            & ~busy_request
            & ~illegal_joint
        )
        rejected = illegal_joint | cooldown_request | busy_request | other_rejected
        new_rejection = rejected & ~state.rejected_request_latched
        illegal_joint &= new_rejection
        cooldown_request &= new_rejection
        busy_request &= new_rejection
        other_rejected &= new_rejection
        terms = CombatFundamentalsTerms(
            approach_progress=(
                config.discount * _engagement_score(next_distance, reach, config)
                - _engagement_score(state.target_distance, reach, config)
            ),
            alignment_progress=(
                jnp.clip(
                    tracking.learner_alignment_gain / jnp.float32(45.0),
                    -1.0,
                    1.0,
                )
            ),
            tracking_state=tracking.causal_alignment.astype(jnp.float32),
            engagement_state=(in_band & aligned).astype(jnp.float32),
            well_timed_attack=proper_start.astype(jnp.float32),
            damage_dealt=jnp.clip(dealt, 0.0, 1.0),
            damage_received=jnp.clip(received, 0.0, 1.0),
            blocked_hit=jnp.clip(
                info.arsenal_info.blocked_hits.astype(jnp.float32), 0.0, 1.0
            ),
            accepted_attack_cost=selected_accepted.astype(jnp.float32),
            missed_attack=missed.astype(jnp.float32),
            cooldown_request=cooldown_request.astype(jnp.float32),
            busy_request=busy_request.astype(jnp.float32),
            rejected_attack=other_rejected.astype(jnp.float32),
            illegal_joint_action=illegal_joint.astype(jnp.float32),
            excess_spin=excess_spin,
            victory=outcome.success.astype(jnp.float32),
            death=outcome.death.astype(jnp.float32),
            simultaneous_death=outcome.simultaneous.astype(jnp.float32),
        )
        reward = base_reward * config.base_reward_scale + _weighted_reward(
            terms, config
        )
        next_state = CombatFundamentalsState(
            next_environment,
            next_distance,
            next_error,
            jnp.where(resolved | done, False, open_attack),
            jnp.where(resolved | done, False, attack_hit),
            jnp.where(resolved | done, 0.0, attack_start_distance),
            jnp.where(resolved | done, 0.0, attack_start_error),
            selected_basic & (state.rejected_request_latched | rejected),
            terms,
        )
        head = (
            next_state,
            next_observation,
            jax.lax.stop_gradient(reward.astype(jnp.float32)),
            done,
            scoped(mask),
        )
        return (*head, info) if detailed else head

    def step(state, observation, action, keys):
        return advance(state, observation, action, keys, detailed=False)

    def step_detailed(state, observation, action, keys):
        return advance(state, observation, action, keys, detailed=True)

    def episode_outcome(state, done):
        return environment.episode_outcome(state.environment, done)

    return PPOEnvironment(
        reset=reset,
        step=step,
        spec=environment.spec,
        episode_outcome=episode_outcome,
        step_detailed=step_detailed,
    )


__all__ = [
    "COMBAT_ATTACK_ENVELOPE_SCHEMA",
    "COMBAT_FUNDAMENTALS_REWARD_SCHEMA",
    "CombatAttackEnvelope",
    "CombatFundamentalsConfig",
    "CombatFundamentalsState",
    "CombatFundamentalsTerms",
    "arsenal_basic_attack_envelope",
    "combat_fundamentals_training_feature",
    "combat_fundamentals_training_profile",
    "make_combat_fundamentals_environment",
]
