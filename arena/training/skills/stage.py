"""Executable reset and binding contracts for two-actor JAX skill stages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from numbers import Integral, Real
import struct
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from hytalegym.jax.combat.arsenal.runtime import reset_arsenal_batch
from hytalegym.jax.combat.arsenal.schema.types import (
    ArsenalEnvironmentState,
    ArsenalRuntimeConfig,
)
from hytalegym.jax.combat.contracts import combat_dynamics_contract_sha256
from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_ACTION_HEAD_NAMES,
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
)
from hytalegym.jax.combat.types import CombatParams
from hytalegym.jax.training.multi_actor import (
    MultiActorArenaState,
    PolicyActorAssignment,
    arsenal_runtime_config_content_sha256,
    validate_policy_actor_assignment,
)
from hytalegym.jax.world import GeometryProvider

from arena.curriculum.strategy import (
    ArenaTrainingStrategy,
    TargetMotionPattern,
)
from arena.training.selfplay.production import assignment_content_sha256
from arena.training.skills.program import (
    TargetMode,
    TargetMotionProgram,
    flee_compass_choice,
    standoff_compass_choice,
    target_motion_factors,
)


RADIAL_PAIR_PLACEMENT_SCHEMA = "arena-radial-pair-placement-v1"
SKILL_RESET_PROGRAM_SCHEMA = "arena-skill-reset-program-v2"
ARENA_SKILL_STAGE_PROGRAM_SCHEMA = "arena-skill-stage-program-v2"
ARENA_SKILL_STAGE_ARTIFACT_SCHEMA = "arena-skill-stage-artifact-v1"
BOUND_ARENA_SKILL_STAGE_IDENTITY_SCHEMA = "arena-bound-skill-stage-identity-v1"
_ACTION_SIZE = sum(ARSENAL_POLICY_ACTION_HEAD_SIZES)


class YawErrorDistribution(str, Enum):
    """Declared device-side law for sampling a reset yaw-error band."""

    UNIFORM_INTERVAL = "uniform_interval"
    SYMMETRIC_MAGNITUDE = "symmetric_magnitude"


class HealthResetLaw(str, Enum):
    """How initial physical-entity health is selected by a reset."""

    CONFIGURED_MAXIMUM = "configured_physical_entity_maximum"
    EXPLICIT_OVERRIDE = "explicit_physical_entity_override"


def _hash(value: Any) -> str:
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


def _sha256(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA-256 string")
    canonical = value.upper()
    if len(canonical) != 64 or any(
        character not in "0123456789ABCDEF" for character in canonical
    ):
        raise ValueError(f"{name} must be a SHA-256 string")
    return canonical


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _bands(
    values: tuple[tuple[float, float], ...],
    name: str,
    *,
    positive: bool = False,
) -> tuple[tuple[float, float], ...]:
    normalized = []
    for index, value in enumerate(tuple(values)):
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError(f"{name} band {index} must be a pair")
        minimum = _finite_float(value[0], f"{name} minimum")
        maximum = _finite_float(value[1], f"{name} maximum")
        if minimum > maximum:
            raise ValueError(f"{name} band minimum cannot exceed maximum")
        if positive and minimum <= 0.0:
            raise ValueError(f"{name} bands must be strictly positive")
        normalized.append((minimum, maximum))
    if not normalized:
        raise ValueError(f"{name} bands cannot be empty")
    return tuple(normalized)


def _batched_key_count(keys: jax.Array) -> int:
    key_data = jax.random.key_data(keys)
    if key_data.ndim != 2 or key_data.shape[1] < 1:
        raise ValueError("skill reset keys must be a batch of JAX PRNG keys")
    if key_data.shape[0] < 1:
        raise ValueError("skill reset key batch cannot be empty")
    return key_data.shape[0]


def _uniform01(keys: jax.Array, domain: int) -> jax.Array:
    folded = jax.vmap(lambda key: jax.random.fold_in(key, domain))(keys)
    return jax.vmap(lambda key: jax.random.uniform(key, (), dtype=jnp.float32))(folded)


@dataclass(frozen=True, slots=True)
class RadialPairPlacementProgram:
    """Lane-stratified physical pair placement on an unobstructed plane."""

    distance_bands: tuple[tuple[float, float], ...]
    center_xz: tuple[float, float] = (0.0, 0.0)
    spawn_y: float = 0.0
    schema: str = RADIAL_PAIR_PLACEMENT_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "distance_bands",
            _bands(self.distance_bands, "radial distance", positive=True),
        )
        center = tuple(self.center_xz)
        if len(center) != 2:
            raise ValueError("radial placement center_xz must have two values")
        object.__setattr__(
            self,
            "center_xz",
            tuple(_finite_float(value, "radial placement center") for value in center),
        )
        object.__setattr__(
            self,
            "spawn_y",
            _finite_float(self.spawn_y, "radial placement spawn_y"),
        )
        if self.schema != RADIAL_PAIR_PLACEMENT_SCHEMA:
            raise ValueError("radial pair placement schema is not current")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "distance_bands": self.distance_bands,
            "center_xz": self.center_xz,
            "spawn_y": self.spawn_y,
            "lane_schedule": "lane_index_modulo_ordered_distance_bands",
            "angle_distribution": "keyed_uniform_negative_pi_to_pi",
            "pair_layout": "physical_actor0_and_actor1_symmetric_about_center",
            "key_domains": {"distance": 0, "angle": 1},
        }

    @property
    def contract_sha256(self) -> str:
        return _hash(self.manifest())


class PairPlacement(NamedTuple):
    """Physical two-entity coordinates selected for one reset batch."""

    entity_position: jax.Array
    separation: jax.Array
    band_index: jax.Array
    valid: jax.Array


def radial_pair_placement(
    program: RadialPairPlacementProgram,
    keys: jax.Array,
) -> PairPlacement:
    """Sample symmetric physical positions without leaving the JAX device."""

    if not isinstance(program, RadialPairPlacementProgram):
        raise TypeError("radial placement needs a RadialPairPlacementProgram")
    batch = _batched_key_count(keys)
    lanes = jnp.arange(batch, dtype=jnp.int32)
    band_index = lanes % len(program.distance_bands)
    bands = jnp.asarray(program.distance_bands, dtype=jnp.float32)
    selected = bands[band_index]
    separation = selected[:, 0] + _uniform01(keys, 0) * (
        selected[:, 1] - selected[:, 0]
    )
    angle = -jnp.pi + jnp.float32(2.0 * math.pi) * _uniform01(keys, 1)
    axis = jnp.stack(
        (jnp.sin(angle), jnp.zeros((batch,), dtype=jnp.float32), -jnp.cos(angle)),
        axis=-1,
    )
    center = jnp.broadcast_to(
        jnp.asarray(
            (program.center_xz[0], program.spawn_y, program.center_xz[1]),
            dtype=jnp.float32,
        ),
        (batch, 3),
    )
    half_delta = jnp.float32(0.5) * separation[:, None] * axis
    return PairPlacement(
        entity_position=jnp.stack((center - half_delta, center + half_delta), axis=1),
        separation=separation,
        band_index=band_index,
        valid=jnp.ones((batch,), dtype=jnp.bool_),
    )


@dataclass(frozen=True, slots=True)
class SkillResetProgram:
    """Content identity and facing distribution for a physical pair reset."""

    placement_contract_sha256: str
    learner_yaw_error_bands: tuple[tuple[float, float], ...]
    target_yaw_error_bands: tuple[tuple[float, float], ...] = ((0.0, 0.0),)
    entity_health: tuple[float, float] | None = None
    health_reset_law: HealthResetLaw = HealthResetLaw.CONFIGURED_MAXIMUM
    learner_yaw_error_distribution: YawErrorDistribution = (
        YawErrorDistribution.UNIFORM_INTERVAL
    )
    target_yaw_error_distribution: YawErrorDistribution = (
        YawErrorDistribution.UNIFORM_INTERVAL
    )
    schema: str = SKILL_RESET_PROGRAM_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "placement_contract_sha256",
            _sha256(self.placement_contract_sha256, "placement contract"),
        )
        learner_bands = _bands(self.learner_yaw_error_bands, "learner yaw error")
        target_bands = _bands(self.target_yaw_error_bands, "target yaw error")
        if any(
            minimum < -180.0 or maximum > 180.0
            for minimum, maximum in learner_bands + target_bands
        ):
            raise ValueError("yaw error bands must stay inside [-180,180]")
        object.__setattr__(self, "learner_yaw_error_bands", learner_bands)
        object.__setattr__(self, "target_yaw_error_bands", target_bands)
        for owner in ("learner", "target"):
            field = f"{owner}_yaw_error_distribution"
            try:
                distribution = YawErrorDistribution(getattr(self, field))
            except ValueError as error:
                raise ValueError(
                    f"{owner} yaw error distribution is unsupported"
                ) from error
            bands = learner_bands if owner == "learner" else target_bands
            if distribution is YawErrorDistribution.SYMMETRIC_MAGNITUDE and any(
                minimum < 0.0 for minimum, _ in bands
            ):
                raise ValueError(
                    f"{owner} symmetric-magnitude yaw bands must be nonnegative"
                )
            object.__setattr__(self, field, distribution)
        try:
            health_law = HealthResetLaw(self.health_reset_law)
        except ValueError as error:
            raise ValueError("skill reset health law is unsupported") from error
        object.__setattr__(self, "health_reset_law", health_law)
        if health_law is HealthResetLaw.CONFIGURED_MAXIMUM:
            if self.entity_health is not None:
                raise ValueError(
                    "configured-maximum health cannot declare an explicit override"
                )
        else:
            if self.entity_health is None:
                raise ValueError("explicit health law requires two override values")
            health = tuple(self.entity_health)
            if len(health) != 2:
                raise ValueError("skill reset must declare exactly two health values")
            normalized_health = tuple(
                _finite_float(value, "skill reset health") for value in health
            )
            if any(value <= 0.0 for value in normalized_health):
                raise ValueError("skill reset health must be positive")
            object.__setattr__(self, "entity_health", normalized_health)
        if self.schema != SKILL_RESET_PROGRAM_SCHEMA:
            raise ValueError("skill reset program schema is not current")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "placement_contract_sha256": self.placement_contract_sha256,
            "learner_yaw_error_bands": self.learner_yaw_error_bands,
            "target_yaw_error_bands": self.target_yaw_error_bands,
            "learner_yaw_error_distribution": (
                self.learner_yaw_error_distribution.value
            ),
            "target_yaw_error_distribution": (self.target_yaw_error_distribution.value),
            "health_reset": {
                "law": self.health_reset_law.value,
                "explicit_physical_entity_override": self.entity_health,
                "configured_sources": (
                    "CombatParams.agent_max_health",
                    "CombatParams.target_max_health",
                ),
                "override_above_configured_maximum": (
                    "clamp_to_configured_maximum_and_mark_health_invalid"
                ),
            },
            "entity_order": "physical_actor0_then_actor1_never_role_swapped",
            "facing": "both_actors_face_peer_then_role_owned_yaw_error",
            "key_domains": {
                "learner_yaw_error_magnitude_or_interval": 2,
                "target_yaw_error_magnitude_or_interval": 3,
                "learner_yaw_error_sign": 4,
                "target_yaw_error_sign": 5,
            },
            "reset_kernel": "reset_arsenal_batch_explicit_entity_state",
        }

    @property
    def contract_sha256(self) -> str:
        return _hash(self.manifest())


class SkillDuelRoles(NamedTuple):
    """One trainable learner and one frozen target in two policy slots."""

    learner_policy_slot: jax.Array
    target_policy_slot: jax.Array
    learner_actor_index: jax.Array
    target_actor_index: jax.Array


def skill_duel_roles(assignment: PolicyActorAssignment) -> SkillDuelRoles:
    """Validate and derive exact learner/target ownership outside the JIT."""

    validate_policy_actor_assignment(assignment, entity_count=2)
    if assignment.actor_index.ndim != 2 or assignment.actor_index.shape[1] != 2:
        raise ValueError("skill stages require exactly two policy slots")
    active = np.asarray(jax.device_get(assignment.active), dtype=np.bool_)
    trainable = np.asarray(jax.device_get(assignment.trainable), dtype=np.bool_)
    actor = np.asarray(jax.device_get(assignment.actor_index), dtype=np.int32)
    policy = np.asarray(jax.device_get(assignment.policy_id), dtype=np.int32)
    if not np.all(active):
        raise ValueError("skill stages require two active actors in every lane")
    if np.any(np.sum(trainable, axis=1) != 1):
        raise ValueError("skill stages require exactly one trainable learner per lane")
    if np.any(np.sort(actor, axis=1) != np.asarray((0, 1), dtype=np.int32)):
        raise ValueError("skill stages require physical actors zero and one")
    learner_slot = np.argmax(trainable, axis=1).astype(np.int32)
    target_slot = 1 - learner_slot
    lanes = np.arange(actor.shape[0], dtype=np.int32)
    if np.any(policy[lanes, learner_slot] == policy[lanes, target_slot]):
        raise ValueError("a frozen target cannot share the learner policy row")
    return SkillDuelRoles(
        learner_policy_slot=jnp.asarray(learner_slot, dtype=jnp.int32),
        target_policy_slot=jnp.asarray(target_slot, dtype=jnp.int32),
        learner_actor_index=jnp.asarray(actor[lanes, learner_slot], dtype=jnp.int32),
        target_actor_index=jnp.asarray(actor[lanes, target_slot], dtype=jnp.int32),
    )


class SkillResetOutput(NamedTuple):
    """Reset state plus lane-local role and placement diagnostics."""

    state: ArsenalEnvironmentState
    placement_valid: jax.Array
    health_valid: jax.Array
    learner_actor_index: jax.Array
    target_actor_index: jax.Array
    separation: jax.Array
    band_index: jax.Array


def _sample_banded_value(
    bands: tuple[tuple[float, float], ...],
    keys: jax.Array,
    domain: int,
    distribution: YawErrorDistribution,
) -> jax.Array:
    batch = _batched_key_count(keys)
    selected = jnp.asarray(bands, dtype=jnp.float32)[
        jnp.arange(batch, dtype=jnp.int32) % len(bands)
    ]
    value = selected[:, 0] + _uniform01(keys, domain) * (
        selected[:, 1] - selected[:, 0]
    )
    if distribution is YawErrorDistribution.SYMMETRIC_MAGNITUDE:
        sign = jnp.where(
            _uniform01(keys, domain + 2) < jnp.float32(0.5),
            jnp.float32(-1.0),
            jnp.float32(1.0),
        )
        value = sign * value
    return value


def reset_skill_duel(
    program: SkillResetProgram,
    keys: jax.Array,
    placement: PairPlacement,
    roles: SkillDuelRoles,
    params: CombatParams,
    config: ArsenalRuntimeConfig,
    *,
    geometry: GeometryProvider | None = None,
) -> SkillResetOutput:
    """Reset a physical two-entity duel through the canonical Arsenal kernel."""

    if not isinstance(program, SkillResetProgram):
        raise TypeError("skill reset needs a SkillResetProgram")
    if not isinstance(placement, PairPlacement):
        raise TypeError("skill reset needs a PairPlacement")
    if not isinstance(roles, SkillDuelRoles):
        raise TypeError("skill reset needs validated SkillDuelRoles")
    batch = _batched_key_count(keys)
    if placement.entity_position.shape != (batch, 2, 3):
        raise ValueError("skill placement entity_position must have shape [B,2,3]")
    if placement.separation.shape != (batch,):
        raise ValueError("skill placement separation must have shape [B]")
    if placement.band_index.shape != (batch,):
        raise ValueError("skill placement band_index must have shape [B]")
    if placement.valid.shape != (batch,):
        raise ValueError("skill placement valid must have shape [B]")
    for name, value in zip(SkillDuelRoles._fields, roles, strict=True):
        if value.shape != (batch,) or value.dtype != jnp.dtype(jnp.int32):
            raise ValueError(f"skill duel role {name} must be int32[B]")
    if config.loadout.weapon_id.shape != (batch, 2):
        raise ValueError("skill reset runtime config must have shape [B,2]")

    position = jnp.asarray(placement.entity_position, dtype=jnp.float32)
    direction = position[:, 1] - position[:, 0]
    actor0_yaw = jnp.degrees(jnp.arctan2(-direction[:, 0], -direction[:, 2]))
    base_yaw = jnp.stack((actor0_yaw, actor0_yaw + 180.0), axis=1)
    learner_error = _sample_banded_value(
        program.learner_yaw_error_bands,
        keys,
        2,
        program.learner_yaw_error_distribution,
    )
    target_error = _sample_banded_value(
        program.target_yaw_error_bands,
        keys,
        3,
        program.target_yaw_error_distribution,
    )
    lanes = jnp.arange(batch, dtype=jnp.int32)
    yaw_error = jnp.zeros((batch, 2), dtype=jnp.float32)
    yaw_error = yaw_error.at[lanes, roles.learner_actor_index].set(learner_error)
    yaw_error = yaw_error.at[lanes, roles.target_actor_index].set(target_error)
    entity_yaw = jnp.mod(base_yaw + yaw_error, jnp.float32(360.0))
    configured_maximum = jnp.broadcast_to(
        jnp.stack(
            (
                jnp.asarray(params.agent_max_health, dtype=jnp.float32),
                jnp.asarray(params.target_max_health, dtype=jnp.float32),
            )
        )[None, :],
        (batch, 2),
    )
    if program.health_reset_law is HealthResetLaw.CONFIGURED_MAXIMUM:
        entity_health = configured_maximum
        health_valid = jnp.ones((batch,), dtype=jnp.bool_)
    else:
        requested_health = jnp.broadcast_to(
            jnp.asarray(program.entity_health, dtype=jnp.float32), (batch, 2)
        )
        health_valid = jnp.all(requested_health <= configured_maximum, axis=1)
        entity_health = jnp.minimum(requested_health, configured_maximum)
    state, _ = reset_arsenal_batch(
        keys,
        params,
        config,
        geometry,
        entity_position=position,
        entity_health=entity_health,
        entity_yaw=entity_yaw,
    )
    return SkillResetOutput(
        state=state,
        placement_valid=jnp.asarray(placement.valid, dtype=jnp.bool_),
        health_valid=health_valid,
        learner_actor_index=roles.learner_actor_index,
        target_actor_index=roles.target_actor_index,
        separation=jnp.asarray(placement.separation, dtype=jnp.float32),
        band_index=jnp.asarray(placement.band_index, dtype=jnp.int32),
    )


def reset_radial_skill_duel(
    placement_program: RadialPairPlacementProgram,
    reset_program: SkillResetProgram,
    keys: jax.Array,
    roles: SkillDuelRoles,
    params: CombatParams,
    config: ArsenalRuntimeConfig,
    *,
    geometry: GeometryProvider | None = None,
) -> SkillResetOutput:
    """Compose the content-matched radial placement and Arsenal reset kernels."""

    if placement_program.contract_sha256 != reset_program.placement_contract_sha256:
        raise ValueError("skill reset placement contract does not match its program")
    return reset_skill_duel(
        reset_program,
        keys,
        radial_pair_placement(placement_program, keys),
        roles,
        params,
        config,
        geometry=geometry,
    )


def _represented_motion_patterns(
    program: TargetMotionProgram,
) -> tuple[TargetMotionPattern, ...]:
    mapping = {
        TargetMode.STATIONARY: TargetMotionPattern.STATIONARY,
        TargetMode.STRAFE: TargetMotionPattern.STRAFE,
        TargetMode.APPROACH: TargetMotionPattern.RADIAL,
        TargetMode.APPROACH_THEN_HOLD: TargetMotionPattern.RADIAL,
        TargetMode.FLEE: TargetMotionPattern.RADIAL,
        TargetMode.FLEE_WEAVE: TargetMotionPattern.MIXED_POLICY,
        TargetMode.FROZEN_POLICY: TargetMotionPattern.MIXED_POLICY,
    }
    return tuple(
        sorted(
            {mapping[mode] for mode in program.lane_modes},
            key=lambda value: value.value,
        )
    )


@dataclass(frozen=True, slots=True)
class ArenaSkillStageProgram:
    """Reusable strategy bound to exact executable target and reset programs."""

    strategy: ArenaTrainingStrategy
    target_motion: TargetMotionProgram
    reset: SkillResetProgram
    trainable_heads: tuple[str, ...] = ARSENAL_POLICY_ACTION_HEAD_NAMES
    frozen_scope: str = "all_policy_parameters_except_selected_actor_head_columns"
    knockback: bool = True
    teacher: str | None = None
    teacher_anneal_gate: str | None = None
    schema: str = ARENA_SKILL_STAGE_PROGRAM_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.strategy, ArenaTrainingStrategy):
            raise TypeError("skill stage strategy must be an ArenaTrainingStrategy")
        if not isinstance(self.target_motion, TargetMotionProgram):
            raise TypeError("skill stage target motion must be a TargetMotionProgram")
        if not isinstance(self.reset, SkillResetProgram):
            raise TypeError("skill stage reset must be a SkillResetProgram")
        heads = tuple(self.trainable_heads)
        if not heads or len(heads) != len(set(heads)):
            raise ValueError("skill stage trainable heads must be non-empty and unique")
        unknown = sorted(set(heads) - set(ARSENAL_POLICY_ACTION_HEAD_NAMES))
        if unknown:
            raise ValueError("unknown trainable action head(s): " + ", ".join(unknown))
        object.__setattr__(self, "trainable_heads", heads)
        if self.frozen_scope != (
            "all_policy_parameters_except_selected_actor_head_columns"
        ):
            raise ValueError("skill stage frozen scope is unsupported")
        if not isinstance(self.knockback, bool):
            raise TypeError("skill stage knockback flag must be boolean")
        if (self.teacher is None) != (self.teacher_anneal_gate is None):
            raise ValueError("skill stage teacher and anneal gate must be declared together")
        for name in ("teacher", "teacher_anneal_gate"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value or value != value.strip()
            ):
                raise ValueError(f"skill stage {name} must be a canonical label")
        represented = _represented_motion_patterns(self.target_motion)
        declared = tuple(
            sorted(self.strategy.target_motion.patterns, key=lambda value: value.value)
        )
        if represented != declared:
            raise ValueError(
                "executable target modes must exactly match declared target-motion patterns"
            )
        if self.schema != ARENA_SKILL_STAGE_PROGRAM_SCHEMA:
            raise ValueError("Arena skill stage program schema is not current")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "strategy": self.strategy.manifest(),
            "strategy_contract_sha256": self.strategy.contract_sha256,
            "target_motion": self.target_motion.manifest(),
            "target_motion_contract_sha256": self.target_motion.contract_sha256,
            "reset": self.reset.manifest(),
            "reset_contract_sha256": self.reset.contract_sha256,
            "duel_roles": "one_trainable_learner_one_frozen_target_two_entities",
            "trainable_heads": self.trainable_heads,
            "frozen_scope": self.frozen_scope,
            "opponent_mode": tuple(
                mode.name.lower() for mode in self.target_motion.lane_modes
            ),
            "knockback": self.knockback,
            "episode_ticks": self.strategy.episode.maximum_ticks,
            "no_progress_patience_ticks": (
                self.strategy.episode.no_progress_patience_ticks
            ),
            "success_contracts": tuple(
                objective.success_contract for objective in self.strategy.objectives
            ),
            "reward_laws": tuple(
                objective.reward_law for objective in self.strategy.objectives
            ),
            "teacher": self.teacher,
            "teacher_anneal_gate": self.teacher_anneal_gate,
        }

    @property
    def contract_sha256(self) -> str:
        return _hash(self.manifest())


def skill_stage_parameter_mask(program: ArenaSkillStageProgram, parameters: Any) -> Any:
    """Select only actor output columns owned by the stage's action heads."""

    if not isinstance(program, ArenaSkillStageProgram):
        raise TypeError("parameter scope needs an ArenaSkillStageProgram")
    mask = jax.tree_util.tree_map(lambda value: jnp.zeros_like(value, dtype=bool), parameters)
    actor = getattr(parameters, "actor", None)
    if actor is None or actor.kernel.shape[-1] != _ACTION_SIZE or actor.bias.shape != (
        _ACTION_SIZE,
    ):
        raise ValueError("skill stage needs a policy actor with the published action width")
    selected = jnp.zeros((_ACTION_SIZE,), dtype=bool)
    offset = 0
    for name, size in zip(
        ARSENAL_POLICY_ACTION_HEAD_NAMES,
        ARSENAL_POLICY_ACTION_HEAD_SIZES,
        strict=True,
    ):
        if name in program.trainable_heads:
            selected = selected.at[offset : offset + size].set(True)
        offset += size
    return mask._replace(
        actor=mask.actor._replace(
            kernel=jnp.broadcast_to(selected, actor.kernel.shape),
            bias=selected,
        )
    )


def skill_stage_artifact(
    program: ArenaSkillStageProgram, initial_parameters: Any, final_parameters: Any,
) -> dict[str, Any]:
    """Record exact frozen/trainable element integrity for one stage run."""

    mask = skill_stage_parameter_mask(program, initial_parameters)
    initial, initial_tree = jax.tree_util.tree_flatten(initial_parameters)
    final, final_tree = jax.tree_util.tree_flatten(final_parameters)
    selected, selected_tree = jax.tree_util.tree_flatten(mask)
    if initial_tree != final_tree or initial_tree != selected_tree:
        raise ValueError("skill stage parameter trees do not match")
    frozen = trainable = changed_trainable = 0
    for before, after, scope in zip(initial, final, selected, strict=True):
        before_value = np.asarray(before)
        after_value = np.asarray(after)
        scope_value = np.asarray(scope, dtype=np.bool_)
        if before_value.shape != after_value.shape or before_value.shape != scope_value.shape:
            raise ValueError("skill stage parameter leaf shapes do not match")
        changed = before_value != after_value
        if bool(np.any(changed & ~scope_value)):
            raise ValueError("skill stage changed a frozen parameter")
        frozen += int(np.sum(~scope_value))
        trainable += int(np.sum(scope_value))
        changed_trainable += int(np.sum(changed & scope_value))
    return {
        "schema": ARENA_SKILL_STAGE_ARTIFACT_SCHEMA,
        "stage": program.manifest(),
        "stage_contract_sha256": program.contract_sha256,
        "frozen_parameter_identity_exact": True,
        "frozen_elements": frozen,
        "trainable_elements": trainable,
        "changed_trainable_elements": changed_trainable,
    }


@dataclass(frozen=True, slots=True)
class BoundArenaSkillStageIdentity:
    """Content identity of one reusable stage bound to concrete runtime data."""

    program_sha256: str
    action_contract_sha256: str
    combat_dynamics_sha256: str
    combat_params_sha256: str
    runtime_config_sha256: str
    assignment_sha256: str
    world_sha256: str
    batch_size: int
    entity_count: int = 2
    policy_slots: int = 2
    schema: str = BOUND_ARENA_SKILL_STAGE_IDENTITY_SCHEMA

    def __post_init__(self) -> None:
        for field in (
            "program_sha256",
            "action_contract_sha256",
            "combat_dynamics_sha256",
            "combat_params_sha256",
            "runtime_config_sha256",
            "assignment_sha256",
            "world_sha256",
        ):
            object.__setattr__(self, field, _sha256(getattr(self, field), field))
        for field in ("batch_size", "entity_count", "policy_slots"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{field} must be an integer")
            object.__setattr__(self, field, int(value))
        if self.batch_size < 1:
            raise ValueError("bound skill stage batch size must be positive")
        if self.entity_count != 2 or self.policy_slots != 2:
            raise ValueError(
                "bound skill stages require exactly two entities and slots"
            )
        if self.schema != BOUND_ARENA_SKILL_STAGE_IDENTITY_SCHEMA:
            raise ValueError("bound Arena skill stage identity schema is not current")

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "program_sha256": self.program_sha256,
            "action_contract_sha256": self.action_contract_sha256,
            "combat_dynamics_sha256": self.combat_dynamics_sha256,
            "combat_params_sha256": self.combat_params_sha256,
            "runtime_config_sha256": self.runtime_config_sha256,
            "assignment_sha256": self.assignment_sha256,
            "world_sha256": self.world_sha256,
            "batch_size": self.batch_size,
            "entity_count": self.entity_count,
            "policy_slots": self.policy_slots,
        }

    @property
    def contract_sha256(self) -> str:
        return _hash(self.manifest())


def combat_params_content_sha256(params: CombatParams) -> str:
    """Hash the exact named numeric Combat parameter leaves used by reset."""

    if not isinstance(params, CombatParams):
        raise TypeError("combat params identity needs CombatParams")
    digest = hashlib.sha256(b"ARENA_SKILL_STAGE_COMBAT_PARAMS\0")
    for name, leaf in zip(params._fields, params, strict=True):
        value = np.asarray(jax.device_get(leaf))
        canonical = np.ascontiguousarray(
            value.astype(
                value.dtype
                if value.dtype.byteorder == "|"
                else value.dtype.newbyteorder("<"),
                copy=False,
            )
        )
        encoded = name.encode("ascii")
        digest.update(struct.pack(">I", len(encoded)))
        digest.update(encoded)
        dtype = canonical.dtype.str.encode("ascii")
        digest.update(struct.pack(">I", len(dtype)))
        digest.update(dtype)
        digest.update(struct.pack(">I", canonical.ndim))
        for dimension in canonical.shape:
            digest.update(struct.pack(">Q", int(dimension)))
        raw = canonical.tobytes(order="C")
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest().upper()


def bind_arena_skill_stage_identity(
    program: ArenaSkillStageProgram,
    params: CombatParams,
    config: ArsenalRuntimeConfig,
    assignment: PolicyActorAssignment,
    *,
    action_contract_sha256: str,
    world_sha256: str,
) -> BoundArenaSkillStageIdentity:
    """Validate two-role semantics and bind every concrete runtime identity."""

    if not isinstance(program, ArenaSkillStageProgram):
        raise TypeError("bound skill stage needs an ArenaSkillStageProgram")
    roles = skill_duel_roles(assignment)
    batch = roles.learner_actor_index.shape[0]
    if config.loadout.weapon_id.shape != (batch, 2):
        raise ValueError("bound skill stage runtime config must have shape [B,2]")
    return BoundArenaSkillStageIdentity(
        program_sha256=program.contract_sha256,
        action_contract_sha256=action_contract_sha256,
        combat_dynamics_sha256=combat_dynamics_contract_sha256(),
        combat_params_sha256=combat_params_content_sha256(params),
        runtime_config_sha256=arsenal_runtime_config_content_sha256(config),
        assignment_sha256=assignment_content_sha256(assignment),
        world_sha256=world_sha256,
        batch_size=batch,
    )


class ScriptedTargetActionMask(NamedTuple):
    """One exact pre-sampling target mask and its support diagnostics."""

    action_mask: jax.Array
    requested_factors: jax.Array
    issued_factors: jax.Array
    mode: jax.Array
    supported: jax.Array


def _one_hot_factor_rows(factors: jax.Array) -> jax.Array:
    parts = []
    for index, size in enumerate(ARSENAL_POLICY_ACTION_HEAD_SIZES):
        parts.append(
            jnp.arange(size, dtype=jnp.int32)[None, :] == factors[:, index, None]
        )
    return jnp.concatenate(parts, axis=-1)


def scripted_target_action_mask(
    program: TargetMotionProgram,
    roles: SkillDuelRoles,
    arena: MultiActorArenaState,
    observation: Any,
    base_mask: jax.Array,
) -> ScriptedTargetActionMask:
    """Force only the scripted target slot before policy-bank sampling."""

    if not isinstance(program, TargetMotionProgram):
        raise TypeError("scripted target mask needs a TargetMotionProgram")
    if TargetMode.FROZEN_POLICY in program.lane_modes:
        raise ValueError("frozen-policy lanes must be sampled through their scope")
    if not isinstance(roles, SkillDuelRoles):
        raise TypeError("scripted target mask needs validated SkillDuelRoles")
    if not isinstance(arena, MultiActorArenaState):
        raise TypeError("scripted target mask needs a MultiActorArenaState")
    observation_mask = getattr(observation, "action_mask", None)
    if observation_mask is None:
        raise TypeError("scripted target observation must expose action_mask")
    mask = jnp.asarray(base_mask)
    batch = roles.target_policy_slot.shape[0]
    if mask.dtype != jnp.dtype(jnp.bool_) or mask.shape != (batch, 2, _ACTION_SIZE):
        raise ValueError(f"scripted target base mask must be bool[B,2,{_ACTION_SIZE}]")
    if observation_mask.shape != mask.shape:
        raise ValueError("scripted target observation and base masks differ in shape")
    if arena.arsenal.combat.tick_count.shape != (batch,):
        raise ValueError("scripted target arena tick must have shape [B]")
    lanes = jnp.arange(batch, dtype=jnp.int32)
    target_mask = mask[lanes, roles.target_policy_slot]
    position = arena.arsenal.combat.position
    target_position = position[lanes, roles.target_actor_index]
    learner_position = position[lanes, roles.learner_actor_index]
    horizontal_distance = jnp.linalg.norm(
        (target_position - learner_position)[:, (0, 2)], axis=-1
    )
    flee_direction = None
    # Body-relative, so these need the target's own body yaw rather than its
    # head yaw -- travel rides the body and the two are separate here.
    target_body_yaw = arena.arsenal.combat.yaw[lanes, roles.target_actor_index]
    if TargetMode.FLEE_ADAPTIVE in program.lane_modes:
        flee_direction = flee_compass_choice(
            learner_position[:, (0, 2)],
            target_position[:, (0, 2)],
            target_body_yaw,
            arena_radius=program.adaptive_arena_radius,
        )
    if TargetMode.FLEE_STANDOFF in program.lane_modes:
        tick_value = arena.arsenal.combat.tick_count.astype(jnp.int32)
        # Lane parity offsets the reversal so the batch does not all swap orbit
        # direction on the same tick, matching the gait cycle's own offset.
        orbit = (
            tick_value // jnp.int32(program.standoff_strafe_period_ticks) + lanes
        ) & jnp.int32(1)
        standoff_direction = standoff_compass_choice(
            learner_position[:, (0, 2)],
            target_position[:, (0, 2)],
            target_body_yaw,
            jnp.where(orbit == 0, jnp.float32(1.0), jnp.float32(-1.0)),
            inner_distance=program.standoff_inner_distance,
            outer_distance=program.standoff_outer_distance,
            arena_radius=program.adaptive_arena_radius,
        )
        # Lanes are exclusive, so whichever mode owns a lane owns its heading;
        # combining them here keeps a mixed-mode program on one direction row.
        # Lane assignment mirrors `target_motion_factors`, which is the single
        # definition of which lane runs which mode.
        if flee_direction is not None:
            declared = jnp.asarray(
                tuple(int(mode) for mode in program.lane_modes), dtype=jnp.int32
            )
            lane_mode = declared[lanes % len(program.lane_modes)]
            standoff_direction = jnp.where(
                lane_mode == jnp.int32(TargetMode.FLEE_STANDOFF),
                standoff_direction,
                flee_direction,
            )
        flee_direction = standoff_direction
    motion = target_motion_factors(
        program,
        arena.arsenal.combat.tick_count,
        target_mask,
        horizontal_distance=horizontal_distance,
        flee_direction=flee_direction,
    )
    forced = _one_hot_factor_rows(motion.factors) & target_mask
    resolved = mask.at[lanes, roles.target_policy_slot].set(forced)
    return ScriptedTargetActionMask(
        action_mask=resolved,
        requested_factors=motion.requested_factors,
        issued_factors=motion.factors,
        mode=motion.mode,
        supported=motion.requested_legal & motion.issued_legal,
    )


__all__ = [
    "ARENA_SKILL_STAGE_ARTIFACT_SCHEMA",
    "ARENA_SKILL_STAGE_PROGRAM_SCHEMA",
    "BOUND_ARENA_SKILL_STAGE_IDENTITY_SCHEMA",
    "RADIAL_PAIR_PLACEMENT_SCHEMA",
    "SKILL_RESET_PROGRAM_SCHEMA",
    "ArenaSkillStageProgram",
    "BoundArenaSkillStageIdentity",
    "HealthResetLaw",
    "PairPlacement",
    "RadialPairPlacementProgram",
    "ScriptedTargetActionMask",
    "SkillDuelRoles",
    "SkillResetOutput",
    "SkillResetProgram",
    "YawErrorDistribution",
    "bind_arena_skill_stage_identity",
    "combat_params_content_sha256",
    "radial_pair_placement",
    "reset_radial_skill_duel",
    "reset_skill_duel",
    "scripted_target_action_mask",
    "skill_stage_artifact",
    "skill_stage_parameter_mask",
    "skill_duel_roles",
]
