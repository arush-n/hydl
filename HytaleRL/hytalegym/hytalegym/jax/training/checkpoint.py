"""Host-side portable policy checkpoints for training and native inference."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import jax.numpy as jnp
import numpy as np

from hytalegym.jax.combat.arsenal.schema.contract import (
    ABILITY_CAPACITY,
    ARSENAL_SCHEMA,
    ARSENAL_VERSION,
    EVENT_CAPACITY,
    OBSERVATION_CAPACITY,
)
from hytalegym.jax.combat.arsenal.profiles import (
    hytale_0_5_7_runtime_capacity,
)
from hytalegym.jax.combat.arsenal.schema.spec import (
    combat_arsenal_contract_sha256,
)
from hytalegym.jax.combat.arsenal.schema.types import ArsenalRuntimeCapacity
from hytalegym.jax.combat.mechanics.schema.contract import (
    COMBAT_MECHANICS_SCHEMA,
    COMBAT_MECHANICS_VERSION,
)
from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_ACTION_SIZE,
    ARSENAL_POLICY_SCHEMA,
    arsenal_policy_contract_sha256,
    arsenal_policy_observation_size,
)
from hytalegym.jax.combat.observation.v3.tokens.world import (
    WorldGeometryPolicyConfig,
    normalize_world_geometry_policy_config,
    world_geometry_policy_config_manifest,
)
from hytalegym.jax.combat.contracts.dynamics import (
    combat_dynamics_contract_sha256,
)
from hytalegym.jax.combat.contracts.policy import (
    active_combat_observation_contract_sha256,
    combat_skill_action_contract_sha256,
)
from hytalegym.jax.combat.types import ACTIVE_OBSERVATION_SIZE, ENTITY_COUNT
from hytalegym.jax.combat.skills import SKILL_COUNT
from hytalegym.jax.training.types import (
    DenseParams,
    GRUParams,
    PPOConfig,
    RecurrentPolicyParams,
)
from hytalegym.rulesets import (
    COMBAT_RULESET_RESOURCE,
    combat_ruleset_sha256,
    load_combat_ruleset,
)
from hytalegym.jax.world import world_geometry_token_contract_sha256


CHECKPOINT_FORMAT_VERSION = 1
COMBAT_OBSERVATION_SCHEMA = f"active_combat_{ACTIVE_OBSERVATION_SIZE}"
COMBAT_ACTION_SCHEMA = f"combat_skills_{SKILL_COUNT}"
ACTIVE_COMBAT_POLICY_SURFACE = "active_combat"
ARSENAL_POLICY_SURFACE = "arsenal"


def current_combat_checkpoint_contract(
    *,
    entity_count: int = ENTITY_COUNT,
    arsenal_runtime_capacity: (
        ArsenalRuntimeCapacity | Mapping[str, int] | None
    ) = None,
    policy_surface: str = ACTIVE_COMBAT_POLICY_SURFACE,
    world_geometry_config: (
        WorldGeometryPolicyConfig | Mapping[str, object] | None
    ) = None,
) -> dict[str, Any]:
    """Return the exact environment identity required for policy transfer."""

    ruleset = load_combat_ruleset()
    runtime_capacity = _normalize_arsenal_runtime_capacity(arsenal_runtime_capacity)
    token_config = normalize_world_geometry_policy_config(world_geometry_config)
    (
        observation_schema,
        observation_size,
        observation_sha256,
        action_schema,
        action_size,
        action_sha256,
    ) = _policy_contract(
        policy_surface,
        token_config,
        entity_count=entity_count,
    )
    contract = {
        "hytale_version": ruleset["hytale_server_version"],
        "combat_model_version": ruleset["version"],
        "combat_mechanics_schema": COMBAT_MECHANICS_SCHEMA,
        "combat_mechanics_version": COMBAT_MECHANICS_VERSION,
        "arsenal_schema": ARSENAL_SCHEMA,
        "arsenal_version": ARSENAL_VERSION,
        "arsenal_contract_sha256": combat_arsenal_contract_sha256(
            entity_count=entity_count,
        ),
        "arsenal_runtime_capacity": runtime_capacity,
        "combat_ruleset_resource": COMBAT_RULESET_RESOURCE,
        # Deliberately fail closed on the exact resource bytes. The numeric
        # ruleset version is descriptive, not a substitute for identity.
        "combat_ruleset_sha256": combat_ruleset_sha256(),
        "combat_dynamics_contract_sha256": combat_dynamics_contract_sha256(),
        "agent_role": ruleset["matchup"]["agent_role"],
        "target_role": ruleset["matchup"]["target_role"],
        "policy_surface": policy_surface,
        "observation": observation_schema,
        "observation_size": observation_size,
        "observation_contract_sha256": observation_sha256,
        "action": action_schema,
        "action_size": action_size,
        "action_contract_sha256": action_sha256,
    }
    if policy_surface == ARSENAL_POLICY_SURFACE:
        contract.update(
            {
                "world_geometry_token_contract_sha256": (
                    world_geometry_token_contract_sha256()
                ),
                "world_geometry_policy_config": (
                    world_geometry_policy_config_manifest(token_config)
                ),
            }
        )
    return contract


def combat_checkpoint_metadata(
    *,
    entity_count: int = ENTITY_COUNT,
    microticks: int,
    combat_target_active: bool,
    seed: int,
    updates: int,
    environment_steps: int,
    arsenal_runtime_capacity: (
        ArsenalRuntimeCapacity | Mapping[str, int] | None
    ) = None,
    policy_surface: str = ACTIVE_COMBAT_POLICY_SURFACE,
    world_geometry_config: (
        WorldGeometryPolicyConfig | Mapping[str, object] | None
    ) = None,
) -> dict[str, Any]:
    """Build complete metadata for a policy trained in the current ruleset."""

    return {
        **current_combat_checkpoint_contract(
            entity_count=entity_count,
            arsenal_runtime_capacity=arsenal_runtime_capacity,
            policy_surface=policy_surface,
            world_geometry_config=world_geometry_config,
        ),
        "microticks": int(microticks),
        "combat_target_active": bool(combat_target_active),
        "seed": int(seed),
        "updates": int(updates),
        "environment_steps": int(environment_steps),
    }


def combat_checkpoint_contract_metadata_view(
    metadata: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Select one unambiguous combat-contract representation.

    Portable population rows historically store the combat contract directly
    in checkpoint metadata. The WorldGen IL/PPO producer stores that same
    canonical contract under ``transfer_contract`` alongside task and trace
    provenance. A present nested contract is authoritative and is never filled
    from flat fields, so an incomplete producer contract still fails closed.
    Equal duplicate fields are tolerated; conflicting duplicates are rejected
    before current-contract validation.

    The returned mapping is a read-only view by convention. This function does
    not modify the caller's metadata or either stored representation.
    """

    if "transfer_contract" not in metadata:
        return metadata
    nested = metadata["transfer_contract"]
    if not isinstance(nested, Mapping):
        raise ValueError("checkpoint metadata transfer_contract must be a mapping")
    conflicts = sorted(
        name
        for name, nested_value in nested.items()
        if name in metadata and metadata[name] != nested_value
    )
    if conflicts:
        raise ValueError(
            "checkpoint metadata has conflicting flat and transfer_contract "
            f"values for: {', '.join(conflicts)}"
        )
    return nested


def combat_checkpoint_contract_errors(
    metadata: Mapping[str, Any],
    *,
    entity_count: int = ENTITY_COUNT,
    arsenal_runtime_capacity: (
        ArsenalRuntimeCapacity | Mapping[str, int] | None
    ) = None,
    policy_surface: str = ACTIVE_COMBAT_POLICY_SURFACE,
    world_geometry_config: (
        WorldGeometryPolicyConfig | Mapping[str, object] | None
    ) = None,
) -> list[str]:
    """Describe every mismatch with the current transfer contract."""

    try:
        contract_metadata = combat_checkpoint_contract_metadata_view(metadata)
    except ValueError as error:
        return [str(error)]
    errors = []
    for name, expected in current_combat_checkpoint_contract(
        entity_count=entity_count,
        arsenal_runtime_capacity=arsenal_runtime_capacity,
        policy_surface=policy_surface,
        world_geometry_config=world_geometry_config,
    ).items():
        actual = contract_metadata.get(name)
        if actual != expected:
            errors.append(f"{name}: checkpoint={actual!r}, current={expected!r}")
    return errors


def _policy_contract(
    policy_surface: str,
    world_geometry_config: WorldGeometryPolicyConfig,
    *,
    entity_count: int,
) -> tuple[str, int, str, str, int, str]:
    if policy_surface == ACTIVE_COMBAT_POLICY_SURFACE:
        return (
            COMBAT_OBSERVATION_SCHEMA,
            ACTIVE_OBSERVATION_SIZE,
            active_combat_observation_contract_sha256(),
            COMBAT_ACTION_SCHEMA,
            SKILL_COUNT,
            combat_skill_action_contract_sha256(),
        )
    if policy_surface == ARSENAL_POLICY_SURFACE:
        contract_sha256 = arsenal_policy_contract_sha256(
            world_geometry_config,
            entity_count=entity_count,
        )
        return (
            ARSENAL_POLICY_SCHEMA,
            arsenal_policy_observation_size(world_geometry_config),
            contract_sha256,
            f"arsenal_factored_logits_{ARSENAL_POLICY_ACTION_SIZE}",
            ARSENAL_POLICY_ACTION_SIZE,
            contract_sha256,
        )
    raise ValueError(
        "policy_surface must be "
        f"{ACTIVE_COMBAT_POLICY_SURFACE!r} or {ARSENAL_POLICY_SURFACE!r}"
    )


def _normalize_arsenal_runtime_capacity(
    capacity: ArsenalRuntimeCapacity | Mapping[str, int] | None,
) -> dict[str, int]:
    value = hytale_0_5_7_runtime_capacity() if capacity is None else capacity
    if isinstance(value, ArsenalRuntimeCapacity):
        abilities = value.abilities_per_entity
        events_per_entity = value.events_per_entity
        events_per_ability = value.events_per_ability
    else:
        abilities = value.get("abilities_per_entity")
        events_per_entity = value.get("events_per_entity")
        events_per_ability = value.get("events_per_ability")
    for name, item in (
        ("abilities_per_entity", abilities),
        ("events_per_entity", events_per_entity),
        ("events_per_ability", events_per_ability),
    ):
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ValueError(
                f"arsenal_runtime_capacity.{name} must be a positive integer"
            )
    if abilities > OBSERVATION_CAPACITY:
        raise ValueError(
            "arsenal runtime abilities exceed learner observation capacity"
        )
    if events_per_entity > ABILITY_CAPACITY * EVENT_CAPACITY:
        raise ValueError("arsenal runtime event bank exceeds catalog capacity")
    if events_per_ability > EVENT_CAPACITY:
        raise ValueError("arsenal runtime per-ability events exceed catalog capacity")
    return {
        "abilities_per_entity": abilities,
        "events_per_entity": events_per_entity,
        "events_per_ability": events_per_ability,
    }


def save_policy_checkpoint(
    path: str | Path,
    policy_params: RecurrentPolicyParams,
    config: PPOConfig,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save inference weights without Python pickle or device objects."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": np.asarray(
            CHECKPOINT_FORMAT_VERSION,
            dtype=np.int32,
        ),
        "config_json": np.asarray(json.dumps(asdict(config))),
        "metadata_json": np.asarray(json.dumps(dict(metadata or {}))),
        "encoder_input_kernel": np.asarray(policy_params.encoder_input.kernel),
        "encoder_input_bias": np.asarray(policy_params.encoder_input.bias),
        "encoder_hidden_kernel": np.asarray(policy_params.encoder_hidden.kernel),
        "encoder_hidden_bias": np.asarray(policy_params.encoder_hidden.bias),
        "gru_input_kernel": np.asarray(policy_params.gru.input_kernel),
        "gru_recurrent_kernel": np.asarray(policy_params.gru.recurrent_kernel),
        "gru_bias": np.asarray(policy_params.gru.bias),
        "actor_kernel": np.asarray(policy_params.actor.kernel),
        "actor_bias": np.asarray(policy_params.actor.bias),
        "critic_kernel": np.asarray(policy_params.critic.kernel),
        "critic_bias": np.asarray(policy_params.critic.bias),
    }
    with destination.open("wb") as checkpoint_file:
        np.savez_compressed(checkpoint_file, **payload)
    return destination


def load_policy_checkpoint(
    path: str | Path,
) -> tuple[RecurrentPolicyParams, PPOConfig, dict[str, Any]]:
    """Load a versioned portable policy checkpoint onto the JAX device."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as checkpoint:
        version = int(checkpoint["format_version"])
        if version != CHECKPOINT_FORMAT_VERSION:
            raise ValueError(
                f"unsupported checkpoint format {version}; "
                f"expected {CHECKPOINT_FORMAT_VERSION}"
            )
        config = PPOConfig(**json.loads(str(checkpoint["config_json"])))
        metadata = json.loads(str(checkpoint["metadata_json"]))
        policy_params = RecurrentPolicyParams(
            encoder_input=DenseParams(
                kernel=_float32(checkpoint["encoder_input_kernel"]),
                bias=_float32(checkpoint["encoder_input_bias"]),
            ),
            encoder_hidden=DenseParams(
                kernel=_float32(checkpoint["encoder_hidden_kernel"]),
                bias=_float32(checkpoint["encoder_hidden_bias"]),
            ),
            gru=GRUParams(
                input_kernel=_float32(checkpoint["gru_input_kernel"]),
                recurrent_kernel=_float32(checkpoint["gru_recurrent_kernel"]),
                bias=_float32(checkpoint["gru_bias"]),
            ),
            actor=DenseParams(
                kernel=_float32(checkpoint["actor_kernel"]),
                bias=_float32(checkpoint["actor_bias"]),
            ),
            critic=DenseParams(
                kernel=_float32(checkpoint["critic_kernel"]),
                bias=_float32(checkpoint["critic_bias"]),
            ),
        )
    _validate_checkpoint_shapes(policy_params, config)
    return policy_params, config, metadata


def _float32(value: np.ndarray):
    if value.dtype != np.float32:
        raise ValueError(f"checkpoint tensor must be float32, got {value.dtype}")
    return jnp.asarray(value, dtype=jnp.float32)


def _validate_checkpoint_shapes(
    params: RecurrentPolicyParams,
    config: PPOConfig,
) -> None:
    expected_shapes = {
        "encoder_input.kernel": (
            config.observation_size,
            config.encoder_size,
        ),
        "encoder_input.bias": (config.encoder_size,),
        "encoder_hidden.kernel": (
            config.encoder_size,
            config.encoder_size,
        ),
        "encoder_hidden.bias": (config.encoder_size,),
        "gru.input_kernel": (
            config.encoder_size,
            3 * config.recurrent_size,
        ),
        "gru.recurrent_kernel": (
            config.recurrent_size,
            3 * config.recurrent_size,
        ),
        "gru.bias": (3 * config.recurrent_size,),
        "actor.kernel": (config.recurrent_size, config.action_size),
        "actor.bias": (config.action_size,),
        "critic.kernel": (config.recurrent_size, 1),
        "critic.bias": (1,),
    }
    actual = {
        "encoder_input.kernel": params.encoder_input.kernel.shape,
        "encoder_input.bias": params.encoder_input.bias.shape,
        "encoder_hidden.kernel": params.encoder_hidden.kernel.shape,
        "encoder_hidden.bias": params.encoder_hidden.bias.shape,
        "gru.input_kernel": params.gru.input_kernel.shape,
        "gru.recurrent_kernel": params.gru.recurrent_kernel.shape,
        "gru.bias": params.gru.bias.shape,
        "actor.kernel": params.actor.kernel.shape,
        "actor.bias": params.actor.bias.shape,
        "critic.kernel": params.critic.kernel.shape,
        "critic.bias": params.critic.bias.shape,
    }
    for name, expected in expected_shapes.items():
        if actual[name] != expected:
            raise ValueError(
                f"checkpoint {name} has shape {actual[name]}, expected {expected}"
            )
