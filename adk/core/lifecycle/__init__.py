"""Contract-stamped checkpoint lifecycle for ADK-built policies."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from hytalegym.jax.training.checkpoint import (
    ARSENAL_POLICY_SURFACE,
    combat_checkpoint_contract_errors,
    combat_checkpoint_metadata,
    load_policy_checkpoint,
    save_policy_checkpoint,
)
from hytalegym.jax.training.types import (
    PPOConfig,
    RecurrentPolicyParams,
)

from adk.contracts.stamp import ContractStamp, require_current
from adk.runtime.env_adapter import BuiltAgent
from adk.runtime.tree import ENVIRONMENT_STATE_SELECTOR_SCHEMA
from adk.training import make_training_config


ADK_CHECKPOINT_SCHEMA = "hytalerl_adk_checkpoint_v3"


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    policy_params: RecurrentPolicyParams
    config: PPOConfig
    metadata: Mapping[str, Any]


def checkpoint_metadata(
    built: BuiltAgent,
    *,
    seed: int,
    updates: int,
    environment_steps: int,
    combat_target_active: bool = True,
    run_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build JSON-safe metadata from the identity captured by ``build()``."""

    _require_current_build(built)
    world_geometry_config = built.world_geometry_config
    metadata = combat_checkpoint_metadata(
        microticks=built.spec.microticks,
        combat_target_active=combat_target_active,
        seed=seed,
        updates=updates,
        environment_steps=environment_steps,
        arsenal_runtime_capacity=built.runtime_capacity,
        policy_surface=ARSENAL_POLICY_SURFACE,
        world_geometry_config=world_geometry_config,
    )
    # The upstream helper reads live contracts at this point. Refuse to save if
    # they differ from the identity captured before training began.
    errors = combat_checkpoint_contract_errors(
        metadata,
        arsenal_runtime_capacity=built.runtime_capacity,
        policy_surface=ARSENAL_POLICY_SURFACE,
        world_geometry_config=world_geometry_config,
    )
    if errors:
        raise RuntimeError(
            "checkpoint contract moved while metadata was captured:\n- "
            + "\n- ".join(errors)
        )
    metadata["adk"] = {
        "schema": ADK_CHECKPOINT_SCHEMA,
        "spec": json.loads(built.spec.canonical_json()),
        "spec_sha256": built.spec.spec_sha256(),
        "stamp": built.stamp.to_dict(),
        "stamp_sha256": built.stamp.stamp_sha256(),
        "scene": built.scene.metadata(),
        "backend": {
            "name": built.backend.backend,
            "device_kind": built.backend.device_kind,
        },
        "runtime": {
            "environment_state_selector_schema": (
                ENVIRONMENT_STATE_SELECTOR_SCHEMA
            ),
        },
    }
    metadata["run"] = dict(run_metadata or {})
    return metadata


def save_checkpoint(
    path: str | Path,
    built: BuiltAgent,
    policy_params: RecurrentPolicyParams,
    config: PPOConfig,
    *,
    seed: int,
    updates: int,
    environment_steps: int,
    combat_target_active: bool = True,
    run_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save only after the build, PPO config, and live contracts still agree."""

    _validate_config(built, config)
    metadata = checkpoint_metadata(
        built,
        seed=seed,
        updates=updates,
        environment_steps=environment_steps,
        combat_target_active=combat_target_active,
        run_metadata=run_metadata,
    )
    return save_policy_checkpoint(
        path,
        policy_params,
        config,
        metadata=metadata,
    )


def load_checkpoint(
    path: str | Path,
    built: BuiltAgent,
) -> LoadedCheckpoint:
    """Load a checkpoint only when every current ADK identity still matches."""

    _require_current_build(built)
    policy_params, config, metadata = load_policy_checkpoint(path)
    errors = _checkpoint_errors(built, config, metadata)
    if errors:
        raise RuntimeError(
            "checkpoint does not match the current ADK build:\n- "
            + "\n- ".join(errors)
        )
    return LoadedCheckpoint(
        policy_params=policy_params,
        config=config,
        metadata=metadata,
    )


def _checkpoint_errors(
    built: BuiltAgent,
    config: PPOConfig,
    metadata: Mapping[str, Any],
) -> list[str]:
    world_geometry_config = built.world_geometry_config
    errors = combat_checkpoint_contract_errors(
        metadata,
        arsenal_runtime_capacity=built.runtime_capacity,
        policy_surface=ARSENAL_POLICY_SURFACE,
        world_geometry_config=world_geometry_config,
    )
    try:
        _validate_config(built, config)
    except (TypeError, ValueError) as error:
        errors.append(str(error))
    adk = metadata.get("adk")
    if not isinstance(adk, Mapping):
        errors.append("adk metadata is missing")
        return errors
    if adk.get("schema") != ADK_CHECKPOINT_SCHEMA:
        errors.append("adk checkpoint schema changed")
    if adk.get("spec_sha256") != built.spec.spec_sha256():
        errors.append("AgentSpec digest differs")
    if adk.get("spec") != json.loads(built.spec.canonical_json()):
        errors.append("AgentSpec payload differs")
    if adk.get("scene") != built.scene.metadata():
        errors.append("JAX scene identity differs")
    backend = adk.get("backend")
    if not isinstance(backend, Mapping) or backend.get("name") != "jax":
        errors.append("checkpoint backend is not jax")
    runtime = adk.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("environment_state_selector_schema")
        != ENVIRONMENT_STATE_SELECTOR_SCHEMA
    ):
        errors.append("environment state-selector schema differs")
    saved_stamp = adk.get("stamp")
    try:
        restored = _stamp_from_mapping(saved_stamp)
    except (TypeError, ValueError, KeyError) as error:
        errors.append(f"ADK stamp is invalid: {error}")
    else:
        drift = restored.mismatches(built.stamp)
        if drift:
            errors.append(f"ADK stamp differs: {drift}")
        if adk.get("stamp_sha256") != restored.stamp_sha256():
            errors.append("ADK stamp digest differs")
    return errors


def _stamp_from_mapping(value: object) -> ContractStamp:
    if not isinstance(value, Mapping):
        raise TypeError("stamp must be a mapping")
    contracts = value["contracts"]
    names = value["action_head_names"]
    sizes = value["action_head_sizes"]
    observation_size = value["observation_size"]
    if not isinstance(contracts, Mapping):
        raise TypeError("stamp contracts must be a mapping")
    if not isinstance(names, (list, tuple)):
        raise TypeError("stamp action head names must be a sequence")
    if not isinstance(sizes, (list, tuple)):
        raise TypeError("stamp action head sizes must be a sequence")
    if any(not isinstance(name, str) for name in names):
        raise TypeError("stamp action head names must contain strings")
    if any(
        isinstance(size, bool) or not isinstance(size, int)
        for size in sizes
    ):
        raise TypeError("stamp action head sizes must contain integers")
    if isinstance(observation_size, bool) or not isinstance(observation_size, int):
        raise TypeError("stamp observation_size must be an integer")
    return ContractStamp(
        contracts=dict(contracts),
        action_head_names=tuple(names),
        action_head_sizes=tuple(sizes),
        observation_size=observation_size,
    )


def _require_current_build(built: BuiltAgent) -> None:
    if not isinstance(built, BuiltAgent):
        raise TypeError("built must be a BuiltAgent")
    if built.backend.backend != "jax":
        raise RuntimeError("checkpoints can only be produced by the JAX lane")
    require_current(built.stamp, built.world_geometry_config)


def _validate_config(built: BuiltAgent, config: PPOConfig) -> None:
    # Reuse the public derivation as the source of required static fields.
    expected = make_training_config(
        built,
        num_envs=built.batch,
        num_minibatches=config.num_minibatches,
        rollout_steps=config.rollout_steps,
        update_epochs=config.update_epochs,
        encoder_size=config.encoder_size,
        recurrent_size=config.recurrent_size,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_epsilon=config.clip_epsilon,
        value_coefficient=config.value_coefficient,
        entropy_coefficient=config.entropy_coefficient,
        max_gradient_norm=config.max_gradient_norm,
    )
    static_fields = (
        "num_envs",
        "observation_size",
        "action_size",
        "action_head_sizes",
        "action_transport",
    )
    differences = [
        name
        for name in static_fields
        if getattr(config, name) != getattr(expected, name)
    ]
    if differences:
        raise ValueError(f"PPO config differs on static fields: {differences}")


__all__ = [
    "ADK_CHECKPOINT_SCHEMA",
    "LoadedCheckpoint",
    "checkpoint_metadata",
    "load_checkpoint",
    "save_checkpoint",
]
