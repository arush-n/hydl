"""Deterministic, pickle-free population checkpoint format v1."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from hytalegym.jax.combat.arsenal.schema.types import (
    ArsenalRuntimeCapacity,
    ArsenalRuntimeConfig,
)
from hytalegym.jax.combat.observation.v3.tokens.world import (
    WorldGeometryPolicyConfig,
)
from hytalegym.jax.policy import DenseParams
from hytalegym.jax.training.checkpoint import (
    ARSENAL_POLICY_SURFACE,
    CHECKPOINT_FORMAT_VERSION,
    current_combat_checkpoint_contract,
    load_policy_checkpoint,
)
from hytalegym.jax.training.types import (
    GRUParams,
    PPOConfig,
    RecurrentPolicyParams,
)

from ..assignment import (
    PolicyActorAssignment,
    validate_policy_actor_assignment,
)
from ..contract import multi_actor_assignment_contract_sha256
from ..population_training import (
    MultiActorPopulationTrainer,
    MultiActorPopulationTrainingState,
    plan_multi_actor_population,
)
from ..runtime import MultiActorArenaState
from .provenance import (
    ARSENAL_RUNTIME_CONFIG_CONTENT_SCHEMA,
    RuntimeTrainingEntitySpec,
    arsenal_runtime_config_content_sha256,
    attest_policy_training_contexts,
    validate_policy_training_context_attestations,
)


POPULATION_CHECKPOINT_SCHEMA = "hytalerl_multi_actor_population_checkpoint_v1"
POPULATION_CHECKPOINT_VERSION = 1
_ARRAYS_FILE = "population.npz"
_MANIFEST_FILE = "manifest.json"
_MANIFEST_SHA_FILE = "manifest.sha256"
_SELECTED_POLICY_FILE = "selected-policy.npz"

_POLICY_ARRAYS = (
    ("policy_encoder_input_kernel", ("encoder_input", "kernel")),
    ("policy_encoder_input_bias", ("encoder_input", "bias")),
    ("policy_encoder_hidden_kernel", ("encoder_hidden", "kernel")),
    ("policy_encoder_hidden_bias", ("encoder_hidden", "bias")),
    ("policy_gru_input_kernel", ("gru", "input_kernel")),
    ("policy_gru_recurrent_kernel", ("gru", "recurrent_kernel")),
    ("policy_gru_bias", ("gru", "bias")),
    ("policy_actor_kernel", ("actor", "kernel")),
    ("policy_actor_bias", ("actor", "bias")),
    ("policy_critic_kernel", ("critic", "kernel")),
    ("policy_critic_bias", ("critic", "bias")),
)


@dataclass(frozen=True)
class LoadedPopulationCheckpointV1:
    """Restored update-boundary state and its verified portable metadata."""

    state: MultiActorPopulationTrainingState
    assignment: PolicyActorAssignment
    config: PPOConfig
    manifest: dict[str, Any]
    selected_policy_path: Path | None


def population_checkpoint_contract_manifest() -> dict[str, Any]:
    """Describe the stable population checkpoint semantics."""

    return {
        "schema": POPULATION_CHECKPOINT_SCHEMA,
        "version": POPULATION_CHECKPOINT_VERSION,
        "files": {
            "arrays": _ARRAYS_FILE,
            "manifest": _MANIFEST_FILE,
            "manifest_sha256": _MANIFEST_SHA_FILE,
            "selected_policy": _SELECTED_POLICY_FILE,
        },
        "state": {
            "policy_bank": "float32 RecurrentPolicyParams with leading K axis",
            "optimizer_states": (
                "one canonical PPO PyTree per sorted trainable policy ID"
            ),
            "counters": (
                "int32 update_count and total_trainable_actor_steps on K_train"
            ),
            "assignment": "exact int32/bool PolicyActorAssignment[B,P]",
            "policy_training_contexts": (
                "optional canonical policy_id to entity/profile/team/controller "
                "upload provenance, derived from the actual runtime config"
            ),
            "policy_training_context_attestations": (
                "per-policy runtime/content identities; caller metadata cannot "
                "supply or override them"
            ),
            "runtime_config_content_sha256": (
                "canonical identity of every numeric ArsenalRuntimeConfig leaf"
            ),
            "runtime_config_content_schema": (
                ARSENAL_RUNTIME_CONFIG_CONTENT_SCHEMA
            ),
        },
        "resume_boundary": (
            "bank, optimizers, and counters resume on a freshly reset arena; "
            "in-flight arena/recurrent/episode state is not serialized"
        ),
        "integrity": (
            "canonical manifest plus deterministic ordered numeric archive; "
            "both byte and decoded-array content hashes are verified"
        ),
    }


def population_checkpoint_contract_sha256() -> str:
    """Return the stable identity of population checkpoint v1."""

    return hashlib.sha256(
        _canonical_json(population_checkpoint_contract_manifest())
    ).hexdigest().upper()


def save_population_checkpoint_v1(
    path: str | Path,
    state: MultiActorPopulationTrainingState,
    assignment: PolicyActorAssignment,
    config: PPOConfig,
    trainer: MultiActorPopulationTrainer,
    *,
    entity_count: int,
    selected_policy_id: int | None = None,
    metadata: Mapping[str, Any] | None = None,
    arsenal_runtime_capacity: (
        ArsenalRuntimeCapacity | Mapping[str, int] | None
    ) = None,
    world_geometry_config: (
        WorldGeometryPolicyConfig | Mapping[str, object] | None
    ) = None,
    runtime_config: ArsenalRuntimeConfig | None = None,
    training_entity_specs: Sequence[RuntimeTrainingEntitySpec] = (),
) -> Path:
    """Save a complete population update boundary without Python pickle."""

    destination = Path(path)
    validate_policy_actor_assignment(assignment, entity_count=entity_count)
    bank_size = _validate_policy_bank(state.policy_bank, config)
    plan = plan_multi_actor_population(
        assignment,
        config,
        bank_size=bank_size,
    )
    _validate_trainer_plan(trainer, plan)
    if len(state.optimizer_states) != len(plan.trainable_policy_ids):
        raise ValueError("optimizer state count differs from trainable policy IDs")

    combat_contract = current_combat_checkpoint_contract(
        entity_count=entity_count,
        arsenal_runtime_capacity=arsenal_runtime_capacity,
        policy_surface=ARSENAL_POLICY_SURFACE,
        world_geometry_config=world_geometry_config,
    )
    _validate_config_contract(config, combat_contract)
    user_metadata = _json_mapping(metadata)
    _validate_metadata_contract(user_metadata, combat_contract)
    if runtime_config is None:
        if training_entity_specs:
            raise ValueError(
                "training entity specs require the actual runtime config"
            )
        raise ValueError(
            "population checkpoints require the actual Arsenal runtime config"
        )
    runtime_config_sha256 = arsenal_runtime_config_content_sha256(runtime_config)
    if training_entity_specs:
        (
            raw_contexts,
            raw_attestations,
            attested_runtime_sha256,
        ) = attest_policy_training_contexts(
            runtime_config,
            assignment,
            tuple(training_entity_specs),
        )
        if attested_runtime_sha256 != runtime_config_sha256:
            raise RuntimeError("runtime provenance derivations disagree")
        policy_training_contexts = _validate_policy_training_contexts(
            list(raw_contexts),
            assignment,
            bank_size=bank_size,
            entity_count=entity_count,
        )
        policy_training_context_attestations = (
            validate_policy_training_context_attestations(
                policy_training_contexts,
                list(raw_attestations),
                runtime_config_content_sha256=runtime_config_sha256,
            )
        )
    else:
        policy_training_contexts = []
        policy_training_context_attestations = []

    arrays: dict[str, np.ndarray] = {}
    for name, path_parts in _POLICY_ARRAYS:
        arrays[name] = _host_array(_nested_attr(state.policy_bank, path_parts))
    arrays.update(
        {
            "assignment_actor_index": _host_array(assignment.actor_index),
            "assignment_policy_id": _host_array(assignment.policy_id),
            "assignment_active": _host_array(assignment.active),
            "assignment_trainable": _host_array(assignment.trainable),
            "update_count": _host_array(state.update_count),
            "total_trainable_actor_steps": _host_array(
                state.total_trainable_actor_steps
            ),
        }
    )
    optimizer_manifest = []
    for policy_id, optimizer_state in zip(
        plan.trainable_policy_ids,
        state.optimizer_states,
        strict=True,
    ):
        leaves, _ = jax.tree_util.tree_flatten(optimizer_state)
        records = []
        for leaf_index, leaf in enumerate(leaves):
            name = f"optimizer_{policy_id:08d}_{leaf_index:04d}"
            arrays[name] = _host_array(leaf)
            records.append(name)
        optimizer_manifest.append(
            {"policy_id": policy_id, "leaves": records}
        )

    update_count = arrays["update_count"]
    actor_steps = arrays["total_trainable_actor_steps"]
    expected_counter_shape = (len(plan.trainable_policy_ids),)
    if update_count.shape != expected_counter_shape or update_count.dtype != np.int32:
        raise ValueError("update_count must be int32[K_train]")
    if actor_steps.shape != expected_counter_shape or actor_steps.dtype != np.int32:
        raise ValueError("total_trainable_actor_steps must be int32[K_train]")

    array_specs = [_array_spec(name, arrays[name]) for name in sorted(arrays)]
    _prepare_empty_directory(destination)
    arrays_path = destination / _ARRAYS_FILE
    _write_deterministic_npz(arrays_path, arrays)
    arrays_sha256 = _file_sha256(arrays_path)
    content_sha256 = _arrays_content_sha256(arrays)

    selected_manifest = None
    if selected_policy_id is not None:
        selected_manifest = _save_selected_policy(
            destination,
            state.policy_bank,
            config,
            plan.trainable_policy_ids,
            update_count,
            actor_steps,
            combat_contract,
            selected_policy_id=selected_policy_id,
            bank_size=bank_size,
            population_content_sha256=content_sha256,
            metadata=user_metadata,
            policy_training_contexts=policy_training_contexts,
            policy_training_context_attestations=(
                policy_training_context_attestations
            ),
        )

    manifest = {
        "schema": POPULATION_CHECKPOINT_SCHEMA,
        "version": POPULATION_CHECKPOINT_VERSION,
        "checkpoint_contract_sha256": population_checkpoint_contract_sha256(),
        "multi_actor_assignment_contract_sha256": (
            multi_actor_assignment_contract_sha256()
        ),
        "combat_contract": combat_contract,
        "entity_count": entity_count,
        "bank_size": bank_size,
        "config": asdict(config),
        "assignment": {
            "actor_index": arrays["assignment_actor_index"].tolist(),
            "policy_id": arrays["assignment_policy_id"].tolist(),
            "active": arrays["assignment_active"].tolist(),
            "trainable": arrays["assignment_trainable"].tolist(),
        },
        "training_plan": {
            "trainable_policy_ids": list(plan.trainable_policy_ids),
            "trainable_actor_rows": [
                [list(row) for row in rows]
                for rows in plan.trainable_actor_rows
            ],
            "trainable_actor_count_per_policy": (
                plan.trainable_actor_count_per_policy
            ),
        },
        "state": {
            "update_count": update_count.tolist(),
            "total_trainable_actor_steps": actor_steps.tolist(),
            "optimizer_states": optimizer_manifest,
            "resume_boundary": "fresh_arena",
        },
        "arrays": {
            "file": _ARRAYS_FILE,
            "sha256": arrays_sha256,
            "content_sha256": content_sha256,
            "entries": array_specs,
        },
        "selected_policy_export": selected_manifest,
        "runtime_config_content_schema": (
            ARSENAL_RUNTIME_CONFIG_CONTENT_SCHEMA
        ),
        "runtime_config_content_sha256": runtime_config_sha256,
        "policy_training_contexts": policy_training_contexts,
        "policy_training_context_attestations": (
            policy_training_context_attestations
        ),
        "metadata": user_metadata,
    }
    manifest_bytes = _canonical_json(manifest) + b"\n"
    (destination / _MANIFEST_FILE).write_bytes(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest().upper()
    (destination / _MANIFEST_SHA_FILE).write_text(
        manifest_sha256 + "\n",
        encoding="ascii",
        newline="\n",
    )
    return destination


def load_population_checkpoint_v1(
    path: str | Path,
    trainer: MultiActorPopulationTrainer,
    arena: MultiActorArenaState,
    *,
    expected_assignment: PolicyActorAssignment,
    expected_config: PPOConfig,
    entity_count: int,
    arsenal_runtime_capacity: (
        ArsenalRuntimeCapacity | Mapping[str, int] | None
    ) = None,
    world_geometry_config: (
        WorldGeometryPolicyConfig | Mapping[str, object] | None
    ) = None,
    expected_runtime_config: ArsenalRuntimeConfig | None = None,
    expected_training_entity_specs: Sequence[RuntimeTrainingEntitySpec] = (),
) -> LoadedPopulationCheckpointV1:
    """Verify and restore v1 state onto a caller-provided fresh arena."""

    source = Path(path)
    manifest_path = source / _MANIFEST_FILE
    manifest_bytes = manifest_path.read_bytes()
    expected_manifest_sha = (source / _MANIFEST_SHA_FILE).read_text(
        encoding="ascii"
    ).strip()
    actual_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest().upper()
    if actual_manifest_sha != expected_manifest_sha:
        raise ValueError("population manifest SHA-256 mismatch")
    manifest = json.loads(manifest_bytes)
    if manifest_bytes != _canonical_json(manifest) + b"\n":
        raise ValueError("population manifest is not canonical JSON")
    if manifest.get("schema") != POPULATION_CHECKPOINT_SCHEMA:
        raise ValueError("unsupported population checkpoint schema")
    if manifest.get("version") != POPULATION_CHECKPOINT_VERSION:
        raise ValueError("unsupported population checkpoint version")
    if (
        manifest.get("checkpoint_contract_sha256")
        != population_checkpoint_contract_sha256()
    ):
        raise ValueError("population checkpoint contract identity mismatch")
    if (
        manifest.get("multi_actor_assignment_contract_sha256")
        != multi_actor_assignment_contract_sha256()
    ):
        raise ValueError("multi-actor assignment contract identity mismatch")
    if manifest.get("entity_count") != entity_count:
        raise ValueError("population checkpoint entity_count mismatch")

    current_contract = current_combat_checkpoint_contract(
        entity_count=entity_count,
        arsenal_runtime_capacity=arsenal_runtime_capacity,
        policy_surface=ARSENAL_POLICY_SURFACE,
        world_geometry_config=world_geometry_config,
    )
    if manifest.get("combat_contract") != current_contract:
        raise ValueError("population checkpoint combat contract is stale")
    if expected_runtime_config is None:
        raise ValueError(
            "population checkpoint restore requires the expected runtime config"
        )
    if (
        manifest.get("runtime_config_content_schema")
        != ARSENAL_RUNTIME_CONFIG_CONTENT_SCHEMA
    ):
        raise ValueError("population checkpoint runtime config schema is stale")
    expected_runtime_sha256 = arsenal_runtime_config_content_sha256(
        expected_runtime_config
    )
    if (
        manifest.get("runtime_config_content_sha256")
        != expected_runtime_sha256
    ):
        raise ValueError("population checkpoint runtime config identity mismatch")
    config = PPOConfig(**manifest["config"])
    if asdict(config) != asdict(expected_config):
        raise ValueError("population checkpoint PPO config mismatch")
    _validate_config_contract(config, current_contract)

    arrays_path = source / manifest["arrays"]["file"]
    if _file_sha256(arrays_path) != manifest["arrays"]["sha256"]:
        raise ValueError("population array archive SHA-256 mismatch")
    arrays = _load_and_validate_arrays(arrays_path, manifest["arrays"]["entries"])
    if _arrays_content_sha256(arrays) != manifest["arrays"]["content_sha256"]:
        raise ValueError("population decoded-array content mismatch")

    assignment = PolicyActorAssignment(
        actor_index=jnp.asarray(arrays["assignment_actor_index"], dtype=jnp.int32),
        policy_id=jnp.asarray(arrays["assignment_policy_id"], dtype=jnp.int32),
        active=jnp.asarray(arrays["assignment_active"], dtype=jnp.bool_),
        trainable=jnp.asarray(arrays["assignment_trainable"], dtype=jnp.bool_),
    )
    validate_policy_actor_assignment(assignment, entity_count=entity_count)
    _require_assignment_equal(assignment, expected_assignment)
    _require_assignment_manifest_equal(assignment, manifest["assignment"])

    policy_bank = _policy_bank_from_arrays(arrays)
    bank_size = _validate_policy_bank(policy_bank, config)
    if bank_size != manifest["bank_size"]:
        raise ValueError("population bank size differs from manifest")
    policy_training_contexts = _validate_policy_training_contexts(
        manifest.get("policy_training_contexts", []),
        assignment,
        bank_size=bank_size,
        entity_count=entity_count,
    )
    policy_training_context_attestations = (
        validate_policy_training_context_attestations(
            policy_training_contexts,
            manifest.get("policy_training_context_attestations", []),
            runtime_config_content_sha256=expected_runtime_sha256,
        )
    )
    if expected_training_entity_specs:
        (
            expected_contexts,
            expected_attestations,
            attested_runtime_sha256,
        ) = attest_policy_training_contexts(
            expected_runtime_config,
            expected_assignment,
            tuple(expected_training_entity_specs),
        )
        if attested_runtime_sha256 != expected_runtime_sha256:
            raise RuntimeError("restored runtime provenance derivations disagree")
        if list(expected_contexts) != policy_training_contexts:
            raise ValueError(
                "population training contexts differ from expected runtime"
            )
        if list(expected_attestations) != policy_training_context_attestations:
            raise ValueError(
                "population training attestations differ from expected runtime"
            )
    elif policy_training_contexts or policy_training_context_attestations:
        raise ValueError(
            "population training provenance requires expected entity specs"
        )
    plan = plan_multi_actor_population(
        assignment,
        config,
        bank_size=bank_size,
    )
    _validate_trainer_plan(trainer, plan)
    _validate_plan_manifest(plan, manifest["training_plan"])
    template = trainer.initialize(policy_bank, arena)
    optimizer_states = _restore_optimizer_states(
        arrays,
        manifest["state"]["optimizer_states"],
        plan.trainable_policy_ids,
        template.optimizer_states,
    )
    update_count = jnp.asarray(arrays["update_count"], dtype=jnp.int32)
    actor_steps = jnp.asarray(
        arrays["total_trainable_actor_steps"], dtype=jnp.int32
    )
    if update_count.tolist() != manifest["state"]["update_count"]:
        raise ValueError("update_count differs from manifest")
    if actor_steps.tolist() != manifest["state"]["total_trainable_actor_steps"]:
        raise ValueError("actor-step count differs from manifest")
    restored = template._replace(
        policy_bank=policy_bank,
        optimizer_states=optimizer_states,
        update_count=update_count,
        total_trainable_actor_steps=actor_steps,
    )
    selected_path = _validate_selected_policy(
        source,
        manifest.get("selected_policy_export"),
        policy_bank,
        config,
        manifest["arrays"]["content_sha256"],
        policy_training_contexts,
        policy_training_context_attestations,
    )
    return LoadedPopulationCheckpointV1(
        state=restored,
        assignment=assignment,
        config=config,
        manifest=manifest,
        selected_policy_path=selected_path,
    )


def _save_selected_policy(
    destination: Path,
    policy_bank: RecurrentPolicyParams,
    config: PPOConfig,
    trainable_policy_ids: tuple[int, ...],
    update_count: np.ndarray,
    actor_steps: np.ndarray,
    combat_contract: Mapping[str, Any],
    *,
    selected_policy_id: int,
    bank_size: int,
    population_content_sha256: str,
    metadata: Mapping[str, Any],
    policy_training_contexts: list[dict[str, Any]],
    policy_training_context_attestations: list[dict[str, Any]],
) -> dict[str, Any]:
    if isinstance(selected_policy_id, bool) or not isinstance(selected_policy_id, int):
        raise TypeError("selected_policy_id must be an integer")
    if selected_policy_id < 0 or selected_policy_id >= bank_size:
        raise ValueError("selected_policy_id is outside the policy bank")
    selected = jax.tree_util.tree_map(
        lambda value: value[selected_policy_id],
        policy_bank,
    )
    selected_arrays = {
        name: _host_array(_nested_attr(selected, path_parts))
        for name, path_parts in _POLICY_ARRAYS
    }
    if selected_policy_id in trainable_policy_ids:
        counter_index = trainable_policy_ids.index(selected_policy_id)
        selected_updates = int(update_count[counter_index])
        selected_steps = int(actor_steps[counter_index])
        selected_trainable = True
    else:
        selected_updates = 0
        selected_steps = 0
        selected_trainable = False
    selected_context = next(
        (
            context
            for context in policy_training_contexts
            if context["policy_id"] == selected_policy_id
        ),
        None,
    )
    selected_attestation = next(
        (
            attestation
            for attestation in policy_training_context_attestations
            if attestation["policy_id"] == selected_policy_id
        ),
        None,
    )
    if (selected_context is None) != (selected_attestation is None):
        raise ValueError("selected policy context and attestation differ")
    selected_metadata = {
        **metadata,
        **combat_contract,
        "population_checkpoint_schema": POPULATION_CHECKPOINT_SCHEMA,
        "population_checkpoint_contract_sha256": (
            population_checkpoint_contract_sha256()
        ),
        "population_content_sha256": population_content_sha256,
        "population_policy_id": selected_policy_id,
        "population_policy_trainable": selected_trainable,
        "population_policy_updates": selected_updates,
        "population_policy_actor_steps": selected_steps,
        "policy_training_context": selected_context,
        "population_policy_training_context": selected_context,
        "policy_training_context_attestation": selected_attestation,
        "policy_training_contexts": policy_training_contexts,
        "policy_training_context_attestations": (
            policy_training_context_attestations
        ),
        "population_policy_content_sha256": (
            _arrays_content_sha256(selected_arrays)
        ),
    }
    selected_path = destination / _SELECTED_POLICY_FILE
    _write_portable_policy_checkpoint(
        selected_path,
        selected,
        config,
        selected_metadata,
    )
    return {
        "file": _SELECTED_POLICY_FILE,
        "sha256": _file_sha256(selected_path),
        "policy_id": selected_policy_id,
        "trainable": selected_trainable,
        "update_count": selected_updates,
        "total_trainable_actor_steps": selected_steps,
        "policy_content_sha256": _arrays_content_sha256(selected_arrays),
        "source_population_content_sha256": population_content_sha256,
        "policy_training_context": selected_context,
        "policy_training_context_attestation": selected_attestation,
    }


def _validate_selected_policy(
    source: Path,
    selected_manifest: Mapping[str, Any] | None,
    policy_bank: RecurrentPolicyParams,
    config: PPOConfig,
    population_content_sha256: str,
    policy_training_contexts: list[dict[str, Any]],
    policy_training_context_attestations: list[dict[str, Any]],
) -> Path | None:
    if selected_manifest is None:
        return None
    selected_path = source / selected_manifest["file"]
    if _file_sha256(selected_path) != selected_manifest["sha256"]:
        raise ValueError("selected-policy export SHA-256 mismatch")
    selected, selected_config, metadata = load_policy_checkpoint(selected_path)
    if asdict(selected_config) != asdict(config):
        raise ValueError("selected-policy PPO config mismatch")
    policy_id = selected_manifest["policy_id"]
    expected = jax.tree_util.tree_map(lambda value: value[policy_id], policy_bank)
    for actual_leaf, expected_leaf in zip(
        jax.tree_util.tree_leaves(selected),
        jax.tree_util.tree_leaves(expected),
        strict=True,
    ):
        if not np.array_equal(
            np.asarray(actual_leaf),
            np.asarray(expected_leaf),
        ):
            raise ValueError("selected-policy tensors differ from population row")
    if metadata.get("population_content_sha256") != population_content_sha256:
        raise ValueError("selected-policy population identity mismatch")
    if metadata.get("population_policy_id") != policy_id:
        raise ValueError("selected-policy ID metadata mismatch")
    expected_context = next(
        (
            context
            for context in policy_training_contexts
            if context["policy_id"] == policy_id
        ),
        None,
    )
    expected_attestation = next(
        (
            attestation
            for attestation in policy_training_context_attestations
            if attestation["policy_id"] == policy_id
        ),
        None,
    )
    if selected_manifest.get("policy_training_context") != expected_context:
        raise ValueError("selected-policy manifest training context mismatch")
    if metadata.get("population_policy_training_context") != expected_context:
        raise ValueError("selected-policy metadata training context mismatch")
    if metadata.get("policy_training_context") != expected_context:
        raise ValueError("selected-policy direct training context mismatch")
    if (
        selected_manifest.get("policy_training_context_attestation")
        != expected_attestation
    ):
        raise ValueError("selected-policy manifest attestation mismatch")
    if (
        metadata.get("policy_training_context_attestation")
        != expected_attestation
    ):
        raise ValueError("selected-policy metadata attestation mismatch")
    return selected_path


def _restore_optimizer_states(
    arrays: Mapping[str, np.ndarray],
    manifest: list[Mapping[str, Any]],
    policy_ids: tuple[int, ...],
    templates: tuple[Any, ...],
) -> tuple[Any, ...]:
    if [row["policy_id"] for row in manifest] != list(policy_ids):
        raise ValueError("optimizer policy ownership differs from training plan")
    restored = []
    for row, template in zip(manifest, templates, strict=True):
        template_leaves, treedef = jax.tree_util.tree_flatten(template)
        names = row["leaves"]
        if len(names) != len(template_leaves):
            raise ValueError("optimizer PyTree leaf count changed")
        leaves = []
        for name, template_leaf in zip(names, template_leaves, strict=True):
            value = arrays[name]
            expected = np.asarray(template_leaf)
            if value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError(f"optimizer leaf {name} shape or dtype changed")
            leaves.append(jnp.asarray(value, dtype=template_leaf.dtype))
        restored.append(jax.tree_util.tree_unflatten(treedef, leaves))
    return tuple(restored)


def _policy_bank_from_arrays(arrays: Mapping[str, np.ndarray]):
    return RecurrentPolicyParams(
        encoder_input=DenseParams(
            kernel=jnp.asarray(arrays["policy_encoder_input_kernel"]),
            bias=jnp.asarray(arrays["policy_encoder_input_bias"]),
        ),
        encoder_hidden=DenseParams(
            kernel=jnp.asarray(arrays["policy_encoder_hidden_kernel"]),
            bias=jnp.asarray(arrays["policy_encoder_hidden_bias"]),
        ),
        gru=GRUParams(
            input_kernel=jnp.asarray(arrays["policy_gru_input_kernel"]),
            recurrent_kernel=jnp.asarray(arrays["policy_gru_recurrent_kernel"]),
            bias=jnp.asarray(arrays["policy_gru_bias"]),
        ),
        actor=DenseParams(
            kernel=jnp.asarray(arrays["policy_actor_kernel"]),
            bias=jnp.asarray(arrays["policy_actor_bias"]),
        ),
        critic=DenseParams(
            kernel=jnp.asarray(arrays["policy_critic_kernel"]),
            bias=jnp.asarray(arrays["policy_critic_bias"]),
        ),
    )


def _validate_policy_bank(
    policy_bank: RecurrentPolicyParams,
    config: PPOConfig,
) -> int:
    named = {
        name: _nested_attr(policy_bank, path_parts)
        for name, path_parts in _POLICY_ARRAYS
    }
    first = next(iter(named.values()))
    if first.ndim < 1 or first.shape[0] < 1:
        raise ValueError("population policy bank must have a nonempty K axis")
    bank_size = int(first.shape[0])
    expected = {
        "policy_encoder_input_kernel": (
            config.observation_size,
            config.encoder_size,
        ),
        "policy_encoder_input_bias": (config.encoder_size,),
        "policy_encoder_hidden_kernel": (
            config.encoder_size,
            config.encoder_size,
        ),
        "policy_encoder_hidden_bias": (config.encoder_size,),
        "policy_gru_input_kernel": (
            config.encoder_size,
            3 * config.recurrent_size,
        ),
        "policy_gru_recurrent_kernel": (
            config.recurrent_size,
            3 * config.recurrent_size,
        ),
        "policy_gru_bias": (3 * config.recurrent_size,),
        "policy_actor_kernel": (config.recurrent_size, config.action_size),
        "policy_actor_bias": (config.action_size,),
        "policy_critic_kernel": (config.recurrent_size, 1),
        "policy_critic_bias": (1,),
    }
    for name, value in named.items():
        if value.shape != (bank_size,) + expected[name]:
            raise ValueError(f"{name} shape does not match population PPO config")
        if value.dtype != jnp.dtype(jnp.float32):
            raise ValueError(f"{name} must be float32")
    return bank_size


def _validate_config_contract(
    config: PPOConfig,
    combat_contract: Mapping[str, Any],
) -> None:
    if config.observation_size != combat_contract["observation_size"]:
        raise ValueError("PPO observation size differs from current contract")
    if config.action_size != combat_contract["action_size"]:
        raise ValueError("PPO action size differs from current contract")
    if config.action_transport != "factors":
        raise ValueError("population checkpoint requires factor actions")


def _validate_trainer_plan(trainer, plan) -> None:
    if trainer.trainable_policy_ids != plan.trainable_policy_ids:
        raise ValueError("trainer policy IDs differ from assignment plan")
    if trainer.trainable_actor_rows != plan.trainable_actor_rows:
        raise ValueError("trainer actor rows differ from assignment plan")
    if (
        trainer.trainable_actor_count_per_policy
        != plan.trainable_actor_count_per_policy
    ):
        raise ValueError("trainer actor count differs from assignment plan")


def _validate_plan_manifest(plan, manifest: Mapping[str, Any]) -> None:
    expected = {
        "trainable_policy_ids": list(plan.trainable_policy_ids),
        "trainable_actor_rows": [
            [list(row) for row in rows] for rows in plan.trainable_actor_rows
        ],
        "trainable_actor_count_per_policy": (
            plan.trainable_actor_count_per_policy
        ),
    }
    if dict(manifest) != expected:
        raise ValueError("population training plan differs from manifest")


def _require_assignment_equal(actual, expected) -> None:
    for name in PolicyActorAssignment._fields:
        if not np.array_equal(
            np.asarray(getattr(actual, name)),
            np.asarray(getattr(expected, name)),
        ):
            raise ValueError(f"population assignment {name} mismatch")


def _require_assignment_manifest_equal(assignment, manifest) -> None:
    expected = {
        name: np.asarray(getattr(assignment, name)).tolist()
        for name in PolicyActorAssignment._fields
    }
    if dict(manifest) != expected:
        raise ValueError("population assignment differs from manifest")


def _load_and_validate_arrays(
    path: Path,
    specs: list[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    expected_names = [row["name"] for row in specs]
    with np.load(path, allow_pickle=False) as archive:
        if sorted(archive.files) != sorted(expected_names):
            raise ValueError("population array names differ from manifest")
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    for spec in specs:
        value = arrays[spec["name"]]
        if list(value.shape) != spec["shape"] or value.dtype.str != spec["dtype"]:
            raise ValueError(f"population array {spec['name']} shape or dtype drift")
        if _single_array_sha256(value) != spec["sha256"]:
            raise ValueError(f"population array {spec['name']} content mismatch")
    return arrays


def _array_spec(name: str, value: np.ndarray) -> dict[str, Any]:
    return {
        "name": name,
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "sha256": _single_array_sha256(value),
    }


def _arrays_content_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = _contiguous_array(arrays[name])
        header = _canonical_json(
            {"name": name, "shape": list(value.shape), "dtype": value.dtype.str}
        )
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        payload = value.tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest().upper()


def _single_array_sha256(value: np.ndarray) -> str:
    contiguous = _contiguous_array(value)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest().upper()


def _write_deterministic_npz(
    path: Path,
    arrays: Mapping[str, np.ndarray],
) -> None:
    with path.open("wb") as output:
        with zipfile.ZipFile(
            output,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for name in sorted(arrays):
                buffer = io.BytesIO()
                value = _contiguous_array(arrays[name])
                np.lib.format.write_array(
                    buffer,
                    value,
                    allow_pickle=False,
                )
                info = zipfile.ZipInfo(
                    filename=f"{name}.npy",
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o600 << 16
                archive.writestr(
                    info,
                    buffer.getvalue(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )


def _write_portable_policy_checkpoint(
    path: Path,
    policy,
    config: PPOConfig,
    metadata: Mapping[str, Any],
) -> None:
    payload = {
        "format_version": np.asarray(
            CHECKPOINT_FORMAT_VERSION,
            dtype=np.int32,
        ),
        "config_json": np.asarray(
            _canonical_json(asdict(config)).decode("ascii")
        ),
        "metadata_json": np.asarray(
            _canonical_json(dict(metadata)).decode("ascii")
        ),
    }
    for name, path_parts in _POLICY_ARRAYS:
        payload[name.removeprefix("policy_")] = _host_array(
            _nested_attr(policy, path_parts)
        )
    _write_deterministic_npz(path, payload)


def _prepare_empty_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError("population checkpoint path must be a directory")
        if any(path.iterdir()):
            raise FileExistsError(
                "population checkpoint directory must be absent or empty"
            )
    else:
        path.mkdir(parents=True)


def _nested_attr(value: Any, path_parts: tuple[str, ...]):
    for part in path_parts:
        value = getattr(value, part)
    return value


def _host_array(value: Any) -> np.ndarray:
    array = np.asarray(jax.device_get(value))
    if array.dtype.hasobject:
        raise ValueError("population checkpoints cannot contain object arrays")
    return _contiguous_array(array)


def _contiguous_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
        return array.copy()
    return np.ascontiguousarray(array)


def _json_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(value or {})
    return json.loads(_canonical_json(payload))


def _validate_metadata_contract(
    metadata: Mapping[str, Any],
    combat_contract: Mapping[str, Any],
) -> None:
    reserved = {
        "policy_training_context",
        "policy_training_contexts",
        "population_policy_training_context",
        "policy_training_context_attestation",
        "policy_training_context_attestations",
        "runtime_config_content_sha256",
    }
    collisions = reserved.intersection(metadata)
    if collisions:
        names = ", ".join(sorted(collisions))
        raise ValueError(
            "population metadata cannot supply producer provenance: " + names
        )
    conflicts = {
        name
        for name, value in combat_contract.items()
        if name in metadata and metadata[name] != value
    }
    if conflicts:
        names = ", ".join(sorted(conflicts))
        raise ValueError(f"population metadata overrides contract: {names}")


def _validate_policy_training_contexts(
    value: Any,
    assignment: PolicyActorAssignment,
    *,
    bank_size: int,
    entity_count: int,
) -> list[dict[str, Any]]:
    """Validate optional upload provenance against active assignment owners."""

    if not isinstance(value, list):
        raise TypeError("policy_training_contexts must be a JSON list")
    if not value:
        return []

    actor_index = np.asarray(jax.device_get(assignment.actor_index))
    policy_id = np.asarray(jax.device_get(assignment.policy_id))
    active = np.asarray(jax.device_get(assignment.active))
    trainable = np.asarray(jax.device_get(assignment.trainable))
    expected: dict[int, set[tuple[int, bool]]] = {}
    for arena, slot in zip(*np.nonzero(active), strict=True):
        selected_policy = int(policy_id[arena, slot])
        expected.setdefault(selected_policy, set()).add(
            (
                int(actor_index[arena, slot]),
                bool(trainable[arena, slot]),
            )
        )

    normalized: list[dict[str, Any]] = []
    seen_policies: set[int] = set()
    for raw_context in value:
        if not isinstance(raw_context, dict):
            raise TypeError("each policy training context must be a JSON object")
        selected_policy = raw_context.get("policy_id")
        if isinstance(selected_policy, bool) or not isinstance(
            selected_policy, int
        ):
            raise TypeError("policy training context policy_id must be an integer")
        if selected_policy < 0 or selected_policy >= bank_size:
            raise ValueError("policy training context policy_id is outside bank")
        if selected_policy in seen_policies:
            raise ValueError("duplicate policy training context policy_id")
        seen_policies.add(selected_policy)
        raw_owners = raw_context.get("owners")
        if not isinstance(raw_owners, list) or not raw_owners:
            raise ValueError("policy training context owners must be nonempty")
        owners: list[dict[str, Any]] = []
        seen_owners: set[tuple[int, bool]] = set()
        for raw_owner in raw_owners:
            if not isinstance(raw_owner, dict):
                raise TypeError("policy training context owner must be an object")
            entity_index = raw_owner.get("entity_index")
            team_id = raw_owner.get("team_id")
            profile = raw_owner.get("profile")
            controller = raw_owner.get("controller")
            owner_trainable = raw_owner.get("trainable")
            if isinstance(entity_index, bool) or not isinstance(entity_index, int):
                raise TypeError("training context entity_index must be an integer")
            if entity_index < 0 or entity_index >= entity_count:
                raise ValueError("training context entity_index is out of range")
            if isinstance(team_id, bool) or not isinstance(team_id, int):
                raise TypeError("training context team_id must be an integer")
            if not isinstance(profile, str) or not profile:
                raise ValueError("training context profile must be nonempty")
            if not isinstance(controller, str) or not controller:
                raise ValueError("training context controller must be nonempty")
            if not isinstance(owner_trainable, bool):
                raise TypeError("training context trainable must be boolean")
            owner_key = (entity_index, owner_trainable)
            if owner_key in seen_owners:
                raise ValueError("duplicate policy training context owner")
            seen_owners.add(owner_key)
            owners.append(
                {
                    "entity_index": entity_index,
                    "profile": profile,
                    "team_id": team_id,
                    "controller": controller,
                    "trainable": owner_trainable,
                }
            )
        if seen_owners != expected.get(selected_policy, set()):
            raise ValueError(
                "policy training context owners differ from assignment"
            )
        normalized.append(
            {
                "policy_id": selected_policy,
                "owners": sorted(
                    owners,
                    key=lambda owner: (
                        owner["entity_index"],
                        owner["trainable"],
                    ),
                ),
            }
        )
    if seen_policies != set(expected):
        raise ValueError(
            "policy training contexts do not cover every active policy ID"
        )
    normalized.sort(key=lambda context: context["policy_id"])
    if value != normalized:
        raise ValueError("policy training contexts are not in canonical order")
    return normalized


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


__all__ = [
    "POPULATION_CHECKPOINT_SCHEMA",
    "POPULATION_CHECKPOINT_VERSION",
    "LoadedPopulationCheckpointV1",
    "load_population_checkpoint_v1",
    "population_checkpoint_contract_manifest",
    "population_checkpoint_contract_sha256",
    "save_population_checkpoint_v1",
]
