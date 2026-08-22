"""Runtime-derived provenance for portable multi-policy checkpoints.

The human-readable profile/team/controller labels in an upload bundle are not
evidence by themselves.  This module proves those labels against the concrete
``ArsenalRuntimeConfig`` used by the rollout and gives the complete runtime a
stable, content-addressed identity.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any, Mapping, NamedTuple, Sequence

import jax
import numpy as np

from hytalegym.jax.combat.arsenal.factory import specialize_ability_loadout
from hytalegym.jax.combat.arsenal.profiles import hytale_0_5_7_entity_loadouts
from hytalegym.jax.combat.arsenal.schema.types import ArsenalRuntimeConfig

from ..assignment import (
    PolicyActorAssignment,
    validate_policy_actor_assignment,
)


POLICY_TRAINING_CONTEXT_SCHEMA = "hytalerl_policy_training_context_v1"
POLICY_TRAINING_CONTEXT_ATTESTATION_SCHEMA = (
    "hytalerl_policy_training_context_attestation_v1"
)
POLICY_TRAINING_CONTEXT_DERIVATION = "runtime_config_entity_specs"
ARSENAL_RUNTIME_CONFIG_CONTENT_SCHEMA = (
    "hytalerl_arsenal_runtime_config_content_v1"
)

_CONTEXT_DOMAIN = b"HYTALERL_POLICY_TRAINING_CONTEXT\0"
_RUNTIME_DOMAIN = b"HYTALERL_ARSENAL_RUNTIME_CONFIG\0"
_SHA256_HEX_LENGTH = 64


class RuntimeTrainingEntitySpec(NamedTuple):
    """One human-readable entity label that must match the runtime exactly."""

    profile: str
    team_id: int
    controller: str


def arsenal_runtime_config_content_sha256(
    runtime_config: ArsenalRuntimeConfig,
) -> str:
    """Hash every typed runtime leaf in a platform-independent byte stream.

    Stream v1 is the fixed domain, a big-endian version and leaf count, then
    each lexicographically sorted leaf as: length-prefixed UTF-8 field path,
    length-prefixed NumPy dtype string, rank, unsigned 64-bit dimensions,
    unsigned 64-bit byte length, and canonical little-endian C-order bytes.
    """

    if not isinstance(runtime_config, ArsenalRuntimeConfig):
        raise TypeError("runtime_config must be an ArsenalRuntimeConfig")
    leaves = sorted(_array_leaves(runtime_config), key=lambda row: row[0])
    if not leaves:
        raise ValueError("runtime config has no numeric leaves")
    digest = hashlib.sha256()
    digest.update(_RUNTIME_DOMAIN)
    digest.update(struct.pack(">I", 1))
    digest.update(struct.pack(">I", len(leaves)))
    seen: set[str] = set()
    for path, value in leaves:
        if path in seen:
            raise ValueError(f"runtime config repeats field path {path}")
        seen.add(path)
        canonical = _canonical_array(value, name=path)
        _write_text(digest, path)
        _write_text(digest, canonical.dtype.str)
        digest.update(struct.pack(">I", canonical.ndim))
        for dimension in canonical.shape:
            digest.update(struct.pack(">Q", int(dimension)))
        raw = canonical.tobytes(order="C")
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest().upper()


def policy_training_context_content_sha256(
    context: Mapping[str, Any],
) -> str:
    """Hash one canonical policy-owner context using the Java-shared stream."""

    normalized = _normalize_context(context)
    digest = hashlib.sha256()
    digest.update(_CONTEXT_DOMAIN)
    digest.update(struct.pack(">I", 1))
    _write_text(digest, POLICY_TRAINING_CONTEXT_SCHEMA)
    digest.update(struct.pack(">q", normalized["policy_id"]))
    owners = normalized["owners"]
    digest.update(struct.pack(">I", len(owners)))
    for owner in owners:
        digest.update(struct.pack(">i", owner["entity_index"]))
        _write_text(digest, owner["profile"])
        digest.update(struct.pack(">i", owner["team_id"]))
        _write_text(digest, owner["controller"])
        digest.update(struct.pack(">B", int(owner["trainable"])))
    return digest.hexdigest().upper()


def attest_policy_training_contexts(
    runtime_config: ArsenalRuntimeConfig,
    assignment: PolicyActorAssignment,
    entity_specs: Sequence[RuntimeTrainingEntitySpec],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], str]:
    """Derive upload contexts only after labels match the concrete runtime.

    The current context schema has no arena index, so each entity label must be
    stable across all arena rows.  A heterogeneous per-arena runtime therefore
    fails closed until a future schema can represent it without collapsing
    distinct training conditions.
    """

    if not isinstance(runtime_config, ArsenalRuntimeConfig):
        raise TypeError("runtime_config must be an ArsenalRuntimeConfig")
    batch, entity_count = runtime_config.loadout.weapon_id.shape
    validate_policy_actor_assignment(assignment, entity_count=entity_count)
    if assignment.actor_index.shape[0] != batch:
        raise ValueError("assignment batch differs from runtime config")
    specs = _normalize_entity_specs(entity_specs, entity_count=entity_count)
    _verify_entity_specs_against_runtime(runtime_config, specs)

    actor_index = np.asarray(jax.device_get(assignment.actor_index))
    policy_id = np.asarray(jax.device_get(assignment.policy_id))
    active = np.asarray(jax.device_get(assignment.active))
    trainable = np.asarray(jax.device_get(assignment.trainable))
    owners_by_policy: dict[int, set[tuple[int, str, int, str, bool]]] = {}
    for arena, slot in zip(*np.nonzero(active), strict=True):
        actor = int(actor_index[arena, slot])
        spec = specs[actor]
        owners_by_policy.setdefault(int(policy_id[arena, slot]), set()).add(
            (
                actor,
                spec.profile,
                spec.team_id,
                spec.controller,
                bool(trainable[arena, slot]),
            )
        )

    contexts: list[dict[str, Any]] = []
    for selected_policy, raw_owners in sorted(owners_by_policy.items()):
        contexts.append(
            {
                "policy_id": selected_policy,
                "owners": [
                    {
                        "entity_index": owner[0],
                        "profile": owner[1],
                        "team_id": owner[2],
                        "controller": owner[3],
                        "trainable": owner[4],
                    }
                    for owner in sorted(raw_owners)
                ],
            }
        )

    runtime_sha256 = arsenal_runtime_config_content_sha256(runtime_config)
    attestations = tuple(
        {
            "schema": POLICY_TRAINING_CONTEXT_ATTESTATION_SCHEMA,
            "derivation": POLICY_TRAINING_CONTEXT_DERIVATION,
            "policy_id": context["policy_id"],
            "runtime_config_content_sha256": runtime_sha256,
            "context_content_sha256": (
                policy_training_context_content_sha256(context)
            ),
        }
        for context in contexts
    )
    return tuple(contexts), attestations, runtime_sha256


def validate_policy_training_context_attestations(
    contexts: Sequence[Mapping[str, Any]],
    attestations: Any,
    *,
    runtime_config_content_sha256: str,
) -> list[dict[str, Any]]:
    """Validate stored attestations against contexts and runtime identity."""

    runtime_sha256 = _sha256(
        runtime_config_content_sha256,
        name="runtime_config_content_sha256",
    )
    if not isinstance(attestations, list):
        raise TypeError("policy training context attestations must be a list")
    if len(attestations) != len(contexts):
        raise ValueError("policy training context attestation count differs")
    expected_by_policy = {
        int(context["policy_id"]): policy_training_context_content_sha256(context)
        for context in contexts
    }
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw in attestations:
        if not isinstance(raw, dict):
            raise TypeError("policy training context attestation must be an object")
        policy_id = raw.get("policy_id")
        if isinstance(policy_id, bool) or not isinstance(policy_id, int):
            raise TypeError("policy training context attestation ID must be integer")
        if policy_id in seen or policy_id not in expected_by_policy:
            raise ValueError("policy training context attestation ID is invalid")
        seen.add(policy_id)
        normalized_row = {
            "schema": raw.get("schema"),
            "derivation": raw.get("derivation"),
            "policy_id": policy_id,
            "runtime_config_content_sha256": _sha256(
                raw.get("runtime_config_content_sha256"),
                name="attested runtime config content SHA-256",
            ),
            "context_content_sha256": _sha256(
                raw.get("context_content_sha256"),
                name="attested context content SHA-256",
            ),
        }
        if normalized_row["schema"] != POLICY_TRAINING_CONTEXT_ATTESTATION_SCHEMA:
            raise ValueError("policy training context attestation schema is stale")
        if normalized_row["derivation"] != POLICY_TRAINING_CONTEXT_DERIVATION:
            raise ValueError("policy training context derivation is unsupported")
        if normalized_row["runtime_config_content_sha256"] != runtime_sha256:
            raise ValueError("policy training context runtime identity differs")
        if (
            normalized_row["context_content_sha256"]
            != expected_by_policy[policy_id]
        ):
            raise ValueError("policy training context content identity differs")
        normalized.append(normalized_row)
    normalized.sort(key=lambda row: row["policy_id"])
    if attestations != normalized:
        raise ValueError("policy training context attestations are not canonical")
    return normalized


def _verify_entity_specs_against_runtime(
    runtime_config: ArsenalRuntimeConfig,
    specs: tuple[RuntimeTrainingEntitySpec, ...],
) -> None:
    batch, entity_count = runtime_config.loadout.weapon_id.shape
    profile_rows = tuple(
        tuple(spec.profile for spec in specs) for _ in range(batch)
    )
    expected_loadout = specialize_ability_loadout(
        hytale_0_5_7_entity_loadouts(profile_rows)
    )
    for field in runtime_config.loadout._fields:
        expected = np.asarray(jax.device_get(getattr(expected_loadout, field)))
        actual = np.asarray(
            jax.device_get(getattr(runtime_config.loadout, field))
        )
        if expected.shape != actual.shape or not np.array_equal(
            expected,
            actual,
            equal_nan=True,
        ):
            raise ValueError(
                "training entity profile labels differ from runtime "
                f"loadout.{field}"
            )

    expected_team = np.broadcast_to(
        np.asarray(tuple(spec.team_id for spec in specs), dtype=np.int32),
        (batch, entity_count),
    )
    actual_team = np.asarray(
        jax.device_get(runtime_config.targeting_rules.team_id),
        dtype=np.int32,
    )
    if not np.array_equal(expected_team, actual_team):
        raise ValueError("training entity team labels differ from runtime")
    expected_controller = np.broadcast_to(
        np.asarray(
            tuple(spec.controller == "scripted" for spec in specs),
            dtype=np.bool_,
        ),
        (batch, entity_count),
    )
    actual_controller = np.asarray(
        jax.device_get(runtime_config.opponent_controller_mask),
        dtype=np.bool_,
    )
    if not np.array_equal(expected_controller, actual_controller):
        raise ValueError("training entity controller labels differ from runtime")


def _normalize_entity_specs(
    value: Sequence[RuntimeTrainingEntitySpec],
    *,
    entity_count: int,
) -> tuple[RuntimeTrainingEntitySpec, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("training entity specs must be a sequence")
    if len(value) != entity_count:
        raise ValueError("training entity specs must cover every runtime entity")
    result: list[RuntimeTrainingEntitySpec] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, RuntimeTrainingEntitySpec):
            raise TypeError(
                f"training entity spec {index} must be RuntimeTrainingEntitySpec"
            )
        if not raw.profile or raw.profile != raw.profile.strip():
            raise ValueError("training entity profile must be nonempty and trimmed")
        if isinstance(raw.team_id, bool) or not isinstance(raw.team_id, int):
            raise TypeError("training entity team_id must be an integer")
        if raw.controller not in {"policy", "scripted"}:
            raise ValueError("training entity controller is unsupported")
        result.append(raw)
    return tuple(result)


def _normalize_context(context: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(context, Mapping):
        raise TypeError("policy training context must be an object")
    policy_id = context.get("policy_id")
    if isinstance(policy_id, bool) or not isinstance(policy_id, int):
        raise TypeError("policy training context ID must be an integer")
    owners = context.get("owners")
    if not isinstance(owners, list) or not owners:
        raise ValueError("policy training context owners must be nonempty")
    normalized = []
    for raw in owners:
        if not isinstance(raw, Mapping):
            raise TypeError("policy training context owner must be an object")
        entity_index = raw.get("entity_index")
        team_id = raw.get("team_id")
        trainable = raw.get("trainable")
        profile = raw.get("profile")
        controller = raw.get("controller")
        if isinstance(entity_index, bool) or not isinstance(entity_index, int):
            raise TypeError("context owner entity_index must be an integer")
        if isinstance(team_id, bool) or not isinstance(team_id, int):
            raise TypeError("context owner team_id must be an integer")
        if not isinstance(trainable, bool):
            raise TypeError("context owner trainable must be boolean")
        if not isinstance(profile, str) or not profile:
            raise ValueError("context owner profile must be nonempty")
        if not isinstance(controller, str) or not controller:
            raise ValueError("context owner controller must be nonempty")
        normalized.append(
            {
                "entity_index": entity_index,
                "profile": profile,
                "team_id": team_id,
                "controller": controller,
                "trainable": trainable,
            }
        )
    normalized.sort(
        key=lambda owner: (
            owner["entity_index"],
            owner["profile"],
            owner["team_id"],
            owner["controller"],
            owner["trainable"],
        )
    )
    return {"policy_id": policy_id, "owners": normalized}


def _array_leaves(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    fields = getattr(value, "_fields", None)
    if fields is not None:
        leaves: list[tuple[str, Any]] = []
        for field in fields:
            path = f"{prefix}.{field}" if prefix else field
            leaves.extend(_array_leaves(getattr(value, field), path))
        return leaves
    if not prefix:
        raise TypeError("runtime config root must expose named fields")
    return [(prefix, value)]


def _canonical_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(jax.device_get(value))
    if array.dtype.hasobject:
        raise TypeError(f"runtime config field {name} has object dtype")
    dtype = (
        array.dtype
        if array.dtype.byteorder == "|"
        else array.dtype.newbyteorder("<")
    )
    return np.ascontiguousarray(array.astype(dtype, copy=False))


def _write_text(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(struct.pack(">I", len(encoded)))
    digest.update(encoded)


def _sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != _SHA256_HEX_LENGTH:
        raise ValueError(f"{name} must be an uppercase SHA-256")
    if value != value.upper() or any(
        character not in "0123456789ABCDEF" for character in value
    ):
        raise ValueError(f"{name} must be an uppercase SHA-256")
    return value


__all__ = [
    "ARSENAL_RUNTIME_CONFIG_CONTENT_SCHEMA",
    "POLICY_TRAINING_CONTEXT_ATTESTATION_SCHEMA",
    "POLICY_TRAINING_CONTEXT_DERIVATION",
    "POLICY_TRAINING_CONTEXT_SCHEMA",
    "RuntimeTrainingEntitySpec",
    "arsenal_runtime_config_content_sha256",
    "attest_policy_training_contexts",
    "policy_training_context_content_sha256",
    "validate_policy_training_context_attestations",
]
