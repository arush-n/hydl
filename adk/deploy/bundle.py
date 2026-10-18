"""Package a trained checkpoint as a loadable NPC for the policy mod.

`port_fidelity.py` produces weights a Java *test* can read. This produces the
directory the **plugin** loads at runtime, so a trained policy becomes an NPC
that a server operator drops in place:

    mods/HytalePolicyAgent-0.1.0.jar
    mods/policy-agent/          <- what this writes
        encoder_input_kernel.bin ... actor_bias.bin
        observation.bin, action_mask.bin
        role.txt                <- which NPC role the policy drives
        decision.txt            <- ticks between decisions
        spawn.txt               <- "Role x y z", spawns one on load
        nojump                  <- presence suppresses the jump head
        agent.json              <- identity; see below
        contract.txt            <- dependency-free Java mirror of that identity

The layout is `PolicyAgentPlugin`'s, read off the source rather than invented.
The plugin stays inert with a warning if the directory is missing, so a bad
export costs a silent NPC rather than the server.

    python -m adk.deploy.bundle trained.npz mods/policy-agent \\
        --role Kweebec_Razorleaf --spawn 52.5 126 -59.5 --decision 2 \\
        --fixture <captured-policy-case>

**`agent.json` is the part the plugin layout lacks.** The bare directory has no
record of which contract the weights were trained against, so a policy trained
on an older observation width loads and produces confident garbage. This writes
the widths, a checkpoint digest, and a canonical digest of the exact eleven raw
policy tensors next to the weights; `verify()` re-checks them, and the Java mod
independently recomputes the tensor digest before arming.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
from typing import Mapping

import numpy as np

from ..tools.port_fidelity import build as build_weights
from .compatibility import (
    DEPLOYMENT_COMPATIBILITY_SCHEMA,
    build_deployment_compatibility,
    compatibility_errors,
    flat_contract_rows as compatibility_flat_contract_rows,
    parse_transfer_mapping,
    profile_content_sha256,
    profile_payload_sha256,
)
from .tensor_identity import (
    POLICY_TENSOR_IDENTITY_SCHEMA,
    POLICY_TENSOR_NAMES,
    policy_tensor_content_sha256,
)


BUNDLE_SCHEMA = "hytalerl_policy_agent_bundle_v6"
TRAINING_PROVENANCE_SCHEMA = "hytalerl_policy_training_provenance_v1"
POLICY_ACTION_SCOPE_IDENTITY_SCHEMA = "hytalerl_policy_action_scope_identity_v1"
BRIDGE_TARGET_SCHEMA = "hytalerl_policy_bridge_target_v1"
BRIDGE_TARGET_FILE = "bridge-target.txt"
_POLICY_ACTION_SCOPE_HASH_DOMAIN = b"HYTALERL_POLICY_ACTION_SCOPE\0"
_LEGACY_PURSUIT_RUN_SCHEMA = "arena-recurrent-pursuit-run-v16"
BRIDGE_TARGET_KEYS = (
    "schema",
    "content_sha256",
    "bridge_jar_sha256",
    "combat_facade_schema",
    "combat_facade_version",
    "world_facade_schema",
    "world_facade_version",
    "world_transport_schema",
    "world_transport_version",
    "world_transport_sha256",
    "actor_capture_schema",
    "actor_capture_version",
    "actor_capture_sha256",
    "group_action_schema",
    "group_action_version",
    "group_action_sha256",
    "actor_capacity",
    "dodge_cooldown_id",
    "dodge_cooldown_seconds",
    "world_evidence_reject_reason",
)
_BRIDGE_TARGET_DOMAIN = b"HYTALERL_POLICY_BRIDGE_TARGET\0"


def _is_canonical_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789ABCDEF" for character in value)
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _policy_action_scope_identity_sha256(
    scope_schema: str,
    action_contract_sha256: str,
    mask_bits: str,
) -> str:
    digest = hashlib.sha256(_POLICY_ACTION_SCOPE_HASH_DOMAIN)
    digest.update(scope_schema.encode("utf-8"))
    digest.update(b"\0")
    digest.update(action_contract_sha256.encode("ascii"))
    digest.update(b"\0")
    digest.update(mask_bits.encode("ascii"))
    return digest.hexdigest().upper()


def _normalize_policy_action_scope(
    value: object,
    *,
    action_contract_sha256: object,
    expected_scope_sha256: object | None = None,
) -> dict[str, object]:
    from arena.training.skills.program import (
        FACTORED_ACTION_SCOPE_SCHEMA,
        FactoredActionScope,
    )
    from hytalegym.jax.combat.observation.v3.policy import (
        ARSENAL_POLICY_ACTION_HEAD_NAMES,
        ARSENAL_POLICY_ACTION_HEAD_SIZES,
    )

    if not isinstance(value, dict):
        raise ValueError("policy_action_scope is missing or is not an object")
    scope_schema = value.get("schema")
    if scope_schema != FACTORED_ACTION_SCOPE_SCHEMA:
        raise ValueError(
            "policy_action_scope.schema "
            f"{scope_schema!r} is unsupported; supported value is "
            f"{FACTORED_ACTION_SCOPE_SCHEMA!r}"
        )
    if not isinstance(action_contract_sha256, str) or not _is_canonical_sha256(
        action_contract_sha256
    ):
        raise ValueError("action_contract_sha256 is missing or malformed")

    heads = value.get("heads")
    if not isinstance(heads, (list, tuple)) or len(heads) != len(
        ARSENAL_POLICY_ACTION_HEAD_NAMES
    ):
        raise ValueError(
            "policy_action_scope.heads must declare every supported action head"
        )
    allowed_choices = []
    for index, (head, name, size) in enumerate(
        zip(
            heads,
            ARSENAL_POLICY_ACTION_HEAD_NAMES,
            ARSENAL_POLICY_ACTION_HEAD_SIZES,
            strict=True,
        )
    ):
        if not isinstance(head, dict):
            raise ValueError(f"policy_action_scope.heads[{index}] is not an object")
        if head.get("name") != name:
            raise ValueError(
                f"policy_action_scope.heads[{index}].name "
                f"{head.get('name')!r} is unsupported; supported value is {name!r}"
            )
        if type(head.get("size")) is not int or head["size"] != size:
            raise ValueError(
                f"policy_action_scope.heads[{index}].size "
                f"{head.get('size')!r} is unsupported; supported value is {size}"
            )
        choices = head.get("allowed_choices")
        if not isinstance(choices, (list, tuple)):
            raise ValueError(
                f"policy_action_scope.heads[{index}].allowed_choices must be a list"
            )
        allowed_choices.append(tuple(choices))

    try:
        scope = FactoredActionScope(tuple(allowed_choices), schema=scope_schema)
    except (TypeError, ValueError) as error:
        raise ValueError(f"policy_action_scope is invalid: {error}") from error
    normalized_manifest = json.loads(_canonical_json(scope.manifest()))
    if _canonical_json(value) != _canonical_json(normalized_manifest):
        raise ValueError(
            "policy_action_scope contains unsupported or inconsistent fields"
        )
    if scope.manifest()["action_contract_sha256"] != action_contract_sha256:
        raise ValueError(
            "policy_action_scope.action_contract_sha256 does not match "
            "transfer_contract.action_contract_sha256"
        )
    if expected_scope_sha256 is not None and expected_scope_sha256 != (
        scope.contract_sha256
    ):
        raise ValueError(
            "policy_action_scope_sha256 does not match the declared scope manifest"
        )

    mask_bits = "".join(
        "1" if choice in choices else "0"
        for size, choices in zip(
            ARSENAL_POLICY_ACTION_HEAD_SIZES,
            scope.allowed_choices,
            strict=True,
        )
        for choice in range(size)
    )
    return {
        "manifest": normalized_manifest,
        "contract_sha256": scope.contract_sha256,
        "identity": {
            "schema": POLICY_ACTION_SCOPE_IDENTITY_SCHEMA,
            "mask_bits": mask_bits,
            "content_sha256": _policy_action_scope_identity_sha256(
                scope_schema,
                action_contract_sha256,
                mask_bits,
            ),
        },
    }


def _full_policy_action_scope() -> dict[str, object]:
    from arena.training.skills.program import FactoredActionScope
    from hytalegym.jax.combat.observation.v3.policy import (
        ARSENAL_POLICY_ACTION_HEAD_SIZES,
    )

    return FactoredActionScope(
        tuple(tuple(range(size)) for size in ARSENAL_POLICY_ACTION_HEAD_SIZES)
    ).manifest()


def _legacy_pursuit_action_scope(
    checkpoint: Path,
    checkpoint_metadata: Mapping[str, object],
    training_report: Path,
) -> tuple[dict[str, object], str]:
    try:
        report = json.loads(training_report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"training report is unreadable: {error}") from error
    if not isinstance(report, dict) or report.get("schema") != (
        _LEGACY_PURSUIT_RUN_SCHEMA
    ):
        schema = report.get("schema") if isinstance(report, dict) else None
        raise ValueError(
            "training report has no policy_action_scope and no supported "
            f"legacy adapter for schema {schema!r}"
        )

    checkpoint_sha256 = _digest(checkpoint)
    report_checkpoint = report.get("checkpoint")
    report_checkpoint_sha256 = (
        report_checkpoint.get("sha256")
        if isinstance(report_checkpoint, dict)
        else None
    )
    lineage = checkpoint_metadata.get("certification_lineage")
    lineage_sha256 = (
        lineage.get("source_sha256") if isinstance(lineage, dict) else None
    )
    if report_checkpoint_sha256 not in {checkpoint_sha256, lineage_sha256}:
        raise ValueError(
            "training report checkpoint SHA-256 does not identify this "
            "checkpoint or its certified source"
        )

    selection = report.get("selection")
    settings = report.get("settings")
    contracts = report.get("contracts")
    if not all(isinstance(value, dict) for value in (selection, settings, contracts)):
        raise ValueError("training report is missing selection/settings/contracts")
    selected_update = selection.get("selected_update")
    if type(selected_update) is not int or selected_update != checkpoint_metadata.get(
        "selected_update"
    ):
        raise ValueError(
            "training report selected_update does not match checkpoint metadata"
        )
    batch = settings.get("batch")
    rollout_steps = settings.get("rollout_steps")
    if type(batch) is not int or type(rollout_steps) is not int:
        raise ValueError("training report batch/rollout_steps are not integers")
    selected_steps = selected_update * batch * rollout_steps
    if (
        checkpoint_metadata.get("selected_environment_steps") != selected_steps
        or checkpoint_metadata.get("environment_steps") != selected_steps
    ):
        raise ValueError(
            "training report environment steps do not match checkpoint metadata"
        )

    report_stage = contracts.get("stage")
    report_stage_sha256 = contracts.get("stage_sha256")
    metadata_stage = checkpoint_metadata.get("pursuit_stage")
    metadata_stage_sha256 = checkpoint_metadata.get("pursuit_stage_sha256")
    if (
        not isinstance(report_stage, dict)
        or _canonical_json(report_stage) != _canonical_json(metadata_stage)
        or report_stage_sha256 != metadata_stage_sha256
    ):
        raise ValueError(
            "training report stage contract does not match checkpoint metadata"
        )

    base_allow_jump = settings.get("evader_allow_jump")
    stage_options = settings.get("stage_options", {})
    if type(base_allow_jump) is not bool or not isinstance(stage_options, dict):
        raise ValueError(
            "training report does not declare a valid pursuit action-scope setting"
        )
    effective_allow_jump = stage_options.get("allow_jump", base_allow_jump)
    if type(effective_allow_jump) is not bool:
        raise ValueError(
            "training report stage_options.allow_jump must be a boolean"
        )

    from arena.training.contracts.pursuit import (
        PursuitStageConfig,
        pursuit_learner_action_scope,
    )

    stage = PursuitStageConfig(
        maximum_ticks=settings.get("evaluation_steps"),
        **{"allow_jump": base_allow_jump, **stage_options},
    )
    if (
        stage.allow_jump != effective_allow_jump
        or _canonical_json(stage.manifest()) != _canonical_json(report_stage)
    ):
        raise ValueError(
            "training report settings do not reproduce its selected stage contract"
        )
    scope = pursuit_learner_action_scope(allow_jump=stage.allow_jump)
    return scope.manifest(), scope.contract_sha256


def _policy_action_scope_errors(
    manifest: dict[str, object],
    transfer_contract: object,
) -> list[str]:
    if not isinstance(transfer_contract, dict):
        return ["policy_action_scope cannot be checked without an action contract"]
    expected_scope_sha256 = manifest.get("policy_action_scope_contract_sha256")
    try:
        normalized = _normalize_policy_action_scope(
            manifest.get("policy_action_scope"),
            action_contract_sha256=transfer_contract.get(
                "action_contract_sha256"
            ),
            expected_scope_sha256=expected_scope_sha256,
        )
    except (TypeError, ValueError) as error:
        return [str(error)]

    problems = []
    if manifest.get("policy_action_scope_identity") != normalized["identity"]:
        problems.append(
            "policy_action_scope_identity differs from the declared allowed choices"
        )
    contract = manifest.get("contract")
    action_size = contract.get("action_mask_size") if isinstance(contract, dict) else None
    mask_bits = normalized["identity"]["mask_bits"]
    if action_size != len(mask_bits):
        problems.append(
            "policy_action_scope mask width "
            f"{len(mask_bits)} != contract action_mask_size {action_size}"
        )
    return problems


def _transfer_contract_errors(contract: object) -> list[str]:
    if not isinstance(contract, dict):
        return ["checkpoint has no exact transfer_contract"]
    from hytalegym.jax.training.checkpoint import (
        ARSENAL_POLICY_SURFACE,
        combat_checkpoint_contract_errors,
    )
    from hytalegym.jax.combat.observation.v3.policy.layout import (
        ARSENAL_POLICY_JVM_PROJECTION_ABI,
    )

    world_manifest = contract.get("world_geometry_policy_config")
    world_config = (
        {
            name: world_manifest[name]
            for name in ("token_capacity", "edge_capacity", "maximum_distance")
            if name in world_manifest
        }
        if isinstance(world_manifest, dict)
        else world_manifest
    )
    errors = combat_checkpoint_contract_errors(
        contract,
        arsenal_runtime_capacity=contract.get("arsenal_runtime_capacity"),
        policy_surface=ARSENAL_POLICY_SURFACE,
        world_geometry_config=world_config,
    )
    if contract.get("observation_contract_sha256") != contract.get(
        "action_contract_sha256"
    ):
        errors.append("checkpoint observation/action contract identities differ")
    declared_abi = contract.get("policy_projection_abi")
    if declared_abi is not None and declared_abi != ARSENAL_POLICY_JVM_PROJECTION_ABI:
        errors.append(
            "policy_projection_abi: checkpoint declares "
            f"{declared_abi!r}, supported value is "
            f"{ARSENAL_POLICY_JVM_PROJECTION_ABI!r}"
        )
    return errors


def _projection_abi_errors(contract: object) -> list[str]:
    from hytalegym.jax.combat.observation.v3.policy.layout import (
        ARSENAL_POLICY_JVM_PROJECTION_ABI,
    )

    if not isinstance(contract, dict):
        return ["policy_projection_abi is missing from the bundle contract"]
    declared = contract.get("policy_projection_abi")
    if declared != ARSENAL_POLICY_JVM_PROJECTION_ABI:
        return [
            "policy_projection_abi: bundle declares "
            f"{declared!r}, supported value is "
            f"{ARSENAL_POLICY_JVM_PROJECTION_ABI!r}"
        ]
    return []


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest().upper()


def _bridge_target_content_sha256(target: Mapping[str, object]) -> str:
    """Canonical cross-language identity for one independently built bridge."""

    digest = hashlib.sha256()
    digest.update(_BRIDGE_TARGET_DOMAIN)
    digest.update((1).to_bytes(4, "big", signed=True))

    def text(value: object) -> None:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big", signed=True))
        digest.update(encoded)

    text(target["schema"])
    for key in (
        "bridge_jar_sha256",
        "combat_facade_schema",
        "combat_facade_version",
        "world_facade_schema",
        "world_facade_version",
        "world_transport_schema",
        "world_transport_version",
        "world_transport_sha256",
        "actor_capture_schema",
        "actor_capture_version",
        "actor_capture_sha256",
        "group_action_schema",
        "group_action_version",
        "group_action_sha256",
        "actor_capacity",
        "dodge_cooldown_id",
    ):
        text(target[key])
    digest.update(struct.pack(">f", float(target["dodge_cooldown_seconds"])))
    text(target["world_evidence_reject_reason"])
    return digest.hexdigest().upper()


def _read_bridge_target(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"bridge target is not a file: {path}")
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("\t")
        if not separator or not key or value != value.strip() or "\0" in value:
            raise ValueError(f"malformed bridge target line: {line!r}")
        if key in fields:
            raise ValueError(f"bridge target repeats field {key!r}")
        fields[key] = value
    if tuple(fields) != BRIDGE_TARGET_KEYS:
        raise ValueError("bridge target fields are not the exact ordered v1 set")
    if fields["schema"] != BRIDGE_TARGET_SCHEMA:
        raise ValueError(f"unsupported bridge target schema {fields['schema']!r}")
    for key in (
        "content_sha256",
        "bridge_jar_sha256",
        "world_transport_sha256",
        "actor_capture_sha256",
        "group_action_sha256",
    ):
        if not _is_canonical_sha256(fields[key]):
            raise ValueError(f"bridge target {key} is not canonical SHA-256")
    for key in (
        "combat_facade_version",
        "world_facade_version",
        "world_transport_version",
        "actor_capture_version",
        "group_action_version",
        "actor_capacity",
    ):
        try:
            value = int(fields[key])
        except ValueError as error:
            raise ValueError(f"bridge target {key} is not an integer") from error
        if value < 1 or str(value) != fields[key]:
            raise ValueError(f"bridge target {key} is not canonical positive integer")
    try:
        cooldown = float(fields["dodge_cooldown_seconds"])
    except ValueError as error:
        raise ValueError("bridge target Dodge cooldown is not a decimal") from error
    if not np.isfinite(cooldown) or cooldown <= 0.0:
        raise ValueError("bridge target Dodge cooldown is outside its boundary")
    for key in (
        "combat_facade_schema",
        "world_facade_schema",
        "world_transport_schema",
        "actor_capture_schema",
        "group_action_schema",
        "dodge_cooldown_id",
        "world_evidence_reject_reason",
    ):
        if not fields[key]:
            raise ValueError(f"bridge target {key} is empty")
    target: dict[str, object] = dict(fields)
    for key in (
        "combat_facade_version",
        "world_facade_version",
        "world_transport_version",
        "actor_capture_version",
        "group_action_version",
        "actor_capacity",
    ):
        target[key] = int(fields[key])
    target["dodge_cooldown_seconds"] = cooldown
    reconstructed = _bridge_target_content_sha256(target)
    if target["content_sha256"] != reconstructed:
        raise ValueError(
            "bridge target content SHA-256 differs from canonical tuple: "
            f"{target['content_sha256']} != {reconstructed}"
        )
    return target


def _bridge_target_errors(
    directory: Path, manifest_target: object
) -> list[str]:
    if manifest_target is None:
        return [] if not (directory / BRIDGE_TARGET_FILE).exists() else [
            "unbound bridge-target.txt is present"
        ]
    if not isinstance(manifest_target, dict):
        return ["agent.json bridge_target is not an object"]
    try:
        installed = _read_bridge_target(directory / BRIDGE_TARGET_FILE)
    except (OSError, ValueError) as error:
        return [f"bridge target is invalid: {error}"]
    return [] if installed == manifest_target else [
        "bridge-target.txt differs from agent.json bridge_target"
    ]


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _counter(metadata: Mapping[str, object], *names: str) -> int:
    for name in names:
        value = metadata.get(name)
        if isinstance(value, bool):
            raise ValueError(f"checkpoint training counter {name} must be an integer")
        if isinstance(value, int):
            if value < 0:
                raise ValueError(
                    f"checkpoint training counter {name} cannot be negative"
                )
            return value
    return 0


def _training_provenance(
    metadata: object,
    *,
    diagnostic: bool,
) -> dict[str, object]:
    """Normalize explicit, content-bound evidence that weights were trained.

    Actor-weight standard deviation is deliberately absent.  It produced both
    directions of error in real artifacts: a four-update policy stayed below
    the old threshold, while an explicitly synthetic diagnostic sat far above
    it.  Positive optimizer/environment counters or a positive demonstration
    count are semantic evidence; diagnostics require an explicit export mode.
    """

    if not isinstance(metadata, Mapping):
        metadata = {}
    explicit = metadata.get("training_provenance")
    source = explicit if isinstance(explicit, Mapping) else metadata
    if explicit is not None and not isinstance(explicit, Mapping):
        raise ValueError("checkpoint training_provenance must be an object")
    if isinstance(explicit, Mapping):
        schema = explicit.get("schema")
        if schema != TRAINING_PROVENANCE_SCHEMA:
            raise ValueError(
                "checkpoint training provenance schema differs: "
                f"{schema!r} != {TRAINING_PROVENANCE_SCHEMA!r}"
            )
    optimizer_updates = _counter(
        source,
        "optimizer_updates",
        "population_policy_updates",
        "updates",
    )
    environment_steps = _counter(
        source,
        "environment_steps",
        "population_policy_actor_steps",
        "actor_steps",
    )
    demonstration_examples = _counter(
        source,
        "demonstration_examples",
        "imitation_examples",
    )
    purpose = "diagnostic" if diagnostic else "trained"
    if not diagnostic and not (
        (optimizer_updates > 0 and environment_steps > 0)
        or demonstration_examples > 0
    ):
        raise ValueError(
            "checkpoint has no explicit positive training provenance; require "
            "optimizer updates plus environment steps, demonstration examples, "
            "or export with diagnostic=True for a controlled diagnostic only"
        )
    return {
        "schema": TRAINING_PROVENANCE_SCHEMA,
        "purpose": purpose,
        "optimizer_updates": optimizer_updates,
        "environment_steps": environment_steps,
        "demonstration_examples": demonstration_examples,
        "source_metadata_sha256": _canonical_json_sha256(dict(metadata)),
    }


def _training_provenance_errors(
    destination: Path,
    value: object,
) -> list[str]:
    if not isinstance(value, dict):
        return ["bundle has no explicit training provenance"]
    problems: list[str] = []
    if value.get("schema") != TRAINING_PROVENANCE_SCHEMA:
        problems.append("training provenance schema differs")
    purpose = value.get("purpose")
    counters: dict[str, int] = {}
    for name in (
        "optimizer_updates",
        "environment_steps",
        "demonstration_examples",
    ):
        item = value.get(name)
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            problems.append(f"training provenance {name} must be nonnegative int")
        else:
            counters[name] = item
    source_sha = value.get("source_metadata_sha256")
    if not _is_canonical_sha256(source_sha):
        problems.append("training source metadata SHA-256 is malformed")
    if purpose == "trained":
        if len(counters) == 3 and not (
            (
                counters["optimizer_updates"] > 0
                and counters["environment_steps"] > 0
            )
            or counters["demonstration_examples"] > 0
        ):
            problems.append("trained bundle has no positive training evidence")
    elif purpose == "diagnostic":
        if not (destination / "live-test-control.txt").is_file():
            problems.append(
                "diagnostic bundle requires live-test-control.txt and cannot "
                "arm as an ordinary policy"
            )
    else:
        problems.append(f"unknown policy purpose {purpose!r}")
    return problems


def _profile_content_sha256(profile: Path) -> str:
    """Compatibility alias for callers of the original profile verifier."""

    return profile_payload_sha256(profile)


def _profile_errors(
    profile: Path,
    contract: dict[str, object],
    *,
    allow_missing_projection_abi: bool = False,
) -> list[str]:
    manifest_path = profile / "profile.json"
    if not manifest_path.is_file():
        return [f"live profile has no profile.json: {profile}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"live profile manifest is unreadable: {error}"]
    problems = []
    expected_policy = contract.get("observation_contract_sha256")
    if manifest.get("arsenal_policy_contract_sha256") != expected_policy:
        problems.append(
            "live profile policy contract does not match the checkpoint: "
            f"profile={manifest.get('arsenal_policy_contract_sha256')!r}, "
            f"checkpoint={expected_policy!r}"
        )
    expected_abi = contract.get("policy_projection_abi")
    declared_abi = manifest.get("policy_projection_abi")
    if declared_abi != expected_abi and not (
        allow_missing_projection_abi and declared_abi is None
    ):
        problems.append(
            "live profile policy_projection_abi does not match the bundle: "
            f"profile={declared_abi!r}, bundle={expected_abi!r}"
        )
    if manifest.get("jax_backend") != "gpu":
        problems.append("live profile was not exported on the required GPU backend")
    for name, file_name, width in (
        ("observation_size", "expected_observation.bin", 4),
        ("action_size", "action_mask.bin", 1),
    ):
        path = profile / file_name
        expected = contract.get(name)
        actual = None if not path.is_file() else path.stat().st_size // width
        if manifest.get(name) != expected or actual != expected:
            problems.append(
                f"live profile {name} disagrees: manifest={manifest.get(name)!r}, "
                f"file={actual!r}, checkpoint={expected!r}"
            )
    actual_content = _profile_content_sha256(profile)
    if manifest.get("content_sha256") != actual_content:
        problems.append("live profile content hash does not match its files")
    return problems


def _flat_contract_errors(
    directory: Path, manifest: dict[str, object]
) -> list[str]:
    """Verify the Java-readable mirror names the same bundle as agent.json."""

    path = directory / "contract.txt"
    if not path.is_file():
        return ["bundle has no contract.txt; the Java mod would stay inert"]
    fields: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            key, separator, value = line.partition("\t")
            if not separator or not key:
                return [f"contract.txt line is not key<TAB>value: {line!r}"]
            if key in fields:
                return [f"contract.txt repeats key {key!r}"]
            fields[key] = value.strip()
    except OSError as error:
        return [f"contract.txt is unreadable: {error}"]

    tensor_identity = manifest.get("policy_tensor_identity")
    tensor_identity = tensor_identity if isinstance(tensor_identity, dict) else {}
    training = manifest.get("training_provenance")
    training = training if isinstance(training, dict) else {}
    compatibility = manifest.get("deployment_compatibility")
    compatibility = compatibility if isinstance(compatibility, dict) else {}
    contract = manifest.get("contract")
    contract = contract if isinstance(contract, dict) else {}
    transfer = manifest.get("transfer_contract")
    transfer = transfer if isinstance(transfer, dict) else {}
    action_scope = manifest.get("policy_action_scope")
    action_scope = action_scope if isinstance(action_scope, dict) else {}
    action_scope_identity = manifest.get("policy_action_scope_identity")
    action_scope_identity = (
        action_scope_identity if isinstance(action_scope_identity, dict) else {}
    )
    expected = {
        "schema": manifest.get("schema"),
        "role": manifest.get("role"),
        "checkpoint_sha256": manifest.get("checkpoint_sha256"),
        "policy_tensor_identity_schema": tensor_identity.get("schema"),
        "policy_tensor_count": len(POLICY_TENSOR_NAMES),
        "policy_tensor_content_sha256": tensor_identity.get("content_sha256"),
        "training_provenance_schema": training.get("schema"),
        "policy_purpose": training.get("purpose"),
        "training_optimizer_updates": training.get("optimizer_updates"),
        "training_environment_steps": training.get("environment_steps"),
        "training_demonstration_examples": training.get(
            "demonstration_examples"
        ),
        "training_source_metadata_sha256": training.get(
            "source_metadata_sha256"
        ),
        "observation_size": contract.get("observation_size"),
        "action_mask_size": contract.get("action_mask_size"),
        "encoder_size": contract.get("encoder_size"),
        "recurrent_size": contract.get("recurrent_size"),
        "actor_kernel_std": manifest.get("actor_kernel_std"),
        "observation_contract_sha256": transfer.get(
            "observation_contract_sha256"
        ),
        "action_contract_sha256": transfer.get("action_contract_sha256"),
        "policy_projection_abi": transfer.get("policy_projection_abi"),
        "policy_action_scope_schema": action_scope.get("schema"),
        "policy_action_scope_identity_schema": action_scope_identity.get(
            "schema"
        ),
        "policy_action_scope_bits": action_scope_identity.get("mask_bits"),
        "policy_action_scope_sha256": action_scope_identity.get(
            "content_sha256"
        ),
    }
    bridge_target = manifest.get("bridge_target")
    if isinstance(bridge_target, dict):
        expected.update({
            "bridge_target_schema": bridge_target.get("schema"),
            "bridge_target_content_sha256": bridge_target.get(
                "content_sha256"
            ),
        })
    if compatibility:
        expected.update(dict(compatibility_flat_contract_rows(compatibility)))
    else:
        expected["deployment_compatibility_schema"] = (
            DEPLOYMENT_COMPATIBILITY_SCHEMA
        )
    problems = []
    for key, value in expected.items():
        expected_value = repr(value) if key == "actor_kernel_std" else str(value)
        if fields.get(key) != expected_value:
            problems.append(
                f"contract.txt {key} {fields.get(key)!r} != "
                f"agent.json {expected_value!r}"
            )
    return problems


def export(checkpoint: Path, destination: Path, *, role: str,
           decision: int | None = None,
           spawn: tuple[float, float, float] | None = None,
           nojump: bool = False,
           profile: Path | None = None,
           fixture: Path | None = None,
           diagnostic: bool = False,
           diagnostic_control: Path | None = None,
           training_report: Path | None = None,
           bridge_target: Path | None = None,
           role_transfer: tuple[str, str] | None = None,
           profile_transfer: tuple[str, str] | None = None,
           transfer_reason: str | None = None) -> dict[str, object]:
    report = build_weights(checkpoint, destination, fixture=fixture)
    bridge_target_manifest = None
    if bridge_target is not None:
        bridge_target = Path(bridge_target)
        bridge_target_manifest = _read_bridge_target(bridge_target)
        shutil.copyfile(bridge_target, destination / BRIDGE_TARGET_FILE)
    source_transfer_contract = report["transfer_contract"]
    transfer_errors = _transfer_contract_errors(source_transfer_contract)
    if transfer_errors:
        raise ValueError(
            "checkpoint is not deployable under the current policy contract: "
            + "; ".join(transfer_errors)
        )
    from hytalegym.jax.combat.observation.v3.policy.layout import (
        ARSENAL_POLICY_JVM_PROJECTION_ABI,
    )

    transfer_contract = dict(source_transfer_contract)
    transfer_contract["policy_projection_abi"] = (
        ARSENAL_POLICY_JVM_PROJECTION_ABI
    )
    checkpoint_metadata = report.get("checkpoint_metadata")
    if not isinstance(checkpoint_metadata, Mapping):
        checkpoint_metadata = {}
    scope_manifest = checkpoint_metadata.get("policy_action_scope")
    scope_contract_sha256 = checkpoint_metadata.get(
        "policy_action_scope_sha256"
    )
    if scope_manifest is None and training_report is not None:
        scope_manifest, scope_contract_sha256 = _legacy_pursuit_action_scope(
            Path(checkpoint),
            checkpoint_metadata,
            Path(training_report),
        )
    if scope_manifest is None and diagnostic:
        scope_manifest = _full_policy_action_scope()
    if scope_manifest is None:
        raise ValueError(
            "checkpoint metadata has no policy_action_scope; retrain with a "
            "scope-aware trainer or pass --training-report for a supported "
            "older pursuit checkpoint"
        )
    action_scope = _normalize_policy_action_scope(
        scope_manifest,
        action_contract_sha256=transfer_contract.get("action_contract_sha256"),
        expected_scope_sha256=scope_contract_sha256,
    )
    if diagnostic_control is not None and not diagnostic:
        raise ValueError("diagnostic_control requires diagnostic=True")
    if diagnostic:
        if diagnostic_control is None:
            raise ValueError(
                "diagnostic export requires an explicit live-test-control file"
            )
        diagnostic_control = Path(diagnostic_control)
        if not diagnostic_control.is_file():
            raise FileNotFoundError(
                f"diagnostic control is not a file: {diagnostic_control}"
            )
        shutil.copyfile(
            diagnostic_control,
            destination / "live-test-control.txt",
        )
    profile_manifest = None
    if profile is not None:
        profile = Path(profile)
        profile_errors = _profile_errors(
            profile, transfer_contract, allow_missing_projection_abi=True
        )
        if profile_errors:
            raise ValueError("live profile is not deployable: " + "; ".join(profile_errors))
        profile_manifest = json.loads(
            (profile / "profile.json").read_text(encoding="utf-8")
        )

    (destination / "role.txt").write_text(role + "\n", encoding="utf-8")
    if decision is not None:
        if decision < 1:
            raise ValueError(f"decision period must be >= 1 tick, got {decision}")
        (destination / "decision.txt").write_text(
            f"{decision}\n", encoding="utf-8")
    if spawn is not None:
        # The plugin parses exactly "Role x y z" and rejects any other field
        # count, so the role is written into the line rather than assumed.
        (destination / "spawn.txt").write_text(
            f"{role} {spawn[0]} {spawn[1]} {spawn[2]}\n", encoding="utf-8")
    marker = destination / "nojump"
    if nojump:
        marker.write_text("", encoding="utf-8")
    elif marker.exists():
        marker.unlink()

    installed_profile = None
    if profile is not None:
        target = destination / "profile"
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(profile, target)
        installed_profile = target
        installed_manifest_path = installed_profile / "profile.json"
        installed_manifest = json.loads(
            installed_manifest_path.read_text(encoding="utf-8")
        )
        installed_manifest["policy_projection_abi"] = (
            ARSENAL_POLICY_JVM_PROJECTION_ABI
        )
        installed_manifest_path.write_text(
            json.dumps(installed_manifest, indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        installed_errors = _profile_errors(installed_profile, transfer_contract)
        if installed_errors:
            raise ValueError(
                "installed live profile is not deployable: "
                + "; ".join(installed_errors)
            )
        profile_manifest = json.loads(
            (installed_profile / "profile.json").read_text(encoding="utf-8")
        )

    tensor_content_sha256 = policy_tensor_content_sha256(destination)
    training_provenance = _training_provenance(
        report.get("checkpoint_metadata"),
        diagnostic=diagnostic,
    )
    deployment_compatibility = build_deployment_compatibility(
        report.get("checkpoint_metadata"),
        transfer_contract,
        role=role,
        profile_manifest=profile_manifest,
        profile_content_sha256_value=(
            None
            if installed_profile is None
            else profile_content_sha256(installed_profile)
        ),
        diagnostic=diagnostic,
        role_transfer=role_transfer,
        profile_transfer=profile_transfer,
        transfer_reason=transfer_reason,
    )
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "role": role,
        "decision_ticks": decision,
        "spawn": None if spawn is None else list(spawn),
        "nojump": nojump,
        "live_profile": profile is not None,
        "live_profile_manifest": profile_manifest,
        "checkpoint_sha256": _digest(checkpoint),
        "policy_tensor_identity": {
            "schema": POLICY_TENSOR_IDENTITY_SCHEMA,
            "ordered_names": list(POLICY_TENSOR_NAMES),
            "content_sha256": tensor_content_sha256,
        },
        "training_provenance": training_provenance,
        "policy_action_scope": action_scope["manifest"],
        "policy_action_scope_contract_sha256": action_scope[
            "contract_sha256"
        ],
        "policy_action_scope_identity": action_scope["identity"],
        "deployment_compatibility": deployment_compatibility,
        "bridge_target": bridge_target_manifest,
        "transfer_contract": transfer_contract,
        "contract": {
            "encoder_size": report["encoder_size"],
            "recurrent_size": report["recurrent_size"],
            "observation_size": int(
                np.fromfile(destination / "observation.bin", dtype="<f4").size),
            "action_mask_size": int(
                np.fromfile(destination / "action_mask.bin", dtype=np.uint8).size),
        },
        # Debugging telemetry only. Explicit training provenance establishes
        # whether the ordinary policy may arm.
        "actor_kernel_std": report["actor_std"],
    }
    (destination / "agent.json").write_text(
        json.dumps(manifest, indent=1) + "\n", encoding="utf-8")

    # Flat mirror for the Java side. `AgentBundle` reads this rather than the
    # JSON because the mod compiles against the server jar alone -- pulling in
    # a JSON parser for this small scalar contract would be the only dependency
    # in the tree. Same convention `case/actions.txt` already uses.
    rows = [
        ("schema", manifest["schema"]),
        ("role", role),
        ("checkpoint_sha256", manifest["checkpoint_sha256"]),
        ("policy_tensor_identity_schema", POLICY_TENSOR_IDENTITY_SCHEMA),
        ("policy_tensor_count", len(POLICY_TENSOR_NAMES)),
        ("policy_tensor_content_sha256", tensor_content_sha256),
        ("training_provenance_schema", TRAINING_PROVENANCE_SCHEMA),
        ("policy_purpose", training_provenance["purpose"]),
        ("training_optimizer_updates", training_provenance["optimizer_updates"]),
        ("training_environment_steps", training_provenance["environment_steps"]),
        (
            "training_demonstration_examples",
            training_provenance["demonstration_examples"],
        ),
        (
            "training_source_metadata_sha256",
            training_provenance["source_metadata_sha256"],
        ),
        ("observation_size", manifest["contract"]["observation_size"]),
        ("action_mask_size", manifest["contract"]["action_mask_size"]),
        ("encoder_size", manifest["contract"]["encoder_size"]),
        ("recurrent_size", manifest["contract"]["recurrent_size"]),
        ("actor_kernel_std", repr(manifest["actor_kernel_std"])),
        (
            "observation_contract_sha256",
            transfer_contract["observation_contract_sha256"],
        ),
        ("action_contract_sha256", transfer_contract["action_contract_sha256"]),
        ("policy_projection_abi", transfer_contract["policy_projection_abi"]),
        (
            "policy_action_scope_schema",
            action_scope["manifest"]["schema"],
        ),
        (
            "policy_action_scope_identity_schema",
            action_scope["identity"]["schema"],
        ),
        (
            "policy_action_scope_bits",
            action_scope["identity"]["mask_bits"],
        ),
        (
            "policy_action_scope_sha256",
            action_scope["identity"]["content_sha256"],
        ),
    ]
    rows.extend(compatibility_flat_contract_rows(deployment_compatibility))
    if bridge_target_manifest is not None:
        rows.extend((
            ("bridge_target_schema", bridge_target_manifest["schema"]),
            (
                "bridge_target_content_sha256",
                bridge_target_manifest["content_sha256"],
            ),
        ))
    (destination / "contract.txt").write_text(
        "\n".join(f"{key}\t{value}" for key, value in rows) + "\n",
        encoding="utf-8")
    return manifest


def _live_widths() -> tuple[int, int]:
    """The widths the Gym publishes right now.

    Defaulted from the live contract rather than written down. These were the
    literals 8271 and 99, and both went stale: the observation gained
    `locomotion_stamina_fraction` and the action surface grew to 124 when
    hotbar switching landed, so `verify()` refused every current bundle with a
    message blaming the policy for being "trained on another contract".
    """

    from hytalegym.jax.combat.observation.v3 import (
        ARSENAL_POLICY_ACTION_HEAD_SIZES,
        arsenal_policy_observation_size,
    )

    return (
        int(arsenal_policy_observation_size()),
        int(sum(ARSENAL_POLICY_ACTION_HEAD_SIZES)),
    )


def verify(destination: str | Path, *, expect_observation: int | None = None,
           expect_actions: int | None = None) -> list[str]:
    """Re-check a bundle. Returns problems; empty means loadable."""

    if expect_observation is None or expect_actions is None:
        live_observation, live_actions = _live_widths()
        expect_observation = (
            live_observation if expect_observation is None else expect_observation
        )
        expect_actions = live_actions if expect_actions is None else expect_actions

    destination = Path(destination)
    problems: list[str] = []
    manifest_path = destination / "agent.json"
    if not manifest_path.is_file():
        return [f"no agent.json in {destination}; contract identity is unknown"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"agent.json is unreadable: {error}"]
    transfer_contract = manifest.get("transfer_contract")
    problems.extend(_transfer_contract_errors(transfer_contract))
    problems.extend(_projection_abi_errors(transfer_contract))
    problems.extend(_policy_action_scope_errors(manifest, transfer_contract))
    problems.extend(
        _training_provenance_errors(
            destination,
            manifest.get("training_provenance"),
        )
    )
    problems.extend(
        compatibility_errors(
            destination,
            manifest.get("deployment_compatibility"),
            role=manifest.get("role"),
            purpose=(
                manifest.get("training_provenance", {}).get("purpose")
                if isinstance(manifest.get("training_provenance"), dict)
                else None
            ),
        )
    )
    problems.extend(
        _bridge_target_errors(destination, manifest.get("bridge_target"))
    )

    if manifest.get("schema") != BUNDLE_SCHEMA:
        problems.append(
            f"bundle schema {manifest.get('schema')!r} != {BUNDLE_SCHEMA!r}"
        )

    tensor_identity = manifest.get("policy_tensor_identity")
    if not isinstance(tensor_identity, dict):
        problems.append("bundle has no canonical policy tensor identity")
    else:
        if tensor_identity.get("schema") != POLICY_TENSOR_IDENTITY_SCHEMA:
            problems.append(
                "policy tensor identity schema differs: "
                f"{tensor_identity.get('schema')!r} != "
                f"{POLICY_TENSOR_IDENTITY_SCHEMA!r}"
            )
        if tensor_identity.get("ordered_names") != list(POLICY_TENSOR_NAMES):
            problems.append("policy tensor order/names differ from the Java model")
        declared_tensor_sha = tensor_identity.get("content_sha256")
        if not _is_canonical_sha256(declared_tensor_sha):
            problems.append("policy tensor content SHA-256 is malformed")
        else:
            try:
                actual_tensor_sha = policy_tensor_content_sha256(destination)
            except (OSError, ValueError) as error:
                problems.append(f"policy tensor content cannot be hashed: {error}")
            else:
                if actual_tensor_sha != declared_tensor_sha:
                    problems.append(
                        "policy tensor content hash does not match the raw weights"
                    )

    if not _is_canonical_sha256(manifest.get("checkpoint_sha256")):
        problems.append("checkpoint SHA-256 is malformed")
    problems.extend(_flat_contract_errors(destination, manifest))

    if manifest.get("live_profile"):
        if isinstance(transfer_contract, dict):
            problems.extend(_profile_errors(destination / "profile", transfer_contract))
        else:
            problems.append("live profile cannot be checked without a transfer contract")

    for name in (*POLICY_TENSOR_NAMES, "observation", "action_mask"):
        if not (destination / f"{name}.bin").is_file():
            problems.append(f"missing {name}.bin")
    if not (destination / "role.txt").is_file():
        problems.append("no role.txt; the plugin would fall back to its default role")

    contract = manifest.get("contract", {})
    if contract.get("observation_size") != expect_observation:
        problems.append(
            f"observation width {contract.get('observation_size')} != "
            f"{expect_observation}; this policy was trained on another contract")
    if contract.get("action_mask_size") != expect_actions:
        problems.append(
            f"action mask width {contract.get('action_mask_size')} != "
            f"{expect_actions}")
    return problems


def certify_java(
    destination: str | Path,
    *,
    java: str | Path,
    classes: str | Path,
    server_jar: str | Path,
) -> dict[str, object]:
    """Run the exported policy through the standalone Java inference gate."""

    destination = Path(destination).resolve()
    paths = {
        "java": Path(java).resolve(),
        "classes": Path(classes).resolve(),
        "server_jar": Path(server_jar).resolve(),
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Java inference inputs missing: {missing}")
    command = (
        str(paths["java"]),
        "-cp",
        os.pathsep.join((str(paths["classes"]), str(paths["server_jar"]))),
        "Main",
        str(destination),
    )
    result = subprocess.run(
        command, capture_output=True, text=True, check=False, timeout=60
    )
    passed = result.returncode == 0 and "PASS - JVM reproduces the JAX policy" in (
        result.stdout
    )
    if not passed:
        detail = (result.stderr or result.stdout).strip()[-2000:]
        raise RuntimeError(
            f"Java policy inference failed with exit {result.returncode}: {detail}"
        )
    return {
        "passed": True,
        "command": list(command),
        "stdout": result.stdout,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("destination", type=Path)
    ap.add_argument("--role", default="Kweebec_Razorleaf",
                    help="NPC role the policy drives; the plugin's own default")
    ap.add_argument("--decision", type=int, default=None,
                    help="ticks between decisions (30 TPS = 33.3ms each)")
    ap.add_argument("--spawn", nargs=3, type=float, metavar=("X", "Y", "Z"),
                    default=None, help="spawn one on load at these coordinates")
    ap.add_argument("--nojump", action="store_true",
                    help="suppress the jump head; airborne NPCs get no "
                         "horizontal translation, so a jump-happy policy "
                         "looks exactly like broken steering")
    ap.add_argument("--profile", type=Path, default=None,
                    help="live evidence profile directory; requires the "
                         "matching bridge facade to be deployed")
    ap.add_argument("--fixture", type=Path, default=None,
                    help="captured observation/action-mask directory used for "
                         "offline Java inference and fallback bundle values")
    ap.add_argument(
        "--training-report",
        type=Path,
        default=None,
        help=(
            "selected training report for a supported older pursuit "
            "checkpoint that did not save policy_action_scope metadata"
        ),
    )
    ap.add_argument(
        "--diagnostic",
        action="store_true",
        help=(
            "mark an untrained diagnostic explicitly; Java then also requires "
            "live-test-control.txt and will not arm it as an ordinary policy"
        ),
    )
    ap.add_argument(
        "--diagnostic-control",
        type=Path,
        help=(
            "exact live-test-control.txt for --diagnostic; copied into the "
            "bundle so Java cannot arm an uncontrolled synthetic policy"
        ),
    )
    ap.add_argument(
        "--bridge-target",
        type=Path,
        help=(
            "exact generic bridge-target.txt tuple; binds the independently "
            "packaged bridge JAR and combat/World/Dodge contracts"
        ),
    )
    ap.add_argument(
        "--role-transfer",
        default=None,
        metavar="SOURCE=TARGET",
        help=(
            "explicit cross-role deployment mapping; exact role matches need "
            "no override and mismatches fail closed without this mapping"
        ),
    )
    ap.add_argument(
        "--profile-transfer",
        default=None,
        metavar="SOURCE=TARGET",
        help=(
            "explicit cross-profile mapping from one selected-row training "
            "owner to the installed live profile"
        ),
    )
    ap.add_argument(
        "--transfer-reason",
        default=None,
        help="required documentation for any cross-role/profile mapping",
    )
    ap.add_argument("--java", type=Path, default=None)
    ap.add_argument("--java-classes", type=Path, default=None)
    ap.add_argument("--server-jar", type=Path, default=None)
    args = ap.parse_args()

    manifest = export(
        args.checkpoint, args.destination, role=args.role,
        decision=args.decision,
        spawn=None if args.spawn is None else tuple(args.spawn),
        nojump=args.nojump, profile=args.profile, fixture=args.fixture,
        diagnostic=args.diagnostic,
        diagnostic_control=args.diagnostic_control,
        training_report=args.training_report,
        bridge_target=args.bridge_target,
        role_transfer=parse_transfer_mapping(
            args.role_transfer, name="--role-transfer"
        ),
        profile_transfer=parse_transfer_mapping(
            args.profile_transfer, name="--profile-transfer"
        ),
        transfer_reason=args.transfer_reason)
    problems = verify(args.destination)

    for key, value in manifest.items():
        print(f"{key:20} {value}")
    print()
    if problems:
        print("NOT LOADABLE:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    java_inputs = (args.java, args.java_classes, args.server_jar)
    if any(java_inputs) and not all(java_inputs):
        ap.error("--java, --java-classes and --server-jar must be passed together")
    if all(java_inputs):
        certificate = certify_java(
            args.destination,
            java=args.java,
            classes=args.java_classes,
            server_jar=args.server_jar,
        )
        print(certificate["stdout"], end="")
    print(f"bundle OK -> copy {args.destination} next to the mod jar as "
          f"mods/policy-agent/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
