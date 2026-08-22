"""Content-bound training-context to Java deployment compatibility.

The policy surface and raw tensor identities deliberately say nothing about
*which* actor/loadout a row learned to control.  This module binds the selected
policy row's producer-attested training owners to the separately supplied Java
role and live profile. A caller-declared context is not accepted as proof of a
trained live profile. Exact matches need no exception. Deliberate cross-role or
cross-profile transfer must name a source-to-target mapping and a reason.

The Java mod independently recomputes the published binary identity below; it
does not trust the digest written by Python.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import struct
from typing import Mapping, Sequence


DEPLOYMENT_COMPATIBILITY_SCHEMA = (
    "hytalerl_policy_deployment_compatibility_v1"
)
TRAINING_CONTEXT_SCHEMA = "hytalerl_policy_training_context_v1"
TRAINING_CONTEXT_ATTESTATION_SCHEMA = (
    "hytalerl_policy_training_context_attestation_v1"
)
TRAINING_CONTEXT_ATTESTATION_DERIVATION = "runtime_config_entity_specs"
ARSENAL_RUNTIME_CONFIG_CONTENT_SCHEMA = (
    "hytalerl_arsenal_runtime_config_content_v1"
)
_IDENTITY_DOMAIN = b"HYTALERL_POLICY_DEPLOYMENT_COMPATIBILITY\0"
_IDENTITY_VERSION = 1
_TRAINING_CONTEXT_DOMAIN = b"HYTALERL_POLICY_TRAINING_CONTEXT\0"
_TRAINING_CONTEXT_VERSION = 1


def _profile_files_sha256(
    profile: Path,
    *,
    include_manifest: bool,
) -> str:
    identity = hashlib.sha256()
    entries = list(Path(profile).iterdir())
    unexpected = [
        path for path in entries if path.is_symlink() or not path.is_file()
    ]
    if unexpected:
        names = ", ".join(sorted(path.name for path in unexpected))
        raise ValueError(
            "live profile must be a flat directory of regular files; "
            f"unexpected entries: {names}"
        )
    paths = sorted(
        (
            path
            for path in entries
            if include_manifest or path.name != "profile.json"
        ),
        key=lambda path: path.name.encode("utf-8"),
    )
    for path in paths:
        identity.update(path.name.encode("utf-8"))
        identity.update(b"\0")
        identity.update(path.read_bytes())
        identity.update(b"\0")
    return identity.hexdigest().upper()


def profile_payload_sha256(profile: Path) -> str:
    """Hash profile payload files for the self-referential profile manifest."""

    return _profile_files_sha256(profile, include_manifest=False)


def profile_content_sha256(profile: Path) -> str:
    """Hash every immediate file installed in the Java profile directory.

    Unlike the profile manifest's own payload checksum, this deployment
    identity includes ``profile.json``.  The selected profile name and its
    policy-contract metadata therefore cannot be replaced while retaining the
    same binary payload and compatibility digest.
    """

    return _profile_files_sha256(profile, include_manifest=True)


def parse_transfer_mapping(
    value: str | None,
    *,
    name: str,
) -> tuple[str, str] | None:
    """Parse one explicit ``SOURCE=TARGET`` deployment mapping."""

    if value is None:
        return None
    source, separator, target = value.partition("=")
    if not separator or not source or not target:
        raise ValueError(f"{name} must be SOURCE=TARGET")
    if source != source.strip() or target != target.strip():
        raise ValueError(f"{name} source and target cannot have edge whitespace")
    return source, target


def _integer(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} is outside [{minimum}, {maximum}]")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty trimmed string")
    if "\t" in value or "\r" in value or "\n" in value:
        raise ValueError(f"{name} cannot contain tabs or newlines")
    return value


def normalize_training_context(
    metadata: object,
    *,
    diagnostic: bool,
) -> dict[str, object]:
    """Return the selected policy's canonical owner rows.

    Current population checkpoints publish ``policy_training_context`` (and
    retain the longer population-prefixed alias).  Ordinary trained exports
    fail closed when neither exists: a profile supplied at deployment is not
    evidence that the checkpoint trained on that profile.  A controlled
    synthetic diagnostic has no training owners by definition.
    """

    if not isinstance(metadata, Mapping):
        metadata = {}
    raw = metadata.get("policy_training_context")
    population_raw = metadata.get("population_policy_training_context")
    if raw is not None and population_raw is not None:
        if not isinstance(raw, Mapping) or not isinstance(
            population_raw, Mapping
        ) or dict(raw) != dict(population_raw):
            raise ValueError(
                "selected policy_training_context disagrees with the "
                "population-selected context"
            )
    if raw is None:
        raw = population_raw
    plural = metadata.get("policy_training_contexts")
    if raw is not None and plural is not None:
        if not isinstance(plural, Sequence) or isinstance(plural, (str, bytes)):
            raise TypeError("policy_training_contexts must be a list")
        selected_policy = (
            raw.get("policy_id") if isinstance(raw, Mapping) else None
        )
        selected = [
            value
            for value in plural
            if isinstance(value, Mapping)
            and value.get("policy_id") == selected_policy
        ]
        if len(selected) != 1 or dict(selected[0]) != dict(raw):
            raise ValueError(
                "selected policy_training_context disagrees with the "
                "population context list"
            )
    if raw is None:
        if diagnostic:
            return {
                "schema": TRAINING_CONTEXT_SCHEMA,
                "policy_id": -1,
                "owners": [],
            }
        raise ValueError(
            "trained checkpoint has no policy_training_context; the selected "
            "row cannot be bound to a Java profile without its owner metadata"
        )
    if not isinstance(raw, Mapping):
        raise TypeError("policy_training_context must be an object")
    policy_id = _integer(
        raw.get("policy_id"),
        name="policy_training_context.policy_id",
        minimum=0,
        maximum=(1 << 63) - 1,
    )
    raw_owners = raw.get("owners")
    if not isinstance(raw_owners, Sequence) or isinstance(
        raw_owners, (str, bytes)
    ) or not raw_owners:
        raise ValueError("policy_training_context.owners must be a nonempty list")
    owners: list[dict[str, object]] = []
    keys: set[tuple[int, str, int, str, bool]] = set()
    for index, raw_owner in enumerate(raw_owners):
        if not isinstance(raw_owner, Mapping):
            raise TypeError(f"policy training owner {index} must be an object")
        trainable = raw_owner.get("trainable")
        if not isinstance(trainable, bool):
            raise TypeError(f"policy training owner {index} trainable must be bool")
        owner = {
            "entity_index": _integer(
                raw_owner.get("entity_index"),
                name=f"policy training owner {index} entity_index",
                minimum=0,
                maximum=(1 << 31) - 1,
            ),
            "profile": _text(
                raw_owner.get("profile"),
                name=f"policy training owner {index} profile",
            ),
            "team_id": _integer(
                raw_owner.get("team_id"),
                name=f"policy training owner {index} team_id",
                minimum=-(1 << 31),
                maximum=(1 << 31) - 1,
            ),
            "controller": _text(
                raw_owner.get("controller"),
                name=f"policy training owner {index} controller",
            ),
            "trainable": trainable,
        }
        key = (
            owner["entity_index"],
            owner["profile"],
            owner["team_id"],
            owner["controller"],
            owner["trainable"],
        )
        if key in keys:
            raise ValueError("policy_training_context repeats an owner row")
        keys.add(key)
        owners.append(owner)
    owners.sort(
        key=lambda owner: (
            owner["entity_index"],
            owner["profile"],
            owner["team_id"],
            owner["controller"],
            owner["trainable"],
        )
    )
    return {
        "schema": TRAINING_CONTEXT_SCHEMA,
        "policy_id": policy_id,
        "owners": owners,
    }


def _write_u32(identity, value: int) -> None:
    identity.update(struct.pack(">I", value))


def _write_i32(identity, value: int) -> None:
    identity.update(struct.pack(">i", value))


def _write_i64(identity, value: int) -> None:
    identity.update(struct.pack(">q", value))


def _write_text(identity, value: str) -> None:
    encoded = value.encode("utf-8")
    _write_u32(identity, len(encoded))
    identity.update(encoded)


def training_context_content_sha256(value: Mapping[str, object]) -> str:
    """Content identity a producer must derive from actual runtime specs."""

    identity = hashlib.sha256(_TRAINING_CONTEXT_DOMAIN)
    _write_u32(identity, _TRAINING_CONTEXT_VERSION)
    _write_text(identity, str(value["schema"]))
    _write_i64(identity, int(value["policy_id"]))
    owners = value["owners"]
    _write_u32(identity, len(owners))
    for owner in owners:
        _write_i32(identity, int(owner["entity_index"]))
        _write_text(identity, str(owner["profile"]))
        _write_i32(identity, int(owner["team_id"]))
        _write_text(identity, str(owner["controller"]))
        identity.update(b"\1" if owner["trainable"] else b"\0")
    return identity.hexdigest().upper()


def _normalize_training_context_attestation(
    metadata: Mapping[str, object],
    context: Mapping[str, object],
) -> dict[str, object]:
    raw = metadata.get("policy_training_context_attestation")
    population_raw = metadata.get(
        "population_policy_training_context_attestation"
    )
    if raw is not None and population_raw is not None:
        if not isinstance(raw, Mapping) or not isinstance(
            population_raw, Mapping
        ) or dict(raw) != dict(population_raw):
            raise ValueError(
                "selected policy training context attestation disagrees with "
                "the population-selected attestation"
            )
    if raw is None:
        raw = population_raw
    plural = metadata.get("policy_training_context_attestations")
    if raw is not None and plural is not None:
        if not isinstance(plural, Sequence) or isinstance(
            plural, (str, bytes)
        ):
            raise TypeError(
                "policy_training_context_attestations must be a list"
            )
        selected = [
            value
            for value in plural
            if isinstance(value, Mapping)
            and value.get("policy_id") == context["policy_id"]
        ]
        if len(selected) != 1 or dict(selected[0]) != dict(raw):
            raise ValueError(
                "selected policy training context attestation disagrees with "
                "the population attestation list"
            )
    if raw is None:
        return {
            "schema": "",
            "derivation": "",
            "runtime_config_content_sha256": "",
            "context_content_sha256": "",
        }
    if not isinstance(raw, Mapping):
        raise TypeError("policy_training_context_attestation must be an object")
    schema = _text(
        raw.get("schema"),
        name="policy training context attestation schema",
    )
    if schema != TRAINING_CONTEXT_ATTESTATION_SCHEMA:
        raise ValueError("policy training context attestation schema differs")
    derivation = _text(
        raw.get("derivation"),
        name="policy training context attestation derivation",
    )
    if derivation != TRAINING_CONTEXT_ATTESTATION_DERIVATION:
        raise ValueError(
            "policy training context attestation was not derived from the "
            "runtime config entity specs"
        )
    policy_id = _integer(
        raw.get("policy_id"),
        name="policy training context attestation policy_id",
        minimum=0,
        maximum=(1 << 63) - 1,
    )
    if policy_id != context["policy_id"]:
        raise ValueError("policy training context attestation policy ID differs")
    runtime_sha = _text(
        raw.get("runtime_config_content_sha256"),
        name="runtime config content SHA-256",
    )
    context_sha = _text(
        raw.get("context_content_sha256"),
        name="training context content SHA-256",
    )
    for name, value in (
        ("runtime config", runtime_sha),
        ("training context", context_sha),
    ):
        if len(value) != 64 or any(
            character not in "0123456789ABCDEF" for character in value
        ):
            raise ValueError(f"{name} content SHA-256 must be canonical")
    actual_context_sha = training_context_content_sha256(context)
    if context_sha != actual_context_sha:
        raise ValueError(
            "policy training context attestation does not identify the "
            "selected owner rows"
        )
    return {
        "schema": schema,
        "derivation": derivation,
        "runtime_config_content_sha256": runtime_sha,
        "context_content_sha256": context_sha,
    }


def compatibility_content_sha256(value: Mapping[str, object]) -> str:
    """Canonical identity independently implemented by the Java loader."""

    source = value["training_context"]
    deployment = value["deployment"]
    transfer = value["transfer"]
    identity = hashlib.sha256(_IDENTITY_DOMAIN)
    _write_u32(identity, _IDENTITY_VERSION)
    _write_text(identity, str(value["schema"]))
    _write_text(identity, str(source["schema"]))
    _write_text(identity, str(source["source_role"]))
    _write_i64(identity, int(source["policy_id"]))
    owners = source["owners"]
    _write_u32(identity, len(owners))
    for owner in owners:
        _write_i32(identity, int(owner["entity_index"]))
        _write_text(identity, str(owner["profile"]))
        _write_i32(identity, int(owner["team_id"]))
        _write_text(identity, str(owner["controller"]))
        identity.update(b"\1" if owner["trainable"] else b"\0")
    attestation = source["attestation"]
    _write_text(identity, str(attestation["schema"]))
    _write_text(identity, str(attestation["derivation"]))
    _write_text(identity, str(attestation["runtime_config_content_sha256"]))
    _write_text(identity, str(attestation["context_content_sha256"]))
    _write_text(identity, str(deployment["role"]))
    _write_text(identity, str(deployment["profile_mode"]))
    _write_text(identity, str(deployment["profile"]))
    _write_text(identity, str(deployment["profile_content_sha256"]))
    _write_text(identity, str(transfer["role_source"]))
    _write_text(identity, str(transfer["profile_source"]))
    _write_text(identity, str(transfer["reason"]))
    return identity.hexdigest().upper()


def build_deployment_compatibility(
    metadata: object,
    transfer_contract: Mapping[str, object],
    *,
    role: str,
    profile_manifest: Mapping[str, object] | None,
    profile_content_sha256_value: str | None,
    diagnostic: bool,
    role_transfer: tuple[str, str] | None = None,
    profile_transfer: tuple[str, str] | None = None,
    transfer_reason: str | None = None,
) -> dict[str, object]:
    """Bind one checkpoint row to the exact separately supplied Java inputs."""

    role = _text(role, name="deployment role")
    if role == "*" and not diagnostic:
        raise ValueError(
            "trained deployment role must be exact; wildcard '*' is reserved "
            "for an explicitly controlled diagnostic"
        )
    source_role = _text(
        transfer_contract.get("agent_role"),
        name="transfer_contract.agent_role",
    )
    context = normalize_training_context(metadata, diagnostic=diagnostic)
    attestation = _normalize_training_context_attestation(
        metadata if isinstance(metadata, Mapping) else {},
        context,
    )
    owners = context["owners"]
    source_profiles = {str(owner["profile"]) for owner in owners}

    if profile_manifest is None:
        profile_mode = "frozen"
        deployment_profile = ""
        profile_sha = ""
        if profile_transfer is not None:
            raise ValueError("profile transfer mapping requires a live profile")
    else:
        profile_mode = "live"
        deployment_profile = _text(
            profile_manifest.get("profile"),
            name="live profile manifest profile",
        )
        profile_sha = _text(
            profile_content_sha256_value,
            name="live profile content SHA-256",
        )
        if len(profile_sha) != 64 or any(
            character not in "0123456789ABCDEF" for character in profile_sha
        ):
            raise ValueError("live profile content SHA-256 must be canonical")

    role_source = ""
    if role == source_role:
        if role_transfer is not None:
            raise ValueError("role transfer mapping is unnecessary for an exact match")
    else:
        if role_transfer != (source_role, role):
            raise ValueError(
                "deployment role differs from training role; require exact "
                f"--role-transfer {source_role}={role}"
            )
        role_source = source_role

    profile_source = ""
    if profile_mode == "live" and not diagnostic:
        if attestation["schema"] != TRAINING_CONTEXT_ATTESTATION_SCHEMA:
            raise ValueError(
                "trained live upload has no runtime-derived policy training "
                "context attestation; migrate the checkpoint producer before "
                "pairing this row with a Java profile"
            )
        if deployment_profile in source_profiles:
            if profile_transfer is not None:
                raise ValueError(
                    "profile transfer mapping is unnecessary for a trained profile"
                )
        else:
            if (
                profile_transfer is None
                or profile_transfer[0] not in source_profiles
                or profile_transfer[1] != deployment_profile
            ):
                available = ", ".join(sorted(source_profiles)) or "<none>"
                raise ValueError(
                    "deployment profile was not owned by the selected policy; "
                    "require an exact --profile-transfer SOURCE=TARGET from "
                    f"one of [{available}]"
                )
            profile_source = profile_transfer[0]
    elif profile_transfer is not None:
        raise ValueError(
            "profile transfer mapping is only valid for a trained live profile"
        )

    needs_reason = bool(role_source or profile_source)
    reason = "" if transfer_reason is None else _text(
        transfer_reason,
        name="transfer reason",
    )
    if needs_reason and not reason:
        raise ValueError("cross-context transfer requires --transfer-reason")
    if not needs_reason and reason:
        raise ValueError("transfer reason is only valid with an explicit mapping")

    result: dict[str, object] = {
        "schema": DEPLOYMENT_COMPATIBILITY_SCHEMA,
        "training_context": {
            **context,
            "source_role": source_role,
            "attestation": attestation,
        },
        "deployment": {
            "role": role,
            "profile_mode": profile_mode,
            "profile": deployment_profile,
            "profile_content_sha256": profile_sha,
        },
        "transfer": {
            "role_source": role_source,
            "profile_source": profile_source,
            "reason": reason,
        },
    }
    result["content_sha256"] = compatibility_content_sha256(result)
    return result


def flat_contract_rows(value: Mapping[str, object]) -> list[tuple[str, object]]:
    source = value["training_context"]
    deployment = value["deployment"]
    transfer = value["transfer"]
    rows: list[tuple[str, object]] = [
        ("deployment_compatibility_schema", value["schema"]),
        ("deployment_compatibility_content_sha256", value["content_sha256"]),
        ("training_context_schema", source["schema"]),
        ("training_context_source_role", source["source_role"]),
        ("training_context_policy_id", source["policy_id"]),
        ("training_context_owner_count", len(source["owners"])),
    ]
    for index, owner in enumerate(source["owners"]):
        prefix = f"training_context_owner_{index}_"
        rows.extend(
            (
                (prefix + "entity_index", owner["entity_index"]),
                (prefix + "profile", owner["profile"]),
                (prefix + "team_id", owner["team_id"]),
                (prefix + "controller", owner["controller"]),
                (prefix + "trainable", str(owner["trainable"]).lower()),
            )
        )
    rows.extend(
        (
            (
                "training_context_attestation_schema",
                source["attestation"]["schema"],
            ),
            (
                "training_context_attestation_derivation",
                source["attestation"]["derivation"],
            ),
            (
                "training_context_runtime_config_content_sha256",
                source["attestation"]["runtime_config_content_sha256"],
            ),
            (
                "training_context_content_sha256",
                source["attestation"]["context_content_sha256"],
            ),
            ("deployment_role", deployment["role"]),
            ("deployment_profile_mode", deployment["profile_mode"]),
            ("deployment_profile", deployment["profile"]),
            (
                "deployment_profile_content_sha256",
                deployment["profile_content_sha256"],
            ),
            ("role_transfer_source", transfer["role_source"]),
            ("profile_transfer_source", transfer["profile_source"]),
            ("transfer_reason", transfer["reason"]),
        )
    )
    return rows


def compatibility_errors(
    directory: Path,
    value: object,
    *,
    role: object,
    purpose: object,
) -> list[str]:
    """Revalidate the JSON contract and exact installed role/profile files."""

    if not isinstance(value, Mapping):
        return ["bundle has no deployment compatibility contract"]
    problems: list[str] = []
    try:
        source = value["training_context"]
        deployment = value["deployment"]
        transfer = value["transfer"]
        if value.get("schema") != DEPLOYMENT_COMPATIBILITY_SCHEMA:
            problems.append("deployment compatibility schema differs")
        if source.get("schema") != TRAINING_CONTEXT_SCHEMA:
            problems.append("policy training context schema differs")
        expected_sha = compatibility_content_sha256(value)
        if value.get("content_sha256") != expected_sha:
            problems.append("deployment compatibility content hash differs")
        if role != deployment.get("role"):
            problems.append("bundle role differs from deployment compatibility role")
        if purpose == "trained" and deployment.get("role") == "*":
            problems.append("trained deployment role cannot be wildcard '*'")
        owners = source.get("owners")
        if purpose == "trained" and not owners:
            problems.append("trained deployment compatibility has no owner rows")
        source_role = source.get("source_role")
        deployment_role = deployment.get("role")
        role_source = transfer.get("role_source")
        reason = transfer.get("reason")
        if deployment_role == source_role:
            if role_source:
                problems.append("exact role match carries an unnecessary transfer")
        elif role_source != source_role or not reason:
            problems.append("cross-role deployment has no explicit mapped reason")

        mode = deployment.get("profile_mode")
        profile_path = Path(directory) / "profile"
        if mode == "frozen":
            if profile_path.is_dir():
                problems.append("frozen compatibility unexpectedly has a profile")
            if deployment.get("profile") or deployment.get(
                "profile_content_sha256"
            ) or transfer.get("profile_source"):
                problems.append("frozen compatibility carries live-profile fields")
        elif mode == "live":
            attestation = source.get("attestation")
            if purpose == "trained" and (
                not isinstance(attestation, Mapping)
                or attestation.get("schema")
                != TRAINING_CONTEXT_ATTESTATION_SCHEMA
                or attestation.get("derivation")
                != TRAINING_CONTEXT_ATTESTATION_DERIVATION
                or not isinstance(
                    attestation.get("runtime_config_content_sha256"), str
                )
                or len(attestation.get("runtime_config_content_sha256", ""))
                != 64
                or any(
                    character not in "0123456789ABCDEF"
                    for character in attestation.get(
                        "runtime_config_content_sha256", ""
                    )
                )
                or attestation.get("context_content_sha256")
                != training_context_content_sha256(source)
            ):
                problems.append(
                    "trained live compatibility has no valid runtime-derived "
                    "training-context attestation"
                )
            if not profile_path.is_dir():
                problems.append("live compatibility has no profile directory")
            else:
                actual = profile_content_sha256(profile_path)
                if deployment.get("profile_content_sha256") != actual:
                    problems.append("deployment profile content hash differs")
            target = deployment.get("profile")
            profiles = {
                owner.get("profile")
                for owner in owners
                if isinstance(owner, Mapping)
            } if isinstance(owners, list) else set()
            profile_source = transfer.get("profile_source")
            if purpose == "trained":
                if target in profiles:
                    if profile_source:
                        problems.append(
                            "trained profile match carries an unnecessary transfer"
                        )
                elif profile_source not in profiles or not reason:
                    problems.append(
                        "cross-profile deployment has no explicit mapped reason"
                    )
            elif profile_source:
                problems.append("diagnostic profile carries a training transfer")
        else:
            problems.append("unknown deployment profile mode")

        role_path = Path(directory) / "role.txt"
        actual_role = (
            role_path.read_text(encoding="utf-8").strip()
            if role_path.is_file()
            else None
        )
        if actual_role != deployment_role:
            problems.append("role.txt differs from deployment compatibility role")
    except (KeyError, OSError, TypeError, ValueError, OverflowError) as error:
        problems.append(f"deployment compatibility is malformed: {error}")
    return problems


__all__ = [
    "DEPLOYMENT_COMPATIBILITY_SCHEMA",
    "ARSENAL_RUNTIME_CONFIG_CONTENT_SCHEMA",
    "TRAINING_CONTEXT_SCHEMA",
    "TRAINING_CONTEXT_ATTESTATION_SCHEMA",
    "build_deployment_compatibility",
    "compatibility_content_sha256",
    "compatibility_errors",
    "flat_contract_rows",
    "normalize_training_context",
    "parse_transfer_mapping",
    "profile_content_sha256",
    "profile_payload_sha256",
    "training_context_content_sha256",
]
