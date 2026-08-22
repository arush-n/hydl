"""Canonical publication stamp for the complete Combat dependency tuple."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
from uuid import uuid4
from zipfile import ZipFile

from hytalegym.jax.combat.arsenal.schema.spec import (
    combat_arsenal_contract_sha256,
)
from hytalegym.jax.combat.observation.v3.policy.surface import (
    action_surface_staging_contract_sha256,
)
from hytalegym.jax.combat.observation.v3.policy.candidates.contract import (
    recipe_candidate_encoding_contract_sha256,
)
from hytalegym.jax.combat.observation.v3.policy import (
    ARSENAL_POLICY_ACTION_HEAD_SIZES,
    ARSENAL_POLICY_ACTION_SIZE,
    ARSENAL_POLICY_OBSERVATION_SIZE,
    arsenal_policy_contract_sha256,
)
from hytalegym.jax.combat.observation.v3.schema.spec import (
    learner_observation_v3_actor_evidence_contract_sha256,
    learner_observation_v3_contract_sha256,
)
from hytalegym.rulesets import combat_ruleset_sha256
from hytalegym.worldgen.region import current_native_evidence_jar_sha256
from hytalegym.worldgen.region.recertification import derive_contract_cascade


_NATIVE_ACTOR_EVIDENCE_CLASS = (
    "com/hytalerlbridge/observation/NativeActorEvidenceFrame.class"
)
_PRIVILEGED_NPC_SNAPSHOT_CLASS = (
    "com/hytalerlbridge/entity/PrivilegedNpcSnapshot.class"
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _repository_git_sha(repository_root: Path) -> str:
    return subprocess.check_output(
        ("git", "rev-parse", "--short", "HEAD"),
        cwd=repository_root,
        text=True,
    ).strip()


def bridge_actor_evidence_contract_sha256(
    bridge_jar: str | Path,
) -> str:
    """Read the actor-evidence contract embedded in one built bridge JAR."""

    value = bridge_string_constant(
        bridge_jar,
        class_entry=_NATIVE_ACTOR_EVIDENCE_CLASS,
        field_name="CONTRACT_SHA256",
    )
    if len(value) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in value
    ):
        raise ValueError(
            "bridge actor-evidence contract is not a SHA-256 digest: "
            f"{value!r}"
        )
    return value.upper()


def bridge_privileged_npc_component_filter(
    bridge_jar: str | Path,
) -> str:
    """Read the privileged-NPC filter embedded in one built bridge JAR."""

    return bridge_string_constant(
        bridge_jar,
        class_entry=_PRIVILEGED_NPC_SNAPSHOT_CLASS,
        field_name="COMPONENT_FILTER",
    )


def bridge_string_constant(
    bridge_jar: str | Path,
    *,
    class_entry: str,
    field_name: str,
) -> str:
    """Read one constant String from one compiled class inside a bridge JAR."""

    path = Path(bridge_jar)
    with ZipFile(path) as archive:
        try:
            class_bytes = archive.read(class_entry)
        except KeyError as error:
            raise ValueError(
                f"bridge JAR has no {class_entry} class entry: {path}"
            ) from error
    return _java_string_constant(class_bytes, field_name=field_name)


def _java_string_constant(class_bytes: bytes, *, field_name: str) -> str:
    """Return one ConstantValue String from a JVM class file."""

    offset = 0

    def take(size: int) -> bytes:
        nonlocal offset
        end = offset + size
        if end > len(class_bytes):
            raise ValueError("truncated JVM class file")
        value = class_bytes[offset:end]
        offset = end
        return value

    def u1() -> int:
        return take(1)[0]

    def u2() -> int:
        return struct.unpack(">H", take(2))[0]

    def u4() -> int:
        return struct.unpack(">I", take(4))[0]

    if take(4) != b"\xca\xfe\xba\xbe":
        raise ValueError("invalid JVM class-file magic")
    take(4)  # minor_version + major_version
    constant_count = u2()
    constants: list[tuple[int, object] | None] = [None] * constant_count
    index = 1
    while index < constant_count:
        tag = u1()
        if tag == 1:  # CONSTANT_Utf8
            constants[index] = (tag, take(u2()).decode("utf-8"))
        elif tag in (3, 4):  # Integer / Float
            take(4)
        elif tag in (5, 6):  # Long / Double occupy two entries.
            take(8)
            index += 1
        elif tag in (7, 8, 16, 19, 20):  # one u2 index
            constants[index] = (tag, u2())
        elif tag in (9, 10, 11, 12, 17, 18):  # two u2 indexes
            take(4)
        elif tag == 15:  # MethodHandle: u1 kind + u2 index
            take(3)
        else:
            raise ValueError(f"unsupported JVM constant-pool tag: {tag}")
        index += 1

    def utf8(constant_index: int) -> str:
        try:
            entry = constants[constant_index]
        except IndexError as error:
            raise ValueError("invalid JVM constant-pool index") from error
        if entry is None or entry[0] != 1:
            raise ValueError("JVM constant is not UTF-8")
        return str(entry[1])

    def string(constant_index: int) -> str:
        try:
            entry = constants[constant_index]
        except IndexError as error:
            raise ValueError("invalid JVM string-constant index") from error
        if entry is None or entry[0] != 8:
            raise ValueError("JVM ConstantValue is not a String")
        return utf8(int(entry[1]))

    take(6)  # access_flags + this_class + super_class
    take(2 * u2())  # interfaces
    field_count = u2()
    for _ in range(field_count):
        take(2)  # access_flags
        name = utf8(u2())
        descriptor = utf8(u2())
        attribute_count = u2()
        constant_value: int | None = None
        for _ in range(attribute_count):
            attribute_name = utf8(u2())
            attribute_length = u4()
            attribute = take(attribute_length)
            if attribute_name == "ConstantValue":
                if attribute_length != 2:
                    raise ValueError("invalid JVM ConstantValue attribute")
                constant_value = struct.unpack(">H", attribute)[0]
        if name == field_name:
            if descriptor != "Ljava/lang/String;" or constant_value is None:
                raise ValueError(
                    f"JVM field {field_name!r} is not a constant String"
                )
            return string(constant_value)
    raise ValueError(f"JVM class has no field {field_name!r}")


def _default_deployed_bridge_path() -> Path | None:
    appdata = os.environ.get("APPDATA")
    if not appdata:
        return None
    return (
        Path(appdata)
        / "Hytale"
        / "install"
        / "release"
        / "package"
        / "game"
        / "latest"
        / "Server"
        / "mods"
        / "HytaleRLBridge-0.1.0.jar"
    )


def verify_combat_bridge_artifact_identity(
    repository_root: str | Path,
    *,
    built_bridge: str | Path | None = None,
    deployed_bridge: str | Path | None = None,
    require_deployed: bool = False,
    require_actor_evidence_contract: bool = True,
    require_privileged_npc_component_filter: bool | None = None,
) -> dict[str, str | None]:
    """Verify bridge bytes and compiled constants consumed by the host."""

    if require_privileged_npc_component_filter is None:
        require_privileged_npc_component_filter = (
            require_actor_evidence_contract
        )
    if require_privileged_npc_component_filter:
        from hytalegym.worldgen.native_privileged_npcs import (
            NATIVE_PRIVILEGED_NPC_COMPONENT_FILTER,
        )

        expected_privileged_npc_filter: str | None = (
            NATIVE_PRIVILEGED_NPC_COMPONENT_FILTER
        )
    else:
        expected_privileged_npc_filter = None

    expected = current_native_evidence_jar_sha256().upper()
    expected_actor_evidence = (
        learner_observation_v3_actor_evidence_contract_sha256().upper()
    )
    built_path = (
        Path(built_bridge)
        if built_bridge is not None
        else (
            Path(repository_root)
            / "hytale-plugin"
            / "build"
            / "libs"
            / "HytaleRLBridge-0.1.0.jar"
        )
    )
    if not built_path.is_file():
        raise FileNotFoundError(f"built bridge JAR is missing: {built_path}")
    built = _sha256_file(built_path)
    if built != expected:
        raise RuntimeError(
            "built bridge differs from the canonical native-evidence pin: "
            f"built={built} expected={expected} path={built_path}"
        )
    built_actor_evidence = (
        bridge_actor_evidence_contract_sha256(built_path)
        if require_actor_evidence_contract
        else None
    )
    built_privileged_npc_filter = (
        bridge_privileged_npc_component_filter(built_path)
        if require_privileged_npc_component_filter
        else None
    )
    if (
        built_actor_evidence is not None
        and built_actor_evidence != expected_actor_evidence
    ):
        raise RuntimeError(
            "built bridge actor-evidence contract differs from the host: "
            f"built={built_actor_evidence} "
            f"host={expected_actor_evidence} path={built_path}"
        )
    if (
        built_privileged_npc_filter is not None
        and built_privileged_npc_filter != expected_privileged_npc_filter
    ):
        raise RuntimeError(
            "built bridge privileged-NPC component filter differs from the "
            f"host: built={built_privileged_npc_filter!r} "
            f"host={expected_privileged_npc_filter!r} path={built_path}"
        )

    deployed_path = (
        Path(deployed_bridge)
        if deployed_bridge is not None
        else _default_deployed_bridge_path()
    )
    deployed: str | None = None
    deployed_actor_evidence: str | None = None
    deployed_privileged_npc_filter: str | None = None
    if deployed_path is not None and deployed_path.is_file():
        deployed = _sha256_file(deployed_path)
        if deployed != expected:
            raise RuntimeError(
                "deployed bridge differs from the canonical native-evidence pin: "
                f"deployed={deployed} expected={expected} path={deployed_path}"
            )
        deployed_actor_evidence = (
            bridge_actor_evidence_contract_sha256(deployed_path)
            if require_actor_evidence_contract
            else None
        )
        deployed_privileged_npc_filter = (
            bridge_privileged_npc_component_filter(deployed_path)
            if require_privileged_npc_component_filter
            else None
        )
        if (
            deployed_actor_evidence is not None
            and deployed_actor_evidence != expected_actor_evidence
        ):
            raise RuntimeError(
                "deployed bridge actor-evidence contract differs from the host: "
                f"deployed={deployed_actor_evidence} "
                f"host={expected_actor_evidence} path={deployed_path}"
            )
        if (
            deployed_privileged_npc_filter is not None
            and deployed_privileged_npc_filter
            != expected_privileged_npc_filter
        ):
            raise RuntimeError(
                "deployed bridge privileged-NPC component filter differs "
                f"from the host: deployed={deployed_privileged_npc_filter!r} "
                f"host={expected_privileged_npc_filter!r} path={deployed_path}"
            )
    elif require_deployed:
        raise FileNotFoundError(
            "deployed bridge JAR is missing: "
            f"{deployed_path if deployed_path is not None else '<undiscoverable>'}"
        )

    return {
        "expected_sha256": expected,
        "built_sha256": built,
        "built_path": str(built_path.resolve()),
        "host_actor_evidence_contract_sha256": expected_actor_evidence,
        "built_actor_evidence_contract_sha256": built_actor_evidence,
        "host_privileged_npc_component_filter": (
            expected_privileged_npc_filter
        ),
        "built_privileged_npc_component_filter": (
            built_privileged_npc_filter
        ),
        "deployed_sha256": deployed,
        "deployed_actor_evidence_contract_sha256": deployed_actor_evidence,
        "deployed_privileged_npc_component_filter": (
            deployed_privileged_npc_filter
        ),
        "deployed_path": (
            str(deployed_path.resolve()) if deployed_path is not None else None
        ),
    }


def combat_publication_snapshot(
    *,
    repository_git_sha: str,
) -> dict[str, object]:
    """Return the Combat-specific identities omitted by the shared cascade."""

    return {
        "schema": "hytalerl_combat_publication_snapshot_v1",
        "repository_git_sha": repository_git_sha,
        "identities": {
            "action_surface": action_surface_staging_contract_sha256(),
            "arsenal_policy": arsenal_policy_contract_sha256(),
            "bridge": current_native_evidence_jar_sha256().upper(),
            "combat_arsenal": combat_arsenal_contract_sha256(),
            "combat_ruleset": combat_ruleset_sha256().upper(),
            "learner_observation_v3": (
                learner_observation_v3_contract_sha256()
            ),
            "learner_observation_v3_actor_evidence": (
                learner_observation_v3_actor_evidence_contract_sha256()
            ),
            "recipe_candidate_encoding": (
                recipe_candidate_encoding_contract_sha256()
            ),
        },
        "shape": {
            "action_head_sizes": list(ARSENAL_POLICY_ACTION_HEAD_SIZES),
            "action_logit_size": ARSENAL_POLICY_ACTION_SIZE,
            "observation_size": ARSENAL_POLICY_OBSERVATION_SIZE,
        },
    }


def combat_contract_stamp(
    repository_root: str | Path,
    *,
    repository_git_sha: str,
) -> dict[str, object]:
    """Join the shared cascade and Combat surface into one immutable stamp."""

    root = Path(repository_root)
    publication = combat_publication_snapshot(
        repository_git_sha=repository_git_sha,
    )
    bridge = str(publication["identities"]["bridge"])
    cascade = derive_contract_cascade(root, bridge)
    identities = {
        "bridge": bridge,
        **cascade["identities"],
        "action_surface": publication["identities"]["action_surface"],
        "combat_ruleset": publication["identities"]["combat_ruleset"],
        "learner_observation_v3_actor_evidence": publication["identities"][
            "learner_observation_v3_actor_evidence"
        ],
        "recipe_candidate_encoding": publication["identities"][
            "recipe_candidate_encoding"
        ],
    }
    return {
        "schema": "hytalerl_combat_contract_stamp_v1",
        "repository_git_sha": repository_git_sha,
        "identities": identities,
        "shape": publication["shape"],
        "publication": cascade["publication"],
    }


def combat_contract_stamp_sha256(
    repository_root: str | Path,
    *,
    repository_git_sha: str,
) -> str:
    """Return the canonical identity of the complete publication stamp."""

    payload = json.dumps(
        combat_contract_stamp(
            repository_root,
            repository_git_sha=repository_git_sha,
        ),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def publish_combat_contract_stamp(
    repository_root: str | Path,
    *,
    repository_git_sha: str,
    built_bridge: str | Path | None = None,
    deployed_bridge: str | Path | None = None,
    require_deployed: bool | None = None,
) -> Path:
    """Atomically publish one complete immutable Combat contract directory.

    Explicit bridge paths support an isolated validation server without
    treating the shared install's ``mods`` directory as the active runtime.
    Both paths are still checked byte-for-byte against the canonical pin and
    against the host actor-evidence constants.
    """

    root = Path(repository_root)
    actual_repository_git_sha = _repository_git_sha(root)
    if repository_git_sha != actual_repository_git_sha:
        raise RuntimeError(
            "Combat publication repository identity is stale: "
            f"declared={repository_git_sha} actual={actual_repository_git_sha}"
        )
    if require_deployed is None:
        require_deployed = os.name == "nt"
    bridge_artifacts = verify_combat_bridge_artifact_identity(
        root,
        built_bridge=built_bridge,
        deployed_bridge=deployed_bridge,
        require_deployed=require_deployed,
    )
    stamp = combat_contract_stamp(
        root,
        repository_git_sha=repository_git_sha,
    )
    snapshot = combat_publication_snapshot(
        repository_git_sha=repository_git_sha,
    )
    bridge = str(stamp["identities"]["bridge"])
    if bridge_artifacts["expected_sha256"] != bridge:
        raise RuntimeError(
            "verified bridge artifact identity changed during Combat "
            "publication"
        )
    cascade = derive_contract_cascade(root, bridge)
    if stamp["publication"] != cascade["publication"] or any(
        stamp["identities"].get(name) != value
        for name, value in cascade["identities"].items()
    ):
        raise RuntimeError("Combat stamp and shared contract cascade disagree")
    # Re-derive after assembling every file. A concurrent contract move must
    # abort before any publication directory becomes visible.
    if stamp != combat_contract_stamp(
        root,
        repository_git_sha=repository_git_sha,
    ) or snapshot != combat_publication_snapshot(
        repository_git_sha=repository_git_sha,
    ):
        raise RuntimeError("Combat contract changed during publication")
    if bridge_artifacts != verify_combat_bridge_artifact_identity(
        root,
        built_bridge=built_bridge,
        deployed_bridge=deployed_bridge,
        require_deployed=require_deployed,
    ):
        raise RuntimeError("bridge artifacts changed during Combat publication")

    digest = hashlib.sha256(
        json.dumps(
            stamp,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest().upper()
    artifacts = root / "artifacts"
    target = artifacts / f"combat-contract-stamp-{digest[:8].lower()}"
    values = {
        "combat-publication.json": snapshot,
        "contract-cascade.json": cascade,
        "contract-stamp.json": stamp,
    }
    if target.exists():
        _verify_combat_publication_directory(target, values)
        return target

    artifacts.mkdir(parents=True, exist_ok=True)
    temporary = artifacts / f".{target.name}.{uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        for name, value in values.items():
            (temporary / name).write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        (temporary / "README.md").write_text(
            "\n".join(
                (
                    "# Combat contract stamp",
                    "",
                    "This directory is an immutable, content-addressed Combat ",
                    "contract publication. `contract-stamp.json` joins the ",
                    "Combat and shared World dependency tuple; ",
                    "`combat-publication.json` records the Combat-only surface; ",
                    "and `contract-cascade.json` records the shared cascade.",
                    "",
                    "It is not native outcome or trained-policy evidence. ",
                    "Generation-specific identities and validation notes belong ",
                    "in `STATUS.md`.",
                    "",
                )
            ),
            encoding="utf-8",
        )
        (temporary / "STATUS.md").write_text(
            "\n".join(
                (
                    "# Status",
                    "",
                    f"Content address: `{digest}`",
                    f"Repository base: `{repository_git_sha}`",
                    f"Canonical bridge: `{bridge}`",
                    "",
                    "The publisher verified the explicitly selected built and ",
                    "runtime bridge bytes before and after assembling the three ",
                    "JSON records. Behavioral evidence is recorded separately.",
                    "",
                )
            ),
            encoding="utf-8",
        )
        _verify_combat_publication_directory(temporary, values)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink()
            temporary.rmdir()
    _verify_combat_publication_directory(target, values)
    return target


def _verify_combat_publication_directory(
    directory: Path,
    values: dict[str, object],
) -> None:
    if not directory.is_dir():
        raise RuntimeError(f"Combat publication path is not a directory: {directory}")
    for name, expected in values.items():
        path = directory / name
        if not path.is_file():
            raise RuntimeError(f"Combat publication file is missing: {path}")
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise RuntimeError(f"Combat publication file drifted: {path}")


__all__ = [
    "bridge_actor_evidence_contract_sha256",
    "bridge_privileged_npc_component_filter",
    "bridge_string_constant",
    "combat_contract_stamp",
    "combat_contract_stamp_sha256",
    "combat_publication_snapshot",
    "publish_combat_contract_stamp",
    "verify_combat_bridge_artifact_identity",
]
