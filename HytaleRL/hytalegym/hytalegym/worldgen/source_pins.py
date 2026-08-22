"""Installed-build staleness gate for source-derived World contracts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping
from zipfile import BadZipFile, ZipFile

from hytalegym.worldgen.region.contract import HYTALE_SERVER_VERSION


SOURCE_PIN_STALENESS_SCHEMA = "hytalerl_world_source_pin_staleness_v9"
SOURCE_PIN_STALENESS_VERSION = 9

_AUDITS = (
    (
        "perception",
        "surrogate_perception_source_audit.py",
        True,
    ),
    (
        "actor_world_state",
        "surrogate_actor_world_state_source_audit.py",
        False,
    ),
    (
        "locomotion",
        "surrogate_locomotion_source_audit.py",
        True,
    ),
    (
        "navigation",
        "surrogate_navigation_source_audit.py",
        True,
    ),
    (
        "doors",
        "surrogate_door_source_audit.py",
        True,
    ),
    (
        "mutable_blocks",
        "surrogate_mutable_block_source_audit.py",
        False,
    ),
    (
        "explosion_mutation",
        "surrogate_explosion_mutation_source_audit.py",
        True,
    ),
    (
        "explosion_dynamics",
        "surrogate_explosion_dynamics_source_audit.py",
        False,
    ),
    (
        "crafting",
        "surrogate_crafting_source_audit.py",
        True,
    ),
    (
        "inventory",
        "surrogate_inventory_source_audit.py",
        False,
    ),
    (
        "entity_geometry",
        "surrogate_entity_geometry_source_audit.py",
        False,
    ),
)
_MANIFEST_PATH = "META-INF/MANIFEST.MF"


def verify_installed_world_source_pins(
    scripts_directory: str | Path,
    assets: str | Path,
    server_jar: str | Path,
    *,
    python_executable: str = sys.executable,
) -> dict[str, object]:
    """Run every World source audit and return one fail-closed report."""

    scripts = Path(scripts_directory).resolve()
    asset_path = Path(assets).resolve()
    server_path = Path(server_jar).resolve()
    if not scripts.is_dir():
        raise FileNotFoundError(f"source-audit directory is unavailable: {scripts}")
    if not asset_path.is_file():
        raise FileNotFoundError(f"installed Assets.zip is unavailable: {asset_path}")
    if not server_path.is_file():
        raise FileNotFoundError(
            f"installed HytaleServer.jar is unavailable: {server_path}"
        )

    reports: dict[str, dict[str, object]] = {}
    for name, script_name, uses_assets in _AUDITS:
        script = scripts / script_name
        if not script.is_file():
            raise FileNotFoundError(f"World source audit is unavailable: {script}")
        command = [
            python_executable,
            str(script),
            "--server-jar",
            str(server_path),
        ]
        if uses_assets:
            command.extend(("--assets", str(asset_path)))
        completed = subprocess.run(
            command,
            cwd=scripts.parents[1],
            env=_source_audit_environment(scripts.parents[1]),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            detail = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(
                f"World source-pin audit failed ({name}): "
                f"{detail[-1000:] or 'no diagnostic'}"
            )
        try:
            report = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"World source-pin audit emitted invalid JSON ({name})"
            ) from error
        if not isinstance(report, dict):
            raise RuntimeError(f"World source-pin audit root is not an object ({name})")
        reports[name] = report
    return world_source_pin_staleness_report(server_path, reports)


def _source_audit_environment(project_root: Path) -> dict[str, str]:
    """Keep the package root absolute when audit subprocesses change cwd."""

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root.resolve())
    return environment


def world_source_pin_staleness_report(
    server_jar: str | Path,
    reports: Mapping[str, Mapping[str, object]],
    *,
    expected_version: str = HYTALE_SERVER_VERSION,
) -> dict[str, object]:
    """Validate installed identity and already-produced audit reports."""

    server_path = Path(server_jar).resolve()
    identity = installed_hytale_server_identity(server_path)
    observed_version = identity["implementation_version"]
    if observed_version != expected_version:
        raise RuntimeError(
            "installed Hytale version invalidates World source pins: "
            f"observed={observed_version} expected={expected_version}"
        )
    expected_names = tuple(name for name, _, _ in _AUDITS)
    if set(reports) != set(expected_names):
        raise RuntimeError(
            "World source-pin report set differs from the required registry"
        )

    server_sha256 = identity["server_jar_sha256"]
    audit_rows = []
    for name in expected_names:
        report = reports[name]
        if report.get("matches_0_5_7_baseline") is not True:
            raise RuntimeError(f"World source pin is stale: {name}")
        observed_server = _sha256_text(report.get("server_sha256"), name)
        expected_server = _sha256_text(
            report.get("expected_server_sha256"),
            name,
        )
        if observed_server != server_sha256 or expected_server != server_sha256:
            raise RuntimeError(f"World source pin targets another server build: {name}")
        semantic = _sha256_text(report.get("semantic_sha256"), name)
        expected_semantic = _sha256_text(
            report.get("expected_semantic_sha256"),
            name,
        )
        if semantic != expected_semantic:
            raise RuntimeError(f"World source semantics are stale: {name}")
        schema = report.get("schema")
        if not isinstance(schema, str) or not schema:
            raise RuntimeError(f"World source audit schema is invalid: {name}")
        audit_rows.append(
            {
                "name": name,
                "schema": schema,
                "semantic_sha256": semantic,
                "verified_class_count": _mapping_size(
                    report.get("class_sha256"),
                    f"{name}.class_sha256",
                ),
                "verified_asset_count": _optional_mapping_size(
                    report.get("asset_sha256"),
                    f"{name}.asset_sha256",
                ),
            }
        )

    result: dict[str, object] = {
        "schema": SOURCE_PIN_STALENESS_SCHEMA,
        "version": SOURCE_PIN_STALENESS_VERSION,
        "status": "current",
        "expected_hytale_version": expected_version,
        "installed_build": identity,
        "audits": audit_rows,
        "invalidation": (
            "version, server JAR, pinned class, pinned asset, or semantic drift"
        ),
    }
    result["report_semantic_sha256"] = _semantic_sha256(result)
    return result


def installed_hytale_server_identity(
    server_jar: str | Path,
) -> dict[str, str]:
    """Read version and revision from the installed server JAR itself."""

    path = Path(server_jar).resolve()
    try:
        with ZipFile(path) as archive:
            manifest = archive.read(_MANIFEST_PATH)
    except (BadZipFile, KeyError, OSError) as error:
        raise RuntimeError("installed server JAR manifest is unavailable") from error
    fields = _manifest_fields(manifest)
    title = fields.get("Implementation-Title")
    version = fields.get("Implementation-Version")
    revision = fields.get("Implementation-Revision-Id")
    if title != "Server" or not version or not revision:
        raise RuntimeError("installed server JAR build identity is incomplete")
    return {
        "implementation_title": title,
        "implementation_version": version,
        "implementation_revision": revision,
        "server_jar_sha256": _file_sha256(path),
    }


def _manifest_fields(raw: bytes) -> dict[str, str]:
    fields: dict[str, str] = {}
    current: str | None = None
    for line in raw.decode("utf-8").splitlines():
        if line.startswith(" ") and current is not None:
            fields[current] += line[1:]
            continue
        if ": " not in line:
            current = None
            continue
        current, value = line.split(": ", 1)
        fields[current] = value
    return fields


def _mapping_size(value: object, label: str) -> int:
    if not isinstance(value, dict) or not value:
        raise RuntimeError(f"World source audit mapping is invalid: {label}")
    return len(value)


def _optional_mapping_size(value: object, label: str) -> int:
    if value is None:
        return 0
    if not isinstance(value, dict) or not value:
        raise RuntimeError(f"World source audit mapping is invalid: {label}")
    return len(value)


def _sha256_text(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise RuntimeError(f"World source audit SHA-256 is invalid: {label}")
    return value.lower()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _semantic_sha256(value: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "SOURCE_PIN_STALENESS_SCHEMA",
    "SOURCE_PIN_STALENESS_VERSION",
    "installed_hytale_server_identity",
    "verify_installed_world_source_pins",
    "world_source_pin_staleness_report",
]
