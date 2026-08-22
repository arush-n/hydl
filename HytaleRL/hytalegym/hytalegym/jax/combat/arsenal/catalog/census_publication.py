"""Stable publication lookup for the data-derived ability census."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from hytalegym.jax.combat.arsenal.catalog.census import (
    hytale_0_5_7_ability_executability_census,
    hytale_0_5_7_ability_executability_census_sha256,
)


PUBLICATION_SCHEMA = (
    "hytalerl_combat_ability_executability_census_publication_v1"
)
REPORT_SCHEMA = "hytalerl_combat_ability_executability_census_v7"
PUBLICATION_FILENAME = "combat-ability-census-current.json"
HYTALE_VERSION = "0.5.7"


def ability_executability_census_publication_path(
    repository_root: str | Path,
) -> Path:
    """Return the stable pointer to the current immutable census report."""

    return Path(repository_root) / "artifacts" / PUBLICATION_FILENAME


def _json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_current_ability_executability_census_publication(
    repository_root: str | Path,
) -> tuple[Path, Path]:
    """Write one immutable census report and atomically move its stable index."""

    root = Path(repository_root)
    census = hytale_0_5_7_ability_executability_census()
    census_sha256 = hytale_0_5_7_ability_executability_census_sha256()
    report_reference = (
        f"combat-ability-census-content-{census_sha256[:8].lower()}/report.json"
    )
    report_path = root / "artifacts" / report_reference
    report = {
        "schema": REPORT_SCHEMA,
        "hytale_version": HYTALE_VERSION,
        "identity_basis": "complete_census_sha256",
        "census_sha256": census_sha256,
        "summary": {
            name: value
            for name, value in census.items()
            if name != "profile_counts"
        },
        "profile_counts": census["profile_counts"],
    }
    report_bytes = _canonical_json(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        if report_path.read_bytes() != report_bytes:
            raise RuntimeError(
                "immutable ability census content already exists with "
                f"different bytes: {report_path}"
            )
    else:
        report_path.write_bytes(report_bytes)

    publication_path = ability_executability_census_publication_path(root)
    publication_path.parent.mkdir(parents=True, exist_ok=True)
    publication = {
        "schema": PUBLICATION_SCHEMA,
        "hytale_version": HYTALE_VERSION,
        "census_sha256": census_sha256,
        "report": report_reference,
        "report_sha256": _file_sha256(report_path),
    }
    temporary_path = publication_path.with_suffix(".json.tmp")
    temporary_path.write_bytes(_canonical_json(publication))
    temporary_path.replace(publication_path)
    return publication_path, report_path


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")


def verify_current_ability_executability_census_publication(
    repository_root: str | Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Verify the stable index, immutable report, and live census currentness."""

    root = Path(repository_root)
    publication_path = ability_executability_census_publication_path(root)
    if not publication_path.is_file():
        raise RuntimeError(
            f"ability census publication index is missing: {publication_path}"
        )
    publication = _json_object(publication_path)
    if publication.get("schema") != PUBLICATION_SCHEMA:
        raise RuntimeError(
            f"ability census publication schema drifted: {publication_path}"
        )
    if publication.get("hytale_version") != HYTALE_VERSION:
        raise RuntimeError(
            f"ability census publication game version drifted: {publication_path}"
        )

    current_sha256 = hytale_0_5_7_ability_executability_census_sha256()
    published_sha256 = publication.get("census_sha256")
    if published_sha256 != current_sha256:
        raise RuntimeError(
            "ability census publication is stale at stable index "
            f"{publication_path}: published={published_sha256}, "
            f"current={current_sha256}"
        )

    expected_reference = (
        f"combat-ability-census-content-{current_sha256[:8].lower()}/report.json"
    )
    report_reference = publication.get("report")
    if report_reference != expected_reference:
        raise RuntimeError(
            "ability census publication does not reference its immutable "
            f"content address: {publication_path}"
        )
    report_path = root / "artifacts" / expected_reference
    if not report_path.is_file():
        raise RuntimeError(f"published ability census is missing: {report_path}")
    if publication.get("report_sha256") != _file_sha256(report_path):
        raise RuntimeError(f"published ability census bytes changed: {report_path}")

    report = _json_object(report_path)
    if report.get("schema") != REPORT_SCHEMA:
        raise RuntimeError(f"published ability census schema drifted: {report_path}")
    if report.get("hytale_version") != HYTALE_VERSION:
        raise RuntimeError(
            f"published ability census game version drifted: {report_path}"
        )
    if report.get("identity_basis") != "complete_census_sha256":
        raise RuntimeError(f"published ability census identity drifted: {report_path}")
    if report.get("census_sha256") != current_sha256:
        raise RuntimeError(f"published ability census digest drifted: {report_path}")

    census = hytale_0_5_7_ability_executability_census()
    expected_summary = {
        name: value for name, value in census.items() if name != "profile_counts"
    }
    if report.get("summary") != expected_summary:
        raise RuntimeError(f"published ability census summary drifted: {report_path}")
    if report.get("profile_counts") != census["profile_counts"]:
        raise RuntimeError(f"published ability profile rows drifted: {report_path}")
    return publication, report


__all__ = [
    "PUBLICATION_FILENAME",
    "PUBLICATION_SCHEMA",
    "REPORT_SCHEMA",
    "HYTALE_VERSION",
    "ability_executability_census_publication_path",
    "write_current_ability_executability_census_publication",
    "verify_current_ability_executability_census_publication",
]
