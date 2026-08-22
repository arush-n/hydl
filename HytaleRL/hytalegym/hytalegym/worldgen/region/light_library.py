"""Manifest-backed native light paired with exact frozen Region artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Mapping, Sequence

from hytalegym.worldgen.region.library import (
    RegionArtifactEntry,
    RegionArtifactLibrary,
)
from hytalegym.worldgen.region.light_snapshot import NativeRegionLightSnapshot
from hytalegym.worldgen.region.native_light import (
    native_region_light_section_contract_sha256,
)


REGION_LIGHT_LIBRARY_SCHEMA = "hytalerl_native_region_light_library_v1"
REGION_LIGHT_LIBRARY_VERSION = 1

_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "version",
        "source_region_library_semantic_sha256",
        "region_light_library_contract_sha256",
        "native_region_light_contract_sha256",
        "artifact_count",
        "entries",
        "library_semantic_sha256",
    }
)
_ENTRY_FIELDS = frozenset(
    {
        "path",
        "file_sha256",
        "source_region_semantic_sha256",
        "semantic_sha256",
        "evidence_bridge_sha256",
    }
)


@dataclass(frozen=True)
class RegionLightLibraryEntry:
    """One native-light sidecar and its exact source Region identity."""

    path: str
    file_sha256: str
    source_region_semantic_sha256: str
    semantic_sha256: str
    evidence_bridge_sha256: str


@dataclass(frozen=True)
class RegionLightLibrary:
    """Complete native-light sidecars in physical Region-library order."""

    manifest_path: Path
    source_region_library_semantic_sha256: str
    semantic_sha256: str
    entries: tuple[RegionLightLibraryEntry, ...]

    @classmethod
    def load(
        cls,
        manifest_path: str | Path,
        region_library: RegionArtifactLibrary,
    ) -> RegionLightLibrary:
        if not isinstance(region_library, RegionArtifactLibrary):
            raise TypeError("region_library must be a RegionArtifactLibrary")
        path = Path(manifest_path).resolve()
        manifest = _json_object(path)
        _exact_fields(manifest, _MANIFEST_FIELDS, "light library")
        if manifest["schema"] != REGION_LIGHT_LIBRARY_SCHEMA:
            raise ValueError("unsupported Region light library")
        if _exact_int(manifest["version"], "version") != (
            REGION_LIGHT_LIBRARY_VERSION
        ):
            raise ValueError("unsupported Region light library version")
        source = _sha256(
            manifest["source_region_library_semantic_sha256"],
            "source Region library",
        )
        if source != region_library.semantic_sha256:
            raise ValueError("light library names a different Region library")
        library_contract = _sha256(
            manifest["region_light_library_contract_sha256"],
            "Region light library contract",
        )
        if library_contract != region_light_library_contract_sha256():
            raise ValueError("Region light library contract changed")
        contract = _sha256(
            manifest["native_region_light_contract_sha256"],
            "native Region light contract",
        )
        if contract != native_region_light_section_contract_sha256():
            raise ValueError("native Region light contract changed")
        count = _positive_int(manifest["artifact_count"], "artifact_count")
        raw_entries = manifest["entries"]
        if not isinstance(raw_entries, list) or len(raw_entries) != count:
            raise ValueError("light entries differ from artifact_count")
        entries = tuple(_entry(value) for value in raw_entries)
        _require_complete_pairing(entries, region_library.entries)
        semantic = _sha256(
            manifest["library_semantic_sha256"],
            "library semantic SHA-256",
        )
        if semantic != _library_semantic_sha256(
            source,
            library_contract,
            contract,
            entries,
        ):
            raise ValueError("light library semantic SHA-256 mismatch")
        result = cls(path, source, semantic, entries)
        result.load_for_region_entries(region_library.entries)
        return result

    def load_for_region_entries(
        self,
        region_entries: Sequence[RegionArtifactEntry],
    ) -> tuple[NativeRegionLightSnapshot, ...]:
        """Load sidecars in the caller's exact physical selection order."""

        requested = tuple(region_entries)
        if any(not isinstance(row, RegionArtifactEntry) for row in requested):
            raise TypeError("region_entries must contain RegionArtifactEntry")
        by_source = {
            row.source_region_semantic_sha256: row for row in self.entries
        }
        root = self.manifest_path.parent
        snapshots = []
        for region_entry in requested:
            try:
                entry = by_source[region_entry.semantic_sha256]
            except KeyError as error:
                raise ValueError(
                    "selected Region lacks a native-light sidecar"
                ) from error
            path = _artifact_path(root, entry.path)
            if _file_sha256(path) != entry.file_sha256:
                raise ValueError(
                    f"native-light file SHA-256 mismatch: {entry.path}"
                )
            snapshot = NativeRegionLightSnapshot.load(path)
            if (
                snapshot.source_region_semantic_sha256
                != region_entry.semantic_sha256
                or snapshot.semantic_digest() != entry.semantic_sha256
                or snapshot.evidence_bridge_sha256
                != entry.evidence_bridge_sha256
                or tuple(int(value) for value in snapshot.core_min_chunk_xz)
                != region_entry.core_min_chunk_xz
            ):
                raise ValueError(
                    "native-light sidecar differs from its Region pairing"
                )
            snapshots.append(snapshot)
        return tuple(snapshots)


def create_region_light_library(
    root: str | Path,
    light_paths: Sequence[str | Path],
    region_library: RegionArtifactLibrary,
    *,
    manifest_name: str = "manifest.json",
) -> RegionLightLibrary:
    """Atomically publish one complete one-to-one native-light catalog."""

    if isinstance(light_paths, (str, Path)):
        raise TypeError("light_paths must be a sequence")
    if not isinstance(region_library, RegionArtifactLibrary):
        raise TypeError("region_library must be a RegionArtifactLibrary")
    destination = Path(root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths = tuple(Path(value).resolve() for value in light_paths)
    if not paths:
        raise ValueError("light_paths must not be empty")
    rows = []
    for path in paths:
        snapshot = NativeRegionLightSnapshot.load(path)
        rows.append(
            {
                "path": _relative_path(destination, path),
                "file_sha256": _file_sha256(path),
                "source_region_semantic_sha256": (
                    snapshot.source_region_semantic_sha256
                ),
                "semantic_sha256": snapshot.semantic_digest(),
                "evidence_bridge_sha256": (
                    snapshot.evidence_bridge_sha256
                ),
            }
        )
    rows.sort(key=lambda value: value["source_region_semantic_sha256"])
    entries = tuple(_entry(value) for value in rows)
    _require_complete_pairing(entries, region_library.entries)
    source = region_library.semantic_sha256
    library_contract = region_light_library_contract_sha256()
    contract = native_region_light_section_contract_sha256()
    manifest = {
        "schema": REGION_LIGHT_LIBRARY_SCHEMA,
        "version": REGION_LIGHT_LIBRARY_VERSION,
        "source_region_library_semantic_sha256": source,
        "region_light_library_contract_sha256": library_contract,
        "native_region_light_contract_sha256": contract,
        "artifact_count": len(entries),
        "entries": rows,
        "library_semantic_sha256": _library_semantic_sha256(
            source,
            library_contract,
            contract,
            entries,
        ),
    }
    manifest_path = _artifact_path(destination, manifest_name)
    _write_json_atomic(manifest_path, manifest)
    return RegionLightLibrary.load(manifest_path, region_library)


def _library_semantic_sha256(
    source_region_library_sha256: str,
    region_light_library_contract_sha256: str,
    native_light_contract_sha256: str,
    entries: Sequence[RegionLightLibraryEntry],
) -> str:
    """Hash meaning only; paths, file hashes, and bridge stay provenance."""

    semantic = {
        "schema": REGION_LIGHT_LIBRARY_SCHEMA,
        "version": REGION_LIGHT_LIBRARY_VERSION,
        "source_region_library_semantic_sha256": (
            source_region_library_sha256
        ),
        "region_light_library_contract_sha256": (
            region_light_library_contract_sha256
        ),
        "native_region_light_contract_sha256": (
            native_light_contract_sha256
        ),
        "entries": [
            {
                "source_region_semantic_sha256": (
                    entry.source_region_semantic_sha256
                ),
                "semantic_sha256": entry.semantic_sha256,
            }
            for entry in entries
        ],
    }
    return hashlib.sha256(
        json.dumps(
            semantic,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def region_light_library_contract() -> dict[str, object]:
    """Return the complete exact-pairing contract for native light."""

    return {
        "schema": REGION_LIGHT_LIBRARY_SCHEMA,
        "version": REGION_LIGHT_LIBRARY_VERSION,
        "pairing": "one_sidecar_per_exact_source_region_semantic_sha256",
        "ordering": "source_region_library_order_at_load",
        "semantic_identity": [
            "source_region_library_semantic_sha256",
            "native_region_light_contract_sha256",
            "ordered_source_region_and_light_semantic_sha256_pairs",
        ],
        "provenance_only": [
            "path",
            "file_sha256",
            "evidence_bridge_sha256",
        ],
        "fail_closed": [
            "missing_or_duplicate_source_region",
            "source_region_semantic_mismatch",
            "core_min_chunk_mismatch",
            "sidecar_file_or_semantic_drift",
            "native_light_contract_drift",
        ],
    }


def region_light_library_contract_sha256() -> str:
    payload = json.dumps(
        region_light_library_contract(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_complete_pairing(
    light_entries: Sequence[RegionLightLibraryEntry],
    region_entries: Sequence[RegionArtifactEntry],
) -> None:
    sources = [row.source_region_semantic_sha256 for row in light_entries]
    expected = [row.semantic_sha256 for row in region_entries]
    if len(sources) != len(set(sources)):
        raise ValueError("native-light source Region is duplicated")
    if set(sources) != set(expected) or len(sources) != len(expected):
        raise ValueError("light library must pair every Region exactly once")


def _entry(value: object) -> RegionLightLibraryEntry:
    if not isinstance(value, Mapping):
        raise ValueError("native-light entry must be an object")
    raw = dict(value)
    _exact_fields(raw, _ENTRY_FIELDS, "native-light entry")
    return RegionLightLibraryEntry(
        path=_relative_manifest_path(raw["path"]),
        file_sha256=_sha256(raw["file_sha256"], "file SHA-256"),
        source_region_semantic_sha256=_sha256(
            raw["source_region_semantic_sha256"],
            "source Region semantic SHA-256",
        ),
        semantic_sha256=_sha256(raw["semantic_sha256"], "semantic SHA-256"),
        evidence_bridge_sha256=_sha256(
            raw["evidence_bridge_sha256"],
            "evidence bridge SHA-256",
        ),
    )


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("native-light manifest must be an object")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            encoding="utf-8",
            newline="\n",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _artifact_path(root: Path, relative: object) -> Path:
    clean = _relative_manifest_path(relative)
    path = (root / Path(PurePosixPath(clean))).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("native-light artifact escapes its library")
    return path


def _relative_path(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "native-light artifacts must be inside the library root"
        ) from error
    return _relative_manifest_path(relative.as_posix())


def _relative_manifest_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("artifact path must be a nonempty string")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("artifact path must be a safe relative POSIX path")
    return str(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase 64-hex")
    return value


def _exact_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _positive_int(value: object, label: str) -> int:
    result = _exact_int(value, label)
    if result < 1:
        raise ValueError(f"{label} must be positive")
    return result


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise ValueError(
            f"{label} fields differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


__all__ = [
    "REGION_LIGHT_LIBRARY_SCHEMA",
    "REGION_LIGHT_LIBRARY_VERSION",
    "RegionLightLibrary",
    "RegionLightLibraryEntry",
    "create_region_light_library",
    "region_light_library_contract",
    "region_light_library_contract_sha256",
]
