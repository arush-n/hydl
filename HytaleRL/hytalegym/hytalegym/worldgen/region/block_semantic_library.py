"""Manifest-backed block semantics paired with frozen Region artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Mapping, Sequence

from hytalegym.worldgen.region.block_semantics import (
    NativeRegionBlockSemanticSnapshot,
    region_block_semantic_contract_sha256,
    region_block_semantic_projection_capture_contract_sha256,
)
from hytalegym.worldgen.region.library import (
    RegionArtifactEntry,
    RegionArtifactLibrary,
)


REGION_BLOCK_SEMANTIC_LIBRARY_SCHEMA = (
    "hytalerl_region_block_semantic_library_v1"
)
REGION_BLOCK_SEMANTIC_LIBRARY_VERSION = 1

_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "version",
        "source_region_library_semantic_sha256",
        "block_semantic_contract_sha256",
        "physical_alignment_contract_sha256",
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
class RegionBlockSemanticLibraryEntry:
    path: str
    file_sha256: str
    source_region_semantic_sha256: str
    semantic_sha256: str
    evidence_bridge_sha256: str


@dataclass(frozen=True)
class RegionBlockSemanticLibrary:
    """Complete semantic sidecars in physical Region-library order."""

    manifest_path: Path
    source_region_library_semantic_sha256: str
    semantic_sha256: str
    entries: tuple[RegionBlockSemanticLibraryEntry, ...]

    @classmethod
    def load(
        cls,
        manifest_path: str | Path,
        region_library: RegionArtifactLibrary,
    ) -> "RegionBlockSemanticLibrary":
        if not isinstance(region_library, RegionArtifactLibrary):
            raise TypeError("region_library must be a RegionArtifactLibrary")
        path = Path(manifest_path).resolve()
        manifest = _json_object(path)
        _exact_fields(manifest, _MANIFEST_FIELDS, "semantic library")
        if manifest["schema"] != REGION_BLOCK_SEMANTIC_LIBRARY_SCHEMA:
            raise ValueError("unsupported Region block-semantic library")
        if (
            _exact_int(manifest["version"], "version")
            != REGION_BLOCK_SEMANTIC_LIBRARY_VERSION
        ):
            raise ValueError("unsupported Region block-semantic version")
        source = _sha256(
            manifest["source_region_library_semantic_sha256"],
            "source Region library",
        )
        if source != region_library.semantic_sha256:
            raise ValueError(
                "block-semantic library names a different Region library"
            )
        contract = _sha256(
            manifest["block_semantic_contract_sha256"],
            "block-semantic contract",
        )
        if contract != region_block_semantic_contract_sha256():
            raise ValueError("block-semantic contract changed")
        alignment = _sha256(
            manifest["physical_alignment_contract_sha256"],
            "physical alignment contract",
        )
        if alignment != (
            region_block_semantic_projection_capture_contract_sha256()
        ):
            raise ValueError("block-semantic physical alignment changed")
        count = _positive_int(manifest["artifact_count"], "artifact_count")
        raw_entries = manifest["entries"]
        if not isinstance(raw_entries, list) or len(raw_entries) != count:
            raise ValueError(
                "block-semantic entries differ from artifact_count"
            )
        entries = tuple(_entry(value) for value in raw_entries)
        _require_complete_pairing(entries, region_library.entries)
        semantic = _sha256(
            manifest["library_semantic_sha256"],
            "library semantic SHA-256",
        )
        if semantic != _library_semantic_sha256(
            source,
            contract,
            alignment,
            entries,
        ):
            raise ValueError(
                "block-semantic library semantic SHA-256 mismatch"
            )
        library = cls(path, source, semantic, entries)
        library.load_for_region_entries(region_library.entries)
        return library

    def load_for_region_entries(
        self,
        region_entries: Sequence[RegionArtifactEntry],
    ) -> tuple[NativeRegionBlockSemanticSnapshot, ...]:
        """Load sidecars in the caller's exact physical selection order."""

        requested = tuple(region_entries)
        if any(
            not isinstance(entry, RegionArtifactEntry) for entry in requested
        ):
            raise TypeError("region_entries must contain RegionArtifactEntry")
        by_source = {
            entry.source_region_semantic_sha256: entry
            for entry in self.entries
        }
        root = self.manifest_path.parent
        snapshots = []
        for region_entry in requested:
            try:
                entry = by_source[region_entry.semantic_sha256]
            except KeyError as error:
                raise ValueError(
                    "selected Region lacks a block-semantic sidecar"
                ) from error
            path = _artifact_path(root, entry.path)
            if _file_sha256(path) != entry.file_sha256:
                raise ValueError(
                    f"block-semantic file SHA-256 mismatch: {entry.path}"
                )
            snapshot = NativeRegionBlockSemanticSnapshot.load(path)
            if (
                snapshot.source_region_semantic_sha256
                != region_entry.semantic_sha256
                or snapshot.semantic_sha256() != entry.semantic_sha256
                or snapshot.evidence_bridge_sha256
                != entry.evidence_bridge_sha256
                or tuple(int(value) for value in snapshot.core_min_chunk_xz)
                != region_entry.core_min_chunk_xz
            ):
                raise ValueError(
                    "block-semantic sidecar differs from its manifest pairing"
                )
            snapshots.append(snapshot)
        return tuple(snapshots)


def create_region_block_semantic_library(
    root: str | Path,
    semantic_paths: Sequence[str | Path],
    region_library: RegionArtifactLibrary,
    *,
    manifest_name: str = "manifest.json",
) -> RegionBlockSemanticLibrary:
    """Atomically publish one complete one-to-one semantic-sidecar catalog."""

    if isinstance(semantic_paths, (str, Path)):
        raise TypeError("semantic_paths must be a sequence")
    if not isinstance(region_library, RegionArtifactLibrary):
        raise TypeError("region_library must be a RegionArtifactLibrary")
    destination = Path(root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths = tuple(Path(value).resolve() for value in semantic_paths)
    if not paths:
        raise ValueError("semantic_paths must not be empty")
    rows = []
    for path in paths:
        snapshot = NativeRegionBlockSemanticSnapshot.load(path)
        rows.append(
            {
                "path": _relative_path(destination, path),
                "file_sha256": _file_sha256(path),
                "source_region_semantic_sha256": (
                    snapshot.source_region_semantic_sha256
                ),
                "semantic_sha256": snapshot.semantic_sha256(),
                "evidence_bridge_sha256": (
                    snapshot.evidence_bridge_sha256
                ),
            }
        )
    rows.sort(key=lambda value: value["source_region_semantic_sha256"])
    entries = tuple(_entry(value) for value in rows)
    _require_complete_pairing(entries, region_library.entries)
    source = region_library.semantic_sha256
    contract = region_block_semantic_contract_sha256()
    alignment = region_block_semantic_projection_capture_contract_sha256()
    manifest = {
        "schema": REGION_BLOCK_SEMANTIC_LIBRARY_SCHEMA,
        "version": REGION_BLOCK_SEMANTIC_LIBRARY_VERSION,
        "source_region_library_semantic_sha256": source,
        "block_semantic_contract_sha256": contract,
        "physical_alignment_contract_sha256": alignment,
        "artifact_count": len(entries),
        "entries": rows,
        "library_semantic_sha256": _library_semantic_sha256(
            source,
            contract,
            alignment,
            entries,
        ),
    }
    _write_json_atomic(
        _artifact_path(destination, manifest_name),
        manifest,
    )
    return RegionBlockSemanticLibrary.load(
        destination / manifest_name,
        region_library,
    )


def _library_semantic_sha256(
    source_region_library_sha256: str,
    block_semantic_contract_sha256: str,
    physical_alignment_contract_sha256: str,
    entries: Sequence[RegionBlockSemanticLibraryEntry],
) -> str:
    """Hash meaning only; paths, file hashes, and bridge provenance stay out."""

    semantic = {
        "schema": REGION_BLOCK_SEMANTIC_LIBRARY_SCHEMA,
        "version": REGION_BLOCK_SEMANTIC_LIBRARY_VERSION,
        "source_region_library_semantic_sha256": (
            source_region_library_sha256
        ),
        "block_semantic_contract_sha256": (
            block_semantic_contract_sha256
        ),
        "physical_alignment_contract_sha256": (
            physical_alignment_contract_sha256
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


def _require_complete_pairing(
    semantic_entries: Sequence[RegionBlockSemanticLibraryEntry],
    region_entries: Sequence[RegionArtifactEntry],
) -> None:
    sources = [
        entry.source_region_semantic_sha256 for entry in semantic_entries
    ]
    expected = [entry.semantic_sha256 for entry in region_entries]
    if len(sources) != len(set(sources)):
        raise ValueError("block-semantic source Region is duplicated")
    if set(sources) != set(expected) or len(sources) != len(expected):
        raise ValueError(
            "block-semantic library must pair every Region exactly once"
        )


def _entry(value: object) -> RegionBlockSemanticLibraryEntry:
    if not isinstance(value, Mapping):
        raise ValueError("block-semantic entry must be an object")
    raw = dict(value)
    _exact_fields(raw, _ENTRY_FIELDS, "block-semantic entry")
    return RegionBlockSemanticLibraryEntry(
        path=_relative_manifest_path(raw["path"]),
        file_sha256=_sha256(raw["file_sha256"], "file SHA-256"),
        source_region_semantic_sha256=_sha256(
            raw["source_region_semantic_sha256"],
            "source Region semantic SHA-256",
        ),
        semantic_sha256=_sha256(
            raw["semantic_sha256"],
            "semantic SHA-256",
        ),
        evidence_bridge_sha256=_sha256(
            raw["evidence_bridge_sha256"],
            "evidence bridge SHA-256",
        ),
    )


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("block-semantic manifest must be an object")
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
        raise ValueError("block-semantic artifact escapes its library")
    return path


def _relative_path(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            "block-semantic artifacts must be inside the library root"
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
            f"{label} fields differ: "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


__all__ = [
    "REGION_BLOCK_SEMANTIC_LIBRARY_SCHEMA",
    "REGION_BLOCK_SEMANTIC_LIBRARY_VERSION",
    "RegionBlockSemanticLibrary",
    "RegionBlockSemanticLibraryEntry",
    "create_region_block_semantic_library",
]
