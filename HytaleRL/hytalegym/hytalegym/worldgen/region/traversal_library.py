"""Pinned native traversal graphs aligned to one exact Region library."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from hytalegym.worldgen.region.library import (
    RegionArtifactEntry,
    RegionArtifactLibrary,
)
from hytalegym.worldgen.region.traversal import (
    REGION_TRAVERSAL_GRAPH_SCHEMA,
    REGION_TRAVERSAL_GRAPH_VERSION,
    NativeRegionTraversalGraph,
    region_traversal_graph_contract,
)

REGION_TRAVERSAL_LIBRARY_SCHEMA = (
    "hytalerl_native_region_traversal_library_v2"
)
REGION_TRAVERSAL_LIBRARY_VERSION = 2
_LIBRARY_SCHEMAS = {
    1: "hytalerl_native_region_traversal_library_v1",
    REGION_TRAVERSAL_LIBRARY_VERSION: REGION_TRAVERSAL_LIBRARY_SCHEMA,
}
_FIELDS = frozenset(
    {
        "schema",
        "version",
        "source_region_library_semantic_sha256",
        "actor_profile",
        "native_evidence_jar_sha256",
        "graph_schema",
        "graph_version",
        "graph_count",
        "entries",
        "library_semantic_sha256",
    }
)
_ENTRY_FIELDS = frozenset(
    {
        "source_region_semantic_sha256",
        "path",
        "file_sha256",
        "graph_semantic_sha256",
    }
)


@dataclass(frozen=True)
class RegionTraversalGraphEntry:
    source_region_semantic_sha256: str
    path: str
    file_sha256: str
    graph_semantic_sha256: str


@dataclass(frozen=True)
class RegionTraversalGraphLibrary:
    """Validated one-to-one graph sidecars for a Region artifact library."""

    manifest_path: Path
    manifest: Mapping[str, Any]
    entries: tuple[RegionTraversalGraphEntry, ...]

    @classmethod
    def load(
        cls,
        manifest_path: str | Path,
        region_library: RegionArtifactLibrary,
    ) -> RegionTraversalGraphLibrary:
        if not isinstance(region_library, RegionArtifactLibrary):
            raise TypeError("region_library must be a RegionArtifactLibrary")
        source = Path(manifest_path).resolve()
        manifest = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or set(manifest) != _FIELDS:
            raise ValueError("Region traversal library fields changed")
        version = _supported_library_version(manifest["version"])
        if manifest["schema"] != _LIBRARY_SCHEMAS[version]:
            raise ValueError("unsupported Region traversal library schema")
        graph_version = _exact_nonnegative_int(
            manifest["graph_version"],
            "Region traversal graph version",
        )
        if version == 1 and graph_version != 1:
            raise ValueError("legacy traversal library requires graph v1")
        graph_contract = region_traversal_graph_contract(graph_version)
        if manifest["graph_schema"] != graph_contract["schema"]:
            raise ValueError("Region traversal graph schema changed")
        if manifest["source_region_library_semantic_sha256"] != (
            region_library.semantic_sha256
        ):
            raise ValueError("Region traversal and terrain libraries differ")
        actor_profile = _nonempty(
            manifest["actor_profile"],
            "actor_profile",
        )
        bridge = _sha256(
            manifest["native_evidence_jar_sha256"],
            "native evidence bridge",
        ).upper()
        raw_entries = manifest["entries"]
        if (
            not isinstance(raw_entries, list)
            or manifest["graph_count"] != len(raw_entries)
        ):
            raise ValueError("Region traversal graph count changed")
        entries = tuple(_entry(value) for value in raw_entries)
        expected_sources = {
            value.semantic_sha256 for value in region_library.entries
        }
        actual_sources = {
            value.source_region_semantic_sha256 for value in entries
        }
        if (
            len(entries) != len(actual_sources)
            or actual_sources != expected_sources
        ):
            raise ValueError(
                "Region traversal library must cover every Region exactly once"
            )
        if manifest["library_semantic_sha256"] != (
            region_traversal_library_semantic_sha256(manifest)
        ):
            raise ValueError("Region traversal library semantic hash changed")
        result = cls(
            manifest_path=source,
            manifest=manifest,
            entries=entries,
        )
        for entry in entries:
            graph = result._load_entry(entry)
            if graph.metadata["actor_profile"] != actor_profile:
                raise ValueError("Region traversal actor profiles differ")
            if graph.native_evidence_jar_sha256 != bridge:
                raise ValueError("Region traversal evidence bridges differ")
        return result

    @property
    def semantic_sha256(self) -> str:
        return str(self.manifest["library_semantic_sha256"])

    @property
    def actor_profile(self) -> str:
        return str(self.manifest["actor_profile"])

    def load_for_region_entries(
        self,
        entries: Sequence[RegionArtifactEntry],
    ) -> tuple[NativeRegionTraversalGraph, ...]:
        by_source = {
            value.source_region_semantic_sha256: value
            for value in self.entries
        }
        return tuple(
            self._load_entry(by_source[value.semantic_sha256])
            for value in entries
        )

    def _load_entry(
        self,
        entry: RegionTraversalGraphEntry,
    ) -> NativeRegionTraversalGraph:
        path = _contained_path(self.manifest_path.parent, entry.path)
        if _file_sha256(path) != entry.file_sha256:
            raise ValueError("Region traversal graph file hash changed")
        graph = NativeRegionTraversalGraph.load(path)
        if graph.source_region_semantic_sha256 != (
            entry.source_region_semantic_sha256
        ):
            raise ValueError("Region traversal graph source changed")
        if graph.semantic_digest() != entry.graph_semantic_sha256:
            raise ValueError("Region traversal graph semantic hash changed")
        return graph


def create_region_traversal_graph_library(
    root: str | Path,
    region_library: RegionArtifactLibrary,
    graph_paths: Sequence[str | Path],
    *,
    manifest_name: str = "traversal-manifest.json",
) -> RegionTraversalGraphLibrary:
    """Atomically publish complete graph sidecars for one Region library."""

    if not isinstance(region_library, RegionArtifactLibrary):
        raise TypeError("region_library must be a RegionArtifactLibrary")
    destination = Path(root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths = tuple(Path(value).resolve() for value in graph_paths)
    if len(paths) != len(region_library.entries):
        raise ValueError("one traversal graph is required per Region artifact")
    rows = []
    actor_profiles = set()
    bridges = set()
    for path in paths:
        graph = NativeRegionTraversalGraph.load(path)
        if graph.metadata["version"] != REGION_TRAVERSAL_GRAPH_VERSION:
            raise ValueError("new traversal libraries require current graphs")
        relative = path.relative_to(destination).as_posix()
        rows.append(
            {
                "source_region_semantic_sha256": (
                    graph.source_region_semantic_sha256
                ),
                "path": relative,
                "file_sha256": _file_sha256(path),
                "graph_semantic_sha256": graph.semantic_digest(),
            }
        )
        actor_profiles.add(str(graph.metadata["actor_profile"]))
        bridges.add(graph.native_evidence_jar_sha256)
    expected = {
        value.semantic_sha256 for value in region_library.entries
    }
    if {value["source_region_semantic_sha256"] for value in rows} != expected:
        raise ValueError("traversal graphs do not match the Region library")
    if len(actor_profiles) != 1 or len(bridges) != 1:
        raise ValueError("traversal graph provenance must be homogeneous")
    rows.sort(key=lambda value: value["source_region_semantic_sha256"])
    manifest: dict[str, Any] = {
        "schema": REGION_TRAVERSAL_LIBRARY_SCHEMA,
        "version": REGION_TRAVERSAL_LIBRARY_VERSION,
        "source_region_library_semantic_sha256": (
            region_library.semantic_sha256
        ),
        "actor_profile": next(iter(actor_profiles)),
        "native_evidence_jar_sha256": next(iter(bridges)),
        "graph_schema": REGION_TRAVERSAL_GRAPH_SCHEMA,
        "graph_version": REGION_TRAVERSAL_GRAPH_VERSION,
        "graph_count": len(rows),
        "entries": rows,
        "library_semantic_sha256": "",
    }
    manifest["library_semantic_sha256"] = (
        region_traversal_library_semantic_sha256(manifest)
    )
    target = _contained_path(destination, manifest_name)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(manifest, output, sort_keys=True, separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return RegionTraversalGraphLibrary.load(target, region_library)


def _entry(value: object) -> RegionTraversalGraphEntry:
    if not isinstance(value, dict) or set(value) != _ENTRY_FIELDS:
        raise ValueError("Region traversal library entry fields changed")
    return RegionTraversalGraphEntry(
        source_region_semantic_sha256=_sha256(
            value["source_region_semantic_sha256"],
            "source Region semantic SHA-256",
        ),
        path=_relative_path(value["path"]),
        file_sha256=_sha256(value["file_sha256"], "graph file SHA-256"),
        graph_semantic_sha256=_sha256(
            value["graph_semantic_sha256"],
            "graph semantic SHA-256",
        ),
    )


def region_traversal_library_semantic_sha256(
    manifest: Mapping[str, Any],
) -> str:
    """Return the version-aware graph-library meaning identity."""

    if not isinstance(manifest, Mapping) or set(manifest) != _FIELDS:
        raise ValueError("Region traversal library fields changed")
    version = _supported_library_version(manifest["version"])
    if manifest["schema"] != _LIBRARY_SCHEMAS[version]:
        raise ValueError("unsupported Region traversal library schema")
    graph_version = _exact_nonnegative_int(
        manifest["graph_version"],
        "Region traversal graph version",
    )
    if version == 1 and graph_version != 1:
        raise ValueError("legacy traversal library requires graph v1")
    expected_graph = region_traversal_graph_contract(graph_version)
    if (
        manifest["graph_schema"] != expected_graph["schema"]
    ):
        raise ValueError("Region traversal graph schema changed")
    source = _sha256(
        manifest["source_region_library_semantic_sha256"],
        "source Region library semantic SHA-256",
    )
    actor_profile = _nonempty(manifest["actor_profile"], "actor_profile")
    _sha256(
        manifest["native_evidence_jar_sha256"],
        "native evidence bridge",
    )
    raw_entries = manifest["entries"]
    if not isinstance(raw_entries, list):
        raise ValueError("Region traversal entries must be a list")
    graph_count = _exact_nonnegative_int(
        manifest["graph_count"],
        "graph_count",
    )
    if graph_count != len(raw_entries):
        raise ValueError("Region traversal graph count changed")
    entries = tuple(_entry(value) for value in raw_entries)
    if version == 1:
        stable = dict(manifest)
        stable["library_semantic_sha256"] = ""
    else:
        stable = {
            "schema": manifest["schema"],
            "version": version,
            "source_region_library_semantic_sha256": source,
            "actor_profile": actor_profile,
            "graph_schema": manifest["graph_schema"],
            "graph_version": graph_version,
            "graph_count": graph_count,
            "graphs": sorted(
                (
                    {
                        "source_region_semantic_sha256": (
                            entry.source_region_semantic_sha256
                        ),
                        "graph_semantic_sha256": (
                            entry.graph_semantic_sha256
                        ),
                    }
                    for entry in entries
                ),
                key=lambda value: value[
                    "source_region_semantic_sha256"
                ],
            ),
        }
    return hashlib.sha256(
        json.dumps(
            stable,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    return region_traversal_library_semantic_sha256(manifest)


def _contained_path(root: Path, relative: str | Path) -> Path:
    path = (root / _relative_path(str(relative))).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("Region traversal path escapes its library") from error
    return path


def _relative_path(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).is_absolute()
        or "\\" in value
        or ".." in Path(value).parts
    ):
        raise ValueError("Region traversal paths must be safe POSIX relatives")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{label} must be a SHA-256")
    return value.lower()


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty")
    return value


def _exact_nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    return value


def _supported_library_version(value: object) -> int:
    version = _exact_nonnegative_int(
        value,
        "Region traversal library version",
    )
    if version not in _LIBRARY_SCHEMAS:
        raise ValueError("unsupported Region traversal library version")
    return version


__all__ = [
    "REGION_TRAVERSAL_LIBRARY_SCHEMA",
    "REGION_TRAVERSAL_LIBRARY_VERSION",
    "RegionTraversalGraphEntry",
    "RegionTraversalGraphLibrary",
    "create_region_traversal_graph_library",
    "region_traversal_library_semantic_sha256",
]
