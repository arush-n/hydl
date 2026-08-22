"""Native traversal sidecar packs for exact WorldGen V2 bundle artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Mapping, Sequence

from hytalegym.worldgen.region.traversal import (
    NativeRegionTraversalGraph,
    region_traversal_graph_contract_sha256,
)

from .bundle import V2JaxArtifact, V2JaxBundle


TRAVERSAL_PACK_SCHEMA = "hytalerl_worldgen_v2_traversal_pack_v1"
TRAVERSAL_PACK_VERSION = 1
TRAVERSAL_PACK_AUTHORITY = "native_MotionControllerWalk_probeMove"

_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "version",
        "authority",
        "source_bundle_semantic_sha256",
        "actor_profile",
        "native_evidence_jar_sha256",
        "graph_contract_sha256",
        "coverage",
        "entries",
        "pack_semantic_sha256",
    }
)
_COVERAGE_FIELDS = frozenset(
    {
        "kind",
        "entry_count",
        "source_artifact_count",
        "complete",
        "structures",
    }
)
_ENTRY_FIELDS = frozenset(
    {
        "structure",
        "seed",
        "source_region_semantic_sha256",
        "path",
        "file_sha256",
        "graph_semantic_sha256",
        "node_count",
        "edge_count",
    }
)
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")


@dataclass(frozen=True)
class V2TraversalEntry:
    """One native traversal graph aligned to one exact V2 Region."""

    structure: str
    seed: int
    source_region_semantic_sha256: str
    path: str
    file_sha256: str
    graph_semantic_sha256: str
    node_count: int
    edge_count: int


@dataclass(frozen=True)
class V2TraversalPack:
    """Validated partial-or-complete native graph coverage for one bundle."""

    manifest_path: Path
    manifest: Mapping[str, Any]
    bundle: V2JaxBundle
    entries: tuple[V2TraversalEntry, ...]

    @classmethod
    def load(
        cls,
        manifest_path: str | Path,
        bundle: V2JaxBundle,
        *,
        verify_graphs: bool = True,
    ) -> "V2TraversalPack":
        if not isinstance(bundle, V2JaxBundle):
            raise TypeError("bundle must be a V2JaxBundle")
        path = Path(manifest_path).resolve()
        value = _json_object(path)
        _exact_fields(value, _MANIFEST_FIELDS, "V2 traversal pack")
        if value["schema"] != TRAVERSAL_PACK_SCHEMA:
            raise ValueError("unsupported V2 traversal pack schema")
        if _exact_int(value["version"], "traversal pack version") != (
            TRAVERSAL_PACK_VERSION
        ):
            raise ValueError("unsupported V2 traversal pack version")
        if value["authority"] != TRAVERSAL_PACK_AUTHORITY:
            raise ValueError("V2 traversal pack lacks native authority")
        if _sha256(
            value["source_bundle_semantic_sha256"],
            "source bundle semantic",
        ) != bundle.semantic_sha256:
            raise ValueError("V2 traversal pack names another source bundle")
        actor_profile = _nonempty(value["actor_profile"], "actor_profile")
        bridge = _sha256(
            value["native_evidence_jar_sha256"],
            "native evidence bridge",
        )
        source_bridge = _sha256(
            bundle.manifest["source"]["bridge_jar_sha256"],
            "source bundle bridge",
        )
        if bridge != source_bridge:
            raise ValueError("V2 traversal and Region evidence bridges differ")
        if _sha256(
            value["graph_contract_sha256"],
            "graph contract",
        ) != region_traversal_graph_contract_sha256():
            raise ValueError("V2 traversal graph contract changed")
        if _sha256(value["pack_semantic_sha256"], "pack semantic") != (
            _pack_digest(value)
        ):
            raise ValueError("V2 traversal pack semantic SHA-256 mismatch")

        rows = value["entries"]
        if not isinstance(rows, list):
            raise TypeError("V2 traversal entries must be a list")
        entries = tuple(_entry(row) for row in rows)
        identities = {
            (row.structure, row.seed, row.source_region_semantic_sha256)
            for row in entries
        }
        if len(identities) != len(entries):
            raise ValueError("V2 traversal entries must be unique")
        bundle_index = {
            (row.structure, row.seed, row.region_semantic_sha256): row
            for row in bundle.artifacts
        }
        if any(identity not in bundle_index for identity in identities):
            raise ValueError("V2 traversal entry does not belong to the bundle")
        if len({row.path for row in entries}) != len(entries):
            raise ValueError("V2 traversal graph paths must be unique")

        coverage = _mapping(value["coverage"], "traversal coverage")
        _exact_fields(coverage, _COVERAGE_FIELDS, "traversal coverage")
        if coverage["kind"] != "explicit_artifact_subset":
            raise ValueError("unsupported V2 traversal coverage kind")
        if _nonnegative_int(coverage["entry_count"], "coverage entry_count") != (
            len(entries)
        ):
            raise ValueError("V2 traversal coverage count changed")
        if _positive_int(
            coverage["source_artifact_count"],
            "coverage source_artifact_count",
        ) != len(bundle.artifacts):
            raise ValueError("V2 traversal source artifact count changed")
        complete = coverage["complete"]
        if not isinstance(complete, bool) or complete != (
            len(entries) == len(bundle.artifacts)
        ):
            raise ValueError("V2 traversal complete-coverage flag changed")
        structures = coverage["structures"]
        if not isinstance(structures, list) or structures != sorted(
            {row.structure for row in entries}
        ):
            raise ValueError("V2 traversal structure coverage changed")

        result = cls(
            manifest_path=path,
            manifest=value,
            bundle=bundle,
            entries=entries,
        )
        if verify_graphs:
            for entry in entries:
                graph = result._load_entry(entry)
                if graph.metadata.get("actor_profile") != actor_profile:
                    raise ValueError("V2 traversal actor profile changed")
                if graph.native_evidence_jar_sha256.lower() != bridge:
                    raise ValueError("V2 traversal graph bridge changed")
                if graph.metadata.get("worldgen_provider") != (
                    bundle.manifest["source"]["worldgen_provider"]
                ):
                    raise ValueError("V2 traversal graph provider changed")
                if _exact_int(graph.metadata.get("seed"), "graph seed") != (
                    entry.seed
                ):
                    raise ValueError("V2 traversal graph seed changed")
        return result

    @property
    def semantic_sha256(self) -> str:
        return str(self.manifest["pack_semantic_sha256"])

    @property
    def complete(self) -> bool:
        return bool(self.manifest["coverage"]["complete"])

    @property
    def actor_profile(self) -> str:
        return str(self.manifest["actor_profile"])

    def covers(self, artifacts: Sequence[V2JaxArtifact]) -> bool:
        available = {
            (row.structure, row.seed, row.source_region_semantic_sha256)
            for row in self.entries
        }
        return all(
            (row.structure, row.seed, row.region_semantic_sha256) in available
            for row in artifacts
        )

    def load_for_artifacts(
        self,
        artifacts: Sequence[V2JaxArtifact],
    ) -> tuple[NativeRegionTraversalGraph, ...]:
        selected = tuple(artifacts)
        by_source = {
            (row.structure, row.seed, row.source_region_semantic_sha256): row
            for row in self.entries
        }
        missing = [
            f"{row.structure}/{row.seed}"
            for row in selected
            if (row.structure, row.seed, row.region_semantic_sha256)
            not in by_source
        ]
        if missing:
            raise ValueError(
                "V2 traversal pack does not cover selected worlds: "
                + ", ".join(missing)
            )
        return tuple(
            self._load_entry(
                by_source[
                    (row.structure, row.seed, row.region_semantic_sha256)
                ]
            )
            for row in selected
        )

    def _load_entry(
        self,
        entry: V2TraversalEntry,
    ) -> NativeRegionTraversalGraph:
        path = _contained_path(self.manifest_path.parent, entry.path)
        if _file_sha256(path) != entry.file_sha256:
            raise ValueError("V2 traversal graph file SHA-256 changed")
        graph = NativeRegionTraversalGraph.load(path)
        if graph.source_region_semantic_sha256 != (
            entry.source_region_semantic_sha256
        ):
            raise ValueError("V2 traversal graph source Region changed")
        if graph.semantic_digest() != entry.graph_semantic_sha256:
            raise ValueError("V2 traversal graph semantic SHA-256 changed")
        if graph.node_count != entry.node_count or graph.edge_count != (
            entry.edge_count
        ):
            raise ValueError("V2 traversal graph census changed")
        return graph


def publish_v2_traversal_pack(
    manifest_path: str | Path,
    bundle: V2JaxBundle,
    graph_paths: Sequence[str | Path],
) -> V2TraversalPack:
    """Publish graph files already stored below the manifest directory."""

    if not isinstance(bundle, V2JaxBundle):
        raise TypeError("bundle must be a V2JaxBundle")
    destination = Path(manifest_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    paths = tuple(Path(value).resolve() for value in graph_paths)
    if not paths:
        raise ValueError("at least one V2 traversal graph is required")
    by_source = {
        row.region_semantic_sha256: row for row in bundle.artifacts
    }
    rows: list[dict[str, Any]] = []
    actor_profiles: set[str] = set()
    bridges: set[str] = set()
    for path in paths:
        relative = path.relative_to(destination.parent).as_posix()
        graph = NativeRegionTraversalGraph.load(path)
        artifact = by_source.get(graph.source_region_semantic_sha256)
        if artifact is None:
            raise ValueError("V2 traversal graph names an unknown Region")
        rows.append(
            {
                "structure": artifact.structure,
                "seed": artifact.seed,
                "source_region_semantic_sha256": (
                    artifact.region_semantic_sha256
                ),
                "path": _relative_path(relative),
                "file_sha256": _file_sha256(path),
                "graph_semantic_sha256": graph.semantic_digest(),
                "node_count": graph.node_count,
                "edge_count": graph.edge_count,
            }
        )
        actor_profiles.add(
            _nonempty(graph.metadata.get("actor_profile"), "actor_profile")
        )
        bridges.add(graph.native_evidence_jar_sha256.lower())
        if _exact_int(graph.metadata.get("seed"), "graph seed") != artifact.seed:
            raise ValueError("V2 traversal graph seed differs from its Region")
        if graph.metadata.get("worldgen_provider") != (
            bundle.manifest["source"]["worldgen_provider"]
        ):
            raise ValueError("V2 traversal graph provider differs from Region")
    if len(actor_profiles) != 1 or len(bridges) != 1:
        raise ValueError("V2 traversal pack provenance must be homogeneous")
    source_bridge = _sha256(
        bundle.manifest["source"]["bridge_jar_sha256"],
        "source bundle bridge",
    )
    if next(iter(bridges)) != source_bridge:
        raise ValueError("V2 traversal and Region evidence bridges differ")
    rows.sort(
        key=lambda row: (
            row["structure"],
            row["seed"],
            row["source_region_semantic_sha256"],
        )
    )
    if len(
        {
            (row["structure"], row["seed"], row["source_region_semantic_sha256"])
            for row in rows
        }
    ) != len(rows):
        raise ValueError("V2 traversal graph sources must be unique")
    manifest: dict[str, Any] = {
        "schema": TRAVERSAL_PACK_SCHEMA,
        "version": TRAVERSAL_PACK_VERSION,
        "authority": TRAVERSAL_PACK_AUTHORITY,
        "source_bundle_semantic_sha256": bundle.semantic_sha256,
        "actor_profile": next(iter(actor_profiles)),
        "native_evidence_jar_sha256": next(iter(bridges)),
        "graph_contract_sha256": region_traversal_graph_contract_sha256(),
        "coverage": {
            "kind": "explicit_artifact_subset",
            "entry_count": len(rows),
            "source_artifact_count": len(bundle.artifacts),
            "complete": len(rows) == len(bundle.artifacts),
            "structures": sorted({str(row["structure"]) for row in rows}),
        },
        "entries": rows,
        "pack_semantic_sha256": "",
    }
    manifest["pack_semantic_sha256"] = _pack_digest(manifest)
    _atomic_json(destination, manifest)
    return V2TraversalPack.load(destination, bundle, verify_graphs=True)


def _entry(value: object) -> V2TraversalEntry:
    row = _mapping(value, "V2 traversal entry")
    _exact_fields(row, _ENTRY_FIELDS, "V2 traversal entry")
    return V2TraversalEntry(
        structure=_nonempty(row["structure"], "structure"),
        seed=_nonnegative_int(row["seed"], "seed"),
        source_region_semantic_sha256=_sha256(
            row["source_region_semantic_sha256"],
            "source Region semantic",
        ),
        path=_relative_path(row["path"]),
        file_sha256=_sha256(row["file_sha256"], "graph file"),
        graph_semantic_sha256=_sha256(
            row["graph_semantic_sha256"],
            "graph semantic",
        ),
        node_count=_positive_int(row["node_count"], "node_count"),
        edge_count=_nonnegative_int(row["edge_count"], "edge_count"),
    )


def _pack_digest(value: Mapping[str, Any]) -> str:
    stable = dict(value)
    stable["pack_semantic_sha256"] = ""
    return hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
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


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} fields changed")


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a SHA-256")
    return value.lower()


def _nonempty(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _exact_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    result = _exact_int(value, label)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _positive_int(value: object, label: str) -> int:
    result = _exact_int(value, label)
    if result < 1:
        raise ValueError(f"{label} must be positive")
    return result


def _relative_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("sidecar path must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("sidecar path must be safe and archive-relative")
    return path.as_posix()


def _contained_path(root: Path, relative: str) -> Path:
    path = (root / PurePosixPath(_relative_path(relative))).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("sidecar path escapes its pack root") from error
    if not path.is_file():
        raise ValueError(f"sidecar file does not exist: {path}")
    return path


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()
