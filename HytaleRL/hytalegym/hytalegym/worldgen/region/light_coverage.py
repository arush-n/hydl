"""Complete native-light availability over one frozen Region library."""

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
from hytalegym.worldgen.region.snapshot import NativeRegionSnapshot
from hytalegym.worldgen.region.stability import (
    RegionSemanticComparison,
    compare_region_semantics,
)


REGION_LIGHT_COVERAGE_SCHEMA = "hytalerl_native_region_light_coverage_v1"
REGION_LIGHT_COVERAGE_VERSION = 1
REGION_LIGHT_UNAVAILABLE_FLUID_DRIFT = (
    "cross_reset_non_damaging_fluid_semantic_drift"
)
REGION_LIGHT_UNAVAILABLE_SEMANTIC_DRIFT = (
    "cross_reset_region_semantic_mismatch"
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "version",
        "source_region_library_semantic_sha256",
        "contract_sha256",
        "artifact_count",
        "available_count",
        "entries",
        "library_semantic_sha256",
    }
)
_ENTRY_FIELDS = frozenset(
    {
        "source_region_semantic_sha256",
        "available",
        "path",
        "file_sha256",
        "evidence_semantic_sha256",
        "evidence_bridge_sha256",
        "unavailable_reason",
        "differing_cell_count",
        "non_damaging_fluid_only_cell_count",
        "other_differing_cell_count",
        "seed_projection_equal",
    }
)


@dataclass(frozen=True)
class RegionLightCoverageEntry:
    """One exact light sidecar or one measured fail-closed rejection."""

    source_region_semantic_sha256: str
    available: bool
    path: str
    file_sha256: str
    evidence_semantic_sha256: str
    evidence_bridge_sha256: str
    unavailable_reason: str
    differing_cell_count: int
    non_damaging_fluid_only_cell_count: int
    other_differing_cell_count: int
    seed_projection_equal: bool


@dataclass(frozen=True)
class RegionLightCoverageLibrary:
    """One-to-one light availability in frozen Region-library order."""

    manifest_path: Path
    source_region_library_semantic_sha256: str
    semantic_sha256: str
    entries: tuple[RegionLightCoverageEntry, ...]

    @classmethod
    def load(
        cls,
        manifest_path: str | Path,
        region_library: RegionArtifactLibrary,
    ) -> RegionLightCoverageLibrary:
        if not isinstance(region_library, RegionArtifactLibrary):
            raise TypeError("region_library must be a RegionArtifactLibrary")
        path = Path(manifest_path).resolve()
        value = _object(json.loads(path.read_text(encoding="utf-8")), "manifest")
        _exact_fields(value, _MANIFEST_FIELDS, "light coverage manifest")
        if value.get("schema") != REGION_LIGHT_COVERAGE_SCHEMA:
            raise ValueError("unsupported Region light coverage schema")
        if _exact_int(value.get("version"), "coverage version") != (
            REGION_LIGHT_COVERAGE_VERSION
        ):
            raise ValueError("unsupported Region light coverage version")
        source = _sha256(
            value.get("source_region_library_semantic_sha256"),
            "source Region library",
        )
        if source != region_library.semantic_sha256:
            raise ValueError("light coverage names a different Region library")
        if _sha256(value.get("contract_sha256"), "coverage contract") != (
            region_light_coverage_contract_sha256()
        ):
            raise ValueError("Region light coverage contract changed")
        raw_entries = value.get("entries")
        if not isinstance(raw_entries, list):
            raise ValueError("Region light coverage entries must be a list")
        entries = tuple(_entry(row) for row in raw_entries)
        if len({row.evidence_bridge_sha256 for row in entries}) != 1:
            raise ValueError("Region light coverage mixes evidence bridges")
        expected_sources = tuple(
            row.semantic_sha256 for row in region_library.entries
        )
        if tuple(row.source_region_semantic_sha256 for row in entries) != (
            expected_sources
        ):
            raise ValueError("light coverage must preserve Region-library order")
        if _nonnegative_int(
            value.get("artifact_count"),
            "artifact count",
        ) != len(entries):
            raise ValueError("light coverage artifact count changed")
        available_count = sum(row.available for row in entries)
        if _nonnegative_int(
            value.get("available_count"),
            "available count",
        ) != available_count:
            raise ValueError("light coverage available count changed")
        semantic = _sha256(
            value.get("library_semantic_sha256"),
            "light coverage semantic SHA-256",
        )
        if semantic != _semantic_sha256(source, entries):
            raise ValueError("light coverage semantic SHA-256 mismatch")
        result = cls(path, source, semantic, entries)
        result.load_optional_for_region_entries(
            region_library.entries,
            region_library=region_library,
        )
        return result

    def load_optional_for_region_entries(
        self,
        region_entries: Sequence[RegionArtifactEntry],
        *,
        region_library: RegionArtifactLibrary,
    ) -> tuple[NativeRegionLightSnapshot | None, ...]:
        """Load selected exact sidecars while retaining unavailable slots."""

        if not isinstance(region_library, RegionArtifactLibrary):
            raise TypeError("region_library must be a RegionArtifactLibrary")
        if (
            region_library.semantic_sha256
            != self.source_region_library_semantic_sha256
        ):
            raise ValueError("light coverage and Region library differ")
        requested = tuple(region_entries)
        if any(not isinstance(row, RegionArtifactEntry) for row in requested):
            raise TypeError("region_entries must contain RegionArtifactEntry")
        by_source = {
            row.source_region_semantic_sha256: row for row in self.entries
        }
        root = self.manifest_path.parent
        snapshots = []
        for region_entry in requested:
            row = by_source.get(region_entry.semantic_sha256)
            if row is None:
                raise ValueError("selected Region is absent from light coverage")
            evidence_path = _artifact_path(root, row.path)
            if _file_sha256(evidence_path) != row.file_sha256:
                raise ValueError("Region light coverage file SHA-256 mismatch")
            if row.available:
                snapshot = NativeRegionLightSnapshot.load(evidence_path)
                if (
                    snapshot.source_region_semantic_sha256
                    != region_entry.semantic_sha256
                    or snapshot.semantic_digest() != row.evidence_semantic_sha256
                    or snapshot.evidence_bridge_sha256
                    != row.evidence_bridge_sha256
                    or tuple(int(value) for value in snapshot.core_min_chunk_xz)
                    != region_entry.core_min_chunk_xz
                ):
                    raise ValueError("native-light coverage sidecar drifted")
                snapshots.append(snapshot)
                continue
            rejected = NativeRegionSnapshot.load(evidence_path)
            expected = NativeRegionSnapshot.load(
                region_library.manifest_path.parent / region_entry.path
            )
            comparison = compare_region_semantics(
                expected,
                rejected,
                require_same_world=False,
                difference_limit=0,
                native_evidence_jar_sha256=row.evidence_bridge_sha256,
            )
            if (
                rejected.semantic_digest() != row.evidence_semantic_sha256
                or comparison.equal
                or comparison.differing_cell_count != row.differing_cell_count
                or comparison.non_damaging_fluid_only_cell_count
                != row.non_damaging_fluid_only_cell_count
                or comparison.other_differing_cell_count
                != row.other_differing_cell_count
                or comparison.seed_projection_equal
                != row.seed_projection_equal
                or row.unavailable_reason
                != _unavailable_reason(comparison)
            ):
                raise ValueError("unavailable Region light evidence drifted")
            snapshots.append(None)
        return tuple(snapshots)


def create_region_light_coverage_library(
    root: str | Path,
    light_paths: Sequence[str | Path],
    rejected_region_paths: Sequence[str | Path],
    region_library: RegionArtifactLibrary,
) -> RegionLightCoverageLibrary:
    """Publish exact native light plus explicit semantic unavailability."""

    if isinstance(light_paths, (str, Path)) or isinstance(
        rejected_region_paths,
        (str, Path),
    ):
        raise TypeError("coverage evidence paths must be sequences")
    if not isinstance(region_library, RegionArtifactLibrary):
        raise TypeError("region_library must be a RegionArtifactLibrary")
    destination = Path(root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    available = {}
    for raw_path in light_paths:
        path = Path(raw_path).resolve()
        snapshot = NativeRegionLightSnapshot.load(path)
        if snapshot.source_region_semantic_sha256 in available:
            raise ValueError("native-light source Region is duplicated")
        available[snapshot.source_region_semantic_sha256] = (path, snapshot)
    rejected = {}
    for raw_path in rejected_region_paths:
        path = Path(raw_path).resolve()
        snapshot = NativeRegionSnapshot.load(path)
        identity = (
            int(snapshot.metadata["seed"]),
            tuple(int(value) for value in snapshot.core_min_chunk_xz),
        )
        if identity in rejected:
            raise ValueError("rejected Region identity is duplicated")
        rejected[identity] = (path, snapshot)
    rows = []
    entries = []
    for region_entry in region_library.entries:
        exact = available.pop(region_entry.semantic_sha256, None)
        if exact is not None:
            path, snapshot = exact
            entry = RegionLightCoverageEntry(
                region_entry.semantic_sha256,
                True,
                _relative_path(destination, path),
                _file_sha256(path),
                snapshot.semantic_digest(),
                snapshot.evidence_bridge_sha256,
                "",
                0,
                0,
                0,
                True,
            )
        else:
            identity = (region_entry.seed, region_entry.core_min_chunk_xz)
            try:
                path, actual = rejected.pop(identity)
            except KeyError as error:
                raise ValueError(
                    "every Region needs exact light or rejected evidence"
                ) from error
            expected = NativeRegionSnapshot.load(
                region_library.manifest_path.parent / region_entry.path
            )
            bridge = _sha256(
                str(actual.metadata.get("native_evidence_jar_sha256")).lower(),
                "rejected evidence bridge",
            )
            comparison = compare_region_semantics(
                expected,
                actual,
                require_same_world=False,
                difference_limit=0,
                native_evidence_jar_sha256=bridge,
            )
            if comparison.equal:
                raise ValueError(
                    "exact regenerated Region must carry native-light evidence"
                )
            entry = RegionLightCoverageEntry(
                region_entry.semantic_sha256,
                False,
                _relative_path(destination, path),
                _file_sha256(path),
                actual.semantic_digest(),
                bridge,
                _unavailable_reason(comparison),
                comparison.differing_cell_count,
                comparison.non_damaging_fluid_only_cell_count,
                comparison.other_differing_cell_count,
                comparison.seed_projection_equal,
            )
        entries.append(entry)
        rows.append(_row(entry))
    if available or rejected:
        raise ValueError("unmatched light coverage evidence remains")
    source = region_library.semantic_sha256
    manifest = {
        "schema": REGION_LIGHT_COVERAGE_SCHEMA,
        "version": REGION_LIGHT_COVERAGE_VERSION,
        "source_region_library_semantic_sha256": source,
        "contract_sha256": region_light_coverage_contract_sha256(),
        "artifact_count": len(entries),
        "available_count": sum(row.available for row in entries),
        "entries": rows,
        "library_semantic_sha256": _semantic_sha256(source, entries),
    }
    manifest_path = destination / "coverage-manifest.json"
    _write_json_atomic(manifest_path, manifest)
    return RegionLightCoverageLibrary.load(manifest_path, region_library)


def region_light_coverage_contract() -> dict[str, object]:
    return {
        "schema": REGION_LIGHT_COVERAGE_SCHEMA,
        "version": REGION_LIGHT_COVERAGE_VERSION,
        "pairing": "one_availability_row_per_frozen_region_in_source_order",
        "available": "exact_native_static_light_sidecar",
        "unavailable": {
            "fluid_only": REGION_LIGHT_UNAVAILABLE_FLUID_DRIFT,
            "other_semantic_mismatch": REGION_LIGHT_UNAVAILABLE_SEMANTIC_DRIFT,
        },
        "fallback": "surrogate_row_with_explicit_provenance",
        "native_mutation_policy": "invalidate_after_any_geometry_override",
        "provenance_only": [
            "path",
            "file_sha256",
            "evidence_bridge_sha256",
            "rejected_region_semantic_sha256",
            "differing_cell_count",
            "non_damaging_fluid_only_cell_count",
            "other_differing_cell_count",
            "seed_projection_equal",
        ],
    }


def region_light_coverage_contract_sha256() -> str:
    return hashlib.sha256(
        json.dumps(
            region_light_coverage_contract(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _entry(value: object) -> RegionLightCoverageEntry:
    row = _object(value, "coverage entry")
    _exact_fields(row, _ENTRY_FIELDS, "light coverage entry")
    result = RegionLightCoverageEntry(
        _sha256(row.get("source_region_semantic_sha256"), "source Region"),
        _boolean(row.get("available"), "available"),
        _relative_manifest_path(row.get("path")),
        _sha256(row.get("file_sha256"), "file SHA-256"),
        _sha256(row.get("evidence_semantic_sha256"), "evidence semantic"),
        _sha256(row.get("evidence_bridge_sha256"), "evidence bridge"),
        _string(row.get("unavailable_reason"), "unavailable reason"),
        _nonnegative_int(
            row.get("differing_cell_count"),
            "differing cell count",
        ),
        _nonnegative_int(
            row.get("non_damaging_fluid_only_cell_count"),
            "fluid-only cell count",
        ),
        _nonnegative_int(
            row.get("other_differing_cell_count"),
            "other differing cell count",
        ),
        _boolean(row.get("seed_projection_equal"), "seed projection equal"),
    )
    if result.available:
        if (
            result.unavailable_reason
            or result.differing_cell_count
            or result.non_damaging_fluid_only_cell_count
            or result.other_differing_cell_count
            or not result.seed_projection_equal
        ):
            raise ValueError("available Region light carries rejection state")
    elif (
        result.unavailable_reason
        not in {
            REGION_LIGHT_UNAVAILABLE_FLUID_DRIFT,
            REGION_LIGHT_UNAVAILABLE_SEMANTIC_DRIFT,
        }
        or result.differing_cell_count <= 0
        or result.non_damaging_fluid_only_cell_count
        + result.other_differing_cell_count
        != result.differing_cell_count
        or (
            result.unavailable_reason == REGION_LIGHT_UNAVAILABLE_FLUID_DRIFT
        )
        != (
            result.other_differing_cell_count == 0
            and result.seed_projection_equal
        )
    ):
        raise ValueError("unavailable Region light lacks semantic-drift evidence")
    return result


def _row(entry: RegionLightCoverageEntry) -> dict[str, object]:
    return {
        field: getattr(entry, field)
        for field in RegionLightCoverageEntry.__dataclass_fields__
    }


def _semantic_sha256(
    source: str,
    entries: Sequence[RegionLightCoverageEntry],
) -> str:
    value = {
        "schema": REGION_LIGHT_COVERAGE_SCHEMA,
        "version": REGION_LIGHT_COVERAGE_VERSION,
        "source_region_library_semantic_sha256": source,
        "contract_sha256": region_light_coverage_contract_sha256(),
        "entries": [
            {
                "source_region_semantic_sha256": row.source_region_semantic_sha256,
                "available": row.available,
                "evidence_semantic_sha256": (
                    row.evidence_semantic_sha256 if row.available else None
                ),
                "unavailable_reason": row.unavailable_reason,
            }
            for row in entries
        ],
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _unavailable_reason(comparison: RegionSemanticComparison) -> str:
    if (
        comparison.only_non_damaging_fluid_differences
        and comparison.seed_projection_equal
    ):
        return REGION_LIGHT_UNAVAILABLE_FLUID_DRIFT
    return REGION_LIGHT_UNAVAILABLE_SEMANTIC_DRIFT


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


def _artifact_path(root: Path, relative: str) -> Path:
    result = (root / Path(PurePosixPath(relative))).resolve()
    if not result.is_relative_to(root):
        raise ValueError("light coverage artifact escapes its library")
    return result


def _relative_path(root: Path, path: Path) -> str:
    try:
        return _relative_manifest_path(path.relative_to(root).as_posix())
    except ValueError as error:
        raise ValueError("light coverage evidence must be inside its root") from error


def _relative_manifest_path(value: object) -> str:
    result = _string(value, "artifact path")
    if not result:
        raise ValueError("artifact path must be nonempty")
    path = PurePosixPath(result)
    if path.is_absolute() or ".." in path.parts or "\\" in result:
        raise ValueError("artifact path must be safe relative POSIX")
    return str(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase 64-hex")
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be Boolean")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


def _nonnegative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{label} must be a non-negative integer")
    return value


def _exact_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


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
    "REGION_LIGHT_COVERAGE_SCHEMA",
    "REGION_LIGHT_COVERAGE_VERSION",
    "REGION_LIGHT_UNAVAILABLE_FLUID_DRIFT",
    "REGION_LIGHT_UNAVAILABLE_SEMANTIC_DRIFT",
    "RegionLightCoverageEntry",
    "RegionLightCoverageLibrary",
    "create_region_light_coverage_library",
    "region_light_coverage_contract",
    "region_light_coverage_contract_sha256",
]
