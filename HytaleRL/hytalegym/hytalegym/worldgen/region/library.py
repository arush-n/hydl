"""Exact, bounded Region artifact catalogs for reproducible training worlds."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import operator
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Mapping, Sequence

from hytalegym.worldgen.region.contract import certified_chunk_api_manifest
from hytalegym.worldgen.region.native import REGION_STABILITY_PASSES
from hytalegym.worldgen.region.snapshot import (
    OFFLINE_COMPLETE,
    REGION_SNAPSHOT_SCHEMA,
    REGION_SNAPSHOT_VERSION,
    NativeRegionSnapshot,
)

REGION_ARTIFACT_LIBRARY_SCHEMA = "hytalerl_region_artifact_library_v1"
REGION_ARTIFACT_LIBRARY_VERSION = 1
REGION_ARTIFACT_SPLIT_METHOD = "sha256_rank_v1"
REGION_ARTIFACT_AUTHORITY = "exact_frozen_native_artifact"
REGION_ARTIFACT_RUNTIME_SOURCE = "artifact_only_no_seed_regeneration"

_SPLITS = frozenset({"train", "heldout"})
_MANIFEST_FIELDS = frozenset(
    {
        "schema",
        "version",
        "authority",
        "runtime_source",
        "artifact_count",
        "heldout_count",
        "split_contract",
        "capture_contract",
        "entries",
        "library_semantic_sha256",
    }
)
_SPLIT_FIELDS = frozenset({"method", "salt_sha256"})
_CAPTURE_FIELDS = frozenset(
    {
        "snapshot_schema",
        "snapshot_version",
        "server_version",
        "chunk_api_sha256",
        "capture_mode",
        "exact_collision_shapes",
        "dynamic_state",
        "same_world_confirmation_passes",
        "native_evidence_jar_sha256",
        "worldgen_provider",
        "worldgen_version",
        "section_protocol_schema",
        "section_protocol_version",
        "source_reach_bound_blocks",
    }
)
_ENTRY_FIELDS = frozenset(
    {
        "path",
        "file_sha256",
        "semantic_sha256",
        "split",
        "seed",
        "core_min_chunk_xz",
    }
)


@dataclass(frozen=True)
class RegionArtifactEntry:
    """One exact artifact and its immutable train/held-out assignment."""

    path: str
    file_sha256: str
    semantic_sha256: str
    split: str
    seed: int
    core_min_chunk_xz: tuple[int, int]


@dataclass(frozen=True)
class RegionArtifactLibrary:
    """Validated manifest whose artifacts are checked without retaining them."""

    manifest_path: Path
    manifest: Mapping[str, Any]
    entries: tuple[RegionArtifactEntry, ...]

    @classmethod
    def load(
        cls,
        manifest_path: str | Path,
        *,
        artifact_capacity: int | None = None,
    ) -> "RegionArtifactLibrary":
        source = Path(manifest_path)
        manifest = _load_json_object(source)
        _require_fields(manifest, _MANIFEST_FIELDS, "Region library manifest")
        if manifest["schema"] != REGION_ARTIFACT_LIBRARY_SCHEMA:
            raise ValueError("unsupported Region artifact library schema")
        if _exact_int(manifest["version"], "library version") != (
            REGION_ARTIFACT_LIBRARY_VERSION
        ):
            raise ValueError("unsupported Region artifact library version")
        if manifest["authority"] != REGION_ARTIFACT_AUTHORITY:
            raise ValueError("Region artifact library authority changed")
        if manifest["runtime_source"] != REGION_ARTIFACT_RUNTIME_SOURCE:
            raise ValueError("Region library must reject seed regeneration")

        count = _positive_int(manifest["artifact_count"], "artifact_count")
        if artifact_capacity is not None:
            capacity = _positive_int(artifact_capacity, "artifact_capacity")
            if count > capacity:
                raise ValueError("Region artifact library exceeds host capacity")
        heldout_count = _positive_int(
            manifest["heldout_count"],
            "heldout_count",
        )
        if heldout_count >= count:
            raise ValueError("Region library requires train and held-out artifacts")

        split_contract = _mapping(
            manifest["split_contract"],
            "split_contract",
        )
        _require_fields(split_contract, _SPLIT_FIELDS, "split_contract")
        if split_contract["method"] != REGION_ARTIFACT_SPLIT_METHOD:
            raise ValueError("unsupported Region artifact split method")
        salt = _sha256(split_contract["salt_sha256"], "split salt")

        capture = _mapping(manifest["capture_contract"], "capture_contract")
        _validate_capture_contract(capture)
        raw_entries = manifest["entries"]
        if not isinstance(raw_entries, list) or len(raw_entries) != count:
            raise ValueError("Region artifact entries differ from artifact_count")
        entries = tuple(_parse_entry(value) for value in raw_entries)
        _validate_entries(entries, heldout_count, salt)
        expected_digest = _manifest_digest(manifest)
        if _sha256(
            manifest["library_semantic_sha256"],
            "library semantic SHA-256",
        ) != expected_digest:
            raise ValueError("Region artifact library semantic SHA-256 mismatch")

        result = cls(
            manifest_path=source.resolve(),
            manifest=_json_copy(manifest),
            entries=entries,
        )
        for entry in entries:
            result._load_entry(entry, capture)
        return result

    @property
    def semantic_sha256(self) -> str:
        return str(self.manifest["library_semantic_sha256"])

    def entries_for_split(self, split: str) -> tuple[RegionArtifactEntry, ...]:
        selected_split = _split(split)
        return tuple(entry for entry in self.entries if entry.split == selected_split)

    def select(
        self,
        split: str,
        count: int,
        *,
        selection_key: int,
    ) -> tuple[RegionArtifactEntry, ...]:
        """Choose a stable, order-independent fixed working set."""

        candidates = self.entries_for_split(split)
        requested = _positive_int(count, "selection count")
        if requested > len(candidates):
            raise ValueError("Region selection exceeds the requested split")
        key = _nonnegative_int(selection_key, "selection_key")
        ranked = sorted(
            candidates,
            key=lambda entry: (
                hashlib.sha256(
                    (
                        f"{REGION_ARTIFACT_SPLIT_METHOD}\0{key}\0"
                        f"{entry.semantic_sha256}"
                    ).encode()
                ).digest(),
                entry.semantic_sha256,
            ),
        )
        return tuple(ranked[:requested])

    def select_seeds(
        self,
        split: str,
        seeds: Sequence[int],
    ) -> tuple[RegionArtifactEntry, ...]:
        """Choose an explicit working set by artifact seed, in the order given.

        :meth:`select` ranks the whole split by digest and takes a prefix, so the
        key decides the entire set at once and a composed set -- "half flat, half
        varied" -- cannot be expressed by it at all. This is the seam for callers
        that have already decided *which* worlds they want and need the ordering
        preserved, because slot order is what ``environment_world_id`` indexes.

        Duplicates are refused rather than deduplicated: a repeated seed would
        silently shrink the working set below the capacity the caller asked for,
        and the resulting atlas would still claim the larger one.
        """

        candidates = {entry.seed: entry for entry in self.entries_for_split(split)}
        requested = tuple(int(seed) for seed in seeds)
        if not requested:
            raise ValueError("explicit Region selection is empty")
        duplicates = sorted({s for s in requested if requested.count(s) > 1})
        if duplicates:
            raise ValueError(f"explicit Region selection repeats seeds {duplicates}")
        missing = [seed for seed in requested if seed not in candidates]
        if missing:
            raise ValueError(
                f"explicit Region selection names seeds absent from the "
                f"{split!r} split: {missing}"
            )
        return tuple(candidates[seed] for seed in requested)

    def load_seed_selection(
        self,
        split: str,
        seeds: Sequence[int],
    ) -> tuple[NativeRegionSnapshot, ...]:
        capture = _mapping(self.manifest["capture_contract"], "capture_contract")
        return tuple(
            self._load_entry(entry, capture)
            for entry in self.select_seeds(split, seeds)
        )

    def load_selection(
        self,
        split: str,
        count: int,
        *,
        selection_key: int,
    ) -> tuple[NativeRegionSnapshot, ...]:
        capture = _mapping(self.manifest["capture_contract"], "capture_contract")
        return tuple(
            self._load_entry(entry, capture)
            for entry in self.select(
                split,
                count,
                selection_key=selection_key,
            )
        )

    def _load_entry(
        self,
        entry: RegionArtifactEntry,
        capture: Mapping[str, Any],
    ) -> NativeRegionSnapshot:
        root = self.manifest_path.parent.resolve()
        path = _artifact_path(root, entry.path)
        if _file_sha256(path) != entry.file_sha256:
            raise ValueError(f"Region artifact file SHA-256 mismatch: {entry.path}")
        snapshot = NativeRegionSnapshot.load(path)
        _validate_snapshot(snapshot, entry, capture)
        return snapshot


def create_region_artifact_library(
    root: str | Path,
    artifact_paths: Sequence[str | Path],
    *,
    heldout_count: int,
    split_salt_sha256: str,
    manifest_name: str = "manifest.json",
) -> RegionArtifactLibrary:
    """Validate exact snapshots and atomically publish one homogeneous catalog."""

    if isinstance(artifact_paths, (str, Path)):
        raise TypeError("artifact_paths must be a sequence of paths")
    destination_root = Path(root).resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    paths = tuple(Path(path).resolve() for path in artifact_paths)
    if not paths:
        raise ValueError("at least one Region artifact is required")
    heldout = _positive_int(heldout_count, "heldout_count")
    if heldout >= len(paths):
        raise ValueError("Region library requires train and held-out artifacts")
    salt = _sha256(split_salt_sha256, "split salt")
    manifest_path = _artifact_path(destination_root, manifest_name)
    if manifest_path in paths:
        raise ValueError("Region library manifest cannot replace an artifact")

    rows: list[tuple[dict[str, Any], Mapping[str, Any]]] = []
    for path in paths:
        relative = _relative_artifact_path(destination_root, path)
        snapshot = NativeRegionSnapshot.load(path)
        capture = _capture_contract(snapshot)
        rows.append(
            (
                {
                    "path": relative,
                    "file_sha256": _file_sha256(path),
                    "semantic_sha256": snapshot.semantic_artifact_digest(),
                    "split": "train",
                    "seed": _nonnegative_int(
                        snapshot.metadata.get("seed"),
                        "snapshot seed",
                    ),
                    "core_min_chunk_xz": [
                        int(snapshot.core_min_chunk_xz[0]),
                        int(snapshot.core_min_chunk_xz[1]),
                    ],
                },
                capture,
            )
        )
    rows.sort(key=lambda row: row[0]["semantic_sha256"])
    first_capture = rows[0][1]
    if any(capture != first_capture for _, capture in rows[1:]):
        raise ValueError("Region library artifacts use different capture contracts")

    heldout_semantics = {
        row["semantic_sha256"]
        for row, _ in sorted(
            rows,
            key=lambda item: (
                _split_rank(salt, item[0]["semantic_sha256"]),
                item[0]["semantic_sha256"],
            ),
        )[:heldout]
    }
    entries = []
    for row, _ in rows:
        row["split"] = (
            "heldout"
            if row["semantic_sha256"] in heldout_semantics
            else "train"
        )
        entries.append(row)
    parsed_entries = tuple(_parse_entry(entry) for entry in entries)
    _validate_entries(parsed_entries, heldout, salt)
    _validate_capture_contract(first_capture)

    manifest: dict[str, Any] = {
        "schema": REGION_ARTIFACT_LIBRARY_SCHEMA,
        "version": REGION_ARTIFACT_LIBRARY_VERSION,
        "authority": REGION_ARTIFACT_AUTHORITY,
        "runtime_source": REGION_ARTIFACT_RUNTIME_SOURCE,
        "artifact_count": len(entries),
        "heldout_count": heldout,
        "split_contract": {
            "method": REGION_ARTIFACT_SPLIT_METHOD,
            "salt_sha256": salt,
        },
        "capture_contract": dict(first_capture),
        "entries": entries,
        "library_semantic_sha256": "",
    }
    manifest["library_semantic_sha256"] = _manifest_digest(manifest)
    _atomic_json_write(manifest_path, manifest)
    return RegionArtifactLibrary.load(manifest_path)


def _capture_contract(snapshot: NativeRegionSnapshot) -> dict[str, Any]:
    metadata = snapshot.metadata
    if metadata.get("capture_mode") != OFFLINE_COMPLETE:
        raise ValueError("Region library accepts only offline-complete artifacts")
    if metadata.get("same_world_stability") != "verified":
        raise ValueError("Region library requires same-world-confirmed artifacts")
    if metadata.get("same_world_confirmation_passes") != REGION_STABILITY_PASSES:
        raise ValueError("Region artifact confirmation pass count changed")
    return {
        "snapshot_schema": REGION_SNAPSHOT_SCHEMA,
        "snapshot_version": REGION_SNAPSHOT_VERSION,
        "server_version": _nonempty(metadata.get("server_version"), "server_version"),
        "chunk_api_sha256": _canonical_sha256(metadata.get("chunk_api")),
        "capture_mode": OFFLINE_COMPLETE,
        "exact_collision_shapes": True,
        "dynamic_state": "static",
        "same_world_confirmation_passes": REGION_STABILITY_PASSES,
        "native_evidence_jar_sha256": _sha256(
            metadata.get("native_evidence_jar_sha256"),
            "native evidence bridge",
        ).upper(),
        "worldgen_provider": _nonempty(
            metadata.get("worldgen_provider"),
            "worldgen_provider",
        ),
        "worldgen_version": _nonempty(
            metadata.get("worldgen_version"),
            "worldgen_version",
        ),
        "section_protocol_schema": _nonempty(
            metadata.get("section_protocol_schema"),
            "section_protocol_schema",
        ),
        "section_protocol_version": _positive_int(
            metadata.get("section_protocol_version"),
            "section_protocol_version",
        ),
        "source_reach_bound_blocks": _finite_nonnegative(
            metadata.get("source_reach_bound_blocks"),
            "source_reach_bound_blocks",
        ),
    }


def _validate_capture_contract(value: Mapping[str, Any]) -> None:
    _require_fields(value, _CAPTURE_FIELDS, "capture_contract")
    if value["snapshot_schema"] != REGION_SNAPSHOT_SCHEMA:
        raise ValueError("Region library snapshot schema changed")
    if _exact_int(value["snapshot_version"], "snapshot_version") != (
        REGION_SNAPSHOT_VERSION
    ):
        raise ValueError("Region library snapshot version changed")
    if value["capture_mode"] != OFFLINE_COMPLETE:
        raise ValueError("Region library capture mode changed")
    if value["exact_collision_shapes"] is not True:
        raise ValueError("Region library requires exact collision shapes")
    if value["dynamic_state"] != "static":
        raise ValueError("Region library requires static artifacts")
    if _exact_int(
        value["same_world_confirmation_passes"],
        "same_world_confirmation_passes",
    ) != REGION_STABILITY_PASSES:
        raise ValueError("Region library confirmation pass count changed")
    if value["chunk_api_sha256"] != _canonical_sha256(
        certified_chunk_api_manifest()
    ):
        raise ValueError("Region library chunk API fingerprint changed")
    _nonempty(value["server_version"], "server_version")
    _sha256(value["native_evidence_jar_sha256"], "native evidence bridge")
    _nonempty(value["worldgen_provider"], "worldgen_provider")
    _nonempty(value["worldgen_version"], "worldgen_version")
    _nonempty(value["section_protocol_schema"], "section_protocol_schema")
    _positive_int(value["section_protocol_version"], "section_protocol_version")
    _finite_nonnegative(
        value["source_reach_bound_blocks"],
        "source_reach_bound_blocks",
    )


def _validate_snapshot(
    snapshot: NativeRegionSnapshot,
    entry: RegionArtifactEntry,
    capture: Mapping[str, Any],
) -> None:
    if _capture_contract(snapshot) != capture:
        raise ValueError(f"Region artifact capture contract changed: {entry.path}")
    if snapshot.semantic_artifact_digest() != entry.semantic_sha256:
        raise ValueError(f"Region artifact semantic SHA-256 mismatch: {entry.path}")
    if snapshot.metadata.get("seed") != entry.seed:
        raise ValueError(f"Region artifact seed changed: {entry.path}")
    actual_core = tuple(int(value) for value in snapshot.core_min_chunk_xz)
    if actual_core != entry.core_min_chunk_xz:
        raise ValueError(f"Region artifact core changed: {entry.path}")


def _parse_entry(value: Any) -> RegionArtifactEntry:
    source = _mapping(value, "Region artifact entry")
    _require_fields(source, _ENTRY_FIELDS, "Region artifact entry")
    core = source["core_min_chunk_xz"]
    if not isinstance(core, list) or len(core) != 2:
        raise ValueError("Region artifact core_min_chunk_xz must contain two values")
    return RegionArtifactEntry(
        path=_relative_path(source["path"]),
        file_sha256=_sha256(source["file_sha256"], "artifact file SHA-256"),
        semantic_sha256=_sha256(
            source["semantic_sha256"],
            "artifact semantic SHA-256",
        ),
        split=_split(source["split"]),
        seed=_nonnegative_int(source["seed"], "artifact seed"),
        core_min_chunk_xz=(
            _exact_int(core[0], "artifact core X"),
            _exact_int(core[1], "artifact core Z"),
        ),
    )


def _validate_entries(
    entries: tuple[RegionArtifactEntry, ...],
    heldout_count: int,
    salt: str,
) -> None:
    semantics = [entry.semantic_sha256 for entry in entries]
    if semantics != sorted(semantics):
        raise ValueError("Region artifact entries must use semantic-hash order")
    if len(set(semantics)) != len(entries):
        raise ValueError("Region artifact semantic hashes must be unique")
    paths = [entry.path for entry in entries]
    if len(set(paths)) != len(entries):
        raise ValueError("Region artifact paths must be unique")
    identities = [(entry.seed, entry.core_min_chunk_xz) for entry in entries]
    if len(set(identities)) != len(entries):
        raise ValueError("Region artifact seed/core identities must be unique")
    expected_heldout = set(
        sorted(
            semantics,
            key=lambda semantic: (
                _split_rank(salt, semantic),
                semantic,
            ),
        )[:heldout_count]
    )
    actual_heldout = {
        entry.semantic_sha256
        for entry in entries
        if entry.split == "heldout"
    }
    if actual_heldout != expected_heldout:
        raise ValueError("Region artifact split assignment changed")


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("library_semantic_sha256", None)
    return _canonical_sha256(payload)


def _split_rank(salt: str, semantic_sha256: str) -> bytes:
    return hashlib.sha256(
        f"{REGION_ARTIFACT_SPLIT_METHOD}\0{salt}\0{semantic_sha256}".encode()
    ).digest()


def _artifact_path(root: Path, relative: str) -> Path:
    path = (root / _relative_path(relative)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Region artifact path escapes the library root")
    return path


def _relative_artifact_path(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError("Region artifacts must live under the library root") from error
    return _relative_path(relative.as_posix())


def _relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Region artifact path must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("Region artifact path must stay under the library root")
    return path.as_posix()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(value, output, sort_keys=True, separators=(",", ":"))
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_json_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read Region library manifest: {path}") from error
    if not isinstance(value, dict):
        raise TypeError("Region library manifest must be an object")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _require_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    label: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} fields differ: "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _split(value: Any) -> str:
    if value not in _SPLITS:
        raise ValueError("Region artifact split must be train or heldout")
    return str(value)


def _exact_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{label} must be an integer") from error


def _positive_int(value: Any, label: str) -> int:
    result = _exact_int(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _nonnegative_int(value: Any, label: str) -> int:
    result = _exact_int(value, label)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be a number") from error
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return result


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{label} must be a SHA-256")
    return value.lower()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))


__all__ = [
    "REGION_ARTIFACT_AUTHORITY",
    "REGION_ARTIFACT_LIBRARY_SCHEMA",
    "REGION_ARTIFACT_LIBRARY_VERSION",
    "REGION_ARTIFACT_RUNTIME_SOURCE",
    "REGION_ARTIFACT_SPLIT_METHOD",
    "RegionArtifactEntry",
    "RegionArtifactLibrary",
    "create_region_artifact_library",
]
