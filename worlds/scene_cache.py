"""On-disk cache keyed by everything that decides a scene's contents.

Building a working set of saved worlds costs minutes and produces the same
bytes for the same inputs, but the in-memory caches in this package die with
the process -- and the training worker restarts whenever guarded source moves,
so in practice every run paid the cost again.

An entry is addressed by a digest over *all* of its inputs, seeds included, so
two builds share an entry only when they would have produced the same scene.
The seeds also appear in the directory name, so a cache directory can be read
by a person without opening a manifest:

    <root>/<library-id>/k<selection-key>-n<node-seed>-w<worlds>-<digest>/
        manifest.json   every input verbatim, plus the digest they produce
        payload.npz     the cached arrays

The digest covers the library's own semantic hash, so regenerating a world
library invalidates its entries rather than serving them against new terrain.
It also covers `FORMAT_VERSION`: changing what a payload holds must not read
an old payload back as if the format had not moved.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


#: Bump when the meaning or membership of a payload changes. Entries written by
#: an older format are then unreachable rather than silently misread.
FORMAT_VERSION = 1

_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = _ROOT / "artifacts" / "worlds" / "scene-cache"
#: Length of the digest kept in a directory name. The full digest is recorded
#: in the manifest; this is a prefix for humans, not the identity.
_DIGEST_IN_NAME = 16


@dataclass(frozen=True)
class SceneKey:
    """One addressable scene build."""

    library_id: str
    selection_key: int
    node_seed: int
    world_count: int
    digest: str
    inputs: Mapping[str, Any]

    @property
    def directory(self) -> Path:
        name = (
            f"k{self.selection_key}-n{self.node_seed}"
            f"-w{self.world_count}-{self.digest[:_DIGEST_IN_NAME]}"
        )
        return CACHE_ROOT / _slug(self.library_id) / name

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "hytalerl_scene_cache_entry_v1",
            "format_version": FORMAT_VERSION,
            "digest": self.digest,
            "library_id": self.library_id,
            "selection_key": self.selection_key,
            "node_seed": self.node_seed,
            "world_count": self.world_count,
            "inputs": _canonical(self.inputs),
        }


def _slug(value: str) -> str:
    """A directory name that cannot escape the cache root."""

    safe = "".join(character if character.isalnum() else "-" for character in value)
    return safe.strip("-") or "unknown"


def _canonical(value: Any) -> Any:
    """Reduce inputs to plain JSON with a stable ordering.

    Tuples and lists both become lists, so a caller that passes one where it
    used to pass the other still addresses the same entry. Sets are sorted:
    their iteration order is not part of what they mean.
    """

    if isinstance(value, Mapping):
        return {str(name): _canonical(value[name]) for name in sorted(value)}
    if isinstance(value, (set, frozenset)):
        return sorted(_canonical(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"scene cache inputs must be JSON-shaped, got {type(value)!r}")


def key_for(
    *,
    library_id: str,
    library_semantic_sha256: str,
    selection_key: int,
    node_seed: int,
    world_count: int,
    **inputs: Any,
) -> SceneKey:
    """Address the entry for one scene build.

    Every argument participates in the digest. A caller that adds an input
    without passing it here would make two different scenes share an entry, so
    pass the whole call, not the parts that look interesting.
    """

    payload = _canonical(
        {
            "format_version": FORMAT_VERSION,
            "library_id": library_id,
            "library_semantic_sha256": library_semantic_sha256,
            "selection_key": int(selection_key),
            "node_seed": int(node_seed),
            "world_count": int(world_count),
            **inputs,
        }
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return SceneKey(
        library_id=library_id,
        selection_key=int(selection_key),
        node_seed=int(node_seed),
        world_count=int(world_count),
        digest=digest,
        inputs=payload,
    )


def load(key: SceneKey) -> dict[str, np.ndarray] | None:
    """Return the cached arrays, or None when this scene is not cached.

    A miss is never an error: a corrupt, truncated or half-written entry reads
    as absent so a run rebuilds instead of failing on a bad cache.
    """

    directory = key.directory
    manifest_path = directory / "manifest.json"
    payload_path = directory / "payload.npz"
    if not manifest_path.is_file() or not payload_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # The directory name holds a digest prefix only, so confirm identity
        # against the manifest before trusting the payload beside it.
        if manifest.get("digest") != key.digest:
            return None
        if int(manifest.get("format_version", -1)) != FORMAT_VERSION:
            return None
        with np.load(payload_path, allow_pickle=False) as stored:
            return {name: stored[name] for name in stored.files}
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def store(key: SceneKey, arrays: Mapping[str, np.ndarray]) -> Path:
    """Write one entry, atomically, and return its directory.

    The payload is written to a temporary directory and moved into place, so a
    crash or a second process mid-write cannot leave a partial entry that a
    later run would read as complete.
    """

    directory = key.directory
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=directory.parent, prefix=".staging-"))
    try:
        np.savez(staging / "payload.npz", **dict(arrays))
        (staging / "manifest.json").write_text(
            json.dumps(key.manifest(), indent=2, sort_keys=True), encoding="utf-8"
        )
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
        os.replace(staging, directory)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return directory


def entries() -> list[dict[str, Any]]:
    """Every readable entry, for inspection and for pruning."""

    found: list[dict[str, Any]] = []
    if not CACHE_ROOT.is_dir():
        return found
    for manifest_path in sorted(CACHE_ROOT.glob("*/*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        payload = manifest_path.parent / "payload.npz"
        manifest["directory"] = str(manifest_path.parent)
        manifest["bytes"] = payload.stat().st_size if payload.is_file() else 0
        found.append(manifest)
    return found


__all__ = [
    "CACHE_ROOT",
    "FORMAT_VERSION",
    "SceneKey",
    "entries",
    "key_for",
    "load",
    "store",
]
