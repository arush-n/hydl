"""Content-addressed native replay projection and split cache."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from time import perf_counter
from typing import Any

import numpy as np

from arena.training.imitation.native_replay import (
    NATIVE_REPLAY_VISUAL_SIZE,
    NativeReplayCorpus,
    NativeReplayTensors,
    load_native_replay_corpus,
    native_replay_action_head_coverage,
    native_replay_tensor_sha256,
    split_native_replay_corpus,
)


NATIVE_REPLAY_CACHE_SCHEMA = "arena_native_replay_compilation_v1"
_SPLIT_METHOD = "latest_seed_per_directed_role_world_group"


@dataclass(frozen=True, slots=True)
class CompiledNativeReplay:
    """Verified projected train/heldout arrays ready for JAX materialization."""

    training: NativeReplayCorpus
    heldout: NativeReplayCorpus
    manifest: dict[str, Any]
    path: Path
    cache_hit: bool
    load_seconds: float

    def report(self) -> dict[str, Any]:
        return {
            "schema": NATIVE_REPLAY_CACHE_SCHEMA,
            "cache_hit": self.cache_hit,
            "path": self.path.as_posix(),
            "cache_key": self.manifest["cache_key"],
            "source_file_sha256": self.manifest["source"]["file_sha256"],
            "source_tensor_sha256": self.manifest["source"]["tensor_sha256"],
            "projected_tensor_sha256": self.manifest["projected"]["tensor_sha256"],
            "projection_contract_sha256": self.manifest["projection"][
                "contract_sha256"
            ],
            "observation_view": self.manifest["projection"]["view"],
            "compile_seconds": self.manifest["timings"]["compile_seconds"],
            "load_seconds": self.load_seconds,
        }


def compile_or_load_native_replay(
    source: str | Path,
    cache_root: str | Path,
    observation_view: str,
) -> CompiledNativeReplay:
    """Compile once, then mmap exact projected splits without padded source copies."""

    started = perf_counter()
    source = Path(source).resolve()
    sidecar_path = source.with_suffix(source.suffix + ".json")
    if not source.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError("native replay source and sidecar must both exist")
    sidecar = _json(sidecar_path)
    source_file_sha = _file_sha256(source)
    if sidecar.get("file_sha256") != source_file_sha:
        raise ValueError("native replay source hash differs from its sidecar")
    roles = tuple(sidecar.get("roles", ()))
    if not roles or any(not isinstance(role, str) or not role for role in roles):
        raise ValueError("native replay sidecar has no valid role vocabulary")
    projection_sha = _projection_contract_sha256(observation_view, roles, sidecar)
    key_payload = {
        "schema": NATIVE_REPLAY_CACHE_SCHEMA,
        "source_file_sha256": source_file_sha,
        "source_tensor_sha256": sidecar.get("tensor_sha256"),
        "observation_view": observation_view,
        "projection_contract_sha256": projection_sha,
    }
    cache_key = _json_sha256(key_payload)
    root = Path(cache_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / cache_key
    manifest_path = destination / "manifest.json"
    if not manifest_path.is_file():
        _compile(
            source,
            destination,
            cache_key,
            source_file_sha,
            observation_view,
            projection_sha,
        )
        cache_hit = False
    else:
        cache_hit = True
    manifest = _json(manifest_path)
    _validate_manifest(
        manifest,
        cache_key=cache_key,
        source_file_sha=source_file_sha,
        source_tensor_sha=sidecar.get("tensor_sha256"),
        observation_view=observation_view,
        projection_sha=projection_sha,
    )
    common = manifest["projected"]
    training = _load_split(destination, manifest["training"], common)
    heldout = _load_split(destination, manifest["heldout"], common)
    if _combined_action_coverage(training, heldout) != manifest[
        "action_head_coverage"
    ]:
        raise ValueError("compiled native replay action coverage drift")
    return CompiledNativeReplay(
        training=training,
        heldout=heldout,
        manifest=manifest,
        path=destination,
        cache_hit=cache_hit,
        load_seconds=perf_counter() - started,
    )


def _compile(
    source_path: Path,
    destination: Path,
    cache_key: str,
    source_file_sha: str,
    observation_view: str,
    projection_sha: str,
) -> None:
    started = perf_counter()
    source = load_native_replay_corpus(source_path)
    projected = _project(source, observation_view)
    split = split_native_replay_corpus(projected)
    readiness = _action_head_readiness(source)
    visual = _visual_context_stats(source)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{cache_key[:12]}-", dir=destination.parent)
    )
    try:
        manifest = {
            "schema": NATIVE_REPLAY_CACHE_SCHEMA,
            "cache_key": cache_key,
            "source": {
                "path": source_path.as_posix(),
                "file_sha256": source_file_sha,
                "source_report_sha256": source.source_report_sha256,
                "tensor_sha256": source.tensor_sha256,
            },
            "projection": {
                "view": observation_view,
                "contract_sha256": projection_sha,
            },
            "projected": _corpus_metadata(projected),
            "split_method": _SPLIT_METHOD,
            "training": _write_split(temporary, "training", split.training),
            "heldout": _write_split(temporary, "heldout", split.heldout),
            "action_head_coverage": native_replay_action_head_coverage(projected),
            "action_head_runtime_readiness": readiness,
            "visual_context": visual,
            "timings": {"compile_seconds": perf_counter() - started},
        }
        manifest["content_sha256"] = _json_sha256(manifest)
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if destination.exists():
            shutil.rmtree(temporary)
        else:
            temporary.rename(destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    finally:
        del split, projected, source
        gc.collect()


def _write_split(root: Path, name: str, corpus: NativeReplayCorpus) -> dict[str, Any]:
    directory = root / name
    directory.mkdir()
    files = {}
    for field, value in zip(NativeReplayTensors._fields, corpus.tensors, strict=True):
        path = directory / f"{field}.npy"
        np.save(path, np.asarray(value), allow_pickle=False)
        array = np.asarray(value)
        files[field] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": _file_sha256(path),
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }
    return {**_corpus_metadata(corpus), "files": files}


def _load_split(
    root: Path, row: dict[str, Any], common: dict[str, Any]
) -> NativeReplayCorpus:
    files = row.get("files", {})
    if set(files) != set(NativeReplayTensors._fields):
        raise ValueError("compiled native replay tensor members differ")
    values = []
    for field in NativeReplayTensors._fields:
        receipt = files[field]
        path = (root / receipt["path"]).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError("compiled native replay tensor path escaped its cache")
        if _file_sha256(path) != receipt.get("sha256"):
            raise ValueError(f"compiled native replay file hash drift: {field}")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.dtype.str != receipt.get("dtype") or list(value.shape) != receipt.get(
            "shape"
        ):
            raise ValueError(f"compiled native replay array contract drift: {field}")
        values.append(value)
    tensors = NativeReplayTensors(*values)
    if native_replay_tensor_sha256(tensors) != row.get("tensor_sha256"):
        raise ValueError("compiled native replay tensor hash drift")
    corpus = NativeReplayCorpus(
        tensors=tensors,
        roles=tuple(common["roles"]),
        feature_layout=tuple(common["feature_layout"]),
        source_report_sha256=common["source_report_sha256"],
        tensor_sha256=row["tensor_sha256"],
        episode_ids=tuple(row["episode_ids"]),
        episode_worlds=tuple(row["episode_worlds"]),
        episode_seeds=tuple(row["episode_seeds"]),
        observation_schema=common["observation_schema"],
        worlds=tuple(common["worlds"]),
        fixtures=tuple(common["fixtures"]),
        worldgen_structures=tuple(common["worldgen_structures"]),
        action_schema=common["action_schema"],
        ability_slot_layout=_ability_layout(common["ability_slot_layout"]),
        ability_slot_contract_sha256=common["ability_slot_contract_sha256"],
    )
    if (
        corpus.episodes != row["episodes"]
        or corpus.rows != row["rows"]
        or corpus.observation_size != common["observation_size"]
    ):
        raise ValueError("compiled native replay shape drift")
    return corpus


def _combined_action_coverage(
    training: NativeReplayCorpus, heldout: NativeReplayCorpus
) -> dict[str, dict[str, Any]]:
    tensors = training.tensors._replace(
        expert_action=np.concatenate(
            (training.tensors.expert_action, heldout.tensors.expert_action), axis=0
        ),
        supervision_mask=np.concatenate(
            (training.tensors.supervision_mask, heldout.tensors.supervision_mask), axis=0
        ),
        valid=np.concatenate((training.tensors.valid, heldout.tensors.valid), axis=0),
    )
    return native_replay_action_head_coverage(
        NativeReplayCorpus(
            tensors=tensors,
            roles=training.roles,
            feature_layout=training.feature_layout,
            source_report_sha256=training.source_report_sha256,
            tensor_sha256="",
            observation_schema=training.observation_schema,
            action_schema=training.action_schema,
            ability_slot_layout=training.ability_slot_layout,
            ability_slot_contract_sha256=training.ability_slot_contract_sha256,
        )
    )


def _corpus_metadata(corpus: NativeReplayCorpus) -> dict[str, Any]:
    return {
        "tensor_sha256": corpus.tensor_sha256,
        "source_report_sha256": corpus.source_report_sha256,
        "observation_schema": corpus.observation_schema,
        "observation_size": corpus.observation_size,
        "action_schema": corpus.action_schema,
        "episodes": corpus.episodes,
        "rows": corpus.rows,
        "roles": list(corpus.roles),
        "feature_layout": list(corpus.feature_layout),
        "episode_ids": list(corpus.episode_ids),
        "episode_worlds": list(corpus.episode_worlds),
        "episode_seeds": list(corpus.episode_seeds),
        "worlds": list(corpus.worlds),
        "fixtures": list(corpus.fixtures),
        "worldgen_structures": list(corpus.worldgen_structures),
        "ability_slot_layout": [
            [list(slot) for slot in role] for role in corpus.ability_slot_layout
        ],
        "ability_slot_contract_sha256": corpus.ability_slot_contract_sha256,
    }


def _project(corpus: NativeReplayCorpus, view: str) -> NativeReplayCorpus:
    from arena.training.imitation.native_transfer import (
        native_arsenal_conditioned_replay_corpus,
        native_arsenal_shared_replay_corpus,
        native_arsenal_visual_conditioned_replay_corpus,
        native_arsenal_visual_shared_replay_corpus,
    )

    projectors = {
        "omniscient": lambda value: value,
        "arsenal_shared": native_arsenal_shared_replay_corpus,
        "arsenal_conditioned": native_arsenal_conditioned_replay_corpus,
        "arsenal_visual_shared": native_arsenal_visual_shared_replay_corpus,
        "arsenal_visual_conditioned": native_arsenal_visual_conditioned_replay_corpus,
    }
    try:
        return projectors[view](corpus)
    except KeyError as error:
        raise ValueError(f"unsupported native replay observation view: {view}") from error


def _projection_contract_sha256(
    view: str, roles: tuple[str, ...], sidecar: dict[str, Any]
) -> str:
    from arena.training.imitation.native_transfer import (
        NATIVE_ARSENAL_SHARED_OBSERVATION_SHA256,
        NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SHA256,
        native_arsenal_conditioned_observation_sha256,
        native_arsenal_visual_conditioned_observation_sha256,
    )

    contracts = {
        "omniscient": _json_sha256(
            {
                "schema": sidecar.get("observation_schema"),
                "feature_layout": sidecar.get("feature_layout"),
            }
        ),
        "arsenal_shared": NATIVE_ARSENAL_SHARED_OBSERVATION_SHA256,
        "arsenal_conditioned": native_arsenal_conditioned_observation_sha256(roles),
        "arsenal_visual_shared": NATIVE_ARSENAL_VISUAL_SHARED_OBSERVATION_SHA256,
        "arsenal_visual_conditioned": (
            native_arsenal_visual_conditioned_observation_sha256(roles)
        ),
    }
    try:
        return contracts[view]
    except KeyError as error:
        raise ValueError(f"unsupported native replay observation view: {view}") from error


def _action_head_readiness(corpus: NativeReplayCorpus) -> dict[str, Any]:
    from arena.training.imitation.native_transfer import native_arsenal_action_head_readiness

    return native_arsenal_action_head_readiness(corpus)


def _visual_context_stats(corpus: NativeReplayCorpus) -> dict[str, Any]:
    observation = np.asarray(corpus.tensors.observation, dtype=np.float32)
    valid = np.asarray(corpus.tensors.valid, dtype=np.bool_)
    if observation.shape[-1] < NATIVE_REPLAY_VISUAL_SIZE:
        return {"enabled": False}
    visual = observation[..., -NATIVE_REPLAY_VISUAL_SIZE:]
    available = valid & (visual[..., -1] > 0.5)
    token_count = np.sum(visual[..., 8 * 44 : 9 * 44] > 0.5, axis=-1)
    frames = int(np.sum(valid))
    available_frames = int(np.sum(available))
    return {
        "enabled": True,
        "available_frames": available_frames,
        "valid_frames": frames,
        "available_fraction": available_frames / frames if frames else 0.0,
        "mean_visible_tokens": (
            float(np.mean(token_count[available])) if available_frames else 0.0
        ),
    }


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    cache_key: str,
    source_file_sha: str,
    source_tensor_sha: object,
    observation_view: str,
    projection_sha: str,
) -> None:
    recorded_content = manifest.get("content_sha256")
    content = {key: value for key, value in manifest.items() if key != "content_sha256"}
    if manifest.get("schema") != NATIVE_REPLAY_CACHE_SCHEMA:
        raise ValueError("native replay cache schema differs")
    if recorded_content != _json_sha256(content):
        raise ValueError("native replay cache manifest hash drift")
    if manifest.get("cache_key") != cache_key:
        raise ValueError("native replay cache key drift")
    if (
        manifest.get("source", {}).get("file_sha256") != source_file_sha
        or manifest.get("source", {}).get("tensor_sha256") != source_tensor_sha
    ):
        raise ValueError("native replay cache source identity drift")
    if (
        manifest.get("projection", {}).get("view") != observation_view
        or manifest.get("projection", {}).get("contract_sha256") != projection_sha
    ):
        raise ValueError("native replay cache projection contract drift")
    if manifest.get("split_method") != _SPLIT_METHOD:
        raise ValueError("native replay cache split contract drift")


def _ability_layout(
    value: list[Any],
) -> tuple[tuple[tuple[str, str, str, str], ...], ...]:
    return tuple(tuple(tuple(slot) for slot in role) for role in value)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest().upper()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


__all__ = [
    "CompiledNativeReplay",
    "NATIVE_REPLAY_CACHE_SCHEMA",
    "compile_or_load_native_replay",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument(
        "--observation-view",
        choices=(
            "omniscient",
            "arsenal_shared",
            "arsenal_conditioned",
            "arsenal_visual_shared",
            "arsenal_visual_conditioned",
        ),
        default="arsenal_visual_conditioned",
    )
    arguments = parser.parse_args()
    compiled = compile_or_load_native_replay(
        arguments.source, arguments.cache_root, arguments.observation_view
    )
    print(json.dumps(compiled.report(), sort_keys=True))


if __name__ == "__main__":
    main()
