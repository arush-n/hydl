"""Capture small exact WorldGen V2 pilots through the isolated HytaleRL bridge."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from _workspace import use_workspace_hytalerl


use_workspace_hytalerl()

from hytalegym.envs.hytale_env import HytaleEnv  # noqa: E402
from hytalegym.worldgen import NativeWorldgenSnapshot  # noqa: E402
from hytalegym.worldgen.native_lease import (  # noqa: E402
    acquire_native_evidence_lease,
)


PRIMARY_STRUCTURES = (
    "Default",
    "Zone1_Plains1",
    "Zone2_Desert1",
    "Zone3_Taiga1",
    "Zone4_Volcanic1",
)


def capture(
    *,
    host: str,
    port: int,
    seed: int,
    structure: str,
    expected_bridge_sha256: str,
    destination: Path,
) -> dict[str, object]:
    env = HytaleEnv(
        task="native_fidelity",
        backend="native",
        world="hytale_generator",
        worldgen_structure=structure,
        host=host,
        port=port,
        ticks_per_step=1,
        max_episode_steps=100,
        npc_role="Trork_Unarmed",
    )
    try:
        started = time.perf_counter()
        observation, info = env.reset(seed=seed)
        reset_seconds = time.perf_counter() - started

        assert info["world_template"] == "hytale_generator", info
        assert info["worldgen_provider"] == "HytaleGenerator", info
        assert info["worldgen_structure"] == structure, info
        assert info["worldgen_seed"] == seed, info
        assert str(info["bridge_sha256"]).lower() == (
            expected_bridge_sha256.lower()
        ), info
        assert info["fidelity_fixture"] == (
            "native_hytale_generator_worldgen"
        ), info
        assert info["worldgen_version"] not in (
            "",
            "unknown",
            "uninitialized",
        ), info
        assert info["block_below_id"] != 0, info
        assert info["on_ground"] is True, info

        nearby_blocks = np.asarray(observation["nearby_blocks"])
        assert np.count_nonzero(nearby_blocks) > 0
        snapshot = NativeWorldgenSnapshot.from_native(
            observation,
            info,
            seed=seed,
        )
        snapshot_path = snapshot.save(destination)
        restored = NativeWorldgenSnapshot.load(snapshot_path)
        assert restored.semantic_digest() == snapshot.semantic_digest()

        return {
            "structure": structure,
            "seed": seed,
            "snapshot_file": snapshot_path.name,
            "snapshot_file_sha256": _sha256(snapshot_path),
            "snapshot_semantic_sha256": snapshot.semantic_digest(),
            "local_frame_sha256": _local_frame_sha256(
                observation,
                nearby_blocks,
            ),
            "spawn": np.asarray(observation["position"], dtype=np.float64)
            .tolist(),
            "nonzero_local_blocks": int(np.count_nonzero(nearby_blocks)),
            "unique_local_block_ids": int(np.unique(nearby_blocks).size),
            "reset_seconds": reset_seconds,
            "worldgen_provider": info["worldgen_provider"],
            "worldgen_version": info["worldgen_version"],
            "native_server_version": info["native_server_version"],
            "bridge_sha256": info["bridge_sha256"],
        }
    finally:
        env.close()


def run(args: argparse.Namespace) -> Path:
    provenance = {
        name: _provenance(path)
        for name, path in (
            ("bridge_jar", args.bridge_jar),
            ("server_jar", args.server_jar),
            ("assets", args.assets),
        )
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    expected_bridge_sha256 = str(provenance["bridge_jar"]["sha256"])

    captures = []
    for index, structure in enumerate(args.structures):
        captures.append(
            capture(
                host=args.host,
                port=args.port,
                seed=args.seed,
                structure=structure,
                expected_bridge_sha256=expected_bridge_sha256,
                destination=output / f"{index:02d}-{structure}.npz",
            )
        )
    replay = capture(
        host=args.host,
        port=args.port,
        seed=args.seed,
        structure=args.structures[0],
        expected_bridge_sha256=expected_bridge_sha256,
        destination=output / f"replay-{args.structures[0]}.npz",
    )

    first = captures[0]
    assert replay["snapshot_semantic_sha256"] == first[
        "snapshot_semantic_sha256"
    ], "same-seed V2 replay changed"
    assert replay["local_frame_sha256"] == first["local_frame_sha256"], (
        "same-seed V2 local frame changed"
    )
    local_frames = {capture["local_frame_sha256"] for capture in captures}
    assert len(local_frames) >= 2, (
        "selected V2 world structures produced one identical local frame"
    )

    report = {
        "schema": "hytalerl_worldgen_v2_native_pilot_v1",
        "version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "host": args.host,
        "port": args.port,
        "seed": args.seed,
        "provenance": provenance,
        "captures": captures,
        "replay": replay,
        "validated": {
            "runtime_bridge_identity": True,
            "provider_identity": True,
            "world_structure_identity": True,
            "seed_identity": True,
            "safe_generated_spawn": True,
            "snapshot_round_trip": True,
            "same_seed_replay": True,
            "distinct_structure_local_frames": len(local_frames),
        },
    }
    destination = output / "report.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _local_frame_sha256(
    observation: dict[str, object],
    nearby_blocks: np.ndarray,
) -> str:
    digest = hashlib.sha256()
    for value in (
        np.asarray(observation["position"], dtype=np.float64),
        np.asarray(nearby_blocks),
        np.asarray(observation["geometry"]["flags"]),
        np.asarray(observation["geometry"]["support"]),
    ):
        contiguous = np.ascontiguousarray(value)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(json.dumps(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _provenance(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"provenance file not found: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--seed", type=int, default=570057)
    parser.add_argument(
        "--structures",
        nargs="+",
        default=list(PRIMARY_STRUCTURES),
    )
    parser.add_argument("--bridge-jar", type=Path, required=True)
    parser.add_argument("--server-jar", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.structures:
        parser.error("--structures must contain at least one world structure")

    with acquire_native_evidence_lease(
        host=args.host,
        port=args.port,
        purpose="worldgen_v2_native_pilot",
        stage=args.output,
        evidence_kinds=("worldgen_v2", "native_geometry"),
    ):
        report = run(args)
    print("WORLDGEN_V2_NATIVE_PILOT_PASS")
    print(report)


if __name__ == "__main__":
    main()
