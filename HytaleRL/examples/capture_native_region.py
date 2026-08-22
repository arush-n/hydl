"""Capture and same-world-confirm Region v1 from an already-running Hytale."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from hytalegym.envs.hytale_env import HytaleEnv
from hytalegym.worldgen.native_lease import acquire_native_evidence_lease
from hytalegym.worldgen.region import (
    capture_native_region,
    current_native_evidence_jar_sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--seed", type=int, default=570057)
    parser.add_argument("--role", default="Kweebec_Razorleaf")
    parser.add_argument("--core-min-chunk-x", type=int)
    parser.add_argument("--core-min-chunk-z", type=int)
    evidence = parser.add_mutually_exclusive_group()
    evidence.add_argument("--native-evidence-jar", type=Path)
    evidence.add_argument("--native-evidence-jar-sha256")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/worldgen/region_0.5.7_seed_570057.npz"),
    )
    args = parser.parse_args()

    with acquire_native_evidence_lease(
        host=args.host,
        port=args.port,
        purpose="capture_native_region",
    ):
        _run(parser, args)


def _run(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> None:
    evidence_hash = _native_evidence_hash(parser, args)

    env = HytaleEnv(
        task="kill_trork",
        backend="native",
        world="hytale",
        host=args.host,
        port=args.port,
        ticks_per_step=1,
        max_episode_steps=100,
        npc_role=args.role,
        combat_target_active=False,
        fidelity_fixture="static_region",
    )
    try:
        env.reset(seed=args.seed)
        snapshot = capture_native_region(
            env,
            core_min_chunk_x=args.core_min_chunk_x,
            core_min_chunk_z=args.core_min_chunk_z,
            native_evidence_jar_sha256=evidence_hash,
            progress=_progress,
        )
        destination = snapshot.save(args.output)
    finally:
        env.close()

    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    print("region:", destination.resolve())
    print("file_sha256:", digest)
    print("semantic_sha256:", snapshot.metadata["region_semantic_sha256"])
    print("cell_palette_size:", snapshot.cell_palette.size)
    print("shape_palette_size:", snapshot.shape_palette.size)
    print("native_evidence_jar_sha256:", evidence_hash)


def _progress(done: int, total: int, _task: object) -> None:
    if done == 1 or done == total or done % 25 == 0:
        print(f"captured {done}/{total} sections")


def _native_evidence_hash(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> str:
    expected = current_native_evidence_jar_sha256()
    if args.native_evidence_jar is not None:
        actual = hashlib.sha256(
            args.native_evidence_jar.read_bytes()
        ).hexdigest().upper()
    elif args.native_evidence_jar_sha256 is not None:
        actual = args.native_evidence_jar_sha256.upper()
    else:
        parser.error(
            "pass --native-evidence-jar for a local deployment or "
            "--native-evidence-jar-sha256 for a remote deployment"
        )
    if actual != expected:
        parser.error(
            "native evidence bridge differs from the current canonical pin: "
            f"expected {expected}, got {actual}"
        )
    return actual


if __name__ == "__main__":
    main()
