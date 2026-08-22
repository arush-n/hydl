"""Capture one exact native-generated Hytale geometry snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

from hytalegym.envs.hytale_env import HytaleEnv
from hytalegym.worldgen import NativeWorldgenSnapshot
from hytalegym.worldgen.native_lease import acquire_native_evidence_lease


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--seed", type=int, default=570057)
    parser.add_argument("--role", default="Kweebec_Razorleaf")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/worldgen/native_0.5.7_seed_570057.npz"),
    )
    args = parser.parse_args()

    with acquire_native_evidence_lease(
        host=args.host,
        port=args.port,
        purpose="capture_native_worldgen",
    ):
        _run(args)


def _run(args: argparse.Namespace) -> None:
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
    )
    try:
        observation, info = env.reset(seed=args.seed)
        snapshot = NativeWorldgenSnapshot.from_native(
            observation,
            info,
            seed=args.seed,
        )
        destination = snapshot.save(args.output)
    finally:
        env.close()

    print("snapshot:", destination.resolve())
    print("semantic_sha256:", snapshot.semantic_digest())
    print("metadata:", dict(snapshot.metadata))


if __name__ == "__main__":
    main()
