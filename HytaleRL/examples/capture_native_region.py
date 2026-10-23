"""Capture and same-world-confirm Region v1 from an already-running Hytale."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path

from hytalegym.envs.hytale_env import HytaleEnv
from hytalegym.worldgen.native_lease import acquire_native_evidence_lease
from hytalegym.worldgen.region import (
    NativeRegionSnapshot,
    capture_native_region,
    current_native_evidence_jar_sha256,
)

BLOCK_TOOL_NAME = "hytalerl_capture_block_semantics"


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
    parser.add_argument(
        "--block-semantics-output",
        type=Path,
        help=(
            "also capture the block-semantic sidecar on this env, right after the physical NPZ "
            "is saved; omit for a physical-only capture"
        ),
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
    block = None
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
        if args.block_semantics_output is not None:
            # The block pass reads the NPZ just written, not the in-memory capture.
            physical = NativeRegionSnapshot.load(destination)
            block = _block_tool().capture_sidecar(
                env,
                physical,
                seed=args.seed,
                region_npz=destination,
                evidence_bridge_sha256=evidence_hash,
                output=args.block_semantics_output,
                progress=_block_progress,
            )
    finally:
        env.close()

    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    print("region:", destination.resolve())
    print("file_sha256:", digest)
    print("semantic_sha256:", snapshot.metadata["region_semantic_sha256"])
    print("cell_palette_size:", snapshot.cell_palette.size)
    print("shape_palette_size:", snapshot.shape_palette.size)
    print("native_evidence_jar_sha256:", evidence_hash)
    if block is not None:
        print(
            "block_semantics:",
            block["output"],
            "semantic_sha256:",
            block["semantic_sha256"],
            "reload_matches:",
            block["reload_matches"],
        )
        if not block["reload_matches"]:
            raise SystemExit("block-semantic sidecar reload differs from the captured digest")


def _progress(done: int, total: int, _task: object) -> None:
    if done == 1 or done == total or done % 25 == 0:
        print(f"captured {done}/{total} sections")


def _block_progress(done: int, total: int, _task: object) -> None:
    if done == 1 or done == total or done % 50 == 0:
        print(f"block_semantics captured {done}/{total} sections", flush=True)


def _block_tool():
    """The block pass is shared with tools/worldgen/capture_block_semantics.py; load that file once by path."""

    if BLOCK_TOOL_NAME not in sys.modules:
        path = (
            Path(__file__).resolve().parents[1]
            / "hytalegym"
            / "tools"
            / "worldgen"
            / "capture_block_semantics.py"
        )
        spec = importlib.util.spec_from_file_location(BLOCK_TOOL_NAME, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[BLOCK_TOOL_NAME] = module
        spec.loader.exec_module(module)
    return sys.modules[BLOCK_TOOL_NAME]


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
