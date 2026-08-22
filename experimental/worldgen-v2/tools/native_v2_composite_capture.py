"""Capture a bounded multi-Region WorldGen V2 environment from one reset.

This is the producer counterpart to :mod:`jax_port.worlds.composite_world`.  It
performs exactly one native reset, captures every adjacent Region tile while
that world remains live, binds a shared capture-group identity, captures block
and fluid identities, and publishes a manifest whose JAX materializer assigns
all tiles one world ID.

The tool never deploys or repins a jar.  The caller must provide the already
deployed evidence bridge and acquire the native port through the normal lease.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import secrets
import sys
import time
from typing import Any, Mapping, Sequence

from _workspace import use_workspace_hytalerl


use_workspace_hytalerl()
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from hytalegym.envs.hytale_env import HytaleEnv  # noqa: E402
from hytalegym.worldgen.native_lease import (  # noqa: E402
    acquire_native_evidence_lease,
    update_owned_native_evidence_lease,
)
from hytalegym.worldgen.region import (  # noqa: E402
    CHUNK_SIZE,
    CORE_BLOCKS_PER_AXIS,
    CORE_CHUNKS_PER_AXIS,
    MIN_Y,
    WORLD_HEIGHT,
    capture_native_region,
    capture_native_region_block_semantics,
)
from jax_port.worlds.composite_world import (  # noqa: E402
    V2CompositeTileSource,
    bind_composite_capture_identity,
    publish_composite_world,
)
from native_v2_region_capture import (  # noqa: E402
    WORLDGEN_PROVIDER,
    WORLD_TEMPLATE,
    _atomic_json,
    _bind_v2_snapshot_identity,
    _canonical_sha256,
    _file_sha256,
    _provenance,
    _require_live_v2_identity,
    _spawn_position,
)
from native_v2_region_fluid_sidecar import (  # noqa: E402
    capture_native_region_fluid_semantics,
)


RECEIPT_SCHEMA = "hytalerl_worldgen_v2_composite_capture_receipt_v1"
RECEIPT_VERSION = 1
MAX_WORLD_AXIS = 512


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Capture a bounded WorldGen V2 environment as adjacent Regions "
            "from exactly one native reset and publish one JAX-world manifest."
        )
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--structure", required=True)
    parser.add_argument("--environment-id", required=True)
    parser.add_argument("--world-width", type=int, required=True)
    parser.add_argument("--world-depth", type=int, required=True)
    parser.add_argument("--core-min-chunk-x", type=int)
    parser.add_argument("--core-min-chunk-z", type=int)
    parser.add_argument("--role", default="Kweebec_Razorleaf")
    parser.add_argument("--split", default="authoring")
    parser.add_argument("--bridge-jar", type=Path, required=True)
    parser.add_argument("--server-jar", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _validate_args(parser, args)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "composite-world.json"
    receipt_path = output / "capture-receipt.json"
    if manifest_path.exists() or receipt_path.exists():
        parser.error("output already contains a composite manifest or receipt")

    provenance = {
        name: _provenance(path)
        for name, path in (
            ("bridge_jar", args.bridge_jar),
            ("server_jar", args.server_jar),
            ("assets", args.assets),
        )
    }
    expected_bridge = str(provenance["bridge_jar"]["sha256"]).upper()
    with acquire_native_evidence_lease(
        host=args.host,
        port=args.port,
        purpose="worldgen_v2_composite_capture",
        stage=output,
        evidence_kinds=(
            "worldgen_v2",
            "native_region",
            "block_semantics",
            "stable_fluid_semantics",
            "same_reset_composite_world",
        ),
    ) as lease:
        receipt = capture_composite_world(
            args,
            output=output,
            provenance=provenance,
            expected_bridge=expected_bridge,
            lease=lease,
        )
    _atomic_json(receipt_path, receipt)
    print("WORLDGEN_V2_COMPOSITE_CAPTURE_PASS", flush=True)
    print(manifest_path, flush=True)
    print(receipt_path, flush=True)


def capture_composite_world(
    args: argparse.Namespace,
    *,
    output: Path,
    provenance: Mapping[str, Mapping[str, Any]],
    expected_bridge: str,
    lease: Any | None = None,
) -> dict[str, Any]:
    """Capture all tiles without calling ``reset`` more than once."""

    started = time.perf_counter()
    env = HytaleEnv(
        task="kill_trork",
        backend="native",
        world=WORLD_TEMPLATE,
        worldgen_structure=args.structure,
        host=args.host,
        port=args.port,
        ticks_per_step=1,
        max_episode_steps=100,
        npc_role=args.role,
        combat_target_active=False,
        fidelity_fixture="static_region",
    )
    try:
        # This is deliberately the only reset in this function. Every Region
        # and both semantic sidecars below are observations of this world.
        observation, live_info = env.reset(seed=args.seed)
        _require_live_v2_identity(
            live_info,
            seed=args.seed,
            structure=args.structure,
            expected_bridge=expected_bridge,
        )
        spawn = _spawn_position(observation)
        plan = composite_capture_plan(
            spawn,
            args.world_width,
            args.world_depth,
            core_min_chunk_x=args.core_min_chunk_x,
            core_min_chunk_z=args.core_min_chunk_z,
        )
        group = _capture_group_sha256(
            seed=args.seed,
            structure=args.structure,
            spawn=spawn,
            expected_bridge=expected_bridge,
        )
        if lease is not None:
            update_owned_native_evidence_lease(
                lease,
                phase="composite_capture",
                phase_status="running",
                detail={
                    "native_reset_count": 1,
                    "capture_group_sha256": group,
                    "tile_grid": list(plan["tile_grid"]),
                    "tile_count": len(plan["tiles"]),
                },
            )

        artifact_root = output / "artifacts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        sources: list[V2CompositeTileSource] = []
        tile_receipts: list[dict[str, Any]] = []
        for index, tile in enumerate(plan["tiles"], start=1):
            tile_x, tile_z = tile["tile_xz"]
            core_x, core_z = tile["core_min_chunk_xz"]
            stem = f"tile-{tile_x}-{tile_z}"
            region_path = artifact_root / f"{stem}.region.npz"
            block_path = artifact_root / f"{stem}.block-semantics.npz"
            fluid_path = artifact_root / f"{stem}.fluid-semantics.npz"
            if any(path.exists() for path in (region_path, block_path, fluid_path)):
                raise ValueError(f"composite tile output already exists: {stem}")
            if lease is not None:
                update_owned_native_evidence_lease(
                    lease,
                    phase="composite_capture",
                    phase_status="running",
                    detail={
                        "native_reset_count": 1,
                        "tile_index": index,
                        "tile_count": len(plan["tiles"]),
                        "tile_xz": [tile_x, tile_z],
                        "core_min_chunk_xz": [core_x, core_z],
                    },
                )
            physical = capture_native_region(
                env,
                core_min_chunk_x=core_x,
                core_min_chunk_z=core_z,
                native_evidence_jar_sha256=expected_bridge,
                progress=_progress(index, len(plan["tiles"]), "physical"),
            )
            physical = _bind_v2_snapshot_identity(
                physical,
                live_info,
                seed=args.seed,
                structure=args.structure,
                expected_bridge=expected_bridge,
                spawn_position=spawn,
            )
            physical = bind_composite_capture_identity(
                physical,
                capture_group_sha256=group,
                tile_xz=(tile_x, tile_z),
                tile_grid=plan["tile_grid"],
            )
            physical.save(region_path)
            block = capture_native_region_block_semantics(
                env,
                physical,
                evidence_bridge_sha256=expected_bridge,
                progress=_progress(index, len(plan["tiles"]), "blocks"),
            )
            block.save(block_path)
            fluid = capture_native_region_fluid_semantics(
                env,
                physical,
                evidence_bridge_sha256=expected_bridge,
                progress=_progress(index, len(plan["tiles"]), "fluids"),
            )
            fluid.save(fluid_path)
            sources.append(
                V2CompositeTileSource(
                    tile_xz=(tile_x, tile_z),
                    region_path=region_path,
                    block_semantic_path=block_path,
                    fluid_semantic_path=fluid_path,
                )
            )
            tile_receipts.append(
                {
                    "tile_xz": [tile_x, tile_z],
                    "core_min_chunk_xz": [core_x, core_z],
                    "region_semantic_sha256": physical.semantic_artifact_digest(),
                    "block_semantic_sha256": block.semantic_sha256(),
                    "fluid_semantic_sha256": fluid.semantic_sha256(),
                }
            )
            print(
                f"captured composite tile {index}/{len(plan['tiles'])} "
                f"tile=({tile_x},{tile_z}) core=({core_x},{core_z})",
                flush=True,
            )

        world = publish_composite_world(
            output / "composite-world.json",
            output,
            sources,
            environment_id=args.environment_id,
            split=args.split,
            bounds_min=plan["bounds_min"],
            bounds_max=plan["bounds_max"],
        )
    finally:
        env.close()

    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "version": RECEIPT_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "world_template": WORLD_TEMPLATE,
        "worldgen_provider": WORLDGEN_PROVIDER,
        "environment_id": args.environment_id,
        "structure": args.structure,
        "seed": args.seed,
        "requested_extent": {
            "width": args.world_width,
            "depth": args.world_depth,
        },
        "native_reset_count": 1,
        "same_reset_tile_capture": True,
        "capture_group_sha256": group,
        "tile_grid": list(plan["tile_grid"]),
        "tile_count": len(plan["tiles"]),
        "tiles": tile_receipts,
        "bounds": {
            "minimum": list(plan["bounds_min"]),
            "maximum": list(plan["bounds_max"]),
        },
        "provenance": dict(provenance),
        "composite_manifest": {
            "path": "composite-world.json",
            "file_sha256": _file_sha256(output / "composite-world.json"),
            "semantic_sha256": world.semantic_sha256,
            "jax_world_count": 1,
            "shared_world_id": 0,
        },
        "capture_seconds": time.perf_counter() - started,
        "receipt_semantic_sha256": "",
    }
    receipt["receipt_semantic_sha256"] = _receipt_sha256(receipt)
    if lease is not None:
        update_owned_native_evidence_lease(
            lease,
            phase="complete",
            phase_status="passed",
            detail={
                "native_reset_count": 1,
                "tile_count": len(plan["tiles"]),
                "composite_semantic_sha256": world.semantic_sha256,
            },
        )
    return receipt


def composite_capture_plan(
    spawn: Sequence[float],
    width: int,
    depth: int,
    *,
    core_min_chunk_x: int | None = None,
    core_min_chunk_z: int | None = None,
) -> dict[str, Any]:
    """Resolve contiguous cores and a spawn-containing bounded window."""

    if len(spawn) != 3 or any(not math.isfinite(float(value)) for value in spawn):
        raise ValueError("spawn must contain three finite coordinates")
    width = _world_axis(width, "world_width")
    depth = _world_axis(depth, "world_depth")
    if (core_min_chunk_x is None) != (core_min_chunk_z is None):
        raise ValueError("both explicit core coordinates must be supplied")
    tiles_x = math.ceil(width / CORE_BLOCKS_PER_AXIS)
    tiles_z = math.ceil(depth / CORE_BLOCKS_PER_AXIS)
    coverage_x = tiles_x * CORE_BLOCKS_PER_AXIS
    coverage_z = tiles_z * CORE_BLOCKS_PER_AXIS
    if core_min_chunk_x is None:
        origin_x = math.floor((float(spawn[0]) - coverage_x / 2) / CHUNK_SIZE)
        origin_z = math.floor((float(spawn[2]) - coverage_z / 2) / CHUNK_SIZE)
    else:
        origin_x = int(core_min_chunk_x)
        origin_z = int(core_min_chunk_z)
    union_min_x = origin_x * CHUNK_SIZE
    union_min_z = origin_z * CHUNK_SIZE
    union_max_x = union_min_x + coverage_x
    union_max_z = union_min_z + coverage_z
    bounds_min_x = min(
        max(float(spawn[0]) - width / 2, union_min_x),
        union_max_x - width,
    )
    bounds_min_z = min(
        max(float(spawn[2]) - depth / 2, union_min_z),
        union_max_z - depth,
    )
    bounds_max_x = bounds_min_x + width
    bounds_max_z = bounds_min_z + depth
    if not (
        bounds_min_x <= float(spawn[0]) <= bounds_max_x
        and bounds_min_z <= float(spawn[2]) <= bounds_max_z
    ):
        raise ValueError("explicit composite core grid does not contain spawn")
    tiles = [
        {
            "tile_xz": (tile_x, tile_z),
            "core_min_chunk_xz": (
                origin_x + tile_x * CORE_CHUNKS_PER_AXIS,
                origin_z + tile_z * CORE_CHUNKS_PER_AXIS,
            ),
        }
        for tile_z in range(tiles_z)
        for tile_x in range(tiles_x)
    ]
    return {
        "tile_grid": (tiles_x, tiles_z),
        "core_origin_chunk_xz": (origin_x, origin_z),
        "tiles": tiles,
        "bounds_min": (bounds_min_x, float(MIN_Y), bounds_min_z),
        "bounds_max": (
            bounds_max_x,
            float(MIN_Y + WORLD_HEIGHT),
            bounds_max_z,
        ),
    }


def _capture_group_sha256(
    *,
    seed: int,
    structure: str,
    spawn: Sequence[float],
    expected_bridge: str,
) -> str:
    value = {
        "schema": "hytalerl_worldgen_v2_same_reset_nonce_v1",
        "seed": seed,
        "structure": structure,
        "spawn": [float(value) for value in spawn],
        "bridge_sha256": expected_bridge.lower(),
        "nonce": secrets.token_hex(32),
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _progress(tile_index: int, tile_count: int, phase: str):
    def report(index: int, total: int, _task: Any) -> None:
        if index == total or index % 50 == 0:
            print(
                f"tile {tile_index}/{tile_count} {phase} {index}/{total}",
                flush=True,
            )

    return report


def _receipt_sha256(value: Mapping[str, Any]) -> str:
    stable = dict(value)
    stable["generated_at_utc"] = ""
    stable["capture_seconds"] = 0.0
    stable["receipt_semantic_sha256"] = ""
    return _canonical_sha256(stable)


def _world_axis(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if not 1 <= value <= MAX_WORLD_AXIS:
        raise ValueError(f"{label} must be in [1, {MAX_WORLD_AXIS}]")
    return value


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    try:
        _world_axis(args.world_width, "world_width")
        _world_axis(args.world_depth, "world_depth")
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    if (args.core_min_chunk_x is None) != (args.core_min_chunk_z is None):
        parser.error("both explicit core coordinates must be supplied")
    if not args.structure.strip() or not args.environment_id.strip():
        parser.error("structure and environment-id must be non-empty")
    for label, path in (
        ("bridge jar", args.bridge_jar),
        ("server jar", args.server_jar),
        ("assets", args.assets),
    ):
        if not path.resolve().is_file():
            parser.error(f"{label} does not exist: {path}")


if __name__ == "__main__":
    main()
