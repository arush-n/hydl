"""Capture authored structure markers for explicitly pinned WorldGen V2 worlds."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

WORKSPACE = Path(__file__).resolve().parents[4]
for source in (
    Path(__file__).resolve().parents[2],
    WORKSPACE / "HytaleRL" / "hytalegym",
):
    if not source.is_dir():
        raise RuntimeError(f"required source directory not found: {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from hytalegym.worldgen.native_lease import (  # noqa: E402
    acquire_native_evidence_lease,
    update_owned_native_evidence_lease,
)
from hytalegym.worldgen.region import (  # noqa: E402
    CHUNK_SIZE,
    CORE_CHUNKS_PER_AXIS,
    MIN_Y,
    WORLD_HEIGHT,
)
from hytalegym.worldgen.region.native import (  # noqa: E402
    capture_native_region_pass,
)
from jax_port.bundles.bundle import V2JaxArtifact, V2JaxBundle  # noqa: E402
from jax_port.capture.capture_traversal import (  # noqa: E402
    _LiveV2Transport,
    _file_sha256,
    _require_live_identity,
    select_artifacts,
)
from jax_port.structures.native_structure_markers import (  # noqa: E402
    NATIVE_STRUCTURE_MARKER_MAX_CAPACITY,
    parse_worldgen_structure_markers,
    worldgen_structure_marker_request,
)
from jax_port.structures.structure_pack import (  # noqa: E402
    V2StructurePack,
    V2StructureRegistry,
    V2StructureSnapshot,
    compile_structure_snapshot,
    publish_v2_structure_pack,
)


STRUCTURE_CAPTURE_RECEIPT_SCHEMA = (
    "hytalerl_worldgen_v2_structure_capture_receipt_v1"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        metavar="STRUCTURE:SEED",
        help="repeat for each exact V2 world to cover",
    )
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument("--bridge-jar", type=Path, required=True)
    parser.add_argument(
        "--capacity",
        type=int,
        default=NATIVE_STRUCTURE_MARKER_MAX_CAPACITY,
    )
    parser.add_argument("--actor-profile", default="Kweebec_Razorleaf")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    bundle = V2JaxBundle.load(args.bundle, args.artifact_root)
    artifacts = select_artifacts(bundle, args.artifact)
    registry = V2StructureRegistry.load(args.registry)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        registry.path.relative_to(output.parent)
    except ValueError as error:
        parser.error(
            "registry must live below the structure-pack directory so the "
            "pack can pin its exact bytes"
        )
        raise AssertionError from error
    bridge_path = args.bridge_jar.resolve()
    if not bridge_path.is_file():
        parser.error(f"bridge JAR does not exist: {bridge_path}")
    bridge_sha256 = _file_sha256(bridge_path)
    source_bridge = str(bundle.manifest["source"]["bridge_jar_sha256"])
    if bridge_sha256 != source_bridge:
        parser.error(
            "bridge JAR differs from the exact V2 source bundle: "
            f"observed={bridge_sha256} expected={source_bridge}"
        )
    if not 1 <= args.capacity <= NATIVE_STRUCTURE_MARKER_MAX_CAPACITY:
        parser.error("capacity must be in [1, 256]")
    if not 1 <= args.port <= 65535:
        parser.error("port must be in [1, 65535]")
    if not isinstance(args.actor_profile, str) or not args.actor_profile.strip():
        parser.error("actor profile must be a non-empty string")

    with acquire_native_evidence_lease(
        host=args.host,
        port=args.port,
        purpose="worldgen_v2_structure_marker_capture",
        stage=output.parent,
        evidence_kinds=("worldgen_v2", "structure_markers"),
    ) as lease:
        pack, rows = capture_structure_pack(
            bundle,
            artifacts,
            registry,
            output,
            host=args.host,
            port=args.port,
            bridge_sha256=bridge_sha256,
            actor_profile=args.actor_profile.strip(),
            capacity=args.capacity,
            resume=bool(args.resume),
            lease=lease,
        )
    receipt = (
        args.receipt.resolve()
        if args.receipt is not None
        else output.with_suffix(".receipt")
    )
    publish_capture_receipt(receipt, pack, rows)
    print("WORLDGEN_V2_STRUCTURE_CAPTURE_PASS", flush=True)
    print(output, flush=True)
    print(receipt, flush=True)


def capture_structure_pack(
    bundle: V2JaxBundle,
    artifacts: Sequence[V2JaxArtifact],
    registry: V2StructureRegistry,
    output: Path,
    *,
    host: str,
    port: int,
    bridge_sha256: str,
    actor_profile: str,
    capacity: int,
    resume: bool,
    lease: Any | None = None,
) -> tuple[V2StructurePack, tuple[dict[str, Any], ...]]:
    snapshot_root = output.parent / f"{output.stem}.snapshots"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    paths = []
    rows = []
    total = len(artifacts)
    for index, artifact in enumerate(artifacts, start=1):
        if lease is not None:
            update_owned_native_evidence_lease(
                lease,
                phase="structure_marker_capture",
                phase_status="running",
                detail={
                    "structure": artifact.structure,
                    "seed": artifact.seed,
                    "artifact_index": index,
                    "artifact_count": total,
                },
            )
        path = snapshot_root / (
            f"{artifact.region_semantic_sha256}.structures"
        )
        if path.exists():
            if not resume:
                raise ValueError(
                    f"structure snapshot exists; pass --resume: {path}"
                )
            snapshot = _load_reusable_snapshot(
                path,
                artifact,
                registry,
                bridge_sha256,
            )
            row = _snapshot_row(snapshot, artifact, reused=True)
            print(
                f"V2_STRUCTURES {index}/{total} {artifact.structure}/"
                f"{artifact.seed} reused",
                flush=True,
            )
        else:
            snapshot, row = _capture_snapshot(
                bundle,
                artifact,
                registry,
                host=host,
                port=port,
                bridge_sha256=bridge_sha256,
                actor_profile=actor_profile,
                capacity=capacity,
                progress=f"{index}/{total}",
            )
            snapshot.save(path)
        paths.append(path)
        rows.append(row)
    return (
        publish_v2_structure_pack(
            output,
            bundle,
            paths,
            registry=registry,
        ),
        tuple(rows),
    )


def _capture_snapshot(
    bundle: V2JaxBundle,
    artifact: V2JaxArtifact,
    registry: V2StructureRegistry,
    *,
    host: str,
    port: int,
    bridge_sha256: str,
    actor_profile: str,
    capacity: int,
    progress: str,
) -> tuple[V2StructureSnapshot, dict[str, Any]]:
    source_path = (bundle.artifact_root / artifact.region_path).resolve()
    if _file_sha256(source_path) != artifact.region_file_sha256:
        raise ValueError("source V2 Region file SHA-256 changed")
    transport = _LiveV2Transport(host, port)
    transport.connect()
    try:
        print(
            f"V2_STRUCTURES {progress} {artifact.structure}/{artifact.seed} reset",
            flush=True,
        )
        info = transport.reset(artifact, actor_profile)
        _require_live_identity(info, artifact, bridge_sha256)
        live = capture_native_region_pass(
            transport,
            requested_core=artifact.core_min_chunk_xz,
        )
        if live.semantic_artifact_digest() != artifact.region_semantic_sha256:
            raise ValueError(
                "live regenerated V2 Region differs from the pinned artifact"
            )
        bounds = marker_capture_bounds(artifact)
        marker_ids = tuple(row.marker_asset_id for row in registry.entries)
        response = transport.request(
            worldgen_structure_marker_request(
                bounds,
                marker_ids,
                capacity=capacity,
            )
        )
        capture = parse_worldgen_structure_markers(
            response,
            expected_bridge_sha256=bridge_sha256,
            expected_seed=artifact.seed,
            expected_marker_asset_ids=marker_ids,
        )
        if capture.worldgen_provider != "HytaleGenerator":
            raise ValueError("structure markers came from another provider")
        snapshot = compile_structure_snapshot(
            artifact,
            capture.complete_markers(),
            registry,
            evidence_bridge_sha256=bridge_sha256,
        )
        return snapshot, _snapshot_row(
            snapshot,
            artifact,
            reused=False,
            live_region_verification="semantic_sha256_equal",
            native_uuid_sha256=hashlib.sha256(
                capture.uuid_bytes.tobytes()
            ).hexdigest(),
        )
    finally:
        transport.close()


def marker_capture_bounds(artifact: V2JaxArtifact) -> tuple[float, ...]:
    core_x = artifact.core_min_chunk_xz[0] * CHUNK_SIZE
    core_z = artifact.core_min_chunk_xz[1] * CHUNK_SIZE
    return (
        float(core_x - CHUNK_SIZE),
        float(MIN_Y),
        float(core_z - CHUNK_SIZE),
        float(core_x + (CORE_CHUNKS_PER_AXIS + 1) * CHUNK_SIZE),
        float(MIN_Y + WORLD_HEIGHT),
        float(core_z + (CORE_CHUNKS_PER_AXIS + 1) * CHUNK_SIZE),
    )


def _load_reusable_snapshot(
    path: Path,
    artifact: V2JaxArtifact,
    registry: V2StructureRegistry,
    bridge_sha256: str,
) -> V2StructureSnapshot:
    snapshot = V2StructureSnapshot.load(path)
    expected = {
        "source_region_semantic_sha256": artifact.region_semantic_sha256,
        "evidence_bridge_sha256": bridge_sha256,
        "registry_semantic_sha256": registry.semantic_sha256,
    }
    actual = {
        "source_region_semantic_sha256": (
            snapshot.source_region_semantic_sha256
        ),
        "evidence_bridge_sha256": snapshot.evidence_bridge_sha256,
        "registry_semantic_sha256": snapshot.registry_semantic_sha256,
    }
    if actual != expected:
        raise ValueError(f"reusable V2 structure snapshot differs: {actual}")
    return snapshot


def _snapshot_row(
    snapshot: V2StructureSnapshot,
    artifact: V2JaxArtifact,
    *,
    reused: bool,
    live_region_verification: str | None = None,
    native_uuid_sha256: str | None = None,
) -> dict[str, Any]:
    return {
        "structure": artifact.structure,
        "seed": artifact.seed,
        "source_region_semantic_sha256": artifact.region_semantic_sha256,
        "snapshot_semantic_sha256": snapshot.semantic_sha256,
        "instance_count": len(snapshot.instances),
        "complete_instance_count": sum(
            row.capture_coverage == "complete" for row in snapshot.instances
        ),
        "clipped_instance_count": sum(
            row.capture_coverage == "clipped_to_capture"
            for row in snapshot.instances
        ),
        "reused": reused,
        "live_region_verification": live_region_verification,
        "native_uuid_sha256": native_uuid_sha256,
    }


def publish_capture_receipt(
    path: Path,
    pack: V2StructurePack,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    value = {
        "schema": STRUCTURE_CAPTURE_RECEIPT_SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_bundle_semantic_sha256": pack.bundle.semantic_sha256,
        "structure_pack_semantic_sha256": pack.semantic_sha256,
        "evidence_bridge_sha256": pack.manifest["evidence_bridge_sha256"],
        "marker_contract_sha256": pack.manifest["marker_contract_sha256"],
        "registry_semantic_sha256": pack.registry.semantic_sha256,
        "artifact_count": len(rows),
        "rows": [dict(row) for row in rows],
    }
    value["receipt_semantic_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
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
    return path


if __name__ == "__main__":
    main()
