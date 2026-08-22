"""Capture native traversal graphs for explicitly pinned WorldGen V2 worlds."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[4]
for source in (
    Path(__file__).resolve().parents[2],
    WORKSPACE / "HytaleRL" / "hytalegym",
):
    if not source.is_dir():
        raise RuntimeError(f"required source directory not found: {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from hytalegym.utils.connection import BridgeConnection  # noqa: E402
from hytalegym.worldgen.native_lease import (  # noqa: E402
    acquire_native_evidence_lease,
    update_owned_native_evidence_lease,
)
from hytalegym.worldgen.native_traversal import (  # noqa: E402
    NATIVE_TRAVERSAL_EDGE_PROBE_MAX_SAMPLES,
    NativeTraversalEdgeProbeCapture,
    native_traversal_edge_probe_request,
    parse_native_traversal_edge_probe,
)
from hytalegym.worldgen.region import (  # noqa: E402
    REGION_TRAVERSAL_HORIZONTAL_TOLERANCE,
    REGION_TRAVERSAL_VERTICAL_TOLERANCE,
    NativeRegionSnapshot,
    compile_region_traversal_candidates,
    native_region_traversal_graph_from_captures,
)
from hytalegym.worldgen.region.native import (  # noqa: E402
    capture_native_region_pass,
)
from hytalegym.worldgen.region.stability import (  # noqa: E402
    require_current_native_runtime_bridge,
)
from jax_port.bundles.bundle import V2JaxArtifact, V2JaxBundle  # noqa: E402
from jax_port.bundles.traversal_pack import (  # noqa: E402
    V2TraversalPack,
    publish_v2_traversal_pack,
)


WORLD_TEMPLATE = "hytale_generator"
WORLDGEN_PROVIDER = "HytaleGenerator"
TRAVERSAL_CAPTURE_RECEIPT_SCHEMA = (
    "hytalerl_worldgen_v2_traversal_capture_receipt_v1"
)


class _LiveV2Transport:
    def __init__(self, host: str, port: int) -> None:
        self.connection = BridgeConnection(host, port, timeout=180.0)

    def connect(self) -> None:
        self.connection.connect()
        response = self.connection.send_and_recv(
            {"type": "config", "tick_rate": 1, "max_episode_steps": 100}
        )
        if response.get("type") != "ack":
            raise ConnectionError("bridge did not acknowledge traversal config")

    def reset(
        self,
        artifact: V2JaxArtifact,
        actor_profile: str,
    ) -> Mapping[str, Any]:
        response = self.connection.send_and_recv(
            reset_request(artifact, actor_profile)
        )
        if response.get("type") != "observation":
            raise ConnectionError("V2 traversal reset returned no observation")
        info = response.get("info")
        if not isinstance(info, Mapping):
            raise ValueError("V2 traversal reset lacks an info object")
        return info

    def capture_region_manifest(
        self,
        core_min_chunk_x: int | None = None,
        core_min_chunk_z: int | None = None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {"type": "region_manifest"}
        if core_min_chunk_x is not None:
            request["core_min_chunk_x"] = int(core_min_chunk_x)
            request["core_min_chunk_z"] = int(core_min_chunk_z)
        response = self.connection.send_and_recv(request)
        if response.get("type") != "region_manifest":
            raise ConnectionError("bridge did not return a Region manifest")
        return response

    def capture_region_section(
        self,
        chunk_x: int,
        chunk_z: int,
        section_y: int,
    ) -> dict[str, Any]:
        response = self.connection.send_and_recv(
            {
                "type": "region_section",
                "chunk_x": int(chunk_x),
                "chunk_z": int(chunk_z),
                "section_y": int(section_y),
            }
        )
        if response.get("type") != "region_section":
            raise ConnectionError("bridge did not return a Region section")
        return response

    def capture_edge_probe(
        self,
        starts: np.ndarray,
        targets: np.ndarray,
        horizontal: np.ndarray,
        vertical: np.ndarray,
    ) -> NativeTraversalEdgeProbeCapture:
        return parse_native_traversal_edge_probe(
            self.connection.send_and_recv(
                native_traversal_edge_probe_request(
                    starts,
                    targets,
                    horizontal,
                    vertical,
                )
            )
        )

    def request(self, message: dict[str, Any]) -> dict[str, Any]:
        return self.connection.send_and_recv(message)

    def close(self) -> None:
        try:
            self.connection.send_and_recv({"type": "close"})
        except Exception:
            pass
        self.connection.close()


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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5557)
    parser.add_argument("--bridge-jar", type=Path, required=True)
    parser.add_argument("--actor-profile", default="Kweebec_Razorleaf")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    bundle = V2JaxBundle.load(args.bundle, args.artifact_root)
    artifacts = select_artifacts(bundle, args.artifact)
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
    if not isinstance(args.actor_profile, str) or not args.actor_profile.strip():
        parser.error("actor profile must be a non-empty string")
    if not 1 <= args.port <= 65535:
        parser.error("port must be in [1, 65535]")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with acquire_native_evidence_lease(
        host=args.host,
        port=args.port,
        purpose="worldgen_v2_traversal_capture",
        stage=output.parent,
        evidence_kinds=("worldgen_v2", "native_traversal"),
    ) as lease:
        pack, rows = capture_traversal_pack(
            bundle,
            artifacts,
            output,
            host=args.host,
            port=args.port,
            bridge_sha256=bridge_sha256,
            actor_profile=args.actor_profile.strip(),
            resume=bool(args.resume),
            lease=lease,
        )
    receipt = (
        args.receipt.resolve()
        if args.receipt is not None
        else output.with_suffix(".receipt")
    )
    publish_capture_receipt(receipt, pack, rows)
    print("WORLDGEN_V2_TRAVERSAL_CAPTURE_PASS", flush=True)
    print(output, flush=True)
    print(receipt, flush=True)


def reset_request(
    artifact: V2JaxArtifact,
    actor_profile: str,
) -> dict[str, Any]:
    """Return the explicit V2 reset; legacy ``world=hytale`` is forbidden."""

    if not isinstance(artifact, V2JaxArtifact):
        raise TypeError("artifact must be a V2JaxArtifact")
    if not isinstance(actor_profile, str) or not actor_profile.strip():
        raise ValueError("actor_profile must be a non-empty string")
    return {
        "type": "reset",
        "task_id": "survive",
        "seed": artifact.seed,
        "options": {
            "backend": "native",
            "world": WORLD_TEMPLATE,
            "worldgen_structure": artifact.structure,
            "npc_role": actor_profile.strip(),
            "combat_target_active": False,
            "fidelity_fixture": "static_region",
        },
    }


def select_artifacts(
    bundle: V2JaxBundle,
    specifications: Sequence[str],
) -> tuple[V2JaxArtifact, ...]:
    if isinstance(specifications, (str, bytes)):
        raise TypeError("artifact specifications must be a sequence")
    selected = []
    for specification in specifications:
        if not isinstance(specification, str) or ":" not in specification:
            raise ValueError("artifact must use STRUCTURE:SEED")
        structure, raw_seed = specification.rsplit(":", 1)
        if not structure:
            raise ValueError("artifact structure cannot be empty")
        try:
            seed = int(raw_seed)
        except ValueError as error:
            raise ValueError("artifact seed must be an integer") from error
        if str(seed) != raw_seed or seed < 0:
            raise ValueError("artifact seed must be canonical and non-negative")
        selected.extend(bundle.select(structure, seeds=[seed]))
    identities = [(row.structure, row.seed) for row in selected]
    if not selected or len(set(identities)) != len(identities):
        raise ValueError("selected traversal artifacts must be non-empty and unique")
    return tuple(selected)


def capture_traversal_pack(
    bundle: V2JaxBundle,
    artifacts: Sequence[V2JaxArtifact],
    output: Path,
    *,
    host: str,
    port: int,
    bridge_sha256: str,
    actor_profile: str,
    resume: bool,
    lease: Any | None = None,
) -> tuple[V2TraversalPack, tuple[dict[str, Any], ...]]:
    graph_root = output.parent / f"{output.stem}.graphs"
    graph_root.mkdir(parents=True, exist_ok=True)
    graph_paths: list[Path] = []
    rows = []
    total = len(artifacts)
    for index, artifact in enumerate(artifacts, start=1):
        if lease is not None:
            update_owned_native_evidence_lease(
                lease,
                phase="traversal_capture",
                phase_status="running",
                detail={
                    "structure": artifact.structure,
                    "seed": artifact.seed,
                    "artifact_index": index,
                    "artifact_count": total,
                },
            )
        graph_path = graph_root / (
            f"{artifact.region_semantic_sha256}.traversal.npz"
        )
        if graph_path.exists():
            if not resume:
                raise ValueError(
                    f"traversal graph exists; pass --resume: {graph_path}"
                )
            graph = _load_reusable_graph(
                graph_path,
                artifact,
                bridge_sha256,
                actor_profile,
                bundle,
            )
            row = _graph_row(graph, artifact, reused=True)
            print(
                f"V2_TRAVERSAL {index}/{total} {artifact.structure}/"
                f"{artifact.seed} reused",
                flush=True,
            )
        else:
            graph, row = _capture_graph(
                bundle,
                artifact,
                host=host,
                port=port,
                bridge_sha256=bridge_sha256,
                actor_profile=actor_profile,
                progress=f"{index}/{total}",
            )
            graph.save(graph_path)
        graph_paths.append(graph_path)
        rows.append(row)
    return (
        publish_v2_traversal_pack(output, bundle, graph_paths),
        tuple(rows),
    )


def _capture_graph(
    bundle: V2JaxBundle,
    artifact: V2JaxArtifact,
    *,
    host: str,
    port: int,
    bridge_sha256: str,
    actor_profile: str,
    progress: str,
):
    snapshot_path = (bundle.artifact_root / artifact.region_path).resolve()
    if _file_sha256(snapshot_path) != artifact.region_file_sha256:
        raise ValueError("source V2 Region file SHA-256 changed")
    snapshot = NativeRegionSnapshot.load(snapshot_path)
    if snapshot.semantic_artifact_digest() != artifact.region_semantic_sha256:
        raise ValueError("source V2 Region semantic SHA-256 changed")
    transport = _LiveV2Transport(host, port)
    transport.connect()
    try:
        print(
            f"V2_TRAVERSAL {progress} {artifact.structure}/{artifact.seed} reset",
            flush=True,
        )
        info = transport.reset(artifact, actor_profile)
        _require_live_identity(info, artifact, bridge_sha256)
        core = artifact.core_min_chunk_xz
        print(
            f"V2_TRAVERSAL {progress} {artifact.structure}/{artifact.seed} "
            "verify-region",
            flush=True,
        )
        live_snapshot = capture_native_region_pass(
            transport,
            requested_core=core,
        )
        if live_snapshot.semantic_artifact_digest() != (
            artifact.region_semantic_sha256
        ):
            raise ValueError(
                "live regenerated V2 Region differs from the pinned artifact"
            )
        profile = _capture_profile(transport, artifact)
        compile_start = time.perf_counter()
        candidates = compile_region_traversal_candidates(
            snapshot,
            actor_bounds=profile.actor_bounds,
            maximum_climb_height=profile.maximum_climb_height,
            maximum_drop_height=profile.maximum_drop_height,
        )
        compile_seconds = time.perf_counter() - compile_start
        captures = []
        edge_count = int(candidates.edge_source.shape[0])
        capture_start = time.perf_counter()
        for start in range(
            0,
            edge_count,
            NATIVE_TRAVERSAL_EDGE_PROBE_MAX_SAMPLES,
        ):
            stop = min(
                start + NATIVE_TRAVERSAL_EDGE_PROBE_MAX_SAMPLES,
                edge_count,
            )
            size = stop - start
            horizontal = np.full(
                size,
                REGION_TRAVERSAL_HORIZONTAL_TOLERANCE,
                dtype=np.float64,
            )
            vertical = np.full(
                size,
                REGION_TRAVERSAL_VERTICAL_TOLERANCE,
                dtype=np.float64,
            )
            capture = transport.capture_edge_probe(
                candidates.edge_start_position[start:stop],
                candidates.edge_target_position[start:stop],
                horizontal,
                vertical,
            )
            if capture.seed != artifact.seed:
                raise ValueError("native traversal seed changed during capture")
            captures.append(capture)
            if stop == edge_count or (
                start // NATIVE_TRAVERSAL_EDGE_PROBE_MAX_SAMPLES + 1
            ) % 25 == 0:
                print(
                    f"V2_TRAVERSAL {progress} {artifact.structure}/"
                    f"{artifact.seed} edges={stop}/{edge_count}",
                    flush=True,
                )
        capture_seconds = time.perf_counter() - capture_start
        graph = native_region_traversal_graph_from_captures(
            candidates,
            captures,
            native_evidence_jar_sha256=bridge_sha256,
            actor_profile=actor_profile,
        )
        return graph, _graph_row(
            graph,
            artifact,
            reused=False,
            region_verification="semantic_sha256_equal",
            compile_seconds=compile_seconds,
            capture_seconds=capture_seconds,
        )
    finally:
        transport.close()


def _capture_profile(
    transport: _LiveV2Transport,
    artifact: V2JaxArtifact,
) -> NativeTraversalEdgeProbeCapture:
    point = np.asarray([artifact.spawn_position], dtype=np.float64)
    tolerance = np.asarray(
        [REGION_TRAVERSAL_HORIZONTAL_TOLERANCE], dtype=np.float64
    )
    return transport.capture_edge_probe(point, point, tolerance, tolerance)


def _require_live_identity(
    info: Mapping[str, Any],
    artifact: V2JaxArtifact,
    bridge_sha256: str,
) -> None:
    require_current_native_runtime_bridge(info, expected=bridge_sha256.upper())
    expected = {
        "world_template": WORLD_TEMPLATE,
        "worldgen_provider": WORLDGEN_PROVIDER,
        "worldgen_structure": artifact.structure,
        "worldgen_seed": artifact.seed,
        "block_ticking_enabled": False,
    }
    mismatches = {
        key: (wanted, info.get(key))
        for key, wanted in expected.items()
        if info.get(key) != wanted
    }
    if mismatches:
        raise RuntimeError(f"live V2 traversal identity differs: {mismatches}")
    if info.get("worldgen_version") in (None, "", "unknown", "uninitialized"):
        raise RuntimeError("live V2 traversal provider lacks a version")


def _load_reusable_graph(
    path: Path,
    artifact: V2JaxArtifact,
    bridge_sha256: str,
    actor_profile: str,
    bundle: V2JaxBundle,
):
    from hytalegym.worldgen.region import NativeRegionTraversalGraph

    graph = NativeRegionTraversalGraph.load(path)
    expected = {
        "source_region_semantic_sha256": artifact.region_semantic_sha256,
        "native_evidence_jar_sha256": bridge_sha256.upper(),
        "actor_profile": actor_profile,
        "seed": artifact.seed,
        "worldgen_provider": bundle.manifest["source"]["worldgen_provider"],
    }
    actual = {
        "source_region_semantic_sha256": graph.source_region_semantic_sha256,
        "native_evidence_jar_sha256": graph.native_evidence_jar_sha256,
        "actor_profile": graph.metadata.get("actor_profile"),
        "seed": graph.metadata.get("seed"),
        "worldgen_provider": graph.metadata.get("worldgen_provider"),
    }
    if actual != expected:
        raise ValueError(f"reusable V2 traversal graph differs: {actual}")
    return graph


def _graph_row(
    graph,
    artifact: V2JaxArtifact,
    *,
    reused: bool,
    region_verification: str | None = None,
    compile_seconds: float | None = None,
    capture_seconds: float | None = None,
) -> dict[str, Any]:
    return {
        "structure": artifact.structure,
        "seed": artifact.seed,
        "source_region_semantic_sha256": artifact.region_semantic_sha256,
        "graph_semantic_sha256": graph.semantic_digest(),
        "node_count": graph.node_count,
        "candidate_edge_count": int(graph.metadata["candidate_count"]),
        "accepted_edge_count": graph.edge_count,
        "reused": reused,
        "live_region_verification": region_verification,
        "compile_seconds": compile_seconds,
        "native_capture_seconds": capture_seconds,
    }


def publish_capture_receipt(
    path: Path,
    pack: V2TraversalPack,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    value = {
        "schema": TRAVERSAL_CAPTURE_RECEIPT_SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_bundle_semantic_sha256": pack.bundle.semantic_sha256,
        "traversal_pack_semantic_sha256": pack.semantic_sha256,
        "native_evidence_jar_sha256": pack.manifest[
            "native_evidence_jar_sha256"
        ],
        "actor_profile": pack.actor_profile,
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
